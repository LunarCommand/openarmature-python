## Non-obvious shapes

Recipes that aren't deducible from the API surface alone. The primitives docs tell you what's possible; this section tells you what's smart.

### Declare a non-clobbering reducer on accumulator list fields

State fields default to `last_write_wins` — each node's write replaces the prior value for that field. For scalar fields (`status: str`, `count: int`) that's usually what you want. For list fields that accumulate contributions across multiple nodes (`messages: list[Message]`, `events: list[Event]`, `results: list[Result]`), it's the wrong default — every node's contribution silently clobbers everything before it.

Declare `append` (or another non-clobbering reducer) at the state class:

```python
from typing import Annotated
from pydantic import Field
from openarmature.graph import State, append

class WorkflowState(State):
    messages: Annotated[list[Message], append] = Field(default_factory=list)
    events: Annotated[list[Event], append] = Field(default_factory=list)
    final_status: str = "pending"   # last_write_wins is fine here
```

The failure mode without `append` is silent and easy to misdiagnose — the final state shows only the last node's contribution to the list, with no error. Common "why is my accumulator empty?" question. `merge` is the equivalent for `dict[str, V]` fields that accumulate keys across nodes.

### Branch on `Response.finish_reason` before reading `message.content`

After `await provider.complete(messages, tools=[...])` returns, the shape of `Response` varies by `finish_reason`:

- `finish_reason == "stop"` — assistant produced a content response. `message.content` carries the text; `message.tool_calls` is empty.
- `finish_reason == "tool_calls"` — assistant emitted tool calls. `message.tool_calls` carries the list; `message.content` is typically empty (model didn't say anything beyond the tool calls).
- `finish_reason == "length"` / `"content_filter"` / `"error"` — completion was cut off or refused; `message.content` may be partial or empty.

Post-LLM logic that reads `message.content` without checking `finish_reason` misses the entire tool-calling path:

```python
response = await provider.complete(messages, tools=tools)

if response.finish_reason == "tool_calls":
    # Dispatch each tool call, append ToolMessage responses, re-call complete()
    for tc in response.message.tool_calls:
        result = dispatch_tool(tc.name, tc.arguments)
        messages.append(ToolMessage(content=result, tool_call_id=tc.id))
    response = await provider.complete(messages, tools=tools)
elif response.finish_reason == "stop":
    handle_text(response.message.content)
else:
    handle_error_or_partial(response)
```

The discriminator is one branch; missing it gives you empty data on tool-call responses and silently wrong behavior on truncations.

### `disable_llm_payload` defaults to `True` — flip it for LLM-aware observability backends

The `OTelObserver` (and any spec-conformant observer reading LLM events) defaults `disable_llm_payload: bool = True` per spec §5.5's "default-off by privacy" framing. Without flipping the flag, LLM spans carry GenAI semconv attributes (token counts, model name, finish reason) but NOT the message payload (input messages, response content, request extras).

That's the right default for general OpenArmature use — payloads may contain PII the user hasn't audited, and storage cost grows with prompt size. But it's the WRONG default if you're wiring up an LLM-aware observability backend (Langfuse, Phoenix, Honeycomb's LLM lens) that renders the message stream as part of its generation view. Backends will show "empty" generations and you'll wonder why.

Flip the flag once at observer construction:

```python
from openarmature.observability import OTelObserver

observer = OTelObserver(
    span_processor=your_exporter,
    disable_llm_payload=False,   # opt in to message-payload attributes
)
compiled.attach_observer(observer)
```

The companion `disable_genai_semconv` flag defaults to `False` — GenAI semconv attributes emit by default since they're how LLM-aware backends render anything at all. Don't flip that one unless you're routing GenAI emission through a different layer.

### Use the bundled `FilesystemCheckpointer` or `SQLiteCheckpointer`, not a hand-rolled serializer

The temptation when persisting graph state is to `json.dumps(state.model_dump())` and write to a file. Don't. The shipped Checkpointer backends handle every contract `openarmature.checkpoint.Checkpointer` defines — round-trip integrity, `parent_states` for inner-save resume, fan-out progress tracking, schema-version migration, listing by `correlation_id`, `CheckpointRecordInvalid` on shape drift. A hand-rolled serializer that "works" on the happy path silently fails the moment a fan-out crash leaves an in-flight save record, and you'll be debugging it for hours before realizing the bundled backend exists.

If your storage requirement isn't local disk (`FilesystemCheckpointer`) or local SQLite (`SQLiteCheckpointer` — also supports `:memory:` and arbitrary file paths), implement the `Checkpointer` Protocol against your backend rather than wrapping state serialization yourself. Custom backends inherit the spec's correctness contract for free.

### Subgraphs > conditional-edge spaghetti when branches don't share state

A common shape is "after this LLM call, route to either a JSON-extraction node or a tool-dispatch node depending on `finish_reason`." The naive solution is two conditional edges from the LLM node, one to each downstream. That works for two branches; it scales poorly past three.

When the branches operate on different sub-shapes of state — e.g., one path is "extract JSON, then validate" while another is "dispatch tools, loop until done, then summarize" — encapsulate each as a `SubgraphNode` and route from the LLM node to the right subgraph. Each subgraph has its own state schema (projected from the parent), its own entry node, and its own internal topology. The parent graph becomes a switchboard with a few edges; the complexity lives one layer down where it composes cleanly.

### Be explicit with `tool_choice`; don't trust the provider's default

`Provider.complete(messages, tools, tool_choice=...)` accepts `"auto"`, `"required"`, `"none"`, or a `ForceTool(name=...)` record. When you omit `tool_choice`, the OpenAI provider's own default applies — usually `"auto"` when `tools` is non-empty, but documented per-provider. A pipeline that wants deterministic tool-calling (a routing node that MUST produce a tool call, a guarded LLM call that MUST NOT call tools) should pin `tool_choice` explicitly rather than relying on the provider default.

Pre-send validation catches the three §5 failure modes (`required` with empty tools, `ForceTool` with empty tools, `ForceTool.name` not in tools) and raises `ProviderInvalidRequest` before the HTTP call. Not all providers honor `tool_choice` — confirm with your provider's docs — but the OpenAI-compatible mapping is in `OpenAIProvider`.

### Always `await graph.drain()` in short-lived processes; supply a `timeout` if observers might hang

`CompiledGraph.invoke()` returns when the graph reaches END or raises; observer events are dispatched onto a per-invocation queue and delivered by a background worker. The graph's execution loop never awaits observer processing. In a long-running service this is invisible — the worker drains naturally. In a CLI, script, or serverless function, the process exits before the worker finishes, and any late observer events (typically the last node's `completed` event plus any `checkpoint_saved` events) get dropped.

Always call `await graph.drain()` before the short-lived process exits. If your observer set includes anything that might hang (a metrics observer with a flaky network endpoint, an OTel exporter behind a slow OTLP collector), supply a `timeout`:

```python
summary = await graph.drain(timeout=5.0)
if summary.timeout_reached:
    log.warning("drain truncated: %d events undelivered", summary.undelivered_count)
```

The compiled graph stays usable for subsequent invocations after a timed-out drain — workers are cancelled cleanly, no partial state leaks.

### Three exception hierarchies; know which one your code catches

`openarmature` exceptions split across three sibling hierarchies:

- `RuntimeGraphError` (in `openarmature.graph`) — node execution failures: `NodeException`, `RoutingError`, `EdgeException`, `ReducerError`, `StateValidationError`. Each has a `category` string matching the spec's canonical error categories.
- `CheckpointError` (in `openarmature.checkpoint`) — persistence failures: `CheckpointNotFound`, `CheckpointSaveFailed`, `CheckpointRecordInvalid`, `CheckpointStateMigrationMissing`, `CheckpointStateMigrationFailed`, `CheckpointStateMigrationChainAmbiguous`.
- `LlmProviderError` (in `openarmature.llm`) — provider call failures: `ProviderAuthentication`, `ProviderInvalidRequest`, `ProviderInvalidResponse`, `ProviderInvalidModel`, `ProviderModelNotLoaded`, `ProviderRateLimit`, `ProviderUnavailable`, `ProviderUnsupportedContentBlock`, `StructuredOutputInvalid`.

Catching `Exception` works but is too broad; catching one hierarchy misses the other two. If you want to branch on category strings (e.g., for retry logic), catch the relevant base — `RuntimeGraphError` covers all five spec runtime categories, `LlmProviderError` covers all nine provider categories, `CheckpointError` covers all six checkpoint categories. The `TRANSIENT_CATEGORIES` frozenset in `openarmature.llm` enumerates which provider categories are retriable.

### Reconcile `started` → `completed` pairs via a per-invocation dict keyed on `(namespace, branch_name, attempt_index, fan_out_index)`

Observers receive `started` and `completed` events as a pair per node attempt, but the engine doesn't carry a `step_id`-like correlation field across the pair (it doesn't need one for its own logic — the events arrive serially per spec §6). Observer code that needs to thread per-call state — start timestamps, request payloads, custom IDs — between the two events has to reconcile manually.

The pair identity is `(namespace, branch_name, attempt_index, fan_out_index)`: that tuple is unique within an invocation (per graph-engine §6 uniqueness invariants — `branch_name` and `fan_out_index` are independent slots, so a node inside a parallel-branches branch needs `branch_name` in the key to avoid colliding with the same-named node in a sibling branch). Carry per-invocation state in a `dict[invocation_id, dict[tuple, value]]` and look up on `completed`:

```python
class StepTimingObserver:
    def __init__(self) -> None:
        # invocation_id -> {(namespace, branch_name, attempt_index, fan_out_index): start_ts}
        self._pending: dict[str, dict[tuple[Any, ...], float]] = {}

    async def __call__(self, event: NodeEvent) -> None:
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        key = (event.namespace, event.branch_name, event.attempt_index, event.fan_out_index)
        if event.phase == "started":
            self._pending.setdefault(invocation_id, {})[key] = time.monotonic()
        elif event.phase == "completed":
            start = self._pending.get(invocation_id, {}).pop(key, None)
            if start is not None:
                duration = time.monotonic() - start
                # … emit timing
            # Sweep when the dict empties (last completed for this invocation).
            if not self._pending.get(invocation_id):
                self._pending.pop(invocation_id, None)
```

The `_pending[invocation_id]` sub-dict naturally tracks in-flight pairs and drains as completions arrive. Sweep the outer entry when the sub-dict empties so long-running services don't accumulate per-invocation entries. If you also subscribe to drain events, that's another sweep opportunity. The same pattern works for any per-call state the observer needs to thread across the pair.

### Filter `openarmature.*`-namespaced events when your observer only cares about user nodes

OA emits observer events under sentinel node-names for its own internal dispatch: `openarmature.llm.complete` for LLM provider calls (proposal 0024), `openarmature.checkpoint.migrate` for state-migration runs (proposal 0014), `openarmature.checkpoint.save` for checkpoint saves (proposal 0010). These events let the OTel / Langfuse observers emit LLM-provider spans, checkpoint-migrate spans, etc. — but a custom observer that only cares about user-defined node activity sees them as noise:

```python
async def __call__(self, event: NodeEvent) -> None:
    # Skip OA-internal events; only react to user node activity.
    if event.namespace and event.namespace[0].startswith("openarmature."):
        return
    # … user-node handling
```

`event.namespace[0]` is the safest discriminator (the leaf `event.node_name` would also work for LLM events but won't match the checkpoint sentinels since those repurpose `node_name` differently). Don't try to filter on `current_invocation_id() is None` — OA-internal events are dispatched within the same invocation context as user-node events, so `invocation_id` is set for both; the namespace-prefix check is the stable contract.

### A `with_state_migration` recipe — register migrations alongside the state class, run on resume

`GraphBuilder.with_state_migration(s)` registers callables that transform an old-schema state record into the current schema. The engine calls them automatically on `invoke(resume_invocation=...)` when the loaded record's `schema_version` doesn't match `state_cls.schema_version`. The migration callable's signature is `(state_dict: dict, from_version: str, to_version: str) -> dict`; it receives the raw deserialized record and returns the new shape.

Wire it up at compile time:

```python
class PipelineState(State):
    schema_version: ClassVar[str] = "2"
    # … v2 fields

def _migrate_v1_to_v2(state_dict: dict, from_version: str, to_version: str) -> dict:
    # Old field "step_count" renamed to "steps_completed" in v2.
    state_dict["steps_completed"] = state_dict.pop("step_count", 0)
    return state_dict

compiled = (
    GraphBuilder(PipelineState)
    .add_node("step", _step_body)
    .add_edge("step", END)
    .set_entry("step")
    .with_state_migration(from_version="1", to_version="2", migrate=_migrate_v1_to_v2)
    .compile()
)
compiled.attach_checkpointer(checkpointer)
```

Important detail: the migration runs once on resume, before any node body fires; the engine dispatches a synthetic `checkpoint_migrated` observer event (per spec §6 cross-ref) so observers can emit a migration span. The migrated state is what `_step_body` sees on resume — you do NOT need to handle both v1 and v2 shapes in node bodies.

When chaining multiple migrations (v1 → v2 → v3), register each step separately via repeated `with_state_migration` calls; the engine walks the chain in version order. If the chain has gaps (registered v1→v2 and v3→v4 but a record is at v2 with `to_version="4"`), the engine raises `CheckpointStateMigrationMissing` at resume time — fail-loud rather than silently skipping.

### Fan-out subgraphs that emit `list[X]` per instance produce `list[list[X]]` at `target_field`

When a fan-out's per-instance state collects a `list[X]` as its `collect_field` (e.g., each instance produces 0..N records), the engine's contribution step is `[s[cfg.collect_field] for s in successes]` — every instance's value becomes one element of the outer list. With `list[X]` per-instance, the parent receives `list[list[X]]`, and the default `append` reducer on the parent's `Annotated[list[X], append]` field preserves the nesting verbatim. Pydantic then fails to validate each `list[X]` element against `X`:

```
attributed_candidates.0  Input should be a valid dictionary or
  instance of ClaimCandidate [input_value=[ClaimCandidate(...)],
  input_type=list]
```

The fix is the `concat_flatten` built-in reducer (proposal 0036) — the list-of-lists analog of `append`. Declare it on the parent's collection field:

```python
from typing import Annotated

from pydantic import Field

from openarmature.graph import State, concat_flatten

class PipelineState(State):
    attributed_candidates: Annotated[list[ClaimCandidate], concat_flatten] = Field(default_factory=list)
```

`concat_flatten` folds the per-instance lists into one flat list (`[*prior, *(item for sublist in update for item in sublist)]`), strict like `append` — it raises `ReducerError` if any element of the update isn't itself a list.

The dict-shaped analog is `merge_all` (also proposal 0036): when each fan-out instance contributes a `dict[str, X]`, the parent's `target_field` receives `list[dict]`, which plain `merge` can't consume. `merge_all` folds the sequence of mappings into the prior with shallow last-write-wins per key:

```python
from typing import Annotated

from pydantic import Field

from openarmature.graph import State, merge_all

class PipelineState(State):
    keyed_results: Annotated[dict[str, Result], merge_all] = Field(default_factory=dict)
```

Single-record-per-instance fan-outs (`collect_field: str`, parent field `Annotated[list[X], append]`) don't hit this — the engine still wraps each instance's value as one element, but `append` flattens it correctly since each element is already an `X`. The two non-flat shapes emerge only when the per-instance value is itself a container: a `list[X]` per instance lands `list[list[X]]` (use `concat_flatten`), and a `dict[str, X]` per instance lands `list[dict]` (use `merge_all`).

If a parent field is populated by BOTH direct node writes AND fan-out collection, that's an architectural ambiguity worth fixing upstream — split into two fields, or pick one path.

### `invoke(metadata=...)` for caller-supplied trace identifiers (tenant IDs, request IDs, feature flags)

Per spec observability §3.4 / proposal 0034, callers attach arbitrary key/value entries at `invoke()` time and the framework propagates them to every observability backend:

```python
await compiled.invoke(
    initial_state,
    metadata={"tenantId": "acme-corp", "requestId": "req-12345", "featureFlag": "v2-canary"},
)
```

The OTel observer emits each entry as an `openarmature.user.<key>` cross-cutting span attribute on every span (invocation, node, subgraph wrapper, fan-out instance, LLM provider). The Langfuse observer merges each entry as a top-level key into `trace.metadata` AND every observation's metadata. Backends that consume OTel attributes (Honeycomb, Datadog APM, HyperDX, Grafana Tempo) pick the entries up for free; backends with typed metadata fields (Langfuse) get them via the per-backend propagation rule.

Boundary validation runs synchronously: keys MUST NOT start with `openarmature.` or `gen_ai.` (reserved namespaces); values MUST be OTel-attribute-compatible scalars (`str` / `int` / `float` / `bool`) or homogeneous arrays of those. Violations raise `ValueError` before any work begins.

Mid-invocation augmentation via the public helper:

```python
from openarmature.observability import set_invocation_metadata

async def my_node(state: MyState) -> dict:
    set_invocation_metadata(productId=state.product_id)
    # subsequent spans (this node's completed, next node's started,
    # any LLM call inside, etc.) carry productId
    return {"score": await compute_score(state)}
```

The augmentation respects fan-out / parallel-branches per-instance scoping — each instance's augmentation lives in its own Context copy and doesn't leak to siblings. Sequential nodes in the same engine task see prior nodes' augmentations forward. The helper validates the same rules as the `invoke()` boundary.
