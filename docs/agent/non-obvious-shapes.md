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
