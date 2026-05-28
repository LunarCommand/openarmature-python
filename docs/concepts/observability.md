# Observability

Two complementary patterns:

- **The `trace` field pattern**: a typed list inside state that nodes
  append to. State-shaped history, accessible from inside the graph,
  visible in the final state. Falls out of existing primitives.
  Covered in [State and reducers](state-and-reducers.md).
- **Observer hooks**: out-of-band events delivered to external code,
  with full pre/post state snapshots, error context, and visibility
  across subgraph boundaries. The control-side equivalent of the
  data-side `trace` field. This page.

The two are complementary, not redundant. `trace` is what state itself
remembers. Observers are what external code sees as state changes.

## An observer is an async callable

```python
from openarmature.graph import NodeEvent


async def my_observer(event: NodeEvent) -> None:
    print(event.phase, event.step, event.namespace, event.node_name)
```

The matching Protocol is `Observer`:

```python
from openarmature.graph import Observer


class StructuredLogger:
    async def __call__(self, event: NodeEvent) -> None: ...


_: Observer = StructuredLogger()  # structural conformance check
```

## Two registration modes

**Graph-attached**: fires on every invocation until removed:

```python
compiled = builder.compile()
handle = compiled.attach_observer(my_observer)
# ...later
handle.remove()                 # idempotent
```

Changes to the registered set during a graph run don't take effect
until the next invocation. The in-flight observer set is fixed at
`invoke()` time.

**Invocation-scoped**: fires only for one specific run:

```python
final = await compiled.invoke(initial, observers=[request_logger])
```

Common pattern: graph-attached for global concerns (Sentry, metrics,
structured tracing); invocation-scoped for per-request concerns (a
request-ID closure, a per-call snapshot ring).

## The NodeEvent shape

```python
@dataclass(frozen=True)
class NodeEvent:
    node_name: str
    namespace: tuple[str, ...]
    step: int
    phase: Literal["started", "completed", "checkpoint_saved"]
    pre_state: State
    post_state: State | None
    error: RuntimeGraphError | None
    parent_states: tuple[State, ...]
    attempt_index: int = 0
    fan_out_index: int | None = None
    fan_out_config: FanOutEventConfig | None = None
    branch_name: str | None = None
```

A walk-through:

- **`phase`**: every node attempt produces a `started` / `completed`
  *pair*. The pair shares `step` and `pre_state`. `started` fires
  before the node body runs; `completed` fires after the reducer
  merge succeeds *and* the outgoing edge has been evaluated. A
  successful pair populates `post_state` on `completed`; a failed
  pair populates `error` on `completed`. **`started` events have
  neither `post_state` nor `error` populated.**

  `checkpoint_saved` is an additional optional phase: when a
  Checkpointer is attached, the engine emits one per successful save
  (post-`completed`, immediately after the save resolves).
  **Default observer subscriptions don't include `checkpoint_saved`**;
  opt in via `phases={"checkpoint_saved"}` when registering (or
  `phases=KNOWN_PHASES`, exported from `openarmature.graph`, to
  subscribe to every phase including `checkpoint_saved`).

- **`node_name`**: the node's local name in its immediate containing
  graph. For nested subgraphs, the inner name, NOT a qualified path.

- **`namespace`**: the qualified path of containing-graph node names
  + the current node's name, outermost-first. For a top-level node:
  `(node_name,)`. For a subgraph-internal node:
  `(outer_subgraph_node_name, inner_name)`. A *tuple of strings*;
  the framework keeps it as a tuple at the API boundary rather than
  joining with a delimiter, so node names can contain any characters
  without parsing ambiguity.

- **`step`**: monotonic counter starting at 0, scoped to one
  outermost invocation. Subgraph-internal nodes increment the same
  counter; subgraph events interleave with outer events. The
  `started`/`completed` pair for one attempt share the same step.

- **`pre_state`** / **`post_state`**: state the node received vs.
  state after the reducer merge. *Shape varies with namespace*: for
  a subgraph-internal node, both are subgraph-state instances, not
  the outer state.

- **`error`**: the wrapped runtime error on `completed` events that
  failed. `event.error.category` gives the canonical error category;
  `event.error.__cause__` gives the original exception. **Edge /
  routing errors land here too**; see *Routing errors and the
  completed event* below.

- **`parent_states`**: one snapshot per containing graph, outermost
  first. Empty tuple for outermost-graph events. Invariant:
  `len(parent_states) == len(namespace) - 1`.

- **`attempt_index`**: 0-based retry attempt counter. `0` for nodes
  not wrapped by retry middleware; `1+` for retries. Retry middleware
  may wrap transitively. A retry on a [parallel-branches
  branch](parallel-branches.md) or fan-out `instance_middleware`
  re-runs the whole subgraph; events from inner nodes carry the
  wrapping retry's attempt counter.

- **`fan_out_index`**: 0-based per-instance index for events inside
  a fan-out instance; `None` outside.

- **`fan_out_config`**: populated on `started` / `completed` events
  for the *fan-out node itself*, carrying the resolved
  `item_count` / `concurrency` / `error_policy` / `parent_node_name`.
  `None` on every other event.

- **`branch_name`**: populated on events from nodes inside a
  [parallel-branches branch](parallel-branches.md), carrying the
  branch's name as declared on the dispatcher. `None` outside.
  Independent of `fan_out_index`; both may be present simultaneously
  when a parallel-branches branch contains a fan-out (or a fan-out
  instance contains a parallel-branches node). The combination
  `(namespace, branch_name, fan_out_index, attempt_index, phase)`
  uniquely identifies each event source. On the OTel mapping
  side, an `openarmature.branch_name` span attribute is added in
  parallel to the existing `openarmature.node.fan_out_index`.

## Routing errors and the completed event

When a conditional edge raises or returns an invalid target:

1. The preceding node runs and its body returns successfully.
2. The reducer merge succeeds.
3. The engine evaluates the outgoing edge.
4. The edge fn raises (`EdgeException`) OR returns something that
   isn't a declared node name or `END` (`RoutingError`).
5. The engine populates that error into the preceding node's
   `completed` event and dispatches it, sharing the
   started/completed pair rather than synthesising a new event.

So edge / routing errors *do* land on a `NodeEvent`, on the
*preceding* node's `completed` event, with `error` populated and
`post_state` left `None`. Observers see the failure attributed to the
right node without a synthetic event.

## Subgraph events bubble up

A subgraph-attached observer sees its own internal node events
whenever the subgraph runs, directly OR as a subgraph inside a
parent. The parent's observers ALSO see those internal events.

Delivery order for an event from a subgraph-internal node:

```
outermost-graph-attached → ... → subgraph-attached → invocation-scoped
```

Within each level, registration order. The subgraph-as-node wrapper
itself does *not* generate its own event; it's transparent to
observers.

## Serial delivery

Observers receive events serially within a single outermost invocation:

- No two observers receive the same event concurrently.
- No observer sees event N+1 until every observer has finished N.

**Why not parallel?** Two reasons. Parallel observers' output
interleaves nondeterministically (log readers can't reconstruct
ordering), and multi-observer error semantics get fiddly
(first-error-wins? collected exceptions?). Serial keeps per-run
output deterministic and error handling trivial. If a single observer
needs internal parallelism it can `asyncio.gather` itself.

A slow observer holds back delivery of subsequent events to siblings.
Two responses: keep the slow exporter as one observer (it serializes
naturally), or push events to an internal queue and return fast.

## Async-from-graph delivery + drain()

The graph's execution loop dispatches events onto a per-invocation
queue and **does not await** observer processing. Event dispatch is
constant-time from the graph's perspective; observers can't slow
node execution down.

This means `await compiled.invoke(...)` returns when the graph
reaches END (or raises), regardless of whether the observer queue has
finished. For long-running services that's fine. For short-lived
processes (scripts, serverless, CLIs), events dispatched late in the
run may not be delivered before the process exits.

`drain()` waits until every dispatched event has been delivered and
returns a `DrainSummary` reporting the outcome:

```python
final = await compiled.invoke(initial)
summary = await compiled.drain()
# DrainSummary(undelivered_count=0, timeout_reached=False)
```

- Per-graph, not per-invoke. Drain awaits *all* prior invocations'
  queues.
- Snapshot at call time. Events from invocations started concurrently
  with `drain()` may or may not be included.
- Subgraph events are part of the parent. A parent drain covers every
  subgraph event from any of its invocations; no need to drain each
  subgraph separately.

If you forget `drain()` in a CLI, the symptom is an empty trace file
or missing log entries.

### Bounded drain (optional timeout)

`drain()` accepts an optional `timeout` parameter (non-negative
seconds) — `await compiled.drain(timeout=5.0)` bounds the wait at five
seconds. When the deadline fires, in-flight workers are cancelled
cleanly so the compiled graph stays usable for subsequent invocations
— partial delivery state from one drain does NOT leak into the next.

The returned `DrainSummary` carries:

- `timeout_reached: bool` — `True` only when the timeout actually
  fired. A drain that finishes before the deadline reports `False`.
- `undelivered_count: int` — events dispatched but not fully delivered
  to every subscribed observer before the deadline. Always `0` when
  `timeout_reached is False`.

Observers **should** be cancellation-safe (idempotent writes,
`try/finally` cleanup) so that interruption by drain timeout does not
leave partial side effects in an inconsistent state.

When to set a timeout: short-lived processes (CLIs, scripts,
serverless functions) where a misbehaving observer holding drain
indefinitely would stall process exit. Long-running services that
control their own lifecycle can leave the timeout off and let drain
wait for natural completion.

## Error isolation

An observer that raises:

- Does NOT propagate its exception to `invoke()`'s caller.
- Does NOT prevent other observers from receiving the same event.
- Does NOT prevent any observer from receiving subsequent events.

Failures are reported via `warnings.warn` (Python's channel for
non-fatal anomalies). A bad observer can't take down the system that's
calling it. The graph run is the source of truth; observability is a
side concern.

## correlation_id is a separate join key

Two identifiers travel with every invocation:

- **`invocation_id`**: unique per `invoke()` call. Identifies *this
  run*. Surfaced on `CheckpointRecord.invocation_id`, observer span
  attributes, log records.
- **`correlation_id`**: a cross-system identifier propagated via
  `ContextVar`. Multiple invocations related by a higher-level
  request (e.g., a parent run that spawns a subgraph via direct
  `await sub.invoke(...)`, or a user-request that drives several
  related graph runs) can share one `correlation_id` while each
  having its own `invocation_id`.

`correlation_id` is the load-bearing join key in the multi-backend
scenario: a Langfuse trace, an OTel trace, and a structured log all
end up with the same `correlation_id` even
though their `invocation_id`s differ. It's exported from the
`openarmature.observability` package as `current_correlation_id` /
`current_invocation_id` (and friends) for code that needs to thread
the IDs explicitly.

## Caller-supplied invocation metadata

`correlation_id` is one string; if you also need to attach
business-domain identifiers — tenant IDs, request IDs, feature
flags, A/B cohort labels — pass them as a structured mapping at
`invoke()` time:

```python
await compiled.invoke(
    initial_state,
    metadata={
        "tenantId": "acme-corp",
        "requestId": "req-12345",
        "featureFlag": "v2-canary",
        "seatCount": 42,
    },
)
```

Every observability backend picks the entries up:

- **OTel** emits each entry as an `openarmature.user.<key>`
  cross-cutting span attribute on every span — invocation, node,
  subgraph wrapper, fan-out instance, LLM provider, retry attempt.
  Backends that consume OTel attributes (Phoenix / Arize, Honeycomb,
  Datadog APM, HyperDX, Grafana Tempo, custom collectors) see them
  uniformly without per-backend wiring.
- **Langfuse** merges each entry as a top-level key into
  `trace.metadata` AND into every `observation.metadata`. The
  Langfuse UI filters on `metadata.<key>` directly, so dashboard
  queries like "show me all traces for `tenantId == acme-corp`"
  work without any custom dashboard config.

Validation runs at the `invoke()` boundary before any work begins.
Two rules:

- **Keys** MUST NOT start with `openarmature.` or `gen_ai.`
  (reserved for spec-normative attribute namespaces; collisions
  would silently overwrite OA-emitted state).
- **Values** MUST be OTel-attribute-compatible scalars (`str`,
  `int`, `float`, `bool`) or homogeneous arrays of those types.
  `None`, nested objects, and mixed-type arrays are rejected.

Violations raise `ValueError` synchronously — no spans emitted, no
work runs.

### Adding entries mid-invocation

From inside a node body, middleware, or observer, augment the
in-scope metadata via the public helper:

```python
from openarmature.observability import set_invocation_metadata

async def evaluate_product(state: PipelineState) -> dict[str, Any]:
    set_invocation_metadata(productId=state.product_id, productCategory=state.category)
    # Spans emitted AFTER this call carry productId + productCategory
    # in addition to whatever the original invoke() metadata supplied.
    response = await provider.complete(messages)
    return {"score": parse_score(response.message.content)}
```

Spans already closed are NOT retroactively updated. Spans emitted
after the call (the current node's `completed` event, the next
node's `started`, any LLM call inside) pick up the new entries.

**Per-async-context scoping.** The metadata mapping lives in a
`ContextVar`, which Python copies on async-task creation. Fan-out
instances and parallel-branches each receive their own copy at
dispatch time — an instance that calls `set_invocation_metadata`
does NOT leak its augmentation to sibling instances. This is the
canonical pattern for per-instance identifiers:

```python
# Each fan-out instance adds its own productId; siblings stay clean
async def evaluate_product(state: ProductState) -> dict[str, Any]:
    set_invocation_metadata(productId=state.product_id)
    return await score_product(state)
```

Augmentation within the parent context (before fan-out dispatch, or
in code that runs serially) flows forward to subsequent spans in
that context, per normal `ContextVar` semantics.

### Reading the in-scope metadata

`openarmature.observability.current_invocation_metadata()` returns
the live mapping (or an empty `MappingProxyType` outside an
invocation). Observers and capability code read this to surface
the entries on backend-specific records; user code typically uses
`set_invocation_metadata` to write and lets the framework propagate.

## OpenTelemetry mapping (opt-in)

Install with the `[otel]` extra:

```bash
pip install 'openarmature[otel]'
```

`OTelObserver` maps node events to OTel spans + structured log
correlation:

- Each node `started` / `completed` pair becomes one span.
- Subgraph hierarchy is reflected in span parent-child structure.
- Spec error categories map to OTel `Status.ERROR` with semantic
  attributes.
- Log records emitted during node execution carry the active span's
  `trace_id` / `span_id` plus an `openarmature.correlation_id`
  attribute, so the join key survives the OTel boundary.

### TracerProvider isolation

`OTelObserver` constructs a **private** `TracerProvider` from the
processor you supply. It never registers globally and never reads
`get_tracer_provider()`. This isolation is intentional.

The motivation is concrete: many production stacks already register a
global `TracerProvider` (Langfuse v3's OpenInference integration is
the recurring example) for their own instrumentation. If openarmature
piggybacked on the global provider, every span the engine emits would
also flow to those other backends, doubling exports, corrupting
hierarchies, and tying openarmature's lifecycle to whichever
unrelated library happened to register first. Isolation prevents
that; the observer's spans only flow through the processor you handed
it.

### Detached trace mode

Some subgraphs or fan-outs are better as their own root trace than as
descendants of the parent's span tree: long-running asynchronous
work, retries that would balloon a parent span, or work that gets
reported to a different backend.

Configure detachment on the observer:

```python
obs = OTelObserver(
    processor=processor,
    detached_subgraphs=frozenset({"long_async_step"}),
    detached_fan_outs=frozenset({"daily_batch"}),
)
```

A detached subgraph or fan-out gets a fresh trace root (new
`trace_id`); the `correlation_id` still propagates through, so
join semantics survive even when trace boundaries don't.

The non-detached default is what you want most of the time: one
trace per outermost invocation, with subgraphs and fan-out instances
as nested spans.

### LLM provider spans

When an `OpenAIProvider` (or any [custom Provider](../model-providers/authoring.md)
that wires the dispatch hook) is used inside a graph with `OTelObserver`
attached, each `provider.complete()` call emits a dedicated span named
`openarmature.llm.complete`, parented under the calling node's span.
The span carries two attribute families.

**`openarmature.llm.*` (always on).** The framework's canonical
namespace: model identifier, finish reason, token counts, prompt
identity from `with_active_prompt(...)`, error category on failure.
Set unconditionally whenever the LLM span itself emits.

**`gen_ai.*` (OpenTelemetry GenAI semantic conventions, default on).**
Cross-vendor attribute names every LLM-aware backend reads
(Langfuse, Phoenix, Honeycomb's LLM lens, OpenInference-aware
tools). Emitted alongside the OA namespace:

- `gen_ai.system` — `"openai"` by default; override per provider
  instance to `"vllm"` / `"lm_studio"` / `"llama_cpp"` / etc. when
  the OpenAI Chat Completions wire format is hitting a non-OpenAI
  endpoint:

  ```python
  provider = OpenAIProvider(
      base_url="http://vllm.internal:8000",
      model="meta-llama/Llama-3-8B-Instruct",
      genai_system="vllm",
  )
  ```

- `gen_ai.request.model` / `gen_ai.response.model` — the bound
  model and (when the provider returns one) the more-specific
  identifier in the response body.
- `gen_ai.request.temperature` / `max_tokens` / `top_p` / `seed` /
  `frequency_penalty` / `presence_penalty` / `stop_sequences` —
  only emitted for fields the caller actually set; absence on
  the span means "not supplied," distinct from a zero value.
- `gen_ai.usage.input_tokens` / `output_tokens` — token counts.
- `gen_ai.response.finish_reasons` — single-element string array.
- `gen_ai.response.id` — when the provider returns one.

Disable the GenAI semconv set with `OTelObserver(disable_genai_semconv=True)`
when an external auto-instrumentation library (OpenInference,
`opentelemetry-instrumentation-openai`) is already the canonical
source on your stack.

### LLM payload attributes

By default, LLM spans do **not** carry the messages sent or the
response content. Opt in with `disable_llm_payload=False`:

```python
observer = OTelObserver(
    span_processor=SimpleSpanProcessor(exporter),
    disable_llm_payload=False,
)
```

This surfaces three attributes:

- `openarmature.llm.input.messages` — JSON-encoded message array
  (the spec §3 message shape: `{role, content, tool_calls?, …}`).
- `openarmature.llm.output.content` — the assistant's response
  content string verbatim. Omitted for tool-call-only responses
  with empty content.
- `openarmature.llm.request.extras` — JSON-encoded `RuntimeConfig`
  extras bag (provider-specific pass-through fields like
  `repetition_penalty` for vLLM, or `top_k` for HuggingFace
  endpoints). Omitted when empty.

**Default-off is deliberate.** The payload may contain PII the user
hasn't audited; opting in is a separate decision from opting into
observability. The flag name keeps symmetry with `disable_llm_spans`:
the default value (`True`) reads as "the observer disables payload
emission by default."

#### Truncation

Each payload attribute is capped at `payload_max_bytes` UTF-8 bytes
(default 64 KiB, minimum 256). When the serialized value exceeds the
cap, the observer emits the largest UTF-8-code-point-aligned prefix
that fits within `cap - len(marker)` bytes followed by the marker:

```
…[truncated, M bytes total]
```

where M is the pre-truncation byte length. The marker is appended
outside any JSON encoding — a truncated attribute is *not* parseable
JSON, which is the clean signal backend code can use to detect
truncation without a separate flag.

#### Inline image redaction (always on)

Image content blocks with `ImageSourceInline` are redacted at the
provider, *before* the payload reaches the observer:

```json
{
  "type": "image",
  "source": {"type": "inline_redacted", "byte_count": 4096},
  "media_type": "image/png",
  "detail": "auto"
}
```

The `media_type` and `detail` fields are preserved at the image-block
level (per llm-provider §3.1.2); only `source` is replaced. URL-form
images pass through unchanged — the URL is a short string and is
informative for trace readers.

Redaction is **not** gated by `disable_llm_payload` and is **not**
configurable. Inline image bytes never leave the provider in event
form, so custom observers consuming
[`LlmEventPayload`](#publishing-llm-events-for-custom-observers)
cannot accidentally leak raw bytes regardless of how they're
written.

### Identifying the service: `Resource`

Pass an `opentelemetry.sdk.resources.Resource` to set
`service.name` / `service.version` / etc. without relying on the
`OTEL_SERVICE_NAME` / `OTEL_RESOURCE_ATTRIBUTES` environment
variables (which had to be set *before* `OTelObserver()`
construction to take effect):

```python
from opentelemetry.sdk.resources import Resource

observer = OTelObserver(
    span_processor=SimpleSpanProcessor(exporter),
    resource=Resource.create({"service.name": "claims-pipeline"}),
)
```

### Fanning out to multiple backends

The `span_processor` argument accepts either a single processor or
a sequence. Multi-destination export (HyperDX + Langfuse from one
observer) is a one-line construct:

```python
observer = OTelObserver(
    span_processor=[
        BatchSpanProcessor(OTLPSpanExporter(endpoint=HYPERDX_URL)),
        BatchSpanProcessor(OTLPSpanExporter(endpoint=LANGFUSE_URL)),
    ],
)
```

Every registered processor receives every span.

### Adding backend-specific attributes: `attribute_enrichers`

When a backend needs attributes the framework doesn't emit
(custom `langfuse.observation.*` keys, Honeycomb derived fields,
etc.), the `attribute_enrichers` hook fires just before every
`span.end()` call:

```python
def langfuse_observation_kind(span, event):
    if span.name == "openarmature.llm.complete":
        span.set_attribute("langfuse.observation.type", "generation")

observer = OTelObserver(
    span_processor=processor,
    attribute_enrichers=[langfuse_observation_kind],
)
```

Each enricher receives the live `Span` plus the `NodeEvent` that
triggered the close (or `None` on synthetic close sites — subgraph
dispatch, detached root, fan-out instance, invocation span,
shutdown drain). Setting attributes inside this hook works
correctly; doing it from a `SpanProcessor.on_end` callback does
not, because the framework has already called `span.end()` and the
OTel SDK silently drops `set_attribute` on ended spans.

Exceptions raised by an enricher are caught and warned, never
propagated.

### Publishing LLM events for custom observers

`openarmature.observability.LLM_NAMESPACE` and
`openarmature.observability.LlmEventPayload` are part of the public
API. A custom observer subscribing to the dispatch stream can
recognize the LLM-event sentinel namespace and read the typed
payload directly:

```python
from openarmature.observability import LLM_NAMESPACE, LlmEventPayload

async def my_llm_observer(event):
    if event.namespace != LLM_NAMESPACE:
        return
    payload = event.pre_state
    if not isinstance(payload, LlmEventPayload):
        return
    # payload.model, payload.input_messages (already image-redacted),
    # payload.output_content, payload.request_params,
    # payload.response_id, payload.active_prompt, ...
```

A custom `Provider` that wants to participate in the same span
emission protocol dispatches `NodeEvent(namespace=LLM_NAMESPACE,
pre_state=LlmEventPayload(...))` via `current_dispatch()`. See
[Authoring providers](../model-providers/authoring.md) for the
full pattern.

### Flushing under fast teardown

`OTelObserver.shutdown()` calls `provider.shutdown()` on the private
`TracerProvider`, which per OTel SDK contract flushes every
registered span processor. Under unusual teardown orderings — for
example, FastAPI's `TestClient` teardown that closes the event loop
before a `BatchSpanProcessor`'s export thread finishes — spans can
appear dropped. Two workarounds:

- Call `observer._provider.force_flush(timeout_millis=...)`
  explicitly before `shutdown()`.
- Use `SimpleSpanProcessor` instead of `BatchSpanProcessor` in
  tests; it exports synchronously and is unaffected by teardown
  timing.

## Langfuse mapping (opt-in)

A second sibling observer maps the same `NodeEvent` stream onto
Langfuse's native Trace + Observation data model — Traces at the
top, Span observations for graph nodes, Generation observations for
LLM calls. Use it instead of (or alongside) the OTel observer when
your trace UI is Langfuse and you want first-class Generation
rendering without going through Langfuse's OTLP ingest.

```python
from openarmature.observability.langfuse import (
    InMemoryLangfuseClient,
    LangfuseObserver,
)

client = InMemoryLangfuseClient()  # or langfuse.Langfuse(...) in prod
observer = LangfuseObserver(client=client)
graph.attach_observer(observer)
```

The `client` is anything matching the `LangfuseClient` Protocol —
the bundled `InMemoryLangfuseClient` (used by the conformance
harness, useful for unit tests), or a real `langfuse.Langfuse()`
instance wrapped in `LangfuseSDKAdapter` for production. Install
the optional extras to bring in the Langfuse SDK:

```bash
pip install 'openarmature[langfuse]'
```

Production wire-up:

```python
from langfuse import Langfuse
from openarmature.observability.langfuse import (
    LangfuseObserver,
    LangfuseSDKAdapter,
)

langfuse_client = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
)
observer = LangfuseObserver(
    client=LangfuseSDKAdapter(langfuse_client),
    disable_llm_payload=False,
)
```

The adapter bridges `langfuse>=4.6`'s unified `start_observation`
API onto our `LangfuseClient` Protocol; the observer code is the
same in tests and production. See
[`examples/10-langfuse-observability`](../examples/10-langfuse-observability.md)
for a runnable demo.

!!! note "Langfuse SDK version compatibility"

    Validated against `langfuse>=4.6,<5`. The v4 SDK introduced an
    OTel-based architecture with `start_observation` /
    `propagate_attributes` replacing the v2/v3 `trace` / `span` /
    `generation` low-level API; the bundled `LangfuseSDKAdapter`
    handles the bridge so the observer surface is stable across
    future v4 patches.

    Earlier SDK versions (v2.x, v3.x) are NOT supported. Projects on
    those versions either upgrade to v4 or supply their own adapter
    matching the `LangfuseClient` Protocol's four methods.

    A runtime `isinstance(adapter, LangfuseClient)` check ships in
    the unit suite — if a future v4 patch breaks the Protocol's
    surface, the test fails loudly.

### What Langfuse sees

- **Trace ID = invocation ID.** The Trace's `id` is the OA
  `invocation_id` verbatim, so cross-system lookup by invocation_id
  finds the Langfuse Trace directly (spec §8.4.1).
- **Trace name.** Defaults to the entry-node name (spec §8.6
  fallback). Caller-supplied invocation labels land in PR 4
  (proposal 0034).
- **Per-observation metadata.** Each Span / Generation carries
  `namespace`, `step`, `attempt_index`, optional `fan_out_index` /
  `branch_name`, and the `correlation_id` cross-cutting join key
  (spec §8.5).
- **Generation fields.** LLM calls become Generation observations
  with `model`, `model_parameters` (the `gen_ai.request.*` request
  parameters lifted by inclusion per §8.4.3), `usage` (input /
  output / total tokens), and `metadata.finish_reason` /
  `system` / `response_model` / `response_id`.

### Payload + truncation

`disable_llm_payload` mirrors the OTel observer's flag — defaults
to `True` for the same privacy reason. Flip to `False` to populate
`generation.input` / `output` / `metadata.request_extras` from the
LLM event payload.

```python
observer = LangfuseObserver(
    client=client,
    disable_llm_payload=False,
    payload_byte_cap=65536,
)
```

When a payload exceeds `payload_byte_cap`, the observer emits the
serialized form with the §5.5.5 truncation marker
(`…[truncated, M bytes total]`) verbatim as a raw string instead of
parsing back to native shape. The unparseable JSON IS the
truncation signal in the Langfuse UI.

### Prompt linkage

When a Prompt's source backend exposes a Langfuse Prompt entity
reference under `Prompt.observability_entities['langfuse_prompt']`,
the Generation observation links to that entity natively (spec
§8.4.4 case 1). Backends that don't surface a Langfuse reference
(filesystem, in-memory, etc.) leave the Generation with
`metadata.prompt` populated but no entity link (case 2).

### Composition with OTel

The two observers are independent §6 event consumers and can be
attached together. They share the `correlation_id` as the
cross-backend join key — find a slow Generation in Langfuse, search
for its `correlation_id` in OTel logs, see the surrounding
infrastructure activity.

```python
otel_observer = OTelObserver(span_processor=...)
langfuse_observer = LangfuseObserver(client=langfuse_client)
graph.attach_observer(otel_observer)
graph.attach_observer(langfuse_observer)
```

Each observer's `disable_llm_spans` / `disable_llm_payload` flag is
independent; one MAY emit while the other suppresses.
