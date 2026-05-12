# Observability

Two complementary patterns:

- **The `trace` field pattern** — a typed list inside state that nodes
  append to. State-shaped history, accessible from inside the graph,
  visible in the final state. Falls out of existing primitives.
  Covered in [State and reducers](state-and-reducers.md).
- **Observer hooks** — out-of-band events delivered to external code,
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

**Graph-attached** — fires on every invocation until removed:

```python
compiled = builder.compile()
handle = compiled.attach_observer(my_observer)
# ...later
handle.remove()                 # idempotent
```

Changes to the registered set during a graph run don't take effect
until the next invocation — the in-flight observer set is fixed at
`invoke()` time.

**Invocation-scoped** — fires only for one specific run:

```python
final = await compiled.invoke(initial, observers=[request_logger])
```

Common pattern: graph-attached for global concerns (Sentry, metrics,
structured tracing); invocation-scoped for per-request concerns (a
request-ID closure, a per-call snapshot ring).

## The NodeEvent shape (spec v0.6.0)

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
```

A walk-through:

- **`phase`** — every node attempt produces a `started` / `completed`
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

- **`node_name`** — the node's local name in its immediate containing
  graph. For nested subgraphs, the inner name, NOT a qualified path.

- **`namespace`** — the qualified path of containing-graph node names
  + the current node's name, outermost-first. For a top-level node:
  `(node_name,)`. For a subgraph-internal node:
  `(outer_subgraph_node_name, inner_name)`. A *tuple of strings* —
  per spec, implementations MUST NOT join with a delimiter at the
  API boundary, so node names may contain any characters.

- **`step`** — monotonic counter starting at 0, scoped to one
  outermost invocation. Subgraph-internal nodes increment the same
  counter; subgraph events interleave with outer events. The
  `started`/`completed` pair for one attempt share the same step.

- **`pre_state`** / **`post_state`** — state the node received vs.
  state after the reducer merge. *Shape varies with namespace*: for
  a subgraph-internal node, both are subgraph-state instances, not
  the outer state.

- **`error`** — the wrapped runtime error on `completed` events that
  failed. `event.error.category` gives the spec category;
  `event.error.__cause__` gives the original exception. **Edge /
  routing errors land here too** — see *Routing errors and the
  completed event* below.

- **`parent_states`** — one snapshot per containing graph, outermost
  first. Empty tuple for outermost-graph events. Invariant:
  `len(parent_states) == len(namespace) - 1`.

- **`attempt_index`** — 0-based retry attempt counter. `0` for nodes
  not wrapped by retry middleware; `1+` for retries.

- **`fan_out_index`** — 0-based per-instance index for events inside
  a fan-out instance; `None` outside.

- **`fan_out_config`** — populated on `started` / `completed` events
  for the *fan-out node itself*, carrying the resolved
  `item_count` / `concurrency` / `error_policy` / `parent_node_name`
  (spec proposal 0013, v0.10.0). `None` on every other event.

## Routing errors and the completed event

When a conditional edge raises or returns an invalid target:

1. The preceding node runs and its body returns successfully.
2. The reducer merge succeeds.
3. The engine evaluates the outgoing edge.
4. The edge fn raises (`EdgeException`) OR returns something that
   isn't a declared node name or `END` (`RoutingError`).
5. The engine populates that error into the preceding node's
   `completed` event and dispatches it — sharing the
   started/completed pair rather than synthesising a new event.

So edge / routing errors *do* land on a `NodeEvent` — on the
*preceding* node's `completed` event, with `error` populated and
`post_state` left `None`. Observers see the failure attributed to the
right node without a synthetic event.

## Subgraph events bubble up

A subgraph-attached observer sees its own internal node events
whenever the subgraph runs — directly OR as a subgraph inside a
parent. The parent's observers ALSO see those internal events.

Delivery order for an event from a subgraph-internal node:

```
outermost-graph-attached → ... → subgraph-attached → invocation-scoped
```

Within each level, registration order. The subgraph-as-node wrapper
itself does *not* generate its own event — it's transparent to
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
constant-time from the graph's perspective — observers can't slow
node execution down.

This means `await compiled.invoke(...)` returns when the graph
reaches END (or raises), regardless of whether the observer queue has
finished. For long-running services that's fine. For short-lived
processes (scripts, serverless, CLIs), events dispatched late in the
run may not be delivered before the process exits.

`drain()` blocks until every dispatched event has been delivered:

```python
final = await compiled.invoke(initial)
await compiled.drain()
```

- Per-graph, not per-invoke. Drain awaits *all* prior invocations'
  queues.
- Snapshot at call time. Events from invocations started concurrently
  with `drain()` may or may not be included.
- Subgraph events are part of the parent. A parent drain covers every
  subgraph event from any of its invocations — no need to drain each
  subgraph separately.

If you forget `drain()` in a CLI, the symptom is an empty trace file
or missing log entries.

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

- **`invocation_id`** — unique per `invoke()` call. Identifies *this
  run*. Surfaced on `CheckpointRecord.invocation_id`, observer span
  attributes, log records.
- **`correlation_id`** — a cross-system identifier propagated via
  `ContextVar`. Multiple invocations related by a higher-level
  request (e.g., a parent run that spawns a subgraph via direct
  `await sub.invoke(...)`, or a user-request that drives several
  related graph runs) can share one `correlation_id` while each
  having its own `invocation_id`.

`correlation_id` is the load-bearing join key in the multi-backend
scenario the charter calls out: a Langfuse trace, an OTel trace, and
a structured log all end up with the same `correlation_id` even
though their `invocation_id`s differ. It's exported from the
`openarmature.observability` package as `current_correlation_id` /
`current_invocation_id` (and friends) for code that needs to thread
the IDs explicitly.

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
processor you supply — it never registers globally and never reads
`get_tracer_provider()`. This is mandated by spec observability §6.

The motivation is concrete: many production stacks already register a
global `TracerProvider` (Langfuse v3's OpenInference integration is
the recurring example) for their own instrumentation. If openarmature
piggybacked on the global provider, every span the engine emits would
also flow to those other backends — doubling exports, corrupting
hierarchies, and tying openarmature's lifecycle to whichever
unrelated library happened to register first. Isolation prevents
that; the observer's spans only flow through the processor you handed
it.

### Detached trace mode

Some subgraphs or fan-outs are better as their own root trace than as
descendants of the parent's span tree — long-running asynchronous
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

The non-detached default is what you want most of the time — one
trace per outermost invocation, with subgraphs and fan-out instances
as nested spans.
