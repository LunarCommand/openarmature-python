# Observability

Two complementary patterns:

- **The `trace` field pattern** — a typed list inside state that nodes
  append to. State-shaped history, accessible from inside the graph,
  visible in the final state. Falls out of existing primitives (State
  + `append` reducer). Covered in [State and reducers](state-and-reducers.md).
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
    print(event.step, event.namespace, event.node_name)
```

The matching Protocol is `Observer`:

```python
from openarmature.graph import Observer


class StructuredLogger:
    async def __call__(self, event: NodeEvent) -> None:
        ...


_: Observer = StructuredLogger()  # structural conformance check
```

**Why async?** Observers typically do IO — write spans to a tracing
backend, push metrics, append to a log. Async lets the delivery loop
coordinate ordering across observers without thread machinery.

## Two registration modes

**Graph-attached** — fires on every invocation until removed:

```python
compiled = builder.compile()
handle = compiled.attach_observer(my_observer)
# ... later
handle.remove()                 # idempotent
```

`attach_observer` returns a `RemoveHandle`. Removal is idempotent;
double-removing is safe. Changes to the registered set during a graph
run don't take effect until the next invocation — the in-flight
observer set is fixed at `invoke()` time.

**Invocation-scoped** — fires only for one specific run:

```python
final = await compiled.invoke(initial, observers=[request_logger])
```

Common pattern: graph-attached for global concerns (Sentry, metrics,
structured tracing); invocation-scoped for per-request concerns (a
request-ID closure, a per-call snapshot ring).

## The NodeEvent shape

A `NodeEvent` is a frozen dataclass with these fields:

- **`node_name`** — the name as registered in this node's *immediate*
  containing graph. For nested subgraphs, it's the inner name, not a
  qualified path.
- **`namespace`** — the qualified path: outermost graph's node name(s)
  down to this node. For top-level: `(node_name,)`. For a
  subgraph-internal node: `(outer_subgraph_node_name, inner_name)`.
  A *tuple of strings* — the spec is explicit that implementations MUST
  NOT join with a delimiter at the API boundary, so node names can
  contain any characters without parsing ambiguity.
- **`step`** — monotonic counter starting at 0, scoped to one outermost
  invocation. Subgraph-internal nodes increment the same counter —
  subgraph events interleave with outer events.
- **`pre_state`** / **`post_state`** — state the node received and
  state after its update merged. *Shape varies with namespace:* for a
  subgraph-internal node, both are subgraph-state instances, not the
  outer state.
- **`error`** — populated only when the node failed. Carries the
  wrapped runtime error; read `event.error.category` for the spec
  category and `event.error.__cause__` for the original exception.
- **`parent_states`** — one snapshot per containing graph, outermost
  first. Empty tuple for outermost-graph events. Invariant:
  `len(parent_states) == len(namespace) - 1`.

**Exactly one of `post_state` or `error` is populated per event.** That's
how an observer distinguishes success from failure.

**Routing errors don't get their own event.** They arise from edge
evaluation, after the prior node's event has already been dispatched.
A routing failure surfaces through the runtime exception path but
doesn't produce a `NodeEvent`.

## Subgraph events bubble up

A subgraph-attached observer sees its own internal node events
whenever the subgraph runs — whether invoked directly OR as a subgraph
inside a parent.

- Subgraph runs alone (`subgraph.invoke(...)`) → subgraph-attached
  observer sees its internal events.
- Subgraph runs inside a parent → the parent's own observers ALSO see
  those internal events, *plus* the subgraph-attached observer still
  sees them.

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

**Why not parallel (`asyncio.gather`)?** Two reasons. First,
parallel observers' output interleaves nondeterministically (log
readers can't reconstruct ordering). Second, multi-observer error
semantics get fiddly (first-error-wins? collected exceptions?). Serial
keeps per-run output deterministic and error handling trivial. If a
single observer needs internal parallelism, it can `asyncio.gather`
itself.

**The consequence:** a slow observer holds back delivery of subsequent
events to siblings. If you're integrating with an exporter that takes
80ms per event, observers behind it queue up. Two responses: keep the
slow exporter as one observer (it serializes naturally), or have the
slow observer push events to its own internal queue and return fast
(decoupling delivery from export).

## Async-from-graph delivery

The graph's execution loop dispatches events onto a per-invocation
queue and **does not await** observer processing. Event dispatch is
constant-time from the graph's perspective — no observer can slow node
execution down. The observer queue runs concurrently as a background
task on the same event loop.

This means `await compiled.invoke(...)` returns when the graph reaches
END (or raises), regardless of whether the observer queue has
finished. For long-running services this is fine — the queue keeps
draining.

For short-lived processes (scripts, serverless, CLIs), events
dispatched late in the run may not be delivered before the process
exits. That's what `drain` is for.

## drain() for short-lived processes

```python
final = await compiled.invoke(initial)
await compiled.drain()
```

`drain()` returns once every event dispatched by prior invocations of
this graph has been delivered to every registered observer.

Three things to know:

1. **Per-graph, not per-invoke.** Drain awaits *all* prior
   invocations' queues, not just the most recent. Multiple `invoke()`
   calls in flight are all covered by one `drain()`.
2. **Snapshot at call time.** Events from invocations started
   concurrently with `drain()` may or may not be included.
3. **Subgraph events are part of the parent.** A parent drain covers
   every subgraph event that was part of any of its invocations — no
   need to drain each subgraph separately.

If you forget `drain()` in a CLI, the symptom is an empty trace file
or missing log entries — events were dispatched but the process exited
before the queue worker could deliver them.

## Error isolation

An observer that raises:

- Does NOT propagate its exception to `invoke()`'s caller.
- Does NOT prevent other observers from receiving the same event.
- Does NOT prevent any observer from receiving subsequent events.

The exception is reported via `warnings.warn` (Python's standard
channel for non-fatal anomalies). Production code that needs
structured handling of observer failures can install a `warnings`
filter or wrap each observer with its own try/except.

The contract is intentionally unforgiving of observer bugs. The graph
run is the source of truth; observability is a side concern. A bad
observer can't take down the system that's calling it.

## OpenTelemetry mapping (opt-in)

Install with the `[otel]` extra:

```bash
pip install 'openarmature[otel]'
```

The OTel observer maps node events to OTel spans + structured log
correlation:

- Each node execution becomes a span.
- Subgraph hierarchy is reflected in span parent-child structure.
- Spec error categories map to OTel `Status.ERROR` with semantic
  attributes.
- Log records emitted during node execution carry the active span's
  `trace_id` / `span_id` + an `openarmature.correlation_id` attribute.

The mapping uses spec-defined attribute names (`openarmature.node.name`,
`openarmature.invocation_id`, etc.) so any OTel backend (Honeycomb,
Tempo, Jaeger, the OTel collector) can render the trees correctly.

The OTel observer is opt-in by extra; the core library has no OTel
dependency, so projects that don't want OTel pay nothing for it.
