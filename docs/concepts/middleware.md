# Middleware

Middleware wraps the dispatch of a single node. The shape is an async
callable `(state, next) -> partial_update`. Anything you want to happen
around a node, without changing the node itself, lives here: retries,
timing, structured logging, request enrichment, error transformation,
circuit-breaking.

```python
from collections.abc import Mapping
from typing import Any

from openarmature.graph import Middleware, NextCall


class LogAround:
    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        print("before")
        partial = await next_(state)
        print("after")
        return partial


_: Middleware = LogAround()  # structural conformance check
```

`next` invokes the next layer of the chain (or the wrapped node, at
the innermost end) and returns the partial update from that layer.
Code before `await next(state)` is the pre-node phase (runs on the way
in); code after is the post-node phase (runs on the way out).

## Four registration sites

You can attach middleware at four places. The same `Middleware` shape
works in all of them.

**Per-node**, on a single function node:

```python
builder.add_node("fetch", fetch_fn, middleware=[RetryMiddleware()])
```

**Per-graph**, applied to every node in the graph:

```python
builder.add_middleware(TimingMiddleware(node_name="...", on_complete=record))
```

**Per-branch**, on a single branch of a parallel-branches node:

```python
from openarmature.graph import BranchSpec

branches = {
    "sentiment": BranchSpec(
        subgraph=sentiment_subgraph,
        middleware=(RetryMiddleware(),),
    ),
    "topic": BranchSpec(subgraph=topic_subgraph),
}
builder.add_parallel_branches_node("classify", branches=branches)
```

The branch middleware wraps the whole branch dispatch as one call. A
retry on a branch retries the entire branch from scratch, not an
individual node inside it.

**Per-fan-out-instance**, on the instance dispatch inside a fan-out
node:

```python
builder.add_fan_out_node(
    "summarize",
    subgraph=summarize_subgraph,
    items_field="articles",
    item_field="article",
    collect_field="article",
    target_field="summaries",
    instance_middleware=[RetryMiddleware()],
)
```

A retry here retries one instance, not the whole fan-out.

## Composition order

When a node has middleware from multiple sites, per-graph composes
*outside* per-node. The runtime chain at a single function node is:

```
[per_graph_outer_to_inner...] → [per_node_outer_to_inner...] → node
```

The first middleware in `builder.add_middleware()` calls is the
outermost layer; the last is closest to the node. Same rule for
per-node: list order is outer-to-inner.

## The subgraph boundary

Middleware does not cross into a subgraph. The parent's middleware
wraps the `SubgraphNode` dispatch as a single atomic call, and the
subgraph's own middleware (configured on the child builder) wraps the
child's internal nodes independently.

In practical terms: a `RetryMiddleware` on a subgraph-as-node retries
the whole child graph from its entry. A `RetryMiddleware` inside the
child retries one of its individual nodes.

## Error semantics

An exception raised by `next(state)` propagates up through `await
next(state)`. Middleware may:

- **Re-raise**: the simplest case. Don't catch, let it bubble.
- **Catch and recover**: catch the exception and return a partial
  update of your own. The rest of the chain continues as if the node
  had returned that partial update normally.
- **Catch and transform**: catch one exception type, raise a different
  one. The new exception propagates up.
- **Call `next` more than once**: this is what retry middleware does.

A middleware MUST NOT mutate the input `state` object in place. To
hand a transformed state down the chain, pass a new state instance to
`next(...)`.

## Built-in: RetryMiddleware

```python
from openarmature.graph import RetryConfig, RetryMiddleware, exponential_jitter_backoff


async def on_retry(exc: Exception, attempt: int) -> None:
    log.warning("retrying after %r (attempt %d)", exc, attempt)


retry = RetryMiddleware(
    RetryConfig(
        max_attempts=3,
        backoff=exponential_jitter_backoff,
        on_retry=on_retry,
    )
)
```

Configured with a `RetryConfig`; four fields, all optional:

- **`max_attempts`** is the total attempt count including the first
  call. `1` disables retry. Default `3`.
- **`classifier`** is a predicate `(exception, state) -> bool`.
  The default (`default_classifier`, importable from
  `openarmature.graph`) treats any exception with a `category`
  attribute matching the project's `TRANSIENT_CATEGORIES` set as
  transient. To retry on additional types, write a classifier that
  delegates to `default_classifier` and falls back to your own check.
- **`backoff`** is a callable `(attempt_index) -> seconds`. The default
  is exponential with jitter (base 1s, cap 30s, full jitter).
  `deterministic_backoff(seconds)` is provided for tests.
- **`on_retry`** is an optional async callback `(exception, attempt)
  -> None`. Fires before each sleep. Useful for emitting a structured
  "about to retry" event.

A retry's attempt counter propagates as a context variable to every
node event emitted from within the retry, including nodes inside
subgraphs and branches that the retry wraps transitively. So an
observer logging a retried node sees `attempt=1`, `attempt=2`, etc. on
the inner events.

## Built-in: TimingMiddleware

```python
from openarmature.graph import TimingMiddleware, TimingRecord


async def record(rec: TimingRecord) -> None:
    metrics.histogram("node_duration_ms", rec.duration_ms, tags={
        "node": rec.node_name,
        "outcome": rec.outcome,
    })


builder.add_node(
    "fetch",
    fetch_fn,
    middleware=[TimingMiddleware(node_name="fetch", on_complete=record)],
)
```

`TimingMiddleware` records the wrapped chain's duration with a
monotonic clock and delivers a `TimingRecord` to your async callback.
The record includes `node_name`, `duration_ms`, `outcome` (`"success"`
or `"exception"`), and `exception_category` (the failing exception's
`category` attribute when present).

Two implementation details worth knowing:

- The callback fires **inline** before the chain's result returns.
  Slow callbacks add to the apparent node duration. Keep them fast
  (queue work, defer I/O).
- The clock is injectable per instance via the `clock` kwarg.
  Test fixtures use this to supply a deterministic stub without
  globally patching `time.monotonic` (which would also distort
  asyncio's scheduling).

## Built-in: FailureIsolationMiddleware

```python
from openarmature.graph import FailureIsolationMiddleware

builder.add_node(
    "extract_segments",
    extract_fn,
    middleware=[
        FailureIsolationMiddleware(
            degraded_update={"segments": []},
            event_name="segment_extraction_degraded",
        ),
    ],
)
```

`FailureIsolationMiddleware` catches an exception escaping the wrapped
chain and returns a degraded partial update instead of letting it abort
the invocation. Reach for it when a node is not load-bearing enough to
kill the whole run: a failed enrichment step degrades to an empty list,
the graph continues, and the failure is still visible in your traces.
It is the named, observable form of the "catch and recover" pattern
from [Error semantics](#error-semantics) above.

Configuration:

- **`degraded_update`** (required) is the partial update returned on a
  caught exception. It may be a static mapping, or a callable
  `state -> partial_update` when the fallback shape depends on the input
  state. The callable is resolved once, at catch time.
- **`event_name`** (required, no default) is a stable identifier for
  this catch site. It rides on the emitted event (below) and any
  downstream logging. There is no default on purpose: a generic name
  like `"failure_isolated"` collapses every degraded path into one
  indistinguishable bucket in a dashboard, so the name is forced at the
  construction site, where the context to name it well is available.
- **`catch`** is an optional set of error categories. When supplied, an
  exception is caught only if the *derived category* of its cause chain
  is in the set: the category of the outermost non-carrier link that
  carries one, resolved *through* the engine's `node_exception` carriers (the same value the
  event reports as `caught_exception.category`). This is the recommended
  gate for category-scoped degradation. At a wrapping placement (a
  subgraph, a fan-out instance, a branch) the engine wraps the real
  failure in a carrier, so a check on the surface exception sees only the
  carrier and misses it; `catch` classifies through the carrier and
  matches the originating category. A bare uncategorized error has no
  derived category and is not matched, so it propagates.
- **`predicate`** is an optional `Exception -> bool` over the *surface*
  (caught) exception. When supplied, only exceptions where it returns true
  are caught; everything else propagates. The default is always-true. It
  composes with `catch` as a conjunction (both must admit), and both
  default permissive, so the both-unset default catches every
  `Exception`. Because `predicate` sees the surface exception, it
  misclassifies a carrier-wrapped failure at a wrapping placement; reach
  for `catch` for category gating, or classify the chain yourself with
  `classify_cause_chain` (below).
- **`on_caught`** is an optional async hook `Exception -> None`, fired
  when the middleware catches. Use it to pump the caught exception to
  caller-specific telemetry beyond the framework event. It fires inline
  before the degraded update returns, and an exception it raises is
  isolated (logged, not propagated) so a buggy hook cannot defeat the
  recovery.

Like `RetryMiddleware`, it catches `Exception` only; `BaseException`
(cancellation, keyboard interrupt) propagates so aborts still work.

### The failure-isolated event

On a catch, the middleware dispatches a `FailureIsolatedEvent` onto the
observer stream. It is a distinct event variant, not a node event: it
carries the `event_name`, the wrapped node's lineage identity, the input
and degraded states, and a `CaughtException` record. That record holds a
derived `category` (when the cause has one) and `message` for simple
consumers, plus a `chain` of cause links (`CauseLink`) from the caught
exception down to the originating raise, with graph-engine carrier
wrappers flagged so a consumer can skip them. Observers narrow on it with
`isinstance(event, FailureIsolatedEvent)`. The bundled OTel
and Langfuse observers render it as a marker span / observation so the
catch shows up alongside the node's own span. The default emission path
is the observer stream only, with no logging-library dependency;
`on_caught` is the escape hatch for anything else.

### Cause-chain classification

The walk behind `catch` and `caught_exception` is exposed as a public
primitive, `classify_cause_chain`, so any consumer classifies a
carrier-wrapped failure the same way the framework does:

```python
from openarmature.graph import classify_cause_chain

result = classify_cause_chain(exc)
result.category   # derived category (outermost non-carrier link with a category), or None
result.message    # the message that category came from
result.chain      # the ordered CauseLink chain, carriers flagged
```

It returns a `CaughtException` (the same record the failure-isolated
event's `caught_exception` field holds) carrying the ordered `chain` (one
`CauseLink` per exception, carriers flagged), the derived `category`, and
its `message`. Use it in a custom `predicate` that needs to see through
carriers, in a router or metric keyed on the originating category, or in a
retry classifier that wants full-chain depth (the default retry classifier
is deliberately single-level, classifying at re-attempt granularity rather
than walking the full chain).

### Composing with RetryMiddleware

The two compose into the canonical "retry transients, then give up
gracefully" pattern. The order is load-bearing: failure isolation is the
**outer** layer, retry is **inner**.

```python
builder.add_node(
    "summarize",
    summarize_fn,
    middleware=[
        FailureIsolationMiddleware(
            degraded_update={"summary": ""},
            event_name="summary_degraded",
        ),
        RetryMiddleware(RetryConfig(max_attempts=3)),
    ],
)
```

Retry sits closest to the node, so it sees raw transient failures first
and retries them. Only what escapes retry (an exhausted budget, or a
non-transient exception retry's classifier declines) reaches the outer
failure isolation, which degrades. Reverse the order and the inner
isolation would swallow transients before retry ever saw them, defeating
the retry entirely.

The [fan-out with retry example](../examples/fan-out-with-retry.md)
applies this composition as `instance_middleware` in its `degrade`
mode: each fan-out instance is wrapped isolation-outer / retry-inner,
so an instance whose retries exhaust degrades to a placeholder result
and the batch finishes instead of aborting.

## Related

- [Parallel branches](parallel-branches.md): per-branch middleware
  and its interaction with parent-graph middleware.
- [Fan-out](fan-out.md): `instance_middleware` and how it composes
  with parent and node-level layers.
- [LLMs](llms.md): how transient-classification flows from provider
  errors into `RetryMiddleware`'s default classifier.
- [Observability](observability.md): observer events emitted around
  middleware-wrapped nodes carry the retry attempt index.
