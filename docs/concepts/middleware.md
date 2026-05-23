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
from openarmature.graph import RetryMiddleware, exponential_jitter_backoff


async def on_retry(exc: Exception, attempt: int) -> None:
    log.warning("retrying after %r (attempt %d)", exc, attempt)


retry = RetryMiddleware(
    max_attempts=3,
    backoff=exponential_jitter_backoff,
    on_retry=on_retry,
)
```

Four plug points, all optional:

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

## Related

- [Parallel branches](parallel-branches.md): per-branch middleware
  and its interaction with parent-graph middleware.
- [Fan-out](fan-out.md): `instance_middleware` and how it composes
  with parent and node-level layers.
- [LLMs](llms.md): how transient-classification flows from provider
  errors into `RetryMiddleware`'s default classifier.
- [Observability](observability.md): observer events emitted around
  middleware-wrapped nodes carry the retry attempt index.
