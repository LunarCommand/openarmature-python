# Fan-out

Run the same subgraph many times in parallel, each instance receiving
a different input, results merged back deterministically.

The "same subgraph at two-or-three call sites" pattern from
[`ExplicitMapping`](composition.md#explicitmapping-declarative)
handles cases where you know the parent fields up front. Fan-out
handles N call sites where N is determined at runtime: "for each
item in `state.urls`, run the scraping subgraph; collect the
results."

## Two modes: per-item or per-count

A fan-out can dispatch instances driven by a list in state
(`items_field` mode) or by a count resolved from state (`count` mode).

**`items_field` mode**: one instance per item in a parent list field:

```python
from openarmature.graph import FanOutConfig, FanOutNode

scrape_all = FanOutNode(
    name="scrape_all",
    config=FanOutConfig(
        subgraph=scrape_subgraph,        # CompiledGraph[ScrapeState]
        items_field="urls",              # parent list field, one instance per item
        item_field="url",                # subgraph field that receives each item
        collect_field="content",         # subgraph field whose value is collected
        target_field="contents",         # parent list field that receives the collection
        concurrency=4,
        error_policy="fail_fast",        # or "collect"
        on_empty="raise",                # or "noop"
    ),
)
builder.add_node("scrape_all", scrape_all)
```

**`count` mode**: fixed-or-dynamic instance count, no list field:

```python
fan_out = FanOutNode(
    name="sample",
    config=FanOutConfig(
        subgraph=sample_subgraph,
        count=8,                          # int or callable: state -> int
        collect_field="reading",
        target_field="readings",
        concurrency=4,
    ),
)
```

Both `count` and `concurrency` accept a callable that takes the
pre-fan-out parent state and returns an int (`None` for `concurrency`
means unbounded). That lets you size the dispatch from state at run
time.

## Per-instance state, inputs and outputs

Each instance gets its own subgraph state, distinct from siblings,
distinct from the parent. By default the instance receives only:

- the dispatched item in the field named by `item_field` (in
  `items_field` mode); and
- the parent-field-name-mapped values declared in `inputs`.

`inputs` is a `Mapping[subgraph_field, parent_field]`. The subgraph
fields not named in `inputs` (and not `item_field`) take their
schema defaults; same closed-by-default-on-the-way-in posture as
the explicit-projection story for ordinary subgraphs.

On exit, each instance's `collect_field` value becomes one element
of the parent's `target_field` list, in instance-index order. To
collect additional per-instance fields, declare
`extra_outputs: Mapping[parent_field, subgraph_field]`; each becomes
its own parent list of the same length, instance-index-aligned.

## Error policy

Two values:

- **`"fail_fast"`** (default): the first instance failure cancels
  the in-flight siblings (`asyncio.gather` semantics) and propagates
  as a `NodeException` wrapping the failing instance's cause, with
  `recoverable_state` set to the parent's pre-fan-out snapshot. Use
  this when one bad result invalidates the rest.
- **`"collect"`**: instance failures are captured; the fan-out runs
  to completion. Failed instances contribute nothing to
  `target_field`. If you declare `errors_field` on the config, each
  failed instance produces a record (`{"fan_out_index": str(idx),
  "category": str}`) appended to that parent list field.

Choose by whether partial results are useful.

## What ends up in the parent

After the fan-out completes, the parent receives a partial update
containing:

- `target_field`: list of `collect_field` values, instance-index order.
- Each parent name in `extra_outputs`: list of values from the named
  subgraph field, instance-index order.
- `count_field` (if configured): the instance count.
- `errors_field` (if configured, `"collect"` policy only): per-instance
  error records.
- `on_empty="noop"` for an empty items_field → all the above with empty
  lists; `count_field` set to 0.

## Empty fan-outs

If `items_field` is set and the parent list is empty (or `count`
resolves to 0):

- `on_empty="raise"` (default): raises `FanOutEmpty` (a runtime
  error category).
- `on_empty="noop"`: emits an empty partial (no instances dispatched,
  no errors).

## Observability per instance

The fan-out node's own `started` / `completed` events carry a
`fan_out_config` payload populated from the resolved
`item_count` / `concurrency` / `error_policy` / `parent_node_name`.

Per-instance events have `fan_out_index = N` (0-based) and a
namespace whose final element is the fan-out node's name; instances
do NOT contribute a separate synthetic namespace element. Backends
disambiguate per-instance spans using `fan_out_index` alongside the
namespace.

## Resume semantics

A fan-out node's `completed` event triggers a save like any other
outermost-graph or subgraph-internal node. **Per-instance internal
events do NOT save** in the shipping version; on resume, the
fan-out re-runs end-to-end if it hadn't completed (atomic restart).

A per-instance fan-out resume mode is planned but not yet shipped.
The `fan_out_progress` field on `CheckpointRecord` is reserved for
its eventual contents. Until it lands, atomic restart is the
shipping behavior.

## When to reach for fan-out

The signal: N similar pieces of work, N depends on state at runtime
(not at build time), the work is independent enough to run
concurrently. If N is known at build time and small (≤3),
`ExplicitMapping` at multiple subgraph sites is simpler. If the
work isn't independent (instance 2 needs instance 1's output),
that's a linear pipeline, not fan-out.

## What fan-out is NOT

- **Not a map-reduce.** No reduce phase beyond the parent's
  reducers. If you need a real reduce, do it in a node *after* the
  fan-out.
- **Not a queue.** All instances dispatch within a single
  invocation; the engine doesn't persist them.
- **Not retry.** If an instance fails and you want a retry,
  wrap the subgraph (or individual nodes inside it) with retry
  middleware. The fan-out's `error_policy` is a fan-in-collection
  decision, not a recovery one.
