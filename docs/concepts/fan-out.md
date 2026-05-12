# Fan-out

Run the same subgraph many times in parallel, each instance receiving
a different input, results merged back deterministically.

The single-graph-twice pattern from
[`ExplicitMapping`](composition.md#explicitmapping-declarative) handles
two-or-three call sites where you know the parent fields up front. Fan-out
handles N call sites where N is determined at runtime — "for each
URL in `state.urls`, run the scraping subgraph; gather the results."

## The shape

A fan-out node is a regular node that, instead of running a function,
runs a *compiled subgraph N times*. You register it with a
`FanOutNode`:

```python
from openarmature.graph import FanOutConfig, FanOutNode

# instance_subgraph: CompiledGraph[InstanceState]
# - one instance per item in the source field
# - results merged back per the projection

fan_out = FanOutNode(
    subgraph=instance_subgraph,
    config=FanOutConfig(
        source_field="urls",                    # parent field — must be a list
        instance_input_field="url",             # subgraph field that receives each item
        concurrency=4,                          # max in-flight instances
        error_policy="continue_on_error",       # or "halt_on_error"
    ),
)

builder.add_node("scrape_all", fan_out)
```

The engine reads `state.urls`, dispatches one subgraph invocation per
URL (up to `concurrency` at a time), and merges the per-instance
results back into the parent via per-field reducers.

## Per-instance state

Each instance gets its own subgraph state, distinct from siblings. The
instance receives:

- The dispatched item (in the field named by `instance_input_field`).
- Any other input fields projected in via the projection strategy
  (same as a regular subgraph).

It produces a final state on its own, no different from a non-fan-out
subgraph invocation. The projection's `project_out` runs for each
instance, producing a partial update merged into the parent.

The parent's reducers handle merging across instances. Two patterns
matter:

- **List accumulation via `append`.** Each instance returns its result
  in a list field; the parent has `append`, so the per-instance
  results concatenate in *instance order*.
- **Map keyed by item id via `merge`.** Each instance returns
  `{"results": {item_id: result}}`; the parent has `merge`, so the
  per-instance partials assemble into one keyed map.

## Concurrency is bounded, not unlimited

`concurrency` caps the number of in-flight instances. Higher values
trade memory + downstream-API pressure for wall-clock latency. Tune to
the slowest external dependency: an LLM endpoint at 4 parallel
requests; a scraping target at 1 to be polite.

The engine drives the bound with `asyncio.Semaphore`. Instances queue
up internally; you don't see them in state until they finish.

## Error policy

What happens when one instance fails? Two modes:

- **`halt_on_error`** — the first instance failure halts the whole
  fan-out node. The engine cancels the in-flight siblings (best
  effort) and raises a `NodeException` wrapping the failing
  instance's error. Strict semantics, useful when one bad result
  invalidates the rest.
- **`continue_on_error`** — instance failures are captured into the
  result; the fan-out continues to completion. Each failed instance's
  partial update includes an error marker (typically a list of
  `errors` in the parent's state). Loose semantics, useful when N-1
  good results are useful even if 1 fails.

Choose based on whether you can use partial results.

## Observability per instance

Observer events fire for every node inside every instance, with
namespaces that include the fan-out index:

- `namespace = ("scrape_all", "fan_out_instance_3", "fetch")` —
  fetch node inside the 4th instance of fan-out node `scrape_all`.
- `event.fan_out_index = 3` — explicit instance index on the event.
- `event.parent_node_name = "scrape_all"` — the fan-out node's name,
  for parent attribution.

This lets observers (and the OTel mapping in particular) build a
hierarchy where each instance is a sibling span under the fan-out
node's span, with per-instance attributes.

## When to reach for fan-out

The signal: you have N similar pieces of work, N depends on state at
runtime (not at build time), and the work is independent enough that
running them concurrently is correct.

If N is known at build time and small (≤3), `ExplicitMapping` at
multiple sites is simpler. If the work isn't independent — instance 2
needs instance 1's output — that's a linear pipeline, not fan-out.

## What fan-out is NOT

- **Not a map-reduce.** No reduce phase beyond the parent's reducers.
  If you need a real reduce, do it in a node *after* the fan-out.
- **Not a queue.** All instances dispatch within a single invocation;
  the engine doesn't persist them.
- **Not retry.** If an instance fails and you want a retry, that's the
  middleware layer, not fan-out.
