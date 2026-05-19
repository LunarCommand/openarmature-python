# Parallel branches

Dispatch M heterogeneous subgraphs concurrently, projected outputs
merged back into the parent via the parent's reducers in branch
insertion order.

Sibling to [fan-out](fan-out.md) (same `for each thing, do work in
parallel` shape), but the *thing* is different per branch: a research
subgraph, a categorize subgraph, a sentiment subgraph (each with its
own state schema, its own middleware, its own observer events),
running in parallel and joining their results into one parent state.

## When to reach for parallel branches

The signal: a fixed set of named operations, each with its own
behavior and state schema, that don't depend on each other. Three
classifiers running independently against the same input. A research
step, a translate step, and a fact-check step that all want the
parent's prompt. M is known at build time and small (typically 2–6),
and each branch is its own subgraph because each has its own
internal pipeline worth modelling separately.

Fan-out is the right pick when you have N similar pieces of work,
N depends on runtime state, and the work is the same across instances.
Parallel branches is the right pick when M is a small fixed set of
different operations that happen to run concurrently.

## The shape

```python
from openarmature.graph import BranchSpec, GraphBuilder

builder.add_parallel_branches_node(
    "dispatcher",
    branches={
        "research": BranchSpec(
            subgraph=research_subgraph,            # CompiledGraph[ResearchState]
            inputs={"question": "prompt"},         # subgraph_field -> parent_field
            outputs={"facts": "facts"},            # parent_field -> subgraph_field
        ),
        "translate": BranchSpec(
            subgraph=translate_subgraph,           # CompiledGraph[TranslateState]
            inputs={"source": "prompt"},
            outputs={"translation": "translated"},
        ),
        "fact_check": BranchSpec(
            subgraph=fact_check_subgraph,          # CompiledGraph[FactCheckState]
            inputs={"claim": "prompt"},
            outputs={"verdict": "verdict"},
        ),
    },
    error_policy="fail_fast",                      # or "collect"
)
```

Each branch's `subgraph` is a compiled graph; `inputs` and `outputs`
mirror the explicit projection shape from
[composition](composition.md#explicitmapping-declarative). The
branches dict's key is the branch name, used as the branch identity
on observer events (see [observability](observability.md)) and in
the per-branch error records that `error_policy: "collect"`
produces.

## Per-branch state, inputs and outputs

Each branch runs its own subgraph against its own state; heterogeneous
schemas are explicit. Subgraph fields named in `inputs` are seeded
from the parent's corresponding field at branch entry; other subgraph
fields take their schema defaults. At branch exit, only the parent
fields named in `outputs` receive contributions; the rest of the
branch's final state is discarded.

When two branches contribute to the same parent field, the parent's
reducer for that field applies both values in **branch insertion
order**: first the branch declared first in the `branches` dict,
then the next, and so on. This is deterministic regardless of which
branch's inner work finishes first.

## Error policy

- **`"fail_fast"`** (default): the first branch failure cancels
  the in-flight siblings and propagates as
  `ParallelBranchesBranchFailed` (a `NodeException` subtype) carrying
  the failing `branch_name` and the original cause as `__cause__`.
  `recoverable_state` is the parent's snapshot at the moment the
  dispatcher entered. **No buffered branch contributions are
  applied**, including those of branches that successfully completed
  before the failure. Buffer-and-apply semantics: contributions are
  held until every branch finishes, then either all apply (success)
  or none apply (fail_fast failure).
- **`"collect"`**: every branch runs to completion. Successful
  branches' contributions merge in insertion order; failed branches'
  `outputs` projections do NOT fire (their named parent fields stay
  at their defaults). If you declare `errors_field` on the dispatcher,
  each failed branch produces a record with at minimum
  `{"branch_name": <name>, "category": <category>}` appended to that
  parent list field; the implementation may include additional keys
  (message, cause_type) and tests should match by the spec-mandated
  keys rather than strict equality.

## Branch middleware

Each `BranchSpec` accepts a `middleware` tuple of middlewares that
wrap that branch's whole subgraph invocation as a unit. Retry
middleware on a branch retries the **whole branch**: a fresh
subgraph invocation each time, fresh inner-node execution. The
wrapping retry's attempt counter propagates to events emitted from
inner nodes (per graph-engine §6 v0.16.1), so observer events
inside the branch correctly show `attempt_index` ticking across
retries.

Branch middleware is independent across branches: branch A may
have `[retry, timing]`; branch B may have `[]`; branch C may have
some custom breaker. Each branch's chain composes in isolation.

## Composition with other constructs

Parallel branches compose with the rest of the engine the way
subgraphs and fan-outs do:

- A branch's subgraph can itself contain a fan-out node; inner-node
  events inside that fan-out carry **both** `branch_name` (this
  branch) and `fan_out_index` (the instance within this branch).
  The two fields are independent.
- The parallel-branches node itself can be invoked from inside a
  fan-out instance, and inner events then carry the outer fan-out's
  `fan_out_index` and the inner branch's `branch_name`.
- Per-graph and per-node middleware on the parallel-branches node
  wrap the dispatcher as a single unit: one `started` event before
  dispatch begins, one `completed` event after all branches finish
  and fan-in lands. The parent's retry middleware retries the **whole
  parallel-branches node**, not individual branches.

## Resume semantics

Parallel-branches nodes use the same **atomic restart** model as
fan-out (per spec §10.7): if a checkpoint resume lands on a
parallel-branches node, all branches re-dispatch from scratch.
Per-branch progress is not individually persisted in v1.

## When parallel branches is NOT the right shape

- **Not the same as N copies of one subgraph.** If you want "run
  this subgraph for each item in a list," reach for
  [fan-out](fan-out.md).
- **Not a router.** A router is a conditional-edge pattern that
  picks one branch based on state. Parallel branches runs *all*
  branches concurrently.
- **Not a coordinator.** Branches don't communicate with each other
  during execution; if branch B's work depends on branch A's
  output, you want a linear pipeline (A → B), not parallel branches.
