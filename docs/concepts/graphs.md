# Graphs: nodes, edges, build, invoke

Four moves turn a state schema into a runnable pipeline:

1. Write **node functions** that read state and return partial updates.
2. Wire them together with **edges** through `GraphBuilder`.
3. **Compile** the builder into an immutable graph.
4. **Invoke** the compiled graph to run it.

## Nodes are async functions

A node is just an async callable with this shape:

```python
from collections.abc import Mapping
from typing import Any
from openarmature.graph import State


class S(State):
    plan: str = ""


async def my_node(state: S) -> Mapping[str, Any]:
    return {"plan": "outline"}
```

Three things to notice:

- **Read the snapshot, return a partial update.** The node doesn't
  construct a new state, doesn't mutate the old one, doesn't worry
  about merging. The engine handles all of that.
- **The return is a `Mapping`, not a `dict` literally.** You can return
  any dict shape that satisfies the type; field names are validated
  against the state schema at merge time (extra keys raise).
- **Empty dict is fine.** `return {}` means "I made no state changes" —
  state passes through, execution moves on per the outgoing edge. Good
  for logging or pure-observation nodes.

**Why async?** The canonical node does IO — LLM call, HTTP request,
tool invocation. An async signature lets the runtime overlap IO when
you eventually add parallel branches or retries. For a purely CPU node,
async costs nothing — you just `return {...}` without an `await`.

You register the node on a builder under a name:

```python
builder.add_node("plan", plan_node)
```

The name is what edges reference. The function itself is the work.

## Edges: exactly one outgoing per node

Each node has **exactly one** outgoing edge. Branching is *not*
expressed with multiple static edges from the same source; it's a
single *conditional* edge whose function chooses the next node. (More
on conditional edges in [Composition](composition.md).)

A static edge is unconditional:

```python
builder.add_edge("plan", "write")    # after `plan` merges, run `write`
builder.add_edge("write", END)       # after `write` merges, halt
```

`END` is a sentinel object — a distinct value, not the string `"END"`:

```python
from openarmature.graph import END
```

Using the literal string `"END"` is fine *as a node name* if you want
one; the sentinel is a separate object so the engine can tell them
apart.

**Why one outgoing edge per node?** It concentrates the routing
decision into one place per source (in the case of conditional edges,
the routing function). Scattering routing across multiple static edges
would require some precedence rule. Compile-checking and reading both
get simpler when there's one rule.

## GraphBuilder is the construction surface

`GraphBuilder` is mutable. Every method returns `self` so you chain:

```python
from openarmature.graph import END, GraphBuilder, State


class S(State):
    plan: str = ""


async def plan(_s: S) -> dict[str, str]:
    return {"plan": "outline"}


graph = (
    GraphBuilder(S)
    .add_node("plan", plan)
    .add_edge("plan", END)
    .set_entry("plan")
    .compile()
)
```

The methods you'll use:

- **`GraphBuilder(state_cls)`** — constructor. The state class
  determines the reducer table at compile time.
- **`.add_node(name, fn)`** — register an async node function.
- **`.add_edge(source, target)`** — static edge. `target` is a node
  name or `END`.
- **`.add_conditional_edge(source, fn)`** — branching edge. `fn(state)`
  is sync and returns a node name or `END`.
- **`.add_subgraph_node(name, compiled, projection=None)`** — register
  a compiled graph as a node inside this graph (see
  [Composition](composition.md)).
- **`.set_entry(name)`** — declare where execution begins.
- **`.compile()`** — validate and return `CompiledGraph`.

**Why split builder and compiled?** Construction and execution are
different problems. Construction is mutable and permissive (add things
in whatever order reads well); execution wants something immutable and
validated. `compile()` is the one-way door between the two — and
structural problems surface at the door so a bad graph can't reach
runtime.

## compile() is the structural firewall

`GraphBuilder.compile()` runs structural checks and raises a
`CompileError` subclass on the first failure. The checks are:

| Error                              | When it fires                                                            |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `ConflictingReducers`              | A field has more than one reducer declared via `Annotated[...]`          |
| `NoDeclaredEntry`                  | `.set_entry(...)` was never called                                       |
| `DanglingEdge`                     | An edge references a node name that was never `.add_node()`'d            |
| `MultipleOutgoingEdges`            | A node has more than one outgoing edge                                   |
| `UnreachableNode`                  | A declared node isn't reachable from the entry via any edge path         |
| `MappingReferencesUndeclaredField` | A subgraph projection mapping names a field absent from the schema       |

Every failure here is a graph-shape problem — the kind of thing that
would otherwise crash mid-execution with a confusing traceback.
Catching them at construction means you *cannot* invoke a malformed
graph.

**Reachability is sound but loose.** Conditional edges over-approximate
during the reachability check: a conditional from node X is treated as
reaching every declared node (the compiler can't statically know the
function's range). So a node reachable only via a never-taken branch is
still considered reachable. No false positives; some slack on the
upper bound.

## invoke() runs the loop

`CompiledGraph.invoke()` runs to completion and returns the final
state:

```python
final = await graph.invoke(S())
```

The per-step loop:

1. Run the current node, await its result.
2. Merge its partial update into state via per-field reducers.
3. Re-validate state against the schema.
4. Evaluate the outgoing edge against the *post-merge* state to pick
   the next node (or `END`).

The output is the final `State` instance — whatever state looks like
when an edge returns `END`.

## Runtime errors carry context

If a node, reducer, edge function, or routing decision fails, the
engine raises one of these:

| Error                  | When it fires                                                       | `recoverable_state`? |
| ---------------------- | ------------------------------------------------------------------- | :------------------: |
| `NodeException`        | A node function raised                                              |         yes          |
| `ReducerError`         | A reducer raised (often a type mismatch on the partial)             |         yes          |
| `EdgeException`        | A conditional edge fn raised                                        |         yes          |
| `RoutingError`         | Conditional edge returned something that isn't a node name or `END` |         yes          |
| `StateValidationError` | Merged state fails schema validation (typo'd field, bad type)       |          no          |

`recoverable_state` is the state at the point of failure —
pre-failing-node for node/edge/routing errors, pre-merge for reducer
errors. Useful for post-crash forensics. State validation errors don't
carry recoverable_state because the merge that triggered the failure
hadn't produced a valid state to recover *to*.
