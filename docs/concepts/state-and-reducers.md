# State and reducers

## State is a typed, frozen Pydantic model

The graph is a pipeline of pure state transitions. Each node receives a
snapshot of state, returns a *partial update* (a dict of just the fields
it wants to change), and the engine merges the update via per-field
reducers. The engine is the only thing that writes to state.

The shape of state is your responsibility. You subclass `State`:

```python
from typing import Annotated
from openarmature.graph import State, append
from pydantic import Field


class GraphState(State):
    topic: str
    plan: str = ""
    output: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)
```

The `State` base class is a pre-configured Pydantic `BaseModel`. Two
guarantees come baked in:

- **Frozen.** `model_config = ConfigDict(frozen=True, ...)`. A node
  can't `state.plan = "..."` even if it tried; the assignment raises.
- **No extra fields.** `extra="forbid"`. A node that returns
  `{"plann": "..."}` (typo) fails loudly with a `StateValidationError`
  instead of silently dropping the key.

Everything else Pydantic gives you — validators, computed fields,
custom types, `Field` metadata — still works. You don't need to set
`model_config` yourself; subclassing `State` is enough.

**Why frozen?** It rules out a whole class of bugs that make multi-step
LLM pipelines miserable: the snapshot a node holds can't be mutated by
anything else while it's running. State changes are an *engine action*
(the merge), not a *node action*. The node's job is "produce this
update"; merging is somebody else's problem.

**Why `extra="forbid"`?** Field-name typos are common during iteration.
Failing loudly at the merge boundary means a typo can't quietly produce
a graph that runs but produces the wrong output.

## State does NOT have history

The engine doesn't retain prior state snapshots. `CompiledGraph.invoke()`
holds one `state` local; each merge reassigns it, and the previous
snapshot becomes unreferenced.

```text
state = initial_state
while not at END:
    partial = await current_node(state)
    state = merge(state, partial)   # prior state now unreferenced
    current_node = next_node_for(state)
```

This is by design. Checkpoint/resume, per-node streaming, persistent
state backends, and human-in-the-loop interrupts are explicit
non-goals for the engine itself. They're pipeline-layer utilities
that compose *on top of* the graph primitives. Keeping the engine
one-job keeps it small.

What you do have for "what happened":

- A **user-built trail inside state.** Idiomatic: an
  `Annotated[list[str], append]` field that nodes write to. Whatever
  your schema captures, that's the history you get after `invoke()`
  returns.
- **Crash context.** The four non-validation runtime errors
  (`NodeException`, `EdgeException`, `ReducerError`, `RoutingError`)
  carry a `recoverable_state` — the state at the point of failure. Good
  for forensics; not a walkable timeline.

If you need a full timeline (debugging, eval, time-travel,
resumability), build it explicitly: fatter `trace`, logging middleware,
or use [Checkpointing](checkpointing.md).

## Reducers: one per field

A reducer is a named callable: `(prior, partial) -> new`. Attach one to
a field via `typing.Annotated`:

```python
from typing import Annotated
from openarmature.graph import State, append
from pydantic import Field


class GraphState(State):
    trace: Annotated[list[str], append] = Field(default_factory=list)
```

On each merge step, the engine looks up the reducer for every updated
field and calls `reducer(prior_value, partial_value)`. Fields without
an annotated reducer fall back to `last_write_wins`.

**The point of per-field reducers:** a node shouldn't know how its
output combines with prior state — that's a property of the field, not
the node. `trace.append`, `meta.merge`, `score.last_write_wins`. The
schema declares the policy once; nodes return their increment; the
engine applies the merge consistently. If two nodes write the same
field and the merge strategy is wrong, the fix is one line on the
schema, not surgery across call sites.

## Three built-in reducers

| Reducer           | Semantics                                | Typical use                           |
| ----------------- | ---------------------------------------- | ------------------------------------- |
| `last_write_wins` | `partial` replaces `prior` *(default)*   | Scalars owned by a single node        |
| `append`          | `[*prior, *partial]` for list fields     | Traces, message history, accumulators |
| `merge`           | `{**prior, **partial}` (shallow)         | Metadata bags, namespaced state       |

```python
from openarmature.graph import append, last_write_wins, merge
```

You can write your own. A reducer is any named callable matching the
`(prior, partial) -> new` contract.

## How reducers execute

A reducer **always returns a new value** — never mutates `prior`. That
matches the frozen-state contract: the prior list/dict may still be a
snapshot somebody else holds.

The built-ins type-check their inputs before running. If a node returns
a string for a `list`-typed field, `append` raises `TypeError` before
the bad value can land in state; the engine wraps it as a
`ReducerError` carrying the field name, reducer name, and producing
node.

Side note: `append`'s name is a bit misleading. It's list
*concatenation* (`[*prior, *partial]`), not `list.append`. It can't
mutate for the same reason `State` is frozen.

## Return the increment, not the full value

For reducer-tracked fields, a node returns *only what it's adding*:

```python
async def plan_node(s: GraphState) -> dict[str, list[str]]:
    return {"trace": ["plan"]}   # add ["plan"] to trace
```

NOT `{"trace": s.trace + ["plan"]}` — that's already what `append`
does. Returning the full list would concatenate twice and duplicate
entries.

## Two reducers on one field → compile error

You *can* try to declare two reducers on a field:

```python
class Bad(State):
    log: Annotated[list[str], append, merge] = Field(default_factory=list)
```

But `GraphBuilder.compile()` fails with `ConflictingReducers("log")` —
the graph never compiles, so you can't reach runtime with an ambiguous
merge policy. The same compile pass picks the one declared reducer per
field; with no declaration, the default is `last_write_wins`.
