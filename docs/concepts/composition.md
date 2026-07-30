# Composition: conditional edges, subgraphs, projection

Three composition mechanisms turn a linear pipeline into a routed
pipeline of reusable sub-pipelines:

1. **Conditional edges** route based on state.
2. **Subgraphs** encapsulate a sub-pipeline as a single node.
3. **Projections** translate state across the subgraph boundary.

None of these add new primitives (a conditional edge is still one
outgoing edge, a subgraph is still a single node) but they change
what a graph can express.

## Conditional edges

A conditional edge is the one outgoing edge from a node whose
*target* is computed from state. You register a sync function that
receives the post-merge state and returns a node name or `END`:

```python
from openarmature.graph import END, EndSentinel, State


class S(State):
    route: str = ""


def route_from_classification(s: S) -> str | EndSentinel:
    if s.route == "research":
        return "research"
    return "quick_answer"


# builder.add_conditional_edge("classify", route_from_classification)
```

**Routing decisions belong in state.** Notice that `classify` writes
its decision into a regular state field (`route`) and the edge fn just
reads it. This is deliberate.

Compared to "classify returns the next node name, the engine stashes
it somewhere invisible," state-driven routing gives you:

- **Visibility.** The decision is a typed field on `State`. It shows
  up in the final state, in `recoverable_state` on crash, in any
  trace/logs your nodes emit.
- **Inspectability.** Downstream nodes can *read* `s.route` too. If a
  branch wants to know "how did we get here?", it doesn't have to
  reconstruct the answer.
- **Testability.** The routing function is a pure `state → string`
  call. Test it without touching the graph or any LLM.

**Why sync?** Conditional edges are routing decisions, not units of
work. If you want `async def`, the right move is to do the IO in the
producing node and write the decision to a state field, exactly what
`classify` does. Keeping edges sync keeps the loop simple to read:
node (async) → merge → edge (sync) → next.

**Timing.** The edge fn sees state *after* the source node's update
has been merged and re-validated. So `s.route` is whatever `classify`
just wrote (or the prior default, if `classify` didn't touch it).

**Failure modes:**

- **Edge fn raises** → `EdgeException`, carries `recoverable_state`.
- **Edge fn returns something that isn't a declared node name or
  `END`** → `RoutingError`, carries `recoverable_state` and the bad
  return value.

**Default-branch patterns:**

- *Permissive fallback*: `return "happy" if cond else "fallback"`. Every
  state value routes somewhere.
- *Halt on unknown*: `return "happy" if cond else END`. If the
  classifier misbehaves, the graph stops cleanly.
- *Route to an error node*: send unexpected states to a real node that
  can log/enrich/halt. Useful when you want the error path to be as
  observable as the happy path.

## Subgraphs

A subgraph is a `CompiledGraph` used as a node inside another graph.
From the outer graph's point of view it behaves like any other node:
one name, one outgoing edge, receives state / returns partial update.

```python
research_subgraph = (
    GraphBuilder(ResearchState)
    .add_node("plan", plan_research)
    .add_node("gather", gather)
    .add_node("synthesize", synthesize)
    .add_edge("plan", "gather")
    .add_edge("gather", "synthesize")
    .add_edge("synthesize", END)
    .set_entry("plan")
    .compile()
)

# In the outer graph:
builder.add_subgraph_node("research", research_subgraph, projection=...)
```

**Encapsulation and reuse:**

- **Encapsulation.** The outer graph knows that `research` produces an
  answer. It doesn't know about `plan`, `gather`, or `synthesize`. If
  the research pipeline gains a `verify` step, only the subgraph
  changes; outer wiring is untouched.
- **Reuse.** The compiled subgraph is a plain Python value. You can
  `await research_subgraph.invoke(...)` directly in a test, drop it
  into a *different* outer graph, or compose it inside yet another
  subgraph.

**Separate state schemas are load-bearing.** The subgraph has its own
`State` subclass, distinct from the parent's. At compile time, the
subgraph's reducer table and field validation are built against its
own schema. Parent fields can't leak in by accident; they aren't in
scope on either side of the boundary. **The only way data crosses is
through the projection.**

**When to reach for one.** Two signals:

- The inner steps form a cohesive sub-computation with its own state
  shape (e.g., research nodes need `angles` / `notes` / `answer`; the
  outer graph doesn't care).
- You want those inner steps to be reusable or testable in isolation.

If neither applies, inline nodes are simpler. Don't add a subgraph
boundary just to have one.

## Projection: the parent ↔ subgraph data seam

A `ProjectionStrategy` is the translation layer at the boundary. It
decides **what the subgraph sees on entry** and **what leaks back out
on exit**. It's a Protocol with two methods:

```python
class ProjectionStrategy(Protocol):
    def project_in(self, parent_state: State, subgraph_state_cls: type[State]) -> State: ...
    def project_out(
        self,
        subgraph_final_state: State,
        parent_state: State,
        subgraph_state_cls: type[State],
    ) -> Mapping[str, Any]: ...
```

Three strategies cover most cases.

### `FieldNameMatching` (the default)

If you don't pass a `projection=` argument, you get this. It behaves
asymmetrically:

- **`project_in`: parent state is ignored.** Returns
  `subgraph_state_cls()`, a fresh instance from the subgraph's
  defaults. If the subgraph has a required field, this constructor
  fails; the subgraph can't run without an explicit projection.
- **`project_out`: field-name intersection.** Looks at the subgraph's
  final state, keeps fields whose names also exist on the parent, and
  returns them as a partial update. The parent's reducers then merge.

The asymmetry, "closed on the way in, open on the way back," is by
design. The author opts *in* to sharing data with the subgraph; the
subgraph's observable outputs route back through the parent's reducers
automatically.

In practice, the default is rarely enough.

### `ExplicitMapping` (declarative)

When the projection is "copy parent.foo into subgraph.bar on the way
in, write subgraph.baz back as parent.qux on the way out," reach for
`ExplicitMapping`:

```python
from openarmature.graph import ExplicitMapping

projection = ExplicitMapping[ParentState, SubgraphState](
    inputs={"topic": "topic_a"},                          # subgraph_field: parent_field
    outputs={"a_summary": "summary", "a_score": "score"}, # parent_field: subgraph_field
)
builder.add_subgraph_node("analyze_a", subgraph, projection=projection)
```

`inputs` and `outputs` are independent; pass either, both, or neither.

**Asymmetry: inputs additive, outputs replacement.** This mirrors the
default's asymmetry.

- `inputs` is *additive over no-projection-in*. Subgraph fields named
  in `inputs` get the corresponding parent field's value; unnamed
  fields get their schema defaults.
- `outputs` *replaces* field-name matching when present. Only pairs
  named in `outputs` are merged back. Unnamed subgraph fields are
  discarded, so no slip of extra fields by accident.

**`None` vs `{}` for `outputs`:**

- `outputs=None` (absent) → fall back to field-name matching for
  project-out. Useful when you want precise inputs but the default's
  output behavior.
- `outputs={}` (empty) → project nothing back. Useful for
  fire-and-forget subgraphs whose results you intentionally drop.

**Compile-time validation.** `ExplicitMapping.validate` runs at
parent-graph compile and raises `MappingReferencesUndeclaredField` if
any mapping names a field that isn't on the relevant schema.
Refactor-safe: if you rename a parent field but forget the mapping,
construction fails, not runtime.

**The case `ExplicitMapping` uniquely unlocks.** Same subgraph at
multiple sites with disjoint parent fields:

```python
analysis = build_analysis_subgraph()  # one CompiledGraph

builder.add_subgraph_node(
    "analyze_a",
    analysis,
    projection=ExplicitMapping[ComparisonState, AnalysisState](
        inputs={"topic": "topic_a"},
        outputs={"a_summary": "summary", "a_score": "score"},
    ),
)
builder.add_subgraph_node(
    "analyze_b",
    analysis,
    projection=ExplicitMapping[ComparisonState, AnalysisState](
        inputs={"topic": "topic_b"},
        outputs={"b_summary": "summary", "b_score": "score"},
    ),
)
```

The two sites address disjoint parent fields, so they cannot collide.
Without explicit mapping, both calls would have to read from and write
to the same parent fields under name matching, making "run the same
subgraph twice on different inputs" structurally impossible.

### `DeclaredSameName` (declared, same-name only)

Most boundaries do not rename anything: the parent and the subgraph
already agree on the field names, and the rename maps are pure
boilerplate. `DeclaredSameName` names the crossing fields as two sets
of bare field names:

```python
from openarmature.graph import DeclaredSameName

projection = DeclaredSameName[ParentState, SubgraphState](
    in_fields={"topic", "corpus"},   # copied parent -> subgraph, same name
    out_fields={"summary"},          # merged subgraph -> parent, same name
)
builder.add_subgraph_node("summarize", subgraph, projection=projection)
```

Per-field behavior is exactly `ExplicitMapping` restricted to same-name
pairs, and both sets are validated against both schemas at compile time.
That is the point: it is the checked middle between the default (terse
but unchecked, so a rename silently stops carrying a field) and the maps
(checked but verbose). Rename a field on either side and you get a
`MappingReferencesUndeclaredField` at compile, not a value quietly
arriving as its schema default.

**It is a complete declaration, with no fallback.** `ExplicitMapping`
distinguishes an *absent* `outputs` (falls back to field-name matching)
from an *empty* one (projects nothing). The declared form drops that
distinction: empty means "nothing" in both directions, so an
accidentally-empty set cannot silently re-enable the implicit behavior.
If you want field-name matching, use the default strategy instead.

**`out_fields` defaults to empty, so omitting it projects nothing back.**
This is the sharpest edge of the new form, and it is the opposite of what
omitting `outputs` does:

```python
DeclaredSameName(in_fields={"topic"})              # projects NOTHING out
ExplicitMapping(inputs={"topic": "topic"})         # name-matches everything out
```

Both compile clean. If you want fields to come back, name them.

Use `ExplicitMapping` when any pair actually renames. A node declares
its projection with one form: a strategy exposing both the sets and the
maps fails compilation with `ConflictingProjectionForms`.

### Watch out: round-tripping a field through a growing reducer

Projection-out merges into the parent through the parent's reducers,
exactly as any node's return does. That consistency has a sharp edge: if
you project a field *in* and then the same field back *out*, the value
merges a second time. With `last_write_wins` that is harmless. With
`append` the list doubles.

The library warns about this at compile time:

```python
import warnings
from openarmature.graph import CompileWarning

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    graph = builder.compile()
# caught[0].message is a ProjectionReducerRoundTrip, whose `category` is
# "projection_reducer_round_trip"
```

The warning changes no runtime behavior, and on its own it does not fail
compilation. It runs last, after every structural check has passed, so it
can never mask a real compile error. Treat it as a prompt to either route
the field through an idempotent reducer, or stop round-tripping it (keep
the subgraph read-only for that field and have the parent read it
directly). If you would rather it be fatal,
`warnings.simplefilter("error", CompileWarning)` promotes it, and then
`compile()` does raise. Note that a `CompileWarning` is a `UserWarning`
subclass, not a `CompileError`, so catch it accordingly. The same applies
if your project already runs under `-W error` or pytest
`filterwarnings = ["error"]`.

What the doubling costs depends on the reducer: `append` and
`concat_flatten` grow the field, while re-applying `merge_all` is
ill-typed and raises at runtime instead.

It fires on the declarative projection surfaces: a subgraph node using
`ExplicitMapping` or `DeclaredSameName`, both of a fan-out's output
channels, and a parallel-branches subgraph branch's `inputs` / `outputs`.
Touching the same parent field through two *different* subgraph fields is
not a round-trip (the value coming back is a distinct, subgraph-computed
one), so that case stays silent.

An **imperative custom strategy** is not covered. The engine can only
read a declared boundary, so if you write your own `project_in` /
`project_out` you are responsible for this hazard yourself.

Fan-out has two ways back to the parent, and both count. The obvious one
is the `extra_outputs` map. The easier one to miss is the **primary
collect channel**, where the round-trip is spelled with scalars instead
of a map:

```python
class Parent(State):
    # concat_flatten, not append: the fan-in hands the parent a LIST of
    # per-instance values, and only concat_flatten can consume that.
    results: Annotated[list[str], concat_flatten] = Field(default_factory=list)

class Instance(State):
    acc: list[str] = Field(default_factory=list)

async def add_one(state: Instance) -> dict[str, Any]:
    return {"acc": [*state.acc, "new"]}

builder.add_fan_out_node(
    "spread",
    subgraph=inner,              # a graph over Instance running add_one
    count=2,
    inputs={"acc": "results"},   # parent results -> each instance's acc
    collect_field="acc",         # ...and each instance's acc
    target_field="results",      # ...collected straight back into results
)
```

The seed value re-merges **once per instance**: start from `["seed"]` over
two instances and `results` ends up
`['seed', 'seed', 'new', 'seed', 'new']`.

Two things to note. `append` on the collect channel does not merely
double: the update is a list of lists, so it fails state validation
outright. And the seeding can also be spelled with `items_field` /
`item_field`, which hands each instance one *element* of the parent list;
that round-trips too, and warns.

Feed `collect_field` from anywhere other than `target_field` and there is
no round-trip, so collecting into a growing reducer is not by itself a
problem. Branches need no equivalent care: a branch has only its
`outputs` map, with no per-branch collect slot.

The check is structural: it cannot prove the subgraph left the value
untouched, so a round-trip that legitimately replaces the value will
still warn.

### Custom projection strategies

If you need behavior beyond name-mapping (synthesize values, project
conditionally, transform on the way through), write a class that
matches the Protocol:

```python
class QuestionProjection:
    def project_in(self, parent_state, subgraph_state_cls):
        return subgraph_state_cls(question=parent_state.question)

    def project_out(self, subgraph_final_state, parent_state, subgraph_state_cls):
        return {
            "answer":  subgraph_final_state.answer,
            "trace":   subgraph_final_state.trace,         # appends into parent's trace
            "tallies": {"research_runs": 1},               # merges into parent's tallies
        }
```

Then `.add_subgraph_node("research", research_subgraph,
projection=QuestionProjection())`.

A few design points worth sitting with:

- **`project_out` returns a partial update, not a full state.** The
  parent's reducers apply: `trace` appends, `tallies` merges, `answer`
  is `last_write_wins`. Clean composition without thinking about it.
- **Unknown fields from `project_out` raise.** Parent's `extra="forbid"`
  catches typos at the merge boundary.
- **The `parent_state` argument of `project_out` is for context, not
  for writing.** You can read it to decide what to project ("only
  return the answer if the parent was in a research route") but you
  can't mutate it.

`ProjectionStrategy` is a `Protocol`, not a base class. A class fits
the shape or it doesn't; the type checker verifies at use sites. If
you have Java instincts ("where's the `implements` keyword?"), reach
for TypeScript or Go interface instincts instead; that's the same
family.

## Related composition primitives

Subgraphs run once per outer-graph entry into them. Two related
primitives run subgraphs multiple times or in parallel; both use
the same projection machinery at their boundaries.

- [Fan-out](fan-out.md): dispatch N copies of *one* compiled subgraph
  against an input collection. Use when you have a list of similar
  items to process independently.
- [Parallel branches](parallel-branches.md): dispatch M *heterogeneous*
  subgraphs concurrently against the same parent state, each with its
  own state schema and (optional) middleware. Use when several
  independent analyses share a single input.
