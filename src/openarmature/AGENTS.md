# OpenArmature — Agent documentation

*This is the agent guide bundled with the openarmature Python package, version 0.10.0 (spec v0.35.0). For the full docs site see [openarmature.ai](https://openarmature.ai). For the canonical spec text see [openarmature.org/capabilities](https://openarmature.org/capabilities/). For project-specific conventions for the code you're editing, see the host project's `AGENTS.md` or `CLAUDE.md`.*

## TL;DR

OpenArmature is a workflow framework for LLM pipelines and tool-calling agents — typed state, compile-time topology checks, observability, and crash-safe checkpoints baked into a graph engine. The graph layer has no concept of LLMs or tools; the same primitives drive deterministic ETL pipelines and tool-calling agents alike. Nodes return partial updates; the engine merges into a frozen state snapshot. Behavior is defined by [openarmature-spec](https://openarmature.org/capabilities/) and verified by conformance fixtures; this package is the reference Python implementation.

**What OpenArmature is NOT:** not a chat framework (no built-in messages channel), not an LLM SDK (Provider is the abstraction layer; OpenAIProvider is the canonical impl), not a state-management library (state is per-invocation, not application-wide), not an evaluation framework (deferred to `openarmature-eval`).

## Capability contracts

_Sourced from openarmature-spec v0.35.0. Each entry below reproduces §1 (Purpose) and §2 (Concepts) of the capability's `spec.md`. For the full spec text (execution model, error semantics, determinism, observer hooks, etc.) see the linked docs site._

### Capability: `graph-engine`

#### 1. Purpose

The graph engine defines how a workflow is structured, how state flows between steps, and how execution
progresses. It is the substrate for both deterministic LLM pipelines and LLM-driven tool-calling agents.

#### 2. Concepts

**State.** A typed schema describing the data flowing through a graph. State is a product type (a record with
named, typed fields). Implementations MUST validate state against the schema at graph boundaries (entry, exit)
and SHOULD validate at node boundaries.

**Node.** A named unit of work. A node receives the current state and returns a partial update — a mapping
from field names to new values. Nodes MUST be asynchronous. A node MUST NOT mutate the state object it
received; it returns a new partial update which the engine merges. In languages whose typed-state
representation is effectively immutable (notably Python with Pydantic) this is directly enforceable; in
languages without value-type enforcement (notably TypeScript) implementations SHOULD defend against
accidental mutation via freezing or immutable data structures.

**Edge.** A directed connection between nodes. Edges are one of:

- **Static edge** — always routes from source node to a fixed destination.
- **Conditional edge** — a function of current state that returns the destination node name (or the sentinel
  `END`).

Each node has exactly one outgoing edge. Branching is always expressed via a conditional edge, not by
declaring multiple static edges from the same source.

**END.** An engine-provided sentinel value used as a routing target to halt execution. `END` is a distinct
engine constant, not a reserved node name, so a user node may happen to be named `"END"` without collision.

**Reducer.** A function that merges a node's partial update into the prior state for a given field. Each state
field has exactly one reducer. The default reducer is _last-write-wins_ (the new value replaces the old).
Implementations MUST provide at least: `last_write_wins`, `append` (for list-typed fields), `merge`
(for mapping-typed fields), `concat_flatten` (for list-typed fields whose updates are lists of lists —
e.g., fan-out target fields collecting list-emitting per-instance values), and `merge_all` (for
mapping-typed fields whose updates are lists of mappings — e.g., fan-out target fields collecting
dict-emitting per-instance values). Users MAY register custom reducers per field.

**`concat_flatten` semantics.** `concat_flatten(prior, update)` returns the concatenation of `prior` with the
one-level flattening of `update`. Both `prior` and `update` MUST be lists, and every element of `update` MUST
itself be a list. Violations raise `ReducerError` per §4 (the engine MUST surface the offending field, the
reducer name, and a root-cause naming the non-list value). Empty `update` is a no-op (returns `prior`
unchanged). Empty sub-lists inside `update` contribute zero elements (the one-to-many fan-out case where an
instance legitimately produces zero records). Implementations MUST NOT auto-detect whether `update` is a list
of lists vs. a flat list — `concat_flatten` is strictly the two-level reducer; callers with mixed-shape
requirements MUST register a custom reducer rather than rely on shape-dependent behavior.

**`merge_all` semantics.** `merge_all(prior, update)` folds the sequence of mappings in `update` into `prior`,
applying the same shallow merge semantics as `merge` (later writes win on key conflict; non-conflicting keys
from `prior` are preserved). For `update = [d_1, d_2, ..., d_n]`, the result is equivalent to applying `merge`
N times sequentially: `merge(merge(...merge(merge(prior, d_1), d_2)...), d_n)`, so within `update`
last-write-wins applies across all N dicts (e.g., if `d_2` and `d_n` both set key `k`, `d_n`'s value wins).
`prior` MUST be a mapping, `update` MUST be a list, and every element of `update` MUST itself be a mapping.
Violations raise `ReducerError` per §4. Empty `update` is a no-op (returns `prior` unchanged). Empty mappings
inside `update` contribute zero keys. Implementations MUST NOT auto-detect whether `update` is a list of
mappings vs. a single mapping — `merge_all` is strictly the list-of-mappings reducer; callers needing both
behaviors on the same field MUST register a custom reducer rather than rely on shape-dependent behavior.

**Subgraph.** A compiled graph used as a node inside another graph. A subgraph executes against its own state
schema and produces a partial update that is merged into the parent's state. The merge uses the same reducer
rules as ordinary nodes — parent reducers, applied to parent fields.

By default, no projection in occurs: the subgraph runs from the initial state defined by its own schema's
field defaults, independent of the parent's current state.

Projection out defaults to **field-name matching**: when the subgraph completes, the values of any subgraph
fields whose names match parent fields are merged into those parent fields via the parent's reducers.
Subgraph fields with no matching parent field are discarded.

**Explicit input/output mapping.** A subgraph-as-node MAY declare an `inputs` mapping, an `outputs` mapping,
or both:

- `inputs`: a mapping from subgraph field name → parent field name. For each entry, the parent field's
  current value is copied to the subgraph's corresponding field at entry. Subgraph fields not named in
  `inputs` receive their schema-declared default — they are NOT filled by field-name matching as a
  fallback.
- `outputs`: a mapping from parent field name → subgraph field name. For each entry, the subgraph's final
  value for the named subgraph field is merged into the corresponding parent field via the parent's
  reducer for that field. Subgraph fields not named in `outputs` are discarded — they do NOT fall through
  to field-name matching.

The two directions are independent: a subgraph-as-node MAY declare `inputs` only, `outputs` only, both, or
neither.

- When `inputs` is absent, the default above applies: no projection in. The subgraph runs from its own
  schema defaults.
- When `inputs` is present, named parent fields are copied to their mapped subgraph fields at entry; all
  other subgraph fields receive their schema-declared defaults.
- When `outputs` is absent, the default above applies: subgraph fields whose names match parent fields are
  merged back via the parent's reducers; non-matching subgraph fields are discarded.
- When `outputs` is present, it **replaces** field-name matching for projection-out: only the
  parent/subgraph field pairs named in `outputs` are merged, via the parent's reducer for the named parent
  field. All other subgraph fields are discarded.

This asymmetry — `inputs` additive, `outputs` replacement — is intentional. It reflects the asymmetry in
the defaults themselves: projection-in is off by default (so `inputs` turns it on for listed fields), while
projection-out is on by default via field-name matching (so `outputs` replaces it to avoid ambiguous mixed
rules).

Compilation MUST fail with category `mapping_references_undeclared_field` if an `inputs` mapping names a
parent field that is not declared in the parent's state schema, or a subgraph field that is not declared in
the subgraph's state schema. The same rule applies symmetrically to `outputs`. Implementations SHOULD
validate at compile time that the types of mapped parent/subgraph field pairs are compatible (per the
language's type system's notion of compatibility); this is SHOULD rather than MUST because type-system
expressiveness varies across languages.

**Compiled graph.** The result of compiling a graph definition. A compiled graph is immutable and executable.
The entry node MUST be declared explicitly by the graph author — there is no implicit "first node added"
default. Compilation MUST fail with a diagnostic error if the graph has: no declared entry node, unreachable
nodes, dangling edges (references to nonexistent nodes), a node with more than one outgoing edge, or a field
with more than one declared reducer.

When reporting a compile-time error, implementations MUST expose one of the following canonical category
identifiers (as an error class, error code, or tagged discriminant, per the language's idiom):

- `no_declared_entry` — no entry node was declared.
- `unreachable_node` — a declared node has no path from the entry.
- `dangling_edge` — an edge references a node name that is not declared.
- `multiple_outgoing_edges` — a node has more than one outgoing edge.
- `conflicting_reducers` — a state field has more than one declared reducer.
- `mapping_references_undeclared_field` — a subgraph-as-node `inputs` or `outputs` mapping names a field
  not declared in the relevant state schema.

### Capability: `pipeline-utilities`

#### 1. Purpose

The pipeline-utilities capability defines a layer of cross-cutting concerns that compose with the
graph-engine without modifying the engine. This first version specifies **middleware** — wrappers
around node execution — and two canonical middleware as concrete instances: **retry** and
**timing**. Both are mandated as part of the pipeline-utilities surface (§6) because their shape
is non-obvious enough to warrant a normative contract; other middleware-shaped concerns (logging,
resource lifecycle, circuit breakers) are implementable as middleware but are not spec-mandated.

Middleware solves the problem of code that should run around many node invocations without being
duplicated in each node's body. Retry, timing, logging, instrumentation, and resource lifecycle are
all middleware-shaped. Observer hooks (graph-engine §6) cover read-only observation of what
happened; middleware covers control over what happens.

The pipeline-utilities capability composes on top of graph-engine. It does NOT modify graph-engine
behavior — middleware sits between the engine's "node dispatch" step and the user's node function,
and is invisible to nodes that don't opt into middleware.

#### 2. Concepts

**Middleware.** An async callable with the shape:

```
async def middleware(state, next) -> partial_update
```

where:

- `state` is the input state the wrapped node would have received (the engine's pre-merge state at
  the time of node dispatch).
- `next` is an async callable taking a single argument (the state to pass to the next layer or the
  original node) and returning the partial update from that layer.
- The middleware MUST return a partial update — a mapping of field names to new values, the same
  shape a node returns.

A middleware MAY:

- Call `next(state)` to invoke the wrapped chain, optionally inspecting or transforming the input
  state first (the transformed state is passed to `next`, NOT to the engine's merge step).
- Inspect, augment, or replace the returned partial update before returning it.
- Short-circuit by NOT calling `next` and returning its own partial update. The rest of the chain
  — subsequent middleware and the wrapped node — does not execute, and this middleware's own
  post-phase (code following `await next(...)`) is skipped. See "Pre-node and post-node phases"
  below for the dual-phase model that makes this possible.
- Catch exceptions raised by `next(state)` and either re-raise, transform, or recover (returning a
  partial update instead of raising).
- Call `next` more than once (e.g., retry middleware). The state passed to subsequent calls MAY be
  the original or a transformed version; the middleware decides.

A middleware MUST NOT:

- Mutate the input `state` object. The same immutability contract that applies to nodes
  (graph-engine §2 Node) applies to middleware. Pass a new state to `next` if a transformation is
  needed.
- Side-effect on engine internals (the reducer registry, edge map, etc.). Middleware operates only
  through the `state` and `next` it receives and the partial update it returns.

**Middleware chain.** An ordered sequence of middleware applied to a single node. The chain composes
outer-to-inner: the first middleware in the chain runs first, calls `next(state)` to invoke the
second, and so on, with the original node at the inner end.

For a chain `[m1, m2, m3]` wrapping node `n`, execution proceeds:

```
m1 sees state, calls next(s) ────► m2 sees state, calls next(s) ────► m3 sees state, calls next(s)
                                                                                  │
                                                                                  ▼
                                                                                 n(state) → partial_update
                                                                                  │
m1 returns partial_update ◄──── m2 returns partial_update ◄──── m3 returns partial_update
```

Each middleware's return value flows back through the previous layer's `next` call return.

**Pre-node and post-node phases.** A middleware function has two phases separated by
`await next(...)`. Code *before* `await next` is the **pre-node phase**, running on the way *into*
the chain (left-to-right in the diagram); code *after* `await next` returns is the **post-node
phase**, running on the way *out* (right-to-left). The wrapped node always runs at the innermost
point — it is never reached partway through the chain.

The two phases are tied to a single position in the chain: if `m1` is outermost, `m1`'s pre-phase
runs first AND `m1`'s post-phase runs last. Pre-order and post-order are not configured
independently. Concretely, a middleware function carries both phases:

```
async def my_middleware(state, next):
    # ── pre-node phase: runs on the way IN ──
    started_at = time.time()

    partial_update = await next(state)   # the rest of the chain (and eventually the node) runs here

    # ── post-node phase: runs on the way OUT ──
    log(f"node took {time.time() - started_at}s")
    return partial_update
```

This is the standard middleware shape used by Express, Koa, ASGI, Tower, Django middleware, and
similar frameworks.

### Capability: `llm-provider`

#### 1. Purpose

The LLM provider capability defines a uniform request/response surface for sending messages to a
Large Language Model and receiving its response. It is the substrate every higher-level LLM
capability composes against — tool systems, prompt management, evaluation harnesses, agent loops.

The substrate is intentionally narrow:

- A provider is **stateless**. It does not maintain conversation history; the caller passes the full
  message list on every call.
- A provider does **not** loop on tool calls. If the assistant returns tool calls, the caller is
  responsible for executing the tools and making a follow-on `complete()` with the results.
- A provider does **not** handle retry, rate limiting, fallback, or routing. Those are pipeline-
  utilities concerns and compose above the provider via middleware.
- A provider is **bound to a single model identifier**. Switching models means constructing a new
  provider, not passing a different argument per call. (Implementations MAY offer convenience
  factories that produce per-model providers from shared credentials; that is a constructor concern,
  not a behavioral one.)

Every constraint above is a deliberate scope cut. The narrower the provider surface, the easier it is
to swap implementations, mock for tests, and stack pipeline utilities on top.

**Transparency.** Per charter §3.1 principle 8 ("Transparency over abstraction"), the provider
abstraction surfaces a normalized shape — `Message`, `Tool`, `Response` — without hiding what the
underlying provider returned. The `Response` record carries the parsed provider response verbatim
alongside the normalized fields (§6 `raw`), and the §7 error categories preserve the underlying
provider exception as cause. Users who need provider-specific fields (logprobs, content-filter
details, vendor-specific extensions) reach through the abstraction directly; structure is added,
never removed.

#### 2. Concepts

**Message.** A typed entry in a conversation. The four message kinds are `system`, `user`,
`assistant`, and `tool`. Each kind carries kind-specific content as defined in §3.

**Tool.** A function the model may request the user execute. A tool definition is a record of `name`,
`description`, and `parameters` (a JSON Schema describing the argument shape).

**Tool call.** A request from an assistant message to invoke a named tool with structured arguments.
The user is responsible for executing the tool and returning the result via a `tool` message bearing
the corresponding `tool_call_id`.

**Provider.** An object that, given a sequence of messages and an optional set of tools, returns a
single assistant message wrapped in a `Response`. A provider is bound to a specific model identifier.

**Response.** The result of a provider call: the assistant message, a finish reason, and usage
information.

### Capability: `observability`

#### 1. Purpose

The observability capability defines normative mappings from OpenArmature's runtime event surface
(graph-engine §6 observer events, specifically the v0.6.0 started/completed event pairs) into
well-known external observability backends. The substrate is provider-neutral; the capability is
where each concrete backend's translation lives.

This spec defines two concrete backend mappings: the **OpenTelemetry** mapping in §3–§7 and the
**Langfuse** mapping in §8. Future proposals add additional backends as further sibling sections
of this same spec; the OTel mapping serves as the reference shape for cross-backend equivalence.

The capability does NOT introduce new graph-engine primitives. It consumes the existing observer
event stream — `started` events open spans, `completed` events close them. An implementation that
emits OTel spans (or Langfuse observations, per §8) is built on top of §6, not into the engine.

#### 2. Concepts

**Span.** A unit of work in OTel — a logically distinct interval with a name, start/end timestamps,
status, attributes, and parent-child relationships. The mapping translates each user-meaningful unit
of work in a graph invocation (the invocation itself, each subgraph, each node execution, each fan-
out instance) into a span.

**Span attributes.** Key/value pairs attached to a span. OTel attribute values are restricted to
scalar types (string, int, float, bool) and arrays thereof. The mapping uses dotted-key namespaces
under the prefix `openarmature.`.

**Span status.** OTel spans carry a status of `OK`, `ERROR`, or `UNSET`. The mapping translates
graph-engine §4 error categories into status `ERROR` with a category-bearing description.

**Trace.** OTel's term for a complete tree of spans rooted at a single trace ID. By default, one
outermost graph invocation produces one trace; subgraphs (whether composed via
`add_subgraph_node` or instantiated by a fan-out per pipeline-utilities §9) participate in the
parent invocation's trace as nested spans. Implementations MUST also support an opt-in
**detached** mode for specific subgraphs or fan-outs (§4.4), where the subgraph or fan-out gets
its own trace and the parent's dispatch span carries an OTel `Link` to that new trace.

**Correlation ID.** A per-invocation identifier that flows across observability backends.
Distinct from `invocation_id` — the `invocation_id` (caller-supplied or framework-generated, per
§5.1) correlates spans within a single backend, while `correlation_id` is application-supplied
(or auto-generated when absent)
and is intended to be visible in every backend the implementation emits to. A user running an
LLM workflow with both an OTel backend (system traces, logs) and a Langfuse backend
(LLM-specific traces) uses the `correlation_id` as a join key between them: find a slow request
in Langfuse, search for its `correlation_id` in OTel logs, and see the surrounding
infrastructure activity. See §3 (architectural contract), §5.6 (OTel attribute realization),
and §8.5 (Langfuse attribute realization).

### Capability: `prompt-management`

#### 1. Purpose

The prompt-management capability defines the contract by which named, versioned templates
are fetched from one or more backends, rendered with caller-supplied variables, and turned
into LLM-ready message sequences. The spec establishes the contracts; implementations and
sibling-package backends ship the concrete forms.

The capability composes with the llm-provider capability (a `PromptResult` carries
`Message` records per llm-provider §3) and with the observability capability (rendered
prompts carry stable identity that observer events MAY surface).

This capability does NOT define:

- The templating language or syntax (Jinja2 in Python, handlebars / template literals in
  TypeScript — per implementation).
- Specific backend implementations beyond a minimum local-filesystem reference.
- Prompt versioning workflows (the spec defines a `version` field on `Prompt`; how
  versions are assigned, incremented, or pinned is per-project discipline).
- Cache invalidation policies (the spec defines hashes that user code MAY use as cache
  keys; the cache itself is out of scope).

#### 2. Concepts

**Prompt.** An unrendered template plus its identity metadata. A prompt is what a backend
returns from a fetch; it carries enough information to be rendered, traced, and
content-addressed without a backend round-trip.

**PromptResult.** The rendered output of applying variables to a prompt. Carries the
rendered `Message` sequence (per llm-provider §3) plus the prompt's identity metadata
(propagated from the source `Prompt`) plus a `rendered_hash` that captures the rendered
content.

**PromptManager.** The user-facing API. Composes one or more `PromptBackend`s and exposes
fetch + render operations. Users interact with the manager; backends are an
implementation detail of the manager's construction.

**PromptBackend.** The protocol implementations and sibling packages plug into. Defines a
single operation: fetch a prompt by name and label. Backends do not render; rendering is
the manager's concern.

**PromptGroup.** A composition pattern for tracing related prompts together: an ordered
sequence of `PromptResult` instances that should appear under one logical grouping in
observability. The canonical N=2 case is "classifier + follow-up"; longer chains
(multi-stage classification, RAG with reranking, self-correction loops, map-reduce over
chunks) work under the same primitive. The group is a thin wrapper over its members and
a span-grouping convention; it is not a fetch or render primitive and performs no
orchestration.

**Fetch vs. render distinction.** Fetching retrieves the template; rendering applies
variables. Splitting the two operations lets users:

- Inspect a template without binding variables (useful for tooling, schema validation,
  prompt-version diffs).
- Cache templates separately from rendered output (template fetch is the I/O-bound step;
  rendering is local).
- Render the same template with different variables in tight loops without re-fetching.

A convenience operation that combines fetch + render is permitted (see §6) but the spec
treats fetch and render as separable.

## Patterns

_Recipes that compose the primitives. Not framework contracts — these are how to do common things idiomatically._

### Bypass if output exists

**Problem.** How do I skip a node whose external output already
exists?

#### Approach

A small custom [middleware](https://openarmature.ai/concepts/middleware/) wraps the
node. Before calling `next_(state)`, the middleware checks "does
my output already exist?" (a filesystem file, a database row, a
content-addressable store entry). If yes, it returns the cached
output as the partial update directly. If no, it calls `next_`
and returns the result.

The node sees its normal `(state) → partial_update` contract.
The middleware is the only thing that knows about idempotency;
all callers of the node compose with it cleanly.

#### Snippet

```python
import os
from collections.abc import Mapping
from typing import Any
from openarmature.graph import GraphBuilder, NextCall, State


class BypassIfRendered:
    """Skip the node if its rendered output already exists on disk."""

    def __init__(self, output_field: str, key_field: str, root: str):
        self.output_field = output_field
        self.key_field = key_field
        self.root = root

    async def __call__(
        self, state: Any, next_: NextCall
    ) -> Mapping[str, Any]:
        key = getattr(state, self.key_field)
        path = f"{self.root}/{key}.bin"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return {self.output_field: f.read()}
        partial = await next_(state)
        # ... persist partial[self.output_field] to path here, or
        #     have the node itself write the file ...
        return partial


class RenderState(State):
    scene_id: str
    rendered_frame: bytes = b""


builder = (
    GraphBuilder(RenderState)
    .add_node(
        "render",
        render_frame_fn,
        middleware=[
            BypassIfRendered(
                output_field="rendered_frame",
                key_field="scene_id",
                root="./renders",
            )
        ],
    )
    # ... rest of graph ...
)
```

The middleware composes with the framework's
[four registration sites](https://openarmature.ai/concepts/middleware/): attach it
per-node (as above), per-graph, per-branch, or
per-fan-out-instance, depending on the scope of the bypass.

#### When this is the right pattern

- The node's work is expensive and idempotent given the same key
  (rendering a frame, calling an external API with content-
  addressable output, downloading a file).
- The "does it exist" check is cheap (a filesystem `stat`, a
  Redis `EXISTS`, a database key lookup).
- You're OK with the node being skipped silently — the partial
  update returned by the middleware is indistinguishable from a
  successful node run.

#### When it isn't

- The check itself is expensive enough that you'd rather just run
  the node. The cost model inverts; the pattern is wrong.
- You need to *force* re-execution on demand (cache invalidation).
  Add a `force_rerun: bool` field on state that the middleware
  consults — but if you're doing that often, the bypass logic
  belongs in the node itself, gated on a state field, not in
  middleware.
- The cached output's freshness depends on inputs the middleware
  can't see (downstream state, time-of-day, etc.). Use a
  dedicated caching layer instead of reimplementing cache
  invalidation in the middleware.

#### Cross-references

- [Middleware](https://openarmature.ai/concepts/middleware/) — middleware shape, the
  four registration sites, composition.
- Spec: [pipeline-utilities](https://openarmature.org/capabilities/pipeline-utilities/)

This pattern is explicitly called out in proposal 0008's
*Alternatives considered* section as a userland recipe rather than
spec'd behavior — this page is its canonical home.

### Parameterized entry point

**Problem.** How do I start the graph at an arbitrary node?

#### Approach

You don't. Make the "entry point" a state-level parameter instead.
A first router node passes through, and a
[conditional edge](https://openarmature.ai/concepts/composition/) routes to wherever
execution should begin. The graph stays a single graph; what
differs across runs is which branch the conditional edge takes.

Combine with [checkpointing](https://openarmature.ai/concepts/checkpointing/) if you
want resume-style behavior — skip nodes whose work is already
captured in state.

#### Snippet

```python
from openarmature.graph import END, EndSentinel, GraphBuilder, State


class MissionState(State):
    starting_stage: str = "plan"  # "plan" | "execute" | "report"
    plan: str = ""
    execution_log: str = ""
    report: str = ""


def route_from_starting_stage(s: MissionState) -> str | EndSentinel:
    return s.starting_stage


async def router(s: MissionState) -> dict:
    return {}  # no state change; conditional edge below routes


async def plan(s: MissionState) -> dict:
    return {
        "plan": "Apollo-style free-return trajectory.",
        "starting_stage": "execute",
    }


async def execute(s: MissionState) -> dict:
    return {"execution_log": "Burn complete. Trajectory nominal."}


async def report(s: MissionState) -> dict:
    return {"report": "Mission objectives met."}


builder = (
    GraphBuilder(MissionState)
    .add_node("router", router)
    .add_node("plan", plan)
    .add_node("execute", execute)
    .add_node("report", report)
    .add_conditional_edge("router", route_from_starting_stage)
    .add_edge("plan", "execute")
    .add_edge("execute", "report")
    .add_edge("report", END)
    .set_entry("router")
)
graph = builder.compile()

### Start at the beginning:
await graph.invoke(MissionState())

### Or skip straight to execute, with the plan already in state:
await graph.invoke(MissionState(starting_stage="execute", plan="..."))
```

The caller pre-populates `starting_stage` (and any prerequisite
fields the chosen branch needs) and the graph routes accordingly.

#### When this is the right pattern

- You have a few canonical entry points and the choice between
  them is data, not control flow.
- You want to skip work already done in a prior run — combine with
  [checkpointing](https://openarmature.ai/concepts/checkpointing/) to pick up where
  you left off.
- Your "different entry points" share state structure and most of
  the downstream graph.

#### When it isn't

- "Start at node X" really means "run a different pipeline." Then
  it's a different compiled graph. Don't bend one graph into two;
  two graphs are easier to test and reason about.
- The number of entry points grows unboundedly. Then you're
  reimplementing routing — consider a higher-level dispatch layer
  that picks which graph to invoke.

#### Cross-references

- [Composition: conditional edges](https://openarmature.ai/concepts/composition/)
- [Checkpointing](https://openarmature.ai/concepts/checkpointing/)
- Spec: [graph-engine](https://openarmature.org/capabilities/graph-engine/)

### Session as checkpoint resume

**Problem.** How do I keep multi-turn agent state across turns?

#### Approach

The framework's [checkpointing](https://openarmature.ai/concepts/checkpointing/)
provides single-invocation crash resume out of the box. Multi-turn
state is the same primitive used differently: the application
keeps a stable `session_id → invocation_id` mapping, and each
turn calls `invoke(resume_invocation=<prior_invocation_id>)` to
pick up where the previous turn left off.

The checkpointer returns the prior state. The new turn proceeds
from there. Session-context fields that accumulate across turns
(message history, retrieved facts, running totals) use a `merge`
or `append` reducer so each turn's contribution adds to what's
already there rather than replacing it.

Each resume mints a new `invocation_id`; the `session_id` is the
join key the application maintains, typically as the
`correlation_id` on `invoke()` (which is preserved unchanged
across resume).

#### Snippet

```python
from typing import Annotated
from pydantic import Field
from openarmature.checkpoint import SQLiteCheckpointer
from openarmature.graph import END, GraphBuilder, State, append, merge
from openarmature.llm import Message


class SessionState(State):
    messages: Annotated[list[Message], append] = Field(default_factory=list)
    facts: Annotated[dict[str, str], merge] = Field(default_factory=dict)
    last_user_input: str = ""


### ... define nodes that read s.messages, append to s.messages,
###     and merge into s.facts ...

checkpointer = SQLiteCheckpointer(path="./sessions.db")
graph = (
    GraphBuilder(SessionState)
    .add_node("plan", plan)
    .add_node("respond", respond)
    .add_edge("plan", "respond")
    .add_edge("respond", END)
    .set_entry("plan")
    .with_checkpointer(checkpointer)
    .compile()
)


### The application maintains its own session table mapping
### session_id -> latest invocation_id. OA's checkpointer doesn't
### know about sessions; the join is the application's
### responsibility. The session_id doubles as correlation_id so
### observability traces share the cross-turn join key.
async def handle_turn(session_id: str, user_input: str) -> str:
    initial = SessionState(last_user_input=user_input)
    prior_invocation_id = sessions_db.get_invocation_id(session_id)

    if prior_invocation_id is None:
        final = await graph.invoke(initial, correlation_id=session_id)
    else:
        final = await graph.invoke(
            initial, resume_invocation=prior_invocation_id
        )

    # Record the new invocation_id for next turn's resume.
    # Read it from the checkpointer's latest record for this
    # correlation_id; exact lookup is application-side bookkeeping.
    sessions_db.set_invocation_id(session_id, latest_for(session_id))

    return final.messages[-1].content
```

`sessions_db` is your application's session-state store (Postgres,
Redis, a flat file, whatever); the checkpointer holds the OA-side
state and the session table holds the join keys.

#### When this is the right pattern

- Your application has long-lived sessions with multiple LLM turns
  and you want the prior state to be the starting point of the
  next turn.
- You're already running a checkpointer for crash resume — this
  pattern is "use it more."
- Cross-turn state has clean reducer semantics: `merge` for
  accumulating dicts, `append` for growing lists.

#### When it isn't

- A session's "state" is bigger than fits comfortably in a single
  graph state shape. Split into multiple graphs and share an
  external store keyed by session.
- Turns are completely independent — there's no value in carrying
  state across them. Then just run each turn as a fresh invoke.
- The application already has its own state-management layer that
  conflicts with OA's frozen-state model. Use OA per-turn without
  cross-turn resume.

#### Cross-references

- [Checkpointing](https://openarmature.ai/concepts/checkpointing/) — backend wiring,
  `resume_invocation`, schema migration.
- [State and reducers](https://openarmature.ai/concepts/state-and-reducers/) — `merge`
  and `append` reducer strategies.
- [`examples/08-checkpointing-and-migration`](https://openarmature.ai/examples/08-checkpointing-and-migration/) —
  single-resume baseline.
- Spec: [pipeline-utilities](https://openarmature.org/capabilities/pipeline-utilities/)

### Tool dispatch as node

**Problem.** How do I run an agent tool-call loop?

#### Approach

A node reads the assistant's last `tool_calls` from the running
message list, dispatches each to a local Python function, appends
`ToolMessage` records back to the message list via an
[`append` reducer](https://openarmature.ai/concepts/state-and-reducers/), and a
[conditional edge](https://openarmature.ai/concepts/composition/) loops back to the
LLM node if the model wants more turns. The exit is the
conditional edge routing to a `present` node (or `END`) when the
assistant returns no `tool_calls`.

No "agent framework" abstraction — the loop is just a graph cycle
on top of [`Tool`, `ToolCall`, `ToolMessage`](https://openarmature.ai/concepts/llms/).

#### Snippet

```python
import json
from typing import Annotated
from pydantic import Field
from openarmature.graph import END, EndSentinel, GraphBuilder, State, append
from openarmature.llm import AssistantMessage, Message, Tool, ToolMessage


class AgentState(State):
    messages: Annotated[list[Message], append] = Field(default_factory=list)
    turn: int = 0


TOOLS = [
    Tool(
        name="lookup_mission",
        description="Look up Apollo or Artemis mission facts.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
]
MAX_TURNS = 5


async def call_llm(s: AgentState) -> dict:
    response = await provider.complete(s.messages, tools=TOOLS)
    return {"messages": [response], "turn": s.turn + 1}


async def dispatch_tools(s: AgentState) -> dict:
    assistant = s.messages[-1]
    assert isinstance(assistant, AssistantMessage)
    results: list[Message] = []
    for tc in assistant.tool_calls or ():
        output = await dispatch_one(tc.name, tc.arguments)  # str or JSON-serializable
        content = output if isinstance(output, str) else json.dumps(output)
        results.append(ToolMessage(content=content, tool_call_id=tc.id))
    return {"messages": results}


def route_after_llm(s: AgentState) -> str | EndSentinel:
    if s.turn >= MAX_TURNS:
        return "present"
    last = s.messages[-1]
    if isinstance(last, AssistantMessage) and last.tool_calls:
        return "dispatch_tools"
    return "present"


async def present(s: AgentState) -> dict:
    return {}  # final formatting / output


builder = (
    GraphBuilder(AgentState)
    .add_node("call_llm", call_llm)
    .add_node("dispatch_tools", dispatch_tools)
    .add_node("present", present)
    .add_conditional_edge("call_llm", route_after_llm)
    .add_edge("dispatch_tools", "call_llm")
    .add_edge("present", END)
    .set_entry("call_llm")
)
graph = builder.compile()
```

The `MAX_TURNS` cap prevents runaway loops; the conditional edge
short-circuits to `present` when the cap is hit or when the model
returns no `tool_calls`.

See [`examples/09-tool-use`](https://openarmature.ai/examples/09-tool-use/) for a
runnable version with full tool definitions, defensive handling
for malformed `ToolCall.arguments`, and trace output.

#### When this is the right pattern

- The model needs to call local Python functions and react to
  their results.
- The loop is bounded — either by `MAX_TURNS`, by the model
  signaling it's done, or by both.
- Tool results are textual or JSON-serializable and fit cleanly
  into `ToolMessage.content`.

#### When it isn't

- Tools have side effects you can't replay safely on resume. Wrap
  each side-effecting tool with the
  [bypass-if-output-exists](#bypass-if-output-exists) pattern so
  a crashed run resumes without re-side-effecting.
- The "tools" are long-running async pipelines, not function
  calls. Model them as subgraphs and let the LLM node route via
  conditional edge to the right subgraph; the loop shape is the
  same but each "tool" is a full pipeline.
- You need streaming tool results back to the model mid-call. The
  current `Tool` / `ToolMessage` shape is request/response;
  streaming is out of scope for this pattern.

#### Cross-references

- [LLMs concept page](https://openarmature.ai/concepts/llms/) — `Tool`, `ToolCall`,
  `ToolMessage` types and the `complete(messages, tools=...)`
  contract.
- [State and reducers](https://openarmature.ai/concepts/state-and-reducers/) —
  `append` reducer semantics.
- [`examples/09-tool-use`](https://openarmature.ai/examples/09-tool-use/) — runnable
  reference implementation.
- Spec: [llm-provider](https://openarmature.org/capabilities/llm-provider/)

## Non-obvious shapes

Recipes that aren't deducible from the API surface alone. The primitives docs tell you what's possible; this section tells you what's smart.

### Declare a non-clobbering reducer on accumulator list fields

State fields default to `last_write_wins` — each node's write replaces the prior value for that field. For scalar fields (`status: str`, `count: int`) that's usually what you want. For list fields that accumulate contributions across multiple nodes (`messages: list[Message]`, `events: list[Event]`, `results: list[Result]`), it's the wrong default — every node's contribution silently clobbers everything before it.

Declare `append` (or another non-clobbering reducer) at the state class:

```python
from typing import Annotated
from pydantic import Field
from openarmature.graph import State, append

class WorkflowState(State):
    messages: Annotated[list[Message], append] = Field(default_factory=list)
    events: Annotated[list[Event], append] = Field(default_factory=list)
    final_status: str = "pending"   # last_write_wins is fine here
```

The failure mode without `append` is silent and easy to misdiagnose — the final state shows only the last node's contribution to the list, with no error. Common "why is my accumulator empty?" question. `merge` is the equivalent for `dict[str, V]` fields that accumulate keys across nodes.

### Branch on `Response.finish_reason` before reading `message.content`

After `await provider.complete(messages, tools=[...])` returns, the shape of `Response` varies by `finish_reason`:

- `finish_reason == "stop"` — assistant produced a content response. `message.content` carries the text; `message.tool_calls` is empty.
- `finish_reason == "tool_calls"` — assistant emitted tool calls. `message.tool_calls` carries the list; `message.content` is typically empty (model didn't say anything beyond the tool calls).
- `finish_reason == "length"` / `"content_filter"` / `"error"` — completion was cut off or refused; `message.content` may be partial or empty.

Post-LLM logic that reads `message.content` without checking `finish_reason` misses the entire tool-calling path:

```python
response = await provider.complete(messages, tools=tools)

if response.finish_reason == "tool_calls":
    # Dispatch each tool call, append ToolMessage responses, re-call complete()
    for tc in response.message.tool_calls:
        result = dispatch_tool(tc.name, tc.arguments)
        messages.append(ToolMessage(content=result, tool_call_id=tc.id))
    response = await provider.complete(messages, tools=tools)
elif response.finish_reason == "stop":
    handle_text(response.message.content)
else:
    handle_error_or_partial(response)
```

The discriminator is one branch; missing it gives you empty data on tool-call responses and silently wrong behavior on truncations.

### `disable_llm_payload` defaults to `True` — flip it for LLM-aware observability backends

The `OTelObserver` (and any spec-conformant observer reading LLM events) defaults `disable_llm_payload: bool = True` per spec §5.5's "default-off by privacy" framing. Without flipping the flag, LLM spans carry GenAI semconv attributes (token counts, model name, finish reason) but NOT the message payload (input messages, response content, request extras).

That's the right default for general OpenArmature use — payloads may contain PII the user hasn't audited, and storage cost grows with prompt size. But it's the WRONG default if you're wiring up an LLM-aware observability backend (Langfuse, Phoenix, Honeycomb's LLM lens) that renders the message stream as part of its generation view. Backends will show "empty" generations and you'll wonder why.

Flip the flag once at observer construction:

```python
from openarmature.observability import OTelObserver

observer = OTelObserver(
    span_processor=your_exporter,
    disable_llm_payload=False,   # opt in to message-payload attributes
)
compiled.attach_observer(observer)
```

The companion `disable_genai_semconv` flag defaults to `False` — GenAI semconv attributes emit by default since they're how LLM-aware backends render anything at all. Don't flip that one unless you're routing GenAI emission through a different layer.

### Use the bundled `FilesystemCheckpointer` or `SQLiteCheckpointer`, not a hand-rolled serializer

The temptation when persisting graph state is to `json.dumps(state.model_dump())` and write to a file. Don't. The shipped Checkpointer backends handle every contract `openarmature.checkpoint.Checkpointer` defines — round-trip integrity, `parent_states` for inner-save resume, fan-out progress tracking, schema-version migration, listing by `correlation_id`, `CheckpointRecordInvalid` on shape drift. A hand-rolled serializer that "works" on the happy path silently fails the moment a fan-out crash leaves an in-flight save record, and you'll be debugging it for hours before realizing the bundled backend exists.

If your storage requirement isn't local disk (`FilesystemCheckpointer`) or local SQLite (`SQLiteCheckpointer` — also supports `:memory:` and arbitrary file paths), implement the `Checkpointer` Protocol against your backend rather than wrapping state serialization yourself. Custom backends inherit the spec's correctness contract for free.

### Subgraphs > conditional-edge spaghetti when branches don't share state

A common shape is "after this LLM call, route to either a JSON-extraction node or a tool-dispatch node depending on `finish_reason`." The naive solution is two conditional edges from the LLM node, one to each downstream. That works for two branches; it scales poorly past three.

When the branches operate on different sub-shapes of state — e.g., one path is "extract JSON, then validate" while another is "dispatch tools, loop until done, then summarize" — encapsulate each as a `SubgraphNode` and route from the LLM node to the right subgraph. Each subgraph has its own state schema (projected from the parent), its own entry node, and its own internal topology. The parent graph becomes a switchboard with a few edges; the complexity lives one layer down where it composes cleanly.

### Be explicit with `tool_choice`; don't trust the provider's default

`Provider.complete(messages, tools, tool_choice=...)` accepts `"auto"`, `"required"`, `"none"`, or a `ForceTool(name=...)` record. When you omit `tool_choice`, the OpenAI provider's own default applies — usually `"auto"` when `tools` is non-empty, but documented per-provider. A pipeline that wants deterministic tool-calling (a routing node that MUST produce a tool call, a guarded LLM call that MUST NOT call tools) should pin `tool_choice` explicitly rather than relying on the provider default.

Pre-send validation catches the three §5 failure modes (`required` with empty tools, `ForceTool` with empty tools, `ForceTool.name` not in tools) and raises `ProviderInvalidRequest` before the HTTP call. Not all providers honor `tool_choice` — confirm with your provider's docs — but the OpenAI-compatible mapping is in `OpenAIProvider`.

### Always `await graph.drain()` in short-lived processes; supply a `timeout` if observers might hang

`CompiledGraph.invoke()` returns when the graph reaches END or raises; observer events are dispatched onto a per-invocation queue and delivered by a background worker. The graph's execution loop never awaits observer processing. In a long-running service this is invisible — the worker drains naturally. In a CLI, script, or serverless function, the process exits before the worker finishes, and any late observer events (typically the last node's `completed` event plus any `checkpoint_saved` events) get dropped.

Always call `await graph.drain()` before the short-lived process exits. If your observer set includes anything that might hang (a metrics observer with a flaky network endpoint, an OTel exporter behind a slow OTLP collector), supply a `timeout`:

```python
summary = await graph.drain(timeout=5.0)
if summary.timeout_reached:
    log.warning("drain truncated: %d events undelivered", summary.undelivered_count)
```

The compiled graph stays usable for subsequent invocations after a timed-out drain — workers are cancelled cleanly, no partial state leaks.

### `install_log_bridge` skips its own handler when the application already attached one to the same `LoggerProvider`

Two distinct classes both named `LoggingHandler` exist in the OTel Python ecosystem and both bridge stdlib log records to the OTel Logs SDK:

- `opentelemetry.sdk._logs.LoggingHandler` (the SDK class). Typically attached by an application's own logging setup — e.g., a FastAPI `setup_logging(...)` step that wires up an OTLP-backed `LoggerProvider` for log export.
- `opentelemetry.instrumentation.logging.handler.LoggingHandler` (the instrumentation class). What `openarmature.observability.otel.install_log_bridge` attaches when it runs.

Different classes, same OTel-Logs export path. If both are attached against the same `LoggerProvider`, every stdlib log record fires through both handlers, both call `provider.get_logger(...).emit(...)`, and `BatchLogRecordProcessor` ships the record TWICE to the OTLP endpoint. The duplication is OTLP-only — a console handler attached separately is unaffected, which makes "OTLP rows are doubled, console isn't" a head-scratcher to diagnose.

`install_log_bridge` detects either handler class against the same provider and skips its own `addHandler` accordingly; the `openarmature.correlation_id` LogRecord factory still installs. The check is provider-scoped, so an application that intentionally attaches a handler against a DIFFERENT `LoggerProvider` (a separate logs pipeline) still gets the OA bridge against the OA provider — the helper only dedups when the SAME provider would receive duplicate emissions.

### Three exception hierarchies; know which one your code catches

`openarmature` exceptions split across three sibling hierarchies:

- `RuntimeGraphError` (in `openarmature.graph`) — node execution failures: `NodeException`, `RoutingError`, `EdgeException`, `ReducerError`, `StateValidationError`. Each has a `category` string matching the spec's canonical error categories.
- `CheckpointError` (in `openarmature.checkpoint`) — persistence failures: `CheckpointNotFound`, `CheckpointSaveFailed`, `CheckpointRecordInvalid`, `CheckpointStateMigrationMissing`, `CheckpointStateMigrationFailed`, `CheckpointStateMigrationChainAmbiguous`.
- `LlmProviderError` (in `openarmature.llm`) — provider call failures: `ProviderAuthentication`, `ProviderInvalidRequest`, `ProviderInvalidResponse`, `ProviderInvalidModel`, `ProviderModelNotLoaded`, `ProviderRateLimit`, `ProviderUnavailable`, `ProviderUnsupportedContentBlock`, `StructuredOutputInvalid`.

Catching `Exception` works but is too broad; catching one hierarchy misses the other two. If you want to branch on category strings (e.g., for retry logic), catch the relevant base — `RuntimeGraphError` covers all five spec runtime categories, `LlmProviderError` covers all nine provider categories, `CheckpointError` covers all six checkpoint categories. The `TRANSIENT_CATEGORIES` frozenset in `openarmature.llm` enumerates which provider categories are retriable.

### Reconcile `started` → `completed` pairs via a per-invocation dict keyed on `(namespace, branch_name, attempt_index, fan_out_index)`

Observers receive `started` and `completed` events as a pair per node attempt, but the engine doesn't carry a `step_id`-like correlation field across the pair (it doesn't need one for its own logic — the events arrive serially per spec §6). Observer code that needs to thread per-call state — start timestamps, request payloads, custom IDs — between the two events has to reconcile manually.

The pair identity is `(namespace, branch_name, attempt_index, fan_out_index)`: that tuple is unique within an invocation (per graph-engine §6 uniqueness invariants — `branch_name` and `fan_out_index` are independent slots, so a node inside a parallel-branches branch needs `branch_name` in the key to avoid colliding with the same-named node in a sibling branch). Carry per-invocation state in a `dict[invocation_id, dict[tuple, value]]` and look up on `completed`:

```python
class StepTimingObserver:
    def __init__(self) -> None:
        # invocation_id -> {(namespace, branch_name, attempt_index, fan_out_index): start_ts}
        self._pending: dict[str, dict[tuple[Any, ...], float]] = {}

    async def __call__(self, event: NodeEvent) -> None:
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        key = (event.namespace, event.branch_name, event.attempt_index, event.fan_out_index)
        if event.phase == "started":
            self._pending.setdefault(invocation_id, {})[key] = time.monotonic()
        elif event.phase == "completed":
            start = self._pending.get(invocation_id, {}).pop(key, None)
            if start is not None:
                duration = time.monotonic() - start
                # … emit timing
            # Sweep when the dict empties (last completed for this invocation).
            if not self._pending.get(invocation_id):
                self._pending.pop(invocation_id, None)
```

The `_pending[invocation_id]` sub-dict naturally tracks in-flight pairs and drains as completions arrive. Sweep the outer entry when the sub-dict empties so long-running services don't accumulate per-invocation entries. If you also subscribe to drain events, that's another sweep opportunity. The same pattern works for any per-call state the observer needs to thread across the pair.

### Filter `openarmature.*`-namespaced events when your observer only cares about user nodes

OA emits observer events under sentinel node-names for its own internal dispatch: `openarmature.llm.complete` for LLM provider calls (proposal 0024), `openarmature.checkpoint.migrate` for state-migration runs (proposal 0014), `openarmature.checkpoint.save` for checkpoint saves (proposal 0010). These events let the OTel / Langfuse observers emit LLM-provider spans, checkpoint-migrate spans, etc. — but a custom observer that only cares about user-defined node activity sees them as noise:

```python
async def __call__(self, event: NodeEvent) -> None:
    # Skip OA-internal events; only react to user node activity.
    if event.namespace and event.namespace[0].startswith("openarmature."):
        return
    # … user-node handling
```

`event.namespace[0]` is the safest discriminator (the leaf `event.node_name` would also work for LLM events but won't match the checkpoint sentinels since those repurpose `node_name` differently). Don't try to filter on `current_invocation_id() is None` — OA-internal events are dispatched within the same invocation context as user-node events, so `invocation_id` is set for both; the namespace-prefix check is the stable contract.

### A `with_state_migration` recipe — register migrations alongside the state class, run on resume

`GraphBuilder.with_state_migration(s)` registers callables that transform an old-schema state record into the current schema. The engine calls them automatically on `invoke(resume_invocation=...)` when the loaded record's `schema_version` doesn't match `state_cls.schema_version`. The migration callable's signature is `(state_dict: dict, from_version: str, to_version: str) -> dict`; it receives the raw deserialized record and returns the new shape.

Wire it up at compile time:

```python
class PipelineState(State):
    schema_version: ClassVar[str] = "2"
    # … v2 fields

def _migrate_v1_to_v2(state_dict: dict, from_version: str, to_version: str) -> dict:
    # Old field "step_count" renamed to "steps_completed" in v2.
    state_dict["steps_completed"] = state_dict.pop("step_count", 0)
    return state_dict

compiled = (
    GraphBuilder(PipelineState)
    .add_node("step", _step_body)
    .add_edge("step", END)
    .set_entry("step")
    .with_state_migration(from_version="1", to_version="2", migrate=_migrate_v1_to_v2)
    .compile()
)
compiled.attach_checkpointer(checkpointer)
```

Important detail: the migration runs once on resume, before any node body fires; the engine dispatches a synthetic `checkpoint_migrated` observer event (per spec §6 cross-ref) so observers can emit a migration span. The migrated state is what `_step_body` sees on resume — you do NOT need to handle both v1 and v2 shapes in node bodies.

When chaining multiple migrations (v1 → v2 → v3), register each step separately via repeated `with_state_migration` calls; the engine walks the chain in version order. If the chain has gaps (registered v1→v2 and v3→v4 but a record is at v2 with `to_version="4"`), the engine raises `CheckpointStateMigrationMissing` at resume time — fail-loud rather than silently skipping.

### Fan-out subgraphs that emit `list[X]` per instance produce `list[list[X]]` at `target_field`

When a fan-out's per-instance state collects a `list[X]` as its `collect_field` (e.g., each instance produces 0..N records), the engine's contribution step is `[s[cfg.collect_field] for s in successes]` — every instance's value becomes one element of the outer list. With `list[X]` per-instance, the parent receives `list[list[X]]`, and the default `append` reducer on the parent's `Annotated[list[X], append]` field preserves the nesting verbatim. Pydantic then fails to validate each `list[X]` element against `X`:

```
attributed_candidates.0  Input should be a valid dictionary or
  instance of ClaimCandidate [input_value=[ClaimCandidate(...)],
  input_type=list]
```

The fix is the `concat_flatten` built-in reducer (proposal 0036) — the list-of-lists analog of `append`. Declare it on the parent's collection field:

```python
from typing import Annotated

from pydantic import Field

from openarmature.graph import State, concat_flatten

class PipelineState(State):
    attributed_candidates: Annotated[list[ClaimCandidate], concat_flatten] = Field(default_factory=list)
```

`concat_flatten` folds the per-instance lists into one flat list (`[*prior, *(item for sublist in update for item in sublist)]`), strict like `append` — it raises `ReducerError` if any element of the update isn't itself a list.

The dict-shaped analog is `merge_all` (also proposal 0036): when each fan-out instance contributes a `dict[str, X]`, the parent's `target_field` receives `list[dict]`, which plain `merge` can't consume. `merge_all` folds the sequence of mappings into the prior with shallow last-write-wins per key:

```python
from typing import Annotated

from pydantic import Field

from openarmature.graph import State, merge_all

class PipelineState(State):
    keyed_results: Annotated[dict[str, Result], merge_all] = Field(default_factory=dict)
```

Single-record-per-instance fan-outs (`collect_field: str`, parent field `Annotated[list[X], append]`) don't hit this — the engine still wraps each instance's value as one element, but `append` flattens it correctly since each element is already an `X`. The two non-flat shapes emerge only when the per-instance value is itself a container: a `list[X]` per instance lands `list[list[X]]` (use `concat_flatten`), and a `dict[str, X]` per instance lands `list[dict]` (use `merge_all`).

If a parent field is populated by BOTH direct node writes AND fan-out collection, that's an architectural ambiguity worth fixing upstream — split into two fields, or pick one path.

### `invoke(metadata=...)` for caller-supplied trace identifiers (tenant IDs, request IDs, feature flags)

Per spec observability §3.4 / proposal 0034, callers attach arbitrary key/value entries at `invoke()` time and the framework propagates them to every observability backend:

```python
await compiled.invoke(
    initial_state,
    metadata={"tenantId": "acme-corp", "requestId": "req-12345", "featureFlag": "v2-canary"},
)
```

The OTel observer emits each entry as an `openarmature.user.<key>` cross-cutting span attribute on every span (invocation, node, subgraph wrapper, fan-out instance, LLM provider). The Langfuse observer merges each entry as a top-level key into `trace.metadata` AND every observation's metadata. Backends that consume OTel attributes (Honeycomb, Datadog APM, HyperDX, Grafana Tempo) pick the entries up for free; backends with typed metadata fields (Langfuse) get them via the per-backend propagation rule.

Boundary validation runs synchronously: keys MUST NOT start with `openarmature.` or `gen_ai.` (reserved namespaces); values MUST be OTel-attribute-compatible scalars (`str` / `int` / `float` / `bool`) or homogeneous arrays of those. Violations raise `ValueError` before any work begins.

Mid-invocation augmentation via the public helper:

```python
from openarmature.observability import set_invocation_metadata

async def my_node(state: MyState) -> dict:
    set_invocation_metadata(productId=state.product_id)
    # subsequent spans (this node's completed, next node's started,
    # any LLM call inside, etc.) carry productId
    return {"score": await compute_score(state)}
```

The augmentation respects fan-out / parallel-branches per-instance scoping — each instance's augmentation lives in its own Context copy and doesn't leak to siblings. Sequential nodes in the same engine task see prior nodes' augmentations forward. The helper validates the same rules as the `invoke()` boundary.

## Example index

_Runnable example programs shipped in the source tree at `examples/`. The full code is not bundled here (each example is 300+ lines); read the file at the listed path to see the canonical shape for that use case._

- **`examples/00-hello-world/main.py`** — Hello-world demo: a 3-node graph where each node makes an LLM call with structured output. Classify a query, then either plan research or write a one-sentence summary.
- **`examples/01-routing-and-subgraphs/main.py`** — openarmature demo: conditional routing + subgraph with a custom projection.
- **`examples/02-explicit-subgraph-mapping/main.py`** — openarmature demo: same compiled subgraph reused at two sites in one parent graph, each site with its own ExplicitMapping.
- **`examples/03-observer-hooks/main.py`** — openarmature demo: observer hooks for structured logging, per-call metrics, and OTel spans.
- **`examples/04-nested-subgraphs/main.py`** — openarmature demo: question answering against a tiny document corpus, with two levels of subgraph nesting.
- **`examples/05-fan-out-with-retry/main.py`** — openarmature demo: summarize a batch of lunar-mission headlines in parallel, with per-headline retries and timing.
- **`examples/06-parallel-branches/main.py`** — openarmature demo: enrich a lunar-mission news article with three independent analyses running concurrently.
- **`examples/07-multimodal-prompt/main.py`** — openarmature demo: two independent analyses of a lunar-mission photograph using versioned prompt templates, a fallback prompt backend, and a multimodal user message.
- **`examples/08-checkpointing-and-migration/main.py`** — openarmature demo: a lunar-mission planning pipeline that checkpoints its progress, then resumes under an upgraded state schema.
- **`examples/09-tool-use/main.py`** — openarmature demo: a lunar-mission assistant that calls local Python functions as tools to answer fact and physics questions about Apollo / Artemis missions.
- **`examples/10-langfuse-observability/main.py`** — openarmature demo: Langfuse observer + prompt linkage on a lunar mission Q&A pipeline.

## Discovery cross-references

If your question isn't covered above, look here:

- **Full docs site:** [openarmature.ai](https://openarmature.ai)
- **Spec text:** [openarmature.org/capabilities](https://openarmature.org/capabilities/)
- **API reference:** [openarmature.ai/reference](https://openarmature.ai/reference/)
- **Host project conventions:** the project's own `AGENTS.md` / `CLAUDE.md`
