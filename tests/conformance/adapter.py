"""Adapter: spec conformance YAML fixtures → openarmature.graph constructs.

The fixture format is documented in
`openarmature-spec/spec/graph-engine/conformance/README.md`. This module
parses one fixture (or one sub-case from the table-style 007 fixture) into a
state class, a compiled graph, and an execution-order trace, so the
parametrized tests in `test_conformance.py` can drive the engine and assert
against the fixture's `expected` block.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field, create_model

from openarmature.graph import (
    END,
    CompiledGraph,
    EndSentinel,
    ExplicitMapping,
    FanOutNode,
    FieldNameMatching,
    GraphBuilder,
    ProjectionStrategy,
    Reducer,
    State,
    SubgraphNode,
    append,
    last_write_wins,
    merge,
)
from openarmature.graph.events import NodeEvent
from openarmature.graph.observer import Observer

if TYPE_CHECKING:
    from openarmature.graph.observer import _InvocationContext

REDUCERS: dict[str, Reducer] = {
    "last_write_wins": last_write_wins,
    "append": append,
    "merge": merge,
}


def _parse_type(s: str) -> Any:
    s = s.strip()
    if s == "string":
        return str
    if s == "int":
        return int
    if s == "float":
        return float
    if s == "bool":
        return bool
    if s.startswith("list<") and s.endswith(">"):
        return list[_parse_type(s[5:-1])]
    if s.startswith("dict<") and s.endswith(">"):
        k, _, v = s[5:-1].partition(",")
        return dict[_parse_type(k), _parse_type(v)]
    raise ValueError(f"unknown fixture type {s!r}")


def build_state_cls(model_name: str, fields_spec: Mapping[str, Mapping[str, Any]]) -> type[State]:
    """Translate a fixture's `state.fields` block into a Pydantic State subclass.

    The `alt_reducer` key (used only by the 007 conflicting_reducers case) is
    treated as a second declared reducer so the resulting field carries two
    reducers in its Annotated metadata — exactly the shape `field_reducers`
    inspects.
    """

    field_defs: dict[str, Any] = {}
    for fname, spec in fields_spec.items():
        py_type = _parse_type(spec["type"])
        reducers = [REDUCERS[spec[k]] for k in ("reducer", "alt_reducer") if k in spec]
        annotation: Any = Annotated[py_type, *reducers] if reducers else py_type

        if "default" in spec:
            raw_default: Any = spec["default"]
            if isinstance(raw_default, list | dict):
                # Mutable default → use a factory so each instance gets its own copy.
                snapshot = copy.deepcopy(cast(Any, raw_default))
                field_defs[fname] = (
                    annotation,
                    Field(default_factory=lambda v=snapshot: copy.deepcopy(v)),
                )
            else:
                field_defs[fname] = (annotation, raw_default)
        else:
            field_defs[fname] = (annotation, ...)

    return create_model(model_name, __base__=State, **field_defs)


def _resolve_target(target: str) -> str | EndSentinel:
    return END if target == "END" else target


def _make_update_fn(
    node_name: str,
    update: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    snapshot = dict(update)

    async def fn(_state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        return copy.deepcopy(snapshot)

    return fn


def _make_raising_fn(
    node_name: str,
    message: str,
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    async def fn(_state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        raise RuntimeError(message)

    return fn


def _resolve_success_compute(success_compute: Mapping[str, Any], state: Any) -> dict[str, Any]:
    """Translate a ``success_compute`` mapping into a partial update.

    For each key/value: if the value is a string AND matches a state
    field, treat it as a field reference (read state.<value>). Otherwise
    treat it as a literal — bool/int/str-not-a-field/etc. Used by
    flaky_by_index and flaky_instance_only test seams.
    """
    state_cls = cast("type[State]", type(state))
    model_fields = cast("dict[str, Any]", state_cls.model_fields)
    state_fields: set[str] = set(model_fields.keys())
    result: dict[str, Any] = {}
    for tgt, val in success_compute.items():
        if isinstance(val, str) and val in state_fields:
            result[tgt] = getattr(state, val)
        else:
            result[tgt] = val
    return result


class _CategorizedException(Exception):
    """A test exception carrying a `category` attribute so the default
    retry classifier can match it."""

    def __init__(self, message: str, category: str) -> None:
        super().__init__(message)
        self.category = category


def _make_pure_update_fn(
    node_name: str,
    update: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`update_pure` test seam — same as `update` but explicitly tagged as
    state-independent. Used by fan-out fixtures whose worker subgraphs
    apply a fixed update that doesn't depend on the input state."""
    snapshot = dict(update)

    async def fn(_state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        return copy.deepcopy(snapshot)

    return fn


def _make_update_from_field_fn(
    node_name: str,
    update: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`update_from_field` test seam — assigns ``state.<source> * multiplier``
    to ``target``. The fixture format is ``{<target>: <source>, multiplier: N}``;
    e.g. ``{result: x, multiplier: 2}`` means ``result = state.x * 2``.
    Used by the doubler / scorer subgraphs in the fan-out fixtures."""
    cfg = dict(update)
    multiplier = int(cfg.pop("multiplier", 1))
    # The remaining single key→value pair is target_field → source_field.
    if len(cfg) != 1:
        raise ValueError(
            f"update_from_field for {node_name!r} expects exactly one "
            f"target→source mapping plus an optional `multiplier`; got {update!r}"
        )
    target_field, source_field = next(iter(cfg.items()))

    async def fn(state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        source_value = getattr(state, source_field)
        return {target_field: source_value * multiplier}

    return fn


def _make_flaky_by_index_fn(
    node_name: str,
    cfg: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`flaky_by_index` test seam used by fan-out fixtures.

    Two config shapes exist in the spec fixtures:

    - ``fail_count_per_idx: N`` — fixture 020. Each fan-out instance
      fails its first N ATTEMPTS with a transient exception (so retry
      middleware exercises the retry path). Counter is per instance —
      reconstructed per node activation since each fan-out instance
      gets its own subgraph invocation, so the closure is fresh.
    - ``fail_when_idx: N`` — fixture 019. The instance whose
      ``state.idx`` (item value) equals N fails; others succeed. No
      retry logic; this is the collect-policy test seam.

    Both shapes share ``category`` and ``success_compute``.
    """
    category = cfg.get("category", "provider_unavailable")
    success_compute = dict(cfg.get("success_compute", {}))
    fail_count_per_idx = cfg.get("fail_count_per_idx")
    fail_when_idx = cfg.get("fail_when_idx")
    # Fixture-global attempt counter. Truly per-instance semantics
    # would require an identifier stable across an instance's retries
    # AND distinct across fan-out instances. id(state) works for
    # NODE-LEVEL retry (state stable across retries) but not for
    # INSTANCE-LEVEL retry (each instance retry constructs a fresh
    # subgraph state). A state-field key would work in principle but
    # the field name is fixture-specific. The current fixtures
    # (020 node-level retry, 019 collect mode) pass with this global
    # counter because the timing aligns; documenting the limitation
    # here so a future fixture exercising the gap surfaces a real
    # failure rather than silently miscounting.
    attempt_counter = [0]

    async def fn(state: Any) -> Mapping[str, Any]:
        idx = attempt_counter[0]
        attempt_counter[0] = idx + 1
        if idx == 0:
            trace.append(node_name)
        # fail_when_idx mode: fail when the instance's item value (read
        # from state.idx by convention) equals the configured int.
        if fail_when_idx is not None:
            instance_idx = getattr(state, "idx", None)
            if instance_idx == int(fail_when_idx):
                raise _CategorizedException(
                    message=f"flaky_by_index fail_when_idx={fail_when_idx}",
                    category=category,
                )
        # fail_count_per_idx mode: fail the first N attempts (counted
        # globally across the fan-out).
        elif fail_count_per_idx is not None and idx < int(fail_count_per_idx):
            raise _CategorizedException(
                message=f"flaky_by_index failure at attempt {idx}",
                category=category,
            )
        if success_compute:
            return _resolve_success_compute(success_compute, state)
        return {}

    return fn


def _make_flaky_instance_only_fn(
    node_name: str,
    cfg: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`flaky_instance_only` test seam — fails on the first call,
    succeeds thereafter. Same fixture-global counter as
    flaky_by_index; see that function's note on the per-instance vs
    per-fixture keying tradeoff. The current fixtures (021
    instance-level retry) align with the global counter at runtime
    even though strict per-instance semantics would need a richer
    identifier."""
    fail_first_only = bool(cfg.get("fail_first_only", True))
    category = cfg.get("category", "provider_unavailable")
    success_compute = dict(cfg.get("success_compute", {}))
    attempt_counter = [0]

    async def fn(state: Any) -> Mapping[str, Any]:
        idx = attempt_counter[0]
        attempt_counter[0] = idx + 1
        if idx == 0:
            trace.append(node_name)
        if fail_first_only and idx == 0:
            raise _CategorizedException(
                message="flaky_instance_only first-attempt failure",
                category=category,
            )
        if success_compute:
            return _resolve_success_compute(success_compute, state)
        return {}

    return fn


def _make_flaky_fn(
    node_name: str,
    flaky: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """Build a flaky node body that fails per a configured failure_sequence
    and finally returns a success_update.

    Per fixture 007's contract: `failure_sequence` is a list of failures
    keyed to attempt index. Each entry has a `transient` flag, a
    `category` (matched by the retry classifier), and a `message`. When
    the sequence is exhausted, subsequent attempts return `success_update`.
    """
    sequence = list(flaky.get("failure_sequence", []))
    success_update = dict(flaky.get("success_update", {}))
    attempt_counter = [0]

    async def fn(_state: Any) -> Mapping[str, Any]:
        idx = attempt_counter[0]
        attempt_counter[0] = idx + 1
        # `execution_order` is engine-step-scoped, not per-attempt — only
        # append on the first attempt so retry middleware re-invocations
        # don't double-count the node.
        if idx == 0:
            trace.append(node_name)
        if idx < len(sequence):
            entry = sequence[idx]
            if entry is None:
                return copy.deepcopy(success_update)
            raise _CategorizedException(
                message=entry.get("message", "flaky"),
                category=entry.get("category", "provider_unavailable"),
            )
        return copy.deepcopy(success_update)

    return fn


@dataclass(frozen=True)
class _TracingFanOutNode(FanOutNode[State, State]):
    """Conformance helper: a FanOutNode that appends its name to a shared
    trace list when the engine runs it. Same role as _TracingSubgraphNode
    for subgraphs — a fan-out node is one engine step from the parent's
    POV (per §9.6), so it should contribute exactly one trace entry."""

    trace_list: list[str] = field(default_factory=list[str])

    async def run_with_context(
        self,
        state: State,
        context: _InvocationContext,
    ) -> Mapping[str, Any]:
        self.trace_list.append(self.name)
        return await super().run_with_context(state, context)


@dataclass(frozen=True)
class _TracingSubgraphNode(SubgraphNode[State, State]):
    """Conformance helper: a SubgraphNode that appends its name to a shared
    trace list when the engine runs it.

    Lets the conformance adapter use real SubgraphNode (so observer-context
    threading works for fixture 013, and compile-time projection validation
    works for the mapping_references_undeclared_field 007 case) while still
    supporting `execution_order` assertions that include the wrapper name —
    the engine itself doesn't dispatch an event for the wrapper per fixture
    013's spec.
    """

    trace_list: list[str] = field(default_factory=list[str])

    async def run(
        self,
        state: State,
        context: _InvocationContext | None = None,
    ) -> Mapping[str, Any]:
        self.trace_list.append(self.name)
        return await super().run(state, context=context)


def _make_conditional_fn(
    if_field: str,
    equals: Any,
    then: str,
    else_: str,
) -> Callable[[Any], str | EndSentinel]:
    then_target = _resolve_target(then)
    else_target = _resolve_target(else_)

    def fn(state: Any) -> str | EndSentinel:
        return then_target if getattr(state, if_field) == equals else else_target

    return fn


@dataclass
class BuiltGraph:
    """Result of translating a fixture into runnable engine constructs."""

    state_cls: type[State]
    builder: GraphBuilder[State]
    trace: list[str]

    def initial_state(self, overrides: Mapping[str, Any]) -> State:
        return self.state_cls(**overrides)


def _projection_for(node_spec: Mapping[str, Any]) -> ProjectionStrategy[State, State]:
    """Pick the projection strategy declared on a subgraph node spec.

    `inputs:` and/or `outputs:` in the YAML → `ExplicitMapping`. Both absent →
    the spec's default `FieldNameMatching`.
    """

    inputs = node_spec.get("inputs")
    outputs = node_spec.get("outputs")
    if inputs is None and outputs is None:
        return FieldNameMatching[State, State]()
    return ExplicitMapping[State, State](inputs=inputs, outputs=outputs)


def build_graph(
    spec: Mapping[str, Any],
    *,
    subgraphs: Mapping[str, CompiledGraph[State]] | None = None,
    trace: list[str] | None = None,
    model_name: str = "FixtureState",
    node_middleware: Mapping[str, Sequence[Any]] | None = None,
    graph_middleware: Sequence[Any] | None = None,
    fan_out_instance_middleware: Mapping[str, Sequence[Any]] | None = None,
) -> BuiltGraph:
    """Translate a graph-shaped fixture block into a `BuiltGraph`.

    `spec` is the top-level fixture mapping for plain fixtures, or the inner
    `graph:` block for the table-style 007 cases. `subgraphs` is the registry
    used by 006-style fixtures to look up a compiled subgraph by its declared
    name.

    `node_middleware` (mapping node name to ordered middleware list) and
    `graph_middleware` (ordered middleware list applied to every node)
    are pipeline-utilities §3 hooks. The translation from a fixture's
    `middleware:` block into actual instances lives in the
    pipeline-utilities test driver.

    Subgraph references in `spec.nodes` resolve to `_TracingSubgraphNode`
    (a SubgraphNode subclass) so the engine threads observer context through
    AND the conformance adapter's `execution_order` trace gets the wrapper
    name appended when it runs.
    """

    state_cls = build_state_cls(model_name, spec["state"]["fields"])
    builder = GraphBuilder(state_cls)
    if "entry" in spec:
        builder.set_entry(spec["entry"])

    trace = trace if trace is not None else []
    subgraphs = subgraphs or {}
    node_middleware = node_middleware or {}
    fan_out_instance_middleware = fan_out_instance_middleware or {}

    for mw in graph_middleware or ():
        builder.add_middleware(mw)

    for node_name, node_spec in spec.get("nodes", {}).items():
        per_node_mw = tuple(node_middleware.get(node_name, ()))
        if "subgraph" in node_spec:
            sub_name = node_spec["subgraph"]
            compiled = subgraphs[sub_name]
            projection = _projection_for(node_spec)
            if node_name in builder._nodes:
                raise ValueError(f"node {node_name!r} already declared")
            builder._nodes[node_name] = _TracingSubgraphNode(
                name=node_name,
                compiled=compiled,
                projection=projection,
                trace_list=trace,
                middleware=per_node_mw,
            )
        elif "fan_out" in node_spec:
            _add_fan_out_node(
                builder,
                node_name,
                node_spec["fan_out"],
                subgraphs,
                trace,
                instance_middleware=fan_out_instance_middleware.get(node_name, ()),
            )
        elif "raises" in node_spec:
            builder.add_node(
                node_name,
                _make_raising_fn(node_name, node_spec["raises"], trace),
                middleware=per_node_mw,
            )
        elif "flaky" in node_spec:
            builder.add_node(
                node_name,
                _make_flaky_fn(node_name, node_spec["flaky"], trace),
                middleware=per_node_mw,
            )
        elif "flaky_by_index" in node_spec:
            builder.add_node(
                node_name,
                _make_flaky_by_index_fn(node_name, node_spec["flaky_by_index"], trace),
                middleware=per_node_mw,
            )
        elif "flaky_instance_only" in node_spec:
            builder.add_node(
                node_name,
                _make_flaky_instance_only_fn(node_name, node_spec["flaky_instance_only"], trace),
                middleware=per_node_mw,
            )
        elif "update" in node_spec:
            builder.add_node(
                node_name,
                _make_update_fn(node_name, node_spec["update"], trace),
                middleware=per_node_mw,
            )
        elif "update_pure" in node_spec:
            builder.add_node(
                node_name,
                _make_pure_update_fn(node_name, node_spec["update_pure"], trace),
                middleware=per_node_mw,
            )
        elif "update_from_field" in node_spec:
            builder.add_node(
                node_name,
                _make_update_from_field_fn(node_name, node_spec["update_from_field"], trace),
                middleware=per_node_mw,
            )
        else:
            raise ValueError(
                f"node {node_name!r} has no recognized directive "
                "(update / update_pure / update_from_field / raises / flaky / "
                "flaky_by_index / flaky_instance_only / fan_out / subgraph)"
            )

    for edge_spec in spec.get("edges", []):
        source = edge_spec["from"]
        if "to" in edge_spec:
            builder.add_edge(source, _resolve_target(edge_spec["to"]))
        elif "condition" in edge_spec:
            cond = edge_spec["condition"]
            builder.add_conditional_edge(
                source,
                _make_conditional_fn(cond["if_field"], cond["equals"], cond["then"], cond["else"]),
            )
        else:
            raise ValueError(f"edge from {source!r} has neither `to` nor `condition`")

    return BuiltGraph(state_cls=state_cls, builder=builder, trace=trace)


# ---------------------------------------------------------------------------
# Observer fixture support (spec v0.3.0 §6, fixtures 012–015)
# ---------------------------------------------------------------------------


@dataclass
class ObserverFixture:
    """Captured per-observer state for assertion against an observer fixture.

    Built once per observer declared in a fixture's `observers:` block. The
    observer callable produced by `make_observer_fn` records every event it
    receives into `events` and (if behavior == "raise") raises after
    recording.

    `phases` is the optional subscription set parsed from the fixture's
    YAML. None means "no `phases:` key was present" — the harness leaves
    the engine to default to both phases.
    """

    name: str
    attach: str  # "graph" | "invocation"
    target: str  # "outer" | <subgraph name>
    behavior: str  # "record" | "raise"
    phases: frozenset[str] | None = None
    events: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


def _record_event(event: NodeEvent) -> dict[str, Any]:
    """Convert a NodeEvent into a dict matching the YAML expected shape."""
    rec: dict[str, Any] = {
        "step": event.step,
        "phase": event.phase,
        "node_name": event.node_name,
        "namespace": list(event.namespace),
        "pre_state": event.pre_state.model_dump(),
        "parent_states": [ps.model_dump() for ps in event.parent_states],
        "attempt_index": event.attempt_index,
    }
    if event.post_state is not None:
        rec["post_state"] = event.post_state.model_dump()
    if event.error is not None:
        rec["error"] = event.error.category
    if event.fan_out_index is not None:
        rec["fan_out_index"] = event.fan_out_index
    return rec


def make_observer_fn(
    fixture: ObserverFixture,
    delivery: list[tuple[str, int, str]],
) -> Observer:
    """Build the async observer callable for an `ObserverFixture`.

    Records every event into `fixture.events` and appends
    `(name, step, phase)` to the shared `delivery` list (the order
    observers are called in across the whole invocation, used to assert
    `delivery_order`). Raising observers record + append before raising,
    so the engine's error isolation can be verified by checking that
    subsequent observers/events still get through.
    """

    async def observer(event: NodeEvent) -> None:
        delivery.append((fixture.name, event.step, event.phase))
        fixture.events.append(_record_event(event))
        if fixture.behavior == "raise":
            raise RuntimeError(f"{fixture.name} raised on event at step {event.step}")

    return observer


def normalize_expected_event(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Fill in defaults for keys the YAML omits, so equality with the
    recorded event dict works as-is. Fixtures don't repeat the
    `attempt_index: 0` and `parent_states: []` defaults for every event;
    the engine emits both unconditionally, so backfill them here.
    """
    e = dict(ev)
    e.setdefault("parent_states", [])
    e.setdefault("attempt_index", 0)
    return e


def _add_fan_out_node(
    builder: GraphBuilder[Any],
    node_name: str,
    cfg: Mapping[str, Any],
    subgraphs: Mapping[str, CompiledGraph[State]],
    trace: list[str],
    *,
    instance_middleware: Sequence[Any] = (),
) -> None:
    """Translate a fixture's ``fan_out:`` block into a builder.add_fan_out_node
    call.

    Resolves callable forms of ``count`` and ``concurrency``:

    - ``state_field_read`` — read an int from a parent state field.
    - ``queue_chunk`` — ``max(1, len(state.<field>) // chunk_size)``.

    These are the only callable shapes the in-scope Phase 3 fixtures
    use. Adding more is straightforward.
    """
    sub_name = cfg["subgraph"]
    sub_compiled = subgraphs[sub_name]

    count_raw = cfg.get("count")
    count: int | Callable[[Any], int] | None = None
    if isinstance(count_raw, dict):
        count = _resolve_callable_int_resolver(cast("dict[str, Any]", count_raw))
    elif count_raw is not None:
        count = int(count_raw)

    conc_raw = cfg.get("concurrency", 10)
    conc: int | Callable[[Any], int | None] | None
    if isinstance(conc_raw, dict):
        conc = cast(
            "Callable[[Any], int | None]",
            _resolve_callable_int_resolver(cast("dict[str, Any]", conc_raw)),
        )
    elif conc_raw is None:
        conc = None
    else:
        conc = int(conc_raw)

    builder.add_fan_out_node(
        node_name,
        subgraph=sub_compiled,
        collect_field=cfg["collect_field"],
        target_field=cfg["target_field"],
        items_field=cfg.get("items_field"),
        item_field=cfg.get("item_field"),
        count=count,
        concurrency=conc,
        error_policy=cfg.get("error_policy", "fail_fast"),
        on_empty=cfg.get("on_empty", "raise"),
        count_field=cfg.get("count_field"),
        inputs=cfg.get("inputs"),
        extra_outputs=cfg.get("extra_outputs"),
        errors_field=cfg.get("errors_field"),
        instance_middleware=instance_middleware,
    )

    # Swap the registered FanOutNode for a tracing variant so the
    # conformance trace records the fan-out as one engine step. The
    # builder's compile-time validation already ran above; we only
    # replace the stored Node instance.
    original = cast("FanOutNode[State, State]", builder._nodes[node_name])
    builder._nodes[node_name] = _TracingFanOutNode(
        name=original.name,
        config=original.config,
        middleware=original.middleware,
        trace_list=trace,
    )


def _resolve_callable_int_resolver(cfg: Mapping[str, Any]) -> Callable[[Any], int]:
    """Build a state-reader callable from a fixture's callable config.

    Supported shapes per the fan-out fixtures:

    - ``{callable: state_field_read, field: <name>}`` — return
      ``state.<name>`` (must be int).
    - ``{callable: queue_chunk, field: <name>, chunk_size: N}`` —
      return ``max(1, len(state.<name>) // N)``.
    """
    kind = cfg["callable"]
    field_name = cfg["field"]
    if kind == "state_field_read":

        def state_field_read(state: Any) -> int:
            return int(getattr(state, field_name))

        return state_field_read
    if kind == "queue_chunk":
        chunk_size = int(cfg["chunk_size"])

        def queue_chunk(state: Any) -> int:
            return max(1, len(getattr(state, field_name)) // chunk_size)

        return queue_chunk
    raise ValueError(f"unknown callable resolver: {kind!r}")
