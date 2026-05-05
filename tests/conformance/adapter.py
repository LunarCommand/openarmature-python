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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field, create_model

from openarmature.graph import (
    END,
    CompiledGraph,
    EndSentinel,
    ExplicitMapping,
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
) -> BuiltGraph:
    """Translate a graph-shaped fixture block into a `BuiltGraph`.

    `spec` is the top-level fixture mapping for plain fixtures, or the inner
    `graph:` block for the table-style 007 cases. `subgraphs` is the registry
    used by 006-style fixtures to look up a compiled subgraph by its declared
    name.

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

    for node_name, node_spec in spec.get("nodes", {}).items():
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
            )
        elif "raises" in node_spec:
            builder.add_node(node_name, _make_raising_fn(node_name, node_spec["raises"], trace))
        elif "update" in node_spec:
            builder.add_node(node_name, _make_update_fn(node_name, node_spec["update"], trace))
        else:
            raise ValueError(f"node {node_name!r} has neither update, raises, nor subgraph")

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
    """

    name: str
    attach: str  # "graph" | "invocation"
    target: str  # "outer" | <subgraph name>
    behavior: str  # "record" | "raise"
    events: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


def _record_event(event: NodeEvent) -> dict[str, Any]:
    """Convert a NodeEvent into a dict matching the YAML expected shape."""
    rec: dict[str, Any] = {
        "step": event.step,
        "node_name": event.node_name,
        "namespace": list(event.namespace),
        "pre_state": event.pre_state.model_dump(),
        "parent_states": [ps.model_dump() for ps in event.parent_states],
    }
    if event.post_state is not None:
        rec["post_state"] = event.post_state.model_dump()
    if event.error is not None:
        rec["error"] = event.error.category
    return rec


def make_observer_fn(
    fixture: ObserverFixture,
    delivery: list[tuple[str, int]],
) -> Observer:
    """Build the async observer callable for an `ObserverFixture`.

    Records every event into `fixture.events` and appends `(name, step)` to
    the shared `delivery` list (the order observers are called in across the
    whole invocation, used to assert `delivery_order`). Raising observers
    record + append before raising, so the engine's error isolation can be
    verified by checking that subsequent observers/events still get through.
    """

    async def observer(event: NodeEvent) -> None:
        delivery.append((fixture.name, event.step))
        fixture.events.append(_record_event(event))
        if fixture.behavior == "raise":
            raise RuntimeError(f"{fixture.name} raised on event at step {event.step}")

    return observer


def normalize_expected_event(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Fill in defaults for keys the YAML omits, so equality with the
    recorded event dict works as-is."""
    e = dict(ev)
    e.setdefault("parent_states", [])
    return e
