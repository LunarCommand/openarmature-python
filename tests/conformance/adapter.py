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
from dataclasses import dataclass
from typing import Annotated, Any, cast

from openarmature.graph import (
    END,
    CompiledGraph,
    EndSentinel,
    FieldNameMatching,
    GraphBuilder,
    ProjectionStrategy,
    Reducer,
    State,
    append,
    last_write_wins,
    merge,
)
from pydantic import Field, create_model

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


def _make_subgraph_fn(
    node_name: str,
    compiled: CompiledGraph,
    trace: list[str],
    projection: ProjectionStrategy,
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """Outer-graph node that delegates to a compiled subgraph.

    Records the outer node name in the parent trace before invoking the
    subgraph; the subgraph's inner nodes record into their own trace, which
    the parametrized test discards (subgraph fixtures only assert outer-graph
    execution order).
    """

    async def fn(state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        sub_initial = projection.project_in(state, compiled.state_cls)
        sub_final = await compiled.invoke(sub_initial)
        return projection.project_out(sub_final, state, compiled.state_cls)

    return fn


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
    builder: GraphBuilder
    trace: list[str]

    def initial_state(self, overrides: Mapping[str, Any]) -> State:
        return self.state_cls(**overrides)


def build_graph(
    spec: Mapping[str, Any],
    *,
    subgraphs: Mapping[str, CompiledGraph] | None = None,
    trace: list[str] | None = None,
    model_name: str = "FixtureState",
) -> BuiltGraph:
    """Translate a graph-shaped fixture block into a `BuiltGraph`.

    `spec` is the top-level fixture mapping for plain fixtures, or the inner
    `graph:` block for the table-style 007 cases. `subgraphs` is the registry
    used by 006-style fixtures to look up a compiled subgraph by its declared
    name.
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
            builder.add_node(
                node_name,
                _make_subgraph_fn(node_name, compiled, trace, FieldNameMatching()),
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
            builder.add_conditional(
                source,
                _make_conditional_fn(cond["if_field"], cond["equals"], cond["then"], cond["else"]),
            )
        else:
            raise ValueError(f"edge from {source!r} has neither `to` nor `condition`")

    return BuiltGraph(state_cls=state_cls, builder=builder, trace=trace)
