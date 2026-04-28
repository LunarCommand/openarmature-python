"""Compile-error edge cases not in the spec's 007 fixture table."""

from typing import Any

import pytest
from openarmature.graph import (
    END,
    DanglingEdge,
    ExplicitMapping,
    GraphBuilder,
    MappingReferencesUndeclaredField,
    State,
)


class S(State):
    v: str = ""


class ChildS(State):
    x: str = ""


async def _noop(_s: Any) -> dict[str, Any]:
    return {}


def test_entry_pointing_to_undeclared_node_is_dangling_edge() -> None:
    """Entry → undeclared node surfaces as DanglingEdge with source `<entry>`."""

    builder = GraphBuilder(S).add_node("a", _noop).add_edge("a", END).set_entry("ghost")

    with pytest.raises(DanglingEdge) as excinfo:
        builder.compile()

    assert excinfo.value.source == "<entry>"
    assert excinfo.value.target == "ghost"


def test_duplicate_node_name_raises_value_error() -> None:
    builder = GraphBuilder(S).add_node("a", _noop)
    with pytest.raises(ValueError, match="already declared"):
        builder.add_node("a", _noop)


def test_duplicate_subgraph_name_raises_value_error() -> None:
    inner = GraphBuilder(S).add_node("x", _noop).add_edge("x", END).set_entry("x").compile()
    builder = GraphBuilder(S).add_node("a", _noop)
    with pytest.raises(ValueError, match="already declared"):
        builder.add_subgraph_node("a", inner)


def test_compile_validates_subgraph_projection_mapping() -> None:
    """Per spec v0.2.0 §2: compilation MUST fail when a subgraph-as-node
    mapping references a field not declared in the relevant state schema."""

    inner = GraphBuilder(ChildS).add_node("i", _noop).add_edge("i", END).set_entry("i").compile()

    builder = (
        GraphBuilder(S)
        .add_subgraph_node(
            "sub",
            inner,
            ExplicitMapping[S, ChildS](inputs={"x": "missing_on_parent"}),
        )
        .add_edge("sub", END)
        .set_entry("sub")
    )

    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        builder.compile()
    assert excinfo.value.category == "mapping_references_undeclared_field"
    assert excinfo.value.direction == "inputs"
    assert excinfo.value.side == "parent"
    assert excinfo.value.field_name == "missing_on_parent"
