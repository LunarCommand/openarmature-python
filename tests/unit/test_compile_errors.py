"""Compile-error edge cases not in the spec's 007 fixture table."""

from typing import Any

import pytest
from openarmature.graph import END, DanglingEdge, GraphBuilder, State


class S(State):
    v: str = ""


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
