"""SubgraphNode coverage.

The conformance adapter routes around `SubgraphNode` via a function-node
wrapper (so it can record outer-graph execution order in the parent trace).
These tests exercise `SubgraphNode.run` directly, plus the
`builder.add_subgraph_node` path that constructs one internally.
"""

from typing import Annotated, Any

from openarmature.graph import (
    END,
    FieldNameMatching,
    GraphBuilder,
    State,
    SubgraphNode,
    append,
)
from pydantic import Field


class Inner(State):
    msg: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


class Outer(State):
    msg: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=lambda: ["outer-start"])


async def test_subgraph_node_run_projects_via_field_name_matching() -> None:
    async def x(_s: Any) -> dict[str, Any]:
        return {"msg": "hello", "trace": ["x"]}

    inner = GraphBuilder(Inner).add_node("x", x).add_edge("x", END).set_entry("x").compile()

    sub = SubgraphNode[Outer, Inner](name="sub", compiled=inner, projection=FieldNameMatching[Outer, Inner]())
    result = await sub.run(Outer())

    # Field-name matching projects shared fields back; subgraph runs from its own defaults.
    assert dict(result) == {"msg": "hello", "trace": ["x"]}


async def test_outer_graph_composes_subgraph_via_add_subgraph_node() -> None:
    async def x(_s: Any) -> dict[str, Any]:
        return {"msg": "from-x", "trace": ["x"]}

    async def y(_s: Any) -> dict[str, Any]:
        return {"msg": "from-y", "trace": ["y"]}

    inner = (
        GraphBuilder(Inner)
        .add_node("x", x)
        .add_node("y", y)
        .add_edge("x", "y")
        .add_edge("y", END)
        .set_entry("x")
        .compile()
    )

    async def outer_a(_s: Any) -> dict[str, Any]:
        return {"trace": ["outer_a"]}

    async def outer_b(_s: Any) -> dict[str, Any]:
        return {"trace": ["outer_b"]}

    outer = (
        GraphBuilder(Outer)
        .add_node("outer_a", outer_a)
        .add_subgraph_node("outer_sub", inner)
        .add_node("outer_b", outer_b)
        .add_edge("outer_a", "outer_sub")
        .add_edge("outer_sub", "outer_b")
        .add_edge("outer_b", END)
        .set_entry("outer_a")
        .compile()
    )

    final = await outer.invoke(Outer())
    # `msg` merges via parent's last_write_wins; `trace` via parent's append reducer.
    assert final.model_dump() == {
        "msg": "from-y",
        "trace": ["outer-start", "outer_a", "x", "y", "outer_b"],
    }
