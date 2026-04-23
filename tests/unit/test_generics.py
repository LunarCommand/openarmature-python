"""Type-level regression tests for the generic graph primitives.

These tests exercise the static types more than runtime behavior — the
`assert_type` calls compile down to no-ops but make pyright fail loudly if
the generic surface regresses (e.g. if `CompiledGraph.invoke` goes back to
returning base `State` instead of the user's concrete subclass).

Run as part of the normal `pytest`/`pyright` flow. Runtime assertions are
kept minimal — the point is the static types.
"""

from typing import Annotated, Any, assert_type

from openarmature.graph import (
    END,
    CompiledGraph,
    FieldNameMatching,
    GraphBuilder,
    ProjectionStrategy,
    State,
    append,
)
from pydantic import Field


class ParentS(State):
    tag: str = "p"
    trace: Annotated[list[str], append] = Field(default_factory=list)


class ChildS(State):
    tag: str = "c"
    trace: Annotated[list[str], append] = Field(default_factory=list)


async def _parent_node(s: ParentS) -> dict[str, Any]:
    return {"tag": s.tag, "trace": ["parent_node"]}


async def _child_node(s: ChildS) -> dict[str, Any]:
    return {"tag": s.tag, "trace": ["child_node"]}


def test_compiled_graph_invoke_preserves_state_type() -> None:
    """GraphBuilder[ParentS].compile() → CompiledGraph[ParentS]; invoke returns ParentS."""
    builder = GraphBuilder(ParentS)
    assert_type(builder, GraphBuilder[ParentS])

    compiled = builder.add_node("a", _parent_node).add_edge("a", END).set_entry("a").compile()
    assert_type(compiled, CompiledGraph[ParentS])


async def test_invoke_return_type_is_concrete_state_subclass() -> None:
    compiled = GraphBuilder(ParentS).add_node("a", _parent_node).add_edge("a", END).set_entry("a").compile()
    final = await compiled.invoke(ParentS())
    assert_type(final, ParentS)
    # Runtime sanity: field access goes through unchallenged.
    assert final.tag == "p"


def test_conditional_edge_fn_receives_parent_state_type() -> None:
    """Edge fn's state parameter is StateT, not Any — mis-typing surfaces at pyright."""

    def route(s: ParentS) -> str:
        # If pyright had lost the generics, `s.tag` would be Any here.
        return "a" if s.tag == "p" else "a"

    builder = GraphBuilder(ParentS).add_node("a", _parent_node)
    builder = builder.add_conditional_edge("a", route).set_entry("a")
    compiled = builder.compile()
    assert_type(compiled, CompiledGraph[ParentS])


async def test_subgraph_node_preserves_both_state_types() -> None:
    """add_subgraph_node's ChildT is inferred from the compiled argument."""
    child = GraphBuilder(ChildS).add_node("c", _child_node).add_edge("c", END).set_entry("c").compile()
    assert_type(child, CompiledGraph[ChildS])

    class PassTag:
        def project_in(self, parent_state: ParentS, subgraph_state_cls: type[ChildS]) -> ChildS:
            return subgraph_state_cls(tag=parent_state.tag)

        def project_out(
            self,
            subgraph_final_state: ChildS,
            parent_state: ParentS,
            subgraph_state_cls: type[ChildS],
        ) -> dict[str, Any]:
            return {"tag": subgraph_final_state.tag, "trace": subgraph_final_state.trace}

    _: ProjectionStrategy[ParentS, ChildS] = PassTag()

    outer = (
        GraphBuilder(ParentS)
        .add_subgraph_node("sub", child, projection=PassTag())
        .add_edge("sub", END)
        .set_entry("sub")
        .compile()
    )
    assert_type(outer, CompiledGraph[ParentS])

    final = await outer.invoke(ParentS(tag="hello"))
    assert_type(final, ParentS)
    assert final.tag == "hello"


def test_field_name_matching_accepts_type_parameters() -> None:
    """Default projection is constructible with explicit type arguments."""
    proj = FieldNameMatching[ParentS, ChildS]()
    assert_type(proj, FieldNameMatching[ParentS, ChildS])
    _: ProjectionStrategy[ParentS, ChildS] = proj
