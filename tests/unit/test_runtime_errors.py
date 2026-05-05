"""Runtime-error categories not exercised by the conformance suite.

Spec §4 defines five runtime categories. The conformance fixtures cover
`node_exception` (009) and `routing_error` (008) directly and reach the
others incidentally via 001–006. These tests target the three categories no
fixture triggers: `edge_exception`, `reducer_error`, and
`state_validation_error`.
"""

from typing import Annotated, Any

import pytest
from openarmature.graph import (
    END,
    EdgeException,
    EndSentinel,
    GraphBuilder,
    ReducerError,
    State,
    StateValidationError,
    append,
)
from pydantic import Field


class S(State):
    log: Annotated[list[str], append] = Field(default_factory=list)
    score: int = 0


async def test_edge_exception_when_conditional_fn_raises() -> None:
    async def node_a(_state: Any) -> dict[str, Any]:
        return {"score": 1}

    def bad_edge(_state: Any) -> str | EndSentinel:
        raise RuntimeError("edge boom")

    g = GraphBuilder(S).add_node("a", node_a).add_conditional_edge("a", bad_edge).set_entry("a").compile()

    with pytest.raises(EdgeException) as excinfo:
        await g.invoke(S())

    err = excinfo.value
    assert err.category == "edge_exception"
    assert err.source_node == "a"
    assert isinstance(err.__cause__, RuntimeError)
    # Recoverable state is the post-update state (after node a wrote score=1).
    assert err.recoverable_state.model_dump() == {"log": [], "score": 1}


async def test_reducer_error_when_append_receives_non_list() -> None:
    async def node_a(_state: Any) -> dict[str, Any]:
        # `log` reducer is `append`, which requires a list update; pass a string.
        return {"log": "not-a-list"}

    g = GraphBuilder(S).add_node("a", node_a).add_edge("a", END).set_entry("a").compile()

    with pytest.raises(ReducerError) as excinfo:
        await g.invoke(S())

    err = excinfo.value
    assert err.category == "reducer_error"
    assert err.field_name == "log"
    assert err.reducer_name == "append"
    assert err.producing_node == "a"
    assert isinstance(err.__cause__, TypeError)
    # Recoverable state is the pre-merge state (before node a's update).
    assert err.recoverable_state.model_dump() == {"log": [], "score": 0}


async def test_state_validation_error_on_type_mismatch() -> None:
    async def node_a(_state: Any) -> dict[str, Any]:
        return {"score": "not-an-int"}

    g = GraphBuilder(S).add_node("a", node_a).add_edge("a", END).set_entry("a").compile()

    with pytest.raises(StateValidationError) as excinfo:
        await g.invoke(S())

    err = excinfo.value
    assert err.category == "state_validation_error"
    assert "score" in err.fields
    # Spec §4: state_validation_error MUST NOT carry recoverable_state.
    assert not hasattr(err, "recoverable_state")


async def test_state_validation_error_on_unknown_field() -> None:
    async def node_a(_state: Any) -> dict[str, Any]:
        return {"undeclared": "value"}

    g = GraphBuilder(S).add_node("a", node_a).add_edge("a", END).set_entry("a").compile()

    with pytest.raises(StateValidationError) as excinfo:
        await g.invoke(S())

    err = excinfo.value
    assert err.category == "state_validation_error"
    assert "undeclared" in err.fields


async def test_subgraph_projection_error_wrapped_as_node_exception() -> None:
    """Errors from a subgraph's projection (project_in / project_out) are
    NOT spec §4 categories on their own. The engine wraps them as
    NodeException tagged with the subgraph wrapper's name so callers see
    a uniform error contract."""

    from openarmature.graph import NodeException, ProjectionStrategy

    class Inner(State):
        x: int = 0

    async def _inner_node(_s: Inner) -> dict[str, Any]:
        return {}

    inner_g = GraphBuilder(Inner).add_node("i", _inner_node).add_edge("i", END).set_entry("i").compile()

    # Parameter names match the ProjectionStrategy Protocol exactly so
    # pyright's strict structural conformance check passes.
    class BoomProjection:
        def project_in(self, parent_state: S, subgraph_state_cls: type[Inner]) -> Inner:
            raise RuntimeError("project_in boom")

        def project_out(
            self,
            subgraph_final_state: Inner,
            parent_state: S,
            subgraph_state_cls: type[Inner],
        ) -> dict[str, Any]:
            return {}

    _: ProjectionStrategy[S, Inner] = BoomProjection()

    g = (
        GraphBuilder(S)
        .add_subgraph_node("sub", inner_g, projection=BoomProjection())
        .add_edge("sub", END)
        .set_entry("sub")
        .compile()
    )

    with pytest.raises(NodeException) as excinfo:
        await g.invoke(S())

    err = excinfo.value
    assert err.category == "node_exception"
    # The wrapper's name, not the inner node's — projection is at the
    # boundary, not inside the subgraph.
    assert err.node_name == "sub"
    assert isinstance(err.__cause__, RuntimeError)
    assert str(err.__cause__) == "project_in boom"
