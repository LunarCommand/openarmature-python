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

    g = GraphBuilder(S).add_node("a", node_a).add_conditional("a", bad_edge).set_entry("a").compile()

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
