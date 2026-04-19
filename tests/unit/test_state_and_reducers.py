"""Small invariants: state metadata helpers, frozen state, reducer type errors."""

from typing import Annotated

import pytest
from openarmature.graph import END, State, append, last_write_wins, merge
from openarmature.graph.state import field_reducers, resolve_reducer
from pydantic import Field, ValidationError


class S(State):
    plain: int = 0
    log: Annotated[list[str], append] = Field(default_factory=list)
    meta: Annotated[dict[str, str], merge] = Field(default_factory=dict)


def test_field_reducers_picks_up_annotated_metadata() -> None:
    fr = field_reducers(S)
    assert fr["plain"] == []
    assert fr["log"] == [append]
    assert fr["meta"] == [merge]


def test_resolve_reducer_defaults_to_last_write_wins_for_undeclared() -> None:
    assert resolve_reducer([]) is last_write_wins


def test_resolve_reducer_returns_first_when_declared() -> None:
    assert resolve_reducer([append]) is append


def test_state_is_frozen_and_rejects_mutation() -> None:
    s = S()
    with pytest.raises(ValidationError):
        s.plain = 1  # type: ignore[misc]


def test_end_sentinel_repr() -> None:
    assert repr(END) == "END"


def test_append_reducer_rejects_non_list_prior() -> None:
    with pytest.raises(TypeError):
        append("not-a-list", ["x"])


def test_append_reducer_rejects_non_list_update() -> None:
    with pytest.raises(TypeError):
        append(["a"], "not-a-list")


def test_merge_reducer_rejects_non_mapping_prior() -> None:
    with pytest.raises(TypeError):
        merge("not-a-dict", {"k": "v"})


def test_merge_reducer_rejects_non_mapping_update() -> None:
    with pytest.raises(TypeError):
        merge({"k": "v"}, "not-a-dict")
