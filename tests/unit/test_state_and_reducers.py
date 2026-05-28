"""Small invariants: state metadata helpers, frozen state, reducer type errors."""

from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from openarmature.graph import (
    END,
    State,
    append,
    concat_flatten,
    last_write_wins,
    merge,
    merge_all,
)
from openarmature.graph.state import field_reducers, resolve_reducer


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


# Proposal 0036 — concat_flatten (list-of-lists → flat list).


def test_concat_flatten_concatenates_and_flattens() -> None:
    assert concat_flatten(["a", "b"], [["c"], ["d", "e"], []]) == ["a", "b", "c", "d", "e"]


def test_concat_flatten_empty_update_is_noop() -> None:
    assert concat_flatten(["a"], []) == ["a"]


def test_concat_flatten_empty_sublists_contribute_nothing() -> None:
    assert concat_flatten([], [[], []]) == []


def test_concat_flatten_rejects_non_list_prior() -> None:
    with pytest.raises(TypeError):
        concat_flatten("not-a-list", [["x"]])


def test_concat_flatten_rejects_non_list_update() -> None:
    with pytest.raises(TypeError):
        concat_flatten([], "not-a-list")


def test_concat_flatten_rejects_non_list_element() -> None:
    with pytest.raises(TypeError):
        concat_flatten([], [["a"], "not_a_list"])


# Proposal 0036 — merge_all (list-of-mappings → folded mapping).


def test_merge_all_folds_with_last_write_wins() -> None:
    result = merge_all(
        {"seed": "prior", "retained": "kept"},
        [{"a": "1"}, {"seed": "overwritten", "b": "2"}, {"a": "1_wins"}],
    )
    assert result == {"seed": "overwritten", "retained": "kept", "a": "1_wins", "b": "2"}


def test_merge_all_empty_update_is_noop() -> None:
    assert merge_all({"k": "v"}, []) == {"k": "v"}


def test_merge_all_empty_mappings_contribute_nothing() -> None:
    assert merge_all({"prior": "value"}, [{}, {}]) == {"prior": "value"}


def test_merge_all_rejects_non_mapping_prior() -> None:
    with pytest.raises(TypeError):
        merge_all("not-a-dict", [{"k": "v"}])


def test_merge_all_rejects_non_list_update() -> None:
    with pytest.raises(TypeError):
        merge_all({}, {"k": "v"})


def test_merge_all_rejects_non_mapping_element() -> None:
    with pytest.raises(TypeError):
        merge_all({}, [{"k": "1"}, "not_a_mapping"])
