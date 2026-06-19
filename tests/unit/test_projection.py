"""Projection strategy unit coverage: FieldNameMatching + ExplicitMapping."""

import pytest
from pydantic import Field

from openarmature.graph import (
    ExplicitMapping,
    FieldNameMatching,
    MappingReferencesUndeclaredField,
    State,
)


class Parent(State):
    shared: str = "parent-shared"
    parent_only: int = 0


class ChildOverlap(State):
    shared: str = "child-shared"
    child_only: list[str] = Field(default_factory=list)


class ChildNoOverlap(State):
    completely_different: str = "x"


# ===== FieldNameMatching =====


def test_project_in_returns_subgraph_defaults() -> None:
    """Default projection-in is no projection — subgraph starts from
    its own field defaults regardless of parent state."""

    proj = FieldNameMatching[Parent, ChildOverlap]()
    sub = proj.project_in(Parent(shared="ignored"), ChildOverlap)
    assert isinstance(sub, ChildOverlap)
    assert sub.shared == "child-shared"


def test_project_out_returns_only_shared_field_names() -> None:
    proj = FieldNameMatching[Parent, ChildOverlap]()
    sub_final = ChildOverlap(shared="child-final", child_only=["a"])
    out = proj.project_out(sub_final, Parent(), ChildOverlap)

    # Only `shared` is in both schemas; `child_only` is dropped.
    assert dict(out) == {"shared": "child-final"}


def test_project_out_is_empty_when_no_overlap() -> None:
    proj = FieldNameMatching[Parent, ChildNoOverlap]()
    sub_final = ChildNoOverlap(completely_different="y")
    out = proj.project_out(sub_final, Parent(), ChildNoOverlap)
    assert dict(out) == {}


def test_field_name_matching_has_no_validate_method() -> None:
    """`validate` is an optional duck-typed compile hook — strategies with
    nothing declarative to check (FieldNameMatching, custom imperative
    projections) simply omit it. `compile()` skips them via `getattr`."""

    assert not hasattr(FieldNameMatching[Parent, ChildOverlap](), "validate")


# ===== ExplicitMapping =====


class ParentEM(State):
    a: int = 5
    b: int = 7
    captured: int = -1
    note: str = "outer"


class ChildEM(State):
    input: int = 3
    result: int = 0
    note: str = "child-default"


def test_explicit_mapping_inputs_copies_named_parent_fields() -> None:
    """`inputs: {input: a}` overlays parent.a onto subgraph.input; other
    subgraph fields receive their schema-declared defaults (no field-name
    fallback)."""

    proj = ExplicitMapping[ParentEM, ChildEM](inputs={"input": "a"})
    sub = proj.project_in(ParentEM(a=42, note="ignored-no-fallback"), ChildEM)

    assert sub.input == 42
    # `note` is declared on both schemas but not in `inputs` — must NOT be
    # filled by name matching. It uses the subgraph's schema default.
    assert sub.note == "child-default"
    assert sub.result == 0


def test_explicit_mapping_outputs_projects_only_named_pairs() -> None:
    """`outputs: {captured: input, ...}` projects only the listed
    parent←subgraph pairs; other subgraph fields are discarded (replacement
    of field-name matching)."""

    proj = ExplicitMapping[ParentEM, ChildEM](outputs={"captured": "input", "a": "result"})
    sub_final = ChildEM(input=42, result=99, note="written-but-discarded")
    out = proj.project_out(sub_final, ParentEM(), ChildEM)

    assert dict(out) == {"captured": 42, "a": 99}


def test_explicit_mapping_outputs_absent_falls_back_to_field_name_matching() -> None:
    """`outputs=None` (absent) falls back to the default field-name matching;
    `outputs={}` (present, empty) projects nothing."""

    sub_final = ChildEM(input=1, result=2, note="from-child")

    fallback = ExplicitMapping[ParentEM, ChildEM](inputs={"input": "a"}).project_out(
        sub_final, ParentEM(), ChildEM
    )
    # Only `note` is shared by name with the parent; `input` and `result` are not.
    assert dict(fallback) == {"note": "from-child"}

    explicit_empty = ExplicitMapping[ParentEM, ChildEM](inputs={"input": "a"}, outputs={}).project_out(
        sub_final, ParentEM(), ChildEM
    )
    assert dict(explicit_empty) == {}


def test_explicit_mapping_no_args_behaves_like_default_strategies_per_direction() -> None:
    """Both `inputs` and `outputs` absent: project_in returns schema defaults;
    project_out falls back to field-name matching."""

    proj = ExplicitMapping[ParentEM, ChildEM]()
    sub_in = proj.project_in(ParentEM(a=1, b=2), ChildEM)
    assert sub_in == ChildEM()

    sub_final = ChildEM(input=10, result=20, note="from-child")
    out = proj.project_out(sub_final, ParentEM(), ChildEM)
    # Only `note` is shared by name.
    assert dict(out) == {"note": "from-child"}


@pytest.mark.parametrize(
    ("inputs", "outputs", "expected_direction", "expected_side", "expected_field"),
    [
        ({"missing_sub": "a"}, None, "inputs", "subgraph", "missing_sub"),
        ({"input": "missing_parent"}, None, "inputs", "parent", "missing_parent"),
        (None, {"missing_parent": "input"}, "outputs", "parent", "missing_parent"),
        (None, {"a": "missing_sub"}, "outputs", "subgraph", "missing_sub"),
    ],
    ids=["inputs-sub", "inputs-parent", "outputs-parent", "outputs-sub"],
)
def test_explicit_mapping_validate_raises_on_undeclared_field(
    inputs: dict[str, str] | None,
    outputs: dict[str, str] | None,
    expected_direction: str,
    expected_side: str,
    expected_field: str,
) -> None:
    proj = ExplicitMapping[ParentEM, ChildEM](inputs=inputs, outputs=outputs)
    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        proj.validate(ParentEM, ChildEM)
    assert excinfo.value.direction == expected_direction
    assert excinfo.value.side == expected_side
    assert excinfo.value.field_name == expected_field


def test_explicit_mapping_validate_passes_when_all_fields_declared() -> None:
    proj = ExplicitMapping[ParentEM, ChildEM](
        inputs={"input": "a"},
        outputs={"captured": "result", "note": "note"},
    )
    proj.validate(ParentEM, ChildEM)
