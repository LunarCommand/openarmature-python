"""FieldNameMatching projection: overlap, partial overlap, no overlap."""

from openarmature.graph import FieldNameMatching, State
from pydantic import Field


class Parent(State):
    shared: str = "parent-shared"
    parent_only: int = 0


class ChildOverlap(State):
    shared: str = "child-shared"
    child_only: list[str] = Field(default_factory=list)


class ChildNoOverlap(State):
    completely_different: str = "x"


def test_project_in_returns_subgraph_defaults() -> None:
    """Per spec v0.1.1 §2: default projection-in is no projection — subgraph
    starts from its own field defaults regardless of parent state."""

    proj = FieldNameMatching()
    sub = proj.project_in(Parent(shared="ignored"), ChildOverlap)
    assert isinstance(sub, ChildOverlap)
    assert sub.shared == "child-shared"


def test_project_out_returns_only_shared_field_names() -> None:
    proj = FieldNameMatching()
    sub_final = ChildOverlap(shared="child-final", child_only=["a"])
    out = proj.project_out(sub_final, Parent(), ChildOverlap)

    # Only `shared` is in both schemas; `child_only` is dropped.
    assert dict(out) == {"shared": "child-final"}


def test_project_out_is_empty_when_no_overlap() -> None:
    proj = FieldNameMatching()
    sub_final = ChildNoOverlap(completely_different="y")
    out = proj.project_out(sub_final, Parent(), ChildNoOverlap)
    assert dict(out) == {}
