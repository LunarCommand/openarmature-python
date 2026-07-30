"""Declared same-name projection boundary + the reducer round-trip warning.

Covers the third projection form (field-name sets, checked and with no
field-name-matching fallback), its compile diagnostics, and the compile-time
warning raised when a projection round-trips a field through a reducer that
grows on re-application.
"""

# Proposal 0094 (graph-engine §2). Behavior + unit coverage ship ahead of
# the pin (spec v0.89.0 is beyond the current v0.88.0 pin); the conformance
# fixtures ride the v0.17.0 pin bump.

import warnings
from typing import Annotated, Any, cast

import pytest
from pydantic import Field

from openarmature.graph import (
    END,
    BranchSpec,
    CompileWarning,
    ConflictingProjectionForms,
    DeclaredSameName,
    ExplicitMapping,
    GraphBuilder,
    MappingReferencesUndeclaredField,
    NoDeclaredEntry,
    ProjectionReducerRoundTrip,
    Reducer,
    State,
    append,
    concat_flatten,
)


class Child(State):
    query: str = ""
    answer: str = ""
    notes: Annotated[list[str], append] = Field(default_factory=list)


class Parent(State):
    query: str = ""
    answer: str = ""
    # append is not round-trip-idempotent: a projected-in-and-back-out value
    # merges a second time and the list doubles.
    notes: Annotated[list[str], append] = Field(default_factory=list)


async def _child_node(state: Child) -> dict[str, Any]:
    return {"answer": f"ans({state.query})", "notes": ["n1"]}


def _child_graph() -> Any:
    return GraphBuilder(Child).add_node("c", _child_node).add_edge("c", END).set_entry("c").compile()


def _parent_with(projection: Any) -> GraphBuilder[Parent]:
    builder: GraphBuilder[Parent] = GraphBuilder(Parent)
    builder.add_subgraph_node("sg", _child_graph(), projection)
    builder.add_edge("sg", END)
    builder.set_entry("sg")
    return builder


def _compile_capturing_warnings(builder: GraphBuilder[Parent]) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        builder.compile()
    return list(captured)


def _compile_capturing_any(builder: Any) -> list[warnings.WarningMessage]:
    """State-type-agnostic variant, for the locally-declared schemas below."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        builder.compile()
    return list(captured)


async def test_declared_boundary_projects_named_fields_only() -> None:
    # In-set copies the parent value into the same-name subgraph field;
    # out-set merges the same-name subgraph field back. A subgraph field in
    # neither set is discarded (``notes`` here), even though its name
    # matches a parent field -- the declared form does not name-match.
    graph = _parent_with(DeclaredSameName(in_fields={"query"}, out_fields={"answer"})).compile()

    final = await graph.invoke(Parent(query="Q"))

    assert final.answer == "ans(Q)"
    assert final.notes == []


async def test_declared_empty_out_set_projects_nothing() -> None:
    # A present-but-empty out-set projects nothing out. The declared form is
    # a complete declaration with no absent-vs-empty subtlety: it does NOT
    # fall back to field-name matching the way the maps' absent ``outputs``
    # does, so an accidentally-empty collection cannot silently re-enable it.
    graph = _parent_with(DeclaredSameName(in_fields={"query"}, out_fields=set())).compile()

    final = await graph.invoke(Parent(query="Q"))

    assert final.answer == ""


async def test_default_form_still_field_name_matches() -> None:
    # The paired default-form case: with no declaration at all, projection
    # out still field-name matches. This is what makes the empty-out-set
    # case above a distinction rather than a coincidence.
    builder: GraphBuilder[Parent] = GraphBuilder(Parent)
    builder.add_subgraph_node("sg", _child_graph())
    builder.add_edge("sg", END)
    builder.set_entry("sg")

    final = await builder.compile().invoke(Parent(query="Q"))

    assert final.answer == "ans()"
    assert final.notes == ["n1"]


@pytest.mark.parametrize(
    ("projection", "expected_side"),
    [
        pytest.param(
            DeclaredSameName[Parent, Child](in_fields={"nope"}),
            "parent",
            id="in_set_absent_on_parent",
        ),
        pytest.param(
            DeclaredSameName[Parent, Child](out_fields={"nope"}),
            "parent",
            id="out_set_absent_on_parent",
        ),
    ],
)
async def test_declared_boundary_drift_is_a_compile_error(projection: Any, expected_side: str) -> None:
    # The whole point of the declared form: a name that does not exist is a
    # compile error instead of a silently dropped field.
    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        _parent_with(projection).compile()

    assert excinfo.value.category == "mapping_references_undeclared_field"
    assert excinfo.value.field_name == "nope"
    assert excinfo.value.side == expected_side


async def test_declared_name_present_on_parent_but_not_subgraph_is_a_compile_error() -> None:
    # Same-name means the name must exist on BOTH schemas; drift on the
    # subgraph side is caught symmetrically.
    class ParentOnly(State):
        only_here: str = ""

    builder: GraphBuilder[ParentOnly] = GraphBuilder(ParentOnly)
    builder.add_subgraph_node("sg", _child_graph(), DeclaredSameName(in_fields={"only_here"}))
    builder.add_edge("sg", END)
    builder.set_entry("sg")

    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        builder.compile()

    assert excinfo.value.side == "subgraph"


async def test_conflicting_projection_forms_fails_compilation() -> None:
    # The two bundled declarative strategies each expose exactly one form,
    # so neither can conflict on its own. The rule is enforced duck-typed
    # over whatever a strategy declares, so a custom strategy exposing both
    # a set form and a map form fails rather than the rule being absent.
    class BothForms(DeclaredSameName[Parent, Child]):
        inputs = {"query": "query"}
        outputs = None

    with pytest.raises(ConflictingProjectionForms) as excinfo:
        _parent_with(BothForms(in_fields={"query"})).compile()

    assert excinfo.value.category == "conflicting_projection_forms"
    assert excinfo.value.node_name == "sg"


async def test_round_trip_into_non_idempotent_reducer_warns() -> None:
    # ``notes`` is projected in and back out through the parent's ``append``
    # reducer, so the unchanged value merges a second time and the list
    # doubles. MUST-level: ``append`` is a canonical reducer whose
    # idempotency is statically determinable.
    captured = _compile_capturing_warnings(
        _parent_with(DeclaredSameName(in_fields={"notes"}, out_fields={"notes"}))
    )

    assert len(captured) == 1
    warning = captured[0].message
    assert isinstance(warning, ProjectionReducerRoundTrip)
    assert warning.category == "projection_reducer_round_trip"
    assert warning.field_name == "notes"
    assert warning.reducer_name == "append"
    assert warning.node_name == "sg"


async def test_round_trip_into_idempotent_reducer_is_silent() -> None:
    # ``answer`` carries the default ``last_write_wins``: re-applying an
    # already-merged value is a no-op, so the round-trip is harmless. The
    # exhaustive-no-warning assertion is what catches an over-warning
    # implementation.
    captured = _compile_capturing_warnings(
        _parent_with(DeclaredSameName(in_fields={"answer"}, out_fields={"answer"}))
    )

    assert captured == []


async def test_same_parent_field_via_different_subgraph_fields_is_not_a_round_trip() -> None:
    # ``notes`` is an inputs SOURCE and an outputs TARGET, but through two
    # different subgraph fields, so the value merged back is a distinct,
    # subgraph-computed value rather than the one carried in. The
    # no-false-positive case.
    captured = _compile_capturing_warnings(
        _parent_with(ExplicitMapping(inputs={"query": "notes"}, outputs={"notes": "answer"}))
    )

    assert captured == []


async def test_round_trip_warns_on_the_fan_out_surface() -> None:
    # The warning applies wherever a projection-out merges through a parent
    # reducer, which includes a fan-out's inputs / extra_outputs pair.
    inner = GraphBuilder(Child).add_node("c", _child_node).add_edge("c", END).set_entry("c").compile()
    builder: GraphBuilder[Parent] = GraphBuilder(Parent)
    builder.add_fan_out_node(
        "spread",
        subgraph=inner,
        count=2,
        collect_field="answer",
        target_field="answer",
        inputs={"notes": "notes"},
        extra_outputs={"notes": "notes"},
    )
    builder.add_edge("spread", END)
    builder.set_entry("spread")

    captured = _compile_capturing_warnings(builder)

    assert len(captured) == 1
    warning = captured[0].message
    assert isinstance(warning, ProjectionReducerRoundTrip)
    assert warning.field_name == "notes"
    assert warning.node_name == "spread"


def _fan_out_builder(*, collect_field: str, target_field: str, inputs: dict[str, str]) -> Any:
    inner = GraphBuilder(Child).add_node("c", _child_node).add_edge("c", END).set_entry("c").compile()
    builder: GraphBuilder[Parent] = GraphBuilder(Parent)
    builder.add_fan_out_node(
        "spread",
        subgraph=inner,
        count=2,
        collect_field=collect_field,
        target_field=target_field,
        inputs=inputs,
    )
    builder.add_edge("spread", END)
    builder.set_entry("spread")
    return builder


async def test_round_trip_warns_on_the_fan_out_collect_channel() -> None:
    # Proposal 0111: the fan-out PRIMARY output channel round-trips too. The
    # collect_field is seeded from target_field via inputs and collected
    # straight back into that same target_field, so the parent value merges
    # again once per instance. collect_field / target_field are scalars, not a
    # map, which is why a map-pair resolver missed this.
    captured = _compile_capturing_warnings(
        _fan_out_builder(collect_field="notes", target_field="notes", inputs={"notes": "notes"})
    )

    assert len(captured) == 1
    warning = captured[0].message
    assert isinstance(warning, ProjectionReducerRoundTrip)
    assert warning.field_name == "notes"
    assert warning.node_name == "spread"


class _FlatParent(State):
    # concat_flatten, not append: the fan-in hands the parent a LIST of
    # per-instance values, which only concat_flatten can consume. An
    # append-reduced target_field fails state validation instead of doubling.
    results: Annotated[list[str], concat_flatten] = Field(default_factory=list)


class _FlatChild(State):
    acc: list[str] = Field(default_factory=list)


async def _flat_node(state: _FlatChild) -> dict[str, Any]:
    return {"acc": [*state.acc, "new"]}


def _flat_fan_out(**kwargs: Any) -> GraphBuilder[_FlatParent]:
    inner = GraphBuilder(_FlatChild).add_node("c", _flat_node).add_edge("c", END).set_entry("c").compile()
    builder: GraphBuilder[_FlatParent] = GraphBuilder(_FlatParent)
    builder.add_fan_out_node("spread", subgraph=inner, **kwargs)
    builder.add_edge("spread", END)
    builder.set_entry("spread")
    return builder


async def test_fan_out_collect_round_trip_re_merges_once_per_instance() -> None:
    # Pins the CONSEQUENCE the warning describes, not just the warning: the
    # seed value comes back once per instance. This is also the worked
    # example in docs/concepts/composition.md, so the doc is pinned here too.
    builder = _flat_fan_out(count=2, collect_field="acc", target_field="results", inputs={"acc": "results"})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        graph = builder.compile()

    assert [type(c.message).__name__ for c in captured] == ["ProjectionReducerRoundTrip"]

    final = await graph.invoke(_FlatParent(results=["seed"]))
    assert final.results == ["seed", "seed", "new", "seed", "new"]


async def test_fan_out_item_seeded_collect_round_trip_warns() -> None:
    # The items_field / item_field spelling seeds each instance with one
    # ELEMENT of the parent list and collects straight back, so it
    # round-trips just as the inputs spelling does. Adopted ahead of an
    # explicit spec condition (0111 §9.3 names only inputs) and flagged.
    builder = _flat_fan_out(
        items_field="results", item_field="acc", collect_field="acc", target_field="results"
    )
    captured = [c for c in _compile_capturing_any(builder) if isinstance(c.message, CompileWarning)]

    assert len(captured) == 1
    assert cast("ProjectionReducerRoundTrip", captured[0].message).field_name == "results"


async def test_fan_out_collect_same_name_without_seeding_is_silent() -> None:
    # The single-variable negative: collect_field == target_field on a
    # non-idempotent reducer, but NOT seeded from it. A naive predicate that
    # only compared collect_field to target_field would warn here, so this is
    # what pins the rule to the round-trip CONDITION.
    captured = _compile_capturing_any(
        _flat_fan_out(count=2, collect_field="acc", target_field="results", inputs={})
    )

    assert [c for c in captured if isinstance(c.message, CompileWarning)] == []


async def test_clean_fan_out_collect_channel_is_silent() -> None:
    # The no-false-positive half of the rule: a collect_field NOT seeded from
    # target_field carries a distinct, subgraph-computed value back, so
    # collecting into a growing reducer is not by itself a round-trip. This
    # pins the trigger as the round-trip CONDITION rather than "collect into a
    # non-idempotent reducer".
    captured = _compile_capturing_warnings(
        _fan_out_builder(collect_field="answer", target_field="notes", inputs={"query": "query"})
    )

    assert captured == []


async def test_round_trip_warns_on_the_parallel_branches_surface() -> None:
    # The third surface: a subgraph branch's inputs / outputs pair.
    inner = GraphBuilder(Child).add_node("c", _child_node).add_edge("c", END).set_entry("c").compile()
    builder: GraphBuilder[Parent] = GraphBuilder(Parent)
    builder.add_parallel_branches_node(
        "split",
        branches={
            "alpha": BranchSpec(
                subgraph=inner,
                inputs={"notes": "notes"},
                outputs={"notes": "notes"},
            )
        },
    )
    builder.add_edge("split", END)
    builder.set_entry("split")

    captured = _compile_capturing_warnings(builder)

    assert len(captured) == 1
    warning = captured[0].message
    assert isinstance(warning, ProjectionReducerRoundTrip)
    assert warning.field_name == "notes"
    assert warning.node_name == "split"


async def test_explicit_mapping_absent_outputs_round_trip_warns() -> None:
    # Adversarial-review finding: with ``outputs`` absent the effective
    # projection-out is field-name matching, NOT "nothing", so a same-name
    # ``inputs`` entry on a shared field round-trips exactly like an
    # explicit outputs pair. This is the most idiomatic spelling of the
    # hazard and the shape 0094's motivation cites.
    captured = _compile_capturing_warnings(_parent_with(ExplicitMapping(inputs={"notes": "notes"})))

    assert len(captured) == 1
    warning = captured[0].message
    assert isinstance(warning, ProjectionReducerRoundTrip)
    assert warning.field_name == "notes"

    # And the doubling it warns about is real.
    builder = _parent_with(ExplicitMapping(inputs={"notes": "notes"}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileWarning)
        graph = builder.compile()
    final = await graph.invoke(Parent(notes=["seed"]))
    assert final.notes == ["seed", "seed", "n1"]


async def test_explicit_mapping_absent_outputs_renamed_field_is_silent() -> None:
    # The fallback carries a field back under its OWN name, so an inputs
    # entry that renames (subgraph field != parent field) is not a
    # round-trip: parent ``notes`` arrives as subgraph ``query``, and it is
    # subgraph ``notes`` (untouched) that name-matches back.
    captured = _compile_capturing_warnings(_parent_with(ExplicitMapping(inputs={"query": "notes"})))

    assert captured == []


async def test_advisory_warning_never_preempts_a_compile_error() -> None:
    # Adversarial-review finding: the advisory must not mask a MUST-fail
    # structural error for a caller whose filters escalate warnings.
    inner = _child_graph()
    builder: GraphBuilder[Parent] = GraphBuilder(Parent)
    builder.add_subgraph_node("sg", inner, DeclaredSameName(in_fields={"notes"}, out_fields={"notes"}))
    # No set_entry -> NoDeclaredEntry, which MUST win over the advisory.
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileWarning)
        with pytest.raises(NoDeclaredEntry):
            builder.compile()


def test_custom_reducer_classification_drives_the_warning() -> None:
    # The SHOULD-level custom-reducer path: an author opting in with
    # round_trip_idempotent = False warns; leaving it unset stays silent,
    # which is the conforming choice for an unclassifiable reducer.
    class GrowingReducer(Reducer):
        name = "growing"
        round_trip_idempotent = False

        def __call__(self, prior: Any, update: Any) -> Any:
            return [*prior, *update]

    class UnclassifiedReducer(Reducer):
        name = "unclassified"

        def __call__(self, prior: Any, update: Any) -> Any:
            return [*prior, *update]

    for reducer, expected in ((GrowingReducer(), 1), (UnclassifiedReducer(), 0)):

        class ParentCustom(State):
            notes: Annotated[list[str], reducer] = Field(default_factory=list)

        class ChildCustom(State):
            notes: list[str] = Field(default_factory=list)

        async def _node(_s: ChildCustom) -> dict[str, Any]:
            return {"notes": ["n1"]}

        inner = GraphBuilder(ChildCustom).add_node("c", _node).add_edge("c", END).set_entry("c").compile()
        builder: GraphBuilder[ParentCustom] = GraphBuilder(ParentCustom)
        builder.add_subgraph_node("sg", inner, DeclaredSameName(in_fields={"notes"}, out_fields={"notes"}))
        builder.add_edge("sg", END)
        builder.set_entry("sg")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            builder.compile()
        assert len(captured) == expected, f"{reducer.name} expected {expected} warning(s)"


def test_declared_same_name_rejects_a_bare_string() -> None:
    # A bare str satisfies Iterable[str], so without this guard
    # in_fields="notes" would silently become {'e','n','o','s','t'}.
    with pytest.raises(TypeError, match="not a bare str"):
        DeclaredSameName[Parent, Child](in_fields="notes")
    with pytest.raises(TypeError, match="not a bare str"):
        DeclaredSameName[Parent, Child](out_fields="notes")


async def test_round_trip_warning_does_not_fail_compilation() -> None:
    # A warning is advisory: compilation succeeds and the graph runs, with
    # the doubling the warning describes actually happening.
    builder = _parent_with(DeclaredSameName(in_fields={"notes"}, out_fields={"notes"}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileWarning)
        graph = builder.compile()

    final = await graph.invoke(Parent(notes=["seed"]))

    # Projected in as ["seed"], the subgraph appends "n1" -> ["seed", "n1"],
    # then that whole list merges back through the parent's append reducer.
    assert final.notes == ["seed", "seed", "n1"]
