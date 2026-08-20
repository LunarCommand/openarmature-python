"""Structural guards on the conformance harness itself."""

# Nothing here asserts engine behaviour. These tests assert that the harness's
# own allowlists still describe what its code does, because both have already
# drifted in ways no fixture run could reveal: a green conformance run is
# identical whether an assertion is live or dead. See the AGENTS.md note
# "Activating a conformance fixture is not done when it passes".

from __future__ import annotations

import ast
import inspect
from typing import Any, cast

import pytest

from . import test_observability as otel_runner
from . import test_observability_langfuse as langfuse_runner

# Drivers whose `expected` keys are deliberately NOT declared in
# `_DRIVER_EXPECTED_KEYS`, with the reason they are safe to leave unguarded.
# The guard is fail-open, so an unregistered driver is invisible to it; listing
# them here is what turns "invisible" into "decided". A new driver appears in
# neither map and fails the test below, forcing that decision at the point it is
# added rather than leaving it dark indefinitely.
#
# A single-fixture driver cannot suffer the defect the guard exists to catch --
# a fixture routed to a SHARED driver that ignores one of its directives -- since
# it was written against exactly one fixture's `expected` block.
_UNGUARDED_SINGLE_FIXTURE_DRIVERS = frozenset(
    {
        "_run_fixture_001",
        "_run_fixture_002",
        "_run_fixture_003",
        "_run_fixture_004",
        "_run_fixture_005",
        "_run_fixture_006",
        "_run_fixture_007",
        "_run_fixture_008",
        "_run_fixture_009",
        "_run_fixture_010",
        "_run_fixture_011",
        "_run_fixture_028",
        "_run_fixture_038",
        "_run_fixture_050",
        "_run_fixture_051",
        "_run_fixture_052",
        "_run_fixture_053",
        "_run_fixture_054",
        "_run_fixture_055",
        "_run_fixture_056",
        "_run_fixture_058",
        "_run_fixture_084",
        "_run_fixture_110",
        "_run_fixture_132",
        "_run_fixture_133",
        "_run_structured_output_error_span_fixture",
    }
)

# Shared drivers that ARE exposed to the mis-routing defect and are not yet
# registered. This is a real gap, recorded rather than hidden: registering one
# means deriving its read set from its body (and its callees), which is the step
# that produced a wrong `observers` claim when done by inspection alone.
_UNGUARDED_SHARED_DRIVERS = frozenset(
    {
        "_run_get_invocation_metadata_fixture",
        "_run_llm_cache_fixture",
        "_run_tool_fixture",
        "_run_typed_event_chain_cases",
    }
)


def test_every_dispatched_driver_is_registered_or_explicitly_unguarded() -> None:
    dispatched = set(otel_runner._driver_for_fixture().values())  # noqa: SLF001
    accounted = (
        set(otel_runner._DRIVER_EXPECTED_KEYS)  # noqa: SLF001
        | _UNGUARDED_SINGLE_FIXTURE_DRIVERS
        | _UNGUARDED_SHARED_DRIVERS
    )
    unaccounted = sorted(dispatched - accounted)
    assert not unaccounted, (
        f"drivers {unaccounted} are dispatched to but appear in neither `_DRIVER_EXPECTED_KEYS` nor "
        f"the unguarded lists in this file. Register the driver's `expected` keys, or add it to an "
        f"unguarded list with the reason -- an unregistered driver silently exempts its fixtures "
        f"from the mis-routing guard."
    )
    # Non-vacuity on the INPUT: an empty dispatch map satisfies the subset check
    # above trivially, which is how a guard over nothing reads as a clean pass.
    assert dispatched, "the dispatch map is empty, so the check above compared nothing"
    # And the lists must not accumulate names for drivers that no longer exist.
    stale = sorted((_UNGUARDED_SINGLE_FIXTURE_DRIVERS | _UNGUARDED_SHARED_DRIVERS) - dispatched)
    assert not stale, f"unguarded lists name drivers nothing dispatches to: {stale}"


def _invariant_names_read_by(func: object, param: str) -> set[str]:
    """String literals `func` looks up on the mapping named `param`."""
    return _invariant_names_read_by_node(
        ast.parse(inspect.getsource(func)),  # pyright: ignore[reportArgumentType]
        param,
    )


def _invariant_helpers_called_by(func: object, module: object) -> list[object]:
    """Module-level callees of `func` that take an `expected_invariants` param."""
    # One level, deliberately. Reading a guard's own body misses names it
    # delegates to a helper, which is not hypothetical: moving 148's two
    # invariants into `_assert_generation_usage_omission` made the check below
    # blind to them the moment it was written. Deeper recursion is the
    # transitive-derivation problem tracked separately; one level covers the
    # delegate-to-a-sibling shape that actually occurs here.
    #
    # Selected by SIGNATURE, not by an `_assert_` name prefix. A prefix filter
    # silently excludes a future guard spelled `_check_...`, and excluding it
    # from the walk is the fail-open direction this whole check exists to close.
    tree = ast.parse(inspect.getsource(func))  # pyright: ignore[reportArgumentType]
    found: list[object] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        helper = getattr(module, node.func.id, None)
        if not callable(helper) or helper in found:
            continue
        try:
            params = inspect.signature(helper).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        if "expected_invariants" in params:
            found.append(helper)
    return found


def _invariant_names_evaluated_by(func: object, param: str) -> set[str]:
    """Names from `_invariant_names_read_by` whose `if` body actually does work."""
    # A name is MENTIONED when `if invariants.get("x"):` appears; it is
    # EVALUATED when that branch asserts something. `_assert_trace` contains a
    # deliberate mention-only arm (`trace_id_equals_invocation_id`, whose body is
    # a comment and `pass`), so counting mentions as evaluation lets a guard be
    # gutted to `if x: pass` while every check here stays green.
    tree = ast.parse(inspect.getsource(func))  # pyright: ignore[reportArgumentType]
    # Guards spell the lookup two ways: inline in the test (`if inv.get("x"):`)
    # and bound to a local first (`x = inv.get("x")` ... `if x:`), which is what
    # a guard reading several names for one early-return does. Resolve the
    # binding so the check does not quietly mandate one of the two idioms.
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            names = _invariant_names_read_by_node(node.value, param)
            if len(names) == 1:
                bound[node.targets[0].id] = next(iter(names))

    evaluated: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = [s for s in node.body if not _is_inert(s)]
        if not body:
            continue
        evaluated |= _invariant_names_read_by_node(node.test, param)
        evaluated |= {bound[n.id] for n in ast.walk(node.test) if isinstance(n, ast.Name) and n.id in bound}
    return evaluated


def _is_inert(stmt: ast.stmt) -> bool:
    """A statement that asserts nothing about the invariant."""
    # A BARE `return` counts as inert, which is not a nicety: a guard reading
    # several names typically opens with `if not a and not b: return`, and that
    # arm mentions every name while checking none of them. Counting it as
    # evaluation let the gutted-to-`pass` mutant survive this whole check.
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Return) and stmt.value is None:
        return True
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _invariant_names_read_by_node(node: ast.AST, param: str) -> set[str]:
    """The `_invariant_names_read_by` extraction, over an arbitrary subtree."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "get"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == param
            and sub.args
            and isinstance(sub.args[0], ast.Constant)
            and isinstance(sub.args[0].value, str)
        ):
            names.add(sub.args[0].value)
        elif (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == param
            and isinstance(sub.slice, ast.Constant)
            and isinstance(sub.slice.value, str)
        ):
            names.add(sub.slice.value)
    return names


# Per-trace invariant names whose arm is deliberately a no-op, with the reason.
# Listed so the stale check can tell "documented as inert" from "silently gutted".
_DOCUMENTARY_PER_TRACE_INVARIANTS = {
    # §8.4.1 says trace.id == invocation_id, and there is no accessor for the
    # invocation_id from outside the observer, so the claim degenerates to
    # "trace.id matches the UUIDv4 pattern" -- already asserted via the
    # `<uuid>` placeholder on the id itself.
    "trace_id_equals_invocation_id",
}


def _span_dependent_invariants_in(func: object) -> set[str]:
    """Invariant names whose `if invariants.get(...)` body reads the span set."""
    tree = ast.parse(inspect.getsource(func))  # pyright: ignore[reportArgumentType]
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {
            call.args[0].value
            for call in ast.walk(node.test)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "get"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "invariants"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        }
        if not names:
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        if any(isinstance(n, ast.Name) and n.id in {"spans", "_spans_carrying"} for n in ast.walk(body)):
            found |= names
    return found


def test_span_dependent_invariants_lists_every_span_reading_claim() -> None:
    # `_assert_span_anchor` only fires for names in `_SPAN_DEPENDENT_INVARIANTS`.
    # A span-reading claim missing from that set gets no positive anchor, so an
    # empty span set satisfies it trivially -- which is what the `startswith`
    # heuristic this set replaced actually did to the two
    # `exceeded_span_signal_*` names.
    derived = _span_dependent_invariants_in(
        otel_runner._assert_metrics_invariants  # noqa: SLF001
    ) | _span_dependent_invariants_in(otel_runner._assert_token_budget_invariants)  # noqa: SLF001
    assert derived, "parsed no span-reading invariant blocks, so this compared nothing"
    missing = sorted(derived - otel_runner._SPAN_DEPENDENT_INVARIANTS)  # noqa: SLF001
    assert not missing, (
        f"{missing} read the span set but are absent from `_SPAN_DEPENDENT_INVARIANTS`, so no "
        f"positive anchor runs for them and a zero-span run satisfies them vacuously."
    )
    stale = sorted(otel_runner._SPAN_DEPENDENT_INVARIANTS - derived)  # noqa: SLF001
    assert not stale, f"`_SPAN_DEPENDENT_INVARIANTS` names claims that no longer read spans: {stale}"


def _drive_metrics_guard(case: dict[str, Any]) -> None:
    otel_runner._assert_metrics_invariants(case, [], [])  # noqa: SLF001


def _drive_token_budget_guard(case: dict[str, Any]) -> None:
    otel_runner._assert_token_budget_invariants(case, [], {}, [])  # noqa: SLF001


@pytest.mark.parametrize("drive", [_drive_metrics_guard, _drive_token_budget_guard])
@pytest.mark.parametrize("invariant", sorted(otel_runner._SPAN_DEPENDENT_INVARIANTS))  # noqa: SLF001
def test_a_zero_span_run_cannot_satisfy_a_span_absence_claim(invariant: str, drive: Any) -> None:
    # The test above keeps `_SPAN_DEPENDENT_INVARIANTS` honest about which names
    # read spans; this one keeps the ANCHOR ITSELF live. Deleting
    # `_assert_span_anchor`, or its call from either guard, leaves every shipping
    # fixture green -- both fixtures that declare a span-absence claim also
    # declare `span_tree`, whose own root-span check fails first, so nothing in
    # the corpus would notice. Driving each guard directly is what makes the
    # anchor's removal detectable at all.
    with pytest.raises(AssertionError, match="pass vacuously"):
        drive({"expected": {"invariants": {invariant: True}}})


def test_per_trace_invariants_matches_what_the_per_trace_guards_evaluate() -> None:
    # `_assert_multi_traces` filters the fixture's invariants down to
    # `_PER_TRACE_INVARIANTS` before delegating. A name `_assert_trace` checks
    # but that is absent from the set is therefore dropped on that path, and the
    # fixture declaring it still passes.
    entry = langfuse_runner._assert_trace  # noqa: SLF001
    helpers = _invariant_helpers_called_by(entry, langfuse_runner)
    # Non-vacuity on the callee walk: `_assert_trace` delegates today, so
    # finding none means the walk broke and the check silently narrowed back to
    # the entry point's own body.
    assert helpers, "found no invariant helpers called by _assert_trace; the callee walk is broken"
    read: set[str] = set()
    evaluated: set[str] = set()
    for func in [entry, *helpers]:
        read |= _invariant_names_read_by(func, "expected_invariants")
        evaluated |= _invariant_names_evaluated_by(func, "expected_invariants")
    assert read, "parsed no invariant lookups out of the per-trace guards, so this compared nothing"
    assert evaluated, "parsed no EVALUATED invariant arms; the body-triviality filter is broken"
    missing = sorted(read - set(langfuse_runner._PER_TRACE_INVARIANTS))  # noqa: SLF001
    assert not missing, (
        f"the per-trace guards check {missing}, but `_PER_TRACE_INVARIANTS` omits them, so the "
        f"multi-trace runner discards those claims silently."
    )
    # The other direction, and on EVALUATED rather than merely mentioned: a set
    # entry whose arm is `if name: pass` reads as covered while asserting
    # nothing, so gutting a guard to its `if` would otherwise stay green.
    stale = sorted(
        set(langfuse_runner._PER_TRACE_INVARIANTS)  # noqa: SLF001
        - evaluated
        - _DOCUMENTARY_PER_TRACE_INVARIANTS
    )
    assert not stale, (
        f"`_PER_TRACE_INVARIANTS` names invariants no per-trace guard evaluates: {stale}. Either "
        f"the guard was gutted, or the arm is deliberately inert and belongs in "
        f"`_DOCUMENTARY_PER_TRACE_INVARIANTS` with its reason."
    )


def test_langfuse_harness_fixtures_are_all_driven_by_the_langfuse_runner() -> None:
    # Activating a Langfuse-mapping fixture takes two independent edits in two
    # files: adding it to `_LANGFUSE_HARNESS_FIXTURES` here, which makes the OTel
    # runner skip it, and to `_LANGFUSE_FIXTURES` there, which makes the Langfuse
    # runner parametrize it. Do only the first and the fixture runs NOWHERE: the
    # OTel runner skips it saying it is tested by the sibling, the sibling never
    # collects it, and the coverage guard counts it as accounted because the
    # skip set is unioned into `accounted`. One green suite, zero assertions,
    # and a skip message actively claiming coverage.
    skipped_here = set(otel_runner._LANGFUSE_HARNESS_FIXTURES)  # noqa: SLF001
    driven_there = set(langfuse_runner._LANGFUSE_FIXTURES)  # noqa: SLF001
    assert skipped_here, "the skip set is empty, so the check below compared nothing"
    undriven = sorted(skipped_here - driven_there)
    assert not undriven, (
        f"{undriven} are skipped by the OTel runner as 'fixture-tested by the Langfuse "
        f"harness', but the Langfuse runner does not collect them, so they run nowhere while "
        f"the coverage guard counts them as accounted. Add them to `_LANGFUSE_FIXTURES`."
    )


# Invariant names the Langfuse runner IMPLEMENTS, mapped to the fixture whose
# claim rests on them. Most invariants in this corpus are documentary -- they
# restate what a concrete directive already pins -- so a blanket "every declared
# name must be read" guard would fail the ~50 working exactly as intended.
# These are the opposite: the fixture's `langfuse_trace` block CANNOT express
# the claim, so the invariant is the only thing asserting it.
_LOAD_BEARING_LANGFUSE_INVARIANTS = {
    # 148: `usage` is compared by iterating the EXPECTED keys, so an omitted
    # `input` key -- the whole claim -- is invisible to the directive.
    "generation_usage_input_omitted_when_prompt_tokens_null": (
        "148-langfuse-generation-usage-omits-input-on-null-counter"
    ),
    "generation_usage_output_and_total_present_when_sound": (
        "148-langfuse-generation-usage-omits-input-on-null-counter"
    ),
    # 156 declares no `level` at all, so this is its only expression. 155 pins
    # `level: ERROR` concretely, which is why only the 156 direction is here.
    "no_warning_level_under_budget": "156-langfuse-token-budget-under-budget-flag-false",
}


def test_load_bearing_langfuse_invariants_are_still_declared() -> None:
    # A guard keyed on an invariant name goes dark the moment the fixture stops
    # declaring that name -- a spec-side rename is the routine way that happens,
    # and a pin bump brings those. Nothing else catches it: the guard early-
    # returns, the `langfuse_trace` block passes on its own, the
    # assert-something check is satisfied by that block's presence, and the
    # unknown-directive check never descends into `expected.invariants`. The
    # fixture reports green while asserting nothing it was activated for.
    import yaml  # noqa: PLC0415

    declared: dict[str, set[str]] = {}
    for path in langfuse_runner._fixture_paths():  # noqa: SLF001
        spec = cast("dict[str, Any]", yaml.safe_load(path.read_text()))
        cases = cast("list[dict[str, Any]]", spec.get("cases") or [spec])
        for case in cases:
            expected = cast("dict[str, Any]", case.get("expected") or {})
            for name in cast("dict[str, Any]", expected.get("invariants") or {}):
                declared.setdefault(name, set()).add(path.stem)
    assert declared, "parsed no invariants out of the corpus, so this compared nothing"

    for name, fixture in sorted(_LOAD_BEARING_LANGFUSE_INVARIANTS.items()):
        assert fixture in declared.get(name, set()), (
            f"{fixture} no longer declares {name!r}, but the Langfuse runner still implements "
            f"a guard keyed on that name. The guard is now dead and the fixture's claim rests "
            f"on nothing. Re-point the guard at the fixture's new spelling."
        )
