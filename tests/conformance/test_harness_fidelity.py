"""Structural guards on the conformance harness itself."""

# Nothing here asserts engine behaviour. These tests assert that the harness's
# own allowlists still describe what its code does, because both have already
# drifted in ways no fixture run could reveal: a green conformance run is
# identical whether an assertion is live or dead. See the AGENTS.md note
# "Activating a conformance fixture is not done when it passes".

from __future__ import annotations

import ast
import inspect
from typing import Any

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
    tree = ast.parse(inspect.getsource(func))  # pyright: ignore[reportArgumentType]
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == param
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == param
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            names.add(node.slice.value)
    return names


def _assert_helpers_called_by(func: object, module: object) -> list[object]:
    """Module-level `_assert_*` functions `func` calls directly."""
    # One level, deliberately. Reading a guard's own body misses names it
    # delegates to a helper, which is not hypothetical: moving 148's two
    # invariants into `_assert_generation_usage_omission` made the check below
    # blind to them the moment it was written. Deeper recursion is the
    # transitive-derivation problem tracked separately; one level covers the
    # delegate-to-a-sibling shape that actually occurs here.
    tree = ast.parse(inspect.getsource(func))  # pyright: ignore[reportArgumentType]
    found: list[object] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            helper = getattr(module, node.func.id, None)
            if node.func.id.startswith("_assert_") and callable(helper) and helper not in found:
                found.append(helper)
    return found


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


def test_per_trace_invariants_covers_everything_assert_trace_reads() -> None:
    # `_assert_multi_traces` filters the fixture's invariants down to
    # `_PER_TRACE_INVARIANTS` before delegating. A name `_assert_trace` checks
    # but that is absent from the set is therefore dropped on that path, and the
    # fixture declaring it still passes.
    entry = langfuse_runner._assert_trace  # noqa: SLF001
    helpers = _assert_helpers_called_by(entry, langfuse_runner)
    # Non-vacuity on the callee walk: `_assert_trace` delegates today, so
    # finding none means the walk broke and the check silently narrowed back to
    # the entry point's own body.
    assert helpers, "found no _assert_* helpers called by _assert_trace; the callee walk is broken"
    read: set[str] = set()
    for func in [entry, *helpers]:
        read |= _invariant_names_read_by(func, "expected_invariants")
    assert read, "parsed no invariant lookups out of the per-trace guards, so this compared nothing"
    missing = sorted(read - set(langfuse_runner._PER_TRACE_INVARIANTS))  # noqa: SLF001
    assert not missing, (
        f"the per-trace guards check {missing}, but `_PER_TRACE_INVARIANTS` omits them, so the "
        f"multi-trace runner discards those claims silently."
    )
    # Stale names are the other direction: a set entry nothing reads means the
    # multi-trace path forwards a claim no guard evaluates.
    stale = sorted(set(langfuse_runner._PER_TRACE_INVARIANTS) - read)  # noqa: SLF001
    assert not stale, f"`_PER_TRACE_INVARIANTS` names invariants no per-trace guard reads: {stale}"
