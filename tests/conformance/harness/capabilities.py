"""Adapter-capability declaration and the per-case `requires_capability` gate."""

# Spec basis: conformance-adapter §5.5 (proposal 0116). Some contracts have arms
# only one kind of adapter can take -- the raise arm needs the adapter to be able
# to establish a Langfuse client's bound provider, which is not portably
# guaranteed -- so a fixture case carries `requires_capability` and each adapter
# declares what it can do. A case whose gate this adapter does not match is a
# RECOGNIZED skip (§8.1): the contract's other arm covers it, so not running it
# is correct rather than a coverage hole.
#
# The declaration lives in conformance.toml rather than here so the public
# conformance record and the test gate cannot disagree: it is the same statement,
# read once.

from __future__ import annotations

import tomllib
import warnings
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any, cast

_CONFORMANCE_TOML = Path(__file__).resolve().parents[3] / "conformance.toml"


class RecognizedSkip(UserWarning):
    """A fixture case that did not run because its capability arm is not ours."""


def report_recognized_skip(fixture_stem: str, case_name: str, reason: str) -> None:
    """Record a gated-out case where a PASSING run still shows it."""
    # §5.5 calls a gated-out case "a recognized skip, not a silently-omitted
    # case". Writing the reason into a local dict does not satisfy that: on a
    # passing run the dict is dropped and the output is identical whether the
    # fixture asserted every case or a third of them. A warning lands in pytest's
    # summary either way, so a gate that quietly swallows the MUST-bearing cases
    # of a fixture is visible without having to already suspect it.
    warnings.warn(
        f"{fixture_stem}::{case_name} did not run -- {reason}",
        RecognizedSkip,
        stacklevel=2,
    )


@cache
def adapter_capabilities() -> Mapping[str, bool]:
    """The capabilities this adapter declares, from conformance.toml."""
    # Values are required to be real booleans rather than coerced. bool("false")
    # is True, so a quoted TOML value would read as capable in the gate while the
    # published record renders false -- the two disagreeing is precisely what
    # single-sourcing the declaration is supposed to make impossible.
    with _CONFORMANCE_TOML.open("rb") as handle:
        manifest = tomllib.load(handle)
    declared: dict[str, Any] = manifest.get("adapter_capabilities") or {}
    for name, value in declared.items():
        if not isinstance(value, bool):
            raise AssertionError(
                f"conformance.toml [adapter_capabilities] {name} = {value!r} is "
                f"{type(value).__name__}, not a boolean. Write it unquoted (true / false) so "
                f"the published record and the gate cannot disagree."
            )
    return dict(declared)


def capability_skip_reason(requires: object) -> str | None:
    """Return why this case is gated out, or ``None`` when it should run."""
    # Pure by design: the caller decides how to record the skip. The conformance
    # runners iterate a fixture's cases inside a single parametrized test, so a
    # `pytest.skip` here would abandon the SIBLING cases too -- the reason the
    # existing per-case deferral hook `continue`s rather than skips.
    #
    # A case naming a capability the manifest does not declare raises. §5.5 lets
    # an adapter leave `langfuse_bound_provider_detection` undeclared and be
    # treated as non-capable; we require the value to be written down anyway,
    # because silence reads identically whether the adapter genuinely lacks the
    # capability or the manifest has not caught up with a newly-added capability
    # name -- and in the second case every gating case switches itself off while
    # still reporting green.
    if requires is None:
        return None
    if not isinstance(requires, Mapping):
        # Checked BEFORE the falsy test, which a malformed value would otherwise
        # slip through: `requires_capability: []` is falsy, so it would read as
        # ungated and run the case against an arm meant for a different adapter
        # class, silently. A non-empty malformed value would instead surface as
        # an AttributeError from `.items()`, which says nothing useful about the
        # fixture-authoring mistake behind it.
        raise AssertionError(
            f"case requires_capability must be a mapping of capability name to boolean, got "
            f"{type(requires).__name__}: {requires!r}. A non-mapping would otherwise read as "
            f"ungated and assert the wrong adapter arm."
        )
    if not requires:
        return None
    declared = adapter_capabilities()
    for name, required in cast("Mapping[str, Any]", requires).items():
        if not isinstance(required, bool):
            raise AssertionError(
                f"case requires_capability {name} = {required!r} is {type(required).__name__}, "
                f"not a boolean; a coerced value would select the wrong arm silently."
            )
        if name not in declared:
            raise AssertionError(
                f"case requires capability {name!r}, which conformance.toml "
                f"[adapter_capabilities] does not declare. Declare it (true or false) so the "
                f"gate is a decision rather than an accident; declared: {sorted(declared)}"
            )
        if declared[name] is not required:
            return (
                f"requires {name}={required}, this adapter declares "
                f"{name}={declared[name]} (recognized skip: the contract's other arm)"
            )
    return None


def assert_some_case_ran(
    fixture_stem: str,
    ran: int,
    excluded: Mapping[str, str],
) -> None:
    """Fail when a fixture executed no cases, whatever excluded them."""
    # A fixture that ran nothing is a vacuous pass wearing a green tick, and both
    # ways of excluding a case -- a capability gate and a per-case deferral -- are
    # designed to be quiet, which is what makes the empty run invisible. So the
    # check spans both channels rather than each policing itself: an all-deferred
    # fixture belongs in the fixture-level deferral table, where it skips visibly.
    #
    # `ran` counts EXECUTIONS, not the difference between a case total and the
    # size of `excluded`. Keying off `excluded` would miss a fixture whose cases
    # share a name (or are unnamed): two excluded cases collapse to one dict
    # entry, the count comes up short, and the empty run passes. `excluded` is
    # for the message only.
    if ran > 0:
        return
    detail = "; ".join(f"{name}: {reason}" for name, reason in sorted(excluded.items())) or "no cases at all"
    raise AssertionError(
        f"{fixture_stem}: executed no cases, so this fixture asserted nothing. Either an "
        f"exclusion is wrong (a stale deferral, a bad capability declaration) or the fixture "
        f"no longer applies to this adapter and should be de-activated at fixture level, "
        f"where it skips visibly. Excluded -- {detail}"
    )


__all__ = [
    "RecognizedSkip",
    "adapter_capabilities",
    "assert_some_case_ran",
    "capability_skip_reason",
    "report_recognized_skip",
]
