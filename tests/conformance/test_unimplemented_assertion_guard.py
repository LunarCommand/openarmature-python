"""The guard on assertion keys declared for parsing but not implemented."""

# Fixtures 157 / 158 arrive with the v0.112.0 pin carrying leak assertions no
# comparator implements yet, and the `expected` model forbids extras, so those
# keys must be declared for the fixtures to parse at all. Declaring a field to
# make a fixture parse is how an assertion quietly becomes dead: the key
# validates, no comparator reads it, and the fixture reads like coverage.
#
# `_reject_unimplemented_assertions` exists to make that loud. Both fixtures are
# deferred, so NOTHING in the corpus reaches the guard today -- which means
# without these tests the guard would itself be unverified, the exact shape it
# was written to prevent.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conformance.harness.expectations import ObservabilityExpected

from . import test_observability_langfuse as langfuse_runner

_FIXTURE = "023-langfuse-generation-rendering"
_CASE = "generation_rendering"


def test_set_covers_every_leak_assertion_the_deferred_fixtures_use() -> None:
    # Derived from the CORPUS, not from the set under test. The parametrized test
    # below takes its cases from the set itself, so dropping a key there would
    # silently shrink the run rather than fail; this reads the fixtures and
    # notices a leak assertion that nothing guards -- including one a future pin
    # adds to 158.
    used: set[str] = set()
    for stem in ("157-langfuse-provider-isolation", "158-langfuse-payload-leak-fail-closed"):
        spec = langfuse_runner._load(langfuse_runner.CONFORMANCE_DIR / f"{stem}.yaml")
        for case in spec.get("cases") or [spec]:
            used |= set(case.get("expected") or {})

    leak_keys = {k for k in used if "langfuse_observations" in k or "payload_bearing" in k}
    assert leak_keys, "no leak assertions found in 157 / 158; this check is reading nothing"
    unguarded = sorted(leak_keys - langfuse_runner._UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS)
    assert not unguarded, (
        f"fixtures 157 / 158 assert {unguarded}, which the expectations model accepts for parsing "
        f"but no comparator implements and nothing guards. Add them to "
        f"_UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS, or implement them."
    )


def test_every_unimplemented_key_is_a_real_model_field() -> None:
    # A typo in the set would make the guard silently never match the key it was
    # meant to catch, so the set is pinned against the model that declares them.
    declared = set(ObservabilityExpected.model_fields)
    unknown = sorted(langfuse_runner._UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS - declared)
    assert not unknown, (
        f"{unknown} are listed as unimplemented assertions but are not fields on "
        f"ObservabilityExpected, so the guard would never fire for them. Fix the spelling or "
        f"drop them from the set."
    )


@pytest.mark.parametrize("key", sorted(langfuse_runner._UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS))
async def test_an_activated_case_reaching_an_unimplemented_assertion_fails(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    # Every key is driven, not a representative one: a guard that covers four of
    # five keys reads identically to one that covers all five.
    original = langfuse_runner._load

    def _loader(path: Path) -> dict[str, Any]:
        spec = original(path)
        for case in spec["cases"]:
            if case.get("name") == _CASE:
                case["expected"][key] = True
        return spec

    monkeypatch.setattr(langfuse_runner, "_load", _loader)
    with pytest.raises(AssertionError, match="does not implement"):
        await langfuse_runner.test_langfuse_fixture(langfuse_runner.CONFORMANCE_DIR / f"{_FIXTURE}.yaml")


async def test_a_case_using_no_unimplemented_assertion_passes() -> None:
    # Non-vacuity for the parametrized test above: the fixture it injects into
    # passes untouched, so the failures there come from the injected key.
    await langfuse_runner.test_langfuse_fixture(langfuse_runner.CONFORMANCE_DIR / f"{_FIXTURE}.yaml")
