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
    accounted = (
        langfuse_runner._IMPLEMENTED_LEAK_ASSERTIONS | langfuse_runner._UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS
    )
    unaccounted = sorted(leak_keys - accounted)
    assert not unaccounted, (
        f"fixtures 157 / 158 assert {unaccounted}, which the expectations model accepts for "
        f"parsing but which is neither implemented nor guarded. Implement it, or add it to "
        f"_UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS so an activated fixture reaching it fails loudly."
    )
    overlap = sorted(
        langfuse_runner._IMPLEMENTED_LEAK_ASSERTIONS & langfuse_runner._UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS
    )
    assert not overlap, f"{overlap} are listed as both implemented and unimplemented"


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


def _inject(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    """Graft `key` onto the first case of whatever fixture the runner loads."""
    original = langfuse_runner._load

    def _loader(path: Path) -> dict[str, Any]:
        spec = original(path)
        # A single-case fixture IS the case: it carries `expected` at top level
        # with no `cases` key, which is the shape that reaches the runner's
        # single-case dispatch path.
        target = spec["cases"][0] if "cases" in spec else spec
        target.setdefault("expected", {})[key] = True
        return spec

    monkeypatch.setattr(langfuse_runner, "_load", _loader)


# One activated fixture per DISPATCH PATH through test_langfuse_fixture. Covering
# only the multi-case loop is how this guard shipped wired into one path of three:
# the tests exercised the path that already worked.
_PATHS = {
    "multi_case_loop": "023-langfuse-generation-rendering",
    "single_case": "022-langfuse-basic-trace",
    "hand_built_134": "134-langfuse-nested-fan-out-parent-resolution",
}


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("key", sorted(langfuse_runner._UNIMPLEMENTED_OBSERVABILITY_ASSERTIONS))
async def test_an_activated_case_reaching_an_unimplemented_assertion_fails(
    monkeypatch: pytest.MonkeyPatch, key: str, path: str
) -> None:
    # Every key against every path: a guard covering four of five keys, or two of
    # three paths, reads identically to one that covers all of them.
    _inject(monkeypatch, key)
    with pytest.raises(AssertionError, match="does not implement"):
        await langfuse_runner.test_langfuse_fixture(langfuse_runner.CONFORMANCE_DIR / f"{_PATHS[path]}.yaml")


@pytest.mark.parametrize("path", sorted(_PATHS))
async def test_each_dispatch_path_passes_when_nothing_is_injected(path: str) -> None:
    # Non-vacuity for the matrix above, per path: each fixture passes untouched,
    # so the failures there come from the injected key rather than from the
    # fixture or the dispatch path itself.
    await langfuse_runner.test_langfuse_fixture(langfuse_runner.CONFORMANCE_DIR / f"{_PATHS[path]}.yaml")


def test_the_named_fixtures_still_take_the_paths_they_are_meant_to() -> None:
    # The matrix is only three-path coverage while these fixtures keep their
    # shapes. A spec edit that gives 022 a `cases:` block would silently collapse
    # two of the three paths onto one, with everything still green.
    single = langfuse_runner._load(langfuse_runner.CONFORMANCE_DIR / f"{_PATHS['single_case']}.yaml")
    multi = langfuse_runner._load(langfuse_runner.CONFORMANCE_DIR / f"{_PATHS['multi_case_loop']}.yaml")
    assert "cases" not in single, f"{_PATHS['single_case']} gained a `cases:` block; pick another"
    assert "cases" in multi, f"{_PATHS['multi_case_loop']} lost its `cases:` block; pick another"
    assert _PATHS["hand_built_134"] == langfuse_runner._FIXTURE_134
