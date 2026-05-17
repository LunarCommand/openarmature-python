"""Phase 0 exit criterion: every fixture in the spec submodule parses into a
typed harness config, AND the parse is round-trip stable (parse → dump →
parse produces an equal model).

Round-trip stability catches the bug class where a directive lands in the
spec but our pydantic model silently drops it via ``extra="forbid"`` not
being applied (or, conversely, where a field is mistakenly typed loose
enough to accept a dict that doesn't actually round-trip cleanly).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .harness import discover_fixtures, load_fixture


def _id(case: tuple[str, Path]) -> str:
    capability, path = case
    return f"{capability}/{path.stem}"


_FIXTURES = list(discover_fixtures())


# Fixtures whose typed-harness directives land in a later PR of the
# 5-proposal batch. The fixture parsers / round-trippers need the new
# directive shapes (state_migration, parallel_branches, NodeEvent
# branch_name) to succeed; those shapes ship with their respective PRs.
# Keyed by the test ID format ``<capability>/<stem>``.
_DEFERRED_FIXTURES: dict[str, str] = {
    # proposal 0011's parallel-branches fixtures (032-038 +
    # graph-engine/021) were removed from this list as part of
    # PR-5; the typed harness parses the parallel_branches:
    # node shape via the new ParallelBranchesSpec directive model.
    # proposal 0014's state-migration fixtures (039-046) were removed
    # from this list as part of PR-4; the CasesFixture model already
    # parses the seeded_record / migrations shape via its permissive
    # extras (CaseSpec uses ``model_config = ConfigDict(extra="allow")``).
    # proposal 0015's llm-provider fixtures (009-020) were removed
    # from this list as part of PR-2; the typed harness parses the
    # content-block message shape via LlmCallSpec's permissive
    # ``messages: list[dict[str, Any]]`` typing without needing
    # model extensions.
}


def test_inventory_is_non_empty() -> None:
    """Sanity guard. The spec submodule should expose 68+ fixtures across
    the four capabilities. If discover returns zero, the submodule pin is
    wrong or the directory layout changed."""
    assert len(_FIXTURES) > 0, "no conformance fixtures discovered"


@pytest.mark.parametrize("case", _FIXTURES, ids=_id)
def test_fixture_parses(case: tuple[str, Path]) -> None:
    """Every fixture parses into one of the three typed variants. The
    discriminator routes to ``LlmProviderFixture``, ``CasesFixture``, or
    ``GraphFixture`` based on top-level keys; ``extra="forbid"`` rejects
    any unknown top-level field."""
    case_id = _id(case)
    if case_id in _DEFERRED_FIXTURES:
        pytest.skip(f"{case_id}: {_DEFERRED_FIXTURES[case_id]}")
    _, path = case
    load_fixture(path)


@pytest.mark.parametrize("case", _FIXTURES, ids=_id)
def test_fixture_round_trips(case: tuple[str, Path]) -> None:
    """Parse → ``model_dump`` → re-parse → equal. Exit criterion for
    Phase 0 per the implementation plan: catches dropped fields the user
    intended to use later."""
    case_id = _id(case)
    if case_id in _DEFERRED_FIXTURES:
        pytest.skip(f"{case_id}: {_DEFERRED_FIXTURES[case_id]}")
    _, path = case
    parsed = load_fixture(path)
    dumped = parsed.model_dump(exclude_none=True)
    # Re-parse via the same loader path so the discriminator runs again.
    from .harness.loader import _FIXTURE_ADAPTER

    reparsed = _FIXTURE_ADAPTER.validate_python(dumped)
    assert parsed == reparsed, f"round-trip mismatch for {path}"
