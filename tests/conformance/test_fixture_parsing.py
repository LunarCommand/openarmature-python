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
    # proposal 0031's Langfuse fixtures (022-024) introduce a
    # ``langfuse_trace:`` expected-block shape unknown to the current
    # typed directives. The Langfuse harness lands in the next PR of
    # the batch (PR 3) and adds the matching directive model;
    # deferring here keeps the parser tests passing in the meantime.
    # Langfuse fixtures (022-024) use a directive shape the cross-
    # capability parser doesn't model (`langfuse_observer:`,
    # `expected.langfuse_trace`, `prompt_backend:`). The capability-
    # specific harness at tests/conformance/test_observability_langfuse.py
    # parses these directly via yaml + tailored helpers.
    "observability/022-langfuse-basic-trace": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
    ),
    "observability/023-langfuse-generation-rendering": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
    ),
    "observability/024-langfuse-prompt-linkage": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
    ),
    # Proposal 0035 (spec v0.26.1) Langfuse graph-topology fixtures
    # (031/032/033) introduce a ``langfuse_traces:`` (plural) expected
    # shape for detached / multi-trace cases. Same deferral rationale as
    # 022-024 — Langfuse shape models live in the dedicated harness.
    "observability/031-langfuse-subgraph-span-hierarchy": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
    ),
    "observability/032-langfuse-fan-out-per-instance-spans": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
    ),
    "observability/033-langfuse-detached-trace-mode": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
    ),
    # Proposal 0034 Langfuse caller-metadata fixture uses the
    # ``langfuse_observer`` / ``langfuse_trace`` directive shapes the
    # cross-capability parser doesn't model. The capability-specific
    # harness at tests/conformance/test_observability_langfuse.py
    # parses these directly via yaml + tailored helpers.
    "observability/027-langfuse-caller-supplied-metadata": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
    ),
    # Proposal 0034 boundary-rejection fixture uses rejection
    # invariants (``invoke_rejects_at_api_boundary``, ``no_spans_emitted``,
    # ``no_langfuse_observations_emitted``) the cross-capability
    # parser doesn't model. Driven by the dedicated runner
    # ``_run_fixture_028`` in test_observability.py.
    "observability/028-caller-metadata-namespace-rejection": (
        "Rejection invariants live in the dedicated _run_fixture_028 runner"
    ),
    # Proposal 0034 fan-out / parallel-branches caller-metadata
    # fixtures need the harness primitive
    # ``augment_metadata_from_field`` (per-instance / per-branch
    # ``set_invocation_metadata`` calls). The 026/027/028 fixtures
    # (cross-cutting + boundary rejection) shipped with PR 4; the
    # augmentation primitive lands in a follow-up.
    "observability/029-caller-metadata-fan-out-per-instance": (
        "Per-instance augmentation harness primitive lands in a follow-up"
    ),
    "observability/030-caller-metadata-parallel-branches-per-branch": (
        "Per-branch augmentation harness primitive lands in a follow-up"
    ),
    # proposal 0033 added typed directive shapes (`secondary_manager`,
    # `label_resolver`, `cases`) the canonical parser doesn't model.
    # The capability-specific harness at
    # tests/conformance/harness/prompt_management.py models the new
    # shapes; defer the cross-capability parser until that catches up.
    "prompt-management/015-label-resolver-fallback-chain": (
        "Label-resolver shape models live in the PM-specific capability harness"
    ),
    "prompt-management/016-prompt-observability-entities-propagation": (
        "Cases shape models live in the PM-specific capability harness"
    ),
    # Proposal 0037 (Anthropic Messages mapping) shipped in spec v0.28.0
    # but python marks it not-yet in conformance.toml — the Anthropic
    # provider isn't implemented in this release. Defer the
    # cross-capability parse tests for the 033-042 fixtures until that
    # lands; the openai-strips-thinking-blocks side (043) is in
    # test_llm_provider.py's own deferral.
    "llm-provider/033-anthropic-basic-message-round-trip": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/034-anthropic-tool-call-flow": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/035-anthropic-image-content-blocks": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/036-anthropic-tool-choice-modes": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/037-anthropic-runtime-config-mapping": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/038-anthropic-max-tokens-required": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/039-anthropic-error-mapping": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/040-anthropic-structured-output-native": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/041-anthropic-structured-output-fallback": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    "llm-provider/042-anthropic-thinking-block-round-trip": (
        "Anthropic provider not implemented (0037 not-yet in conformance.toml)"
    ),
    # Proposal 0040 (open-span metadata update) — task #22 implements
    # the §6 augmentation-event mechanism + un-defers 029/030 + 034.
    "observability/034-caller-metadata-open-span-update-serial": (
        "Open-span augmentation-event mechanism lands with #22 (0040 not-yet)"
    ),
    # Proposal 0039 (caller-supplied invocation_id) Langfuse trace.id
    # derivation fixtures use the langfuse_trace expected shape the
    # cross-capability parser doesn't model. The derivation itself is
    # pinned by unit tests in test_observability_langfuse_adapter.py
    # against the same spec vector fixture 036 uses
    # (sha256("run_abc123")[:16].hex == 29b50a6c08dabfeaeb1696301f4fabe1);
    # wiring into the langfuse-specific conformance harness is a
    # follow-up.
    "observability/035-caller-invocation-id-uuid": (
        "Cross-capability parser doesn't model langfuse_trace; derivation pinned by unit tests"
    ),
    "observability/036-caller-invocation-id-non-uuid": (
        "Cross-capability parser doesn't model langfuse_trace; derivation pinned by unit tests"
    ),
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
