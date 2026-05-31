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
_PROMPT_0046_REASON = "Proposal 0046 not yet implemented (PR 12)"
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
    # Proposal 0034 / 0040 fan-out / parallel-branches caller-metadata
    # fixtures. The augmentation MECHANISM is implemented in v0.11.0
    # (#22) and covered end-to-end by unit tests
    # (test_observability_otel.py + test_observability_langfuse.py).
    # The CONFORMANCE FIXTURES stay deferred for harness-shape gaps:
    # - 029 omits ``collect_field`` / ``target_field`` on the fan-out
    #   cfg AND a ``state:`` block on the inner subgraph (both
    #   required by the cross-cap adapter).
    # - 030 expects per-branch dispatch spans in the Langfuse tree;
    #   the spec direction for that span layer is pending in coord
    #   thread ``discuss-otel-parallel-branches-dispatch-span``.
    "observability/029-caller-metadata-fan-out-per-instance": (
        "Fixture-shape gaps (no collect_field/target_field/state); mechanism covered by unit tests"
    ),
    "observability/030-caller-metadata-parallel-branches-per-branch": (
        "Per-branch dispatch span shape pending spec; mechanism covered by unit tests"
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
    # Proposal 0038 (Google Gemini wire-format mapping) shipped in spec
    # v0.32.0 but python marks it not-yet in conformance.toml — the
    # Gemini provider isn't implemented in this release. Defer the
    # cross-capability parse tests for the 044-053 fixtures; the
    # `mapping: gemini` discriminator is harness-extension territory.
    "llm-provider/044-gemini-basic-message-round-trip": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/045-gemini-function-call-flow": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/046-gemini-image-content-blocks": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/047-gemini-tool-choice-modes": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/048-gemini-runtime-config-mapping": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/049-gemini-error-mapping": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/050-gemini-structured-output-native": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/051-gemini-structured-output-fallback": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/052-gemini-thought-signature-round-trip": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    "llm-provider/053-cross-provider-signature-strip": (
        "Gemini provider not implemented (0038 not-yet in conformance.toml)"
    ),
    # Proposal 0040 (open-span metadata update) — task #22 implements
    # the §6 augmentation-event mechanism + un-defers 029/030 + 034.
    # Fixture 034 lands in the Langfuse-specific harness directly
    # via the augment_metadata directive (see
    # ``tests/conformance/test_observability_langfuse.py``); the
    # cross-capability parser still doesn't model langfuse_trace, so
    # defer the parser-side activation per the 022-024 pattern.
    "observability/034-caller-metadata-open-span-update-serial": (
        "Langfuse shape models live in the dedicated test_observability_langfuse harness"
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
    # Proposal 0043 (trace.input/output) — fixture 037 uses langfuse_trace
    # expected shape + hook-based directives the cross-capability parser
    # doesn't model. Behavior pinned by unit tests at
    # tests/unit/test_observability_langfuse.py::test_trace_input_output_*.
    "observability/037-langfuse-trace-input-output": (
        "Cross-capability parser doesn't model langfuse_trace; behavior pinned by unit tests"
    ),
    # Proposal 0044 (parallel-branches OTel dispatch span) — fixture
    # 038 uses ``expected_otel_spans`` shape the cross-cap parser
    # doesn't model.  Behavior is exercised end-to-end by the langfuse
    # observability harness via fixture 030 (Langfuse-side per-branch
    # spans) once that fixture activates.
    # Proposal 0046 (chat-prompt rendering) shorthand for the 15-entry
    # sweep below.  Centralizing keeps the per-fixture lines below the
    # 110-char ruff E501 budget.
    "observability/038-otel-parallel-branches-dispatch-span": (
        "Cross-capability parser doesn't model expected_otel_spans; OTel-side behavior pinned by unit tests"
    ),
    # Proposal 0045 (nested-lineage augmentation) — fixture 039 uses
    # graph-topology + augmentation directives the cross-cap parser
    # doesn't fully model.  PR 11 activates the langfuse-side coverage
    # directly via the dedicated harness.
    "observability/039-nested-lineage-augmentation": (
        "Cross-capability parser doesn't model nested-lineage directives; PR 11 lands the activation"
    ),
    # Proposal 0046 (chat-prompt rendering, v0.38.0) — chat prompt
    # fixtures 017-031 use new prompt-management YAML shapes the
    # cross-cap parser hasn't modeled.  Activation lands in PR 12.
    "prompt-management/017-chat-prompt-per-segment-render": _PROMPT_0046_REASON,
    "prompt-management/018-chat-prompt-placeholder-injection": _PROMPT_0046_REASON,
    "prompt-management/019-chat-prompt-placeholder-empty-list": _PROMPT_0046_REASON,
    "prompt-management/020-chat-prompt-per-segment-strict-undefined": _PROMPT_0046_REASON,
    "prompt-management/021-chat-prompt-empty-segment": _PROMPT_0046_REASON,
    "prompt-management/022-chat-prompt-unfilled-placeholder": _PROMPT_0046_REASON,
    "prompt-management/023-chat-prompt-content-blocks-text-image-url": _PROMPT_0046_REASON,
    "prompt-management/024-chat-prompt-content-blocks-inline-image": _PROMPT_0046_REASON,
    "prompt-management/025-chat-prompt-role-block-compatibility": _PROMPT_0046_REASON,
    "prompt-management/026-chat-prompt-observability-entities": _PROMPT_0046_REASON,
    "prompt-management/027-chat-prompt-empty-rendered-messages": _PROMPT_0046_REASON,
    "prompt-management/028-chat-prompt-duplicate-placeholder-names": _PROMPT_0046_REASON,
    "prompt-management/029-chat-prompt-content-blocks-empty-cases": _PROMPT_0046_REASON,
    "prompt-management/030-chat-prompt-placeholder-name-validation": _PROMPT_0046_REASON,
    "prompt-management/031-text-prompt-placeholders-ignored": _PROMPT_0046_REASON,
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
