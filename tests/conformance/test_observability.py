"""Run spec observability conformance fixtures (001-011) against OTelObserver.

Driven fixtures:

- **001-basic-trace** — full span shape.
- **002-subgraph-hierarchy** — synthetic dispatch span +
  inner-node parenting.
- **003-error-status** — ERROR status mapping for the
  ``node_exception`` case.
- **005-llm-provider-span-nested** — LLM span +
  ``disable_llm_spans`` opt-out + TracerProvider isolation.
- **007-retry-attempt-spans** — sibling attempt spans with
  per-attempt ``attempt_index`` under retry middleware.
- **008-detached-trace-mode** — detached subgraph
  + detached fan-out + cross-trace ``correlation_id``.
- **009-correlation-id-cross-cutting** — every span
  carries ``openarmature.correlation_id``; back-to-back
  invocations get distinct UUIDv4s.
- **010-log-correlation** — log records emitted from
  inside node bodies pick up the active node span's
  ``trace_id``/``span_id`` via the engine-side
  ``prepare_sync`` → OTel context attach pipeline; both nested
  and detached-trace cases.
- **011-determinism** — deterministic span content
  (hierarchy, names, status, attributes minus the canonical
  non-deterministic-by-design list) is identical across runs.

Per-fixture wiring notes live in
``docs/phase-6-1-conformance-fillin.md``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import yaml

# Skip the entire module if the ``otel`` extras aren't installed —
# importing ``openarmature.observability.otel`` raises ImportError at
# import time when the extras are missing, which would fail
# collection rather than skipping cleanly. Mirrors the pattern in
# ``tests/unit/test_observability_otel.py``.
pytest.importorskip("opentelemetry.sdk.trace")

from openarmature.observability.otel import OTelObserver  # noqa: E402

from .adapter import build_graph  # noqa: E402

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from openarmature.llm.response import RuntimeConfig


# OTel SDK 1.x makes ``set_tracer_provider`` one-shot: once a non-default
# provider is set, subsequent calls are no-ops (the SDK logs a warning
# and returns). The set is guarded by a ``Once`` primitive at
# ``opentelemetry.trace._TRACER_PROVIDER_SET_ONCE``, not just by the
# value of ``_TRACER_PROVIDER``. Restoring via the public API silently
# fails after a prior set, leaking the test's global provider into
# subsequent tests that also touch the OTel global. This helper resets
# BOTH the value and the Once via the SDK's private API so a sibling
# test running after this one starts from a clean global state.
def _reset_otel_global_tracer_provider(restore_to: object) -> None:
    from opentelemetry import trace as otel_trace

    once = otel_trace._TRACER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
    with once._lock:  # pyright: ignore[reportPrivateUsage]
        if isinstance(restore_to, otel_trace.ProxyTracerProvider):
            otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
            once._done = False  # pyright: ignore[reportPrivateUsage]
        else:
            otel_trace._TRACER_PROVIDER = restore_to  # type: ignore[attr-defined]
            once._done = True  # pyright: ignore[reportPrivateUsage]


CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "observability" / "conformance"
)


_SUPPORTED_FIXTURES = frozenset(
    {
        # v0.36.0 — proposal 0044 (parallel-branches OTel dispatch
        # span). Asserts the per-branch dispatch span synthesis +
        # §5.7 attribute surface end-to-end against a two-branch
        # parallel-branches graph with calls_llm in each branch.
        "038-otel-parallel-branches-dispatch-span",
        # v0.42.0 — proposal 0050 call-level-retry per-attempt LLM span
        # surface. Single-attempt default: one span, attempt_index 0.
        "057-llm-attempt-index-single-attempt-default",
        "001-otel-basic-trace",
        "002-otel-subgraph-hierarchy",
        "003-otel-error-status",
        "004-otel-routing-error-attribution",
        "005-otel-llm-provider-span-nested",
        "006-otel-fan-out-instance-attribution",
        "007-otel-retry-attempt-spans",
        "008-otel-detached-trace-mode",
        "009-otel-correlation-id-cross-cutting",
        "010-otel-log-correlation",
        "011-otel-determinism",
        # v0.17.0 — proposal 0024 (friction-roundup #1, #2, #6).
        "012-otel-llm-payload-default-off",
        "013-otel-llm-payload-enabled",
        "014-otel-llm-payload-truncation",
        "015-otel-llm-payload-image-redaction",
        "016-otel-llm-request-params",
        "017-otel-llm-request-params-partial",
        "018-otel-llm-request-extras",
        "019-otel-llm-genai-semconv",
        "020-otel-llm-genai-system-override",
        "021-otel-llm-disable-genai-semconv",
        # v0.24.0 — proposal 0032 (three new declared RuntimeConfig
        # fields surfaced as gen_ai.request.* attributes).
        "025-otel-llm-request-params-extended",
        # v0.10.0 — proposal 0034 (caller-supplied invocation metadata
        # cross-cutting on every span). 026 verifies the
        # ``openarmature.user.*`` attribute family lands on the
        # invocation span, every node span, and the LLM provider span.
        "026-otel-caller-supplied-metadata",
        # 028 — proposal 0034 API-boundary rejection: caller-supplied
        # metadata keys under reserved namespaces (openarmature.*,
        # gen_ai.*) MUST raise at the ``invoke()`` boundary before
        # any work begins. Two cases (one per reserved prefix).
        "028-caller-metadata-namespace-rejection",
        # v0.41.0 — proposal 0047 (§5.5.3.1 OA-namespace cache
        # attributes). Three fixtures cover cache-hit emission (040),
        # absence (041 — no prompt_tokens_details on the wire), and
        # reported-zero (042 — distinct from absent).
        "040-llm-cache-attribute-emission",
        "041-llm-cache-attribute-absence",
        "042-llm-cache-attribute-reported-zero",
        # v0.41.0 — proposal 0049 (typed LlmCompletionEvent variant on
        # the observer event union). Seven fixtures exercise dispatch
        # shape (050), type discrimination (051), opt-in caller
        # metadata (052), success-only scope on failure paths (053),
        # fan_out_index / branch_name population (054 / 055), and
        # strict-serial arrival ordering (056).
        "050-llm-completion-event-dispatch",
        "051-llm-completion-event-type-discrimination",
        "052-llm-completion-event-caller-metadata-opt-in",
        "053-llm-completion-event-no-event-on-failure",
        "054-llm-completion-event-fan-out-index-population",
        "055-llm-completion-event-branch-name-population",
        "056-llm-completion-event-strict-serial-ordering",
        # proposal 0057 LlmCompletionEvent field population (060-068) +
        # proposal 0058 LlmFailedEvent (069-073). Driven through the typed-
        # event-collector runner (the same machinery as 050-056) plus a
        # multi-node-chain variant for 067/071. Fixture-harness catch-up
        # tier 1. Four of the family stay unit-tested for now (see
        # _UNIT_TESTED_FIXTURES): 066 (corrected >=2-member group ships at spec
        # v0.74.1, picked up at the v0.16.0 pin), 069 (asserts a request model
        # the fixture doesn't declare), 070 (missing-tool_call_id message is
        # non-constructible in python -- enforced at construction, not the
        # complete() boundary), 073 (fixture asserts the vendor body error.type
        # verbatim, but python's error_type is the OA exception class name).
        "060-llm-completion-event-input-messages-populated",
        "061-llm-completion-event-output-content-populated",
        "062-llm-completion-event-request-params-populated",
        "063-llm-completion-event-request-extras-populated",
        "064-llm-completion-event-active-prompt-populated",
        "065-llm-completion-event-active-prompt-null",
        "067-llm-completion-event-call-id-always-present-and-distinct",
        "068-llm-completion-event-response-model-distinct-from-request",
        "071-llm-failure-event-call-id-distinct-from-completion-event",
        "072-llm-failure-event-mutual-exclusion-with-completion-event",
        # Fixture-harness catch-up tier 2a: trace-shape Langfuse fixtures
        # driven through a LangfuseObserver + InMemoryLangfuseClient recorder.
        # 022/031/032 assert the Trace + observation tree (proposal 0031/0035/
        # 0061); 035/036 the caller-invocation-id -> trace.id derivation
        # (proposal 0039); 059 the implementation-attribution trace metadata
        # (proposal 0052). 023/024 (Langfuse Generation) are tier 2b.
        "022-langfuse-basic-trace",
        "031-langfuse-subgraph-span-hierarchy",
        "032-langfuse-fan-out-per-instance-spans",
        "035-caller-invocation-id-uuid",
        "036-caller-invocation-id-non-uuid",
        "059-implementation-attribution-langfuse",
        # proposal 0052 attribution fixture (case 1) + proposal 0061
        # (case 2: the §5.1 attribution lands on the detached trace's own
        # openarmature.invocation span). Wired together now that 0061
        # (v0.61.0) resolves the detached-invocation-span shape case 2
        # presupposed — the whole fixture was unwired pending that.
        "058-implementation-attribution-otel",
        # v0.62.0 — proposal 0064 (Langfuse trace.sessionId / trace.userId
        # population). Cases 2/3/4 (not session-bound + userId promotion)
        # run; session-bound cases 1/5 defer until the sessions capability
        # (0020) supplies openarmature.session_id.
        "084-langfuse-session-user-promotion",
        # v0.67.0 — proposal 0076 (tool-call request observability on the
        # LLM span). The model's output tool calls surface as ungated
        # identity (count / names / ids) plus a gated full serialization
        # on openarmature.llm.complete. Driven through the generic
        # LLM-payload fixture runner.
        "085-llm-tool-call-request-attributes",
        "086-llm-tool-call-request-absent",
        "087-llm-tool-call-request-survives-payload-gating",
        # v0.68.0 — proposal 0067 (GenAI metrics, observability §11). The
        # LLM-path fixtures: token + duration histograms (088), error.type
        # on duration (090), and the enable_metrics-off no-op (091).
        # Captured via a private MeterProvider + InMemoryMetricReader (the
        # §6.9 metric-capture primitive). 089 (embeddings) is deferred.
        "088-llm-metrics-token-and-duration",
        "090-metrics-error-type-on-duration",
        "091-metrics-disabled-no-measurements",
        # v0.69.0 — proposal 0063 (tool-execution observability). A
        # calls_tool node enters the with_tool_call scope; the typed
        # ToolCallEvent / ToolCallFailedEvent drive the OTel tool span +
        # the Langfuse Tool observation.
        "092-tool-call-event-dispatch",
        "093-tool-call-failed-event-dispatch",
        "094-tool-call-event-mutual-exclusion",
        "095-tool-call-id-links-to-llm-request",
        "096-tool-call-payload-gating",
        "097-otel-tool-span-attributes",
        "098-langfuse-tool-observation",
        # v0.70.1 — proposal 0075 callable-branch span shape (observability
        # §5.7). The ORIGINAL fixture 110 (span shape + skip-emits-no-span);
        # the branch_count assertion arrives with the v0.73.1 pin (v0.16.0).
        "110-otel-callable-branch-span",
    }
)


_EMBEDDING_DEFER = (
    "embedding capability (proposal 0059) unimplemented until v0.16.0; "
    "no embedding event/provider to record from"
)

_RERANK_DEFER = (
    "rerank capability (proposal 0060) unimplemented until v0.16.0; no rerank event/provider to record from"
)


# Pinned observability fixtures NOT run by this YAML harness, each with an
# explicit reason. The coverage guard (test_observability_fixture_coverage_
# is_complete) fails on any pinned fixture absent from _SUPPORTED_FIXTURES +
# the three sets below, so a future unwired spec fixture cannot silently
# pytest.skip past CI.
#
# _DEFERRED_FIXTURES — not run because the capability is unimplemented.
_DEFERRED_FIXTURES: dict[str, str] = {
    # Proposal 0045 IS implemented (v0.11.0), but the nested-case Langfuse
    # fixture stays deferred: it needs runtime-state item-list lookup for
    # nested fan-outs plus an augment_metadata_from_outer_item directive
    # the harness doesn't model yet.
    "039-nested-lineage-augmentation": (
        "nested-case Langfuse harness wiring not yet implemented (proposal 0045 nested fan-out)"
    ),
    # Embedding observability (proposals 0059 / 0067 §11). The embedding
    # capability is unshipped until v0.16.0; the LLM-path equivalents run.
    **{
        fixture_id: _EMBEDDING_DEFER
        for fixture_id in (
            "074-embedding-event-dispatch",
            "075-embedding-failure-event-dispatch-on-provider-unavailable",
            "076-embedding-event-mutual-exclusion",
            "077-embedding-event-call-id-distinct",
            "078-embedding-event-input-strings-populated",
            "079-embedding-event-request-params-populated",
            "080-embedding-event-input-count-and-dimensions-populated",
            "081-embedding-event-active-prompt-populated",
            "082-otel-embedding-span-attributes",
            "083-langfuse-embedding-observation",
            "089-embedding-metrics-token-and-duration",
        )
    },
    # Rerank observability (proposal 0060, v0.70.0). The rerank protocol is
    # unshipped in python until v0.16.0; no rerank provider/event exists.
    **{
        fixture_id: _RERANK_DEFER
        for fixture_id in (
            "099-rerank-event-dispatch",
            "100-rerank-failure-event-dispatch-on-provider-unavailable",
            "101-rerank-event-mutual-exclusion",
            "102-rerank-event-call-id-distinct",
            "103-rerank-event-query-and-documents-populated",
            "104-rerank-event-request-params-populated",
            "105-rerank-event-top-k-and-result-count-populated",
            "106-rerank-event-active-prompt-populated",
            "107-otel-rerank-span-attributes",
            "108-langfuse-rerank-observation",
            "109-rerank-metrics-token-and-duration",
        )
    },
}


# _UNIT_TESTED_FIXTURES — implemented behavior covered by the dedicated unit
# suite rather than wired into this YAML harness. Value names the proposal +
# the covering file.
_UNIT_TESTED_FIXTURES: dict[str, str] = {
    fixture_id: reason
    for fixture_ids, reason in (
        # Fixture-harness catch-up tier 2a wired the trace-shape Langfuse
        # fixtures (022/031/032), the invocation-id fixtures (035/036), and the
        # attribution fixture (059). 023/024 (Langfuse Generation) are tier 2b;
        # 033 (detached multi-trace) is tier 4.
        (
            ("023-langfuse-generation-rendering", "024-langfuse-prompt-linkage"),
            "proposal 0031 Langfuse generation/prompt-linkage; covered by test_observability_langfuse.py",
        ),
        (
            ("033-langfuse-detached-trace-mode",),
            "proposal 0035/0061 Langfuse detached-trace mode; covered by test_observability_langfuse.py",
        ),
        (
            (
                "027-langfuse-caller-supplied-metadata",
                "029-caller-metadata-fan-out-per-instance",
                "034-caller-metadata-open-span-update-serial",
            ),
            "proposal 0034/0040 caller metadata; covered by test_observability_langfuse.py",
        ),
        (
            ("030-caller-metadata-parallel-branches-per-branch",),
            "proposal 0040 per-branch caller metadata; covered by test_observability_otel.py",
        ),
        (
            ("037-langfuse-trace-input-output",),
            "proposal 0043 trace input/output; covered by test_observability_langfuse.py",
        ),
        (
            (
                "043-get-invocation-metadata-roundtrip",
                "044-get-invocation-metadata-fan-out-scoping",
                "045-get-invocation-metadata-retry-scoping",
                "046-get-invocation-metadata-outside-invocation",
            ),
            "proposal 0048 get_invocation_metadata; covered by test_observability_metadata.py",
        ),
        # Fixture-harness catch-up tier 1 wired the rest of the 0057/0058
        # family into _SUPPORTED_FIXTURES; these three stay here, each blocked
        # on a spec-side fixture change that python picks up at the v0.16.0 pin
        # bump.
        (
            ("066-llm-completion-event-active-prompt-group-populated",),
            # At the current v0.70.1 pin the fixture's group has a single
            # member, which python's PromptGroup (prompt-management §10,
            # >=2 members) correctly rejects. The corrected >=2-member fixture
            # ships at spec v0.74.1; wire it with the v0.16.0 pin bump.
            "proposal 0057 active_prompt_group; corrected >=2-member fixture "
            "ships at spec v0.74.1, wired with the v0.16.0 pin bump; covered by "
            "test_llm_provider.py",
        ),
        (
            ("069-llm-failure-event-dispatch-on-provider-unavailable",),
            # Asserts model "gpt-test" on the failed event but declares no
            # request-side model the harness can bind (no calls_llm.model, and
            # the 503 body carries no model). Needs a spec fixture fix to
            # declare the request model, cf. 068.
            "proposal 0058 LlmFailedEvent; fixture asserts a request model it "
            "doesn't declare; covered by test_llm_provider.py",
        ),
        (
            ("070-llm-failure-event-dispatch-on-provider-invalid-request",),
            # The fixture's malformed message (tool role, no tool_call_id) is
            # non-constructible in python: ToolMessage.tool_call_id is a required
            # field, so the "MUST be present" rule (llm-provider §3) is enforced
            # at construction, not the complete() boundary. python drives
            # provider_invalid_request via the unmatched-tool_call_id shape.
            "proposal 0058 provider_invalid_request; fixture's missing-"
            "tool_call_id message is non-constructible in python; covered by "
            "test_llm_provider.py",
        ),
        (
            ("073-llm-failure-event-error-type-vendor-specific",),
            # The fixture asserts the vendor body ``error.type`` verbatim per
            # case (rate_limit_exceeded / RateLimitError) plus a null case.
            # python deliberately sources error_type from the OA exception
            # class name (e.g. "ProviderRateLimit") -- a spec-permitted
            # "exception class name" style, but it never echoes the body type
            # nor emits null. The behavior is contract-conformant; the fixture
            # over-constrains beyond the permissive field contract.
            "proposal 0058 LlmFailedEvent.error_type; python uses the exception "
            "class name, the fixture asserts the vendor body error.type; covered "
            "by test_llm_provider.py",
        ),
    )
    for fixture_id in fixture_ids
}


# _CONVENTION_ONLY_FIXTURES — proposal 0048 §9 queryable-observer pattern is
# convention-only (no new abstract surface on Observer), satisfied via
# docs/concepts/observability.md, so there is no library API to assert.
_CONVENTION_ONLY_FIXTURES: dict[str, str] = {
    fixture_id: (
        "proposal 0048 §9 queryable-observer pattern is convention-only "
        "(no library surface); satisfied by docs/concepts/observability.md"
    )
    for fixture_id in (
        "047-queryable-observer-pattern",
        "048-queryable-observer-async-safety",
        "049-queryable-observer-lifecycle-drop",
    )
}


# UUIDv4 canonical form: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (where y in {8,9,a,b}).
_UUIDV4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _fixture_paths() -> list[Path]:
    return sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def test_observability_fixture_coverage_is_complete() -> None:
    # Fail-on-unknown guard. Every pinned observability conformance fixture
    # MUST be either run (_SUPPORTED_FIXTURES) or explicitly accounted for:
    # _DEFERRED_FIXTURES (future capability), _UNIT_TESTED_FIXTURES (covered
    # by the unit suite, not this YAML harness), or _CONVENTION_ONLY_FIXTURES
    # (doc-satisfied, no library surface). A new spec fixture that is none of
    # these fails HERE rather than silently pytest.skip-ping past CI.
    all_ids = {p.stem for p in _fixture_paths()}
    accounted = (
        set(_SUPPORTED_FIXTURES)
        | _DEFERRED_FIXTURES.keys()
        | _UNIT_TESTED_FIXTURES.keys()
        | _CONVENTION_ONLY_FIXTURES.keys()
    )
    unaccounted = sorted(all_ids - accounted)
    assert not unaccounted, (
        "unaccounted observability conformance fixtures: wire each into "
        "_SUPPORTED_FIXTURES once it runs, or document it in _DEFERRED_FIXTURES "
        "(future capability) / _UNIT_TESTED_FIXTURES (covered by the unit suite) "
        f"/ _CONVENTION_ONLY_FIXTURES (doc-satisfied): {unaccounted}"
    )
    # An accounting entry whose fixture no longer exists on disk (renamed at
    # a pin bump) should be removed.
    stale = sorted(accounted - all_ids)
    assert not stale, f"accounting entries with no fixture file (remove): {stale}"
    # A fixture cannot be both run and documented-as-not-run.
    not_run = _DEFERRED_FIXTURES.keys() | _UNIT_TESTED_FIXTURES.keys() | _CONVENTION_ONLY_FIXTURES.keys()
    overlap = sorted(set(_SUPPORTED_FIXTURES) & not_run)
    assert not overlap, f"fixtures both run and documented-as-not-run (pick one): {overlap}"


# ---------------------------------------------------------------------------
# Per-fixture dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_observability_fixture(fixture_path: Path) -> None:
    fixture_id = fixture_path.stem
    skip_reason = (
        _DEFERRED_FIXTURES.get(fixture_id)
        or _UNIT_TESTED_FIXTURES.get(fixture_id)
        or _CONVENTION_ONLY_FIXTURES.get(fixture_id)
    )
    if skip_reason is not None:
        pytest.skip(f"{fixture_id}: {skip_reason}")
    if fixture_id not in _SUPPORTED_FIXTURES:
        # Unaccounted: neither wired nor documented. The coverage guard
        # (test_observability_fixture_coverage_is_complete) fails loudly
        # listing every such fixture; the individual case skips here.
        pytest.skip(f"{fixture_id}: unaccounted -- see the coverage guard")

    spec = _load(fixture_path)
    if fixture_id == "001-otel-basic-trace":
        await _run_fixture_001(spec)
    elif fixture_id == "002-otel-subgraph-hierarchy":
        await _run_fixture_002(spec)
    elif fixture_id == "003-otel-error-status":
        await _run_fixture_003(spec)
    elif fixture_id == "004-otel-routing-error-attribution":
        await _run_fixture_004(spec)
    elif fixture_id == "005-otel-llm-provider-span-nested":
        await _run_fixture_005(spec)
    elif fixture_id == "006-otel-fan-out-instance-attribution":
        await _run_fixture_006(spec)
    elif fixture_id == "007-otel-retry-attempt-spans":
        await _run_fixture_007(spec)
    elif fixture_id == "008-otel-detached-trace-mode":
        await _run_fixture_008(spec)
    elif fixture_id == "009-otel-correlation-id-cross-cutting":
        await _run_fixture_009(spec)
    elif fixture_id == "010-otel-log-correlation":
        await _run_fixture_010(spec)
    elif fixture_id == "011-otel-determinism":
        await _run_fixture_011(spec)
    elif fixture_id == "028-caller-metadata-namespace-rejection":
        await _run_fixture_028(spec)
    elif fixture_id == "038-otel-parallel-branches-dispatch-span":
        await _run_fixture_038(spec)
    elif fixture_id == "110-otel-callable-branch-span":
        await _run_fixture_110(spec)
    elif fixture_id in {
        "040-llm-cache-attribute-emission",
        "041-llm-cache-attribute-absence",
        "042-llm-cache-attribute-reported-zero",
    }:
        await _run_llm_cache_fixture(spec)
    elif fixture_id == "050-llm-completion-event-dispatch":
        await _run_fixture_050(spec)
    elif fixture_id == "051-llm-completion-event-type-discrimination":
        await _run_fixture_051(spec)
    elif fixture_id == "052-llm-completion-event-caller-metadata-opt-in":
        await _run_fixture_052(spec)
    elif fixture_id == "053-llm-completion-event-no-event-on-failure":
        await _run_fixture_053(spec)
    elif fixture_id == "054-llm-completion-event-fan-out-index-population":
        await _run_fixture_054(spec)
    elif fixture_id == "055-llm-completion-event-branch-name-population":
        await _run_fixture_055(spec)
    elif fixture_id == "056-llm-completion-event-strict-serial-ordering":
        await _run_fixture_056(spec)
    elif fixture_id in {
        "060-llm-completion-event-input-messages-populated",
        "061-llm-completion-event-output-content-populated",
        "062-llm-completion-event-request-params-populated",
        "063-llm-completion-event-request-extras-populated",
        "064-llm-completion-event-active-prompt-populated",
        "065-llm-completion-event-active-prompt-null",
        "068-llm-completion-event-response-model-distinct-from-request",
    }:
        await _run_typed_event_cases(spec)
    elif fixture_id == "072-llm-failure-event-mutual-exclusion-with-completion-event":
        await _run_typed_event_cases(spec, expect_failure=True)
    elif fixture_id == "067-llm-completion-event-call-id-always-present-and-distinct":
        await _run_typed_event_chain_cases(spec)
    elif fixture_id == "071-llm-failure-event-call-id-distinct-from-completion-event":
        await _run_typed_event_chain_cases(spec, expect_failure=True)
    elif fixture_id == "058-implementation-attribution-otel":
        await _run_fixture_058(spec)
    elif fixture_id == "084-langfuse-session-user-promotion":
        await _run_fixture_084(spec)
    elif fixture_id in {
        "022-langfuse-basic-trace",
        "031-langfuse-subgraph-span-hierarchy",
        "032-langfuse-fan-out-per-instance-spans",
        "059-implementation-attribution-langfuse",
    }:
        await _run_langfuse_trace_fixture(spec)
    elif fixture_id in {
        "035-caller-invocation-id-uuid",
        "036-caller-invocation-id-non-uuid",
    }:
        await _run_invocation_id_fixture(spec)
    elif fixture_id in {
        "012-otel-llm-payload-default-off",
        "013-otel-llm-payload-enabled",
        "014-otel-llm-payload-truncation",
        "015-otel-llm-payload-image-redaction",
        "016-otel-llm-request-params",
        "017-otel-llm-request-params-partial",
        "018-otel-llm-request-extras",
        "019-otel-llm-genai-semconv",
        "020-otel-llm-genai-system-override",
        "021-otel-llm-disable-genai-semconv",
        "025-otel-llm-request-params-extended",
        "026-otel-caller-supplied-metadata",
        "057-llm-attempt-index-single-attempt-default",
        "085-llm-tool-call-request-attributes",
        "086-llm-tool-call-request-absent",
        "087-llm-tool-call-request-survives-payload-gating",
    }:
        await _run_llm_payload_fixture(spec)
    elif fixture_id in {
        "088-llm-metrics-token-and-duration",
        "090-metrics-error-type-on-duration",
        "091-metrics-disabled-no-measurements",
    }:
        await _run_metrics_fixture(spec)
    elif fixture_id in {
        "092-tool-call-event-dispatch",
        "093-tool-call-failed-event-dispatch",
        "094-tool-call-event-mutual-exclusion",
        "095-tool-call-id-links-to-llm-request",
        "096-tool-call-payload-gating",
        "097-otel-tool-span-attributes",
        "098-langfuse-tool-observation",
    }:
        await _run_tool_fixture(spec)
    else:
        raise AssertionError(f"no driver for supported fixture {fixture_id!r}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_observer() -> tuple[OTelObserver, Any]:
    """Build a fresh OTelObserver + InMemorySpanExporter pair."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    return observer, exporter


async def _run_graph(
    spec: Mapping[str, Any],
    observer: OTelObserver,
    *,
    correlation_id: str | None = None,
) -> Any:
    """Build + invoke a graph from a fixture spec; return the final
    state. Caller is responsible for calling ``observer.shutdown()``
    afterwards."""
    trace: list[str] = []
    built = build_graph(spec, trace=trace)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    final = await compiled.invoke(initial_state, correlation_id=correlation_id)
    await compiled.drain()
    return final


def _all_correlation_ids(spans: Any) -> set[str]:
    """Pull the ``openarmature.correlation_id`` attribute off every
    span; returns the unique set. Accepts any iterable of spans
    (``InMemorySpanExporter.get_finished_spans`` returns a tuple)."""
    return {cast("str", dict(s.attributes or {}).get("openarmature.correlation_id")) for s in spans}


# ---------------------------------------------------------------------------
# Fixture 001 — basic trace shape
# ---------------------------------------------------------------------------


async def _run_fixture_001(spec: Mapping[str, Any]) -> None:
    observer, exporter = _build_observer()
    final = await _run_graph(spec, observer, correlation_id=spec.get("caller_correlation_id"))
    observer.shutdown()
    spans = exporter.get_finished_spans()
    assert len(spans) == 4, (
        f"expected 4 spans (invocation + 3 nodes); got {len(spans)}: {[s.name for s in spans]}"
    )
    by_name = {s.name: s for s in spans}
    assert "openarmature.invocation" in by_name
    inv = by_name["openarmature.invocation"]
    assert inv.parent is None
    inv_attrs = dict(inv.attributes or {})
    assert inv_attrs.get("openarmature.graph.entry_node") == spec["entry"]
    cid = inv_attrs.get("openarmature.correlation_id")
    assert isinstance(cid, str) and len(cid) > 0
    inv_ctx = inv.context
    assert inv_ctx is not None
    invocation_span_id = inv_ctx.span_id
    for node_name in spec["nodes"]:
        assert node_name in by_name, f"missing span for {node_name!r}"
        node_span = by_name[node_name]
        node_parent = node_span.parent
        assert node_parent is not None and node_parent.span_id == invocation_span_id
        node_attrs = dict(node_span.attributes or {})
        assert node_attrs.get("openarmature.node.name") == node_name
        assert list(node_attrs.get("openarmature.node.namespace") or []) == [node_name]
        assert isinstance(node_attrs.get("openarmature.node.step"), int)
        assert node_attrs.get("openarmature.node.attempt_index") == 0
        assert node_attrs.get("openarmature.correlation_id") == cid
    expected_trace = ["a", "b", "c"]
    assert final.trace == expected_trace  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixture 002 — subgraph hierarchy
# ---------------------------------------------------------------------------


async def _run_fixture_002(spec: Mapping[str, Any]) -> None:
    """The subgraph wrapper synthesizes a dispatch span;
    inner-node spans parent under it; the dispatch span parents
    under the invocation."""
    observer, exporter = _build_observer()
    subgraphs = _compile_subgraphs(spec)
    trace_log: list[str] = []
    built = build_graph(spec, subgraphs=subgraphs, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    by_name: dict[str, list[Any]] = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)

    # Invocation span at the root.
    inv_list = by_name.get("openarmature.invocation") or []
    assert len(inv_list) == 1, f"expected 1 invocation span; got {len(inv_list)}"
    inv = inv_list[0]
    assert inv.parent is None
    assert inv.context is not None
    invocation_span_id = inv.context.span_id

    # Top-level outer nodes parent under invocation.
    for outer_node in ("outer_in", "outer_out"):
        outer_spans = by_name.get(outer_node) or []
        assert len(outer_spans) == 1, f"expected 1 span for {outer_node!r}; got {len(outer_spans)}"
        node = outer_spans[0]
        assert node.parent is not None and node.parent.span_id == invocation_span_id, (
            f"{outer_node!r} MUST parent under invocation span"
        )

    # The subgraph wrapper synthesizes a dispatch span at namespace
    # ("outer_sub",); its parent is the invocation span.
    sub_dispatch_spans = by_name.get("outer_sub") or []
    assert len(sub_dispatch_spans) == 1, (
        f"expected 1 synthetic subgraph dispatch span for outer_sub; got {len(sub_dispatch_spans)}"
    )
    sub_dispatch = sub_dispatch_spans[0]
    assert sub_dispatch.parent is not None and sub_dispatch.parent.span_id == invocation_span_id, (
        "subgraph dispatch span MUST parent under the invocation span per §4.5"
    )
    assert sub_dispatch.context is not None
    sub_dispatch_id = sub_dispatch.context.span_id
    sub_dispatch_attrs = dict(sub_dispatch.attributes or {})
    # Per observability §5.3 + coord thread `clarify-subgraph-name-
    # semantics` Option A: `openarmature.subgraph.name` carries the
    # compiled subgraph's identity, NOT the wrapper node name. The
    # conformance adapter sets ``subgraph_identity = "inner"`` when
    # compiling the fixture's ``subgraph: { name: inner }`` block.
    assert sub_dispatch_attrs.get("openarmature.subgraph.name") == "inner"

    # Inner-node spans parent under the subgraph dispatch span and
    # carry the nested namespace.
    for inner_node in ("inner_x", "inner_y"):
        inner_spans = by_name.get(inner_node) or []
        assert len(inner_spans) == 1, f"expected 1 span for {inner_node!r}; got {len(inner_spans)}"
        inner = inner_spans[0]
        assert inner.parent is not None and inner.parent.span_id == sub_dispatch_id, (
            f"{inner_node!r} MUST parent under the subgraph dispatch span per §4.5"
        )
        inner_attrs = dict(inner.attributes or {})
        assert list(inner_attrs.get("openarmature.node.namespace") or []) == ["outer_sub", inner_node], (
            f"{inner_node!r} namespace MUST be ['outer_sub', '{inner_node}']; got "
            f"{inner_attrs.get('openarmature.node.namespace')!r}"
        )


# ---------------------------------------------------------------------------
# Fixture 003 — error status mapping (node_exception case)
# ---------------------------------------------------------------------------


async def _run_fixture_003(spec: Mapping[str, Any]) -> None:
    """A node-exception failure produces an ERROR span
    with the canonical category in the description, an exception
    event recorded, and the ``openarmature.error.category``
    attribute. Sibling spans before the failure stay OK; the
    invocation span ends ERROR (OTel doesn't auto-propagate child
    status to parents, so the OTelObserver explicitly sets ERROR
    on the invocation span when any child errors per
    ``_handle_completed``)."""
    from opentelemetry.trace import StatusCode

    from openarmature.graph import RuntimeGraphError

    observer, exporter = _build_observer()
    trace_log: list[str] = []
    built = build_graph(spec, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    with pytest.raises(RuntimeGraphError):
        await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    by_name = {s.name: s for s in spans}

    ok_node = by_name.get("ok_node")
    assert ok_node is not None
    assert ok_node.status.status_code == StatusCode.OK, (
        f"ok_node status MUST be OK; got {ok_node.status.status_code}"
    )

    fail_node = by_name.get("fail_node")
    assert fail_node is not None
    assert fail_node.status.status_code == StatusCode.ERROR, (
        f"fail_node status MUST be ERROR; got {fail_node.status.status_code}"
    )
    assert fail_node.status.description == "node_exception", (
        f"fail_node status_description MUST be 'node_exception'; got {fail_node.status.description!r}"
    )
    fail_attrs = dict(fail_node.attributes or {})
    assert fail_attrs.get("openarmature.error.category") == "node_exception"
    # Exception event recorded on the span via record_exception.
    exception_events = [e for e in fail_node.events if e.name == "exception"]
    event_names = [e.name for e in fail_node.events]
    assert len(exception_events) >= 1, (
        f"fail_node MUST have at least one 'exception' event recorded; got {event_names}"
    )

    # Invocation span ends ERROR when any child errors per spec
    # §4.2 / fixture 003. The OTelObserver sets ERROR explicitly in
    # ``_handle_completed`` (OTel doesn't auto-propagate child status
    # to parents).
    inv = by_name.get("openarmature.invocation")
    assert inv is not None
    assert inv.status.status_code == StatusCode.ERROR, (
        f"invocation span status MUST be ERROR when a child errored; got {inv.status.status_code}"
    )


# ---------------------------------------------------------------------------
# Fixture 004 — routing-error attribution (proposal 0012 / spec v0.9.0)
# ---------------------------------------------------------------------------


async def _run_fixture_004(spec: Mapping[str, Any]) -> None:
    """Routing errors land on the preceding node's ``completed`` event
    with ``error`` populated
    (sharing the started/completed pair rather than producing a
    separate one). The OTel observer's existing
    ``_handle_completed`` ERROR-mapping path picks this up
    automatically — no observer-side change needed for the swap.

    Driver verifies: the ``pick`` node's span ends ERROR with
    ``status_description == "routing_error"``, an ``exception``
    event recorded, and the ``openarmature.error.category``
    attribute. No span for the edge function (no ``edge_spans``) —
    edge logic is folded into the preceding node span."""
    from opentelemetry.trace import StatusCode

    from openarmature.graph import RuntimeGraphError

    observer, exporter = _build_observer()
    trace_log: list[str] = []
    built = build_graph(spec, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    with pytest.raises(RuntimeGraphError) as excinfo:
        await compiled.invoke(initial_state)
    assert excinfo.value.category == "routing_error"
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    by_name = {s.name: s for s in spans}

    pick = by_name.get("pick")
    assert pick is not None
    assert pick.status.status_code == StatusCode.ERROR, (
        f"preceding node 'pick' span MUST be ERROR; got {pick.status.status_code}"
    )
    assert pick.status.description == "routing_error", (
        f"preceding node 'pick' span status_description MUST be 'routing_error'; "
        f"got {pick.status.description!r}"
    )
    pick_attrs = dict(pick.attributes or {})
    assert pick_attrs.get("openarmature.error.category") == "routing_error"
    # Exception event recorded on the span via record_exception.
    exception_events = [e for e in pick.events if e.name == "exception"]
    event_names = [e.name for e in pick.events]
    assert len(exception_events) >= 1, (
        f"'pick' MUST have at least one 'exception' event recorded; got {event_names}"
    )

    # Per fixture 004's "no_edge_spans: true" — the edge function
    # itself does not produce a separate span; the routing error is
    # folded into the preceding node's span.
    edge_span_names = {"edge", "openarmature.edge", "edge_function"}
    edge_spans = [s for s in spans if s.name in edge_span_names]
    assert edge_spans == [], (
        f"there MUST be no separate edge-function spans per §4.2; got {[s.name for s in edge_spans]}"
    )

    # Unreachable nodes never fire spans (they were unreached).
    for unreachable in ("unreachable_a", "unreachable_b"):
        assert unreachable not in by_name, f"{unreachable!r} MUST not produce a span — never reached"

    # Invocation span ends ERROR per the §4.2 invocation-status
    # propagation contract.
    inv = by_name.get("openarmature.invocation")
    assert inv is not None
    assert inv.status.status_code == StatusCode.ERROR, (
        f"invocation span MUST end ERROR when a child errors; got {inv.status.status_code}"
    )


# ---------------------------------------------------------------------------
# Fixture 007 — retry attempt spans
# ---------------------------------------------------------------------------


async def _run_fixture_006(spec: Mapping[str, Any]) -> None:
    """Non-detached fan-out instances synthesize per-instance dispatch
    spans nested between
    the fan-out node span and the inner-node spans. The fan-out node
    span carries ``item_count`` / ``concurrency`` / ``error_policy``
    from ``NodeEvent.fan_out_config``; per-instance spans carry
    ``fan_out_index`` and ``parent_node_name``."""
    # Subgraphs declared at the spec level (outside ``cases:``) — the
    # ``worker`` subgraph used by every case in this fixture lives
    # there. Compile once and reuse across cases.
    _patch_unsupported_directives(spec)
    subgraphs = _compile_subgraphs(spec)
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_006_case(case, subgraphs)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_006_case(case: Mapping[str, Any], subgraphs: Mapping[str, Any]) -> None:
    _patch_unsupported_directives(case)

    observer, exporter = _build_observer()
    trace_log: list[str] = []
    built = build_graph(case, subgraphs=dict(subgraphs), trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))
    await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    # Span-tree shape per fixture:
    #   invocation
    #   └─ process (fan-out NODE) — item_count=3, concurrency=2, error_policy="collect"
    #      ├─ process (instance, fan_out_index=0) — parent_node_name="process"
    #      │  └─ compute
    #      ├─ process (instance, fan_out_index=1) — parent_node_name="process"
    #      │  └─ compute
    #      └─ process (instance, fan_out_index=2) — parent_node_name="process"
    #         └─ compute
    process_spans = [s for s in spans if s.name == "process"]
    # Expect 4: 1 fan-out node + 3 per-instance dispatch spans.
    assert len(process_spans) == 4, (
        f"expected 4 'process' spans (1 fan-out node + 3 per-instance dispatch); got {len(process_spans)}"
    )
    # Fan-out node span carries the three §5.4 attributes.
    fan_out_node_spans = [
        s
        for s in process_spans
        if dict(s.attributes or {}).get("openarmature.fan_out.item_count") is not None
    ]
    assert len(fan_out_node_spans) == 1, (
        f"expected exactly 1 fan-out NODE span (with item_count attribute); got {len(fan_out_node_spans)}"
    )
    fan_out_node_span = fan_out_node_spans[0]
    fan_out_attrs = dict(fan_out_node_span.attributes or {})
    assert fan_out_attrs.get("openarmature.fan_out.item_count") == 3
    assert fan_out_attrs.get("openarmature.fan_out.concurrency") == 2
    assert fan_out_attrs.get("openarmature.fan_out.error_policy") == "collect"

    # Per-instance dispatch spans: 3 of them, each with
    # fan_out_index 0..2 and parent_node_name "process".
    per_instance_spans = [s for s in process_spans if s != fan_out_node_span]
    assert len(per_instance_spans) == 3
    fan_out_indices: set[int] = set()
    for s in per_instance_spans:
        attrs = dict(s.attributes or {})
        idx = attrs.get("openarmature.node.fan_out_index")
        assert isinstance(idx, int), f"per-instance span MUST carry fan_out_index; got attrs={attrs}"
        fan_out_indices.add(idx)
        assert attrs.get("openarmature.fan_out.parent_node_name") == "process", (
            f"per-instance span MUST carry parent_node_name='process'; got {attrs}"
        )
        # Each per-instance dispatch span parents under the fan-out
        # node span (proposal 0013 + §5.4 nesting).
        assert s.parent is not None and s.parent.span_id == fan_out_node_span.context.span_id, (
            "per-instance dispatch span MUST parent under the fan-out node span"
        )
    assert fan_out_indices == {0, 1, 2}, (
        f"per-instance fan_out_index range MUST be 0..2; got {sorted(fan_out_indices)}"
    )

    # Each per-instance dispatch span has a 'compute' child (the
    # inner-node work).
    per_instance_ids = {s.context.span_id for s in per_instance_spans}
    compute_spans = [s for s in spans if s.name == "compute"]
    assert len(compute_spans) == 3, f"expected 3 compute spans; got {len(compute_spans)}"
    for cs in compute_spans:
        assert cs.parent is not None and cs.parent.span_id in per_instance_ids, (
            "compute span MUST parent under a per-instance dispatch span"
        )


async def _run_fixture_007(spec: Mapping[str, Any]) -> None:
    """Two sub-cases:

    1. ``three_attempts_third_succeeds`` — retry succeeds on
       attempt 2; expect 3 sibling attempt spans (ERROR, ERROR, OK).
    2. ``retry_exhausts_all_three_spans_error`` — retry exhausts;
       expect 3 sibling attempt spans (all ERROR); invoke raises.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_007_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_007_case(case: Mapping[str, Any]) -> None:
    from opentelemetry.trace import StatusCode

    from openarmature.graph import RuntimeGraphError
    from openarmature.graph.middleware import RetryConfig, RetryMiddleware
    from openarmature.graph.middleware.retry import deterministic_backoff

    observer, exporter = _build_observer()
    # The fixture's flaky directive uses ``fail_count: N`` shape
    # (fail attempts 0..N-1, succeed on attempt N) which the
    # adapter doesn't translate; rewrite to the adapter's
    # ``failure_sequence`` shape before building.
    flaky_node_name = cast("str", case["entry"])
    nodes = cast("dict[str, Any]", case["nodes"])
    flaky_node = cast("dict[str, Any]", nodes[flaky_node_name])
    flaky_directive = cast("dict[str, Any]", flaky_node["flaky"])
    fail_count = int(flaky_directive["fail_count"])
    fail_category = cast("str", flaky_directive.get("category", "provider_unavailable"))
    on_success = cast("dict[str, Any]", flaky_directive.get("on_success", {}))
    flaky_node["flaky"] = {
        "failure_sequence": [
            {"category": fail_category, "message": f"flaky attempt {i}"} for i in range(fail_count)
        ],
        "success_update": on_success,
    }
    # Translate the per-node retry middleware. The adapter accepts
    # ``node_middleware`` mapping; the YAML's
    # ``nodes.flaky.middleware: [{type: retry, ...}]`` maps in.
    middleware_specs = cast("list[dict[str, Any]]", flaky_node.pop("middleware", []) or [])
    node_middleware: dict[str, list[Any]] = {}
    for mw_spec in middleware_specs:
        if mw_spec["type"] != "retry":
            raise AssertionError(f"fixture 007: unexpected middleware type {mw_spec['type']!r}")
        backoff_cfg = cast(
            "dict[str, Any]", mw_spec.get("backoff") or {"type": "deterministic", "seconds": 0}
        )
        if backoff_cfg["type"] != "deterministic":
            raise AssertionError(f"fixture 007: unsupported backoff type {backoff_cfg['type']!r}")
        backoff = deterministic_backoff(float(backoff_cfg.get("seconds", 0)))
        classifier_cfg = cast("dict[str, Any] | None", mw_spec.get("classifier"))
        if classifier_cfg is not None:
            transient = frozenset(cast("list[str]", classifier_cfg.get("transient_categories", [])))

            def _classifier(exc: Exception, _state: Any, _transient: frozenset[str] = transient) -> bool:
                return getattr(exc, "category", None) in _transient

            classifier_fn: Any = _classifier
        else:
            classifier_fn = None
        node_middleware.setdefault(flaky_node_name, []).append(
            RetryMiddleware(
                RetryConfig(
                    max_attempts=int(mw_spec.get("max_attempts", 3)),
                    backoff=backoff,
                    classifier=classifier_fn,
                )
            )
        )

    trace_log: list[str] = []
    built = build_graph(case, trace=trace_log, node_middleware=node_middleware)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))
    expected_error = case.get("expected_error")
    if expected_error is not None:
        with pytest.raises(RuntimeGraphError):
            await compiled.invoke(initial_state)
    else:
        await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    attempt_spans = [s for s in spans if s.name == flaky_node_name]
    assert len(attempt_spans) == 3, (
        f"expected 3 sibling attempt spans for {flaky_node_name!r}; got {len(attempt_spans)}"
    )
    # Each attempt span has a distinct attempt_index in 0..2 and
    # they all share the invocation as parent (siblings).
    attempt_indices: list[int] = []
    parent_span_ids: set[int] = set()
    for span in attempt_spans:
        attrs = dict(span.attributes or {})
        idx = attrs.get("openarmature.node.attempt_index")
        assert isinstance(idx, int)
        attempt_indices.append(idx)
        assert span.parent is not None
        parent_span_ids.add(span.parent.span_id)
    assert sorted(attempt_indices) == [0, 1, 2], (
        f"attempt_index values MUST be 0..2; got {sorted(attempt_indices)}"
    )
    assert len(parent_span_ids) == 1, (
        "all attempt spans MUST share the same parent (sibling-level under the invocation); "
        f"got {len(parent_span_ids)} distinct parents"
    )

    # Status assertions.
    by_attempt = {
        cast("int", dict(s.attributes or {})["openarmature.node.attempt_index"]): s for s in attempt_spans
    }
    if expected_error is not None:
        # All three attempts ERROR.
        for idx in (0, 1, 2):
            assert by_attempt[idx].status.status_code == StatusCode.ERROR, (
                f"attempt {idx} status MUST be ERROR (retry exhausted); "
                f"got {by_attempt[idx].status.status_code}"
            )
    else:
        # Attempts 0 + 1 ERROR, attempt 2 OK.
        for idx in (0, 1):
            assert by_attempt[idx].status.status_code == StatusCode.ERROR, (
                f"attempt {idx} status MUST be ERROR (failed before retry succeeded); "
                f"got {by_attempt[idx].status.status_code}"
            )
        assert by_attempt[2].status.status_code == StatusCode.OK, (
            f"attempt 2 status MUST be OK (success on third attempt); got {by_attempt[2].status.status_code}"
        )


# ---------------------------------------------------------------------------
# Fixture 011 — determinism
# ---------------------------------------------------------------------------


# Spec-canonical attributes that are non-deterministic by design and
# MUST be excluded from determinism-comparison runs. Per spec
# coordination on the fixture (08-spec-prep-sync-confirmed reaffirms
# the §3.2 + §5.1 distinction): caller-supplied correlation_id is
# deterministic; auto-generated UUIDv4 is not. Fixture 011 omits
# ``caller_correlation_id``, so the auto-generated correlation_id IS
# in the ignore set for this fixture.
_DETERMINISM_IGNORED_ATTRS: frozenset[str] = frozenset(
    {
        "openarmature.invocation_id",
        "openarmature.correlation_id",
    }
)


async def _run_fixture_011(spec: Mapping[str, Any]) -> None:
    """Deterministic span content is identical across two
    invocations of the same graph with the same input. The
    signature compared per-span:
    ``(name, status_code, parent_name, attrs ∖ ignored_set)``.
    Parent linkage is encoded as the parent span's NAME rather
    than its span_id (span_ids are non-deterministic per OTel SDK's
    default RandomIdGenerator); a hierarchy regression where a
    node reparented to a different ancestor surfaces as a
    parent_name divergence."""
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_011_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_011_case(case: Mapping[str, Any]) -> None:
    # Translate the fixture's ``when:`` conditional-edge syntax
    # (``when: {field: counter, gt: 0}``) into the adapter's
    # ``condition: {if_field, equals, then, else}`` shape. The
    # adapter doesn't have a ``gt`` builder, but the deterministic
    # input means ``counter == 1`` always — so ``gt: 0`` is
    # functionally equivalent to ``equals: 1`` for this fixture's
    # flow. The determinism comparison itself doesn't depend on
    # which adapter construct represents the edge; the same
    # branch always fires under identical inputs. (Generic
    # ``gt``/``lt``/etc. edge-condition support is tracked under
    # the Harness backlog in
    # ``openarmature-coord/docs/phase-6-1-conformance-fillin.md``.)
    case_for_build = _translate_011_when_edges(case)

    invocations = int(case.get("invocations", 2))
    assert invocations == 2, f"fixture 011: expected invocations=2; got {invocations}"

    runs: list[list[Any]] = []
    for _ in range(invocations):
        observer, exporter = _build_observer()
        trace_log: list[str] = []
        built = build_graph(case_for_build, trace=trace_log)
        compiled = built.builder.compile()
        compiled.attach_observer(observer)
        initial_state = built.initial_state(case.get("initial_state", {}))
        await compiled.invoke(initial_state)
        await compiled.drain()
        observer.shutdown()
        runs.append(list(exporter.get_finished_spans()))

    assert len(runs[0]) == len(runs[1]), (
        f"deterministic input MUST produce equal span counts; got {len(runs[0])} vs {len(runs[1])}"
    )

    # Compare each span's structural signature across runs. Span
    # span_ids are non-deterministic, so we encode the parent
    # linkage by looking up parent.span_id in the same run's
    # by-id map and including the parent's NAME in the signature.
    # That way a hierarchy regression (e.g., a node reparented
    # from invocation to a sibling) shows up as a signature
    # difference even though both spans' own attributes are
    # unchanged.
    def _signature(
        span: Any, by_id: Mapping[int, Any]
    ) -> tuple[str, str, str | None, tuple[tuple[str, Any], ...]]:
        attrs = dict(span.attributes or {})
        deterministic_items = sorted(
            (k, _normalize_attr_value(v)) for k, v in attrs.items() if k not in _DETERMINISM_IGNORED_ATTRS
        )
        parent_name: str | None = None
        if span.parent is not None:
            parent_span = by_id.get(span.parent.span_id)
            if parent_span is not None:
                parent_name = cast("str", parent_span.name)
        return (
            cast("str", span.name),
            str(span.status.status_code),
            parent_name,
            tuple(deterministic_items),
        )

    by_id_run_0: dict[int, Any] = {}
    for s in runs[0]:
        if s.context is not None:
            by_id_run_0[s.context.span_id] = s
    by_id_run_1: dict[int, Any] = {}
    for s in runs[1]:
        if s.context is not None:
            by_id_run_1[s.context.span_id] = s
    sig_run_0 = sorted(_signature(s, by_id_run_0) for s in runs[0])
    sig_run_1 = sorted(_signature(s, by_id_run_1) for s in runs[1])
    assert sig_run_0 == sig_run_1, (
        f"deterministic span content MUST match across runs; "
        f"first divergence: run_0={sig_run_0!r} vs run_1={sig_run_1!r}"
    )


async def _run_fixture_028(spec: Mapping[str, Any]) -> None:
    """Caller-supplied metadata keys under reserved namespaces
    (``openarmature.*``, ``gen_ai.*``) MUST
    raise at the ``invoke()`` boundary before any work begins.
    The harness asserts:

    - The invocation raises ``ValueError`` synchronously.
    - No OTel spans are emitted (the OTel observer attached
      to the graph never saw a single event).
    - No Langfuse observations are emitted (the Langfuse
      observer attached likewise saw nothing).
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    from openarmature.graph import END, GraphBuilder  # noqa: PLC0415
    from openarmature.observability.langfuse import (  # noqa: PLC0415
        InMemoryLangfuseClient,
        LangfuseObserver,
    )

    # Case-level deferrals for fixture 028. The spec extends the
    # fixture with new negative-control cases as new reserved keys
    # land in §3.4; impl-side coverage for those cases lands in the
    # PR that implements the corresponding key reservation. Stored as
    # a set since the skip is a ``continue`` (no ``pytest.skip``
    # reason surface); rationale lives in the per-block comment above
    # each name.
    _deferred_cases: set[str] = {
        # Proposal 0052 (implementation attribution, spec v0.44.0)
        # extends the reserved-set 24 → 26 names with
        # ``implementation_name`` / ``implementation_version``.
        # Coverage lands in PR 3 of the v0.12.0 cycle alongside the
        # ``openarmature.implementation.*`` invocation-span attribute
        # emission.
        "rejects_reserved_oa_name_implementation_name",
        "rejects_reserved_oa_name_implementation_version",
    }
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        if case_name in _deferred_cases:
            continue
        try:
            # Build a minimal graph from the case's nodes/edges. The
            # fixture's node is a noop update — we never expect it to
            # run since the boundary rejects before any worker spins
            # up.
            from .adapter import build_state_cls  # noqa: PLC0415

            state_cls = build_state_cls("RejectionFixtureState", case["state"]["fields"])
            builder = GraphBuilder(state_cls)
            nodes_spec = cast("dict[str, Any]", case["nodes"])
            for node_name, node_spec in nodes_spec.items():
                node_dict = cast("dict[str, Any]", node_spec)
                update_block = cast("dict[str, Any]", node_dict["update"])
                augment_block = cast("dict[str, Any] | None", node_dict.get("augment_metadata"))

                def _make_body(
                    payload: dict[str, Any],
                    augment: dict[str, Any] | None,
                ) -> Any:
                    # Per spec §3.4 + proposal 0040: the augment_metadata
                    # primitive injects a ``set_invocation_metadata(**augment)``
                    # call at the top of the node body. Used by 028's
                    # mid-invocation-rejection case (reserved name `step`)
                    # and by 034 for the open-span update demonstration.
                    from openarmature.observability.metadata import (  # noqa: PLC0415
                        set_invocation_metadata,
                    )

                    async def _body(_s: Any) -> dict[str, Any]:
                        if augment is not None:
                            set_invocation_metadata(**augment)
                        return dict(payload)

                    return _body

                builder.add_node(node_name, _make_body(update_block, augment_block))
            for edge in cast("list[dict[str, str]]", case["edges"]):
                target_raw = edge["to"]
                target = END if target_raw == "END" else target_raw
                builder.add_edge(edge["from"], target)
            builder.set_entry(cast("str", case["entry"]))
            graph = builder.compile()

            exporter = InMemorySpanExporter()
            otel_observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
            langfuse_client = InMemoryLangfuseClient()
            langfuse_observer = LangfuseObserver(client=langfuse_client)
            graph.attach_observer(otel_observer)
            graph.attach_observer(langfuse_observer)

            caller_metadata = cast("dict[str, Any]", case["caller_metadata"])
            expected = cast("dict[str, Any]", case["expected"])
            expects_boundary_rejection = expected.get("invoke_rejects_at_api_boundary", False)
            expects_call_site_rejection = expected.get("augment_rejects_at_call_site", False)
            try:
                if expects_boundary_rejection:
                    # Boundary-rejection path: invoke()'s caller_metadata
                    # validator rejects before any work begins. Covers
                    # both the prefix-namespace rejection (openarmature.*
                    # / gen_ai.*, from 0034) and the exact-key-name
                    # rejection (0041's §8.4 reserved set). Both error
                    # messages contain "reserved".
                    with pytest.raises(ValueError, match="reserved"):
                        await graph.invoke(state_cls(), metadata=caller_metadata)
                elif expects_call_site_rejection:
                    # Mid-invocation rejection path: caller_metadata
                    # passes the boundary; the node body's
                    # ``set_invocation_metadata(**augment)`` raises a
                    # ValueError at the call site. The engine wraps the
                    # node-body raise in NodeException whose
                    # ``__cause__`` is the ValueError. The §3.4 contract
                    # is that the helper raises at the call site — the
                    # reserved key MUST NOT reach any emission, hence
                    # no spans / no Langfuse observations afterward.
                    from openarmature.graph import NodeException  # noqa: PLC0415

                    with pytest.raises(NodeException) as exc_info:
                        await graph.invoke(state_cls(), metadata=caller_metadata)
                    cause = exc_info.value.__cause__
                    assert isinstance(cause, ValueError), (
                        f"expected NodeException.__cause__ to be ValueError; got {type(cause).__name__}"
                    )
                    assert "reserved" in str(cause), f"expected 'reserved' in cause message; got {cause!s}"
                else:
                    raise AssertionError(
                        "case has neither invoke_rejects_at_api_boundary nor augment_rejects_at_call_site set"
                    )
                await graph.drain()
            finally:
                otel_observer.shutdown()

            if expected.get("no_spans_emitted"):
                spans = exporter.get_finished_spans()
                assert len(spans) == 0, f"expected zero spans, got {[s.name for s in spans]}"
            if expected.get("no_langfuse_observations_emitted"):
                # Trace MAY exist (lazy-open on first event); the
                # invariant is "no observations are emitted." Since
                # invoke rejects before any event fires, neither
                # trace nor observations should be created.
                assert len(langfuse_client.traces) == 0, (
                    f"expected zero Langfuse traces, got {sorted(langfuse_client.traces.keys())}"
                )
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


def _normalize_attr_value(value: Any) -> Any:
    """OTel attribute values can be tuple or list shapes for sequence
    types depending on how they were set; normalize for comparison."""
    if isinstance(value, list):
        return tuple(cast("list[Any]", value))
    if isinstance(value, tuple):
        return cast("tuple[Any, ...]", value)
    return value


def _translate_011_when_edges(case: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite fixture 011's ``when: {field: counter, gt: 0}``
    edges into the adapter's ``condition: {if_field, equals,
    then, else}`` shape. The deterministic input always satisfies
    the branch, so the comparison can be ``equals: 1``."""
    new_case = cast("dict[str, Any]", copy.deepcopy(case))
    new_edges: list[Any] = []
    branch_when_edge: dict[str, Any] | None = None
    branch_default_edge: dict[str, Any] | None = None
    for edge in cast("list[dict[str, Any]]", new_case.get("edges", [])):
        if "when" in edge:
            branch_when_edge = edge
        elif edge.get("from") == "branch" and "when" not in edge:
            branch_default_edge = edge
        else:
            new_edges.append(edge)
    if branch_when_edge is not None and branch_default_edge is not None:
        when = cast("dict[str, Any]", branch_when_edge["when"])
        if_field = cast("str", when["field"])
        # gt: 0 with the deterministic input (counter == 1) →
        # equals: 1 is equivalent for this fixture's flow.
        new_edges.append(
            {
                "from": "branch",
                "condition": {
                    "if_field": if_field,
                    "equals": 1,
                    "then": branch_when_edge["to"],
                    "else": branch_default_edge["to"],
                },
            }
        )
    elif branch_when_edge is not None or branch_default_edge is not None:
        raise AssertionError("fixture 011: expected paired when/default edges from 'branch'")
    new_case["edges"] = new_edges
    return new_case


# ---------------------------------------------------------------------------
# Fixture 009 — correlation_id cross-cutting
# ---------------------------------------------------------------------------


async def _run_fixture_009(spec: Mapping[str, Any]) -> None:
    """Three sub-cases, each in the ``cases:`` block:

    1. caller-supplied correlation_id used verbatim on every span.
    2. auto-generated UUIDv4 used uniformly across all spans.
    3. Two back-to-back invocations get DIFFERENT correlation_ids,
       both UUIDv4 form.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_009_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_009_case(case: Mapping[str, Any]) -> None:
    case_name = case["name"]
    if case_name == "context_reset_between_invocations":
        # Two back-to-back invocations of the same compiled graph;
        # each MUST get its own UUIDv4 (distinct from the other).
        observer, exporter = _build_observer()
        # Build the graph ONCE so both invocations share it.
        from .adapter import build_graph as _bg

        built = _bg(case)
        compiled = built.builder.compile()
        compiled.attach_observer(observer)
        for _ in range(int(case.get("invocations", 2))):
            await compiled.invoke(built.initial_state(case.get("initial_state", {})))
            await compiled.drain()
        observer.shutdown()
        spans = exporter.get_finished_spans()

        # Group spans by trace_id (each invocation has its own trace).
        by_trace: dict[int, list[Any]] = {}
        for s in spans:
            tid = s.context.trace_id
            by_trace.setdefault(tid, []).append(s)
        assert len(by_trace) == 2, f"expected 2 distinct traces (one per invocation); got {len(by_trace)}"
        # Each invocation's spans share one correlation_id.
        per_invocation_cids: list[str] = []
        for trace_spans in by_trace.values():
            cids = _all_correlation_ids(trace_spans)
            assert len(cids) == 1, f"each invocation MUST uniformly carry one correlation_id; got {cids}"
            cid = next(iter(cids))
            assert _UUIDV4_RE.match(cid), f"auto-generated correlation_id MUST be UUIDv4; got {cid!r}"
            per_invocation_cids.append(cid)
        # Cross-invocation: distinct.
        assert per_invocation_cids[0] != per_invocation_cids[1], (
            "back-to-back invocations MUST get distinct correlation_ids"
        )
        return

    # Sub-cases 1 & 2 (single-invocation).
    observer, exporter = _build_observer()
    await _run_graph(case, observer, correlation_id=case.get("caller_correlation_id"))
    observer.shutdown()
    spans = exporter.get_finished_spans()
    cids = _all_correlation_ids(spans)
    assert len(cids) == 1, f"every span MUST carry the same correlation_id; got {cids}"
    cid = next(iter(cids))
    expected = case.get("caller_correlation_id")
    if expected is not None:
        # Caller-supplied → exact match.
        assert cid == expected, (
            f"caller_correlation_id MUST be used verbatim; got {cid!r}, expected {expected!r}"
        )
    else:
        # Auto-generated → UUIDv4.
        assert _UUIDV4_RE.match(cid), f"auto-generated correlation_id MUST be UUIDv4; got {cid!r}"


# ---------------------------------------------------------------------------
# Fixture 005, 008 placeholders — driven below in subsequent commits
# ---------------------------------------------------------------------------


async def _run_fixture_005(spec: Mapping[str, Any]) -> None:
    """Three sub-cases:

    1. ``default`` — LLM span emits with its attributes, parented under
       the calling node.
    2. ``disable_llm_spans`` — opt-out suppresses the LLM span entirely.
    3. ``external_auto_instrumentation_active`` — second exporter on
       the OTel global provider; openarmature spans MUST NOT leak to
       it (the load-bearing TracerProvider isolation guarantee).
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_005_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_005_case(case: Mapping[str, Any]) -> None:
    case_name = case["name"]
    disable_llm_spans = bool(case.get("disable_llm_spans", False))
    caller_global_active = bool(case.get("caller_global_otel_active", False))

    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    # Optional second exporter on the OTel global provider — sub-case 3.
    # Save the prior global provider so we can restore it after the
    # case (otherwise the global state leaks to subsequent tests).
    global_exporter: InMemorySpanExporter | None = None
    prior_global = otel_trace.get_tracer_provider() if caller_global_active else None
    if caller_global_active:
        global_exporter = InMemorySpanExporter()
        global_provider = TracerProvider()
        global_provider.add_span_processor(SimpleSpanProcessor(global_exporter))
        # OTel SDK 1.x's ``set_tracer_provider`` is guarded by a
        # ``_TRACER_PROVIDER_SET_ONCE`` primitive — once a non-default
        # provider is set, subsequent calls are silent no-ops (with a
        # WARNING log "Overriding of current TracerProvider is not
        # allowed"). If a prior test in the suite-run order left a
        # non-default provider behind, the call below would no-op and
        # this case's ``global_exporter`` would receive 0 spans. Reset
        # both the value AND the Once explicitly so this case's set
        # always wins. The finally block below restores ``prior_global``
        # via the same direct reset so the next test starts clean.
        once = otel_trace._TRACER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
        with once._lock:  # pyright: ignore[reportPrivateUsage]
            otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
            once._done = False  # pyright: ignore[reportPrivateUsage]
        otel_trace.set_tracer_provider(global_provider)

    try:
        private_exporter = InMemorySpanExporter()
        observer = OTelObserver(
            span_processor=SimpleSpanProcessor(private_exporter),
            disable_llm_spans=disable_llm_spans,
        )

        # Build a graph whose entry node calls a mock LLM provider.
        graph, _ = _build_graph_with_mock_llm(case)
        graph.attach_observer(observer)

        # Drive the graph. The ``calls_llm`` node body reads the mock
        # responses set up in ``_build_graph_with_mock_llm`` (httpx
        # MockTransport keyed off the response queue).
        initial_state_cls = graph.state_cls
        final = await graph.invoke(initial_state_cls())
        await graph.drain()
        observer.shutdown()
        private_spans = private_exporter.get_finished_spans()

        # Sub-case 3: external span emitted by the harness through the
        # global tracer (simulating auto-instrumentation).
        if caller_global_active:
            global_tracer = otel_trace.get_tracer("external-instrumentation")
            with global_tracer.start_as_current_span("external.llm.call"):
                pass
            assert global_exporter is not None
            global_spans = global_exporter.get_finished_spans()
            assert len(global_spans) == 1, (
                f"global exporter MUST see exactly one external span; got {len(global_spans)}"
            )
            assert global_spans[0].name == "external.llm.call"
            # The load-bearing isolation check.
            for s in global_spans:
                assert not s.name.startswith("openarmature."), (
                    f"openarmature spans MUST NOT leak to the global provider; got {s.name!r}"
                )
    finally:
        if caller_global_active and prior_global is not None:
            # OTel SDK 1.x makes set_tracer_provider one-shot: once a
            # non-default provider is set, subsequent calls are no-ops.
            # Restore by resetting the private Once + state directly so
            # the global doesn't leak into subsequent tests.
            _reset_otel_global_tracer_provider(prior_global)

    # Common assertions: the LLM span presence/absence + (when
    # present) attributes + parent-child to the calling node.
    llm_spans = [s for s in private_spans if s.name == "openarmature.llm.complete"]
    if disable_llm_spans:
        assert not llm_spans, (
            f"disable_llm_spans=True MUST suppress LLM span emission; got {len(llm_spans)} llm spans"
        )
        # ask_llm node span still emits.
        assert any(s.name == "ask_llm" for s in private_spans)
        return

    # default + external_auto_instrumentation_active — LLM span
    # MUST emit with the spec §5.5 attributes.
    assert len(llm_spans) == 1, (
        f"expected one LLM span; got {len(llm_spans)}: {[s.name for s in private_spans]}"
    )
    llm = llm_spans[0]
    attrs = dict(llm.attributes or {})
    assert attrs.get("openarmature.llm.model") == "test-model"
    if case_name == "default":
        # Sub-case 1 asserts the full attribute set.
        assert attrs.get("openarmature.llm.finish_reason") == "stop"
        assert attrs.get("openarmature.llm.usage.prompt_tokens") == 5
        assert attrs.get("openarmature.llm.usage.completion_tokens") == 1
        assert attrs.get("openarmature.llm.usage.total_tokens") == 6
    # Parent: the ask_llm node span.
    ask_llm = next((s for s in private_spans if s.name == "ask_llm"), None)
    assert ask_llm is not None, "expected ask_llm node span"
    llm_parent = llm.parent
    ask_llm_ctx = ask_llm.context
    assert llm_parent is not None and ask_llm_ctx is not None
    assert llm_parent.span_id == ask_llm_ctx.span_id, (
        "openarmature.llm.complete MUST be parented under the calling node span"
    )
    # Final state was updated by the calls_llm node (msg = "hello"
    # for default; "hi back" for disable_llm_spans path that we
    # already returned from above).
    assert "msg" in dir(final)


def _build_graph_with_mock_llm(case: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    """Build a graph whose entry node invokes ``OpenAIProvider.complete``
    against an ``httpx.MockTransport`` preloaded with the fixture's
    ``mock_llm`` responses."""
    import json

    import httpx

    from openarmature.graph import GraphBuilder
    from openarmature.llm import OpenAIProvider, UserMessage

    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    transport = httpx.MockTransport(_handler)
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test",
        transport=transport,
    )

    # Build a State subclass with the fixture's declared fields.
    from .adapter import build_state_cls

    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    state_cls = build_state_cls("LlmFixtureState", state_fields)

    # Node body: calls the LLM provider and writes the response into
    # the configured field.
    nodes = cast("dict[str, Any]", case["nodes"])
    entry_name = cast("str", case["entry"])
    calls_llm_spec = cast("dict[str, Any]", nodes[entry_name]["calls_llm"])
    stores_in = cast("str", calls_llm_spec.get("stores_response_in", "msg"))
    messages_spec = cast("list[dict[str, str]]", calls_llm_spec.get("messages", []))
    messages: Sequence[Any] = [
        UserMessage(content=m["content"]) for m in messages_spec if m.get("role") == "user"
    ]

    async def ask_llm_body(_s: Any) -> dict[str, str]:
        response = await provider.complete(messages)
        return {stores_in: response.message.content or ""}

    builder = (
        GraphBuilder(state_cls)
        .add_node(entry_name, ask_llm_body)
        .add_edge(entry_name, _resolve_target_for_005(case))
        .set_entry(entry_name)
    )
    return builder.compile(), mock_responses


def _resolve_target_for_005(case: Mapping[str, Any]) -> Any:
    """Fixture 005's edges go to END. Return the END sentinel."""
    from openarmature.graph import END

    edges = cast("list[dict[str, Any]]", case.get("edges") or [])
    if not edges:
        return END
    target = edges[0].get("to")
    return END if target == "END" else target


async def _run_fixture_038(spec: Mapping[str, Any]) -> None:
    """A two-branch parallel-branches fixture where each branch's inner
    ``ask`` node makes an LLM call.

    The OTel observer MUST synthesize a per-branch dispatch span between
    the parallel-branches NODE span and each branch's inner-node spans;
    the attribute surface (``branch_count`` + ``error_policy`` on
    the NODE span, ``branch_name`` + ``parent_node_name`` on each
    dispatch span, ``branch_name`` on inner-branch leaf spans) MUST
    appear; per-branch dispatch spans MUST close before the NODE span
    in branch-declaration order.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    assert len(cases) == 1, f"fixture 038 expects exactly one case; got {len(cases)}"
    case = cases[0]
    case_name = cast("str", case["name"])
    try:
        await _run_fixture_038_case(case)
    except AssertionError as e:
        raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_038_case(case: Mapping[str, Any]) -> None:
    import json

    import httpx
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from openarmature.graph import END, BranchSpec, GraphBuilder
    from openarmature.llm import OpenAIProvider, UserMessage

    from .adapter import build_state_cls

    # ---- Build a queue-backed mock LLM transport.  The fixture's
    # per-branch ``ask`` nodes share a single OpenAIProvider keyed off
    # an httpx MockTransport.  Both branches dispatch concurrently and
    # this fixture asserts span topology + §5.7 attributes only (not
    # response-content routing), so a FIFO mock returning any queued
    # response per call is correct.  Build the queue from the
    # fixture-declared ``calls_llm.response`` values; the per-user-msg
    # bookkeeping below is a side effect of parsing the fixture, not
    # used by the handler.
    branches_spec = cast("dict[str, Any]", case["nodes"]["dispatcher"]["parallel_branches"]["branches"])
    branch_response_by_user_msg: dict[str, str] = {}
    for branch_name, branch_spec in branches_spec.items():
        sub_id = cast("str", branch_spec["subgraph"])
        sub = cast("dict[str, Any]", case["subgraphs"][sub_id])
        ask_calls_llm = cast("dict[str, Any]", sub["nodes"]["ask"]["calls_llm"])
        user_msg = "answer the question"
        if "messages" in ask_calls_llm:
            messages = cast("list[dict[str, str]]", ask_calls_llm["messages"])
            user_msg = next((m["content"] for m in messages if m.get("role") == "user"), user_msg)
        branch_response_by_user_msg[user_msg + f"::{branch_name}"] = cast(
            "str", ask_calls_llm.get("response", f"{branch_name} response")
        )

    fallback_responses = list(branch_response_by_user_msg.values())

    def _handler(_request: httpx.Request) -> httpx.Response:
        # Both branches dispatch concurrently; response-to-branch
        # mapping is non-deterministic by design.  The fixture asserts
        # span topology + §5.7 attributes, NOT response-content
        # routing — so a FIFO mock returning ANY of the queued
        # responses is correct.  Don't add response-content assertions
        # without first switching the mock to a content-routed shape.
        if not fallback_responses:
            raise AssertionError("mock_llm queue exhausted")
        next_response = fallback_responses.pop(0)
        body = {
            "id": "test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": next_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(
            200, content=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )

    transport = httpx.MockTransport(_handler)
    provider = OpenAIProvider(
        base_url="http://mock-llm.test", model="test-model", api_key="test", transport=transport
    )

    # ---- Build the inner subgraphs (one per branch).  Each inner
    # subgraph has a single ``ask`` node that calls the mock provider.
    subgraphs: dict[str, Any] = {}
    for sub_id, sub_spec in cast("dict[str, Any]", case["subgraphs"]).items():
        inner_fields = cast("dict[str, dict[str, Any]]", sub_spec["state"]["fields"])
        inner_state_cls = build_state_cls(f"Inner_{sub_id}", inner_fields)
        ask_calls_llm = cast("dict[str, Any]", sub_spec["nodes"]["ask"]["calls_llm"])
        stores_in = cast("str", ask_calls_llm.get("stores_response_in", "msg"))
        messages_in: tuple[Any, ...] = tuple(
            UserMessage(content=m["content"])
            for m in cast("list[dict[str, str]]", ask_calls_llm.get("messages", []))
            if m.get("role") == "user"
        ) or (UserMessage(content="answer the question"),)

        # Need to bind ``stores_in`` and the messages into the closure
        # for each subgraph independently — default-arg binding is the
        # idiomatic late-binding sidestep for loop-scope closures.
        async def _ask_body(
            _s: Any,
            stores: str = stores_in,
            messages: tuple[Any, ...] = messages_in,
        ) -> dict[str, str]:
            response = await provider.complete(list(messages))
            return {stores: response.message.content or ""}

        subgraphs[sub_id] = (
            GraphBuilder(inner_state_cls)
            .add_node("ask", _ask_body)
            .add_edge("ask", END)
            .set_entry("ask")
            .compile()
        )

    # ---- Build the outer graph with the parallel-branches node.
    outer_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    outer_state_cls = build_state_cls("Outer_038", outer_fields)
    error_policy = cast(
        "str", case["nodes"]["dispatcher"]["parallel_branches"].get("error_policy", "fail_fast")
    )
    branches = {
        branch_name: BranchSpec(subgraph=subgraphs[cast("str", branch_spec["subgraph"])])
        for branch_name, branch_spec in branches_spec.items()
    }
    builder = (
        GraphBuilder(outer_state_cls)
        .add_parallel_branches_node("dispatcher", branches=branches, error_policy=error_policy)  # type: ignore[arg-type]
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
    )
    graph = builder.compile()

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    graph.attach_observer(observer)
    try:
        await graph.invoke(outer_state_cls())
        await graph.drain()
    finally:
        observer.shutdown()
        await provider.aclose()

    spans = exporter.get_finished_spans()

    # ---- span_tree assertions
    expected_tree = cast("list[dict[str, Any]]", case["expected"]["span_tree"])
    # Find the invocation root.
    inv_root = next(
        (s for s in spans if s.name == "openarmature.invocation" and cast("Any", s.parent) is None), None
    )
    assert inv_root is not None, f"invocation root span missing; got {[s.name for s in spans]}"
    _assert_span_tree_matches(spans, [inv_root], expected_tree)

    # ---- Invariants
    invariants = cast("dict[str, Any]", case["expected"].get("invariants") or {})
    dispatch_spans = [
        s
        for s in spans
        if (s.attributes or {}).get("openarmature.parallel_branches.parent_node_name") is not None
    ]
    node_span = next(
        (
            s
            for s in spans
            if s.name == "dispatcher"
            and (s.attributes or {}).get("openarmature.parallel_branches.branch_count") is not None
        ),
        None,
    )
    if invariants.get("same_named_inner_spans_disambiguated_by_dispatch_parent"):
        ask_spans = [s for s in spans if s.name == "ask"]
        assert len(ask_spans) == 2, f"expected 2 inner ask spans; got {len(ask_spans)}"
        dispatch_span_ids = {cast("Any", d.context).span_id for d in dispatch_spans}
        ask_parents = {cast("Any", s.parent).span_id for s in ask_spans if s.parent is not None}
        assert ask_parents.issubset(dispatch_span_ids), (
            "same-named ask spans MUST parent under distinct per-branch dispatch spans"
        )
        assert len(ask_parents) == 2, "each ask span MUST parent under a DIFFERENT dispatch span"
    if invariants.get("dispatch_spans_close_before_node_span"):
        assert node_span is not None
        node_end = node_span.end_time
        for d in dispatch_spans:
            assert d.end_time is not None and node_end is not None and d.end_time <= node_end, (
                f"dispatch span {d.name!r} MUST close before parallel-branches NODE span"
            )
    declaration_order = cast("list[str] | None", invariants.get("dispatch_spans_close_in_declaration_order"))
    if declaration_order is not None:
        dispatch_by_name = {d.name: d for d in dispatch_spans}
        ends = [
            (name, dispatch_by_name[name].end_time) for name in declaration_order if name in dispatch_by_name
        ]
        found = [n for n, _ in ends]
        assert len(ends) == len(declaration_order), (
            f"declaration_order references {declaration_order!r} but only {found} dispatch spans found"
        )
        for (prev_name, prev_end), (name, end) in zip(ends, ends[1:], strict=False):
            assert prev_end is not None and end is not None and prev_end <= end, (
                f"dispatch span {prev_name!r} (end={prev_end}) MUST close before {name!r} (end={end})"
            )


def _assert_span_tree_matches(
    all_spans: Sequence[Any], actual_roots: Sequence[Any], expected_nodes: Sequence[Mapping[str, Any]]
) -> None:
    """Recursive structural match: for every expected node, find a
    matching actual span by name + attribute-subset; recurse on its
    children.  Children are matched as a SET (order independent —
    parallel-branches dispatch order isn't span-emission order)."""
    actual_by_name: dict[str, list[Any]] = {}
    for root in actual_roots:
        actual_by_name.setdefault(root.name, []).append(root)

    for expected in expected_nodes:
        expected_name = cast("str", expected["name"])
        candidates = actual_by_name.get(expected_name, [])
        expected_attrs = cast("dict[str, Any]", expected.get("attributes") or {})

        def _matches(span: Any, eattrs: dict[str, Any] = expected_attrs) -> bool:
            attrs = dict(span.attributes or {})
            return all(attrs.get(k) == v for k, v in eattrs.items())

        matching = [c for c in candidates if _matches(c)]
        assert len(matching) >= 1, (
            f"no span found matching expected node name={expected_name!r} attrs={expected_attrs!r}; "
            f"candidates: {[c.name for c in candidates]}"
        )
        # Ambiguous-match guard: when multiple candidates pass the
        # name + attribute-subset filter, fail loudly rather than
        # silently picking the first one.  Future fixtures with
        # multiple same-named siblings at the same tree level MUST
        # provide a disambiguator attribute in the expected node's
        # ``attributes`` block.
        assert len(matching) == 1, (
            f"ambiguous match for expected node name={expected_name!r} "
            f"attrs={expected_attrs!r}: {len(matching)} candidates pass "
            f"the name + attribute-subset filter.  Add a disambiguating "
            f"attribute to the expected node's ``attributes:`` block."
        )
        matched = matching[0]
        actual_by_name[expected_name] = [c for c in candidates if c is not matched]
        expected_status = cast("str | None", expected.get("status"))
        if expected_status is not None:
            actual_status_name = matched.status.status_code.name
            # OTel's StatusCode default is UNSET — observers set OK
            # explicitly on success ends, but the spec's expected
            # ``status: OK`` semantic is "non-error", so accept
            # UNSET as OK.  ERROR vs OK is the load-bearing
            # distinction.
            if expected_status == "OK" and actual_status_name in {"OK", "UNSET"}:
                pass
            else:
                assert actual_status_name == expected_status, (
                    f"{expected_name!r} status: expected {expected_status!r}, got {actual_status_name!r}"
                )
        expected_children = cast("list[dict[str, Any]] | None", expected.get("children"))
        if expected_children is not None:
            matched_span_id = matched.context.span_id
            actual_children = [
                s for s in all_spans if s.parent is not None and s.parent.span_id == matched_span_id
            ]
            _assert_span_tree_matches(all_spans, actual_children, expected_children)


async def _run_fixture_110(spec: Mapping[str, Any]) -> None:
    # Proposal 0075 callable-branch span shape (observability §5.7): an
    # inline-callable parallel branch renders as ONE per-branch dispatch span
    # keyed by openarmature.node.branch_name with NO inner-node spans; a
    # when-skipped branch emits no span. Bundled OTel observer (default config),
    # as for fixtures 038 / 082.
    for case in cast("list[dict[str, Any]]", spec["cases"]):
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_110_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_110_case(case: Mapping[str, Any]) -> None:
    observer, exporter = _build_observer()
    final = await _run_graph(case, observer)
    observer.shutdown()

    # ---- final_state: the dispatched callable branches applied their
    # updates; the when-skipped branch contributed nothing.
    expected_final = cast("dict[str, Any]", case["expected"].get("final_state") or {})
    for field_name, expected_value in expected_final.items():
        actual = getattr(final, field_name)
        assert actual == expected_value, f"final_state.{field_name}: {actual!r} != {expected_value!r}"

    spans = exporter.get_finished_spans()
    expected_tree = cast("list[dict[str, Any]]", case["expected"]["span_tree"])
    inv_root = next((s for s in spans if s.name == "openarmature.invocation" and s.parent is None), None)
    assert inv_root is not None, f"invocation root span missing; got {[s.name for s in spans]}"
    _assert_span_tree_matches(spans, [inv_root], expected_tree)

    # ---- when-skipped branches emit NO span. The span_tree match is
    # subset-based, so assert the skip explicitly: any declared branch absent
    # from the (dispatched) span_tree must have produced no span.
    def _names(nodes: list[dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for n in nodes:
            out.add(cast("str", n["name"]))
            out |= _names(cast("list[dict[str, Any]]", n.get("children") or []))
        return out

    dispatched = _names(expected_tree)
    nodes = cast("dict[str, Any]", case["nodes"])
    pb_node = next((ns for ns in nodes.values() if "parallel_branches" in ns), None)
    if pb_node is not None:
        declared = set(cast("dict[str, Any]", pb_node["parallel_branches"]["branches"]).keys())
        for branch in declared - dispatched:
            assert [s for s in spans if s.name == branch] == [], (
                f"when-skipped branch {branch!r} MUST emit no span"
            )


async def _run_fixture_008(spec: Mapping[str, Any]) -> None:
    """Two sub-cases: detached subgraph (one Link, two traces, shared
    correlation_id) and detached fan-out (one trace per instance,
    each with a Link from the fan-out node span)."""
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_008_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_detached_case_graph(
    case: Mapping[str, Any], *, expect_raise: bool = False
) -> Sequence[ReadableSpan]:
    """Build + invoke a detached-mode fixture case; return the finished
    spans. Shared by fixture 008 (detached trace mode) and fixture 058
    case 2 (attribution on the detached invocation span).

    ``expect_raise`` swallows the ``invoke()`` exception for the
    error-status case (the detached subgraph raises and propagates); the
    spans are captured regardless.
    """
    # The fixture configures detached subgraphs by the SUBGRAPH'S IDENTITY
    # NAME (the key in ``subgraphs:``), but the OTel observer keys on the
    # WRAPPER NODE'S NAME in the parent graph (graph-engine §6 namespace
    # convention; see the fixture 029 spec note). Translate by looking up
    # the wrapper node that references each detached subgraph identity.
    detached_subgraph_identities = set(cast("list[str]", case.get("detached_subgraphs") or []))
    nodes = cast("dict[str, Any]", case.get("nodes") or {})
    wrapper_names_for_detached: set[str] = set()
    for wrapper_name, node_spec in nodes.items():
        sub_id = cast("dict[str, Any]", node_spec).get("subgraph")
        if isinstance(sub_id, str) and sub_id in detached_subgraph_identities:
            wrapper_names_for_detached.add(wrapper_name)
    detached_subgraphs = frozenset(wrapper_names_for_detached)
    detached_fan_outs = frozenset(cast("list[str]", case.get("detached_fan_outs") or []))

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        detached_subgraphs=detached_subgraphs,
        detached_fan_outs=detached_fan_outs,
    )

    # Patch test-seam directives the adapter doesn't translate
    # (``update_pure_from_state`` etc.) with a benign no-op; these
    # assertions only inspect span structure, not computed values.
    _patch_unsupported_directives(case)

    subgraphs = _compile_subgraphs(case)
    trace_log: list[str] = []
    built = build_graph(case, subgraphs=subgraphs, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))
    try:
        await compiled.invoke(initial_state)
    except Exception:
        if not expect_raise:
            raise
    await compiled.drain()
    observer.shutdown()
    return exporter.get_finished_spans()


def _invocation_id_of(span: Any) -> Any:
    """The ``openarmature.invocation_id`` attribute off a span (or None)."""
    return dict(span.attributes or {}).get("openarmature.invocation_id")


def _has_exception_event(span: Any) -> bool:
    """Whether the span recorded an OTel exception event."""
    events = cast("list[Any]", span.events or [])
    return any(getattr(e, "name", None) == "exception" for e in events)


async def _run_fixture_008_case(case: Mapping[str, Any]) -> None:
    case_name = case["name"]
    expect_raise = case_name == "detached_subgraph_raises_error_status_on_both_spans"
    spans = await _run_detached_case_graph(case, expect_raise=expect_raise)

    if case_name == "detached_subgraph_two_traces_one_link":
        # Group by trace_id. Span context is non-None for any span
        # the SDK actually exported, so the cast keeps pyright quiet
        # on the `.trace_id` access.
        by_trace: dict[int, list[Any]] = {}
        for s in spans:
            ctx = cast("Any", s.context)
            by_trace.setdefault(ctx.trace_id, []).append(s)
        assert len(by_trace) == 2, (
            f"expected 2 distinct traces (parent + detached subgraph); "
            f"got {len(by_trace)}: {[s.name for s in spans]}"
        )
        # Cross-trace correlation_id consistency (§3).
        cids = _all_correlation_ids(spans)
        assert len(cids) == 1, f"correlation_id MUST flow unchanged across detached boundary; got {cids}"
        # Find the parent dispatch span (it's the one with a Link).
        dispatch_spans = [s for s in spans if s.name == "dispatch"]
        # Two "dispatch" spans: one in parent trace (with Link), one in
        # detached trace (under the detached invocation span). Pick the
        # one with links.
        parent_dispatch = next((s for s in dispatch_spans if s.links), None)
        assert parent_dispatch is not None, "expected a 'dispatch' span carrying a Link to the detached trace"
        assert len(parent_dispatch.links) == 1, (
            f"dispatch span MUST carry exactly one Link; got {len(parent_dispatch.links)}"
        )
        link_target_trace_id = parent_dispatch.links[0].context.trace_id
        # The link's trace_id matches the detached trace's actual trace_id.
        parent_trace_id = cast("Any", parent_dispatch.context).trace_id
        detached_dispatch = next(
            (s for s in dispatch_spans if not s.links and cast("Any", s.context).trace_id != parent_trace_id),
            None,
        )
        assert detached_dispatch is not None
        detached_trace_id = cast("Any", detached_dispatch.context).trace_id
        assert link_target_trace_id == detached_trace_id, (
            f"Link target trace_id MUST match the detached trace's trace_id; "
            f"got link={link_target_trace_id!r}, detached={detached_trace_id!r}"
        )
        # Proposal 0061: the detached trace roots in its OWN
        # openarmature.invocation span sharing the parent's invocation_id
        # (invariant detached_invocation_id_equals_parent).
        inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
        assert len(inv_spans) == 2, f"expected parent + detached invocation spans; got {len(inv_spans)}"
        parent_inv = next((s for s in inv_spans if cast("Any", s.context).trace_id == parent_trace_id), None)
        detached_inv = next(
            (s for s in inv_spans if cast("Any", s.context).trace_id == detached_trace_id), None
        )
        assert detached_inv is not None, "detached trace MUST root in an openarmature.invocation span"
        assert parent_inv is not None
        parent_iid = _invocation_id_of(parent_inv)
        assert parent_iid is not None and _invocation_id_of(detached_inv) == parent_iid, (
            "detached invocation span MUST share the parent's invocation_id (§4.3)"
        )
        return

    if case_name == "detached_fan_out_one_trace_per_instance":
        # Group by trace_id.
        by_trace = {}
        for s in spans:
            ctx = cast("Any", s.context)
            by_trace.setdefault(ctx.trace_id, []).append(s)
        # 4 traces total: 1 parent + 3 instance traces.
        assert len(by_trace) == 4, f"expected 4 traces (parent + 3 instances); got {len(by_trace)}"
        # All 4 share the same correlation_id.
        cids = _all_correlation_ids(spans)
        assert len(cids) == 1, (
            f"correlation_id MUST be uniform across parent + detached instance traces; got {cids}"
        )
        # Find the fan-out node span — it's in the parent trace and
        # carries 3 Links.
        fan_out_node_spans = [s for s in spans if s.name == "per_document_scoring"]
        # Three of these are inside detached instance roots; one is in
        # the parent trace and is the one with Links.
        parent_fan_out = next((s for s in fan_out_node_spans if s.links), None)
        assert parent_fan_out is not None, "expected a fan-out span with Links in parent trace"
        assert len(parent_fan_out.links) == 3, (
            f"fan-out span MUST carry one Link per instance (3); got {len(parent_fan_out.links)}"
        )
        # Proposal 0061: each instance trace roots in its OWN
        # openarmature.invocation span sharing the parent's invocation_id.
        parent_trace_id = cast("Any", parent_fan_out.context).trace_id
        inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
        assert len(inv_spans) == 4, f"expected parent + 3 instance invocation spans; got {len(inv_spans)}"
        parent_inv = next((s for s in inv_spans if cast("Any", s.context).trace_id == parent_trace_id), None)
        assert parent_inv is not None
        parent_iid = _invocation_id_of(parent_inv)
        instance_invs = [s for s in inv_spans if cast("Any", s.context).trace_id != parent_trace_id]
        assert len(instance_invs) == 3, (
            f"expected 3 detached instance invocation spans; got {len(instance_invs)}"
        )
        for inv in instance_invs:
            assert parent_iid is not None and _invocation_id_of(inv) == parent_iid, (
                "each detached instance invocation span MUST share the parent's invocation_id"
            )
        return

    if case_name == "detached_subgraph_raises_error_status_on_both_spans":
        # Proposal 0061 §4.2: a raising detached subgraph surfaces ERROR
        # on BOTH the parent's dispatch span and the detached invocation
        # span — distinct traces, shared invocation_id, each with the §4
        # category + an OTel exception event.
        dispatch_spans = [s for s in spans if s.name == "dispatch"]
        parent_dispatch = next((s for s in dispatch_spans if s.links), None)
        assert parent_dispatch is not None, "expected a parent 'dispatch' span with a Link"
        assert parent_dispatch.status.status_code.name == "ERROR", (
            "parent dispatch span MUST carry ERROR for a raising detached subgraph"
        )
        assert _has_exception_event(parent_dispatch), "parent dispatch span MUST record the exception"
        assert dict(parent_dispatch.attributes or {}).get("openarmature.error.category") == "node_exception"
        parent_trace_id = cast("Any", parent_dispatch.context).trace_id
        detached_trace_id = parent_dispatch.links[0].context.trace_id
        assert detached_trace_id != parent_trace_id, "detached + parent traces MUST be distinct"
        inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
        detached_inv = next(
            (s for s in inv_spans if cast("Any", s.context).trace_id == detached_trace_id), None
        )
        parent_inv = next((s for s in inv_spans if cast("Any", s.context).trace_id == parent_trace_id), None)
        assert detached_inv is not None, "detached trace MUST root in an openarmature.invocation span"
        assert parent_inv is not None
        assert detached_inv.status.status_code.name == "ERROR", (
            "detached invocation span MUST carry the detached unit's ERROR status (§4.2)"
        )
        assert _has_exception_event(detached_inv), "detached invocation span MUST record the exception"
        assert dict(detached_inv.attributes or {}).get("openarmature.error.category") == "node_exception"
        parent_iid = _invocation_id_of(parent_inv)
        assert parent_iid is not None and _invocation_id_of(detached_inv) == parent_iid, (
            "detached invocation span MUST share the parent's invocation_id"
        )
        return

    raise AssertionError(f"unknown sub-case {case_name!r}")


def _assert_attribution_on_invocation_only(spans: Any) -> None:
    """Assert every openarmature.invocation span carries non-empty
    implementation.name (the canonical python value) and
    implementation.version, and that no inner span carries either."""
    # Proposal 0052 §5.1: the attribution attributes are invocation-span-
    # only (not §5.6 cross-cutting), so inner spans MUST NOT carry them.
    inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
    assert inv_spans, "expected at least one openarmature.invocation span"
    for inv in inv_spans:
        iattrs = dict(inv.attributes or {})
        name = iattrs.get("openarmature.implementation.name")
        version = iattrs.get("openarmature.implementation.version")
        assert name == "openarmature-python", (
            f"implementation.name MUST equal the canonical python value; got {name!r}"
        )
        assert isinstance(version, str) and len(version) > 0, (
            "implementation.version MUST be a non-empty string"
        )
    for s in spans:
        if s.name == "openarmature.invocation":
            continue
        sattrs = dict(s.attributes or {})
        assert "openarmature.implementation.name" not in sattrs, (
            f"inner span {s.name!r} MUST NOT carry implementation.name"
        )
        assert "openarmature.implementation.version" not in sattrs, (
            f"inner span {s.name!r} MUST NOT carry implementation.version"
        )


async def _run_fixture_058(spec: Mapping[str, Any]) -> None:
    """Implementation-attribution attributes. Case 1: present on the
    invocation span, absent on inner spans. Case 2: the attribution
    lands on the detached trace's OWN openarmature.invocation span, with
    the subgraph-wrapper span nested under it."""
    # Case 1 covers proposal 0052 (§5.1 attribution); case 2 additionally
    # depends on proposal 0061 (the detached trace gains its own
    # invocation-span root for the attributes to land on).
    for case in cast("list[dict[str, Any]]", spec["cases"]):
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_058_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_058_case(case: Mapping[str, Any]) -> None:
    case_name = case["name"]
    if case_name == "implementation_attribution_attributes_present_on_invocation_span":
        observer, exporter = _build_observer()
        await _run_graph(case, observer)
        observer.shutdown()
        _assert_attribution_on_invocation_only(exporter.get_finished_spans())
        return

    if case_name == "detached_subgraph_attribution_propagates_to_child_trace_invocation_span":
        spans = await _run_detached_case_graph(case)
        # Two invocation spans (parent + detached); §5.1 attribution on
        # both, absent on every inner span across both traces.
        inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
        assert len(inv_spans) == 2, f"expected parent + detached invocation spans; got {len(inv_spans)}"
        _assert_attribution_on_invocation_only(spans)
        # Proposal 0061: the detached subgraph-wrapper span nests BETWEEN
        # the detached invocation span and the inner node (058 case 2
        # added the previously-missing wrapper layer).
        wrapper = next(
            (s for s in spans if dict(s.attributes or {}).get("openarmature.subgraph.detached") is True),
            None,
        )
        assert wrapper is not None, "expected a detached subgraph-wrapper span"
        detached_trace_id = cast("Any", wrapper.context).trace_id
        detached_inv = next(
            (s for s in inv_spans if cast("Any", s.context).trace_id == detached_trace_id), None
        )
        assert detached_inv is not None, "detached trace MUST root in an openarmature.invocation span"
        assert (
            wrapper.parent is not None and wrapper.parent.span_id == cast("Any", detached_inv.context).span_id
        ), "detached subgraph-wrapper span MUST nest under the detached invocation span"
        inner = [
            s
            for s in spans
            if s.parent is not None and s.parent.span_id == cast("Any", wrapper.context).span_id
        ]
        assert inner, "the inner node MUST nest under the subgraph-wrapper span (no skipped layer)"
        return

    raise AssertionError(f"unknown sub-case {case_name!r}")


def _patch_unsupported_directives(spec: Mapping[str, Any]) -> None:
    """Replace test-seam directives the conformance adapter doesn't
    yet translate (``update_pure_from_state`` etc.) with a benign
    ``update_pure: {}`` no-op. The observability fixtures only
    assert span structure (parent-child, Links, trace_ids,
    correlation_id), not state values, so the swap is safe."""

    def patch_nodes(graph_block: Mapping[str, Any] | None) -> None:
        if not graph_block:
            return
        nodes = cast("dict[str, Any]", graph_block.get("nodes") or {})
        for node_spec_any in nodes.values():
            if not isinstance(node_spec_any, dict):
                continue
            node_spec = cast("dict[str, Any]", node_spec_any)
            for unsupported in (
                "update_pure_from_state",
                "calls_llm",
            ):
                if unsupported in node_spec:
                    node_spec.pop(unsupported)
                    node_spec.setdefault("update_pure", {})

    patch_nodes(spec)
    if "subgraph" in spec:
        patch_nodes(cast("Mapping[str, Any]", spec["subgraph"]))
    for sub in cast("dict[str, Any]", spec.get("subgraphs") or {}).values():
        patch_nodes(cast("Mapping[str, Any]", sub))


def _compile_subgraphs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build any subgraphs declared by the fixture and return a
    name→compiled-graph registry the adapter consumes."""
    subgraph_specs: dict[str, Any] = {}
    if "subgraph" in spec:
        single = cast("Mapping[str, Any]", spec["subgraph"])
        name = single.get("name") or "subgraph"
        subgraph_specs[name] = single
    if "subgraphs" in spec:
        for k, v in cast("dict[str, Any]", spec["subgraphs"]).items():
            subgraph_specs[k] = v
    compiled_subgraphs: dict[str, Any] = {}
    for name, sub_spec in subgraph_specs.items():
        sub_built = build_graph(sub_spec, trace=[])
        compiled_subgraphs[name] = sub_built.builder.compile()
    return compiled_subgraphs


# ---------------------------------------------------------------------------
# Fixture 031 — span/log assertions deferred
#
# Lives in this file (not test_checkpoint.py) because the assertions
# verify OTel span attributes across the original + resumed runs of
# the same checkpoint fixture. The checkpoint harness already covers the
# record-level half (correlation_id preserved, invocation_id changes);
# this picks up the cross-run span-attribute half.
# ---------------------------------------------------------------------------


_PIPELINE_CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "pipeline-utilities" / "conformance"
)


async def test_phase5_fixture_031_span_assertions() -> None:
    """Every span across BOTH the original and resumed runs MUST carry
    the same ``openarmature.correlation_id``; ``invocation_id`` differs
    across the two runs (each is its own invocation in the
    observability sense)."""
    fixture_path = _PIPELINE_CONFORMANCE_DIR / "031-checkpoint-correlation-id-preserved-across-resume.yaml"
    spec = _load(fixture_path)
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_031_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_031_case(case: Mapping[str, Any]) -> None:
    from openarmature.checkpoint import CheckpointRecord, InMemoryCheckpointer
    from openarmature.checkpoint.protocol import Checkpointer
    from openarmature.graph import RuntimeGraphError

    class _CapturingCheckpointer:
        """Mirrors the local capture pattern in test_checkpoint.py
        but inlined here so the observability test doesn't depend on
        that file's internal helpers."""

        def __init__(self) -> None:
            self._inner = InMemoryCheckpointer()
            self.saves: list[CheckpointRecord] = []

        async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
            self.saves.append(record)
            await self._inner.save(invocation_id, record)

        async def load(self, invocation_id: str) -> CheckpointRecord | None:
            return await self._inner.load(invocation_id)

        async def list(self, filter: Any = None) -> Any:
            return await self._inner.list(filter)

        async def delete(self, invocation_id: str) -> None:
            await self._inner.delete(invocation_id)

    capturing = _CapturingCheckpointer()
    observer, exporter = _build_observer()

    trace: list[str] = []
    built = build_graph(case, trace=trace)
    builder = built.builder
    builder.with_checkpointer(cast("Checkpointer", capturing))
    compiled = builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))

    # First run — expected to abort.
    expected_error = cast("Mapping[str, Any]", case["first_run_expected_error"])
    caller_cid = case.get("caller_correlation_id")
    with pytest.raises(RuntimeGraphError) as excinfo:
        await compiled.invoke(initial_state, correlation_id=caller_cid)
    assert excinfo.value.category == expected_error["category"]
    await compiled.drain()

    # Capture first run's invocation_id from the latest save.
    assert capturing.saves, "expected at least one save before the abort"
    first_invocation_id = capturing.saves[-1].invocation_id
    first_correlation_id = capturing.saves[-1].correlation_id
    if caller_cid is not None:
        assert first_correlation_id == caller_cid, (
            f"first run MUST preserve caller-supplied correlation_id; "
            f"got {first_correlation_id!r}, expected {caller_cid!r}"
        )
    else:
        # Auto-generated → UUIDv4 form.
        assert _UUIDV4_RE.match(first_correlation_id), (
            f"auto-generated correlation_id MUST be UUIDv4; got {first_correlation_id!r}"
        )

    # Resume — should succeed.
    capturing.saves.clear()
    await compiled.invoke(initial_state, resume_invocation=first_invocation_id)
    await compiled.drain()
    observer.shutdown()

    # ----- Span assertions (the §10.4 / §3 invariants) -----
    spans = exporter.get_finished_spans()
    # Every span across both runs MUST carry the same correlation_id.
    cids = _all_correlation_ids(spans)
    assert len(cids) == 1, f"correlation_id MUST be uniform across both runs; got {cids}"
    cid = next(iter(cids))
    assert cid == first_correlation_id, (
        f"resumed run MUST preserve original correlation_id; got {cid!r}, original {first_correlation_id!r}"
    )
    # The original and resumed runs MUST have DIFFERENT trace_ids
    # (each invocation is its own trace per §5.1's
    # ``invocation_id`` semantics — different invocation_id ↔
    # different OTel trace_id under the default in-trace
    # parent-child rules).
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 2, (
        f"original and resumed runs MUST produce DIFFERENT trace_ids "
        f"(per §10.4 step 4 + §5.1); got {len(trace_ids)} distinct trace_ids"
    )


# ---------------------------------------------------------------------------
# Fixture 084 — Langfuse session/user promotion (proposal 0064)
# ---------------------------------------------------------------------------


async def _run_fixture_084(spec: Mapping[str, Any]) -> None:
    from openarmature.observability.langfuse import (  # noqa: PLC0415
        InMemoryLangfuseClient,
        LangfuseObserver,
    )

    # Proposal 0064 §8.4.1. Cases 1 + 5 are session-bound: they supply
    # session_id at invoke(), which needs the sessions capability
    # (proposal 0020, §5.6) to surface openarmature.session_id. That is
    # unimplemented in python until v0.19.0, so trace.sessionId has no
    # source and these cases defer (per-case continue). Cases 2/3/4 (not
    # session-bound + the userId promotion) run now.
    _deferred_cases = {
        "session_bound_sets_trace_session_id",
        "multi_invocation_shared_session_groups",
    }
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        if case_name in _deferred_cases:
            continue
        try:
            client = InMemoryLangfuseClient()
            observer = LangfuseObserver(client=client)
            trace: list[str] = []
            built = build_graph(case, trace=trace)
            compiled = built.builder.compile()
            compiled.attach_observer(observer)
            initial_state = built.initial_state(case.get("initial_state", {}))
            caller_metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
            if caller_metadata is not None:
                await compiled.invoke(initial_state, metadata=caller_metadata)
            else:
                await compiled.invoke(initial_state)
            await compiled.drain()
            observer.shutdown()

            assert len(client.traces) == 1, f"expected 1 trace, got {len(client.traces)}"
            lf_trace = next(iter(client.traces.values()))
            expected = cast("dict[str, Any]", case["expected"]["langfuse_trace"])
            # trace.sessionId is unset for the runnable cases (no session
            # source until 0020).
            assert lf_trace.session_id == expected.get("sessionId"), (
                f"sessionId: got {lf_trace.session_id!r}, expected {expected.get('sessionId')!r}"
            )
            # trace.userId: promoted from the userId caller key (case 3),
            # unset otherwise (cases 2/4).
            assert lf_trace.user_id == expected.get("userId"), (
                f"userId: got {lf_trace.user_id!r}, expected {expected.get('userId')!r}"
            )
            # Additive promotion + unaffected metadata: every concrete
            # (non-placeholder) expected metadata key also lands top-level.
            expected_md = cast("dict[str, Any]", expected.get("metadata") or {})
            for key, val in expected_md.items():
                if isinstance(val, str) and val.startswith("<") and val.endswith(">"):
                    continue
                assert lf_trace.metadata.get(key) == val, (
                    f"metadata.{key}: got {lf_trace.metadata.get(key)!r}, expected {val!r}"
                )
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


_LANGFUSE_MATCHER_SUBKEYS = frozenset({"harness_parameterized", "non_empty_string"})


def _langfuse_value_matches(
    actual: Any,
    expected: Any,
    *,
    bindings: dict[str, Any],
    params: Mapping[str, Any],
) -> bool:
    """Match a Langfuse trace/observation value against a fixture expectation:
    an inline placeholder token, the assertion sub-key dict, or plain equality.
    """
    # The value-matcher idioms are the conformance-adapter §5.10 vocabulary.
    if isinstance(expected, str) and expected.startswith("<") and expected.endswith(">"):
        return _langfuse_placeholder_matches(actual, expected, bindings)
    # A NON-empty mapping whose keys are all matcher sub-keys is an assertion
    # dict; an empty dict (or a dict with other keys) is matched by equality.
    if (
        isinstance(expected, Mapping)
        and expected
        and set(cast("Mapping[str, Any]", expected)).issubset(_LANGFUSE_MATCHER_SUBKEYS)
    ):
        return _langfuse_matcher_subkeys_match(actual, cast("Mapping[str, Any]", expected), params)
    return bool(actual == expected)


def _langfuse_placeholder_matches(actual: Any, token: str, bindings: dict[str, Any]) -> bool:
    """Inline placeholder tokens: ``<any-string>`` (non-empty), ``<uuid-hex>``
    (32-hex dashes-stripped), and first-occurrence binding tokens like
    ``<corr_id_1>`` (bind on first sighting, assert equality after -- the
    correlation-id-consistency check). The §5.10 ``<uuid>`` (canonical) token
    is added when a wired fixture first needs it.
    """
    if token == "<any-string>":
        return isinstance(actual, str) and actual != ""
    if token == "<uuid-hex>":
        return isinstance(actual, str) and re.fullmatch(r"[0-9a-f]{32}", actual) is not None
    if token in bindings:
        return actual == bindings[token]
    if actual is None:
        return False
    bindings[token] = actual
    return True


def _langfuse_matcher_subkeys_match(actual: Any, spec: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """Assertion sub-keys (059): ``non_empty_string`` and ``harness_parameterized``
    (value equals the named harness-injected parameter)."""
    if spec.get("non_empty_string") is True and not (isinstance(actual, str) and actual != ""):
        return False
    if "harness_parameterized" in spec:
        param_name = cast("str", spec["harness_parameterized"])
        if actual != params.get(param_name):
            return False
    return True


def _assert_langfuse_trace_shape(
    trace: Any,
    expected: Mapping[str, Any],
    *,
    bindings: dict[str, Any],
    params: Mapping[str, Any],
) -> None:
    """Assert a Langfuse Trace's id / name / metadata / observation tree against
    the fixture's ``expected.langfuse_trace`` block. Each clause is asserted only
    when present (059 asserts metadata only; 022/031/032 assert all four).
    """
    if "id" in expected:
        # python's in-memory LangfuseTrace.id is the RAW invocation_id (the
        # §8.4.1 verbatim OA-side id); the fixture asserts the DERIVED Langfuse
        # trace id (uuid-hex / sha256[:16]). Bridge via langfuse_trace_id, the
        # impl's own derivation rule (trace_id.py).
        from openarmature.observability.langfuse import langfuse_trace_id

        derived_id = langfuse_trace_id(trace.id)
        assert _langfuse_value_matches(derived_id, expected["id"], bindings=bindings, params=params), (
            f"derived trace.id {derived_id!r} (from raw {trace.id!r}) did not match {expected['id']!r}"
        )
    if "name" in expected:
        assert _langfuse_value_matches(trace.name, expected["name"], bindings=bindings, params=params), (
            f"trace.name {trace.name!r} did not match {expected['name']!r}"
        )
    for key, val in cast("dict[str, Any]", expected.get("metadata") or {}).items():
        assert _langfuse_value_matches(trace.metadata.get(key), val, bindings=bindings, params=params), (
            f"trace.metadata.{key} {trace.metadata.get(key)!r} did not match {val!r}"
        )
    observations = cast("list[dict[str, Any]] | None", expected.get("observations"))
    if observations is not None:
        _assert_langfuse_observation_tree(trace, observations, bindings=bindings, params=params)


async def _run_langfuse_trace_fixture(spec: Mapping[str, Any]) -> None:
    """Driver for the trace-shape Langfuse fixtures: 022/031/032 (single-dict)
    and 059 (cases). Each builds a graph via the adapter, records into an
    InMemoryLangfuseClient, and asserts the Trace + observation tree.
    """
    if "cases" in spec:
        for case in cast("list[dict[str, Any]]", spec["cases"]):
            case_name = cast("str", case["name"])
            try:
                await _run_langfuse_trace_case(case)
            except AssertionError as e:
                raise AssertionError(f"case {case_name!r}: {e}") from e
    else:
        await _run_langfuse_trace_case(spec)


async def _run_langfuse_trace_case(case: Mapping[str, Any]) -> None:
    import openarmature
    from openarmature.observability.langfuse import InMemoryLangfuseClient, LangfuseObserver

    _patch_unsupported_directives(case)
    client = InMemoryLangfuseClient()
    lf_kwargs: dict[str, Any] = {"client": client}
    cfg = cast("dict[str, Any]", case.get("langfuse_observer_config") or case.get("langfuse_observer") or {})
    if "disable_state_payload" in cfg:
        lf_kwargs["disable_state_payload"] = bool(cfg["disable_state_payload"])
    if "disable_provider_payload" in cfg:
        lf_kwargs["disable_provider_payload"] = bool(cfg["disable_provider_payload"])
    observer = LangfuseObserver(**lf_kwargs)

    subgraphs = _compile_subgraphs(case)
    built = build_graph(case, subgraphs=dict(subgraphs), trace=[])
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))
    await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()

    assert len(client.traces) == 1, f"expected 1 Langfuse trace; got {len(client.traces)}"
    trace = next(iter(client.traces.values()))
    bindings: dict[str, Any] = {}
    params = {"implementation_name": openarmature.__implementation_name__}
    expected = cast("dict[str, Any]", case["expected"]["langfuse_trace"])
    _assert_langfuse_trace_shape(trace, expected, bindings=bindings, params=params)


async def _run_invocation_id_fixture(spec: Mapping[str, Any]) -> None:
    """Driver for the caller-invocation-id fixtures (035/036). Builds a simple
    calls_llm graph, invokes with ``invocation_id=caller_invocation_id``, and
    asserts the Langfuse ``trace.id`` equals the fixture's pinned derivation
    (python derives it; the harness checks the result) plus 036's raw id in
    ``trace.metadata``.
    """
    for case in cast("list[dict[str, Any]]", spec["cases"]):
        case_name = cast("str", case["name"])
        try:
            await _run_invocation_id_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_invocation_id_case(case: Mapping[str, Any]) -> None:
    from openarmature.observability.langfuse import (
        InMemoryLangfuseClient,
        LangfuseObserver,
        langfuse_trace_id,
    )

    graph, state_cls, provider = _build_simple_llm_graph(case, populate_caller_metadata=False)
    client = InMemoryLangfuseClient()
    graph.attach_observer(LangfuseObserver(client=client))
    state = _make_state_instance(case, state_cls)
    caller_id = cast("str", case["caller_invocation_id"])
    try:
        await graph.invoke(state, invocation_id=caller_id)
        await graph.drain()
    finally:
        await provider.aclose()

    assert len(client.traces) == 1, f"expected 1 Langfuse trace; got {len(client.traces)}"
    trace = next(iter(client.traces.values()))
    expected_trace = cast("dict[str, Any]", case["expected"]["langfuse_trace"])
    # The fixture's trace.id is the DERIVED Langfuse id; the in-memory recorder
    # keys by the raw invocation_id. Bridge via the impl's langfuse_trace_id.
    derived_id = langfuse_trace_id(trace.id)
    assert derived_id == expected_trace["id"], (
        f"derived trace.id {derived_id!r} (from raw {trace.id!r}) != {expected_trace['id']!r}"
    )
    for key, val in cast("dict[str, Any]", expected_trace.get("metadata") or {}).items():
        actual = trace.metadata.get(key)
        # The real SDK derives trace.id and preserves the raw invocation_id in
        # metadata for reverse lookup; the in-memory recorder instead keeps the
        # raw id AS trace.id. Recover it from there when metadata omits it (036).
        if actual is None and key == "invocation_id":
            actual = trace.id
        assert actual == val, f"trace.metadata.{key} {actual!r} != {val!r}"


# ---------------------------------------------------------------------------
# Fixture 010 — log correlation
#
# Two sub-cases. Both build the graph by hand rather than going through the
# adapter — fixture 010's ``emits_log:`` directive isn't an adapter primitive
# (the adapter recognizes ``update_pure``, ``subgraph``, etc., and silently
# ignores anything else), and the sub-cases are small enough that hand-built
# python is clearer than threading a new directive through the adapter.
# ---------------------------------------------------------------------------


def _setup_isolated_log_bridge() -> tuple[Any, Any, Any]:
    """Spin up an OTel ``LoggerProvider`` + ``InMemoryLogRecordExporter`` and
    install the log bridge against the root logger, snapshotting the prior
    log state so the caller can restore it in ``finally`` (the bridge mutates
    process-global ``logging`` state — handlers, factory).

    Returns ``(exporter, provider, restore_state)`` where ``restore_state``
    is a snapshot to pass to :func:`_restore_log_state`.
    """
    import logging as _logging  # noqa: PLC0415

    from opentelemetry.sdk._logs import LoggerProvider  # noqa: PLC0415
    from opentelemetry.sdk._logs.export import (  # noqa: PLC0415
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )

    from openarmature.observability.otel import install_log_bridge  # noqa: PLC0415

    root = _logging.getLogger()
    snapshot = (list(root.handlers), list(root.filters), _logging.getLogRecordFactory())

    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    install_log_bridge(provider)
    return exporter, provider, snapshot


def _restore_log_state(snapshot: Any) -> None:
    """Pair to :func:`_setup_isolated_log_bridge` — restores the root logger's
    handler list, filters, and ``LogRecord`` factory to the snapshot taken
    before ``install_log_bridge`` ran."""
    import logging as _logging  # noqa: PLC0415

    handlers, filters, factory = snapshot
    root = _logging.getLogger()
    root.handlers[:] = handlers
    root.filters[:] = filters
    _logging.setLogRecordFactory(factory)


def _enable_test_logger_at_info() -> tuple[Any, int]:
    """Bring the fixture-010 test logger up to ``INFO`` so YAML's
    ``level: INFO`` records actually flow through Python's logger-level
    filter to the bridge handler. Returns ``(logger, prior_level)`` to
    pair with a restore in ``finally``."""
    import logging as _logging  # noqa: PLC0415

    test_logger = _logging.getLogger("openarmature.test.fixture_010")
    prior_level = test_logger.level
    test_logger.setLevel(_logging.INFO)
    return test_logger, prior_level


async def _run_fixture_010(spec: Mapping[str, Any]) -> None:
    """Two sub-cases: nested-trace log correlation (single graph, all logs
    share the parent trace_id) and detached-subgraph log correlation
    (logs across the detached boundary carry distinct trace_ids but the
    same correlation_id)."""
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_010_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_010_case(case: Mapping[str, Any]) -> None:
    case_name = cast("str", case["name"])
    if case_name == "log_records_carry_trace_span_correlation_ids":
        await _run_fixture_010_nested_trace(case)
    elif case_name == "detached_subgraph_log_uses_detached_trace_id_keeps_correlation_id":
        await _run_fixture_010_detached(case)
    else:
        raise AssertionError(f"unknown fixture 010 sub-case: {case_name!r}")


async def _run_fixture_010_nested_trace(case: Mapping[str, Any]) -> None:
    """Sub-case 1: 2 nodes ``a`` → ``b``, both emit logs from the FIRST line
    of their body. The log bridge MUST report all logs in the parent
    trace_id, with each log's span_id matching the active node span at
    emission, and all carrying the invocation's correlation_id."""
    from openarmature.graph import END, GraphBuilder, State  # noqa: PLC0415

    nodes_spec = cast("dict[str, Any]", case["nodes"])
    correlation_id = cast("str", case["caller_correlation_id"])
    # Spec YAML is the single source of truth for the log bodies; derive
    # them up front rather than hard-coding so a fixture rename doesn't
    # silently break the driver's record filtering.
    node_emit_messages: dict[str, str] = {
        name: cast("str", cast("dict[str, Any]", nodes_spec[name])["emits_log"]["message"])
        for name in nodes_spec
    }

    class _S(State):
        x: int = 0

    test_logger, prior_level = _enable_test_logger_at_info()

    def _make_body(node_name: str) -> Any:
        spec = cast("dict[str, Any]", nodes_spec[node_name])
        emit_msg = cast("str", spec["emits_log"]["message"])
        update = cast("dict[str, Any]", spec["update_pure"])

        async def body(_s: _S) -> dict[str, Any]:
            # FIRST line, before any await — the load-bearing case
            # the engine attach via ``prepare_sync`` exists to cover.
            test_logger.info(emit_msg)
            return dict(update)

        return body

    builder = GraphBuilder(_S)
    for node_name in nodes_spec:
        builder.add_node(node_name, _make_body(node_name))
    for edge in cast("list[dict[str, Any]]", case["edges"]):
        from_node = cast("str", edge["from"])
        to = edge["to"]
        builder.add_edge(from_node, END if to == "END" else cast("str", to))
    builder.set_entry(cast("str", case["entry"]))
    compiled = builder.compile()

    observer, span_exporter = _build_observer()
    log_exporter, log_provider, snapshot = _setup_isolated_log_bridge()
    try:
        compiled.attach_observer(observer)
        await compiled.invoke(_S(), correlation_id=correlation_id)
        await compiled.drain()
        observer.shutdown()
        log_provider.force_flush()

        records = log_exporter.get_finished_logs()
        # Filter to OUR test loggers so concurrent test setup noise
        # doesn't contaminate the assertions. Expected message set
        # comes from the spec YAML, not hard-coded strings.
        expected_messages = set(node_emit_messages.values())
        ours = [r for r in records if str(r.log_record.body) in expected_messages]
        assert len(ours) == 2, (
            f"expected 2 log records (one per node body); got {len(ours)}: "
            f"{[str(r.log_record.body) for r in ours]}"
        )

        # Group by body for predictable lookup, indexing by the spec's
        # emit-message values.
        by_body = {str(r.log_record.body): r for r in ours}
        a_log = by_body[node_emit_messages["a"]]
        b_log = by_body[node_emit_messages["b"]]

        # Invariant: all_logs_same_trace_id.
        trace_ids = {a_log.log_record.trace_id, b_log.log_record.trace_id}
        assert len(trace_ids) == 1, f"all logs MUST share a trace_id (single nested trace); got {trace_ids}"

        # Invariant: log_span_ids_match_active_span_at_emission.
        spans = span_exporter.get_finished_spans()
        node_span_ids: dict[str, int] = {}
        for s in spans:
            if s.name in {"a", "b"}:
                node_span_ids[s.name] = s.context.span_id
        assert a_log.log_record.span_id == node_span_ids["a"], (
            f"node-a log MUST carry node-a span's span_id; "
            f"got log span_id={a_log.log_record.span_id}, span={node_span_ids['a']}"
        )
        assert b_log.log_record.span_id == node_span_ids["b"], (
            f"node-b log MUST carry node-b span's span_id; "
            f"got log span_id={b_log.log_record.span_id}, span={node_span_ids['b']}"
        )

        # Invariant: all_logs_carry_correlation_id.
        for r in ours:
            attrs = dict(r.log_record.attributes or {})
            assert attrs.get("openarmature.correlation_id") == correlation_id, (
                f"every log MUST carry openarmature.correlation_id={correlation_id!r}; "
                f"got {attrs.get('openarmature.correlation_id')!r}"
            )
    finally:
        test_logger.setLevel(prior_level)
        _restore_log_state(snapshot)


async def _run_fixture_010_detached(case: Mapping[str, Any]) -> None:
    """Sub-case 2: outer invocation has a detached subgraph. Logs emitted
    inside the detached subgraph carry the DETACHED trace's trace_id —
    NOT the parent's — while the correlation_id flows unchanged across
    the boundary."""
    from openarmature.graph import END, GraphBuilder, State  # noqa: PLC0415

    correlation_id = cast("str", case["caller_correlation_id"])
    sub_specs = cast("dict[str, Any]", case["subgraphs"])
    inner_spec = cast("dict[str, Any]", sub_specs["detached_inner"])
    outer_nodes = cast("dict[str, Any]", case["nodes"])

    # Detached subgraph identity → wrapper-node-name translation, same
    # convention as fixture 008. The fixture YAML lists subgraph identities
    # in ``detached_subgraphs:``; OTelObserver keys on the wrapper node's
    # name in the parent graph.
    detached_identities = set(cast("list[str]", case.get("detached_subgraphs") or []))
    wrapper_names: set[str] = set()
    for wrapper_name, node_spec in outer_nodes.items():
        sub_id = cast("dict[str, Any]", node_spec).get("subgraph")
        if isinstance(sub_id, str) and sub_id in detached_identities:
            wrapper_names.add(wrapper_name)
    detached_subgraphs = frozenset(wrapper_names)

    test_logger, prior_level = _enable_test_logger_at_info()

    # Inner subgraph (detached_inner): 1 node ``inner`` with
    # ``update_pure: {y: 1}`` + ``emits_log: "inside detached subgraph"``.
    class _Inner(State):
        y: int = 0

    inner_node_spec = cast("dict[str, Any]", inner_spec["nodes"]["inner"])
    inner_emit = cast("str", inner_node_spec["emits_log"]["message"])
    inner_update = cast("dict[str, Any]", inner_node_spec["update_pure"])

    async def _inner_body(_s: _Inner) -> dict[str, Any]:
        test_logger.info(inner_emit)
        return dict(inner_update)

    inner_compiled = (
        GraphBuilder(_Inner)
        .add_node("inner", _inner_body)
        .add_edge("inner", END)
        .set_entry("inner")
        .compile()
    )

    # Outer graph: ``outer_dispatch`` is a SubgraphNode wrapper around
    # ``inner_compiled`` AND emits a log "before subgraph dispatch".
    # SubgraphNode wrappers don't get ``prepare_sync`` per spec — the
    # outer log is emitted via per-node middleware that fires inside
    # the wrapper's chain. Without an attached span at wrapper scope,
    # the outer log's trace_id is OTel's "no active span" sentinel
    # (0); the inner log's trace_id is the detached trace's. The
    # invariant ``log_trace_ids_differ_when_detached`` holds either
    # way.
    class _Outer(State):
        z: int = 0

    outer_node_spec = cast("dict[str, Any]", outer_nodes["outer_dispatch"])
    outer_emit = cast("str", outer_node_spec["emits_log"]["message"])

    async def _outer_log_middleware(s: Any, next_call: Any) -> Mapping[str, Any]:
        test_logger.info(outer_emit)
        return cast("Mapping[str, Any]", await next_call(s))

    outer_compiled = (
        GraphBuilder(_Outer)
        .add_subgraph_node("outer_dispatch", inner_compiled, middleware=[_outer_log_middleware])
        .add_edge("outer_dispatch", END)
        .set_entry("outer_dispatch")
        .compile()
    )

    observer, _span_exporter = _build_observer_with_detached(detached_subgraphs)
    log_exporter, log_provider, snapshot = _setup_isolated_log_bridge()
    try:
        outer_compiled.attach_observer(observer)
        await outer_compiled.invoke(_Outer(), correlation_id=correlation_id)
        await outer_compiled.drain()
        observer.shutdown()
        log_provider.force_flush()

        records = log_exporter.get_finished_logs()
        ours = [r for r in records if str(r.log_record.body) in {outer_emit, inner_emit}]
        assert len(ours) == 2, (
            f"expected 2 log records (outer + inner); got {len(ours)}: "
            f"{[str(r.log_record.body) for r in ours]}"
        )

        by_body = {str(r.log_record.body): r for r in ours}
        outer_log = by_body[outer_emit]
        inner_log = by_body[inner_emit]

        # Invariant: log_trace_ids_differ_when_detached.
        assert outer_log.log_record.trace_id != inner_log.log_record.trace_id, (
            f"detached-subgraph log MUST carry the detached trace's trace_id, "
            f"DIFFERENT from the parent log; both got {outer_log.log_record.trace_id}"
        )

        # Invariant: all_logs_carry_correlation_id.
        for r in ours:
            attrs = dict(r.log_record.attributes or {})
            assert attrs.get("openarmature.correlation_id") == correlation_id, (
                f"every log MUST carry openarmature.correlation_id={correlation_id!r}; "
                f"got {attrs.get('openarmature.correlation_id')!r}"
            )
    finally:
        test_logger.setLevel(prior_level)
        _restore_log_state(snapshot)


def _build_observer_with_detached(detached_subgraphs: frozenset[str]) -> tuple[OTelObserver, Any]:
    """Variant of :func:`_build_observer` that takes a detached_subgraphs
    set — needed for fixture 010 sub-case 2."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        detached_subgraphs=detached_subgraphs,
    )
    return observer, exporter


# ---------------------------------------------------------------------------
# v0.17.0 LLM-payload + GenAI-semconv fixtures (012-021)
# ---------------------------------------------------------------------------


async def _run_llm_payload_fixture(spec: Mapping[str, Any]) -> None:
    """Generic driver for the ten LLM-attribute fixtures.

    Each fixture is single-case (GraphFixture shape) with a top-level
    ``cases:`` list of one entry; the case carries the graph + the
    ``calls_llm`` config + the optional observer/provider flags.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        try:
            await _run_llm_payload_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case.get('name')!r}: {e}") from e


async def _run_llm_payload_case(case: Mapping[str, Any]) -> None:
    """Build + invoke the graph, then walk the expected span tree
    asserting via the LLM-attribute helpers (parse-shape, truncation,
    redaction-substring-absence)."""
    import json

    import httpx
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    from openarmature.graph import END, GraphBuilder
    from openarmature.llm import OpenAIProvider

    from .adapter import build_state_cls
    from .harness.llm_attribute_assertions import (
        assert_attribute_does_not_contain,
        assert_attribute_parses_as_messages,
        assert_attribute_parses_as_object,
        assert_attribute_truncation,
        assert_attributes_absent,
        record_synthesized_base64_prefix,
        reset_synthesized_base64_prefixes,
    )

    reset_synthesized_base64_prefixes()

    # ---- Resolve harness primitives (content_repeat, base64_data_synthetic)
    nodes_spec = cast("dict[str, Any]", case["nodes"])
    entry_name = cast("str", case["entry"])
    # Most LLM-payload fixtures are single-node (the entry IS the
    # calls_llm node); fixture 026 has a non-LLM ``prep`` step
    # before the LLM call. Find whichever node carries ``calls_llm``
    # and treat the others as plain ``update`` nodes.
    llm_node_name = next(
        (name for name, spec in nodes_spec.items() if isinstance(spec, dict) and "calls_llm" in spec),
        entry_name,
    )
    calls_llm_spec = cast("dict[str, Any]", nodes_spec[llm_node_name]["calls_llm"])
    raw_messages = cast("list[dict[str, Any]]", calls_llm_spec.get("messages", []))
    materialized_messages, full_input_serialization = _materialize_messages(
        raw_messages,
        record_base64_prefix=record_synthesized_base64_prefix,
    )

    # ---- RuntimeConfig from the calls_llm.config block
    config_spec = cast("dict[str, Any] | None", calls_llm_spec.get("config"))
    runtime_config: RuntimeConfig | None = _build_runtime_config(config_spec)

    # ---- Provider knobs (provider.genai_system override)
    provider_spec = cast("dict[str, Any] | None", case.get("provider"))
    genai_system = "openai"
    if provider_spec and isinstance(provider_spec.get("genai_system"), str):
        genai_system = cast("str", provider_spec["genai_system"])

    # ---- Mock LLM transport
    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test",
        transport=httpx.MockTransport(_handler),
        genai_system=genai_system,
    )

    # ---- State + node body
    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    state_cls = build_state_cls("LlmPayloadFixtureState", state_fields)
    stores_in = cast("str", calls_llm_spec.get("stores_response_in", "msg"))

    async def ask_llm_body(_s: Any) -> dict[str, str]:
        response = await provider.complete(
            cast("Sequence[Any]", materialized_messages),
            config=runtime_config,
        )
        return {stores_in: response.message.content or ""}

    # Build the graph: the calls_llm node uses ``ask_llm_body``; any
    # other node carries an ``update:`` block translated to a simple
    # async function that returns it verbatim. Edges come from the
    # fixture's ``edges:`` list when present (multi-node case); the
    # single-node case falls back to ``entry → END``.
    builder = GraphBuilder(state_cls)
    for node_name, node_spec in nodes_spec.items():
        if node_name == llm_node_name:
            builder.add_node(node_name, ask_llm_body)
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        update_block = cast("dict[str, Any] | None", node_dict.get("update"))
        if update_block is None:
            raise AssertionError(
                f"non-LLM node {node_name!r} in LLM fixture has neither "
                f"`calls_llm` nor `update`; harness needs an extension"
            )

        def _make_update_body(payload: dict[str, Any]) -> Any:
            async def _body(_s: Any) -> dict[str, Any]:
                return dict(payload)

            return _body

        builder.add_node(node_name, _make_update_body(update_block))
    edges_spec = cast("list[dict[str, str]] | None", case.get("edges"))
    if edges_spec is None:
        builder.add_edge(llm_node_name, END)
    else:
        for edge in edges_spec:
            target_raw = edge["to"]
            target = END if target_raw == "END" else target_raw
            builder.add_edge(edge["from"], target)
    builder.set_entry(entry_name)
    graph = builder.compile()

    # ---- Observer
    exporter = InMemorySpanExporter()
    observer_kwargs: dict[str, Any] = {"span_processor": SimpleSpanProcessor(exporter)}
    if "disable_provider_payload" in case:
        observer_kwargs["disable_provider_payload"] = bool(case["disable_provider_payload"])
    if "disable_genai_semconv" in case:
        observer_kwargs["disable_genai_semconv"] = bool(case["disable_genai_semconv"])
    if "disable_llm_spans" in case:
        observer_kwargs["disable_llm_spans"] = bool(case["disable_llm_spans"])
    observer = OTelObserver(**observer_kwargs)
    graph.attach_observer(observer)

    # ---- Run + collect spans
    initial_state_cls = graph.state_cls
    invoke_kwargs: dict[str, Any] = {}
    caller_metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
    if caller_metadata is not None:
        invoke_kwargs["metadata"] = caller_metadata
    await graph.invoke(initial_state_cls(), **invoke_kwargs)
    await graph.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    # ---- Walk expected.span_tree and check per-span assertions
    expected = cast("dict[str, Any]", case["expected"])
    expected_tree = cast("list[dict[str, Any]]", expected.get("span_tree") or [])
    _check_payload_span_tree(
        spans,
        expected_tree,
        full_input_serialization=full_input_serialization,
        assert_attributes_absent=assert_attributes_absent,
        assert_attribute_parses_as_messages=assert_attribute_parses_as_messages,
        assert_attribute_parses_as_object=assert_attribute_parses_as_object,
        assert_attribute_does_not_contain=assert_attribute_does_not_contain,
        assert_attribute_truncation=assert_attribute_truncation,
    )


def _materialize_messages(
    raw_messages: list[dict[str, Any]],
    *,
    record_base64_prefix: Any,
) -> tuple[list[Any], str | None]:
    """Resolve harness directives (``content_repeat``,
    ``base64_data_synthetic``) into real ``Message`` instances.

    Returns the message list AND the canonical full-serialization
    string for the materialized payload — the truncation fixture
    needs the latter for its ``prefix_of_full_serialization`` check.
    """
    from openarmature.llm.messages import UserMessage

    out: list[Any] = []
    full_serial_target: str | None = None
    for msg in raw_messages:
        role = msg.get("role")
        # ``content_repeat`` may live at the message level (fixture 014:
        # ``{role: user, content_repeat: {char, bytes}}``) — no ``content``
        # key in that case; synthesize a string of N repeated chars.
        content: Any
        if "content_repeat" in msg:
            repeat = cast("dict[str, Any]", msg["content_repeat"])
            content = cast("str", repeat["char"]) * int(repeat["bytes"])
        else:
            content = msg.get("content")
        if role == "user":
            materialized = _materialize_user_content(
                content,
                record_base64_prefix=record_base64_prefix,
            )
            out.append(UserMessage(content=materialized))
        elif role == "system":
            from openarmature.llm.messages import SystemMessage

            out.append(SystemMessage(content=cast("str", content)))
        else:
            raise AssertionError(f"unsupported role in payload fixture: {role!r}")

    # Compute the full serialization (what the observer would emit
    # before truncation). The provider's _serialize_messages_for_payload
    # is the canonical encoder; mirror its shape via the same import.
    from openarmature.llm.providers.openai import _serialize_messages_for_payload

    plain = _serialize_messages_for_payload(out)
    import json

    full_serial_target = json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return out, full_serial_target


def _materialize_user_content(content: Any, *, record_base64_prefix: Any) -> Any:
    """Resolve the user message's content. Strings pass through; lists
    of blocks materialize the harness directives in each block.

    ``content_repeat: {char, bytes}`` on a string-only message synthesizes
    a repeated-character string of N bytes. ``base64_data_synthetic:
    {bytes}`` on an inline image source synthesizes a deterministic
    base64 blob; the prefix is recorded via the supplied callable so
    the ``attribute_does_not_contain`` assertion can verify absence.
    """
    from openarmature.llm.messages import (
        ImageBlock,
        ImageSourceInline,
        ImageSourceURL,
        TextBlock,
    )

    # Compact form: ``content`` is a dict with ``content_repeat`` —
    # synthesize a string of N repeated chars.
    if isinstance(content, dict) and "content_repeat" in content:
        repeat = cast("dict[str, Any]", content["content_repeat"])
        char = cast("str", repeat["char"])
        nbytes = int(repeat["bytes"])
        return char * nbytes
    if isinstance(content, str):
        return content
    # List of content blocks.
    blocks: list[Any] = []
    for block in cast("list[dict[str, Any]]", content):
        btype = block.get("type")
        if btype == "text":
            blocks.append(TextBlock(text=cast("str", block["text"])))
        elif btype == "image":
            source_spec = cast("dict[str, Any]", block["source"])
            stype = source_spec.get("type")
            if stype == "inline":
                synth = cast("dict[str, Any] | None", source_spec.get("base64_data_synthetic"))
                if synth is not None:
                    nbytes = int(synth["bytes"])
                    blob = _synth_base64(nbytes)
                    record_base64_prefix(blob)
                    source = ImageSourceInline(base64_data=blob)
                else:
                    source = ImageSourceInline(base64_data=cast("str", source_spec["base64_data"]))
            elif stype == "url":
                source = ImageSourceURL(url=cast("str", source_spec["url"]))
            else:
                raise AssertionError(f"unsupported image source type: {stype!r}")
            blocks.append(
                ImageBlock(
                    source=source,
                    media_type=cast("str | None", block.get("media_type")),
                    detail=cast("Any", block.get("detail")),
                )
            )
        else:
            raise AssertionError(f"unsupported content block type: {btype!r}")
    # Compact form: a single ``content_repeat`` entry inside a list.
    return blocks


def _synth_base64(nbytes: int) -> str:
    """Synthesize a deterministic base64 blob of exactly ``nbytes`` bytes.

    Fixture 015 uses 4096 bytes; deterministic so the synthesized prefix
    can be recorded once and the ``attribute_does_not_contain`` helper
    verifies the same prefix is absent from the redacted attribute.
    """
    # Repeated-letter base64 — valid base64 chars, deterministic, length
    # exactly nbytes.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    # Use a single character so the prefix-check signal is strong; the
    # bytes are not a real PNG (the redaction rule is about SHAPE).
    return alphabet[0] * nbytes


def _check_payload_span_tree(
    spans: Any,
    expected_tree: list[dict[str, Any]],
    *,
    full_input_serialization: str | None,
    assert_attributes_absent: Any,
    assert_attribute_parses_as_messages: Any,
    assert_attribute_parses_as_object: Any,
    assert_attribute_does_not_contain: Any,
    assert_attribute_truncation: Any,
) -> None:
    """Walk ``expected_tree`` and verify each expected span's attribute
    block matches the spans in ``spans``."""
    spans_by_name: dict[str, list[Any]] = {}
    for s in spans:
        spans_by_name.setdefault(s.name, []).append(s)

    def _walk(expected_entries: list[dict[str, Any]]) -> None:
        for entry in expected_entries:
            name = cast("str", entry["name"])
            candidates = spans_by_name.get(name, [])
            assert candidates, f"expected a span named {name!r}; got {sorted(spans_by_name.keys())}"
            # The fixtures we cover have unique span names in each tree.
            span = candidates[0]
            attrs = dict(span.attributes or {})
            # ``attributes:`` block — exact match per key.
            for k, v in cast("dict[str, Any]", entry.get("attributes") or {}).items():
                actual: Any = attrs.get(k)
                # OTel attribute arrays come back as tuples; normalize.
                if isinstance(v, list) and isinstance(actual, tuple):
                    actual = list(cast("tuple[Any, ...]", actual))
                assert actual == v, f"span {name!r} attribute {k!r} mismatch: expected {v!r}, got {actual!r}"
            # ``attributes_absent:`` list of names that MUST NOT appear.
            absent = entry.get("attributes_absent")
            if absent:
                assert_attributes_absent(attrs, cast("list[str]", absent))
            # ``attributes_present:`` list of names that MUST appear
            # (presence-only, value not asserted). Fixture 087 case 2
            # uses this for the gated openarmature.llm.output.tool_calls,
            # whose serialized value is checked structurally in the
            # mirror unit test rather than bytewise here.
            present = entry.get("attributes_present")
            if present:
                for attr_name in cast("list[str]", present):
                    assert attr_name in attrs, (
                        f"span {name!r} MUST carry attribute {attr_name!r}; got {sorted(attrs)}"
                    )
            # ``attribute_parses_as_messages:`` shape assertion.
            parses_as_messages = entry.get("attribute_parses_as_messages")
            if parses_as_messages:
                assert_attribute_parses_as_messages(attrs, cast("dict[str, Any]", parses_as_messages))
            # ``attribute_parses_as_object:`` shape assertion.
            parses_as_object = entry.get("attribute_parses_as_object")
            if parses_as_object:
                assert_attribute_parses_as_object(attrs, cast("dict[str, Any]", parses_as_object))
            # ``attribute_does_not_contain:`` substring absence.
            does_not_contain = entry.get("attribute_does_not_contain")
            if does_not_contain:
                assert_attribute_does_not_contain(attrs, cast("dict[str, Any]", does_not_contain))
            # ``attribute_truncation:`` §5.5.5 contract.
            truncation = entry.get("attribute_truncation")
            if truncation:
                full_map: dict[str, str] = {}
                # The fixture is single-attribute; supply the full
                # serialization under the same key for the
                # prefix_of_full_serialization clause.
                if full_input_serialization is not None:
                    for attr_name in cast("dict[str, Any]", truncation):
                        full_map[attr_name] = full_input_serialization
                assert_attribute_truncation(attrs, cast("dict[str, Any]", truncation), full_map)
            # Recurse into children.
            children = cast("list[dict[str, Any]] | None", entry.get("children"))
            if children:
                _walk(children)

    _walk(expected_tree)


# ---------------------------------------------------------------------------
# Proposal 0067 — GenAI metrics fixtures (088 / 090 / 091)
# ---------------------------------------------------------------------------
#
# The §6.9 metric-capture primitive: a private MeterProvider with an
# InMemoryMetricReader, injected into the OTelObserver so recorded
# measurements can be asserted. The driver reuses ``_build_simple_llm_graph``
# (mock transport + provider) and asserts ``expected.metrics`` against the
# recorded data points. Duration values + bucket assignment are not asserted
# (§11.4); token.usage values are (the fixed-usage mock).


async def _run_metrics_fixture(spec: Mapping[str, Any]) -> None:
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        try:
            await _run_metrics_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case.get('name')!r}: {e}") from e


async def _run_metrics_case(case: Mapping[str, Any]) -> None:
    from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from openarmature.graph import NodeException

    graph, state_cls, provider = _build_simple_llm_graph(case, populate_caller_metadata=False)

    reader = InMemoryMetricReader()
    meter_provider = SdkMeterProvider(metric_readers=[reader])
    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        enable_metrics=bool(case.get("enable_metrics", False)),
        meter_provider=meter_provider,
    )
    graph.attach_observer(observer)

    state = _make_state_instance(case, state_cls)
    expected_error = cast("dict[str, Any] | None", case.get("expected_error"))
    try:
        if expected_error is not None:
            # A failed provider call raises (LlmFailedEvent dispatched
            # alongside per 0058); the failed-attempt event is still
            # enqueued before the raise, so the duration metric records.
            with pytest.raises(NodeException):
                await graph.invoke(state)
        else:
            await graph.invoke(state)
        await graph.drain()
    finally:
        await provider.aclose()
        observer.shutdown()

    points = _collect_metric_points(reader)
    expected_metrics = cast("list[dict[str, Any]]", case["expected"].get("metrics") or [])
    _assert_metric_points(points, expected_metrics)


def _collect_metric_points(reader: Any) -> list[tuple[str, float, int, dict[str, Any]]]:
    """Flatten an InMemoryMetricReader's data into
    ``(instrument_name, point_sum, point_count, point_attributes)``
    tuples. Observations with identical attribute sets aggregate into one
    histogram data point (sum + count)."""
    data = reader.get_metrics_data()
    points: list[tuple[str, float, int, dict[str, Any]]] = []
    if data is None:
        return points
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                for pt in metric.data.data_points:
                    points.append((metric.name, pt.sum, pt.count, dict(pt.attributes)))
    return points


def _assert_metric_points(
    points: list[tuple[str, float, int, dict[str, Any]]],
    expected_metrics: list[dict[str, Any]],
) -> None:
    """Match each ``expected.metrics`` entry (instrument + exact
    dimensions, plus value for token.usage) against a recorded data
    point. An empty expected list asserts no measurements (fixture
    091)."""
    if not expected_metrics:
        assert points == [], f"expected no measurements; got {points}"
        return
    for expected in expected_metrics:
        instrument = cast("str", expected["instrument"])
        exp_dims = cast("dict[str, Any]", expected.get("dimensions") or {})
        candidates = [p for p in points if p[0] == instrument and p[3] == exp_dims]
        assert candidates, (
            f"no {instrument!r} observation with dimensions {exp_dims}; "
            f"recorded: {[(p[0], p[3]) for p in points]}"
        )
        # token.usage asserts the recorded value (from the fixed-usage
        # mock); duration asserts presence + dimensions only (§11.4).
        if "value" in expected:
            assert candidates[0][1] == expected["value"], (
                f"{instrument!r} {exp_dims} value: expected {expected['value']}, got sum {candidates[0][1]}"
            )


# ---------------------------------------------------------------------------
# Proposal 0063 — tool-execution observability fixtures (092-098)
# ---------------------------------------------------------------------------
#
# A calls_tool node enters the with_tool_call instrumentation scope around a
# mock tool (returns -> ToolCallEvent; raises -> ToolCallFailedEvent +
# re-raise). One graph builder serves all three assertion shapes:
# typed-event-collector (092-095), OTel span_tree (096/097), and the Langfuse
# Tool observation tree (098). Dispatch is by which expected.* key appears.


def _make_tool_node_body(spec: Mapping[str, Any]) -> Any:
    from openarmature.observability import with_tool_call

    tool_name = cast("str", spec["tool_name"])
    arguments = cast("dict[str, Any] | None", spec.get("arguments"))
    tool_call_id = cast("str | None", spec.get("tool_call_id"))
    stores_in = cast("str | None", spec.get("stores_result_in"))
    mock = cast("dict[str, Any]", spec["mock_tool"])

    async def body(_state: Any) -> Mapping[str, Any]:
        with with_tool_call(tool_name, arguments, tool_call_id=tool_call_id) as scope:
            raises = cast("dict[str, Any] | None", mock.get("raises"))
            if raises is not None:
                # Synthesize an exception whose class name == error_type
                # so the event captures the fixture's error_type verbatim.
                exc_type = cast("str", raises.get("error_type", "ToolError"))
                exc_cls = type(exc_type, (Exception,), {})
                raise exc_cls(cast("str", raises.get("message", "")))
            result = mock.get("returns")
            scope.set_result(result)
        return {stores_in: result} if stores_in else {}

    return body


def _make_tool_fixture_llm_body(spec: Mapping[str, Any], provider: Any) -> Any:
    from openarmature.llm import UserMessage

    stores_in = cast("str", spec.get("stores_response_in", "msg"))
    messages = [
        UserMessage(content=cast("str", m["content"]))
        for m in cast("list[dict[str, Any]]", spec.get("messages", []))
        if m.get("role") == "user"
    ]

    async def body(_state: Any) -> Mapping[str, str]:
        response = await provider.complete(messages)
        return {stores_in: response.message.content or ""}

    return body


def _build_tool_graph(case: Mapping[str, Any]) -> tuple[Any, type[Any], list[Any]]:
    """Build a graph whose nodes are calls_tool / calls_llm / update.
    Returns (compiled_graph, state_cls, providers-to-close)."""
    import json

    import httpx

    from openarmature.graph import END, GraphBuilder
    from openarmature.llm import OpenAIProvider

    from .adapter import build_state_cls

    state_cls = build_state_cls(
        "ToolFixtureState", cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    )
    builder = GraphBuilder(state_cls)
    providers: list[Any] = []
    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    nodes = cast("dict[str, Any]", case["nodes"])
    for node_name, node_spec in nodes.items():
        nd = cast("dict[str, Any]", node_spec)
        if "calls_tool" in nd:
            builder.add_node(node_name, _make_tool_node_body(cast("dict[str, Any]", nd["calls_tool"])))
        elif "calls_llm" in nd:
            provider = OpenAIProvider(
                base_url="http://mock-llm.test",
                model="test-model",
                api_key="test",
                transport=httpx.MockTransport(_handler),
            )
            providers.append(provider)
            builder.add_node(
                node_name, _make_tool_fixture_llm_body(cast("dict[str, Any]", nd["calls_llm"]), provider)
            )
        elif "update" in nd:
            update_block = cast("dict[str, Any]", nd["update"])

            async def _update_body(_s: Any, _payload: dict[str, Any] = update_block) -> dict[str, Any]:
                return dict(_payload)

            builder.add_node(node_name, _update_body)
        else:
            raise AssertionError(f"tool fixture node {node_name!r} has no recognized directive")

    for edge in cast("list[dict[str, str]]", case["edges"]):
        target = END if edge["to"] == "END" else edge["to"]
        builder.add_edge(edge["from"], target)
    builder.set_entry(cast("str", case["entry"]))
    return builder.compile(), state_cls, providers


def _assert_langfuse_observation_tree(
    trace: Any,
    expected: list[dict[str, Any]],
    parent_id: str | None = None,
    *,
    bindings: dict[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> None:
    """Recursively match expected observations against the trace's flat
    observation list (linked by parent_observation_id). type + name are
    matched exactly; level / input / output exactly when present; metadata is
    subset-matched. When ``bindings``/``params`` are supplied, metadata values
    go through the value-matcher (placeholder tokens + sub-key matchers);
    otherwise they are compared exactly (the tool-fixture path)."""
    # Mutable copy: each matched observation is consumed so two
    # same-shape expected siblings can't both bind to one actual.
    remaining = list(trace.children_of(parent_id))
    use_matcher = bindings is not None or params is not None
    for exp in expected:
        exp_type = cast("str", exp["type"])
        exp_name = cast("str | None", exp.get("name"))
        match = next(
            (o for o in remaining if o.type == exp_type and (exp_name is None or o.name == exp_name)),
            None,
        )
        assert match is not None, (
            f"no {exp_type!r} observation named {exp_name!r} under parent {parent_id!r}; "
            f"got {[(o.type, o.name) for o in remaining]}"
        )
        remaining.remove(match)
        if "level" in exp:
            assert match.level == exp["level"], f"{exp_name!r}: level {match.level!r} != {exp['level']!r}"
        if "input" in exp:
            assert match.input == exp["input"], f"{exp_name!r}: input {match.input!r} != {exp['input']!r}"
        if "output" in exp:
            assert match.output == exp["output"], (
                f"{exp_name!r}: output {match.output!r} != {exp['output']!r}"
            )
        for key, val in cast("dict[str, Any]", exp.get("metadata") or {}).items():
            if use_matcher:
                assert _langfuse_value_matches(
                    match.metadata.get(key), val, bindings=bindings or {}, params=params or {}
                ), f"{exp_name!r}: metadata.{key} {match.metadata.get(key)!r} did not match {val!r}"
            else:
                assert match.metadata.get(key) == val, (
                    f"{exp_name!r}: metadata.{key} {match.metadata.get(key)!r} != {val!r}"
                )
        children = cast("list[dict[str, Any]] | None", exp.get("children"))
        if children:
            _assert_langfuse_observation_tree(
                trace, children, parent_id=match.id, bindings=bindings, params=params
            )


async def _run_tool_fixture(spec: Mapping[str, Any]) -> None:
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        try:
            await _run_tool_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case.get('name')!r}: {e}") from e


async def _run_tool_case(case: Mapping[str, Any]) -> None:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from openarmature.graph import NodeException
    from openarmature.observability.langfuse import InMemoryLangfuseClient, LangfuseObserver

    from .harness.llm_attribute_assertions import (
        assert_attribute_does_not_contain,
        assert_attribute_parses_as_messages,
        assert_attribute_parses_as_object,
        assert_attribute_truncation,
        assert_attributes_absent,
    )

    expected = cast("dict[str, Any]", case["expected"])
    expected_error = cast("dict[str, Any] | None", case.get("expected_error"))
    graph, state_cls, providers = _build_tool_graph(case)
    state = _make_state_instance(case, state_cls)

    collectors: dict[str, _TypedEventCollector] = {}
    exporter: Any = None
    otel_observer: Any = None
    langfuse_client: Any = None
    if "observers" in expected:
        collectors, _ = _parse_typed_observers(case)
        for collector in collectors.values():
            graph.attach_observer(collector)
    if "span_tree" in expected:
        exporter = InMemorySpanExporter()
        otel_kwargs: dict[str, Any] = {"span_processor": SimpleSpanProcessor(exporter)}
        if "disable_provider_payload" in case:
            otel_kwargs["disable_provider_payload"] = bool(case["disable_provider_payload"])
        otel_observer = OTelObserver(**otel_kwargs)
        graph.attach_observer(otel_observer)
    if "langfuse_trace" in expected:
        langfuse_client = InMemoryLangfuseClient()
        lf_kwargs: dict[str, Any] = {"client": langfuse_client}
        if "disable_provider_payload" in case:
            lf_kwargs["disable_provider_payload"] = bool(case["disable_provider_payload"])
        graph.attach_observer(LangfuseObserver(**lf_kwargs))

    try:
        if expected_error is not None:
            with pytest.raises(NodeException):
                await graph.invoke(state)
        else:
            await graph.invoke(state)
        await graph.drain()
    finally:
        for provider in providers:
            await provider.aclose()
        if otel_observer is not None:
            otel_observer.shutdown()

    if "observers" in expected:
        for obs_name, obs_spec in cast("dict[str, Any]", expected["observers"]).items():
            _assert_observer_expectations(obs_name, collectors[obs_name], cast("dict[str, Any]", obs_spec))
    if "span_tree" in expected and exporter is not None:
        _check_payload_span_tree(
            exporter.get_finished_spans(),
            cast("list[dict[str, Any]]", expected["span_tree"]),
            full_input_serialization=None,
            assert_attributes_absent=assert_attributes_absent,
            assert_attribute_parses_as_messages=assert_attribute_parses_as_messages,
            assert_attribute_parses_as_object=assert_attribute_parses_as_object,
            assert_attribute_does_not_contain=assert_attribute_does_not_contain,
            assert_attribute_truncation=assert_attribute_truncation,
        )
    if "langfuse_trace" in expected and langfuse_client is not None:
        assert len(langfuse_client.traces) == 1, (
            f"expected 1 Langfuse trace; got {sorted(langfuse_client.traces)}"
        )
        trace = next(iter(langfuse_client.traces.values()))
        _assert_langfuse_observation_tree(
            trace, cast("list[dict[str, Any]]", expected["langfuse_trace"].get("observations") or [])
        )


# ---------------------------------------------------------------------------
# Proposal 0049 — typed LlmCompletionEvent fixtures (050-056)
# ---------------------------------------------------------------------------
#
# Activates the seven 050-056 conformance fixtures introduced by spec
# proposal 0049. The fixtures exercise the typed event variant on the
# observer event union: dispatch shape, type-discrimination filtering,
# the OPTIONAL caller_invocation_metadata opt-in, the success-only
# scope, fan_out_index + branch_name population, and strict-serial
# arrival ordering.
#
# Harness directives introduced:
#   - ``typed_observers`` at the case top level — list of observer
#     definitions with ``kind: typed_event_collector`` plus optional
#     flags (``filter_event_type``, ``include_caller_metadata``,
#     ``retains_arrival_order``).
#   - ``caller_metadata`` at the case top level — a mapping passed to
#     ``graph.invoke(metadata=...)`` for the run.
#   - Assertion shapes under ``expected.observers.<name>``:
#     ``contains_event``, ``contains_exactly_one_event_of_type``,
#     ``contains_event_of_type``, ``contains_exactly_n_events_of_type``,
#     ``does_not_contain_event_of_type``,
#     ``captured_event_field_values_cover``,
#     ``every_captured_event_has``, ``relative_order_of_events_matching``.


def _mock_model_from_first_response(case: Mapping[str, Any]) -> str | None:
    """Return the ``model`` declared on the first ``mock_llm`` response
    body (if any). Used to bind the test provider to the same model
    the fixture's expected events report.
    """
    responses = cast("list[dict[str, Any]] | None", case.get("mock_llm")) or []
    if not responses:
        return None
    body = cast("dict[str, Any] | None", responses[0].get("body")) or {}
    model = body.get("model")
    return model if isinstance(model, str) else None


class _TypedEventCollector:
    """Observer adapter that captures events for fixture introspection.

    Implements the Observer protocol. Appends every event to an
    internal list. When ``filter_event_type`` is set, only events
    whose class name matches are captured; ``None`` captures every
    event (preserves cross-event-type arrival order). The
    ``include_caller_metadata`` directive on the fixture YAML is
    consumed at parse time to set the provider's
    ``populate_caller_metadata`` knob and is not threaded through
    the collector instance. ``retains_arrival_order`` is preserved
    for spec-text fidelity but has no runtime effect (lists preserve
    insertion order).
    """

    def __init__(self, *, filter_event_type: str | None = None) -> None:
        self.filter_event_type = filter_event_type
        self.events: list[Any] = []

    async def __call__(self, event: Any) -> None:
        # LlmRetryAttemptEvent is python-internal (it drives the OTel
        # per-attempt span surface), not a spec-normative observer
        # event, so the conformance collector excludes it from the
        # captured stream that spec fixtures assert against.
        from openarmature.graph import LlmRetryAttemptEvent  # noqa: PLC0415

        if isinstance(event, LlmRetryAttemptEvent):
            return
        if self.filter_event_type is not None:
            if type(event).__name__ != self.filter_event_type:
                return
        self.events.append(event)


def _parse_typed_observers(
    case: Mapping[str, Any],
) -> tuple[dict[str, _TypedEventCollector], bool]:
    """Parse the ``typed_observers`` directive into a name → collector
    mapping plus the aggregate ``populate_caller_metadata`` flag for
    the provider (True when ANY collector's ``include_caller_metadata``
    flag is set).
    """
    raw = cast("list[dict[str, Any]] | None", case.get("typed_observers")) or []
    collectors: dict[str, _TypedEventCollector] = {}
    populate_caller_metadata = False
    for entry in raw:
        name = cast("str", entry["name"])
        kind = cast("str", entry.get("kind", "typed_event_collector"))
        assert kind == "typed_event_collector", f"unsupported typed_observer kind: {kind!r}"
        collectors[name] = _TypedEventCollector(
            filter_event_type=cast("str | None", entry.get("filter_event_type")),
        )
        if bool(entry.get("include_caller_metadata", False)):
            populate_caller_metadata = True
    return collectors, populate_caller_metadata


def _build_simple_llm_graph(
    case: Mapping[str, Any],
    *,
    populate_caller_metadata: bool,
) -> tuple[Any, type[Any], Any]:
    """Build a single-node graph that calls the LLM provider against a
    mock transport. Matches the simple entry → ask → END pattern used
    by fixtures 050, 051, 052, 053, 056. Returns ``(compiled_graph,
    state_cls, provider)`` — the caller owns the provider's lifecycle
    and MUST call ``await provider.aclose()`` after invoke completes
    to release the underlying httpx.AsyncClient connection pool.
    """
    from openarmature.graph import END, GraphBuilder
    from openarmature.llm import OpenAIProvider

    from .adapter import build_state_cls

    transport = _make_mock_transport(case)
    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    state_cls = build_state_cls("LlmTypedFixtureState", state_fields)

    nodes = cast("dict[str, Any]", case["nodes"])
    entry_name = cast("str", case["entry"])
    node_spec = cast("dict[str, Any]", nodes[entry_name])
    calls_llm_spec = cast("dict[str, Any]", node_spec["calls_llm"])
    stores_in = cast("str", calls_llm_spec.get("stores_response_in", "msg"))

    # Bind the provider to the request-side model. Priority: the node's
    # declared ``calls_llm.model`` (the requested identifier per spec
    # §5.5.7 -- 068 needs this to differ from the provider-returned
    # response_model), else the model the first mock response reports
    # (050-056 path), else a default.
    bound_model = (
        cast("str | None", calls_llm_spec.get("model"))
        or _mock_model_from_first_response(case)
        or "test-model"
    )
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model=bound_model,
        api_key="test",
        transport=transport,
        populate_caller_metadata=populate_caller_metadata,
    )

    runtime_config = _build_runtime_config(cast("dict[str, Any] | None", calls_llm_spec.get("config")))

    # A node may render a prompt before the call (064): the rendered
    # PromptResult is stamped active for the complete() call and supplies
    # the messages when the node declares none explicitly.
    renders_prompt_name = cast("str | None", node_spec.get("renders_prompt"))
    prompt_result = _render_prompt_result(case, renders_prompt_name) if renders_prompt_name else None

    messages_spec = cast("list[dict[str, str]]", calls_llm_spec.get("messages", []))
    if messages_spec:
        messages = _materialize_typed_messages(messages_spec)
    elif prompt_result is not None:
        messages = list(prompt_result.messages)
    else:
        messages = []

    async def ask_body(_s: Any) -> dict[str, str]:
        response = await _complete_with_optional_prompt(
            provider, messages, config=runtime_config, prompt_result=prompt_result
        )
        return {stores_in: response.message.content or ""}

    builder = (
        GraphBuilder(state_cls).add_node(entry_name, ask_body).add_edge(entry_name, END).set_entry(entry_name)
    )
    return builder.compile(), state_cls, provider


def _make_mock_transport(case: Mapping[str, Any]) -> Any:
    """Build an httpx.MockTransport that replays the case's ``mock_llm``
    response queue in order, one response popped per request.
    """
    import json

    import httpx

    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(_handler)


def _build_runtime_config(config_spec: Mapping[str, Any] | None) -> RuntimeConfig | None:
    """Build a RuntimeConfig from a fixture's ``calls_llm.config`` block, or
    None when absent.
    """
    if not config_spec:
        return None
    from openarmature.llm.response import RuntimeConfig

    # The canonical sampling keys (observability §5.5.2) map to RuntimeConfig
    # fields; everything under ``extras`` is the provider-specific extras bag
    # (062 request_params, 063 request_extras). Mirrors the LLM-payload runner.
    extras = cast("dict[str, Any]", config_spec.get("extras") or {})
    kwargs: dict[str, Any] = {
        k: v
        for k, v in config_spec.items()
        if k
        in {
            "temperature",
            "max_tokens",
            "top_p",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
        }
    }
    kwargs.update(extras)
    return RuntimeConfig(**kwargs)


def _require_text_content(role: object, content: object) -> str:
    """Assert a fixture message's ``content`` is a present, non-empty string and
    return it (the system/user roles require this).
    """
    # Assert with the role + value rather than coercing, so a fixture mistake
    # surfaces on the real field instead of a downstream model ValueError.
    assert isinstance(content, str) and content != "", (
        f"{role} message content MUST be a present non-empty string; got {content!r}"
    )
    return content


def _materialize_typed_messages(messages_spec: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Build typed Message objects from a fixture's ``calls_llm.messages`` list,
    for the system / user / assistant roles the typed-event fixtures use.
    """
    from openarmature.llm import AssistantMessage, SystemMessage, UserMessage

    # 060 sends a system + user pair the event echoes back in full, so dropping
    # non-user roles would under-populate input_messages.
    out: list[Any] = []
    for m in messages_spec:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            out.append(SystemMessage(content=_require_text_content(role, content)))
        elif role == "user":
            out.append(UserMessage(content=_require_text_content(role, content)))
        elif role == "assistant":
            # Assistant content is optional (tool-call-only messages carry none).
            out.append(AssistantMessage(content=cast("str", content or "")))
        else:
            raise AssertionError(f"unsupported message role in typed-event fixture: {role!r}")
    return out


def _render_prompt_result(case: Mapping[str, Any], prompt_name: str) -> Any:
    """Build a PromptResult from ``prompt_backend.prompts.<name>`` rendered
    against ``render_variables``.
    """
    from datetime import UTC, datetime

    from openarmature.llm import Message, UserMessage
    from openarmature.prompts import PromptResult, compute_rendered_hash

    # The renders_prompt: directive (064): the 5-field identity (name / version
    # / label / template_hash / rendered_hash) is what the event's active_prompt
    # asserts; the rendered messages drive the actual call.
    prompts = cast("dict[str, dict[str, Any]]", case["prompt_backend"]["prompts"])
    entry = prompts[prompt_name]
    variables = cast("dict[str, Any]", case.get("render_variables") or {})
    template = cast("str", entry.get("template", ""))
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", str(value)).replace("{{ " + key + " }}", str(value))
    messages: list[Message] = [UserMessage(content=rendered)]
    now = datetime.now(UTC)
    return PromptResult(
        name=cast("str", entry["name"]),
        version=cast("str", entry["version"]),
        label=cast("str", entry["label"]),
        template_hash=cast("str", entry["template_hash"]),
        rendered_hash=compute_rendered_hash(messages),
        messages=messages,
        variables=variables,
        fetched_at=now,
        rendered_at=now,
    )


async def _complete_with_optional_prompt(
    provider: Any,
    messages: Sequence[Any],
    *,
    config: Any,
    prompt_result: Any,
) -> Any:
    """Call ``provider.complete`` inside the active-prompt context when the node
    rendered a prompt, otherwise call it directly.
    """
    if prompt_result is not None:
        from openarmature.prompts import with_active_prompt

        # Inside with_active_prompt so the provider stamps active_prompt onto
        # the typed event.
        with with_active_prompt(prompt_result):
            return await provider.complete(messages, config=config)
    return await provider.complete(messages, config=config)


def _build_chain_llm_graph(
    case: Mapping[str, Any],
    *,
    populate_caller_metadata: bool,
) -> tuple[Any, type[Any], Any]:
    """Build a multi-node graph where every node with a ``calls_llm`` block
    calls one shared provider against one mock-response queue, wired per the
    case's ``edges``. Used by the chain fixtures 067 (three success calls) and
    071 (success then failure). Returns ``(compiled, state_cls, provider)``;
    the caller owns ``provider.aclose()``.
    """
    from openarmature.graph import END, GraphBuilder
    from openarmature.llm import OpenAIProvider

    from .adapter import build_state_cls

    bound_model = _mock_model_from_first_response(case) or "test-model"
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model=bound_model,
        api_key="test",
        transport=_make_mock_transport(case),
        populate_caller_metadata=populate_caller_metadata,
    )

    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    state_cls = build_state_cls("LlmChainFixtureState", state_fields)

    nodes = cast("dict[str, Any]", case["nodes"])
    builder = GraphBuilder(state_cls)

    def _make_node_body(messages: list[Any], stores_in: str, config: Any) -> Any:
        async def _body(_s: Any) -> dict[str, str]:
            response = await provider.complete(messages, config=config)
            return {stores_in: response.message.content or ""}

        return _body

    for node_name, raw in nodes.items():
        node = cast("dict[str, Any]", raw)
        if "calls_llm" not in node:
            raise AssertionError(
                f"_build_chain_llm_graph only supports calls_llm nodes; {node_name!r} has none"
            )
        calls_llm_spec = cast("dict[str, Any]", node["calls_llm"])
        stores_in = cast("str", calls_llm_spec.get("stores_response_in", "msg"))
        messages = _materialize_typed_messages(
            cast("list[dict[str, str]]", calls_llm_spec.get("messages", []))
        )
        config = _build_runtime_config(cast("dict[str, Any] | None", calls_llm_spec.get("config")))
        builder.add_node(node_name, _make_node_body(messages, stores_in, config))

    for edge in cast("list[dict[str, str]]", case.get("edges") or []):
        target = edge["to"]
        builder.add_edge(edge["from"], END if target == "END" else target)
    builder.set_entry(cast("str", case["entry"]))

    return builder.compile(), state_cls, provider


def _assert_expected_error_if_present(case: Mapping[str, Any], exc: Exception) -> None:
    """When the case carries an ``expected_error`` block (071/072), assert the
    raised exception's cause chain carries the declared ``category`` AND
    originates at the declared ``raised_from`` node.
    """
    # category (llm-provider §7) sits on the inner LlmProviderError; raised_from
    # sits on the engine's NodeException wrapper as node_name -- both in one
    # chain. Complements contains_event: LlmFailedEvent: per 0058's exception-
    # flow-preserved contract both must hold (the event fires AND the exception
    # still raises), so this is not a substitute for the event check.
    expected_error = cast("dict[str, Any] | None", case.get("expected_error"))
    if not expected_error:
        return
    category = cast("str", expected_error["category"])
    raised_from = cast("str | None", expected_error.get("raised_from"))
    found_category = False
    node_match = raised_from is None
    err: Any = exc
    while err is not None:
        if getattr(err, "category", None) == category:
            found_category = True
        if raised_from is not None and getattr(err, "node_name", None) == raised_from:
            node_match = True
        err = getattr(err, "__cause__", None)
    if not found_category:
        raise AssertionError(
            f"expected_error category {category!r} not found in the raised exception cause chain"
        )
    if not node_match:
        raise AssertionError(
            f"expected_error raised_from {raised_from!r} not found "
            f"(no NodeException for that node in the cause chain)"
        )


def _assert_call_id_invariants(
    case: Mapping[str, Any],
    collectors: Mapping[str, _TypedEventCollector],
) -> None:
    """Machine-check the call_id presence/distinctness invariants that 067/071
    declare in their ``invariants`` block. A no-op for cases with no call_id
    invariant.
    """
    # The fixtures' ``expected`` blocks assert only event counts, which don't
    # capture the per-call call_id freshness contract they're named for; this
    # closes that gap. Scoped to the terminal events (LlmCompletionEvent /
    # LlmFailedEvent): the per-attempt LlmRetryAttemptEvent shares its call's
    # call_id, so including it would false-collide.
    invariants = cast("dict[str, Any]", case.get("invariants") or {})
    if not any("call_id" in key for key in invariants):
        return
    # Gather terminal events across every collector, deduped by identity, so a
    # filtered-only collector still yields the call_ids (no silent no-op).
    seen: set[int] = set()
    ids: list[str] = []
    for collector in collectors.values():
        for event in collector.events:
            if type(event).__name__ in {"LlmCompletionEvent", "LlmFailedEvent"} and id(event) not in seen:
                seen.add(id(event))
                ids.append(cast("str", event.call_id))
    for cid in ids:
        assert isinstance(cid, str) and cid != "", (
            f"call_id invariant: every terminal event's call_id MUST be a non-empty string; got {cid!r}"
        )
    assert len(ids) == len(set(ids)), (
        f"call_id invariant: terminal-event call_ids MUST be pairwise distinct; got {ids!r}"
    )


async def _run_typed_event_chain_case(
    case: Mapping[str, Any],
    *,
    expect_failure: bool = False,
) -> None:
    """Runner for the multi-node-chain typed-event cases (067 success chain,
    071 success-then-failure chain). Mirrors _run_typed_event_fixture_case but
    builds the graph via _build_chain_llm_graph.
    """
    collectors, populate_caller_metadata = _parse_typed_observers(case)
    graph, state_cls, provider = _build_chain_llm_graph(
        case, populate_caller_metadata=populate_caller_metadata
    )
    try:
        extra: _AllEventsCollector | None = None
        if expect_failure and not any(c.filter_event_type is None for c in collectors.values()):
            extra = _AllEventsCollector()
        final, exc = await _invoke_typed_fixture(case, collectors, graph, state_cls, extra_observer=extra)
        if expect_failure:
            assert exc is not None, "failure-path chain fixture expected an exception"
            _assert_expected_error_if_present(case, exc)
        elif final is None:
            raise AssertionError("expected a non-None final state on success path")
        expected = cast("dict[str, Any]", case.get("expected") or {})
        observer_expectations = cast("dict[str, Any]", expected.get("observers") or {})
        for name, expectations in observer_expectations.items():
            collector = collectors.get(name)
            if collector is None:
                raise AssertionError(f"fixture references unknown observer {name!r}")
            _assert_observer_expectations(name, collector, cast("dict[str, Any]", expectations))
        _assert_call_id_invariants(case, collectors)
    finally:
        await provider.aclose()


async def _run_typed_event_cases(spec: Mapping[str, Any], *, expect_failure: bool = False) -> None:
    """Iterate the simple single-node typed-event cases (060-065, 068 success;
    072 failure), each through _run_typed_event_fixture_case.
    """
    for case in cast("list[dict[str, Any]]", spec["cases"]):
        case_name = cast("str", case["name"])
        try:
            await _run_typed_event_fixture_case(case, expect_failure=expect_failure)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_typed_event_chain_cases(spec: Mapping[str, Any], *, expect_failure: bool = False) -> None:
    """Iterate the multi-node-chain typed-event cases (067 success, 071
    failure)."""
    for case in cast("list[dict[str, Any]]", spec["cases"]):
        case_name = cast("str", case["name"])
        try:
            await _run_typed_event_chain_case(case, expect_failure=expect_failure)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


def _make_state_instance(case: Mapping[str, Any], state_cls: type[Any]) -> Any:
    """Construct a State instance from the case's ``initial_state`` plus
    field defaults declared on the fixture state schema.
    """
    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    initial = cast("dict[str, Any]", case.get("initial_state") or {})
    state_kwargs: dict[str, Any] = {}
    for field_name, field_spec in state_fields.items():
        if field_name in initial:
            state_kwargs[field_name] = initial[field_name]
        elif "default" in field_spec:
            state_kwargs[field_name] = field_spec["default"]
    return state_cls(**state_kwargs)


async def _invoke_typed_fixture(
    case: Mapping[str, Any],
    collectors: Mapping[str, _TypedEventCollector],
    graph: Any,
    state_cls: type[Any],
    *,
    extra_observer: Any | None = None,
) -> tuple[Any | None, Exception | None]:
    """Attach collectors, invoke the graph with the case's
    ``caller_metadata`` (when present), and return the final state +
    any propagated exception. ``extra_observer`` (when provided, e.g.,
    the failure-path NodeEvent collector) is attached alongside.
    Errors are captured rather than raised so failure-path fixtures
    (053) can assert on observer state.
    """
    from openarmature.graph import NodeException

    handles = [graph.attach_observer(c) for c in collectors.values()]
    if extra_observer is not None:
        handles.append(graph.attach_observer(extra_observer))
    metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
    state_instance = _make_state_instance(case, state_cls)
    try:
        # ``is not None`` so an explicit ``caller_metadata: {}`` is
        # passed through to graph.invoke() (truthy check would collapse
        # the empty-mapping case to "no metadata" and drop the kwarg).
        if metadata is not None:
            final = await graph.invoke(state_instance, metadata=metadata)
        else:
            final = await graph.invoke(state_instance)
        return final, None
    except NodeException as exc:
        return None, exc
    finally:
        for handle in handles:
            handle.remove()
        await graph.drain()


# Known assertion shape keys on ``expected.observers.<name>``. Used by
# ``_assert_observer_expectations`` to detect fixture typos (an unknown
# key would otherwise silently skip the assertion).
_OBSERVER_ASSERTION_KEYS = frozenset(
    {
        "contains_event",
        "contains_exactly_one_event_of_type",
        "contains_event_of_type",
        "contains_exactly_n_events_of_type",
        # proposal 0063 (092-094) spelling for an exact-count assertion,
        # same shape as contains_exactly_n_events_of_type.
        "event_count",
        # proposal 0058 (071/072): list form of the scalar event_count, one
        # {event_type, count} entry per asserted type in the same observer
        # block.
        "event_counts",
        "does_not_contain_event_of_type",
        "captured_event_field_values_cover",
        "every_captured_event_has",
        "relative_order_of_events_matching",
        # Informational flags carried by fixtures but not driving an
        # assertion in this harness; tracked here so they don't trip
        # the unknown-key guard.
        "sentinel_node_event_emission_is_impl_defined",
    }
)


def _assert_observer_expectations(
    name: str,
    collector: _TypedEventCollector,
    spec: Mapping[str, Any],
) -> None:
    """Apply each observer-level assertion shape from a fixture's
    ``expected.observers.<name>`` block against the collector's
    captured events. Catches fixture typos by rejecting unknown
    assertion keys.
    """
    unknown_keys = set(spec.keys()) - _OBSERVER_ASSERTION_KEYS
    assert not unknown_keys, (
        f"observer {name!r}: unknown assertion key(s) {sorted(unknown_keys)!r}; "
        f"supported keys are {sorted(_OBSERVER_ASSERTION_KEYS)!r}"
    )
    events = collector.events
    if "contains_event" in spec:
        sub = cast("dict[str, Any]", spec["contains_event"])
        _assert_contains_event(name, events, sub)
    if "contains_exactly_one_event_of_type" in spec:
        type_name = cast("str", spec["contains_exactly_one_event_of_type"])
        matching = [e for e in events if type(e).__name__ == type_name]
        assert len(matching) == 1, (
            f"observer {name!r}: expected exactly one {type_name} event; got {len(matching)}"
        )
    if "contains_event_of_type" in spec:
        type_name = cast("str", spec["contains_event_of_type"])
        matching = [e for e in events if type(e).__name__ == type_name]
        assert len(matching) >= 1, f"observer {name!r}: expected at least one {type_name} event; got 0"
    if "contains_exactly_n_events_of_type" in spec:
        sub = cast("dict[str, Any]", spec["contains_exactly_n_events_of_type"])
        type_name = cast("str", sub["event_type"])
        expected_count = int(cast("int", sub["count"]))
        matching = [e for e in events if type(e).__name__ == type_name]
        assert len(matching) == expected_count, (
            f"observer {name!r}: expected exactly {expected_count} {type_name} events; got {len(matching)}"
        )
    if "event_count" in spec:
        sub = cast("dict[str, Any]", spec["event_count"])
        type_name = cast("str", sub["event_type"])
        expected_count = int(cast("int", sub["count"]))
        matching = [e for e in events if type(e).__name__ == type_name]
        assert len(matching) == expected_count, (
            f"observer {name!r}: expected exactly {expected_count} {type_name} events; got {len(matching)}"
        )
    if "event_counts" in spec:
        for item in cast("list[dict[str, Any]]", spec["event_counts"]):
            type_name = cast("str", item["event_type"])
            expected_count = int(cast("int", item["count"]))
            matching = [e for e in events if type(e).__name__ == type_name]
            assert len(matching) == expected_count, (
                f"observer {name!r}: expected {expected_count} {type_name} events; got {len(matching)}"
            )
    if "does_not_contain_event_of_type" in spec:
        type_name = cast("str", spec["does_not_contain_event_of_type"])
        matching = [e for e in events if type(e).__name__ == type_name]
        assert len(matching) == 0, f"observer {name!r}: expected zero {type_name} events; got {len(matching)}"
    if "captured_event_field_values_cover" in spec:
        sub = cast("dict[str, Any]", spec["captured_event_field_values_cover"])
        field_name = cast("str", sub["field"])
        expected_values = cast("list[Any]", sub["values"])
        captured_values = [getattr(e, field_name) for e in events if hasattr(e, field_name)]
        # "Cover" is set-equality semantics: the captured field-value
        # set MUST equal the expected set. Using ``set`` rather than
        # sorted-by-str avoids the latter's ambiguity around None +
        # primitives and matches what the spec text means by "cover".
        try:
            captured_set = set(captured_values)
            expected_set = set(expected_values)
        except TypeError as e:  # unhashable values (lists, dicts) in either side
            raise AssertionError(
                f"observer {name!r}: captured_event_field_values_cover requires hashable "
                f"field values for field {field_name!r}; got captured={captured_values!r}"
            ) from e
        assert captured_set == expected_set, (
            f"observer {name!r}: field {field_name!r} values across captured events: "
            f"got {sorted(captured_values, key=repr)}, "
            f"expected {sorted(expected_values, key=repr)}"
        )
    if "every_captured_event_has" in spec:
        sub = cast("dict[str, Any]", spec["every_captured_event_has"])
        for event in events:
            for field_name, expected_value in sub.items():
                if not hasattr(event, field_name):
                    raise AssertionError(
                        f"observer {name!r}: every_captured_event_has names field "
                        f"{field_name!r} that does not exist on {type(event).__name__}; "
                        f"check for typos in the fixture YAML or add a filter_event_type "
                        f"to scope the captured set"
                    )
                actual = getattr(event, field_name)
                assert actual == expected_value, (
                    f"observer {name!r}: expected every event to have "
                    f"{field_name}={expected_value!r}; got {actual!r} on {type(event).__name__}"
                )
    if "relative_order_of_events_matching" in spec:
        sub = cast("dict[str, Any]", spec["relative_order_of_events_matching"])
        _assert_relative_order(name, events, sub)


def _assert_contains_event(
    observer_name: str,
    events: Sequence[Any],
    spec: Mapping[str, Any],
) -> None:
    """Match ``spec.event_type`` + ``spec.fields`` against captured
    events. At least one event must match every declared field. None
    values in the fixture YAML are matched exactly.
    """
    type_name = cast("str", spec["event_type"])
    expected_fields = cast("dict[str, Any]", spec.get("fields") or {})
    # ``fields_absent_keys`` (062, conformance-adapter §3.2): the named field
    # MUST be a mapping AND none of the listed keys may appear in it. A
    # matching event must satisfy both ``fields`` and ``fields_absent_keys``.
    absent_keys_spec = cast("dict[str, list[str]]", spec.get("fields_absent_keys") or {})
    matching_type = [e for e in events if type(e).__name__ == type_name]
    assert matching_type, (
        f"observer {observer_name!r}: contains_event expected at least one {type_name}; got none"
    )
    for event in matching_type:
        if _event_fields_match(event, expected_fields) and _event_fields_absent_keys(event, absent_keys_spec):
            return
    raise AssertionError(
        f"observer {observer_name!r}: no {type_name} event matched fields {expected_fields!r} "
        f"with absent keys {absent_keys_spec!r}; "
        f"captured: {[_event_to_repr(e) for e in matching_type]}"
    )


def _event_fields_absent_keys(event: Any, absent_spec: Mapping[str, Sequence[str]]) -> bool:
    """Return True when, for each ``field -> [keys]`` in ``absent_spec``, the
    event's field is a mapping containing none of the listed keys. Raises when
    the fixture names a field that doesn't exist on the event (typo guard,
    matching ``_event_fields_match``).
    """
    # Absence-is-meaningful (conformance-adapter §3.2): a key present with a
    # null value still counts as present and fails the check.
    for field_name, keys in absent_spec.items():
        if not hasattr(event, field_name):
            raise AssertionError(
                f"fields_absent_keys references field {field_name!r} that does not exist on "
                f"{type(event).__name__}; check for typos in the fixture YAML"
            )
        actual = getattr(event, field_name)
        if not isinstance(actual, Mapping):
            return False
        actual_map = cast("Mapping[str, Any]", actual)
        for key in keys:
            if key in actual_map:
                return False
    return True


def _event_fields_match(event: Any, expected: Mapping[str, Any]) -> bool:
    """Return True when every key in ``expected`` matches the event's field.

    Comparison delegates to ``_value_matches``, which handles the fixture
    idioms: the ``<any-string>`` value-token, list-vs-tuple sequences
    (``namespace``), and nested mappings compared against either a Mapping or
    a record's attributes (``usage`` Usage, ``active_prompt`` PromptResult).

    Raises AssertionError when the fixture names a field that doesn't exist on
    the event type. Upstream filtering by event type means a missing attribute
    signals a fixture-side typo (e.g. ``node_nam: null`` instead of
    ``node_name: null``), not a None value worth silently matching.
    """
    for field_name, expected_value in expected.items():
        if not hasattr(event, field_name):
            raise AssertionError(
                f"fixture references field {field_name!r} that does not exist on "
                f"{type(event).__name__}; check for typos in the fixture YAML"
            )
        if not _value_matches(getattr(event, field_name), expected_value):
            return False
    return True


def _value_matches(actual: Any, expected: Any) -> bool:
    """Match one captured value against a fixture's expected value.

    - ``<any-string>``: any non-empty string (an empty string fails); used by
      064's ``rendered_hash``.
    - A list expectation against a tuple (the event carries ``namespace`` as a
      tuple) compares as sequences.
    - A mapping expectation compares against either a Mapping or a record's
      attributes (``usage`` -> Usage instance, ``active_prompt`` ->
      PromptResult), recursing so inner tokens still apply.
    - Everything else is plain equality (None matched exactly).
    """
    # <any-string> (conformance-adapter §3.2) matches any NON-EMPTY string; an
    # empty string is non-null but MUST fail (spec ruling on Q3).
    if expected == "<any-string>":
        return isinstance(actual, str) and actual != ""
    if isinstance(expected, list) and isinstance(actual, tuple):
        actual = list(cast("tuple[Any, ...]", actual))
    if isinstance(expected, Mapping):
        if actual is None:
            return False
        for key, sub_expected in cast("Mapping[str, Any]", expected).items():
            if isinstance(actual, Mapping):
                actual_mapping = cast("Mapping[str, Any]", actual)
                if key not in actual_mapping:
                    return False
                sub_actual = actual_mapping[key]
            elif hasattr(actual, key):
                sub_actual = getattr(actual, key)
            else:
                return False
            if not _value_matches(sub_actual, sub_expected):
                return False
        return True
    return bool(actual == expected)


def _event_to_repr(event: Any) -> dict[str, Any]:
    """Compact field dump for assertion error messages."""
    keys = ("invocation_id", "node_name", "namespace", "model", "provider", "finish_reason")
    return {k: getattr(event, k, None) for k in keys}


def _assert_relative_order(
    observer_name: str,
    events: Sequence[Any],
    spec: Mapping[str, Any],
) -> None:
    """Filter events by the fixture's ``filter`` map, then assert the
    resulting subsequence's first len(expected_order) entries match
    the expected (event_type, optional phase) pattern.
    """
    filter_spec = cast("dict[str, Any]", spec.get("filter") or {})
    expected_order = cast("list[dict[str, Any]]", spec.get("expected_order") or [])

    def _matches(event: Any, filt: Mapping[str, Any]) -> bool:
        for key, value in filt.items():
            actual = getattr(event, key, None)
            if actual != value:
                return False
        return True

    filtered = [e for e in events if _matches(e, filter_spec)]
    assert len(filtered) >= len(expected_order), (
        f"observer {observer_name!r}: relative_order expected at least {len(expected_order)} "
        f"filtered events; got {len(filtered)} (captured types: "
        f"{[type(e).__name__ for e in filtered]})"
    )
    for index, expected in enumerate(expected_order):
        event = filtered[index]
        expected_type = cast("str", expected["event_type"])
        assert type(event).__name__ == expected_type, (
            f"observer {observer_name!r}: relative_order at position {index} expected "
            f"{expected_type}; got {type(event).__name__}"
        )
        if "phase" in expected:
            expected_phase = expected["phase"]
            actual_phase = getattr(event, "phase", None)
            assert actual_phase == expected_phase, (
                f"observer {observer_name!r}: relative_order at position {index} expected "
                f"phase {expected_phase!r}; got {actual_phase!r}"
            )


def _assert_node_completed_event_carries_error(
    events: Sequence[Any],
    spec: Mapping[str, Any],
) -> None:
    """Failure-path assertion (fixture 053): the calling node's
    completed NodeEvent carries an error whose cause chain bottoms
    out in an llm-provider category matching the expectation.
    The engine wraps the underlying ProviderUnavailable (etc.) in a
    NodeException; walk ``__cause__`` to reach the categorized cause.
    """
    node_name = cast("str", spec["node_name"])
    expected_category = cast("str", spec["error_category"])
    for event in events:
        if type(event).__name__ != "NodeEvent":
            continue
        if getattr(event, "node_name", None) != node_name:
            continue
        if getattr(event, "phase", None) != "completed":
            continue
        error: Any = getattr(event, "error", None)
        # Walk the cause chain for a category attribute. The
        # NodeException wrapper itself is uncategorized; the
        # LlmProviderError underneath carries the canonical category.
        while error is not None:
            category = getattr(error, "category", None)
            if category == expected_category:
                return
            error = getattr(error, "__cause__", None)
    raise AssertionError(
        f"no NodeEvent(completed, node_name={node_name!r}) with error category {expected_category!r} found"
    )


# Use a single collector that captures EVERY event for fixtures that
# need to assert on NodeEvent(completed, error=...). Distinct from the
# fixture's named typed_observers; attached temporarily and detached
# before fixture assertions run.
class _AllEventsCollector:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def __call__(self, event: Any) -> None:
        self.events.append(event)


async def _run_llm_cache_fixture(spec: Mapping[str, Any]) -> None:
    """Run the cache-attribute fixtures (040, 041, 042). All three
    share the same simple-shape graph and assert
    on ``Response.usage`` cache fields plus the LLM provider span's
    ``openarmature.llm.cache_read.input_tokens`` /
    ``openarmature.llm.cache_creation.input_tokens`` attribute set.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_llm_cache_fixture_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_llm_cache_fixture_case(case: Mapping[str, Any]) -> None:
    """Build a simple LLM-calling graph, capture the response, and
    assert on response_usage + llm_span_attributes /
    llm_span_attributes_absent expectations.
    """
    import json

    import httpx
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    from openarmature.graph import END, GraphBuilder
    from openarmature.llm import OpenAIProvider, UserMessage
    from openarmature.llm.response import Response
    from openarmature.observability.otel import OTelObserver

    from .adapter import build_state_cls

    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model=_mock_model_from_first_response(case) or "test-model",
        api_key="test",
        transport=httpx.MockTransport(_handler),
    )

    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    state_cls = build_state_cls("LlmCacheFixtureState", state_fields)

    nodes = cast("dict[str, Any]", case["nodes"])
    entry_name = cast("str", case["entry"])
    calls_llm_spec = cast("dict[str, Any]", nodes[entry_name]["calls_llm"])
    stores_in = cast("str", calls_llm_spec.get("stores_response_in", "answer"))
    messages_spec = cast("list[dict[str, str]]", calls_llm_spec.get("messages", []))
    messages = [UserMessage(content=m["content"]) for m in messages_spec if m.get("role") == "user"]

    captured_responses: list[Response] = []

    async def ask_body(_s: Any) -> dict[str, str]:
        response = await provider.complete(messages)
        captured_responses.append(response)
        return {stores_in: response.message.content or ""}

    builder = (
        GraphBuilder(state_cls).add_node(entry_name, ask_body).add_edge(entry_name, END).set_entry(entry_name)
    )
    graph = builder.compile()

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    graph.attach_observer(observer)
    try:
        await graph.invoke(state_cls())
    finally:
        await graph.drain()
        observer.shutdown()
        # OpenAIProvider owns an httpx.AsyncClient; closing it releases
        # the connection pool. Matches the convention used by fixture
        # 005 / 038 runners elsewhere in this file.
        await provider.aclose()

    expected = cast("dict[str, Any]", case["expected"])

    # ---- Response.usage assertion
    expected_usage = cast("dict[str, Any] | None", expected.get("response_usage"))
    if expected_usage is not None:
        # The cache-attribute fixtures (040/041/042) are single-LLM-call
        # by shape — one ``ask`` node, one mocked response. A future
        # fixture extending to multi-call would need this assertion to
        # loop over captured_responses rather than indexing [0].
        assert len(captured_responses) == 1, (
            f"response_usage assertion expects exactly one LLM call; captured {len(captured_responses)}"
        )
        actual_usage = captured_responses[0].usage
        for field_name, expected_value in expected_usage.items():
            actual = getattr(actual_usage, field_name)
            assert actual == expected_value, (
                f"response_usage.{field_name}: expected {expected_value!r}, got {actual!r}"
            )

    # ---- LLM span attribute assertions
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1, f"expected exactly one LLM provider span; got {len(llm_spans)}"
    llm_span_attrs = dict(llm_spans[0].attributes or {})

    expected_attrs = cast("dict[str, Any] | None", expected.get("llm_span_attributes"))
    if expected_attrs is not None:
        for attr_name, expected_value in expected_attrs.items():
            actual = llm_span_attrs.get(attr_name)
            assert actual == expected_value, (
                f"llm_span_attributes[{attr_name!r}]: expected {expected_value!r}, got {actual!r}"
            )

    absent_attrs = cast("list[str] | None", expected.get("llm_span_attributes_absent"))
    if absent_attrs is not None:
        for attr_name in absent_attrs:
            assert attr_name not in llm_span_attrs, (
                f"llm_span_attributes_absent: {attr_name!r} unexpectedly present "
                f"with value {llm_span_attrs[attr_name]!r}"
            )


async def _run_typed_event_fixture_case(
    case: Mapping[str, Any],
    *,
    expect_failure: bool = False,
) -> None:
    """Shared runner for the 050-056 simple-shape cases. Parses
    typed_observers, builds the graph, invokes with caller_metadata,
    runs the assertion shapes.

    Failure-path fixtures (053) need access to the calling node's
    ``completed`` NodeEvent (with the wrapped error) to assert error
    category. Unfiltered named collectors capture every event by
    construction; failure-path runs without an unfiltered named
    collector attach a separate ``_AllEventsCollector`` to provide
    the same surface.
    """
    collectors, populate_caller_metadata = _parse_typed_observers(case)
    graph, state_cls, provider = _build_simple_llm_graph(
        case, populate_caller_metadata=populate_caller_metadata
    )
    try:
        extra: _AllEventsCollector | None = None
        if expect_failure and not any(c.filter_event_type is None for c in collectors.values()):
            extra = _AllEventsCollector()
        final, exc = await _invoke_typed_fixture(case, collectors, graph, state_cls, extra_observer=extra)

        expected = cast("dict[str, Any]", case.get("expected") or {})
        if expect_failure:
            assert exc is not None, "failure-path fixture expected an exception"
            _assert_expected_error_if_present(case, exc)
            node_completed = cast("dict[str, Any] | None", expected.get("node_completed_event_carries_error"))
            if node_completed:
                # Source for the assertion: an unfiltered named collector
                # when present, otherwise the failure-path-only extra
                # ``_AllEventsCollector``.
                unfiltered_named = next((c for c in collectors.values() if c.filter_event_type is None), None)
                source = (
                    unfiltered_named.events
                    if unfiltered_named is not None
                    else (extra.events if extra is not None else [])
                )
                _assert_node_completed_event_carries_error(source, node_completed)
        else:
            if final is None:
                raise AssertionError("expected a non-None final state on success path")
        observer_expectations = cast("dict[str, Any]", expected.get("observers") or {})
        for name, expectations in observer_expectations.items():
            collector = collectors.get(name)
            if collector is None:
                raise AssertionError(f"fixture references unknown observer {name!r}")
            _assert_observer_expectations(name, collector, cast("dict[str, Any]", expectations))
        _assert_call_id_invariants(case, collectors)
    finally:
        # _build_simple_llm_graph hands ownership of the provider's
        # httpx.AsyncClient to the runner; close it to release the
        # connection pool.
        await provider.aclose()


async def _run_fixture_050(spec: Mapping[str, Any]) -> None:
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_typed_event_fixture_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_051(spec: Mapping[str, Any]) -> None:
    await _run_fixture_050(spec)


async def _run_fixture_052(spec: Mapping[str, Any]) -> None:
    await _run_fixture_050(spec)


async def _run_fixture_053(spec: Mapping[str, Any]) -> None:
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_typed_event_fixture_case(case, expect_failure=True)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_056(spec: Mapping[str, Any]) -> None:
    await _run_fixture_050(spec)


async def _run_fixture_054(spec: Mapping[str, Any]) -> None:
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_typed_event_fanout_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_055(spec: Mapping[str, Any]) -> None:
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_typed_event_branches_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_typed_event_fanout_case(case: Mapping[str, Any]) -> None:
    """Fan-out case (054): outer fan-out node dispatching one inner
    subgraph per item; each inner subgraph runs an LLM-calling node.
    """
    import json

    import httpx

    from openarmature.graph import END, GraphBuilder
    from openarmature.llm import OpenAIProvider, UserMessage

    from .adapter import build_state_cls

    collectors, populate_caller_metadata = _parse_typed_observers(case)

    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model=_mock_model_from_first_response(case) or "test-model",
        api_key="test",
        transport=httpx.MockTransport(_handler),
        populate_caller_metadata=populate_caller_metadata,
    )

    inner_subgraphs = cast("dict[str, Any]", case["inner_subgraphs"])
    inner_id = next(iter(inner_subgraphs))
    inner_spec = cast("dict[str, Any]", inner_subgraphs[inner_id])
    ask_spec = cast("dict[str, Any]", inner_spec["nodes"]["ask"]["calls_llm"])
    stores_in = cast("str", ask_spec.get("stores_response_in", "score"))
    msgs_spec = cast("list[dict[str, str]]", ask_spec.get("messages", []))
    inner_messages = [UserMessage(content=m["content"]) for m in msgs_spec if m.get("role") == "user"]

    # Inner subgraph state: an ``input`` field receives the fan-out
    # item (a dict per the fixture's products list); ``score`` holds
    # the LLM response. The fan-out wires items_field=products →
    # item_field=input on dispatch, and inner score → outer results
    # on collect.
    inner_state_cls = build_state_cls(
        f"FanOutInner_{inner_id}",
        {
            "input": {"type": "dict", "default": {}},
            stores_in: {"type": "string", "default": ""},
        },
    )

    async def _ask_body(_s: Any) -> dict[str, str]:
        response = await provider.complete(inner_messages)
        return {stores_in: response.message.content or ""}

    inner_builder = (
        GraphBuilder(inner_state_cls).add_node("ask", _ask_body).add_edge("ask", END).set_entry("ask")
    )
    inner_compiled = inner_builder.compile()

    outer_state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    # The outer state needs a transient ``products`` field for the
    # fan-out source plus the existing ``results`` accumulator. The
    # inner ``score`` is a string (LLM output); the fixture YAML
    # declares ``results: list<dict>`` which mismatches the string
    # we collect. Override to plain ``list`` so the collect step
    # type-checks without forcing the inner subgraph to wrap each
    # score in a dict.
    #
    # TODO: revisit this override if a future spec revision adds
    # final-state assertions to the 054 fixture — the override
    # would silently pass against the wrong shape. The current
    # fixture asserts only on observer events.
    outer_fields_extended = dict(outer_state_fields)
    outer_fields_extended["results"] = {"type": "list", "reducer": "append", "default": []}
    outer_fields_extended.setdefault("products", {"type": "list", "default": []})
    outer_state_cls = build_state_cls("FanOutOuter", outer_fields_extended)

    outer_builder = (
        GraphBuilder(outer_state_cls)
        .add_fan_out_node(
            "fan_out_node",
            subgraph=inner_compiled,
            items_field="products",
            item_field="input",
            collect_field=stores_in,
            target_field="results",
        )
        .add_edge("fan_out_node", END)
        .set_entry("fan_out_node")
    )
    outer_compiled = outer_builder.compile()

    initial = cast("dict[str, Any]", case.get("initial_state") or {})
    state_kwargs: dict[str, Any] = {}
    for field_name, field_spec in outer_fields_extended.items():
        if field_name in initial:
            state_kwargs[field_name] = initial[field_name]
        elif "default" in field_spec:
            state_kwargs[field_name] = field_spec["default"]
    state_instance = outer_state_cls(**state_kwargs)

    handles = [outer_compiled.attach_observer(c) for c in collectors.values()]
    try:
        metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
        # ``is not None`` so an explicit empty mapping reaches invoke().
        if metadata is not None:
            await outer_compiled.invoke(state_instance, metadata=metadata)
        else:
            await outer_compiled.invoke(state_instance)
    finally:
        for handle in handles:
            handle.remove()
        await outer_compiled.drain()
        # Release the underlying httpx.AsyncClient connection pool.
        await provider.aclose()

    expected = cast("dict[str, Any]", case.get("expected") or {})
    observer_expectations = cast("dict[str, Any]", expected.get("observers") or {})
    for name, expectations in observer_expectations.items():
        collector = collectors.get(name)
        if collector is None:
            raise AssertionError(f"fixture references unknown observer {name!r}")
        _assert_observer_expectations(name, collector, cast("dict[str, Any]", expectations))


async def _run_typed_event_branches_case(case: Mapping[str, Any]) -> None:
    """Parallel-branches case (055): each named branch runs an
    LLM-calling node.
    """
    import json

    import httpx

    from openarmature.graph import END, BranchSpec, GraphBuilder
    from openarmature.llm import OpenAIProvider, UserMessage

    from .adapter import build_state_cls

    collectors, populate_caller_metadata = _parse_typed_observers(case)

    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model=_mock_model_from_first_response(case) or "test-model",
        api_key="test",
        transport=httpx.MockTransport(_handler),
        populate_caller_metadata=populate_caller_metadata,
    )

    inner_subgraphs = cast("dict[str, Any]", case["inner_subgraphs"])

    def _build_branch(branch_name: str) -> Any:
        inner_spec = cast("dict[str, Any]", inner_subgraphs[branch_name])
        ask_spec = cast("dict[str, Any]", inner_spec["nodes"]["ask"]["calls_llm"])
        stores_in = cast("str", ask_spec.get("stores_response_in", "score"))
        msgs_spec = cast("list[dict[str, str]]", ask_spec.get("messages", []))
        msgs = [UserMessage(content=m["content"]) for m in msgs_spec if m.get("role") == "user"]
        inner_state_cls = build_state_cls(
            f"Branch_{branch_name}",
            {stores_in: {"type": "string", "default": ""}},
        )

        async def _body(_s: Any, _msgs: Any = msgs, _stores: str = stores_in) -> dict[str, str]:
            response = await provider.complete(list(_msgs))
            return {_stores: response.message.content or ""}

        builder = GraphBuilder(inner_state_cls).add_node("ask", _body).add_edge("ask", END).set_entry("ask")
        return builder.compile()

    branches_spec = cast("dict[str, Any]", case["nodes"]["branches_node"]["parallel_branches"])
    branch_names = cast("list[str]", branches_spec["branches"])
    branches_map = {name: BranchSpec(subgraph=_build_branch(name)) for name in branch_names}

    outer_state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    outer_state_cls = build_state_cls("ParallelBranchesOuter", outer_state_fields)

    outer_builder = (
        GraphBuilder(outer_state_cls)
        .add_parallel_branches_node("branches_node", branches=branches_map)
        .add_edge("branches_node", END)
        .set_entry("branches_node")
    )
    outer_compiled = outer_builder.compile()

    initial = cast("dict[str, Any]", case.get("initial_state") or {})
    state_kwargs: dict[str, Any] = {}
    for field_name, field_spec in outer_state_fields.items():
        if field_name in initial:
            state_kwargs[field_name] = initial[field_name]
        elif "default" in field_spec:
            state_kwargs[field_name] = field_spec["default"]
    state_instance = outer_state_cls(**state_kwargs)

    handles = [outer_compiled.attach_observer(c) for c in collectors.values()]
    try:
        metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
        # ``is not None`` so an explicit empty mapping reaches invoke().
        if metadata is not None:
            await outer_compiled.invoke(state_instance, metadata=metadata)
        else:
            await outer_compiled.invoke(state_instance)
    finally:
        for handle in handles:
            handle.remove()
        await outer_compiled.drain()
        # Release the underlying httpx.AsyncClient connection pool.
        await provider.aclose()

    expected = cast("dict[str, Any]", case.get("expected") or {})
    observer_expectations = cast("dict[str, Any]", expected.get("observers") or {})
    for name, expectations in observer_expectations.items():
        collector = collectors.get(name)
        if collector is None:
            raise AssertionError(f"fixture references unknown observer {name!r}")
        _assert_observer_expectations(name, collector, cast("dict[str, Any]", expectations))
