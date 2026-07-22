"""Every fixture in the spec submodule parses into a
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

from ._deferral import skip_if_deferred
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
    # Proposal 0046 (chat-prompt rendering, v0.38.0) — fixtures 017-031
    # activate in PR 12 against the prompt-management harness's
    # extended ``chat_template`` + ``placeholders`` directive shapes.
    # ----- v0.12.0 cycle spec-pin bump (v0.38.0 -> v0.45.0) ----------------
    # Proposal 0047 (implicit prefix-cache wire-byte stability, v0.39.0)
    # — fixtures 054-055 require the wire-byte hashing directive shape.
    # Queued for v0.13.0 LLM provider hardening batch.
    "llm-provider/054-openai-wire-byte-stability": ("Proposal 0047 wire-byte stability; queued for v0.13.0"),
    "llm-provider/055-anthropic-wire-byte-stability": (
        "Proposal 0047 wire-byte stability; queued for v0.13.0"
    ),
    # Proposal 0047 also adds cache attribute emission fixtures
    # (observability/040-042). Same v0.13.0 batch.
    "observability/040-llm-cache-attribute-emission": (
        "Proposal 0047 cache attribute emission; queued for v0.13.0"
    ),
    "observability/041-llm-cache-attribute-absence": (
        "Proposal 0047 cache attribute emission; queued for v0.13.0"
    ),
    "observability/042-llm-cache-attribute-reported-zero": (
        "Proposal 0047 cache attribute emission; queued for v0.13.0"
    ),
    # Proposal 0048 (read-symmetric metadata + queryable observer,
    # v0.40.0): fixtures 043-049 introduce new directive shapes the
    # cross-capability parser does not model
    # (``augment_metadata``, ``capture_invocation_metadata_into``,
    # ``capture_queryable_observer_read_into``,
    # ``per_attempt_behavior``, top-level ``queryable_observers`` /
    # ``inner_subgraphs`` / ``caller_metadata`` / ``direct_call`` /
    # ``sequential_invocations`` / ``informative``, plus
    # ``final_state_bounds`` / ``direct_call_result`` /
    # ``per_invocation`` in the expected block). The python
    # implementation already satisfies the §3.4 read contract via
    # ``current_invocation_metadata`` plus the §9 queryable observer
    # pattern (convention-only); v0.12.0 adds ``get_invocation_metadata``
    # as the canonical alias and the §9 documentation. Behavior is
    # pinned by:
    #   - ``tests/unit/test_observability_metadata.py``: read
    #     roundtrip + alias identity + mid-invocation augmentation
    #     visible to next node + outside-invocation empty.
    #   - Predecessor proposal 0034/0040 conformance fixtures
    #     (observability/026, 027, 029, 030, 034 — already
    #     implemented and exercised by the runtime harness) cover
    #     per-async-context scoping under fan-out + parallel-branches.
    #   - Per-attempt scoping under retry is the OPEN gap documented
    #     on the 0048 manifest entry (status = "partial"); pinning
    #     lands in the follow-on retry-metadata-reset PR.
    # Fixture-shape activation is queued for a future PR.
    "observability/043-get-invocation-metadata-roundtrip": (
        "Proposal 0048 fixture-shape models pending; contract pinned by unit tests"
    ),
    "observability/044-get-invocation-metadata-fan-out-scoping": (
        "Proposal 0048 fixture-shape models pending; contract pinned by unit tests"
    ),
    "observability/045-get-invocation-metadata-retry-scoping": (
        "Proposal 0048 fixture-shape models pending; contract pinned by unit tests"
    ),
    "observability/046-get-invocation-metadata-outside-invocation": (
        "Proposal 0048 fixture-shape models pending; contract pinned by unit tests"
    ),
    "observability/047-queryable-observer-pattern": (
        "Proposal 0048 queryable observer fixture-shape models pending; pattern is convention-only"
    ),
    "observability/048-queryable-observer-async-safety": (
        "Proposal 0048 informative async-safety fixture; queryable observer is convention-only"
    ),
    "observability/049-queryable-observer-lifecycle-drop": (
        "Proposal 0048 queryable observer lifecycle fixture-shape models pending; 0054 lands the drain pair"
    ),
    # Proposal 0049 (typed LLM completion event, v0.41.0) — fixtures
    # 050-056 require the ``LlmCompletionEvent`` typed-event directive
    # shape. Queued for v0.13.0 LLM provider hardening batch.
    "observability/050-llm-completion-event-dispatch": (
        "Proposal 0049 typed LLM completion event; queued for v0.13.0"
    ),
    "observability/051-llm-completion-event-type-discrimination": (
        "Proposal 0049 typed LLM completion event; queued for v0.13.0"
    ),
    "observability/052-llm-completion-event-caller-metadata-opt-in": (
        "Proposal 0049 typed LLM completion event; queued for v0.13.0"
    ),
    "observability/053-llm-completion-event-no-event-on-failure": (
        "Proposal 0049 typed LLM completion event; queued for v0.13.0"
    ),
    "observability/054-llm-completion-event-fan-out-index-population": (
        "Proposal 0049 typed LLM completion event; queued for v0.13.0"
    ),
    "observability/055-llm-completion-event-branch-name-population": (
        "Proposal 0049 typed LLM completion event; queued for v0.13.0"
    ),
    "observability/056-llm-completion-event-strict-serial-ordering": (
        "Proposal 0049 typed LLM completion event; queued for v0.13.0"
    ),
    # Proposal 0063 (tool-execution observability, v0.69.0) — the
    # typed-collector fixtures share the ``expected.observers`` shape of
    # 050-056 (which the harness schema models loosely, hence the same
    # parser-deferral); they still RUN via test_observability. The
    # OTel-span (096/097) and Langfuse (098) tool fixtures parse fine and
    # are not deferred.
    "observability/092-tool-call-event-dispatch": (
        "Proposal 0063 typed-event-collector shape; runs in test_observability"
    ),
    "observability/093-tool-call-failed-event-dispatch": (
        "Proposal 0063 typed-event-collector shape; runs in test_observability"
    ),
    "observability/094-tool-call-event-mutual-exclusion": (
        "Proposal 0063 typed-event-collector shape; runs in test_observability"
    ),
    "observability/095-tool-call-id-links-to-llm-request": (
        "Proposal 0063 typed-event-collector shape; runs in test_observability"
    ),
    # Proposal 0057 (LlmCompletionEvent field-set extension, v0.51.0)
    # — fixtures 060-068 share the same ``typed_observers`` directive
    # shape as 050-056 and inherit the same parser-deferral status
    # pending the harness model's typed-event-collector schema work.
    "observability/060-llm-completion-event-input-messages-populated": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/061-llm-completion-event-output-content-populated": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/062-llm-completion-event-request-params-populated": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/063-llm-completion-event-request-extras-populated": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/064-llm-completion-event-active-prompt-populated": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/065-llm-completion-event-active-prompt-null": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/066-llm-completion-event-active-prompt-group-populated": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/067-llm-completion-event-call-id-always-present-and-distinct": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    "observability/068-llm-completion-event-response-model-distinct-from-request": (
        "Proposal 0057 typed event request-side fields; queued for v0.13.0"
    ),
    # Proposal 0058 (LlmFailedEvent typed variant, v0.53.0) — fixtures
    # 069-073 share the same typed_observers / typed_event_collector
    # directive shape as 050-068 and inherit the same parser-deferral
    # status pending the harness model's typed-event-collector schema.
    # The behavior is pinned by unit tests in
    # ``tests/unit/test_llm_provider.py`` (provider emission) plus
    # ``tests/unit/test_observability_otel.py`` and
    # ``test_observability_langfuse.py`` (observer rendering).
    "observability/069-llm-failure-event-dispatch-on-provider-unavailable": (
        "Proposal 0058 typed LLM failure event; harness typed_event_collector schema pending"
    ),
    "observability/070-llm-failure-event-dispatch-on-provider-invalid-request": (
        "Proposal 0058 typed LLM failure event; harness typed_event_collector schema pending"
    ),
    "observability/071-llm-failure-event-call-id-distinct-from-completion-event": (
        "Proposal 0058 typed LLM failure event; harness typed_event_collector schema pending"
    ),
    "observability/072-llm-failure-event-mutual-exclusion-with-completion-event": (
        "Proposal 0058 typed LLM failure event; harness typed_event_collector schema pending"
    ),
    "observability/073-llm-failure-event-error-type-vendor-specific": (
        "Proposal 0058 typed LLM failure event; harness typed_event_collector schema pending"
    ),
    # Proposal 0023 (canonical state reducers, accepted before v0.53.0
    # but not implemented by this release) — fixtures 035-038 introduce
    # the new dict-form reducer directive (``dedupe_append: {}``,
    # ``merge_by_key: {key: 'id'}``); the harness's reducer field still
    # expects a string. Queued for the canonical-state-reducers impl
    # batch alongside its conformance manifest entry.
    "graph-engine/034-reducer-bounded-append": (
        "Proposal 0023 canonical state reducers; impl not yet shipped"
    ),
    "graph-engine/035-reducer-dedupe-append": (
        "Proposal 0023 canonical state reducers; impl not yet shipped"
    ),
    "graph-engine/036-reducer-merge-by-key": ("Proposal 0023 canonical state reducers; impl not yet shipped"),
    "graph-engine/037-reducer-configuration-invalid-max-len": (
        "Proposal 0023 canonical state reducers; impl not yet shipped"
    ),
    "graph-engine/038-reducer-error-non-list-update": (
        "Proposal 0023 canonical state reducers; impl not yet shipped"
    ),
    # Proposal 0050 call-level retry — llm-provider fixtures 056-058
    # assert the per-attempt LLM span surface (N spans +
    # ``openarmature.llm.attempt_index``) that python deferred under
    # decision (b); 0050 is marked ``partial`` accordingly. They stay
    # deferred until a future LlmRetryAttemptEvent cycle implements
    # per-attempt spans. (pipeline-utilities failure-isolation fixtures
    # 058-063 now parse + run via test_pipeline_utilities.py; 061 is
    # execution-deferred there for the attempt_index reconciliation.)
    "llm-provider/056-call-level-retry-transient": (
        "Proposal 0050 call-level retry asserts the deferred per-attempt span surface (0050 partial)"
    ),
    "llm-provider/057-call-level-retry-exhaustion": (
        "Proposal 0050 call-level retry asserts the deferred per-attempt span surface (0050 partial)"
    ),
    "llm-provider/058-call-level-retry-non-transient-no-retry": (
        "Proposal 0050 call-level retry asserts the deferred per-attempt span surface (0050 partial)"
    ),
    # Proposal 0052 (implementation attribution attributes, v0.44.0):
    # observability/059 is the Langfuse-side mapping fixture; uses the
    # ``langfuse_observer_config`` + ``harness_parameterized`` directive
    # shapes the cross-capability parser doesn't model. The python
    # implementation ships in v0.12.0 (manifest 0052 = implemented);
    # behavior is pinned by unit tests in
    # ``tests/unit/test_observability_metadata.py``,
    # ``tests/unit/test_observability_otel.py``, and
    # ``tests/unit/test_observability_langfuse.py``. Fixture-shape
    # activation is queued for a future PR slotted after the upcoming
    # spec conformance-adapter capability ratifies the directive
    # vocabulary.  058 (the OTel-side mapping fixture) parses cleanly
    # against the existing ``span_tree`` + ``attributes_absent``
    # directive shapes and is therefore NOT deferred from parsing;
    # runtime exec is gated by ``_SUPPORTED_FIXTURES`` in
    # ``test_observability.py`` until the harness wires up the
    # canonical-value parameterization.
    "observability/059-implementation-attribution-langfuse": (
        "Proposal 0052 fixture-shape models pending; contract pinned by unit tests"
    ),
    # ----- v0.12.0 cycle spec-pin bump (v0.45.0 -> v0.46.0) -------------
    # Proposal 0054 (per-invocation observer event drain, v0.46.0):
    # six graph-engine fixtures introduce new directive shapes the
    # cross-capability parser does not model (``observers[].behavior``,
    # ``nodes.<name>.invoke_drain_events_for``, the accumulator-observer
    # contract, the per-node ``node_drain_summaries`` /
    # ``node_accumulator_snapshots`` assertion blocks). The python
    # implementation already ships
    # ``CompiledGraph.drain_events_for(invocation_id, *, timeout)`` in
    # v0.12.0 (conformance manifest 0054 = implemented); behavior is
    # pinned by ``tests/unit/test_drain.py`` (basic synchronization,
    # worker-NOT-cancelled-on-timeout, invocation-scope isolation,
    # zero-timeout non-blocking check, unknown id, negative + NaN
    # boundary rejection). Fixture-shape activation is queued for a
    # future PR slotted after the upcoming spec conformance-adapter
    # capability ratifies the directive vocabulary.
    "graph-engine/028-drain-events-for-basic-synchronization": (
        "Proposal 0054 fixture-shape models pending; contract pinned by unit tests"
    ),
    "graph-engine/029-drain-events-for-snapshot-semantic": (
        "Proposal 0054 fixture-shape models pending; contract pinned by unit tests"
    ),
    "graph-engine/030-drain-events-for-timeout": (
        "Proposal 0054 fixture-shape models pending; contract pinned by unit tests"
    ),
    "graph-engine/031-drain-events-for-invocation-scope": (
        "Proposal 0054 fixture-shape models pending; contract pinned by unit tests"
    ),
    "graph-engine/032-drain-events-for-fan-out-coverage": (
        "Proposal 0054 fixture-shape models pending; contract pinned by unit tests"
    ),
    "graph-engine/033-drain-events-for-parallel-branches-coverage": (
        "Proposal 0054 fixture-shape models pending; contract pinned by unit tests"
    ),
    # Proposal 0059 (retrieval-provider / embedding) — the embedding
    # typed-collector fixtures (074-081) share the ``expected.observers``
    # shape of 050-068 / 092-095 (which the harness schema models loosely,
    # hence the same parser-deferral); they still RUN via test_observability.
    # The OTel-span (082) and Langfuse (083, 137) embedding fixtures parse
    # fine against the existing span_tree / langfuse_trace shapes and are NOT
    # deferred (they run in test_observability too).
    "observability/074-embedding-event-dispatch": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    "observability/075-embedding-failure-event-dispatch-on-provider-unavailable": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    "observability/076-embedding-event-mutual-exclusion": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    "observability/077-embedding-event-call-id-distinct": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    "observability/078-embedding-event-input-strings-populated": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    "observability/079-embedding-event-request-params-populated": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    "observability/080-embedding-event-input-count-and-dimensions-populated": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    "observability/081-embedding-event-active-prompt-populated": (
        "Proposal 0059 typed-event-collector shape; runs in test_observability"
    ),
    # Proposal 0060 (retrieval-provider rerank, v0.70.0): the rerank typed-
    # collector fixtures (099-106) share the ``expected.observers`` shape of the
    # embedding typed-collector fixtures (074-081) / the LLM 050-068 (which the
    # harness schema models loosely, hence the same parser-deferral); they still
    # RUN via test_observability. Their ``calls_rerank`` directive now parses
    # (0060a); the parse-deferral is purely the typed-event-collector shape. The
    # OTel-span (107), Langfuse (108), and rerank-metrics (109) fixtures parse
    # fine against the existing span_tree / langfuse_trace / metrics shapes now
    # that calls_rerank is modelled and are NOT deferred here -- they run in
    # test_observability, where the rerank rendering stays deferred until 0060b.
    "observability/099-rerank-event-dispatch": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    "observability/100-rerank-failure-event-dispatch-on-provider-unavailable": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    "observability/101-rerank-event-mutual-exclusion": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    "observability/102-rerank-event-call-id-distinct": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    "observability/103-rerank-event-query-and-documents-populated": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    "observability/104-rerank-event-request-params-populated": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    "observability/105-rerank-event-top-k-and-result-count-populated": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    "observability/106-rerank-event-active-prompt-populated": (
        "Proposal 0060 typed-event-collector shape; runs in test_observability"
    ),
    # Proposal 0075 (callable-branch span, fixture 110 added v0.70.1): the case
    # mixes a graph-style ``expected.final_state`` with the observability
    # ``span_tree``; the cross-capability parser's ObservabilityExpected model
    # forbids final_state. RUNS via _run_fixture_110 in test_observability (the
    # same defer-from-parse-but-runs pattern as fixture 038).
    "observability/110-otel-callable-branch-span": (
        "Cross-capability parser doesn't model final_state + span_tree together; runs in test_observability"
    ),
    # ----- v0.16.0 spec-pin bump (v0.70.1 -> v0.84.0) -----------------------
    # New fixtures whose directive shapes the cross-capability parser doesn't
    # model, for proposals deferred to their own later v0.16.0 PRs. Each runs
    # (or stays accounted) in its capability runner once that proposal lands.
    # Proposal 0062 (LLM completion streaming, v0.71.0) -- the stream-flag
    # llm-provider wire fixtures + the per-chunk LlmTokenEvent observability
    # fixtures. (117 parses cleanly and is accounted in test_observability.)
    "llm-provider/059-openai-streaming-wire": "Proposal 0062 streaming; not implemented",
    "llm-provider/060-stream-unsupported-mapping-rejects": "Proposal 0062 streaming; not implemented",
    "observability/111-llm-token-event-dispatch-on-stream": "Proposal 0062 streaming; not implemented",
    "observability/112-llm-token-event-absent-without-stream": "Proposal 0062 streaming; not implemented",
    "observability/113-streamed-tool-call-reassembles-no-token-events": (
        "Proposal 0062 streaming; not implemented"
    ),
    "observability/114-llm-token-event-then-failure-mid-stream": "Proposal 0062 streaming; not implemented",
    "observability/115-llm-token-event-call-id-links-to-completion": (
        "Proposal 0062 streaming; not implemented"
    ),
    "observability/116-llm-token-event-call-level-retry-one-call-id": (
        "Proposal 0062 streaming; not implemented"
    ),
    "observability/118-llm-token-event-reasoning-delta-kind": "Proposal 0062 streaming; not implemented",
    # Proposal 0075 (callable branches) coverage round-out fixture 119
    # (v0.73.1); the cross-capability parser doesn't model its graph-style
    # shape (cf. 110). Accounted in test_observability.
    "observability/119-otel-callable-branch-attempt-index-under-node-retry": (
        "Proposal 0075 callable-branch coverage round-out; harness shape not modelled"
    ),
    # Proposal 0082 (structured-output failure diagnostics, v0.77.0) --
    # fixtures 120-122 share the same typed_event_collector shape as 069-073
    # that the cross-cap parser doesn't model; the response-side surface IS
    # implemented and runtime-driven in test_observability.
    "observability/120-llm-failure-event-structured-output-truncation": (
        "Proposal 0082 structured-output failure event; harness typed_event_collector schema pending"
    ),
    "observability/121-llm-failure-event-structured-output-schema-mismatch": (
        "Proposal 0082 structured-output failure event; harness typed_event_collector schema pending"
    ),
    "observability/122-llm-failure-event-response-side-null-on-non-body-failure": (
        "Proposal 0082 structured-output failure event; harness typed_event_collector schema pending"
    ),
    # Proposal 0083 (per-prompt token-budget observability, v0.78.0) -- the
    # token_budget directive shape + budget-exceeded expectations.
    "observability/126-token-budget-input-exceeded": "Proposal 0083 token-budget; not implemented",
    "observability/127-token-budget-total-exceeded": "Proposal 0083 token-budget; not implemented",
    "observability/128-token-budget-under-budget-no-warning": "Proposal 0083 token-budget; not implemented",
    "observability/129-token-budget-absent-unchanged": "Proposal 0083 token-budget; not implemented",
    "observability/130-langfuse-token-budget-warning-level": "Proposal 0083 token-budget; not implemented",
    "observability/131-token-budget-on-structured-output-failure": (
        "Proposal 0083 token-budget; not implemented"
    ),
    # Proposal 0084 (nested-fan-out span lineage, v0.81.0) -- the
    # lineage-chain directive shapes. (132 parses cleanly; accounted in
    # test_observability.)
    "observability/133-otel-nested-fan-out-orphan-llm-fallback": (
        "Proposal 0084 nested-fan-out span lineage; not implemented"
    ),
    "observability/134-langfuse-nested-fan-out-parent-resolution": (
        "Proposal 0084 nested-fan-out span lineage; not implemented"
    ),
    # Proposal 0087 (within-node directive execution order, v0.82.0).
    # 135 reuses the augment_metadata / capture_invocation_metadata_into
    # directive shapes (as 043-046 above) the cross-cap parser does not
    # model; runtime-driven via test_observability, so parse stays deferred.
    "observability/135-within-node-directive-execution-order": (
        "Proposal 0087 implemented; reuses the 043-046 augment/capture "
        "directive shapes the cross-cap parser does not model, "
        "runtime-driven in test_observability"
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
    skip_if_deferred(case_id, _DEFERRED_FIXTURES)
    _, path = case
    load_fixture(path)


@pytest.mark.parametrize("case", _FIXTURES, ids=_id)
def test_fixture_round_trips(case: tuple[str, Path]) -> None:
    """Parse → ``model_dump`` → re-parse → equal. Catches dropped
    fields the user intended to use later."""
    case_id = _id(case)
    skip_if_deferred(case_id, _DEFERRED_FIXTURES)
    _, path = case
    parsed = load_fixture(path)
    dumped = parsed.model_dump(exclude_none=True)
    # Re-parse via the same loader path so the discriminator runs again.
    from .harness.loader import _FIXTURE_ADAPTER

    reparsed = _FIXTURE_ADAPTER.validate_python(dumped)
    assert parsed == reparsed, f"round-trip mismatch for {path}"
