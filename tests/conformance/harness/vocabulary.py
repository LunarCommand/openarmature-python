# Spec: conformance-adapter §5 *Definition homes* and §8.2 (proposal 0120,
# spec v0.113.0). A directive's definition lives in §5 or in a per-directory
# harness note; together those are the recognized vocabulary, and a key outside
# it must be rejected rather than skipped.

"""The case-level directive vocabulary this harness recognizes.

Two obligations sit behind the registry below, and they are different:

* a key nothing recognizes must be rejected, so a directive the spec adds
  cannot arrive and be ignored;
* a key the harness recognizes but never reads must also be rejected, because
  a case declaring it passes with the knob at its default while still counting
  as coverage.

The second is the one that has actually cost us. ``expected_wire_url`` was
declared by a running fixture, documented in that fixture's header as locking
the resolved route, and read by nothing.
"""

from __future__ import annotations

# Directives the harness recognizes beyond what the fixture models declare, at
# the document root and at case level both.
#
# Enumerated rather than derived from the corpus. A list built by walking the
# fixtures is a transcription of what happens to be there, which cannot tell a
# directive that is honoured from one that is merely present -- the same reason
# `_DRIVER_EXPECTED_KEYS` is derived from driver bodies instead.
RECOGNIZED_DIRECTIVES: frozenset[str] = frozenset(
    {
        # Provider and mock wiring.
        "mock_provider",
        "mock_embedding",
        "mock_rerank",
        "mock_llm_stream",
        "provider",
        "calls",
        # Observer construction and opt-outs.
        "typed_observers",
        "queryable_observers",
        "langfuse_observer",
        # Arrives in use at the v0.118.2 pin. The harness read it before any
        # fixture declared it, which is what shipping 0121 ahead of the pin meant.
        "otel_observer",
        "langfuse_observer_config",
        "langfuse_client",
        "caller_global_otel_active",
        "disable_llm_spans",
        "disable_provider_payload",
        "disable_genai_semconv",
        "enable_metrics",
        # Caller-supplied identity and scoping.
        "caller_metadata",
        "caller_invocation_id",
        "session_id",
        # Graph composition beyond the modelled forms.
        "inner_subgraphs",
        "detached_subgraphs",
        "detached_fan_outs",
        "direct_call",
        # Prompt management.
        "manager",
        "backends",
        "prompt_backend",
        "render_variables",
        # Checkpointing, migration and crash injection.
        "seeded_record",
        "migrations",
        "crash_injection",
        "runtime_state_subclass",
        "every_save_assertions",
        "clock_stub",
        # Wire-level assertions (llm-provider and retrieval-provider).
        "expected_wire_headers",
        "expected_wire_request_absent_keys",
        "expected_wire_request_checks",
        "expected_wire_request_count",
        "expected_wire_url",
        # Expected-outcome forms not modelled on `CaseSpec`.
        "expected_compile_warning",
        "expected_construction_error",
        "expected_chain_ambiguity_error",
        "expected_message_equal",
        "expected_shared_prefix",
        "first_run_expected",
        # Capability gating (section 5.5).
        "requires_capability",
        # Document-root directives. retrieval-provider fixtures are validated
        # against no typed model at all -- its runner is a bare yaml.safe_load --
        # so this set is their only vocabulary.
        "mapping",
        "call",
        "openai_embedding_provider",
        "cohere_embedding_provider",
        "jina_embedding_provider",
        "tei_embedding_provider",
        "cohere_rerank_provider",
        "jina_rerank_provider",
        "tei_rerank_provider",
        "expected_wire_bytes_identical",
        "sequential_invocations",
        "informative",
    }
)

# Keys a runner reads by a route a read-position scan cannot see, each paired
# with a machine-checkable justification rather than prose.
#
# `model:<Model>.<field>` -- consumed as an attribute off a typed fixture model,
# so the literal never appears in a subscript. `keylist:<module>.<NAME>` -- the
# harness iterates a collection of key names and subscripts with the loop
# variable, so the literal sits in that collection instead.
#
# Both forms are verified by import: a field that stops existing, or a key that
# leaves the collection, fails rather than sitting here as a stale claim.
READ_VIA: dict[str, str] = {
    "manager": "model:tests.conformance.harness.prompt_management.PromptManagementFixture.manager",
    "backends": "model:tests.conformance.harness.prompt_management.PromptManagementFixture.backends",
    "disable_genai_semconv": "keylist:tests.conformance.test_observability._OTEL_OBSERVER_DIRECTIVE_KEYS",
}


# Recognized keys no runner reads, each tied to what makes that acceptable.
#
# The tie is the point. An exemption that merely named the key would go stale
# silently the moment its fixture started running; naming the deferral instead
# means un-deferring the fixture fails this check until the key is either wired
# or the entry removed. Nothing here may name a fixture that runs.
UNAPPLIED_PENDING_DEFERRAL: dict[str, tuple[str, ...]] = {
    "mock_llm_stream": (
        "111-llm-token-event-dispatch-on-stream",
        "113-streamed-tool-call-reassembles-no-token-events",
        "114-llm-token-event-then-failure-mid-stream",
        "115-llm-token-event-call-id-links-to-completion",
        "116-llm-token-event-call-level-retry-one-call-id",
    ),
    "queryable_observers": (
        "047-queryable-observer-pattern",
        "048-queryable-observer-async-safety",
    ),
    "expected_message_equal": ("032-cross-variable-substring-stability",),
    "expected_shared_prefix": ("032-cross-variable-substring-stability",),
    "expected_wire_bytes_identical": (
        "054-openai-wire-byte-stability",
        "055-anthropic-wire-byte-stability",
    ),
    "sequential_invocations": ("049-queryable-observer-lifecycle-drop",),
    "informative": ("048-queryable-observer-async-safety",),
}

# Recognized keys carried only by a case its driver skips, rather than by a
# deferred fixture. Keyed by `(fixture, case)` so the pair has to stay true:
# the fixture runs, and that one case does not.
UNAPPLIED_PENDING_CASE_DEFERRAL: dict[str, tuple[tuple[str, str], ...]] = {
    "session_id": (("084-langfuse-session-user-promotion", "session_bound_sets_trace_session_id"),),
}


# Capability directories that ship conformance fixtures and have no runner,
# because the capability itself is unimplemented here.
#
# Declared rather than left absent. A directory nothing names reads exactly like
# one nobody got to, and the two need telling apart: `conformance-adapter/`
# arrives at spec v0.114.0 carrying a fixture an adapter is expected to pass, and
# nothing on this side would have noticed it appeared.
UNIMPLEMENTED_CAPABILITIES: dict[str, str] = {
    "harness": "abstract harness contract; no surface in src/openarmature",
    "harness-chat": "chat-loop sub-spec; rests on harness, sessions and suspension",
    "sessions": "proposal 0020 SessionStore / SessionState; scheduled for v0.19.0",
    "suspension": "node-side suspend/resume; no surface in src/openarmature",
}


# Capability directories whose fixtures have arrived at the current pin but whose
# adoption has not landed.
#
# A third state, and it needs to be distinct from UNIMPLEMENTED_CAPABILITIES:
# that dict says the capability does not exist here and nothing is coming, which
# is a claim about the library. This says the fixtures are real, the adoption is
# in flight, and the entry is expected to go away. Folding the two together would
# let work in progress read as a permanent absence.
PENDING_ADOPTION: dict[str, str] = {}


# The nested vocabulary: keys inside an `expected` block (at case level, under
# `resume`, per `invocations[]` entry and per `calls[]` entry), inside `resume`,
# inside each `calls[]` entry, and inside each node spec, subgraph bodies
# included. Spec's v0.17.0 release review found running fixtures asserting
# nothing at exactly this depth.
#
# A position's vocabulary is the fields of the models named here plus the extra
# keys those models do not declare. Models are named as strings and resolved by
# import, so a renamed model or field fails rather than shrinking the set.
NESTED_MODELS: dict[str, tuple[str, ...]] = {
    "expected": (
        "tests.conformance.harness.expectations.GraphEngineExpected",
        "tests.conformance.harness.expectations.PipelineUtilitiesExpected",
        "tests.conformance.harness.expectations.ObservabilityExpected",
        "tests.conformance.harness.expectations.LlmProviderExpected",
        "tests.conformance.harness.prompt_management.FixtureExpectedPerCall",
        "tests.conformance.harness.prompt_management.FixtureExpectedTopLevel",
    ),
    "resume": (),
    "calls[]": (
        "tests.conformance.harness.directives.LlmCallSpec",
        "tests.conformance.harness.prompt_management.FixtureCall",
    ),
    "nodes.*": ("tests.conformance.harness.directives.NodeSpec",),
}

NESTED_EXTRAS: dict[str, frozenset[str]] = {
    "expected": frozenset(
        {
            # graph-engine: the per-invocation drain (proposal 0054).
            "final_accumulator_state",
            "node_accumulator_snapshot_invariants",
            "node_accumulator_snapshots",
            "node_drain_summaries",
            # pipeline-utilities: what a resume re-runs and skips.
            "instances_executed_during_resume",
            "instances_skipped_during_resume",
            "nodes_executed_during_resume",
            "nodes_skipped_during_resume",
            "migrations_run",
            "successful_attempt_index_during_resume",
            # llm-provider.
            "caller_messages_unmodified",
            "llm_spans",
            "no_token_events_emitted",
            "provider_call_count",
            "wire_requests",
            # observability.
            "augment_rejects_at_call_site",
            "direct_call_result",
            "final_state_bounds",
            "first_trace_unchanged",
            "invocation_id",
            "invoke_rejects_at_api_boundary",
            "llm_span_attributes",
            "llm_span_attributes_absent",
            "no_langfuse_observations_emitted",
            "no_spans_emitted",
            "node_completed_event_carries_error",
            "per_invocation",
            "response_usage",
        }
    ),
    "resume": frozenset(
        {
            "caller_invocation_id",
            "expected",
            "expected_chain_ambiguity_error",
            "expected_error",
            "from_first_run",
            "from_seeded_record",
            "invariants",
            "resume_with_modified_items",
        }
    ),
    "calls[]": frozenset(
        {
            "config",
            "expected_wire_request",
            "expected_wire_request_checks",
            "response_schema",
            "retry_middleware",
        }
    ),
    "nodes.*": frozenset(
        {
            "augment_metadata",
            "capture_invocation_metadata_into",
            "capture_queryable_observer_read_into",
            "invoke_drain_events_for",
            "noop",
            "per_attempt_behavior",
            "renders_prompt_group",
            "retry_middleware",
            "subgraph_call",
            "then_assert_bucket_absent_into",
            "then_drop_for_current_invocation",
        }
    ),
}

# Nested keys a running fixture declares and no owning runner reads, keyed by
# `(position, key)` to the fixtures declaring them. The ledger may only shrink:
# a new unread key fails, and an entry that stops being unread -- because the key
# was wired, or the fixture stopped running -- fails until it is removed.
KNOWN_UNREAD_NESTED: dict[tuple[str, str], tuple[str, ...]] = {
    # graph-engine drain (028-033)
    ("expected", "final_accumulator_state"): ("029-drain-events-for-snapshot-semantic",),
    ("nodes.*", "invoke_drain_events_for"): (
        "028-drain-events-for-basic-synchronization",
        "029-drain-events-for-snapshot-semantic",
        "030-drain-events-for-timeout",
        "031-drain-events-for-invocation-scope",
        "032-drain-events-for-fan-out-coverage",
        "033-drain-events-for-parallel-branches-coverage",
    ),
    ("expected", "node_accumulator_snapshot_invariants"): (
        "032-drain-events-for-fan-out-coverage",
        "033-drain-events-for-parallel-branches-coverage",
    ),
    ("expected", "node_accumulator_snapshots"): (
        "028-drain-events-for-basic-synchronization",
        "029-drain-events-for-snapshot-semantic",
        "031-drain-events-for-invocation-scope",
    ),
    ("expected", "node_drain_summaries"): (
        "028-drain-events-for-basic-synchronization",
        "029-drain-events-for-snapshot-semantic",
        "030-drain-events-for-timeout",
        "031-drain-events-for-invocation-scope",
        "032-drain-events-for-fan-out-coverage",
        "033-drain-events-for-parallel-branches-coverage",
    ),
    # pipeline-utilities resume, records and events
    ("expected", "concurrency_invariant"): ("022-fan-out-count-and-concurrency-modes",),
    ("expected", "expected_attempt_events"): ("061-failure-isolation-retry-three-piece-composition",),
    ("expected", "latest_record_assertions"): ("026-checkpoint-record-shape",),
    ("expected", "nodes_executed_during_resume"): (
        "025-checkpoint-resume-from-completed-position",
        "029-checkpoint-subgraph-resume",
        "070-crash-injection-after-node-resume",
    ),
    ("expected", "nodes_skipped_during_resume"): (
        "025-checkpoint-resume-from-completed-position",
        "029-checkpoint-subgraph-resume",
        "070-crash-injection-after-node-resume",
    ),
    ("expected", "observer_events"): (
        "011-middleware-determinism",
        "015-retry-per-attempt-observer-events",
    ),
    # graph-engine observer isolation
    ("expected", "no_propagated_error"): ("015-observer-error-isolation",),
    # observability per-fixture drivers. Some assert the claim by hardcoding it
    # rather than reading the key, so a changed fixture value would go unnoticed.
    ("nodes.*", "also_emits_via_global_tracer"): ("005-otel-llm-provider-span-nested",),
    ("expected", "determinism_check"): ("011-otel-determinism",),
    ("resume", "from_first_run"): ("037-langfuse-trace-input-output",),
    ("expected", "invocation_count"): ("009-otel-correlation-id-cross-cutting",),
    ("expected", "no_edge_spans"): ("004-otel-routing-error-attribution",),
    ("expected", "no_llm_provider_span"): ("005-otel-llm-provider-span-nested",),
    ("expected", "no_openarmature_spans_on_global"): ("005-otel-llm-provider-span-nested",),
    ("expected", "parent_trace"): ("008-otel-detached-trace-mode",),
    ("expected", "span_tree_global"): ("005-otel-llm-provider-span-nested",),
    ("expected", "span_tree_private"): ("005-otel-llm-provider-span-nested",),
    ("nodes.*", "subgraph_call"): ("039-nested-lineage-augmentation",),
    ("expected", "traces"): (
        "008-otel-detached-trace-mode",
        "058-implementation-attribution-otel",
    ),
}
