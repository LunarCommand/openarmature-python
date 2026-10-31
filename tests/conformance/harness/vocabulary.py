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

from typing import NamedTuple

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
            "log_records_absent",
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

# Case-level directives whose value is a mapping, or a list of mappings. Each one
# the corpus uses sits in exactly one of the four groups below, so a new one
# fails until it is placed.
#
# Walked already: a position above, a container of positions, or a subgraph
# body, which the subgraph-level check covers.
WALKED_MAPPINGS: frozenset[str] = frozenset(
    {
        "calls",
        "cases",
        "expected",
        "inner_subgraphs",
        "nodes",
        "resume",
        "secondary_calls",
        "subgraph",
        "subgraph_with_idx",
        "subgraphs",
        "tertiary_calls",
    }
)

# Mappings whose keys this check does not police, each with the reason.
OPEN_MAPPINGS: dict[str, str] = {
    "caller_metadata": "metadata keys the caller chooses",
    "expected_backend_state": "keyed by backend name",
    "expected_wire_headers": "HTTP header names",
    "expected_wire_request": "wire request body fields",
    "expected_wire_request_checks": "`<wire field>_absent` checks, read by pattern",
    "initial_state": "state field names, which the case's state schema declares",
    "render_variables": "template variable names",
    "requires_capability": "capability names; the gate rejects one conformance.toml does not declare",
}

# Outside the recognized vocabulary by conformance-adapter section 5's own rule.
OUT_OF_SCOPE_MAPPINGS: dict[str, str] = {
    "invariants": "section 5.9 fixture-specific predicate names",
}


class MappingVocabulary(NamedTuple):
    models: tuple[str, ...] = ()
    extras: frozenset[str] = frozenset()


# Every other case-level mapping is a closed vocabulary and a position of its
# own, named for the directive, or `<directive>[]` for each entry of a list. The
# same two obligations apply as one level down: an unknown key fails, and a key a
# running fixture declares must be read.
CASE_MAPPINGS: dict[str, MappingVocabulary] = {
    "backends[]": MappingVocabulary(("tests.conformance.harness.prompt_management.FixtureBackendSpec",)),
    "call": MappingVocabulary(
        ("tests.conformance.harness.directives.LlmCallSpec",),
        frozenset({"config", "response_schema", "retry", "stream", "tool_choice"}),
    ),
    "checkpointer": MappingVocabulary(extras=frozenset({"fan_out_internal_save_batching", "kind"})),
    "clock_stub": MappingVocabulary(extras=frozenset({"advance_ms_per_call", "type"})),
    "cohere_embedding_provider": MappingVocabulary(extras=frozenset({"api_key", "base_url", "model"})),
    "cohere_rerank_provider": MappingVocabulary(extras=frozenset({"api_key", "base_url", "model"})),
    "crash_injection": MappingVocabulary(extras=frozenset({"after_fan_out_instance", "after_node"})),
    "direct_call": MappingVocabulary(extras=frozenset({"operation"})),
    "edges[]": MappingVocabulary(("tests.conformance.harness.directives.EdgeSpec",)),
    "every_save_assertions": MappingVocabulary(extras=frozenset({"schema_version"})),
    "expected_construction_error": MappingVocabulary(extras=frozenset({"category"})),
    "expected_error": MappingVocabulary(
        extras=frozenset(
            {
                "category",
                "execution_order",
                "fan_out_category",
                "flaky_call_count",
                "message",
                "raised_from",
                "recoverable_state",
                "transient",
            }
        )
    ),
    "expected_message_equal": MappingVocabulary(extras=frozenset({"between", "message_indices"})),
    "expected_shared_prefix": MappingVocabulary(extras=frozenset({"between", "message_index", "prefix"})),
    "expected_wire_bytes_identical": MappingVocabulary(extras=frozenset({"calls"})),
    "first_run_expected": MappingVocabulary(extras=frozenset({"langfuse_trace"})),
    "first_run_expected_error": MappingVocabulary(extras=frozenset({"category", "raised_from"})),
    "graph": MappingVocabulary(
        ("tests.conformance.harness.fixtures.SubgraphDefinition",),
        frozenset({"subgraph", "subgraphs"}),
    ),
    "invocations[]": MappingVocabulary(
        extras=frozenset({"drain", "expected", "initial_state", "name", "session_id"})
    ),
    "invoke": MappingVocabulary(extras=frozenset({"drain"})),
    "invoke_with": MappingVocabulary(extras=frozenset({"resume_invocation"})),
    "jina_embedding_provider": MappingVocabulary(extras=frozenset({"api_key", "base_url", "model"})),
    "jina_rerank_provider": MappingVocabulary(extras=frozenset({"api_key", "base_url", "model"})),
    "label_resolver": MappingVocabulary(
        ("tests.conformance.harness.prompt_management.FixtureLabelResolverSpec",)
    ),
    "langfuse_client": MappingVocabulary(
        extras=frozenset({"accept_shared_provider", "mode", "preexisting_same_key_client", "provider"})
    ),
    "langfuse_observer": MappingVocabulary(
        extras=frozenset(
            {
                "disable_provider_payload",
                "disable_state_payload",
                "payload_byte_cap",
                "trace_input_from_state",
                "trace_output_from_state",
            }
        )
    ),
    "langfuse_observer_config": MappingVocabulary(extras=frozenset({"disable_state_payload"})),
    "manager": MappingVocabulary(("tests.conformance.harness.prompt_management.FixtureManagerSpec",)),
    "middleware": MappingVocabulary(("tests.conformance.harness.directives.MiddlewareConfig",)),
    "migrations[]": MappingVocabulary(extras=frozenset({"from_version", "migrate", "to_version"})),
    "mock_embedding[]": MappingVocabulary(extras=frozenset({"body", "raises", "status"})),
    "mock_llm[]": MappingVocabulary(
        ("tests.conformance.harness.directives.MockResponse",), frozenset({"raises"})
    ),
    "mock_llm_stream[]": MappingVocabulary(extras=frozenset({"wire_attempts"})),
    "mock_provider": MappingVocabulary(
        ("tests.conformance.harness.directives.MockProviderConfig",), frozenset({"capabilities"})
    ),
    "mock_rerank[]": MappingVocabulary(extras=frozenset({"body", "raises", "status"})),
    "observers[]": MappingVocabulary(("tests.conformance.harness.directives.ObserverSpec",)),
    "openai_embedding_provider": MappingVocabulary(
        extras=frozenset({"api_key", "base_url", "chunk_size", "document_prefix", "model", "query_prefix"})
    ),
    "otel_observer": MappingVocabulary(
        extras=frozenset({"attached", "disable_genai_semconv", "disable_provider_payload"})
    ),
    "prompt_backend": MappingVocabulary(extras=frozenset({"prompts", "type"})),
    "provider": MappingVocabulary(extras=frozenset({"genai_system", "type"})),
    "queryable_observers[]": MappingVocabulary(
        extras=frozenset({"kind", "name", "persists_across_invocations"})
    ),
    "runtime_state_subclass": MappingVocabulary(extras=frozenset({"schema_version"})),
    "saved_record_assertions": MappingVocabulary(
        extras=frozenset(
            {
                "completed_positions",
                "fan_out_node_in_completed_positions",
                "fan_out_progress",
                "parent_states_outermost_first",
                "parent_states_present",
                "state",
            }
        )
    ),
    "secondary_manager": MappingVocabulary(
        ("tests.conformance.harness.prompt_management.FixtureManagerSpec",)
    ),
    "seeded_record": MappingVocabulary(
        extras=frozenset(
            {"completed_positions", "fan_out_progress", "parent_states", "schema_version", "state"}
        )
    ),
    "sequential_invocations": MappingVocabulary(
        extras=frozenset({"invocations", "observer_persists_across"})
    ),
    "state": MappingVocabulary(("tests.conformance.harness.directives.StateSchema",)),
    "tei_embedding_provider": MappingVocabulary(
        extras=frozenset(
            {"base_url", "chunk_size", "document_prefix", "input_type_prompt_map", "model", "query_prefix"}
        )
    ),
    "tei_rerank_provider": MappingVocabulary(extras=frozenset({"base_url", "chunk_size", "model"})),
    "tertiary_label_resolver": MappingVocabulary(
        ("tests.conformance.harness.prompt_management.FixtureLabelResolverSpec",)
    ),
    "tertiary_manager": MappingVocabulary(
        ("tests.conformance.harness.prompt_management.FixtureManagerSpec",)
    ),
    "typed_observers[]": MappingVocabulary(
        extras=frozenset(
            {"filter_event_type", "include_caller_metadata", "kind", "name", "retains_arrival_order"}
        )
    ),
}

_BACKOFF = MappingVocabulary(extras=frozenset({"seconds", "type"}))
_MIDDLEWARE_MODELS = (
    "tests.conformance.harness.directives.ErrorRecoveryMiddleware",
    "tests.conformance.harness.directives.FailureIsolationMiddleware",
    "tests.conformance.harness.directives.RetryMiddleware",
    "tests.conformance.harness.directives.ShortCircuitMiddleware",
    "tests.conformance.harness.directives.TimingMiddleware",
    "tests.conformance.harness.directives.TraceRecorderMiddleware",
)

# Inside a node spec, and inside each closed node mapping below it, a key whose
# value is a mapping, or a list of mappings, is placed by its path, such as
# `nodes.*.update`. No node key holds a subgraph body or falls outside the
# vocabulary rule, so a node mapping is either open or closed.
NODE_OPEN_MAPPINGS: dict[str, str] = {
    "nodes.*.augment_metadata": "metadata keys the node chooses",
    "nodes.*.inputs": "state field names, mapped from parent to subgraph",
    "nodes.*.outputs": "state field names, mapped from subgraph to parent",
    "nodes.*.update": "state field names",
    "nodes.*.update_from_field": "target state field names, plus the reserved `multiplier`",
    "nodes.*.update_pure": "state field names",
    "nodes.*.update_pure_from_state": "target state field names",
    "nodes.*.calls_llm.response_schema": "a JSON Schema the call passes to the provider",
    "nodes.*.calls_tool.arguments": "the tool's arguments",
    "nodes.*.fan_out.augment_metadata_from_field": "metadata keys, each mapped to an item field",
    "nodes.*.fan_out.extra_outputs": "state field names",
    "nodes.*.fan_out.inputs": "state field names, mapped from parent to subgraph",
    "nodes.*.flaky.on_success": "state field names",
    "nodes.*.flaky.success_update": "state field names",
    "nodes.*.flaky_by_index.success_compute": "state field names",
    "nodes.*.flaky_instance_only.success_compute": "state field names",
    "nodes.*.flaky_per_index.success_compute": "state field names",
    "nodes.*.flaky_resume_aware.on_success": "state field names",
    "nodes.*.middleware[].degraded_update": "state field names",
    "nodes.*.per_attempt_behavior[].augment_metadata": "metadata keys the attempt chooses",
    "nodes.*.subgraph_call.outputs": "state field names, mapped from subgraph to parent",
    "nodes.*.calls_llm.tools[].parameters": "a JSON Schema for the tool's arguments",
    "nodes.*.calls_llm.tools[].function.parameters": "a JSON Schema for the tool's arguments",
    "nodes.*.calls_tool.mock_tool.returns": "the tool's return value",
    "nodes.*.fan_out.instance_middleware[].degraded_update": "state field names",
    "nodes.*.parallel_branches.branches.*.augment_metadata": "metadata keys the branch chooses",
    "nodes.*.parallel_branches.branches.*.augment_metadata_from_outer_item": (
        "metadata keys, each mapped to an outer item field"
    ),
    "nodes.*.parallel_branches.branches.*.middleware[].degraded_update": "state field names",
    "nodes.*.parallel_branches.branches.*.outputs": "state field names, mapped from branch to parent",
}

# A closed node mapping is a position of its own, held to the same two
# obligations: `<path>` for a mapping, `<path>[]` for each entry of a list, and
# `<path>.*` for each value of a mapping keyed by names the fixture chooses. The
# models are the field types of the mapping that holds it; they do not declare
# every key, and the extras add the rest.
NODE_MAPPINGS: dict[str, MappingVocabulary] = {
    "nodes.*.also_emits_via_global_tracer": MappingVocabulary(
        ("tests.conformance.harness.directives.GlobalTracerSpec",)
    ),
    "nodes.*.calls_embed": MappingVocabulary(
        ("tests.conformance.harness.directives.CallsEmbedSpec",),
        frozenset({"config", "input_from_rendered_prompt", "model"}),
    ),
    "nodes.*.calls_llm": MappingVocabulary(
        ("tests.conformance.harness.directives.CallsLlmSpec",),
        frozenset({"model", "response", "response_schema", "retry", "stream", "tools"}),
    ),
    "nodes.*.calls_llm_from_wrapper": MappingVocabulary(
        ("tests.conformance.harness.directives.CallsLlmFromWrapperSpec",)
    ),
    "nodes.*.calls_rerank": MappingVocabulary(
        ("tests.conformance.harness.directives.CallsRerankSpec",),
        frozenset({"query_from_rendered_prompt"}),
    ),
    "nodes.*.calls_tool": MappingVocabulary(("tests.conformance.harness.directives.CallsToolSpec",)),
    "nodes.*.capture_queryable_observer_read_into": MappingVocabulary(
        extras=frozenset({"method", "observer_name", "state_field"})
    ),
    "nodes.*.emits_log": MappingVocabulary(("tests.conformance.harness.directives.EmitsLogSpec",)),
    "nodes.*.fan_out": MappingVocabulary(
        ("tests.conformance.harness.directives.FanOutSpec",),
        frozenset({"augment_metadata_from_field", "extra_outputs", "inner_subgraph", "inputs"}),
    ),
    "nodes.*.flaky": MappingVocabulary(
        ("tests.conformance.harness.directives.FlakySpec",), frozenset({"category", "fail_count"})
    ),
    "nodes.*.flaky_by_index": MappingVocabulary(("tests.conformance.harness.directives.FlakyByIndexSpec",)),
    "nodes.*.flaky_instance_only": MappingVocabulary(
        ("tests.conformance.harness.directives.FlakyInstanceOnlySpec",)
    ),
    "nodes.*.flaky_per_index": MappingVocabulary(("tests.conformance.harness.directives.FlakyPerIndexSpec",)),
    "nodes.*.flaky_resume_aware": MappingVocabulary(
        ("tests.conformance.harness.directives.FlakyResumeAwareSpec",)
    ),
    "nodes.*.invoke_drain_events_for": MappingVocabulary(
        ("tests.conformance.harness.directives.DrainEventsForSpec",)
    ),
    "nodes.*.middleware[]": MappingVocabulary(_MIDDLEWARE_MODELS),
    "nodes.*.middleware[].backoff": _BACKOFF,
    "nodes.*.middleware[].classifier": MappingVocabulary(extras=frozenset({"transient_categories"})),
    "nodes.*.parallel_branches": MappingVocabulary(
        ("tests.conformance.harness.directives.ParallelBranchesSpec",)
    ),
    "nodes.*.per_attempt_behavior[]": MappingVocabulary(
        extras=frozenset(
            {"attempt", "augment_metadata", "capture_invocation_metadata_into", "raises", "succeeds"}
        )
    ),
    "nodes.*.raises": MappingVocabulary(extras=frozenset({"exception_kind", "message"})),
    "nodes.*.renders_prompt_group": MappingVocabulary(
        extras=frozenset({"active_prompt", "group_name", "members"})
    ),
    "nodes.*.retry_middleware": MappingVocabulary(extras=frozenset({"classifier", "max_attempts"})),
    "nodes.*.subgraph_call": MappingVocabulary(extras=frozenset({"outputs", "subgraph"})),
    "nodes.*.then_drop_for_current_invocation": MappingVocabulary(extras=frozenset({"observer_name"})),
    "nodes.*.calls_llm.messages[]": MappingVocabulary(
        extras=frozenset({"content", "content_repeat", "role", "tool_call_id"})
    ),
    "nodes.*.calls_llm.messages[].content[]": MappingVocabulary(
        extras=frozenset({"detail", "media_type", "source", "text", "type"})
    ),
    "nodes.*.calls_llm.messages[].content[].source": MappingVocabulary(
        extras=frozenset({"base64_data_synthetic", "type"})
    ),
    "nodes.*.calls_llm.messages[].content[].source.base64_data_synthetic": MappingVocabulary(
        extras=frozenset({"bytes"})
    ),
    "nodes.*.calls_llm.messages[].content_repeat": MappingVocabulary(extras=frozenset({"bytes", "char"})),
    "nodes.*.calls_llm.retry": MappingVocabulary(
        extras=frozenset({"backoff", "max_attempts", "per_attempt_override", "reask"})
    ),
    "nodes.*.calls_llm.retry.backoff": _BACKOFF,
    "nodes.*.calls_llm.retry.per_attempt_override[]": MappingVocabulary(
        ("openarmature.llm.response.RuntimeConfig",)
    ),
    "nodes.*.calls_llm.retry.reask": MappingVocabulary(extras=frozenset({"template"})),
    "nodes.*.calls_llm.tools[]": MappingVocabulary(
        extras=frozenset({"description", "function", "name", "parameters", "type"})
    ),
    "nodes.*.calls_llm.tools[].function": MappingVocabulary(
        extras=frozenset({"description", "name", "parameters"})
    ),
    "nodes.*.calls_llm_from_wrapper.messages[]": MappingVocabulary(extras=frozenset({"content", "role"})),
    "nodes.*.calls_tool.mock_tool": MappingVocabulary(("tests.conformance.harness.directives.MockToolSpec",)),
    "nodes.*.calls_tool.mock_tool.raises": MappingVocabulary(extras=frozenset({"error_type", "message"})),
    "nodes.*.fan_out.concurrency": MappingVocabulary(extras=frozenset({"callable", "field"})),
    "nodes.*.fan_out.count": MappingVocabulary(extras=frozenset({"callable", "chunk_size", "field"})),
    # `retry` is 053's entry, `{retry: {max_attempts: 3}}`, where every other
    # fixture writes `{type: retry, ...}`. The runner does not apply it; see the
    # unread ledger.
    "nodes.*.fan_out.instance_middleware[]": MappingVocabulary(
        _MIDDLEWARE_MODELS, frozenset({"catch", "retry"})
    ),
    "nodes.*.fan_out.instance_middleware[].backoff": _BACKOFF,
    "nodes.*.fan_out.instance_middleware[].retry": MappingVocabulary(extras=frozenset({"max_attempts"})),
    "nodes.*.flaky.failure_sequence[]": MappingVocabulary(
        ("tests.conformance.harness.directives.FailureSpec",), frozenset({"cause"})
    ),
    "nodes.*.flaky.failure_sequence[].cause": MappingVocabulary(
        extras=frozenset({"category", "cause", "message"})
    ),
    "nodes.*.parallel_branches.branches.*": MappingVocabulary(
        ("tests.conformance.harness.directives.ParallelBranchSpec",),
        frozenset({"augment_metadata", "augment_metadata_from_outer_item"}),
    ),
    "nodes.*.parallel_branches.branches.*.middleware[]": MappingVocabulary(_MIDDLEWARE_MODELS),
    "nodes.*.parallel_branches.branches.*.middleware[].backoff": _BACKOFF,
    "nodes.*.parallel_branches.branches.*.when": MappingVocabulary(extras=frozenset({"field"})),
}

# Node mappings a runner hands whole to an engine type that forbids unknown
# fields, keyed by path to that type. The type's fields are the vocabulary, and
# handing the mapping over is the read.
NODE_FORWARDED_MAPPINGS: dict[str, str] = {
    "nodes.*.calls_embed.config": "openarmature.retrieval.EmbeddingRuntimeConfig",
    "nodes.*.calls_llm.config": "openarmature.llm.response.RuntimeConfig",
    "nodes.*.calls_rerank.config": "openarmature.retrieval.RerankRuntimeConfig",
}

# The runner functions that hand each forwarded mapping to its type, held both
# ways: every function in an owning runner that builds the type from a mapping is
# listed, and each one passes the mapping whole.
FORWARDING_BUILDERS: dict[str, tuple[str, ...]] = {
    "nodes.*.calls_embed.config": (
        "tests.conformance.test_observability._build_embedding_runtime_config",
        "tests.conformance.test_retrieval_provider._build_config",
    ),
    "nodes.*.calls_llm.config": (
        "tests.conformance.test_observability._build_runtime_config",
        "tests.conformance.test_observability_langfuse._runtime_config_from_spec",
    ),
    "nodes.*.calls_rerank.config": (
        "tests.conformance.test_observability._build_rerank_runtime_config",
        "tests.conformance.test_retrieval_provider._build_rerank_config",
    ),
}

# A path whose mapping has the shape of a registered position, mapped to it.
NODE_MAPPING_ALIASES: dict[str, str] = {
    # A callable branch runs its `call` as the branch's node body.
    "nodes.*.parallel_branches.branches.*.call": "nodes.*",
}

# Nested keys a running fixture declares and no owning runner reads, keyed by
# `(position, key)` to the fixtures declaring them. The ledger may only shrink:
# a new unread key fails, and an entry that stops being unread -- because the key
# was wired, or the fixture stopped running -- fails until it is removed.
KNOWN_UNREAD_NESTED: dict[tuple[str, str], tuple[str, ...]] = {
    # pipeline-utilities: the fixture expects the recovered node's event to carry
    # post_state, which graph-engine section 6 dispatch timing and fixture 061 do
    # not allow. Awaiting a spec ruling.
    ("expected", "expected_observer_event"): ("006-middleware-error-recovery",),
    # llm-provider: the runners assert each case's concrete directives and none
    # of its section 5.9 predicates, so a case is covered by its concrete
    # directives alone until a per-predicate pass reads them.
    ("expected", "invariants"): (
        "056-call-level-retry-transient",
        "057-call-level-retry-exhaustion",
        "058-call-level-retry-non-transient-no-retry",
        "061-call-level-retry-per-attempt-override",
        "062-call-level-reask-success",
        "063-call-level-reask-budget-exhausted",
        "064-call-level-reask-off-by-default",
        "065-call-level-reask-compose-override",
        "066-call-level-reask-transient-interleave",
        "068-usage-counter-malformed-partial",
        "069-usage-counter-malformed-all",
        "072-managed-response-format-collision",
        "074-managed-structural-model-collision",
        "075-managed-declared-scalar-collision",
        "076-managed-declared-stop-merge",
        "081-managed-declared-stop-malformed",
    ),
    # observability: 035 and 036 run only in the Langfuse runner, which checks the
    # trace and not the span attribute this key names.
    ("expected", "invocation_id"): (
        "035-caller-invocation-id-uuid",
        "036-caller-invocation-id-non-uuid",
    ),
    # observability: the detached-trace drivers check their trees by hand rather
    # than reading these keys, so a changed fixture value would go unnoticed.
    ("expected", "parent_trace"): ("008-otel-detached-trace-mode",),
    ("expected", "traces"): (
        "008-otel-detached-trace-mode",
        "058-implementation-attribution-otel",
    ),
    # pipeline-utilities: 065's compile-error case raises while its fan-out node
    # is added, before the edges are read.
    ("graph", "edges"): ("065-fan-out-failure-isolation-degrade-contribution",),
    # graph-engine: each entry's name only labels it; the runner runs the entries
    # in order.
    ("invocations[]", "name"): (
        "024-drain-timeout-clean-state-for-next-invocation",
        "030-drain-events-for-timeout",
        "031-drain-events-for-invocation-scope",
    ),
    # observability: only 084's multi-invocation session case declares these,
    # and that case waits for the sessions capability (proposal 0020).
    ("invocations[]", "initial_state"): ("084-langfuse-session-user-promotion",),
    ("invocations[]", "session_id"): ("084-langfuse-session-user-promotion",),
    # pipeline-utilities: 013's run stops at its expected_error, so the timing
    # record this category feeds is never asserted, and the node raises without
    # a category.
    ("nodes.*", "error_category"): ("013-timing-middleware-failure-path",),
    # observability: 010's driver logs every message at INFO and reads only the
    # message.
    ("nodes.*.emits_log", "level"): ("010-otel-log-correlation",),
    # pipeline-utilities: the checkpoint runner drives 053 through a
    # flaky_per_index seam and does not apply its instance middleware, whose
    # `{retry: {...}}` shape no other fixture uses. Awaiting a spec ruling on the
    # shape.
    ("nodes.*.fan_out.instance_middleware[]", "retry"): (
        "053-checkpoint-fan-out-instance-middleware-retry-resume",
    ),
    ("nodes.*.fan_out.instance_middleware[].retry", "max_attempts"): (
        "053-checkpoint-fan-out-instance-middleware-retry-resume",
    ),
    # pipeline-utilities: the flaky node body raises each entry's category, the
    # default classifier decides whether that is transient, and nothing checks
    # the outcome against this flag.
    ("nodes.*.flaky.failure_sequence[]", "transient"): (
        "007-retry-middleware-success-on-second-attempt",
        "008-retry-middleware-exhausted",
        "009-retry-middleware-non-retryable-passthrough",
        "011-middleware-determinism",
        "014-timing-and-retry-composition",
        "015-retry-per-attempt-observer-events",
        "016-retry-state-aware-classifier",
        "036-parallel-branches-with-branch-middleware-retry",
        "058-failure-isolation-static-degraded",
        "059-failure-isolation-callable-degraded",
        "060-failure-isolation-predicate-filtering",
        "061-failure-isolation-retry-three-piece-composition",
        "062-failure-isolation-on-caught-callback",
        "064-failure-isolation-cause-fidelity-at-wrapping-sites",
        "065-fan-out-failure-isolation-degrade-contribution",
        "066-failure-isolation-cause-chain",
        "068-failure-mock-cause-chain",
        "069-fan-out-degrade-refinements",
        "071-fan-out-degrade-strict-reducer-raise",
        "072-failure-isolation-catch-cause-chain",
        "075-parallel-branches-callable-failure-isolation",
    ),
    # observability: 045's driver maps the abstract classifier to the default
    # retry classifier and reads only max_attempts.
    ("nodes.*.retry_middleware", "classifier"): ("045-get-invocation-metadata-retry-scoping",),
    # observability: 020's driver always builds the OpenAI-compatible provider and
    # reads only genai_system.
    ("provider", "type"): ("020-otel-llm-genai-system-override",),
    # pipeline-utilities: the checkpoint runner checks a saved record's positions,
    # parent states and fan-out progress, not its state.
    ("saved_record_assertions", "state"): (
        "025-checkpoint-resume-from-completed-position",
        "070-crash-injection-after-node-resume",
    ),
    # retrieval-provider: the runner drives each provider directly, with no
    # engine, so no state schema is built.
    ("state", "fields"): (
        "001-embed-positive-control",
        "002-embed-model-binding-error",
        "003-embed-malformed-response-mismatched-vector-count",
        "004-embed-malformed-response-inconsistent-dimensions",
        "005-embed-input-order-preserved",
        "006-rerank-positive-control",
        "007-rerank-model-binding-error",
        "008-rerank-malformed-response-out-of-range-index",
        "009-rerank-malformed-response-duplicate-index",
        "010-rerank-top-k-respected",
        "011-rerank-top-k-violation",
        "012-rerank-per-result-echo-variance",
        "013-embed-input-type-prompt-name",
        "014-rerank-tei-within-batch",
        "015-rerank-tei-chunk-and-stitch",
        "016-rerank-tei-truncate-fail-loud",
        "017-embed-tei-wire",
        "018-rerank-jina-wire",
        "019-rerank-jina-return-documents",
        "020-embed-jina-input-type-task",
        "021-rerank-jina-truncation-fail-loud",
        "022-jina-rate-limit",
        "023-embed-openai-compatible-wire",
        "024-embed-openai-compatible-dimensions",
        "025-embed-openai-compatible-base-url",
        "026-embed-openai-compatible-input-type-noop",
        "027-embed-openai-compatible-client-side-prefix",
        "028-cohere-rerank-wire",
        "029-cohere-return-documents-noop",
        "030-cohere-top-k-top-n",
        "031-cohere-rate-limit",
        "032-embed-cohere-wire",
        "033-embed-cohere-input-type",
        "034-embed-cohere-input-type-rejected",
        "035-embed-cohere-output-dimension",
        "036-embed-cohere-truncate-fail-loud",
        "037-embed-cohere-chunk-and-stitch",
        "038-embed-tei-chunk-and-stitch",
        "039-embed-cohere-embedding-types-merge",
        "040-embed-malformed-usage-not-reported",
        "041-rerank-malformed-usage-not-reported",
        "042-embed-chunk-malformed-usage-not-reported",
        "043-embed-openai-chunk-and-stitch",
        "044-embed-empty-string-response-id-nulled",
        "045-rerank-jina-bare-400-invalid-request",
        "046-embed-cohere-managed-truncate-collision",
        "047-embed-tei-managed-truncate-collision",
        "048-rerank-jina-managed-truncation-collision",
        "049-embed-openai-base64-decode",
        "050-embed-openai-base64-malformed",
        "051-embed-jina-task-collision",
        "052-embed-openai-dimensions-collision",
        "053-embed-cohere-embedding-types-malformed",
        "054-embed-cohere-declared-field-collisions",
        "055-embed-tei-declared-field-collisions",
        "056-embed-tei-prompt-name-not-produced",
    ),
    # retrieval-provider: the runner installs one collector per case and reads
    # only each entry's kind.
    ("typed_observers[]", "name"): (
        "013-embed-input-type-prompt-name",
        "020-embed-jina-input-type-task",
        "026-embed-openai-compatible-input-type-noop",
        "027-embed-openai-compatible-client-side-prefix",
        "033-embed-cohere-input-type",
        "040-embed-malformed-usage-not-reported",
        "041-rerank-malformed-usage-not-reported",
        "042-embed-chunk-malformed-usage-not-reported",
        "044-embed-empty-string-response-id-nulled",
    ),
}
