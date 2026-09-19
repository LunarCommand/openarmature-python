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
PENDING_ADOPTION: dict[str, str] = {
    "conformance-adapter": (
        "proposal 0123. Fixture 001 arrived with spec v0.114.0 and spec ruled it in scope, "
        "reported as adapter conformance distinct from the five runtime capabilities. "
        "The runner lands with the 0123 adoption"
    ),
}
