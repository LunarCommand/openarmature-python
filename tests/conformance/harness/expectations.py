"""Typed ``expected:`` block models — per-capability shapes for the
fixture's assertion payload.

The four capabilities have non-overlapping expected shapes (an
observability fixture wouldn't have ``checkpoint_saves``; a graph-engine
fixture wouldn't have ``span_tree``). Modelling each cleanly catches
fixture authors mixing keys across capabilities, and gives runtime code
in :mod:`runtime` typed access to the assertion payload it needs.

Phase 0 typing depth: TOP-LEVEL keys per capability are exhaustively
typed (catches new directives the spec adds). The nested payload values
underneath (e.g., individual span tree entries, observer event details)
are kept loose as ``list[Any]`` / ``dict[str, Any]`` because the runtime
phases that consume them are the right place to tighten — Phase 1
will type observer-event entries, Phase 5 will type span_tree, etc.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Discriminator, Tag


class _ForbidExtras(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# graph-engine expected block
# ---------------------------------------------------------------------------


class GraphEngineExpected(_ForbidExtras):
    """Expected block for graph-engine fixtures (001–018).

    Top-level keys union'd across every fixture in
    ``spec/graph-engine/conformance/`` at v0.8.0.
    """

    final_state: dict[str, Any] | None = None
    execution_order: list[str] | None = None
    expected_error: dict[str, Any] | None = None
    # Two shapes seen in fixtures:
    # - dict[observer_name, list[event_dict]] — most fixtures
    # - list[event_dict] flat — pipeline-utilities/011 (single-observer)
    # Permissive ``Any`` until Phase 1 (engine retrofit) tightens.
    observer_events: Any = None
    delivery_order: list[dict[str, Any]] | None = None
    observer_event_invariants: dict[str, Any] | None = None
    # 015 — invoke() returns normally; obs_raiser's exceptions surface to
    # warnings rather than propagate.
    no_propagated_error: bool | None = None
    # 018 — registering an observer with `phases: []` raises at
    # registration time per spec §6.
    empty_phases_raises_at_registration: bool | None = None


# ---------------------------------------------------------------------------
# llm-provider expected block
# ---------------------------------------------------------------------------


class LlmProviderResponseAssertion(_ForbidExtras):
    """Assertion payload for a successful ``complete()`` call."""

    message: dict[str, Any] | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw_check: dict[str, Any] | None = None


class LlmProviderRaisesAssertion(BaseModel):
    """Assertion payload for a call that's expected to raise.

    Permissive — fixtures attach assertion-specific knobs like
    ``retry_after_seconds`` (rate-limit fixture) without restructuring
    the type. The runtime in Phase 2 validates the keys it reads.
    """

    model_config = ConfigDict(extra="allow")

    category: str
    message: str | None = None
    cause: dict[str, Any] | None = None


class LlmProviderExpected(_ForbidExtras):
    """Expected block for llm-provider fixtures.

    A call's ``expected:`` carries ``response`` (success path),
    ``raises`` (error path), or ``success`` (boolean for the ``ready``
    operation). Mutually exclusive in practice.
    """

    response: LlmProviderResponseAssertion | None = None
    raises: LlmProviderRaisesAssertion | None = None
    success: bool | None = None


# ---------------------------------------------------------------------------
# pipeline-utilities expected block
# ---------------------------------------------------------------------------


class PipelineUtilitiesExpected(_ForbidExtras):
    """Expected block for pipeline-utilities fixtures (001–031).

    Spans middleware, fan-out, and checkpointing; the union is wide.
    """

    final_state: dict[str, Any] | None = None
    execution_order: list[str] | None = None
    expected_error: dict[str, Any] | None = None
    # Two shapes seen in fixtures:
    # - dict[observer_name, list[event_dict]] — most fixtures
    # - list[event_dict] flat — pipeline-utilities/011 (single-observer)
    # Permissive ``Any`` until Phase 1 (engine retrofit) tightens.
    observer_events: Any = None
    observer_event_invariants: dict[str, Any] | None = None
    # Singular form used by 015 — assert one specific event shape.
    expected_observer_event: dict[str, Any] | None = None
    # Checkpointing fixtures (024–031).
    checkpoint_saves: list[dict[str, Any]] | None = None
    latest_record_assertions: dict[str, Any] | None = None
    invariants: dict[str, Any] | None = None
    # Fan-out fixtures (017–023).
    concurrency_invariant: dict[str, Any] | None = None
    # Timing middleware fixtures (012–014).
    timing_records: list[dict[str, Any]] | None = None
    # Trace recorder middleware fixtures (001–003). Two shapes:
    # - dict[recorder_name, list[record]] when multiple recorders (001).
    # - list[record] flat when a single recorder.
    trace_records: Any = None


# ---------------------------------------------------------------------------
# observability expected block
# ---------------------------------------------------------------------------


class ObservabilityExpected(_ForbidExtras):
    """Expected block for observability fixtures (001–011).

    Span trees come in three flavours depending on what the fixture
    verifies:

    - ``span_tree`` — single-trace, single-exporter (the common case).
    - ``span_tree_private`` / ``span_tree_global`` — dual-exporter
      isolation case (fixture 005 + the global-tracer companion).
    - ``traces`` — multi-trace case (detached subgraphs/fan-outs in
      fixture 008 produce multiple root traces).
    """

    span_tree: list[dict[str, Any]] | None = None
    span_tree_private: list[dict[str, Any]] | None = None
    span_tree_global: list[dict[str, Any]] | None = None
    traces: list[dict[str, Any]] | None = None
    parent_trace: dict[str, Any] | None = None
    detached_trace_count: int | None = None
    # Logs Bridge (fixture 010).
    log_records: list[dict[str, Any]] | None = None
    # Negative assertions used across 005, 008, 010, 011.
    no_global_provider_spans: bool | None = None
    no_openarmature_spans_on_global: bool | None = None
    no_edge_spans: bool | None = None
    no_llm_provider_span: bool | None = None
    # Invariants block (fixture 011 determinism).
    invariants: dict[str, Any] | None = None
    determinism_check: dict[str, Any] | None = None
    # Multi-invocation fixtures (009 cross-cutting, 011 determinism).
    invocation_count: int | None = None


# ---------------------------------------------------------------------------
# Discriminated union — pick by which capability-specific keys appear
# ---------------------------------------------------------------------------


_GRAPH_ENGINE_KEYS = frozenset(
    {
        "no_propagated_error",
        "empty_phases_raises_at_registration",
    }
)
_LLM_PROVIDER_KEYS = frozenset({"response", "raises", "success"})
_PIPELINE_UTILITIES_KEYS = frozenset(
    {
        "checkpoint_saves",
        "latest_record_assertions",
        "concurrency_invariant",
        "timing_records",
        "trace_records",
        "expected_observer_event",
    }
)
_OBSERVABILITY_KEYS = frozenset(
    {
        "span_tree",
        "span_tree_private",
        "span_tree_global",
        "traces",
        "parent_trace",
        "detached_trace_count",
        "log_records",
        "no_global_provider_spans",
        "no_openarmature_spans_on_global",
        "no_edge_spans",
        "no_llm_provider_span",
        "determinism_check",
        "invocation_count",
    }
)


def _discriminate_expected(
    value: Any,
) -> Literal["graph_engine", "llm_provider", "pipeline_utilities", "observability"]:
    """Pick the per-capability expected shape from the dict's keys.

    Capability-specific keys take priority. For shape-overlap (e.g.
    ``final_state`` is in both graph-engine and pipeline-utilities), the
    fixture's location on disk is the authoritative tag — but expected
    blocks themselves don't know that, so we discriminate on the keys
    that ARE distinctive and fall back to graph-engine for plain
    ``final_state``-only fixtures.

    Pydantic invokes this callable on both the validation and
    serialization paths. On serialization the value is already one of the
    concrete variants, so route by ``isinstance`` first; otherwise the
    dump path falls through to ``graph_engine`` and warns that the
    serialized variant doesn't match the union's first arm.
    """
    if isinstance(value, GraphEngineExpected):
        return "graph_engine"
    if isinstance(value, LlmProviderExpected):
        return "llm_provider"
    if isinstance(value, PipelineUtilitiesExpected):
        return "pipeline_utilities"
    if isinstance(value, ObservabilityExpected):
        return "observability"
    if not isinstance(value, dict):
        return "graph_engine"
    keys: set[str] = {str(k) for k in cast("dict[str, Any]", value)}
    if keys & _LLM_PROVIDER_KEYS and not keys & _GRAPH_ENGINE_KEYS:
        return "llm_provider"
    if keys & _OBSERVABILITY_KEYS:
        return "observability"
    if keys & _PIPELINE_UTILITIES_KEYS:
        return "pipeline_utilities"
    return "graph_engine"


ExpectedBlock = Annotated[
    Annotated[GraphEngineExpected, Tag("graph_engine")]
    | Annotated[LlmProviderExpected, Tag("llm_provider")]
    | Annotated[PipelineUtilitiesExpected, Tag("pipeline_utilities")]
    | Annotated[ObservabilityExpected, Tag("observability")],
    Discriminator(_discriminate_expected),
]


__all__ = [
    "ExpectedBlock",
    "GraphEngineExpected",
    "LlmProviderExpected",
    "LlmProviderRaisesAssertion",
    "LlmProviderResponseAssertion",
    "ObservabilityExpected",
    "PipelineUtilitiesExpected",
]
