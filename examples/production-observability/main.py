"""openarmature demo: production observability with dual OTel +
Langfuse observers, caller hooks for trace.input/output, and the
canonical TimingMiddleware.

**Use case:** A single-turn lunar-mission Q&A endpoint instrumented
the way you'd ship it: BOTH the OTel observer and the Langfuse
observer attached to the same graph, caller hooks deriving
domain-shaped trace.input/output from State (instead of dumping the
raw State object), the built-in TimingMiddleware recording per-node
duration, and multi-tenant caller-supplied metadata (tenantId /
requestId / featureFlag) propagating to both observers in one
``invoke()`` call.

**Demonstrates (mapped to shipped features):**

- Dual observers on one graph (proposal 0031 + the no-double-export
  posture from the README pitch). Both consume the same NodeEvent
  stream independently; nothing in node code knows there are two.
- ``trace_input_from_state`` / ``trace_output_from_state`` caller
  hooks on ``LangfuseObserver`` (proposal 0043 §8.4.1). The hooks
  derive domain dicts (``{"question": ...}`` / ``{"answer": ...,
  "model": ...}``) instead of letting the observer dump the raw
  State.  Production teams use this to keep PII out of trace
  payloads while still surfacing operational signal.
- Built-in ``TimingMiddleware`` from ``openarmature.graph``
  wrapping the respond node.  An ``on_complete`` callback receives
  a ``TimingRecord(node_name, duration_ms, outcome,
  exception_category)`` and prints a one-line timing summary; in
  production the callback would queue to a metrics backend
  (StatsD, Prometheus pushgateway, OTLP metrics exporter).
- ``invoke(metadata={...})`` carrying multi-tenant identifiers.
  The OTel observer emits each entry as an
  ``openarmature.user.<key>`` attribute on every span; the Langfuse
  observer merges them as top-level ``trace.metadata`` keys plus
  per-observation metadata.  One call site, two backends, no
  per-observer wiring.
- ``InMemoryLangfuseClient`` captures the Trace + Observation tree
  in-process so the demo can print it at the end without needing a
  real Langfuse account.  ``InMemorySpanExporter`` does the
  symmetric job for OTel.  Production code swaps in
  ``LangfuseSDKAdapter(Langfuse(...))`` and
  ``BatchSpanProcessor(OTLPSpanExporter(...))`` respectively; the
  observer call surface doesn't change.
- **Queryable accumulator observer + per-invocation drain.** A
  third observer (``LlmUsageAccumulator``) rolls up LLM token
  totals per invocation. A terminal ``persist`` node calls
  ``graph.drain_events_for(state.invocation_id)`` to synchronize on
  the deliver loop, then reads the accumulator's bucket and drops
  it. Without the drain, the bucket would be missing the most
  recent LLM event's tokens (the deliver loop hasn't reached them
  yet). This is the canonical shape for per-invocation cost
  attribution at request scope, replacing the round-trip-through-
  State workarounds that pre-v0.12.0 deployments used. The pattern
  is convention-only at the observer level: ``Observer`` itself
  stays a single-callable protocol; the queryable accumulator just
  exposes its own read methods (``get_bucket`` / ``drop``) that the
  persist node knows about.

Complementary to the observer-hooks example (three observers
side-by-side) and the langfuse-observability example (Langfuse
observer + LangfusePromptBackend prompt linkage).  This example's
headline is the production-shape wiring + per-invocation cost
attribution, not the hook surface or the prompt linkage.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. Host root only.
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't
  authenticate).

Run with:

    uv sync --group examples --all-extras
    LLM_API_KEY=sk-... uv run python examples/production-observability/main.py

(``--all-extras`` pulls in ``opentelemetry-sdk`` for the OTel observer
and ``[langfuse]`` for the Langfuse observer's record types.)
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    NodeEvent,
    NodeException,
    ObserverEvent,
    State,
)
from openarmature.graph.middleware import TimingMiddleware, TimingRecord
from openarmature.llm import (
    LlmProviderError,
    OpenAIProvider,
    RuntimeConfig,
    SystemMessage,
    UserMessage,
)
from openarmature.observability import LLM_NAMESPACE, LlmEventPayload
from openarmature.observability.correlation import current_invocation_id
from openarmature.observability.langfuse import (
    InMemoryLangfuseClient,
    LangfuseObservation,
    LangfuseObserver,
    LangfuseTrace,
)
from openarmature.observability.otel import OTelObserver

# ---------------------------------------------------------------------------
# Provider (lazy-init)
# ---------------------------------------------------------------------------

_provider_instance: OpenAIProvider | None = None


def _get_provider() -> OpenAIProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OpenAIProvider(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("LLM_API_KEY") or None,
        )
    return _provider_instance


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class BriefingState(State):
    question: str
    answer: str = ""
    model_used: str = ""


# ---------------------------------------------------------------------------
# Queryable accumulator observer (per-invocation LLM token rollup)
# ---------------------------------------------------------------------------
# A third observer alongside the OTel + Langfuse pair.  Its job is to
# accumulate per-invocation LLM token usage in memory so a terminal
# persist node can read the totals at request scope (rather than
# round-tripping every count through State).  The Observer protocol is
# a single async callable; the accumulator adds its own read methods
# (``get_bucket`` / ``drop``) on the instance for the persist node to
# consume.  Convention only; openarmature does not ship a base class
# for accumulators.
#
# The accumulator subscribes to every event but only records the LLM-
# namespace ones (provider-emitted ``openarmature.llm.complete`` event
# pair carrying an LlmEventPayload on ``pre_state``).  Per-invocation
# isolation is by ``current_invocation_id()`` — read inside the
# observer callback from the worker's Context, populated by the
# engine at worker create time. Concurrent invocations on one
# observer each get their own bucket.


@dataclass
class _UsageBucket:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0


class LlmUsageAccumulator:
    """Per-invocation LLM token rollup."""

    def __init__(self) -> None:
        # Concurrent invocations on one observer each land in their
        # own bucket.  Production deployments with high concurrency
        # would want an eviction policy on top to bound memory; the
        # demo's persist node drops the bucket explicitly after read.
        self._by_invocation: dict[str, _UsageBucket] = {}

    async def __call__(self, event: ObserverEvent) -> None:
        if not isinstance(event, NodeEvent):
            return
        if event.namespace != LLM_NAMESPACE:
            return
        # Only the completed half of the pair carries the token counts.
        if event.phase != "completed":
            return
        if not isinstance(event.pre_state, LlmEventPayload):
            return
        # NodeEvent doesn't carry invocation_id on the dataclass;
        # observers read it from the ContextVar, which the
        # deliver-loop worker's Context carries from the engine task
        # at worker create-time (per-invocation worker, per-invocation
        # Context).
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        payload = event.pre_state
        bucket = self._by_invocation.setdefault(invocation_id, _UsageBucket())
        if payload.prompt_tokens is not None:
            bucket.prompt_tokens += payload.prompt_tokens
        if payload.completion_tokens is not None:
            bucket.completion_tokens += payload.completion_tokens
        if payload.total_tokens is not None:
            bucket.total_tokens += payload.total_tokens
        bucket.call_count += 1

    # Consumers MUST synchronize on ``drain_events_for`` before
    # calling ``get_bucket`` if completeness matters — without the
    # drain the deliver loop may still hold pending events whose
    # tokens have not been added yet. ``None`` is returned when
    # nothing has been recorded yet (e.g., an invocation with no
    # LLM calls).
    def get_bucket(self, invocation_id: str) -> _UsageBucket | None:
        """Read the accumulated bucket for an invocation."""
        return self._by_invocation.get(invocation_id)

    # The accumulator does NOT auto-drop on
    # ``InvocationCompletedEvent`` — a terminal node legitimately
    # needs to read the bucket BEFORE the invocation completes, and
    # auto-drop would race the read. Callers invoke ``drop()``
    # explicitly after reading.
    def drop(self, invocation_id: str) -> None:
        """Release the bucket for an invocation."""
        self._by_invocation.pop(invocation_id, None)


# Module-level singletons make the persist node closure-free and
# match how ``_provider_instance`` is handled.  In an application
# server, these would live on a request-scoped or app-scoped
# container instead.
_accumulator: LlmUsageAccumulator | None = None
_compiled_graph: CompiledGraph[BriefingState] | None = None


# ---------------------------------------------------------------------------
# Caller hooks for Langfuse trace.input / trace.output
# ---------------------------------------------------------------------------
# Per proposal 0043 §8.4.1, the LangfuseObserver lets callers derive
# domain-shaped trace.input and trace.output from State rather than
# letting the framework dump the raw State object.  The hooks fire
# once per invocation: trace_input_from_state on InvocationStartedEvent,
# trace_output_from_state on InvocationCompletedEvent.  Production
# teams use this to keep PII out of trace payloads while still
# surfacing the operational signal a Langfuse UI viewer needs.


def _trace_input(state: BriefingState) -> dict[str, Any]:
    return {"question": state.question}


def _trace_output(state: BriefingState) -> dict[str, Any]:
    return {"answer": state.answer, "model": state.model_used}


# ---------------------------------------------------------------------------
# TimingMiddleware on_complete callback
# ---------------------------------------------------------------------------
# The canonical TimingMiddleware (openarmature.graph.middleware) wraps
# a node's execution and dispatches a TimingRecord to this callback
# when the chain returns or raises.  Keep callbacks fast: a slow
# callback adds to the apparent node duration since it fires inline
# before the chain's result returns to the engine.
#
# Production deployments queue the record into a metrics exporter
# (Prometheus pushgateway, StatsD, OTLP metrics) rather than print.


async def _emit_timing(record: TimingRecord) -> None:
    extra = ""
    if record.exception_category is not None:
        extra = f" [category={record.exception_category}]"
    print(f"[timing] {record.node_name}: {record.duration_ms:.1f}ms ({record.outcome}){extra}")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def respond(state: BriefingState) -> dict[str, Any]:
    """Single LLM call that answers the briefing question."""
    response = await _get_provider().complete(
        [
            SystemMessage(
                content=(
                    "You are a lunar-mission expert assistant.  Answer "
                    "questions about Apollo and Artemis missions concisely "
                    "and factually.  Keep responses to two or three sentences."
                ),
            ),
            UserMessage(content=state.question),
        ],
        config=RuntimeConfig(temperature=0.0, max_tokens=300),
    )
    return {
        "answer": response.message.content or "",
        # ``response.response_model`` is the version-specific
        # identifier the provider echoes back on the response body
        # (e.g., ``gpt-4o-mini-2024-07-18``); falling back to
        # "unknown" guards against providers that omit the field.
        "model_used": response.response_model or "unknown",
    }


# Terminal node. State is intentionally unused — this node's job is
# to synchronize on the observer deliver loop and report a derived
# rollup, not to read or modify pipeline state.
#
# ``drain_events_for`` blocks until every event dispatched up to this
# point has reached every attached observer. Without it the
# accumulator's bucket may still be missing the most-recent LLM
# event's tokens — the deliver loop hasn't processed them yet when
# the node body runs. Snapshot semantic: the drain awaits only
# events dispatched BEFORE the call (this node's own ``started``
# event included), not events that fire after the call begins
# (notably this node's own ``completed`` event, which only fires
# after the body returns — that's how the call avoids deadlocking
# on itself).
#
# Default timeout is 5.0s; the demo tightens to 2.0s so a stuck
# observer surfaces fast. Production teams pick the threshold against
# their observer SLO. Returning a timeout summary instead of raising
# lets the caller record an SLO breach and proceed with whatever
# data is available, rather than failing the whole invocation.
async def persist(_state: BriefingState) -> dict[str, Any]:
    """Drain the deliver loop, read the LLM-usage rollup, drop the bucket."""
    assert _compiled_graph is not None
    assert _accumulator is not None
    invocation_id = current_invocation_id()
    assert invocation_id is not None
    summary = await _compiled_graph.drain_events_for(invocation_id, timeout=2.0)
    if summary.timeout_reached:
        # Production: emit an SLO-breach metric.  Demo: surface the
        # gap inline so a reader sees what an incomplete drain looks
        # like.
        print(f"[persist] drain incomplete: {summary.undelivered_count} events still pending after 2.0s")
    bucket = _accumulator.get_bucket(invocation_id)
    _accumulator.drop(invocation_id)
    if bucket is None:
        print("[persist] no LLM usage recorded for this invocation")
        return {}
    # In production, this is where you'd write the canonical
    # invocation artifact to durable storage: a JSON record with the
    # answer + per-invocation token cost + caller metadata + trace
    # IDs for cross-system join.  The demo prints the rollup so the
    # pattern is legible.
    print(
        f"[persist] LLM usage: prompt={bucket.prompt_tokens}, "
        f"completion={bucket.completion_tokens}, total={bucket.total_tokens} "
        f"across {bucket.call_count} call(s)"
    )
    return {}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph() -> CompiledGraph[BriefingState]:
    """Two-node graph: respond -> persist -> END.

    TimingMiddleware wraps the respond node so wall-clock duration
    is captured per call.  The persist node runs synchronously after
    respond returns; it drains the deliver loop for the current
    invocation, reads the LLM-usage accumulator's bucket, drops the
    bucket, and prints a cost summary.  No other middleware
    (RetryMiddleware lives in the fan-out-with-retry / parallel-
    branches examples; this one's scope is observability)."""
    timing = TimingMiddleware(node_name="respond", on_complete=_emit_timing)
    return (
        GraphBuilder(BriefingState)
        .add_node("respond", respond, middleware=[timing])
        .add_node("persist", persist)
        .add_edge("respond", "persist")
        .add_edge("persist", END)
        .set_entry("respond")
        .compile()
    )


# ---------------------------------------------------------------------------
# Observer wiring (dual)
# ---------------------------------------------------------------------------
# Both observers consume the same NodeEvent stream independently.
# The Langfuse observer uses an in-memory client so the demo can
# print the captured Trace tree at the end without a real Langfuse
# account.  The OTel observer uses an in-memory span exporter for
# symmetric reasons; production code attaches a BatchSpanProcessor
# with a real OTLP exporter pointed at HyperDX / Honeycomb / Tempo /
# any OTLP-compatible backend.
#
# Caller hooks attach to LangfuseObserver via constructor kwargs.
# ``disable_llm_payload=False`` opts in to capturing the input
# messages + output content on Generation observations so the demo
# output is meaningful; default-True is the spec privacy posture.


def _build_otel_observer(exporter: InMemorySpanExporter) -> OTelObserver:
    # ``disable_llm_payload=False`` opts in to capturing input messages
    # + output content on the LLM-call span (same flag the Langfuse
    # observer below flips for the same reason).  The example's whole
    # point is showing both backends seeing the same logical events;
    # leaving them asymmetric (Langfuse captures the conversation, OTel
    # doesn't) would undercut that.  Default-True is the privacy
    # posture for general use; flip it deliberately when the operator
    # has audited the payload PII risk.
    return OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        resource=Resource.create({"service.name": "openarmature-production-observability"}),
        disable_llm_payload=False,
    )


def _build_langfuse_observer(client: InMemoryLangfuseClient) -> LangfuseObserver:
    return LangfuseObserver(
        client=client,
        disable_llm_payload=False,
        trace_input_from_state=_trace_input,
        trace_output_from_state=_trace_output,
    )


# ---------------------------------------------------------------------------
# Captured-output rendering
# ---------------------------------------------------------------------------
# Production traces land in Honeycomb / HyperDX / Tempo / Phoenix /
# Langfuse Cloud where the backend UI does the rendering work.  For
# the in-process demo we walk the captured records and pretty-print
# enough of the shape that a reader can see what each backend would
# have ingested.


# Invocation-span-only attributes (spec 5.1).  Surface these only on
# the root ``openarmature.invocation`` span line; inner spans don't
# carry them (they're invocation-level constants, not cross-cutting
# 5.6 attributes).
_INVOCATION_SPAN_KEYS = (
    "openarmature.graph.entry_node",
    "openarmature.graph.spec_version",
    "openarmature.implementation.name",
    "openarmature.implementation.version",
)

# Per-node + cross-cutting attributes (5.6 + GenAI semconv).  Surface
# these on inner-node spans only; they propagate to the invocation
# span too but showing them there is redundant once they appear on
# every node line below.
_INNER_SPAN_KEYS = (
    "openarmature.node.name",
    "openarmature.user.tenantId",
    "openarmature.user.requestId",
    "openarmature.user.featureFlag",
    "gen_ai.system",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
)


def _format_otel_spans(spans: list[ReadableSpan]) -> str:
    """One line per span: name, duration, key attributes.

    The ``openarmature.invocation`` root span closes on observer
    ``shutdown()`` and surfaces only its invocation-level
    attributes (spec 5.1 — entry_node, spec_version, implementation
    name + version).  Inner-node spans surface the cross-cutting
    caller metadata + GenAI semconv attributes; printing them on
    the invocation line too would just repeat data shown three
    more times below.
    """
    if not spans:
        return "  (no spans captured)"
    lines: list[str] = []
    # Sort by start time so the timeline reads naturally.
    spans_sorted = sorted(spans, key=lambda s: s.start_time or 0)
    for span in spans_sorted:
        attrs = span.attributes or {}
        keys = _INVOCATION_SPAN_KEYS if span.name == "openarmature.invocation" else _INNER_SPAN_KEYS
        relevant = {k: v for k in keys if (v := attrs.get(k)) is not None}
        duration_ms = 0.0
        if span.start_time is not None and span.end_time is not None:
            duration_ms = (span.end_time - span.start_time) / 1_000_000.0
        attr_str = ", ".join(f"{k}={v!r}" for k, v in relevant.items())
        lines.append(f"  [{span.name}] {duration_ms:.1f}ms  {attr_str}")
    return "\n".join(lines)


def _format_langfuse_trace(trace: LangfuseTrace) -> str:
    """Pretty-print the captured Trace + Observation tree.

    Mirrors what the Langfuse production UI renders for the same
    invocation: trace.input / trace.output (sourced via the caller
    hooks), top-level metadata (caller-supplied + spec keys), and
    the Observation tree underneath.
    """
    lines: list[str] = []
    lines.append(f"Trace id={trace.id}")
    lines.append(f"      name={trace.name!r}")
    lines.append(f"      input={trace.input!r}")
    lines.append(f"      output={trace.output!r}")
    lines.append(f"      metadata={trace.metadata!r}")
    for obs in trace.children_of(None):
        _format_observation(lines, trace, obs, indent="  ")
    return "\n".join(lines)


def _format_observation(
    lines: list[str], trace: LangfuseTrace, obs: LangfuseObservation, indent: str
) -> None:
    lines.append(f"{indent}[{obs.type}] {obs.name!r}")
    if obs.input is not None:
        lines.append(f"{indent}    input={obs.input!r}")
    if obs.output is not None:
        lines.append(f"{indent}    output={obs.output!r}")
    if obs.model is not None:
        lines.append(f"{indent}    model={obs.model!r}")
    if obs.usage is not None:
        lines.append(f"{indent}    usage={obs.usage!r}")
    for child in trace.children_of(obs.id):
        _format_observation(lines, trace, child, indent=indent + "  ")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


async def main() -> None:
    # Pass a question on the command line to override the default,
    # e.g. ``... main.py "When did Apollo 17 splash down?"``.
    question = " ".join(sys.argv[1:]) or "What was the primary objective of Apollo 11?"

    # In-memory captures so the demo can print what BOTH backends
    # would have seen.  Swap with production exporters / clients
    # without touching node or graph code.
    span_exporter = InMemorySpanExporter()
    langfuse_client = InMemoryLangfuseClient()

    # Module-level singletons populated before invoke so the persist
    # node (which is plain-async, no closure) can reach the graph for
    # drain_events_for and the accumulator for the read + drop.
    global _accumulator, _compiled_graph
    _accumulator = LlmUsageAccumulator()
    _compiled_graph = build_graph()
    # Keep the OTel observer reachable so we can ``shutdown()`` it
    # after drain — the root ``openarmature.invocation`` span only
    # closes on shutdown, and the in-memory exporter only surfaces
    # closed spans through ``get_finished_spans()``.  Production
    # deployments do the same dance at process exit.
    otel_observer = _build_otel_observer(span_exporter)
    _compiled_graph.attach_observer(otel_observer)
    _compiled_graph.attach_observer(_build_langfuse_observer(langfuse_client))
    _compiled_graph.attach_observer(_accumulator)
    graph = _compiled_graph

    # Caller-supplied multi-tenant metadata.  Both observers pick
    # the entries up: OTel attaches them as ``openarmature.user.*``
    # span attributes; Langfuse merges them as top-level
    # ``trace.metadata`` keys plus per-observation metadata.
    metadata = {
        "tenantId": "demo-acme",
        "requestId": str(uuid.uuid4()),
        "featureFlag": "v2-canary",
    }

    print("=== openarmature production-observability demo ===")
    print(f"question:    {question}")
    print(f"tenant id:   {metadata['tenantId']}")
    print(f"request id:  {metadata['requestId']}")
    print(f"feature flag: {metadata['featureFlag']}")
    print()

    # Caller-side error boundary at the invoke() seam.  ``exc.__cause__``
    # carries the underlying error via Python's standard exception
    # chain; an ``LlmProviderError`` surfaces its canonical
    # ``.category`` string (``provider_rate_limit``,
    # ``provider_invalid_request``, etc.) so the failure mode is
    # immediately greppable.  Both observers still capture what they
    # saw (the captures get printed below) so a reader sees how each
    # backend records a failed invocation.  Production code would
    # either retry transient categories via ``RetryMiddleware`` on
    # the node, fallback inside the node body, or surface the
    # category to the caller as the example does here.
    final: BriefingState | None = None
    try:
        final = await graph.invoke(
            BriefingState(question=question),
            metadata=metadata,
        )
    except NodeException as exc:
        cause = exc.__cause__
        if isinstance(cause, LlmProviderError):
            category = cause.category
        else:
            category = type(cause).__name__ if cause is not None else "<unknown>"
        print()
        print(f"*** node {exc.node_name!r} failed ({category}): {cause} ***")
        print()
    finally:
        # drain() is required for short-lived processes: invoke()
        # returns when the graph reaches END regardless of whether
        # the observer queue has finished draining.  shutdown() on
        # the OTel observer closes the root ``openarmature.invocation``
        # span so it lands in the exporter alongside the per-node
        # spans; the Langfuse observer has no analog because it
        # writes Trace + Observation entities synchronously through
        # the client.
        await graph.drain()
        otel_observer.shutdown()
        await _get_provider().aclose()

    if final is not None:
        print(f"answer:      {final.answer}")
        print(f"model:       {final.model_used}")
        print()

    # OTel side: pretty-print the captured spans timeline so a
    # reader can see what an OTLP backend would have ingested.
    print("--- captured OTel spans ---")
    print(_format_otel_spans(list(span_exporter.get_finished_spans())))
    print()

    # Langfuse side: pretty-print the Trace + Observation tree.
    # Exactly one Trace per invocation (the observer opens it on
    # the first node event; trace.id equals the invocation_id so
    # cross-system lookups land directly).
    print("--- captured Langfuse trace ---")
    assert len(langfuse_client.traces) == 1, f"expected exactly one trace, got {len(langfuse_client.traces)}"
    trace = next(iter(langfuse_client.traces.values()))
    print(_format_langfuse_trace(trace))


if __name__ == "__main__":
    asyncio.run(main())
