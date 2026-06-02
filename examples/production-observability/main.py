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

Complementary to the observer-hooks example (three observers
side-by-side) and the langfuse-observability example (Langfuse
observer + LangfusePromptBackend prompt linkage).  This example's
headline is the production-shape wiring, not the hook surface or
the prompt linkage.

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
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from openarmature.graph import END, GraphBuilder, NodeException, State
from openarmature.graph.middleware import TimingMiddleware, TimingRecord
from openarmature.llm import (
    LlmProviderError,
    OpenAIProvider,
    RuntimeConfig,
    SystemMessage,
    UserMessage,
)
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


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    """Single-node graph: respond -> END.

    TimingMiddleware wraps the respond node so wall-clock duration
    is captured per call.  No other middleware (RetryMiddleware lives
    in the fan-out-with-retry / parallel-branches examples; this
    one's scope is observability)."""
    timing = TimingMiddleware(node_name="respond", on_complete=_emit_timing)
    return (
        GraphBuilder(BriefingState)
        .add_node("respond", respond, middleware=[timing])
        .add_edge("respond", END)
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


def _format_otel_spans(spans: list[ReadableSpan]) -> str:
    """One line per span: name, duration, key attributes."""
    if not spans:
        return "  (no spans captured)"
    lines: list[str] = []
    # Sort by start time so the timeline reads naturally.
    spans_sorted = sorted(spans, key=lambda s: s.start_time or 0)
    for span in spans_sorted:
        attrs = span.attributes or {}
        # Pull a few interesting attributes for the summary; the
        # full set is in span.attributes for any reader who wants it.
        keys_of_interest = (
            "openarmature.node.name",
            "openarmature.user.tenantId",
            "openarmature.user.requestId",
            "openarmature.user.featureFlag",
            "gen_ai.system",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        )
        relevant = {k: v for k in keys_of_interest if (v := attrs.get(k)) is not None}
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

    graph = build_graph()
    graph.attach_observer(_build_otel_observer(span_exporter))
    graph.attach_observer(_build_langfuse_observer(langfuse_client))

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
        # the observer queue has finished draining.
        await graph.drain()
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
