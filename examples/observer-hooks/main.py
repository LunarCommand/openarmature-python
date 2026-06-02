"""openarmature demo: observer hooks for structured logging, per-call metrics, and OTel spans.

**Use case:** Add observability to a small three-stage answer pipeline (an
outer `draft → review → finalize` flow where `review` is its own subgraph)
without changing any node code. Three observer flavors run side-by-side:

  1. A **graph-attached console tracer** that prints every node-boundary event
     to stderr as a structured one-liner.
  2. An **invocation-scoped metrics collector** that tallies counts for THIS
     specific call.
  3. The **OTel observer** wired to a console span exporter, so the same
     boundaries surface as OpenTelemetry spans.

**Demonstrates:** Observer hooks; registering graph-attached and
invocation-scoped observers, the `NodeEvent` shape, namespace chaining
across a subgraph boundary, the `drain()` call required for short-lived
processes, and how observers see structured pre/post state without nodes
having to log anything themselves. Also covers the OTel mapping: the
`OTelObserver` is just another observer registration; the same events
turn into spans on its private TracerProvider.

LLM calls go through ``openarmature.llm.OpenAIProvider``.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples --all-extras
    cd examples/observer-hooks
    LLM_API_KEY=sk-... uv run python main.py "what year did the moon landing happen"
    LLM_API_KEY=sk-... uv run python main.py "explain why NASA is returning to the moon with Artemis"

(``--all-extras`` pulls in ``opentelemetry-sdk`` for the OTel observer.)

**Production swap: real OTLP exporter (e.g. HyperDX).**

The example wires ``OTelObserver`` to a ``SimpleSpanProcessor`` +
``ConsoleSpanExporter`` so every span prints to stdout. That is fine
for a short-lived demo and wrong for production: synchronous export
blocks each node boundary, and printing is not ingestion. For a real
backend (HyperDX, Honeycomb, Tempo, any OTLP-HTTP collector), swap to
``BatchSpanProcessor`` + ``OTLPSpanExporter`` pointing at your
collector and supplying its auth header. The HyperDX shape::

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    otel_observer = OTelObserver(
        span_processor=BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="https://in-otel.hyperdx.io/v1/traces",
                # HyperDX accepts the API key as a bare ``authorization``
                # value. Other collectors expect ``Bearer <token>``;
                # check your destination's docs. The bracket-form
                # ``os.environ[...]`` is intentional: unlike ``LLM_API_KEY``
                # (which permits None for unauthenticated local servers),
                # a missing HyperDX key would silently send unauthenticated
                # requests, so fail-loud at boot is the right shape.
                headers={"authorization": os.environ["HYPERDX_API_KEY"]},
            )
        ),
        resource=Resource.create({"service.name": "openarmature-demo-answers"}),
    )

Same observer call surface; only the processor + exporter change. The
``OTLPSpanExporter`` lives in the ``opentelemetry-exporter-otlp-proto-http``
package (not in ``[otel]`` extras yet; install it directly while OA
gauges demand). Before short-lived processes exit, call
``await graph.drain()`` (drains the observer's per-invocation event
queue so spans see their ``completed`` events) and then
``otel_observer.force_flush()`` (synchronous; pushes
``BatchSpanProcessor``'s tail through the exporter). The drain + flush
pair ensures the tail lands before teardown.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from typing import Annotated, Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    ExplicitMapping,
    GraphBuilder,
    NodeEvent,
    Observer,
    ObserverEvent,
    State,
    append,
)
from openarmature.llm import OpenAIProvider, SystemMessage, UserMessage
from openarmature.observability.otel import OTelObserver

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


# ----------------------------------------------------------------------------
# State schemas
# ----------------------------------------------------------------------------
# The outer `AnswerState` and the subgraph's `ReviewState` overlap on the
# fields we want to flow across the boundary (`draft`, `revised`, `trace`)
# and each adds its own (`question` outside, `critique` inside). Keeping
# the fields aligned by name lets the subgraph's outputs flow back through
# default field-name matching; except for `draft`, which we
# DO need to project IN. Hence the `inputs={"draft": "draft"}` mapping
# below; absent `outputs` falls back to field-name matching for the way
# back, projecting `revised` and `trace`.


class AnswerState(State):
    """Outer graph: takes a question through draft → review → finalize."""

    question: str
    draft: str = ""
    revised: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


class ReviewState(State):
    """Subgraph: critique a draft, then revise it."""

    draft: str = ""
    critique: str = ""
    revised: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# LLM helper
# ----------------------------------------------------------------------------


async def _chat(system: str, user: str) -> str:
    response = await _get_provider().complete(
        [SystemMessage(content=system), UserMessage(content=user)],
    )
    return (response.message.content or "").strip()


# ----------------------------------------------------------------------------
# Outer-graph nodes
# ----------------------------------------------------------------------------


async def draft_node(s: AnswerState) -> Mapping[str, Any]:
    content = await _chat(
        system="Answer the question in two or three sentences. No preamble.",
        user=s.question,
    )
    return {"draft": content, "trace": ["draft"]}


async def finalize(_s: AnswerState) -> Mapping[str, Any]:
    """Outer endpoint marker. The subgraph's revision is already in
    `revised` via projection; this node just records that the run is done.
    """
    return {"trace": ["finalize"]}


# ----------------------------------------------------------------------------
# Subgraph nodes (review)
# ----------------------------------------------------------------------------


async def critique(s: ReviewState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "Read the draft answer below and write one short paragraph "
            "criticizing it; what's missing, what's vague, what could be "
            "tightened. No preamble."
        ),
        user=s.draft,
    )
    return {"critique": content, "trace": ["critique"]}


async def revise(s: ReviewState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "Rewrite the draft to address the critique. Keep it concise, two or three sentences. No preamble."
        ),
        user=f"Draft:\n{s.draft}\n\nCritique:\n{s.critique}",
    )
    return {"revised": content, "trace": ["revise"]}


def build_review_subgraph() -> CompiledGraph[ReviewState]:
    return (
        GraphBuilder(ReviewState)
        .add_node("critique", critique)
        .add_node("revise", revise)
        .add_edge("critique", "revise")
        .add_edge("revise", END)
        .set_entry("critique")
        .compile()
    )


# ----------------------------------------------------------------------------
# Observer 1: console tracer (graph-attached)
# ----------------------------------------------------------------------------
# A bare async function. Conforms to the `Observer` Protocol
# structurally: any `async def(event: NodeEvent) -> None` works.
# Graph-attached observers fire on every invocation of the compiled
# graph until removed.


async def console_tracer(event: ObserverEvent) -> None:
    """Print one structured line per node boundary to stderr.

    Format: `[step=N] namespace.path → fields_changed_in_this_step`
    On error, format flips to `... ✗ error_category`.

    Mid-invocation ``set_invocation_metadata`` augmentations also
    reach observers as ``MetadataAugmentationEvent`` instances; this
    tracer ignores them.
    """
    if not isinstance(event, NodeEvent):
        return
    namespace = ".".join(event.namespace)
    if event.error is not None:
        print(
            f"[step={event.step}] {namespace} ✗ {event.error.category}",
            file=sys.stderr,
        )
        return

    pre = event.pre_state.model_dump()
    post = event.post_state.model_dump() if event.post_state is not None else {}
    # Keys whose values changed during this node; gives the reader a
    # field-level view of what each node DID without nodes having to log.
    changed = {k: post[k] for k in post if pre.get(k) != post[k]}
    print(f"[step={event.step}] {namespace} → {changed}", file=sys.stderr)


# Static type check: a bare `async def` matches the `Observer` Protocol
# structurally. Catches signature drift at type-check time.
_: Observer = console_tracer


# ----------------------------------------------------------------------------
# Observer 2: per-invocation metrics collector
# ----------------------------------------------------------------------------
# A class with an `async __call__` method. Same Protocol; class-shaped
# observers are useful when you want per-invocation state that isn't a
# global. We pass the instance to `invoke(observers=[...])` so it fires
# only for THAT call; graph-attached observers are persistent across
# calls; invocation-scoped observers are per-call.


class InvocationMetrics:
    """Counts events and errors for one invocation; collects unique
    namespaces visited (a quick way to see whether a subgraph ran)."""

    def __init__(self) -> None:
        self.events: int = 0
        self.errors: int = 0
        self.namespaces: set[tuple[str, ...]] = set()

    async def __call__(self, event: ObserverEvent) -> None:
        if not isinstance(event, NodeEvent):
            return
        self.events += 1
        if event.error is not None:
            self.errors += 1
        self.namespaces.add(event.namespace)


# ----------------------------------------------------------------------------
# Outer graph construction
# ----------------------------------------------------------------------------


def build_graph() -> CompiledGraph[AnswerState]:
    review = build_review_subgraph()
    return (
        GraphBuilder(AnswerState)
        .add_node("draft", draft_node)
        .add_subgraph_node(
            "review",
            review,
            projection=ExplicitMapping[AnswerState, ReviewState](
                # Pass the outer draft IN. Without this, the subgraph would
                # critique an empty string from its own schema default.
                # Leaving `outputs` absent falls back to default field-name
                # matching, which projects subgraph.revised → parent.revised
                # and subgraph.trace → parent.trace via the parent's append
                # reducer. (Subgraph.critique is discarded; no parent field
                # of that name.)
                inputs={"draft": "draft"},
            ),
        )
        .add_node("finalize", finalize)
        .add_edge("draft", "review")
        .add_edge("review", "finalize")
        .add_edge("finalize", END)
        .set_entry("draft")
        .compile()
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
# The shape of an observer-aware run:
#
#   1. Build the graph.
#   2. Attach graph-level observers (console_tracer + OTelObserver here)
#     ; these fire on every invoke of this compiled graph.
#   3. For each invoke, optionally pass invocation-scoped observers
#      (metrics here); they fire only for THAT invocation.
#   4. await drain() before exiting. The graph dispatches events to a
#      background queue; without drain, a short-lived process can exit
#      before the queue's worker has delivered them. In a long-running
#      service this isn't necessary because the event loop keeps running.
#
# The OTel observer is just another observer registration. It speaks the
# same `Observer` Protocol; the difference is what it does with each event:
# it opens / closes spans on a private TracerProvider, threading parent /
# child / fan-out relationships. Wiring it next to the bare async function
# above shows the point: observability backends are pluggable behind one
# uniform hook.


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "what year did the moon landing happen"

    # OTel observer with a console span exporter; every span prints to
    # stdout as a JSON blob when it closes. SimpleSpanProcessor exports
    # synchronously which is right for a short-lived demo; production
    # would use BatchSpanProcessor against a real OTLP exporter. The
    # provider here is PRIVATE to the observer; the global
    # TracerProvider is untouched, so this won't pollute any OTel
    # setup the surrounding application already has.
    #
    # ``resource`` stamps ``service.name`` on every emitted span so
    # downstream backends (Honeycomb, Tempo, HyperDX, Langfuse) can
    # filter by service. Setting it on the observer is the explicit
    # path; the OTel SDK alternative (reading the
    # ``OTEL_SERVICE_NAME`` / ``OTEL_RESOURCE_ATTRIBUTES`` env vars)
    # has to be set BEFORE the observer constructs, which is easy
    # to get wrong.
    #
    # LLM-call spans (one per ``provider.complete()``) carry the
    # OpenTelemetry GenAI semantic conventions automatically:
    # ``gen_ai.system``, ``gen_ai.request.model``,
    # ``gen_ai.response.{model,id,finish_reasons}``,
    # ``gen_ai.usage.{input,output}_tokens``. Cross-vendor backends
    # (Langfuse, Phoenix, Honeycomb's LLM lens) render them
    # correctly without a per-service attribute-mapping shim.
    otel_observer = OTelObserver(
        span_processor=SimpleSpanProcessor(ConsoleSpanExporter()),
        resource=Resource.create({"service.name": "openarmature-demo-answers"}),
    )

    graph = build_graph()
    graph.attach_observer(console_tracer)
    graph.attach_observer(otel_observer)

    metrics = InvocationMetrics()
    try:
        final = await graph.invoke(
            AnswerState(question=question),
            observers=[metrics],
        )
    finally:
        # Required for short-lived processes: invoke() returns when the
        # graph reaches END regardless of whether the observer queue has
        # finished. The try/finally also matters on the failure path:
        # the engine dispatches a failure event with `error` populated
        # BEFORE propagating, and that event is exactly what a debugging
        # user would want to see. Without `finally`, an invoke that
        # raises would lose those late events.
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()

    print()
    print(f"question: {final.question}")
    print()
    print(f"draft:\n{final.draft}")
    print()
    print(f"revised:\n{final.revised}")
    print()
    print("per-invocation metrics:")
    print(f"  events seen:        {metrics.events}")
    print(f"  errors observed:    {metrics.errors}")
    print(f"  unique namespaces:  {len(metrics.namespaces)}")
    print(f"  trace order:        {final.trace}")


if __name__ == "__main__":
    asyncio.run(main())
