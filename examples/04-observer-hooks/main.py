"""openarmature demo: observer hooks for structured logging + per-call metrics.

**Use case:** Add observability to a small three-stage answer pipeline (an
outer `draft → review → finalize` flow where `review` is its own subgraph)
without changing any node code. A graph-attached console tracer prints
every node-boundary event to stderr; an invocation-scoped metrics
collector tallies counts for THIS specific call.

**Demonstrates:** Observer hooks (spec v0.3 / proposal 0003) — registering
graph-attached and invocation-scoped observers, the `NodeEvent` shape,
namespace chaining across a subgraph boundary, the `drain()` call required
for short-lived processes, and how observers see structured pre/post state
without nodes having to log anything themselves.

Run with:
    uv run python main.py "what year did the moon landing happen"
    uv run python main.py "explain the rise of espresso culture"
    uv run python main.py                          # → uses default question
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Annotated, Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    ExplicitMapping,
    GraphBuilder,
    NodeEvent,
    Observer,
    State,
    append,
)

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ"

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")


# ----------------------------------------------------------------------------
# State schemas
# ----------------------------------------------------------------------------
# The outer `AnswerState` and the subgraph's `ReviewState` overlap on the
# fields we want to flow across the boundary (`draft`, `revised`, `trace`)
# and each adds its own (`question` outside, `critique` inside). Keeping
# the fields aligned by name lets the subgraph's outputs flow back through
# the spec's default field-name matching — except for `draft`, which we
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
# LLM helper (plumbing — not openarmature)
# ----------------------------------------------------------------------------


async def _chat(system: str, user: str) -> str:
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=user),
    ]
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


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
            "criticizing it — what's missing, what's vague, what could be "
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
# A bare async function. Conforms to the `Observer` Protocol structurally —
# any `async def(event: NodeEvent) -> None` works. Graph-attached observers
# fire on every invocation of the compiled graph until removed.


async def console_tracer(event: NodeEvent) -> None:
    """Print one structured line per node boundary to stderr.

    Format: `[step=N] namespace.path → fields_changed_in_this_step`
    On error, format flips to `... ✗ error_category`.
    """
    namespace = ".".join(event.namespace)
    if event.error is not None:
        print(
            f"[step={event.step}] {namespace} ✗ {event.error.category}",
            file=sys.stderr,
        )
        return

    pre = event.pre_state.model_dump()
    post = event.post_state.model_dump() if event.post_state is not None else {}
    # Keys whose values changed during this node — gives the reader a
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
# only for THAT call — graph-attached observers are persistent across
# calls; invocation-scoped observers are per-call.


class InvocationMetrics:
    """Counts events and errors for one invocation; collects unique
    namespaces visited (a quick way to see whether a subgraph ran)."""

    def __init__(self) -> None:
        self.events: int = 0
        self.errors: int = 0
        self.namespaces: set[tuple[str, ...]] = set()

    async def __call__(self, event: NodeEvent) -> None:
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
                # reducer. (Subgraph.critique is discarded — no parent field
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
#   2. Attach graph-level observers (console_tracer here) — these fire on
#      every invoke of this compiled graph.
#   3. For each invoke, optionally pass invocation-scoped observers
#      (metrics here) — they fire only for THAT invocation.
#   4. await drain() before exiting. The graph dispatches events to a
#      background queue; without drain, a short-lived process can exit
#      before the queue's worker has delivered them. In a long-running
#      service this isn't necessary because the event loop keeps running.


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "what year did the moon landing happen"

    graph = build_graph()
    graph.attach_observer(console_tracer)

    metrics = InvocationMetrics()
    try:
        final = await graph.invoke(
            AnswerState(question=question),
            observers=[metrics],
        )
    finally:
        # Required for short-lived processes: invoke() returns when the
        # graph reaches END regardless of whether the observer queue has
        # finished. The try/finally also matters on the failure path —
        # per spec v0.3 §6, the engine dispatches a failure event with
        # `error` populated BEFORE propagating, and that event is exactly
        # what a debugging user would want to see. Without `finally`, an
        # invoke that raises would lose those late events.
        await graph.drain()

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
