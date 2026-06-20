"""openarmature demo: enrich a lunar-mission news article with several
independent analyses running concurrently.

**Use case:** Given a news article about a lunar mission, produce
side-by-side outputs: a one-sentence summary, an overall sentiment label,
a short list of topic tags, an estimated reading time, and an optional
translation. The analyses don't depend on each other, so dispatch them
in parallel. Some are full subgraphs with their own state schema
(summary, sentiment, topics); one is a lightweight inline function over
the shared article (reading time); and one runs only when asked for
(translation). That mix is exactly the shape parallel-branches is for.

Where fan-out (the fan-out-with-retry example) runs N copies of ONE subgraph against
different inputs, parallel-branches runs M heterogeneous branches
against the same input. Different schemas, different middleware,
different topologies per branch; one dispatch.

**What's interesting in the implementation:**

- ``GraphBuilder.add_parallel_branches_node`` registers M
  ``BranchSpec``s under named keys (``summary``, ``sentiment``,
  ``topics``, ``reading_time``, ``translation`` here). A branch gives
  its work as either a compiled ``subgraph`` (with input/output
  projection) or an inline ``call``, and may carry its own middleware
  and an optional ``when`` predicate.
- A branch can be a whole subgraph OR a single function. The summary,
  sentiment, and topics branches are ``subgraph=...`` branches, each
  with its own state schema and a projection mapping the parent's
  ``article`` into the branch's input field. The reading-time branch is
  a ``call=...`` branch: an inline async function over the parent state
  that returns parent fields directly, no subgraph or projection. Reach
  for ``call`` when a leg is really just "run this one function."
- The translation branch carries a ``when`` predicate and runs only
  when ``target_language`` is set on the state. With it unset (the
  default here) the branch is skipped entirely: no dispatch, no
  contribution, no observer events. Flip ``target_language`` to run it.
- The branches have DIFFERENT state schemas. The summary subgraph's
  state has a ``summary`` field; the sentiment subgraph's has a
  ``label`` field; the topics subgraph's has a ``tags`` list. Each is
  scoped to its job. The projection mapping translates the parent's
  ``article`` into each branch's input field name.
- The sentiment branch wraps its subgraph in ``RetryMiddleware`` to
  show per-branch middleware composition. The other branches run bare.
  Per-branch middleware is heterogeneous; branch A may have
  retry + timing, branch B nothing, branch C something custom.
- Branch insertion order determines fan-in order: when two branches
  contribute to the same parent field, the parent's reducer applies
  them in the order the branches were declared in the ``branches``
  mapping (not in completion order). The three branches here write
  disjoint parent fields, so the order doesn't affect the result,
  but the property holds and would matter if they overlapped.
- A ``branch_attribution_observer`` reads ``NodeEvent.branch_name``
  on inner-node events. ``branch_name`` is populated only for
  events INSIDE a branch's subgraph; outermost nodes (receive,
  enrich, present) have ``branch_name=None``. This is the
  per-event attribution that lets observability backends route
  metrics / spans by branch.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/parallel-branches
    LLM_API_KEY=sk-... uv run python main.py
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from openarmature.graph import (
    END,
    BranchSpec,
    CompiledGraph,
    GraphBuilder,
    NodeEvent,
    ObserverEvent,
    State,
    append,
)
from openarmature.graph.middleware import (
    RetryConfig,
    RetryMiddleware,
    deterministic_backoff,
)
from openarmature.llm import OpenAIProvider, SystemMessage, UserMessage

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


async def _chat(system: str, user: str) -> str:
    response = await _get_provider().complete(
        [SystemMessage(content=system), UserMessage(content=user)],
    )
    return (response.message.content or "").strip()


# ---------------------------------------------------------------------------
# Sample article. A real app would pull this from a feed, a queue, an API.
# ---------------------------------------------------------------------------

ARTICLE = (
    "NASA's Artemis II crew capsule Integrity splashed down in the Pacific "
    "Ocean this evening, ending a ten-day flight that carried four "
    "astronauts on a free-return trajectory around the Moon and back. The "
    "flight was the first crewed mission beyond low Earth orbit since "
    "Apollo 17 in 1972. Agency officials described the result as a "
    "successful test of the Orion spacecraft's deep-space systems and "
    "cautioned that the Artemis III surface-landing timeline remains "
    "dependent on the on-ground refurbishment cadence and lander-system "
    "milestones. Even so, the splashdown was greeted with relief by "
    "partner space agencies and renewed calls in policy circles for "
    "sustained federal funding of the lunar return program."
)


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class ArticleState(State):
    """Outer: an article goes in, the enrichment fields come out.

    ``target_language`` is an input flag: set it to enable the optional
    translation branch (a ``when`` predicate gates that branch on it).
    """

    article: str = ""
    target_language: str = ""
    summary: str = ""
    sentiment: str = ""
    topics: list[str] = Field(default_factory=list)
    reading_time_seconds: int = 0
    translation: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


class SummaryState(State):
    """Summary branch: one-sentence rewrite of the article."""

    text: str = ""
    summary: str = ""


class SentimentState(State):
    """Sentiment branch: overall tone of the article."""

    text: str = ""
    label: str = ""


class TopicsState(State):
    """Topics branch: a short list of topic tags."""

    text: str = ""
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Branch subgraphs; each is one node, but each has its own scope.
# ---------------------------------------------------------------------------


async def write_summary(s: SummaryState) -> Mapping[str, Any]:
    content = await _chat(
        system=("Summarize the article in one tight sentence (~20 words). No preamble, no quoting."),
        user=s.text,
    )
    return {"summary": content}


async def classify_sentiment(s: SentimentState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "Classify the overall sentiment of the article. Reply with ONE "
            "word from this set: positive, negative, neutral, mixed. "
            "Lowercase, no punctuation."
        ),
        user=s.text,
    )
    label = content.strip().lower().strip(".")
    return {"label": label}


async def extract_topics(s: TopicsState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "Extract three short topic tags for the article. Reply with "
            "exactly three lines, one tag per line, no numbering or bullets. "
            "Tags should be 1-3 words each."
        ),
        user=s.text,
    )
    tags = [line.strip(" -*•\t") for line in content.splitlines() if line.strip()][:3]
    return {"tags": tags}


def build_summary_subgraph() -> CompiledGraph[SummaryState]:
    return (
        GraphBuilder(SummaryState)
        .add_node("write_summary", write_summary)
        .add_edge("write_summary", END)
        .set_entry("write_summary")
        .compile()
    )


def build_sentiment_subgraph() -> CompiledGraph[SentimentState]:
    return (
        GraphBuilder(SentimentState)
        .add_node("classify_sentiment", classify_sentiment)
        .add_edge("classify_sentiment", END)
        .set_entry("classify_sentiment")
        .compile()
    )


def build_topics_subgraph() -> CompiledGraph[TopicsState]:
    return (
        GraphBuilder(TopicsState)
        .add_node("extract_topics", extract_topics)
        .add_edge("extract_topics", END)
        .set_entry("extract_topics")
        .compile()
    )


# ---------------------------------------------------------------------------
# Callable branches; each is just a function over the parent state. No
# subgraph, no state schema, no projection: the function reads the article
# off the parent state and returns parent-shaped fields directly.
# ---------------------------------------------------------------------------


async def estimate_reading_time(s: ArticleState) -> Mapping[str, Any]:
    """A lightweight branch that needs no LLM and no subgraph: estimate
    reading time from the word count at ~200 words per minute. The kind
    of leg that the subgraph form would be too heavy for."""
    words = len(s.article.split())
    return {"reading_time_seconds": round(words / 200 * 60)}


async def translate_article(s: ArticleState) -> Mapping[str, Any]:
    """An LLM branch that runs only when a target language is requested
    (gated by a ``when`` predicate on its BranchSpec)."""
    content = await _chat(
        system=(f"Translate the article into {s.target_language}. Output only the translation, no preamble."),
        user=s.article,
    )
    return {"translation": content}


# ---------------------------------------------------------------------------
# Outer graph
# ---------------------------------------------------------------------------


async def receive(s: ArticleState) -> Mapping[str, Any]:
    del s
    return {"trace": ["receive"]}


async def present(s: ArticleState) -> Mapping[str, Any]:
    del s
    return {"trace": ["present"]}


async def branch_attribution_observer(event: ObserverEvent) -> None:
    """Print which branch each event came from.

    NodeEvent carries ``branch_name`` on every event from inside a
    parallel-branches branch: the inner nodes of a subgraph branch, and
    the single branch-unit event of a callable branch. It's the per-event
    attribution that says "this came from branch X." Outermost-graph nodes
    (receive, enrich, present) carry no branch_name. A skipped ``when``
    branch emits no events, so it never appears here. The observer skips
    events with no branch attribution and prints ``(branch=…) node_name``
    for the rest.
    """
    if not isinstance(event, NodeEvent):
        return
    if event.branch_name is None or event.phase != "started":
        return
    print(f"  [observer] (branch={event.branch_name}) node {event.node_name!r} started")


def build_graph() -> CompiledGraph[ArticleState]:
    summary = build_summary_subgraph()
    sentiment = build_sentiment_subgraph()
    topics = build_topics_subgraph()

    # Only the sentiment branch retries. Realistic in production: the
    # classification call is short and cheap to retry, but you may not want
    # the same policy on a longer summarize call (where a retry doubles
    # cost) or on a topic-extract that has different transient profile.
    sentiment_retry = RetryMiddleware(
        RetryConfig(
            max_attempts=3,
            backoff=deterministic_backoff(0.2),
        )
    )

    return (
        GraphBuilder(ArticleState)
        .add_node("receive", receive)
        .add_parallel_branches_node(
            "enrich",
            branches={
                "summary": BranchSpec(
                    subgraph=summary,
                    inputs={"text": "article"},
                    outputs={"summary": "summary"},
                ),
                "sentiment": BranchSpec(
                    subgraph=sentiment,
                    inputs={"text": "article"},
                    outputs={"sentiment": "label"},
                    middleware=(sentiment_retry,),
                ),
                "topics": BranchSpec(
                    subgraph=topics,
                    inputs={"text": "article"},
                    outputs={"topics": "tags"},
                ),
                # A callable branch: an inline function over the parent
                # state, no subgraph / projection. Its return value IS the
                # contribution.
                "reading_time": BranchSpec(call=estimate_reading_time),
                # A conditional callable branch: skipped at dispatch unless
                # a target language is requested.
                "translation": BranchSpec(
                    call=translate_article,
                    when=lambda s: bool(s.target_language),
                ),
            },
        )
        .add_node("present", present)
        .add_edge("receive", "enrich")
        .add_edge("enrich", "present")
        .add_edge("present", END)
        .set_entry("receive")
        .compile()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    graph = build_graph()
    graph.attach_observer(branch_attribution_observer)

    print("=" * 72)
    print("Lunar-mission article enrichment; independent analyses in parallel")
    print("=" * 72)
    print()
    print(f"Article ({len(ARTICLE)} chars):")
    print()
    print(ARTICLE)
    print()

    wall_start = time.monotonic()
    try:
        # target_language is unset, so the translation branch's `when`
        # predicate is false and that branch is skipped. Set it (e.g.
        # ArticleState(article=ARTICLE, target_language="Spanish")) to run it.
        final = await graph.invoke(ArticleState(article=ARTICLE))
        wall_ms = (time.monotonic() - wall_start) * 1000.0
        print("=" * 72)
        print("Enrichment results")
        print("=" * 72)
        print()
        print(f"  summary:      {final.summary}")
        print(f"  sentiment:    {final.sentiment}")
        print(f"  topics:       {final.topics}")
        print(f"  reading time: {final.reading_time_seconds}s")
        translation = final.translation or "(skipped by `when`; set target_language to enable)"
        print(f"  translation:  {translation}")
        print()
        print(f"  wall-clock: {wall_ms:7.1f} ms")
        print()
        print("The branches ran in parallel; wall-clock is closer to the slowest")
        print("single branch than to the sum of them all. The subgraph branches")
        print("(summary, sentiment, topics) and the inline-callable reading-time")
        print("branch all contributed; the translation branch was skipped by its")
        print("`when` predicate, so it did no work and emitted no events.")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
