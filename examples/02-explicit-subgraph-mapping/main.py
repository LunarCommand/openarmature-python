"""openarmature demo: same compiled subgraph reused at two sites in one parent
graph, each site with its own ExplicitMapping.

**Use case:** Compare two topics ("Apollo program vs Artemis program",
"Apollo 11 vs Apollo 17") by running the same analysis subgraph on each,
then synthesizing a verdict.

**Demonstrates:** One compiled subgraph reused at two parent sites with
per-site `ExplicitMapping` — the canonical way to express "run the same
subgraph twice on disjoint parent fields" without writing per-site
projection classes that mirror each other.

Without explicit input/output mapping, both sites would have to read from
and write to the same parent fields under name matching — making "run the
same subgraph twice on different inputs" structurally impossible. The two
analyze_a/analyze_b sites here share the SAME compiled subgraph value but
project different parent fields in and different parent fields out.

LLM calls go through ``openarmature.llm.OpenAIProvider``.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/02-explicit-subgraph-mapping
    LLM_API_KEY=sk-... uv run python main.py "Apollo 11" "Apollo 17"
    LLM_API_KEY=sk-... uv run python main.py "Apollo program vs Artemis program"
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    ExplicitMapping,
    GraphBuilder,
    State,
    append,
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


# ----------------------------------------------------------------------------
# State schemas: parent and subgraph
# ----------------------------------------------------------------------------
# The parent compares two topics — call them A and B — and needs to capture a
# summary and a score for EACH topic. So the parent schema declares paired
# fields: a_summary/a_score and b_summary/b_score.
#
# The subgraph speaks in a single set of names — `topic`, `summary`, `score` —
# because it has no idea which side of the comparison it's running for. The
# mapping at each call site is what wires the subgraph's neutral names to the
# parent's per-side fields.
#
# This separation is the whole point. If the parent and subgraph shared field
# names (`summary`, `score`) and we relied on default field-name
# matching, the two subgraph calls would BOTH write to a single
# `parent.summary` field — the second call would clobber the first, and the
# comparator at the end would have no way to see both. With explicit mapping
# the two sites address disjoint parent fields and can't collide.


class ComparisonState(State):
    """Outer graph: holds two topics, captures per-side analysis, emits a verdict."""

    topic_a: str
    topic_b: str
    a_summary: str = ""
    a_score: int = 0
    b_summary: str = ""
    b_score: int = 0
    verdict: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


class AnalysisState(State):
    """Subgraph: takes a single topic, produces a one-line summary and a 1-10 score."""

    topic: str = ""  # projected IN from a parent field via inputs mapping
    summary: str = ""  # projected OUT to a parent field via outputs mapping
    score: int = 0  # projected OUT to a parent field via outputs mapping
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
# Subgraph nodes
# ----------------------------------------------------------------------------
# Two nodes — `summarize` then `score` — running against the subgraph's own
# state. They're written entirely against `AnalysisState`; they don't know
# (and can't know) that the parent calls them twice with different mappings.
# That's the encapsulation a subgraph buys.


async def summarize(s: AnalysisState) -> Mapping[str, Any]:
    content = await _chat(
        system="In one tight sentence, summarize what the user-supplied topic IS. No preamble.",
        user=s.topic,
    )
    return {"summary": content, "trace": ["summarize"]}


async def score(s: AnalysisState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "Given a topic and a one-line summary, rate the topic from 1 to 10 on overall "
            "interestingness/usefulness. Reply with just the integer, no other text."
        ),
        user=f"Topic: {s.topic}\nSummary: {s.summary}",
    )
    match = re.search(r"\d+", content)
    n = int(match.group()) if match else 5
    return {"score": max(1, min(10, n)), "trace": ["score"]}


def build_analysis_subgraph() -> CompiledGraph[AnalysisState]:
    return (
        GraphBuilder(AnalysisState)
        .add_node("summarize", summarize)
        .add_node("score", score)
        .add_edge("summarize", "score")
        .add_edge("score", END)
        .set_entry("summarize")
        .compile()
    )


# ----------------------------------------------------------------------------
# Outer-graph node: synthesize a verdict from both per-side analyses
# ----------------------------------------------------------------------------


async def synthesize(s: ComparisonState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "Two topics have been analyzed. Pick a winner (or call it a tie) in one short "
            "paragraph. Be specific about WHY, citing the summaries."
        ),
        user=(
            f"Topic A: {s.topic_a}\n"
            f"  summary: {s.a_summary}\n"
            f"  score:   {s.a_score}/10\n\n"
            f"Topic B: {s.topic_b}\n"
            f"  summary: {s.b_summary}\n"
            f"  score:   {s.b_score}/10\n"
        ),
    )
    return {"verdict": content, "trace": ["synthesize"]}


# ----------------------------------------------------------------------------
# Outer graph: ONE compiled subgraph at TWO sites with DIFFERENT mappings
# ----------------------------------------------------------------------------
# The point of this entire example lives in the next ~15 lines: the same
# compiled `analysis` subgraph is registered twice as a node, and each
# registration carries its own `ExplicitMapping`.
#
# `analyze_a` says "feed me parent.topic_a as my `topic`; write my `summary`
# back as parent.a_summary and my `score` as parent.a_score."
#
# `analyze_b` says the same thing but with the B-side parent fields.
#
# The two sites address disjoint parent fields. They CANNOT collide.
#
# Why is this only doable with explicit mapping?
#
#   - The default field-name matching can't help: it operates on names alone,
#     with no way to express "this site reads A, that site reads B." Both
#     sites would write `parent.summary` and clobber each other.
#
#   - A custom `ProjectionStrategy` (the 01-routing-and-subgraphs approach)
#     would have to differ per call site — you'd write two distinct projection
#     classes that do the same thing in mirror image. That's exactly the
#     boilerplate `ExplicitMapping` removes.
#
#   - The subgraph can't rename its own fields to avoid the clash: its schema
#     is fixed at compile time and shared across both call sites. Wrong layer.
#
# Note also that `trace` is included in both `outputs` mappings — so each
# subgraph run's per-node trace is appended to the parent's `trace` field via
# the parent's `append` reducer. The final `trace` will show the subgraph
# nodes running TWICE (once per analyze site), interleaved with the outer
# nodes.


def build_graph() -> CompiledGraph[ComparisonState]:
    analysis = build_analysis_subgraph()

    return (
        GraphBuilder(ComparisonState)
        .add_subgraph_node(
            "analyze_a",
            analysis,
            projection=ExplicitMapping[ComparisonState, AnalysisState](
                inputs={"topic": "topic_a"},
                outputs={"a_summary": "summary", "a_score": "score", "trace": "trace"},
            ),
        )
        .add_subgraph_node(
            "analyze_b",
            analysis,
            projection=ExplicitMapping[ComparisonState, AnalysisState](
                inputs={"topic": "topic_b"},
                outputs={"b_summary": "summary", "b_score": "score", "trace": "trace"},
            ),
        )
        .add_node("synthesize", synthesize)
        .add_edge("analyze_a", "analyze_b")
        .add_edge("analyze_b", "synthesize")
        .add_edge("synthesize", END)
        .set_entry("analyze_a")
        .compile()
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


async def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 2:
        topic_a, topic_b = args[0], args[1]
    elif len(args) == 1 and " vs " in args[0].lower():
        topic_a, topic_b = re.split(r" vs ", args[0], maxsplit=1, flags=re.IGNORECASE)
    else:
        topic_a, topic_b = "Apollo 11", "Apollo 17"

    graph = build_graph()
    try:
        final = await graph.invoke(ComparisonState(topic_a=topic_a, topic_b=topic_b))

        print(f"topic A: {final.topic_a}")
        print(f"  summary: {final.a_summary}")
        print(f"  score:   {final.a_score}/10")
        print()
        print(f"topic B: {final.topic_b}")
        print(f"  summary: {final.b_summary}")
        print(f"  score:   {final.b_score}/10")
        print()
        print(f"verdict:\n{final.verdict}")
        print()
        print(f"trace: {final.trace}")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
