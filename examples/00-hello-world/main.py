"""Hello-world demo: a 3-node graph that classifies a query with an LLM
(via structured output) and routes to one of two follow-up nodes.

**Demonstrates:**

- Typed ``State`` with three reducer policies (``last_write_wins``,
  ``append``, ``merge``).
- ``OpenAIProvider`` from ``openarmature.llm`` against any
  OpenAI-compatible endpoint.
- Structured output via a Pydantic class — the model's response comes
  back as a validated ``Classification`` instance, not a string.
- Conditional routing as a pure function of state (``route``).
- ``attach_observer`` for boundary visibility.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` — defaults to ``https://api.openai.com``. **Host
  root only** — the impl adds ``/v1/chat/completions`` and
  ``/v1/models`` itself, so do NOT include ``/v1`` in this value.
- ``LLM_MODEL`` — defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` — required (your OpenAI API key, or empty for
  local servers that don't authenticate).

Run with:

    uv sync --group examples
    LLM_API_KEY=sk-... uv run python examples/00-hello-world/main.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    NodeEvent,
    State,
    append,
    merge,
)
from openarmature.llm import OpenAIProvider, UserMessage


class Classification(BaseModel):
    """The Pydantic schema the model is constrained to produce.

    Passed as ``response_schema`` to ``provider.complete()``; the
    framework converts to JSON Schema, instructs the provider to
    return matching content, validates the response, and yields a
    ``Classification`` instance via ``Response.parsed``.
    """

    intent: Literal["research", "summarize"]
    rationale: str


class PipelineState(State):
    query: str
    classification: Classification | None = None
    sources: Annotated[list[str], append] = Field(default_factory=list)
    metadata: Annotated[dict[str, str], merge] = Field(default_factory=dict)


_provider = OpenAIProvider(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
    model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    api_key=os.environ.get("LLM_API_KEY"),
)


async def classify(state: PipelineState) -> Mapping[str, Any]:
    response = await _provider.complete(
        [
            UserMessage(
                content=(
                    f"Route this query to either 'research' (look something up) or "
                    f"'summarize' (condense known material): {state.query!r}"
                )
            )
        ],
        response_schema=Classification,
    )
    return {"classification": response.parsed, "metadata": {"classified_by": "llm"}}


async def research(state: PipelineState) -> Mapping[str, Any]:
    return {"sources": ["wikipedia", "arxiv"], "metadata": {"tool": "search"}}


async def summarize(state: PipelineState) -> Mapping[str, Any]:
    return {"sources": ["cache"], "metadata": {"tool": "summarizer"}}


def route(state: PipelineState) -> str:
    if state.classification is None:
        raise RuntimeError("classify did not populate state.classification")
    return state.classification.intent


async def trace(event: NodeEvent) -> None:
    # OpenAIProvider emits NodeEvent-shaped events for LLM-span
    # tracking under a sentinel namespace; those have post_state=None.
    # Filter to events that carry a state snapshot before reading it.
    if event.phase == "completed" and event.error is None and event.post_state is not None:
        print(f"{event.node_name}: sources={event.post_state.sources}")


def build_graph() -> CompiledGraph[PipelineState]:
    return (
        GraphBuilder(PipelineState)
        .add_node("classify", classify)
        .add_node("research", research)
        .add_node("summarize", summarize)
        .add_conditional_edge("classify", route)
        .add_edge("research", END)
        .add_edge("summarize", END)
        .set_entry("classify")
        .compile()
    )


async def main() -> None:
    graph = build_graph()
    graph.attach_observer(trace)
    try:
        final = await graph.invoke(PipelineState(query="what is RAG?"))
        print(f"\nclassification: {final.classification}")
        print(f"sources: {final.sources}")
        print(f"metadata: {final.metadata}")
    finally:
        await graph.drain()


if __name__ == "__main__":
    asyncio.run(main())
