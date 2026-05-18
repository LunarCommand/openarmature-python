"""Hello-world demo: a 3-node graph where each node makes an LLM call
with structured output. Classify a query, then either plan research or
write a one-sentence summary.

**Demonstrates:**

- Typed ``State`` with three reducer policies (``last_write_wins``,
  ``append``, ``merge``).
- ``OpenAIProvider`` from ``openarmature.llm`` against any
  OpenAI-compatible endpoint.
- Both ``response_schema`` forms:
  - Pydantic class (``Classification``, ``Summary``): typed
    instance on ``Response.parsed``.
  - JSON Schema dict (``research``): raw dict on ``Response.parsed``.
- Conditional routing on a parsed field (``route`` reads
  ``state.classification.intent``).
- ``attach_observer`` for boundary visibility.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL``: defaults to ``https://api.openai.com``. **Host
  root only**; the impl adds ``/v1/chat/completions`` and
  ``/v1/models`` itself, so do NOT include ``/v1`` in this value.
- ``LLM_MODEL``: defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY``: required (your OpenAI API key, or empty for
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


# Pydantic schemas the model is constrained to produce. Passing a
# class as ``response_schema`` makes the framework convert to JSON
# Schema, instruct the provider to return matching content, validate
# the response, and yield an instance via ``Response.parsed``.
class Classification(BaseModel):
    intent: Literal["research", "summarize"]
    rationale: str


class Summary(BaseModel):
    one_liner: str
    confidence: float


# State holds intermediate artifacts from each LLM call. ``research``
# uses a dict schema (rather than a class), so its parsed value is a
# raw dict, typed here as ``dict[str, Any] | None``.
class PipelineState(State):
    query: str
    classification: Classification | None = None
    research_plan: dict[str, Any] | None = None
    summary: Summary | None = None
    sources: Annotated[list[str], append] = Field(default_factory=list)
    metadata: Annotated[dict[str, str], merge] = Field(default_factory=dict)


# Lazy initialization: the provider is constructed on first call from
# inside a node body, not at import time. That avoids opening an
# httpx.AsyncClient connection pool when tools (test harnesses, doc
# builders, IDE inspection) import this module without running main().
_provider_instance: OpenAIProvider | None = None


def _get_provider() -> OpenAIProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OpenAIProvider(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            # ``or None`` so an exported-but-empty LLM_API_KEY falls
            # through to no-auth (matters for local servers like vLLM
            # that reject an empty bearer header).
            api_key=os.environ.get("LLM_API_KEY") or None,
        )
    return _provider_instance


async def classify(state: PipelineState) -> Mapping[str, Any]:
    # response_schema=class form: parsed comes back as a Classification
    # instance. The model picks the branch (research vs summarize) and
    # the routing function below reads it as a typed field.
    response = await _get_provider().complete(
        [
            UserMessage(
                content=(
                    f"Route this query to either 'research' (find new information) "
                    f"or 'summarize' (condense known material): {state.query!r}"
                )
            )
        ],
        response_schema=Classification,
    )
    return {"classification": response.parsed, "metadata": {"classified_by": "llm"}}


async def research(state: PipelineState) -> Mapping[str, Any]:
    # response_schema=dict form: parsed comes back as a plain dict.
    # Same wire shape as the class form: the framework converts a
    # class via .model_json_schema() under the hood. Use dict when
    # you want raw shape without declaring a Pydantic model.
    response = await _get_provider().complete(
        [
            UserMessage(
                content=(
                    f"Plan research for the query {state.query!r}. List up to 3 "
                    f"specific topics to investigate and up to 3 follow-up questions."
                )
            )
        ],
        response_schema={
            "type": "object",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}},
                "follow_up_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topics", "follow_up_questions"],
            "additionalProperties": False,
        },
    )
    return {
        "research_plan": response.parsed,
        "sources": ["wikipedia", "arxiv"],
        "metadata": {"tool": "research"},
    }


async def summarize(state: PipelineState) -> Mapping[str, Any]:
    # Pydantic-class form again: parsed is a Summary instance with
    # a typed one_liner and a confidence float.
    response = await _get_provider().complete(
        [
            UserMessage(
                content=(
                    f"Summarize {state.query!r} in one sentence. Set confidence "
                    f"between 0 and 1 reflecting how well-established the answer is."
                )
            )
        ],
        response_schema=Summary,
    )
    return {
        "summary": response.parsed,
        "sources": ["cache"],
        "metadata": {"tool": "summarize"},
    }


def route(state: PipelineState) -> str:
    if state.classification is None:
        raise RuntimeError("classify did not populate state.classification")
    return state.classification.intent


async def trace(event: NodeEvent) -> None:
    # OpenAIProvider emits NodeEvent-shaped events for LLM-span
    # tracking under a sentinel namespace; those have post_state=None.
    # Filter to events that carry a PipelineState snapshot before
    # reading it. The isinstance check both narrows the type for
    # static checkers (post_state is typed as the base State, not
    # PipelineState) and acts as a defensive guard against any
    # foreign-state observer event the engine might dispatch.
    if event.phase == "completed" and event.error is None and isinstance(event.post_state, PipelineState):
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
        final = await graph.invoke(PipelineState(query="why did Apollo 13 abort its lunar landing?"))
        print(f"\nclassification: {final.classification}")
        if final.research_plan is not None:
            print(f"research_plan: {final.research_plan}")
        if final.summary is not None:
            print(f"summary: {final.summary}")
        print(f"sources: {final.sources}")
        print(f"metadata: {final.metadata}")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
