"""openarmature demo: question answering against a tiny document corpus, with
two levels of subgraph nesting.

**Use case:** Given a question and a small corpus of documents, find the
answer. Three layers of responsibility:

1. **Outer (coordinator).** Takes the user's question, delegates to the
   doc-QA subgraph, and polishes the final answer for the user.
2. **Doc-QA subgraph (middle).** Picks the single most relevant document
   from the corpus, delegates the section-level work to its own subgraph,
   and synthesizes a clean answer from what came back.
3. **Section-extract subgraph (inner).** Given a single document and the
   question, finds the relevant paragraph and extracts the answer text.

Each layer has its own state schema reflecting its scope: the outer cares
about a question + final answer, the middle picks one document from CORPUS
and synthesizes, the inner cares about a single doc + an extracted span.
That separation is the whole reason the middle and inner pieces are
subgraphs and not flat nodes; each is a self-contained, reusable
sub-pipeline with its own inputs and outputs.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/nested-subgraphs
    LLM_API_KEY=sk-... uv run python main.py "what year did humans first land on the moon?"
    LLM_API_KEY=sk-... uv run python main.py "what happened on Apollo 13?"
    LLM_API_KEY=sk-... uv run python main.py "who was on the Artemis II crew?"
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    ExplicitMapping,
    GraphBuilder,
    NodeEvent,
    ObserverEvent,
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


async def _chat(system: str, user: str) -> str:
    response = await _get_provider().complete(
        [SystemMessage(content=system), UserMessage(content=user)],
    )
    return (response.message.content or "").strip()


# ---------------------------------------------------------------------------
# A tiny baked-in corpus. In a real app this would come from a retriever
# or a vector store; here it's three short documents so the example runs
# without any external setup.
# ---------------------------------------------------------------------------

CORPUS: list[dict[str, str]] = [
    {
        "title": "Apollo 11",
        "body": (
            "Apollo 11 was the United States spaceflight that first landed humans on the Moon. "
            "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar "
            "Module Eagle on July 20, 1969. Armstrong became the first person to step onto the "
            "lunar surface six hours and 39 minutes later, on July 21 at 02:56 UTC. The mission "
            "fulfilled a national goal proposed by President Kennedy in 1961."
        ),
    },
    {
        "title": "Apollo 13",
        "body": (
            "Apollo 13 was the seventh crewed mission in the Apollo program and the third intended "
            "to land on the Moon. The lunar landing was aborted after an oxygen tank in the service "
            "module ruptured two days after launch in April 1970, crippling power and life support. "
            "The crew of Jim Lovell, Jack Swigert, and Fred Haise used the lunar module Aquarius as "
            "a lifeboat and looped around the Moon on a free-return trajectory before splashing down "
            "safely in the Pacific. The mission is remembered as a successful failure."
        ),
    },
    {
        "title": "Artemis II",
        "body": (
            "Artemis II was the first crewed mission of NASA's Artemis program, launching from "
            "Kennedy Space Center on April 1, 2026 atop the Space Launch System rocket. The "
            "ten-day flight carried astronauts Reid Wiseman, Victor Glover, Christina Koch, and "
            "Jeremy Hansen aboard the Orion spacecraft Integrity on a free-return trajectory around "
            "the Moon and back. It was the first crewed flight beyond low Earth orbit since Apollo "
            "17 in 1972. The capsule splashed down in the Pacific Ocean on April 10, 2026, marking "
            "a successful test flight ahead of the Artemis III lunar landing mission."
        ),
    },
]


# ---------------------------------------------------------------------------
# State schemas; one per layer, each scoped to its layer's job.
# ---------------------------------------------------------------------------


class OuterState(State):
    """User-facing state: a question goes in, an answer comes out."""

    question: str
    answer: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


class DocQAState(State):
    """Middle: the doc-QA subgraph picks a doc and synthesizes an answer.

    The corpus itself is module-level configuration, not per-invocation
    state. Nodes reach into ``CORPUS`` directly rather than carrying it
    through state; typical for application config that doesn't change
    between calls.
    """

    question: str = ""
    selected_title: str = ""
    selected_body: str = ""
    raw_answer: str = ""
    answer: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


class SectionState(State):
    """Inner: the section-extract subgraph narrows to one paragraph then
    pulls the answer text out of it."""

    question: str = ""
    doc_body: str = ""
    relevant_section: str = ""
    extracted: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inner subgraph: section-extract (one doc → answer text)
# ---------------------------------------------------------------------------


async def find_section(s: SectionState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "You are given a document and a question. Find the single paragraph "
            "in the document most likely to contain the answer. Return ONLY that "
            "paragraph verbatim, no preamble."
        ),
        user=f"Question: {s.question}\n\nDocument:\n{s.doc_body}",
    )
    return {"relevant_section": content, "trace": ["find_section"]}


async def extract_answer(s: SectionState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "You are given a question and a paragraph that contains the answer. "
            "Extract just the answer in one short phrase or sentence. No preamble, "
            "no quoting the source."
        ),
        user=f"Question: {s.question}\n\nParagraph:\n{s.relevant_section}",
    )
    return {"extracted": content, "trace": ["extract_answer"]}


def build_section_extract() -> CompiledGraph[SectionState]:
    return (
        GraphBuilder(SectionState)
        .add_node("find_section", find_section)
        .add_node("extract_answer", extract_answer)
        .add_edge("find_section", "extract_answer")
        .add_edge("extract_answer", END)
        .set_entry("find_section")
        .compile()
    )


# ---------------------------------------------------------------------------
# Middle subgraph: doc-QA (corpus → answer)
# ---------------------------------------------------------------------------


async def pick_doc(s: DocQAState) -> Mapping[str, Any]:
    """Ask the LLM which corpus document is most relevant to the question."""
    titles_and_bodies = "\n\n".join(f"{d['title']}:\n{d['body']}" for d in CORPUS)
    content = await _chat(
        system=(
            "You are given a question and several documents. Reply with EXACTLY "
            "the title of the single document most relevant to answering the "
            "question. No quotes, no punctuation, just the title."
        ),
        user=f"Question: {s.question}\n\nDocuments:\n\n{titles_and_bodies}",
    )
    reply = content.strip().strip('"').strip("'").lower()
    # Permissive match: the model may paraphrase ("Apollo 11 article") or
    # return only part of the title. Accept either direction of containment
    # over the lowercased strings; strict equality is too brittle for
    # free-form output. A production app would constrain the model with
    # response_schema (see 00-hello-world) so the reply is guaranteed to be
    # a valid title.
    match = next(
        (d for d in CORPUS if d["title"].lower() in reply or reply in d["title"].lower()),
        None,
    )
    if match is None:
        raise RuntimeError(
            f"pick_doc: model returned {content!r} which doesn't match any "
            f"corpus title ({[d['title'] for d in CORPUS]!r})."
        )
    return {"selected_title": match["title"], "selected_body": match["body"], "trace": ["pick_doc"]}


async def synthesize(s: DocQAState) -> Mapping[str, Any]:
    """Polish the extracted answer into one user-facing sentence."""
    content = await _chat(
        system=(
            "You are given a question and a short raw answer extracted from a "
            "document. Rewrite the answer as one clean sentence that stands on "
            "its own. No preamble."
        ),
        user=f"Question: {s.question}\n\nRaw answer:\n{s.raw_answer}",
    )
    return {"answer": content, "trace": ["synthesize"]}


def build_doc_qa(section_extract: CompiledGraph[SectionState]) -> CompiledGraph[DocQAState]:
    return (
        GraphBuilder(DocQAState)
        .add_node("pick_doc", pick_doc)
        .add_subgraph_node(
            "section_extract",
            section_extract,
            # The middle hands its selected doc to the inner subgraph, then
            # receives back the extracted text into ``raw_answer`` for the
            # synthesize step.
            projection=ExplicitMapping[DocQAState, SectionState](
                inputs={"question": "question", "doc_body": "selected_body"},
                outputs={"raw_answer": "extracted", "trace": "trace"},
            ),
        )
        .add_node("synthesize", synthesize)
        .add_edge("pick_doc", "section_extract")
        .add_edge("section_extract", "synthesize")
        .add_edge("synthesize", END)
        .set_entry("pick_doc")
        .compile()
    )


# ---------------------------------------------------------------------------
# Outer graph: coordinator
# ---------------------------------------------------------------------------


async def receive(s: OuterState) -> Mapping[str, Any]:
    """Marker node so the trace shows the outer received the question."""
    del s
    return {"trace": ["receive"]}


async def format_final(s: OuterState) -> Mapping[str, Any]:
    """Light polish on the synthesized answer before returning to the user."""
    content = await _chat(
        system=(
            "Lightly copy-edit the following answer for clarity. Keep it short "
            "and preserve meaning. Return only the edited answer."
        ),
        user=s.answer,
    )
    return {"answer": content, "trace": ["format_final"]}


def build_graph() -> CompiledGraph[OuterState]:
    section_extract = build_section_extract()
    doc_qa = build_doc_qa(section_extract)
    return (
        GraphBuilder(OuterState)
        .add_node("receive", receive)
        .add_subgraph_node(
            "doc_qa",
            doc_qa,
            # The outer feeds its question and the corpus down to the
            # doc-QA subgraph and receives back the synthesized answer.
            projection=ExplicitMapping[OuterState, DocQAState](
                inputs={"question": "question"},
                outputs={"answer": "answer", "trace": "trace"},
            ),
        )
        .add_node("format_final", format_final)
        .add_edge("receive", "doc_qa")
        .add_edge("doc_qa", "format_final")
        .add_edge("format_final", END)
        .set_entry("receive")
        .compile()
    )


# ---------------------------------------------------------------------------
# Observer; formats events so the descent through layers is visible.
#
# The same observer fires for every node at every depth; including the
# inner section-extract subgraph at depth 3. Indentation in the printed
# output makes the descent and return obvious.
# ---------------------------------------------------------------------------


def _fmt_state(state: Any) -> str:
    """Compact one-line dump of whichever state class the event carries."""
    if state is None:
        return "-"
    dumped = state.model_dump()
    # Hide the trace (already visible in the printed order). Print the
    # remaining fields as a compact summary.
    skip = {"trace"}
    parts: list[str] = []
    for key, value in dumped.items():
        if key in skip:
            continue
        # Truncate long string values so the line stays scannable.
        if isinstance(value, str) and len(value) > 60:
            value = value[:57] + "..."
        parts.append(f"{key}={value!r}")
    return " ".join(parts) if parts else "(empty)"


async def depth_observer(event: ObserverEvent) -> None:
    if not isinstance(event, NodeEvent):
        return
    depth = len(event.namespace)
    indent = "  " * (depth - 1)
    ns = " > ".join(event.namespace)

    if event.phase == "started":
        print(f"{indent}[step {event.step}] depth={depth}  {ns}")
        print(f"{indent}    started   {_fmt_state(event.pre_state)}")
    else:
        if event.error is not None:
            print(f"{indent}    completed ERROR={type(event.error).__name__}: {event.error}")
        else:
            print(f"{indent}    completed {_fmt_state(event.post_state)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "what year did humans first land on the moon?"

    outer = build_graph()
    outer.attach_observer(depth_observer)

    print("=" * 72)
    print(f"Question: {question}")
    print("=" * 72)
    print()

    try:
        final = await outer.invoke(OuterState(question=question))
        print()
        print("=" * 72)
        print(f"Answer: {final.answer}")
        print("=" * 72)
        print()
        print(f"Trace: {final.trace}")
    finally:
        await outer.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
