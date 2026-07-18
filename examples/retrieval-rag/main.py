"""Retrieval-augmented answering over a lunar knowledge base.

A question about the Moon is answered from a small corpus of passages,
using the two-stage retrieval pattern before generation:

1. **Index** the corpus once, offline: batch-embed every passage into a
   vector (``OpenAIEmbeddingProvider.embed`` over a list returns one
   vector per passage, in input order).
2. **Retrieve** per query: embed the question, rank the corpus by cosine
   similarity, and keep the top handful of candidates. Cheap and broad.
3. **Rerank** those candidates with a cross-encoder
   (``CohereRerankProvider.rerank``), which scores each candidate
   against the query directly. More accurate than similarity, so it
   reorders the shortlist and drops the weakest.
4. **Generate** a grounded answer from the reranked passages with an LLM.

Retrieval gives recall (find plausibly-relevant passages fast); reranking
gives precision (put the genuinely-relevant ones first). Feeding only the
reranked top few to the model keeps the context tight and on-topic.

**Demonstrates:**

- ``OpenAIEmbeddingProvider`` from ``openarmature.retrieval``: batch
  ``embed`` for the index, single ``embed`` for the query, one vector
  per input in input order.
- ``EmbeddingRuntimeConfig(input_type=...)``: the query/document knob.
  On OpenAI it is a wire no-op (the model is symmetric), but the same
  call selects the right representation on an asymmetric provider (TEI,
  Cohere, Jina), so setting it keeps the pipeline portable.
- ``CohereRerankProvider`` from ``openarmature.retrieval``: ``rerank``
  returns ``ScoredDocument`` results sorted by relevance. Cohere does
  not echo the document text, so results are mapped back to the
  candidate list by ``ScoredDocument.index`` rather than ``.document``.
- The retrieval providers driven inside graph nodes, so their
  ``EmbeddingEvent`` / ``RerankEvent`` reach the attached ``trace``
  observer (which prints them), the same way LLM completions do; the
  offline index build runs outside the graph and is not observed.
- An ``OpenAIProvider`` answer node grounded in the reranked passages.

**Configuration** (env vars):

- ``OPENAI_API_KEY``: required (used for both embeddings and the answer).
- ``OPENAI_BASE_URL``: host root, defaults to ``https://api.openai.com``.
  Point it at any OpenAI-compatible endpoint; the impl appends the
  ``/v1`` routes, so do not include ``/v1``.
- ``OPENAI_EMBED_MODEL``: defaults to ``text-embedding-3-small``.
- ``OPENAI_CHAT_MODEL``: defaults to ``gpt-4o-mini``.
- ``COHERE_API_KEY``: required (used for reranking).
- ``COHERE_RERANK_MODEL``: defaults to ``rerank-v3.5``.

Run with:

    uv sync --group examples
    OPENAI_API_KEY=sk-... COHERE_API_KEY=... uv run python examples/retrieval-rag/main.py
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from openarmature.graph import (
    END,
    CompiledGraph,
    EmbeddingEvent,
    GraphBuilder,
    ObserverEvent,
    RerankEvent,
    State,
)
from openarmature.llm import OpenAIProvider, RuntimeConfig, UserMessage
from openarmature.retrieval import (
    CohereRerankProvider,
    EmbeddingRuntimeConfig,
    OpenAIEmbeddingProvider,
    RerankRuntimeConfig,
)

# A tiny lunar knowledge base. In a real app the corpus is far larger and
# its vectors live in a vector store; here it is inline so the demo is
# self-contained. The passages deliberately overlap in topic so reranking
# has something to disambiguate.
CORPUS: list[str] = [
    "The Moon's south pole holds water ice in permanently shadowed crater floors "
    "that never see sunlight, a candidate resource for future crews.",
    "Apollo 11 landed in the Sea of Tranquility in July 1969; Armstrong and Aldrin "
    "spent about two and a half hours walking on the surface.",
    "The lunar maria are vast basaltic plains that formed when ancient impact basins "
    "flooded with lava, giving the near side its dark patches.",
    "A lunar day lasts about 29.5 Earth days, so any point on the surface sees roughly "
    "two weeks of sunlight followed by two weeks of night.",
    "The Moon is slowly receding from Earth at about 3.8 centimeters per year, measured "
    "by bouncing lasers off the retroreflectors Apollo crews left behind.",
    "Regolith, the layer of loose dust and broken rock covering the Moon, is abrasive "
    "and clings to everything, a persistent hazard for equipment and spacesuits.",
    "Apollo 13 aborted its landing after an oxygen tank ruptured; the crew looped around "
    "the Moon and returned safely using the lunar module as a lifeboat.",
    "Permanently shadowed regions near the poles are among the coldest places in the "
    "solar system, cold enough to trap water ice for billions of years.",
]

# How many candidates cosine retrieval hands to the reranker, and how many
# survive reranking to ground the answer. Retrieve broad, rerank narrow.
_RETRIEVE_K = 4
_RERANK_K = 2


# Lazy, module-level singletons: constructed on first use inside a node
# body (or the index build), never at import time, so importing this
# module for inspection does not open an httpx connection pool. Same
# pattern as the other examples.
_embedder: OpenAIEmbeddingProvider | None = None
_reranker: CohereRerankProvider | None = None
_answerer: OpenAIProvider | None = None

# The offline-built index: corpus passage vectors, in CORPUS order. Built
# once in main() before the graph runs.
_corpus_vectors: list[list[float]] = []


def _get_embedder() -> OpenAIEmbeddingProvider:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddingProvider(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
            model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=os.environ.get("OPENAI_API_KEY") or None,
        )
    return _embedder


def _get_reranker() -> CohereRerankProvider:
    global _reranker
    if _reranker is None:
        _reranker = CohereRerankProvider(
            model=os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5"),
            api_key=os.environ.get("COHERE_API_KEY") or None,
        )
    return _reranker


def _get_answerer() -> OpenAIProvider:
    global _answerer
    if _answerer is None:
        _answerer = OpenAIProvider(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"),
            model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY") or None,
        )
    return _answerer


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # OpenAI embeddings are unit-normalized, so the dot product already
    # equals cosine similarity; the full formula is spelled out here so the
    # example stays correct for any provider whose vectors are not
    # pre-normalized.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SearchState(State):
    query: str
    # Corpus indices of the cosine-retrieved candidates, best-first.
    candidate_indices: list[int] = []
    # Corpus indices after reranking, best-first (a reordered, trimmed
    # subset of candidate_indices).
    ranked_indices: list[int] = []
    answer: str | None = None


async def retrieve(state: SearchState) -> Mapping[str, Any]:
    # The corpus index is process-global state, built once by _build_index()
    # before the graph runs (in a real app it would live in a vector store,
    # not a module global). Guard the dependency explicitly: without it,
    # retrieve would return no candidates and the failure would surface two
    # nodes later as an opaque "empty documents" rerank error that never
    # names the real cause.
    if not _corpus_vectors:
        raise RuntimeError("corpus index not built; call _build_index() before invoking the graph")

    # Embed the query and rank the corpus index by cosine similarity.
    # input_type="query" is the asymmetric-embedding knob: a no-op on
    # OpenAI's symmetric model, but the same code selects the query
    # representation on TEI / Cohere / Jina, so the pipeline ports without
    # change. embed() takes a list and returns one vector per input; here
    # the list is a single query.
    response = await _get_embedder().embed(
        [state.query],
        config=EmbeddingRuntimeConfig(input_type="query"),
    )
    query_vector = response.vectors[0]

    scored = sorted(
        range(len(_corpus_vectors)),
        key=lambda i: _cosine(query_vector, _corpus_vectors[i]),
        reverse=True,
    )
    return {"candidate_indices": scored[:_RETRIEVE_K]}


async def rerank(state: SearchState) -> Mapping[str, Any]:
    # Cross-encoder reranking of the retrieved shortlist. The reranker
    # scores each candidate against the query directly (more accurate than
    # vector similarity) and returns results sorted best-first. Cohere does
    # not echo the document text, so results map back to the corpus by
    # ScoredDocument.index, not .document. return_documents is set to show
    # the knob; it is a no-op on the Cohere wire.
    candidates = [CORPUS[i] for i in state.candidate_indices]
    response = await _get_reranker().rerank(
        state.query,
        candidates,
        top_k=_RERANK_K,
        config=RerankRuntimeConfig(return_documents=True),
    )
    # result.index points into `candidates`; translate back to corpus index.
    ranked = [state.candidate_indices[result.index] for result in response.results]
    return {"ranked_indices": ranked}


async def answer(state: SearchState) -> Mapping[str, Any]:
    # Ground the answer in the reranked passages only. Passing the tight,
    # reordered top-K (rather than everything retrieved) keeps the context
    # on-topic and the answer faithful.
    context = "\n".join(f"- {CORPUS[i]}" for i in state.ranked_indices)
    response = await _get_answerer().complete(
        [
            UserMessage(
                content=(
                    f"Answer the question using only these passages. If they do not "
                    f"contain the answer, say so.\n\nPassages:\n{context}\n\n"
                    f"Question: {state.query}"
                )
            )
        ],
        config=RuntimeConfig(temperature=0.0),
    )
    return {"answer": response.message.content}


async def trace(event: ObserverEvent) -> None:
    # Every embed / rerank call made inside a graph node dispatches a typed
    # EmbeddingEvent / RerankEvent to any attached observer, the same way LLM
    # completions do, so retrieval is a first-class boundary in your telemetry.
    # A real deployment routes these to OTel or Langfuse (see the observability
    # examples); this one just prints them. The offline index build runs outside
    # the graph, so its embed dispatches nothing and does not appear here.
    if isinstance(event, EmbeddingEvent):
        print(f"   [obs] embed  {event.node_name}: {event.input_count} in / {event.dimensions}d")
    elif isinstance(event, RerankEvent):
        print(f"   [obs] rerank {event.node_name}: {event.document_count} in / top {event.result_count}")


def build_graph() -> CompiledGraph[SearchState]:
    return (
        GraphBuilder(SearchState)
        .add_node("retrieve", retrieve)
        .add_node("rerank", rerank)
        .add_node("answer", answer)
        .add_edge("retrieve", "rerank")
        .add_edge("rerank", "answer")
        .add_edge("answer", END)
        .set_entry("retrieve")
        .compile()
    )


async def _build_index() -> None:
    # Offline index build: batch-embed the whole corpus once. Runs outside
    # any graph, so it dispatches no observer events (unlike the per-query
    # embed inside retrieve()). input_type="document" is the counterpart to
    # the query knob above. embed() over a list returns one vector per
    # passage, in CORPUS order, so the index lines up positionally.
    global _corpus_vectors
    response = await _get_embedder().embed(
        CORPUS,
        config=EmbeddingRuntimeConfig(input_type="document"),
    )
    _corpus_vectors = response.vectors


async def main() -> None:
    # build_graph() only compiles the topology and opens no network client,
    # so it is safe before the try. Everything that constructs a provider
    # (the index build and the per-query runs) goes inside the try, so a
    # failure on any of them, including a bad or missing API key on the very
    # first embed, still closes the httpx clients in the finally.
    graph = build_graph()
    graph.attach_observer(trace)
    try:
        await _build_index()
        print(f"indexed {len(_corpus_vectors)} passages\n")
        for query in (
            "Why did Apollo 13 not land on the Moon?",
            "Where might future crews find water on the Moon?",
        ):
            final = await graph.invoke(SearchState(query=query))
            print(f"Q: {query}")
            print(f"   retrieved: {final.candidate_indices}")
            print(f"   reranked:  {final.ranked_indices}")
            print(f"   A: {final.answer}\n")
    finally:
        await graph.drain()
        if _embedder is not None:
            await _embedder.aclose()
        if _reranker is not None:
            await _reranker.aclose()
        if _answerer is not None:
            await _answerer.aclose()


if __name__ == "__main__":
    asyncio.run(main())
