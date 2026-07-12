"""Integration tests for the Jina wire mappings against the live Jina API.

Gated by the ``JINA_API_KEY`` env var: the tests run only when it is set (a
live Jina API key). Skipped in CI and local runs that don't have it in scope;
runs end-to-end against the hosted Jina endpoint when invoked with the key set.

``base_url`` is read from ``JINA_BASE_URL`` (default ``https://api.jina.ai``),
and the bound models from ``JINA_EMBED_MODEL`` / ``JINA_RERANK_MODEL``
(sensible Jina defaults, overridable via env). Nothing is hardcoded to a
specific deployment.
"""

from __future__ import annotations

import os

import pytest

from openarmature.retrieval import JinaEmbeddingProvider, JinaRerankProvider, RerankRuntimeConfig

_API_KEY = os.environ.get("JINA_API_KEY")
_BASE_URL = os.environ.get("JINA_BASE_URL", "https://api.jina.ai")
_EMBED_MODEL = os.environ.get("JINA_EMBED_MODEL", "jina-embeddings-v3")
_RERANK_MODEL = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")

requires_jina = pytest.mark.skipif(not _API_KEY, reason="Requires JINA_API_KEY (live Jina API key)")


@pytest.mark.integration
@requires_jina
async def test_jina_embed_returns_vectors_with_usage() -> None:
    """embed() on a couple of strings returns one real vector per input, in
    input order, with a usage record (Jina meters embeddings by tokens)."""
    provider = JinaEmbeddingProvider(model=_EMBED_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.embed(["the moon orbits the earth", "lunar regolith is abrasive"])
    finally:
        await provider.aclose()

    assert len(response.vectors) == 2
    dim = len(response.vectors[0])
    assert dim > 0
    assert all(len(v) == dim for v in response.vectors)
    assert response.dimensions == dim
    # Jina reports usage.total_tokens -> input_tokens (a record, not fabricated).
    assert response.usage is not None
    assert response.usage.input_tokens > 0


@pytest.mark.integration
@requires_jina
async def test_jina_rerank_returns_sorted_results_with_usage() -> None:
    """rerank() on a small pool returns results sorted by relevance descending,
    each index valid into the input documents, with a token usage record."""
    documents = [
        "The Sea of Tranquility was the Apollo 11 landing site.",
        "Cheese is made from milk.",
        "The lunar south pole holds water ice in permanently shadowed craters.",
    ]
    provider = JinaRerankProvider(model=_RERANK_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.rerank("Where on the moon is there water ice?", documents)
    finally:
        await provider.aclose()

    scores = [r.relevance_score for r in response.results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= r.index < len(documents) for r in response.results)
    assert len({r.index for r in response.results}) == len(response.results)
    # Jina meters rerank by tokens -> input_tokens; search_units stays null.
    assert response.usage is not None
    assert response.usage.input_tokens is not None
    assert response.usage.search_units is None


@pytest.mark.integration
@requires_jina
async def test_jina_rerank_return_documents_populates_echo() -> None:
    """rerank() with return_documents=True surfaces Jina's document echo on
    ScoredDocument.document. Jina returns the echo as a TextDoc object
    ({"text": ...}); the mapping must unwrap it to the string, so every result's
    document is a non-empty string mapping back to an input -- not silently
    dropped to None."""
    documents = [
        "The Sea of Tranquility was the Apollo 11 landing site.",
        "The lunar south pole holds water ice in permanently shadowed craters.",
    ]
    provider = JinaRerankProvider(model=_RERANK_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.rerank(
            "Where on the moon is there water ice?",
            documents,
            config=RerankRuntimeConfig(return_documents=True),
        )
    finally:
        await provider.aclose()

    # Every result carries the unwrapped string echo, mapping back to an input.
    assert all(isinstance(r.document, str) and r.document for r in response.results)
    assert all(r.document in documents for r in response.results)
