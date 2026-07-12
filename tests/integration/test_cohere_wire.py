"""Integration tests for the Cohere wire mappings against the live Cohere API.

Gated by the ``COHERE_API_KEY`` env var: the tests run only when it is set (a
live Cohere API key). Skipped in CI and local runs that don't have it in scope;
runs end-to-end against the hosted Cohere endpoint when invoked with the key set.

``base_url`` is read from ``COHERE_BASE_URL`` (default ``https://api.cohere.com``),
the bound rerank model from ``COHERE_RERANK_MODEL`` and the bound embed model
from ``COHERE_EMBED_MODEL`` (sensible Cohere defaults, overridable via env).
Nothing is hardcoded to a specific deployment.
"""

from __future__ import annotations

import os

import pytest

from openarmature.retrieval import (
    CohereEmbeddingProvider,
    CohereRerankProvider,
    EmbeddingRuntimeConfig,
    RerankRuntimeConfig,
)

_API_KEY = os.environ.get("COHERE_API_KEY")
_BASE_URL = os.environ.get("COHERE_BASE_URL", "https://api.cohere.com")
_RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")
_EMBED_MODEL = os.environ.get("COHERE_EMBED_MODEL", "embed-v4.0")

requires_cohere = pytest.mark.skipif(not _API_KEY, reason="Requires COHERE_API_KEY (live Cohere API key)")


@pytest.mark.integration
@requires_cohere
async def test_cohere_rerank_returns_sorted_results_with_search_units() -> None:
    """rerank() on a small pool returns results sorted by relevance descending,
    each index valid into the input documents, with a search_units usage record
    (Cohere meters rerank by search units, not tokens)."""
    documents = [
        "The Sea of Tranquility was the Apollo 11 landing site.",
        "Cheese is made from milk.",
        "The lunar south pole holds water ice in permanently shadowed craters.",
    ]
    provider = CohereRerankProvider(model=_RERANK_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.rerank("Where on the moon is there water ice?", documents)
    finally:
        await provider.aclose()

    scores = [r.relevance_score for r in response.results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= r.index < len(documents) for r in response.results)
    assert len({r.index for r in response.results}) == len(response.results)
    # Cohere meters rerank by search_units -> search_units; input_tokens stays null.
    assert response.usage is not None
    assert response.usage.search_units is not None
    assert response.usage.input_tokens is None


@pytest.mark.integration
@requires_cohere
async def test_cohere_rerank_return_documents_is_wire_noop() -> None:
    """return_documents=True is a silent no-op on the Cohere wire: the results
    still come back (no error), and ScoredDocument.document is null on every
    result -- Cohere never echoes document text on /v2/rerank."""
    documents = [
        "The lunar maria are dark basaltic plains.",
        "Photosynthesis occurs in chloroplasts.",
    ]
    provider = CohereRerankProvider(model=_RERANK_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.rerank(
            "What are the dark plains on the moon?",
            documents,
            config=RerankRuntimeConfig(return_documents=True),
        )
    finally:
        await provider.aclose()

    assert len(response.results) == len(documents)
    assert all(r.document is None for r in response.results)


@pytest.mark.integration
@requires_cohere
async def test_cohere_embed_returns_vectors_in_input_order_with_input_tokens() -> None:
    """embed() on a couple of strings returns one vector per input in input
    order, all of equal dimensionality, with an input_tokens usage record
    (Cohere meters embedding by input tokens)."""
    inputs = [
        "The lunar south pole holds water ice in permanently shadowed craters.",
        "The Sea of Tranquility was the Apollo 11 landing site.",
    ]
    provider = CohereEmbeddingProvider(model=_EMBED_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.embed(inputs)
    finally:
        await provider.aclose()

    assert len(response.vectors) == len(inputs)
    assert len({len(v) for v in response.vectors}) == 1
    assert response.dimensions == len(response.vectors[0])
    # Cohere meters embedding by input tokens -> input_tokens.
    assert response.usage is not None
    assert response.usage.input_tokens > 0


@pytest.mark.integration
@requires_cohere
async def test_cohere_embed_output_dimension_controls_vector_length() -> None:
    """A dimensions=... call maps to Cohere's output_dimension (Matryoshka), so
    the returned vectors have exactly that length (on a model that supports it,
    e.g. embed-v4.0)."""
    provider = CohereEmbeddingProvider(model=_EMBED_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.embed(
            ["The lunar maria are dark basaltic plains."],
            config=EmbeddingRuntimeConfig(dimensions=256),
        )
    finally:
        await provider.aclose()

    assert response.dimensions == 256
    assert all(len(v) == 256 for v in response.vectors)
