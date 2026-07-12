"""Integration tests for the Cohere wire mappings against the live Cohere API.

Gated by the ``COHERE_API_KEY`` env var: the tests run only when it is set (a
live Cohere API key). Skipped in CI and local runs that don't have it in scope;
runs end-to-end against the hosted Cohere endpoint when invoked with the key set.

``base_url`` is read from ``COHERE_BASE_URL`` (default ``https://api.cohere.com``),
and the bound rerank model from ``COHERE_RERANK_MODEL`` (a sensible Cohere
default, overridable via env). Nothing is hardcoded to a specific deployment.
"""

from __future__ import annotations

import os

import pytest

from openarmature.retrieval import CohereRerankProvider, RerankRuntimeConfig

_API_KEY = os.environ.get("COHERE_API_KEY")
_BASE_URL = os.environ.get("COHERE_BASE_URL", "https://api.cohere.com")
_RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")

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
