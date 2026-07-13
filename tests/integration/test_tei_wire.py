"""Integration tests for the TEI wire mappings against a live TEI runtime.

Gated by the ``TEI_EMBED_URL`` / ``TEI_RERANK_URL`` env vars: the embedding
tests run only when ``TEI_EMBED_URL`` points at a live TEI embedding instance,
the rerank tests only when ``TEI_RERANK_URL`` points at a live TEI reranker.
Skipped in CI and local runs that don't have them in scope; runs end-to-end
against a local TEI when invoked with the vars set (the reviewer sets e.g.
``TEI_EMBED_URL=http://localhost:8083 TEI_RERANK_URL=http://localhost:8082``).

The ports are read from the env vars, never hardcoded. The bound model
identifier is a label only (TEI hosts one model per instance and ignores a
model field in the body); override via ``TEI_EMBED_MODEL`` / ``TEI_RERANK_MODEL``
when a specific label is wanted.
"""

from __future__ import annotations

import os

import pytest

from openarmature.retrieval import TeiEmbeddingProvider, TeiRerankProvider

_EMBED_URL = os.environ.get("TEI_EMBED_URL")
_RERANK_URL = os.environ.get("TEI_RERANK_URL")
_EMBED_MODEL = os.environ.get("TEI_EMBED_MODEL", "tei-embed-live")
_RERANK_MODEL = os.environ.get("TEI_RERANK_MODEL", "tei-rerank-live")

requires_embed = pytest.mark.skipif(
    not _EMBED_URL, reason="Requires TEI_EMBED_URL (live TEI embedding instance)"
)
requires_rerank = pytest.mark.skipif(
    not _RERANK_URL, reason="Requires TEI_RERANK_URL (live TEI reranker instance)"
)


@pytest.mark.integration
@requires_embed
async def test_tei_embed_returns_vectors_with_null_usage() -> None:
    """embed() on a couple of strings returns one real vector per input,
    in input order, with no fabricated usage record."""
    provider = TeiEmbeddingProvider(base_url=str(_EMBED_URL), model=_EMBED_MODEL)
    try:
        response = await provider.embed(["the moon orbits the earth", "lunar regolith is abrasive"])
    finally:
        await provider.aclose()

    assert len(response.vectors) == 2
    dim = len(response.vectors[0])
    assert dim > 0
    assert all(len(v) == dim for v in response.vectors)
    assert response.dimensions == dim
    # TEI /embed returns no usage object -> usage is null (never fabricated).
    assert response.usage is None
    assert response.response_id is None


@pytest.mark.integration
@requires_embed
async def test_tei_embed_chunk_and_stitch_over_chunk_size() -> None:
    """A chunk_size smaller than the input list exercises the mandatory
    /embed chunk-and-stitch: one vector per input, in input order, with the
    same dimensionality across the chunk boundary and no fabricated usage."""
    inputs = [
        "the moon orbits the earth",
        "lunar regolith is abrasive",
        "the far side never faces earth",
    ]
    # chunk_size 2 over 3 inputs => 2 /embed requests (sizes 2, 1).
    provider = TeiEmbeddingProvider(base_url=str(_EMBED_URL), model=_EMBED_MODEL, chunk_size=2)
    try:
        response = await provider.embed(inputs)
    finally:
        await provider.aclose()

    assert len(response.vectors) == len(inputs)
    dim = len(response.vectors[0])
    assert dim > 0
    assert all(len(v) == dim for v in response.vectors)
    assert response.dimensions == dim
    # TEI /embed reports no usage object and no id -> both null across the stitch.
    assert response.usage is None
    assert response.response_id is None


@pytest.mark.integration
@requires_rerank
async def test_tei_rerank_returns_sorted_results() -> None:
    """rerank() on a small pool returns results sorted by relevance
    descending, each index valid into the input documents."""
    documents = [
        "The Sea of Tranquility was the Apollo 11 landing site.",
        "Cheese is made from milk.",
        "The lunar south pole holds water ice in permanently shadowed craters.",
    ]
    provider = TeiRerankProvider(base_url=str(_RERANK_URL), model=_RERANK_MODEL)
    try:
        response = await provider.rerank("Where on the moon is there water ice?", documents)
    finally:
        await provider.aclose()

    scores = [r.relevance_score for r in response.results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= r.index < len(documents) for r in response.results)
    assert len({r.index for r in response.results}) == len(response.results)
    assert response.usage is None
    assert response.response_id is None


@pytest.mark.integration
@requires_rerank
async def test_tei_rerank_chunk_and_stitch_global_sort() -> None:
    """A chunk_size smaller than the pool exercises the mandatory
    chunk-and-stitch: results are globally sorted with absolute indices and
    exactly min(top_k, len) results are returned."""
    documents = [
        "The moon has no atmosphere to speak of.",
        "A lunar day lasts about 29.5 earth days.",
        "Regolith blankets the lunar surface.",
        "The far side of the moon faces away from earth.",
        "Moonquakes were detected by the Apollo seismometers.",
    ]
    top_k = 3
    # chunk_size 2 over 5 documents => 3 /rerank requests (sizes 2, 2, 1).
    provider = TeiRerankProvider(base_url=str(_RERANK_URL), model=_RERANK_MODEL, chunk_size=2)
    try:
        response = await provider.rerank("What covers the lunar surface?", documents, top_k=top_k)
    finally:
        await provider.aclose()

    assert len(response.results) == min(top_k, len(documents))
    scores = [r.relevance_score for r in response.results]
    assert scores == sorted(scores, reverse=True)
    # Absolute indices into the original 5-document list, no duplicates.
    assert all(0 <= r.index < len(documents) for r in response.results)
    assert len({r.index for r in response.results}) == len(response.results)
