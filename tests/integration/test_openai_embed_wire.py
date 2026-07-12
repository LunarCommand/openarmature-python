"""Integration tests for the OpenAI-compatible embeddings wire mapping.

Gated by the ``OPENAI_API_KEY`` env var: the tests run only when it is set.
Skipped in CI and local runs that don't have it in scope; runs end-to-end
against the live ``/v1/embeddings`` endpoint when invoked with the key set.

``base_url`` is read from ``OPENAI_BASE_URL`` (default ``https://api.openai.com``,
origin only), and the bound model from ``OPENAI_EMBED_MODEL`` (default
``text-embedding-3-small``). Point ``OPENAI_BASE_URL`` at any OpenAI-compatible
backend (vLLM / LocalAI / TEI's OpenAI surface) to exercise the same mapping.
"""

from __future__ import annotations

import os

import pytest

from openarmature.retrieval import EmbeddingRuntimeConfig, OpenAIEmbeddingProvider

_API_KEY = os.environ.get("OPENAI_API_KEY")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

requires_openai = pytest.mark.skipif(
    not _API_KEY, reason="Requires OPENAI_API_KEY (live OpenAI-compatible key)"
)


@pytest.mark.integration
@requires_openai
async def test_openai_embed_returns_vectors_with_usage() -> None:
    """embed() on a couple of strings returns one real vector per input, in
    input order, with a token usage record (usage.prompt_tokens -> input_tokens)."""
    provider = OpenAIEmbeddingProvider(model=_EMBED_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.embed(["the moon orbits the earth", "lunar regolith is abrasive"])
    finally:
        await provider.aclose()

    assert len(response.vectors) == 2
    dim = len(response.vectors[0])
    assert dim > 0
    assert all(len(v) == dim for v in response.vectors)
    assert response.dimensions == dim
    assert response.usage is not None
    assert response.usage.input_tokens > 0


@pytest.mark.integration
@requires_openai
async def test_openai_embed_dimensions_truncates_vector_length() -> None:
    """embed(config={dimensions: N}) returns vectors of length N -- the
    text-embedding-3 family supports Matryoshka dimension truncation on the
    wire `dimensions` field."""
    provider = OpenAIEmbeddingProvider(model=_EMBED_MODEL, api_key=str(_API_KEY), base_url=_BASE_URL)
    try:
        response = await provider.embed(
            ["the sea of tranquility"], config=EmbeddingRuntimeConfig(dimensions=256)
        )
    finally:
        await provider.aclose()

    assert len(response.vectors) == 1
    assert len(response.vectors[0]) == 256
    assert response.dimensions == 256
