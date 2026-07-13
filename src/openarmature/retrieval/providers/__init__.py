"""Bundled retrieval-provider reference implementations."""

from __future__ import annotations

from .cohere import CohereEmbeddingProvider, CohereRerankProvider
from .jina import JinaEmbeddingProvider, JinaRerankProvider
from .openai import OpenAIEmbeddingProvider
from .tei import TeiEmbeddingProvider, TeiRerankProvider

__all__ = [
    "CohereEmbeddingProvider",
    "CohereRerankProvider",
    "JinaEmbeddingProvider",
    "JinaRerankProvider",
    "OpenAIEmbeddingProvider",
    "TeiEmbeddingProvider",
    "TeiRerankProvider",
]
