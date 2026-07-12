"""Bundled retrieval-provider reference implementations."""

from __future__ import annotations

from .cohere import CohereRerankProvider
from .jina import JinaEmbeddingProvider, JinaRerankProvider
from .openai import OpenAIEmbeddingProvider
from .tei import TeiEmbeddingProvider, TeiRerankProvider

__all__ = [
    "CohereRerankProvider",
    "JinaEmbeddingProvider",
    "JinaRerankProvider",
    "OpenAIEmbeddingProvider",
    "TeiEmbeddingProvider",
    "TeiRerankProvider",
]
