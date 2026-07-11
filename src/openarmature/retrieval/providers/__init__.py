"""Bundled retrieval-provider reference implementations."""

from __future__ import annotations

from .cohere import CohereRerankProvider
from .openai import OpenAIEmbeddingProvider
from .tei import TeiEmbeddingProvider, TeiRerankProvider

__all__ = [
    "CohereRerankProvider",
    "OpenAIEmbeddingProvider",
    "TeiEmbeddingProvider",
    "TeiRerankProvider",
]
