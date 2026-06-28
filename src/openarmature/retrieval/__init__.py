"""The retrieval-provider capability.

The embedding provider protocol, its response types, and the bundled
reference embedding provider. The rerank protocol is a sibling surface on
the same capability.
"""

from __future__ import annotations

from .provider import (
    EmbeddingProvider,
    validate_embedding_input,
    validate_embedding_response,
)
from .providers.openai import OpenAIEmbeddingProvider
from .response import EmbeddingResponse, EmbeddingRuntimeConfig, EmbeddingUsage

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResponse",
    "EmbeddingRuntimeConfig",
    "EmbeddingUsage",
    "OpenAIEmbeddingProvider",
    "validate_embedding_input",
    "validate_embedding_response",
]
