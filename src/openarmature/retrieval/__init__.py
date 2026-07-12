"""The retrieval-provider capability.

The embedding + rerank provider protocols, their response types, and the
bundled reference providers (an OpenAI-compatible embedding provider, a
Cohere-shape reranker, and the TEI embedding + rerank providers). Embedding and
rerank are sibling surfaces on the same capability.
"""

from __future__ import annotations

from .provider import (
    EmbeddingProvider,
    RerankProvider,
    validate_embedding_input,
    validate_embedding_response,
    validate_rerank_input,
    validate_rerank_response,
)
from .providers.cohere import CohereRerankProvider
from .providers.jina import JinaEmbeddingProvider, JinaRerankProvider
from .providers.openai import OpenAIEmbeddingProvider
from .providers.tei import TeiEmbeddingProvider, TeiRerankProvider
from .response import (
    EmbeddingResponse,
    EmbeddingRuntimeConfig,
    EmbeddingUsage,
    RerankResponse,
    RerankRuntimeConfig,
    RerankUsage,
    ScoredDocument,
)

__all__ = [
    "CohereRerankProvider",
    "EmbeddingProvider",
    "EmbeddingResponse",
    "EmbeddingRuntimeConfig",
    "EmbeddingUsage",
    "JinaEmbeddingProvider",
    "JinaRerankProvider",
    "OpenAIEmbeddingProvider",
    "RerankProvider",
    "RerankResponse",
    "RerankRuntimeConfig",
    "RerankUsage",
    "ScoredDocument",
    "TeiEmbeddingProvider",
    "TeiRerankProvider",
    "validate_embedding_input",
    "validate_embedding_response",
    "validate_rerank_input",
    "validate_rerank_response",
]
