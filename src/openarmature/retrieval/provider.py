# Spec: realizes retrieval-provider §3 (EmbeddingProvider protocol +
# operations) + §4 (the embedding response invariants) and §5 (RerankProvider
# protocol + operations) + §6 (the rerank response invariants). The §7 error
# categories are shared cross-capability with llm-provider, so both paths
# raise the existing llm-provider error classes (the retrieval-applicable
# subset) rather than defining their own taxonomy.

"""EmbeddingProvider / RerankProvider protocols + input / response validation.

An ``EmbeddingProvider`` / ``RerankProvider`` is stateless; every call
carries the full input. It does not cache, retry, or fall back (those
compose above via pipeline-utilities middleware). Each protocol exposes two
operations:

- ``async ready() -> None``: verifies the bound model is reachable.
- ``async embed(input, *, config=None) -> EmbeddingResponse`` / ``async
  rerank(query, documents, *, top_k=None, config=None) -> RerankResponse``:
  performs a single call, preserving order semantics.

This module also exports the validators that bracket each per-call flow:
:func:`validate_embedding_input` / :func:`validate_rerank_input` (pre-send)
and :func:`validate_embedding_response` / :func:`validate_rerank_response`
(post-receive). They raise ``provider_invalid_request`` and
``provider_invalid_response`` respectively.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openarmature.llm.errors import ProviderInvalidRequest, ProviderInvalidResponse

from .response import (
    EmbeddingResponse,
    EmbeddingRuntimeConfig,
    RerankResponse,
    RerankRuntimeConfig,
    ScoredDocument,
)


class EmbeddingProvider(Protocol):
    """The shape of any retrieval-provider embedding implementation.

    Implementations are bound to a single embedding model identifier;
    switching models means constructing a new provider, not passing a
    different argument per call.
    """

    async def ready(self) -> None:
        """Verify the bound embedding model is reachable and serving."""
        ...

    async def embed(
        self,
        input: Sequence[str],
        *,
        config: EmbeddingRuntimeConfig | None = None,
    ) -> EmbeddingResponse:
        """Embed ``input`` into one vector per string, in input order.

        Args:
            input: The strings to embed. Always a list, even for a
                single-string caller (wrap as a one-element list). Not
                mutated by the implementation.
            config: Optional per-call request parameters.

        Returns an :class:`EmbeddingResponse` whose ``vectors[i]`` is the
        embedding of ``input[i]``; order is preserved, never permuted.
        """
        ...


def validate_embedding_input(input: Sequence[str]) -> None:
    """Validate the input list before sending.

    Raises :class:`ProviderInvalidRequest` when the input list is empty.
    """
    if not input:
        raise ProviderInvalidRequest("embedding input list must not be empty")


def validate_embedding_response(
    vectors: Sequence[Sequence[float]],
    input_count: int,
) -> int:
    """Validate the response invariants and return the dimensionality.

    Raises :class:`ProviderInvalidResponse` when the vector count does not
    match the input count, when the response carries no vectors, or when
    the vectors are not all the same length. Returns the (consistent)
    dimensionality on success.
    """
    if len(vectors) != input_count:
        raise ProviderInvalidResponse(f"provider returned {len(vectors)} vectors for {input_count} inputs")
    if not vectors:
        raise ProviderInvalidResponse("provider returned no vectors")
    dimensions = len(vectors[0])
    for vector in vectors:
        if len(vector) != dimensions:
            raise ProviderInvalidResponse("provider returned vectors with inconsistent dimensionality")
    return dimensions


class RerankProvider(Protocol):
    """The shape of any retrieval-provider rerank implementation.

    Implementations are bound to a single rerank model identifier;
    switching models means constructing a new provider, not passing a
    different argument per call.
    """

    async def ready(self) -> None:
        """Verify the bound rerank model is reachable and serving."""
        ...

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        config: RerankRuntimeConfig | None = None,
    ) -> RerankResponse:
        """Score ``documents`` against ``query``, sorted by relevance.

        Args:
            query: The query string the documents are scored against.
            documents: The candidate documents. Always a list, even for a
                single-document caller (wrap as a one-element list). Not
                mutated by the implementation.
            top_k: The maximum number of results to return. ``None`` means
                "all" (up to ``len(documents)``).
            config: Optional per-call request parameters.

        Returns a :class:`RerankResponse` whose ``results`` are sorted by
        ``relevance_score`` descending; each result's ``index`` keys back
        into the input ``documents`` list.
        """
        ...


def validate_rerank_input(
    query: str,
    documents: Sequence[str],
    top_k: int | None,
) -> None:
    """Validate the rerank inputs before sending.

    Raises :class:`ProviderInvalidRequest` when the query is empty, the
    documents list is empty, or ``top_k`` is supplied and not positive.
    ``top_k`` may exceed ``len(documents)``; that is allowed.
    """
    if not query:
        raise ProviderInvalidRequest("rerank query must not be empty")
    if not documents:
        raise ProviderInvalidRequest("rerank documents list must not be empty")
    if top_k is not None and top_k <= 0:
        raise ProviderInvalidRequest(f"rerank top_k must be positive (got {top_k})")


def validate_rerank_response(
    results: Sequence[ScoredDocument],
    document_count: int,
    top_k: int | None,
) -> None:
    """Validate the rerank response invariants.

    Raises :class:`ProviderInvalidResponse` when a result's ``index`` is
    out of range for the input documents, when an ``index`` appears twice,
    or when ``top_k`` is supplied and the provider returned more results
    than requested.
    """
    seen: set[int] = set()
    for result in results:
        index = result.index
        if index < 0 or index >= document_count:
            raise ProviderInvalidResponse(
                f"rerank result index {index} out of range for {document_count} documents"
            )
        if index in seen:
            raise ProviderInvalidResponse(f"rerank response has duplicate index {index}")
        seen.add(index)
    if top_k is not None and len(results) > top_k:
        raise ProviderInvalidResponse(f"rerank provider returned {len(results)} results for top_k={top_k}")


__all__ = [
    "EmbeddingProvider",
    "RerankProvider",
    "validate_embedding_input",
    "validate_embedding_response",
    "validate_rerank_input",
    "validate_rerank_response",
]
