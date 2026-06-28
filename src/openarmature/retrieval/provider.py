# Spec: realizes retrieval-provider §3 (EmbeddingProvider protocol +
# operations) and §4 (the response invariants). The §7 error categories
# are shared cross-capability with llm-provider, so the embedding path
# raises the existing llm-provider error classes (the embedding-applicable
# subset) rather than defining its own taxonomy.

"""EmbeddingProvider Protocol + input / response validation.

An ``EmbeddingProvider`` is stateless; every call carries the full input
list. It does not cache, retry, or fall back (those compose above via
pipeline-utilities middleware). A provider exposes two operations:

- ``async ready() -> None``: verifies the bound model is reachable.
- ``async embed(input, *, config=None) -> EmbeddingResponse``: performs a
  single embedding call, preserving input order.

This module also exports two validators that bracket the per-call flow:
:func:`validate_embedding_input` (pre-send, list non-empty) and
:func:`validate_embedding_response` (post-receive, the vector-count and
dimensionality invariants). They raise ``provider_invalid_request`` and
``provider_invalid_response`` respectively.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openarmature.llm.errors import ProviderInvalidRequest, ProviderInvalidResponse

from .response import EmbeddingResponse, EmbeddingRuntimeConfig


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
        embedding of ``input[i]`` -- order is preserved, never permuted.
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


__all__ = [
    "EmbeddingProvider",
    "validate_embedding_input",
    "validate_embedding_response",
]
