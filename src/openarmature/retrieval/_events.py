# Shared observability-event builders for the retrieval reference providers.
# The four bundled providers (openai / tei / cohere / jina) construct the same
# typed EmbeddingEvent / EmbeddingFailedEvent / RerankEvent / RerankFailedEvent
# from identical ContextVar identity snapshots; the only per-provider variation
# is the provider identifier (gen_ai.system), the bound model, and the response
# fields. These free functions carry that construction once so a provider passes
# its identity + response in rather than duplicating the body.

"""Shared event builders for the retrieval reference providers.

Free functions that construct the typed embedding / rerank observer events
from a provider's identity (``provider`` / ``model`` /
``populate_caller_metadata``), the parsed response (or raised error), and the
per-call request context. The identity / scoping fields are sourced from the
calling-node correlation ContextVars at build time; the outcome fields come
from the response or exception.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openarmature.graph.events import (
    EmbeddingEvent,
    EmbeddingFailedEvent,
    RerankEvent,
    RerankFailedEvent,
)
from openarmature.llm.errors import LlmProviderError
from openarmature.observability.correlation import (
    current_attempt_index,
    current_branch_name,
    current_correlation_id,
    current_fan_out_index,
    current_invocation_id,
    current_namespace_prefix,
)
from openarmature.observability.metadata import AttributeValue, current_invocation_metadata

from .response import EmbeddingResponse, RerankResponse


def build_embedding_event(
    response: EmbeddingResponse,
    latency_ms: float,
    *,
    provider: str,
    model: str,
    populate_caller_metadata: bool,
    call_id: str,
    input_strings: list[str],
    request_params: dict[str, Any],
    request_extras: dict[str, Any],
    active_prompt: Any,
    active_prompt_group: Any,
) -> EmbeddingEvent:
    """Construct the typed EmbeddingEvent for the success path.

    Sources identity / scoping from the calling-node ContextVars and outcome
    fields from the response.
    """
    namespace = current_namespace_prefix()
    node_name = namespace[-1] if namespace else ""
    invocation_id = current_invocation_id() or ""
    caller_metadata: Mapping[str, AttributeValue] | None = None
    if populate_caller_metadata:
        caller_metadata = dict(current_invocation_metadata())
    return EmbeddingEvent(
        invocation_id=invocation_id,
        correlation_id=current_correlation_id(),
        node_name=node_name,
        namespace=namespace,
        attempt_index=current_attempt_index(),
        fan_out_index=current_fan_out_index(),
        branch_name=current_branch_name(),
        provider=provider,
        model=model,
        response_id=response.response_id,
        response_model=response.model,
        usage=response.usage,
        latency_ms=latency_ms,
        input_strings=input_strings,
        input_count=len(input_strings),
        dimensions=response.dimensions,
        # Populated unconditionally on success per observability §5.5.9;
        # privacy gating is observer-side at rendering (symmetric with
        # input_strings). Sources EmbeddingEvent.output_vectors from the
        # parsed response vectors for the §8.4.5 embedding.output mapping.
        output_vectors=response.vectors,
        request_params=request_params,
        request_extras=request_extras,
        active_prompt=active_prompt,
        active_prompt_group=active_prompt_group,
        call_id=call_id,
        caller_invocation_metadata=caller_metadata,
    )


def build_embedding_failed_event(
    exc: LlmProviderError,
    latency_ms: float,
    *,
    provider: str,
    model: str,
    populate_caller_metadata: bool,
    call_id: str,
    input_strings: list[str],
    request_params: dict[str, Any],
    request_extras: dict[str, Any],
    active_prompt: Any,
    active_prompt_group: Any,
) -> EmbeddingFailedEvent:
    """Construct the typed EmbeddingFailedEvent for the failure path.

    ``error_type`` defaults to the exception class name (the "upstream
    exception class name" style).
    """
    namespace = current_namespace_prefix()
    node_name = namespace[-1] if namespace else ""
    invocation_id = current_invocation_id() or ""
    caller_metadata: Mapping[str, AttributeValue] | None = None
    if populate_caller_metadata:
        caller_metadata = dict(current_invocation_metadata())
    return EmbeddingFailedEvent(
        invocation_id=invocation_id,
        correlation_id=current_correlation_id(),
        node_name=node_name,
        namespace=namespace,
        attempt_index=current_attempt_index(),
        fan_out_index=current_fan_out_index(),
        branch_name=current_branch_name(),
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        input_strings=input_strings,
        request_params=request_params,
        request_extras=request_extras,
        active_prompt=active_prompt,
        active_prompt_group=active_prompt_group,
        call_id=call_id,
        error_category=exc.category,
        error_type=type(exc).__name__,
        error_message=str(exc),
        caller_invocation_metadata=caller_metadata,
    )


def build_rerank_event(
    response: RerankResponse,
    latency_ms: float,
    *,
    provider: str,
    model: str,
    populate_caller_metadata: bool,
    call_id: str,
    query: str,
    documents: list[str],
    top_k: int | None,
    request_params: dict[str, Any],
    request_extras: dict[str, Any],
    active_prompt: Any,
    active_prompt_group: Any,
) -> RerankEvent:
    """Construct the typed RerankEvent for the success path.

    One event per rerank() call: ``documents`` is the full input and the
    results are the stitched output, not per-chunk. Sources identity / scoping
    from the calling-node ContextVars and outcome fields from the response.
    """
    namespace = current_namespace_prefix()
    node_name = namespace[-1] if namespace else ""
    invocation_id = current_invocation_id() or ""
    caller_metadata: Mapping[str, AttributeValue] | None = None
    if populate_caller_metadata:
        caller_metadata = dict(current_invocation_metadata())
    return RerankEvent(
        invocation_id=invocation_id,
        correlation_id=current_correlation_id(),
        node_name=node_name,
        namespace=namespace,
        attempt_index=current_attempt_index(),
        fan_out_index=current_fan_out_index(),
        branch_name=current_branch_name(),
        provider=provider,
        model=model,
        response_id=response.response_id,
        response_model=response.model,
        usage=response.usage,
        latency_ms=latency_ms,
        query=query,
        documents=documents,
        document_count=len(documents),
        top_k=top_k,
        result_count=len(response.results),
        # Populated unconditionally on success per proposal 0089; privacy
        # gating is observer-side at rendering (symmetric with query /
        # documents). Sources output_results from the parsed response.
        output_results=list(response.results),
        request_params=request_params,
        request_extras=request_extras,
        active_prompt=active_prompt,
        active_prompt_group=active_prompt_group,
        call_id=call_id,
        caller_invocation_metadata=caller_metadata,
    )


def build_rerank_failed_event(
    exc: LlmProviderError,
    latency_ms: float,
    *,
    provider: str,
    model: str,
    populate_caller_metadata: bool,
    call_id: str,
    query: str,
    documents: list[str],
    top_k: int | None,
    request_params: dict[str, Any],
    request_extras: dict[str, Any],
    active_prompt: Any,
    active_prompt_group: Any,
) -> RerankFailedEvent:
    """Construct the typed RerankFailedEvent for the failure path.

    ``error_type`` defaults to the exception class name (the "upstream
    exception class name" style).
    """
    namespace = current_namespace_prefix()
    node_name = namespace[-1] if namespace else ""
    invocation_id = current_invocation_id() or ""
    caller_metadata: Mapping[str, AttributeValue] | None = None
    if populate_caller_metadata:
        caller_metadata = dict(current_invocation_metadata())
    return RerankFailedEvent(
        invocation_id=invocation_id,
        correlation_id=current_correlation_id(),
        node_name=node_name,
        namespace=namespace,
        attempt_index=current_attempt_index(),
        fan_out_index=current_fan_out_index(),
        branch_name=current_branch_name(),
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        query=query,
        documents=documents,
        document_count=len(documents),
        top_k=top_k,
        request_params=request_params,
        request_extras=request_extras,
        active_prompt=active_prompt,
        active_prompt_group=active_prompt_group,
        call_id=call_id,
        error_category=exc.category,
        error_type=type(exc).__name__,
        error_message=str(exc),
        caller_invocation_metadata=caller_metadata,
    )


__all__ = [
    "build_embedding_event",
    "build_embedding_failed_event",
    "build_rerank_event",
    "build_rerank_failed_event",
]
