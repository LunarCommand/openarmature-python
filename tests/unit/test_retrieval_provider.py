"""Unit tests for the retrieval-provider embedding capability.

Covers behavior the spec conformance fixtures (001-005) do not pin: the
base_url guard, the readiness-probe modes, and the bundled observers'
safe handling of the typed embedding events (the spans / observations are
a follow-up; until then the observers must skip the events rather than
fall through to the NodeEvent phase dispatch).
"""

from __future__ import annotations

from typing import Literal

import httpx
import pytest

from openarmature.graph.events import EmbeddingEvent, EmbeddingFailedEvent
from openarmature.llm.errors import ProviderInvalidModel, ProviderInvalidResponse
from openarmature.retrieval import OpenAIEmbeddingProvider


def _ok_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/v1/embeddings"):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "m",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 1},
            },
        )
    return httpx.Response(200, json={"object": "list", "data": [{"id": "m"}]})


def test_base_url_rejects_v1_suffix() -> None:
    with pytest.raises(ValueError, match="host root"):
        OpenAIEmbeddingProvider(base_url="https://api.openai.com/v1", model="m")
    # The host root is accepted (no doubled /v1/v1).
    OpenAIEmbeddingProvider(base_url="https://api.openai.com", model="m")


def test_readiness_probe_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="readiness_probe must be one of"):
        OpenAIEmbeddingProvider(base_url="http://x", model="m", readiness_probe="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "expected_paths"),
    [
        ("embed", ["/v1/embeddings"]),
        ("models", ["/v1/models"]),
        ("both", ["/v1/models", "/v1/embeddings"]),
    ],
)
async def test_readiness_probe_modes_hit_expected_paths(
    mode: Literal["embed", "models", "both"], expected_paths: list[str]
) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return _ok_handler(request)

    provider = OpenAIEmbeddingProvider(
        base_url="http://x",
        model="m",
        readiness_probe=mode,
        transport=httpx.MockTransport(handler),
    )
    await provider.ready()
    assert paths == expected_paths
    await provider.aclose()


async def test_readiness_models_probe_raises_when_model_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": [{"id": "other-model"}]})

    provider = OpenAIEmbeddingProvider(
        base_url="http://x",
        model="m",
        readiness_probe="models",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderInvalidModel):
        await provider.ready()
    await provider.aclose()


async def test_readiness_models_probe_raises_on_malformed_catalog() -> None:
    # A 200 OK whose body is not a JSON object, or whose 'data' is not a
    # list, is a wire-format problem (proxy/backend), not a missing model:
    # it surfaces as ProviderInvalidResponse, not ProviderInvalidModel.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": "not-a-list"})

    provider = OpenAIEmbeddingProvider(
        base_url="http://x",
        model="m",
        readiness_probe="models",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderInvalidResponse):
        await provider.ready()
    await provider.aclose()


def _embedding_event() -> EmbeddingEvent:
    return EmbeddingEvent(
        invocation_id="inv",
        correlation_id=None,
        node_name="embed_node",
        namespace=("embed_node",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
        provider="openai",
        model="m",
        response_id=None,
        response_model=None,
        usage=None,
        latency_ms=1.0,
        input_strings=["x"],
        input_count=1,
        dimensions=2,
        output_vectors=[[0.1, 0.2]],
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
        call_id="c",
    )


def _embedding_failed_event() -> EmbeddingFailedEvent:
    return EmbeddingFailedEvent(
        invocation_id="inv",
        correlation_id=None,
        node_name="embed_node",
        namespace=("embed_node",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
        provider="openai",
        model="m",
        latency_ms=1.0,
        input_strings=["x"],
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
        call_id="c",
        error_category="provider_unavailable",
        error_message="boom",
    )


async def test_otel_observer_embedding_no_op_without_invocation_context() -> None:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openarmature.observability.otel import OTelObserver

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    # The observer renders openarmature.embedding.complete spans from these
    # events (covered end-to-end by conformance fixture 082). Here they are
    # delivered outside any invocation: the observer reads the invocation_id
    # from the ContextVar (None here) and returns early, so no span is opened.
    await observer(_embedding_event())
    await observer(_embedding_failed_event())
    assert len(exporter.get_finished_spans()) == 0


async def test_langfuse_observer_embedding_no_op_without_invocation_context() -> None:
    from openarmature.observability.langfuse import InMemoryLangfuseClient, LangfuseObserver

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    # The observer renders dedicated Embedding observations from these events
    # (covered end-to-end by conformance fixtures 083 / 137). Delivered outside
    # any invocation, the observer returns early on the None invocation_id, so
    # no trace / observation is created.
    await observer(_embedding_event())
    await observer(_embedding_failed_event())
    assert client.traces == {}
