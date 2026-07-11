"""Unit tests for the retrieval-provider embedding + rerank capability.

Covers behavior the spec conformance fixtures do not pin: the base_url
guards, the readiness-probe modes, the rerank validators + request-body
mapping + typed-event dispatch, and the bundled observers' safe handling of
the typed embedding / rerank events (the spans / observations are a follow-up;
until then the observers must skip the events rather than fall through to the
NodeEvent phase dispatch).
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
import pytest

from openarmature.graph.events import (
    EmbeddingEvent,
    EmbeddingFailedEvent,
    RerankEvent,
    RerankFailedEvent,
)
from openarmature.graph.observer import ObserverEvent
from openarmature.llm.errors import (
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderUnavailable,
)
from openarmature.observability.correlation import _reset_active_dispatch, _set_active_dispatch
from openarmature.retrieval import (
    CohereRerankProvider,
    EmbeddingRuntimeConfig,
    OpenAIEmbeddingProvider,
    RerankRuntimeConfig,
    ScoredDocument,
    TeiEmbeddingProvider,
    TeiRerankProvider,
    validate_rerank_input,
    validate_rerank_response,
)


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


# ---------------------------------------------------------------------------
# Rerank capability (proposal 0060)
# ---------------------------------------------------------------------------


def _rerank_body(
    *,
    id: str = "rerank-id",
    model: str | None = "rerank-test",
    results: list[dict[str, Any]],
    search_units: int | None = 1,
    input_tokens: int | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"id": id, "results": results}
    if model is not None:
        body["model"] = model
    billed: dict[str, Any] = {}
    if search_units is not None:
        billed["search_units"] = search_units
    if input_tokens is not None:
        billed["input_tokens"] = input_tokens
    if billed:
        body["meta"] = {"billed_units": billed}
    return body


def _rerank_provider(
    handler: Any,
    *,
    model: str = "rerank-test",
) -> CohereRerankProvider:
    return CohereRerankProvider(
        base_url="http://mock-rerank.test",
        model=model,
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )


# --- validators -------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "documents", "top_k"),
    [
        ("", ["a"], None),  # empty query
        ("q", [], None),  # empty documents
        ("q", ["a"], 0),  # top_k == 0
        ("q", ["a"], -3),  # top_k < 0
    ],
)
def test_validate_rerank_input_rejects(query: str, documents: list[str], top_k: int | None) -> None:
    with pytest.raises(ProviderInvalidRequest):
        validate_rerank_input(query, documents, top_k)


def test_validate_rerank_input_allows_top_k_exceeding_documents() -> None:
    # top_k MAY exceed len(documents) -- that is allowed, not an error.
    validate_rerank_input("q", ["a", "b"], 10)


def test_validate_rerank_response_rejects_out_of_range_index() -> None:
    with pytest.raises(ProviderInvalidResponse):
        validate_rerank_response([ScoredDocument(index=5, relevance_score=0.9)], 2, None)


def test_validate_rerank_response_rejects_duplicate_index() -> None:
    results = [
        ScoredDocument(index=0, relevance_score=0.9),
        ScoredDocument(index=0, relevance_score=0.4),
    ]
    with pytest.raises(ProviderInvalidResponse):
        validate_rerank_response(results, 3, None)


def test_validate_rerank_response_rejects_more_results_than_top_k() -> None:
    results = [ScoredDocument(index=i, relevance_score=0.5) for i in range(3)]
    with pytest.raises(ProviderInvalidResponse):
        validate_rerank_response(results, 3, top_k=2)


# --- rerank() behavior ------------------------------------------------------


async def test_rerank_sorts_results_and_populates_usage() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_rerank_body(
                results=[
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.1},
                ],
                search_units=1,
            ),
        )

    provider = _rerank_provider(handler)
    response = await provider.rerank("q", ["berlin", "paris", "madrid"])
    # Provider returned UNSORTED; the adapter sorts by relevance descending.
    assert [(r.index, r.relevance_score) for r in response.results] == [(1, 0.9), (0, 0.5), (2, 0.1)]
    assert response.model == "rerank-test"
    assert response.response_id == "rerank-id"
    assert response.usage is not None
    assert response.usage.search_units == 1
    assert response.usage.input_tokens is None
    # Cohere returns an object envelope, so raw is a dict (not the bare-array
    # form TEI uses); narrow the dict | list union before indexing.
    assert isinstance(response.raw, dict)
    assert response.raw["id"] == "rerank-id"
    await provider.aclose()


async def test_rerank_document_echo_passthrough_and_null() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_rerank_body(
                results=[
                    {"index": 0, "relevance_score": 0.9, "document": "alpha doc"},
                    {"index": 2, "relevance_score": 0.5, "document": "gamma doc"},
                    {"index": 1, "relevance_score": 0.2},
                ],
            ),
        )

    provider = _rerank_provider(handler)
    response = await provider.rerank("q", ["alpha doc", "beta doc", "gamma doc"])
    by_index = {r.index: r.document for r in response.results}
    # Echo preserved verbatim where present; None where omitted (never
    # auto-filled from the input documents list).
    assert by_index == {0: "alpha doc", 2: "gamma doc", 1: None}
    await provider.aclose()


async def test_rerank_no_usage_object_yields_null_usage() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_rerank_body(results=[{"index": 0, "relevance_score": 0.9}], search_units=None),
        )

    provider = _rerank_provider(handler)
    response = await provider.rerank("q", ["only"])
    # No usage object surfaced -> usage is None (never a fabricated all-null
    # record).
    assert response.usage is None
    await provider.aclose()


async def test_rerank_out_of_range_index_raises() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_rerank_body(
                results=[
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 5, "relevance_score": 0.4},
                ],
            ),
        )

    provider = _rerank_provider(handler)
    with pytest.raises(ProviderInvalidResponse):
        await provider.rerank("q", ["a", "b"])
    await provider.aclose()


async def test_rerank_top_k_maps_to_top_n_and_return_documents_not_sent() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json=_rerank_body(results=[{"index": 0, "relevance_score": 0.9}]),
        )

    provider = _rerank_provider(handler)
    # return_documents=True is a silent no-op on the Cohere wire (no such field);
    # max_tokens_per_doc rides the extras pass-through bag (model_validate so
    # the undeclared extra is accepted, mirroring the conformance config path).
    config = RerankRuntimeConfig.model_validate({"return_documents": True, "max_tokens_per_doc": 100})
    await provider.rerank("q", ["a", "b", "c"], top_k=2, config=config)
    body = captured[0]
    assert body["model"] == "rerank-test"
    assert body["query"] == "q"
    assert body["documents"] == ["a", "b", "c"]
    assert body["top_n"] == 2
    assert "return_documents" not in body
    assert body["max_tokens_per_doc"] == 100
    await provider.aclose()


# --- typed-event dispatch ---------------------------------------------------


def _collecting_dispatch() -> tuple[list[ObserverEvent], Any]:
    events: list[ObserverEvent] = []

    def _dispatch(event: ObserverEvent) -> None:
        events.append(event)

    token = _set_active_dispatch(_dispatch)
    return events, token


async def test_rerank_success_dispatches_rerank_event_with_fields() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_rerank_body(
                results=[
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.9},
                ],
                search_units=3,
            ),
        )

    provider = _rerank_provider(handler)
    events, token = _collecting_dispatch()
    try:
        await provider.rerank("q", ["a", "b"], top_k=2, config=RerankRuntimeConfig(return_documents=True))
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    rerank_events = [e for e in events if isinstance(e, RerankEvent)]
    failed = [e for e in events if isinstance(e, RerankFailedEvent)]
    assert len(rerank_events) == 1
    assert failed == []  # mutually exclusive
    event = rerank_events[0]
    assert event.provider == "cohere"
    assert event.model == "rerank-test"
    assert event.query == "q"
    assert event.documents == ["a", "b"]
    assert event.document_count == 2
    assert event.top_k == 2
    assert event.result_count == 2
    # output_results populated unconditionally on success (proposal 0089).
    assert [(r.index, r.relevance_score) for r in event.output_results] == [(1, 0.9), (0, 0.5)]
    assert event.usage is not None
    assert event.usage.search_units == 3
    # return_documents was explicitly supplied -> it appears in request_params
    # even though it is a wire no-op.
    assert event.request_params == {"return_documents": True}
    assert event.call_id


async def test_rerank_request_params_empty_when_no_config() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_rerank_body(results=[{"index": 0, "relevance_score": 0.9}]))

    provider = _rerank_provider(handler)
    events, token = _collecting_dispatch()
    try:
        await provider.rerank("q", ["a"])
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    event = next(e for e in events if isinstance(e, RerankEvent))
    assert event.request_params == {}


async def test_rerank_failure_dispatches_rerank_failed_event_only() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "down"})

    provider = _rerank_provider(handler)
    events, token = _collecting_dispatch()
    try:
        with pytest.raises(ProviderUnavailable):
            await provider.rerank("q", ["a", "b"])
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    failed = [e for e in events if isinstance(e, RerankFailedEvent)]
    success = [e for e in events if isinstance(e, RerankEvent)]
    assert len(failed) == 1
    assert success == []  # mutually exclusive
    event = failed[0]
    assert event.error_category == "provider_unavailable"
    assert event.error_type == "ProviderUnavailable"
    assert event.query == "q"
    assert event.documents == ["a", "b"]
    assert event.document_count == 2
    assert event.call_id


async def test_rerank_invalid_request_dispatches_failed_event_before_send() -> None:
    # An empty query fails pre-send validation; the RerankFailedEvent still
    # fires (dispatched alongside the exception, not in place of it).
    def handler(_req: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        raise AssertionError("provider must not be contacted on pre-send validation failure")

    provider = _rerank_provider(handler)
    events, token = _collecting_dispatch()
    try:
        with pytest.raises(ProviderInvalidRequest):
            await provider.rerank("", ["a"])
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    failed = [e for e in events if isinstance(e, RerankFailedEvent)]
    assert len(failed) == 1
    assert failed[0].error_category == "provider_invalid_request"


# --- construction guards + readiness ----------------------------------------


def test_rerank_base_url_rejects_v2_suffix() -> None:
    with pytest.raises(ValueError, match="host root"):
        CohereRerankProvider(base_url="https://api.cohere.com/v2", model="m")
    # The host root is accepted (no doubled /v2/v2).
    CohereRerankProvider(base_url="https://api.cohere.com", model="m")


async def test_rerank_ready_probe_surfaces_invalid_model_on_404() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "model not found"})

    provider = _rerank_provider(handler, model="nonexistent")
    with pytest.raises(ProviderInvalidModel):
        await provider.ready()
    await provider.aclose()


def _rerank_event() -> RerankEvent:
    return RerankEvent(
        invocation_id="inv",
        correlation_id=None,
        node_name="rerank_node",
        namespace=("rerank_node",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
        provider="cohere",
        model="rerank-test",
        response_id=None,
        response_model=None,
        usage=None,
        latency_ms=1.0,
        query="q",
        documents=["a"],
        document_count=1,
        top_k=None,
        result_count=1,
        output_results=[ScoredDocument(index=0, relevance_score=0.9)],
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
        call_id="c",
    )


def _rerank_failed_event() -> RerankFailedEvent:
    return RerankFailedEvent(
        invocation_id="inv",
        correlation_id=None,
        node_name="rerank_node",
        namespace=("rerank_node",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
        provider="cohere",
        model="rerank-test",
        latency_ms=1.0,
        query="q",
        documents=["a"],
        document_count=1,
        top_k=None,
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
        call_id="c",
        error_category="provider_unavailable",
        error_message="boom",
    )


async def test_otel_observer_rerank_no_op_without_invocation_context() -> None:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openarmature.observability.otel import OTelObserver

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    # The rerank span rendering lands in 0060b; for now the observer safely
    # skips the rerank events rather than falling through to the NodeEvent
    # phase dispatch (which would raise on the missing ``phase`` attribute).
    await observer(_rerank_event())
    await observer(_rerank_failed_event())
    assert len(exporter.get_finished_spans()) == 0


async def test_langfuse_observer_rerank_no_op_without_invocation_context() -> None:
    from openarmature.observability.langfuse import InMemoryLangfuseClient, LangfuseObserver

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    # The Langfuse Retriever observation lands in 0060b; for now the observer
    # safely skips the rerank events.
    await observer(_rerank_event())
    await observer(_rerank_failed_event())
    assert client.traces == {}


# ---------------------------------------------------------------------------
# TEI wire mapping (proposal 0077)
# ---------------------------------------------------------------------------


def _tei_embed_provider(handler: Any, **kwargs: Any) -> TeiEmbeddingProvider:
    return TeiEmbeddingProvider(
        base_url="http://tei-embed.test",
        model="tei-embed-test",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _tei_rerank_provider(handler: Any, *, chunk_size: int = 32) -> TeiRerankProvider:
    return TeiRerankProvider(
        base_url="http://tei-rerank.test",
        model="tei-rerank-test",
        chunk_size=chunk_size,
        transport=httpx.MockTransport(handler),
    )


# --- /embed input_type realization ------------------------------------------


@pytest.mark.parametrize(
    ("input_type", "expected_prompt"),
    [("query", "query"), ("document", "passage"), (None, None)],
)
async def test_tei_embed_input_type_maps_to_prompt_name(
    input_type: str | None, expected_prompt: str | None
) -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/embed"
        return httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4]])

    provider = _tei_embed_provider(handler, input_type_prompt_map={"query": "query", "document": "passage"})
    config = EmbeddingRuntimeConfig(input_type=input_type) if input_type is not None else None
    response = await provider.embed(["how tall is the tower?"], config=config)
    await provider.aclose()

    body = captured[0]
    assert body["inputs"] == ["how tall is the tower?"]
    if expected_prompt is None:
        # Absent input_type => no prompt_name; byte-identical to the symmetric
        # path (the body is exactly {"inputs": [...]}).
        assert set(body) == {"inputs"}
    else:
        assert body["prompt_name"] == expected_prompt
    # TEI /embed returns no usage object -> usage is null (never fabricated).
    assert response.usage is None
    assert response.response_id is None
    assert response.model == "tei-embed-test"
    assert response.dimensions == 4


async def test_tei_embed_dimensions_on_wire_omitted_when_unset() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=[[0.1, 0.2, 0.3, 0.4]])

    provider = _tei_embed_provider(handler)
    r0 = await provider.embed(["x"], config=EmbeddingRuntimeConfig(dimensions=4))
    # dimensions maps onto the wire when set; a second call without it omits it.
    await provider.embed(["y"])
    await provider.aclose()

    assert captured[0] == {"inputs": ["x"], "dimensions": 4}
    assert captured[1] == {"inputs": ["y"]}
    # raw is the verbatim deserialized response -- TEI's bare vector array, not
    # wrapped (proposal 0096: raw is the top-level shape the provider returned).
    assert r0.raw == [[0.1, 0.2, 0.3, 0.4]]


async def test_tei_embed_client_side_prefix_fallback() -> None:
    # No input_type_prompt_map, only client-side prefixes: the prefix is
    # prepended to the wire inputs, no prompt_name is sent, and the event's
    # input_strings carry the ORIGINAL (un-prefixed) caller input.
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=[[0.1, 0.2]])

    provider = _tei_embed_provider(handler, query_prefix="query: ", document_prefix="passage: ")
    events, token = _collecting_dispatch()
    try:
        await provider.embed(["moon phases"], config=EmbeddingRuntimeConfig(input_type="query"))
        await provider.embed(["the moon is bright"], config=EmbeddingRuntimeConfig(input_type="document"))
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    assert captured[0]["inputs"] == ["query: moon phases"]
    assert "prompt_name" not in captured[0]
    assert captured[1]["inputs"] == ["passage: the moon is bright"]
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    # The event carries the caller's original input, not the prefixed wire form.
    assert embed_events[0].input_strings == ["moon phases"]
    assert embed_events[0].request_params == {"input_type": "query"}


async def test_tei_embed_bad_shape_raises_invalid_response() -> None:
    # TEI /embed must return a bare vector array; a JSON object is malformed.
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [[0.1, 0.2]]})

    provider = _tei_embed_provider(handler)
    with pytest.raises(ProviderInvalidResponse):
        await provider.embed(["x"])
    await provider.aclose()


@pytest.mark.parametrize("status", [413, 422])
async def test_tei_embed_over_length_maps_to_invalid_request(status: int) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "too long", "error_type": "Validation"})

    provider = _tei_embed_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.embed(["a very long input"])
    await provider.aclose()


# --- /rerank wire + chunk-and-stitch ----------------------------------------


async def test_tei_rerank_wire_body_and_return_text() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/rerank"
        return httpx.Response(
            200,
            json=[
                {"index": 0, "score": 0.42, "text": "doc a"},
                {"index": 1, "score": 0.91, "text": "doc b"},
            ],
        )

    provider = _tei_rerank_provider(handler)
    response = await provider.rerank(
        "q", ["doc a", "doc b"], config=RerankRuntimeConfig(return_documents=True)
    )
    await provider.aclose()

    # truncate: false always (fail-loud); return_text tracks return_documents;
    # texts maps directly onto documents.
    assert captured[0] == {"query": "q", "texts": ["doc a", "doc b"], "truncate": False, "return_text": True}
    # Response sorted descending; echoed text surfaces verbatim on document.
    assert [(r.index, r.document) for r in response.results] == [(1, "doc b"), (0, "doc a")]
    assert response.usage is None
    assert response.response_id is None


async def test_tei_rerank_single_batch_sorts_and_null_usage() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        # Unsorted, chunk-local indices; TEI reports no id / no usage.
        return httpx.Response(
            200,
            json=[{"index": 0, "score": 0.42}, {"index": 1, "score": 0.91}, {"index": 2, "score": 0.08}],
        )

    provider = _tei_rerank_provider(handler)
    response = await provider.rerank("q", ["berlin", "paris", "madrid"])
    await provider.aclose()

    assert [(r.index, r.relevance_score) for r in response.results] == [(1, 0.91), (0, 0.42), (2, 0.08)]
    assert all(r.document is None for r in response.results)
    assert response.usage is None
    # raw is the single response's verbatim bare array (not wrapped, not a
    # 1-element list) -- proposal 0096.
    assert response.raw == [
        {"index": 0, "score": 0.42},
        {"index": 1, "score": 0.91},
        {"index": 2, "score": 0.08},
    ]


async def test_tei_rerank_chunk_and_stitch_global_sort_top_k() -> None:
    # Mirror fixture 015: 9 documents, chunk_size 4 => 3 requests with texts
    # sizes [4, 4, 1]. Each chunk's response uses CHUNK-LOCAL indices; the
    # mapping re-bases them to absolute positions, globally sorts across
    # chunks, and honors top_k=4. A per-chunk sort or a missing re-base would
    # produce the wrong answer.
    captured: list[dict[str, Any]] = []
    chunk_responses = iter(
        [
            [
                {"index": 2, "score": 0.10},
                {"index": 0, "score": 0.20},
                {"index": 3, "score": 0.55},
                {"index": 1, "score": 0.95},
            ],
            [
                {"index": 3, "score": 0.05},
                {"index": 1, "score": 0.30},
                {"index": 0, "score": 0.80},
                {"index": 2, "score": 0.88},
            ],
            [{"index": 0, "score": 0.65}],
        ]
    )

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=next(chunk_responses))

    documents = [f"doc {i}" for i in range(9)]
    provider = _tei_rerank_provider(handler, chunk_size=4)
    events, token = _collecting_dispatch()
    try:
        response = await provider.rerank("which is fastest?", documents, top_k=4)
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # Exactly 3 requests, texts sizes [4, 4, 1], consecutive slices, same query.
    assert len(captured) == 3
    assert [len(b["texts"]) for b in captured] == [4, 4, 1]
    assert captured[0]["texts"] == documents[0:4]
    assert captured[1]["texts"] == documents[4:8]
    assert captured[2]["texts"] == documents[8:9]
    assert all(b["query"] == "which is fastest?" for b in captured)
    assert all(b["truncate"] is False for b in captured)
    # Global top-4 with ABSOLUTE indices, drawn from chunks A, B, B, C.
    assert [(r.index, r.relevance_score) for r in response.results] == [
        (1, 0.95),
        (6, 0.88),
        (4, 0.80),
        (8, 0.65),
    ]
    # One RerankEvent per rerank() call (not per chunk); documents is the full
    # input; result_count is the final stitched count.
    rerank_events = [e for e in events if isinstance(e, RerankEvent)]
    assert len(rerank_events) == 1
    assert rerank_events[0].document_count == 9
    assert rerank_events[0].result_count == 4
    assert len(rerank_events[0].output_results) == 4
    # raw is the verbatim per-request list (chunk-and-stitch): the 3 bare
    # per-chunk responses in request order, not wrapped (proposal 0096).
    assert response.raw == [
        [
            {"index": 2, "score": 0.10},
            {"index": 0, "score": 0.20},
            {"index": 3, "score": 0.55},
            {"index": 1, "score": 0.95},
        ],
        [
            {"index": 3, "score": 0.05},
            {"index": 1, "score": 0.30},
            {"index": 0, "score": 0.80},
            {"index": 2, "score": 0.88},
        ],
        [{"index": 0, "score": 0.65}],
    ]


@pytest.mark.parametrize("status", [413, 422])
async def test_tei_rerank_over_length_maps_to_invalid_request(status: int) -> None:
    # §8.1 fail-loud: truncate: false makes TEI error on over-length input
    # (413 / 422); the mapping surfaces provider_invalid_request.
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(status, json={"error": "too long", "error_type": "Validation"})

    provider = _tei_rerank_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.rerank("summarize", ["a very long document"])
    await provider.aclose()
    # The wire request still carried truncate: false.
    assert captured[0]["truncate"] is False


async def test_tei_rerank_chunk_local_index_out_of_range_raises() -> None:
    # A chunk response index outside the chunk range is malformed.
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"index": 5, "score": 0.9}])

    provider = _tei_rerank_provider(handler, chunk_size=4)
    with pytest.raises(ProviderInvalidResponse):
        await provider.rerank("q", ["a", "b"])
    await provider.aclose()


def test_tei_base_url_strips_trailing_slash() -> None:
    provider = TeiEmbeddingProvider(base_url="http://tei-embed.test/", model="m")
    assert provider.base_url == "http://tei-embed.test"
    rerank = TeiRerankProvider(base_url="http://tei-rerank.test/", model="m")
    assert rerank.base_url == "http://tei-rerank.test"


def test_tei_chunk_size_must_be_positive() -> None:
    # A non-positive chunk_size is rejected at construction on both TEI
    # providers: a zero would raise a raw ValueError from range() mid-call
    # (escaping the provider's error handling), a negative would silently drop
    # every document from the rerank pool.
    for bad in (0, -1):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TeiRerankProvider(base_url="http://tei-rerank.test", model="m", chunk_size=bad)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TeiEmbeddingProvider(base_url="http://tei-embed.test", model="m", chunk_size=bad)
