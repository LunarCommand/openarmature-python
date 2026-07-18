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
import logging
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
    ProviderRateLimit,
    ProviderUnavailable,
)
from openarmature.observability.correlation import _reset_active_dispatch, _set_active_dispatch
from openarmature.retrieval import (
    CohereEmbeddingProvider,
    CohereRerankProvider,
    EmbeddingRuntimeConfig,
    JinaEmbeddingProvider,
    JinaRerankProvider,
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


# --- OpenAI /v1/embeddings wire mapping (§8.3, fixtures 023-027) -------------


def _openai_embed_provider(
    handler: Any, *, model: str = "text-embedding-test", **kwargs: Any
) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        base_url="https://api.openai.invalid",
        model=model,
        api_key="test-openai-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _openai_embed_body(
    *,
    model: str = "text-embedding-test",
    data: list[dict[str, Any]],
    prompt_tokens: int = 6,
) -> dict[str, Any]:
    return {
        "object": "list",
        "model": model,
        "data": data,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


async def test_openai_embed_wire_body_array_form_and_input_order() -> None:
    captured: list[dict[str, Any]] = []
    auth: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        auth.append(req.headers.get("Authorization"))
        assert req.url.path == "/v1/embeddings"
        # `data` emitted OUT of input order (index 2, 0, 1) to make the
        # index-keyed input-order mapping load-bearing.
        return httpx.Response(
            200,
            json=_openai_embed_body(
                data=[
                    {"object": "embedding", "index": 2, "embedding": [0.21, 0.22, 0.23, 0.24]},
                    {"object": "embedding", "index": 0, "embedding": [0.01, 0.02, 0.03, 0.04]},
                    {"object": "embedding", "index": 1, "embedding": [0.11, 0.12, 0.13, 0.14]},
                ],
                prompt_tokens=6,
            ),
        )

    provider = _openai_embed_provider(handler)
    response = await provider.embed(["alpha string", "beta string", "gamma string"])
    await provider.aclose()

    # The wire body is exactly the array form {model, input} -- no dimensions /
    # encoding_format / input_type / task.
    assert captured[0] == {
        "model": "text-embedding-test",
        "input": ["alpha string", "beta string", "gamma string"],
    }
    assert auth[0] == "Bearer test-openai-key"
    # Vectors keyed by INPUT order, not data[] array position.
    assert response.vectors == [
        [0.01, 0.02, 0.03, 0.04],
        [0.11, 0.12, 0.13, 0.14],
        [0.21, 0.22, 0.23, 0.24],
    ]
    assert response.dimensions == 4
    assert response.usage is not None
    assert response.usage.input_tokens == 6
    # OpenAI embeddings carry no id.
    assert response.response_id is None


async def test_openai_embed_dimensions_on_wire_and_in_request_params() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json=_openai_embed_body(
                data=[{"object": "embedding", "index": 0, "embedding": [0.01, 0.02, 0.03, 0.04]}],
                prompt_tokens=2,
            ),
        )

    provider = _openai_embed_provider(handler)
    events, token = _collecting_dispatch()
    try:
        response = await provider.embed(["alpha string"], config=EmbeddingRuntimeConfig(dimensions=4))
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # dimensions maps onto the wire alongside {model, input}; no input_type /
    # task / encoding_format.
    assert captured[0] == {"model": "text-embedding-test", "input": ["alpha string"], "dimensions": 4}
    assert response.dimensions == 4
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    assert embed_events[0].request_params == {"dimensions": 4}


async def test_openai_embed_input_type_is_wire_noop_on_symmetric_provider() -> None:
    # No query_prefix / document_prefix bound (pure-symmetric OpenAI): input_type
    # is a true wire no-op -- byte-identical body, input verbatim -- yet still
    # reaches EmbeddingEvent.request_params (empty when absent).
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json=_openai_embed_body(
                data=[{"object": "embedding", "index": 0, "embedding": [0.10, 0.20, 0.30, 0.40]}],
                prompt_tokens=8,
            ),
        )

    provider = _openai_embed_provider(handler)
    events, token = _collecting_dispatch()
    try:
        await provider.embed(
            ["how tall is the eiffel tower?"], config=EmbeddingRuntimeConfig(input_type="query")
        )
        await provider.embed(["how tall is the eiffel tower?"])
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # The with-input_type body is byte-identical to the no-input_type body:
    # exactly {model, input}, input un-prefixed, no input_type / task field.
    assert captured[0] == {"model": "text-embedding-test", "input": ["how tall is the eiffel tower?"]}
    assert captured[0] == captured[1]
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    # input_type reaches request_params when supplied, empty when absent.
    assert embed_events[0].request_params == {"input_type": "query"}
    assert embed_events[1].request_params == {}


async def test_openai_embed_client_side_prefix_prepends_per_input_type() -> None:
    # An asymmetric model behind a compatible endpoint: query_prefix /
    # document_prefix bound at construction, so input_type selects which prefix
    # to prepend client-side. The wire `input` carries the prefixed text; the
    # event carries the UN-prefixed caller intent + input.
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json=_openai_embed_body(
                model="e5-base-en-test",
                data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}],
                prompt_tokens=9,
            ),
        )

    provider = _openai_embed_provider(
        handler, model="e5-base-en-test", query_prefix="query: ", document_prefix="passage: "
    )
    events, token = _collecting_dispatch()
    try:
        await provider.embed(
            ["how tall is the eiffel tower?"], config=EmbeddingRuntimeConfig(input_type="query")
        )
        await provider.embed(
            ["the eiffel tower is 330 metres tall."], config=EmbeddingRuntimeConfig(input_type="document")
        )
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # input_type "query" -> query_prefix prepend; "document" -> document_prefix.
    # No input_type / task field on the wire (the distinction lives in the text).
    assert captured[0] == {"model": "e5-base-en-test", "input": ["query: how tall is the eiffel tower?"]}
    assert captured[1] == {
        "model": "e5-base-en-test",
        "input": ["passage: the eiffel tower is 330 metres tall."],
    }
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    assert embed_events[0].request_params == {"input_type": "query"}
    assert embed_events[0].input_strings == ["how tall is the eiffel tower?"]
    assert embed_events[1].request_params == {"input_type": "document"}


async def test_openai_embed_base_url_override_routes_to_compatible_backend() -> None:
    # base_url override (a vLLM-style origin): the mapping appends the fixed
    # /v1/embeddings route to the overridden origin (no doubled /v1).
    urls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        urls.append(str(req.url))
        return httpx.Response(
            200,
            json=_openai_embed_body(
                data=[
                    {"object": "embedding", "index": 0, "embedding": [0.01, 0.02, 0.03, 0.04]},
                    {"object": "embedding", "index": 1, "embedding": [0.11, 0.12, 0.13, 0.14]},
                ],
                prompt_tokens=4,
            ),
        )

    provider = OpenAIEmbeddingProvider(
        base_url="http://vllm.invalid",
        model="text-embedding-test",
        api_key="test-openai-key",
        transport=httpx.MockTransport(handler),
    )
    response = await provider.embed(["alpha string", "beta string"])
    await provider.aclose()

    assert urls[0] == "http://vllm.invalid/v1/embeddings"
    assert len(response.vectors) == 2


async def test_openai_embed_base_url_defaults_to_openai_origin() -> None:
    # base_url is optional per §8.3: an unspecified base_url binds the OpenAI
    # origin (host root; the provider appends /v1/embeddings itself).
    provider = OpenAIEmbeddingProvider(model="text-embedding-test", api_key="k")
    try:
        assert provider.base_url == "https://api.openai.com"
    finally:
        await provider.aclose()


async def test_openai_embed_chunk_and_stitch_over_2048_cap() -> None:
    # 2049 inputs, OpenAI's 2048-input cap => exactly 2 requests with input
    # sizes [2048, 1] over the consecutive slices input[0:2048] / input[2048:],
    # identical per-call params on each. The per-chunk data[] are stitched in
    # input order across the chunk boundary; input_tokens summed; the response
    # identity (id + model) from the FIRST chunk; raw the list of the 2
    # per-chunk objects (0096). One EmbeddingEvent per embed() call.
    inputs = [f"doc-{i:04d}" for i in range(2049)]
    chunk_a = [[i / 10000, 0.5] for i in range(2048)]
    chunk_b = [[2048 / 10000, 0.5]]
    responses = iter(
        [
            _openai_embed_body(
                data=[{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(chunk_a)],
                prompt_tokens=4096,
            )
            | {"id": "chunk-a"},
            _openai_embed_body(
                data=[{"object": "embedding", "index": 0, "embedding": chunk_b[0]}],
                prompt_tokens=2,
            )
            | {"id": "chunk-b"},
        ]
    )
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/v1/embeddings"
        return httpx.Response(200, json=next(responses))

    provider = _openai_embed_provider(handler)
    try:
        response = await provider.embed(inputs)
    finally:
        await provider.aclose()

    # Exactly 2 requests, input sizes [2048, 1], consecutive slices, identical
    # per-call params on each. The mapping MUST NOT send an over-cap request.
    assert len(captured) == 2
    assert [len(b["input"]) for b in captured] == [2048, 1]
    assert captured[0]["input"] == inputs[0:2048]
    assert captured[1]["input"] == inputs[2048:2049]
    for b in captured:
        assert b["model"] == "text-embedding-test"
        assert "dimensions" not in b
    # Stitched in input order across the boundary; usage summed; the response
    # identity from the first chunk.
    assert response.vectors == chunk_a + chunk_b
    assert len(response.vectors) == 2049
    assert response.dimensions == 2
    assert response.usage is not None
    assert response.usage.input_tokens == 4098
    assert response.response_id == "chunk-a"
    assert response.model == "text-embedding-test"
    # raw is the list of the 2 per-chunk objects verbatim, in request order.
    assert isinstance(response.raw, list)
    assert len(response.raw) == 2


async def test_openai_embed_single_request_when_within_cap() -> None:
    # <= 2048 inputs is the single-request path (no chunking), and raw is that
    # one response object -- NOT a one-element list (0096).
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json=_openai_embed_body(data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]),
        )

    provider = _openai_embed_provider(handler)
    try:
        response = await provider.embed(["alpha string"])
    finally:
        await provider.aclose()

    assert len(captured) == 1
    assert isinstance(response.raw, dict)


async def test_openai_embed_chunked_usage_is_all_or_nothing() -> None:
    # Usage is summed ONLY when every chunk reports it. A 2049-input call
    # (2 chunks) where chunk 1 reports prompt_tokens but chunk 2 omits the usage
    # block yields usage = null for the WHOLE call, not a confident partial sum:
    # the total is unaccountable, and a partial sum would both undercount and
    # diverge from the single-request path (which returns null for the same
    # unaccountable figure). Guards against the chunked/unchunked inconsistency.
    inputs = [f"doc-{i:04d}" for i in range(2049)]
    chunk_a = [[i / 10000, 0.5] for i in range(2048)]
    body_a = _openai_embed_body(
        data=[{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(chunk_a)],
        prompt_tokens=4096,
    )
    body_b = _openai_embed_body(data=[{"object": "embedding", "index": 0, "embedding": [0.2048, 0.5]}])
    del body_b["usage"]  # chunk 2 reports no usage block at all
    responses = iter([body_a, body_b])

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/embeddings"
        return httpx.Response(200, json=next(responses))

    provider = _openai_embed_provider(handler)
    try:
        response = await provider.embed(inputs)
    finally:
        await provider.aclose()

    assert len(response.vectors) == 2049
    # Chunk 2 reported nothing, so the whole-call total is unaccountable -> null.
    assert response.usage is None


# --- nonneg_int: warn on a present-but-corrupt usage figure (proposal 0093) --
#
# The shared usage-figure reader distinguishes "not reported" (None -- the
# ordinary no-usage case, every TEI call) from "present but unusable" (a
# misbehaving provider / gateway populated the field with garbage). The former
# is quiet; the latter logs a WARNING so a corrupt count is not silently
# indistinguishable from the healthy absent case. Both still return None.


def test_nonneg_int_passes_through_a_non_negative_int_without_warning(caplog: Any) -> None:
    from openarmature.retrieval._wire import nonneg_int

    with caplog.at_level(logging.WARNING):
        assert nonneg_int(6, field="prompt_tokens") == 6
        assert nonneg_int(0, field="prompt_tokens") == 0
    assert caplog.records == []


def test_nonneg_int_absent_is_quiet(caplog: Any) -> None:
    from openarmature.retrieval._wire import nonneg_int

    # None is "not reported" -- the common no-usage case. It MUST NOT warn, else
    # every TEI call would log noise.
    with caplog.at_level(logging.WARNING):
        assert nonneg_int(None, field="prompt_tokens") is None
    assert caplog.records == []


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param("six", id="string"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(6.0, id="float"),
    ],
)
def test_nonneg_int_present_but_corrupt_warns(corrupt: Any, caplog: Any) -> None:
    from openarmature.retrieval._wire import nonneg_int

    with caplog.at_level(logging.WARNING):
        assert nonneg_int(corrupt, field="prompt_tokens") is None
    # Exactly one WARNING, naming the field so the log is actionable.
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "prompt_tokens" in caplog.text


# --- Nullable usage, OpenAI-compatible embeddings (proposal 0093, §4) --------
#
# usage is a record or null. An absent or malformed usage block MUST NOT be
# fabricated into a record (nor a zero -- a zero asserts "zero tokens billed",
# which the provider never claimed). The hosted OpenAI wire always reports
# usage; the null path is what an OpenAI-COMPATIBLE backend that omits the
# block produces. Fixtures 139 / 140 render this case through the observers.


async def test_openai_embed_absent_usage_block_yields_null_usage() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        body = _openai_embed_body(data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}])
        del body["usage"]
        return httpx.Response(200, json=body)

    provider = _openai_embed_provider(handler)
    try:
        response = await provider.embed(["alpha string"])
    finally:
        await provider.aclose()

    # No usage object on the wire -> no record. The vectors are unaffected.
    assert response.usage is None
    assert response.vectors == [[0.1, 0.2]]


@pytest.mark.parametrize(
    "prompt_tokens",
    [
        pytest.param("six", id="string"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="null"),
    ],
)
async def test_openai_embed_malformed_prompt_tokens_yields_null_usage(prompt_tokens: Any) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        body = _openai_embed_body(data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}])
        body["usage"] = {"prompt_tokens": prompt_tokens}
        return httpx.Response(200, json=body)

    provider = _openai_embed_provider(handler)
    try:
        response = await provider.embed(["alpha string"])
    finally:
        await provider.aclose()

    # A corrupt figure reads as "not reported" rather than failing the call:
    # the embed succeeded and the vectors are sound, so a bad secondary
    # accounting field surfaces as unknown (usage = null), not as a
    # provider_invalid_response. Matches the Cohere / Jina mappings.
    assert response.usage is None
    assert response.vectors == [[0.1, 0.2]]


async def test_openai_embed_reported_zero_prompt_tokens_yields_a_record() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_embed_body(
                data=[{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                prompt_tokens=0,
            ),
        )

    provider = _openai_embed_provider(handler)
    try:
        response = await provider.embed(["alpha string"])
    finally:
        await provider.aclose()

    # A REPORTED zero is a claim the provider actually made -- distinct from an
    # absent figure. It yields a record carrying 0, NOT usage = null.
    assert response.usage is not None
    assert response.usage.input_tokens == 0


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


async def test_cohere_rerank_document_echo_shared_rule() -> None:
    # §6 (0097): Cohere is on the shared document_echo rule -- a TextDoc object
    # echo unwraps to its text; a text-less object (or absent) yields null.
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_rerank_body(
                results=[
                    {"index": 0, "relevance_score": 0.9, "document": {"text": "echoed"}},
                    {"index": 1, "relevance_score": 0.5, "document": {"image": "..."}},
                ]
            ),
        )

    provider = _rerank_provider(handler)
    response = await provider.rerank("q", ["a", "b"], config=RerankRuntimeConfig(return_documents=True))
    await provider.aclose()
    assert {r.index: r.document for r in response.results} == {0: "echoed", 1: None}


async def test_cohere_rerank_non_object_document_echo_raises() -> None:
    # §6 (0097): a non-object scalar echo is wire corruption -> provider_invalid_response.
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_rerank_body(results=[{"index": 0, "relevance_score": 0.9, "document": 123}]),
        )

    provider = _rerank_provider(handler)
    with pytest.raises(ProviderInvalidResponse):
        await provider.rerank("q", ["a"], config=RerankRuntimeConfig(return_documents=True))
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


async def test_rerank_base_url_rejects_v2_suffix() -> None:
    with pytest.raises(ValueError, match="host root"):
        CohereRerankProvider(base_url="https://api.cohere.com/v2", model="m")
    # The host root is accepted (no doubled /v2/v2). The /v2-suffix rejection
    # above raises in normalize_base_url before a client is created, so only the
    # accepted construction here needs closing.
    provider = CohereRerankProvider(base_url="https://api.cohere.com", model="m")
    await provider.aclose()


async def test_rerank_base_url_defaults_to_cohere_origin() -> None:
    # base_url is optional per §8.4: an unspecified base_url binds the Cohere
    # origin (host root; the provider appends /v2/rerank itself).
    provider = CohereRerankProvider(model="rerank-test")
    try:
        assert provider.base_url == "https://api.cohere.com"
    finally:
        await provider.aclose()


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
# Cohere embed wire mapping (proposal 0091, §8.4 /v2/embed)
# ---------------------------------------------------------------------------


def _cohere_embed_provider(
    handler: Any, *, model: str = "cohere-embed-test", **kwargs: Any
) -> CohereEmbeddingProvider:
    return CohereEmbeddingProvider(
        base_url="https://api.cohere.invalid",
        model=model,
        api_key="test-cohere-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _cohere_embed_body(
    *,
    id: str = "cohere-embed-id",
    vectors: list[list[float]],
    input_tokens: int | None = 6,
) -> dict[str, Any]:
    body: dict[str, Any] = {"id": id, "embeddings": {"float": vectors}}
    if input_tokens is not None:
        body["meta"] = {"billed_units": {"input_tokens": input_tokens}}
    return body


async def test_cohere_embed_wire_body_and_input_order() -> None:
    captured: list[dict[str, Any]] = []
    auth: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        auth.append(req.headers.get("Authorization"))
        assert req.url.path == "/v2/embed"
        return httpx.Response(
            200,
            json=_cohere_embed_body(
                id="cohere-embed-032",
                vectors=[[0.01, 0.02, 0.03, 0.04], [0.11, 0.12, 0.13, 0.14]],
                input_tokens=6,
            ),
        )

    provider = _cohere_embed_provider(handler)
    response = await provider.embed(["alpha", "beta"])
    await provider.aclose()

    # The wire body is exactly {model, input_type, texts, embedding_types,
    # truncate} -- input_type is the mandatory-field default (search_document,
    # no input_type supplied), no output_dimension (no dimensions supplied).
    assert captured[0] == {
        "model": "cohere-embed-test",
        "input_type": "search_document",
        "texts": ["alpha", "beta"],
        "embedding_types": ["float"],
        "truncate": "NONE",
    }
    assert auth[0] == "Bearer test-cohere-key"
    # embeddings.float assembled in input order (positional, not index-keyed).
    assert response.vectors == [[0.01, 0.02, 0.03, 0.04], [0.11, 0.12, 0.13, 0.14]]
    assert response.dimensions == 4
    assert response.usage is not None
    assert response.usage.input_tokens == 6
    assert response.response_id == "cohere-embed-032"
    # Cohere echoes no model field -> the bound id.
    assert response.model == "cohere-embed-test"
    # Single call -> raw is the verbatim object (not a 1-element list).
    assert response.raw == _cohere_embed_body(
        id="cohere-embed-032",
        vectors=[[0.01, 0.02, 0.03, 0.04], [0.11, 0.12, 0.13, 0.14]],
        input_tokens=6,
    )


async def test_cohere_embed_input_type_query_and_document_map_to_wire() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_cohere_embed_body(vectors=[[0.1, 0.2, 0.3, 0.4]], input_tokens=8))

    provider = _cohere_embed_provider(handler)
    events, token = _collecting_dispatch()
    try:
        await provider.embed(
            ["how tall is the eiffel tower?"], config=EmbeddingRuntimeConfig(input_type="query")
        )
        await provider.embed(
            ["the eiffel tower is 330 metres tall."], config=EmbeddingRuntimeConfig(input_type="document")
        )
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # The closed set maps query -> search_query, document -> search_document.
    assert captured[0]["input_type"] == "search_query"
    assert captured[1]["input_type"] == "search_document"
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    # request_params carries the caller's ORIGINAL value, not the wire mapping.
    assert embed_events[0].request_params == {"input_type": "query"}
    assert embed_events[1].request_params == {"input_type": "document"}


async def test_cohere_embed_input_type_absent_sends_search_document_default() -> None:
    # Cohere v2 requires input_type, so the absent case sends the search_document
    # default -- but request_params stays EMPTY (absence-is-meaningful: the wire
    # default MUST NOT leak back onto the event as if the caller set it).
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_cohere_embed_body(vectors=[[0.5, 0.6, 0.7, 0.8]], input_tokens=9))

    provider = _cohere_embed_provider(handler)
    events, token = _collecting_dispatch()
    try:
        await provider.embed(["the eiffel tower is 330 metres tall."])
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    assert captured[0]["input_type"] == "search_document"
    assert captured[0]["embedding_types"] == ["float"]
    assert captured[0]["truncate"] == "NONE"
    assert "output_dimension" not in captured[0]
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    assert embed_events[0].request_params == {}


async def test_cohere_embed_unrecognized_input_type_rejected_pre_send() -> None:
    # An OA input_type outside the closed query / document set is a pre-send
    # provider_invalid_request; no /v2/embed request is issued (no silent
    # coercion to the search_document default -- that default is absent-only).
    def handler(_req: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        raise AssertionError("provider must not be contacted on pre-send validation failure")

    provider = _cohere_embed_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.embed(["alpha"], config=EmbeddingRuntimeConfig(input_type="summarization"))
    await provider.aclose()


async def test_cohere_embed_output_dimension_from_dimensions() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json=_cohere_embed_body(
                vectors=[[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]], input_tokens=2
            ),
        )

    provider = _cohere_embed_provider(handler)
    events, token = _collecting_dispatch()
    try:
        response = await provider.embed(["alpha string"], config=EmbeddingRuntimeConfig(dimensions=8))
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # dimensions maps to Cohere's output_dimension wire field (Matryoshka), NOT
    # the extras bag; the response dimensions equals the inner-vector length.
    assert captured[0]["output_dimension"] == 8
    assert response.dimensions == 8
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    assert embed_events[0].request_params == {"dimensions": 8}


@pytest.mark.parametrize("status", [400, 422])
async def test_cohere_embed_over_length_maps_to_invalid_request(status: int) -> None:
    # §8.4 fail-loud: truncate "NONE" makes Cohere error on an over-length input
    # (400 / 422); the mapping surfaces provider_invalid_request rather than
    # returning a silently truncated vector. The wire still carried truncate NONE.
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(status, json={"message": "input exceeds the maximum token length"})

    provider = _cohere_embed_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.embed(["a very long document that exceeds the embedding model context window"])
    await provider.aclose()
    assert captured[0]["truncate"] == "NONE"


async def test_cohere_embed_no_usage_object_yields_null_usage() -> None:
    # Cohere reports usage, but a compatible backend that omits meta yields
    # usage = null (§4 -- never fabricate a record or a zero).
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_cohere_embed_body(vectors=[[0.1, 0.2]], input_tokens=None))

    provider = _cohere_embed_provider(handler)
    response = await provider.embed(["alpha"])
    await provider.aclose()
    assert response.usage is None


async def test_cohere_embed_chunk_and_stitch_over_96_cap() -> None:
    # 100 inputs, Cohere's 96-input cap => exactly 2 requests with texts sizes
    # [96, 4] over the consecutive slices input[0:96] / input[96:100], identical
    # per-call params on each. The per-chunk embeddings.float are stitched in
    # input order across the chunk boundary; input_tokens summed; response_id
    # from the FIRST chunk; raw the list of the 2 per-chunk objects (0096). One
    # EmbeddingEvent per embed() call (the whole stitched call, not per chunk).
    inputs = [f"doc-{i:03d}" for i in range(100)]
    chunk_a = [[i / 1000, 0.5] for i in range(96)]
    chunk_b = [[i / 1000, 0.5] for i in range(96, 100)]
    responses = iter(
        [
            _cohere_embed_body(id="chunk-a", vectors=chunk_a, input_tokens=480),
            _cohere_embed_body(id="chunk-b", vectors=chunk_b, input_tokens=20),
        ]
    )
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/v2/embed"
        return httpx.Response(200, json=next(responses))

    provider = _cohere_embed_provider(handler)
    events, token = _collecting_dispatch()
    try:
        response = await provider.embed(inputs)
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # Exactly 2 requests, texts sizes [96, 4], consecutive slices, identical
    # per-call params on each (no output_dimension on either).
    assert len(captured) == 2
    assert [len(b["texts"]) for b in captured] == [96, 4]
    assert captured[0]["texts"] == inputs[0:96]
    assert captured[1]["texts"] == inputs[96:100]
    for b in captured:
        assert b["model"] == "cohere-embed-test"
        assert b["input_type"] == "search_document"
        assert b["embedding_types"] == ["float"]
        assert b["truncate"] == "NONE"
        assert "output_dimension" not in b
    # Stitched in input order across the boundary; usage summed; first-chunk id.
    assert response.vectors == chunk_a + chunk_b
    assert len(response.vectors) == 100
    assert response.dimensions == 2
    assert response.usage is not None
    assert response.usage.input_tokens == 500
    assert response.response_id == "chunk-a"
    assert response.model == "cohere-embed-test"
    # raw is the list of the 2 per-chunk objects verbatim, in request order.
    assert response.raw == [
        _cohere_embed_body(id="chunk-a", vectors=chunk_a, input_tokens=480),
        _cohere_embed_body(id="chunk-b", vectors=chunk_b, input_tokens=20),
    ]
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    assert len(embed_events) == 1
    assert embed_events[0].input_count == 100
    assert embed_events[0].dimensions == 2


async def test_cohere_embed_embedding_types_extra_merges_float_not_clobbers() -> None:
    # §8.4: embedding_types MERGES rather than clobbers -- the mapping guarantees
    # "float" (it reads embeddings.float), but a caller's other precisions ride
    # the extras bag and are preserved. A caller list missing "float" gets it
    # appended; a malformed / absent extra falls back to ["float"].
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=_cohere_embed_body(id="c", vectors=[[0.1, 0.2]], input_tokens=3))

    cfg_both = EmbeddingRuntimeConfig.model_validate({"embedding_types": ["float", "int8"]})
    cfg_int8 = EmbeddingRuntimeConfig.model_validate({"embedding_types": ["int8"]})
    cfg_bad = EmbeddingRuntimeConfig.model_validate({"embedding_types": [{"x": 1}]})
    provider = _cohere_embed_provider(handler)
    # Caller adds int8 alongside float -> both reach the wire (float NOT clobbered).
    await provider.embed(["x"], config=cfg_both)
    # Caller omits float -> "float" is appended (the mapping must read embeddings.float).
    await provider.embed(["x"], config=cfg_int8)
    # No embedding_types extra -> the ["float"] default.
    await provider.embed(["x"])
    # Malformed (non-string elements) -> falls back to ["float"], not sent verbatim.
    await provider.embed(["x"], config=cfg_bad)
    await provider.aclose()

    assert captured[0]["embedding_types"] == ["float", "int8"]
    assert captured[1]["embedding_types"] == ["int8", "float"]
    assert captured[2]["embedding_types"] == ["float"]
    assert captured[3]["embedding_types"] == ["float"]


async def test_cohere_embed_per_chunk_count_mismatch_raises_despite_matching_total() -> None:
    # §4 / §8: the positional stitch validates EACH chunk's vector count, not
    # just the stitched total. 100 inputs -> chunks [96, 4]; chunk A returns 95
    # vectors (total could still reach 100 via a compensating chunk B), which
    # the total-count check alone would pass -- but the per-chunk guard raises on
    # chunk A's 95 != 96 mismatch, preventing a silent positional misalignment.
    def handler(req: httpx.Request) -> httpx.Response:
        n = len(json.loads(req.content)["texts"])
        count = 95 if n == 96 else 5
        return httpx.Response(
            200,
            json=_cohere_embed_body(id="c", vectors=[[float(i)] for i in range(count)], input_tokens=1),
        )

    provider = _cohere_embed_provider(handler)
    with pytest.raises(ProviderInvalidResponse):
        await provider.embed([f"input-{i}" for i in range(100)])
    await provider.aclose()


async def test_cohere_embed_base_url_rejects_v2_suffix() -> None:
    with pytest.raises(ValueError, match="host root"):
        CohereEmbeddingProvider(base_url="https://api.cohere.com/v2", model="m")
    # The host root is accepted (no doubled /v2/v2). The /v2-suffix rejection
    # above raises in normalize_base_url before a client is created, so only the
    # accepted construction here needs closing.
    provider = CohereEmbeddingProvider(base_url="https://api.cohere.com", model="m")
    await provider.aclose()


async def test_cohere_embed_base_url_defaults_to_cohere_origin() -> None:
    # base_url is optional per §8.4: an unspecified base_url binds the Cohere
    # origin (host root; the provider appends /v2/embed itself).
    provider = CohereEmbeddingProvider(model="cohere-embed-test")
    try:
        assert provider.base_url == "https://api.cohere.com"
    finally:
        await provider.aclose()


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


# --- /embed chunk-and-stitch (proposal 0092) --------------------------------


async def test_tei_embed_chunk_and_stitch_by_chunk_size() -> None:
    # Mirror fixture 038: 5 inputs, chunk_size 2 => exactly 3 /embed requests
    # with `inputs` sizes [2, 2, 1] over the consecutive slices input[0:2],
    # input[2:4], input[4:5], identical per-call params on each (bare {inputs}
    # form, no prompt_name / dimensions). The per-chunk bare vector arrays are
    # stitched IN INPUT ORDER across the chunk boundaries; TEI reports no usage
    # and no id, so usage + response_id are null. One EmbeddingEvent per embed()
    # call (the whole stitched call, not per chunk).
    inputs = [f"e{i}" for i in range(5)]
    chunks = [
        [[0.0, 0.5], [0.1, 0.5]],
        [[0.2, 0.5], [0.3, 0.5]],
        [[0.4, 0.5]],
    ]
    responses = iter(chunks)
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/embed"
        return httpx.Response(200, json=next(responses))

    provider = _tei_embed_provider(handler, chunk_size=2)
    events, token = _collecting_dispatch()
    try:
        response = await provider.embed(inputs)
    finally:
        await provider.aclose()
        _reset_active_dispatch(token)

    # Exactly 3 requests, `inputs` sizes [2, 2, 1], consecutive slices, identical
    # per-call params on each (bare {inputs} form; no prompt_name / dimensions).
    assert len(captured) == 3
    assert [len(b["inputs"]) for b in captured] == [2, 2, 1]
    assert captured[0]["inputs"] == inputs[0:2]
    assert captured[1]["inputs"] == inputs[2:4]
    assert captured[2]["inputs"] == inputs[4:5]
    for b in captured:
        assert set(b) == {"inputs"}
    # Stitched in input order across the boundaries (vector i == input i).
    assert response.vectors == [[0.0, 0.5], [0.1, 0.5], [0.2, 0.5], [0.3, 0.5], [0.4, 0.5]]
    assert len(response.vectors) == 5
    assert response.dimensions == 2
    # TEI /embed reports no usage object and no id -> both stitch to null.
    assert response.usage is None
    assert response.response_id is None
    assert response.model == "tei-embed-test"
    # raw is the list of the 3 per-chunk bare vector arrays, in request order.
    assert response.raw == chunks
    embed_events = [e for e in events if isinstance(e, EmbeddingEvent)]
    assert len(embed_events) == 1
    assert embed_events[0].input_count == 5
    assert embed_events[0].dimensions == 2


async def test_tei_embed_chunk_params_identical_across_chunks() -> None:
    # >chunk_size inputs with an input_type_prompt_map + input_type=query: every
    # /embed chunk carries the SAME prompt_name -- identical per-call params, only
    # `inputs` differs across chunks (a param that drifted across chunks fails).
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        captured.append(body)
        return httpx.Response(200, json=[[0.1, 0.2] for _ in body["inputs"]])

    provider = _tei_embed_provider(
        handler, chunk_size=2, input_type_prompt_map={"query": "query", "document": "passage"}
    )
    await provider.embed([f"q{i}" for i in range(5)], config=EmbeddingRuntimeConfig(input_type="query"))
    await provider.aclose()

    assert [len(b["inputs"]) for b in captured] == [2, 2, 1]
    # prompt_name identical on every chunk request; the wire key set is stable.
    assert all(b["prompt_name"] == "query" for b in captured)
    assert all(set(b) == {"inputs", "prompt_name"} for b in captured)


async def test_tei_embed_per_chunk_count_mismatch_raises() -> None:
    # The positional stitch validates EACH chunk's vector count. chunk_size 2
    # over 5 inputs => chunks [2, 2, 1]; the first chunk returns 1 vector for its
    # 2 inputs, which raises provider_invalid_response before a silent positional
    # misalignment (the shared helper's per-chunk guard).
    def handler(req: httpx.Request) -> httpx.Response:
        n = len(json.loads(req.content)["inputs"])
        count = 1 if n == 2 else n
        return httpx.Response(200, json=[[0.1, 0.2] for _ in range(count)])

    provider = _tei_embed_provider(handler, chunk_size=2)
    with pytest.raises(ProviderInvalidResponse):
        await provider.embed([f"e{i}" for i in range(5)])
    await provider.aclose()


async def test_tei_embed_single_request_path_when_within_chunk_size() -> None:
    # len(input) <= chunk_size issues exactly ONE /embed request (the pre-0092
    # single-request behavior preserved): 3 inputs, chunk_size 4 => 1 request,
    # and raw is the single bare array, not a one-element list.
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    provider = _tei_embed_provider(handler, chunk_size=4)
    response = await provider.embed(["a", "b", "c"])
    await provider.aclose()

    assert len(captured) == 1
    assert captured[0]["inputs"] == ["a", "b", "c"]
    assert response.vectors == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    # A single-request call's raw is the ONE bare array, not a one-element list.
    assert response.raw == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert response.usage is None
    assert response.response_id is None


async def test_chunk_and_stitch_embed_rejects_non_positive_cap() -> None:
    # The shared helper guards a caller passing cap <= 0 (a misconfigured per-call
    # limit): fail loudly rather than a raw range() error (0) or an empty stitch.
    from openarmature.retrieval._wire import chunk_and_stitch_embed

    async def _noop(chunk: list[str]) -> tuple[list[list[float]], None, None, None, list[Any]]:
        return [], None, None, None, []

    for bad_cap in (0, -1):
        with pytest.raises(ValueError, match="cap must be positive"):
            await chunk_and_stitch_embed(["a"], model="m", cap=bad_cap, embed_chunk=_noop)


async def test_chunk_and_stitch_embed_rejects_empty_input() -> None:
    # An empty input reaching the helper is a caller error -> provider_invalid_request
    # (validate_embedding_input), NOT a misclassified provider_invalid_response.
    from openarmature.retrieval._wire import chunk_and_stitch_embed

    async def _noop(chunk: list[str]) -> tuple[list[list[float]], None, None, None, list[Any]]:
        return [], None, None, None, []

    with pytest.raises(ProviderInvalidRequest):
        await chunk_and_stitch_embed([], model="m", cap=4, embed_chunk=_noop)


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


async def test_tei_rerank_non_object_text_echo_raises() -> None:
    # §6 (0097): TEI is on the shared document_echo rule -- its `text` echo is a
    # bare string in practice, but a non-object scalar (wire corruption) raises
    # provider_invalid_response rather than folding to null.
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"index": 0, "score": 0.9, "text": 123}])

    provider = _tei_rerank_provider(handler)
    with pytest.raises(ProviderInvalidResponse):
        await provider.rerank("q", ["a"], config=RerankRuntimeConfig(return_documents=True))
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


# --- Jina embed + rerank (§8.2 hosted wire mapping) --------------------------


def _jina_embed_provider(handler: Any, **kwargs: Any) -> JinaEmbeddingProvider:
    return JinaEmbeddingProvider(
        model="jina-embeddings-test",
        api_key="test-jina-key",
        base_url="https://api.jina.invalid",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _jina_rerank_provider(handler: Any, **kwargs: Any) -> JinaRerankProvider:
    return JinaRerankProvider(
        model="jina-reranker-test",
        api_key="test-jina-key",
        base_url="https://api.jina.invalid",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("input_type", "expected_task"),
    [("query", "retrieval.query"), ("document", "retrieval.passage"), (None, None)],
)
async def test_jina_embed_input_type_maps_to_task(input_type: str | None, expected_task: str | None) -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "model": "jina-embeddings-test",
                "usage": {"total_tokens": 8},
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}],
            },
        )

    provider = _jina_embed_provider(handler)
    config = EmbeddingRuntimeConfig(input_type=input_type) if input_type is not None else None
    response = await provider.embed(["how tall is the tower?"], config=config)
    await provider.aclose()

    body = captured[0]
    assert body["model"] == "jina-embeddings-test"
    assert body["input"] == ["how tall is the tower?"]
    # truncate: false is always sent (fail-loud).
    assert body["truncate"] is False
    if expected_task is None:
        # Absent input_type => no task (the symmetric default); the body is
        # exactly {model, input, truncate}.
        assert set(body) == {"model", "input", "truncate"}
    else:
        assert body["task"] == expected_task
    assert response.model == "jina-embeddings-test"
    assert response.dimensions == 4


async def test_jina_embed_unrecognized_input_type_raises_before_send() -> None:
    # An input_type outside the closed set (query / document) is a pre-send
    # provider_invalid_request; NO request is issued.
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={"model": "m", "data": []})

    provider = _jina_embed_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.embed(["x"], config=EmbeddingRuntimeConfig(input_type="clustering"))
    await provider.aclose()
    # No POST reached the transport (pre-send validation).
    assert captured == []


async def test_jina_embed_object_envelope_ordered_by_index_and_usage() -> None:
    # Jina returns an object envelope with data[] out of order; the mapping
    # orders vectors by data[i].index into input order, and maps
    # usage.total_tokens onto EmbeddingUsage.input_tokens (never fabricated).
    body_json = {
        "model": "jina-embeddings-test",
        "usage": {"total_tokens": 11},
        "data": [
            {"index": 1, "embedding": [0.5, 0.6]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ],
    }

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body_json)

    provider = _jina_embed_provider(handler)
    response = await provider.embed(["alpha", "beta"])
    await provider.aclose()

    # data[1] (index 0) is alpha; data[0] (index 1) is beta -- input order.
    assert response.vectors == [[0.1, 0.2], [0.5, 0.6]]
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    # raw is the verbatim response dict (object envelope, not a bare array).
    assert isinstance(response.raw, dict)
    assert response.raw == body_json


async def test_jina_embed_dimensions_on_wire_and_null_usage_when_absent() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        # No usage block -> usage stays null (never fabricated).
        return httpx.Response(
            200, json={"model": "m", "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]}]}
        )

    provider = _jina_embed_provider(handler)
    response = await provider.embed(["x"], config=EmbeddingRuntimeConfig(dimensions=4))
    await provider.aclose()

    assert captured[0]["dimensions"] == 4
    assert "task" not in captured[0]
    assert response.usage is None


@pytest.mark.parametrize("return_documents", [True, False])
async def test_jina_rerank_wire_body_return_documents_and_truncation(return_documents: bool) -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        assert req.url.path == "/v1/rerank"
        result: dict[str, Any] = {"index": 0, "relevance_score": 0.9}
        if return_documents:
            result["document"] = "doc a"
        return httpx.Response(
            200, json={"model": "jina-reranker-test", "usage": {"total_tokens": 5}, "results": [result]}
        )

    provider = _jina_rerank_provider(handler)
    config = RerankRuntimeConfig(return_documents=return_documents)
    response = await provider.rerank("q", ["doc a", "doc b"], config=config)
    await provider.aclose()

    body = captured[0]
    # return_documents is sent EXPLICITLY (Jina's wire default is true, OA's is
    # False); truncation: false is sent explicitly (fail-loud); top_n is omitted
    # when no top_k is supplied.
    assert body["return_documents"] is return_documents
    assert body["truncation"] is False
    assert "top_n" not in body
    assert body["model"] == "jina-reranker-test"
    assert body["documents"] == ["doc a", "doc b"]
    # The echo surfaces verbatim when Jina returns it; null when it does not
    # (never auto-filled from the input documents list).
    assert response.results[0].document == ("doc a" if return_documents else None)


async def test_jina_rerank_document_echo_unwraps_textdoc_object() -> None:
    # Jina's real /v1/rerank echoes the document as a TextDoc OBJECT
    # ({"text": ...}) when return_documents=true, not a bare string -- its
    # OpenAPI types `document` as anyOf[string, TextDoc, ImageDoc, null]. The
    # mapping unwraps either shape onto ScoredDocument.document (§6); a text-less
    # object (an ImageDoc) or an absent/null echo yields null, an empty string is
    # present (surfaced as ""), and the verbatim object is preserved on
    # RerankResponse.raw.
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jina-reranker-test",
                "usage": {"total_tokens": 9},
                "results": [
                    {"index": 0, "relevance_score": 0.9, "document": {"text": "doc a"}},
                    {"index": 1, "relevance_score": 0.5, "document": "doc b"},
                    {"index": 2, "relevance_score": 0.4, "document": {"image": "..."}},
                    {"index": 3, "relevance_score": 0.3, "document": ""},
                    {"index": 4, "relevance_score": 0.1, "document": None},
                ],
            },
        )

    provider = _jina_rerank_provider(handler)
    config = RerankRuntimeConfig(return_documents=True)
    response = await provider.rerank("q", ["a", "b", "c", "d", "e"], config=config)
    await provider.aclose()

    # TextDoc object -> its text; bare string -> itself; text-less object -> null;
    # empty string -> "" (present, distinct from absent); null -> null.
    echoes = {r.index: r.document for r in response.results}
    assert echoes == {0: "doc a", 1: "doc b", 2: None, 3: "", 4: None}
    # The verbatim TextDoc object is still reachable on raw.
    assert isinstance(response.raw, dict)
    assert response.raw["results"][0]["document"] == {"text": "doc a"}


async def test_jina_rerank_non_object_document_echo_raises() -> None:
    # 0097: a non-object scalar echo (number / array / bool -- outside Jina's
    # documented anyOf[string, TextDoc, ImageDoc, null]) is wire corruption and
    # raises provider_invalid_response, NOT folded to null.
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jina-reranker-test",
                "usage": {"total_tokens": 4},
                "results": [{"index": 0, "relevance_score": 0.9, "document": 123}],
            },
        )

    provider = _jina_rerank_provider(handler)
    with pytest.raises(ProviderInvalidResponse):
        await provider.rerank("q", ["doc a"], config=RerankRuntimeConfig(return_documents=True))
    await provider.aclose()


async def test_jina_rerank_relevance_score_parse_sorted_and_usage() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        # UNSORTED, Cohere-shaped relevance_score; Jina reports total_tokens.
        return httpx.Response(
            200,
            json={
                "model": "jina-reranker-test",
                "usage": {"total_tokens": 57},
                "results": [
                    {"index": 0, "relevance_score": 0.42},
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 2, "relevance_score": 0.08},
                ],
            },
        )

    provider = _jina_rerank_provider(handler)
    response = await provider.rerank("q", ["berlin", "paris", "madrid"])
    await provider.aclose()

    assert [(r.index, r.relevance_score) for r in response.results] == [(1, 0.91), (0, 0.42), (2, 0.08)]
    # usage.total_tokens -> RerankUsage.input_tokens; search_units always null
    # (Jina meters rerank by tokens, not search units).
    assert response.usage is not None
    assert response.usage.input_tokens == 57
    assert response.usage.search_units is None
    assert response.response_id is None
    # raw is the verbatim response dict.
    assert isinstance(response.raw, dict)
    assert response.raw["usage"] == {"total_tokens": 57}


async def test_jina_rerank_top_n_maps_from_top_k() -> None:
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "model": "m",
                "usage": {"total_tokens": 3},
                "results": [{"index": 0, "relevance_score": 0.9}],
            },
        )

    provider = _jina_rerank_provider(handler)
    await provider.rerank("q", ["a", "b", "c"], top_k=2)
    await provider.aclose()
    assert captured[0]["top_n"] == 2


@pytest.mark.parametrize("surface", ["embed", "rerank"])
async def test_jina_429_maps_to_rate_limit(surface: str) -> None:
    # §8.2 *Errors*: 429 -> provider_rate_limit on BOTH surfaces (NOT
    # provider_unavailable, the misclassification the fixture pins).
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"detail": "Rate limit exceeded.", "code": "RATE_TOKEN_LIMIT_EXCEEDED"}
        )

    with pytest.raises(ProviderRateLimit):
        if surface == "embed":
            provider = _jina_embed_provider(handler)
            try:
                await provider.embed(["x"])
            finally:
                await provider.aclose()
        else:
            rprovider = _jina_rerank_provider(handler)
            try:
                await rprovider.rerank("q", ["a", "b"])
            finally:
                await rprovider.aclose()


async def test_jina_rerank_over_length_422_maps_to_invalid_request() -> None:
    # §8.2 fail-loud: truncation: false makes Jina error on over-length input
    # (422); the mapping surfaces provider_invalid_request.
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            422,
            json={
                "detail": "Input validation failed.",
                "errors": [{"field": "documents", "type": "too_long"}],
            },
        )

    provider = _jina_rerank_provider(handler)
    with pytest.raises(ProviderInvalidRequest):
        await provider.rerank("summarize", ["a very long document"])
    await provider.aclose()
    # The wire request still carried truncation: false.
    assert captured[0]["truncation"] is False


async def test_jina_bearer_auth_header_present() -> None:
    # Jina is a hosted vendor; the api_key is sent as Authorization: Bearer.
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={
                "model": "m",
                "usage": {"total_tokens": 1},
                "data": [{"index": 0, "embedding": [0.1, 0.2]}],
            },
        )

    provider = _jina_embed_provider(handler)
    await provider.embed(["x"])
    await provider.aclose()
    assert captured[0].headers.get("Authorization") == "Bearer test-jina-key"


def test_jina_base_url_default_and_v1_guard() -> None:
    # base_url defaults to the Jina endpoint origin and strips trailing slashes;
    # a trailing /v1 is rejected (the doubled /v1/v1 footgun).
    embed = JinaEmbeddingProvider(model="m", api_key="k")
    assert embed.base_url == "https://api.jina.ai"
    rerank = JinaRerankProvider(model="m", api_key="k", base_url="https://gw.example/")
    assert rerank.base_url == "https://gw.example"
    for cls in (JinaEmbeddingProvider, JinaRerankProvider):
        with pytest.raises(ValueError, match="v1/v1"):
            cls(model="m", api_key="k", base_url="https://api.jina.ai/v1")
