"""Focused tests for the llm-provider module.

The conformance suite (``tests/conformance/test_llm_provider.py``)
covers the spec's behavioral surface end-to-end against the
fixtures. These unit tests fill gaps the conformance suite doesn't
exercise directly: per-role construction errors that fixtures only
hit through the boundary, the tool-call-id verbatim guarantee,
the canonical category-string contract, and the
``classify_http_error`` mapping table.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from openarmature.llm import (
    PROVIDER_AUTHENTICATION,
    PROVIDER_INVALID_MODEL,
    PROVIDER_INVALID_REQUEST,
    PROVIDER_INVALID_RESPONSE,
    PROVIDER_MODEL_NOT_LOADED,
    PROVIDER_RATE_LIMIT,
    PROVIDER_UNAVAILABLE,
    TRANSIENT_CATEGORIES,
    AssistantMessage,
    LlmProviderError,
    OpenAIProvider,
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderModelNotLoaded,
    ProviderRateLimit,
    ProviderUnavailable,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
    classify_http_error,
    validate_message_list,
    validate_tools,
)

# ---------------------------------------------------------------------------
# Per-role message construction (Pydantic-on-Message layer)
# ---------------------------------------------------------------------------


def test_system_message_empty_content_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemMessage(content="")


def test_user_message_empty_content_rejected() -> None:
    with pytest.raises(ValidationError):
        UserMessage(content="")


def test_assistant_empty_content_without_tool_calls_rejected() -> None:
    with pytest.raises(ValidationError):
        AssistantMessage(content="")


def test_assistant_empty_content_with_tool_calls_ok() -> None:
    # Content MAY be empty when tool_calls is non-empty per spec §3.
    msg = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})],
    )
    assert msg.content == ""
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].id == "call_1"


def test_tool_message_construction_does_not_check_id_match() -> None:
    # Per spec §3 "Validation timing": the per-message layer does NOT
    # check whether tool_call_id corresponds to an earlier assistant
    # ToolCall.id — that's the boundary check's job. So a ToolMessage
    # with a fabricated id is constructible on its own.
    msg = ToolMessage(content="result", tool_call_id="never-issued")
    assert msg.tool_call_id == "never-issued"


# ---------------------------------------------------------------------------
# Tool-call id verbatim preservation (spec §3 MUST NOT rewrite)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_id",
    [
        "call_abc123",
        "call_3f9c2c12-3a44-4f6a-9d2e-fbb12c0e2d77",  # uuid w/ hyphens
        "bifrost_abc-def",  # vendor-prefixed
        "TOOL/CALL/01",  # path-shaped
        "id with spaces",  # whitespace tolerated
        "",  # empty string — opaque correlator, no normalization
    ],
)
def test_tool_call_id_preserved_verbatim(raw_id: str) -> None:
    tc = ToolCall(id=raw_id, name="f", arguments={})
    assert tc.id == raw_id
    # And through Pydantic dump/load round trip — no normalizer should
    # change the id.
    dumped = tc.model_dump()
    assert dumped["id"] == raw_id
    rebuilt = ToolCall.model_validate(dumped)
    assert rebuilt.id == raw_id


# ---------------------------------------------------------------------------
# List-level boundary validation (validate_message_list)
# ---------------------------------------------------------------------------


def test_validate_empty_list_rejected() -> None:
    with pytest.raises(ProviderInvalidRequest, match="non-empty"):
        validate_message_list([])


def test_validate_first_message_must_be_system_or_user() -> None:
    with pytest.raises(ProviderInvalidRequest, match="first message MUST"):
        validate_message_list([AssistantMessage(content="hi")])


def test_validate_system_message_only_at_first_position() -> None:
    msgs = [
        UserMessage(content="hi"),
        SystemMessage(content="surprise"),
    ]
    with pytest.raises(ProviderInvalidRequest, match="system message MUST be the first"):
        validate_message_list(msgs)


def test_validate_last_message_must_be_user_or_tool() -> None:
    msgs = [
        UserMessage(content="hi"),
        AssistantMessage(content="hello"),
    ]
    with pytest.raises(ProviderInvalidRequest, match="last message"):
        validate_message_list(msgs)


def test_validate_tool_message_id_must_match_earlier_assistant() -> None:
    msgs = [
        UserMessage(content="run echo"),
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="call_real", name="echo", arguments={})],
        ),
        ToolMessage(content="ok", tool_call_id="call_FAKE"),
    ]
    with pytest.raises(ProviderInvalidRequest, match="does not match any earlier"):
        validate_message_list(msgs)


def test_validate_tool_message_id_matches_real_call_passes() -> None:
    msgs = [
        UserMessage(content="run echo"),
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="call_real", name="echo", arguments={})],
        ),
        ToolMessage(content="ok", tool_call_id="call_real"),
    ]
    validate_message_list(msgs)  # no raise


def test_validate_minimal_user_only_passes() -> None:
    validate_message_list([UserMessage(content="hi")])


def test_validate_system_then_user_passes() -> None:
    validate_message_list([SystemMessage(content="be helpful"), UserMessage(content="hi")])


# ---------------------------------------------------------------------------
# validate_tools — duplicate name rejection
# ---------------------------------------------------------------------------


def test_validate_tools_none_or_empty() -> None:
    validate_tools(None)
    validate_tools([])


def test_validate_tools_unique_names_pass() -> None:
    validate_tools(
        [
            Tool(name="a", description="", parameters={}),
            Tool(name="b", description="", parameters={}),
        ]
    )


def test_validate_tools_duplicate_names_rejected() -> None:
    with pytest.raises(ProviderInvalidRequest, match="duplicate tool name"):
        validate_tools(
            [
                Tool(name="echo", description="", parameters={}),
                Tool(name="echo", description="", parameters={}),
            ]
        )


# ---------------------------------------------------------------------------
# OpenAIProvider base_url validation
# ---------------------------------------------------------------------------


def test_openai_provider_rejects_v1_suffix() -> None:
    with pytest.raises(ValueError, match=r"base_url must not end with '/v1'"):
        OpenAIProvider(base_url="http://localhost:8090/v1", model="m", api_key="k")


def test_openai_provider_rejects_v1_suffix_with_trailing_slash() -> None:
    with pytest.raises(ValueError, match=r"base_url must not end with '/v1'"):
        OpenAIProvider(base_url="http://localhost:8090/v1/", model="m", api_key="k")


def test_openai_provider_rejects_openai_cloud_with_v1() -> None:
    # The motivating real-world case: api.openai.com/v1 is in the
    # OpenAI docs as the API endpoint, but for OpenAIProvider's
    # base_url the /v1 must be omitted.
    with pytest.raises(ValueError, match=r"base_url must not end with '/v1'"):
        OpenAIProvider(base_url="https://api.openai.com/v1", model="gpt-4", api_key="k")


def test_openai_provider_accepts_host_root() -> None:
    provider = OpenAIProvider(base_url="https://api.openai.com", model="gpt-4", api_key="k")
    assert provider.base_url == "https://api.openai.com"


def test_openai_provider_accepts_host_root_with_trailing_slash() -> None:
    provider = OpenAIProvider(base_url="http://localhost:8090/", model="m", api_key="k")
    assert provider.base_url == "http://localhost:8090"


def test_openai_provider_accepts_non_v1_path() -> None:
    # Proxy prefixes (Cloudflare AI Gateway, internal reverse proxies)
    # are intentional and left alone.
    provider = OpenAIProvider(
        base_url="https://gateway.example.com/openai-proxy",
        model="m",
        api_key="k",
    )
    assert provider.base_url == "https://gateway.example.com/openai-proxy"


def test_openai_provider_accepts_v1_in_middle_of_path() -> None:
    # Only a trailing /v1 is rejected — proxies that include /v1
    # somewhere mid-path are intentional.
    provider = OpenAIProvider(
        base_url="https://gateway.example.com/v1/openai-proxy",
        model="m",
        api_key="k",
    )
    assert provider.base_url == "https://gateway.example.com/v1/openai-proxy"


def test_openai_provider_rejects_v1_with_query_string() -> None:
    # The trailing slash on the path is followed by a query string,
    # so a URL-level rstrip("/") doesn't normalize it. The parsed
    # path's own trailing slash MUST be stripped before the suffix
    # check or this case slips through.
    with pytest.raises(ValueError, match=r"base_url must not end with '/v1'"):
        OpenAIProvider(base_url="https://host/v1/?token=abc", model="m", api_key="k")


def test_openai_provider_rejects_v1_with_fragment() -> None:
    # Same shape as the query-string case but with a URL fragment.
    with pytest.raises(ValueError, match=r"base_url must not end with '/v1'"):
        OpenAIProvider(base_url="https://host/v1/#frag", model="m", api_key="k")


# ---------------------------------------------------------------------------
# Error categories — canonical string contract + __cause__ preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "expected_category"),
    [
        (ProviderAuthentication, PROVIDER_AUTHENTICATION),
        (ProviderUnavailable, PROVIDER_UNAVAILABLE),
        (ProviderInvalidModel, PROVIDER_INVALID_MODEL),
        (ProviderModelNotLoaded, PROVIDER_MODEL_NOT_LOADED),
        (ProviderRateLimit, PROVIDER_RATE_LIMIT),
        (ProviderInvalidResponse, PROVIDER_INVALID_RESPONSE),
        (ProviderInvalidRequest, PROVIDER_INVALID_REQUEST),
    ],
)
def test_error_category_matches_canonical_string(cls: type[LlmProviderError], expected_category: str) -> None:
    err = cls("boom")
    assert err.category == expected_category


def test_provider_rate_limit_retry_after_accessor() -> None:
    err = ProviderRateLimit("429", retry_after=30.0)
    assert err.retry_after == 30.0
    bare = ProviderRateLimit("429")
    assert bare.retry_after is None


def test_error_cause_preserved() -> None:
    underlying = RuntimeError("wire failure")
    try:
        try:
            raise underlying
        except RuntimeError as exc:
            raise ProviderUnavailable("unreachable") from exc
    except ProviderUnavailable as out:
        assert out.__cause__ is underlying


# ---------------------------------------------------------------------------
# TRANSIENT_CATEGORIES — must match exactly the spec's transient set
# ---------------------------------------------------------------------------


def test_transient_categories_exact_set() -> None:
    assert TRANSIENT_CATEGORIES == frozenset(
        {
            PROVIDER_RATE_LIMIT,
            PROVIDER_UNAVAILABLE,
            PROVIDER_MODEL_NOT_LOADED,
        }
    )


def test_transient_categories_excludes_terminal_categories() -> None:
    for terminal in (
        PROVIDER_AUTHENTICATION,
        PROVIDER_INVALID_MODEL,
        PROVIDER_INVALID_REQUEST,
        PROVIDER_INVALID_RESPONSE,
    ):
        assert terminal not in TRANSIENT_CATEGORIES


# ---------------------------------------------------------------------------
# classify_http_error — wire-mapping table (spec §8.1.3)
# ---------------------------------------------------------------------------


def _wire_response(
    status: int, body: dict[str, object] | None = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    """Build an httpx.Response with the given status/body/headers."""
    return httpx.Response(
        status,
        json=body if body is not None else {},
        headers=headers or {},
    )


def test_classify_401_to_authentication() -> None:
    err = classify_http_error(_wire_response(401))
    assert isinstance(err, ProviderAuthentication)


def test_classify_403_to_authentication() -> None:
    err = classify_http_error(_wire_response(403))
    assert isinstance(err, ProviderAuthentication)


def test_classify_400_to_invalid_request() -> None:
    err = classify_http_error(_wire_response(400))
    assert isinstance(err, ProviderInvalidRequest)


def test_classify_404_with_model_not_found_to_invalid_model() -> None:
    err = classify_http_error(
        _wire_response(
            404,
            {"error": {"code": "model_not_found", "message": "no such model"}},
        )
    )
    assert isinstance(err, ProviderInvalidModel)


def test_classify_404_without_model_marker_to_unavailable() -> None:
    err = classify_http_error(_wire_response(404))
    assert isinstance(err, ProviderUnavailable)


def test_classify_429_with_retry_after_to_rate_limit() -> None:
    err = classify_http_error(_wire_response(429, {"error": {"message": "slow down"}}, {"Retry-After": "30"}))
    assert isinstance(err, ProviderRateLimit)
    assert err.retry_after == 30.0


def test_classify_429_without_retry_after_to_rate_limit_no_value() -> None:
    err = classify_http_error(_wire_response(429))
    assert isinstance(err, ProviderRateLimit)
    assert err.retry_after is None


def test_classify_503_with_model_not_loaded_to_model_not_loaded() -> None:
    err = classify_http_error(
        _wire_response(
            503,
            {"error": {"type": "model_not_loaded", "message": "loading"}},
        )
    )
    assert isinstance(err, ProviderModelNotLoaded)


def test_classify_503_without_marker_to_unavailable() -> None:
    err = classify_http_error(_wire_response(503))
    assert isinstance(err, ProviderUnavailable)


def test_classify_500_to_unavailable() -> None:
    err = classify_http_error(_wire_response(500))
    assert isinstance(err, ProviderUnavailable)


def test_classify_502_to_unavailable() -> None:
    err = classify_http_error(_wire_response(502))
    assert isinstance(err, ProviderUnavailable)


def test_classify_504_to_unavailable() -> None:
    err = classify_http_error(_wire_response(504))
    assert isinstance(err, ProviderUnavailable)


# ---------------------------------------------------------------------------
# complete() input non-mutation (spec §5: "messages MUST NOT be mutated")
# ---------------------------------------------------------------------------


async def test_complete_does_not_mutate_messages_or_tools() -> None:
    """The input messages/tools list MUST NOT be mutated by complete().
    Snapshot the inputs (deep-copy via Pydantic round-trip), run a
    happy-path call, and assert the input objects remain equal to
    their pre-call snapshots after the call returns.
    """

    def _ok(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_ok),
    )
    try:
        messages = [
            SystemMessage(content="be helpful"),
            UserMessage(content="hello"),
        ]
        tools = [Tool(name="echo", description="", parameters={})]
        before_messages = [m.model_copy(deep=True) for m in messages]
        before_tools = [t.model_copy(deep=True) for t in tools]
        await provider.complete(messages, tools)
        # Same instances; same field values; lists not reordered or resized.
        assert len(messages) == len(before_messages)
        assert len(tools) == len(before_tools)
        for actual, expected in zip(messages, before_messages, strict=True):
            assert actual == expected
        for actual_t, expected_t in zip(tools, before_tools, strict=True):
            assert actual_t == expected_t
    finally:
        await provider.aclose()


# ---------------------------------------------------------------------------
# Usage non-negative constraint (spec §6: "non-negative integer or None")
# ---------------------------------------------------------------------------


def test_usage_negative_token_count_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=-1, completion_tokens=0, total_tokens=0)


async def test_complete_negative_usage_surfaces_as_invalid_response() -> None:
    """A wire response carrying a negative token count MUST surface as
    ``provider_invalid_response`` rather than silently passing through —
    spec §6 token counts are non-negative integers."""

    def _bad(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": -5, "completion_tokens": 1, "total_tokens": 1},
            },
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_bad),
    )
    try:
        with pytest.raises(ProviderInvalidResponse, match="invalid usage record"):
            await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()


# RuntimeConfig.from_partial — Python ergonomic introduced alongside
# proposal 0032. Wire-layer null-skip already drops Nones; this just
# lets callers splat a partial dict without filtering at the call site.


def test_runtime_config_from_partial_drops_nones() -> None:
    from openarmature.llm import RuntimeConfig

    config = RuntimeConfig.from_partial(temperature=0.7, max_tokens=None, top_p=0.9, seed=None)

    assert config.temperature == 0.7
    assert config.top_p == 0.9
    # max_tokens and seed default to None on a base RuntimeConfig, so
    # `is None` alone doesn't prove the drop. model_fields_set carries
    # only explicitly-set fields, so its absence proves from_partial
    # filtered the None kwargs before __init__ ran.
    assert "max_tokens" not in config.model_fields_set
    assert "seed" not in config.model_fields_set
    assert config.max_tokens is None
    assert config.seed is None


def test_runtime_config_from_partial_forwards_extras() -> None:
    from openarmature.llm import RuntimeConfig

    config = RuntimeConfig.from_partial(temperature=0.5, repetition_penalty=1.05, top_k=None)

    assert config.temperature == 0.5
    assert (config.model_extra or {}) == {"repetition_penalty": 1.05}


def test_runtime_config_from_partial_empty() -> None:
    from openarmature.llm import RuntimeConfig

    config = RuntimeConfig.from_partial()

    assert config.temperature is None
    assert config.frequency_penalty is None
    assert config.stop_sequences is None


# ---------------------------------------------------------------------------
# ready() readiness_probe modes
# ---------------------------------------------------------------------------
# Verifies the v0.12.0 default-flip: chat_completions is the strict probe
# (actually exercises inference), models is the opt-in catalog-only probe,
# both runs catalog then chat. Conformance fixture 007 owns the catalog-only
# semantics — these tests cover the new wire paths and the dispatch.


async def test_ready_chat_completions_200_passes() -> None:
    def _ok(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "."},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_ok),
    )
    try:
        await provider.ready()
    finally:
        await provider.aclose()


async def test_ready_chat_completions_405_surfaces_unavailable() -> None:
    # The Bifrost-style proxy case: the catalog endpoint may answer 200
    # (which the older default probe accepted), but POST /v1/chat/completions
    # returns 405 because the proxy doesn't actually serve completions.
    # classify_http_error routes 405 through ProviderUnavailable.
    def _405(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/completions"
        return httpx.Response(405, json={"error": {"message": "method not allowed"}})

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_405),
    )
    try:
        with pytest.raises(ProviderUnavailable):
            await provider.ready()
    finally:
        await provider.aclose()


async def test_ready_chat_completions_401_surfaces_authentication() -> None:
    def _401(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_401),
    )
    try:
        with pytest.raises(ProviderAuthentication):
            await provider.ready()
    finally:
        await provider.aclose()


async def test_ready_chat_completions_404_model_not_found_surfaces_invalid_model() -> None:
    def _404(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"code": "model_not_found", "message": "no such model 'm'"}},
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_404),
    )
    try:
        with pytest.raises(ProviderInvalidModel):
            await provider.ready()
    finally:
        await provider.aclose()


async def test_ready_both_runs_catalog_then_chat() -> None:
    # The both-mode contract: catalog probe first (so catalog-only failures
    # short-circuit before the billable chat call), then chat probe. This
    # test verifies both endpoints get hit in order on the happy path.
    seen: list[str] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "m", "object": "model"}]},
            )
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "."},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_handler),
        readiness_probe="both",
    )
    try:
        await provider.ready()
    finally:
        await provider.aclose()
    assert seen == ["/v1/models", "/v1/chat/completions"]


async def test_ready_both_catalog_missing_model_short_circuits() -> None:
    # When catalog returns 200 but the bound model isn't in the list, the
    # catalog probe MUST raise ProviderInvalidModel before any chat call.
    # The single ``seen``-tracking handler answers any path with the catalog
    # JSON, so a leaked chat call would silently 200; the real proof of
    # short-circuit is the seen == ["/v1/models"] assertion at the bottom.
    seen: list[str] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "other-model", "object": "model"}],
            },
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_handler),
        readiness_probe="both",
    )
    try:
        with pytest.raises(ProviderInvalidModel):
            await provider.ready()
    finally:
        await provider.aclose()
    assert seen == ["/v1/models"]


async def test_ready_models_mode_still_hits_catalog() -> None:
    # Explicit opt-in to the old default. Verifies the dispatch routes
    # models-mode through the catalog endpoint and ignores the chat path.
    seen: list[str] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "m", "object": "model"}]},
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_handler),
        readiness_probe="models",
    )
    try:
        await provider.ready()
    finally:
        await provider.aclose()
    assert seen == ["/v1/models"]


async def test_ready_both_catalog_200_chat_405_surfaces_unavailable() -> None:
    # The actual Bifrost case via ``both`` mode: catalog probe sees 200 from
    # ``/v1/models`` with the bound model present, then the chat probe gets
    # 405. classify_http_error routes 405 to ProviderUnavailable. This is
    # the failure shape ``both`` mode is meant to surface that the catalog-
    # only probe missed.
    seen: list[str] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "m", "object": "model"}]},
            )
        return httpx.Response(405, json={"error": {"message": "method not allowed"}})

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_handler),
        readiness_probe="both",
    )
    try:
        with pytest.raises(ProviderUnavailable):
            await provider.ready()
    finally:
        await provider.aclose()
    assert seen == ["/v1/models", "/v1/chat/completions"]


async def test_ready_chat_completions_network_error_surfaces_unavailable() -> None:
    # httpx network-layer failures (ConnectError, ReadTimeout, etc.) on the
    # chat probe wrap into ProviderUnavailable, same as on the catalog
    # probe. Fixture 007's network_failure case covers the catalog side;
    # this covers the chat-probe side that the catalog fixture can't reach
    # under the new default.
    def _raises(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_raises),
    )
    try:
        with pytest.raises(ProviderUnavailable, match="connection refused"):
            await provider.ready()
    finally:
        await provider.aclose()


async def test_ready_chat_completions_503_model_not_loaded() -> None:
    # 503 with a model-not-loaded body routes through classify_http_error
    # to ProviderModelNotLoaded. Covered indirectly by the classifier's
    # own tests, but a single test here pins the dispatch from the chat
    # probe specifically.
    def _503(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"error": {"type": "model_not_loaded", "message": "model is not loaded yet"}},
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_503),
    )
    try:
        with pytest.raises(ProviderModelNotLoaded):
            await provider.ready()
    finally:
        await provider.aclose()
