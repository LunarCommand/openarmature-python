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

import json
from collections.abc import Callable
from contextvars import Token
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from openarmature.graph.events import LlmCompletionEvent, LlmFailedEvent, LlmRetryAttemptEvent, NodeEvent
from openarmature.graph.middleware import RetryConfig, deterministic_backoff
from openarmature.graph.observer import ObserverEvent
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
    ProviderUnsupportedContentBlock,
    StructuredOutputInvalid,
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
from openarmature.observability.correlation import (
    _set_active_dispatch,
    _set_attempt_index,
    _set_branch_name,
    _set_correlation_id,
    _set_fan_out_index,
    _set_invocation_id,
    _set_namespace_prefix,
)
from openarmature.observability.metadata import set_invocation_metadata

_DispatchToken = Token[Callable[[ObserverEvent], None] | None]

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
    token counts are non-negative integers."""

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


# ---------------------------------------------------------------------------
# Usage cache-stat fields (proposal 0047 — llm-provider §6 extension)
# ---------------------------------------------------------------------------


def test_usage_cache_fields_default_to_none() -> None:
    # Backwards-compat: existing Usage constructions that don't pass
    # cache fields produce instances with cached_tokens = None and
    # cache_creation_tokens = None (the "not reported" state, distinct
    # from a "reported zero" value of 0).
    usage = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert usage.cached_tokens is None
    assert usage.cache_creation_tokens is None


def test_usage_negative_cached_tokens_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cached_tokens=-1)


def test_usage_negative_cache_creation_tokens_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, cache_creation_tokens=-1)


def _make_openai_response_with_usage(usage_body: dict[str, object]) -> httpx.MockTransport:
    """Build a MockTransport returning a minimal Chat Completions
    response with the given ``usage`` body. Helper for the cache-stat
    end-to-end tests below.
    """

    def _handler(_req: httpx.Request) -> httpx.Response:
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
                "usage": usage_body,
            },
        )

    return httpx.MockTransport(_handler)


async def test_complete_sources_cached_tokens_from_nested_prompt_tokens_details() -> None:
    # Cache-hit reported with a positive value. Spec §8.1.2: the
    # OpenAI-compat mapping sources cached_tokens from
    # usage.prompt_tokens_details.cached_tokens.
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 75},
        }
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        response = await provider.complete([UserMessage(content="hi")])
        assert response.usage.cached_tokens == 75
        # cache_creation_tokens is not sourced by the OpenAI-compat
        # mapping per spec §8.1.2.
        assert response.usage.cache_creation_tokens is None
    finally:
        await provider.aclose()


async def test_complete_reports_zero_cached_tokens_distinct_from_absent() -> None:
    # The spec mandates the absent-vs-reported-zero distinction:
    # absent (None) means the provider didn't report; 0 means the
    # provider reported zero hits. Locks down the distinction.
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        response = await provider.complete([UserMessage(content="hi")])
        assert response.usage.cached_tokens == 0
    finally:
        await provider.aclose()


async def test_complete_cached_tokens_absent_when_prompt_tokens_details_missing() -> None:
    # Common pre-cache path: vLLM without --enable-prompt-tokens-details,
    # OpenAI responses pre-cache-support, etc. No prompt_tokens_details
    # nesting at all → cached_tokens stays None.
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        response = await provider.complete([UserMessage(content="hi")])
        assert response.usage.cached_tokens is None
    finally:
        await provider.aclose()


async def test_complete_cached_tokens_absent_when_nested_key_missing() -> None:
    # Defensive: prompt_tokens_details dict exists (provider may report
    # other details there, e.g., audio_tokens) but cached_tokens is
    # absent within it. Sourcing path stays defensive — no KeyError,
    # cached_tokens stays None.
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"audio_tokens": 0},
        }
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        response = await provider.complete([UserMessage(content="hi")])
        assert response.usage.cached_tokens is None
    finally:
        await provider.aclose()


async def test_complete_excludes_unset_cached_tokens_when_wire_did_not_report() -> None:
    # When the wire response doesn't carry prompt_tokens_details (or
    # carries it without a cached_tokens key), the parser leaves the
    # Pydantic field unset. model_dump(exclude_unset=True) then omits
    # cached_tokens entirely, giving downstream consumers a clean
    # wire-shape projection. Attribute access still returns None per
    # the spec's absent-vs-reported distinction.
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        response = await provider.complete([UserMessage(content="hi")])
        assert response.usage.cached_tokens is None
        dumped = response.usage.model_dump(exclude_unset=True)
        assert "cached_tokens" not in dumped
        # Conversely, when the wire DID report (covered separately
        # above), the field IS set and appears in the projection.
    finally:
        await provider.aclose()


async def test_complete_includes_cached_tokens_in_exclude_unset_dump_when_wire_reported() -> None:
    # Companion to the no-wire-report case above: when the wire reports
    # prompt_tokens_details.cached_tokens, the field IS marked set and
    # appears in model_dump(exclude_unset=True). Locks down the
    # bidirectional projection for downstream consumers.
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 75},
        }
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        response = await provider.complete([UserMessage(content="hi")])
        dumped = response.usage.model_dump(exclude_unset=True)
        assert dumped.get("cached_tokens") == 75
    finally:
        await provider.aclose()


async def test_complete_cached_tokens_absent_when_prompt_tokens_details_not_a_dict() -> None:
    # Defensive against malformed wire responses: if prompt_tokens_details
    # is a non-dict scalar / string / list, the isinstance guard in the
    # parser treats it as absent rather than crashing. cached_tokens
    # stays None.
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": "unexpected_shape",
        }
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        response = await provider.complete([UserMessage(content="hi")])
        assert response.usage.cached_tokens is None
    finally:
        await provider.aclose()


async def test_complete_negative_cached_tokens_surfaces_as_invalid_response() -> None:
    # Same invariant the existing test pins for prompt_tokens — a
    # wire response carrying a negative cache count MUST surface as
    # ``provider_invalid_response`` rather than silently passing through.
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": -1},
        }
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        with pytest.raises(ProviderInvalidResponse, match="invalid usage record"):
            await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()


async def test_complete_populates_cached_tokens_on_typed_event() -> None:
    # Proposal 0047: Response.usage.cached_tokens MUST flow onto the
    # typed LlmCompletionEvent's usage record so observers driving
    # §5.5.3.1 cache attribute emission have the cache stat available.
    # This locks the field at the provider-event boundary; the
    # conformance fixture 040 covers the end-to-end OTel span
    # attribute path.
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
            "prompt_tokens_details": {"cached_tokens": 42},
        }
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.usage is not None
    assert typed.usage.cached_tokens == 42
    # The OpenAI-compat mapping leaves cache_creation_tokens absent
    # per spec §8.1.2; verify the field stays None on the typed event.
    assert typed.usage.cache_creation_tokens is None


async def test_complete_leaves_cached_tokens_none_when_provider_silent() -> None:
    # Companion to the populated case: when the wire response omits
    # prompt_tokens_details, Response.usage.cached_tokens stays None
    # and the typed event's Usage record reflects that.
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.usage is not None
    assert typed.usage.cached_tokens is None
    assert typed.usage.cache_creation_tokens is None


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


def test_readiness_probe_unknown_mode_rejected_at_construction() -> None:
    # Literal type is a static hint, not a runtime guard. Unknown modes
    # would otherwise silently no-op both dispatch branches in ready()
    # and report ready, so reject at construction.
    with pytest.raises(ValueError, match="readiness_probe must be one of"):
        OpenAIProvider(
            base_url="http://test",
            model="m",
            api_key="k",
            readiness_probe="bogus",  # pyright: ignore[reportArgumentType]
        )


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


async def test_ready_models_429_surfaces_rate_limit() -> None:
    # The catalog probe routes non-200 through classify_http_error, so
    # 429 with a Retry-After lands as ProviderRateLimit carrying the
    # parsed delay. Pre-refactor _probe_models would have flattened this
    # to ProviderUnavailable.
    def _429(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"error": {"message": "rate limited"}},
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_429),
        readiness_probe="models",
    )
    try:
        with pytest.raises(ProviderRateLimit) as excinfo:
            await provider.ready()
    finally:
        await provider.aclose()
    assert excinfo.value.retry_after == 30.0


async def test_ready_models_503_model_not_loaded_surfaces_canonical_category() -> None:
    # 503 with a model-not-loaded marker now lands as
    # ProviderModelNotLoaded on the catalog probe too, not the previous
    # generic ProviderUnavailable.
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
        readiness_probe="models",
    )
    try:
        with pytest.raises(ProviderModelNotLoaded):
            await provider.ready()
    finally:
        await provider.aclose()


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


async def test_ready_chat_completions_200_with_error_payload_surfaces_invalid_response() -> None:
    # The residual false-green class: a proxy returning 200 with an
    # error payload (no ``choices`` field) would pass a simple status
    # check but indicates a deeply broken inference path. The chat probe
    # now parses the response shape so this fails with
    # ProviderInvalidResponse rather than reporting ready.
    def _200_error(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "something is wrong"})

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_200_error),
    )
    try:
        with pytest.raises(ProviderInvalidResponse):
            await provider.ready()
    finally:
        await provider.aclose()


async def test_ready_chat_completions_200_with_non_json_body_surfaces_invalid_response() -> None:
    # Same false-green class, JSON-parse leg: a proxy returning 200 with
    # a non-JSON body (HTML error page, plain text) must not pass.
    def _200_html(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>error</html>", headers={"content-type": "text/html"})

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_200_html),
    )
    try:
        with pytest.raises(ProviderInvalidResponse, match="non-JSON"):
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


# ---------------------------------------------------------------------------
# LlmCompletionEvent dual-emit (proposal 0049)
# ---------------------------------------------------------------------------


def _collecting_dispatch() -> tuple[list[ObserverEvent], _DispatchToken]:
    """Install a collecting dispatch callback into the
    ``current_dispatch`` ContextVar and return ``(events, token)``.
    The caller is responsible for resetting the token in a try/finally.
    """
    events: list[ObserverEvent] = []

    def _dispatch(event: ObserverEvent) -> None:
        events.append(event)

    token = _set_active_dispatch(_dispatch)
    return events, token


def _release_dispatch(token: _DispatchToken) -> None:
    from openarmature.observability.correlation import _reset_active_dispatch

    _reset_active_dispatch(token)


async def test_complete_success_emits_only_typed_event() -> None:
    # v0.13.0 dropped the success-side sentinel emission. The provider
    # now emits a single typed LlmCompletionEvent on success; no
    # sentinel NodeEvent pair (started or completed) fires for
    # successful calls. External observers consuming LLM events on
    # the success path MUST filter via isinstance(event,
    # LlmCompletionEvent).
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    node_events = [e for e in events if isinstance(e, NodeEvent)]
    typed_events = [e for e in events if isinstance(e, LlmCompletionEvent)]
    assert node_events == []
    assert len(typed_events) == 1


async def test_complete_failure_emits_typed_llm_failed_event_only() -> None:
    # Per proposal 0058: failures emit a typed LlmFailedEvent on the
    # observer queue ALONGSIDE the exception (the exception still
    # raises out of complete() — caller-side flow unchanged). Per
    # proposal 0049 §3 alternative 3: LlmCompletionEvent stays
    # success-only — no LlmCompletionEvent fires on failure. v0.13.0
    # dropped sentinel-namespace NodeEvent emission for LLM events
    # entirely; no NodeEvent fires on success OR failure.
    from openarmature.graph.events import LlmCompletionEvent, LlmFailedEvent, NodeEvent

    def _503(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_503)
    )
    try:
        with pytest.raises(ProviderUnavailable):
            await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    node_events = [e for e in events if isinstance(e, NodeEvent)]
    completion_events = [e for e in events if isinstance(e, LlmCompletionEvent)]
    failed_events = [e for e in events if isinstance(e, LlmFailedEvent)]
    assert node_events == []
    assert completion_events == []
    assert len(failed_events) == 1
    assert failed_events[0].error_category == "provider_unavailable"
    assert failed_events[0].error_type == "ProviderUnavailable"
    # Proposal 0082: the response-side surface is null for a non-structured
    # (no-response) failure category.
    assert failed_events[0].output_content is None
    assert failed_events[0].finish_reason is None
    assert failed_events[0].usage is None
    assert failed_events[0].response_id is None
    assert failed_events[0].response_model is None


async def test_complete_structured_output_failure_event_carries_response_surface() -> None:
    # Proposal 0082: a structured_output_invalid failure is a completion
    # whose validation gate failed, so the LlmFailedEvent carries the
    # response-side surface (finish_reason for retry triage, output_content,
    # usage, response identity), and error_message carries the failing
    # locator (the failure_description) rather than just the terse summary.
    from openarmature.graph.events import LlmFailedEvent
    from openarmature.llm import StructuredOutputInvalid

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }

    def _handler(_req: httpx.Request) -> httpx.Response:
        # Valid JSON missing the required 'age' -> schema-validation failure,
        # finish_reason "stop" (a clean-finish malformed output, not a truncation).
        return httpx.Response(
            200,
            json={
                "id": "cc-xyz",
                "object": "chat.completion",
                "model": "gpt-test-v2",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"name": "Alice"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            },
        )

    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        with pytest.raises(StructuredOutputInvalid) as excinfo:
            await provider.complete([UserMessage(content="hi")], response_schema=schema)
    finally:
        await provider.aclose()
        _release_dispatch(token)

    failed = [e for e in events if isinstance(e, LlmFailedEvent)]
    assert len(failed) == 1
    ev = failed[0]
    assert ev.error_category == "structured_output_invalid"
    assert ev.finish_reason == "stop"
    assert ev.output_content == '{"name": "Alice"}'
    assert ev.usage is not None
    assert ev.usage.completion_tokens == 8
    assert ev.response_id == "cc-xyz"
    assert ev.response_model == "gpt-test-v2"
    # error_message carries the failing-field locator, not just the terse
    # category summary (proposal 0082).
    assert "age" in ev.error_message
    assert str(excinfo.value) in ev.error_message
    assert ev.error_message != str(excinfo.value)


def test_structured_output_builder_projects_empty_content_to_none() -> None:
    # Proposal 0082 cross-observer parity (defensive): a structured failure whose
    # raw_content is empty projects output_content to None in the built event
    # (like the success path's `content or None`), so both observers omit it
    # identically rather than one rendering "" and the other dropping it on a
    # truthiness gate. The OpenAI wire can't currently produce empty content on
    # the structured path -- an empty assistant message with no tool calls fails
    # earlier as provider_invalid_response, and a tool_calls response skips
    # structured validation -- so this pins the projection at the builder.
    from openarmature.llm import StructuredOutputInvalid

    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k")
    exc = StructuredOutputInvalid(
        "empty content", response_schema={}, raw_content="", failure_description="empty"
    )
    event = provider._build_llm_failed_event(  # noqa: SLF001
        exc,
        latency_ms=1.0,
        call_id="cc",
        input_messages=[],
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
    )
    assert event.error_category == "structured_output_invalid"
    assert event.output_content is None


async def test_complete_populates_output_tool_calls_on_typed_events() -> None:
    # Proposal 0076: provider.complete() populates output_tool_calls
    # (the ToolCall records) on BOTH the terminal LlmCompletionEvent and
    # the per-attempt LlmRetryAttemptEvent — the source the OTel
    # observer renders the §5.5.1 / §5.5.10 output tool-call attributes
    # from. The per-attempt event drives the LLM span; the terminal
    # event carries the field for spec-conformance + the Langfuse path.
    def _tool_call_response(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cc-0076",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                                },
                                {
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {"name": "get_time", "arguments": '{"tz": "EST"}'},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
            },
        )

    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_tool_call_response)
    )
    try:
        await provider.complete([UserMessage(content="weather and time?")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    completion = next(e for e in events if isinstance(e, LlmCompletionEvent))
    attempt = next(e for e in events if isinstance(e, LlmRetryAttemptEvent))
    # output_content is None on a tool-call-only response; the calls
    # live in output_tool_calls instead.
    assert completion.output_content is None
    for ev in (completion, attempt):
        assert [tc.name for tc in ev.output_tool_calls] == ["get_weather", "get_time"]
        assert [tc.id for tc in ev.output_tool_calls] == ["call_a", "call_b"]
        assert ev.output_tool_calls[0].arguments == {"city": "NYC"}


# ---------------------------------------------------------------------------
# Call-level retry (proposal 0050)
# ---------------------------------------------------------------------------


def _ok_chat_completion() -> dict[str, object]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _fail_n_then_ok(calls: list[int], fail_count: int) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        if calls[0] <= fail_count:
            return httpx.Response(503, json={"error": {"message": "down"}})
        return httpx.Response(200, json=_ok_chat_completion())

    return handler


async def test_call_level_retry_succeeds_after_transient() -> None:
    calls = [0]
    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_fail_n_then_ok(calls, fail_count=1)),
    )
    try:
        response = await provider.complete(
            [UserMessage(content="hi")],
            retry=RetryConfig(max_attempts=2, backoff=deterministic_backoff(0)),
        )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    # One transient failure then success: the wire call was retried.
    assert calls[0] == 2
    assert response.message.content == "ok"
    # Terminal-only: one LlmCompletionEvent, no LlmFailedEvent for the
    # intermediate transient attempt.
    assert len([e for e in events if isinstance(e, LlmCompletionEvent)]) == 1
    assert [e for e in events if isinstance(e, LlmFailedEvent)] == []


async def test_call_level_retry_exhaustion_emits_one_failed_event() -> None:
    calls = [0]
    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_fail_n_then_ok(calls, fail_count=99)),
    )
    try:
        with pytest.raises(ProviderUnavailable):
            await provider.complete(
                [UserMessage(content="hi")],
                retry=RetryConfig(max_attempts=3, backoff=deterministic_backoff(0)),
            )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    # Exhausted all 3 attempts, then propagated. Terminal-only: one
    # LlmFailedEvent (not one per attempt), no LlmCompletionEvent.
    assert calls[0] == 3
    assert [e for e in events if isinstance(e, LlmCompletionEvent)] == []
    assert len([e for e in events if isinstance(e, LlmFailedEvent)]) == 1


async def test_call_level_retry_skips_non_transient() -> None:
    calls = [0]
    events, token = _collecting_dispatch()

    def _400(_req: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return httpx.Response(400, json={"error": {"message": "bad"}})

    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_400)
    )
    try:
        with pytest.raises(ProviderInvalidRequest):
            await provider.complete(
                [UserMessage(content="hi")],
                retry=RetryConfig(max_attempts=5, backoff=deterministic_backoff(0)),
            )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    # provider_invalid_request is non-transient: no retry, single attempt.
    assert calls[0] == 1
    assert len([e for e in events if isinstance(e, LlmFailedEvent)]) == 1


async def test_call_level_retry_invokes_on_retry_per_attempt() -> None:
    calls = [0]
    retries: list[tuple[str, int]] = []

    async def _on_retry(exc: Exception, attempt: int) -> None:
        retries.append((type(exc).__name__, attempt))

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_fail_n_then_ok(calls, fail_count=2)),
    )
    try:
        await provider.complete(
            [UserMessage(content="hi")],
            retry=RetryConfig(max_attempts=3, backoff=deterministic_backoff(0), on_retry=_on_retry),
        )
    finally:
        await provider.aclose()

    # Two transient failures then success: on_retry fires once per
    # retried attempt (before each backoff), with the 0-based index.
    assert calls[0] == 3
    assert retries == [("ProviderUnavailable", 0), ("ProviderUnavailable", 1)]


# ---------------------------------------------------------------------------
# Proposal 0058: per-category field-mapping + pre-send + mutual exclusion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc_factory", "expected_cls_name", "expected_category"),
    [
        (lambda: ProviderAuthentication("boom"), "ProviderAuthentication", "provider_authentication"),
        (lambda: ProviderUnavailable("boom"), "ProviderUnavailable", "provider_unavailable"),
        (lambda: ProviderInvalidModel("boom"), "ProviderInvalidModel", "provider_invalid_model"),
        (lambda: ProviderModelNotLoaded("boom"), "ProviderModelNotLoaded", "provider_model_not_loaded"),
        (lambda: ProviderRateLimit("boom"), "ProviderRateLimit", "provider_rate_limit"),
        (lambda: ProviderInvalidResponse("boom"), "ProviderInvalidResponse", "provider_invalid_response"),
        (lambda: ProviderInvalidRequest("boom"), "ProviderInvalidRequest", "provider_invalid_request"),
        (
            lambda: ProviderUnsupportedContentBlock("boom", block_type="image"),
            "ProviderUnsupportedContentBlock",
            "provider_unsupported_content_block",
        ),
        (
            lambda: StructuredOutputInvalid(
                "boom",
                response_schema={},
                raw_content="",
                failure_description="",
            ),
            "StructuredOutputInvalid",
            "structured_output_invalid",
        ),
    ],
)
def test_build_llm_failed_event_maps_category_and_type_per_exception(
    exc_factory: Callable[[], LlmProviderError], expected_cls_name: str, expected_category: str
) -> None:
    # Proposal 0058 field mapping: every §7 LlmProviderError subclass
    # populates the typed event's error_category from its ``category``
    # class attribute and error_type from the class name. Locks down
    # the mapping for all 9 categories so future additions to the
    # error hierarchy can't silently drop this.
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k")
    exc = exc_factory()
    event = provider._build_llm_failed_event(  # noqa: SLF001
        exc,
        latency_ms=12.0,
        call_id="cc-test",
        input_messages=[],
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
    )
    assert event.error_category == expected_category
    assert event.error_type == expected_cls_name
    # error_message is str(exc) for all categories except
    # structured_output_invalid, which appends the failure_description
    # locator (proposal 0082); startswith covers both.
    assert event.error_message.startswith("boom")
    assert event.latency_ms == 12.0
    assert event.call_id == "cc-test"


async def test_complete_pre_send_validation_emits_llm_failed_event_before_propagating() -> None:
    # Proposal 0058: §7 category exceptions raised from the pre-send
    # validation layer (before any wire contact) MUST dispatch
    # LlmFailedEvent on the observer queue alongside the exception.
    # ProviderInvalidRequest from _normalize_response_schema's non-
    # BaseModel-class rejection is the cleanest pre-send trigger
    # because it bypasses every wire concern.
    class _NotABaseModel:
        pass

    events, token = _collecting_dispatch()
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k")
    try:
        with pytest.raises(ProviderInvalidRequest):
            await provider.complete(
                [UserMessage(content="hi")],
                response_schema=cast("type", _NotABaseModel),
            )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    failed_events = [e for e in events if isinstance(e, LlmFailedEvent)]
    completion_events = [e for e in events if isinstance(e, LlmCompletionEvent)]
    assert completion_events == []
    assert len(failed_events) == 1
    assert failed_events[0].error_category == "provider_invalid_request"
    assert failed_events[0].error_type == "ProviderInvalidRequest"


async def test_llm_completion_and_failed_events_are_mutually_exclusive() -> None:
    # Proposal 0058 mutual-exclusion contract: implementations MUST
    # NOT emit both LlmCompletionEvent and LlmFailedEvent for the same
    # provider.complete() call. Verify the disjoint-count rule on
    # both success and failure paths within the same test so a future
    # restructure that accidentally emits both surfaces here.
    success_events, success_token = _collecting_dispatch()
    success_transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    success_provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=success_transport
    )
    try:
        await success_provider.complete([UserMessage(content="hi")])
    finally:
        await success_provider.aclose()
        _release_dispatch(success_token)

    success_completion = [e for e in success_events if isinstance(e, LlmCompletionEvent)]
    success_failed = [e for e in success_events if isinstance(e, LlmFailedEvent)]
    assert len(success_completion) == 1
    assert success_failed == []

    def _503(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    failure_events, failure_token = _collecting_dispatch()
    failure_provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_503)
    )
    try:
        with pytest.raises(ProviderUnavailable):
            await failure_provider.complete([UserMessage(content="hi")])
    finally:
        await failure_provider.aclose()
        _release_dispatch(failure_token)

    failure_completion = [e for e in failure_events if isinstance(e, LlmCompletionEvent)]
    failure_failed = [e for e in failure_events if isinstance(e, LlmFailedEvent)]
    assert failure_completion == []
    assert len(failure_failed) == 1


async def test_llm_completion_event_carries_typed_outcome_fields() -> None:
    # Field sourcing: provider / model / usage / request_id / finish_reason
    # / latency_ms are all populated from the response + instance state.
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 50},
        }
    )
    provider = OpenAIProvider(
        base_url="http://test", model="m-test", api_key="k", transport=transport, genai_system="vllm"
    )
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed_events = [e for e in events if isinstance(e, LlmCompletionEvent)]
    assert len(typed_events) == 1
    typed = typed_events[0]
    assert typed.provider == "vllm"
    assert typed.model == "m-test"
    assert typed.finish_reason == "stop"
    assert typed.response_id == "x"  # the helper returns id="x"
    # usage flows through the shared Usage shape; cache field surfaces
    # via the typed event without separate plumbing per the
    # proposal-0047 + proposal-0049 architectural pair.
    assert typed.usage is not None
    assert typed.usage.cached_tokens == 50
    assert typed.latency_ms is not None
    assert typed.latency_ms >= 0.0


async def test_llm_completion_event_carries_input_messages_and_output_content() -> None:
    # Proposal 0057 request-side fields: input_messages carries the
    # serialized message list; output_content carries the assistant
    # message's text. Both populated unconditionally on the typed
    # event (privacy gating sits at observer rendering).
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete(
            [SystemMessage(content="Be helpful."), UserMessage(content="hi")],
        )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.input_messages == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hi"},
    ]
    # The mock response returns content="ok" — see _make_openai_response_with_usage.
    assert typed.output_content == "ok"


async def test_llm_completion_event_output_content_none_on_tool_call_response() -> None:
    # Per llm-provider §6 mutual-exclusion: tool-call responses leave
    # AssistantMessage.content as the empty string. The typed event
    # projects that to None.
    def _handler(_req: httpx.Request) -> httpx.Response:
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
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "echo", "arguments": '{"x": 1}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_handler),
    )
    try:
        await provider.complete(
            [UserMessage(content="call echo")],
            tools=[Tool(name="echo", description="", parameters={})],
        )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.output_content is None
    assert typed.finish_reason == "tool_calls"


async def test_llm_completion_event_active_prompt_populated_from_context() -> None:
    # Proposal 0057 active_prompt: complete() invoked inside a
    # with_active_prompt block stamps the active PromptResult onto the
    # typed event (the provider reads current_prompt_result()). Covers
    # conformance fixture 064 -- the populated record on the EVENT, not
    # just the observer's span rendering of an injected field.
    from datetime import UTC, datetime

    from openarmature.prompts import PromptResult, with_active_prompt

    now = datetime.now(UTC)
    pr = PromptResult(
        name="greeting",
        version="1",
        label="production",
        template_hash="sha256:tmpl",
        rendered_hash="sha256:rendered",
        messages=[UserMessage(content="hi")],
        variables={"user": "Alice"},
        fetched_at=now,
        rendered_at=now,
    )
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        with with_active_prompt(pr):
            await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.active_prompt == pr


async def test_llm_completion_event_active_prompt_group_populated_from_context() -> None:
    # Proposal 0057 active_prompt_group: complete() inside a
    # with_active_prompt_group block stamps the active PromptGroup onto
    # the typed event (the provider reads current_prompt_group()). Covers
    # conformance fixture 066.
    from datetime import UTC, datetime

    from openarmature.prompts import PromptGroup, PromptResult, with_active_prompt_group

    now = datetime.now(UTC)

    def _pr(name: str) -> PromptResult:
        return PromptResult(
            name=name,
            version="1",
            label="production",
            template_hash="sha256:tmpl",
            rendered_hash="sha256:rendered",
            messages=[UserMessage(content="hi")],
            variables={"user": "Alice"},
            fetched_at=now,
            rendered_at=now,
        )

    # PromptGroup requires N>=2 members.
    group = PromptGroup(group_name="greetings", members=[_pr("greeting"), _pr("farewell")])
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        with with_active_prompt_group(group):
            await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.active_prompt_group == group


async def test_llm_completion_event_request_params_only_carries_supplied_keys() -> None:
    # Proposal 0057 request_params shape: absence-is-meaningful. Only
    # caller-supplied gen_ai.request.* keys appear; unset RuntimeConfig
    # fields are omitted from the mapping (NOT included with None
    # values).
    from openarmature.llm import RuntimeConfig

    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete(
            [UserMessage(content="hi")],
            config=RuntimeConfig(temperature=0.7, max_tokens=64),
        )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    # Only the two caller-supplied keys; not top_p / seed / etc.
    assert dict(typed.request_params) == {"temperature": 0.7, "max_tokens": 64}


async def test_llm_completion_event_request_extras_flows_through() -> None:
    # Proposal 0057 request_extras: RuntimeConfig extras pass-through
    # in native mapping form (not JSON-encoded).
    from openarmature.llm import RuntimeConfig

    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        # ``guided_decoding`` is a vLLM-specific extra; RuntimeConfig
        # accepts undeclared fields via extra="allow". Use model_validate
        # so pyright doesn't flag the undeclared kwarg.
        await provider.complete(
            [UserMessage(content="hi")],
            config=RuntimeConfig.model_validate({"guided_decoding": {"choice": ["a", "b"]}}),
        )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert dict(typed.request_extras) == {"guided_decoding": {"choice": ["a", "b"]}}


async def test_llm_completion_event_response_model_distinct_from_request_model() -> None:
    # Proposal 0057 response_model: provider-returned identifier,
    # distinct from the request-bound model. The OpenAI Chat Completions
    # spec lets the provider return a more specific identifier
    # (e.g. requested gpt-4o → response model gpt-4o-2024-08-06).
    def _handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cc-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-2024-08-06",  # distinct from bound model
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

    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test", model="gpt-4o", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.model == "gpt-4o"  # request-side bound model
    assert typed.response_model == "gpt-4o-2024-08-06"  # provider-returned


async def test_llm_completion_event_call_id_always_present_and_distinct_across_calls() -> None:
    # Proposal 0057 call_id contract: always present, freshly minted
    # per provider.complete() call. Two calls produce two distinct
    # call_ids.
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete([UserMessage(content="hi")])
        await provider.complete([UserMessage(content="hi again")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed_events = [e for e in events if isinstance(e, LlmCompletionEvent)]
    assert len(typed_events) == 2
    assert typed_events[0].call_id
    assert typed_events[1].call_id
    assert typed_events[0].call_id != typed_events[1].call_id


async def test_llm_completion_event_input_messages_redacts_inline_image_bytes() -> None:
    # Privacy contract: inline image bytes are redacted from
    # input_messages before population. The serializer replaces the
    # ImageSourceInline source with {"type": "inline_redacted",
    # "byte_count": N}; the raw base64_data must never appear on the
    # typed event. Catches regressions in _serialize_messages_for_payload
    # that would leak bytes through the typed-event surface.
    from openarmature.llm import ImageBlock, ImageSourceInline, TextBlock

    inline_bytes = "ZmFrZS1iYXNlNjQtZGF0YQ=="  # arbitrary base64
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete(
            [
                UserMessage(
                    content=[
                        TextBlock(text="Describe this."),
                        ImageBlock(
                            source=ImageSourceInline(base64_data=inline_bytes),
                            media_type="image/png",
                        ),
                    ]
                )
            ]
        )
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    # Raw base64 bytes MUST NOT appear anywhere in input_messages.
    serialized = repr(typed.input_messages)
    assert inline_bytes not in serialized, "inline image bytes leaked into LlmCompletionEvent.input_messages"
    # Sanity: the redaction marker IS present.
    assert "inline_redacted" in serialized
    assert "byte_count" in serialized


async def test_caller_invocation_metadata_populated_by_default() -> None:
    # Python default flips proposal 0049's spec-recommended off-by-default
    # so the bundled OTel/Langfuse observers can emit caller-metadata
    # span attributes (§5.6) without callers having to opt in.
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        set_invocation_metadata(user_id="u-123")
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.caller_invocation_metadata is not None
    assert typed.caller_invocation_metadata.get("user_id") == "u-123"


async def test_caller_invocation_metadata_omitted_when_opted_out() -> None:
    # The spec's OPTIONAL contract on caller_invocation_metadata is
    # still honored: pass ``populate_caller_metadata=False`` to suppress
    # the snapshot.
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=transport,
        populate_caller_metadata=False,
    )
    try:
        set_invocation_metadata(user_id="u-123")
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.caller_invocation_metadata is None


async def test_llm_completion_event_request_id_none_when_response_omits_id() -> None:
    # Spec proposal 0049: request_id is the provider-returned response
    # id when present; None otherwise. Pin the None case explicitly.
    def _handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                # Note: no "id" field on the response body.
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

    events, token = _collecting_dispatch()
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.response_id is None


async def test_complete_success_emits_per_attempt_then_terminal_typed_event() -> None:
    # A successful no-retry complete() emits two typed events: a
    # per-attempt LlmRetryAttemptEvent (attempt 0, driving the OTel
    # per-attempt span surface) followed by the terminal
    # LlmCompletionEvent. Both are typed — this locks the no-sentinel
    # shape so a regression re-adding a sentinel NodeEvent on success
    # would surface here as an extra NodeEvent. Spec fixture 056 pins
    # the terminal event's bracketing (it arrives between the CALLING
    # NODE's started/completed pair) end-to-end.
    events, token = _collecting_dispatch()
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)

    assert [type(e).__name__ for e in events] == ["LlmRetryAttemptEvent", "LlmCompletionEvent"]
    first, terminal = events
    assert isinstance(first, LlmRetryAttemptEvent)
    assert first.llm_attempt_index == 0
    assert isinstance(terminal, LlmCompletionEvent)


async def test_llm_completion_event_sources_node_identity_from_calling_context() -> None:
    # When the provider is called inside a node body, the typed event
    # sources node_name / namespace / attempt_index / fan_out_index /
    # branch_name from the calling-node ContextVars. The five tests
    # above all ran outside any node body (default empty/None values);
    # this test installs the ContextVars manually to confirm the
    # sourcing path actually reaches the typed event.
    events, token = _collecting_dispatch()
    namespace_token = _set_namespace_prefix(("outer", "scoring"))
    attempt_token = _set_attempt_index(2)
    fan_out_token = _set_fan_out_index(3)
    branch_token = _set_branch_name("fast")
    invocation_token = _set_invocation_id("inv-abc")
    correlation_token = _set_correlation_id("corr-xyz")
    transport = _make_openai_response_with_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k", transport=transport)
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()
        _release_dispatch(token)
        # Reset the calling-node ContextVars in the same order would
        # work but the engine's normal teardown handles this in
        # production; for the test, the order doesn't matter since
        # we're at end-of-scope.
        from openarmature.observability.correlation import (
            _reset_attempt_index,
            _reset_branch_name,
            _reset_correlation_id,
            _reset_fan_out_index,
            _reset_invocation_id,
            _reset_namespace_prefix,
        )

        _reset_correlation_id(correlation_token)
        _reset_invocation_id(invocation_token)
        _reset_branch_name(branch_token)
        _reset_fan_out_index(fan_out_token)
        _reset_attempt_index(attempt_token)
        _reset_namespace_prefix(namespace_token)

    typed = next(e for e in events if isinstance(e, LlmCompletionEvent))
    assert typed.invocation_id == "inv-abc"
    assert typed.correlation_id == "corr-xyz"
    assert typed.namespace == ("outer", "scoring")
    # node_name is the last element of the namespace per the spec
    # field-table description ("the user-defined node that issued
    # the call").
    assert typed.node_name == "scoring"
    assert typed.attempt_index == 2
    assert typed.fan_out_index == 3
    assert typed.branch_name == "fast"


# ---------------------------------------------------------------------------
# Proposal 0047: intra-impl wire-byte stability
# ---------------------------------------------------------------------------


async def test_wire_byte_equality_across_dict_key_insertion_order_on_tool_parameters() -> None:
    # Spec 0047 §8 intra-impl wire-byte stability: two structurally-
    # equivalent calls whose tool.parameters dicts differ only in
    # key insertion order MUST produce byte-identical wire bytes.
    # Caller-supplied JSON Schemas are the primary source of byte
    # drift under APC; locking them down here pins the contract.
    from openarmature.llm import Tool

    captured: list[bytes] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(bytes(req.content))
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

    # Tool A: parameters dict built with one key order.
    tool_a = Tool(
        name="lookup",
        description="Look something up.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    # Tool B: SAME schema, but ``properties`` keys + top-level keys
    # in a different insertion order.
    tool_b = Tool(
        name="lookup",
        description="Look something up.",
        parameters={
            "required": ["query"],
            "properties": {
                "limit": {"type": "integer"},
                "query": {"type": "string"},
            },
            "type": "object",
        },
    )
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([UserMessage(content="hi")], tools=[tool_a])
        await provider.complete([UserMessage(content="hi")], tools=[tool_b])
    finally:
        await provider.aclose()

    assert len(captured) == 2
    assert captured[0] == captured[1], (
        f"wire bytes differ under permuted dict keys:\n  A: {captured[0]!r}\n  B: {captured[1]!r}"
    )


async def test_wire_byte_equality_across_runtime_config_extras_dict_order() -> None:
    # Spec 0047 §8: RuntimeConfig.extras keys flow through with sorted
    # ordering even when the caller supplied them in a different
    # insertion order. Catches the vLLM ``guided_decoding={"choice":
    # ["a", "b"]}``-style extras where dict-typed values are the
    # primary cache-stability hit.
    from openarmature.llm import RuntimeConfig

    captured: list[bytes] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(bytes(req.content))
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

    config_a = RuntimeConfig.model_validate(
        {"guided_decoding": {"choice": ["a", "b"], "backend": "outlines"}}
    )
    config_b = RuntimeConfig.model_validate(
        {"guided_decoding": {"backend": "outlines", "choice": ["a", "b"]}}
    )
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([UserMessage(content="hi")], config=config_a)
        await provider.complete([UserMessage(content="hi")], config=config_b)
    finally:
        await provider.aclose()

    assert len(captured) == 2
    assert captured[0] == captured[1]


async def test_wire_byte_array_ordering_preserved() -> None:
    # Spec 0047 §8 / Q5: array ORDER is caller-supplied and MUST be
    # preserved — only dict KEYS get sorted. Verify that swapping
    # the order of items in ``stop_sequences`` produces DIFFERENT
    # wire bytes (the canonicalizer must not silently sort the list).
    from openarmature.llm import RuntimeConfig

    captured: list[bytes] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(bytes(req.content))
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

    config_a = RuntimeConfig(stop_sequences=["foo", "bar"])
    config_b = RuntimeConfig(stop_sequences=["bar", "foo"])
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([UserMessage(content="hi")], config=config_a)
        await provider.complete([UserMessage(content="hi")], config=config_b)
    finally:
        await provider.aclose()

    assert len(captured) == 2
    assert captured[0] != captured[1], (
        "caller-supplied list order MUST be preserved on the wire; "
        f"got identical bytes for [foo,bar] and [bar,foo]: {captured[0]!r}"
    )


async def test_wire_byte_equality_across_tool_call_arguments_dict_order() -> None:
    # Spec 0047 §8: tool_call.arguments is a caller-supplied dict
    # JSON-encoded into a string field. The encoded string MUST be
    # byte-stable across equivalent dicts with different key orders.
    captured: list[bytes] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(bytes(req.content))
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

    assistant_a = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="c1", name="lookup", arguments={"query": "x", "limit": 5})],
    )
    assistant_b = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="c1", name="lookup", arguments={"limit": 5, "query": "x"})],
    )
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete(
            [
                UserMessage(content="hi"),
                assistant_a,
                ToolMessage(content="result", tool_call_id="c1"),
            ]
        )
        await provider.complete(
            [
                UserMessage(content="hi"),
                assistant_b,
                ToolMessage(content="result", tool_call_id="c1"),
            ]
        )
    finally:
        await provider.aclose()

    assert len(captured) == 2
    assert captured[0] == captured[1]


async def test_wire_byte_equality_response_format_schema_under_key_permutation() -> None:
    # Spec 0047 §8 / Q5: response_format.json_schema.schema is a
    # caller-supplied JSON Schema that flows through the same
    # canonicalization path as tool.parameters. Verify byte-equality
    # under recursive key permutation including nested ``properties``.
    captured: list[bytes] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(bytes(req.content))
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
                        "message": {"role": "assistant", "content": '{"answer": "ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    schema_a: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }
    schema_b: dict[str, Any] = {
        "additionalProperties": False,
        "required": ["answer"],
        "properties": {
            "score": {"type": "number"},
            "answer": {"type": "string"},
        },
        "type": "object",
    }
    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([UserMessage(content="hi")], response_schema=schema_a)
        await provider.complete([UserMessage(content="hi")], response_schema=schema_b)
    finally:
        await provider.aclose()

    assert len(captured) == 2
    assert captured[0] == captured[1]


def test_canonicalize_dict_keys_sorts_recursively_and_preserves_lists() -> None:
    # Locks down the helper's contract directly — defensive in case
    # the wire-byte tests above ever miss a regression that surfaces
    # only in deeply nested or list-of-objects shapes.
    from openarmature.llm.providers.openai import _canonicalize_dict_keys

    src: dict[str, Any] = {
        "z": 1,
        "a": {
            "y": [{"d": 4, "c": 3}, {"b": 2, "a": 1}],
            "x": "v",
        },
    }
    result = _canonicalize_dict_keys(src)
    # Top-level keys sorted.
    assert list(result.keys()) == ["a", "z"]
    # Nested dict keys sorted.
    assert list(result["a"].keys()) == ["x", "y"]
    # List ordering preserved (the two objects stay in source order).
    assert result["a"]["y"][0] == {"c": 3, "d": 4}
    assert result["a"]["y"][1] == {"a": 1, "b": 2}
    # Inside-list dicts have sorted keys.
    assert list(result["a"]["y"][0].keys()) == ["c", "d"]
    assert list(result["a"]["y"][1].keys()) == ["a", "b"]


async def test_wire_body_top_level_keys_arrive_sorted() -> None:
    # Direct assertion on the belt-and-suspenders pass at the end of
    # _build_request_body — independent of any single apply site. Walks
    # the captured JSON body and confirms every dict at every nesting
    # level has lexicographically-sorted keys. Catches a regression
    # where a future code path adds a key after the belt-and-suspenders
    # pass would have run, or where the pass itself gets removed.
    captured: list[bytes] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(bytes(req.content))
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
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([UserMessage(content="hi")])
    finally:
        await provider.aclose()

    assert len(captured) == 1
    body = json.loads(captured[0])

    def _assert_sorted(node: Any, path: str) -> None:
        if isinstance(node, dict):
            keys = list(cast("dict[str, Any]", node).keys())
            assert keys == sorted(keys), f"keys at {path} not sorted: {keys}"
            for k, v in cast("dict[str, Any]", node).items():
                _assert_sorted(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(cast("list[Any]", node)):
                _assert_sorted(v, f"{path}[{i}]")

    _assert_sorted(body, "<root>")


async def test_wire_byte_equality_across_image_content_blocks() -> None:
    # The image content-block wire shape (``_block_to_wire``) is
    # fully OA-controlled — no caller-supplied dict passes through.
    # But the canonicalization pass at the body root walks through
    # it, and we want byte-equality across equivalent calls to be
    # observably stable (a future refactor that introduces a caller-
    # supplied source dict at this boundary would need to keep this
    # test passing). Two calls with the same image + same surrounding
    # text produce identical wire bytes.
    from openarmature.llm.messages import ImageBlock, ImageSourceURL, TextBlock

    captured: list[bytes] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(bytes(req.content))
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

    def _msg() -> UserMessage:
        return UserMessage(
            content=[
                TextBlock(text="what is this?"),
                ImageBlock(source=ImageSourceURL(url="https://example.com/img.png"), detail="auto"),
            ]
        )

    provider = OpenAIProvider(
        base_url="http://test", model="m", api_key="k", transport=httpx.MockTransport(_handler)
    )
    try:
        await provider.complete([_msg()])
        await provider.complete([_msg()])
    finally:
        await provider.aclose()

    assert len(captured) == 2
    assert captured[0] == captured[1]
