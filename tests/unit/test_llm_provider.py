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
