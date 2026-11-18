"""Thinking and redacted-thinking blocks on assistant messages.

Fixture 043 covers the OpenAI mapping stripping one thinking block beside
one text block. These cover construction, the strip's other shapes, how
the blocks render in a typed event's input messages, and the event's
output content for a block list.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openarmature.llm import (
    AssistantMessage,
    ImageBlock,
    ImageSourceURL,
    OpenAIProvider,
    RedactedThinkingBlock,
    Response,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    Usage,
    UserMessage,
    serialize_messages,
)
from openarmature.llm._event_fields import _event_request
from openarmature.llm.providers.openai import _message_to_wire

_THOUGHT = ThinkingBlock(text="Let me reason.", signature="sig-abc")
_REDACTED = RedactedThinkingBlock(data="opaque-blob")
_CALL = ToolCall(id="call_1", name="lookup", arguments={"q": "moon"})

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_a_thinking_block_needs_text_and_may_omit_its_signature() -> None:
    assert ThinkingBlock(text="t").signature is None
    with pytest.raises(ValidationError, match="non-empty"):
        ThinkingBlock(text="")


def test_an_assistant_message_carries_a_block_list() -> None:
    msg = AssistantMessage(content=[_THOUGHT, TextBlock(text="42.")])
    assert msg.content == [_THOUGHT, TextBlock(text="42.")]


def test_an_assistant_block_list_parses_from_dicts() -> None:
    msg = AssistantMessage.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "t", "signature": "s"},
                {"type": "redacted_thinking", "data": "d"},
                {"type": "text", "text": "x"},
            ],
        }
    )
    assert [type(b) for b in msg.content] == [ThinkingBlock, RedactedThinkingBlock, TextBlock]


def test_an_empty_block_list_needs_tool_calls() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        AssistantMessage(content=[])
    assert AssistantMessage(content=[], tool_calls=[_CALL]).content == []


def test_an_image_block_is_rejected_on_an_assistant_message() -> None:
    with pytest.raises(ValidationError):
        AssistantMessage(content=[ImageBlock(source=ImageSourceURL(url="https://example.com/m.png"))])  # pyright: ignore[reportArgumentType]


def test_a_thinking_block_is_rejected_on_a_user_message() -> None:
    with pytest.raises(ValidationError):
        UserMessage(content=[_THOUGHT])  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# The OpenAI mapping strips them (llm-provider §8.1.1)
# ---------------------------------------------------------------------------


def test_the_text_blocks_go_out_joined_in_order() -> None:
    msg = AssistantMessage(content=[TextBlock(text="A"), _THOUGHT, _REDACTED, TextBlock(text="B")])
    assert _message_to_wire(msg) == {"role": "assistant", "content": "AB"}


def test_thinking_beside_tool_calls_leaves_null_content() -> None:
    wire = _message_to_wire(AssistantMessage(content=[_THOUGHT], tool_calls=[_CALL]))
    assert wire["content"] is None
    assert wire["tool_calls"][0]["id"] == "call_1"


def test_thinking_alone_goes_out_as_empty_content_without_error() -> None:
    assert _message_to_wire(AssistantMessage(content=[_THOUGHT, _REDACTED])) == {
        "role": "assistant",
        "content": "",
    }


# ---------------------------------------------------------------------------
# Typed events
# ---------------------------------------------------------------------------


def test_input_messages_keep_the_reasoning_text_and_drop_the_tokens() -> None:
    [rendered] = serialize_messages([AssistantMessage(content=[_THOUGHT, _REDACTED, TextBlock(text="42.")])])
    assert rendered == {
        "role": "assistant",
        "content": [
            {"type": "thinking", "text": "Let me reason."},
            {"type": "redacted_thinking"},
            {"type": "text", "text": "42."},
        ],
    }


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param([_THOUGHT, TextBlock(text="4"), TextBlock(text="2.")], "42.", id="text-blocks-joined"),
        pytest.param([_THOUGHT, _REDACTED], None, id="no-text"),
    ],
)
async def test_output_content_is_the_text_of_a_block_list(
    content: list[ThinkingBlock | RedactedThinkingBlock | TextBlock], expected: str | None
) -> None:
    provider = OpenAIProvider(base_url="http://test", model="m", api_key="k")
    response = Response(
        message=AssistantMessage(content=content),
        finish_reason="stop",
        usage=Usage(prompt_tokens=None, completion_tokens=None, total_tokens=None),
        raw={},
    )
    try:
        event = provider._build_llm_completion_event(  # pyright: ignore[reportPrivateUsage]
            response,
            1.0,
            call_id="c",
            request=_event_request([UserMessage(content="go")], None),
            active_prompt=None,
            active_prompt_group=None,
            token_budget=None,
        )
    finally:
        await provider.aclose()
    assert event.output_content == expected
