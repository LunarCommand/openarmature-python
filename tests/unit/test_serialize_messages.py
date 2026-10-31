"""``serialize_messages`` renders each message into the shape the typed LLM
events carry, with an inline image's bytes replaced by a placeholder."""

from __future__ import annotations

import json

from openarmature.llm import (
    AssistantMessage,
    ImageBlock,
    ImageSourceInline,
    ImageSourceURL,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
    serialize_messages,
)

_INLINE = "iVBORw0KGgoAAAANSUhEUg=="


def test_each_role_renders_to_its_plain_dict_shape() -> None:
    serialized = serialize_messages(
        [
            SystemMessage(content="be brief"),
            UserMessage(content="what phase is the moon in?"),
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="call-1", name="moon_phase", arguments={"date": "2026-10-30"})],
            ),
            ToolMessage(content="waxing gibbous", tool_call_id="call-1"),
        ]
    )
    assert serialized == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "what phase is the moon in?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "name": "moon_phase", "arguments": {"date": "2026-10-30"}}],
        },
        {"role": "tool", "content": "waxing gibbous", "tool_call_id": "call-1"},
    ]


def test_an_inline_image_becomes_a_placeholder_and_a_url_image_passes_through() -> None:
    serialized = serialize_messages(
        [
            UserMessage(
                content=[
                    TextBlock(text="compare these craters"),
                    ImageBlock(
                        source=ImageSourceInline(base64_data=_INLINE), media_type="image/png", detail="high"
                    ),
                    ImageBlock(source=ImageSourceURL(url="https://example.test/tycho.png")),
                ]
            )
        ]
    )
    assert serialized == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these craters"},
                {
                    "type": "image",
                    "source": {"type": "inline_redacted", "byte_count": len(_INLINE)},
                    "media_type": "image/png",
                    "detail": "high",
                },
                {"type": "image", "source": {"type": "url", "url": "https://example.test/tycho.png"}},
            ],
        }
    ]
    assert _INLINE not in json.dumps(serialized)
