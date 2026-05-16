"""Focused tests for the content-block surface.

The conformance suite (``tests/conformance/test_llm_provider.py``)
covers the spec's behavioral surface end-to-end against fixtures
009–020. These unit tests fill gaps the conformance fixtures don't
exercise directly: per-class construction validation, the inline-
image-needs-media_type rule, detail-default-None wire-omission
behavior, content-rejection HTTP-error mapping heuristics, and
construction from the dict-form a fixture YAML loader would feed in.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from openarmature.llm import (
    ImageBlock,
    ImageSourceInline,
    ImageSourceURL,
    ProviderInvalidRequest,
    ProviderUnsupportedContentBlock,
    TextBlock,
    UserMessage,
)
from openarmature.llm.providers.openai import (
    _block_to_wire,
    _extract_rejected_block_type,
    _looks_like_content_rejection,
    classify_http_error,
)

# ---------------------------------------------------------------------------
# TextBlock construction
# ---------------------------------------------------------------------------


def test_text_block_accepts_non_empty_text() -> None:
    block = TextBlock(text="hello")
    assert block.type == "text"
    assert block.text == "hello"


def test_text_block_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TextBlock(text="")


# ---------------------------------------------------------------------------
# ImageBlock construction
# ---------------------------------------------------------------------------


def test_image_block_url_source_no_media_type() -> None:
    # URL sources have media_type inferred from the URL payload, so
    # media_type may be omitted on the spec block.
    block = ImageBlock(source=ImageSourceURL(url="https://example.com/a.png"))
    assert block.type == "image"
    assert isinstance(block.source, ImageSourceURL)
    assert block.media_type is None


def test_image_block_inline_source_requires_media_type() -> None:
    with pytest.raises(ValueError, match="media_type is required when source is inline"):
        ImageBlock(source=ImageSourceInline(base64_data="AAA="))


def test_image_block_inline_source_with_media_type() -> None:
    block = ImageBlock(
        source=ImageSourceInline(base64_data="AAA="),
        media_type="image/png",
    )
    assert isinstance(block.source, ImageSourceInline)
    assert block.media_type == "image/png"


def test_image_block_detail_defaults_to_none() -> None:
    block = ImageBlock(source=ImageSourceURL(url="https://example.com/a.png"))
    assert block.detail is None


def test_image_block_detail_accepts_known_values() -> None:
    for detail in ("auto", "low", "high"):
        block = ImageBlock(
            source=ImageSourceURL(url="https://example.com/a.png"),
            detail=detail,  # type: ignore[arg-type]
        )
        assert block.detail == detail


def test_image_block_detail_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        ImageBlock(
            source=ImageSourceURL(url="https://example.com/a.png"),
            detail="foo",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# UserMessage with content blocks
# ---------------------------------------------------------------------------


def test_user_message_accepts_string_content() -> None:
    msg = UserMessage(content="hello")
    assert msg.content == "hello"


def test_user_message_rejects_empty_string_content() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        UserMessage(content="")


def test_user_message_accepts_block_sequence() -> None:
    msg = UserMessage(content=[TextBlock(text="hello")])
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1


def test_user_message_rejects_empty_block_sequence() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        UserMessage(content=[])


def test_user_message_accepts_dict_form_via_discriminator() -> None:
    # The YAML loader feeds raw dicts; Pydantic's discriminated union
    # over ContentBlock's `type` field parses each dict to the right
    # variant. _build_message in test_llm_provider.py relies on this.
    raw_blocks: list[Any] = [
        {"type": "text", "text": "describe"},
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/a.png"},
        },
    ]
    msg = UserMessage(content=raw_blocks)
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextBlock)
    assert isinstance(msg.content[1], ImageBlock)


# ---------------------------------------------------------------------------
# _block_to_wire mapping
# ---------------------------------------------------------------------------


def test_block_to_wire_text() -> None:
    wire = _block_to_wire(TextBlock(text="hello"))
    assert wire == {"type": "text", "text": "hello"}


def test_block_to_wire_image_url_no_detail() -> None:
    wire = _block_to_wire(ImageBlock(source=ImageSourceURL(url="https://example.com/a.png")))
    assert wire == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
    }


def test_block_to_wire_image_url_with_detail() -> None:
    wire = _block_to_wire(
        ImageBlock(
            source=ImageSourceURL(url="https://example.com/a.png"),
            detail="high",
        )
    )
    assert wire == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png", "detail": "high"},
    }


def test_block_to_wire_image_inline_constructs_data_uri() -> None:
    wire = _block_to_wire(
        ImageBlock(
            source=ImageSourceInline(base64_data="QUJD"),
            media_type="image/jpeg",
        )
    )
    assert wire == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }


def test_block_to_wire_image_inline_with_detail() -> None:
    wire = _block_to_wire(
        ImageBlock(
            source=ImageSourceInline(base64_data="QUJD"),
            media_type="image/png",
            detail="low",
        )
    )
    assert wire == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD", "detail": "low"},
    }


# ---------------------------------------------------------------------------
# Content-rejection HTTP-error mapping
# ---------------------------------------------------------------------------


def _mock_400(
    *,
    code: str | None = None,
    error_type: str | None = None,
    message: str = "bad request",
) -> httpx.Response:
    body: dict[str, Any] = {"error": {"message": message}}
    if code is not None:
        body["error"]["code"] = code
    if error_type is not None:
        body["error"]["type"] = error_type
    return httpx.Response(400, content=json.dumps(body).encode("utf-8"))


def test_classify_400_with_known_content_code_maps_to_unsupported() -> None:
    exc = classify_http_error(
        _mock_400(
            code="image_content_not_supported",
            error_type="invalid_request_error",
            message="This model does not support image inputs.",
        )
    )
    assert isinstance(exc, ProviderUnsupportedContentBlock)
    assert exc.block_type == "image"
    assert exc.reason is not None and "image" in exc.reason.lower()


def test_classify_400_substring_fallback_via_error_message() -> None:
    exc = classify_http_error(
        _mock_400(
            code="some_other_error",
            message="This model does not support image inputs at this size.",
        )
    )
    assert isinstance(exc, ProviderUnsupportedContentBlock)


def test_classify_400_unrelated_400_stays_invalid_request() -> None:
    # A normal HTTP 400 (schema violation, missing field, etc.) must
    # still map to ProviderInvalidRequest. The content-rejection
    # heuristic is conservative — it only fires on known codes /
    # types / message patterns.
    exc = classify_http_error(_mock_400(code="invalid_field", message="messages: missing"))
    assert isinstance(exc, ProviderInvalidRequest)
    assert not isinstance(exc, ProviderUnsupportedContentBlock)


def test_extract_rejected_block_type_picks_up_image() -> None:
    assert _extract_rejected_block_type("image_content_not_supported", None) == "image"


def test_extract_rejected_block_type_picks_up_audio_from_message() -> None:
    assert _extract_rejected_block_type(None, "audio is not supported") == "audio"


def test_looks_like_content_rejection_negative_cases() -> None:
    # Unrelated codes and messages should NOT trigger the heuristic.
    assert _looks_like_content_rejection("invalid_field", None, "field missing") is False
    assert _looks_like_content_rejection(None, None, None) is False
