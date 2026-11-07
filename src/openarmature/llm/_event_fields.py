"""The fields an LLM provider puts on its typed events: the observed
messages, the request parameters and the extras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openarmature.observability.llm_event import serialize_tool_calls

from .messages import AssistantMessage, ImageSourceInline, SystemMessage, TextBlock, UserMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .messages import Message
    from .response import RuntimeConfig


# Inline image sources are redacted in this step per observability
# §5.5.5: ImageSourceInline → {"type": "inline_redacted",
# "byte_count": N} where N is the byte length of the original base64
# string. media_type stays at the image-block level per llm-provider
# §3.1.2; detail is preserved when present.
#
# Redaction lives here (provider-side) rather than observer-side so
# inline image bytes never leave the provider in event form —
# defense-in-depth that applies to every observer consuming the
# payload, not just OA's own. URL-form images pass through unchanged.
def serialize_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Render messages into the plain-dict shape the typed LLM events carry
    in ``input_messages``.

    An inline image's bytes never appear in the result: its source becomes
    ``{"type": "inline_redacted", "byte_count": N}``, where ``N`` is the
    length of the base64 data, and its ``media_type`` and ``detail`` are
    kept. A URL image passes through unchanged.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                rendered_blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        rendered_blocks.append({"type": "text", "text": block.text})
                    else:  # ImageBlock
                        # The ImageBlock validator already guarantees
                        # media_type when source is inline.
                        if isinstance(block.source, ImageSourceInline):
                            byte_count = len(block.source.base64_data)
                            source_record: dict[str, Any] = {
                                "type": "inline_redacted",
                                "byte_count": byte_count,
                            }
                        else:
                            source_record = {"type": "url", "url": block.source.url}
                        image_record: dict[str, Any] = {
                            "type": "image",
                            "source": source_record,
                        }
                        if block.media_type is not None:
                            image_record["media_type"] = block.media_type
                        if block.detail is not None:
                            image_record["detail"] = block.detail
                        rendered_blocks.append(image_record)
                out.append({"role": "user", "content": rendered_blocks})
        elif isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = serialize_tool_calls(msg.tool_calls)
            out.append(entry)
        else:  # ToolMessage
            out.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
    return out


# Only set fields appear in the result. Absence is meaningful per
# observability §5.5.2: "the field was not supplied for this call"
# — distinct from "supplied with a zero value."
def _request_params_from_config(config: RuntimeConfig | None) -> dict[str, Any]:
    """Extract the cross-vendor request parameters from a
    ``RuntimeConfig`` for emission as ``gen_ai.request.*`` attributes."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if config.temperature is not None:
        out["temperature"] = config.temperature
    if config.max_tokens is not None:
        out["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        out["top_p"] = config.top_p
    if config.seed is not None:
        out["seed"] = config.seed
    # Three fields promoted in proposal 0032; surfaced under their
    # cross-vendor declared names so the observer emits
    # gen_ai.request.{frequency_penalty,presence_penalty,stop_sequences}.
    if config.frequency_penalty is not None:
        out["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        out["presence_penalty"] = config.presence_penalty
    if config.stop_sequences is not None:
        out["stop_sequences"] = config.stop_sequences
    return out


def _request_extras_from_config(config: RuntimeConfig | None) -> dict[str, Any]:
    """Return the ``RuntimeConfig`` extras pass-through bag as a plain
    dict; empty when no extras are set or when ``config`` is None."""
    if config is None:
        return {}
    return dict(config.extras)


@dataclass(frozen=True)
class _EventRequest:
    """The request-side fields of a typed LLM event, describing one request."""

    input_messages: list[dict[str, Any]]
    request_params: dict[str, Any]
    request_extras: dict[str, Any]
    structured_output_path: str | None = None


def _event_request(messages: Sequence[Message], config: RuntimeConfig | None) -> _EventRequest:
    """Describe a request by its messages and config, with no structured-output path."""
    return _EventRequest(
        input_messages=serialize_messages(messages),
        request_params=_request_params_from_config(config),
        request_extras=_request_extras_from_config(config),
    )


__all__ = [
    "_EventRequest",
    "_event_request",
    "_request_extras_from_config",
    "_request_params_from_config",
    "serialize_messages",
]
