# Spec: realizes llm-provider §3 (Message + ToolCall typed surface +
# validation timing) and §4 (Tool definition). Tool-call ids preserved
# verbatim — no rewrite or normalization, per spec §3.

"""Message, Tool, ToolCall: the typed conversation surface.

A conversation is an ordered list of messages, one of four kinds
discriminated by ``role``: ``system``, ``user``, ``assistant``,
``tool``. Each kind has different per-role constraints (system/user
need non-empty content, assistant carries optional tool_calls, tool
needs a tool_call_id matching an earlier assistant ToolCall).

Pydantic enforces the per-role constraints at message construction
time. List-level invariants (like "every tool message's
``tool_call_id`` matches an earlier assistant ``ToolCall.id``") are
checked at the ``complete()`` boundary, not at construction (a
single Message can't see the rest of the list). Both layers are
required.

Tool-call ids are preserved verbatim; implementations MUST NOT
rewrite or normalize provider-supplied ids. The ``id`` field is a
plain ``str`` with no normalizer, so a UUID with hyphens, a
vendor-prefixed id (``bifrost_abc-def``), or any other string shape
round-trips unchanged.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCall(BaseModel):
    """An assistant's request to invoke a named tool.

    ``id`` is an opaque correlator within a single message list.
    Implementations MUST preserve provider-supplied ids verbatim;
    neither rewriting nor normalizing.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    # Per spec §3: under non-error responses, MUST be a parsed mapping
    # conforming to the tool's parameters schema. Under
    # ``finish_reason: "error"``, MAY be ``null`` if the implementation
    # could not parse the provider's bytes as JSON.
    arguments: dict[str, Any] | None


class Tool(BaseModel):
    """A function the model may request the user execute.

    ``parameters`` is a JSON Schema (object schema) describing the
    argument record. Kept as a plain ``dict[str, Any]`` rather than a
    typed schema class so the "JSON Schema, not language-native
    types" intent surfaces directly; implementations may offer
    ergonomic constructors that compile from native types (Pydantic
    ``model_json_schema()``) but the surface is JSON Schema.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


# Spec: realizes llm-provider §5 `tool_choice` discriminated-union
# (proposal 0025). The string-literal modes (`"auto"`, `"required"`,
# `"none"`) and the `ForceTool` record share the `ToolChoice` alias.
# Implementations validate `tool_choice` against `tools` before send
# (see ``validate_tool_choice`` in :mod:`provider`); violations raise
# ``ProviderInvalidRequest`` per §7.
class ForceTool(BaseModel):
    """Force the model to call exactly the named tool.

    Use the record form of the §5 `tool_choice` discriminated union
    when you need the model to call a specific tool by name. ``type``
    is the spec-level discriminator (``"tool"``); the wire mapping
    (§8.1.1) renames it to ``"function"`` for the OpenAI body. The
    ``name`` MUST match a ``Tool.name`` in the supplied ``tools``
    list; ``validate_tool_choice`` enforces this at pre-send time and
    raises ``ProviderInvalidRequest`` on violation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Frozen + extras-forbidden so a ``ForceTool`` instance is safely
    # hashable and structurally pinned. The ``Literal["tool"]`` default
    # makes ``ForceTool(name="search")`` ergonomic at the call site
    # while preserving the spec-level discriminator on the type.
    type: Literal["tool"] = "tool"
    name: str


# Per spec §5: `tool_choice` is one of:
# - ``"auto"`` — the model decides.
# - ``"required"`` — the model MUST call at least one tool.
# - ``"none"`` — the model MUST NOT call tools.
# - ``ForceTool(name=X)`` — the model MUST call the named tool.
# A union of the three string literals plus the record form.
# Callers pass ``tool_choice=None`` (the default) to omit the field
# from the wire — the provider's own default applies, preserving
# pre-0025 behavior.
ToolChoice = Literal["auto", "required", "none"] | ForceTool


# ---------------------------------------------------------------------------
# Per-role message classes
# ---------------------------------------------------------------------------


class _MessageBase(BaseModel):
    """Internal base — each per-role message subclass narrows ``role``
    via Literal and enforces its constraints."""

    model_config = ConfigDict(extra="forbid")


class SystemMessage(_MessageBase):
    """System messages have non-empty ``content``; no tool_calls; no
    tool_call_id."""

    role: Literal["system"] = "system"
    content: str

    @model_validator(mode="after")
    def _check_content(self) -> SystemMessage:
        if not self.content:
            raise ValueError("system message: content MUST be a non-empty string")
        return self


class TextBlock(BaseModel):
    """Text content block. The content-array equivalent of a plain
    text-string user message; a user message with exactly one
    ``TextBlock(text=T)`` is normatively equivalent to one with
    ``content=T``.

    Attributes:
        type: The discriminator literal ``"text"``.
        text: A non-empty string.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str

    @model_validator(mode="after")
    def _check_text(self) -> TextBlock:
        if not self.text:
            raise ValueError("text block: text MUST be a non-empty string")
        return self


class ImageSourceURL(BaseModel):
    """URL-referenced image source. The URL is passed to the provider
    unchanged; the framework does not fetch, cache, or transform it.

    Attributes:
        type: The discriminator literal ``"url"``.
        url: The image URL. MAY be ``http(s)://``, ``data:`` (RFC 2397
            inline data URI), or another scheme the provider documents
            support for.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["url"] = "url"
    url: str


class ImageSourceInline(BaseModel):
    """Inline base64-encoded image source. The framework does not
    inspect, transcode, or re-encode the bytes; the parent ``ImageBlock``
    MUST carry a ``media_type`` for inline sources.

    Attributes:
        type: The discriminator literal ``"inline"``.
        base64_data: The base64-encoded image bytes.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["inline"] = "inline"
    base64_data: str


# Discriminated union over the two image-source shapes. The
# discriminator is the source's ``type`` field, matching the spec's
# "single image block carries exactly one source — url XOR inline.
# The discriminator is the type field on the source itself."
ImageSource = Annotated[
    ImageSourceURL | ImageSourceInline,
    Field(discriminator="type"),
]


class ImageBlock(BaseModel):
    """Image content block. Carries one source (URL or inline base64),
    a conditional ``media_type`` (required for inline sources; ignored
    for URL sources), and an optional ``detail`` hint.

    The class-level default of ``detail=None`` preserves the
    omit-by-default wire behavior: providers apply their own
    conceptual default (``"auto"``) when ``detail`` is absent from the
    wire payload. To force the wire to carry an explicit ``"auto"``,
    set ``detail="auto"`` on the block.

    Attributes:
        type: The discriminator literal ``"image"``.
        source: One of ``ImageSourceURL`` or ``ImageSourceInline``.
        media_type: IANA media type. Required when source is inline.
            Permitted but redundant when source is a URL (the URL
            payload carries the content-type); the OpenAI wire path
            currently does not surface it for URL sources, but
            provider implementations MAY consume it as a hint.
            Providers MUST accept ``image/png``, ``image/jpeg``,
            ``image/webp`` at minimum and MAY accept additional
            ``image/*`` types they document support for.
        detail: Image-processing fidelity hint. One of ``"auto"``,
            ``"low"``, ``"high"``. ``None`` (the default) omits the
            field from the wire.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    source: ImageSource
    media_type: str | None = None
    detail: Literal["auto", "low", "high"] | None = None

    @model_validator(mode="after")
    def _check_media_type_for_inline(self) -> ImageBlock:
        if isinstance(self.source, ImageSourceInline) and self.media_type is None:
            raise ValueError("image block: media_type is required when source is inline")
        return self


# Discriminated union over the two content-block shapes. The
# discriminator is the block's ``type`` field, matching the spec's
# "typed record with a discriminator field identifying the block
# type."
ContentBlock = Annotated[
    TextBlock | ImageBlock,
    Field(discriminator="type"),
]


class UserMessage(_MessageBase):
    """User messages carry content as either a non-empty text string
    or a non-empty ordered sequence of content blocks (text and/or
    image). No tool_calls; no tool_call_id."""

    role: Literal["user"] = "user"
    content: str | list[ContentBlock]

    @model_validator(mode="after")
    def _check_content(self) -> UserMessage:
        if isinstance(self.content, str):
            if not self.content:
                raise ValueError("user message: content MUST be a non-empty string")
        else:
            if not self.content:
                raise ValueError("user message: content MUST be a non-empty list of content blocks")
        return self


class AssistantMessage(_MessageBase):
    """Assistant messages MAY carry ``tool_calls``. If ``tool_calls``
    is present and non-empty, ``content`` MAY be empty (the assistant
    is purely calling tools); otherwise ``content`` MUST be a
    non-empty string. ``tool_call_id`` MUST be absent."""

    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def _check_content_or_tools(self) -> AssistantMessage:
        has_tool_calls = bool(self.tool_calls)
        if not has_tool_calls and not self.content:
            raise ValueError(
                "assistant message: content MUST be a non-empty string when tool_calls is absent or empty"
            )
        return self


class ToolMessage(_MessageBase):
    """Tool messages carry the textual result of a tool call.
    ``tool_call_id`` MUST be present and match the ``id`` of an
    earlier assistant ToolCall in the same message list. The
    list-level matching is checked at the ``complete()`` boundary by
    :func:`provider.validate_message_list`, not at construction."""

    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


# Discriminated union over the four role-typed shapes. Pydantic uses
# the ``role`` field as the discriminator at parse time so a raw dict
# routes to the right subclass automatically.
Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


__all__ = [
    "AssistantMessage",
    "ContentBlock",
    "ForceTool",
    "ImageBlock",
    "ImageSource",
    "ImageSourceInline",
    "ImageSourceURL",
    "Message",
    "SystemMessage",
    "TextBlock",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolMessage",
    "UserMessage",
]
