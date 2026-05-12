# Spec: realizes llm-provider §3 (Message + Tool typed surface) and §4
# (validation timing). Tool-call ids preserved verbatim — no rewrite
# or normalization, per spec §3.

"""Message, Tool, ToolCall — the typed conversation surface.

A conversation is an ordered list of messages, one of four kinds
discriminated by ``role``: ``system``, ``user``, ``assistant``,
``tool``. Each kind has different per-role constraints (system/user
need non-empty content, assistant carries optional tool_calls, tool
needs a tool_call_id matching an earlier assistant ToolCall).

Pydantic enforces the per-role constraints at message construction
time. List-level invariants — like "every tool message's
``tool_call_id`` matches an earlier assistant ``ToolCall.id``" — are
checked at the ``complete()`` boundary, not at construction (a
single Message can't see the rest of the list). Both layers are
required.

Tool-call ids are preserved verbatim — implementations MUST NOT
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
    Implementations MUST preserve provider-supplied ids verbatim —
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
    types" intent surfaces directly — implementations may offer
    ergonomic constructors that compile from native types (Pydantic
    ``model_json_schema()``) but the surface is JSON Schema.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


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


class UserMessage(_MessageBase):
    """User messages have non-empty ``content``; no tool_calls; no
    tool_call_id."""

    role: Literal["user"] = "user"
    content: str

    @model_validator(mode="after")
    def _check_content(self) -> UserMessage:
        if not self.content:
            raise ValueError("user message: content MUST be a non-empty string")
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
    "Message",
    "SystemMessage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "UserMessage",
]
