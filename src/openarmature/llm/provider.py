# Spec: realizes llm-provider §3 (Message + validation timing),
# §5 (Provider Protocol operations), §7 (canonical error categories).

"""Provider Protocol + list-level message validation.

A ``Provider`` is stateless — every call carries the full message
list. It does not loop on tool calls (the caller is responsible for
executing tools and making a follow-on ``complete()`` with results)
and it does not retry on transient errors (that's middleware's job).

A provider MUST expose two operations:

- ``async ready() -> None`` — verifies the bound model is reachable.
  A successful return implies the next ``complete()`` would not
  raise errors that surface mismatched configuration or unloaded
  state.
- ``async complete(messages, tools=None, config=None) -> Response``
  — performs a single completion. Stateless, reentrant, MUST NOT
  mutate its inputs.

This module also exports :func:`validate_message_list`: a list-level
invariant check that complements per-message Pydantic validation. A
single ``Message`` can't see the rest of the list, so the boundary
check enforces:

- The list is non-empty.
- The first message MAY be ``system``; otherwise the list begins
  with ``user``.
- The last message before the call MUST be ``user`` or ``tool``.
- Every ``tool`` message's ``tool_call_id`` matches the ``id`` of an
  earlier assistant ``ToolCall``.

Violations raise ``provider_invalid_request``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .errors import ProviderInvalidRequest
from .messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    Tool,
    ToolMessage,
    UserMessage,
)
from .response import Response, RuntimeConfig


class Provider(Protocol):
    """The shape of any llm-provider implementation.

    Implementations are bound to a single model identifier; switching
    models means constructing a new provider, not passing a different
    argument per call.
    """

    async def ready(self) -> None:
        """Verify the bound model is reachable and serving."""
        ...

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        config: RuntimeConfig | None = None,
    ) -> Response:
        """Perform a single completion call.

        ``messages`` MUST NOT be mutated. ``complete()`` does NOT loop
        on tool calls — if the response's ``finish_reason`` is
        ``"tool_calls"``, the caller is responsible for executing the
        tools and making a follow-on call with ``tool`` messages
        appended. ``complete()`` does NOT retry; transient errors
        propagate.
        """
        ...


def validate_message_list(messages: Sequence[Message]) -> None:
    """Validate list-level invariants.

    Per-message constraints (system/user need non-empty content,
    assistant content-or-tool_calls, etc.) are enforced by Pydantic
    on the per-role Message classes at construction time. This
    function adds the list-level invariants Pydantic-on-Message
    can't see.

    Raises :class:`ProviderInvalidRequest` on the first violation.
    """
    if not messages:
        raise ProviderInvalidRequest("messages: MUST be non-empty")

    first = messages[0]
    if not isinstance(first, SystemMessage | UserMessage):
        raise ProviderInvalidRequest(
            f"messages: first message MUST be system or user (got role={first.role!r})"
        )

    # System messages are only permitted at the first position. Per
    # spec §3, at most one system message and only at the start of the
    # conversation. A system message in the middle of the list is a
    # request error.
    #
    # Note: this function does NOT enforce strict role alternation
    # (e.g., user → assistant → user). Some servers (notably vLLM with
    # Mistral templates, and other strict chat-template models) reject
    # non-alternating sequences server-side and return 400 — that
    # surfaces as ProviderInvalidRequest per §7, with the server's
    # error message preserved. Pre-rejecting at the client boundary
    # would over-restrict providers like OpenAI and Anthropic that
    # handle templating permissively.
    for idx, msg in enumerate(messages):
        if idx > 0 and isinstance(msg, SystemMessage):
            raise ProviderInvalidRequest(
                f"messages: system message MUST be the first message in the list (found at index {idx})"
            )

    last = messages[-1]
    if not isinstance(last, UserMessage | ToolMessage):
        raise ProviderInvalidRequest(
            f"messages: last message before complete() MUST be user or tool (got role={last.role!r})"
        )

    # Build the set of tool-call ids the assistant has issued, walking
    # the list in order. Each tool message is checked against the
    # ids declared by assistant messages that appeared earlier. The
    # set grows as we walk so a tool message can only reference a
    # tool call from a strictly preceding assistant message.
    declared_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                declared_tool_call_ids.add(tc.id)
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id not in declared_tool_call_ids:
                raise ProviderInvalidRequest(
                    f"messages: tool message tool_call_id={msg.tool_call_id!r} "
                    "does not match any earlier assistant ToolCall.id"
                )


def validate_tools(tools: Sequence[Tool] | None) -> None:
    """Validate tool-list invariants. Tool names MUST be unique
    within a single ``complete()`` call."""
    if not tools:
        return
    seen: set[str] = set()
    for t in tools:
        if t.name in seen:
            raise ProviderInvalidRequest(
                f"tools: duplicate tool name {t.name!r} (must be unique within a call)"
            )
        seen.add(t.name)


__all__ = [
    "Provider",
    "validate_message_list",
    "validate_tools",
]
