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
from typing import Any, Protocol, cast

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


# ---------------------------------------------------------------------------
# Schema helpers — used by structured-output Provider implementations
# ---------------------------------------------------------------------------


# Spec llm-provider §5 requires the response_schema argument to
# complete() to be a valid JSON Schema with a top-level type "object".
# The pre-send check here is the structural minimum; deeper validity
# (recursive JSON Schema correctness, vendor extensions) is delegated
# to the runtime validator at parse time.
def validate_response_schema(schema: object) -> None:
    """Pre-send validation for a JSON Schema passed as the
    ``response_schema`` argument to ``complete()``.

    Raises :class:`ProviderInvalidRequest` if the schema is not a dict
    or does not declare a top-level object type.
    """
    if not isinstance(schema, dict):
        raise ProviderInvalidRequest(f"response_schema: MUST be a dict (got {type(schema).__name__})")
    schema_dict = cast("dict[str, Any]", schema)
    schema_type = schema_dict.get("type")
    if schema_type != "object":
        raise ProviderInvalidRequest(
            f"response_schema: top-level type MUST be 'object' (got {schema_type!r})"
        )


# Strict mode (OpenAI's response_format strict:true and the analogous
# native-decoding paths in Anthropic / Gemini) requires the schema to
# satisfy two rules at every nested level:
#   1. additionalProperties is NOT true (false or absent).
#   2. every key in `properties` is listed in `required`.
# strict_mode_supported() walks the schema tree (object properties,
# array items, anyOf/oneOf/allOf branches, $ref targets with cycle
# protection) and returns True only if BOTH rules hold across the full
# tree. An unresolvable $ref or unknown-shape branch returns False —
# the safer choice when we can't statically verify the constraint.
def strict_mode_supported(schema: dict[str, Any]) -> bool:
    """Whether a JSON Schema satisfies the strict-mode constraints used
    by native-decoding LLM wire paths.

    Returns True iff for every nested (sub)schema in the tree
    ``additionalProperties`` is not ``true`` and every key in
    ``properties`` appears in ``required``. False on any violation, on
    an unresolvable ``$ref``, or on an unknown shape.

    Args:
        schema: The root JSON Schema dict.

    Returns:
        ``True`` if the schema cleanly supports strict mode; ``False``
        otherwise.
    """
    return _strict_mode_check(schema, root=schema, visited=set())


def _strict_mode_check(
    schema: Any,
    *,
    root: dict[str, Any],
    visited: set[str],
) -> bool:
    if not isinstance(schema, dict):
        return False
    schema_dict = cast("dict[str, Any]", schema)

    # $ref resolution. Cycle protection: a $ref already in `visited`
    # has been (or is being) validated up the chain; returning True
    # avoids infinite recursion without weakening the rule.
    ref = schema_dict.get("$ref")
    if isinstance(ref, str):
        if ref in visited:
            return True
        visited.add(ref)
        target = _resolve_ref(ref, root)
        if target is None:
            return False
        return _strict_mode_check(target, root=root, visited=visited)

    # Combinator branches — every branch must independently satisfy
    # the strict-mode constraints. anyOf/oneOf/allOf members may
    # themselves be arbitrary schemas; recursing handles nested
    # objects inside each.
    for combinator in ("anyOf", "oneOf", "allOf"):
        branches = schema_dict.get(combinator)
        if branches is None:
            continue
        if not isinstance(branches, list):
            return False
        for branch in cast("list[Any]", branches):
            if not _strict_mode_check(branch, root=root, visited=visited):
                return False

    schema_type = schema_dict.get("type")
    is_object_type = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in cast("list[Any]", schema_type)
    )
    is_array_type = schema_type == "array" or (
        isinstance(schema_type, list) and "array" in cast("list[Any]", schema_type)
    )

    if is_object_type:
        if schema_dict.get("additionalProperties") is True:
            return False
        properties = schema_dict.get("properties")
        if properties is not None and not isinstance(properties, dict):
            return False
        properties_dict = cast("dict[str, Any]", properties or {})
        required = schema_dict.get("required")
        if required is not None and not isinstance(required, list):
            return False
        required_set: set[str] = set(cast("list[str]", required or []))
        for prop_name, prop_schema in properties_dict.items():
            if prop_name not in required_set:
                return False
            if not _strict_mode_check(prop_schema, root=root, visited=visited):
                return False

    if is_array_type:
        items = schema_dict.get("items")
        if isinstance(items, dict):
            if not _strict_mode_check(items, root=root, visited=visited):
                return False
        elif isinstance(items, list):
            # Tuple-form items: each entry is its own schema.
            for item in cast("list[Any]", items):
                if not _strict_mode_check(item, root=root, visited=visited):
                    return False

    return True


# Internal-only $ref resolver. Handles JSON Pointer fragments rooted
# at the document (`#/$defs/Foo`, `#/definitions/Foo`); external refs
# (anything not starting with `#/`) are unresolvable here and return
# None. JSON Pointer escape rules (`~0` for `~`, `~1` for `/`) are
# unescaped per RFC 6901.
def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    current: Any = root
    for part in parts:
        decoded = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or decoded not in cast("dict[str, Any]", current):
            return None
        current = cast("dict[str, Any]", current)[decoded]
    if isinstance(current, dict):
        return cast("dict[str, Any]", current)
    return None


__all__ = [
    "Provider",
    "strict_mode_supported",
    "validate_message_list",
    "validate_response_schema",
    "validate_tools",
]
