# Spec: realizes llm-provider §3 (Message + validation timing),
# §5 (Provider Protocol operations), §7 (canonical error categories).

"""Provider Protocol + list-level message validation.

A ``Provider`` is stateless; every call carries the full message
list. It does not loop on tool calls (the caller is responsible for
executing tools and making a follow-on ``complete()`` with results)
and it does not retry on transient errors (that's middleware's job).

A provider MUST expose two operations:

- ``async ready() -> None``: verifies the bound model is reachable.
  A successful return implies the next ``complete()`` would not
  raise errors that surface mismatched configuration or unloaded
  state.
- ``async complete(messages, tools=None, config=None, response_schema=None) -> Response``
  performs a single completion. Stateless, reentrant, MUST NOT mutate
  its inputs. When ``response_schema`` is supplied (a JSON Schema
  dict or Pydantic class), the implementation constrains the model's
  output and populates ``Response.parsed``.

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
from urllib.parse import unquote

import jsonschema
from jsonschema.validators import validator_for
from pydantic import BaseModel

from .errors import ProviderInvalidRequest
from .messages import (
    AssistantMessage,
    ForceTool,
    Message,
    SystemMessage,
    Tool,
    ToolChoice,
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
        response_schema: dict[str, Any] | type[BaseModel] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Response:
        """Perform a single completion call.

        Returns a :class:`Response` carrying the assistant message,
        finish reason, usage, and raw payload. When ``response_schema``
        is supplied and the model returns structured content,
        ``Response.parsed`` carries the validated value.

        Args:
            messages: The conversation to send. MUST NOT be mutated by
                the implementation.
            tools: Optional tool definitions the model may call.
            config: Optional per-call sampling parameters.
            response_schema: Optional JSON Schema (dict) or Pydantic
                model class describing the expected output shape. When
                supplied, the implementation constrains the model's
                output to the schema and populates ``Response.parsed``
                with the validated value.
            tool_choice: Optional tool-choice constraint (spec §5). One
                of ``"auto"``, ``"required"``, ``"none"``, or a
                :class:`ForceTool` record. When ``None`` (the default)
                the wire ``tool_choice`` field is omitted and the
                provider's own default applies. Pre-send validation
                routes through ``provider_invalid_request``.
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


# The string literals allowed under the §5 `tool_choice` shape.
# Pyright catches non-literal strings at type-check time via the
# ``ToolChoice = Literal[...] | ForceTool`` alias, but Python does
# not enforce Literal at runtime — untyped callers (tests, dynamic
# harnesses, ad-hoc scripts) can pass an arbitrary string. The
# runtime check below is the API-boundary defense against that.
_ALLOWED_TOOL_CHOICE_MODES: frozenset[str] = frozenset({"auto", "required", "none"})


# Spec: realizes llm-provider §5 `tool_choice` pre-send validation
# rules (proposal 0025). The three failure modes route through the
# existing §7 ``provider_invalid_request`` category; no new error
# categories per the spec's "no new category" framing. Validation
# fires BEFORE any HTTP request is sent (fixture 031's mock_provider
# returns an empty response list on these cases to fail the test
# if a request escapes the validation gate).
def validate_tool_choice(
    tool_choice: ToolChoice | None,
    tools: Sequence[Tool] | None,
) -> None:
    """Validate ``tool_choice`` against ``tools`` per spec §5.

    Raises :class:`ProviderInvalidRequest` (the §7
    ``provider_invalid_request`` category) on:

    - ``tool_choice`` supplied as a string that is not one of
      ``"auto"`` / ``"required"`` / ``"none"`` (runtime defense
      against untyped callers; the Literal alias catches well-typed
      ones at type-check time).
    - ``tool_choice="required"`` supplied with empty / absent
      ``tools``.
    - ``tool_choice=ForceTool(name=X)`` supplied with empty / absent
      ``tools``.
    - ``tool_choice=ForceTool(name=X)`` supplied with ``X`` not in the
      supplied tools list.

    No-op when ``tool_choice`` is ``None`` (the default — preserves
    pre-0025 behavior; the wire field is omitted and the provider's
    own default applies). ``tool_choice="auto"`` and
    ``tool_choice="none"`` have no ``tools``-related preconditions.
    """
    if tool_choice is None:
        return
    if isinstance(tool_choice, str) and tool_choice not in _ALLOWED_TOOL_CHOICE_MODES:
        raise ProviderInvalidRequest(
            f'tool_choice {tool_choice!r} is not one of "auto" / "required" / "none"'
        )
    has_tools = bool(tools)
    if tool_choice == "required" and not has_tools:
        raise ProviderInvalidRequest('tool_choice="required" requires non-empty tools')
    if isinstance(tool_choice, ForceTool):
        if not has_tools:
            raise ProviderInvalidRequest(
                f"tool_choice ForceTool(name={tool_choice.name!r}) requires non-empty tools"
            )
        # ``tools`` is non-empty here per the preceding guard. The list
        # is also guaranteed non-None inside this branch.
        names = {t.name for t in tools or ()}
        if tool_choice.name not in names:
            raise ProviderInvalidRequest(
                f"tool_choice name {tool_choice.name!r} not in tools (declared: {sorted(names)})"
            )


# ---------------------------------------------------------------------------
# Schema helpers — used by structured-output Provider implementations
# ---------------------------------------------------------------------------


# Spec llm-provider §5 requires the response_schema argument to
# complete() to be a valid JSON Schema with a top-level type "object".
# The boundary check here validates BOTH constraints: structural
# (must be a dict with top-level type: "object") AND full JSON Schema
# validity via Draft202012Validator.check_schema(). The runtime
# validator on the parse path only handles instance-against-schema
# failures; malformed schemas fail here rather than escaping at decode
# time as jsonschema.SchemaError.
def validate_response_schema(schema: object) -> None:
    """Pre-send validation for a JSON Schema passed as the
    ``response_schema`` argument to ``complete()``.

    Raises :class:`ProviderInvalidRequest` if the schema is not a dict,
    does not declare a top-level object type, or is not a valid JSON
    Schema document.
    """
    if not isinstance(schema, dict):
        raise ProviderInvalidRequest(f"response_schema: MUST be a dict (got {type(schema).__name__})")
    schema_dict = cast("dict[str, Any]", schema)
    schema_type = schema_dict.get("type")
    if schema_type != "object":
        raise ProviderInvalidRequest(
            f"response_schema: top-level type MUST be 'object' (got {schema_type!r})"
        )
    # Full JSON Schema validity check at the boundary so a malformed
    # schema raises ProviderInvalidRequest here instead of escaping as
    # jsonschema.SchemaError at decode time. ValidationError covers
    # instance-against-schema failures and is handled separately on the
    # parse path.
    try:
        # Pick the validator class the runtime would use, so the
        # boundary check uses the same metaschema as
        # jsonschema.validate(). validator_for reads the schema's
        # $schema URL; absent that, it defaults to the latest
        # supported draft.
        validator_cls = validator_for(schema_dict)
        validator_cls.check_schema(schema_dict)
    except jsonschema.SchemaError as exc:
        raise ProviderInvalidRequest(f"response_schema: not a valid JSON Schema: {exc.message}") from exc
    # check_schema() validates the schema's own syntax but does not
    # traverse $ref targets. Walk all refs in the schema and confirm
    # each resolves to a subschema within the document, so external or
    # broken refs fail here rather than escaping at parse time as
    # raw referencing-library exceptions.
    _check_refs_resolvable(schema_dict)


# Subschema-bearing keywords by container shape. Used by
# _check_refs_resolvable to walk only positions that the runtime
# treats as schemas. Anything outside these is data (default, const,
# enum, annotations like description / $comment, unknown / extension
# keywords like x-*) where a nested "$ref" key is just a value.
_SINGLE_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "propertyNames",
        "items",
        "contains",
        "if",
        "then",
        "else",
        "not",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_MAP_OF_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    }
)
_LIST_OF_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    }
)


def _check_refs_resolvable(schema: dict[str, Any]) -> None:
    """Walk subschema positions in the document and raise
    ProviderInvalidRequest for any $ref value that cannot be resolved
    internally. Skips data positions (default, const, enum,
    annotations, unknown / extension keywords) where a "$ref" key is
    just a value and the runtime would not try to resolve it.
    """

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_dict = cast("dict[str, Any]", node)
        ref = node_dict.get("$ref")
        if isinstance(ref, str) and _resolve_ref(ref, schema) is _UNRESOLVABLE:
            raise ProviderInvalidRequest(
                f"response_schema: unresolvable $ref {ref!r}; only internal "
                "refs (#/... or #) are supported by the provider's validator"
            )
        for key, value in node_dict.items():
            if key in _SINGLE_SUBSCHEMA_KEYWORDS:
                walk(value)
            elif key in _MAP_OF_SUBSCHEMA_KEYWORDS:
                if isinstance(value, dict):
                    for subschema in cast("dict[str, Any]", value).values():
                        walk(subschema)
            elif key in _LIST_OF_SUBSCHEMA_KEYWORDS:
                if isinstance(value, list):
                    for subschema in cast("list[Any]", value):
                        walk(subschema)

    walk(schema)


# Strict mode (OpenAI's response_format strict:true and the analogous
# native-decoding paths in Anthropic / Gemini) requires the schema to
# satisfy two rules at every nested level:
#   1. additionalProperties is EXPLICITLY false. OpenAI rejects schemas
#      where the key is absent, since absence means JSON Schema's
#      default of permitting extras.
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
    ``additionalProperties`` is explicitly ``false`` (an omitted key
    counts as non-strict, since JSON Schema's default is to permit
    extras) and every key in ``properties`` appears in ``required``.
    False on any violation, on an unresolvable ``$ref``, or on an
    unknown shape.

    Args:
        schema: The root JSON Schema dict.

    Returns:
        ``True`` if the schema cleanly supports strict mode; ``False``
        otherwise.
    """
    return _strict_mode_check(schema, root=schema, visited=set())


# JSON Schema primitive types: terminal-strict-compatible because they
# carry no nested structure to verify. Object/array types have their
# own branch checks; anything else (const, enum, unknown keywords,
# empty {}) is conservatively non-strict.
_PRIMITIVE_TYPES = frozenset({"string", "integer", "number", "boolean", "null"})


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
        if target is _UNRESOLVABLE:
            return False
        # _strict_mode_check on a non-dict (e.g. boolean subschema)
        # returns False via the `if not isinstance(schema, dict)` line
        # at the top — the conservative answer for strict-mode compat.
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
        if schema_dict.get("additionalProperties") is not False:
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
        # Missing or unrecognized items: contents are unconstrained and
        # may include shapes the walker can't statically verify. Strict
        # mode rejects that case.
        if items is None:
            return False
        if isinstance(items, dict):
            if not _strict_mode_check(items, root=root, visited=visited):
                return False
        elif isinstance(items, list):
            # Tuple-form items: each entry is its own schema.
            for item in cast("list[Any]", items):
                if not _strict_mode_check(item, root=root, visited=visited):
                    return False
        else:
            # items present but not dict or list (e.g. items: true) is
            # not a strict-compatible shape.
            return False

    # Determine whether the schema declared a shape we know how to
    # verify. Object/array branches above already returned False on
    # any internal violation; reaching here means all internal checks
    # passed. Combinators with all branches passing are likewise OK.
    # Primitive types are terminal. Anything else (empty schema,
    # `const`/`enum`-only, unknown keywords) is conservatively
    # non-strict — the walker can't statically verify it.
    has_combinator = any(k in schema_dict for k in ("anyOf", "oneOf", "allOf"))
    if is_object_type or is_array_type or has_combinator:
        return True
    if isinstance(schema_type, str) and schema_type in _PRIMITIVE_TYPES:
        return True
    if isinstance(schema_type, list) and all(
        isinstance(t, str) and t in _PRIMITIVE_TYPES for t in cast("list[Any]", schema_type)
    ):
        return True
    return False


# Sentinel returned by _resolve_ref when the ref isn't internally
# resolvable (path doesn't exist in the document, or the ref is
# external / relative). Distinguishing "unresolvable" from "resolved
# to a non-dict value" matters because boolean subschemas (true /
# false) are valid JSON Schema and we want to surface them to the
# caller rather than reject them at the boundary.
_UNRESOLVABLE: Any = object()


# Internal-only $ref resolver. Handles JSON Pointer fragments rooted
# at the document (`#/$defs/Foo`, `#/definitions/Foo`); external refs
# (anything not starting with `#/`) return the unresolvable sentinel.
# JSON Pointer escape rules (`~0` for `~`, `~1` for `/`) are unescaped
# per RFC 6901.
def _resolve_ref(ref: str, root: dict[str, Any]) -> Any:
    # Bare "#" is the JSON Pointer for the document root; "#/" prefixes
    # an internal path. Anything else (external URIs, relative refs we
    # can't resolve without a base) is treated as unresolvable.
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return _UNRESOLVABLE
    parts = ref[2:].split("/")
    current: Any = root
    for part in parts:
        # Per RFC 6901 §6: a JSON Pointer used as a URI fragment is
        # percent-encoded; percent-decoding happens BEFORE the
        # `~1` / `~0` JSON-Pointer unescape pair.
        decoded = unquote(part).replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or decoded not in cast("dict[str, Any]", current):
            return _UNRESOLVABLE
        current = cast("dict[str, Any]", current)[decoded]
    # Return whatever's at the resolved path — dict, bool, or otherwise.
    # Callers decide what to do with non-dict subschemas: strict-mode
    # validation conservatively rejects them; ref-resolvability
    # validation accepts them.
    return current


__all__ = [
    "Provider",
    "strict_mode_supported",
    "validate_message_list",
    "validate_response_schema",
    "validate_tool_choice",
    "validate_tools",
]
