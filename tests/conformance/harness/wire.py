"""Generic helpers for conformance fixtures that assert on captured wire
requests and on the attributes of raised exceptions.

These helpers are capability-agnostic: any fixture format that uses
``expected_wire_request`` (literal compare with wildcards),
``expected_wire_request_checks`` (sibling boolean checks), or
``expected.raises.carries`` (error-attribute introspection) can drive
into the same helpers.

The ``"*"`` literal in an ``expected_wire_request`` string slot is a
wildcard: the actual value MUST be present and a non-empty string, but
the specific value is exempted from literal comparison. This convention
is documented in the spec's llm-provider conformance fixtures
(021/026/027) and inherited by any future capability that needs the
same shape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import httpx

WILDCARD = "*"


def request_body(captured: httpx.Request) -> dict[str, Any]:
    """Decode a captured httpx request's body as a JSON object."""
    parsed = json.loads(captured.content)
    if not isinstance(parsed, dict):
        raise AssertionError(f"wire body is not a JSON object: {parsed!r}")
    return cast("dict[str, Any]", parsed)


def match_wire_body(
    actual: Any,
    expected: Any,
    *,
    path: str = "$",
) -> None:
    """Recursive deep-equal between an actual wire-body value and an
    expected shape. Strings equal to ``"*"`` in the expected value match
    any non-empty string in the actual value. ``expected_wire_request``
    is a literal compare: keys present in ``actual`` but absent from
    ``expected`` are NOT allowed. Partial assertions belong in the
    sibling ``expected_wire_request_checks`` block.

    Raises :class:`AssertionError` with a JSON-pointer-style path on
    mismatch.
    """
    if isinstance(expected, str) and expected == WILDCARD:
        if not (isinstance(actual, str) and actual):
            raise AssertionError(
                f"wire mismatch at {path}: expected non-empty string (wildcard), got {actual!r}"
            )
        return

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise AssertionError(f"wire mismatch at {path}: expected object, got {type(actual).__name__}")
        expected_map = cast("Mapping[str, Any]", expected)
        actual_map = cast("Mapping[str, Any]", actual)
        extra = set(actual_map) - set(expected_map)
        if extra:
            raise AssertionError(f"wire mismatch at {path}: unexpected extra keys in actual: {sorted(extra)}")
        for key, exp_v in expected_map.items():
            if key not in actual_map:
                raise AssertionError(f"wire mismatch at {path}: missing key {key!r}")
            match_wire_body(actual_map[key], exp_v, path=f"{path}.{key}")
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise AssertionError(f"wire mismatch at {path}: expected list, got {type(actual).__name__}")
        expected_list = cast("list[Any]", expected)
        actual_list = cast("list[Any]", actual)
        if len(actual_list) != len(expected_list):
            raise AssertionError(
                f"wire mismatch at {path}: length differs "
                f"(actual={len(actual_list)}, expected={len(expected_list)})"
            )
        for idx, (a, e) in enumerate(zip(actual_list, expected_list, strict=True)):
            match_wire_body(a, e, path=f"{path}[{idx}]")
        return

    if actual != expected:
        raise AssertionError(f"wire mismatch at {path}: actual={actual!r}, expected={expected!r}")


def assert_response_format_absent(body: Mapping[str, Any]) -> None:
    """Assert the wire body has no ``response_format`` key."""
    if "response_format" in body:
        raise AssertionError(
            f"wire check failed: response_format present (value={body['response_format']!r}), expected absent"
        )


def assert_tool_choice_absent(body: Mapping[str, Any]) -> None:
    """Assert the wire body has no ``tool_choice`` key.

    When the caller omits ``tool_choice`` from the ``complete()`` call,
    the wire body MUST omit the field entirely so the OpenAI provider's
    own default applies. Mirrors
    :func:`assert_response_format_absent`'s pattern.
    """
    if "tool_choice" in body:
        raise AssertionError(
            f"wire check failed: tool_choice present (value={body['tool_choice']!r}), expected absent"
        )


def assert_system_references_schema(body: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Assert the first wire message is a system message whose content
    references the supplied JSON Schema (via substring match of the
    canonical-JSON form).
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise AssertionError(
            "wire check failed: expected a non-empty messages list to verify system-message presence"
        )
    first = cast("list[Any]", messages)[0]
    if not isinstance(first, dict):
        raise AssertionError(
            f"wire check failed: first message is not an object (got {first!r}), "
            "cannot verify schema-directive reference"
        )
    first_dict = cast("dict[str, Any]", first)
    if first_dict.get("role") != "system":
        raise AssertionError(
            f"wire check failed: first message is not system (got {first_dict!r}), "
            "cannot verify schema-directive reference"
        )
    content = first_dict.get("content")
    if not isinstance(content, str):
        raise AssertionError(
            f"wire check failed: system message content is not a string (got {type(content).__name__})"
        )
    schema_json = json.dumps(schema, sort_keys=True)
    if schema_json not in content:
        raise AssertionError(
            "wire check failed: system message content does not contain the serialized schema; "
            f"content={content!r}"
        )


def assert_error_carries(exc: BaseException, carries: Mapping[str, Any]) -> None:
    """Introspect attributes of a raised exception against an
    expected-carries block. Supported keys:

    - ``<attribute>_present: true`` — attribute MUST be set to a
      truthy non-None value (e.g., ``response_schema_present``,
      ``failure_description_present``).
    - ``<attribute>: <value>`` — attribute value equals the supplied
      value (e.g., ``raw_response_content: '...'``).
    - ``<attribute>_mentions: <substring>`` — string attribute value
      contains the supplied substring (e.g.,
      ``failure_description_mentions: 'age'``).
    """
    for key, expected in carries.items():
        if key.endswith("_present"):
            attr = key[: -len("_present")]
            actual = _get_carries_attr(exc, attr)
            if bool(expected) and (actual is None or actual == ""):
                raise AssertionError(f"carries check failed: expected {attr!r} to be present, got {actual!r}")
            if not bool(expected) and (actual is not None and actual != ""):
                raise AssertionError(f"carries check failed: expected {attr!r} to be absent, got {actual!r}")
        elif key.endswith("_mentions"):
            attr = key[: -len("_mentions")]
            actual = _get_carries_attr(exc, attr)
            if not isinstance(actual, str):
                raise AssertionError(
                    f"carries check failed: {attr!r} is not a string (got {type(actual).__name__}); "
                    f"cannot substring-match {expected!r}"
                )
            if expected not in actual:
                raise AssertionError(
                    f"carries check failed: {attr!r}={actual!r} does not mention {expected!r}"
                )
        else:
            actual = _get_carries_attr(exc, key)
            if isinstance(expected, Mapping):
                # A mapping-valued field (e.g. usage) is a subset match: each
                # named key must equal the actual's value, reading a pydantic
                # record's fields (Usage) or a plain mapping. Keys the fixture
                # omits are ignored (the record MAY carry optional extras).
                actual_map = _as_carries_mapping(actual)
                if actual_map is None:
                    raise AssertionError(
                        f"carries check failed: {key!r} is not a mapping/record "
                        f"(got {type(actual).__name__}); cannot subset-match {expected!r}"
                    )
                for subkey, subval in cast("Mapping[str, Any]", expected).items():
                    if actual_map.get(subkey) != subval:
                        raise AssertionError(
                            f"carries check failed: {key!r}[{subkey!r}] "
                            f"actual={actual_map.get(subkey)!r}, expected={subval!r}"
                        )
            elif actual != expected:
                raise AssertionError(
                    f"carries check failed: {key!r} actual={actual!r}, expected={expected!r}"
                )


def _as_carries_mapping(value: Any) -> Mapping[str, Any] | None:
    """Coerce a carries attribute to a mapping for subset comparison: a plain
    Mapping passes through; a pydantic record (e.g. Usage) is dumped to a dict;
    anything else returns None (not comparable as a mapping)."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, Any]", value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return cast("Mapping[str, Any]", dump())
    return None


def _get_carries_attr(exc: BaseException, name: str) -> Any:
    # Allow fixture-naming-friendly aliases for the carries block. The
    # spec fixtures use ``raw_response_content`` (the wire-side label);
    # the Python exception class names its attribute ``raw_content``.
    aliases = {"raw_response_content": "raw_content"}
    canonical = aliases.get(name, name)
    return getattr(exc, canonical, None)
