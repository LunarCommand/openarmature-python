# Spec: realizes llm-provider §5 `tool_choice` validation and wire
# mapping (proposal 0025). These tests cover the three pre-send
# validation failure modes from fixture 031 + the wire mapping rows
# from §8.1.1 + the `ForceTool` round-trip.

"""Unit tests for `tool_choice` validation and wire mapping.

Per spec llm-provider §5 (amended by proposal 0025): `tool_choice`
is one of `"auto"`, `"required"`, `"none"`, or a `ForceTool` record;
violations of the three pre-send validation rules (required-with-
empty-tools, force-specific-with-empty-tools, force-specific-with-
name-not-in-tools) raise `ProviderInvalidRequest` (§7's existing
category — no new error category per the proposal's framing).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openarmature.llm import (
    ForceTool,
    ProviderInvalidRequest,
    Tool,
    validate_tool_choice,
)


def _tool(name: str) -> Tool:
    return Tool(name=name, description=f"the {name} tool", parameters={"type": "object", "properties": {}})


# ---------------------------------------------------------------------------
# validate_tool_choice
# ---------------------------------------------------------------------------


def test_validate_none_is_noop() -> None:
    # No tool_choice supplied — no precondition to check, no raise.
    # Backward-compat with pre-0025 callers.
    validate_tool_choice(None, None)
    validate_tool_choice(None, [_tool("search")])


def test_validate_auto_passes_with_or_without_tools() -> None:
    # `auto` has no tools-related precondition.
    validate_tool_choice("auto", None)
    validate_tool_choice("auto", [])
    validate_tool_choice("auto", [_tool("search")])


def test_validate_none_string_passes_with_or_without_tools() -> None:
    # `none` mode means "MUST NOT call tools" — no precondition on
    # the tools list itself (tools can be supplied; the constraint
    # is request-side, observed via the wire).
    validate_tool_choice("none", None)
    validate_tool_choice("none", [])
    validate_tool_choice("none", [_tool("search")])


def test_validate_required_rejects_empty_tools() -> None:
    # Fixture 031 case 1: `required` mode demands the model call
    # at least one tool, but there are none to call.
    with pytest.raises(ProviderInvalidRequest, match="required.*non-empty tools"):
        validate_tool_choice("required", None)
    with pytest.raises(ProviderInvalidRequest, match="required.*non-empty tools"):
        validate_tool_choice("required", [])


def test_validate_required_passes_with_tools() -> None:
    validate_tool_choice("required", [_tool("search")])
    validate_tool_choice("required", [_tool("a"), _tool("b")])


def test_validate_force_specific_rejects_empty_tools() -> None:
    # Fixture 031 case 2: force-specific transitivity — the named
    # tool must exist in the supplied list; an empty list can't
    # contain anything.
    fc = ForceTool(name="search")
    with pytest.raises(ProviderInvalidRequest, match="requires non-empty tools"):
        validate_tool_choice(fc, None)
    with pytest.raises(ProviderInvalidRequest, match="requires non-empty tools"):
        validate_tool_choice(fc, [])


def test_validate_force_specific_rejects_name_not_in_tools() -> None:
    # Fixture 031 case 3: the forced tool name doesn't appear in
    # the supplied tools list.
    fc = ForceTool(name="search")
    with pytest.raises(ProviderInvalidRequest, match="not in tools"):
        validate_tool_choice(fc, [_tool("summarize")])


def test_validate_force_specific_passes_with_matching_tool() -> None:
    fc = ForceTool(name="search")
    validate_tool_choice(fc, [_tool("search")])
    validate_tool_choice(fc, [_tool("summarize"), _tool("search")])


# ---------------------------------------------------------------------------
# ForceTool shape
# ---------------------------------------------------------------------------


def test_force_tool_defaults_type_field() -> None:
    # Ergonomic default: `ForceTool(name="search")` is equivalent to
    # `ForceTool(type="tool", name="search")`. The `type` field's
    # Literal["tool"] default makes the call site terse without
    # losing the spec-level discriminator on the type.
    fc = ForceTool(name="search")
    assert fc.type == "tool"
    assert fc.name == "search"


def test_force_tool_is_frozen() -> None:
    # Frozen so callers can safely use the instance across
    # multiple complete() calls without worrying about accidental
    # mutation.
    fc = ForceTool(name="search")
    with pytest.raises(ValidationError):
        fc.name = "other"  # type: ignore[misc]


def test_force_tool_rejects_extras() -> None:
    # extra="forbid" so callers don't typo a future field name and
    # get silent acceptance.
    with pytest.raises(ValidationError):
        ForceTool(name="search", unknown_field="x")  # type: ignore[call-arg]


def test_force_tool_rejects_wrong_type_value() -> None:
    # The Literal["tool"] constraint catches a caller who tried to
    # pass the OpenAI wire shape ("function") into the spec-level
    # type. Wire renames happen inside `_build_request_body`, not
    # on the `ForceTool` instance.
    with pytest.raises(ValidationError):
        ForceTool(type="function", name="search")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wire mapping (§8.1.1)
# ---------------------------------------------------------------------------


def test_wire_mapping_omits_field_when_none() -> None:
    # Pre-0025 backward-compat: no tool_choice supplied means the
    # field is absent from the wire body. The OpenAI provider's
    # own default applies.
    from openarmature.llm.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key", base_url="https://example.com", model="test-model")
    body = provider._build_request_body(
        messages=[],
        tools=None,
        config=None,
        schema_dict=None,
        tool_choice=None,
    )
    assert "tool_choice" not in body


def test_wire_mapping_string_modes_pass_through() -> None:
    from openarmature.llm.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key", base_url="https://example.com", model="test-model")
    for mode in ("auto", "required", "none"):
        body = provider._build_request_body(
            messages=[],
            tools=None,
            config=None,
            schema_dict=None,
            tool_choice=mode,  # type: ignore[arg-type]
        )
        assert body["tool_choice"] == mode


def test_wire_mapping_force_tool_renames_tag() -> None:
    # Per §8.1.1: ForceTool(name=X) maps to
    # {type: "function", function: {name: X}}. The spec-level
    # discriminator `type: "tool"` renames to the wire-level
    # `type: "function"` and the name nests under a `function`
    # sub-object.
    from openarmature.llm.providers.openai import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key", base_url="https://example.com", model="test-model")
    body = provider._build_request_body(
        messages=[],
        tools=None,
        config=None,
        schema_dict=None,
        tool_choice=ForceTool(name="search"),
    )
    assert body["tool_choice"] == {"type": "function", "function": {"name": "search"}}
