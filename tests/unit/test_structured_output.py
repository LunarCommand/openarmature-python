"""Focused tests for the structured-output surface.

The conformance suite (``tests/conformance/test_llm_provider.py``)
covers the spec's behavioral surface end-to-end against fixtures
021–028. These unit tests fill gaps the conformance fixtures don't
exercise directly: the strict-mode heuristic's tree-walk edge cases
(anyOf, $ref, cycles), the schema-name derivation, the
message-augmentation directive helper, and the Pydantic-class
overload's class-in → instance-out shape.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from openarmature.llm import (
    OpenAIProvider,
    ProviderInvalidRequest,
    StructuredOutputInvalid,
    SystemMessage,
    UserMessage,
    strict_mode_supported,
    validate_response_schema,
)
from openarmature.llm.providers.openai import (
    _augment_messages_with_schema_directive,
    _derive_schema_name,
)

# ---------------------------------------------------------------------------
# validate_response_schema
# ---------------------------------------------------------------------------


def test_validate_response_schema_accepts_object_top_level() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    validate_response_schema(schema)  # no raise


def test_validate_response_schema_rejects_non_dict() -> None:
    with pytest.raises(ProviderInvalidRequest, match="MUST be a dict"):
        validate_response_schema("not a dict")  # type: ignore[arg-type]


def test_validate_response_schema_rejects_non_object_top_level() -> None:
    with pytest.raises(ProviderInvalidRequest, match="top-level type MUST be 'object'"):
        validate_response_schema({"type": "string"})


def test_validate_response_schema_rejects_missing_type() -> None:
    with pytest.raises(ProviderInvalidRequest, match="top-level type MUST be 'object'"):
        validate_response_schema({"properties": {"x": {"type": "integer"}}})


def test_validate_response_schema_rejects_malformed_schema() -> None:
    # `"type": "foobar"` is not a valid JSON Schema type keyword; the
    # boundary check should catch this and raise ProviderInvalidRequest
    # rather than letting jsonschema.SchemaError leak at parse time.
    with pytest.raises(ProviderInvalidRequest, match="not a valid JSON Schema"):
        validate_response_schema(
            {
                "type": "object",
                "properties": {"x": {"type": "foobar"}},
                "required": ["x"],
                "additionalProperties": False,
            }
        )


# ---------------------------------------------------------------------------
# strict_mode_supported
# ---------------------------------------------------------------------------


def test_strict_mode_all_required_passes() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is True


def test_strict_mode_missing_required_fails() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],  # "b" not required → violates strict
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_additional_properties_true_fails() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": True,
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_missing_additional_properties_fails() -> None:
    # OpenAI strict mode requires additionalProperties: false to be
    # EXPLICITLY set; absence (the default for Pydantic-derived schemas)
    # is not strict-compatible.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_recurses_into_nested_object() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "required": [],  # nested object violates rule
            },
        },
        "required": ["outer"],
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_anyof_branch_must_satisfy() -> None:
    # anyOf member violating the constraint → False
    schema = {
        "type": "object",
        "properties": {
            "x": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "object", "properties": {"y": {"type": "string"}}},  # no required
                ]
            },
        },
        "required": ["x"],
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_resolves_internal_ref() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            }
        },
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "required": ["inner"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is True


def test_strict_mode_unresolvable_ref_fails() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "https://example.com/external-schema.json"}},
        "required": ["x"],
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_handles_ref_cycle() -> None:
    # Self-referential schema: each entry has a "children" key pointing
    # back to the same definition. Without cycle protection this would
    # recurse forever.
    schema: dict[str, Any] = {
        "type": "object",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "children": {"$ref": "#/$defs/Node"},
                },
                "required": ["value", "children"],
                "additionalProperties": False,
            }
        },
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is True


# ---------------------------------------------------------------------------
# _derive_schema_name
# ---------------------------------------------------------------------------


def test_derive_schema_name_uses_title_when_present() -> None:
    schema: dict[str, Any] = {"type": "object", "title": "PersonRecord", "properties": {}, "required": []}
    assert _derive_schema_name(schema) == "PersonRecord"


def test_derive_schema_name_falls_back_to_hash_when_no_title() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    name = _derive_schema_name(schema)
    assert name.startswith("oa_schema_")
    assert len(name) == len("oa_schema_") + 16


def test_derive_schema_name_is_deterministic() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert _derive_schema_name(schema) == _derive_schema_name(schema)


def test_derive_schema_name_ignores_empty_title() -> None:
    schema = {"type": "object", "title": "", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    assert _derive_schema_name(schema).startswith("oa_schema_")


def test_derive_schema_name_falls_back_on_title_with_spaces() -> None:
    # OpenAI's name field rejects spaces; the hash fallback fires.
    schema = {
        "type": "object",
        "title": "Person Record",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    assert _derive_schema_name(schema).startswith("oa_schema_")


def test_derive_schema_name_falls_back_on_title_too_long() -> None:
    # OpenAI's name field has a 64-char cap; longer titles fall back.
    schema = {
        "type": "object",
        "title": "A" * 65,
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }
    assert _derive_schema_name(schema).startswith("oa_schema_")


# ---------------------------------------------------------------------------
# _augment_messages_with_schema_directive
# ---------------------------------------------------------------------------


SAMPLE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
}


def test_augment_prepends_when_no_system_message() -> None:
    original = [UserMessage(content="hello")]
    out = _augment_messages_with_schema_directive(original, SAMPLE_SCHEMA)
    assert len(out) == 2
    assert isinstance(out[0], SystemMessage)
    assert isinstance(out[1], UserMessage)
    assert out[1] is original[0]  # user message reused unchanged


def test_augment_extends_existing_system_message() -> None:
    original = [SystemMessage(content="you are helpful"), UserMessage(content="hello")]
    out = _augment_messages_with_schema_directive(original, SAMPLE_SCHEMA)
    assert len(out) == 2
    assert isinstance(out[0], SystemMessage)
    assert "you are helpful" in out[0].content
    assert "JSON Schema" in out[0].content


def test_augment_does_not_mutate_caller_list() -> None:
    original = [UserMessage(content="hello")]
    snapshot = [m.model_dump(mode="json") for m in original]
    _augment_messages_with_schema_directive(original, SAMPLE_SCHEMA)
    after = [m.model_dump(mode="json") for m in original]
    assert after == snapshot


def test_augment_includes_serialized_schema_substring() -> None:
    out = _augment_messages_with_schema_directive([UserMessage(content="x")], SAMPLE_SCHEMA)
    schema_json = json.dumps(SAMPLE_SCHEMA, sort_keys=True)
    assert schema_json in out[0].content


# ---------------------------------------------------------------------------
# Pydantic-class overload
# ---------------------------------------------------------------------------


class PersonModel(BaseModel):
    name: str
    age: int


def _mock_chat_completion_response(content: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "id": "test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        return httpx.Response(200, content=json.dumps(body).encode("utf-8"))

    return httpx.MockTransport(handler)


async def test_non_basemodel_class_raises_provider_invalid_request() -> None:
    transport = _mock_chat_completion_response('{"x":1}')
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        transport=transport,
    )
    try:
        with pytest.raises(ProviderInvalidRequest, match="BaseModel subclass"):
            await provider.complete(
                [UserMessage(content="x")],
                response_schema=str,  # type: ignore[arg-type]
            )
    finally:
        await provider.aclose()


async def test_pydantic_class_returns_validated_instance() -> None:
    transport = _mock_chat_completion_response('{"name":"Alice","age":30}')
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        transport=transport,
    )
    try:
        response = await provider.complete(
            [UserMessage(content="generate a person")],
            response_schema=PersonModel,
        )
    finally:
        await provider.aclose()
    assert isinstance(response.parsed, PersonModel)
    assert response.parsed.name == "Alice"
    assert response.parsed.age == 30


async def test_pydantic_validation_failure_wraps_in_structured_output_invalid() -> None:
    # "thirty" is not a valid int for the age field.
    transport = _mock_chat_completion_response('{"name":"Alice","age":"thirty"}')
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        transport=transport,
    )
    try:
        with pytest.raises(StructuredOutputInvalid) as excinfo:
            await provider.complete(
                [UserMessage(content="generate a person")],
                response_schema=PersonModel,
            )
    finally:
        await provider.aclose()
    err = excinfo.value
    assert err.raw_content == '{"name":"Alice","age":"thirty"}'
    assert "age" in err.failure_description


async def test_pydantic_class_wire_body_matches_dict_form() -> None:
    # The wire body produced by class-in MUST equal the wire body
    # produced by passing the equivalent JSON Schema dict.
    captured_class: list[httpx.Request] = []
    captured_dict: list[httpx.Request] = []

    def handler_class(request: httpx.Request) -> httpx.Response:
        captured_class.append(request)
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": '{"name":"A","age":1}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode("utf-8"),
        )

    def handler_dict(request: httpx.Request) -> httpx.Response:
        captured_dict.append(request)
        return handler_class(request)

    transport_class = httpx.MockTransport(handler_class)
    transport_dict = httpx.MockTransport(handler_dict)

    p_class = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        transport=transport_class,
    )
    p_dict = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        transport=transport_dict,
    )

    schema_from_class = PersonModel.model_json_schema()

    try:
        await p_class.complete([UserMessage(content="x")], response_schema=PersonModel)
        await p_dict.complete([UserMessage(content="x")], response_schema=schema_from_class)
    finally:
        await p_class.aclose()
        await p_dict.aclose()

    body_class = json.loads(captured_class[0].content)
    body_dict = json.loads(captured_dict[0].content)
    assert body_class["response_format"] == body_dict["response_format"]


# ---------------------------------------------------------------------------
# uses_prompt_augmentation_fallback inspect property
# ---------------------------------------------------------------------------


def test_inspect_property_native_default() -> None:
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
    )
    assert provider.uses_prompt_augmentation_fallback is False


def test_inspect_property_fallback_when_forced() -> None:
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        force_prompt_augmentation_fallback=True,
    )
    assert provider.uses_prompt_augmentation_fallback is True
