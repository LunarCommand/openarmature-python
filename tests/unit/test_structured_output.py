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
from typing import Any, cast

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


def test_validate_response_schema_accepts_ref_to_boolean_subschema() -> None:
    # Boolean true/false are valid JSON Schema subschemas. A $ref
    # whose target is a boolean must resolve cleanly (not raise
    # ProviderInvalidRequest as if it were unresolvable).
    schema: dict[str, Any] = {
        "type": "object",
        "$defs": {"Any": True},
        "properties": {"x": {"$ref": "#/$defs/Any"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    validate_response_schema(schema)  # no raise


def test_validate_response_schema_rejects_external_ref() -> None:
    # External or otherwise unresolvable $refs would surface at
    # validate() time as raw referencing-library exceptions; the
    # boundary check should reject them with the canonical
    # ProviderInvalidRequest category.
    with pytest.raises(ProviderInvalidRequest, match="unresolvable"):
        validate_response_schema(
            {
                "type": "object",
                "properties": {"x": {"$ref": "https://example.com/schema.json"}},
                "required": ["x"],
                "additionalProperties": False,
            }
        )


def test_validate_response_schema_ignores_ref_under_data_keywords() -> None:
    # JSON Schema permits arbitrary data under keywords like
    # ``default``, ``const``, ``enum``, ``$comment``, and unknown /
    # extension keywords (``x-*``). A ``"$ref"`` key in those positions
    # is data, not a schema reference, and must not be resolved.
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {
                "type": "string",
                "default": {"$ref": "this-is-data-not-a-ref"},
            }
        },
        "required": ["x"],
        "additionalProperties": False,
    }
    validate_response_schema(schema)  # no raise


def test_validate_response_schema_accepts_percent_encoded_ref() -> None:
    # JSON Pointer fragments are URI-encoded; spaces in a $defs key
    # appear as %20 on the wire. _resolve_ref must percent-decode
    # before applying JSON Pointer's ~0/~1 unescape rules.
    schema: dict[str, Any] = {
        "type": "object",
        "$defs": {"Name With Spaces": {"type": "string"}},
        "properties": {"x": {"$ref": "#/$defs/Name%20With%20Spaces"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    validate_response_schema(schema)  # no raise


def test_validate_response_schema_accepts_draft07_schema() -> None:
    # A schema declaring draft-07 (still common in tooling) must pass
    # the boundary check via the draft-07 metaschema rather than be
    # rejected by a hard-coded 2020-12 metaschema.
    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    validate_response_schema(schema)  # no raise


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
    # Root is strict-compatible (additionalProperties: false, all
    # properties in required) so the walk DOES reach the nested
    # object. The nested object violates the rule; breaking the
    # recursion would break this test rather than be hidden by a
    # root-level fail.
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "required": [],  # nested object violates the rule
                "additionalProperties": False,
            },
        },
        "required": ["outer"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_anyof_branch_must_satisfy() -> None:
    # Root is strict-compatible so the walk reaches the anyOf branches.
    # One branch is a non-strict object (no required, no
    # additionalProperties: false) — the failure must come from there,
    # not from the root.
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "properties": {"y": {"type": "string"}},
                        # no required, no additionalProperties: false →
                        # branch violation
                    },
                ]
            },
        },
        "required": ["x"],
        "additionalProperties": False,
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
    # Root is strict-compatible so the walk reaches the $ref inside
    # properties.x. The external ref is unresolvable, so the walker
    # returns False from the ref branch (not from a root-level fail).
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"x": {"$ref": "https://example.com/external-schema.json"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_empty_property_schema_fails() -> None:
    # A property schema of {} (matches anything) cannot be statically
    # verified as strict-compatible. The walker should return False
    # rather than fall through to True.
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"x": {}},
        "required": ["x"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_array_without_items_fails() -> None:
    # An array without items has unconstrained content; the walker
    # can't statically verify nested shapes, so strict mode rejects.
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"tags": {"type": "array"}},
        "required": ["tags"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is False


def test_strict_mode_primitive_property_passes() -> None:
    # Primitive types (string, integer, number, boolean, null) carry no
    # nested structure to verify, so they are terminal-strict-compatible.
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is True


def test_strict_mode_resolves_bare_root_ref() -> None:
    # JSON Pointer "#" is a valid reference to the document root
    # (RFC 6901). A schema using $ref: "#" for self-recursion should
    # resolve through and inherit the root's strict-mode status.
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "string"}, "self": {"$ref": "#"}},
        "required": ["value", "self"],
        "additionalProperties": False,
    }
    assert strict_mode_supported(schema) is True


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


async def test_pydantic_class_path_rejects_coercible_string_for_int() -> None:
    # The dict-schema path rejects {"age": "30"} against an integer
    # field via strict jsonschema validation. The class path was
    # previously accepting the same input via Pydantic's default
    # coercive model_validate ("30" → 30). Both paths now run
    # jsonschema first, so both reject the coercion case.
    transport = _mock_chat_completion_response('{"name":"Alice","age":"30"}')
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
    assert "age" in excinfo.value.failure_description
    assert "integer" in excinfo.value.failure_description


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
    assert body_class == body_dict


# ---------------------------------------------------------------------------
# uses_prompt_augmentation_fallback inspect property
# ---------------------------------------------------------------------------


async def test_inspect_property_native_default() -> None:
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
    )
    try:
        assert provider.uses_prompt_augmentation_fallback is False
    finally:
        await provider.aclose()


async def test_inspect_property_fallback_when_forced() -> None:
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        force_prompt_augmentation_fallback=True,
    )
    try:
        assert provider.uses_prompt_augmentation_fallback is True
    finally:
        await provider.aclose()


async def test_fallback_mode_preserves_response_format_on_free_form_calls() -> None:
    # The fallback gate is structured-output-only. A free-form call
    # (response_schema=None) on a fallback-mode provider must preserve
    # a caller-supplied ``response_format`` from RuntimeConfig extras,
    # because the fallback contract only governs structured-output
    # calls.
    from openarmature.llm import RuntimeConfig

    transport = _mock_chat_completion_response('{"ok":true}')
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test-key",
        transport=transport,
        force_prompt_augmentation_fallback=True,
    )
    captured_body_response_format: dict[str, Any] | None = None
    original_post = provider._client.post

    async def capturing_post(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_body_response_format
        body = kwargs.get("json")
        if isinstance(body, dict):
            captured_body_response_format = cast("dict[str, Any]", body).get("response_format")
        return await original_post(*args, **kwargs)

    # Avoid touching the captured-request shape directly; intercept at
    # the client.post level so we see the constructed JSON body.
    provider._client.post = capturing_post  # type: ignore[method-assign]
    try:
        caller_extra = {"type": "json_object"}
        config = RuntimeConfig(response_format=caller_extra)  # type: ignore[call-arg]
        await provider.complete(
            [UserMessage(content="hello")],
            config=config,
        )
    finally:
        await provider.aclose()
    assert captured_body_response_format == caller_extra
