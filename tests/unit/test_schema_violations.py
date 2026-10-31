"""Every violation a structured-output response commits, each at its own location.

Fixture 082 covers the common shapes through substring checks, which cannot
show that a violation was left out. These pin whole messages, for the shapes it
does not reach.
"""

# Spec: llm-provider §5 (draft 2020-12 by default) and §7 (every violation, each
# with an RFC 6901 URI-fragment location and the constraint that failed).

from __future__ import annotations

from typing import Any

import jsonschema.validators
import pytest
from jsonschema import Draft7Validator, Draft202012Validator
from jsonschema import _utils as jsonschema_utils
from pydantic import BaseModel, ValidationError, field_validator

from openarmature.llm._schema_violations import describe, model_violations, schema_violations
from openarmature.llm.errors import StructuredOutputInvalid
from openarmature.llm.provider import _validator_class
from openarmature.llm.providers.openai import _parse_and_validate


def _described(instance: Any, schema: dict[str, Any]) -> str:
    return describe(schema_violations(instance, schema))


@pytest.mark.parametrize(
    ("instance", "schema", "expected"),
    [
        pytest.param(
            {"site": {"foxtrot": "x"}},
            {
                "type": "object",
                "properties": {
                    "site": {
                        "anyOf": [
                            {"type": "object", "properties": {"foxtrot": {"type": "number"}}},
                            {"type": "null"},
                        ]
                    }
                },
            },
            "1 violation: #/site/foxtrot: expected type number, got string",
            id="anyof-member-of-another-type-left-out",
        ),
        pytest.param(
            {"v": {"w": "s"}},
            {
                "type": "object",
                "properties": {
                    "v": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"w": {"anyOf": [{"type": "number"}, {"type": "boolean"}]}},
                            },
                            {"type": "null"},
                        ]
                    }
                },
            },
            "1 violation: #/v/w: matches none of the anyOf member types (number, boolean)",
            id="nested-anyof",
        ),
        pytest.param(
            {"z": 1},
            {"type": "object", "properties": {"z": False}},
            "1 violation: #/z: value is not allowed (false schema)",
            id="false-property",
        ),
        pytest.param(
            {"a": 1, "b": 2, "c": 3},
            {"type": "object", "properties": {"a": {}}, "unevaluatedProperties": False},
            "2 violations: #/b: property is not allowed (unevaluatedProperties: false); "
            "#/c: property is not allowed (unevaluatedProperties: false)",
            id="unevaluated-properties",
        ),
        pytest.param(
            {"t": [1, 2, 3]},
            {"type": "object", "properties": {"t": {"prefixItems": [True], "unevaluatedItems": False}}},
            "2 violations: #/t/1: item is not allowed (unevaluatedItems: false); "
            "#/t/2: item is not allowed (unevaluatedItems: false)",
            id="unevaluated-items",
        ),
        pytest.param(
            {"ok": 1, "toolong": 2},
            {"type": "object", "propertyNames": {"maxLength": 3}},
            "1 violation: #: property name 'toolong' fails maxLength: 'toolong' is too long",
            id="property-names-at-the-object",
        ),
        pytest.param(
            # additionalItems means nothing in 2020-12, so this violation exists
            # only if the declared draft is honored.
            {"a": ["x", 1]},
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": [{"type": "string"}], "additionalItems": False}
                },
            },
            "1 violation: #/a/1: item is not allowed (additionalItems: false)",
            id="declared-draft-honored",
        ),
        pytest.param(
            {"a/b": 1, "m~n": 1, "c%d": 1},
            {
                "type": "object",
                "properties": {
                    "a/b": {"type": "string"},
                    "m~n": {"type": "string"},
                    "c%d": {"type": "string"},
                },
            },
            "3 violations: #/a~1b: expected type string, got integer; "
            "#/c%25d: expected type string, got integer; "
            "#/m~0n: expected type string, got integer",
            id="pointer-escaping",
        ),
        pytest.param(
            {"x": "s"},
            {
                "type": "object",
                "allOf": [
                    {"properties": {"x": {"type": "number"}}},
                    {"properties": {"x": {"type": "number"}}},
                ],
            },
            "1 violation: #/x: expected type number, got string",
            id="allof-duplicate-listed-once",
        ),
    ],
)
def test_each_violation_is_located_and_described(
    instance: Any, schema: dict[str, Any], expected: str
) -> None:
    assert _described(instance, schema) == expected


def test_a_false_data_value_is_not_read_as_a_schema() -> None:
    # A false subschema is rewritten as {"not": {}}, at subschema positions only.
    # Read as schemas, these three false values would reject the content.
    schema = {
        "type": "object",
        "properties": {"c": {"const": False}, "u": {"uniqueItems": False}, "e": {"enum": [False]}},
    }
    assert schema_violations({"c": False, "u": [1, 1], "e": False}, schema) == []


def test_array_indexes_sort_numerically() -> None:
    violations = schema_violations(
        {"rows": [0] * 11}, {"type": "object", "properties": {"rows": {"items": {"type": "string"}}}}
    )
    assert [v.pointer for v in violations][-2:] == ["#/rows/9", "#/rows/10"]


def test_a_schema_declaring_no_draft_is_read_as_2020_12(monkeypatch: pytest.MonkeyPatch) -> None:
    # jsonschema's own fallback is its newest draft, which is 2020-12 today, so
    # pretend it has moved on. Killed by dropping the explicit default.
    monkeypatch.setattr(jsonschema.validators, "_LATEST_VERSION", Draft7Validator)
    assert _validator_class({"type": "object"}) is Draft202012Validator


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param('{"a": NaN}', "not well-formed JSON: NaN is not a JSON value", id="nan-is-malformed"),
        pytest.param("[1, 2]", "1 violation: #: expected type object, got array", id="non-object-top-level"),
    ],
)
def test_content_failures_before_any_property(content: str, expected: str) -> None:
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate(content, {"type": "object"}, None)
    assert excinfo.value.error_message == expected


def test_a_failure_only_the_model_catches_sits_at_its_field() -> None:
    class Pet(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def _no_digits(cls, value: str) -> str:
            if any(ch.isdigit() for ch in value):
                raise ValueError("name must not contain digits")
            return value

    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate('{"name": "rex9"}', Pet.model_json_schema(), Pet)
    assert excinfo.value.error_message == "1 violation: #/name: Value error, name must not contain digits"


def test_a_model_error_inside_a_union_sits_at_its_place_in_the_response() -> None:
    # Pydantic's locations name the union member's type ("Cat", "int"); only the
    # keys and indexes are places in the response. Killed by keeping every segment.
    class Cat(BaseModel):
        meow: str

    class Dog(BaseModel):
        bark: int

    class Owner(BaseModel):
        pet: Cat | Dog
        tags: list[int | str]

    instance = {"pet": {"meow": 1}, "tags": [1, None]}
    with pytest.raises(ValidationError) as excinfo:
        Owner.model_validate(instance)
    assert describe(model_violations(excinfo.value, instance)) == (
        "4 violations: #/pet/bark: Field required; #/pet/meow: Input should be a valid string; "
        "#/tags/1: Input should be a valid integer; #/tags/1: Input should be a valid string"
    )


def test_unevaluated_keywords_fall_back_to_the_parent_without_the_private_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-key evaluation lives in a private jsonschema module. Without it the
    # violation stays at the parent, whose message names each excluded property.
    monkeypatch.delattr(jsonschema_utils, "find_evaluated_property_keys_by_schema")
    schema = {"type": "object", "properties": {"a": True}, "unevaluatedProperties": False}
    assert _described({"a": 1, "b": 2}, schema) == (
        "1 violation: #: fails unevaluatedProperties: "
        "Unevaluated properties are not allowed ('b' was unexpected)"
    )


def test_the_private_jsonschema_helpers_still_exist() -> None:
    # A tripwire. The unevaluated keywords are located with these private
    # helpers, and without them the fallback above takes over in silence. This
    # fails on a jsonschema release that moves them, so losing the precision is
    # a decision.
    assert hasattr(jsonschema_utils, "find_evaluated_property_keys_by_schema")
    assert hasattr(jsonschema_utils, "find_evaluated_item_indexes_by_schema")
