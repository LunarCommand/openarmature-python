"""Every violation a structured-output response commits, each at its own location.

Fixture 082 covers the common shapes through substring checks, which cannot
show that a violation was left out. These pin whole messages, for the shapes it
does not reach.
"""

# Spec: llm-provider §5 (draft 2020-12 by default) and §7 (every violation, each
# with an RFC 6901 URI-fragment location and the constraint that failed).

from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from typing import Annotated, Any, Literal

import jsonschema.validators
import pytest
from jsonschema import Draft7Validator, Draft202012Validator
from jsonschema import _utils as jsonschema_utils
from jsonschema.exceptions import UnknownType
from pydantic import AfterValidator, BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

from openarmature.llm._schema_violations import _missing, describe, model_violations, schema_violations
from openarmature.llm.errors import StructuredOutputInvalid
from openarmature.llm.provider import _validator_class
from openarmature.llm.providers.openai import _parse_and_validate

_DRAFT_03 = "http://json-schema.org/draft-03/schema#"
_DRAFT_04 = "http://json-schema.org/draft-04/schema#"
_DRAFT_06 = "http://json-schema.org/draft-06/schema#"
_DRAFT_07 = "http://json-schema.org/draft-07/schema#"
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"


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
            {"items": 1, "additionalProperties": 2},
            {"type": "object", "properties": {"items": False, "additionalProperties": False}},
            "2 violations: #/additionalProperties: value is not allowed (false schema); "
            "#/items: value is not allowed (false schema)",
            id="false-properties-named-like-keywords",
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
            "1 violation: #/toolong: property name 'toolong' fails maxLength 3: 'toolong' is too long",
            id="property-name-at-its-property",
        ),
        pytest.param(
            {"propertyNames": {}},
            {"type": "object", "properties": {"propertyNames": {"required": ["a", "b"]}}},
            "2 violations: #/propertyNames/a: required property is missing; "
            "#/propertyNames/b: required property is missing",
            id="property-called-property-names",
        ),
        pytest.param(
            # The second "properties" is a name, so the propertyNames after it is
            # the keyword.
            {"properties": {"toolong": 1}},
            {
                "type": "object",
                "properties": {"properties": {"type": "object", "propertyNames": {"maxLength": 3}}},
            },
            "1 violation: #/properties/toolong: "
            "property name 'toolong' fails maxLength 3: 'toolong' is too long",
            id="property-names-under-a-property-called-properties",
        ),
        pytest.param(
            {"zzz": 1},
            {"type": "object", "propertyNames": {"anyOf": [{"pattern": "^a"}, {"maxLength": 2}]}},
            "2 violations: #/zzz: property name 'zzz' fails maxLength 2: 'zzz' is too long; "
            "#/zzz: property name 'zzz' fails pattern '^a': 'zzz' does not match '^a'",
            id="union-under-property-names-expanded",
        ),
        pytest.param(
            {"abc": 1},
            {"type": "object", "propertyNames": {"oneOf": [{"maxLength": 5}, {"pattern": "^a"}]}},
            "1 violation: #/abc: property name 'abc' matches more than one oneOf member",
            id="property-name-matching-two-oneof-members",
        ),
        pytest.param(
            {"zzz": 1},
            {"type": "object", "propertyNames": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
            "1 violation: #/zzz: property name 'zzz' matches none of the anyOf member types (integer, null)",
            id="property-name-matching-no-member-type",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            {"type": "object", "propertyNames": False},
            "2 violations: #/a: property name 'a' is not allowed (propertyNames: false); "
            "#/b: property name 'b' is not allowed (propertyNames: false)",
            id="false-property-names",
        ),
        pytest.param(
            {"zzz": 1},
            {"type": "object", "propertyNames": {"anyOf": [False, {"maxLength": 2}]}},
            "2 violations: #/zzz: property name 'zzz' fails maxLength 2: 'zzz' is too long; "
            "#/zzz: property name 'zzz' is not allowed (false schema)",
            id="false-union-member-under-property-names",
        ),
        pytest.param(
            # additionalItems means nothing in 2020-12, so this violation exists
            # only if the declared draft is honored.
            {"a": ["x", 1]},
            {
                "$schema": _DRAFT_07,
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": [{"type": "string"}], "additionalItems": False}
                },
            },
            "1 violation: #/a/1: item is not allowed (additionalItems: false)",
            id="declared-draft-honored",
        ),
        pytest.param(
            # items is a list here, so the false entry is not items: false.
            {"t": ["a", 1]},
            {
                "$schema": _DRAFT_07,
                "type": "object",
                "properties": {"t": {"type": "array", "items": [{"type": "string"}, False]}},
            },
            "1 violation: #/t/1: value is not allowed (false schema)",
            id="false-entry-in-a-list-of-items",
        ),
        pytest.param(
            # Draft-03 has no `not`, so {"not": {}} would accept anything, and the
            # excluded properties are found from the schema instead. x1 matches a
            # pattern, so it is allowed.
            {"a": "x", "b": 1, "c": 2, "x1": 3},
            {
                "$schema": _DRAFT_03,
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "patternProperties": {"^x": {}},
                "additionalProperties": False,
            },
            "2 violations: #/b: property is not allowed (additionalProperties: false); "
            "#/c: property is not allowed (additionalProperties: false)",
            id="draft-03-additional-properties-false",
        ),
        pytest.param(
            {"t": ["x", 1, 2]},
            {
                "$schema": _DRAFT_03,
                "type": "object",
                "properties": {
                    "t": {"type": "array", "items": [{"type": "string"}], "additionalItems": False}
                },
            },
            "2 violations: #/t/1: item is not allowed (additionalItems: false); "
            "#/t/2: item is not allowed (additionalItems: false)",
            id="draft-03-additional-items-false",
        ),
        pytest.param(
            # jsonschema validates a subschema by its own $schema, and draft-03
            # has no `not`, so a rewritten false here would accept anything.
            {"site": {"extra": 1}},
            {
                "type": "object",
                "properties": {
                    "site": {"$schema": _DRAFT_03, "type": "object", "additionalProperties": False}
                },
            },
            "1 violation: #/site/extra: property is not allowed (additionalProperties: false)",
            id="embedded-draft-03",
        ),
        pytest.param(
            {"t": [1, 2]},
            {
                "$schema": _DRAFT_03,
                "type": "object",
                "properties": {"t": {"$schema": _DRAFT_2020_12, "type": "array", "items": False}},
            },
            "2 violations: #/t/0: item is not allowed (items: false); "
            "#/t/1: item is not allowed (items: false)",
            id="embedded-2020-12-in-draft-03",
        ),
        pytest.param(
            {},
            {
                "$schema": _DRAFT_03,
                "type": "object",
                "properties": {"a": {"type": "string", "required": True}},
            },
            "1 violation: #/a: fails required: 'a' is a required property",
            id="draft-03-boolean-required",
        ),
        pytest.param(
            {"a": 1},
            {"$schema": _DRAFT_03, "type": "object", "dependencies": {"a": "b"}},
            "1 violation: #/b: property required by 'a' is missing",
            id="draft-03-single-name-dependency",
        ),
        pytest.param(
            # A fragment may hold "?" as it is, so it is not percent-encoded.
            {"a/b": 1, "m~n": 1, "c%d": 1, "what?": 1},
            {
                "type": "object",
                "properties": {
                    "a/b": {"type": "string"},
                    "m~n": {"type": "string"},
                    "c%d": {"type": "string"},
                    "what?": {"type": "string"},
                },
            },
            "4 violations: #/a~1b: expected type string, got integer; "
            "#/c%25d: expected type string, got integer; "
            "#/m~0n: expected type string, got integer; "
            "#/what?: expected type string, got integer",
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
        pytest.param(
            # jsonschema leaves $ref out of the schema path, so both required
            # lists report at the same path.
            {},
            {"type": "object", "$defs": {"A": {"required": ["a"]}}, "$ref": "#/$defs/A", "required": ["b"]},
            "2 violations: #/a: required property is missing; #/b: required property is missing",
            id="referenced-and-sibling-required",
        ),
    ],
)
def test_each_violation_is_located_and_described(
    instance: Any, schema: dict[str, Any], expected: str
) -> None:
    assert _described(instance, schema) == expected


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        pytest.param(_DRAFT_04, "1 violation: #/site/lat: required property is missing", id="draft-04"),
        pytest.param(_DRAFT_06, "1 violation: #/site/lat: required property is missing", id="draft-06"),
        pytest.param(_DRAFT_07, "1 violation: #/site/lat: required property is missing", id="draft-07"),
        # 2020-12 applies every keyword beside $ref, so the member is null-only.
        pytest.param(
            None, "1 violation: #/site: matches none of the anyOf member types (null)", id="2020-12"
        ),
    ],
)
def test_a_type_beside_ref_counts_only_where_the_draft_applies_it(
    declared: str | None, expected: str
) -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "definitions": {"A": {"type": "object", "required": ["lat"]}},
        "properties": {"site": {"anyOf": [{"$ref": "#/definitions/A", "type": "null"}, {"type": "null"}]}},
    }
    if declared is not None:
        schema["$schema"] = declared
    assert _described({"site": {}}, schema) == expected


def test_a_false_data_value_is_not_read_as_a_schema() -> None:
    # A false subschema is rewritten as {"not": {}}, at subschema positions only.
    # Read as schemas, these three false values would reject the content.
    schema = {
        "type": "object",
        "properties": {"c": {"const": False}, "u": {"uniqueItems": False}, "e": {"enum": [False]}},
    }
    assert schema_violations({"c": False, "u": [1, 1], "e": False}, schema) == []


def test_an_error_no_case_recognizes_is_still_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    # Content jsonschema rejects is never accepted, whatever the cases make of
    # its error. Killed by dropping the fallback in schema_violations.
    def recognize_nothing(*_: Any) -> Iterator[Any]:
        return iter(())

    monkeypatch.setattr("openarmature.llm._schema_violations._violations", recognize_nothing)
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert _described({"a": "x"}, schema) == (
        "1 violation: #/a: fails type 'integer': 'x' is not of type 'integer'"
    )


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
        # Python reads both as infinity, which no JSON number is.
        pytest.param('{"a": 1e400}', "number out of range: 1e400", id="past-a-float"),
        pytest.param('{"a": -1e400}', "number out of range: -1e400", id="past-a-float-negative"),
        pytest.param("[" * 200_000, "JSON nested too deeply to parse", id="nested-too-deeply"),
    ],
)
def test_content_failures_before_any_property(content: str, expected: str) -> None:
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate(content, {"type": "object"}, None)
    assert excinfo.value.error_message == expected


def test_content_that_is_not_an_object_is_rejected_whatever_the_draft_applies() -> None:
    # Drafts 3 to 7 ignore every sibling of a $ref, so this root type never
    # applies. Killed by dropping the object check in response_violations.
    schema: dict[str, Any] = {
        "$schema": _DRAFT_07,
        "type": "object",
        "$ref": "#/definitions/site",
        "definitions": {"site": {}},
    }
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate("null", schema, None)
    assert excinfo.value.error_message == "1 violation: #: expected type object, got null"


class _Ledger(BaseModel):
    m: dict[str, int]


@pytest.mark.parametrize("schema_class", [None, _Ledger])
def test_a_key_that_is_a_lone_surrogate_still_has_a_pointer(schema_class: type[BaseModel] | None) -> None:
    # JSON text can escape a lone surrogate, which UTF-8 cannot encode. Killed by
    # percent-encoding strictly, which fails while the pointer is written.
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate('{"m": {"\\ud800": "x"}}', _Ledger.model_json_schema(), schema_class)
    assert excinfo.value.error_message == "1 violation: #/m/%ED%A0%80: expected type integer, got string"


def _positive(value: int) -> int:
    if value <= 0:
        raise ValueError("must be positive")
    return value


class _PositiveLedger(BaseModel):
    m: dict[str, Annotated[int, AfterValidator(_positive)]]


def test_a_model_error_at_a_lone_surrogate_key_sits_at_that_key() -> None:
    # Pydantic writes the key in its location with U+FFFD in place of the
    # surrogate. Killed by matching keys exactly, which leaves it at #/m.
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate(
            '{"m": {"a\\ud800": -1, "ok": 1}}', _PositiveLedger.model_json_schema(), _PositiveLedger
        )
    assert excinfo.value.error_message == "1 violation: #/m/a%ED%A0%80: Value error, must be positive"


def test_a_remote_reference_is_never_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    # A $ref inside draft-07 dependencies is one validate_response_schema does
    # not reach. Killed by validating with jsonschema's default registry, which
    # fetches it.
    fetched: list[Any] = []

    def urlopen(*args: Any, **_: Any) -> Any:
        fetched.append(args)
        raise OSError("no network here")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    schema = {
        "$schema": _DRAFT_07,
        "type": "object",
        "dependencies": {"a": {"$ref": "https://example.com/remote.json"}},
    }
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate('{"a": 1}', schema, None)
    assert excinfo.value.error_message.startswith("could not be validated: ")
    assert fetched == []


@pytest.mark.parametrize(
    ("content", "schema", "cause"),
    [
        pytest.param(
            '{"a": [' + ", ".join(["0"] * 1000) + "]}",
            {"$schema": _DRAFT_03, "type": "object", "properties": {"a": {"type": "money"}}},
            UnknownType,
            id="unknown-draft-03-type",
        ),
        pytest.param(
            '{"c":' * 300 + "{}" + "}" * 300,
            {"type": "object", "properties": {"c": {"$ref": "#"}}, "required": ["x"]},
            RecursionError,
            id="past-the-recursion-limit",
        ),
    ],
)
def test_validation_that_cannot_finish_is_still_structured_output_invalid(
    content: str, schema: dict[str, Any], cause: type[BaseException]
) -> None:
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate(content, schema, None)
    assert excinfo.value.error_message.startswith("could not be validated: ")
    assert isinstance(excinfo.value.__cause__, cause)
    # jsonschema's UnknownType text quotes the instance whole.
    assert len(excinfo.value.error_message) < len(content)


def test_a_message_does_not_grow_with_the_value_it_quotes() -> None:
    # jsonschema's enum message lists every allowed value. Killed by writing the
    # message whole, which makes the longer enum's line longer.
    def described(size: int) -> str:
        schema = {"type": "object", "properties": {"code": {"enum": [f"code-{n:04d}" for n in range(size)]}}}
        return _described({"code": "x"}, schema)

    shorter, longer = described(100), described(1000)
    assert len(shorter) == len(longer)
    assert longer.startswith("1 violation: #/code: fails enum: 'x' is not one of ['code-0000', ")


def test_two_constraints_cut_to_the_same_words_stay_two_violations() -> None:
    # Both messages start with the same forty values, past the point where they
    # are cut. Killed by de-duplicating on the cut text.
    shared = [f"code-{n:04d}" for n in range(40)]
    schema = {
        "type": "object",
        "properties": {"c": {"allOf": [{"enum": [*shared, "x-one"]}, {"enum": [*shared, "x-two"]}]}},
    }
    assert [violation.pointer for violation in schema_violations({"c": "nope"}, schema)] == ["#/c", "#/c"]


@pytest.mark.parametrize(
    ("site", "lists"),
    [
        pytest.param({"type": "object", "required": ["a", "b", "c"]}, 1, id="object"),
        pytest.param(
            {"anyOf": [{"type": "object", "required": ["a", "b", "c"]}, {"type": "null"}]},
            1,
            id="union-member",
        ),
        # Killed by keying the expansion on the object alone, which skips the
        # second list.
        pytest.param(
            {"type": "object", "allOf": [{"required": ["a", "b"]}, {"required": ["c"]}]},
            2,
            id="two-lists",
        ),
    ],
)
def test_a_required_list_is_expanded_once_per_object(
    monkeypatch: pytest.MonkeyPatch, site: dict[str, Any], lists: int
) -> None:
    # jsonschema reports each missing name on its own, and each report carries the
    # whole list. Killed by expanding a list for every report.
    calls: list[Any] = []

    def counted(*args: Any) -> Iterator[Any]:
        calls.append(args)
        return _missing(*args)

    monkeypatch.setattr("openarmature.llm._schema_violations._missing", counted)
    schema = {"type": "object", "properties": {"s": site}}
    assert _described({"s": {}}, schema) == (
        "3 violations: #/s/a: required property is missing; #/s/b: required property is missing; "
        "#/s/c: required property is missing"
    )
    assert len(calls) == lists


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


def test_a_missing_value_with_no_location_sits_at_the_root() -> None:
    # A model-level validator can report "missing" with an empty location.
    # Killed by reading the missing name from the location regardless.
    class Contact(BaseModel):
        email: str | None = None
        phone: str | None = None

        @model_validator(mode="after")
        def _reachable(self) -> Contact:
            if not (self.email or self.phone):
                raise PydanticCustomError("missing", "email or phone is required")
            return self

    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate("{}", Contact.model_json_schema(), Contact)
    assert excinfo.value.error_message == "1 violation: #: email or phone is required"


def test_a_model_failure_that_cannot_be_described_keeps_pydantic_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pet(BaseModel):
        name: str

        @field_validator("name")
        @classmethod
        def _no_digits(cls, value: str) -> str:
            if any(ch.isdigit() for ch in value):
                raise ValueError("name must not contain digits")
            return value

    def undescribable(*_: Any) -> list[Any]:
        raise RuntimeError("no location")

    monkeypatch.setattr("openarmature.llm.providers.openai.model_violations", undescribable)
    with pytest.raises(StructuredOutputInvalid) as excinfo:
        _parse_and_validate('{"name": "rex9"}', Pet.model_json_schema(), Pet)
    assert excinfo.value.error_message.startswith("1 validation error for Pet")
    assert "name must not contain digits" in excinfo.value.error_message


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


def test_a_member_name_that_is_also_a_key_is_not_read_as_a_place() -> None:
    # Each union member's name here ("Cat", the tag "image_url", "int") is also
    # a key in the response. Killed by reading a segment as a place whenever the
    # key exists, and (for tags) by ignoring the input Pydantic reports.
    class Cat(BaseModel):
        meow: str

        @field_validator("meow")
        @classmethod
        def _no_digits(cls, value: str) -> str:
            if any(ch.isdigit() for ch in value):
                raise ValueError("meow must not contain digits")
            return value

    class Dog(BaseModel):
        bark: int

    class Url(BaseModel):
        url: str

    class ImagePart(BaseModel):
        type: Literal["image_url"]
        image_url: Url
        detail: str

        @field_validator("detail")
        @classmethod
        def _known_detail(cls, value: str) -> str:
            if value not in ("low", "high"):
                raise ValueError("detail must be low or high")
            return value

    class TextPart(BaseModel):
        type: Literal["text"]
        text: str

    class Owner(BaseModel):
        pet: Cat | Dog
        parts: list[Annotated[ImagePart | TextPart, Field(discriminator="type")]]
        tags: list[int | str]

    instance = {
        "pet": {"Cat": 0, "meow": "rex9"},
        "parts": [{"type": "image_url", "image_url": {"url": "u"}, "detail": "mid"}],
        "tags": [{"int": 2}],
    }
    with pytest.raises(ValidationError) as excinfo:
        Owner.model_validate(instance)
    assert describe(model_violations(excinfo.value, instance)) == (
        "5 violations: #/parts/0/detail: Value error, detail must be low or high; "
        "#/pet/bark: Field required; #/pet/meow: Value error, meow must not contain digits; "
        "#/tags/0: Input should be a valid integer; #/tags/0: Input should be a valid string"
    )


def test_a_tie_is_read_as_a_member_name_when_the_input_is_not_in_the_response() -> None:
    # A before-validator can hand Pydantic a value the response does not hold,
    # so no reading ends at the reported input, and both readings here pass
    # through two places. Killed by preferring the place reading on a tie.
    error = ValidationError.from_exception_data(
        "Owner", [{"type": "string_type", "loc": ("pet", "Cat", "meow"), "input": 5}]
    )
    instance = {"pet": {"Cat": 0, "meow": "x"}}
    assert describe(model_violations(error, instance)) == (
        "1 violation: #/pet/meow: Input should be a valid string"
    )


@pytest.mark.parametrize("failure", ["missing", "raising"])
def test_unevaluated_keywords_fall_back_to_the_parent_when_the_private_helpers_fail(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    # The per-key evaluation lives in a private jsonschema module. When it is
    # gone, or raises (as it does for a reference it cannot resolve), the
    # violation stays at the parent, whose message names each excluded property.
    if failure == "missing":
        monkeypatch.delattr(jsonschema_utils, "find_evaluated_property_keys_by_schema")
    else:

        def unresolvable(*_: Any) -> Any:
            raise LookupError("no such anchor")

        monkeypatch.setattr(jsonschema_utils, "find_evaluated_property_keys_by_schema", unresolvable)
    schema = {"type": "object", "properties": {"a": True}, "unevaluatedProperties": False}
    assert _described({"a": 1, "b": 2}, schema) == (
        "1 violation: #: fails unevaluatedProperties: "
        "Unevaluated properties are not allowed ('b' was unexpected)"
    )


@pytest.mark.parametrize(
    ("instance", "schema", "pointers"),
    [
        pytest.param(
            {"a": 1, "b": 2},
            {
                "$id": "https://example.com/root",
                "type": "object",
                "properties": {"a": True},
                "unevaluatedProperties": False,
            },
            ["#/b"],
            id="root-resource",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            {
                "type": "object",
                "properties": {"$id": {"type": "string"}, "a": True},
                "unevaluatedProperties": False,
            },
            ["#/b"],
            id="property-named-id",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            {
                "type": "object",
                "properties": {"a": True},
                "default": {"$dynamicRef": "#x"},
                "unevaluatedProperties": False,
            },
            ["#/b"],
            id="dynamic-reference-in-data",
        ),
        pytest.param(
            {"inner": {"a": 1, "b": 2}},
            {
                "type": "object",
                "properties": {
                    "inner": {
                        "$id": "https://example.com/inner",
                        "type": "object",
                        "properties": {"a": True},
                        "unevaluatedProperties": False,
                    }
                },
            },
            ["#/inner"],
            id="nested-resource",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            {
                "type": "object",
                "$defs": {"base": {"$dynamicAnchor": "base", "properties": {"a": True}}},
                "$dynamicRef": "#base",
                "unevaluatedProperties": False,
            },
            ["#"],
            id="dynamic-reference",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            {
                "$schema": "https://json-schema.org/draft/2019-09/schema",
                "type": "object",
                "properties": {"a": True},
                "unevaluatedProperties": False,
            },
            ["#"],
            id="draft-2019-09",
        ),
    ],
)
def test_unevaluated_keywords_are_placed_per_key_only_where_the_helpers_apply(
    instance: Any, schema: dict[str, Any], pointers: list[str]
) -> None:
    # The helpers resolve every reference from the root validator, and 2019-09
    # evaluates with helpers of its own. Each parent case is killed by dropping
    # its check, the root resource by also treating a root $id as nested, and the
    # name and data cases by reading every value as a schema.
    # Which keys jsonschema counts as unevaluated under $dynamicRef differs
    # between its releases, so only the placement is asserted.
    violations = schema_violations(instance, schema)
    assert [violation.pointer for violation in violations] == pointers
    assert all("unevaluatedProperties" in violation.description for violation in violations)


def test_the_private_jsonschema_helpers_still_exist() -> None:
    # A tripwire. The unevaluated keywords are located with these private
    # helpers, and without them the fallback above takes over in silence. This
    # fails on a jsonschema release that moves them, so losing the precision is
    # a decision.
    assert hasattr(jsonschema_utils, "find_evaluated_property_keys_by_schema")
    assert hasattr(jsonschema_utils, "find_evaluated_item_indexes_by_schema")
