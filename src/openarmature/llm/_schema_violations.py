"""Every violation a structured-output response commits against its schema."""

# llm-provider §7: a structured_output_invalid error_message describes every
# violation, each with its location as an RFC 6901 §6 URI-fragment JSON Pointer
# and the constraint that failed. jsonschema reports most violations where §7
# locates them; the ones it reports at the enclosing object or array are moved
# to their own location here.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import quote

from jsonschema import ValidationError
from jsonschema.exceptions import UnknownType
from jsonschema.protocols import Validator
from pydantic import ValidationError as ModelValidationError

from .provider import (
    _LIST_OF_SUBSCHEMA_KEYWORDS,
    _MAP_OF_SUBSCHEMA_KEYWORDS,
    _SINGLE_SUBSCHEMA_KEYWORDS,
    _UNRESOLVABLE,
    _resolve_ref,
    _validator_class,
)

Path = tuple[str | int, ...]


@dataclass(frozen=True)
class Violation:
    """One failing assertion: where it sits in the response, and what failed."""

    path: Path
    description: str

    @property
    def pointer(self) -> str:
        """The location as a JSON Pointer in URI-fragment form, such as ``#/site/lat``."""
        return "#" + "".join(f"/{_fragment_token(token)}" for token in self.path)


def schema_violations(instance: Any, schema: dict[str, Any]) -> list[Violation]:
    """Every violation ``instance`` commits against ``schema``, in a stable order."""
    prepared = cast("dict[str, Any]", _false_as_never(schema))
    validator = _validator_class(schema)(prepared)
    found: dict[tuple[Path, str], Violation] = {}
    for error in validator.iter_errors(instance):
        for violation in _violations(error, validator, prepared):
            found.setdefault((violation.path, violation.description), violation)
    return sorted(found.values(), key=_order)


def model_violations(exc: ModelValidationError, instance: Any) -> list[Violation]:
    """The violations a Pydantic model's own validation reported, at their places in ``instance``."""
    found: dict[tuple[Path, str], Violation] = {}
    for error in exc.errors():
        path = _instance_path(error["loc"], instance, missing=error["type"] == "missing")
        violation = Violation(path, error["msg"])
        found.setdefault((violation.path, violation.description), violation)
    return sorted(found.values(), key=_order)


# Pydantic's error location names a union member's type as well as the keys and
# indexes it passed through: ("pet", "Cat", "meow"), ("tags", 1, "int"). Only
# the keys and indexes are places in the response, plus the name of a missing
# property, which is where it would sit.
def _instance_path(loc: tuple[int | str, ...], instance: Any, *, missing: bool) -> Path:
    path: list[str | int] = []
    node: Any = instance
    for position, token in enumerate(loc):
        if isinstance(node, dict) and isinstance(token, str):
            mapping = cast("dict[str, Any]", node)
            if token in mapping or (missing and position == len(loc) - 1):
                path.append(token)
                node = mapping.get(token)
        elif isinstance(node, list) and isinstance(token, int):
            items = cast("list[Any]", node)
            if 0 <= token < len(items):
                path.append(token)
                node = items[token]
    return tuple(path)


def describe(violations: list[Violation]) -> str:
    """One line naming every violation, for ``StructuredOutputInvalid.error_message``."""
    count = f"{len(violations)} violation{'' if len(violations) == 1 else 's'}"
    return f"{count}: " + "; ".join(f"{v.pointer}: {v.description}" for v in violations)


# A false subschema rejects everything, as {"not": {}} does. jsonschema reports a
# false additionalProperties or items once, at the parent, but descends into a
# schema-valued one and reports each excluded property or element at its own
# location, which is where §7 places them. Only subschema positions are
# rewritten: a false const, enum entry or uniqueItems is data. The unevaluated
# keywords are left alone, since jsonschema reports them at the parent either way.
_SINGLE = (_SINGLE_SUBSCHEMA_KEYWORDS - {"unevaluatedItems", "unevaluatedProperties"}) | {"additionalItems"}
_MAP = _MAP_OF_SUBSCHEMA_KEYWORDS | {"dependencies"}


def _false_as_never(node: Any) -> Any:
    if node is False:
        return {"not": {}}
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in cast("dict[str, Any]", node).items():
        if key in _SINGLE and isinstance(value, dict | bool):
            out[key] = _false_as_never(value)
        elif key == "items" and isinstance(value, list):
            out[key] = [_false_as_never(entry) for entry in cast("list[Any]", value)]
        elif key in _MAP and isinstance(value, dict):
            out[key] = {
                name: entry if isinstance(entry, list) else _false_as_never(entry)
                for name, entry in cast("dict[str, Any]", value).items()
            }
        elif key in _LIST_OF_SUBSCHEMA_KEYWORDS and isinstance(value, list):
            out[key] = [_false_as_never(entry) for entry in cast("list[Any]", value)]
        else:
            out[key] = value
    return out


def _violations(error: ValidationError, validator: Validator, root: dict[str, Any]) -> Iterator[Violation]:
    path: Path = tuple(error.absolute_path)
    keyword = error.validator
    instance: Any = error.instance
    if "propertyNames" in error.absolute_schema_path:
        # A name has no location of its own, so the object it names carries it.
        yield Violation(path, f"property name {error.instance!r} fails {keyword}: {error.message}")
    elif keyword in ("anyOf", "oneOf"):
        yield from _union_violations(error, validator, root, path)
    elif keyword == "required":
        yield from _missing(
            path, error.instance, cast("list[Any]", error.validator_value), "required property is missing"
        )
    elif keyword in ("dependentRequired", "dependencies"):
        if isinstance(instance, dict):
            present = cast("dict[str, Any]", instance)
            for name, needed in cast("dict[str, Any]", error.validator_value).items():
                if name in present and isinstance(needed, list):
                    needed_names = cast("list[Any]", needed)
                    yield from _missing(
                        path, present, needed_names, f"property required by {name!r} is missing"
                    )
    elif keyword in ("unevaluatedProperties", "unevaluatedItems"):
        yield from _unevaluated(error, validator, path)
    elif keyword == "not" and error.validator_value == {}:
        yield Violation(path, _excluded(error))
    elif keyword == "type":
        expected = _type_names(error.validator_value)
        yield Violation(path, f"expected type {expected}, got {_json_type(error.instance)}")
    else:
        yield Violation(path, f"fails {keyword}{_bound(error.validator_value)}: {error.message}")


def _union_violations(
    error: ValidationError, validator: Validator, root: dict[str, Any], path: Path
) -> Iterator[Violation]:
    if not error.context:
        # A oneOf more than one member satisfies: the composite's own constraint.
        yield Violation(path, f"matches more than one {error.validator} member")
        return
    members = cast("list[Any]", error.validator_value)
    admitted = [
        index for index, member in enumerate(members) if _admits(validator, root, member, error.instance)
    ]
    if not admitted:
        names: list[str] = []
        for member in members:
            names.extend(name for name in _own_types(root, member) or () if name not in names)
        yield Violation(path, f"matches none of the {error.validator} member types ({', '.join(names)})")
        return
    by_member: dict[Any, list[ValidationError]] = defaultdict(list)
    for sub_error in error.context:
        by_member[sub_error.relative_schema_path[0]].append(sub_error)
    for index in admitted:
        for sub_error in by_member[index]:
            yield from _violations(sub_error, validator, root)


def _admits(validator: Validator, root: dict[str, Any], member: Any, instance: Any) -> bool:
    types = _own_types(root, member)
    if types is None:
        return True
    try:
        return any(validator.is_type(instance, name) for name in types)
    except UnknownType:
        return True


def _own_types(
    root: dict[str, Any], member: Any, seen: frozenset[str] = frozenset()
) -> tuple[str, ...] | None:
    """A union member's own ``type``, after resolving references, or None when it has none."""
    if not isinstance(member, dict):
        return None
    member_dict = cast("dict[str, Any]", member)
    declared = member_dict.get("type")
    if isinstance(declared, str):
        return (declared,)
    if isinstance(declared, list):
        return tuple(name for name in cast("list[Any]", declared) if isinstance(name, str))
    ref = member_dict.get("$ref")
    if isinstance(ref, str) and ref not in seen:
        target = _resolve_ref(ref, root)
        if target is not _UNRESOLVABLE:
            return _own_types(root, target, seen | {ref})
    return None


def _missing(path: Path, instance: Any, names: list[Any], description: str) -> Iterator[Violation]:
    if isinstance(instance, dict):
        for name in names:
            if name not in instance:
                yield Violation((*path, name), description)


# jsonschema reports unevaluatedProperties / unevaluatedItems: false once, at the
# parent, and keeps the per-key evaluation in a private module. It is used here to
# place each excluded property or element at its own location. If it moves, the
# violation stays at the parent, whose message still names each one.
def _unevaluated(error: ValidationError, validator: Validator, path: Path) -> Iterator[Violation]:
    instance: Any = error.instance
    if error.validator_value is False:
        try:
            from jsonschema import _utils  # noqa: PLC0415

            jsonschema_utils = cast("Any", _utils)
            if error.validator == "unevaluatedProperties" and isinstance(instance, dict):
                keys = cast("dict[str, Any]", instance)
                evaluated = set(
                    jsonschema_utils.find_evaluated_property_keys_by_schema(validator, keys, error.schema)
                )
                excluded: list[str | int] = [key for key in keys if key not in evaluated]
                description = "property is not allowed (unevaluatedProperties: false)"
            elif error.validator == "unevaluatedItems" and isinstance(instance, list):
                items = cast("list[Any]", instance)
                evaluated_indexes = set(
                    jsonschema_utils.find_evaluated_item_indexes_by_schema(validator, items, error.schema)
                )
                excluded = [index for index in range(len(items)) if index not in evaluated_indexes]
                description = "item is not allowed (unevaluatedItems: false)"
            else:
                excluded = []
                description = ""
        except (AttributeError, ImportError, TypeError):
            excluded = []
            description = ""
        if excluded:
            for token in excluded:
                yield Violation((*path, token), description)
            return
    yield Violation(path, f"fails {error.validator}: {error.message}")


def _excluded(error: ValidationError) -> str:
    schema_path = list(error.absolute_schema_path)
    parent = schema_path[-2] if len(schema_path) >= 2 else None
    if parent == "additionalProperties":
        return "property is not allowed (additionalProperties: false)"
    if parent in ("items", "additionalItems"):
        return f"item is not allowed ({parent}: false)"
    return "value is not allowed (false schema)"


def _type_names(value: Any) -> str:
    if isinstance(value, list):
        return " or ".join(str(name) for name in cast("list[Any]", value))
    return str(value)


def _json_type(instance: Any) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, dict):
        return "object"
    return type(instance).__name__


def _bound(value: Any) -> str:
    # The keyword's own value, where it is a bound (minItems 2, pattern '^a').
    # jsonschema's message for a count rule does not state it.
    if isinstance(value, bool):
        return ""
    if isinstance(value, int | float):
        return f" {value}"
    if isinstance(value, str):
        return f" {value!r}"
    return ""


# RFC 6901 §3 escapes "~" and "/" inside a reference token; §6 then percent-encodes
# the pointer for use as a URI fragment.
def _fragment_token(token: str | int) -> str:
    return quote(str(token).replace("~", "~0").replace("/", "~1"), safe="!$&'()*+,;=:@")


def _order(violation: Violation) -> tuple[Any, ...]:
    # Array indexes sort numerically and before names, so #/rows/2 precedes #/rows/10.
    path_key = tuple((0, token) if isinstance(token, int) else (1, str(token)) for token in violation.path)
    return (path_key, violation.description)
