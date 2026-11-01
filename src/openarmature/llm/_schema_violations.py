"""Every violation a structured-output response commits against its schema."""

# llm-provider §7: a structured_output_invalid error_message describes every
# violation, each with its location as an RFC 6901 §6 URI-fragment JSON Pointer
# and the constraint that failed. jsonschema reports most violations where §7
# locates them; the ones it reports at the enclosing object or array are moved
# to their own location here.

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import quote

from jsonschema import (
    Draft3Validator,
    Draft4Validator,
    Draft6Validator,
    Draft7Validator,
    Draft202012Validator,
    ValidationError,
)
from jsonschema.exceptions import UnknownType
from jsonschema.protocols import Validator
from pydantic import ValidationError as ModelValidationError
from referencing import Registry

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
    validator_class = _validator_class(schema)
    prepared = schema
    # Draft-03 has no `not`, so its false subschemas stay as written, and
    # _extras places what a false additionalProperties or additionalItems excludes.
    if "not" in validator_class.VALIDATORS:
        prepared = cast("dict[str, Any]", _false_as_never(schema))
    # jsonschema's default registry fetches a remote $ref over the network, and
    # validate_response_schema does not reach every position a $ref can sit in.
    validator = validator_class(prepared, registry=Registry())
    expanded: set[tuple[Path, Path]] = set()
    found: dict[tuple[Path, str], Violation] = {}
    for error in validator.iter_errors(instance):
        if _repeats(error, expanded):
            continue
        # Every error rejects the content, so one that none of the cases below
        # recognize is reported as jsonschema states it.
        violations = list(_violations(error, validator, prepared, expanded))
        for violation in violations or [_as_stated(error, tuple(error.absolute_path))]:
            found.setdefault((violation.path, violation.description), violation)
    return sorted(found.values(), key=_order)


# jsonschema reports a missing property once per name, and every one of those
# reports carries the whole list, so each list is expanded only the first time.
def _repeats(error: ValidationError, expanded: set[tuple[Path, Path]]) -> bool:
    if error.validator not in ("required", "dependentRequired", "dependencies"):
        return False
    key = (tuple(error.absolute_path), tuple(error.absolute_schema_path))
    if key in expanded:
        return True
    expanded.add(key)
    return False


def response_violations(instance: Any, schema: dict[str, Any]) -> list[Violation]:
    """Every violation ``instance`` commits as a structured-output response, which is always an object."""
    violations = schema_violations(instance, schema)
    if isinstance(instance, dict):
        return violations
    # llm-provider §5: the response is an object. Drafts 3 to 7 ignore a root
    # $ref's siblings, so the schema's own root type may never have applied.
    not_an_object = Violation((), f"expected type object, got {_json_type(instance)}")
    return sorted({*violations, not_an_object}, key=_order)


def model_violations(exc: ModelValidationError, instance: Any) -> list[Violation]:
    """The violations a Pydantic model's own validation reported, at their places in ``instance``."""
    found: dict[tuple[Path, str], Violation] = {}
    for error in exc.errors():
        # A model-level validator can report a missing value with no location.
        missing = error["type"] == "missing" and bool(error["loc"])
        path = _instance_path(error["loc"], instance, error["input"], missing=missing)
        violation = Violation(path, error["msg"])
        found.setdefault((violation.path, violation.description), violation)
    return sorted(found.values(), key=_order)


# Pydantic's error location names a union member as well as the keys and indexes
# it passed through: ("pet", "Cat", "meow"), ("tags", 1, "int"). A member's name
# can also be a key in the response, so each segment is read both ways, and the
# reading that ends at the object Pydantic reports as the input is the place.
# Failing that, the reading through more places wins, and on a tie the segment
# is read as a member's name. A missing property sits where it would be.
def _instance_path(loc: tuple[int | str, ...], instance: Any, reported: Any, *, missing: bool) -> Path:
    places = loc[:-1] if missing else loc
    best: dict[tuple[int, int], tuple[tuple[bool, int], Path]] = {}

    def read(position: int, node: Any) -> tuple[tuple[bool, int], Path]:
        if position == len(places):
            return (node is reported, 0), ()
        key = (position, id(node))
        if key not in best:
            score, path = read(position + 1, node)
            place, child = _child(node, places[position])
            if child is not _ABSENT:
                (found, count), rest = read(position + 1, child)
                if (found, count + 1) > score:
                    score, path = (found, count + 1), (place, *rest)
            best[key] = (score, path)
        return best[key]

    path = read(0, instance)[1]
    return (*path, loc[-1]) if missing else path


_ABSENT = object()


def _child(node: Any, token: int | str) -> tuple[int | str, Any]:
    if isinstance(node, dict) and isinstance(token, str):
        entries = cast("dict[str, Any]", node)
        if token in entries:
            return token, entries[token]
        if "\ufffd" in token:
            # Pydantic writes a lone surrogate in a key as U+FFFD, one per byte.
            for name, value in entries.items():
                if name.encode("utf-8", "surrogatepass").decode("utf-8", "replace") == token:
                    return name, value
    if isinstance(node, list) and isinstance(token, int):
        items = cast("list[Any]", node)
        if 0 <= token < len(items):
            return token, items[token]
    return token, _ABSENT


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


# A schema path also names each property, pattern and dependency it passes
# through, and a property can be called items or propertyNames. The token after
# a keyword in _MAP is such a name, so it is skipped. A list index stays, so a
# false entry in a list of subschemas is not read as its keyword being false.
def _keywords(schema_path: Iterable[str | int]) -> list[str | int]:
    keywords: list[str | int] = []
    tokens = iter(schema_path)
    for token in tokens:
        keywords.append(token)
        if token in _MAP:
            next(tokens, None)
    return keywords


def _violations(
    error: ValidationError, validator: Validator, root: dict[str, Any], expanded: set[tuple[Path, Path]]
) -> Iterator[Violation]:
    path: Path = tuple(error.absolute_path)
    keyword = error.validator
    instance: Any = error.instance
    under_property_names = "propertyNames" in _keywords(error.absolute_schema_path)
    subject = ""
    if under_property_names:
        # llm-provider §7: a property the schema does not allow sits at its own
        # location, and the description says it is the name that failed.
        path = (*path, instance)
        subject = f"property name {instance!r} "
    if keyword in ("anyOf", "oneOf"):
        yield from _union_violations(error, validator, root, path, subject, expanded)
    elif keyword == "not" and error.validator_value == {}:
        yield Violation(path, _excluded(error, subject))
    elif under_property_names:
        yield _as_stated(error, path, subject)
    elif keyword in ("additionalProperties", "additionalItems") and error.validator_value is False:
        yield from _extras(error, path)
    elif keyword == "required":
        yield from _missing(path, error.instance, error.validator_value, "required property is missing")
    elif keyword in ("dependentRequired", "dependencies"):
        if isinstance(instance, dict):
            present = cast("dict[str, Any]", instance)
            for name, needed in cast("dict[str, Any]", error.validator_value).items():
                if name in present:
                    # Draft-03 allows a single name in place of a list.
                    names = [needed] if isinstance(needed, str) else needed
                    yield from _missing(path, present, names, f"property required by {name!r} is missing")
    elif keyword in ("unevaluatedProperties", "unevaluatedItems"):
        yield from _unevaluated(error, validator, root, path)
    elif keyword == "type":
        expected = _type_names(error.validator_value)
        yield Violation(path, f"expected type {expected}, got {_json_type(error.instance)}")
    else:
        yield _as_stated(error, path)


def _as_stated(error: ValidationError, path: Path, subject: str = "") -> Violation:
    return Violation(
        path, f"{subject}fails {error.validator}{_bound(error.validator_value)}: {_clip(error.message)}"
    )


# Draft-03 has no `not`, so a false additionalProperties or additionalItems stays
# as written, and jsonschema reports everything it excludes once, at the parent.
def _extras(error: ValidationError, path: Path) -> Iterator[Violation]:
    instance: Any = error.instance
    schema = cast("dict[str, Any]", error.schema)
    if error.validator == "additionalProperties" and isinstance(instance, dict):
        declared = cast("dict[str, Any]", schema.get("properties", {}))
        patterns = cast("dict[str, Any]", schema.get("patternProperties", {}))
        for key in cast("dict[str, Any]", instance):
            if key not in declared and not any(re.search(pattern, key) for pattern in patterns):
                yield Violation((*path, key), "property is not allowed (additionalProperties: false)")
    elif error.validator == "additionalItems" and isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, list):
            for index in range(len(cast("list[Any]", items)), len(cast("list[Any]", instance))):
                yield Violation((*path, index), "item is not allowed (additionalItems: false)")


def _union_violations(
    error: ValidationError,
    validator: Validator,
    root: dict[str, Any],
    path: Path,
    subject: str,
    expanded: set[tuple[Path, Path]],
) -> Iterator[Violation]:
    if not error.context:
        # A oneOf more than one member satisfies: the composite's own constraint.
        yield Violation(path, f"{subject}matches more than one {error.validator} member")
        return
    members = cast("list[Any]", error.validator_value)
    admitted = [
        index for index, member in enumerate(members) if _admits(validator, root, member, error.instance)
    ]
    if not admitted:
        names: list[str] = []
        for member in members:
            names.extend(name for name in _own_types(validator, root, member) or () if name not in names)
        yield Violation(
            path, f"{subject}matches none of the {error.validator} member types ({', '.join(names)})"
        )
        return
    by_member: dict[Any, list[ValidationError]] = defaultdict(list)
    for sub_error in error.context:
        by_member[sub_error.relative_schema_path[0]].append(sub_error)
    for index in admitted:
        for sub_error in by_member[index]:
            if not _repeats(sub_error, expanded):
                yield from _violations(sub_error, validator, root, expanded)


def _admits(validator: Validator, root: dict[str, Any], member: Any, instance: Any) -> bool:
    types = _own_types(validator, root, member)
    if types is None:
        return True
    try:
        return any(validator.is_type(instance, name) for name in types)
    except UnknownType:
        return True


# Drafts 3 to 7 apply a $ref alone and ignore every keyword beside it.
_REF_ALONE = (Draft3Validator, Draft4Validator, Draft6Validator, Draft7Validator)


def _own_types(
    validator: Validator, root: dict[str, Any], member: Any, seen: frozenset[str] = frozenset()
) -> tuple[str, ...] | None:
    """A union member's own ``type``, after resolving references, or None when it has none."""
    if not isinstance(member, dict):
        return None
    member_dict = cast("dict[str, Any]", member)
    ref = member_dict.get("$ref")
    if ref is None or not isinstance(validator, _REF_ALONE):
        declared = member_dict.get("type")
        if isinstance(declared, str):
            return (declared,)
        if isinstance(declared, list):
            return tuple(name for name in cast("list[Any]", declared) if isinstance(name, str))
    if isinstance(ref, str) and ref not in seen:
        target = _resolve_ref(ref, root)
        if target is not _UNRESOLVABLE:
            return _own_types(validator, root, target, seen | {ref})
    return None


def _missing(path: Path, instance: Any, names: Any, description: str) -> Iterator[Violation]:
    # Draft-03's required is a boolean on the property's own schema, so it yields
    # nothing here and is reported as jsonschema states it, at the property.
    if isinstance(instance, dict) and isinstance(names, list):
        for name in cast("list[Any]", names):
            if name not in instance:
                yield Violation((*path, name), description)


# jsonschema reports unevaluatedProperties / unevaluatedItems: false once, at the
# parent, and keeps the per-key evaluation in a private module. It is used here to
# place each excluded property or element at its own location, in a 2020-12 schema
# whose references all resolve from the root: the helpers are handed the root
# validator, and 2019-09 evaluates with different ones. Otherwise, or if the
# helpers fail, the violation stays at the parent, whose message names each one.
def _unevaluated(
    error: ValidationError, validator: Validator, root: dict[str, Any], path: Path
) -> Iterator[Violation]:
    instance: Any = error.instance
    excluded: list[str | int] = []
    description = ""
    try:
        if (
            error.validator_value is False
            and isinstance(validator, Draft202012Validator)
            and not _has_nested_scope(root)
        ):
            from jsonschema import _utils  # noqa: PLC0415

            jsonschema_utils = cast("Any", _utils)
            if error.validator == "unevaluatedProperties" and isinstance(instance, dict):
                keys = cast("dict[str, Any]", instance)
                evaluated = set(
                    jsonschema_utils.find_evaluated_property_keys_by_schema(validator, keys, error.schema)
                )
                excluded = [key for key in keys if key not in evaluated]
                description = "property is not allowed (unevaluatedProperties: false)"
            elif error.validator == "unevaluatedItems" and isinstance(instance, list):
                items = cast("list[Any]", instance)
                evaluated_indexes = set(
                    jsonschema_utils.find_evaluated_item_indexes_by_schema(validator, items, error.schema)
                )
                excluded = [index for index in range(len(items)) if index not in evaluated_indexes]
                description = "item is not allowed (unevaluatedItems: false)"
    except Exception:
        excluded = []
    if excluded:
        for token in excluded:
            yield Violation((*path, token), description)
        return
    yield _as_stated(error, path)


# A nested $id starts a resource whose references resolve against it, and a
# dynamic reference depends on the scopes it was reached through, so the root
# validator can follow neither.
def _has_nested_scope(node: Any, *, nested: bool = False) -> bool:
    if isinstance(node, dict):
        entries = cast("dict[str, Any]", node)
        if (nested and "$id" in entries) or "$dynamicRef" in entries:
            return True
        return any(_has_nested_scope(value, nested=True) for value in entries.values())
    if isinstance(node, list):
        return any(_has_nested_scope(entry, nested=True) for entry in cast("list[Any]", node))
    return False


def _excluded(error: ValidationError, subject: str) -> str:
    keywords = _keywords(error.absolute_schema_path)
    parent = keywords[-2] if len(keywords) >= 2 else None
    if subject:
        reason = "propertyNames: false" if parent == "propertyNames" else "false schema"
        return f"{subject}is not allowed ({reason})"
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


# jsonschema's messages repr the content they reject and the keyword's value in
# full, so a large array or enum would repeat whole in every violation. The
# location and the constraint come before the message and are never cut.
_MESSAGE_LIMIT = 200


def _clip(text: str) -> str:
    return text if len(text) <= _MESSAGE_LIMIT else text[: _MESSAGE_LIMIT - 3] + "..."


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
# what RFC 3986's fragment rule does not allow. A lone surrogate, which JSON text
# can carry, is encoded as its surrogate bytes, so that every key has a pointer.
def _fragment_token(token: str | int) -> str:
    return quote(
        str(token).replace("~", "~0").replace("/", "~1"), safe="!$&'()*+,;=:@?", errors="surrogatepass"
    )


def _order(violation: Violation) -> tuple[Any, ...]:
    # Array indexes sort numerically and before names, so #/rows/2 precedes #/rows/10.
    path_key = tuple((0, token) if isinstance(token, int) else (1, str(token)) for token in violation.path)
    return (path_key, violation.description)
