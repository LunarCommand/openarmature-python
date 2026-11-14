# Spec: helpers shared by the llm-provider §8.X wire mappings: the §6
# response_schema normalization, the §5/§7 parse-and-validate step behind
# structured_output_invalid, the §8.1.5.1 schema directive, and the §8
# wire-byte canonicalizer.

"""Helpers shared by the bundled LLM providers."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import httpx
from pydantic import BaseModel, ValidationError

from .._event_fields import _EventRequest
from .._schema_violations import _clip, describe, model_violations, response_violations
from ..errors import LlmProviderError, ProviderInvalidRequest, StructuredOutputInvalid
from ..messages import Message, SystemMessage
from ..provider import validate_response_schema
from ..response import ParsedValue


# Normalize a response_schema argument to a dict (plus the optional
# BaseModel subclass form for the post-parse instance return). Accepts
# either form per the Provider Protocol; raises ProviderInvalidRequest
# on invalid shapes (non-dict, non-object-top-level for the dict form;
# pre-validated by validate_response_schema).
def _normalize_response_schema(
    response_schema: dict[str, Any] | type[BaseModel] | None,
) -> tuple[dict[str, Any] | None, type[BaseModel] | None]:
    if response_schema is None:
        return None, None
    if isinstance(response_schema, type):
        # Defensive runtime check: the Protocol signature accepts
        # type[BaseModel], but Python doesn't enforce that at the call
        # boundary. Reject non-BaseModel classes with a canonical error
        # instead of letting AttributeError leak from model_json_schema.
        if not issubclass(response_schema, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ProviderInvalidRequest(
                f"response_schema: class form MUST be a Pydantic BaseModel subclass "
                f"(got {response_schema.__name__})"
            )
        schema_dict = response_schema.model_json_schema()
        validate_response_schema(schema_dict)
        return schema_dict, response_schema
    validate_response_schema(response_schema)
    return response_schema, None


# Raised before any of the request is written: no connection opened, none came
# free in the pool, the proxy refused the tunnel, or no transport serves the
# URL's scheme.
_UNSENT_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)


# graph-engine §6: a request that was never sent reports no path. A failure
# after the connection opened keeps it, since the provider may have received
# some or all of the request.
def _as_sent(request: _EventRequest, exc: LlmProviderError | None) -> _EventRequest:
    if exc is not None and isinstance(exc.__cause__, _UNSENT_TRANSPORT_ERRORS):
        return replace(request, structured_output_path=None)
    return request


# observability §5.5.7: an attempt on the native path that finished normally,
# with no stop sequence to end it early, returned content that is not a JSON
# object. A constraint that bound cannot produce that, since llm-provider §5
# requires an object at the schema's top level.
def _schema_not_honored(request: _EventRequest, exc: LlmProviderError | None, *, stop_sent: bool) -> bool:
    return (
        isinstance(exc, StructuredOutputInvalid)
        and request.structured_output_path == "native"
        and exc.finish_reason == "stop"
        and not stop_sent
        and not _is_json_object(exc.output_content)
    )


# Read as the validation reads it, so NaN, a number past a float's range or
# nesting too deep to parse is malformed here too.
def _is_json_object(content: str) -> bool:
    try:
        return isinstance(_load_json(content), dict)
    except (ValueError, RecursionError):
        return False


def _load_json(content: str) -> Any:
    return json.loads(content, parse_constant=_reject_non_json_constant, parse_float=_finite_float)


# Parse the model's content string as JSON, then validate it against the
# schema. The dict-schema path returns the dict; the BaseModel-class path then
# runs the class's own validation, which produces the instance.
def _parse_and_validate(
    content: str,
    schema_dict: dict[str, Any],
    schema_class: type[BaseModel] | None,
) -> ParsedValue:
    try:
        loaded = _load_json(content)
    except (ValueError, RecursionError) as exc:
        raise StructuredOutputInvalid(
            "response content is not valid JSON",
            response_schema=schema_dict,
            output_content=content,
            error_message=_unparsed(exc),
        ) from exc
    try:
        violations = response_violations(loaded, schema_dict)
        error_message = describe(violations) if violations else ""
    except Exception as exc:
        # llm-provider §7: validation that cannot finish is still a
        # structured_output_invalid. It raises for, say, an unknown draft-03
        # type, content nested past the recursion limit, or a reference
        # validate_response_schema does not walk.
        raise StructuredOutputInvalid(
            "response could not be validated against the supplied schema",
            response_schema=schema_dict,
            output_content=content,
            error_message=f"could not be validated: {_clip(str(exc))}",
        ) from exc
    if violations:
        raise StructuredOutputInvalid(
            "response failed JSON Schema validation",
            response_schema=schema_dict,
            output_content=content,
            error_message=error_message,
        )
    # response_violations rejects content that is not an object.
    parsed_dict = cast("dict[str, Any]", loaded)
    if schema_class is None:
        return parsed_dict
    # The JSON Schema check above runs first so the class path is as strict
    # as the dict path: Pydantic's default model_validate is coercive (it
    # accepts "30" for an int field).
    try:
        return schema_class.model_validate(parsed_dict)
    except ValidationError as exc:
        raise StructuredOutputInvalid(
            "response failed Pydantic validation",
            response_schema=schema_dict,
            output_content=content,
            error_message=_model_failure(exc, parsed_dict),
        ) from exc


def _model_failure(exc: ValidationError, instance: dict[str, Any]) -> str:
    try:
        return describe(model_violations(exc, instance))
    except Exception:
        # Pydantic's own text still names each failure, without a pointer.
        return str(exc)


# llm-provider §7: content that cannot be parsed is one violation with no
# location. RFC 8259 §6 and §9 let a parser limit the range of numbers and the
# depth of nesting, and content past those limits is reported the same way.
def _unparsed(exc: ValueError | RecursionError) -> str:
    if isinstance(exc, RecursionError):
        return "JSON nested too deeply to parse"
    if isinstance(exc, _OutOfRange):
        return f"number out of range: {exc}"
    return f"not well-formed JSON: {exc}"


def _reject_non_json_constant(name: str) -> Any:
    # Python's parser accepts NaN and Infinity; RFC 8259 has no literals for them.
    raise ValueError(f"{name} is not a JSON value")


class _OutOfRange(ValueError):
    pass


def _finite_float(text: str) -> float:
    # Python reads a number past a float's range, such as 1e400, as infinity.
    value = float(text)
    if not math.isfinite(value):
        raise _OutOfRange(text)
    return value


_SCHEMA_DIRECTIVE_TEMPLATE = (
    "You MUST return only valid JSON that conforms to the following JSON Schema. "
    "Do not include prose, markdown fences, or any text outside the JSON object.\n\n"
    "JSON Schema:\n{schema_json}"
)


# Construct a fresh message list with a schema directive added. The
# directive is appended to the existing system message's content when
# present, or prepended as a new system message otherwise. The caller's
# original list is never mutated; Message instances are reused, and
# this helper does not modify them in place (the message models are
# not frozen Pydantic models, so the safety is structural, not
# enforced by the type). The serialized schema appears verbatim in
# the directive so callers that need to verify the directive
# references the schema (conformance harnesses, observability spans)
# can substring-match the canonical JSON form.
def _augment_messages_with_schema_directive(
    messages: Sequence[Message],
    schema_dict: dict[str, Any],
) -> list[Message]:
    directive = _SCHEMA_DIRECTIVE_TEMPLATE.format(schema_json=json.dumps(schema_dict, sort_keys=True))
    out: list[Message] = list(messages)
    if out and isinstance(out[0], SystemMessage):
        existing = out[0]
        merged = SystemMessage(content=f"{existing.content}\n\n{directive}")
        out[0] = merged
    else:
        out.insert(0, SystemMessage(content=directive))
    return out


# 0047 §8 wire-byte stability canonicalizer. Sorts dict keys at every nesting
# level and preserves list ORDER, which is caller-supplied; object keys inside
# arrays still sort via the dict branch. Applied at every user-supplied-dict
# boundary so equivalent inputs produce byte-identical wire output, which is
# what makes prompt-cache hits reliable.
#
# No depth cap of its own: a pathologically nested dict raises RecursionError
# here rather than producing broken wire bytes downstream.
#
# Byte-stability rests on dict insertion order (PEP 468) and on httpx
# serializing through the stdlib ``json.dumps``, which respects it. If httpx
# ever randomized key order the wire-byte tests in
# ``tests/unit/test_llm_provider.py`` would fail loudly.
def _canonicalize_dict_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize_dict_keys(value[k]) for k in sorted(cast("dict[str, Any]", value))}
    if isinstance(value, list):
        return [_canonicalize_dict_keys(v) for v in cast("list[Any]", value)]
    return value


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value to a float seconds count.

    HTTP allows seconds-int OR HTTP-date; this implementation handles
    the seconds-int form (the OpenAI/vendor norm) and ignores
    HTTP-date.

    Reusable by third-party Provider implementations that need to
    surface ``Retry-After`` to ``ProviderRateLimit.retry_after``.
    """
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


__all__ = [
    "_as_sent",
    "_augment_messages_with_schema_directive",
    "_canonicalize_dict_keys",
    "_normalize_response_schema",
    "_parse_and_validate",
    "_schema_not_honored",
    "parse_retry_after",
]
