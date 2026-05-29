# Spec: realizes observability §3.4 + §5.6 (proposal 0034).
# Caller-supplied invocation metadata is held in a ContextVar per
# §3.4's "MUST propagate via the language's idiomatic context
# primitive"; per-async-context copy-on-write semantics give fan-out
# instances and parallel branches independent metadata views without
# explicit threading. Validation lives here so the same rules apply
# at the ``invoke()`` boundary and at mid-invocation augmentation
# via ``set_invocation_metadata``.

"""Caller-supplied invocation metadata (proposal 0034).

Two surfaces:

- :func:`current_invocation_metadata` — public reader; returns the
  metadata mapping in scope for the current async context, or the
  empty mapping outside any invocation.
- :func:`set_invocation_metadata` — public augmentation helper.
  Merges the supplied entries into the current context's metadata
  (additive; existing keys are overwritten). Affects observations /
  spans emitted AFTER the call returns.

Plus the engine-internal lifecycle helpers (``_set_invocation_metadata`` /
``_reset_invocation_metadata``) that ``CompiledGraph.invoke`` drives
around the outermost call.

Validation rules (apply at every entry point):

- Keys MUST be strings.
- Keys MUST NOT start with ``openarmature.`` or ``gen_ai.`` (reserved
  for spec-normative attribute namespaces; collisions would silently
  overwrite OA-emitted state at the observer layer).
- Keys MUST NOT exactly match a reserved OA-emitted top-level metadata
  key name (the §8.4 Langfuse set plus ``invocation_id``; proposal
  0041) for the same collision reason.
- Values MUST be OTel-attribute-compatible scalars: ``str``, ``int``,
  ``float``, ``bool``, or a homogeneous list/tuple of those types.
  ``None``, nested objects, and mixed-type arrays are rejected.

All boundary violations raise :class:`ValueError` with a message
naming the offending key.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Any, cast

# OTel-compatible attribute value type per spec §3.4 + OTel's
# AnyValue contract. Homogeneous arrays only; the validator enforces
# the homogeneity at runtime.
AttributeValue = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]

# Sentinel empty mapping that the ContextVar default holds. Using a
# read-only proxy keeps the "no metadata in scope" branch
# allocation-free across calls and prevents accidental mutation of
# the default by any caller that holds the reference.
_EMPTY_METADATA: MappingProxyType[str, AttributeValue] = MappingProxyType({})

_invocation_metadata_var: ContextVar[MappingProxyType[str, AttributeValue]] = ContextVar(
    "openarmature.invocation_metadata", default=_EMPTY_METADATA
)

# Reserved key prefixes per §3.4. Keys with these prefixes are
# off-limits to caller-supplied metadata; the engine rejects at the
# boundary so observers never see a colliding key.
_RESERVED_PREFIXES: tuple[str, ...] = ("openarmature.", "gen_ai.")

# Reserved exact key NAMES per §3.4 (proposal 0041): the top-level
# metadata keys an OA-emitted §8 backend mapping writes alongside
# caller keys (the §8.4 Langfuse set, plus invocation_id). A caller
# key matching one exactly would silently overwrite an OA field in a
# backend's flat top-level metadata, so it is rejected at the boundary
# the same way as the prefix reservation. Backend-set-independent:
# rejected regardless of which observers are attached.
_RESERVED_KEY_NAMES: frozenset[str] = frozenset(
    {
        "correlation_id",
        "entry_node",
        "spec_version",
        "detached_child_trace_ids",
        "namespace",
        "step",
        "attempt_index",
        "fan_out_index",
        "subgraph_name",
        "fan_out_item_count",
        "fan_out_concurrency",
        "fan_out_error_policy",
        "fan_out_parent_node_name",
        "prompt_group_name",
        "request_extras",
        "finish_reason",
        "system",
        "response_model",
        "response_id",
        "prompt",
        "invocation_id",
    }
)


def current_invocation_metadata() -> MappingProxyType[str, AttributeValue]:
    """Return the caller-supplied invocation metadata in scope, or the
    empty mapping outside any invocation.

    Observers and capability code (LLM provider span hook, Langfuse
    observer, OTel observer) read this to surface the mapping on
    backend-specific records. The returned mapping is read-only;
    callers MUST NOT mutate it. Use :func:`set_invocation_metadata`
    to add entries.
    """
    return _invocation_metadata_var.get()


def set_invocation_metadata(**entries: AttributeValue) -> None:
    """Merge ``entries`` into the current async context's invocation
    metadata. Additive: existing keys with the same names are
    overwritten; other keys are preserved.

    Per spec §3.4: affects spans / observations emitted AFTER the
    call returns; spans already closed are NOT retroactively updated.
    Implementations MAY update open root-level surfaces (e.g., the
    Langfuse Trace's metadata) where the backend SDK supports it;
    Langfuse's ``trace.update`` is the canonical example. The
    framework's helper here just maintains the ContextVar; per-
    backend update propagation is the observer's concern.

    Raises :class:`ValueError` if any key violates the reserved-
    namespace rule or any value is not OTel-attribute-compatible.

    Outside any active invocation, this still updates the
    ContextVar (a fresh per-context override), but the value will
    not be observed by any backend since no observer is in scope.
    The empty-invocation case is supported for symmetry; users
    typically call this from inside a node body, middleware, or
    observer where an invocation is already in flight.
    """
    if not entries:
        return
    for key, value in entries.items():
        _validate_metadata_key(key)
        _validate_metadata_value(key, value)
    merged: dict[str, AttributeValue] = dict(_invocation_metadata_var.get())
    merged.update(entries)
    _invocation_metadata_var.set(MappingProxyType(merged))


def validate_invocation_metadata(mapping: object) -> MappingProxyType[str, AttributeValue]:
    """Validate a caller-supplied metadata mapping and return the
    read-only view the engine stashes on the ContextVar.

    Public so the engine (`CompiledGraph.invoke`) calls this at the
    boundary BEFORE any work begins; per spec §3.4 the rejection
    surfaces as a synchronous error to the caller of ``invoke()``
    rather than as a backend-emission failure.

    Returns the validated read-only mapping. Raises :class:`ValueError`
    on any rule violation (with a message naming the offending key).
    """
    if mapping is None:
        return _EMPTY_METADATA
    if not isinstance(mapping, dict):
        raise ValueError(f"invocation metadata must be a dict (or None); got {type(mapping).__name__}")
    typed_mapping = cast("dict[Any, Any]", mapping)
    validated: dict[str, AttributeValue] = {}
    for key, value in typed_mapping.items():
        _validate_metadata_key(key)
        _validate_metadata_value(key, value)
        validated[key] = value
    return MappingProxyType(validated)


def _validate_metadata_key(key: Any) -> None:
    if not isinstance(key, str):
        raise ValueError(f"invocation metadata key must be a string; got {type(key).__name__}")
    for reserved in _RESERVED_PREFIXES:
        if key.startswith(reserved):
            raise ValueError(
                f"invocation metadata key {key!r} uses reserved namespace prefix {reserved!r}; "
                f"reserved prefixes are for spec-normative attributes (openarmature.*, gen_ai.*)"
            )
    if key in _RESERVED_KEY_NAMES:
        raise ValueError(
            f"invocation metadata key {key!r} is reserved: it exactly matches a top-level "
            f"metadata key OA emits to observability backends, and would overwrite it. "
            f"Rename the key."
        )


def _validate_metadata_value(key: str, value: Any) -> None:
    # Scalars first — bool is checked BEFORE int because bool is a
    # subclass of int in Python and the spec treats them as distinct
    # AttributeValue variants.
    if isinstance(value, bool):
        return
    if isinstance(value, (str, int, float)):
        return
    if isinstance(value, (list, tuple)):
        seq = cast("list[Any] | tuple[Any, ...]", value)
        if not seq:
            # Empty arrays are accepted; the homogeneity check is
            # trivially satisfied.
            return
        element_type: type | None = None
        for element in seq:
            # bool BEFORE int again for the array case.
            etype: type
            if isinstance(element, bool):
                etype = bool
            elif isinstance(element, (str, int, float)):
                etype = type(element)
            else:
                raise ValueError(
                    f"invocation metadata key {key!r}: array element has unsupported type "
                    f"{type(element).__name__}; OTel AnyValue arrays accept only "
                    f"str / int / float / bool elements"
                )
            if element_type is None:
                element_type = etype
            elif element_type is not etype:
                raise ValueError(
                    f"invocation metadata key {key!r}: array elements MUST be homogeneous; "
                    f"saw {element_type.__name__} and {etype.__name__}"
                )
        return
    raise ValueError(
        f"invocation metadata key {key!r}: value type {type(value).__name__} is not "
        f"OTel-attribute-compatible; allowed: str, int, float, bool, or a homogeneous list/tuple of those"
    )


def _set_invocation_metadata(
    value: MappingProxyType[str, AttributeValue],
) -> Token[MappingProxyType[str, AttributeValue]]:
    """Set the invocation metadata for the current invocation.
    Internal — callers OUTSIDE the engine should not touch this; the
    engine paves the lifecycle in ``CompiledGraph.invoke`` around the
    outermost call.

    Use :func:`set_invocation_metadata` for mid-invocation
    augmentation (the public augmentation helper validates and
    merges); this internal setter is for the boundary stash only.
    """
    return _invocation_metadata_var.set(value)


def _reset_invocation_metadata(
    token: Token[MappingProxyType[str, AttributeValue]],
) -> None:
    _invocation_metadata_var.reset(token)


__all__ = [
    "AttributeValue",
    "current_invocation_metadata",
    "set_invocation_metadata",
    "validate_invocation_metadata",
    "_reset_invocation_metadata",
    "_set_invocation_metadata",
]
