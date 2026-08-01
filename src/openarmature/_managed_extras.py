# Spec: realizes the llm-provider §6 *Managed-field collision* rule
# (proposals 0105 + 0108), inherited by retrieval-provider §10. Shared by the
# llm and retrieval wire mappings, which is why it sits at the package root
# rather than under either capability -- the rule spans both, and the mechanics
# are identical. It raises the shared ``ProviderInvalidRequest`` (llm.errors,
# which retrieval already depends on), so there is no new dependency edge.
#
# §6 forwards an undeclared extras field to the wire body untouched. A field the
# mapping itself manages is the exception: one it sets for its own correctness
# (0105 -- e.g. a fail-loud truncate flag, the structural model / messages) or
# produces as the wire realization of a declared config field (0108 -- e.g.
# temperature, stop from stop_sequences, Jina task from input_type). When a
# caller's extras key names a managed field, untouched pass-through does not
# apply and the field's shape decides:
#   - additive / list-shaped (stop, embedding_types): merge the caller's
#     value(s) onto the managed value(s), managed-first, de-duplicated
#     first-occurrence-wins; a scalar extra is coerced to a one-element list (the
#     wire scalar-or-array shape), and a caller value already present collapses
#     (the merge arm's form of the matching no-op).
#   - non-additive (a scalar mode-switch, or an object the mapping constructs):
#     a caller value equal to the managed value is a redundant no-op; a
#     conflicting one is rejected pre-send (provider_invalid_request). Never
#     silently dropped, never silently overriding the managed value.
# A conditionally-managed field is managed only while the mapping produces it
# (response_format on the structured-output path; Jina task while input_type is
# set): the caller includes the key in ``managed`` only when produced, so an
# extra naming a not-currently-produced field rides untouched (the escape
# hatch). An unmanaged key keeps §6's untouched pass-through verbatim.

"""Fold provider-specific extras into a wire body, honoring managed fields.

The wire mappings build a request body, then hand their caller-supplied extras
to :func:`apply_managed_extras`, which reconciles any extras key that collides
with a field the mapping manages. The rule the reconciliation follows is
described in the module comment above.
"""

from collections.abc import Mapping
from typing import Any, Literal, cast

# NOTE: ProviderInvalidRequest is imported lazily inside the reject branch, not
# at module top. The `openarmature.llm` package eagerly wires its providers, and
# those providers import THIS module, so a top-level `from openarmature.llm...`
# here forms a cycle (_managed_extras -> llm -> providers.openai ->
# _managed_extras) that fails when this module is imported before the llm
# package is warm. The error is raised only on a real collision, so the lazy
# import costs nothing on the common path.

# A managed key's arm. "merge" for an additive / list-shaped field; "reject"
# for a non-additive scalar or object.
ManagedArm = Literal["merge", "reject"]


def apply_managed_extras(
    body: dict[str, Any],
    extras: Mapping[str, Any],
    managed: Mapping[str, ManagedArm],
    *,
    managed_values: Mapping[str, Any] | None = None,
) -> None:
    """Fold ``extras`` into ``body``, reconciling managed-field collisions.

    ``body`` already holds the mapping's produced values (and its structural
    keys). ``managed`` maps each managed wire key to its arm (``"merge"`` or
    ``"reject"``); a key absent from ``managed`` is unmanaged and rides
    untouched. ``managed_values`` supplies the managed value for a key the
    mapping does not place in ``body`` because it relies on a wire default, so a
    matching extras value stays a no-op that leaves the body minimal; where
    absent, the managed value is ``body[key]``.

    Mutates ``body`` in place; ``extras`` is not modified. Raises
    :class:`ProviderInvalidRequest` on a conflicting non-additive collision.
    """
    # ``managed_values`` carries the value for a relied-upon wire default the
    # mapping does not emit (TEI /embed's truncate: false fail-loud default), so
    # a matching extra no-ops and leaves the body minimal rather than adding it.
    defaults = managed_values or {}
    for key, value in extras.items():
        arm = managed.get(key)
        if arm is None:
            if key in body:
                # The mapping produced this wire key but did not declare it in
                # ``managed``, so the collision rule cannot see it: setdefault
                # would silently drop the extra and plain assignment would
                # silently override the produced value, and 0105 forbids both.
                # This is a mapping-author bug, not a caller error, so fail loud
                # with a plain ValueError (never the caller-facing
                # ProviderInvalidRequest) rather than mishandle the collision.
                raise ValueError(
                    f"wire mapping produced body key {key!r} but did not declare "
                    f"it in `managed`; every produced key must be managed so a "
                    f"colliding extra is merged or rejected, never silently dropped"
                )
            # Unmanaged: untouched pass-through (the key is not one the mapping
            # produced, so there is nothing to collide with).
            body[key] = value
        elif arm == "merge":
            body[key] = _merge_list(body.get(key), value)
        else:  # "reject"
            managed_value = defaults[key] if key in defaults else body.get(key)
            if managed_value == value:
                continue  # redundant no-op: caller re-sent the managed value
            from openarmature.llm.errors import ProviderInvalidRequest

            raise ProviderInvalidRequest(
                f"extras key {key!r} conflicts with the mapping-managed wire "
                f"field {key!r} (managed value {_summarize(managed_value)}, "
                f"extras value {_summarize(value)}); a managed field cannot be "
                f"overridden via extras"
            )


def _merge_list(managed_value: Any, extra_value: Any) -> list[Any]:
    """Merge an extras value onto a managed list value: managed entries first,
    then the extra's, de-duplicated first-occurrence-wins. A scalar on either
    side is coerced to a one-element list (the wire's scalar-or-array shape,
    e.g. OpenAI ``stop``)."""
    base = _as_list(managed_value)
    out = list(base)
    for item in _as_list(extra_value):
        if item not in out:
            out.append(item)
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return cast("list[Any]", value)
    return [value]


def _summarize(value: Any) -> str:
    """Render a managed / extras value for a reject-collision error message. A
    scalar renders verbatim; a list or dict (a full message transcript, a
    response_format schema, a tool array) renders as a type-and-size summary so
    the error does not embed the entire payload in its string."""
    if isinstance(value, list):
        return f"<list of {len(cast('list[Any]', value))}>"
    if isinstance(value, dict):
        return f"<dict of {len(cast('dict[Any, Any]', value))} keys>"
    return repr(value)
