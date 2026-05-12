# Spec: state is the typed product type from graph-engine §2 Concepts,
# validated at graph boundaries. Immutability (Pydantic frozen) enforces
# that nodes cannot mutate the object they receive; ``extra="forbid"``
# enforces "typed product type validated at graph boundaries."

"""Typed state schemas.

State is a typed, immutable product type. Nodes receive a snapshot
and return partial updates; the engine merges via per-field reducers.

Per-field reducers are declared via Annotated metadata::

    class S(State):
        history: Annotated[list[str], append]
        score: int  # default = last_write_wins
"""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from .reducers import Reducer, last_write_wins


class State(BaseModel):
    """Base for graph state schemas. Immutable; reducers attach via Annotated."""

    # ``extra="forbid"`` makes node updates that name an undeclared
    # field surface as a ``state_validation_error`` at the merge step,
    # matching spec §2's "typed product type validated at graph
    # boundaries" intent.
    model_config = ConfigDict(frozen=True, extra="forbid")


def field_reducers(state_cls: type[State]) -> Mapping[str, list[Reducer]]:
    """Return `{field_name: [declared reducers]}` for each field on `state_cls`.

    Returned as a list so callers (graph compilation) can detect
    `conflicting_reducers` (more than one reducer declared on the same field)
    before silently picking one.
    """

    result: dict[str, list[Reducer]] = {}
    for name, field_info in state_cls.model_fields.items():
        result[name] = [m for m in field_info.metadata if isinstance(m, Reducer)]
    return result


def resolve_reducer(declared: list[Reducer]) -> Reducer:
    """Resolve a field's reducer from the list returned by `field_reducers`.

    Falls back to `last_write_wins` if no reducer was declared. The caller is
    responsible for raising `ConflictingReducers` when `len(declared) > 1`;
    this helper assumes that check has already happened.
    """

    return declared[0] if declared else last_write_wins


__all__ = ["State", "field_reducers", "resolve_reducer"]
