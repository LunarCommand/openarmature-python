"""Typed state schemas.

Per spec §2 Concepts (State): state is a typed product type validated at graph
boundaries. State is immutable (Pydantic frozen) so nodes cannot mutate the
object they receive.

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

    model_config = ConfigDict(frozen=True)


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
