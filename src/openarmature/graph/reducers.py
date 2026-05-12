# Spec: realizes graph-engine §2 (Reducer concept) — last_write_wins,
# append, and merge are the three built-ins the spec requires.

"""Reducers for merging node updates into state.

Each state field has exactly one reducer; the default is
``last_write_wins``. The three built-ins are ``last_write_wins``,
``append`` (for list-typed fields), and ``merge`` (for mapping-typed
fields).
"""

from collections.abc import Mapping
from typing import Any


class Reducer:
    """Base class for state-field reducers.

    Each reducer carries a canonical `name` used in error messages and
    introspection. Subclasses override `__call__` to merge a node's partial
    update for a single field into the prior value.
    """

    name: str

    def __call__(self, prior: Any, update: Any) -> Any:
        raise NotImplementedError


class _LastWriteWins(Reducer):
    name = "last_write_wins"

    def __call__(self, prior: Any, update: Any) -> Any:
        return update


class _Append(Reducer):
    name = "append"

    def __call__(self, prior: Any, update: Any) -> list[Any]:
        if not isinstance(prior, list):
            raise TypeError(f"append reducer requires a list prior; got {type(prior).__name__}")
        if not isinstance(update, list):
            raise TypeError(f"append reducer requires a list update; got {type(update).__name__}")
        return [*prior, *update]


class _Merge(Reducer):
    name = "merge"

    def __call__(self, prior: Any, update: Any) -> dict[Any, Any]:
        if not isinstance(prior, Mapping):
            raise TypeError(f"merge reducer requires a mapping prior; got {type(prior).__name__}")
        if not isinstance(update, Mapping):
            raise TypeError(f"merge reducer requires a mapping update; got {type(update).__name__}")
        return {**prior, **update}


last_write_wins: Reducer = _LastWriteWins()
append: Reducer = _Append()
merge: Reducer = _Merge()
