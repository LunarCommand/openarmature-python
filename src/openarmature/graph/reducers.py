# Spec: realizes graph-engine §2 (Reducer concept). Proposal 0036
# (spec v0.27.0) expanded the required-built-in set from three to
# five — adding concat_flatten and merge_all for the fan-out
# collection shapes (a fan-out subgraph emitting list[X] per
# instance lands list[list[X]] at the parent target_field;
# emitting dict[str, X] lands list[dict] — neither of which append
# or merge can consume).

"""Reducers for merging node updates into state.

Each state field has exactly one reducer; the default is
``last_write_wins``. The five built-ins are ``last_write_wins``,
``append`` (for list-typed fields), ``merge`` (for mapping-typed
fields), ``concat_flatten`` (for list-of-lists → flat list), and
``merge_all`` (for list-of-mappings → folded mapping).
"""

from collections.abc import Mapping
from typing import Any, cast


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


class _ConcatFlatten(Reducer):
    # Proposal 0036: the list-of-lists analog of ``append``. The
    # fan-out engine lands ``list[list[X]]`` at the parent
    # target_field when each instance's collect_field is itself a
    # ``list[X]``; this reducer flattens one level onto the prior.
    # Strict like ``append`` — every element of ``update`` MUST be a
    # list. The TypeError surfaces as a ``ReducerError`` (graph-engine
    # §4) once the engine wraps it.
    name = "concat_flatten"

    def __call__(self, prior: Any, update: Any) -> list[Any]:
        if not isinstance(prior, list):
            raise TypeError(f"concat_flatten reducer requires a list prior; got {type(prior).__name__}")
        if not isinstance(update, list):
            raise TypeError(f"concat_flatten reducer requires a list update; got {type(update).__name__}")
        update_list = cast("list[Any]", update)
        for i, element in enumerate(update_list):
            if not isinstance(element, list):
                raise TypeError(
                    f"concat_flatten reducer requires every update element to be a list; "
                    f"update[{i}] is {type(element).__name__}"
                )
        prior_list = cast("list[Any]", prior)
        return [*prior_list, *(item for sublist in update_list for item in cast("list[Any]", sublist))]


class _MergeAll(Reducer):
    # Proposal 0036: the list-of-mappings analog of ``merge``. The
    # fan-out engine lands ``list[dict]`` at the parent target_field
    # when each instance's collect_field is a ``dict[str, X]``; this
    # reducer folds the sequence into the prior with shallow
    # last-write-wins per key (equivalent to applying ``merge`` N
    # times sequentially). Strict like ``merge`` — every element of
    # ``update`` MUST be a mapping.
    name = "merge_all"

    def __call__(self, prior: Any, update: Any) -> dict[Any, Any]:
        if not isinstance(prior, Mapping):
            raise TypeError(f"merge_all reducer requires a mapping prior; got {type(prior).__name__}")
        if not isinstance(update, list):
            raise TypeError(f"merge_all reducer requires a list update; got {type(update).__name__}")
        update_list = cast("list[Any]", update)
        result: dict[Any, Any] = dict(cast("Mapping[Any, Any]", prior))
        for i, element in enumerate(update_list):
            if not isinstance(element, Mapping):
                raise TypeError(
                    f"merge_all reducer requires every update element to be a mapping; "
                    f"update[{i}] is {type(element).__name__}"
                )
            result.update(cast("Mapping[Any, Any]", element))
        return result


last_write_wins: Reducer = _LastWriteWins()
append: Reducer = _Append()
merge: Reducer = _Merge()
concat_flatten: Reducer = _ConcatFlatten()
merge_all: Reducer = _MergeAll()
