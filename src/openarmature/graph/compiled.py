"""Compiled graph + execute loop.

Per spec §3 Execution model: execution begins at the entry node; each step
runs a node, merges its partial update via per-field reducers, then evaluates
the outgoing edge against the post-update state to choose the next node (or
END to halt).

Per spec §4 Error semantics: node, edge, reducer, and routing errors carry
recoverable state; state validation errors do not.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .edges import END, ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    EdgeException,
    NodeException,
    ReducerError,
    RoutingError,
    StateValidationError,
)
from .nodes import Node
from .reducers import Reducer
from .state import State


def _merge_partial(
    prior: State,
    partial: Mapping[str, Any],
    reducers: Mapping[str, Reducer],
    producing_node: str,
) -> State:
    """Apply per-field reducers to merge a node's partial update into prior state.

    Re-validates the resulting state against the schema (per spec §2 SHOULD
    validate at node boundaries). Wraps reducer failures as `ReducerError` and
    schema failures as `StateValidationError`.
    """

    new_values = prior.model_dump()
    for field_name, partial_value in partial.items():
        reducer = reducers.get(field_name)
        if reducer is None:
            # Unknown field — surface as a schema validation failure below.
            new_values[field_name] = partial_value
            continue
        try:
            new_values[field_name] = reducer(new_values[field_name], partial_value)
        except Exception as e:
            raise ReducerError(
                field_name=field_name,
                reducer_name=reducer.name,
                producing_node=producing_node,
                cause=e,
                recoverable_state=prior,
            ) from e

    try:
        return type(prior).model_validate(new_values)
    except ValidationError as e:
        offending = sorted({str(err["loc"][0]) for err in e.errors() if err["loc"]})
        raise StateValidationError(
            f"state validation failed after node {producing_node!r}: {e}",
            fields=offending,
            cause=e,
        ) from e


@dataclass(frozen=True)
class CompiledGraph:
    """An immutable, executable graph produced by `GraphBuilder.compile()`."""

    state_cls: type[State]
    entry: str
    nodes: Mapping[str, Node]
    edges: Mapping[str, StaticEdge | ConditionalEdge]
    reducers: Mapping[str, Reducer]

    async def invoke(self, initial_state: State) -> State:
        """Run the graph from `initial_state` to END and return the final state.

        Raises one of the runtime error categories from spec §4 on failure.
        """

        state = initial_state
        current = self.entry

        while True:
            node = self.nodes[current]

            # Run the node. Wrap user exceptions as NodeException with
            # recoverable_state = state at point of failure (pre-update).
            try:
                partial = await node.run(state)
            except Exception as e:
                raise NodeException(node_name=current, cause=e, recoverable_state=state) from e

            # Merge partial into state via reducers (may raise ReducerError or
            # StateValidationError; both already carry the right context).
            state = _merge_partial(state, partial, self.reducers, current)

            # Evaluate the outgoing edge against the post-update state.
            edge = self.edges[current]
            if isinstance(edge, StaticEdge):
                target: str | EndSentinel = edge.target
            else:
                try:
                    target = edge.fn(state)
                except Exception as e:
                    raise EdgeException(source_node=current, cause=e, recoverable_state=state) from e

            if target is END:
                return state

            if not isinstance(target, str) or target not in self.nodes:
                raise RoutingError(source_node=current, returned=target, recoverable_state=state)

            current = target
