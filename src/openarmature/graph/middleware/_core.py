"""Middleware infrastructure: protocol, chain composition, registration.

Per spec pipeline-utilities §2 Concepts: a middleware is an async callable
with the shape ``(state, next) -> partial_update``. The middleware chain
composes outer-to-inner — the first middleware in the list runs first,
calls ``next(state)`` to invoke the next layer, and so on with the
wrapped node at the inner end. Code before ``await next(...)`` is the
pre-node phase (running on the way IN); code after is the post-node
phase (running on the way OUT).

Per §3 Registration: per-graph middleware composes OUTSIDE per-node
middleware — the runtime chain is
``[per_graph_outer_to_inner...] → [per_node_outer_to_inner...] → node``.

Per §4: middleware does NOT cross the subgraph boundary. The parent's
middleware wraps the SubgraphNode dispatch as a single atomic call;
the subgraph's own middleware wraps its internal nodes independently.

Per §5: errors raised inside the chain (from the node or from inner
middleware) propagate through ``await next(...)``. Middleware may catch
and recover (returning a partial update) or re-raise. Uncaught exceptions
become a graph-engine §4 ``node_exception`` once they reach the engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..state import State


class NextCall(Protocol):
    """The ``next`` callable a middleware receives.

    Calling it with a state invokes the next layer of the chain (or the
    wrapped node, at the inner end) and returns the partial update from
    that layer. Middleware MAY transform the state passed to ``next`` —
    the transformed state flows down the chain but does NOT replace the
    engine's pre-merge state at the outermost level (per §2: "the
    transformed state is passed to ``next``, NOT to the engine's merge
    step").
    """

    async def __call__(self, state: Any, /) -> Mapping[str, Any]:
        """Invoke the next layer with ``state``, return its partial update."""
        ...


class Middleware(Protocol):
    """An async callable that wraps the dispatch of a single node.

    Per spec v0.4.0 pipeline-utilities §2: the shape is
    ``(state, next) -> partial_update``. The middleware MUST return a
    mapping of field names to values — same shape a node returns. It may:

    - Inspect or transform ``state`` before calling ``next(state)``.
    - Inspect or transform the partial update returned from ``next``.
    - Short-circuit by NOT calling ``next`` and returning its own partial
      (the rest of the chain — subsequent middleware and the wrapped node
      — does not execute).
    - Catch exceptions raised by ``next(state)`` and either re-raise,
      transform, or recover (returning a partial update instead of
      raising).
    - Call ``next`` more than once (e.g., retry middleware).

    A middleware MUST NOT mutate the input ``state`` object — pass a new
    state to ``next`` if a transformation is needed.
    """

    async def __call__(
        self,
        state: Any,
        next: NextCall,
        /,
    ) -> Mapping[str, Any]:
        """Wrap the chain layer below this one."""
        ...


# Type alias for a chain-callable: the runtime form of a composed
# middleware list, called with the engine's pre-merge state and returning
# the (possibly post-phase-transformed) partial update. Structurally
# identical to ``NextCall`` — ``NextCall`` is the user-facing protocol
# describing what middleware receives; ``ChainCall`` is the engine-side
# alias for the same shape used during chain assembly.
ChainCall = NextCall


def compose_chain(
    middlewares: Sequence[Middleware],
    innermost: ChainCall,
) -> ChainCall:
    """Build a runtime chain function from a list of middleware + an
    innermost callable.

    ``middlewares`` is in outer-to-inner order; ``innermost`` is the
    terminal layer the chain ultimately invokes (in the engine's case,
    the per-attempt dispatch wrapper around ``node.run`` + merge + event
    dispatch).

    The returned callable takes a state and returns the final
    partial-update (after post-phase transformations from the chain).
    Calling it once = one full chain traversal = at LEAST one call to
    ``innermost`` (more if a middleware calls ``next`` repeatedly, e.g.
    retry).

    Performance note: this is called fresh per dispatch from
    ``CompiledGraph._step_function_node``, producing one closure layer
    per middleware on every node step. For typical workloads
    (single-digit middleware × hundreds of node activations) this is
    negligible. Under heavy fan-out (Phase 3+), e.g. 10K instances × 5
    inner nodes × 3 middlewares = 150K closure constructions per
    invocation; worth measuring with realistic workloads when the
    fan-out runtime lands. The optimization shape (cache the chain at
    compile time, inject only the per-dispatch attempt counter via a
    thin wrapper) is straightforward but premature without numbers.
    """
    chain: ChainCall = innermost
    for mw in reversed(middlewares):
        chain = _wrap(mw, chain)
    return chain


def _wrap(mw: Middleware, next_layer: ChainCall) -> ChainCall:
    """Bind one middleware to the layer below it.

    The returned callable receives a state, hands it to ``mw`` along with
    a ``next`` callable that invokes ``next_layer``. The closure captures
    ``next_layer`` so each chain build is independent.
    """

    async def wrapped(state: State) -> Mapping[str, Any]:
        return await mw(state, next_layer)

    return wrapped


__all__ = [
    "ChainCall",
    "Middleware",
    "NextCall",
    "compose_chain",
]
