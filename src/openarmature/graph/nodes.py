# Spec: realizes graph-engine §2 (Node concept) and pipeline-utilities
# §3 (Registration: per-node middleware). Per-graph middleware composes
# OUTSIDE the per-node list at runtime per §3.

"""Graph nodes.

A node is a named unit of work. Nodes are asynchronous and don't
mutate the state they receive; they return a partial update which
the engine merges via reducers.

The `Node` Protocol exists so subgraphs can compose as nodes
alongside plain function-backed nodes (see `subgraph.SubgraphNode`).
Both are parameterized on `StateT` so the outer graph's state type
flows through to node functions at type-check time.

Each node carries an optional ordered tuple of `Middleware` declared
at its registration site (per-node middleware).
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .middleware import Middleware
from .state import State


class Node[StateT: State](Protocol):
    """A unit of work in a compiled graph."""

    @property
    def name(self) -> str:
        """The name this node was registered under in its containing graph."""
        raise NotImplementedError

    @property
    def middleware(self) -> tuple[Middleware, ...]:
        """Per-node middleware applied at this node's registration site,
        outer-to-inner. Composed inside any per-graph middleware."""
        raise NotImplementedError

    async def run(self, state: StateT) -> Mapping[str, Any]:
        """Execute against `state` and return a partial update to be merged via reducers."""
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionNode[StateT: State]:
    """A node backed by an async callable."""

    name: str
    fn: Callable[[StateT], Awaitable[Mapping[str, Any]]]
    middleware: tuple[Middleware, ...] = field(default_factory=tuple[Middleware, ...])

    async def run(self, state: StateT) -> Mapping[str, Any]:
        """Invoke the wrapped async callable and return its partial
        update. Called by the engine inside any per-node and per-graph
        middleware chain."""
        return await self.fn(state)
