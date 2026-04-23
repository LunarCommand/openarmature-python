"""Graph nodes.

Per spec §2 Concepts (Node): a node is a named unit of work. Nodes MUST be
asynchronous and MUST NOT mutate the state they receive — they return a
partial update which the engine merges via reducers.

The `Node` Protocol exists so subgraphs can compose as nodes alongside
plain function-backed nodes (see `subgraph.SubgraphNode`). Both are
parameterized on `StateT` so the outer graph's state type flows through
to node functions at type-check time.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .state import State


class Node[StateT: State](Protocol):
    """A unit of work in a compiled graph."""

    @property
    def name(self) -> str: ...

    async def run(self, state: StateT) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FunctionNode[StateT: State]:
    """A node backed by an async callable."""

    name: str
    fn: Callable[[StateT], Awaitable[Mapping[str, Any]]]

    async def run(self, state: StateT) -> Mapping[str, Any]:
        return await self.fn(state)
