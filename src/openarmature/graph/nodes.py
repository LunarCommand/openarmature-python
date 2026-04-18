"""Graph nodes.

Per spec §2 Concepts (Node): a node is a named unit of work. Nodes MUST be
asynchronous and MUST NOT mutate the state they receive — they return a
partial update which the engine merges via reducers.

The `Node` Protocol exists so subgraphs can compose as nodes alongside
plain function-backed nodes (see `subgraph.SubgraphNode`).
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class Node(Protocol):
    """A unit of work in a compiled graph."""

    name: str

    async def run(self, state: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FunctionNode:
    """A node backed by an async callable."""

    name: str
    fn: Callable[[Any], Awaitable[Mapping[str, Any]]]

    async def run(self, state: Any) -> Mapping[str, Any]:
        return await self.fn(state)
