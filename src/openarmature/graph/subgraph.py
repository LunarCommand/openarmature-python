"""Subgraphs as nodes.

Per spec v0.1.1 §2 Subgraph: a compiled graph is used as a node inside another
graph. The subgraph runs against its own state schema; projection between
parent and subgraph is delegated to a `ProjectionStrategy` (default:
`FieldNameMatching`).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .projection import FieldNameMatching, ProjectionStrategy
from .state import State

if TYPE_CHECKING:
    from .compiled import CompiledGraph


@dataclass(frozen=True)
class SubgraphNode:
    """A node backed by a compiled subgraph."""

    name: str
    compiled: "CompiledGraph"
    projection: ProjectionStrategy = field(default_factory=FieldNameMatching)

    async def run(self, state: State) -> Mapping[str, Any]:
        sub_initial = self.projection.project_in(state, self.compiled.state_cls)
        sub_final = await self.compiled.invoke(sub_initial)
        return self.projection.project_out(sub_final, state, self.compiled.state_cls)
