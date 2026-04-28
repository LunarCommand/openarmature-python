"""Graph builder: mutable construction → compile to immutable `CompiledGraph`.

Per spec §2: compilation MUST fail if the graph has no declared entry,
unreachable nodes, dangling edges, a node with more than one outgoing edge,
or a field with more than one declared reducer.

`GraphBuilder[StateT]` is parameterized on the graph's state type. Node
functions, conditional-edge functions, and the returned `CompiledGraph[StateT]`
all carry `StateT` forward so consumers get typed `invoke()` return values and
a type-checked `state` parameter on every callback — without `cast(...)` calls.
"""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Self, cast

from .compiled import CompiledGraph
from .edges import ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    ConflictingReducers,
    DanglingEdge,
    MultipleOutgoingEdges,
    NoDeclaredEntry,
    UnreachableNode,
)
from .nodes import FunctionNode, Node
from .projection import FieldNameMatching, ProjectionStrategy
from .reducers import Reducer
from .state import State, field_reducers, resolve_reducer
from .subgraph import SubgraphNode


class GraphBuilder[StateT: State]:
    """Mutable builder for a graph; call `compile()` to produce a `CompiledGraph`."""

    def __init__(self, state_cls: type[StateT]) -> None:
        self.state_cls: type[StateT] = state_cls
        self._nodes: dict[str, Node[StateT]] = {}
        self._edges: list[StaticEdge | ConditionalEdge[StateT]] = []
        self._entry: str | None = None

    def add_node(
        self,
        name: str,
        fn: Callable[[StateT], Awaitable[Mapping[str, Any]]],
    ) -> Self:
        if name in self._nodes:
            raise ValueError(f"node {name!r} already declared")
        self._nodes[name] = FunctionNode[StateT](name=name, fn=fn)
        return self

    def add_subgraph_node[ChildT: State](
        self,
        name: str,
        compiled: CompiledGraph[ChildT],
        projection: ProjectionStrategy[StateT, ChildT] | None = None,
    ) -> Self:
        if name in self._nodes:
            raise ValueError(f"node {name!r} already declared")
        proj: ProjectionStrategy[StateT, ChildT] = (
            projection if projection is not None else FieldNameMatching[StateT, ChildT]()
        )
        self._nodes[name] = SubgraphNode[StateT, ChildT](name=name, compiled=compiled, projection=proj)
        return self

    def add_edge(self, source: str, target: str | EndSentinel) -> Self:
        self._edges.append(StaticEdge(source=source, target=target))
        return self

    def add_conditional_edge(
        self,
        source: str,
        fn: Callable[[StateT], str | EndSentinel],
    ) -> Self:
        self._edges.append(ConditionalEdge[StateT](source=source, fn=fn))
        return self

    def set_entry(self, name: str) -> Self:
        self._entry = name
        return self

    def compile(self) -> CompiledGraph[StateT]:
        # 1. ConflictingReducers — state schema check.
        per_field = field_reducers(self.state_cls)
        for fname, declared in per_field.items():
            if len(declared) > 1:
                raise ConflictingReducers(fname)
        resolved: dict[str, Reducer] = {
            fname: resolve_reducer(declared) for fname, declared in per_field.items()
        }

        # 2. MappingReferencesUndeclaredField — projection strategies on every
        #    subgraph-as-node validate against the parent and child schemas.
        #    Default `FieldNameMatching.validate` is a no-op.
        #    ChildT is erased once SubgraphNode is stored as Node[StateT]; the
        #    cast restores enough type info to call `validate` without pyright
        #    flagging unknown member types. The runtime values still match —
        #    `compiled.state_cls` is exactly the type the projection expects.
        for node in self._nodes.values():
            if isinstance(node, SubgraphNode):
                sub = cast("SubgraphNode[StateT, State]", node)
                sub.projection.validate(self.state_cls, sub.compiled.state_cls)

        # 3. NoDeclaredEntry.
        if self._entry is None:
            raise NoDeclaredEntry()

        # 4. Entry must point to a declared node (treat as DanglingEdge).
        if self._entry not in self._nodes:
            raise DanglingEdge(source="<entry>", target=self._entry)

        # 5. DanglingEdge — both endpoints of every edge must be declared.
        for edge in self._edges:
            if edge.source not in self._nodes:
                raise DanglingEdge(source=edge.source, target=edge.source)
            if isinstance(edge, StaticEdge) and isinstance(edge.target, str):
                if edge.target not in self._nodes:
                    raise DanglingEdge(source=edge.source, target=edge.target)

        # 6. MultipleOutgoingEdges + index by source for the reachability pass.
        edges_by_source: dict[str, StaticEdge | ConditionalEdge[StateT]] = {}
        for edge in self._edges:
            if edge.source in edges_by_source:
                raise MultipleOutgoingEdges(edge.source)
            edges_by_source[edge.source] = edge

        # 7. UnreachableNode — BFS from entry. Conditional edges over-approximate
        #    by reaching every declared node (we cannot statically know the fn's
        #    range), which keeps the check sound (no false positives).
        reachable = self._reachable_nodes(edges_by_source)
        for node_name in self._nodes:
            if node_name not in reachable:
                raise UnreachableNode(node_name)

        return CompiledGraph[StateT](
            state_cls=self.state_cls,
            entry=self._entry,
            nodes=dict(self._nodes),
            edges=edges_by_source,
            reducers=resolved,
        )

    def _reachable_nodes(
        self,
        edges_by_source: Mapping[str, StaticEdge | ConditionalEdge[StateT]],
    ) -> set[str]:
        assert self._entry is not None
        reachable: set[str] = {self._entry}
        frontier = [self._entry]
        all_names = set(self._nodes.keys())
        while frontier:
            current = frontier.pop()
            edge = edges_by_source.get(current)
            if edge is None:
                continue
            if isinstance(edge, StaticEdge):
                if isinstance(edge.target, str) and edge.target not in reachable:
                    reachable.add(edge.target)
                    frontier.append(edge.target)
            else:
                for name in all_names - reachable:
                    reachable.add(name)
                    frontier.append(name)
        return reachable
