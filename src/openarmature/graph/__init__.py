"""Public API for the OpenArmature graph engine.

Re-exports the surface a user touches when building and running a graph: the
state schema base, reducers, the builder/compiled pair, edge primitives and
the END sentinel, the node/subgraph/projection seams, and the canonical
compile-time and runtime error categories from spec §2 and §4.
"""

from .builder import GraphBuilder
from .compiled import CompiledGraph
from .edges import END, ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    CompileError,
    ConflictingReducers,
    DanglingEdge,
    EdgeException,
    GraphError,
    MappingReferencesUndeclaredField,
    MultipleOutgoingEdges,
    NoDeclaredEntry,
    NodeException,
    ReducerError,
    RoutingError,
    RuntimeGraphError,
    StateValidationError,
    UnreachableNode,
)
from .nodes import FunctionNode, Node
from .projection import ExplicitMapping, FieldNameMatching, ProjectionStrategy
from .reducers import Reducer, append, last_write_wins, merge
from .state import State
from .subgraph import SubgraphNode

__all__ = [
    "END",
    "CompileError",
    "CompiledGraph",
    "ConditionalEdge",
    "ConflictingReducers",
    "DanglingEdge",
    "EdgeException",
    "EndSentinel",
    "ExplicitMapping",
    "FieldNameMatching",
    "FunctionNode",
    "GraphBuilder",
    "GraphError",
    "MappingReferencesUndeclaredField",
    "MultipleOutgoingEdges",
    "Node",
    "NodeException",
    "NoDeclaredEntry",
    "ProjectionStrategy",
    "Reducer",
    "ReducerError",
    "RoutingError",
    "RuntimeGraphError",
    "State",
    "StateValidationError",
    "StaticEdge",
    "SubgraphNode",
    "UnreachableNode",
    "append",
    "last_write_wins",
    "merge",
]
