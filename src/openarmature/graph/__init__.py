# Spec: this package implements the graph-engine capability. Compile-time
# and runtime error categories come from graph-engine §2 and §4.

"""Public API for the OpenArmature graph engine.

Re-exports the surface a user touches when building and running a
graph: the state schema base, reducers, the builder/compiled pair,
edge primitives and the END sentinel, the node/subgraph/projection
seams, and the canonical compile-time and runtime error categories.
"""

from .builder import GraphBuilder
from .compiled import CompiledGraph
from .edges import END, ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    CompileError,
    ConflictingReducers,
    DanglingEdge,
    EdgeException,
    FanOutCountModeAmbiguous,
    FanOutEmpty,
    FanOutFieldNotList,
    FanOutInvalidConcurrency,
    FanOutInvalidCount,
    GraphError,
    MappingReferencesUndeclaredField,
    MultipleOutgoingEdges,
    NoDeclaredEntry,
    NodeException,
    ParallelBranchesBranchFailed,
    ParallelBranchesNoBranches,
    ReducerError,
    RoutingError,
    RuntimeGraphError,
    StateValidationError,
    UnreachableNode,
)
from .events import MetadataAugmentationEvent, NodeEvent
from .fan_out import FanOutConfig, FanOutNode
from .middleware import (
    Middleware,
    NextCall,
    RetryMiddleware,
    TimingMiddleware,
    TimingRecord,
    default_classifier,
    deterministic_backoff,
    exponential_jitter_backoff,
)
from .nodes import FunctionNode, Node
from .observer import DrainSummary, Observer, RemoveHandle, SubscribedObserver
from .parallel_branches import BranchSpec, ParallelBranchesNode
from .projection import ExplicitMapping, FieldNameMatching, ProjectionStrategy
from .reducers import Reducer, append, concat_flatten, last_write_wins, merge, merge_all
from .state import State
from .subgraph import SubgraphNode

__all__ = [
    "END",
    "CompileError",
    "CompiledGraph",
    "ConditionalEdge",
    "ConflictingReducers",
    "DanglingEdge",
    "DrainSummary",
    "EdgeException",
    "EndSentinel",
    "ExplicitMapping",
    "FanOutConfig",
    "FanOutCountModeAmbiguous",
    "FanOutEmpty",
    "FanOutFieldNotList",
    "FanOutInvalidConcurrency",
    "FanOutInvalidCount",
    "FanOutNode",
    "FieldNameMatching",
    "FunctionNode",
    "GraphBuilder",
    "GraphError",
    "MappingReferencesUndeclaredField",
    "MetadataAugmentationEvent",
    "Middleware",
    "MultipleOutgoingEdges",
    "NextCall",
    "Node",
    "NodeEvent",
    "NodeException",
    "NoDeclaredEntry",
    "Observer",
    "ParallelBranchesBranchFailed",
    "ParallelBranchesNoBranches",
    "ParallelBranchesNode",
    "BranchSpec",
    "ProjectionStrategy",
    "Reducer",
    "ReducerError",
    "RemoveHandle",
    "RetryMiddleware",
    "RoutingError",
    "RuntimeGraphError",
    "State",
    "StateValidationError",
    "StaticEdge",
    "SubgraphNode",
    "SubscribedObserver",
    "TimingMiddleware",
    "TimingRecord",
    "UnreachableNode",
    "append",
    "concat_flatten",
    "default_classifier",
    "deterministic_backoff",
    "exponential_jitter_backoff",
    "last_write_wins",
    "merge",
    "merge_all",
]
