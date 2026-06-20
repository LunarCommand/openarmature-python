# Spec: this package implements the graph-engine capability. Compile-time
# and runtime error categories come from graph-engine §2 and §4.

"""Public API for the OpenArmature graph engine.

Re-exports the surface a user touches when building and running a
graph: the state schema base, reducers, the builder/compiled pair,
edge primitives and the END sentinel, the node/subgraph/projection
seams, and the canonical compile-time and runtime error categories.
"""

from .builder import GraphBuilder
from .cause_chain import CaughtException, CauseLink, classify_cause_chain
from .compiled import CompiledGraph
from .edges import END, ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    CompileError,
    ConflictingReducers,
    DanglingEdge,
    EdgeException,
    FanOutCountModeAmbiguous,
    FanOutDegradedUpdateMissingCollectField,
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
from .events import (
    FailureIsolatedEvent,
    InvocationCompletedEvent,
    InvocationStartedEvent,
    LlmCompletionEvent,
    LlmFailedEvent,
    LlmRetryAttemptEvent,
    MetadataAugmentationEvent,
    NodeEvent,
)
from .fan_out import FanOutConfig, FanOutNode
from .middleware import (
    DegradedUpdate,
    FailureIsolationMiddleware,
    Middleware,
    NextCall,
    RetryConfig,
    RetryMiddleware,
    TimingMiddleware,
    TimingRecord,
    default_classifier,
    deterministic_backoff,
    exponential_jitter_backoff,
)
from .nodes import FunctionNode, Node
from .observer import DrainSummary, Observer, ObserverEvent, RemoveHandle, SubscribedObserver
from .parallel_branches import BranchSpec, ParallelBranchesNode
from .projection import ExplicitMapping, FieldNameMatching, ProjectionStrategy
from .reducers import Reducer, append, concat_flatten, last_write_wins, merge, merge_all
from .state import State
from .subgraph import SubgraphNode

__all__ = [
    "END",
    "CaughtException",
    "CauseLink",
    "CompileError",
    "CompiledGraph",
    "ConditionalEdge",
    "ConflictingReducers",
    "DanglingEdge",
    "DegradedUpdate",
    "DrainSummary",
    "EdgeException",
    "EndSentinel",
    "ExplicitMapping",
    "FailureIsolatedEvent",
    "FailureIsolationMiddleware",
    "FanOutConfig",
    "FanOutCountModeAmbiguous",
    "FanOutDegradedUpdateMissingCollectField",
    "FanOutEmpty",
    "FanOutFieldNotList",
    "FanOutInvalidConcurrency",
    "FanOutInvalidCount",
    "FanOutNode",
    "FieldNameMatching",
    "FunctionNode",
    "GraphBuilder",
    "GraphError",
    "InvocationCompletedEvent",
    "InvocationStartedEvent",
    "LlmCompletionEvent",
    "LlmFailedEvent",
    "LlmRetryAttemptEvent",
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
    "ObserverEvent",
    "ParallelBranchesBranchFailed",
    "ParallelBranchesNoBranches",
    "ParallelBranchesNode",
    "BranchSpec",
    "ProjectionStrategy",
    "Reducer",
    "ReducerError",
    "RemoveHandle",
    "RetryConfig",
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
    "classify_cause_chain",
    "concat_flatten",
    "default_classifier",
    "deterministic_backoff",
    "exponential_jitter_backoff",
    "last_write_wins",
    "merge",
    "merge_all",
]
