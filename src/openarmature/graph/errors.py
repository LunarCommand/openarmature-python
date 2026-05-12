# Spec: realizes graph-engine §2 (compile-time errors) and §4
# (runtime errors). The four runtime categories other than
# ``state_validation_error`` carry a ``recoverable_state`` attribute
# per §4. Fan-out-specific error categories (both compile and runtime)
# mirror pipeline-utilities §9.

"""Errors raised by the graph engine.

Each error class carries a ``category`` class attribute matching the
canonical category identifier. The four runtime categories other
than ``state_validation_error`` carry a ``recoverable_state``
attribute.
"""

from typing import Any


class GraphError(Exception):
    """Base for all graph-engine errors."""


# ===== Compile-time errors =====


class CompileError(GraphError):
    """Base for compile-time errors."""

    category: str


class NoDeclaredEntry(CompileError):
    category = "no_declared_entry"

    def __init__(self) -> None:
        super().__init__("graph has no declared entry node")


class UnreachableNode(CompileError):
    category = "unreachable_node"

    def __init__(self, node_name: str) -> None:
        super().__init__(f"node {node_name!r} is unreachable from the entry")
        self.node_name = node_name


class DanglingEdge(CompileError):
    category = "dangling_edge"

    def __init__(self, source: str, target: str) -> None:
        super().__init__(f"edge from {source!r} references undeclared node {target!r}")
        self.source = source
        self.target = target


class MultipleOutgoingEdges(CompileError):
    category = "multiple_outgoing_edges"

    def __init__(self, source: str) -> None:
        super().__init__(f"node {source!r} has more than one outgoing edge; use a conditional edge to branch")
        self.source = source


class ConflictingReducers(CompileError):
    category = "conflicting_reducers"

    def __init__(self, field_name: str) -> None:
        super().__init__(f"field {field_name!r} has more than one declared reducer")
        self.field_name = field_name


class MappingReferencesUndeclaredField(CompileError):
    """Raised when a subgraph-as-node ``inputs`` or ``outputs``
    mapping names a field that is not declared in the relevant state
    schema."""

    category = "mapping_references_undeclared_field"

    def __init__(self, *, direction: str, side: str, field_name: str) -> None:
        super().__init__(f"subgraph {direction!r} mapping references undeclared {side} field {field_name!r}")
        self.direction = direction
        self.side = side
        self.field_name = field_name


class FanOutCountModeAmbiguous(CompileError):
    """Raised when a fan-out node specifies both ``items_field`` and
    ``count``, or neither. Exactly one is required."""

    category = "fan_out_count_mode_ambiguous"

    def __init__(self, node_name: str, message: str) -> None:
        super().__init__(f"fan-out node {node_name!r}: {message}")
        self.node_name = node_name


class FanOutFieldNotList(CompileError):
    """Raised when a fan-out node's ``items_field`` does not refer to
    a declared list-typed field on the parent state schema."""

    category = "fan_out_field_not_list"

    def __init__(self, node_name: str, field_name: str) -> None:
        super().__init__(f"fan-out node {node_name!r}: items_field {field_name!r} is not a list-typed field")
        self.node_name = node_name
        self.field_name = field_name


# ===== Runtime errors =====


class RuntimeGraphError(GraphError):
    """Base for runtime errors. The four non-validation categories carry a
    ``recoverable_state`` attribute."""

    category: str


class NodeException(RuntimeGraphError):
    category = "node_exception"
    node_name: str
    recoverable_state: Any

    def __init__(self, node_name: str, cause: BaseException, recoverable_state: Any) -> None:
        super().__init__(f"node {node_name!r} raised {type(cause).__name__}: {cause}")
        self.node_name = node_name
        self.recoverable_state = recoverable_state
        self.__cause__ = cause


class EdgeException(RuntimeGraphError):
    category = "edge_exception"
    source_node: str
    recoverable_state: Any

    def __init__(self, source_node: str, cause: BaseException, recoverable_state: Any) -> None:
        super().__init__(f"edge from {source_node!r} raised {type(cause).__name__}: {cause}")
        self.source_node = source_node
        self.recoverable_state = recoverable_state
        self.__cause__ = cause


class ReducerError(RuntimeGraphError):
    category = "reducer_error"
    field_name: str
    reducer_name: str
    producing_node: str
    recoverable_state: Any

    def __init__(
        self,
        field_name: str,
        reducer_name: str,
        producing_node: str,
        cause: BaseException,
        recoverable_state: Any,
    ) -> None:
        super().__init__(
            f"reducer {reducer_name!r} for field {field_name!r} "
            f"(producing node {producing_node!r}) raised "
            f"{type(cause).__name__}: {cause}"
        )
        self.field_name = field_name
        self.reducer_name = reducer_name
        self.producing_node = producing_node
        self.recoverable_state = recoverable_state
        self.__cause__ = cause


class RoutingError(RuntimeGraphError):
    category = "routing_error"
    source_node: str
    returned: object
    recoverable_state: Any

    def __init__(self, source_node: str, returned: object, recoverable_state: Any) -> None:
        super().__init__(
            f"conditional edge from {source_node!r} returned {returned!r}, "
            "which is neither a declared node nor END"
        )
        self.source_node = source_node
        self.returned = returned
        self.recoverable_state = recoverable_state


class FanOutEmpty(NodeException):
    """Raised when a fan-out node resolves to zero instances while
    its ``on_empty`` config is ``"raise"`` (the default).

    Surfaces as a regular ``node_exception`` (so it integrates with
    the existing error propagation and recoverable-state machinery)
    but exposes an additional ``fan_out_category`` attribute so
    callers can distinguish empty-fan-out from generic node failures.
    """

    fan_out_category = "fan_out_empty"

    def __init__(self, node_name: str, recoverable_state: Any) -> None:
        # Construct a synthetic cause so the NodeException message and
        # __cause__ chain stay consistent with the rest of §4.
        cause = RuntimeError(f"fan-out node {node_name!r} resolved to zero instances and on_empty='raise'")
        super().__init__(
            node_name=node_name,
            cause=cause,
            recoverable_state=recoverable_state,
        )


class FanOutInvalidCount(NodeException):
    """Raised when a fan-out node's ``count`` callable returns a
    negative integer at runtime. Same node-exception shape as
    :class:`FanOutEmpty`, with
    ``fan_out_category = "fan_out_invalid_count"``."""

    fan_out_category = "fan_out_invalid_count"

    def __init__(self, node_name: str, returned: int, recoverable_state: Any) -> None:
        cause = ValueError(f"fan-out node {node_name!r}: count callable returned {returned!r} (must be >= 0)")
        super().__init__(node_name=node_name, cause=cause, recoverable_state=recoverable_state)
        self.returned = returned


class FanOutInvalidConcurrency(NodeException):
    """Raised when a fan-out node's ``concurrency`` callable returns
    zero or a negative integer at runtime. Same node-exception shape
    as :class:`FanOutEmpty`."""

    fan_out_category = "fan_out_invalid_concurrency"

    def __init__(self, node_name: str, returned: int | None, recoverable_state: Any) -> None:
        cause = ValueError(
            f"fan-out node {node_name!r}: concurrency callable returned {returned!r} "
            "(must be a positive integer or None)"
        )
        super().__init__(node_name=node_name, cause=cause, recoverable_state=recoverable_state)
        self.returned = returned


class StateValidationError(RuntimeGraphError):
    """State failed schema validation at a graph boundary.

    Unlike the other runtime errors, this category does NOT carry
    ``recoverable_state`` — at entry there is no prior state to
    recover; at exit the failing state IS the final state.
    """

    category = "state_validation_error"

    def __init__(self, message: str, fields: list[str], cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.fields = fields
        if cause is not None:
            self.__cause__ = cause
