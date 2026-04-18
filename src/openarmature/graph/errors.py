"""Errors raised by the graph engine.

Each error class carries a `category` class attribute matching the canonical
identifier mandated by spec §2 (compile-time) and §4 (runtime). Per spec §4,
the four runtime categories other than state_validation_error MUST carry a
recoverable_state attribute.
"""

from typing import Any


class GraphError(Exception):
    """Base for all graph-engine errors."""


# ===== Compile-time errors (spec §2) =====


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


# ===== Runtime errors (spec §4) =====


class RuntimeGraphError(GraphError):
    """Base for runtime errors. The four non-validation categories carry a
    `recoverable_state` attribute per spec §4."""

    category: str


class NodeException(RuntimeGraphError):
    category = "node_exception"

    def __init__(self, node_name: str, cause: BaseException, recoverable_state: Any) -> None:
        super().__init__(f"node {node_name!r} raised {type(cause).__name__}: {cause}")
        self.node_name = node_name
        self.recoverable_state = recoverable_state
        self.__cause__ = cause


class EdgeException(RuntimeGraphError):
    category = "edge_exception"

    def __init__(self, source_node: str, cause: BaseException, recoverable_state: Any) -> None:
        super().__init__(f"edge from {source_node!r} raised {type(cause).__name__}: {cause}")
        self.source_node = source_node
        self.recoverable_state = recoverable_state
        self.__cause__ = cause


class ReducerError(RuntimeGraphError):
    category = "reducer_error"

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

    def __init__(self, source_node: str, returned: object, recoverable_state: Any) -> None:
        super().__init__(
            f"conditional edge from {source_node!r} returned {returned!r}, "
            "which is neither a declared node nor END"
        )
        self.source_node = source_node
        self.returned = returned
        self.recoverable_state = recoverable_state


class StateValidationError(RuntimeGraphError):
    """State failed schema validation at a graph boundary.

    Per spec §4 this category does NOT carry recoverable_state — at entry there
    is no prior state to recover; at exit the failing state IS the final state.
    """

    category = "state_validation_error"

    def __init__(self, message: str, fields: list[str], cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.fields = fields
        if cause is not None:
            self.__cause__ = cause
