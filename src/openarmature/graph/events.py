"""Node-boundary observer events.

Per spec v0.3.0 §6 (proposal 0003): a NodeEvent is delivered to registered
observers once per node execution, carrying enough context to reconstruct
where in the (potentially nested) execution path the node sat and what the
state looked like before/after the node's update merged.

Frozen dataclass — observers receive a snapshot, not a live handle.
"""

from dataclasses import dataclass

from .errors import RuntimeGraphError
from .state import State


@dataclass(frozen=True)
class NodeEvent:
    """A single node-boundary event delivered to observers.

    Per spec v0.3.0 §6:

    - `node_name` is the name under which this node was registered in its
      immediate containing graph.
    - `namespace` is an ordered sequence of node names from the outermost
      graph down to this node. For a node in the outermost graph,
      `namespace` is `(node_name,)`. For nested subgraphs, the chain
      extends.
    - `step` is a monotonically-increasing counter starting at 0, scoped
      to a single outermost-invocation. Subgraph-internal nodes increment
      the same counter.
    - `pre_state` is the state the node received, before reducer merge.
    - `post_state` is the state after the node's partial update merged
      successfully. Populated only on success.
    - `error` is the wrapped runtime error (NodeException, ReducerError,
      or StateValidationError) when the node failed. Read `event.error.category`
      for the spec category identifier and `event.error.__cause__` for the
      original user/framework exception. Populated only on failure.
    - `parent_states` carries one state snapshot per containing graph,
      outermost first; for a node in the outermost graph it's an empty
      tuple. Invariant: `len(parent_states) == len(namespace) - 1`.

    Exactly one of `post_state` or `error` is populated per event.
    """

    node_name: str
    namespace: tuple[str, ...]
    step: int
    pre_state: State
    post_state: State | None
    error: RuntimeGraphError | None
    parent_states: tuple[State, ...]


__all__ = ["NodeEvent"]
