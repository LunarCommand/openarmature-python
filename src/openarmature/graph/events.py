"""Node-boundary observer events.

Per spec v0.6.0 §6 (proposal 0005): each node attempt produces a
started/completed event PAIR. The engine dispatches the started event
before invoking the wrapped node function and the completed event after
the reducer merge succeeds (with `post_state` populated) or after the
node, reducer, or state validation fails (with `error` populated).

Frozen dataclass — observers receive a snapshot, not a live handle.
"""

from dataclasses import dataclass
from typing import Literal

from .errors import RuntimeGraphError
from .state import State


@dataclass(frozen=True)
class FanOutEventConfig:
    """Spec §6 + §5.4 (per spec proposal 0013, v0.10.0):
    fan-out node events carry the resolved configuration so backend
    observers can attribute the fan-out node span (``item_count`` /
    ``concurrency`` / ``error_policy``) and synthesize per-instance
    spans with the right ``parent_node_name``.

    Populated ONLY on ``started`` and ``completed`` events for a
    fan-out node itself (partition by node type, not event category —
    INCLUDES retried attempts of a fan-out node when retry middleware
    wraps it). All other events leave ``NodeEvent.fan_out_config``
    null.

    Field shapes:

    - ``item_count`` — non-negative int. The resolved instance count
      per pipeline-utilities §9 (matches ``count_field`` value when
      configured; matches ``len(items_field)`` in items_field mode).
    - ``concurrency`` — positive int OR ``None`` (unbounded). Per
      pipeline-utilities §9.2: zero or negative is rejected at config
      resolution time as ``fan_out_invalid_concurrency``. Backend
      mappings translate ``None`` to a sentinel at the attribute layer
      (e.g., ``openarmature.fan_out.concurrency = 0`` per
      observability §5.4) — that translation is observer-internal,
      not engine-internal.
    - ``error_policy`` — one of ``"fail_fast"`` or ``"collect"`` per
      pipeline-utilities §9.4.
    - ``parent_node_name`` — the fan-out node's name in the parent
      graph. Carried here for caching by backend observers when
      attributing per-instance spans (§5.4 mandates
      ``openarmature.fan_out.parent_node_name`` on per-instance spans;
      the engine surfaces the name once on the fan-out node's started
      event, the observer caches and applies on every per-instance
      span it synthesizes).

    All four fields MUST be present when ``fan_out_config`` is
    populated. Only ``concurrency`` is nullable.
    """

    item_count: int
    concurrency: int | None
    error_policy: str
    parent_node_name: str


@dataclass(frozen=True)
class NodeEvent:
    """A single node-boundary event delivered to observers.

    Per spec v0.6.0 §6:

    - `phase` is `"started"` (dispatched before the node runs) or
      `"completed"` (dispatched after the node returns or raises and the
      merge runs/fails). Each node attempt produces exactly one of each
      in that order. Per pipeline-utilities §10.8, the engine ALSO
      dispatches a `"checkpoint_saved"` event on the same shape after
      a successful Checkpointer.save call — observers MUST opt in
      explicitly via `phases={"checkpoint_saved"}` to receive these
      (default subscription is `{"started", "completed"}` only, so
      legacy observers don't see them).
    - `node_name` is the name under which this node was registered in its
      immediate containing graph.
    - `namespace` is an ordered sequence of node names from the outermost
      graph down to this node. For a node in the outermost graph,
      `namespace` is `(node_name,)`. For nested subgraphs, the chain
      extends.
    - `step` is a monotonically-increasing counter starting at 0, scoped
      to a single outermost-invocation. Subgraph-internal nodes increment
      the same counter. The started/completed pair for one attempt share
      the same step.
    - `pre_state` is the state the node received, before reducer merge.
      Populated on both phases (identical across the pair).
    - `post_state` is the state after the node's partial update merged
      successfully. Populated only on `completed` events that succeeded.
    - `error` is the wrapped runtime error (NodeException, ReducerError,
      or StateValidationError) when the node failed. Populated only on
      `completed` events that failed.
    - `parent_states` carries one state snapshot per containing graph,
      outermost first; for a node in the outermost graph it's an empty
      tuple. Invariant: `len(parent_states) == len(namespace) - 1`.
    - `attempt_index` is the 0-based index of this attempt among any
      retries. `0` for nodes not wrapped by retry middleware.
    - `fan_out_index` is the 0-based index of this fan-out instance among
      its siblings. `None` for nodes not inside a fan-out.
    - `fan_out_config` carries resolved fan-out configuration on events
      from a fan-out NODE itself (per spec proposal 0013, v0.10.0). See
      :class:`FanOutEventConfig`. ``None`` on every other event.

    Invariants:
    - On `started` events, `post_state` and `error` MUST both be None.
    - On `completed` events, exactly one of `post_state` and `error` is
      populated.
    """

    node_name: str
    namespace: tuple[str, ...]
    step: int
    phase: Literal["started", "completed", "checkpoint_saved"]
    pre_state: State
    post_state: State | None
    error: RuntimeGraphError | None
    parent_states: tuple[State, ...]
    attempt_index: int = 0
    fan_out_index: int | None = None
    fan_out_config: FanOutEventConfig | None = None


__all__ = ["FanOutEventConfig", "NodeEvent"]
