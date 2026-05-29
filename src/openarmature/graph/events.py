# Spec: realizes graph-engine §6 (started/completed event pair model
# from proposal 0005, v0.6.0). FanOutEventConfig is the fan-out node
# event payload added by proposal 0013 (v0.10.0).

"""Node-boundary observer events.

Each node attempt produces a started/completed event PAIR. The engine
dispatches the started event before invoking the wrapped node function
and the completed event after the reducer merge succeeds (with
``post_state`` populated) or after the node, reducer, or state
validation fails (with ``error`` populated).

Frozen dataclass; observers receive a snapshot, not a live handle.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from openarmature.observability.metadata import AttributeValue

from .errors import RuntimeGraphError
from .state import State

# Sentinel empty metadata mapping for events constructed without a
# live caller-metadata snapshot (test helpers, synthetic events).
# Read-only proxy keeps the default allocation-free.
_EMPTY_METADATA: MappingProxyType[str, AttributeValue] = MappingProxyType({})


# Spec: realizes observability §5.4 fan-out attributes via the
# event-payload mechanism added by proposal 0013 (v0.10.0). Backend
# observers cache ``parent_node_name`` off the fan-out node's
# started event and apply it on every per-instance span they
# synthesize (observability §5.4 mandates
# ``openarmature.fan_out.parent_node_name`` on per-instance spans).
@dataclass(frozen=True)
class FanOutEventConfig:
    """Resolved fan-out configuration carried on a fan-out node's
    own events.

    Fan-out node events carry the resolved configuration so backend
    observers can attribute the fan-out node span (``item_count`` /
    ``concurrency`` / ``error_policy``) and synthesize per-instance
    spans with the right ``parent_node_name``.

    Populated ONLY on ``started`` and ``completed`` events for a
    fan-out node itself (partition by node type, not event category;
    INCLUDES retried attempts of a fan-out node when retry middleware
    wraps it). All other events leave ``NodeEvent.fan_out_config``
    null.

    Field shapes:

    - ``item_count``: non-negative int. The resolved instance count
      (matches ``count_field`` value when configured; matches
      ``len(items_field)`` in items_field mode).
    - ``concurrency``: positive int OR ``None`` (unbounded). Zero or
      negative is rejected at config resolution time as
      ``fan_out_invalid_concurrency``. Backend mappings may translate
      ``None`` to a sentinel at the attribute layer (e.g.,
      ``openarmature.fan_out.concurrency = 0``); that translation is
      observer-internal, not engine-internal.
    - ``error_policy``: one of ``"fail_fast"`` or ``"collect"``.
    - ``parent_node_name``: the fan-out node's name in the parent
      graph. Carried here for caching by backend observers when
      attributing per-instance spans.

    All four fields MUST be present when ``fan_out_config`` is
    populated. Only ``concurrency`` is nullable.
    """

    item_count: int
    concurrency: int | None
    error_policy: str
    parent_node_name: str


# Spec: realizes graph-engine §6 NodeEvent (started/completed pair
# model from proposal 0005, v0.6.0). The ``checkpoint_saved`` phase
# is the Python shape for §10.8 save events (§10.8 SHOULDs an event
# emit but leaves the shape implementation-defined). ``fan_out_config``
# is the observability §5.4 / proposal 0013 (v0.10.0) addition.
@dataclass(frozen=True)
class NodeEvent:
    """A single node-boundary event delivered to observers.

    - ``phase`` is ``"started"`` (dispatched before the node runs) or
      ``"completed"`` (dispatched after the node returns or raises
      and the merge runs/fails). Each node attempt produces exactly
      one of each in that order. The engine ALSO dispatches a
      ``"checkpoint_saved"`` event on the same shape after a
      successful ``Checkpointer.save`` call; observers MUST opt in
      explicitly via ``phases={"checkpoint_saved"}`` to receive these
      (default subscription is ``{"started", "completed"}`` only, so
      legacy observers don't see them).
    - ``node_name`` is the name under which this node was registered
      in its immediate containing graph.
    - ``namespace`` is an ordered sequence of node names from the
      outermost graph down to this node. For a node in the outermost
      graph, ``namespace`` is ``(node_name,)``. For nested subgraphs,
      the chain extends.
    - ``step`` is a monotonically-increasing counter starting at 0,
      scoped to a single outermost invocation. Subgraph-internal nodes
      increment the same counter. The started/completed pair for one
      attempt share the same step.
    - ``pre_state`` is the state the node received, before reducer
      merge. Populated on both phases (identical across the pair).
    - ``post_state`` is the state after the node's partial update
      merged successfully. Populated only on ``completed`` events
      that succeeded.
    - ``error`` is the wrapped runtime error (``NodeException``,
      ``ReducerError``, or ``StateValidationError``) when the node
      failed. Populated only on ``completed`` events that failed.
    - ``parent_states`` carries one state snapshot per containing
      graph, outermost first; for a node in the outermost graph it's
      an empty tuple. Invariant:
      ``len(parent_states) == len(namespace) - 1``.
    - ``attempt_index`` is the 0-based index of this attempt among
      any retries. ``0`` for nodes not wrapped by retry middleware.
    - ``fan_out_index`` is the 0-based index of this fan-out instance
      among its siblings. ``None`` for nodes not inside a fan-out.
    - ``fan_out_config`` carries resolved fan-out configuration on
      events from a fan-out NODE itself. See
      :class:`FanOutEventConfig`. ``None`` on every other event.
    - ``branch_name`` is the non-empty string name of the
      parallel-branches branch this event came from. ``None`` for
      nodes outside any branch. Per graph-engine §6 / pipeline-
      utilities §11, the combination of ``namespace``,
      ``branch_name``, ``fan_out_index``, ``attempt_index``, and
      ``phase`` jointly uniquely identifies an event source.
      ``branch_name`` and ``fan_out_index`` are independent; both
      MAY be present when a branch's subgraph contains a fan-out
      (or a fan-out instance contains a parallel-branches node).

    Invariants:

    - On ``started`` events, ``post_state`` and ``error`` MUST both
      be ``None``.
    - On ``completed`` events, exactly one of ``post_state`` and
      ``error`` is populated.

    **Synthetic phases.** ``"checkpoint_saved"`` (pipeline-utilities
    §10.8) and ``"checkpoint_migrated"`` (proposal 0014 §6
    cross-ref) repurpose this dataclass for non-node events. Both
    are opt-in via ``phases={...}`` on observer registration;
    default subscriptions are ``{"started", "completed"}`` only, so
    legacy observers never see them. Conventions on synthetic
    events:

    - ``checkpoint_saved``: ``pre_state`` carries the saved
      post-merge state (still a real ``State`` instance for this
      phase), ``post_state`` is ``None``. ``step`` matches the
      saving node's step.
    - ``checkpoint_migrated``: ``step=-1`` (no graph-step
      sequencing; migrations run before any node fires).
      ``node_name="openarmature.checkpoint.migrate"`` and
      ``namespace=("openarmature.checkpoint.migrate",)`` are
      dotted-pseudo identifiers, not real node names. ``pre_state``
      carries a private ``_MigrationSummary`` dataclass with
      ``from_version`` / ``to_version`` / ``chain_length``, NOT a
      ``State`` instance. ``parent_states`` is the empty tuple.

    Because ``pre_state`` is no longer guaranteed to be a ``State``
    on the synthetic phases, its type is declared as ``Any`` and
    observer authors who subscribe to those phases MUST narrow
    per-phase before reading ``pre_state``.
    """

    node_name: str
    namespace: tuple[str, ...]
    step: int
    phase: Literal[
        "started",
        "completed",
        "checkpoint_saved",
        # Synthetic phase per spec §6 cross-ref in proposal 0014:
        # fires once at the start of a versioned resume to carry
        # the migration chain's metadata. ``pre_state`` on this
        # phase carries a ``_MigrationSummary`` (not a ``State``);
        # the field type stays permissive on this dataclass and
        # the OTel observer narrows defensively via ``isinstance``.
        "checkpoint_migrated",
    ]
    pre_state: Any
    post_state: State | None
    error: RuntimeGraphError | None
    parent_states: tuple[State, ...]
    attempt_index: int = 0
    fan_out_index: int | None = None
    fan_out_config: FanOutEventConfig | None = None
    # Per pipeline-utilities §11 / graph-engine §6 (proposal 0011):
    # optional non-empty string populated only on events from nodes
    # that execute inside a parallel-branches branch. The
    # combination of ``namespace``, ``branch_name``,
    # ``fan_out_index``, ``attempt_index``, and ``phase`` jointly
    # uniquely identifies an event source. ``branch_name`` and
    # ``fan_out_index`` are independent; both MAY be present
    # simultaneously when a branch's subgraph contains a fan-out
    # (and vice versa).
    branch_name: str | None = None
    # Per observability §5.3 + the coord-thread
    # ``clarify-subgraph-name-semantics`` resolution: chain of
    # compiled-subgraph identities parallel to the wrapper-depth
    # positions of ``namespace``. Index ``i`` is the identity for
    # the wrapper at ``namespace[i]`` (or ``None`` when that
    # wrapper has no tracked identity); chain length equals the
    # depth of wrapper nesting (always ``< len(namespace)`` since
    # the last element of ``namespace`` is the current node, not
    # a wrapper). Observers read by depth and emit it as
    # ``observation.metadata.subgraph_name`` (Langfuse) /
    # ``openarmature.subgraph.name`` (OTel), falling back to the
    # empty string when ``None`` per §5.3's "if the implementation
    # tracks one" clause.
    subgraph_identities: tuple[str | None, ...] = ()
    # Per observability §3.4 + §5.6 (proposal 0034): snapshot of the
    # caller-supplied invocation metadata at event-construction
    # time. The engine reads ``current_invocation_metadata()`` when
    # it constructs the event (in the engine task / node body's
    # Context); the observer reads from the snapshot on the event
    # rather than re-reading the ContextVar at observer time —
    # critical because the observer runs on the engine's
    # ``deliver_loop`` task whose Context is frozen at invoke time
    # (asyncio.create_task copies the parent Context at task
    # creation), so the live ContextVar value in the deliver_loop
    # would NOT reflect mid-invocation augmentations made by node
    # bodies running in the main engine task. Observers emit each
    # entry as ``openarmature.user.<key>`` (OTel, §5.6) /
    # ``metadata.<key>`` (Langfuse, §8.4.1+§8.4.2).
    caller_invocation_metadata: Mapping[str, AttributeValue] = field(default_factory=lambda: _EMPTY_METADATA)


# Spec: realizes observability §3.4 + graph-engine §6 augmentation
# event mechanism (proposal 0040). Emitted by
# ``set_invocation_metadata`` when called mid-invocation; carries the
# delta + the augmenting context's lineage identity so observers can
# resolve which of their open observations belong to the augmenting
# context's subtree and apply the entries in place.
@dataclass(frozen=True)
class MetadataAugmentationEvent:
    """A metadata-augmentation event delivered to observers.

    Emitted by :func:`openarmature.observability.metadata.set_invocation_metadata`
    when called mid-invocation. Carries:

    - ``entries``: the delta merged into the per-async-context
      invocation metadata mapping by the call. Read-only view.
    - ``namespace`` / ``attempt_index`` / ``fan_out_index`` /
      ``branch_name``: the four lineage fields that jointly identify
      the augmenting execution context (the calling node's identity
      tuple). When ``set_invocation_metadata`` is called from outside
      a node body, ``namespace`` is the empty tuple, ``attempt_index``
      is ``0``, and both ``fan_out_index`` and ``branch_name`` are
      ``None`` — the invocation-level identity.

    Distinct from :class:`NodeEvent` because there is no node phase,
    no pre/post state, and no error: this event reports a side-channel
    augmentation, not a node-attempt boundary. Per graph-engine §6 the
    event is NOT subject to the observer ``phases`` filter (which only
    governs ``NodeEvent`` phases); the delivery worker forwards it to
    every subscribed observer. Observers that handle it iterate their
    open observations whose lineage is an ancestor of (or equal to)
    the augmenting context's lineage and apply the entries as
    ``openarmature.user.<key>`` (OTel, §5.6) /
    ``metadata.<key>`` (Langfuse, §8.4.1+§8.4.2).
    """

    entries: Mapping[str, AttributeValue]
    namespace: tuple[str, ...]
    attempt_index: int = 0
    fan_out_index: int | None = None
    branch_name: str | None = None


__all__ = ["FanOutEventConfig", "MetadataAugmentationEvent", "NodeEvent"]
