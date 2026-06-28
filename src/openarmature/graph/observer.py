"""Observer hooks: protocol, subscription, delivery queue, per-invocation context.

Each node attempt produces a started/completed event pair, and
observers register with an optional ``phases`` set so they can
subscribe to one phase or both. The graph never awaits
observer processing.

This module defines:

- `Observer`: the callable shape an observer satisfies.
- `SubscribedObserver`: pairs an `Observer` with the phase set it
  subscribes to. Public; users construct one directly when passing
  phase-filtered observers to `invoke(observers=...)`.
- `RemoveHandle`: returned by `CompiledGraph.attach_observer` so the
  caller can detach later.
- `_InvocationContext`: the cross-graph state threaded through one
  outermost-invocation, including any nested subgraphs. Carries the
  queue, observer chain (graph-attached, outermost → innermost) and the
  invocation-scoped observers, plus a shared step counter, namespace
  prefix, and parent-state stack.
- `_QueuedItem`: an event paired with its delivery observer list.
- `_dispatch`: enqueues an event for the worker to deliver.
- `deliver_loop`: the worker coroutine. Reads items from the queue and
  calls each observer in order, filtering by subscribed phase and
  isolating exceptions via `warnings.warn`.
"""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from .events import (
    EmbeddingEvent,
    EmbeddingFailedEvent,
    FailureIsolatedEvent,
    InvocationCompletedEvent,
    InvocationStartedEvent,
    LlmCompletionEvent,
    LlmFailedEvent,
    LlmRetryAttemptEvent,
    MetadataAugmentationEvent,
    NodeEvent,
    ToolCallEvent,
    ToolCallFailedEvent,
)
from .state import State

# Union of every event variant an Observer may receive. NodeEvent is
# the original §6 started/completed/checkpoint shape; the other
# variants are side-channel events that bypass the phase filter and
# reach every subscribed observer — MetadataAugmentationEvent
# (proposal 0040 mid-invocation metadata augmentation),
# InvocationStartedEvent / InvocationCompletedEvent (proposal 0043
# trace.input/output sourcing), LlmCompletionEvent (proposal 0049
# typed LLM provider call event, dispatched on every successful LLM
# completion), LlmFailedEvent (proposal 0058 typed LLM failure event,
# dispatched alongside the §7 exception when provider.complete raises),
# LlmRetryAttemptEvent (proposal 0050 per-attempt LLM span event,
# python-internal, dispatched once per in-call attempt under call-level
# retry to drive the per-attempt OTel span surface),
# and FailureIsolatedEvent (proposal 0050 §6.3 framework-emitted event,
# dispatched by FailureIsolationMiddleware when it catches an exception
# escaping the inner chain and substitutes a degraded partial update);
# and EmbeddingEvent / EmbeddingFailedEvent (proposal 0059 typed embedding
# provider call events, dispatched on every EmbeddingProvider.embed()).
ObserverEvent = (
    NodeEvent
    | MetadataAugmentationEvent
    | InvocationStartedEvent
    | InvocationCompletedEvent
    | LlmCompletionEvent
    | LlmFailedEvent
    | LlmRetryAttemptEvent
    | FailureIsolatedEvent
    | ToolCallEvent
    | ToolCallFailedEvent
    | EmbeddingEvent
    | EmbeddingFailedEvent
)


class Observer(Protocol):
    """The shape of a callable that receives observer events.

    `Observer` is a structural Protocol; any async callable matching the
    signature qualifies, no subclass required. Plain functions, bound
    methods, and class instances with `__call__` all work::

        async def log_observer(event: NodeEvent | MetadataAugmentationEvent) -> None:
            if isinstance(event, NodeEvent):
                print(event.node_name, event.phase)

        compiled.attach_observer(log_observer)

    Contract:

    - Observers MUST be async so the delivery queue can await each
      one and coordinate ordering. The graph itself never awaits
      observers.
    - Observers MUST NOT alter state, routing, or any other aspect
      of the graph run. Read-only side effects (logging, metrics,
      span emission) only.

    The event parameter is positional-only (`event, /`) so structural
    conformance doesn't pin you to that name; any of `event`, `_event`,
    `e`, etc. matches.

    The variants reaching observers are the :data:`ObserverEvent` members.
    The signature is that union; observers ``isinstance``-narrow on the
    first line and choose which variants they handle.

    - :class:`NodeEvent` — the started/completed/checkpoint phase
      events. Subject to the ``phases`` filter on
      :class:`SubscribedObserver`; observers whose phase set excludes
      ``event.phase`` do NOT receive it.
    - :class:`MetadataAugmentationEvent` — emitted by
      :func:`openarmature.observability.metadata.set_invocation_metadata`
      when called mid-invocation. Carries the augmenting context's
      lineage tuple (``namespace``, ``attempt_index``,
      ``fan_out_index``, ``branch_name``) so rich backends can update
      their open observations in place
      (``span.set_attribute(openarmature.user.<key>, v)`` for OTel,
      ``observation.update(metadata=...)`` for Langfuse). This variant
      is NOT subject to the ``phases`` filter — every
      subscribed observer sees it and isinstance-narrows to decide
      whether to act. Simple user observers typically early-return
      after ``isinstance(event, NodeEvent)`` checks.
    - :class:`InvocationStartedEvent` — emitted once per invocation
      before any node fires. Carries the engine-constructed
      ``initial_state`` so Trace-level backends (Langfuse) can
      populate ``trace.input`` via the three-lever decision tree. NOT
      subject to the ``phases`` filter; OTel-only
      observers ignore it via the isinstance gate.
    - :class:`InvocationCompletedEvent` — emitted once per invocation
      after the last node fires (on both the success path and the
      failure path). Carries ``final_state`` + a closed
      ``status: {"completed", "failed"}`` enum so Trace-level
      backends can populate ``trace.output``. NOT subject to the
      ``phases`` filter; OTel-only observers ignore it via the
      isinstance gate.
    - :class:`LlmCompletionEvent` — dispatched on every successful LLM
      provider call. Carries the typed identity / request / response
      field set for LLM-aware backends. NOT subject to the ``phases``
      filter; non-LLM observers ignore it via the isinstance gate.
    - :class:`LlmFailedEvent` — the failure-side counterpart,
      dispatched alongside the provider exception when an LLM call
      raises. NOT subject to the ``phases`` filter.
    - :class:`FailureIsolatedEvent` — dispatched by
      ``FailureIsolationMiddleware`` when it catches an exception and
      substitutes a degraded partial update. NOT subject to the
      ``phases`` filter.

    Optional ``prepare_sync`` extension
    -----------------------------------
    An observer MAY additionally define a synchronous method::

        def prepare_sync(self, event: NodeEvent, /) -> None: ...

    that the engine calls IN THE ENGINE TASK, BEFORE queueing the
    event for the async ``__call__``. This exists for observers that
    need to set up state (e.g., open a span and stash a handle in a
    ContextVar) that the engine itself must read synchronously
    before running the node body (otherwise logs emitted on the
    first line of the body wouldn't see the right span).

    ``prepare_sync`` is **opt-in via ``hasattr``**; no subclass or
    Protocol method required. Observers that don't define it skip
    the synchronous prep entirely; observers that do define it run
    only for ``"started"``-phase events, with errors warned-not-
    propagated (same isolation contract as the async path).
    ``prepare_sync`` is never invoked for
    :class:`MetadataAugmentationEvent` (the synchronous-prep contract
    is anchored on the ``started`` phase, which only ``NodeEvent``
    carries).
    """

    async def __call__(self, event: ObserverEvent, /) -> None: ...


# Per spec v0.6.0 §6: the two valid phase strings. Used as the default
# subscription set when a caller doesn't restrict by phase.
# Default subscription — what a bare ``Observer`` callable receives
# without an explicit ``phases`` argument. Stays ``{"started",
# "completed"}`` so legacy observers don't unexpectedly receive
# checkpoint events. Subscribing to ``"checkpoint_saved"`` is opt-in.
ALL_PHASES: frozenset[str] = frozenset({"started", "completed"})

# All phase values the engine produces (per spec graph-engine §6 +
# pipeline-utilities §10.8 + proposal 0014 §6 cross-ref). Used by
# the registration-time validator to reject typos like
# ``phases={"complete"}``.
#
# The two synthetic phases (``checkpoint_saved`` and
# ``checkpoint_migrated``) repurpose the ``NodeEvent`` shape for
# non-node events — see the ``NodeEvent`` docstring for conventions.
# Both are opt-in via explicit ``phases={...}``; the default
# subscription ``ALL_PHASES`` above is ``{"started", "completed"}``
# only, so legacy observers never receive them.
KNOWN_PHASES: frozenset[str] = frozenset({"started", "completed", "checkpoint_saved", "checkpoint_migrated"})


@dataclass(frozen=True)
class SubscribedObserver:
    """An observer paired with its phase subscription set.

    Observers register with an optional ``phases`` parameter naming
    the phase strings they want to receive. The default is
    ``ALL_PHASES``, historically named when there were only two
    phases; it now means "the default subscription"
    (``{"started", "completed"}``). The ``"checkpoint_saved"`` phase
    is opt-in: subscribe to it explicitly via
    ``phases={"checkpoint_saved"}`` (or include it in a custom set).
    ``KNOWN_PHASES`` is the full "every phase the engine can produce"
    set used by the registration-time validator.

    Empty phase sets are forbidden; passing one raises
    ``ValueError`` at registration time so misconfiguration surfaces
    immediately.

    Construct one of these directly when handing phase-filtered
    observers to ``CompiledGraph.invoke(observers=...)``. For the
    single-observer ``attach_observer`` path, pass ``phases=`` as a
    keyword argument and the engine wraps it for you.
    """

    observer: Observer
    phases: frozenset[str] = ALL_PHASES

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("phases must be non-empty")
        invalid = self.phases - KNOWN_PHASES
        if invalid:
            raise ValueError(f"unknown phase(s): {sorted(invalid)}; allowed: {sorted(KNOWN_PHASES)}")


def _coerce_subscribed(
    observer: Observer | SubscribedObserver,
    *,
    phases: Iterable[str] | None = None,
) -> SubscribedObserver:
    """Normalize a registration argument into a `SubscribedObserver`.

    - A bare `Observer` callable becomes a `SubscribedObserver` with
      either the supplied `phases` or `ALL_PHASES` (the default
      subscription, `{"started", "completed"}`; subscribing to
      `"checkpoint_saved"` is opt-in via an explicit ``phases``).
    - An existing `SubscribedObserver` passes through unchanged; supplying
      a `phases` kwarg in that case is a misuse and raises.
    """
    if isinstance(observer, SubscribedObserver):
        if phases is not None:
            raise ValueError("cannot override phases on a SubscribedObserver; construct a new one")
        return observer
    return SubscribedObserver(
        observer=observer,
        phases=frozenset(phases) if phases is not None else ALL_PHASES,
    )


@dataclass(frozen=True)
class RemoveHandle:
    """Returned by ``CompiledGraph.attach_observer``. Call
    ``.remove()`` to detach the observer. Idempotent: calling
    ``.remove()`` after the observer is already detached is a no-op.

    Changes to the registered observer set during a graph run do NOT
    take effect until the next invocation.
    """

    _observers: list[SubscribedObserver]
    _observer: SubscribedObserver

    def remove(self) -> None:
        """Detach the observer from its compiled graph. Idempotent: a
        second call is a no-op rather than an error. The change takes
        effect on the next ``invoke()``; in-flight invocations keep
        the observer set they started with."""
        try:
            self._observers.remove(self._observer)
        except ValueError:
            # Idempotency: the observer is already detached. Per the
            # docstring, a second .remove() call is a no-op rather than
            # an error.
            pass


@dataclass(frozen=True)
class _QueuedItem:
    """An event paired with the exact ordered observer list that should
    receive it. The list is computed at dispatch time so events from
    different depths in nested subgraphs carry the correct observer chain
    without the worker needing to know the graph topology.

    ``event`` is the union of ``NodeEvent`` (started / completed /
    checkpoint phases), ``MetadataAugmentationEvent`` (side-channel
    augmentation), and the two invocation-boundary events
    ``InvocationStartedEvent`` / ``InvocationCompletedEvent``
    (Trace-level input/output sourcing). The delivery
    worker branches by type to apply the right delivery contract
    (phase-filter for ``NodeEvent``, no filter for the other three).
    """

    event: ObserverEvent
    observers: tuple[SubscribedObserver, ...]


# A sentinel value the engine puts on the queue to signal the worker to
# return after draining the events ahead of it. None is unambiguous —
# the queue carries ``NodeEvent``, ``MetadataAugmentationEvent``, and
# the two ``Invocation*Event`` variants wrapped in ``_QueuedItem``,
# never None.
_DRAIN_SENTINEL = None


# Spec: realizes graph-engine §6 Drain undelivered-count bookkeeping
# (proposal 0010). Per-invocation mutable counters; `_dispatch` bumps
# `dispatched` after a successful `queue.put_nowait`; `deliver_loop`
# bumps `delivered` after the per-event observer for-loop completes.
# `undelivered = dispatched - delivered` at any point in time — and
# specifically at `CompiledGraph.drain()` cancellation time when the
# timeout has elapsed and pending workers' counters get summed into
# the returned `DrainSummary`.
@dataclass
class _DrainCounters:
    dispatched: int = 0
    delivered: int = 0
    # Per spec graph-engine §6 *Per-invocation drain* (proposal 0054):
    # ``drain_events_for(invocation_id, *, timeout)`` callers register
    # ``(target_delivered_count, Future)`` pairs here; the deliver
    # loop fulfils any whose target has been reached after each
    # ``delivered`` increment. The list is touched only from the
    # single event-loop task running ``deliver_loop`` plus the
    # caller of ``drain_events_for`` — no cross-thread access — so a
    # plain list is sufficient.
    drain_wakers: list[tuple[int, asyncio.Future[None]]] = field(
        default_factory=list[tuple[int, asyncio.Future[None]]]
    )


# Spec: realizes graph-engine §6 Drain summary return shape (proposal
# 0010). The two declared fields are the spec-mandated minimum;
# implementations MAY add richer detail in future PRs (per-observer
# counts, sampled event metadata) without breaking the v0.19.0 shape.
@dataclass(frozen=True)
class DrainSummary:
    """Outcome of a `CompiledGraph.drain()` call.

    Returned from `drain()` regardless of whether a `timeout` was
    supplied. When no timeout was supplied, or the timeout did not
    fire, `undelivered_count == 0` and `timeout_reached is False`.
    When the timeout fired, `undelivered_count` reports the number of
    events that were dispatched to the delivery worker but not fully
    delivered to every subscribed observer before cancellation, and
    `timeout_reached is True`.

    These two fields are the required minimum. Implementations MAY
    extend the shape with diagnostic detail (per-observer counts,
    sampled event metadata) in subsequent versions; this version ships
    the minimum.
    """

    undelivered_count: int
    timeout_reached: bool


# Spec: realizes pipeline-utilities §10.11 per-instance progress
# tracking in the engine. These are the MUTABLE internal-state
# counterparts to the FROZEN public ``FanOutProgress`` /
# ``FanOutInstanceProgress`` shapes the saved CheckpointRecord exposes.
# ``_maybe_save_checkpoint`` projects this mutable state into the
# frozen public shape when building a record.
@dataclass
class _FanOutInstanceState:
    """Mutable per-instance state inside a fan-out, updated by the
    engine as the instance progresses. ``state`` transitions
    not_started -> in_flight -> completed.

    - ``result`` holds the per-instance contribution to the fan-out
      accumulator, set when ``state == "completed"``: "the value
      contributed to the ``target_field`` bucket" (success path) or
      "the error entry contributed to the
      ``errors_field`` bucket" (collect-mode failure). The harness
      projects this into the frozen ``FanOutInstanceProgress.result``
      verbatim.
    - ``result_is_error`` distinguishes success contributions
      (``False``) from collect-mode error contributions (``True``).
      Internal flag — not exposed on the public
      ``FanOutInstanceProgress`` shape because ``result`` is exposed
      as a single typed entry per the parent state schema.
      ``FanOutNode.run_with_context`` consults this on resume to
      route the rolled-forward contribution through the
      ``errors_field`` bucket rather than ``target_field``.
    - ``extra_outputs`` holds the per-instance values for the fan-out's
      ``extra_outputs`` mapping (parent-field -> sub-field) so that
      per-instance resume preserves the FULL per-instance contribution
      (not just the ``target_field`` slice). Internal — not exposed on
      the public ``FanOutInstanceProgress`` shape because ``result``
      is a single accumulator entry.
    - ``completed_inner_positions`` accumulates ``NodePosition`` entries
      from inner nodes that complete inside this instance's subgraph
      execution. Captures the instance's progress for observational
      purposes when an in_flight save snapshot fires; not used as a
      resume re-entry point (the instance re-enters at its subgraph's
      declared entry node).
    """

    state: Literal["completed", "in_flight", "not_started"] = "not_started"
    result: Any = None
    result_is_error: bool = False
    extra_outputs: dict[str, Any] = field(default_factory=dict[str, Any])
    completed_inner_positions: list[Any] = field(default_factory=list[Any])  # list[NodePosition]


@dataclass
class _FanOutExecutionState:
    """Mutable per-fan-out execution state. One entry per in-flight
    fan-out node in the invocation; lives on
    ``_InvocationContext.fan_out_progress_state`` keyed by
    ``(namespace, fan_out_node_name)``. The namespace component
    disambiguates same-named fan-outs in different subgraph descents.
    """

    fan_out_node_name: str
    namespace: tuple[str, ...]
    instance_count: int
    instances: list[_FanOutInstanceState]


@dataclass
class _InvocationContext:
    """Per-invocation state threaded through the engine and into subgraphs.

    Mutable: the step counter increments. The observer chain extends as
    subgraphs are entered. New child contexts are produced via
    `descend_into_subgraph` and share the same queue + step counter; the
    namespace and parent-state stacks are extended by-value.
    """

    queue: asyncio.Queue[_QueuedItem | None]
    # Graph-attached observers in delivery order: outermost graph first,
    # nested subgraph attached observers appended as we descend.
    graph_attached: tuple[SubscribedObserver, ...]
    # Set once at the outermost invoke; carried unchanged into subgraphs.
    invocation_scoped: tuple[SubscribedObserver, ...]
    # Shared mutable single-element list — a simple way to share an int by
    # reference across recursive subgraph contexts without leaking a class.
    step_counter: list[int] = field(default_factory=lambda: [0])
    namespace_prefix: tuple[str, ...] = ()
    parent_states_prefix: tuple[State, ...] = ()
    # Per observability §5.3 + the coord-thread `clarify-subgraph-name-
    # semantics` resolution. Parallel to ``namespace_prefix`` — index
    # ``i`` is the compiled-subgraph identity for the wrapper at
    # ``namespace_prefix[i]``, or ``None`` for wrappers constructed
    # without an identity. Used by observers to emit
    # ``metadata.subgraph_name`` (Langfuse) and
    # ``openarmature.subgraph.name`` (OTel) on the wrapper observation
    # / span at each depth. The chain shape lets nested subgraphs
    # carry distinct identities at distinct depths even though
    # v0.10.0's conformance fixtures only exercise single-level
    # nesting.
    subgraph_identities: tuple[str | None, ...] = ()
    # Per pipeline-utilities §9 + graph-engine §6: nodes inside a
    # fan-out instance fire events tagged with the instance's 0-based
    # index. Set when descending into a fan-out instance, inherited
    # unchanged through any further subgraph descents inside that
    # instance, and absent (None) for nodes outside any fan-out.
    fan_out_index: int | None = None

    # Per proposal 0045 (v0.37.0): per-depth lineage chains.  Mirror
    # ``namespace_prefix`` depth — position ``i`` is the
    # fan_out_index (resp. branch_name) for the dispatch boundary
    # at namespace depth ``i+1``, or ``None`` if that boundary is a
    # subgraph wrapper / serial node (not a fan-out, not a
    # parallel-branches branch).  The chains are extended by one
    # entry at every ``descend_into_*`` call; the engine then drives
    # the chain ContextVars from these fields at every node-execution
    # site so ``set_invocation_metadata`` sees the full chain.
    fan_out_index_chain: tuple[int | None, ...] = ()
    branch_name_chain: tuple[str | None, ...] = ()

    # ----------------------------------------------------------------
    # Checkpointing fields (spec pipeline-utilities §10)
    #
    # ``invocation_id`` and ``correlation_id`` are minted once at the
    # outermost ``invoke`` call (or restored from a saved record on
    # resume) and propagated unchanged through every descent. The
    # checkpointer reference is set when a backend is registered; it
    # is intentionally **None inside fan-out instances** so per-instance
    # internal saves are gated off (§10.7 atomic-restart). The mutable
    # ``completed_positions`` list is shared across descents so the
    # save call sites can append the just-completed position before
    # the engine's next step. ``resume_skip_set`` is a frozen set of
    # namespace tuples whose corresponding nodes have already
    # completed in a prior run and MUST be skipped on this resumed
    # invocation.
    # ----------------------------------------------------------------
    invocation_id: str = ""
    correlation_id: str = ""
    checkpointer: Any = None  # Checkpointer | None; typed Any to avoid an
    # import cycle between graph and checkpoint packages.
    completed_positions: list[Any] = field(default_factory=list[Any])  # list[NodePosition]
    resume_skip_set: frozenset[tuple[str, ...]] = field(default_factory=lambda: frozenset[tuple[str, ...]]())
    # The invocation_id we LOADED FROM on a resumed run — distinct from
    # ``invocation_id`` (the freshly-minted id for this resumed run per
    # §10.4 step 4). ``None`` outside the resume path. Threaded through
    # so inner-descent state-validation failures can populate
    # CheckpointRecordInvalid with the source record's id.
    resume_invocation: str | None = None
    # Resume-with-saved-inner-state plumbing: when the loaded record's
    # latest save fired from inside a subgraph (parent_states populated),
    # the engine restores the OUTER state from parent_states[0] but ALSO
    # needs the saved inner state(s) when re-descending into the
    # in-flight subgraph(s). This map is keyed by descent depth — depth
    # 1 = first subgraph level, depth 2 = nested two deep, etc. The
    # subgraph descent path consumes (pops) its matching depth before
    # falling back to the normal projection. After consumption, fresh
    # descents at the same depth project as usual. Shared mutable dict
    # propagates across descents.
    pending_resume_states: dict[int, Any] = field(default_factory=dict[int, Any])
    # Per spec §10.11: mutable per-fan-out progress tracking. Keyed by
    # ``(namespace, fan_out_node_name)`` — disambiguates same-named
    # fan-outs in different subgraph descents. ``FanOutNode`` populates
    # entries before descending into instances; updates state as
    # instances progress; the entry stays in the dict for the duration
    # of the fan-out so concurrent saves see consistent sibling state.
    # ``_maybe_save_checkpoint`` projects this into the frozen
    # ``FanOutProgress`` shape on the saved CheckpointRecord.
    # Keyed by (namespace, fan_out_node_name, enclosing_fan_out_instance_lineage)
    # -- the lineage (non-None outer fan_out_index chain) disambiguates a fan-out
    # nested inside an outer fan-out instance across concurrent outer instances.
    fan_out_progress_state: dict[tuple[tuple[str, ...], str, tuple[int, ...]], _FanOutExecutionState] = field(
        default_factory=dict[tuple[tuple[str, ...], str, tuple[int, ...]], _FanOutExecutionState]
    )
    # Per spec §6 Drain (proposal 0010): shared mutable counters that
    # the worker reads at drain-cancel time to report undelivered events
    # in the returned ``DrainSummary``. Subgraphs share the parent's
    # counters because subgraphs share the parent's queue + worker, so
    # the parent context's counts naturally cover subgraph events.
    drain_counters: _DrainCounters = field(default_factory=_DrainCounters)
    # Per spec §10.2 (proposal 0028): the canonical source for
    # ``CheckpointRecord.schema_version``. Set once at the outermost
    # ``invoke`` to the compiled graph's declared state class
    # (``CompiledGraph.state_cls``); propagated unchanged through every
    # descent (subgraphs, fan-out instances, parallel branches). All
    # save sites within an invocation MUST read ``schema_version`` from
    # this class — NOT from ``type(state)`` at save time — so the
    # value is consistent across the outer dispatch save, fan-out
    # instance internal saves, and the fan-out node's own completion
    # save. The distinction matters only when a user passes a State
    # subclass that shadows ``schema_version``; the declared class is
    # the only consistent choice for §10.12 migration lookups.
    # ``Any`` rather than ``type[State]`` to avoid an import cycle
    # between graph and observer; callers narrow at the read site.
    state_cls: Any = None
    # Per proposal 0043 (observability §8.4.1 trace.output sourcing):
    # shared mutable single-element box tracking the most recently
    # entered node's name. The outermost ``invoke()`` reads it on
    # exit to populate ``InvocationCompletedEvent.final_node`` on
    # both the success path (last node before END routing) and the
    # failure path (the node whose execution raised). Shared by
    # reference across subgraph / fan-out / parallel-branches
    # descents so the inner-most node's name wins on failure (the
    # real culprit, not the wrapper).
    final_node_box: list[str] = field(default_factory=list[str])
    # Per proposal 0043 (observability §8.4.1 *Resume semantics* +
    # "partial final state captured at the failure point" clause).
    # Tracks the most recent successful step's post-merge state at THIS
    # context level so the outermost ``invoke()`` can populate
    # ``InvocationCompletedEvent.final_state`` on the failure path with
    # the partial outer state, not the bare ``starting_state``. On the
    # success path the box is unused — the engine's return value is the
    # canonical ``final_state``. **Distinct from ``final_node_box``**:
    # the latest-state box is per-level (each subgraph / fan-out
    # instance / parallel-branches branch gets its own fresh box),
    # because the OUTER Langfuse trace cares about the outer-graph's
    # state type, and an inner state has a different type. The
    # ``final_node_box`` shares by reference because the spec wants the
    # innermost failing node's name (the real culprit); state has the
    # opposite contract — the outermost level's state is what the
    # outer trace.output hook receives.
    latest_state_box: list[Any] = field(default_factory=list[Any])

    def full_observers(self) -> tuple[SubscribedObserver, ...]:
        """Return the ordered observer list to deliver for events from
        this depth: graph-attached (outermost → innermost), then
        invocation-scoped (passed to the outermost invoke)."""
        return self.graph_attached + self.invocation_scoped

    def descend_into_subgraph(
        self,
        subgraph_node_name: str,
        parent_state: State,
        sub_attached: tuple[SubscribedObserver, ...],
        *,
        subgraph_identity: str | None = None,
    ) -> _InvocationContext:
        """Build the context for a subgraph-as-node call.

        The returned context shares the queue and step counter (so step
        numbering is monotonic across the boundary) but has an extended
        namespace prefix, parent-state stack, and graph-attached observer
        chain. Invocation-scoped observers carry through unchanged.
        ``fan_out_index`` is inherited so a subgraph descent inside a
        fan-out instance still tags inner events with the index.

        Checkpointing fields propagate unchanged: subgraph-internal
        nodes save to the same backend with the same invocation_id
        (one save per inner-node completion).
        """
        return _InvocationContext(
            queue=self.queue,
            graph_attached=self.graph_attached + sub_attached,
            invocation_scoped=self.invocation_scoped,
            step_counter=self.step_counter,
            namespace_prefix=self.namespace_prefix + (subgraph_node_name,),
            parent_states_prefix=self.parent_states_prefix + (parent_state,),
            subgraph_identities=self.subgraph_identities + (subgraph_identity,),
            fan_out_index=self.fan_out_index,
            # Per proposal 0045: subgraph wrappers don't add a
            # fan-out or branch axis — extend both chains by
            # ``None`` at this depth.
            fan_out_index_chain=self.fan_out_index_chain + (None,),
            branch_name_chain=self.branch_name_chain + (None,),
            invocation_id=self.invocation_id,
            correlation_id=self.correlation_id,
            checkpointer=self.checkpointer,
            completed_positions=self.completed_positions,
            resume_skip_set=self.resume_skip_set,
            pending_resume_states=self.pending_resume_states,
            resume_invocation=self.resume_invocation,
            fan_out_progress_state=self.fan_out_progress_state,
            drain_counters=self.drain_counters,
            state_cls=self.state_cls,
            final_node_box=self.final_node_box,
            # latest_state_box is INTENTIONALLY NOT propagated — each
            # context level tracks its own outer-state-typed latest
            # successful step. See the field docstring above.
        )

    def descend_into_fan_out_instance(
        self,
        fan_out_node_name: str,
        parent_state: State,
        sub_attached: tuple[SubscribedObserver, ...],
        fan_out_index: int,
        *,
        subgraph_identity: str | None = None,
    ) -> _InvocationContext:
        """Build the context for one fan-out instance's subgraph invocation.

        Same shape as ``descend_into_subgraph`` but stamps the fan-out
        index onto the new context so every inner-node event carries it.
        The index is the instance's 0-based position.

        Fan-out instance internal nodes DO produce checkpoint saves.
        The
        checkpointer reference propagates unchanged so an inner node's
        ``completed`` event triggers a save; the engine's save path
        projects the shared ``fan_out_progress_state`` into the record's
        per-instance progress field. ``resume_skip_set`` is dropped:
        inner-position skipping is governed by the per-instance
        ``completed_inner_positions`` field on the loaded record's
        ``fan_out_progress`` entry, not by the outer skip-set (which
        would conflate inner and outer positions otherwise).
        """
        return _InvocationContext(
            queue=self.queue,
            graph_attached=self.graph_attached + sub_attached,
            invocation_scoped=self.invocation_scoped,
            step_counter=self.step_counter,
            namespace_prefix=self.namespace_prefix + (fan_out_node_name,),
            parent_states_prefix=self.parent_states_prefix + (parent_state,),
            subgraph_identities=self.subgraph_identities + (subgraph_identity,),
            fan_out_index=fan_out_index,
            # Per proposal 0045: fan-out instance descent extends the
            # fan_out_index chain with the instance's index; the
            # branch chain extends with ``None`` (no branch axis here).
            fan_out_index_chain=self.fan_out_index_chain + (fan_out_index,),
            branch_name_chain=self.branch_name_chain + (None,),
            invocation_id=self.invocation_id,
            correlation_id=self.correlation_id,
            checkpointer=self.checkpointer,
            completed_positions=self.completed_positions,
            resume_skip_set=frozenset(),
            pending_resume_states={},
            resume_invocation=self.resume_invocation,
            # Propagate the shared per-fan-out tracking dict so an
            # inner-instance node can update its own entry and so the
            # outer save sees consistent sibling state.
            fan_out_progress_state=self.fan_out_progress_state,
            drain_counters=self.drain_counters,
            state_cls=self.state_cls,
            final_node_box=self.final_node_box,
            # latest_state_box is INTENTIONALLY NOT propagated — each
            # context level tracks its own outer-state-typed latest
            # successful step. See the field docstring above.
        )

    def descend_into_parallel_branch(
        self,
        parallel_branches_node_name: str,
        parent_state: State,
        sub_attached: tuple[SubscribedObserver, ...],
        *,
        branch_name: str,
    ) -> _InvocationContext:
        """Build the context for one parallel-branches branch's
        subgraph invocation.

        The parallel-branches node looks to outer middleware like a
        single dispatch; inner-branch
        events come from the branch's subgraph execution. Stamps the
        namespace prefix with the parallel-branches node name so
        inner events nest under it (mirrors
        ``descend_into_fan_out_instance``'s namespace stamping).

        Branch identity (the SCALAR innermost branch_name) lives on
        the ``observability.correlation._branch_name_var`` ContextVar
        — set inside the branch's task closure so ``copy_context``
        inherits it through the subgraph's execution.  The PER-DEPTH
        ``branch_name_chain`` is extended here on the
        context so the engine can drive the chain ContextVar at
        every inner-node execution site.

        Atomic-restart: drops the checkpointer
        and pending_resume_states (a crash mid-dispatch re-runs the
        whole parallel-branches node from scratch on resume; the
        branches' inner saves wouldn't be useful).
        """
        return _InvocationContext(
            queue=self.queue,
            graph_attached=self.graph_attached + sub_attached,
            invocation_scoped=self.invocation_scoped,
            step_counter=self.step_counter,
            namespace_prefix=self.namespace_prefix + (parallel_branches_node_name,),
            parent_states_prefix=self.parent_states_prefix + (parent_state,),
            # Parallel-branches don't reify a single inner subgraph
            # identity at the wrapper position — each branch can hold a
            # different subgraph — so we extend the chain with ``None``
            # at this depth. Per-branch identity handling (if ever
            # needed) is a future addition.
            subgraph_identities=self.subgraph_identities + (None,),
            fan_out_index=self.fan_out_index,
            # Per proposal 0045: parallel-branches branch descent
            # extends the branch chain with this branch's name; the
            # fan_out_index chain extends with ``None`` (no fan-out
            # axis here).
            fan_out_index_chain=self.fan_out_index_chain + (None,),
            branch_name_chain=self.branch_name_chain + (branch_name,),
            invocation_id=self.invocation_id,
            correlation_id=self.correlation_id,
            checkpointer=None,
            completed_positions=self.completed_positions,
            resume_skip_set=frozenset(),
            pending_resume_states={},
            resume_invocation=self.resume_invocation,
            fan_out_progress_state=self.fan_out_progress_state,
            drain_counters=self.drain_counters,
            state_cls=self.state_cls,
            final_node_box=self.final_node_box,
            # latest_state_box is INTENTIONALLY NOT propagated — each
            # context level tracks its own outer-state-typed latest
            # successful step. See the field docstring above.
        )

    def take_step(self) -> int:
        """Atomically (single-threaded asyncio) read-and-increment the
        shared step counter. Returns the value to assign to the just-
        executed node's event."""
        n = self.step_counter[0]
        self.step_counter[0] = n + 1
        return n


def _dispatch(
    context: _InvocationContext,
    event: ObserverEvent,
) -> None:
    """Enqueue an event for the delivery worker.

    Handles the :data:`ObserverEvent` variants. The principal ones:

    - :class:`NodeEvent`: the started/completed/checkpoint pair model.
      For ``"started"``-phase events, also calls any subscribed
      observer's optional ``prepare_sync(event)`` synchronously — in
      the engine task, BEFORE queueing — so observers that need to
      publish per-event state the engine itself reads in the same
      engine-task scope (e.g., the OTel observer setting
      ``current_active_observer_span`` for the engine to attach into
      the OTel context) can do so before the node body runs.
    - :class:`MetadataAugmentationEvent`: a side-channel augmentation
      event emitted by
      ``set_invocation_metadata`` mid-invocation. Bypasses the
      ``prepare_sync`` branch entirely — the sync-prep contract is
      anchored on ``"started"``, which only ``NodeEvent`` carries.
      Queued onto the same serial worker so observers see it in
      strict order with the surrounding node events.
    - :class:`InvocationStartedEvent` /
      :class:`InvocationCompletedEvent`: invocation-boundary events the
      engine enqueues at invocation entry / exit so Trace-level
      backends can populate ``trace.input`` / ``trace.output`` via the
      three-lever decision tree.
      Bypass ``prepare_sync`` (same rationale as
      ``MetadataAugmentationEvent``: not a node-phase event).

    Phase-gated forwarding: ``prepare_sync`` only fires when ``"started"``
    is in the subscribed observer's ``phases`` set, mirroring how the
    async ``deliver_loop`` filters dispatch. A user who explicitly
    subscribes only to ``{"completed"}`` doesn't get the synchronous
    prep — the wrapper acts as a uniform phase shield across both
    sync prep and async dispatch.

    Errors from ``prepare_sync`` follow the same isolation contract
    as the async path: don't propagate, don't break siblings, don't
    block the queueing or subsequent events. Reported via
    ``warnings.warn``.

    No-op when no observers exist for this depth — avoids paying the queue
    overhead for graphs that don't observe anything.
    """
    observers = context.full_observers()
    if not observers:
        return
    if isinstance(event, NodeEvent) and event.phase == "started":
        for subscribed in observers:
            if "started" not in subscribed.phases:
                continue
            prepare_sync = getattr(subscribed.observer, "prepare_sync", None)
            if prepare_sync is None:
                continue
            try:
                result = prepare_sync(event)
            except Exception as e:
                warnings.warn(
                    f"observer prepare_sync raised {type(e).__name__}: {e}",
                    stacklevel=2,
                )
                continue
            if inspect.isawaitable(result):
                # ``prepare_sync`` is opt-in via ``hasattr`` (not a
                # Protocol method) so pyright can't catch a user's
                # ``async def prepare_sync`` signature drift up front.
                # The call here would silently return an unawaited
                # coroutine — the prep work wouldn't run AND Python
                # would emit a delayed "coroutine was never awaited"
                # warning at GC time. Close the awaitable to suppress
                # that secondary noise and surface the misconfiguration
                # via our own explicit warn so it fails loudly at the
                # call site. ``getattr`` rather than ``hasattr``+method
                # access keeps pyright's strict-mode happy on the
                # ``Awaitable`` type (``.close`` lives on
                # ``Coroutine``, not the broader ``Awaitable``).
                close_method = getattr(result, "close", None)
                if close_method is not None:
                    try:
                        close_method()
                    except Exception as close_error:
                        # Cleanup is best-effort: a raise here MUST NOT
                        # propagate or block sibling observers. Surface
                        # via ``warnings.warn`` so the swallow is at
                        # least observable if it ever fires (CodeQL
                        # py/empty-except clears on this surface too).
                        warnings.warn(
                            f"observer prepare_sync close cleanup raised "
                            f"{type(close_error).__name__}: {close_error}",
                            stacklevel=2,
                        )
                warnings.warn(
                    f"observer prepare_sync returned an awaitable "
                    f"({type(result).__name__}); prepare_sync MUST be sync "
                    f"(define as `def`, not `async def`). The returned "
                    f"awaitable will not be awaited and is NOT guaranteed "
                    f"to complete before the node body starts; log "
                    f"correlation may miss this node's span.",
                    stacklevel=2,
                )
    context.queue.put_nowait(_QueuedItem(event=event, observers=observers))
    # Per spec §6 Drain (proposal 0010): increment AFTER the put so a
    # raise from ``put_nowait`` (queue full on a bounded queue — we
    # don't bound, but the invariant holds) doesn't desync the counter.
    context.drain_counters.dispatched += 1


async def deliver_loop(
    queue: asyncio.Queue[_QueuedItem | None],
    counters: _DrainCounters,
) -> None:
    """Background worker: read queued events, deliver to observers serially.

    - No two observers receive the same event concurrently (we await
      each).
    - No observer receives event N+1 until everyone has finished N
      (the loop processes one item fully before pulling the next).
    - For :class:`NodeEvent`, observers whose ``phases`` set excludes
      the event's phase do NOT receive it. Phase filter applies at
      delivery, not dispatch; the engine still produces both events
      for every attempt.
    - For :class:`MetadataAugmentationEvent` and the two
      invocation-boundary events :class:`InvocationStartedEvent` /
      :class:`InvocationCompletedEvent`, the ``phases`` filter is
      bypassed entirely — none of those are
      node-phase events, so every subscribed observer receives them
      regardless of ``phases``. Observers ``isinstance``-narrow on
      the first line and choose whether to act.
    - Observer exceptions don't propagate, don't break siblings,
      don't block subsequent events. Reported via ``warnings.warn``.

    The loop terminates when it receives ``_DRAIN_SENTINEL`` (None).
    """
    while True:
        item = await queue.get()
        if item is None:
            return
        event = item.event
        for subscribed in item.observers:
            if isinstance(event, NodeEvent) and event.phase not in subscribed.phases:
                continue
            try:
                await subscribed.observer(event)
            except Exception as e:
                warnings.warn(
                    f"observer raised {type(e).__name__}: {e}",
                    stacklevel=1,
                )
        # Per spec §6 Drain (proposal 0010): increment AFTER the
        # observer for-loop completes for this event, so an event
        # cancelled mid-for-loop is counted as undelivered
        # (``dispatched - delivered`` includes it). The phase-filter
        # ``continue`` above does NOT skip the increment — an event
        # filtered out for every observer is still considered
        # delivered (we did all the work there was to do for it).
        counters.delivered += 1
        # Per spec §6 *Per-invocation drain* (proposal 0054): wake any
        # ``drain_events_for`` waiter whose ``target_delivered_count``
        # has been reached. Mutate the list in place; the only other
        # toucher is ``drain_events_for`` itself, running in the same
        # event-loop task family. The ``not fut.done()`` guard absorbs
        # the case where the waiter's own ``asyncio.wait_for`` timed
        # out and cancelled the Future before the deliver loop got
        # here.
        if counters.drain_wakers:
            still_pending: list[tuple[int, asyncio.Future[None]]] = []
            for target, fut in counters.drain_wakers:
                if counters.delivered >= target:
                    if not fut.done():
                        fut.set_result(None)
                    continue
                still_pending.append((target, fut))
            counters.drain_wakers = still_pending


__all__ = [
    "ALL_PHASES",
    "DrainSummary",
    "Observer",
    "ObserverEvent",
    "RemoveHandle",
    "SubscribedObserver",
    # Engine-internal but listed so pyright sees them as exported (they're
    # imported by `compiled.py` and `subgraph.py`). The underscore prefix
    # is the user-facing "don't import these" signal.
    "_DRAIN_SENTINEL",
    "_DrainCounters",
    "_FanOutExecutionState",
    "_FanOutInstanceState",
    "_InvocationContext",
    "_QueuedItem",
    "_coerce_subscribed",
    "_dispatch",
    "deliver_loop",
]
