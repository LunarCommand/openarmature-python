"""Observer hooks: protocol, subscription, delivery queue, per-invocation context.

Each node attempt produces a started/completed event pair, and
observers register with an optional ``phases`` set so they can
subscribe to one phase or both. The graph never awaits
observer processing.

This module defines:

- `Observer`: the callable shape an observer satisfies.
- `SubscribedObserver`: pairs an `Observer` with the phase set it
  subscribes to. Public — users construct one directly when passing
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
  isolating exceptions via `warnings.warn` per spec.
"""

from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .events import NodeEvent
from .state import State


class Observer(Protocol):
    """The shape of a callable that receives node-boundary events.

    `Observer` is a structural Protocol — any async callable matching the
    signature qualifies, no subclass required. Plain functions, bound
    methods, and class instances with `__call__` all work::

        async def log_observer(event: NodeEvent) -> None:
            print(event.node_name, event.phase)

        compiled.attach_observer(log_observer)

    Contract:

    - Observers MUST be async so the delivery queue can await each
      one and coordinate ordering. The graph itself never awaits
      observers.
    - Observers MUST NOT alter state, routing, or any other aspect
      of the graph run — read-only side effects (logging, metrics,
      span emission) only.

    The event parameter is positional-only (`event, /`) so structural
    conformance doesn't pin you to that name — any of `event`, `_event`,
    `e`, etc. matches.

    Optional ``prepare_sync`` extension
    -----------------------------------
    An observer MAY additionally define a synchronous method::

        def prepare_sync(self, event: NodeEvent, /) -> None: ...

    that the engine calls IN THE ENGINE TASK, BEFORE queueing the
    event for the async ``__call__``. This exists for observers that
    need to set up state — e.g., open a span and stash a handle in
    a ContextVar — that the engine itself must read synchronously
    before running the node body (otherwise logs emitted on the
    first line of the body wouldn't see the right span).

    ``prepare_sync`` is **opt-in via ``hasattr``** — no subclass or
    Protocol method required. Observers that don't define it skip
    the synchronous prep entirely; observers that do define it run
    only for ``"started"``-phase events, with errors warned-not-
    propagated (same isolation contract as the async path).
    """

    async def __call__(self, event: NodeEvent, /) -> None: ...


# Per spec v0.6.0 §6: the two valid phase strings. Used as the default
# subscription set when a caller doesn't restrict by phase.
# Default subscription — what a bare ``Observer`` callable receives
# without an explicit ``phases`` argument. Stays ``{"started",
# "completed"}`` so legacy observers don't unexpectedly receive
# checkpoint events. Subscribing to ``"checkpoint_saved"`` is opt-in.
ALL_PHASES: frozenset[str] = frozenset({"started", "completed"})

# All phase values the engine produces (per spec graph-engine §6 +
# pipeline-utilities §10.8). Used by the registration-time validator
# to reject typos like ``phases={"complete"}``.
KNOWN_PHASES: frozenset[str] = frozenset({"started", "completed", "checkpoint_saved", "checkpoint_migrated"})


@dataclass(frozen=True)
class SubscribedObserver:
    """An observer paired with its phase subscription set.

    Observers register with an optional ``phases`` parameter naming
    the phase strings they want to receive. The default is
    ``ALL_PHASES`` — historically named when there were only two
    phases, now meaning "the default subscription"
    (``{"started", "completed"}``). The ``"checkpoint_saved"`` phase
    is opt-in: subscribe to it explicitly via
    ``phases={"checkpoint_saved"}`` (or include it in a custom set).
    ``KNOWN_PHASES`` is the full "every phase the engine can produce"
    set used by the registration-time validator.

    Empty phase sets are forbidden — passing one raises
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
      subscription — `{"started", "completed"}`; subscribing to
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
    ``.remove()`` to detach the observer. Idempotent — calling
    ``.remove()`` after the observer is already detached is a no-op.

    Changes to the registered observer set during a graph run do NOT
    take effect until the next invocation.
    """

    _observers: list[SubscribedObserver]
    _observer: SubscribedObserver

    def remove(self) -> None:
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
    """

    event: NodeEvent
    observers: tuple[SubscribedObserver, ...]


# A sentinel value the engine puts on the queue to signal the worker to
# return after draining the events ahead of it. None is unambiguous —
# observers receive `NodeEvent` instances, never None.
_DRAIN_SENTINEL = None


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
    # Per pipeline-utilities §9 + graph-engine §6: nodes inside a
    # fan-out instance fire events tagged with the instance's 0-based
    # index. Set when descending into a fan-out instance, inherited
    # unchanged through any further subgraph descents inside that
    # instance, and absent (None) for nodes outside any fan-out.
    fan_out_index: int | None = None

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

    def full_observers(self) -> tuple[SubscribedObserver, ...]:
        """Return the ordered observer list to deliver for events from
        this depth. Per spec §6: graph-attached (outermost → innermost),
        then invocation-scoped (passed to the outermost invoke)."""
        return self.graph_attached + self.invocation_scoped

    def descend_into_subgraph(
        self,
        subgraph_node_name: str,
        parent_state: State,
        sub_attached: tuple[SubscribedObserver, ...],
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
        (per spec §10.3 — one save per inner-node completion).
        """
        return _InvocationContext(
            queue=self.queue,
            graph_attached=self.graph_attached + sub_attached,
            invocation_scoped=self.invocation_scoped,
            step_counter=self.step_counter,
            namespace_prefix=self.namespace_prefix + (subgraph_node_name,),
            parent_states_prefix=self.parent_states_prefix + (parent_state,),
            fan_out_index=self.fan_out_index,
            invocation_id=self.invocation_id,
            correlation_id=self.correlation_id,
            checkpointer=self.checkpointer,
            completed_positions=self.completed_positions,
            resume_skip_set=self.resume_skip_set,
            pending_resume_states=self.pending_resume_states,
            resume_invocation=self.resume_invocation,
        )

    def descend_into_fan_out_instance(
        self,
        fan_out_node_name: str,
        parent_state: State,
        sub_attached: tuple[SubscribedObserver, ...],
        fan_out_index: int,
    ) -> _InvocationContext:
        """Build the context for one fan-out instance's subgraph invocation.

        Same shape as ``descend_into_subgraph`` but stamps the fan-out
        index onto the new context so every inner-node event carries it.
        Per spec §9 the index is the instance's 0-based position.

        Per pipeline-utilities §10.3 / §10.7: fan-out instance internal
        events do NOT produce checkpoint saves in v1. We achieve that
        by clearing ``checkpointer`` to None on the descent so the
        save gate inside the inner _step_function_node is False; the
        rest of the checkpoint context (invocation_id, correlation_id,
        etc.) still propagates so observability spans inside the
        instance can correlate. ``resume_skip_set`` is also dropped:
        a resumed invocation re-runs the entire fan-out from scratch
        per §10.7 atomic-restart.
        """
        return _InvocationContext(
            queue=self.queue,
            graph_attached=self.graph_attached + sub_attached,
            invocation_scoped=self.invocation_scoped,
            step_counter=self.step_counter,
            namespace_prefix=self.namespace_prefix + (fan_out_node_name,),
            parent_states_prefix=self.parent_states_prefix + (parent_state,),
            fan_out_index=fan_out_index,
            invocation_id=self.invocation_id,
            correlation_id=self.correlation_id,
            checkpointer=None,
            completed_positions=self.completed_positions,
            resume_skip_set=frozenset(),
            # Fan-out instances are atomic-restart per §10.7 — no
            # saved inner state to thread in. Drop the map.
            pending_resume_states={},
            resume_invocation=self.resume_invocation,
        )

    def take_step(self) -> int:
        """Atomically (single-threaded asyncio) read-and-increment the
        shared step counter. Returns the value to assign to the just-
        executed node's event."""
        n = self.step_counter[0]
        self.step_counter[0] = n + 1
        return n


def _dispatch(context: _InvocationContext, event: NodeEvent) -> None:
    """Enqueue a node event for the delivery worker.

    For ``"started"``-phase events, also call any subscribed observer's
    optional ``prepare_sync(event)`` synchronously — in the engine task,
    BEFORE queueing — so observers that need to publish per-event state
    the engine itself reads in the same engine-task scope (e.g., the
    OTel observer setting ``current_active_observer_span`` for the
    engine to attach into the OTel context) can do so before the node
    body runs.

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
    if event.phase == "started":
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


async def deliver_loop(queue: asyncio.Queue[_QueuedItem | None]) -> None:
    """Background worker: read queued events, deliver to observers serially.

    - No two observers receive the same event concurrently (we await
      each).
    - No observer receives event N+1 until everyone has finished N
      (the loop processes one item fully before pulling the next).
    - Observers whose ``phases`` set excludes the event's phase do
      NOT receive it. Phase filter applies at delivery, not dispatch
      — the engine still produces both events for every attempt.
    - Observer exceptions don't propagate, don't break siblings,
      don't block subsequent events. Reported via ``warnings.warn``.

    The loop terminates when it receives ``_DRAIN_SENTINEL`` (None).
    """
    while True:
        item = await queue.get()
        if item is None:
            return
        for subscribed in item.observers:
            if item.event.phase not in subscribed.phases:
                continue
            try:
                await subscribed.observer(item.event)
            except Exception as e:
                warnings.warn(
                    f"observer raised {type(e).__name__}: {e}",
                    stacklevel=1,
                )


__all__ = [
    "ALL_PHASES",
    "Observer",
    "RemoveHandle",
    "SubscribedObserver",
    # Engine-internal but listed so pyright sees them as exported (they're
    # imported by `compiled.py` and `subgraph.py`). The underscore prefix
    # is the user-facing "don't import these" signal.
    "_DRAIN_SENTINEL",
    "_InvocationContext",
    "_QueuedItem",
    "_coerce_subscribed",
    "_dispatch",
    "deliver_loop",
]
