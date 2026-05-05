"""Observer hooks: protocol, subscription, delivery queue, per-invocation context.

Per spec v0.6.0 §6 (proposal 0005): each node attempt produces a started/
completed event pair, and observers register with an optional `phases`
set so they can subscribe to one phase or both. The graph never awaits
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
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from .events import NodeEvent
from .state import State


class Observer(Protocol):
    """An async callable invoked once per node-boundary event.

    Per spec v0.6.0 §6: observers MUST be asynchronous so the delivery
    queue can await each one to coordinate completion. Observers MUST
    NOT alter state, routing, or any other aspect of the graph run.

    The parameter is positional-only (`event, /`) so structural conformance
    isn't tied to a specific parameter name — implementations can use
    `event`, `_event`, or any other name.
    """

    async def __call__(self, event: NodeEvent, /) -> None:
        """Receive a single node-boundary event."""
        raise NotImplementedError


# Per spec v0.6.0 §6: the two valid phase strings. Used as the default
# subscription set when a caller doesn't restrict by phase.
ALL_PHASES: frozenset[str] = frozenset({"started", "completed"})


@dataclass(frozen=True)
class SubscribedObserver:
    """An observer paired with its phase subscription set.

    Per spec v0.6.0 §6: observers register with an optional `phases`
    parameter naming the phase strings they want to receive. The default
    (`ALL_PHASES`) means "deliver every event." Empty phase sets are
    forbidden — passing one raises `ValueError` at registration time per
    the spec's "implementations SHOULD raise" guidance, hardened to MUST
    here so misconfiguration surfaces immediately.

    Construct one of these directly when handing phase-filtered observers
    to `CompiledGraph.invoke(observers=...)`. For the single-observer
    `attach_observer` path, pass `phases=` as a keyword argument and the
    engine wraps it for you.
    """

    observer: Observer
    phases: frozenset[str] = ALL_PHASES

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("phases must be non-empty; spec §6 forbids empty phase subscriptions")
        invalid = self.phases - ALL_PHASES
        if invalid:
            raise ValueError(f"unknown phase(s): {sorted(invalid)}; allowed: 'started', 'completed'")


def _coerce_subscribed(
    observer: Observer | SubscribedObserver,
    *,
    phases: Iterable[str] | None = None,
) -> SubscribedObserver:
    """Normalize a registration argument into a `SubscribedObserver`.

    - A bare `Observer` callable becomes a `SubscribedObserver` with
      either the supplied `phases` or `ALL_PHASES` (default).
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
    """Returned by `CompiledGraph.attach_observer`. Call `.remove()` to
    detach the observer. Idempotent — calling `.remove()` after the
    observer is already detached is a no-op.

    Per spec v0.6.0 §6: changes to the registered observer set during a
    graph run do NOT take effect until the next invocation.
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
        """
        return _InvocationContext(
            queue=self.queue,
            graph_attached=self.graph_attached + sub_attached,
            invocation_scoped=self.invocation_scoped,
            step_counter=self.step_counter,
            namespace_prefix=self.namespace_prefix + (subgraph_node_name,),
            parent_states_prefix=self.parent_states_prefix + (parent_state,),
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

    No-op when no observers exist for this depth — avoids paying the queue
    overhead for graphs that don't observe anything.
    """
    observers = context.full_observers()
    if not observers:
        return
    context.queue.put_nowait(_QueuedItem(event=event, observers=observers))


async def deliver_loop(queue: asyncio.Queue[_QueuedItem | None]) -> None:
    """Background worker: read queued events, deliver to observers serially.

    Per spec v0.6.0 §6:
    - No two observers receive the same event concurrently (we await each).
    - No observer receives event N+1 until everyone has finished N (the
      loop processes one item fully before pulling the next).
    - Observers whose `phases` set excludes the event's phase do NOT
      receive it. Phase filter applies at delivery, not dispatch — the
      engine still produces both events for every attempt.
    - Observer exceptions don't propagate, don't break siblings, don't
      block subsequent events. Reported via `warnings.warn`.

    The loop terminates when it receives `_DRAIN_SENTINEL` (None).
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
