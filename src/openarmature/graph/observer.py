"""Observer hooks: protocol, delivery queue, per-invocation context.

Per spec v0.3.0 §6 (proposal 0003): a registered observer receives a
`NodeEvent` once per node execution, asynchronously with respect to the
graph's execution loop. The graph never awaits observer processing.

This module defines:

- `Observer`: the callable shape an observer satisfies.
- `RemoveHandle`: returned by `CompiledGraph.attach_observer` so the caller
  can detach later without reference-equality games.
- `_InvocationContext`: the cross-graph state threaded through one
  outermost-invocation, including any nested subgraphs. Carries the queue,
  observer chain (graph-attached, outermost → innermost) and the
  invocation-scoped observers, plus a shared step counter, namespace
  prefix, and parent-state stack.
- `_QueuedItem`: an event paired with its delivery observer list.
- `_dispatch`: enqueues an event for the worker to deliver.
- `deliver_loop`: the worker coroutine. Reads items from the queue and
  calls each observer in order, isolating exceptions via
  `warnings.warn` per spec.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, field
from typing import Protocol

from .events import NodeEvent
from .state import State


class Observer(Protocol):
    """An async callable invoked once per node execution.

    Per spec v0.3.0 §6: observers MUST be asynchronous so the delivery
    queue can await each one to coordinate completion. Observers MUST
    NOT alter state, routing, or any other aspect of the graph run.

    The parameter is positional-only (`event, /`) so structural conformance
    isn't tied to a specific parameter name — implementations can use
    `event`, `_event`, or any other name.
    """

    async def __call__(self, event: NodeEvent, /) -> None: ...


@dataclass(frozen=True)
class RemoveHandle:
    """Returned by `CompiledGraph.attach_observer`. Call `.remove()` to
    detach the observer. Idempotent — calling `.remove()` after the
    observer is already detached is a no-op.

    Per spec v0.3.0 §6: changes to the registered observer set during a
    graph run do NOT take effect until the next invocation.
    """

    _observers: list[Observer]
    _observer: Observer

    def remove(self) -> None:
        try:
            self._observers.remove(self._observer)
        except ValueError:
            pass


@dataclass(frozen=True)
class _QueuedItem:
    """An event paired with the exact ordered observer list that should
    receive it. The list is computed at dispatch time so events from
    different depths in nested subgraphs carry the correct observer chain
    without the worker needing to know the graph topology.
    """

    event: NodeEvent
    observers: tuple[Observer, ...]


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
    graph_attached: tuple[Observer, ...]
    # Set once at the outermost invoke; carried unchanged into subgraphs.
    invocation_scoped: tuple[Observer, ...]
    # Shared mutable single-element list — a simple way to share an int by
    # reference across recursive subgraph contexts without leaking a class.
    step_counter: list[int] = field(default_factory=lambda: [0])
    namespace_prefix: tuple[str, ...] = ()
    parent_states_prefix: tuple[State, ...] = ()

    def full_observers(self) -> tuple[Observer, ...]:
        """Return the ordered observer list to deliver for events from
        this depth. Per spec §6: graph-attached (outermost → innermost),
        then invocation-scoped (passed to the outermost invoke)."""
        return self.graph_attached + self.invocation_scoped

    def descend_into_subgraph(
        self,
        subgraph_node_name: str,
        parent_state: State,
        sub_attached: tuple[Observer, ...],
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

    Per spec v0.3.0 §6:
    - No two observers receive the same event concurrently (we await each).
    - No observer receives event N+1 until everyone has finished N (the
      loop processes one item fully before pulling the next).
    - Observer exceptions don't propagate, don't break siblings, don't
      block subsequent events. Reported via `warnings.warn`.

    The loop terminates when it receives `_DRAIN_SENTINEL` (None).
    """
    while True:
        item = await queue.get()
        if item is None:
            return
        for observer in item.observers:
            try:
                await observer(item.event)
            except Exception as e:
                warnings.warn(
                    f"observer raised {type(e).__name__}: {e}",
                    stacklevel=1,
                )


__all__ = [
    "Observer",
    "RemoveHandle",
    # Engine-internal but listed so pyright sees them as exported (they're
    # imported by `compiled.py` and `subgraph.py`). The underscore prefix
    # is the user-facing "don't import these" signal.
    "_DRAIN_SENTINEL",
    "_InvocationContext",
    "_QueuedItem",
    "_dispatch",
    "deliver_loop",
]
