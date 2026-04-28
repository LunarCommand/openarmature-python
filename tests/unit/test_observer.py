"""Unit tests for the observer delivery queue mechanics.

Per spec v0.3.0 §6: delivery is strictly serial, ordered, and isolates
observer exceptions. These tests exercise the queue/worker pair in
isolation — no graph engine — so behavior bugs surface here rather than
inside fixture failures.
"""

import asyncio
import warnings

from openarmature.graph import Observer, State
from openarmature.graph.events import NodeEvent
from openarmature.graph.observer import (
    _DRAIN_SENTINEL,
    RemoveHandle,
    _dispatch,
    _InvocationContext,
    _QueuedItem,
    deliver_loop,
)


class DummyState(State):
    v: str = ""


def _make_event(name: str, step: int = 0) -> NodeEvent:
    return NodeEvent(
        node_name=name,
        namespace=(name,),
        step=step,
        pre_state=DummyState(),
        post_state=DummyState(v=f"after-{name}"),
        error=None,
        parent_states=(),
    )


async def _drain(queue: asyncio.Queue[_QueuedItem | None], worker: asyncio.Task[None]) -> None:
    queue.put_nowait(_DRAIN_SENTINEL)
    await worker


# ===== Basic delivery + ordering =====


async def test_events_delivered_in_queue_order() -> None:
    received: list[str] = []

    async def observer(event: NodeEvent) -> None:
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue))
    for name in ("a", "b", "c"):
        queue.put_nowait(_QueuedItem(event=_make_event(name), observers=(observer,)))
    await _drain(queue, worker)

    assert received == ["a", "b", "c"]


async def test_multiple_observers_fire_in_registration_order() -> None:
    received: list[str] = []

    async def obs1(event: NodeEvent) -> None:
        received.append(f"obs1:{event.node_name}")

    async def obs2(event: NodeEvent) -> None:
        received.append(f"obs2:{event.node_name}")

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue))
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=(obs1, obs2)))
    queue.put_nowait(_QueuedItem(event=_make_event("b"), observers=(obs1, obs2)))
    await _drain(queue, worker)

    # All observers for event A finish before any observer sees event B —
    # the spec's "no observer receives event N+1 until everyone has
    # finished N" rule.
    assert received == ["obs1:a", "obs2:a", "obs1:b", "obs2:b"]


# ===== Error isolation =====


async def test_observer_exception_does_not_propagate_to_caller() -> None:
    async def boom(_event: NodeEvent) -> None:
        raise RuntimeError("nope")

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue))
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=(boom,)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await _drain(queue, worker)

    # The exception is reported via warnings, not raised to the caller.
    assert any("RuntimeError" in str(w.message) for w in caught)


async def test_raising_observer_does_not_block_siblings_on_same_event() -> None:
    received: list[str] = []

    async def obs1(_event: NodeEvent) -> None:
        raise RuntimeError("obs1 boom")

    async def obs2(event: NodeEvent) -> None:
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue))
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=(obs1, obs2)))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        await _drain(queue, worker)

    assert received == ["a"]


async def test_raising_observer_does_not_block_subsequent_events() -> None:
    received: list[str] = []

    async def always_raises(_event: NodeEvent) -> None:
        raise RuntimeError("always boom")

    async def silent(event: NodeEvent) -> None:
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue))
    for name in ("a", "b", "c"):
        queue.put_nowait(_QueuedItem(event=_make_event(name), observers=(always_raises, silent)))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        await _drain(queue, worker)

    assert received == ["a", "b", "c"]


# ===== Sentinel + termination =====


async def test_sentinel_terminates_worker_after_processing_queued_events() -> None:
    received: list[str] = []

    async def observer(event: NodeEvent) -> None:
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue))
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=(observer,)))
    queue.put_nowait(_QueuedItem(event=_make_event("b"), observers=(observer,)))
    queue.put_nowait(_DRAIN_SENTINEL)

    await asyncio.wait_for(worker, timeout=1.0)
    # Both events delivered before the sentinel terminated the worker.
    assert received == ["a", "b"]


# ===== _dispatch =====


async def test_dispatch_skips_when_no_observers_for_depth() -> None:
    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    ctx = _InvocationContext(queue=queue, graph_attached=(), invocation_scoped=())

    _dispatch(ctx, _make_event("a"))

    assert queue.empty()


async def test_dispatch_enqueues_with_full_observer_chain_in_order() -> None:
    async def graph_obs(_event: NodeEvent) -> None: ...
    async def invocation_obs(_event: NodeEvent) -> None: ...

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    ctx = _InvocationContext(queue=queue, graph_attached=(graph_obs,), invocation_scoped=(invocation_obs,))

    _dispatch(ctx, _make_event("a"))

    item = queue.get_nowait()
    assert item is not None
    # graph_attached comes first, then invocation_scoped per spec.
    assert item.observers == (graph_obs, invocation_obs)


# ===== _InvocationContext.descend_into_subgraph =====


async def test_descend_extends_chain_namespace_and_parent_states() -> None:
    async def outer_obs(_event: NodeEvent) -> None: ...
    async def sub_obs(_event: NodeEvent) -> None: ...
    async def invocation_obs(_event: NodeEvent) -> None: ...

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    outer = _InvocationContext(queue=queue, graph_attached=(outer_obs,), invocation_scoped=(invocation_obs,))

    parent = DummyState(v="parent-snapshot")
    sub = outer.descend_into_subgraph(subgraph_node_name="sub", parent_state=parent, sub_attached=(sub_obs,))

    assert sub.queue is queue
    assert sub.step_counter is outer.step_counter
    assert sub.graph_attached == (outer_obs, sub_obs)
    assert sub.invocation_scoped == (invocation_obs,)
    assert sub.namespace_prefix == ("sub",)
    assert sub.parent_states_prefix == (parent,)


async def test_take_step_shares_counter_across_descended_contexts() -> None:
    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    outer = _InvocationContext(queue=queue, graph_attached=(), invocation_scoped=())

    assert outer.take_step() == 0
    assert outer.take_step() == 1

    sub = outer.descend_into_subgraph("sub", DummyState(), ())
    assert sub.take_step() == 2
    # Outer continues from where the subgraph left off.
    assert outer.take_step() == 3


# ===== RemoveHandle =====


def test_remove_handle_detaches_observer() -> None:
    async def obs(_event: NodeEvent) -> None: ...

    observers: list[Observer] = [obs]
    handle = RemoveHandle(_observers=observers, _observer=obs)

    assert obs in observers
    handle.remove()
    assert obs not in observers


def test_remove_handle_is_idempotent() -> None:
    async def obs(_event: NodeEvent) -> None: ...

    observers: list[Observer] = [obs]
    handle = RemoveHandle(_observers=observers, _observer=obs)

    handle.remove()
    handle.remove()  # second call is a no-op, doesn't raise
    assert obs not in observers
