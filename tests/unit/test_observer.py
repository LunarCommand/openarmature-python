"""Unit tests for the observer delivery queue mechanics.

Delivery is strictly serial, ordered, isolates
observer exceptions, and filters by per-observer phase subscription.
These tests exercise the queue/worker pair in isolation — no graph
engine — so behavior bugs surface here rather than inside fixture
failures.
"""

import asyncio
import warnings
from types import MappingProxyType
from typing import Literal

from openarmature.graph import (
    MetadataAugmentationEvent,
    NodeEvent,
    Observer,
    ObserverEvent,
    State,
    SubscribedObserver,
)
from openarmature.graph.observer import (
    _DRAIN_SENTINEL,
    RemoveHandle,
    _dispatch,
    _DrainCounters,
    _InvocationContext,
    _QueuedItem,
    deliver_loop,
)
from openarmature.observability.metadata import set_invocation_metadata


class DummyState(State):
    v: str = ""


def _make_event(
    name: str,
    step: int = 0,
    phase: Literal["started", "completed"] = "completed",
) -> NodeEvent:
    return NodeEvent(
        node_name=name,
        namespace=(name,),
        step=step,
        phase=phase,
        pre_state=DummyState(),
        post_state=DummyState(v=f"after-{name}") if phase == "completed" else None,
        error=None,
        parent_states=(),
    )


def _wrap(observer: Observer) -> SubscribedObserver:
    """Wrap a bare observer for the default both-phases subscription —
    most queue-mechanics tests don't care about phase filtering."""
    return SubscribedObserver(observer=observer)


async def _drain(queue: asyncio.Queue[_QueuedItem | None], worker: asyncio.Task[None]) -> None:
    queue.put_nowait(_DRAIN_SENTINEL)
    await worker


# ===== Basic delivery + ordering =====


async def test_events_delivered_in_queue_order() -> None:
    received: list[str] = []

    async def observer(event: ObserverEvent) -> None:
        assert isinstance(event, NodeEvent)
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    subscribed = (_wrap(observer),)
    for name in ("a", "b", "c"):
        queue.put_nowait(_QueuedItem(event=_make_event(name), observers=subscribed))
    await _drain(queue, worker)

    assert received == ["a", "b", "c"]


async def test_multiple_observers_fire_in_registration_order() -> None:
    received: list[str] = []

    async def obs1(event: ObserverEvent) -> None:
        assert isinstance(event, NodeEvent)
        received.append(f"obs1:{event.node_name}")

    async def obs2(event: ObserverEvent) -> None:
        assert isinstance(event, NodeEvent)
        received.append(f"obs2:{event.node_name}")

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    subscribed = (_wrap(obs1), _wrap(obs2))
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=subscribed))
    queue.put_nowait(_QueuedItem(event=_make_event("b"), observers=subscribed))
    await _drain(queue, worker)

    # All observers for event A finish before any observer sees event B —
    # the spec's "no observer receives event N+1 until everyone has
    # finished N" rule.
    assert received == ["obs1:a", "obs2:a", "obs1:b", "obs2:b"]


# ===== Error isolation =====


async def test_observer_exception_does_not_propagate_to_caller() -> None:
    async def boom(_event: ObserverEvent) -> None:
        raise RuntimeError("nope")

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=(_wrap(boom),)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await _drain(queue, worker)

    # The exception is reported via warnings, not raised to the caller.
    assert any("RuntimeError" in str(w.message) for w in caught)


async def test_raising_observer_does_not_block_siblings_on_same_event() -> None:
    received: list[str] = []

    async def obs1(_event: ObserverEvent) -> None:
        raise RuntimeError("obs1 boom")

    async def obs2(event: ObserverEvent) -> None:
        assert isinstance(event, NodeEvent)
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=(_wrap(obs1), _wrap(obs2))))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        await _drain(queue, worker)

    assert received == ["a"]


async def test_raising_observer_does_not_block_subsequent_events() -> None:
    received: list[str] = []

    async def always_raises(_event: ObserverEvent) -> None:
        raise RuntimeError("always boom")

    async def silent(event: ObserverEvent) -> None:
        assert isinstance(event, NodeEvent)
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    subscribed = (_wrap(always_raises), _wrap(silent))
    for name in ("a", "b", "c"):
        queue.put_nowait(_QueuedItem(event=_make_event(name), observers=subscribed))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        await _drain(queue, worker)

    assert received == ["a", "b", "c"]


# ===== Phase filtering (spec v0.6.0 §6) =====


async def test_phase_filter_skips_unsubscribed_phase() -> None:
    received: list[tuple[str, str]] = []

    async def obs(event: ObserverEvent) -> None:
        assert isinstance(event, NodeEvent)
        received.append((event.node_name, event.phase))

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    completed_only = (SubscribedObserver(observer=obs, phases=frozenset({"completed"})),)
    queue.put_nowait(_QueuedItem(event=_make_event("a", phase="started"), observers=completed_only))
    queue.put_nowait(_QueuedItem(event=_make_event("a", phase="completed"), observers=completed_only))
    await _drain(queue, worker)

    # The observer subscribed to `completed` only — the started event
    # was delivered to the queue but filtered at the worker.
    assert received == [("a", "completed")]


async def test_subscribed_observer_rejects_empty_phases() -> None:
    async def obs(_event: ObserverEvent) -> None:
        pass

    try:
        SubscribedObserver(observer=obs, phases=frozenset())
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty phases")


async def test_subscribed_observer_rejects_unknown_phase() -> None:
    async def obs(_event: ObserverEvent) -> None:
        pass

    try:
        SubscribedObserver(observer=obs, phases=frozenset({"started", "bogus"}))
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown phase")


# ===== Sentinel + termination =====


async def test_sentinel_terminates_worker_after_processing_queued_events() -> None:
    received: list[str] = []

    async def observer(event: ObserverEvent) -> None:
        assert isinstance(event, NodeEvent)
        received.append(event.node_name)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    subscribed = (_wrap(observer),)
    queue.put_nowait(_QueuedItem(event=_make_event("a"), observers=subscribed))
    queue.put_nowait(_QueuedItem(event=_make_event("b"), observers=subscribed))
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
    async def graph_obs(_event: ObserverEvent) -> None:
        pass

    async def invocation_obs(_event: ObserverEvent) -> None:
        pass

    graph_subscribed = _wrap(graph_obs)
    invocation_subscribed = _wrap(invocation_obs)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    ctx = _InvocationContext(
        queue=queue,
        graph_attached=(graph_subscribed,),
        invocation_scoped=(invocation_subscribed,),
    )

    _dispatch(ctx, _make_event("a"))

    item = queue.get_nowait()
    assert item is not None
    # graph_attached comes first, then invocation_scoped per spec.
    assert item.observers == (graph_subscribed, invocation_subscribed)


# ===== _InvocationContext.descend_into_subgraph =====


async def test_descend_extends_chain_namespace_and_parent_states() -> None:
    async def outer_obs(_event: ObserverEvent) -> None:
        pass

    async def sub_obs(_event: ObserverEvent) -> None:
        pass

    async def invocation_obs(_event: ObserverEvent) -> None:
        pass

    outer_subscribed = _wrap(outer_obs)
    sub_subscribed = _wrap(sub_obs)
    invocation_subscribed = _wrap(invocation_obs)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    outer = _InvocationContext(
        queue=queue,
        graph_attached=(outer_subscribed,),
        invocation_scoped=(invocation_subscribed,),
    )

    parent = DummyState(v="parent-snapshot")
    sub = outer.descend_into_subgraph(
        subgraph_node_name="sub",
        parent_state=parent,
        sub_attached=(sub_subscribed,),
    )

    assert sub.queue is queue
    assert sub.step_counter is outer.step_counter
    assert sub.graph_attached == (outer_subscribed, sub_subscribed)
    assert sub.invocation_scoped == (invocation_subscribed,)
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
    async def obs(_event: ObserverEvent) -> None:
        pass

    subscribed = _wrap(obs)
    observers: list[SubscribedObserver] = [subscribed]
    handle = RemoveHandle(_observers=observers, _observer=subscribed)

    assert subscribed in observers
    handle.remove()
    assert subscribed not in observers


def test_remove_handle_is_idempotent() -> None:
    async def obs(_event: ObserverEvent) -> None:
        pass

    subscribed = _wrap(obs)
    observers: list[SubscribedObserver] = [subscribed]
    handle = RemoveHandle(_observers=observers, _observer=subscribed)

    handle.remove()
    handle.remove()  # second call is a no-op, doesn't raise
    assert subscribed not in observers


# ===== Metadata-augmentation event delivery (proposal 0040) =====


async def test_metadata_augmentation_event_bypasses_phase_filter() -> None:
    """Augmentation events flow through ``__call__`` on the union-typed
    Observer Protocol and ignore the per-observer ``phases`` set
    entirely (they aren't phase events). Observers that only care
    about NodeEvent ``isinstance``-narrow and early-return.
    """
    augment_received: list[MetadataAugmentationEvent] = []
    node_received: list[NodeEvent] = []

    async def observer(event: ObserverEvent) -> None:
        if isinstance(event, MetadataAugmentationEvent):
            augment_received.append(event)
        elif isinstance(event, NodeEvent):
            node_received.append(event)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    # Subscribe to ``completed`` only — to prove the augmentation event
    # bypasses the phase filter (it has no phase).
    completed_only = (SubscribedObserver(observer=observer, phases=frozenset({"completed"})),)
    augmentation = MetadataAugmentationEvent(
        entries=MappingProxyType({"region": "us-east-1"}),
        namespace=("router", "classify"),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
    )
    queue.put_nowait(_QueuedItem(event=augmentation, observers=completed_only))
    await _drain(queue, worker)

    assert augment_received == [augmentation]
    assert node_received == []


async def test_metadata_augmentation_observer_exception_is_isolated() -> None:
    """A raise on the augmentation event follows the same isolation
    contract as a raise on a NodeEvent — warned, sibling observers
    still run, the worker keeps draining."""
    sibling_received: list[MetadataAugmentationEvent] = []

    async def boom(_event: ObserverEvent) -> None:
        raise RuntimeError("boom")

    async def good(event: ObserverEvent) -> None:
        if isinstance(event, MetadataAugmentationEvent):
            sibling_received.append(event)

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    worker = asyncio.create_task(deliver_loop(queue, _DrainCounters()))
    augmentation = MetadataAugmentationEvent(
        entries=MappingProxyType({"k": "v"}),
        namespace=(),
    )
    subscribed = (_wrap(boom), _wrap(good))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        queue.put_nowait(_QueuedItem(event=augmentation, observers=subscribed))
        await _drain(queue, worker)

    assert sibling_received == [augmentation]
    assert any("observer raised RuntimeError" in str(w.message) for w in captured)


async def test_set_invocation_metadata_emits_augmentation_event_via_dispatch() -> None:
    """``set_invocation_metadata`` reads the current_dispatch closure
    (engine-installed in real runs) and constructs a
    MetadataAugmentationEvent carrying the delta + lineage from the
    correlation ContextVars."""
    from openarmature.observability.correlation import (
        _reset_active_dispatch,
        _reset_attempt_index,
        _reset_branch_name,
        _reset_fan_out_index,
        _reset_namespace_prefix,
        _set_active_dispatch,
        _set_attempt_index,
        _set_branch_name,
        _set_fan_out_index,
        _set_namespace_prefix,
    )

    captured: list[ObserverEvent] = []

    def dispatch(event: ObserverEvent) -> None:
        captured.append(event)

    dispatch_token = _set_active_dispatch(dispatch)
    namespace_token = _set_namespace_prefix(("outer", "leaf"))
    fan_out_token = _set_fan_out_index(2)
    branch_token = _set_branch_name("primary")
    attempt_token = _set_attempt_index(1)
    try:
        set_invocation_metadata(region="us-east-1", retries=3)
    finally:
        _reset_attempt_index(attempt_token)
        _reset_branch_name(branch_token)
        _reset_fan_out_index(fan_out_token)
        _reset_namespace_prefix(namespace_token)
        _reset_active_dispatch(dispatch_token)

    assert len(captured) == 1
    event = captured[0]
    assert isinstance(event, MetadataAugmentationEvent)
    assert dict(event.entries) == {"region": "us-east-1", "retries": 3}
    assert event.namespace == ("outer", "leaf")
    assert event.attempt_index == 1
    assert event.fan_out_index == 2
    assert event.branch_name == "primary"


def test_set_invocation_metadata_outside_invocation_skips_dispatch() -> None:
    """Without a current_dispatch installed (no engine in scope),
    ``set_invocation_metadata`` still updates the ContextVar but
    does NOT raise and does NOT attempt to enqueue an event."""
    # Sanity: by default outside any engine the dispatch ContextVar is
    # None, so the call should be a no-op on the queue side.
    set_invocation_metadata(local_key="local_value")
