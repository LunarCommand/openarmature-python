# Spec: realizes graph-engine §6 Drain bounded-wait contract
# (proposal 0010). These tests cover the five core paths the spec
# fixtures 022-025 also exercise, plus the no-active-workers and
# clean-state-after-timeout cases isolated from the conformance
# harness.

"""Unit tests for `CompiledGraph.drain(timeout=...)` + `DrainSummary`.

Per spec graph-engine §6 (amended by proposal 0010): drain accepts an
optional timeout, returns a `DrainSummary` with at least
`undelivered_count` + `timeout_reached`, MUST cancel workers cleanly
so the graph remains usable for subsequent invocations.
"""

import asyncio
import time
from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    DrainSummary,
    GraphBuilder,
    NodeEvent,
    ObserverEvent,
    State,
    append,
)


class _S(State):
    v: int = 0


async def _set_v_1(_s: _S) -> Mapping[str, Any]:
    return {"v": 1}


async def _set_v_2(_s: _S) -> Mapping[str, Any]:
    return {"v": 2}


async def _set_v_3(_s: _S) -> Mapping[str, Any]:
    return {"v": 3}


def _build_compiled() -> CompiledGraph[_S]:
    """A trivial three-node linear graph: a → b → c → END. Each node
    bumps the counter so observer events fire for every transition."""
    builder: GraphBuilder[_S] = GraphBuilder(_S)
    builder.set_entry("a")
    builder.add_node("a", _set_v_1)
    builder.add_node("b", _set_v_2)
    builder.add_node("c", _set_v_3)
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", END)
    return builder.compile()


async def test_drain_with_no_active_workers_returns_clean_summary() -> None:
    # Spec: `drain()` MUST return a `DrainSummary` even when there
    # are no active workers. The undelivered count is 0 and the
    # timeout-reached flag is False — consistent shape across all
    # call paths.
    compiled = _build_compiled()
    summary = await compiled.drain()
    assert summary == DrainSummary(undelivered_count=0, timeout_reached=False)


async def test_drain_without_timeout_waits_for_all_events() -> None:
    # Spec §6 (v0.3.0 contract preserved): drain without a timeout
    # waits indefinitely until all events deliver. Returns a
    # DrainSummary with the consistent shape.
    received: list[str] = []

    async def slow_obs(event: ObserverEvent) -> None:
        # ~50ms per event; the 3-node graph fires 8 events
        # (3 nodes × started + completed = 6 NodeEvents, plus
        # InvocationStarted + InvocationCompleted from proposal
        # 0043). The observer only counts NodeEvents — boundary
        # events early-return.
        await asyncio.sleep(0.05)
        if not isinstance(event, NodeEvent):
            return
        received.append(event.node_name)

    compiled = _build_compiled()
    compiled.attach_observer(slow_obs)
    await compiled.invoke(_S())
    started = time.monotonic()
    summary = await compiled.drain()
    elapsed = time.monotonic() - started

    assert summary.timeout_reached is False
    assert summary.undelivered_count == 0
    # 6 NodeEvents reach the receiver (the two boundary events early-
    # return inside the observer, but they're still counted as
    # delivered by the drain summary).
    assert len(received) == 6
    # Drain blocked for the observer's total work — 8 events × 50ms.
    # Allow generous slack for scheduler / CI variance.
    assert elapsed >= 0.35


async def test_drain_with_timeout_not_reached_for_fast_observers() -> None:
    # Spec §6: a generous timeout doesn't fire when observers are
    # fast. Summary reports clean delivery.
    received: list[str] = []

    async def fast_obs(event: ObserverEvent) -> None:
        if not isinstance(event, NodeEvent):
            return
        received.append(event.node_name)

    compiled = _build_compiled()
    compiled.attach_observer(fast_obs)
    await compiled.invoke(_S())
    summary = await compiled.drain(timeout=5.0)

    assert summary == DrainSummary(undelivered_count=0, timeout_reached=False)
    assert len(received) == 6


async def test_drain_with_timeout_fires_reports_undelivered() -> None:
    # Spec §6: a tight timeout against a slow observer fires; the
    # summary reports `timeout_reached=True` and a positive
    # `undelivered_count`. Drain MUST return within the timeout (with
    # generous slack for cancellation settlement).
    received: list[str] = []

    async def slow_obs(event: ObserverEvent) -> None:
        # 200ms per event vs 100ms drain timeout; at most 0-1 events
        # complete before the deadline fires.
        await asyncio.sleep(0.2)
        if not isinstance(event, NodeEvent):
            return
        received.append(event.node_name)

    compiled = _build_compiled()
    compiled.attach_observer(slow_obs)
    await compiled.invoke(_S())
    started = time.monotonic()
    summary = await compiled.drain(timeout=0.1)
    elapsed = time.monotonic() - started

    assert summary.timeout_reached is True
    # 6 NodeEvents + 2 boundary events = 8 enqueued; at most 0-1
    # deliver before the 100ms deadline.
    assert summary.undelivered_count >= 6
    # The hard deadline is non-negotiable. Allow ~250ms of slack for
    # cancellation settling + CI scheduler variance — the dispatched
    # event's await still resolves under cancellation, and the
    # gather(return_exceptions=True) on cancelled tasks settles
    # within an event-loop tick or two.
    assert elapsed < 0.5


async def test_drain_after_timeout_leaves_graph_usable() -> None:
    # Spec §6: "workers MUST be cancelled or otherwise terminated
    # such that the compiled graph remains usable for subsequent
    # invocations — partial delivery state from one drain MUST NOT
    # leak into the next invocation." Drives the load-bearing
    # cross-invocation cleanliness contract.
    call_count = [0]
    received_invocation_two: list[str] = []

    async def obs(event: ObserverEvent) -> None:
        # First invocation: slow enough to force the timeout to
        # fire. Second invocation: fast, so drain completes cleanly.
        # The mode is controlled by `call_count[0]`: we bump it
        # between invocations.
        if call_count[0] == 0:
            await asyncio.sleep(0.1)
            return
        if not isinstance(event, NodeEvent):
            return
        received_invocation_two.append(event.node_name)

    compiled = _build_compiled()
    compiled.attach_observer(obs)

    await compiled.invoke(_S())
    summary_one = await compiled.drain(timeout=0.05)
    assert summary_one.timeout_reached is True
    # `_active_workers` MUST be empty after the cancelled workers'
    # done-callbacks fire (gathered under return_exceptions). If a
    # cancelled worker leaked into the dict, the second drain would
    # snapshot it too and wait on it again.
    assert len(compiled._active_workers) == 0

    # Switch observer to the fast path.
    call_count[0] = 1
    await compiled.invoke(_S())
    summary_two = await compiled.drain()
    assert summary_two == DrainSummary(undelivered_count=0, timeout_reached=False)
    # All 6 events from the SECOND invocation deliver cleanly — no
    # carry-over from the first invocation's cancelled worker.
    assert len(received_invocation_two) == 6


async def test_drain_summary_is_frozen_dataclass() -> None:
    # Spec §6: `DrainSummary` shape is structurally identical across
    # all drain call paths. The dataclass is frozen so callers can
    # safely cache / compare instances.
    a = DrainSummary(undelivered_count=0, timeout_reached=False)
    b = DrainSummary(undelivered_count=0, timeout_reached=False)
    assert a == b
    with pytest.raises(AttributeError):
        a.undelivered_count = 5  # type: ignore[misc]


async def test_drain_with_zero_timeout_fires_immediately() -> None:
    # `timeout=0.0` is a valid non-negative duration; drain returns
    # immediately with whatever the worker hasn't gotten to yet.
    received: list[str] = []

    async def obs(event: ObserverEvent) -> None:
        await asyncio.sleep(0.05)
        if not isinstance(event, NodeEvent):
            return
        received.append(event.node_name)

    compiled = _build_compiled()
    compiled.attach_observer(obs)
    await compiled.invoke(_S())
    summary = await compiled.drain(timeout=0.0)

    assert summary.timeout_reached is True
    # All 8 events (6 NodeEvents + 2 boundary events) are still in
    # flight or queued — none delivered before the zero-second
    # deadline fired.
    assert summary.undelivered_count == 8
    assert len(received) == 0


async def test_drain_rejects_negative_timeout() -> None:
    # Spec §6: timeout is "a non-negative duration in seconds". A
    # negative value is a user mistake — surface it as ValueError at
    # the API boundary rather than silently treating it like an
    # immediate cancel.
    compiled = _build_compiled()
    with pytest.raises(ValueError, match="non-negative"):
        await compiled.drain(timeout=-1.0)


async def test_drain_rejects_nan_timeout() -> None:
    # NaN compares False against everything, so `not (timeout >= 0)`
    # catches it just like negative values. Without the validation it
    # would silently fall through `asyncio.wait` as an immediate cancel.
    compiled = _build_compiled()
    with pytest.raises(ValueError, match="non-negative"):
        await compiled.drain(timeout=float("nan"))


# ---------------------------------------------------------------------------
# Per-invocation drain (proposal 0054, spec graph-engine §6 *Per-invocation
# drain*). Tests below mirror the spec fixtures 028-033's case shapes.
# ---------------------------------------------------------------------------


async def _capture_node_invocation_id() -> str:
    """Helper used inside node bodies. Returns the current invocation_id
    (never None inside a node body)."""
    from openarmature.observability.correlation import current_invocation_id

    inv = current_invocation_id()
    assert inv is not None
    return inv


async def test_drain_events_for_unknown_invocation_returns_clean_summary() -> None:
    # Unknown invocation_id (no active worker, or the invocation
    # already drained and the worker exited) returns an empty summary
    # rather than raising. The shape mirrors drain()'s no-active-workers
    # path for consistency.
    compiled = _build_compiled()
    summary = await compiled.drain_events_for("nonexistent-id")
    assert summary == DrainSummary(undelivered_count=0, timeout_reached=False)


async def test_drain_events_for_basic_synchronization() -> None:
    # Mirrors spec fixture 028: a slow observer is still in flight when
    # a node calls drain_events_for; the drain blocks until the
    # snapshotted set has fully delivered, then returns clean.
    captured_invocation_id: list[str] = []
    deliveries: list[str] = []

    async def slow_obs(event: ObserverEvent) -> None:
        await asyncio.sleep(0.02)
        if isinstance(event, NodeEvent):
            deliveries.append(f"{event.node_name}:{event.phase}")

    async def _capture_then_drain(_s: _S) -> Mapping[str, Any]:
        inv = await _capture_node_invocation_id()
        captured_invocation_id.append(inv)
        return {"v": 1}

    builder: GraphBuilder[_S] = GraphBuilder(_S)
    builder.set_entry("capture")
    builder.add_node("capture", _capture_then_drain)
    builder.add_edge("capture", END)
    compiled = builder.compile()
    compiled.attach_observer(slow_obs)

    await compiled.invoke(_S())
    # Drain blocks until every pre-call event has delivered.
    summary = await compiled.drain_events_for(captured_invocation_id[0], timeout=5.0)

    assert summary == DrainSummary(undelivered_count=0, timeout_reached=False)
    # The capture node's started+completed pair delivered.
    assert "capture:started" in deliveries
    assert "capture:completed" in deliveries


async def test_drain_events_for_timeout_does_not_cancel_worker() -> None:
    # KEY divergence from drain(): per-invocation drain timeout MUST
    # NOT cancel the deliver worker on the SAME graph. After the
    # tight-timeout drain returns, the deliver loop continues
    # processing the queue, so the events that were undelivered at
    # timeout time eventually reach the observer. Mirrors spec
    # fixture 030.
    #
    # The decisive contract check is that the originally-undelivered
    # events land in the observer's delivery list AFTER the
    # timed-out drain returns — that can only happen if the deliver
    # worker kept running. A clean second drain on the same
    # invocation corroborates that the worker is making forward
    # progress.
    deliveries: list[str] = []

    async def slow_obs(event: ObserverEvent) -> None:
        # ~30ms per event. The 4-event invocation needs ~120ms total;
        # the first drain's 10ms timeout fires well before delivery
        # completes.
        await asyncio.sleep(0.03)
        if isinstance(event, NodeEvent):
            deliveries.append(f"{event.node_name}:{event.phase}")

    captured_inv: list[str] = []

    async def _capture(_s: _S) -> Mapping[str, Any]:
        captured_inv.append(await _capture_node_invocation_id())
        return {"v": 1}

    builder: GraphBuilder[_S] = GraphBuilder(_S)
    builder.set_entry("n")
    builder.add_node("n", _capture)
    builder.add_edge("n", END)
    compiled = builder.compile()
    compiled.attach_observer(slow_obs)
    await compiled.invoke(_S())

    # First drain: tight timeout against the slow observer fires
    # before any deliveries complete.
    started = time.monotonic()
    summary_1 = await compiled.drain_events_for(captured_inv[0], timeout=0.01)
    elapsed_1 = time.monotonic() - started

    assert summary_1.timeout_reached is True
    assert summary_1.undelivered_count > 0
    # Drain returned within the deadline (no cancellation overhead).
    assert elapsed_1 < 0.3
    # At this moment NO NodeEvents have been delivered yet (the
    # observer's first sleep was still in flight at timeout time).
    assert deliveries == []

    # Second drain on the SAME invocation_id with a generous timeout.
    # If the deliver worker had been cancelled by the first timeout,
    # this would either also time out (worker stuck) or return clean
    # via the unknown-invocation path (worker gone from
    # _active_workers). Either way the deliveries list would stay
    # empty. With the worker kept alive, the loop catches up and
    # this returns clean with every event delivered.
    summary_2 = await compiled.drain_events_for(captured_inv[0], timeout=5.0)
    assert summary_2 == DrainSummary(undelivered_count=0, timeout_reached=False)

    # Decisive check: every NodeEvent for the invocation reached
    # the observer. Two events per node (started + completed) ×
    # one node = 2. If the worker had been cancelled at first
    # timeout, deliveries would have stopped at 0 or 1.
    assert "n:started" in deliveries
    assert "n:completed" in deliveries


async def test_drain_events_for_invocation_scope_isolation() -> None:
    # Two serial invocations on the same compiled graph. Each drain
    # sees only its own events. Mirrors spec fixture 031.
    #
    # The contract: drain_events_for(inv_a) awaits ONLY events
    # tagged with inv_a; drain_events_for(inv_b) awaits ONLY events
    # tagged with inv_b. Per spec §5.1 each invocation gets a fresh
    # invocation_id, so the two ids differ and the observer's
    # delivery log can be partitioned cleanly by invocation.
    captured: list[str] = []
    delivery_log: list[tuple[str, str]] = []

    async def obs(event: ObserverEvent) -> None:
        await asyncio.sleep(0.01)
        if isinstance(event, NodeEvent):
            # Capture (invocation_id, node_name) so we can assert
            # which deliveries happened under which invocation.
            from openarmature.observability.correlation import current_invocation_id

            inv = current_invocation_id() or "?"
            delivery_log.append((inv, event.node_name))

    async def _capture(_s: _S) -> Mapping[str, Any]:
        captured.append(await _capture_node_invocation_id())
        return {"v": 1}

    builder: GraphBuilder[_S] = GraphBuilder(_S)
    builder.set_entry("n")
    builder.add_node("n", _capture)
    builder.add_edge("n", END)
    compiled = builder.compile()
    compiled.attach_observer(obs)

    # First invocation + drain.
    await compiled.invoke(_S())
    inv_a = captured[-1]
    summary_a = await compiled.drain_events_for(inv_a, timeout=5.0)
    assert summary_a == DrainSummary(undelivered_count=0, timeout_reached=False)

    # Second invocation gets a fresh invocation_id (spec §5.1) and
    # its own drain.
    await compiled.invoke(_S())
    inv_b = captured[-1]
    assert inv_b != inv_a
    summary_b = await compiled.drain_events_for(inv_b, timeout=5.0)
    assert summary_b == DrainSummary(undelivered_count=0, timeout_reached=False)

    # Partition the delivery log by invocation_id. Each drain's
    # snapshot covered exactly its own invocation's events; no
    # cross-contamination.
    a_entries = [(inv, name) for (inv, name) in delivery_log if inv == inv_a]
    b_entries = [(inv, name) for (inv, name) in delivery_log if inv == inv_b]
    # One node ("n") fires started + completed under each invocation.
    assert len(a_entries) == 2
    assert len(b_entries) == 2
    # Strict isolation: every entry's invocation_id matches the
    # partition it landed in.
    assert all(inv == inv_a for (inv, _) in a_entries)
    assert all(inv == inv_b for (inv, _) in b_entries)
    # No entry escaped the partition.
    assert len(a_entries) + len(b_entries) == len(delivery_log)


async def test_drain_events_for_rejects_negative_timeout() -> None:
    compiled = _build_compiled()
    with pytest.raises(ValueError, match="non-negative"):
        await compiled.drain_events_for("any-id", timeout=-1.0)


async def test_drain_events_for_rejects_nan_timeout() -> None:
    compiled = _build_compiled()
    with pytest.raises(ValueError, match="non-negative"):
        await compiled.drain_events_for("any-id", timeout=float("nan"))


async def test_drain_events_for_zero_timeout_is_non_blocking_check() -> None:
    # A zero timeout fires immediately if the snapshot target isn't
    # met. Mirrors drain(timeout=0.0)'s non-blocking semantics.
    captured: list[str] = []

    async def slow_obs(_event: ObserverEvent) -> None:
        await asyncio.sleep(0.2)

    async def _capture(_s: _S) -> Mapping[str, Any]:
        captured.append(await _capture_node_invocation_id())
        return {"v": 1}

    builder: GraphBuilder[_S] = GraphBuilder(_S)
    builder.set_entry("n")
    builder.add_node("n", _capture)
    builder.add_edge("n", END)
    compiled = builder.compile()
    compiled.attach_observer(slow_obs)
    await compiled.invoke(_S())

    started = time.monotonic()
    summary = await compiled.drain_events_for(captured[-1], timeout=0.0)
    elapsed = time.monotonic() - started

    assert summary.timeout_reached is True
    assert summary.undelivered_count > 0
    assert elapsed < 0.05


async def test_drain_events_for_snapshot_semantic_does_not_wait_for_own_completed_event() -> None:
    # Mirrors spec fixture 029. A node body calling
    # ``drain_events_for`` from inside itself MUST NOT block on its
    # own ``completed`` event — the snapshot is the ``dispatched``
    # count AT CALL TIME, before the node's completed fires. Without
    # the snapshot semantic the call would deadlock (the node body
    # is awaiting the completed event, but the completed event only
    # fires after the node body returns).
    captured_summary: list[DrainSummary] = []
    delivered_observer_events: list[str] = []

    async def slow_obs(event: ObserverEvent) -> None:
        await asyncio.sleep(0.01)
        if isinstance(event, NodeEvent):
            delivered_observer_events.append(f"{event.node_name}:{event.phase}")

    compiled_ref: list[CompiledGraph[_S]] = []

    async def _drain_from_inside(_s: _S) -> Mapping[str, Any]:
        inv = await _capture_node_invocation_id()
        summary = await compiled_ref[0].drain_events_for(inv, timeout=2.0)
        captured_summary.append(summary)
        return {"v": 1}

    builder: GraphBuilder[_S] = GraphBuilder(_S)
    builder.set_entry("drain_node")
    builder.add_node("drain_node", _drain_from_inside)
    builder.add_edge("drain_node", END)
    compiled = builder.compile()
    compiled_ref.append(compiled)
    compiled.attach_observer(slow_obs)

    # The whole invoke MUST complete within the outer timeout — if
    # drain_events_for waited for the calling node's own completed
    # event, this would deadlock and asyncio.wait_for would fire.
    await asyncio.wait_for(compiled.invoke(_S()), timeout=5.0)

    # The in-node drain returned cleanly with no undelivered events
    # at its snapshot point.
    assert captured_summary[0] == DrainSummary(undelivered_count=0, timeout_reached=False)


async def test_drain_events_for_covers_fan_out_instance_events() -> None:
    # Mirrors spec fixture 032. Fan-out instances share the parent
    # invocation's ``_DrainCounters`` (subgraph descents pass the
    # parent's counters down via ``descend_into_subgraph``), so
    # ``drain_events_for(outer_invocation_id)`` MUST cover every
    # event emitted under any of the fan-out's instance subgraph
    # descents. Without shared counters the drain would miss
    # instance events entirely.

    class _ParentState(State):
        items: list[int] = Field(default_factory=list[int])
        results: Annotated[list[int], append] = Field(default_factory=list[int])

    class _InstanceState(State):
        item: int = 0
        result: int = 0

    async def _double(state: _InstanceState) -> Mapping[str, Any]:
        return {"result": state.item * 2}

    inner = (
        GraphBuilder(_InstanceState)
        .set_entry("double")
        .add_node("double", _double)
        .add_edge("double", END)
        .compile()
    )

    captured_invocation_id: list[str] = []
    delivered: list[str] = []

    async def slow_obs(event: ObserverEvent) -> None:
        await asyncio.sleep(0.005)
        if isinstance(event, NodeEvent):
            delivered.append(f"{event.namespace}/{event.node_name}:{event.phase}")

    async def _capture_and_persist(_s: _ParentState) -> Mapping[str, Any]:
        captured_invocation_id.append(await _capture_node_invocation_id())
        return {}

    builder: GraphBuilder[_ParentState] = GraphBuilder(_ParentState)
    builder.set_entry("fan_out")
    builder.add_fan_out_node(
        "fan_out",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
    )
    builder.add_node("persist", _capture_and_persist)
    builder.add_edge("fan_out", "persist")
    builder.add_edge("persist", END)
    compiled = builder.compile()
    compiled.attach_observer(slow_obs)

    await compiled.invoke(_ParentState(items=[1, 2, 3]))
    summary = await compiled.drain_events_for(captured_invocation_id[0], timeout=5.0)

    assert summary == DrainSummary(undelivered_count=0, timeout_reached=False)
    # Each of the 3 instances' ``double`` node fired a started +
    # completed pair under its own namespace; all six MUST have
    # delivered by the time drain_events_for returned.
    instance_completions = [e for e in delivered if "/double:completed" in e]
    assert len(instance_completions) == 3


async def test_drain_events_for_covers_parallel_branches_events() -> None:
    # Mirrors spec fixture 033. Parallel-branches branches share the
    # parent invocation's ``_DrainCounters`` (same plumbing as
    # fan-out instances), so ``drain_events_for(outer_invocation_id)``
    # MUST cover every event from every branch's inner subgraph.

    from openarmature.graph import BranchSpec

    class _ParentState(State):
        a_out: str = ""
        b_out: str = ""

    class _BranchAState(State):
        v: str = ""

    class _BranchBState(State):
        v: str = ""

    async def _do_a(_s: _BranchAState) -> Mapping[str, Any]:
        return {"v": "a-done"}

    async def _do_b(_s: _BranchBState) -> Mapping[str, Any]:
        return {"v": "b-done"}

    branch_a = (
        GraphBuilder(_BranchAState)
        .set_entry("a_node")
        .add_node("a_node", _do_a)
        .add_edge("a_node", END)
        .compile()
    )
    branch_b = (
        GraphBuilder(_BranchBState)
        .set_entry("b_node")
        .add_node("b_node", _do_b)
        .add_edge("b_node", END)
        .compile()
    )

    captured_invocation_id: list[str] = []
    delivered: list[str] = []

    async def slow_obs(event: ObserverEvent) -> None:
        await asyncio.sleep(0.005)
        if isinstance(event, NodeEvent):
            delivered.append(f"{event.namespace}/{event.node_name}:{event.phase}")

    async def _capture(_s: _ParentState) -> Mapping[str, Any]:
        captured_invocation_id.append(await _capture_node_invocation_id())
        return {}

    builder: GraphBuilder[_ParentState] = GraphBuilder(_ParentState)
    builder.set_entry("dispatcher")
    builder.add_parallel_branches_node(
        "dispatcher",
        branches={
            "branch_a": BranchSpec(subgraph=branch_a, outputs={"a_out": "v"}),
            "branch_b": BranchSpec(subgraph=branch_b, outputs={"b_out": "v"}),
        },
    )
    builder.add_node("persist", _capture)
    builder.add_edge("dispatcher", "persist")
    builder.add_edge("persist", END)
    compiled = builder.compile()
    compiled.attach_observer(slow_obs)

    await compiled.invoke(_ParentState())
    summary = await compiled.drain_events_for(captured_invocation_id[0], timeout=5.0)

    assert summary == DrainSummary(undelivered_count=0, timeout_reached=False)
    # Both branches' inner nodes (``a_node`` + ``b_node``) MUST have
    # delivered their completed events by the time drain_events_for
    # returned.
    a_done = [e for e in delivered if "/a_node:completed" in e]
    b_done = [e for e in delivered if "/b_node:completed" in e]
    assert len(a_done) == 1
    assert len(b_done) == 1
