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
from typing import Any

import pytest

from openarmature.graph import (
    END,
    CompiledGraph,
    DrainSummary,
    GraphBuilder,
    NodeEvent,
    ObserverEvent,
    State,
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
