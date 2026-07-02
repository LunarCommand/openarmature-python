# Spec proposal 0063: tool-call instrumentation scope.
"""Unit tests for the tool-call instrumentation scope.

Exercise ``with_tool_call`` directly by installing a collecting dispatch
callback + a node-scope identity into the correlation ContextVars, the
same mechanism the engine sets up per invocation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from openarmature.graph.events import ToolCallEvent, ToolCallFailedEvent
from openarmature.observability import with_tool_call
from openarmature.observability.correlation import (
    _reset_active_dispatch,
    _reset_invocation_id,
    _reset_namespace_prefix,
    _set_active_dispatch,
    _set_invocation_id,
    _set_namespace_prefix,
)


@contextmanager
def _scope(namespace: tuple[str, ...] = ("run_tool",)) -> Iterator[list[Any]]:
    """Install a collecting dispatch + invocation/namespace identity;
    yield the captured-events list."""
    events: list[Any] = []
    d_tok = _set_active_dispatch(lambda e: events.append(e))
    i_tok = _set_invocation_id("inv-1")
    n_tok = _set_namespace_prefix(namespace)
    try:
        yield events
    finally:
        _reset_namespace_prefix(n_tok)
        _reset_invocation_id(i_tok)
        _reset_active_dispatch(d_tok)


async def test_with_tool_call_dispatches_tool_call_event_on_success() -> None:
    async def _tool() -> dict[str, int]:
        return {"temperature_c": 20}

    with _scope() as events:
        with with_tool_call("get_weather", {"city": "Paris"}, tool_call_id="call_abc") as scope:
            scope.set_result(await _tool())

    tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_events) == 1
    ev = tool_events[0]
    assert ev.tool_name == "get_weather"
    assert ev.tool_call_id == "call_abc"
    assert ev.arguments == {"city": "Paris"}
    assert ev.result == {"temperature_c": 20}
    assert ev.node_name == "run_tool"
    assert ev.namespace == ("run_tool",)
    assert ev.attempt_index == 0
    assert ev.fan_out_index is None
    assert ev.branch_name is None
    assert ev.invocation_id == "inv-1"
    assert ev.latency_ms is not None
    assert ev.call_id  # minted per execution
    assert [e for e in events if isinstance(e, ToolCallFailedEvent)] == []


async def test_with_tool_call_dispatches_failed_event_and_reraises() -> None:
    with _scope() as events:
        # The scope observes; it does NOT swallow — the exception
        # propagates out of the `with` to the caller.
        with pytest.raises(TimeoutError):
            with with_tool_call("get_weather", {"city": "Paris"}, tool_call_id="call_def"):
                raise TimeoutError("tool timed out")

    failed = [e for e in events if isinstance(e, ToolCallFailedEvent)]
    assert len(failed) == 1
    ev = failed[0]
    assert ev.error_type == "TimeoutError"
    assert ev.error_message == "tool timed out"
    assert ev.tool_name == "get_weather"
    assert ev.tool_call_id == "call_def"
    assert ev.arguments == {"city": "Paris"}
    assert ev.latency_ms is not None
    # The deliberate departure: no error_category on a tool failure.
    assert not hasattr(ev, "error_category")
    assert [e for e in events if isinstance(e, ToolCallEvent)] == []


async def test_with_tool_call_tool_call_id_null_for_standalone() -> None:
    # A node-body utility the caller instruments without an originating
    # LLM tool request carries tool_call_id = None.
    with _scope() as events:
        with with_tool_call("compute_total", {"items": 3}) as scope:
            scope.set_result(42)

    ev = next(e for e in events if isinstance(e, ToolCallEvent))
    assert ev.tool_call_id is None
    assert ev.result == 42


async def test_with_tool_call_forgotten_result_is_null_not_a_crash() -> None:
    # If the caller never reports a result, the event carries None
    # (distinct from a tool that returned None — both render null, but
    # neither raises).
    with _scope() as events:
        with with_tool_call("noop", None):
            pass

    ev = next(e for e in events if isinstance(e, ToolCallEvent))
    assert ev.result is None
    assert ev.arguments is None


async def test_with_tool_call_runs_body_without_observers() -> None:
    # Outside any invocation (no dispatch installed) the scope still runs
    # the body and does not raise; it simply emits no event.
    ran = False
    with with_tool_call("t", {}) as scope:
        ran = True
        scope.set_result(1)
    assert ran
