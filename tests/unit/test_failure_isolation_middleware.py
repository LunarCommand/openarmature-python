"""Unit + integration tests for FailureIsolationMiddleware (proposal 0050 §6.3).

Covers the middleware's catch / degrade / predicate / on_caught
contract, the framework-emitted ``FailureIsolatedEvent`` and its field
population, the three-piece composition with ``RetryMiddleware``, and
rendering by the bundled OTel + Langfuse observers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from pydantic import Field

from openarmature.graph import (
    END,
    CaughtException,
    CauseLink,
    FailureIsolatedEvent,
    FailureIsolationMiddleware,
    GraphBuilder,
    ObserverEvent,
    RetryConfig,
    RetryMiddleware,
    State,
    append,
    deterministic_backoff,
)
from openarmature.graph.errors import NodeException, ParallelBranchesBranchFailed
from openarmature.graph.middleware import NextCall
from openarmature.observability.correlation import (
    _reset_active_dispatch,
    _reset_namespace_prefix,
    _set_active_dispatch,
    _set_namespace_prefix,
)


class _TransientError(Exception):
    category = "provider_rate_limit"


class _NonTransientError(Exception):
    category = "provider_invalid_request"


def _raises(exc: BaseException) -> NextCall:
    """A ``next`` callable that raises ``exc`` when invoked."""

    async def _next(_state: Any) -> Mapping[str, Any]:
        raise exc

    return _next


# ---------------------------------------------------------------------------
# Catch + degrade
# ---------------------------------------------------------------------------


async def test_static_degraded_update_returned_on_catch() -> None:
    mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="x_failed")
    out = await mw("state-in", _raises(ValueError("boom")))
    assert out == {"result": []}


async def test_callable_degraded_update_receives_pre_state() -> None:
    seen: list[Any] = []

    def degrade(state: Any) -> Mapping[str, Any]:
        seen.append(state)
        return {"result": ["fallback"]}

    mw = FailureIsolationMiddleware(degraded_update=degrade, event_name="x_failed")
    out = await mw("state-in", _raises(ValueError("boom")))
    assert out == {"result": ["fallback"]}
    assert seen == ["state-in"]


async def test_success_passes_through_untouched() -> None:
    async def _ok(_state: Any) -> Mapping[str, Any]:
        return {"result": ["real"]}

    mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="x_failed")
    out = await mw("state-in", _ok)
    assert out == {"result": ["real"]}


# ---------------------------------------------------------------------------
# Predicate filtering
# ---------------------------------------------------------------------------


async def test_predicate_true_catches() -> None:
    mw = FailureIsolationMiddleware(
        degraded_update={"result": []},
        event_name="x_failed",
        predicate=lambda exc: isinstance(exc, ValueError),
    )
    out = await mw("s", _raises(ValueError("boom")))
    assert out == {"result": []}


async def test_predicate_false_propagates() -> None:
    mw = FailureIsolationMiddleware(
        degraded_update={"result": []},
        event_name="x_failed",
        predicate=lambda exc: isinstance(exc, ValueError),
    )
    with pytest.raises(KeyError):
        await mw("s", _raises(KeyError("nope")))


# ---------------------------------------------------------------------------
# on_caught hook
# ---------------------------------------------------------------------------


async def test_on_caught_fires_once_and_degrade_still_returned() -> None:
    caught: list[BaseException] = []

    async def on_caught(exc: Exception) -> None:
        caught.append(exc)

    exc = ValueError("boom")
    mw = FailureIsolationMiddleware(
        degraded_update={"result": []},
        event_name="x_failed",
        on_caught=on_caught,
    )
    out = await mw("s", _raises(exc))
    assert out == {"result": []}
    assert caught == [exc]


async def test_on_caught_raise_is_isolated_event_emitted_degrade_returned() -> None:
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))

    async def bad_hook(_exc: Exception) -> None:
        raise RuntimeError("hook boom")

    mw = FailureIsolationMiddleware(
        degraded_update={"result": []},
        event_name="x_failed",
        on_caught=bad_hook,
    )
    try:
        with pytest.warns(UserWarning, match="on_caught raised RuntimeError"):
            out = await mw("s", _raises(ValueError("boom")))
    finally:
        _reset_active_dispatch(disp_token)

    # A buggy on_caught is isolated: the degrade still happens, and the
    # event was already dispatched (emit precedes on_caught).
    assert out == {"result": []}
    assert len(events) == 1
    assert isinstance(events[0], FailureIsolatedEvent)


# ---------------------------------------------------------------------------
# Cancellation propagates
# ---------------------------------------------------------------------------


async def test_base_exception_propagates() -> None:
    mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="x_failed")
    with pytest.raises(asyncio.CancelledError):
        await mw("s", _raises(asyncio.CancelledError()))


# ---------------------------------------------------------------------------
# Framework event emission (via the dispatch ContextVar, no engine)
# ---------------------------------------------------------------------------


async def test_emits_failure_isolated_event_with_fields() -> None:
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    ns_token = _set_namespace_prefix(("segment",))
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="segment_failed")
        out = await mw("state-in", _raises(_TransientError("rate limited")))
    finally:
        _reset_namespace_prefix(ns_token)
        _reset_active_dispatch(disp_token)

    assert out == {"result": []}
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, FailureIsolatedEvent)
    assert ev.event_name == "segment_failed"
    assert ev.namespace == ("segment",)
    assert ev.attempt_index == 0
    assert ev.pre_state == "state-in"
    assert ev.post_state == {"result": []}
    assert isinstance(ev.caught_exception, CaughtException)
    assert ev.caught_exception.category == "provider_rate_limit"
    assert ev.caught_exception.message == "rate limited"


async def test_bare_exception_has_null_category() -> None:
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="x_failed")
        await mw("s", _raises(ValueError("plain")))
    finally:
        _reset_active_dispatch(disp_token)

    assert len(events) == 1
    assert events[0].caught_exception.category is None
    assert events[0].caught_exception.message == "plain"


# ---------------------------------------------------------------------------
# Cause fidelity at carrier-wrapper sites (proposal 0065)
# ---------------------------------------------------------------------------


async def test_node_exception_carrier_resolves_to_originating_category() -> None:
    # At a non-node placement the engine wraps the originating error as a
    # node_exception carrier before the middleware catches it; the event
    # reports the originating category, NOT the masking node_exception.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    carrier = NodeException(node_name="work", cause=_TransientError("rate limited"), recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        out = await mw("s", _raises(carrier))
    finally:
        _reset_active_dispatch(disp_token)

    assert out == {"result": []}
    assert isinstance(events[0], FailureIsolatedEvent)
    assert events[0].caught_exception.category == "provider_rate_limit"
    assert events[0].caught_exception.message == "rate limited"


async def test_nested_carriers_resolve_to_originating_category() -> None:
    # Nested subgraph boundaries stack node_exception carriers; resolution
    # walks all of them to the originating cause.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    inner = NodeException(node_name="inner", cause=_TransientError("rate limited"), recoverable_state={})
    outer = NodeException(node_name="outer", cause=inner, recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(outer))
    finally:
        _reset_active_dispatch(disp_token)

    assert events[0].caught_exception.category == "provider_rate_limit"


async def test_branch_carrier_subtype_resolves_to_originating_category() -> None:
    # The §11.7 branch site catches a ParallelBranchesBranchFailed (a
    # NodeException subtype); resolution still reaches the originating cause.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    carrier = ParallelBranchesBranchFailed(
        node_name="dispatcher",
        cause=_TransientError("rate limited"),
        recoverable_state={},
        branch_name="only",
    )
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(carrier))
    finally:
        _reset_active_dispatch(disp_token)

    assert events[0].caught_exception.category == "provider_rate_limit"


async def test_carrier_over_uncategorized_cause_is_null() -> None:
    # Resolving through the carrier reaches a cause with no category, so the
    # reported category is null (the existing bare-exception rule).
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    carrier = NodeException(node_name="work", cause=ValueError("boom"), recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(carrier))
    finally:
        _reset_active_dispatch(disp_token)

    assert events[0].caught_exception.category is None
    assert events[0].caught_exception.message == "boom"


async def test_uncategorized_surface_resolves_to_categorized_cause() -> None:
    # A node that wraps a categorized provider error in an uncategorized
    # domain error: the event surfaces the underlying provider category
    # (agreeing with what §6.1's classifier retries on), and the message
    # tracks that same cause for coherence.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    surface = ValueError("wrapped")
    surface.__cause__ = _TransientError("rate limited")
    carrier = NodeException(node_name="work", cause=surface, recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(carrier))
    finally:
        _reset_active_dispatch(disp_token)

    assert events[0].caught_exception.category == "provider_rate_limit"
    assert events[0].caught_exception.message == "rate limited"


async def test_categorized_surface_wins_over_deeper_cause() -> None:
    # A node that deliberately re-categorizes (raises a categorized error
    # FROM a categorized cause): the nearest category wins, so the node's
    # re-categorization is respected rather than the deeper cause.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    surface = _NonTransientError("misconfigured")
    surface.__cause__ = _TransientError("rate limited")
    carrier = NodeException(node_name="work", cause=surface, recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(carrier))
    finally:
        _reset_active_dispatch(disp_token)

    assert events[0].caught_exception.category == "provider_invalid_request"
    assert events[0].caught_exception.message == "misconfigured"


async def test_cyclic_cause_chain_terminates() -> None:
    # Defensive: a self-referential __cause__ chain must not hang the
    # degrade path. Resolution terminates and the node still degrades.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    a = NodeException(node_name="a", cause=ValueError("seed"), recoverable_state={})
    b = NodeException(node_name="b", cause=a, recoverable_state={})
    a.__cause__ = b  # cycle: a -> b -> a
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        out = await mw("s", _raises(a))
    finally:
        _reset_active_dispatch(disp_token)

    assert out == {"result": []}
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Cause chain (proposal 0068)
# ---------------------------------------------------------------------------


async def test_chain_records_carrier_then_categorized_cause() -> None:
    # Instance-style placement: one engine carrier over a categorized
    # originating cause. The chain records the carrier (flagged) then the
    # non-carrier cause; the derived category is the non-carrier's.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    carrier = NodeException(node_name="work", cause=_TransientError("rate limited"), recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(carrier))
    finally:
        _reset_active_dispatch(disp_token)

    assert events[0].caught_exception.chain == (
        CauseLink(category="node_exception", message=str(carrier), carrier=True),
        CauseLink(category="provider_rate_limit", message="rate limited", carrier=False),
    )


async def test_chain_node_level_is_single_non_carrier_link() -> None:
    # Node-level placement: the middleware catches the raw error (no carrier),
    # so the chain is a single non-carrier link.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(_TransientError("rate limited")))
    finally:
        _reset_active_dispatch(disp_token)

    assert events[0].caught_exception.chain == (
        CauseLink(category="provider_rate_limit", message="rate limited", carrier=False),
    )


async def test_chain_records_nested_carriers_then_cause() -> None:
    # Nested carriers (carrier -> carrier -> categorized cause), the case the
    # pinned fixture 066 does not cover: both carriers are recorded and
    # flagged, then the originating link, and the derived category is the
    # originating one.
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    inner = NodeException(node_name="inner", cause=_TransientError("rate limited"), recoverable_state={})
    outer = NodeException(node_name="outer", cause=inner, recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(outer))
    finally:
        _reset_active_dispatch(disp_token)

    chain = events[0].caught_exception.chain
    assert [(link.category, link.carrier) for link in chain] == [
        ("node_exception", True),
        ("node_exception", True),
        ("provider_rate_limit", False),
    ]
    assert events[0].caught_exception.category == "provider_rate_limit"


async def test_chain_carries_both_non_carrier_links_on_recategorization() -> None:
    # Two non-carrier links (a re-categorized surface over a deeper cause): the
    # chain carries both, and the derived category is the OUTERMOST non-carrier
    # (proposal 0068's surface-wins derivation).
    events: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: events.append(e))
    surface = _NonTransientError("misconfigured")
    surface.__cause__ = _TransientError("rate limited")
    carrier = NodeException(node_name="work", cause=surface, recoverable_state={})
    try:
        mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="iso")
        await mw("s", _raises(carrier))
    finally:
        _reset_active_dispatch(disp_token)

    chain = events[0].caught_exception.chain
    assert [(link.category, link.carrier) for link in chain] == [
        ("node_exception", True),
        ("provider_invalid_request", False),
        ("provider_rate_limit", False),
    ]
    assert events[0].caught_exception.category == "provider_invalid_request"
    assert events[0].caught_exception.message == "misconfigured"


async def test_no_event_outside_invocation() -> None:
    # current_dispatch() is None outside an invocation; the degrade still
    # happens, no event is emitted, and nothing raises.
    mw = FailureIsolationMiddleware(degraded_update={"result": []}, event_name="x_failed")
    out = await mw("s", _raises(ValueError("boom")))
    assert out == {"result": []}


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class _DocState(State):
    points: Annotated[list[str], append] = Field(default_factory=list)
    note: str = ""


async def _extract_raises(_s: _DocState) -> Mapping[str, Any]:
    raise _TransientError("provider down")


def _isolated_extract_graph() -> Any:
    return (
        GraphBuilder(_DocState)
        .add_node(
            "extract",
            _extract_raises,
            middleware=[
                FailureIsolationMiddleware(
                    degraded_update={"note": "degraded"},
                    event_name="extract_failed",
                )
            ],
        )
        .add_edge("extract", END)
        .set_entry("extract")
        .compile()
    )


async def test_degrade_dont_crash_via_invoke() -> None:
    graph = _isolated_extract_graph()
    events: list[ObserverEvent] = []

    async def rec(event: ObserverEvent) -> None:
        events.append(event)

    final = await graph.invoke(_DocState(), observers=[rec])
    await graph.drain()

    # The node degraded gracefully; the invocation succeeded with the
    # configured fallback applied rather than raising.
    assert final.note == "degraded"

    isolated = [e for e in events if isinstance(e, FailureIsolatedEvent)]
    assert len(isolated) == 1
    assert isolated[0].event_name == "extract_failed"
    assert isolated[0].namespace == ("extract",)
    assert isolated[0].caught_exception.category == "provider_rate_limit"
    assert isolated[0].post_state == {"note": "degraded"}


async def test_three_piece_composition_with_retry() -> None:
    attempts = {"n": 0}

    async def _flaky(_s: _DocState) -> Mapping[str, Any]:
        attempts["n"] += 1
        raise _TransientError("still down")

    graph = (
        GraphBuilder(_DocState)
        .add_node(
            "flaky",
            _flaky,
            # Outer-to-inner: failure isolation OUTER, retry INNER.
            middleware=[
                FailureIsolationMiddleware(
                    degraded_update={"note": "gave_up"},
                    event_name="flaky_failed",
                ),
                RetryMiddleware(RetryConfig(max_attempts=3, backoff=deterministic_backoff(0.0))),
            ],
        )
        .add_edge("flaky", END)
        .set_entry("flaky")
        .compile()
    )

    isolated: list[FailureIsolatedEvent] = []

    async def _capture(event: ObserverEvent) -> None:
        if isinstance(event, FailureIsolatedEvent):
            isolated.append(event)

    graph.attach_observer(_capture)
    final = await graph.invoke(_DocState())
    await graph.drain()

    # Retry exhausted its 3 attempts; failure isolation then caught the
    # propagated exhaustion exception and substituted the degraded value.
    assert attempts["n"] == 3
    assert final.note == "gave_up"
    # The event's attempt_index is the final / exhausting attempt (2 after
    # attempts 0/1/2), not the post-reset baseline (proposal 0050 §6.3
    # lineage correlation).
    assert len(isolated) == 1
    assert isolated[0].attempt_index == 2


# ---------------------------------------------------------------------------
# Bundled-observer rendering
# ---------------------------------------------------------------------------


async def test_otel_renders_failure_isolated_span() -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from openarmature.observability.otel.observer import OTelObserver

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    graph = _isolated_extract_graph()
    graph.attach_observer(observer)

    await graph.invoke(_DocState())
    await graph.drain()
    observer.shutdown()

    spans = exporter.get_finished_spans()
    marker = next((s for s in spans if s.name == "openarmature.failure_isolated"), None)
    assert marker is not None, f"no failure_isolated span; got {[s.name for s in spans]}"
    attrs = dict(marker.attributes or {})
    assert attrs.get("openarmature.failure_isolation.event_name") == "extract_failed"
    assert attrs.get("openarmature.failure_isolation.node") == "extract"
    assert attrs.get("openarmature.error.category") == "provider_rate_limit"


async def test_langfuse_renders_failure_isolated_observation() -> None:
    from openarmature.observability.langfuse.client import InMemoryLangfuseClient
    from openarmature.observability.langfuse.observer import LangfuseObserver

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    graph = _isolated_extract_graph()
    graph.attach_observer(observer)

    await graph.invoke(_DocState())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    marker = next(
        (o for o in trace.observations if o.name == "openarmature.failure_isolated"),
        None,
    )
    assert marker is not None, f"no failure_isolated observation; got {[o.name for o in trace.observations]}"
    assert marker.metadata.get("failure_isolation_event_name") == "extract_failed"
    assert marker.metadata.get("failure_isolation_node") == "extract"
    assert marker.metadata.get("error_category") == "provider_rate_limit"
