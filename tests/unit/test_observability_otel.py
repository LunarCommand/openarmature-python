"""OTel-specific observability unit tests (extras-gated).

Skipped cleanly when the ``otel`` extras aren't installed — the
import-time check in
``openarmature.observability.otel.__init__`` raises ImportError on
missing deps.

These tests fill the gaps the conformance harness defers:

- §6 TracerProvider isolation — the load-bearing "spans don't leak
  into the OTel global provider" guarantee.
- §5 attribute population on every span type.
- §4.2 status mapping for every §4 error category.
- §5.5 LLM provider span via the ContextVar dispatch hook (queue-
  mediated; no synchronous direct dispatch).
- §4.4 detached trace mode key separation in the span stack.
- §10.8 checkpoint_saved → ``openarmature.checkpoint.save`` zero-
  duration span.
- §7 log bridge filter + correlation_id injection.
"""

from __future__ import annotations

import logging

import pytest

# Skip the entire module if otel extras aren't installed.
pytest.importorskip("opentelemetry.sdk.trace")

from typing import cast

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from openarmature.checkpoint import InMemoryCheckpointer
from openarmature.graph import (
    END,
    GraphBuilder,
    NodeException,
    State,
)
from openarmature.observability.otel import OTelObserver, install_log_bridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LinearState(State):
    a: int = 0
    b: int = 0


async def _node_a(_s: _LinearState) -> dict[str, int]:
    return {"a": 1}


async def _node_b(_s: _LinearState) -> dict[str, int]:
    return {"b": 2}


def _build_linear_graph(
    observer: OTelObserver | None = None,
) -> tuple[
    object,
    InMemorySpanExporter,
]:
    """Build a 2-node linear graph wired to a fresh OTelObserver +
    in-memory exporter; returns (compiled_graph, exporter)."""
    exporter = InMemorySpanExporter()
    if observer is None:
        observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_LinearState)
        .add_node("node_a", _node_a)
        .add_node("node_b", _node_b)
        .add_edge("node_a", "node_b")
        .add_edge("node_b", END)
        .set_entry("node_a")
        .compile()
    )
    g.attach_observer(observer)
    return g, exporter


# ---------------------------------------------------------------------------
# §6 TracerProvider isolation
# ---------------------------------------------------------------------------


async def test_observer_uses_private_provider_not_global() -> None:
    """Spec §6 TracerProvider isolation: the OTelObserver MUST use a
    PRIVATE TracerProvider; spans MUST NOT appear on the OTel global
    provider's exporter (this is the load-bearing guarantee against
    duplicate spans from external auto-instrumentation libraries)."""
    # Save the prior global provider so we can restore it after the
    # test — pytest fixture-scoping doesn't cover OTel global state.
    prior_global = otel_trace.get_tracer_provider()
    # Install a separate exporter on the OTel global provider.
    global_exporter = InMemorySpanExporter()
    global_provider = TracerProvider()
    global_provider.add_span_processor(SimpleSpanProcessor(global_exporter))
    otel_trace.set_tracer_provider(global_provider)

    try:
        # Drive a graph through OTelObserver.
        private_exporter = InMemorySpanExporter()
        observer = OTelObserver(span_processor=SimpleSpanProcessor(private_exporter))
        g, _ = _build_linear_graph(observer)
        await g.invoke(_LinearState())  # type: ignore[attr-defined]
        await g.drain()  # type: ignore[attr-defined]
        observer.shutdown()

        private_spans = private_exporter.get_finished_spans()
        global_spans = global_exporter.get_finished_spans()
        assert len(private_spans) > 0, "private provider must have received spans"
        assert len(global_spans) == 0, (
            f"global provider MUST NOT receive openarmature spans; got {[s.name for s in global_spans]}"
        )
    finally:
        otel_trace.set_tracer_provider(prior_global)


# ---------------------------------------------------------------------------
# §5 attribute population
# ---------------------------------------------------------------------------


async def test_node_span_carries_required_attributes() -> None:
    """Spec §5.2: every node span MUST carry the four
    ``openarmature.node.*`` base attributes."""
    g, exporter = _build_linear_graph()
    await g.invoke(_LinearState(), correlation_id="test-cid")  # type: ignore[attr-defined]
    await g.drain()  # type: ignore[attr-defined]
    spans = exporter.get_finished_spans()
    node_spans = [s for s in spans if s.name in {"node_a", "node_b"}]
    assert len(node_spans) == 2
    for span in node_spans:
        attrs = dict(span.attributes or {})
        assert attrs.get("openarmature.node.name") == span.name
        assert isinstance(attrs.get("openarmature.node.namespace"), tuple | list)
        assert isinstance(attrs.get("openarmature.node.step"), int)
        assert attrs.get("openarmature.node.attempt_index") == 0
        # Cross-cutting correlation_id (§5.6).
        assert attrs.get("openarmature.correlation_id") == "test-cid"


async def test_invocation_span_carries_required_attributes() -> None:
    """Spec §5.1: invocation span MUST carry
    ``openarmature.graph.entry_node`` + ``openarmature.graph.spec_version``."""
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g, _ = _build_linear_graph(observer)
    await g.invoke(_LinearState())  # type: ignore[attr-defined]
    await g.drain()  # type: ignore[attr-defined]
    # Invocation span closes on observer shutdown — the engine has
    # no per-invocation lifecycle hook on observers, so the user
    # closes the observer when done with their batch of invocations.
    observer.shutdown()
    spans = exporter.get_finished_spans()
    inv = next((s for s in spans if s.name == "openarmature.invocation"), None)
    assert inv is not None
    attrs = dict(inv.attributes or {})
    assert attrs.get("openarmature.graph.entry_node") == "node_a"
    assert isinstance(attrs.get("openarmature.graph.spec_version"), str)


# ---------------------------------------------------------------------------
# §4.2 status mapping
# ---------------------------------------------------------------------------


class _FailState(State):
    a: int = 0


async def _failing_node(_s: _FailState) -> dict[str, int]:
    raise RuntimeError("boom")


async def test_failing_node_span_carries_error_status() -> None:
    """Spec §4.2: a node-exception failure produces a span with
    ERROR status, an exception event recorded, and the
    ``openarmature.error.category`` attribute on the span."""
    from opentelemetry.trace import StatusCode

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_FailState)
        .add_node("flaky", _failing_node)
        .add_edge("flaky", END)
        .set_entry("flaky")
        .compile()
    )
    g.attach_observer(observer)
    with pytest.raises(NodeException):
        await g.invoke(_FailState())
    await g.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()
    flaky = next((s for s in spans if s.name == "flaky"), None)
    assert flaky is not None
    assert flaky.status.status_code == StatusCode.ERROR
    attrs = dict(flaky.attributes or {})
    assert attrs.get("openarmature.error.category") == "node_exception"


# ---------------------------------------------------------------------------
# §10.8 checkpoint_saved → 0-duration span
# ---------------------------------------------------------------------------


async def test_checkpoint_save_emits_zero_duration_span() -> None:
    """Spec §10.8: a checkpoint save SHOULD emit a §6-style observer
    event surfaced as a span. Our implementation emits a
    ``openarmature.checkpoint.save`` span on every save."""
    cp = InMemoryCheckpointer()
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_LinearState)
        .add_node("node_a", _node_a)
        .add_edge("node_a", END)
        .set_entry("node_a")
        .with_checkpointer(cp)
        .compile()
    )
    # Subscribe to the checkpoint_saved phase (default subscription
    # excludes it; OTelObserver attaches with the explicit set).
    g.attach_observer(observer, phases={"started", "completed", "checkpoint_saved"})
    await g.invoke(_LinearState())
    await g.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()
    save_spans = [s for s in spans if s.name == "openarmature.checkpoint.save"]
    assert len(save_spans) == 1
    save_span = save_spans[0]
    # Zero-duration: end_time - start_time near 0 (exact equality
    # depends on monotonic clock resolution; permit small jitter).
    end_t = save_span.end_time
    start_t = save_span.start_time
    assert end_t is not None and start_t is not None
    duration = end_t - start_t
    assert duration < 1_000_000, f"expected near-zero duration; got {duration}ns"


# ---------------------------------------------------------------------------
# §5.5 disable_llm_spans
# ---------------------------------------------------------------------------


async def test_disable_llm_spans_skips_llm_provider_span() -> None:
    """Spec §5.5: ``disable_llm_spans=True`` MUST suppress the
    LLM-provider span emission while leaving all other spans intact."""
    from openarmature.graph.events import NodeEvent

    # We don't drive a real provider here; instead we emit a synthetic
    # LLM event through the observer's __call__ and assert no span was
    # produced. This isolates the disable_llm_spans branch from the
    # provider's own queue-dispatch wiring.
    from openarmature.llm.providers.openai import _LlmEventState

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        disable_llm_spans=True,
    )
    started = NodeEvent(
        node_name="openarmature.llm.complete",
        namespace=("openarmature.llm.complete",),
        step=-1,
        phase="started",
        pre_state=_LlmEventState(call_id="test-call-1", model="test-m"),
        post_state=None,
        error=None,
        parent_states=(),
    )
    completed = NodeEvent(
        node_name="openarmature.llm.complete",
        namespace=("openarmature.llm.complete",),
        step=-1,
        phase="completed",
        pre_state=_LlmEventState(call_id="test-call-1", model="test-m", finish_reason="stop"),
        post_state=None,
        error=None,
        parent_states=(),
    )
    await observer(started)
    await observer(completed)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert llm_spans == []


# ---------------------------------------------------------------------------
# §7 log bridge: correlation_id injection
# ---------------------------------------------------------------------------


def test_log_bridge_filter_injects_correlation_id() -> None:
    """Spec §7: every log record emitted during an invocation MUST
    carry ``openarmature.correlation_id``. The bridge's filter reads
    the ContextVar and attaches it to the LogRecord."""
    from openarmature.observability.correlation import (
        _reset_correlation_id,
        _set_correlation_id,
    )
    from openarmature.observability.otel.logs import _CorrelationIdFilter

    flt = _CorrelationIdFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=None,
        exc_info=None,
    )
    # Outside an invocation: no correlation_id attribute set.
    assert flt.filter(record) is True
    assert not hasattr(record, "openarmature.correlation_id")

    # Inside an invocation: filter attaches the ContextVar value.
    record2 = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello",
        args=None,
        exc_info=None,
    )
    token = _set_correlation_id("my-cid-42")
    try:
        flt.filter(record2)
    finally:
        _reset_correlation_id(token)
    assert getattr(record2, "openarmature.correlation_id") == "my-cid-42"


def test_install_log_bridge_is_idempotent() -> None:
    """Re-calling :func:`install_log_bridge` MUST NOT register a
    duplicate handler — the bridge owns the only OA-flagged
    LoggingHandler on the root logger."""
    from opentelemetry.sdk._logs import LoggerProvider

    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_filters = list(root.filters)
    try:
        provider = LoggerProvider()
        install_log_bridge(provider)
        handler_count_before = len(root.handlers)
        install_log_bridge(provider)
        handler_count_after = len(root.handlers)
        assert handler_count_before == handler_count_after
    finally:
        # install_log_bridge mutates the process-wide root logger;
        # restore the prior handler + filter set so this test does
        # not leak state into others.
        root.handlers[:] = prior_handlers
        root.filters[:] = prior_filters


# ---------------------------------------------------------------------------
# Phase 6.1: concurrency-safe state scoping + §5.5 calling-node attribution
# ---------------------------------------------------------------------------


async def test_shared_observer_concurrent_invocations_dont_collide() -> None:
    """A single observer shared across concurrent invocations MUST
    keep their span trees isolated. Per spec §5.1 each invocation
    has its own ``invocation_id`` and therefore its own
    ``trace_id``; with shared internal state keyed by
    ``invocation_id`` the observer no longer collides on overlapping
    namespaces, no longer closes another in-flight invocation's span
    on a new event, and produces N distinct trace_ids for N
    concurrent invocations on the same compiled graph."""
    import asyncio

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_LinearState)
        .add_node("node_a", _node_a)
        .add_node("node_b", _node_b)
        .add_edge("node_a", "node_b")
        .add_edge("node_b", END)
        .set_entry("node_a")
        .compile()
    )
    g.attach_observer(observer)

    n = 5
    results = await asyncio.gather(*[g.invoke(_LinearState()) for _ in range(n)])
    await g.drain()
    observer.shutdown()
    assert len(results) == n

    spans = exporter.get_finished_spans()
    invocation_spans = [s for s in spans if s.name == "openarmature.invocation"]
    assert len(invocation_spans) == n, (
        f"expected one invocation span per concurrent invocation; got {len(invocation_spans)}"
    )
    # Each invocation has its own trace_id.
    trace_ids: set[int] = set()
    for s in invocation_spans:
        assert s.context is not None
        trace_ids.add(s.context.trace_id)
    assert len(trace_ids) == n, (
        f"each concurrent invocation MUST have its own trace_id; got {len(trace_ids)} for {n} invocations"
    )
    # Every span in the export belongs to one of those trace_ids
    # (no orphans pointing at a stale trace).
    for s in spans:
        assert s.context is not None
        assert s.context.trace_id in trace_ids, (
            f"span {s.name!r} carries unknown trace_id {s.context.trace_id}"
        )
    # Each trace has the expected node count: one invocation span +
    # node_a + node_b = 3 spans.
    by_trace: dict[int, list[str]] = {tid: [] for tid in trace_ids}
    for s in spans:
        assert s.context is not None
        by_trace[s.context.trace_id].append(s.name)
    for tid, names_list in by_trace.items():
        names = sorted(names_list)
        assert names == ["node_a", "node_b", "openarmature.invocation"], (
            f"trace {tid:x} span set MUST be exactly the invocation + node_a + node_b; got {names}"
        )


async def test_concurrent_fan_out_no_lifo_violation() -> None:
    """Regression check: under fan-out with multiple concurrent
    instances, started/completed events for different instances
    interleave on the observer's call queue. The Phase 6.0
    architecture used cross-event ``opentelemetry.context.attach``
    tokens that produced LIFO violations on out-of-order detach
    (suppressed by try/except guards in round-4 / round-7). Phase
    6.1 derives parents from internal maps within a single event
    handler — no tokens cross event boundaries — so the underlying
    hazard goes away. This test drives a fan-out with three
    instances and asserts the run completes without the warnings
    that the suppressed guards would have produced."""
    import warnings

    class _ParentState(State):
        items: list[int] = []
        results: list[int] = []

    class _ChildState(State):
        item: int = 0
        out: int = 0

    async def _double(s: _ChildState) -> dict[str, int]:
        # Yield to give other instances a chance to interleave their
        # started/completed events on the observer queue.
        import asyncio

        await asyncio.sleep(0)
        return {"out": s.item * 2}

    inner = (
        GraphBuilder(_ChildState)
        .add_node("double", _double)
        .add_edge("double", END)
        .set_entry("double")
        .compile()
    )
    parent = (
        GraphBuilder(_ParentState)
        .add_fan_out_node(
            "fan",
            subgraph=inner,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
            concurrency=3,
        )
        .add_edge("fan", END)
        .set_entry("fan")
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    compiled = parent.compile()
    compiled.attach_observer(observer)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = await compiled.invoke(_ParentState(items=[1, 2, 3, 4, 5]))
    await compiled.drain()
    observer.shutdown()

    assert result.results == [2, 4, 6, 8, 10]
    # Sanity: per-instance node spans landed (one ``double`` span
    # per item, all sharing the same trace_id since the fan-out is
    # not configured detached).
    spans = exporter.get_finished_spans()
    double_spans = [s for s in spans if s.name == "double"]
    assert len(double_spans) == 5, f"expected 5 per-instance node spans; got {len(double_spans)}"


async def test_concurrent_fan_out_llm_spans_parent_under_calling_instance() -> None:
    """Spec §5.5 under concurrent fan-out: each instance's
    ``openarmature.llm.complete`` span MUST parent under that
    instance's calling node, not a sibling instance's. The Phase 6.1
    calling-node identity (namespace_prefix + attempt_index +
    fan_out_index threaded via ContextVar onto the LLM event
    payload) is what makes this attribution correct."""
    import asyncio

    import httpx

    from openarmature.llm.messages import UserMessage
    from openarmature.llm.providers.openai import OpenAIProvider

    def _ok(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_ok),
    )

    class _ParentState(State):
        items: list[int] = []
        outs: list[str] = []

    class _ChildState(State):
        item: int = 0
        out: str = ""

    async def _ask(s: _ChildState) -> dict[str, str]:
        # Yield first so peer instances can interleave.
        await asyncio.sleep(0)
        resp = await provider.complete([UserMessage(content=str(s.item))])
        return {"out": str(resp.message.content or "")}

    inner = GraphBuilder(_ChildState).add_node("ask", _ask).add_edge("ask", END).set_entry("ask").compile()
    parent = (
        GraphBuilder(_ParentState)
        .add_fan_out_node(
            "fan",
            subgraph=inner,
            collect_field="out",
            target_field="outs",
            items_field="items",
            item_field="item",
            concurrency=4,
        )
        .add_edge("fan", END)
        .set_entry("fan")
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    compiled = parent.compile()
    compiled.attach_observer(observer)

    n = 4
    try:
        await compiled.invoke(_ParentState(items=list(range(n))))
        await compiled.drain()
    finally:
        await provider.aclose()
    observer.shutdown()

    spans = exporter.get_finished_spans()
    by_id: dict[int, ReadableSpan] = {}
    for s in spans:
        assert s.context is not None
        by_id[s.context.span_id] = s
    llm_spans = [s for s in spans if s.name == "openarmature.llm.complete"]
    ask_spans = [s for s in spans if s.name == "ask"]
    assert len(llm_spans) == n, f"expected one LLM span per instance; got {len(llm_spans)}"
    assert len(ask_spans) == n, f"expected one ``ask`` span per instance; got {len(ask_spans)}"

    # Build a map from fan_out_index → ask span_id (each instance's
    # node carries its own ``openarmature.node.fan_out_index`` attribute).
    ask_by_index: dict[int, int] = {}
    for s in ask_spans:
        assert s.context is not None and s.attributes is not None
        idx_attr = s.attributes["openarmature.node.fan_out_index"]
        assert isinstance(idx_attr, int)
        ask_by_index[idx_attr] = s.context.span_id
    assert set(ask_by_index.keys()) == set(range(n))

    # For each LLM span, confirm the parent span_id is one of the
    # ``ask`` spans (calling instance's node), not a sibling
    # fan-out instance's span.
    parented_ask_ids: set[int] = set()
    for llm in llm_spans:
        assert llm.parent is not None, "LLM span MUST have a parent"
        parent_span = by_id.get(llm.parent.span_id)
        assert parent_span is not None, f"LLM span parent_id {llm.parent.span_id} not in exported set"
        assert parent_span.name == "ask", (
            f"LLM span MUST parent under ``ask`` (the calling node), got {parent_span.name!r}"
        )
        parented_ask_ids.add(llm.parent.span_id)

    # Every LLM span parents under a UNIQUE ``ask`` span — i.e., no
    # collision where two LLM calls attributed to the same instance.
    assert len(parented_ask_ids) == n, (
        f"each LLM call MUST parent under its own calling instance; "
        f"got {len(parented_ask_ids)} distinct parents for {n} calls"
    )


async def test_llm_call_inside_retried_node_parents_per_attempt() -> None:
    """Spec §5.5 under retry: when an LLM ``complete()`` call
    happens inside a node body wrapped with retry middleware, each
    attempt's LLM span MUST parent under THAT attempt's node span,
    not a hardcoded ``attempt_index=0``. Phase 6.1's
    ``current_attempt_index`` ContextVar (set inside the per-attempt
    ``innermost`` scope) is what makes this work."""
    import httpx

    from openarmature.graph.middleware import RetryMiddleware
    from openarmature.llm.errors import ProviderRateLimit
    from openarmature.llm.messages import UserMessage
    from openarmature.llm.providers.openai import OpenAIProvider

    def _ok(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    provider = OpenAIProvider(
        base_url="http://test",
        model="m",
        api_key="k",
        transport=httpx.MockTransport(_ok),
    )

    class _S(State):
        attempts: int = 0

    # Mutable counter so the node body can observe its own attempt
    # index and decide whether to fail. Two failures + one success.
    flaky_state = {"calls": 0}

    async def _flaky(s: _S) -> dict[str, int]:
        flaky_state["calls"] += 1
        # Always issue an LLM call BEFORE the conditional raise so a
        # span fires for every attempt, including the failing ones.
        await provider.complete([UserMessage(content="hi")])
        if flaky_state["calls"] < 3:
            raise ProviderRateLimit("transient")
        return {"attempts": flaky_state["calls"]}

    g = (
        GraphBuilder(_S)
        .add_node("flaky", _flaky, middleware=[RetryMiddleware(max_attempts=3, backoff=lambda _i: 0.0)])
        .add_edge("flaky", END)
        .set_entry("flaky")
        .compile()
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g.attach_observer(observer)

    try:
        result = await g.invoke(_S())
        await g.drain()
    finally:
        await provider.aclose()
    observer.shutdown()

    assert result.attempts == 3
    spans = exporter.get_finished_spans()
    by_id: dict[int, ReadableSpan] = {}
    for s in spans:
        assert s.context is not None
        by_id[s.context.span_id] = s

    # Three ``flaky`` spans (one per attempt), three LLM spans.
    flaky_spans = [s for s in spans if s.name == "flaky"]
    llm_spans = [s for s in spans if s.name == "openarmature.llm.complete"]
    assert len(flaky_spans) == 3, f"expected 3 attempt spans; got {len(flaky_spans)}"
    assert len(llm_spans) == 3, f"expected 3 LLM spans; got {len(llm_spans)}"

    # Map attempt_index → flaky span_id.
    flaky_by_attempt: dict[int, int] = {}
    for s in flaky_spans:
        assert s.context is not None and s.attributes is not None
        idx = s.attributes["openarmature.node.attempt_index"]
        assert isinstance(idx, int)
        flaky_by_attempt[idx] = s.context.span_id
    assert set(flaky_by_attempt.keys()) == {0, 1, 2}

    # Every LLM span MUST parent under one of the ``flaky`` spans
    # (NOT under the invocation span, which would mean
    # attempt_index=0 was hardcoded and the lookup fell through).
    flaky_span_ids = set(flaky_by_attempt.values())
    parented_under: set[int] = set()
    for llm in llm_spans:
        assert llm.parent is not None, "LLM span MUST have a parent"
        parented_under.add(llm.parent.span_id)
    assert parented_under <= flaky_span_ids, (
        f"every LLM span MUST parent under an attempt's ``flaky`` span; "
        f"got LLM parents {parented_under} not all in flaky set {flaky_span_ids}"
    )
    # And the THREE LLM spans parent under THREE DISTINCT ``flaky``
    # spans — one per attempt — proving the calling_attempt_index
    # threading actually disambiguates per-attempt.
    assert len(parented_under) == 3, (
        f"each attempt's LLM call MUST parent under its OWN attempt's span; "
        f"got {len(parented_under)} distinct parents for 3 LLM calls"
    )
    # Spot-check: every attempt is represented.
    parented_attempts: set[int] = set()
    for pid in parented_under:
        attrs = by_id[pid].attributes
        assert attrs is not None
        idx = cast("int", attrs["openarmature.node.attempt_index"])
        parented_attempts.add(idx)
    assert parented_attempts == {0, 1, 2}
