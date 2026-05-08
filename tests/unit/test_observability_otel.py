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

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
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
