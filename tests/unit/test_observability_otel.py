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
from pathlib import Path

import pytest
from pydantic import Field

# Skip the entire module if otel extras aren't installed.
pytest.importorskip("opentelemetry.sdk.trace")

from typing import Any, cast

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


# OTel SDK 1.x makes ``set_tracer_provider`` one-shot: once a non-default
# provider is set, subsequent ``set_tracer_provider`` calls are no-ops
# (the SDK logs a warning and returns). The set is guarded by a ``Once``
# primitive at ``opentelemetry.trace._TRACER_PROVIDER_SET_ONCE``, not
# just by the value of ``_TRACER_PROVIDER``. Restoring via the public
# API silently fails after a prior set, leaking the test's global
# provider into subsequent tests that also touch the OTel global (e.g.,
# the conformance fixture 005 sub-case verifying private/global
# isolation). Tests that need to manipulate the global provider use
# this helper to reset BOTH the value and the Once.
def _reset_otel_global_tracer_provider(restore_to: object) -> None:
    once = otel_trace._TRACER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
    with once._lock:  # pyright: ignore[reportPrivateUsage]
        if isinstance(restore_to, otel_trace.ProxyTracerProvider):
            # No real provider was set before this test; return the
            # global to "unset" state so the next set_tracer_provider
            # call works as if it were the first.
            otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
            once._done = False  # pyright: ignore[reportPrivateUsage]
        else:
            otel_trace._TRACER_PROVIDER = restore_to  # type: ignore[attr-defined]
            once._done = True  # pyright: ignore[reportPrivateUsage]


async def test_observer_uses_private_provider_not_global() -> None:
    """Spec §6 TracerProvider isolation: the OTelObserver MUST use a
    PRIVATE TracerProvider; spans MUST NOT appear on the OTel global
    provider's exporter (this is the load-bearing guarantee against
    duplicate spans from external auto-instrumentation libraries)."""
    # Save prior global state and install a separate exporter on the
    # OTel global provider. Pytest fixture-scoping doesn't cover the
    # OTel global, so we restore it manually in the finally block.
    prior_global = otel_trace.get_tracer_provider()
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
        _reset_otel_global_tracer_provider(prior_global)


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


async def test_checkpoint_migrate_emits_span_with_chain_metadata(tmp_path: Path) -> None:
    """Spec §6 cross-ref in proposal 0014: a versioned resume whose
    migration chain runs SHOULD emit an
    ``openarmature.checkpoint.migrate`` span carrying
    ``from_version`` / ``to_version`` (final) / ``chain_length``."""
    from openarmature.checkpoint import (
        CheckpointRecord,
        SQLiteCheckpointer,
    )

    # JSON-mode SQLite is migration-eligible (the dict-state form the
    # registry consumes is what the load path produces).
    cp = SQLiteCheckpointer(tmp_path / "ck.db", serialization="json")

    class _MigState(State):
        schema_version = "v2"
        x: int = 0
        new_field: str = "v2_default"

    async def _noop(_s: _MigState) -> dict[str, int]:
        return {}

    # Seed a v1 record so the resume triggers the v1→v2 migration.
    invocation_id = "mig-resume"
    await cp.save(
        invocation_id,
        CheckpointRecord(
            invocation_id=invocation_id,
            correlation_id="cid",
            state={"x": 9},
            completed_positions=(),
            parent_states=(),
            last_saved_at=0.0,
            schema_version="v1",
        ),
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_MigState)
        .add_node("noop", _noop)
        .add_edge("noop", END)
        .set_entry("noop")
        .with_checkpointer(cp)
        .with_state_migration("v1", "v2", lambda s: {**s, "new_field": "v2_default"})
        .compile()
    )
    g.attach_observer(observer, phases={"started", "completed", "checkpoint_migrated"})
    await g.invoke(
        _MigState.model_construct(),
        resume_invocation=invocation_id,
    )
    await g.drain()
    observer.shutdown()

    migrate_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.checkpoint.migrate"]
    assert len(migrate_spans) == 1
    span = migrate_spans[0]
    attrs = dict(span.attributes or {})
    assert attrs.get("openarmature.checkpoint.migrate.from_version") == "v1"
    assert attrs.get("openarmature.checkpoint.migrate.to_version") == "v2"
    assert attrs.get("openarmature.checkpoint.migrate.chain_length") == 1


async def test_checkpoint_migrate_span_absent_on_version_match(tmp_path: Path) -> None:
    """Spec §10.12.3 fast path: when the saved record's schema_version
    equals the current state class's schema_version, the migration
    registry is NOT consulted. The OTel observer MUST NOT emit a
    ``openarmature.checkpoint.migrate`` span in that case."""
    from openarmature.checkpoint import CheckpointRecord, SQLiteCheckpointer

    cp = SQLiteCheckpointer(tmp_path / "ck.db", serialization="json")

    class _MatchState(State):
        schema_version = "v1"
        x: int = 0

    async def _noop(_s: _MatchState) -> dict[str, int]:
        return {}

    invocation_id = "match-resume"
    await cp.save(
        invocation_id,
        CheckpointRecord(
            invocation_id=invocation_id,
            correlation_id="cid",
            state={"x": 7},
            completed_positions=(),
            parent_states=(),
            last_saved_at=0.0,
            schema_version="v1",  # matches current class
        ),
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_MatchState)
        .add_node("noop", _noop)
        .add_edge("noop", END)
        .set_entry("noop")
        .with_checkpointer(cp)
        .compile()
    )
    g.attach_observer(observer, phases={"started", "completed", "checkpoint_migrated"})
    await g.invoke(_MatchState.model_construct(), resume_invocation=invocation_id)
    await g.drain()
    observer.shutdown()

    migrate_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.checkpoint.migrate"]
    assert migrate_spans == []


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


async def test_active_prompt_propagates_to_llm_span_attributes() -> None:
    """Spec prompt-management §11: when an LLM call fires inside a
    ``with_active_prompt`` context, the OTel observer MUST surface
    ``openarmature.prompt.*`` attributes on the LLM-call span.
    ``with_active_prompt_group`` adds ``openarmature.prompt.group_name``."""
    from datetime import UTC, datetime

    from openarmature.graph.events import NodeEvent
    from openarmature.llm.messages import UserMessage
    from openarmature.llm.providers.openai import LlmEventPayload
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import (
        Prompt,
        PromptGroup,
        PromptResult,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))

    now = datetime.now(UTC)
    prompt = Prompt(
        name="greeting",
        version="v1",
        label="production",
        template="Hello, {{ user }}!",
        template_hash="sha256:tpl",
        fetched_at=now,
    )
    result = PromptResult(
        name=prompt.name,
        version=prompt.version,
        label=prompt.label,
        template_hash=prompt.template_hash,
        rendered_hash="sha256:rendered",
        messages=[UserMessage(content="Hello, Alice!")],
        variables={"user": "Alice"},
        fetched_at=now,
        rendered_at=now,
    )
    group = PromptGroup(group_name="classifier_chain", members=[result, result])

    token = _set_invocation_id("inv-1")
    try:
        # Proposal 0024 / friction-roundup #3: the provider captures
        # ``current_prompt_result()`` and ``current_prompt_group()``
        # at dispatch time and puts them on the LLM event payload.
        # The observer reads from the payload, NOT from the live
        # ContextVar — that ContextVar is unreachable from the
        # dispatch worker's task-local Context. This test verifies
        # the observer correctly surfaces prompt attributes when the
        # payload carries them; the cross-task regression case is
        # covered separately by an end-to-end test.
        started = NodeEvent(
            node_name="openarmature.llm.complete",
            namespace=("openarmature.llm.complete",),
            step=-1,
            phase="started",
            pre_state=LlmEventPayload(
                call_id="test-call-prompt",
                model="test-m",
                active_prompt=result,
                active_prompt_group=group,
            ),
            post_state=None,
            error=None,
            parent_states=(),
        )
        completed = NodeEvent(
            node_name="openarmature.llm.complete",
            namespace=("openarmature.llm.complete",),
            step=-1,
            phase="completed",
            pre_state=LlmEventPayload(
                call_id="test-call-prompt",
                model="test-m",
                finish_reason="stop",
                active_prompt=result,
                active_prompt_group=group,
            ),
            post_state=None,
            error=None,
            parent_states=(),
        )
        await observer(started)
        await observer(completed)
    finally:
        _reset_invocation_id(token)

    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    attrs = llm_spans[0].attributes or {}
    assert attrs.get("openarmature.prompt.name") == "greeting"
    assert attrs.get("openarmature.prompt.version") == "v1"
    assert attrs.get("openarmature.prompt.label") == "production"
    assert attrs.get("openarmature.prompt.template_hash") == "sha256:tpl"
    assert attrs.get("openarmature.prompt.rendered_hash") == "sha256:rendered"
    assert attrs.get("openarmature.prompt.group_name") == "classifier_chain"


async def test_llm_span_has_no_prompt_attributes_when_no_active_prompt() -> None:
    """Without ``with_active_prompt``, the LLM-call span MUST NOT carry
    ``openarmature.prompt.*`` attributes."""
    from openarmature.graph.events import NodeEvent
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.observability.llm_event import LlmEventPayload

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))

    token = _set_invocation_id("inv-2")
    try:
        started = NodeEvent(
            node_name="openarmature.llm.complete",
            namespace=("openarmature.llm.complete",),
            step=-1,
            phase="started",
            pre_state=LlmEventPayload(call_id="test-call-noprompt", model="test-m"),
            post_state=None,
            error=None,
            parent_states=(),
        )
        completed = NodeEvent(
            node_name="openarmature.llm.complete",
            namespace=("openarmature.llm.complete",),
            step=-1,
            phase="completed",
            pre_state=LlmEventPayload(call_id="test-call-noprompt", model="test-m", finish_reason="stop"),
            post_state=None,
            error=None,
            parent_states=(),
        )
        await observer(started)
        await observer(completed)
    finally:
        _reset_invocation_id(token)
    observer.shutdown()

    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    attrs = llm_spans[0].attributes or {}
    assert not any(k.startswith("openarmature.prompt.") for k in attrs)


async def test_disable_llm_spans_skips_llm_provider_span() -> None:
    """Spec §5.5: ``disable_llm_spans=True`` MUST suppress the
    LLM-provider span emission while leaving all other spans intact."""
    from openarmature.graph.events import NodeEvent

    # We don't drive a real provider here; instead we emit a synthetic
    # LLM event through the observer's __call__ and assert no span was
    # produced. This isolates the disable_llm_spans branch from the
    # provider's own queue-dispatch wiring.
    from openarmature.observability.llm_event import LlmEventPayload

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        disable_llm_spans=True,
    )
    # ``step=-1`` mirrors the synthetic value ``OpenAIProvider._llm_event``
    # mints (openai.py:643) — LLM-provider events aren't tied to graph step
    # sequencing.
    started = NodeEvent(
        node_name="openarmature.llm.complete",
        namespace=("openarmature.llm.complete",),
        step=-1,
        phase="started",
        pre_state=LlmEventPayload(call_id="test-call-1", model="test-m"),
        post_state=None,
        error=None,
        parent_states=(),
    )
    completed = NodeEvent(
        node_name="openarmature.llm.complete",
        namespace=("openarmature.llm.complete",),
        step=-1,
        phase="completed",
        pre_state=LlmEventPayload(call_id="test-call-1", model="test-m", finish_reason="stop"),
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


def test_log_record_factory_injects_correlation_id() -> None:
    """Spec §7: every log record emitted during an invocation MUST
    carry ``openarmature.correlation_id``. The bridge installs a
    process-global :class:`logging.LogRecord` factory (rather than
    a logger-level filter) so the attribute lands on every record
    regardless of which logger originated it — Python's logging
    propagates records up the logger tree's HANDLERS but skips
    ancestor FILTERS, so a filter on root would miss any
    child-logger emit.

    Tests both null-cid (outside invocation) and live-cid paths."""
    from openarmature.observability.correlation import (
        _reset_correlation_id,
        _set_correlation_id,
    )
    from openarmature.observability.otel.logs import (
        _install_correlation_id_factory,
    )

    prior_factory = logging.getLogRecordFactory()
    try:
        _install_correlation_id_factory()
        factory = logging.getLogRecordFactory()

        # Outside an invocation: no correlation_id attribute set.
        record = factory(
            "any.child.logger",
            logging.INFO,
            "",
            0,
            "hello",
            None,
            None,
        )
        assert not hasattr(record, "openarmature.correlation_id")

        # Inside an invocation: factory attaches the ContextVar
        # value to every newly constructed record.
        token = _set_correlation_id("my-cid-42")
        try:
            record2 = factory(
                "any.child.logger",
                logging.INFO,
                "",
                0,
                "hello",
                None,
                None,
            )
        finally:
            _reset_correlation_id(token)
        assert getattr(record2, "openarmature.correlation_id") == "my-cid-42"
    finally:
        # Restore the prior factory — process-global state.
        logging.setLogRecordFactory(prior_factory)


def test_install_log_bridge_is_idempotent() -> None:
    """Re-calling :func:`install_log_bridge` MUST NOT register a
    duplicate handler on the root logger AND MUST NOT stack a
    second LogRecord factory wrapper on top of the
    already-installed one.

    Wrapped in ``warnings.catch_warnings("error")`` to lock in the
    Phase 6.1 PR-B migration: this is the canonical surface where
    the deprecated ``opentelemetry.sdk._logs.LoggingHandler`` used
    to emit a ``DeprecationWarning``. Any future regression that
    re-introduces the deprecated path fires here immediately."""
    import warnings

    from opentelemetry.sdk._logs import LoggerProvider

    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_filters = list(root.filters)
    prior_factory = logging.getLogRecordFactory()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            provider = LoggerProvider()
            install_log_bridge(provider)
            handler_count_before = len(root.handlers)
            factory_after_first = logging.getLogRecordFactory()
            install_log_bridge(provider)
            handler_count_after = len(root.handlers)
            factory_after_second = logging.getLogRecordFactory()
            assert handler_count_before == handler_count_after
            # Factory identity is preserved across re-calls — no
            # second wrapper stacked on top of the first.
            assert factory_after_first is factory_after_second
    finally:
        # install_log_bridge mutates process-wide state; restore so
        # this test does not leak into others.
        root.handlers[:] = prior_handlers
        root.filters[:] = prior_filters
        logging.setLogRecordFactory(prior_factory)


def test_install_log_bridge_skips_when_sdk_handler_already_attached() -> None:
    """Downstream report (HyperDX integration): if an application's
    own logging setup attached
    :class:`opentelemetry.sdk._logs.LoggingHandler` against the same
    :class:`LoggerProvider` BEFORE ``install_log_bridge`` runs, the
    helper MUST NOT attach a second
    :class:`opentelemetry.instrumentation.logging.handler.LoggingHandler`
    against the same provider — both classes bridge to the same OTel
    Logs SDK and a second attach causes every record to ship to OTLP
    twice. The correlation_id factory still installs."""
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs import LoggingHandler as _SDKLoggingHandler

    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_factory = logging.getLogRecordFactory()
    try:
        provider = LoggerProvider()
        # Simulate the application's setup: attach the SDK handler
        # against `provider` BEFORE OA's bridge runs.
        sdk_handler = _SDKLoggingHandler(level=logging.NOTSET, logger_provider=provider)
        root.addHandler(sdk_handler)
        handler_count_before = len(root.handlers)

        install_log_bridge(provider)

        # No new handler attached — the SDK handler already bridges
        # to `provider`, so installing the instrumentation handler
        # would duplicate every emission.
        assert len(root.handlers) == handler_count_before, (
            f"install_log_bridge MUST NOT add a second OTel-Logs handler when an "
            f"SDK handler is already wired to the same provider; "
            f"got {len(root.handlers)} handlers (was {handler_count_before})"
        )
        # The correlation_id factory MUST install regardless — that's
        # what the helper is for once handler bridging is already
        # taken care of by the application.
        current_factory = logging.getLogRecordFactory()
        assert getattr(current_factory, "_openarmature_correlation_factory", False), (
            "correlation_id factory MUST install even when the OTel-Logs handler "
            "is skipped (application already attached one)"
        )
    finally:
        root.handlers[:] = prior_handlers
        logging.setLogRecordFactory(prior_factory)


def test_install_log_bridge_adds_handler_when_pre_attached_uses_different_provider() -> None:
    """An application MAY intentionally attach an SDK handler against
    a DIFFERENT :class:`LoggerProvider` (e.g., a console-only logs
    setup separate from the OA-managed OTLP provider). The
    idempotency check is scoped to the SAME provider, so OA's helper
    DOES attach its own handler against the OA provider in that
    case — no false-positive dedup that would silently break the OA
    bridge."""
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs import LoggingHandler as _SDKLoggingHandler

    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_factory = logging.getLogRecordFactory()
    try:
        # Application's pre-attached SDK handler points at a DIFFERENT
        # LoggerProvider — its own logs pipeline.
        unrelated_provider = LoggerProvider()
        unrelated_handler = _SDKLoggingHandler(level=logging.NOTSET, logger_provider=unrelated_provider)
        root.addHandler(unrelated_handler)
        handler_count_before = len(root.handlers)

        # OA's bridge installs against its OWN provider.
        oa_provider = LoggerProvider()
        install_log_bridge(oa_provider)

        # One new handler MUST appear — the OA-installed
        # instrumentation handler against `oa_provider`. The
        # pre-existing unrelated handler is unaffected.
        assert len(root.handlers) == handler_count_before + 1, (
            f"install_log_bridge MUST attach when no handler bridges to the "
            f"target provider; got {len(root.handlers)} (was {handler_count_before})"
        )
    finally:
        root.handlers[:] = prior_handlers
        logging.setLogRecordFactory(prior_factory)


def test_log_bridge_exports_records_with_correlation_id() -> None:
    """Spec §7 end-to-end: a log record emitted on a CHILD logger
    under ``current_correlation_id`` flows through the bridge to
    the OTel ``LoggerProvider``'s exporter with
    ``openarmature.correlation_id`` populated. Child-logger emit
    is the load-bearing case — Python's logging propagates child
    records up to root's handlers but skips root's filters, so a
    filter-on-root placement (the prior implementation) misses
    every reasonable user's logger.

    Wrapped in ``warnings.catch_warnings("error")`` so the PR-B
    migration's "no more deprecation warning" guarantee is
    asserted on the affirmative export path too."""
    import warnings

    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import (
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )

    from openarmature.observability.correlation import (
        _reset_correlation_id,
        _set_correlation_id,
    )

    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_filters = list(root.filters)
    prior_factory = logging.getLogRecordFactory()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exporter = InMemoryLogRecordExporter()
            provider = LoggerProvider()
            provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
            install_log_bridge(provider)

            # Emit on a CHILD logger to verify the factory
            # placement (which fires uniformly at record
            # construction) actually delivers — a filter-on-root
            # placement would not.
            child_logger = logging.getLogger("openarmature.test_log_bridge.child")
            token = _set_correlation_id("test-cid-export-1")
            try:
                child_logger.warning("hello from %s", "test")
            finally:
                _reset_correlation_id(token)

            # SimpleLogRecordProcessor flushes synchronously, but
            # force-flush as a belt-and-suspenders guard so any
            # buffered emit lands in the exporter before assertions.
            provider.force_flush()
        records = exporter.get_finished_logs()
        # Filter to the record(s) emitted on our test logger — the
        # root may receive other records from concurrent test setup.
        ours = [r for r in records if r.log_record.body == "hello from test"]
        assert len(ours) == 1, (
            f"expected exactly one exported record for our test logger; "
            f"got {len(ours)} (full set: {[r.log_record.body for r in records]})"
        )
        attrs = dict(ours[0].log_record.attributes or {})
        assert attrs.get("openarmature.correlation_id") == "test-cid-export-1", (
            f"correlation_id MUST appear on the exported OTel LogRecord attributes; "
            f"got {attrs.get('openarmature.correlation_id')!r}"
        )
    finally:
        root.handlers[:] = prior_handlers
        root.filters[:] = prior_filters
        logging.setLogRecordFactory(prior_factory)


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
        items: list[int] = Field(default_factory=list[int])
        results: list[int] = Field(default_factory=list[int])

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


async def test_log_on_first_line_of_node_body_carries_node_span() -> None:
    """The load-bearing case ``prepare_sync`` exists to fix.

    Without ``prepare_sync``, the engine queues the started event for
    async dispatch, then enters the node body — by the time the OTel
    observer's ``__call__`` opens the span on the worker task, the
    node body has already executed (or is mid-await). A log emitted
    on the FIRST line of the body, before any ``await``, would not
    see the observer's span via OTel ``get_current()``.

    With ``prepare_sync``, the observer creates the span synchronously
    in the engine task BEFORE queueing, publishes it via
    ``current_active_observer_span``, and the engine attaches it to
    the OTel context around the node body. The first-line log picks
    up the right ``trace_id``/``span_id``.

    This test exists in unit/ (not just buried in the conformance
    fixture 010 driver) so a failure here jumps straight to
    ``prepare_sync``-related changes during a regression hunt.
    """
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import (
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )

    test_logger = logging.getLogger("openarmature.test.first_line_log")

    class _S(State):
        x: int = 0

    async def first_line_log_node(_s: _S) -> dict[str, Any]:
        # FIRST line, before any ``await`` — without ``prepare_sync``
        # in the engine task, OTel ``get_current()`` would return an
        # invalid span here and the log would have ``trace_id=0`` /
        # ``span_id=0``.
        test_logger.info("emitted before any await")
        return {"x": 1}

    span_exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(span_exporter))
    log_exporter = InMemoryLogRecordExporter()
    log_provider = LoggerProvider()
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    # Snapshot prior log state so this test doesn't bleed into others
    # — install_log_bridge mutates process-global ``logging`` state.
    root = logging.getLogger()
    prior_handlers = list(root.handlers)
    prior_filters = list(root.filters)
    prior_factory = logging.getLogRecordFactory()
    prior_test_level = test_logger.level
    test_logger.setLevel(logging.INFO)

    try:
        install_log_bridge(log_provider)
        g = (
            GraphBuilder(_S)
            .add_node("node_a", first_line_log_node)
            .add_edge("node_a", END)
            .set_entry("node_a")
            .compile()
        )
        g.attach_observer(observer)
        await g.invoke(_S(), correlation_id="first-line-test")
        await g.drain()
        observer.shutdown()
        log_provider.force_flush()

        records = log_exporter.get_finished_logs()
        ours = [r for r in records if str(r.log_record.body) == "emitted before any await"]
        assert len(ours) == 1, (
            f"expected exactly one log record; got {len(ours)}: {[str(r.log_record.body) for r in records]}"
        )
        log_record = ours[0].log_record

        spans = span_exporter.get_finished_spans()
        node_a_spans = [s for s in spans if s.name == "node_a"]
        assert len(node_a_spans) == 1, f"expected one node_a span; got {len(node_a_spans)}"
        node_a_span = node_a_spans[0]
        assert node_a_span.context is not None
        node_span_id = node_a_span.context.span_id
        node_trace_id = node_a_span.context.trace_id

        # Load-bearing: the prepare_sync hook attached the observer
        # span synchronously so the first-line log saw it via OTel
        # ``get_current()``.
        assert log_record.span_id == node_span_id, (
            f"first-line log MUST carry node_a span's span_id "
            f"(prepare_sync attaches the span synchronously in the engine task); "
            f"got log span_id={log_record.span_id}, node span_id={node_span_id}"
        )
        assert log_record.trace_id == node_trace_id, (
            f"first-line log MUST carry node_a span's trace_id; "
            f"got log trace_id={log_record.trace_id}, node trace_id={node_trace_id}"
        )
    finally:
        root.handlers[:] = prior_handlers
        root.filters[:] = prior_filters
        logging.setLogRecordFactory(prior_factory)
        test_logger.setLevel(prior_test_level)


# ---------------------------------------------------------------------------
# Friction-roundup #3 regression: prompt context propagates across the
# dispatch-worker task boundary
# ---------------------------------------------------------------------------


async def test_prompt_context_propagates_cross_task_via_provider_complete() -> None:
    """End-to-end #3 regression: open ``with_active_prompt`` inside a
    node body, call ``provider.complete()``, and assert the LLM span
    carries ``openarmature.prompt.name``.

    Pre-fix this test failed because:

    - ``invoke()`` calls ``asyncio.create_task(deliver_loop(queue))``
      BEFORE any node body runs. The worker's Context is snapshotted
      at task-creation time, so it never sees ContextVars set later
      inside a node body.
    - The observer used to read ``current_prompt_result()`` from the
      worker task — it returned ``None`` because the worker's snapshot
      doesn't have ``_active_prompt`` set.

    Post-fix the provider captures ``current_prompt_result()`` at
    dispatch time (in the node task's Context, where
    ``with_active_prompt`` IS active) and puts the snapshot on the
    ``LlmEventPayload``. The observer reads from the payload, not from
    a ContextVar.
    """
    import json
    from datetime import UTC, datetime

    import httpx
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from openarmature.graph import END, GraphBuilder, State
    from openarmature.llm import OpenAIProvider, UserMessage
    from openarmature.prompts import (
        Prompt,
        PromptResult,
        with_active_prompt,
    )

    def _handler(_request: httpx.Request) -> httpx.Response:
        body = {
            "id": "cc-test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi back"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(
            200,
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    provider = OpenAIProvider(
        base_url="http://mock.test",
        model="test-model",
        api_key="k",
        transport=httpx.MockTransport(_handler),
    )

    now = datetime.now(UTC)
    prompt = Prompt(
        name="greeting",
        version="v1",
        label="production",
        template="Hello, {{ user }}!",
        template_hash="sha256:tpl",
        fetched_at=now,
    )
    rendered = PromptResult(
        name=prompt.name,
        version=prompt.version,
        label=prompt.label,
        template_hash=prompt.template_hash,
        rendered_hash="sha256:rendered",
        messages=[UserMessage(content="Hello, Alice!")],
        variables={"user": "Alice"},
        fetched_at=now,
        rendered_at=now,
    )

    class _S(State):
        reply: str = ""

    async def ask_llm(_s: _S) -> dict[str, str]:
        # The ContextVar set here lives in the node task. Pre-fix, the
        # dispatch worker (a separate task) could not see this set.
        with with_active_prompt(rendered):
            response = await provider.complete(rendered.messages)
        return {"reply": response.message.content}

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    graph = (
        GraphBuilder(_S).add_node("ask_llm", ask_llm).add_edge("ask_llm", END).set_entry("ask_llm")
    ).compile()
    graph.attach_observer(observer)
    try:
        await graph.invoke(_S())
        await graph.drain()
    finally:
        observer.shutdown()
        await provider.aclose()

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1, f"expected one LLM span; got {len(llm_spans)}"
    attrs = dict(llm_spans[0].attributes or {})
    # Pre-fix these were all None; post-fix all populated from the
    # dispatch-time PromptResult snapshot.
    assert attrs.get("openarmature.prompt.name") == "greeting"
    assert attrs.get("openarmature.prompt.version") == "v1"
    assert attrs.get("openarmature.prompt.label") == "production"
    assert attrs.get("openarmature.prompt.template_hash") == "sha256:tpl"
    assert attrs.get("openarmature.prompt.rendered_hash") == "sha256:rendered"


def test_force_flush_delegates_to_provider() -> None:
    # Public force_flush wraps TracerProvider.force_flush so downstream
    # users don't reach into observer._provider to drain the
    # BatchSpanProcessor buffer in fast-teardown harnesses.
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    try:
        assert observer.force_flush() is True
        assert observer.force_flush(timeout_ms=1000) is True
    finally:
        observer.shutdown()
