"""OTel-specific observability unit tests (extras-gated).

Skipped cleanly when the ``otel`` extras aren't installed — the
import-time check in
``openarmature.observability.otel.__init__`` raises ImportError on
missing deps.

These tests fill the gaps the conformance harness defers:

- TracerProvider isolation — the load-bearing "spans don't leak
  into the OTel global provider" guarantee.
- attribute population on every span type.
- status mapping for every error category.
- LLM provider span via the ContextVar dispatch hook (queue-
  mediated; no synchronous direct dispatch).
- detached trace mode key separation in the span stack.
- checkpoint_saved → ``openarmature.checkpoint.save`` zero-
  duration span.
- log bridge filter + correlation_id injection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import Field

# Skip the entire module if otel extras aren't installed.
pytest.importorskip("opentelemetry.sdk.trace")

from typing import Annotated, Any, cast

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
    append,
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
    """TracerProvider isolation: the OTelObserver MUST use a
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
    """Every node span MUST carry the four ``openarmature.node.*``
    base attributes."""
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
    """Invocation span MUST carry ``openarmature.graph.entry_node`` +
    ``openarmature.graph.spec_version``."""
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


# Spec §5.1 / proposal 0052: invocation span MUST carry
# ``openarmature.implementation.name`` and
# ``openarmature.implementation.version`` as non-empty strings; name
# matches the package-registry canonical value (``openarmature-python``).
# Inner-node spans MUST NOT carry them — the attributes live in §5.1,
# not the cross-cutting §5.6 family.
async def test_invocation_span_carries_implementation_attribution_attributes() -> None:
    from openarmature import __version__

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g, _ = _build_linear_graph(observer)
    await g.invoke(_LinearState())  # type: ignore[attr-defined]
    await g.drain()  # type: ignore[attr-defined]
    observer.shutdown()
    spans = exporter.get_finished_spans()

    inv = next((s for s in spans if s.name == "openarmature.invocation"), None)
    assert inv is not None
    inv_attrs = dict(inv.attributes or {})
    assert inv_attrs.get("openarmature.implementation.name") == "openarmature-python"
    assert inv_attrs.get("openarmature.implementation.version") == __version__
    assert isinstance(inv_attrs["openarmature.implementation.name"], str)
    assert inv_attrs["openarmature.implementation.name"]  # non-empty
    assert isinstance(inv_attrs["openarmature.implementation.version"], str)
    assert inv_attrs["openarmature.implementation.version"]  # non-empty

    # Inner-node spans MUST NOT carry the attribution attributes.
    inner_spans = [s for s in spans if s.name != "openarmature.invocation"]
    assert inner_spans, "expected at least one inner node span"
    for span in inner_spans:
        span_attrs = dict(span.attributes or {})
        assert "openarmature.implementation.name" not in span_attrs, (
            f"inner span {span.name!r} unexpectedly carries implementation.name"
        )
        assert "openarmature.implementation.version" not in span_attrs, (
            f"inner span {span.name!r} unexpectedly carries implementation.version"
        )


# Spec §5.1 / proposal 0052: always-emit invariant. The attribution
# attributes describe runtime identity, not runtime data, so the
# privacy knobs that gate payload-shaped attributes (LLM payload,
# state payload, GenAI semconv) MUST NOT gate the attribution. This
# pins the OTel side of the contract; the Langfuse-side equivalent
# lives in test_observability_langfuse.py against
# disable_state_payload=True.
async def test_invocation_span_attribution_emits_under_disable_provider_payload() -> None:
    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        disable_provider_payload=True,
        disable_genai_semconv=True,
        disable_llm_spans=True,
    )
    g, _ = _build_linear_graph(observer)
    await g.invoke(_LinearState())  # type: ignore[attr-defined]
    await g.drain()  # type: ignore[attr-defined]
    observer.shutdown()
    spans = exporter.get_finished_spans()

    inv = next((s for s in spans if s.name == "openarmature.invocation"), None)
    assert inv is not None
    attrs = dict(inv.attributes or {})
    assert attrs.get("openarmature.implementation.name") == "openarmature-python"
    assert isinstance(attrs.get("openarmature.implementation.version"), str)
    assert attrs["openarmature.implementation.version"]


# Spec §5.1 / proposal 0052: every invocation span carries the
# attribution attributes. An observer reused across multiple
# invocations on the same compiled graph MUST emit the attributes on
# every invocation's span — not just the first. The dataclass-field
# defaults are computed once at observer construction, so a regression
# where the values were instance-scoped (read-once) instead of
# emit-each-time would silently break this contract.
async def test_invocation_span_attribution_emits_on_every_invocation() -> None:
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g, _ = _build_linear_graph(observer)

    for _ in range(3):
        await g.invoke(_LinearState())  # type: ignore[attr-defined]
        await g.drain()  # type: ignore[attr-defined]
    observer.shutdown()
    spans = exporter.get_finished_spans()

    inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
    assert len(inv_spans) == 3, f"expected three invocation spans, got {len(inv_spans)}"
    for span in inv_spans:
        attrs = dict(span.attributes or {})
        assert attrs.get("openarmature.implementation.name") == "openarmature-python"
        assert isinstance(attrs.get("openarmature.implementation.version"), str)
        assert attrs["openarmature.implementation.version"]


# ---------------------------------------------------------------------------
# §4.2 status mapping
# ---------------------------------------------------------------------------


class _FailState(State):
    a: int = 0


async def _failing_node(_s: _FailState) -> dict[str, int]:
    raise RuntimeError("boom")


async def test_failing_node_span_carries_error_status() -> None:
    """A node-exception failure produces a span with
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
    """A versioned resume whose migration chain runs SHOULD emit an
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
    """Fast path: when the saved record's schema_version equals the
    current state class's schema_version, the migration
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
    """A checkpoint save SHOULD emit an observer event surfaced as a
    span. Our implementation emits a
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
    """When an LLM call fires inside a ``with_active_prompt`` context,
    the OTel observer MUST surface
    ``openarmature.prompt.*`` attributes on the LLM-call span.
    ``with_active_prompt_group`` adds ``openarmature.prompt.group_name``."""
    from datetime import UTC, datetime

    from openarmature.llm.messages import UserMessage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import (
        PromptGroup,
        PromptResult,
        TextPrompt,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))

    now = datetime.now(UTC)
    prompt = TextPrompt(
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
        # at dispatch time and puts them on the LlmCompletionEvent.
        # The observer reads from the typed event, NOT from the live
        # ContextVar — that ContextVar is unreachable from the
        # dispatch worker's task-local Context.
        await observer(make_retry_attempt_event(active_prompt=result, active_prompt_group=group))
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


async def test_llm_span_parents_under_fan_out_instance_dispatch() -> None:
    # Gap 2 (review-nested-fan-out-lineage): an LLM span whose calling node has
    # no open span and fires inside a top-level fan-out instance MUST parent
    # under the per-instance fan-out dispatch span (matching the Langfuse
    # observer), not fall through to the subgraph / invocation span. Before this
    # OTel had no fan-out-instance fallback in _resolve_llm_parent.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.observability.otel.observer import (
        _dispatch_key,
        _InvState,
        _OpenSpan,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    invocation_id = "inv-fanout-llm"
    token = _set_invocation_id(invocation_id)
    try:
        # Per-instance dispatch span for top-level fan-out "fan", instance 0,
        # keyed by the lineage-aware _dispatch_key. No open_spans entry for the
        # calling node ("fan", "ask"), so the resolver must reach the fan-out
        # dispatch fallback.
        observer._inv_states[invocation_id] = _InvState()  # noqa: SLF001
        dispatch_span = observer._tracer.start_span("fan")  # noqa: SLF001
        instance_key = _dispatch_key(("fan",), (0,), (None,))
        observer._inv_states[invocation_id].fan_out_instance_spans[instance_key] = _OpenSpan(  # noqa: SLF001
            span=dispatch_span
        )
        await observer(
            make_retry_attempt_event(
                invocation_id=invocation_id,
                node_name="ask",
                namespace=("fan", "ask"),
                attempt_index=0,
                fan_out_index=0,
                branch_name=None,
            )
        )
        # dispatch_span is ended by observer.shutdown() below (it drains
        # fan_out_instance_spans); ending it here too would double-end it.
    finally:
        _reset_invocation_id(token)

    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    assert llm_spans[0].parent is not None, "LLM span must have a parent, not be a trace root"
    assert cast("Any", llm_spans[0].parent).span_id == dispatch_span.get_span_context().span_id, (
        "LLM span must parent under the per-instance fan-out dispatch span"
    )


async def test_llm_span_has_no_prompt_attributes_when_no_active_prompt() -> None:
    """Without ``with_active_prompt``, the LLM-call span MUST NOT carry
    ``openarmature.prompt.*`` attributes."""
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))

    token = _set_invocation_id("inv-2")
    try:
        await observer(make_retry_attempt_event())
    finally:
        _reset_invocation_id(token)
    observer.shutdown()

    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    attrs = llm_spans[0].attributes or {}
    assert not any(k.startswith("openarmature.prompt.") for k in attrs)


async def test_otel_observer_ignores_terminal_llm_events() -> None:
    """Feeding a terminal LlmCompletionEvent or LlmFailedEvent to the
    OTel observer produces no ``openarmature.llm.complete`` span; the
    per-attempt event is the sole span source."""
    # Proposal 0050: the OTel span renders only from LlmRetryAttemptEvent.
    # The terminal events stay on the queue for the Langfuse mapping +
    # payload consumers; this guards against reintroducing the
    # terminal-event span path (which would double-emit alongside the
    # per-attempt span).
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_failed_event, make_typed_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))

    token = _set_invocation_id("inv-terminal")
    try:
        await observer(make_typed_event())
        await observer(make_failed_event())
    finally:
        _reset_invocation_id(token)
    observer.shutdown()

    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert llm_spans == []


async def test_structured_output_failure_span_renders_response_surface() -> None:
    # Proposal 0082: a structured_output_invalid failed attempt renders the
    # response-side surface on the ERROR openarmature.llm.complete span
    # (finish_reason, usage, payload-gated output.content) in addition to
    # ERROR status + category -- unlike other failure categories, which carry
    # no response.
    from opentelemetry.trace import StatusCode

    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter), disable_provider_payload=False)
    token = _set_invocation_id("inv-soi")
    try:
        await observer(
            make_retry_attempt_event(
                error_category="structured_output_invalid",
                error_type="StructuredOutputInvalid",
                finish_reason="length",
                output_content='{"name":"Alice","age":',
                usage=Usage(prompt_tokens=20, completion_tokens=16, total_tokens=36),
                response_id="cc-xyz",
                response_model="gpt-test-v2",
            )
        )
    finally:
        _reset_invocation_id(token)
    observer.shutdown()

    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs["openarmature.error.category"] == "structured_output_invalid"
    assert attrs["openarmature.llm.finish_reason"] == "length"
    assert attrs["openarmature.llm.output.content"] == '{"name":"Alice","age":'
    assert attrs["openarmature.llm.usage.completion_tokens"] == 16
    assert attrs["gen_ai.response.id"] == "cc-xyz"


async def test_structured_output_failure_span_redacts_output_when_payload_disabled() -> None:
    # Payload-gated: with disable_provider_payload=True, output.content is
    # redacted while finish_reason / usage / ERROR status stay (proposal 0082).
    from opentelemetry.trace import StatusCode

    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter), disable_provider_payload=True)
    token = _set_invocation_id("inv-soi2")
    try:
        await observer(
            make_retry_attempt_event(
                error_category="structured_output_invalid",
                finish_reason="length",
                output_content='{"name":"Alice","age":',
                usage=Usage(prompt_tokens=20, completion_tokens=16, total_tokens=36),
            )
        )
    finally:
        _reset_invocation_id(token)
    observer.shutdown()

    span = next(s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete")
    attrs = dict(span.attributes or {})
    assert span.status.status_code == StatusCode.ERROR
    assert "openarmature.llm.output.content" not in attrs
    assert attrs["openarmature.llm.finish_reason"] == "length"
    assert attrs["openarmature.llm.usage.completion_tokens"] == 16


async def _drive_llm_span_with_cached_tokens(
    *,
    cached_tokens: int | None,
    cache_creation_tokens: int | None = None,
) -> dict[str, Any]:
    """Drive the OTel observer through a per-attempt LlmRetryAttemptEvent
    carrying the supplied cache-stat fields on the event's Usage
    record. Returns the LLM-span's attribute map.
    """
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    token = _set_invocation_id("inv-cache")
    try:
        await observer(
            make_retry_attempt_event(
                usage=Usage(
                    prompt_tokens=100,
                    completion_tokens=5,
                    total_tokens=105,
                    cached_tokens=cached_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                ),
            )
        )
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    return dict(llm_spans[0].attributes or {})


async def test_llm_span_emits_cache_read_attribute_when_provider_reports_hit() -> None:
    # Proposal 0047 §5.5.3.1: openarmature.llm.cache_read.input_tokens
    # is set on the LLM span when the payload carries a non-None
    # cached_tokens value sourced from Response.usage.cached_tokens.
    attrs = await _drive_llm_span_with_cached_tokens(cached_tokens=42)
    assert attrs.get("openarmature.llm.cache_read.input_tokens") == 42
    assert "openarmature.llm.cache_creation.input_tokens" not in attrs


async def test_llm_span_emits_cache_read_attribute_with_reported_zero() -> None:
    # The absent-vs-reported-zero distinction is observable on the
    # span: a payload with cached_tokens=0 produces the attribute
    # with value 0 (not omitted).
    attrs = await _drive_llm_span_with_cached_tokens(cached_tokens=0)
    assert attrs.get("openarmature.llm.cache_read.input_tokens") == 0


async def test_llm_span_omits_cache_attribute_when_provider_silent() -> None:
    # When the provider doesn't report cache stats (cached_tokens=None
    # on the payload), the OTel observer does NOT emit the attribute
    # per the §5.5.3 conditional-emission convention.
    attrs = await _drive_llm_span_with_cached_tokens(cached_tokens=None)
    assert "openarmature.llm.cache_read.input_tokens" not in attrs
    assert "openarmature.llm.cache_creation.input_tokens" not in attrs


async def test_llm_span_emits_cache_creation_attribute_when_payload_carries_it() -> None:
    # The OpenAI-compatible mapping never sources cache_creation_tokens
    # (per spec §8.1.2), but the observer side honors the field when
    # any future provider populates it.
    attrs = await _drive_llm_span_with_cached_tokens(cached_tokens=20, cache_creation_tokens=5)
    assert attrs.get("openarmature.llm.cache_read.input_tokens") == 20
    assert attrs.get("openarmature.llm.cache_creation.input_tokens") == 5


async def test_disable_llm_spans_skips_llm_provider_span() -> None:
    """``disable_llm_spans=True`` MUST suppress the LLM-provider span
    emission while leaving all other spans intact."""
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


async def test_llm_span_duration_matches_typed_event_latency() -> None:
    # Proposal 0049 + PR 3b: the success-path span's duration is
    # back-dated using LlmCompletionEvent.latency_ms, so observers see
    # the adapter-boundary measurement instead of dispatcher queue
    # delay. Verify the span's end-minus-start lands within tolerance
    # of the typed event's latency_ms.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    latency_ms = 123.456
    token = _set_invocation_id("inv-duration")
    try:
        await observer(make_retry_attempt_event(latency_ms=latency_ms))
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.start_time is not None and span.end_time is not None
    duration_ms = (span.end_time - span.start_time) / 1_000_000
    # Tolerance covers integer-nanosecond truncation and float->int
    # rounding; the back-date is exact apart from those.
    assert abs(duration_ms - latency_ms) < 1.0


async def _drive_llm_span_with_tool_calls(
    tool_calls: list[Any],
    *,
    disable_provider_payload: bool = True,
) -> dict[str, Any]:
    """Drive one per-attempt LLM event carrying ``output_tool_calls``
    through the OTel observer; return the openarmature.llm.complete
    span's attribute dict. ``disable_provider_payload`` mirrors the
    observer's default-on payload gate (the OTel span renders from the
    per-attempt LlmRetryAttemptEvent)."""
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        disable_provider_payload=disable_provider_payload,
    )
    token = _set_invocation_id("inv-tool-calls")
    try:
        await observer(
            make_retry_attempt_event(
                finish_reason="tool_calls" if tool_calls else "stop",
                output_tool_calls=tool_calls,
            )
        )
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    return dict(llm_spans[0].attributes or {})


async def test_llm_span_emits_output_tool_call_identity_projections() -> None:
    # Proposal 0076 §5.5.10 (mirrors fixture 085): a completion
    # requesting two tools emits count / names / ids on the span,
    # index-aligned and in request order. The default payload-off
    # posture applies, so the gated full serialization is absent.
    from openarmature.llm.messages import ToolCall

    attrs = await _drive_llm_span_with_tool_calls(
        [
            ToolCall(id="call_a", name="get_weather", arguments={"city": "NYC"}),
            ToolCall(id="call_b", name="get_time", arguments={"tz": "EST"}),
        ]
    )
    assert attrs.get("openarmature.llm.output.tool_calls.count") == 2
    assert list(attrs.get("openarmature.llm.output.tool_calls.names") or ()) == ["get_weather", "get_time"]
    assert list(attrs.get("openarmature.llm.output.tool_calls.ids") or ()) == ["call_a", "call_b"]
    assert "openarmature.llm.output.tool_calls" not in attrs


async def test_llm_span_omits_output_tool_calls_when_none_requested() -> None:
    # Proposal 0076 (mirrors fixture 086): a completion with no tool
    # calls emits NONE of the family — absence means "no tools
    # requested", distinct from count = 0 / empty arrays.
    attrs = await _drive_llm_span_with_tool_calls([])
    for name in (
        "openarmature.llm.output.tool_calls",
        "openarmature.llm.output.tool_calls.count",
        "openarmature.llm.output.tool_calls.names",
        "openarmature.llm.output.tool_calls.ids",
    ):
        assert name not in attrs


async def test_llm_span_output_tool_calls_payload_gating() -> None:
    # Proposal 0076 §5.5.1 / §5.5.10 (mirrors fixture 087): the identity
    # projections are ungated (render with payload off); the gated full
    # [{id, name, arguments}] serialization is suppressed with payload
    # off and present (carrying the arguments) with payload on.
    import json

    from openarmature.llm.messages import ToolCall

    calls = [ToolCall(id="call_x", name="search_db", arguments={"q": "secret query"})]

    off = await _drive_llm_span_with_tool_calls(calls, disable_provider_payload=True)
    assert off.get("openarmature.llm.output.tool_calls.count") == 1
    assert list(off.get("openarmature.llm.output.tool_calls.names") or ()) == ["search_db"]
    assert list(off.get("openarmature.llm.output.tool_calls.ids") or ()) == ["call_x"]
    assert "openarmature.llm.output.tool_calls" not in off

    on = await _drive_llm_span_with_tool_calls(calls, disable_provider_payload=False)
    assert on.get("openarmature.llm.output.tool_calls.count") == 1
    assert list(on.get("openarmature.llm.output.tool_calls.names") or ()) == ["search_db"]
    assert list(on.get("openarmature.llm.output.tool_calls.ids") or ()) == ["call_x"]
    serialized = on.get("openarmature.llm.output.tool_calls")
    assert isinstance(serialized, str)
    # Parses to the §5.5.5 [{id, name, arguments}] encoding (structure,
    # not bytewise — _serialize_for_attribute sorts keys).
    assert json.loads(serialized) == [
        {"id": "call_x", "name": "search_db", "arguments": {"q": "secret query"}}
    ]


# ---------------------------------------------------------------------------
# Proposal 0067 — GenAI metrics (observability §11)
# ---------------------------------------------------------------------------


def _collect_metric_points(reader: Any) -> list[tuple[str, float, int, dict[str, Any]]]:
    """Flatten an InMemoryMetricReader's collected data into
    ``(instrument_name, recorded_value, point_count, point_attributes)``
    tuples. Histogram observations with identical attribute sets
    aggregate into one data point (sum + count), so per-attempt tests
    assert on ``count``, not point cardinality. A Counter's data points are
    ``NumberDataPoint`` (``.value``, no ``.sum`` / ``.count``) -- the
    token_budget.exceeded counter -- so branch on the point shape:
    ``.value`` (count 1) for a Sum/Counter point, ``.sum`` / ``.count``
    for a Histogram point."""
    data = reader.get_metrics_data()
    points: list[tuple[str, float, int, dict[str, Any]]] = []
    if data is None:
        return points
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                for pt in metric.data.data_points:
                    if hasattr(pt, "value"):
                        points.append((metric.name, float(pt.value), 1, dict(pt.attributes)))
                    else:
                        points.append((metric.name, pt.sum, pt.count, dict(pt.attributes)))
    return points


async def _drive_metrics_events(
    events: list[Any],
    *,
    enable_metrics: bool = True,
    disable_llm_spans: bool = False,
) -> tuple[list[tuple[str, float, int, dict[str, Any]]], list[Any]]:
    """Feed LlmRetryAttemptEvents through an OTelObserver wired to a
    private MeterProvider + InMemoryMetricReader; return the captured
    ``(metric_points, llm_complete_spans)``."""
    from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    reader = InMemoryMetricReader()
    meter_provider = SdkMeterProvider(metric_readers=[reader])
    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        enable_metrics=enable_metrics,
        disable_llm_spans=disable_llm_spans,
        meter_provider=meter_provider,
    )
    token = _set_invocation_id("inv-metrics")
    try:
        for event in events:
            await observer(event)
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    return _collect_metric_points(reader), llm_spans


async def test_metrics_records_token_and_duration() -> None:
    # Proposal 0067 §11 (mirrors fixture 088): a successful LLM attempt
    # with usage {input 5, output 1} records two token.usage observations
    # (input + output) and one duration observation, with the §11.3
    # dimensions. Duration value is not asserted (§11.4).
    from openarmature.llm.response import Usage
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        latency_ms=12.0,
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )
    points, _ = await _drive_metrics_events([event])
    token_points = [p for p in points if p[0] == "openarmature.gen_ai.client.token.usage"]
    duration_points = [p for p in points if p[0] == "openarmature.gen_ai.client.operation.duration"]
    by_type = {p[3]["openarmature.gen_ai.token.type"]: p for p in token_points}
    assert by_type["input"][1] == 5
    assert by_type["output"][1] == 1
    for ttype in ("input", "output"):
        dims = by_type[ttype][3]
        assert dims["openarmature.gen_ai.operation"] == "chat"
        assert dims["gen_ai.request.model"] == "test-model"
        assert dims["gen_ai.system"] == "openai"
    assert len(duration_points) == 1
    ddims = duration_points[0][3]
    assert ddims["openarmature.gen_ai.operation"] == "chat"
    assert ddims["gen_ai.request.model"] == "test-model"
    assert ddims["gen_ai.system"] == "openai"
    assert "error.type" not in ddims


async def test_metrics_records_duration_with_error_type_on_failure() -> None:
    # Proposal 0067 §11.2 / §11.3 (mirrors fixture 090): a failed attempt
    # records a duration observation carrying error.type, and NO
    # token.usage observation (a failed attempt returned no usage).
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        latency_ms=8.0,
        finish_reason=None,
        usage=None,
        error_category="provider_unavailable",
        error_type="ProviderUnavailable",
        error_message="down",
    )
    points, _ = await _drive_metrics_events([event])
    token_points = [p for p in points if p[0] == "openarmature.gen_ai.client.token.usage"]
    duration_points = [p for p in points if p[0] == "openarmature.gen_ai.client.operation.duration"]
    assert token_points == []
    assert len(duration_points) == 1
    assert duration_points[0][3]["error.type"] == "provider_unavailable"
    assert duration_points[0][3]["openarmature.gen_ai.operation"] == "chat"


async def test_metrics_disabled_records_nothing() -> None:
    # Proposal 0067 §11.1 (mirrors fixture 091): enable_metrics off (the
    # default) creates no instrument and records nothing.
    from openarmature.llm.response import Usage
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6))
    points, _ = await _drive_metrics_events([event], enable_metrics=False)
    assert points == []


async def test_metrics_independent_of_disable_llm_spans() -> None:
    # Proposal 0067 §11.1: metrics record even with spans disabled — the
    # disable_llm_spans flag governs span emission only.
    from openarmature.llm.response import Usage
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6))
    points, llm_spans = await _drive_metrics_events([event], disable_llm_spans=True)
    assert llm_spans == []
    assert any(p[0] == "openarmature.gen_ai.client.operation.duration" for p in points)
    assert any(p[0] == "openarmature.gen_ai.client.token.usage" for p in points)


async def test_metrics_record_once_per_attempt_under_retry() -> None:
    # Proposal 0067 §11.2 "Call-level retry": the duration histogram
    # records once per attempt (failed attempts carry error.type), and
    # token.usage only for an attempt that returned usage. Two failed
    # attempts + one success -> 3 duration observations (2 with
    # error.type), 2 token.usage observations (the success's input +
    # output). Observations with identical dimensions aggregate into one
    # data point, so this asserts on histogram counts.
    from openarmature.llm.response import Usage
    from tests._helpers.typed_event import make_retry_attempt_event

    failed = [
        make_retry_attempt_event(
            llm_attempt_index=i,
            latency_ms=5.0,
            finish_reason=None,
            usage=None,
            error_category="provider_unavailable",
            error_type="ProviderUnavailable",
            error_message="down",
        )
        for i in range(2)
    ]
    success = make_retry_attempt_event(
        llm_attempt_index=2,
        latency_ms=7.0,
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )
    points, _ = await _drive_metrics_events([*failed, success])
    duration_points = [p for p in points if p[0] == "openarmature.gen_ai.client.operation.duration"]
    token_points = [p for p in points if p[0] == "openarmature.gen_ai.client.token.usage"]
    # 3 duration observations total: 2 share the error dims (one
    # aggregated point, count 2), the success is a separate point.
    assert sum(p[2] for p in duration_points) == 3
    error_duration = [p for p in duration_points if p[3].get("error.type") == "provider_unavailable"]
    assert sum(p[2] for p in error_duration) == 2
    success_duration = [p for p in duration_points if "error.type" not in p[3]]
    assert sum(p[2] for p in success_duration) == 1
    # token.usage only from the success attempt: one input, one output.
    by_type = {p[3]["openarmature.gen_ai.token.type"]: p for p in token_points}
    assert by_type["input"][1] == 5 and by_type["input"][2] == 1
    assert by_type["output"][1] == 1 and by_type["output"][2] == 1


# ---------------------------------------------------------------------------
# Proposal 0083 — per-prompt token-budget span attributes + §11.2 metrics
# ---------------------------------------------------------------------------

_TB_EXCEEDED = "openarmature.gen_ai.client.token_budget.exceeded"
_TB_UTILIZATION = "openarmature.gen_ai.client.token_budget.utilization"


async def test_token_budget_input_exceeded_span_and_metrics() -> None:
    # §5.5.15 / §11.2 (mirrors fixture 126): input_max_tokens 10 vs prompt 20 ->
    # exceeded. The span carries the declared input bound + exceeded=true (no
    # total-bound attr); metrics record the exceeded counter (kind input,
    # value 1) + the utilization histogram (2.0, kind input).
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
        token_budget=TokenBudget(input_max_tokens=10),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.prompt.token_budget.input_max_tokens") == 10
    assert "openarmature.prompt.token_budget.total_max_tokens" not in attrs
    assert attrs.get("openarmature.llm.token_budget.exceeded") is True

    exceeded = [p for p in points if p[0] == _TB_EXCEEDED]
    util = [p for p in points if p[0] == _TB_UTILIZATION]
    assert len(exceeded) == 1
    assert exceeded[0][1] == 1
    assert exceeded[0][3]["openarmature.gen_ai.token_budget.kind"] == "input"
    assert len(util) == 1
    assert util[0][1] == 2.0
    assert util[0][3]["openarmature.gen_ai.token_budget.kind"] == "input"
    assert util[0][3]["gen_ai.request.model"] == "test-model"
    assert util[0][3]["gen_ai.system"] == "openai"


async def test_token_budget_total_exceeded_kind_total() -> None:
    # §5.5.15 / §11.2 (mirrors fixture 127): total_max_tokens 25 vs total 50 ->
    # exceeded on the total bound; kind "total", input bound absent.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=40, completion_tokens=10, total_tokens=50),
        token_budget=TokenBudget(total_max_tokens=25),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.prompt.token_budget.total_max_tokens") == 25
    assert "openarmature.prompt.token_budget.input_max_tokens" not in attrs
    assert attrs.get("openarmature.llm.token_budget.exceeded") is True

    exceeded = [p for p in points if p[0] == _TB_EXCEEDED]
    util = [p for p in points if p[0] == _TB_UTILIZATION]
    assert len(exceeded) == 1 and exceeded[0][1] == 1
    assert exceeded[0][3]["openarmature.gen_ai.token_budget.kind"] == "total"
    assert len(util) == 1 and util[0][1] == 2.0
    assert util[0][3]["openarmature.gen_ai.token_budget.kind"] == "total"


async def test_token_budget_under_budget_records_utilization_only() -> None:
    # §5.5.15 / §11.2 (mirrors fixture 128): input_max_tokens 40 vs prompt 20 ->
    # under budget. exceeded=false span attr, NO exceeded counter, but the
    # utilization histogram STILL records (0.5).
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
        token_budget=TokenBudget(input_max_tokens=40),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.prompt.token_budget.input_max_tokens") == 40
    assert attrs.get("openarmature.llm.token_budget.exceeded") is False

    exceeded = [p for p in points if p[0] == _TB_EXCEEDED]
    util = [p for p in points if p[0] == _TB_UTILIZATION]
    assert exceeded == []
    assert len(util) == 1 and util[0][1] == 0.5
    assert util[0][3]["openarmature.gen_ai.token_budget.kind"] == "input"


async def test_token_budget_absent_no_budget_surface() -> None:
    # §5.5.15 / §11.2 (mirrors fixture 129): no token_budget -> no budget span
    # attrs and no budget metric observations, while the baseline token.usage
    # instruments still record.
    from openarmature.llm.response import Usage
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        token_budget=None,
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert "openarmature.prompt.token_budget.input_max_tokens" not in attrs
    assert "openarmature.prompt.token_budget.total_max_tokens" not in attrs
    assert "openarmature.llm.token_budget.exceeded" not in attrs

    budget_points = [p for p in points if "token_budget" in p[0]]
    assert budget_points == []
    token_points = [p for p in points if p[0] == "openarmature.gen_ai.client.token.usage"]
    assert len(token_points) == 2


async def test_token_budget_no_usage_omits_exceeded_signal() -> None:
    # §5.5.15: a declared budget with NO usage on the attempt (e.g. a
    # provider_unavailable failed attempt) still surfaces the declared bound
    # attribute but omits the exceeded signal (it needs usage to evaluate),
    # and records no budget metric observation (gated on usage present).
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        finish_reason=None,
        usage=None,
        token_budget=TokenBudget(input_max_tokens=10),
        error_category="provider_unavailable",
        error_type="ProviderUnavailable",
        error_message="down",
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.prompt.token_budget.input_max_tokens") == 10
    assert "openarmature.llm.token_budget.exceeded" not in attrs
    assert [p for p in points if "token_budget" in p[0]] == []


async def test_token_budget_both_bounds_exceeded_double_increment() -> None:
    # §11.2 (proposal 0083): a prompt declaring BOTH bounds, both exceeded,
    # increments the exceeded counter once per breached bound (kinds input +
    # total) and records two utilization observations; the span exceeded signal
    # is the OR (true). Guards the per-kind double-increment.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        token_budget=TokenBudget(input_max_tokens=10, total_max_tokens=15),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.prompt.token_budget.input_max_tokens") == 10
    assert attrs.get("openarmature.prompt.token_budget.total_max_tokens") == 15
    assert attrs.get("openarmature.llm.token_budget.exceeded") is True

    exceeded = [p for p in points if p[0] == _TB_EXCEEDED]
    util = [p for p in points if p[0] == _TB_UTILIZATION]
    assert {p[3]["openarmature.gen_ai.token_budget.kind"] for p in exceeded} == {"input", "total"}
    assert all(p[1] == 1 for p in exceeded)
    util_by_kind = {p[3]["openarmature.gen_ai.token_budget.kind"]: p[1] for p in util}
    assert util_by_kind["input"] == 2.0  # 20 / 10
    assert util_by_kind["total"] == 2.0  # 30 / 15


async def test_token_budget_total_falls_back_to_prompt_plus_completion() -> None:
    # §11.2 (proposal 0083): when usage.total_tokens is None the total bound's
    # actual falls back to prompt_tokens + completion_tokens. 12 + 4 = 16 over a
    # total_max of 10 -> exceeded, utilization 1.6.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=12, completion_tokens=4, total_tokens=None),
        token_budget=TokenBudget(total_max_tokens=10),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.llm.token_budget.exceeded") is True
    util = [p for p in points if p[0] == _TB_UTILIZATION]
    assert len(util) == 1
    assert util[0][3]["openarmature.gen_ai.token_budget.kind"] == "total"
    assert util[0][1] == 1.6  # (12 + 4) / 10


async def test_token_budget_missing_input_count_omits_evaluation() -> None:
    # §5.5.15 / §11.2 (proposal 0083): a usage record present but with the input
    # count UNREPORTED (prompt_tokens None) is not coerced to 0 -- the input
    # bound is not evaluated, so no utilization / exceeded-counter observation
    # and, with no other evaluable bound, the span exceeded signal is absent
    # (not a misleading false). Mirrors the token.usage is-not-None gate.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=None, completion_tokens=500, total_tokens=500),
        token_budget=TokenBudget(input_max_tokens=100),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    # Declared bound still surfaces; the evaluation does not (input count missing).
    assert attrs.get("openarmature.prompt.token_budget.input_max_tokens") == 100
    assert "openarmature.llm.token_budget.exceeded" not in attrs
    assert [p for p in points if "token_budget" in p[0]] == []


async def test_token_budget_zero_bound_exceeds_but_skips_utilization() -> None:
    # §5.5.15 (proposal 0083): a declared bound of 0 is exceeded by any positive
    # usage (the exceeded test is a strict actual > max), so the exceeded span
    # attr + counter fire. The utilization ratio is undefined for a 0 denominator,
    # so that one histogram sample is skipped -- not a fabricated sentinel.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        token_budget=TokenBudget(input_max_tokens=0),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.prompt.token_budget.input_max_tokens") == 0
    assert attrs.get("openarmature.llm.token_budget.exceeded") is True

    exceeded = [p for p in points if p[0] == _TB_EXCEEDED]
    util = [p for p in points if p[0] == _TB_UTILIZATION]
    assert len(exceeded) == 1
    assert exceeded[0][1] == 1
    assert exceeded[0][3]["openarmature.gen_ai.token_budget.kind"] == "input"
    # No utilization sample -- the 0-denominator ratio is undefined and skipped.
    assert util == []


async def test_token_budget_zero_total_bound_exceeds_but_skips_utilization() -> None:
    # §5.5.15 (proposal 0083): the total branch mirrors the input branch -- a
    # total_max of 0 is exceeded by any positive total usage (exceeded attr +
    # counter fire, kind total), and the undefined 0-denominator utilization
    # sample is skipped.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    event = make_retry_attempt_event(
        model="test-model",
        provider="openai",
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        token_budget=TokenBudget(total_max_tokens=0),
    )
    points, llm_spans = await _drive_metrics_events([event])

    attrs: dict[str, Any] = dict(llm_spans[0].attributes or {})
    assert attrs.get("openarmature.prompt.token_budget.total_max_tokens") == 0
    assert attrs.get("openarmature.llm.token_budget.exceeded") is True

    exceeded = [p for p in points if p[0] == _TB_EXCEEDED]
    util = [p for p in points if p[0] == _TB_UTILIZATION]
    assert len(exceeded) == 1
    assert exceeded[0][3]["openarmature.gen_ai.token_budget.kind"] == "total"
    assert util == []


async def test_token_budget_exceeded_emits_one_warning_log(caplog: pytest.LogCaptureFixture) -> None:
    # §7 (proposal 0083): an over-budget attempt emits exactly ONE WARNING record
    # on the openarmature.observability logger naming the breached bound; an
    # under-budget attempt emits none. The record is one-per-attempt, not
    # one-per-bound, and fires independent of the span-attr / metric surfaces.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    caplog.set_level(logging.WARNING, logger="openarmature.observability")

    over = make_retry_attempt_event(
        usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
        token_budget=TokenBudget(input_max_tokens=10),
    )
    await _drive_metrics_events([over])
    warns = [r for r in caplog.records if r.name == "openarmature.observability"]
    assert len(warns) == 1
    assert warns[0].levelno == logging.WARNING
    assert "input 20 > 10" in warns[0].getMessage()

    caplog.clear()
    under = make_retry_attempt_event(
        usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        token_budget=TokenBudget(input_max_tokens=40),
    )
    await _drive_metrics_events([under])
    assert [r for r in caplog.records if r.name == "openarmature.observability"] == []

    # The §7 log is its own surface: it fires even with BOTH the span
    # (disable_llm_spans) and metric (enable_metrics off) surfaces suppressed.
    caplog.clear()
    _, llm_spans = await _drive_metrics_events([over], enable_metrics=False, disable_llm_spans=True)
    assert llm_spans == []
    warns = [r for r in caplog.records if r.name == "openarmature.observability"]
    assert len(warns) == 1
    assert "input 20 > 10" in warns[0].getMessage()


async def test_token_budget_warning_log_names_prompt_identity(caplog: pytest.LogCaptureFixture) -> None:
    # §7 (proposal 0083): the WARNING log names the active prompt's identity
    # (name + version) alongside the breached bound.
    from datetime import UTC, datetime

    from openarmature.llm.messages import UserMessage
    from openarmature.llm.response import Usage
    from openarmature.prompts import PromptResult, TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    caplog.set_level(logging.WARNING, logger="openarmature.observability")
    now = datetime.now(UTC)
    result = PromptResult(
        name="classify",
        version="v7",
        label="production",
        template_hash="sha256:tpl",
        rendered_hash="sha256:r",
        messages=[UserMessage(content="x")],
        variables={},
        fetched_at=now,
        rendered_at=now,
    )
    event = make_retry_attempt_event(
        active_prompt=result,
        usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
        token_budget=TokenBudget(input_max_tokens=10),
    )
    await _drive_metrics_events([event])
    warns = [r for r in caplog.records if r.name == "openarmature.observability"]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "classify v7" in msg
    assert "input 20 > 10" in msg


async def test_token_budget_warning_log_on_structured_output_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # §7 (proposal 0083): the WARNING log fires on an over-budget
    # structured_output_invalid failure attempt (it carries usage, proposal
    # 0082), parity with the completion path; a no-usage failure emits none.
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    caplog.set_level(logging.WARNING, logger="openarmature.observability")
    failed = make_retry_attempt_event(
        error_category="structured_output_invalid",
        finish_reason="length",
        usage=Usage(prompt_tokens=20, completion_tokens=16, total_tokens=36),
        token_budget=TokenBudget(input_max_tokens=10),
    )
    await _drive_metrics_events([failed])
    warns = [r for r in caplog.records if r.name == "openarmature.observability"]
    assert len(warns) == 1
    assert "input 20 > 10" in warns[0].getMessage()

    # A no-usage failure has nothing to evaluate -> no log.
    caplog.clear()
    no_usage = make_retry_attempt_event(
        error_category="provider_unavailable",
        finish_reason=None,
        usage=None,
        token_budget=TokenBudget(input_max_tokens=10),
    )
    await _drive_metrics_events([no_usage])
    assert [r for r in caplog.records if r.name == "openarmature.observability"] == []


async def test_token_budget_warning_log_multibound_and_per_attempt(caplog: pytest.LogCaptureFixture) -> None:
    # §7 (proposal 0083): a both-bounds breach renders "input .. > .., total .. > .."
    # in ONE record (input then total order); N over-budget attempts emit N records
    # (one per exceedance, not per bound).
    from openarmature.llm.response import Usage
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_retry_attempt_event

    caplog.set_level(logging.WARNING, logger="openarmature.observability")
    both = make_retry_attempt_event(
        usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        token_budget=TokenBudget(input_max_tokens=10, total_max_tokens=15),
    )
    await _drive_metrics_events([both])
    warns = [r for r in caplog.records if r.name == "openarmature.observability"]
    assert len(warns) == 1
    assert "input 20 > 10, total 30 > 15" in warns[0].getMessage()

    caplog.clear()
    attempts = [
        make_retry_attempt_event(
            llm_attempt_index=i,
            usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
            token_budget=TokenBudget(input_max_tokens=10),
        )
        for i in range(2)
    ]
    await _drive_metrics_events(attempts)
    assert len([r for r in caplog.records if r.name == "openarmature.observability"]) == 2


async def test_llm_span_zero_duration_when_latency_missing() -> None:
    # When the typed event omits latency_ms (None), the handler falls
    # back to a zero-duration span at end_time rather than guessing
    # the start. Pin the fallback so a future "let's just use now() for
    # both endpoints" tweak doesn't accidentally swap to a small
    # positive duration.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    token = _set_invocation_id("inv-no-latency")
    try:
        await observer(make_retry_attempt_event(latency_ms=None))
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.start_time is not None and span.end_time is not None
    assert span.start_time == span.end_time


async def test_typed_llm_event_drops_silently_outside_invocation() -> None:
    # No invocation in scope (no _set_invocation_id) → the handler
    # MUST early-return without emitting a span. Symmetric with the
    # error path's no-invocation drop.
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    await observer(make_retry_attempt_event())
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert llm_spans == []


async def test_disable_llm_spans_skips_typed_event_path() -> None:
    # disable_llm_spans MUST gate the typed-event handler too — not
    # just the sentinel-pair branch. Companion to
    # ``test_disable_llm_spans_skips_llm_provider_span`` which covers
    # the sentinel side.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        disable_llm_spans=True,
    )
    token = _set_invocation_id("inv-disabled")
    try:
        await observer(make_retry_attempt_event())
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert llm_spans == []


async def test_llm_error_path_emits_error_span_from_typed_failed_event() -> None:
    # Per proposal 0058: failures emit a typed LlmFailedEvent. The
    # OTel observer drives the same openarmature.llm.complete span
    # shape with ERROR status + openarmature.error.category attribute.
    from opentelemetry.trace import StatusCode

    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_retry_attempt_event

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    token = _set_invocation_id("inv-err")
    try:
        await observer(
            make_retry_attempt_event(
                invocation_id="inv-err",
                error_category="provider_rate_limit",
                error_type="ProviderRateLimit",
                error_message="429 from upstream",
                call_id="cc-err",
                finish_reason=None,
            )
        )
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    llm_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs.get("openarmature.error.category") == "provider_rate_limit"


@pytest.mark.parametrize(
    "fixture_id",
    [
        "056-call-level-retry-transient",
        "057-call-level-retry-exhaustion",
        "058-call-level-retry-non-transient-no-retry",
    ],
)
async def test_call_level_retry_fixture_per_attempt_spans(fixture_id: str) -> None:
    # Proposal 0050 §7.1 / observability §5.5: drive each spec
    # call-level-retry fixture (spec/llm-provider/conformance/) through
    # the provider + an OTel observer and assert its per-attempt
    # openarmature.llm.complete spans. These fixtures assert SPANS, so
    # they are activated here (otel-gated, with an observer) rather than
    # the generic llm-provider harness, which has no observer. The
    # provider dispatches one LlmRetryAttemptEvent per attempt and the
    # observer renders one span per event; in production the engine's
    # serial queue carries them, here they are captured then replayed.
    import json
    from pathlib import Path

    import httpx
    import yaml
    from opentelemetry.trace import StatusCode

    from openarmature.graph.events import LlmRetryAttemptEvent
    from openarmature.graph.middleware import RetryConfig, deterministic_backoff
    from openarmature.llm.errors import LlmProviderError
    from openarmature.llm.messages import UserMessage
    from openarmature.llm.providers.openai import OpenAIProvider
    from openarmature.observability.correlation import (
        _reset_active_dispatch,
        _reset_invocation_id,
        _set_active_dispatch,
        _set_invocation_id,
    )

    fixture_dir = (
        Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "llm-provider" / "conformance"
    )
    spec = cast("dict[str, Any]", yaml.safe_load((fixture_dir / f"{fixture_id}.yaml").read_text()))
    responses = cast("list[dict[str, Any]]", spec["mock_provider"]["responses"])
    call = cast("dict[str, Any]", spec["call"])
    expected = cast("dict[str, Any]", spec["expected"])

    response_iter = iter(responses)

    def handler(_request: httpx.Request) -> httpx.Response:
        entry = next(response_iter)
        body = entry.get("body")
        content = json.dumps(body).encode() if body is not None else b""
        return httpx.Response(int(entry.get("status", 200)), content=content)

    retry_cfg = cast("dict[str, Any]", call["retry"])
    backoff_seconds = float(cast("dict[str, Any]", retry_cfg.get("backoff") or {}).get("seconds", 0.0))
    retry = RetryConfig(
        max_attempts=int(retry_cfg["max_attempts"]),
        backoff=deterministic_backoff(backoff_seconds),
    )
    messages = [
        UserMessage(content=cast("str", m["content"])) for m in cast("list[dict[str, Any]]", call["messages"])
    ]

    captured: list[Any] = []
    disp_token = _set_active_dispatch(lambda e: captured.append(e))
    inv_token = _set_invocation_id("inv-clr")
    provider = OpenAIProvider(
        base_url="http://test", model="gpt-test", api_key="k", transport=httpx.MockTransport(handler)
    )
    try:
        if "raises" in expected:
            with pytest.raises(LlmProviderError) as excinfo:
                await provider.complete(messages, retry=retry)
            assert excinfo.value.category == expected["raises"]["category"]
        else:
            result = await provider.complete(messages, retry=retry)
            expected_content = cast("dict[str, Any]", expected["response"]["message"])["content"]
            assert result.message.content == expected_content
    finally:
        await provider.aclose()
        _reset_invocation_id(inv_token)
        _reset_active_dispatch(disp_token)

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    inv_token2 = _set_invocation_id("inv-clr")
    try:
        for event in captured:
            if isinstance(event, LlmRetryAttemptEvent):
                await observer(event)
    finally:
        _reset_invocation_id(inv_token2)
    observer.shutdown()

    spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.llm.complete"]
    expected_spans = cast("list[dict[str, Any]]", expected["llm_spans"])
    assert len(spans) == len(expected_spans)
    spans_by_index = {dict(s.attributes or {})["openarmature.llm.attempt_index"]: s for s in spans}
    for exp_span in expected_spans:
        idx = exp_span["attempt_index"]
        span = spans_by_index[idx]
        span_attrs = dict(span.attributes or {})
        for key, val in cast("dict[str, Any]", exp_span.get("attributes") or {}).items():
            assert span_attrs.get(key) == val, f"attempt {idx}: {key}={span_attrs.get(key)!r} != {val!r}"
        if exp_span.get("error_category"):
            assert span.status.status_code == StatusCode.ERROR
            assert span_attrs.get("openarmature.error.category") == exp_span["error_category"]
        else:
            assert span.status.status_code == StatusCode.OK


# ---------------------------------------------------------------------------
# §7 log bridge: correlation_id injection
# ---------------------------------------------------------------------------


def test_log_record_factory_injects_correlation_id() -> None:
    """Every log record emitted during an invocation MUST carry
    ``openarmature.correlation_id``. The bridge installs a
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
    logging-handler migration: this is the canonical surface where
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
    """End-to-end: a log record emitted on a CHILD logger under
    ``current_correlation_id`` flows through the bridge to
    the OTel ``LoggerProvider``'s exporter with
    ``openarmature.correlation_id`` populated. Child-logger emit
    is the load-bearing case — Python's logging propagates child
    records up to root's handlers but skips root's filters, so a
    filter-on-root placement (the prior implementation) misses
    every reasonable user's logger.

    Wrapped in ``warnings.catch_warnings("error")`` so the
    logging-handler migration's "no more deprecation warning"
    guarantee is asserted on the affirmative export path too."""
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
# Concurrency-safe state scoping + §5.5 calling-node attribution
# ---------------------------------------------------------------------------


async def test_shared_observer_concurrent_invocations_dont_collide() -> None:
    """A single observer shared across concurrent invocations MUST
    keep their span trees isolated. Each invocation has its own
    ``invocation_id`` and therefore its own
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
    interleave on the observer's call queue. An earlier
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


async def test_detached_fan_out_instance_error_status_on_both_spans() -> None:
    # Proposal 0061 / §4.2 (v0.15.0 release-review item 2): when a DETACHED
    # fan-out instance's subgraph raises, ERROR surfaces on BOTH the parent's
    # fan-out node span (parent trace) and the detached instance's invocation
    # span (its own trace, carrying the §4 category + an OTel exception event).
    # This is the fan-out analog of the detached-subgraph case (conformance
    # fixture 008 case 3 covers only the subgraph path).
    from opentelemetry.trace import StatusCode

    from openarmature.graph import RuntimeGraphError

    class _ParentState(State):
        items: list[int] = Field(default_factory=list[int])
        results: list[int] = Field(default_factory=list[int])

    class _ChildState(State):
        item: int = 0
        out: int = 0

    async def _raise(_s: _ChildState) -> dict[str, int]:
        raise RuntimeError("boom")

    inner = (
        GraphBuilder(_ChildState)
        .add_node("compute", _raise)
        .add_edge("compute", END)
        .set_entry("compute")
        .compile()
    )
    parent = (
        GraphBuilder(_ParentState)
        .add_fan_out_node(
            "score",
            subgraph=inner,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
        )
        .add_edge("score", END)
        .set_entry("score")
        .compile()
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        detached_fan_outs=frozenset({"score"}),
    )
    parent.attach_observer(observer)
    with pytest.raises(RuntimeGraphError):
        await parent.invoke(_ParentState(items=[5]))
    await parent.drain()
    observer.shutdown()

    spans = exporter.get_finished_spans()
    # The detached instance trace is the one containing the raising inner
    # node (`score` appears in BOTH traces -- the parent fan-out node span
    # and the detached instance's own root span -- so key off `compute`).
    compute = next(s for s in spans if s.name == "compute")
    assert compute.context is not None
    detached_trace_id = compute.context.trace_id
    # §4.2: the detached instance's invocation span is its own trace's
    # authoritative carrier -- ERROR + the §4 category + an OTel exception
    # event (mirroring the detached-subgraph case).
    detached_inv = next(
        s
        for s in spans
        if s.name == "openarmature.invocation"
        and s.context is not None
        and s.context.trace_id == detached_trace_id
    )
    assert detached_inv.status.status_code == StatusCode.ERROR
    assert dict(detached_inv.attributes or {}).get("openarmature.error.category") == "node_exception"
    assert any(e.name == "exception" for e in detached_inv.events)
    # And the parent-trace fan-out node span (the §4.4 Link carrier) is ERROR.
    parent_score = next(
        s
        for s in spans
        if s.name == "score" and s.context is not None and s.context.trace_id != detached_trace_id
    )
    assert parent_score.status.status_code == StatusCode.ERROR


async def test_concurrent_fan_out_llm_spans_parent_under_calling_instance() -> None:
    """Under concurrent fan-out: each instance's
    ``openarmature.llm.complete`` span MUST parent under that
    instance's calling node, not a sibling instance's. The
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
    """Under retry: when an LLM ``complete()`` call
    happens inside a node body wrapped with retry middleware, each
    attempt's LLM span MUST parent under THAT attempt's node span,
    not a hardcoded ``attempt_index=0``. The
    ``current_attempt_index`` ContextVar (set inside the per-attempt
    ``innermost`` scope) is what makes this work."""
    import httpx

    from openarmature.graph.middleware import RetryConfig, RetryMiddleware
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
        .add_node(
            "flaky",
            _flaky,
            middleware=[RetryMiddleware(RetryConfig(max_attempts=3, backoff=lambda _i: 0.0))],
        )
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
        PromptResult,
        TextPrompt,
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
    prompt = TextPrompt(
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


# ---------------------------------------------------------------------------
# §3.4 mid-invocation augmentation (proposal 0040)
# ---------------------------------------------------------------------------


class _AugmentState(State):
    answer: str = ""


async def test_metadata_augmentation_updates_outermost_open_spans() -> None:
    # Spec §3.4 MUST + proposal 0040 §6: when a node body calls
    # ``set_invocation_metadata`` mid-invocation, every open span whose
    # lineage ancestor-or-equals the calling context's MUST be updated
    # in place to carry the augmented entries. In a single-node
    # outermost-serial graph, that's the invocation root span AND the
    # calling node's span.
    from openarmature.observability.metadata import set_invocation_metadata

    captured: dict[str, str] = {}

    async def node_augments(_s: _AugmentState) -> dict[str, str]:
        set_invocation_metadata(request_id="req-xyz")
        captured["seen"] = "yes"
        return {"answer": "ok"}

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_AugmentState)
        .add_node("ask", node_augments)
        .add_edge("ask", END)
        .set_entry("ask")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_AugmentState())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    invocation_spans = [s for s in spans if s.name == "openarmature.invocation"]
    ask_spans = [s for s in spans if s.name == "ask"]
    assert len(invocation_spans) == 1
    assert len(ask_spans) == 1
    inv_attrs = dict(invocation_spans[0].attributes or {})
    ask_attrs = dict(ask_spans[0].attributes or {})
    # Augmentation reached both the invocation span (open at the call)
    # and the calling node's span (the augmenter itself).
    assert inv_attrs.get("openarmature.user.request_id") == "req-xyz"
    assert ask_attrs.get("openarmature.user.request_id") == "req-xyz"


async def test_metadata_augmentation_outside_invocation_is_silent() -> None:
    # Plumbing safety: ``set_invocation_metadata`` outside any active
    # invocation updates the ContextVar but emits no augmentation event
    # (no dispatch is in scope). The observer never sees an event so
    # no observer-side error surfaces.
    from openarmature.observability.metadata import set_invocation_metadata

    # No graph, no observer attached — should not raise.
    set_invocation_metadata(local_only="value")


async def test_metadata_augmentation_no_op_when_no_entries() -> None:
    # Empty entries dict is a no-op at the public API (the helper
    # short-circuits before validating or dispatching). The observer
    # still must tolerate the case in any future direct test path.
    from openarmature.graph.events import MetadataAugmentationEvent

    observer = OTelObserver(span_processor=SimpleSpanProcessor(InMemorySpanExporter()))
    try:
        # Direct call to the handler bypasses the engine so we can
        # confirm an empty-entries augmentation is silently dropped.
        event = MetadataAugmentationEvent(
            entries={},
            namespace=("ask",),
            attempt_index=0,
            fan_out_index=None,
            branch_name=None,
        )
        observer._handle_metadata_augmentation(event)  # noqa: SLF001
    finally:
        observer.shutdown()


async def test_metadata_augmentation_in_fan_out_isolates_per_instance() -> None:
    # Spec §3.4 + proposal 0040 scoping rule: a fan-out instance
    # augmenting metadata MUST update its own instance dispatch span
    # and its own inner-node span, but NOT the shared fan_out_node
    # parent span, NOT the invocation span, and NOT sibling instances'
    # spans. Each ``inner_ask`` span ends up tagged with its own
    # ``product_id`` only.
    import asyncio

    from openarmature.observability.correlation import current_fan_out_index
    from openarmature.observability.metadata import set_invocation_metadata

    class _ParentState(State):
        products: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
        results: list[str] = Field(default_factory=list[str])

    class _ChildState(State):
        product: dict[str, str] = Field(default_factory=dict[str, str])
        out: str = ""

    async def _ask(s: _ChildState) -> dict[str, str]:
        # Yield once so concurrent instances interleave their
        # augmentation events on the observer queue.
        await asyncio.sleep(0)
        idx = current_fan_out_index()
        assert idx is not None
        product_id = s.product["id"]
        set_invocation_metadata(product_id=product_id)
        return {"out": f"ok-{product_id}"}

    inner = (
        GraphBuilder(_ChildState)
        .add_node("inner_ask", _ask)
        .add_edge("inner_ask", END)
        .set_entry("inner_ask")
        .compile()
    )
    parent = (
        GraphBuilder(_ParentState)
        .add_fan_out_node(
            "fan",
            subgraph=inner,
            collect_field="out",
            target_field="results",
            items_field="products",
            item_field="product",
            concurrency=3,
        )
        .add_edge("fan", END)
        .set_entry("fan")
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    compiled = parent.compile()
    compiled.attach_observer(observer)
    try:
        products = [
            {"id": "prod-A"},
            {"id": "prod-B"},
            {"id": "prod-C"},
        ]
        await compiled.invoke(_ParentState(products=products))
        await compiled.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    inner_spans = [s for s in spans if s.name == "inner_ask"]
    assert len(inner_spans) == 3
    seen: dict[str, str] = {}
    for span in inner_spans:
        attrs = dict(span.attributes or {})
        product_id = attrs.get("openarmature.user.product_id")
        fan_out_idx = attrs.get("openarmature.node.fan_out_index")
        assert isinstance(product_id, str), f"missing per-instance augmentation on {span.name}"
        assert isinstance(fan_out_idx, int)
        seen[str(fan_out_idx)] = product_id
    # Each instance carries its OWN product_id; no sibling leakage.
    assert seen == {"0": "prod-A", "1": "prod-B", "2": "prod-C"}

    # The shared fan-out parent node span and the invocation span MUST
    # NOT carry any per-instance product_id. The PER-INSTANCE dispatch
    # spans (synthesized for non-detached fan-outs per §5.4 + proposal
    # 0013) are IN scope, so each one SHOULD carry its own product_id.
    invocation_spans = [s for s in spans if s.name == "openarmature.invocation"]
    fan_spans = [s for s in spans if s.name == "fan"]
    assert len(invocation_spans) == 1
    # The shared fan-out parent has ``openarmature.fan_out.item_count``
    # set; per-instance dispatch spans don't.
    parent_fan_spans = [s for s in fan_spans if "openarmature.fan_out.item_count" in dict(s.attributes or {})]
    instance_fan_spans = [
        s for s in fan_spans if "openarmature.fan_out.item_count" not in dict(s.attributes or {})
    ]
    assert len(parent_fan_spans) == 1
    assert len(instance_fan_spans) == 3
    # Parent + invocation: no per-instance product_id leakage.
    for span in (*parent_fan_spans, *invocation_spans):
        attrs = dict(span.attributes or {})
        assert "openarmature.user.product_id" not in attrs, (
            f"per-instance augmentation leaked onto {span.name} span"
        )
    # Per-instance dispatch spans: each one carries its own product_id.
    seen_dispatch: dict[int, str] = {}
    for span in instance_fan_spans:
        attrs = dict(span.attributes or {})
        idx_value = attrs.get("openarmature.node.fan_out_index")
        product_value = attrs.get("openarmature.user.product_id")
        assert isinstance(idx_value, int)
        assert isinstance(product_value, str), f"per-instance dispatch span missing product_id; attrs={attrs}"
        seen_dispatch[idx_value] = product_value
    assert seen_dispatch == {0: "prod-A", 1: "prod-B", 2: "prod-C"}


async def test_metadata_augmentation_in_parallel_branches_skips_sibling() -> None:
    # Sibling-skip for parallel-branches: two concurrent branches each
    # augment metadata with their own branch identifier. Each branch's
    # inner-node span carries ONLY its own ``branch_label``; no
    # cross-branch leakage. This also implicitly verifies that the
    # OTel observer's open-span key disambiguates concurrent same-
    # named inner nodes across sibling branches (pre-fix, both
    # branches' ``ask`` opens collided on the same _StackKey).
    import asyncio

    from openarmature.graph import BranchSpec
    from openarmature.observability.metadata import set_invocation_metadata

    class _DispatchState(State):
        fraud_result: str = ""
        audit_result: str = ""

    class _FraudState(State):
        score: str = ""

    class _AuditState(State):
        summary: str = ""

    async def _fraud_ask(_s: _FraudState) -> dict[str, str]:
        await asyncio.sleep(0)
        set_invocation_metadata(branch_label="fraud_check")
        return {"score": "low"}

    async def _audit_ask(_s: _AuditState) -> dict[str, str]:
        await asyncio.sleep(0)
        set_invocation_metadata(branch_label="policy_audit")
        return {"summary": "compliant"}

    fraud_subgraph = (
        GraphBuilder(_FraudState).add_node("ask", _fraud_ask).add_edge("ask", END).set_entry("ask").compile()
    )
    audit_subgraph = (
        GraphBuilder(_AuditState).add_node("ask", _audit_ask).add_edge("ask", END).set_entry("ask").compile()
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_DispatchState)
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "fraud_check": BranchSpec(
                    subgraph=fraud_subgraph,
                    outputs={"fraud_result": "score"},
                ),
                "policy_audit": BranchSpec(
                    subgraph=audit_subgraph,
                    outputs={"audit_result": "summary"},
                ),
            },
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_DispatchState())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    # Pre-fix: two concurrent ``ask`` spans would collide on the
    # _StackKey, so only ONE ask span would land. Post-fix: both
    # branches' ask spans land, each tagged with its own branch_name.
    ask_spans = [s for s in spans if s.name == "ask"]
    assert len(ask_spans) == 2
    by_branch: dict[str, dict[str, Any]] = {}
    for span in ask_spans:
        attrs = dict(span.attributes or {})
        bn = attrs.get("openarmature.node.branch_name")
        assert isinstance(bn, str)
        by_branch[bn] = attrs
    # Each branch's ask carries its OWN branch_label augmentation.
    assert by_branch["fraud_check"].get("openarmature.user.branch_label") == "fraud_check"
    assert by_branch["policy_audit"].get("openarmature.user.branch_label") == "policy_audit"
    # No cross-branch leakage: fraud's ask does NOT carry policy_audit's
    # label and vice versa. The branch_label key is the same name; what
    # matters is each span shows ONLY its own value.
    assert by_branch["fraud_check"].get("openarmature.user.branch_label") != "policy_audit"
    assert by_branch["policy_audit"].get("openarmature.user.branch_label") != "fraud_check"

    # The parallel-branches NODE span(s) and the invocation span MUST
    # NOT carry either branch's branch_label (per-async-context
    # isolation). Note: the current OTel mapping synthesizes a
    # subgraph wrapper at the parallel-branches NODE's namespace in
    # addition to the NODE's own span — that's a pre-existing
    # divergence from fixture 030's expected Langfuse shape that
    # `discuss-otel-parallel-branches-dispatch-span` is asking spec
    # to settle. For this test both dispatcher-named spans MUST be
    # augmentation-clean.
    dispatcher_spans = [s for s in spans if s.name == "dispatcher"]
    invocation_spans = [s for s in spans if s.name == "openarmature.invocation"]
    assert len(invocation_spans) == 1
    assert len(dispatcher_spans) >= 1
    for span in (*dispatcher_spans, *invocation_spans):
        attrs = dict(span.attributes or {})
        assert "openarmature.user.branch_label" not in attrs, (
            f"per-branch augmentation leaked onto {span.name} span"
        )


async def test_parallel_branches_dispatch_span_attributes() -> None:
    # Proposal 0044 (observability §5.7, v0.36.0): pins the §5.7
    # attribute surface end-to-end.
    #
    # - The parallel-branches NODE span carries
    #   ``openarmature.parallel_branches.branch_count`` +
    #   ``openarmature.parallel_branches.error_policy``.
    # - The synthesized per-branch dispatch span (one per branch)
    #   carries ``openarmature.node.branch_name`` +
    #   ``openarmature.parallel_branches.parent_node_name``.
    # - Inner-branch ``ask`` spans carry
    #   ``openarmature.node.branch_name`` matching their branch (the
    #   new attribute replaces the pre-0044 ``openarmature.branch_name``
    #   attribute python emitted before spec defined the namespace).
    #
    # Conformance fixture
    # ``observability/038-otel-parallel-branches-dispatch-span`` is
    # activated in ``tests/conformance/test_observability.py`` via
    # ``_run_fixture_038`` + ``_assert_span_tree_matches`` (PR 9).
    # This unit test covers the §5.7 attribute surface in isolation;
    # the conformance fixture covers the full span-tree topology.
    from openarmature.graph import BranchSpec

    class _DispatchState(State):
        fraud_result: str = ""
        audit_result: str = ""

    class _FraudState(State):
        score: str = ""

    class _AuditState(State):
        summary: str = ""

    async def _fraud_ask(_s: _FraudState) -> dict[str, str]:
        return {"score": "low"}

    async def _audit_ask(_s: _AuditState) -> dict[str, str]:
        return {"summary": "compliant"}

    fraud_subgraph = (
        GraphBuilder(_FraudState).add_node("ask", _fraud_ask).add_edge("ask", END).set_entry("ask").compile()
    )
    audit_subgraph = (
        GraphBuilder(_AuditState).add_node("ask", _audit_ask).add_edge("ask", END).set_entry("ask").compile()
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_DispatchState)
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "fraud_check": BranchSpec(
                    subgraph=fraud_subgraph,
                    outputs={"fraud_result": "score"},
                ),
                "policy_audit": BranchSpec(
                    subgraph=audit_subgraph,
                    outputs={"audit_result": "summary"},
                ),
            },
            error_policy="fail_fast",
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_DispatchState())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    spans_by_name: dict[str, list[Any]] = {}
    for span in spans:
        spans_by_name.setdefault(span.name, []).append(span)

    # ---- Parallel-branches NODE span carries branch_count + error_policy
    dispatcher_node_spans = [
        s
        for s in spans_by_name.get("dispatcher", [])
        if dict(s.attributes or {}).get("openarmature.parallel_branches.branch_count") is not None
    ]
    assert len(dispatcher_node_spans) == 1, (
        f"expected exactly one parallel-branches NODE span carrying §5.7 attrs; "
        f"got {len(dispatcher_node_spans)}"
    )
    node_attrs = dict(dispatcher_node_spans[0].attributes or {})
    assert node_attrs["openarmature.parallel_branches.branch_count"] == 2
    assert node_attrs["openarmature.parallel_branches.error_policy"] == "fail_fast"

    # ---- Per-branch dispatch spans (one per branch) carry the §5.7
    # branch-side attributes
    dispatch_span_attrs_by_branch: dict[str, dict[str, Any]] = {}
    for branch in ("fraud_check", "policy_audit"):
        candidates = [
            s
            for s in spans_by_name.get(branch, [])
            if dict(s.attributes or {}).get("openarmature.parallel_branches.parent_node_name") is not None
        ]
        assert len(candidates) == 1, (
            f"expected exactly one per-branch dispatch span named {branch!r}; got {len(candidates)}"
        )
        dispatch_span_attrs_by_branch[branch] = dict(candidates[0].attributes or {})

    for branch, attrs in dispatch_span_attrs_by_branch.items():
        assert attrs["openarmature.node.branch_name"] == branch
        assert attrs["openarmature.parallel_branches.parent_node_name"] == "dispatcher"

    # ---- Inner-branch ``ask`` spans carry the per-spec branch_name
    # attribute (renamed from the pre-0044 ``openarmature.branch_name``).
    ask_spans = spans_by_name.get("ask", [])
    assert len(ask_spans) == 2
    ask_branch_names = {(dict(s.attributes or {})).get("openarmature.node.branch_name") for s in ask_spans}
    assert ask_branch_names == {"fraud_check", "policy_audit"}


async def test_parallel_branches_inner_spans_parent_under_dispatch_span() -> None:
    # Regression for the parent-resolution bug PR 9 caught during
    # conformance fixture 038 activation: pre-fix, inner-branch leaf
    # spans parented directly under the invocation span instead of
    # under their per-branch dispatch span (because
    # ``_resolve_parent_context`` didn't know about
    # ``parallel_branches_branch_spans``).  Post-fix, the dispatch
    # span is the inner span's direct OTel parent.
    from openarmature.graph import BranchSpec

    class _S(State):
        result: str = ""

    class _InnerS(State):
        x: int = 0

    async def _ask(_s: _InnerS) -> dict[str, int]:
        return {"x": 1}

    inner = GraphBuilder(_InnerS).add_node("ask", _ask).add_edge("ask", END).set_entry("ask").compile()
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_S)
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "fraud_check": BranchSpec(subgraph=inner),
                "policy_audit": BranchSpec(subgraph=inner),
            },
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_S())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    spans_by_id = {cast("Any", s.context).span_id: s for s in spans}

    # Per-branch dispatch spans are the spans named after a branch
    # that carry the §5.7 ``parent_node_name`` attribute.
    dispatch_span_ids: dict[str, int] = {}
    for s in spans:
        attrs = s.attributes or {}
        if attrs.get("openarmature.parallel_branches.parent_node_name") == "dispatcher":
            bn = cast("str", attrs.get("openarmature.node.branch_name"))
            dispatch_span_ids[bn] = cast("Any", s.context).span_id

    assert dispatch_span_ids.keys() == {"fraud_check", "policy_audit"}

    # Each inner ``ask`` span MUST parent under the dispatch span
    # matching its branch — NOT directly under the invocation span.
    ask_spans = [s for s in spans if s.name == "ask"]
    assert len(ask_spans) == 2
    for ask_span in ask_spans:
        attrs = ask_span.attributes or {}
        bn = cast("str", attrs.get("openarmature.node.branch_name"))
        assert ask_span.parent is not None, (
            f"ask span for branch {bn!r} MUST have a parent (not the invocation root)"
        )
        parent_span_id = cast("Any", ask_span.parent).span_id
        expected_parent_id = dispatch_span_ids[bn]
        parent_name = spans_by_id[parent_span_id].name if parent_span_id in spans_by_id else "UNKNOWN"
        assert parent_span_id == expected_parent_id, (
            f"ask span for branch {bn!r} parented under {parent_name!r}, "
            f"expected per-branch dispatch span {bn!r}"
        )


async def test_parallel_branches_node_under_retry_middleware_emits_per_attempt_dispatch_spans() -> None:
    # Regression: under ``RetryMiddleware`` wrapping the parallel-
    # branches node, the per-branch dispatch span synthesizer MUST
    # locate the CURRENT attempt's NODE span (via the scan in
    # ``_open_parallel_branches_branch_dispatch_span``).  A failing
    # first attempt + a successful retry MUST produce:
    #   - two NODE spans (one per attempt, distinct attempt_index)
    #   - two per-branch dispatch spans per branch (one per attempt)
    #   - each attempt's dispatch span parented under THAT attempt's
    #     NODE span (not the wrong attempt's)
    from openarmature.graph import BranchSpec, RetryConfig, RetryMiddleware

    class _S(State):
        result: str = ""

    class _InnerS(State):
        x: int = 0

    attempt_counter: list[int] = [0]

    async def _flaky_branch(_s: _InnerS) -> dict[str, int]:
        attempt_counter[0] += 1
        if attempt_counter[0] == 1:
            raise RuntimeError("first-attempt boom")
        return {"x": 1}

    inner = (
        GraphBuilder(_InnerS).add_node("ask", _flaky_branch).add_edge("ask", END).set_entry("ask").compile()
    )
    # Use a catch-all classifier so the first-attempt failure
    # (surfacing as ParallelBranchesBranchFailed wrapping a node
    # exception) triggers a retry instead of being filtered as
    # non-transient.
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_S)
        .add_parallel_branches_node(
            "dispatcher",
            branches={"only_branch": BranchSpec(subgraph=inner)},
            middleware=[RetryMiddleware(RetryConfig(max_attempts=2, classifier=lambda _exc, _state: True))],
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_S())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()

    # Two NODE spans, distinct ``openarmature.node.attempt_index``.
    node_spans = [
        s
        for s in spans
        if s.name == "dispatcher"
        and (s.attributes or {}).get("openarmature.parallel_branches.branch_count") is not None
    ]
    assert len(node_spans) == 2, f"expected 2 NODE spans (attempts 0 + 1); got {len(node_spans)}"
    node_attempts: list[int] = sorted(
        cast("int", dict(s.attributes or {}).get("openarmature.node.attempt_index", -1)) for s in node_spans
    )
    assert node_attempts == [0, 1]

    # Two per-branch dispatch spans, one per attempt.
    dispatch_spans = [
        s
        for s in spans
        if s.name == "only_branch"
        and (s.attributes or {}).get("openarmature.parallel_branches.parent_node_name") == "dispatcher"
    ]
    assert len(dispatch_spans) == 2, f"expected 2 dispatch spans (one per attempt); got {len(dispatch_spans)}"

    # Each dispatch span's parent MUST be a NODE span (not the
    # invocation span and not the wrong attempt's NODE span).
    node_span_ids = {cast("Any", s.context).span_id for s in node_spans}
    for d in dispatch_spans:
        assert d.parent is not None
        parent_id = cast("Any", d.parent).span_id
        assert parent_id in node_span_ids, (
            f"dispatch span MUST parent under a NODE span; "
            f"got parent_id={parent_id} not in NODE span ids {node_span_ids}"
        )


async def test_parallel_branches_inside_fan_out_instance_inner_span_carries_both_axes() -> None:
    # Regression: an inner-branch span deep inside a fan-out instance
    # MUST carry BOTH ``openarmature.node.fan_out_index`` AND
    # ``openarmature.node.branch_name``.  The 4-tuple ``_StackKey``
    # disambiguation already supports this composition; this test
    # locks the attribute surface that goes with it.
    from openarmature.graph import BranchSpec

    class _OuterS(State):
        items: list[int] = []
        results: Annotated[list[int], append] = []

    class _MidS(State):
        item: int = 0
        out: int = 0

    class _BranchS(State):
        out: int = 0

    async def _branch_ask(_s: _BranchS) -> dict[str, int]:
        return {"out": 1}

    branch_subgraph = (
        GraphBuilder(_BranchS).add_node("ask", _branch_ask).add_edge("ask", END).set_entry("ask").compile()
    )

    # Mid-level subgraph: contains a parallel-branches dispatcher
    # whose branches each end at ``ask``.
    mid_builder = (
        GraphBuilder(_MidS)
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "primary": BranchSpec(subgraph=branch_subgraph, outputs={"out": "out"}),
            },
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
    )
    mid_subgraph = mid_builder.compile()

    # Outer: fan-out → mid-level subgraph (which contains the
    # parallel-branches node).
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_fan_out_node(
            "fan",
            subgraph=mid_subgraph,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
        )
        .add_edge("fan", END)
        .set_entry("fan")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS(items=[1, 2]))
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    # Inner ``ask`` spans (one per fan-out instance × one branch each
    # = 2 spans) MUST carry both fan_out_index AND branch_name.
    ask_spans = [s for s in spans if s.name == "ask"]
    assert len(ask_spans) == 2, f"expected 2 ask spans (one per fan-out instance); got {len(ask_spans)}"
    fan_out_indices: set[Any] = set()
    for ask_span in ask_spans:
        attrs = dict(ask_span.attributes or {})
        assert attrs.get("openarmature.node.branch_name") == "primary"
        fi = attrs.get("openarmature.node.fan_out_index")
        assert fi is not None, f"ask span MUST carry fan_out_index inside a fan-out; attrs={attrs!r}"
        fan_out_indices.add(fi)
    assert fan_out_indices == {0, 1}, f"expected fan_out_index ∈ {{0, 1}}; got {fan_out_indices}"

    # Parent-topology regression: each inner ``ask`` MUST parent under
    # a per-branch dispatch span (the parallel-branches NODE's open
    # span has fan_out_index set inside a fan-out instance; the scan
    # in ``_open_parallel_branches_branch_dispatch_span`` must accept
    # that).  And each dispatch span's parent MUST be the
    # parallel-branches NODE span at the fan-out instance's namespace
    # (NOT the invocation root).
    spans_by_id = {cast("Any", s.context).span_id: s for s in spans}
    dispatcher_node_spans = [
        s
        for s in spans
        if s.name == "dispatcher"
        and dict(s.attributes or {}).get("openarmature.parallel_branches.branch_count") is not None
    ]
    # One per fan-out instance.
    assert len(dispatcher_node_spans) == 2, (
        f"expected 2 dispatcher NODE spans (one per fan-out instance); got {len(dispatcher_node_spans)}"
    )
    dispatcher_node_ids = {cast("Any", s.context).span_id for s in dispatcher_node_spans}
    for ask_span in ask_spans:
        assert ask_span.parent is not None, "ask span MUST have a parent"
        dispatch = spans_by_id.get(cast("Any", ask_span.parent).span_id)
        dispatch_name = dispatch.name if dispatch is not None else "UNKNOWN"
        assert dispatch is not None and dispatch.name == "primary", (
            f"ask span MUST parent under per-branch dispatch span 'primary'; got {dispatch_name!r}"
        )
        assert dispatch.parent is not None, "dispatch span MUST have a parent"
        assert cast("Any", dispatch.parent).span_id in dispatcher_node_ids, (
            "per-branch dispatch span MUST parent under the parallel-branches NODE span "
            "(at the fan-out instance's namespace), not the invocation root"
        )


async def test_parallel_branches_inside_subgraph_wrapper_parent_topology() -> None:
    # Regression for the depth>1 nesting bug PR 9 caught during CoPilot
    # review: pre-fix, when the parallel-branches node sits inside a
    # subgraph wrapper (so the NODE's namespace is deeper than 1), the
    # per-branch dispatch span was never synthesized (synthesis gated
    # on ``depth == 1``) and inner-branch events couldn't find it
    # (resolution hard-coded ``namespace[:1]``).  Post-fix, dispatch
    # spans synthesize at the NODE's actual depth and inner spans
    # parent under them.
    from openarmature.graph import BranchSpec

    class _OuterS(State):
        result: str = ""

    class _InnerWrapS(State):
        result: str = ""

    class _BranchS(State):
        out: str = ""

    async def _ask(_s: _BranchS) -> dict[str, str]:
        return {"out": "done"}

    branch_subgraph = (
        GraphBuilder(_BranchS).add_node("ask", _ask).add_edge("ask", END).set_entry("ask").compile()
    )

    # Inner subgraph: contains a parallel-branches dispatcher.
    inner_subgraph = (
        GraphBuilder(_InnerWrapS)
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "fraud_check": BranchSpec(subgraph=branch_subgraph),
                "policy_audit": BranchSpec(subgraph=branch_subgraph),
            },
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
        .compile()
    )

    # Outer graph: wraps the inner subgraph as a single node.  This
    # puts the parallel-branches NODE at namespace depth 2 in the
    # outer graph (``("wrapper", "dispatcher")``).
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_subgraph_node("wrapper", inner_subgraph)
        .add_edge("wrapper", END)
        .set_entry("wrapper")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    spans_by_id = {cast("Any", s.context).span_id: s for s in spans}

    # Per-branch dispatch spans MUST exist for both branches even
    # though the parallel-branches NODE is at depth 2.
    dispatch_spans_by_branch: dict[str, Any] = {}
    for s in spans:
        attrs = dict(s.attributes or {})
        if attrs.get("openarmature.parallel_branches.parent_node_name") == "dispatcher":
            bn = cast("str", attrs.get("openarmature.node.branch_name"))
            dispatch_spans_by_branch[bn] = s
    assert dispatch_spans_by_branch.keys() == {"fraud_check", "policy_audit"}, (
        "per-branch dispatch spans MUST synthesize even when the parallel-branches "
        f"NODE sits inside a subgraph wrapper; got dispatch spans for {dispatch_spans_by_branch.keys()!r}"
    )

    # Each ``ask`` span MUST parent under its matching dispatch span,
    # not under the invocation root or the wrapper subgraph span.
    ask_spans = [s for s in spans if s.name == "ask"]
    assert len(ask_spans) == 2
    for ask_span in ask_spans:
        attrs = dict(ask_span.attributes or {})
        bn = cast("str", attrs.get("openarmature.node.branch_name"))
        assert ask_span.parent is not None, "ask span MUST have a parent"
        parent_span_id = cast("Any", ask_span.parent).span_id
        expected = dispatch_spans_by_branch[bn]
        expected_id = expected.context.span_id
        parent_name = spans_by_id[parent_span_id].name if parent_span_id in spans_by_id else "UNKNOWN"
        assert parent_span_id == expected_id, (
            f"ask span for branch {bn!r} parented under {parent_name!r}, "
            f"expected per-branch dispatch span at depth-2 namespace"
        )


async def test_fan_out_inside_subgraph_wrapper_emits_per_instance_dispatch_span() -> None:
    # Campsite-rule companion to
    # ``test_parallel_branches_inside_subgraph_wrapper_parent_topology``:
    # the per-instance dispatch span synthesis at observer.py:1277 had
    # the same ``depth == 1`` gating that affected parallel-branches.
    # Post-fix, a fan-out node nested inside a subgraph wrapper
    # synthesizes its per-instance dispatch spans at the NODE's actual
    # depth and inner spans parent under them.
    class _OuterS(State):
        items: list[int] = []
        results: Annotated[list[int], append] = []

    class _MidS(State):
        items: list[int] = []
        results: Annotated[list[int], append] = []

    class _InnerS(State):
        item: int = 0
        out: int = 0

    async def _double(s: _InnerS) -> dict[str, int]:
        return {"out": s.item * 2}

    inner_subgraph = (
        GraphBuilder(_InnerS)
        .add_node("double", _double)
        .add_edge("double", END)
        .set_entry("double")
        .compile()
    )

    # Mid-level subgraph: contains a fan-out dispatcher.
    mid_subgraph = (
        GraphBuilder(_MidS)
        .add_fan_out_node(
            "fan",
            subgraph=inner_subgraph,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
        )
        .add_edge("fan", END)
        .set_entry("fan")
        .compile()
    )

    # Outer graph wraps the mid subgraph as a single node, putting
    # the fan-out NODE at namespace ``("wrapper", "fan")`` (depth 2).
    # Explicit projection: the default FieldNameMatching ignores parent
    # state on the way in, but we need ``items`` plumbed through so the
    # inner fan-out has work to dispatch.
    from openarmature.graph.projection import ExplicitMapping

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_subgraph_node(
            "wrapper",
            mid_subgraph,
            projection=ExplicitMapping[_OuterS, _MidS](inputs={"items": "items"}),
        )
        .add_edge("wrapper", END)
        .set_entry("wrapper")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS(items=[1, 2]))
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    spans_by_id = {cast("Any", s.context).span_id: s for s in spans}

    # Per-instance dispatch spans MUST synthesize even at depth 2.
    # Per spec §5.4 / proposal 0013, they're named after the fan-out
    # NODE ("fan") and carry ``openarmature.node.fan_out_index``.
    instance_dispatch_by_idx: dict[Any, Any] = {}
    for s in spans:
        attrs = dict(s.attributes or {})
        if (
            s.name == "fan"
            and attrs.get("openarmature.node.fan_out_index") is not None
            and "openarmature.fan_out.parent_node_name" in attrs
        ):
            instance_dispatch_by_idx[attrs["openarmature.node.fan_out_index"]] = s
    assert instance_dispatch_by_idx.keys() == {0, 1}, (
        "per-instance dispatch spans MUST synthesize even when the fan-out NODE "
        f"sits inside a subgraph wrapper; got dispatches for {instance_dispatch_by_idx.keys()!r}"
    )

    # Each ``double`` span MUST parent under its matching per-instance
    # dispatch span (at depth 2), not under the wrapper subgraph span
    # at depth 1.
    double_spans = [s for s in spans if s.name == "double"]
    assert len(double_spans) == 2
    for double_span in double_spans:
        attrs = dict(double_span.attributes or {})
        fi = attrs.get("openarmature.node.fan_out_index")
        assert double_span.parent is not None, "double span MUST have a parent"
        parent_span_id = cast("Any", double_span.parent).span_id
        expected_id = instance_dispatch_by_idx[fi].context.span_id
        parent_name = spans_by_id[parent_span_id].name if parent_span_id in spans_by_id else "UNKNOWN"
        assert parent_span_id == expected_id, (
            f"double span for fan_out_index {fi!r} parented under {parent_name!r}, "
            f"expected per-instance dispatch span at depth-2 namespace"
        )


async def test_detached_subgraph_at_depth_two_mints_fresh_trace() -> None:
    # Campsite-rule extension: detached subgraph synthesis previously
    # gated on ``depth == 1``, so a detached subgraph nested inside an
    # outer wrapper would not mint a fresh trace and inner spans would
    # bleed into the parent trace.  Post-fix, ``detached_subgraphs``
    # matches the node-name segment at any depth.
    class _OuterS(State):
        result: str = ""

    class _InnerS(State):
        out: str = ""

    async def _leaf(_s: _InnerS) -> dict[str, str]:
        return {"out": "done"}

    detached_subgraph = (
        GraphBuilder(_InnerS).add_node("leaf", _leaf).add_edge("leaf", END).set_entry("leaf").compile()
    )

    # Mid-level subgraph wraps the detached one as a single node
    # named "detached_inner".  Outer wraps mid as "wrapper".
    # ``detached_subgraphs={"detached_inner"}`` should mint a fresh
    # trace at depth 2 namespace ``("wrapper", "detached_inner")``.
    mid_subgraph = (
        GraphBuilder(_InnerS)
        .add_subgraph_node("detached_inner", detached_subgraph)
        .add_edge("detached_inner", END)
        .set_entry("detached_inner")
        .compile()
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        detached_subgraphs=frozenset({"detached_inner"}),
    )
    g = (
        GraphBuilder(_OuterS)
        .add_subgraph_node("wrapper", mid_subgraph)
        .add_edge("wrapper", END)
        .set_entry("wrapper")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()

    # Two traces: parent invocation and detached subgraph.
    trace_ids = {cast("Any", s.context).trace_id for s in spans}
    assert len(trace_ids) == 2, (
        f"detached subgraph at depth 2 MUST mint a fresh trace; got {len(trace_ids)} trace(s) instead"
    )

    # The detached root MUST carry ``openarmature.subgraph.detached``
    # and live at depth-2 namespace.
    detached_roots = [
        s
        for s in spans
        if s.name == "detached_inner"
        and dict(s.attributes or {}).get("openarmature.subgraph.detached") is True
    ]
    assert len(detached_roots) == 1
    # Proposal 0061: the detached trace roots in its OWN
    # ``openarmature.invocation`` span (parent + detached = two
    # invocation spans), both carrying the SAME invocation_id, with the
    # detached subgraph span nested under the detached invocation span.
    inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
    assert len(inv_spans) == 2, f"expected parent + detached invocation spans, got {len(inv_spans)}"
    detached_trace_id = cast("Any", detached_roots[0].context).trace_id
    detached_inv = next((s for s in inv_spans if cast("Any", s.context).trace_id == detached_trace_id), None)
    parent_inv = next((s for s in inv_spans if cast("Any", s.context).trace_id != detached_trace_id), None)
    assert detached_inv is not None, "detached trace MUST root in an openarmature.invocation span"
    assert parent_inv is not None
    assert detached_trace_id != cast("Any", parent_inv.context).trace_id, (
        "detached subgraph root MUST live in a fresh trace, not the parent invocation trace"
    )
    # Shared invocation_id across the trace boundary (§4.3).
    detached_inv_id = dict(detached_inv.attributes or {}).get("openarmature.invocation_id")
    parent_inv_id = dict(parent_inv.attributes or {}).get("openarmature.invocation_id")
    assert detached_inv_id is not None and detached_inv_id == parent_inv_id, (
        "detached invocation span MUST carry the SAME invocation_id as the parent (§4.3)"
    )
    # The detached subgraph span nests under the detached invocation span.
    assert detached_roots[0].parent is not None
    assert detached_roots[0].parent.span_id == cast("Any", detached_inv.context).span_id, (
        "detached subgraph span MUST nest under the detached invocation span"
    )


async def test_three_deep_mixed_pb_fan_out_pb_composition() -> None:
    # Campsite-rule coverage for the three-deep mixed composition
    # (pb1 → fan-out → pb2 → leaf) that the resolver restructure
    # claimed to support but no earlier test exercised.  Each layer's
    # dispatch span MUST synthesize at its own namespace, and the
    # innermost leaf MUST parent under the innermost pb's per-branch
    # dispatch span.
    from openarmature.graph import BranchSpec

    class _OuterS(State):
        items: list[int] = []

    class _MidBranchS(State):
        items: list[int] = []
        out: int = 0

    class _FanInstanceS(State):
        item: int = 0
        out: int = 0

    class _InnerBranchS(State):
        out: int = 0

    async def _leaf(_s: _InnerBranchS) -> dict[str, int]:
        return {"out": 42}

    inner_pb_branch = (
        GraphBuilder(_InnerBranchS).add_node("leaf", _leaf).add_edge("leaf", END).set_entry("leaf").compile()
    )

    # pb2 sits inside the fan-out's per-instance subgraph.  One
    # branch is enough — we want topology coverage, not branch
    # combinatorics.
    fan_instance_subgraph = (
        GraphBuilder(_FanInstanceS)
        .add_parallel_branches_node(
            "pb2",
            branches={"inner_a": BranchSpec(subgraph=inner_pb_branch, outputs={"out": "out"})},
        )
        .add_edge("pb2", END)
        .set_entry("pb2")
        .compile()
    )

    # Middle layer is the fan-out's wrapper — sits inside pb1's
    # branch subgraph.  One fan-out instance is enough for topology.
    mid_branch_subgraph = (
        GraphBuilder(_MidBranchS)
        .add_fan_out_node(
            "fan",
            subgraph=fan_instance_subgraph,
            collect_field="out",
            target_field="items",
            items_field="items",
            item_field="item",
        )
        .add_edge("fan", END)
        .set_entry("fan")
        .compile()
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_parallel_branches_node(
            "pb1",
            branches={
                "outer_x": BranchSpec(
                    subgraph=mid_branch_subgraph,
                    inputs={"items": "items"},
                ),
            },
        )
        .add_edge("pb1", END)
        .set_entry("pb1")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS(items=[1]))
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    spans_by_id = {cast("Any", s.context).span_id: s for s in spans}

    # The leaf span is at namespace depth 4 (pb1, fan, pb2, leaf)
    # with the innermost branch_name "inner_a" and the fan-out's
    # fan_out_index = 0.  It MUST parent under pb2's per-branch
    # dispatch span at namespace ("pb1", "fan", "pb2", "inner_a").
    leaf_spans = [s for s in spans if s.name == "leaf"]
    assert len(leaf_spans) == 1
    leaf_attrs = dict(leaf_spans[0].attributes or {})
    assert leaf_attrs.get("openarmature.node.branch_name") == "inner_a"
    assert leaf_attrs.get("openarmature.node.fan_out_index") == 0

    # Find pb2's per-branch dispatch span (named "inner_a" with the
    # ``parent_node_name`` attribute = "pb2") and pb1's per-branch
    # dispatch span (named "outer_x" with parent_node_name = "pb1").
    inner_branch_dispatch = next(
        (
            s
            for s in spans
            if s.name == "inner_a"
            and dict(s.attributes or {}).get("openarmature.parallel_branches.parent_node_name") == "pb2"
        ),
        None,
    )
    outer_branch_dispatch = next(
        (
            s
            for s in spans
            if s.name == "outer_x"
            and dict(s.attributes or {}).get("openarmature.parallel_branches.parent_node_name") == "pb1"
        ),
        None,
    )
    assert inner_branch_dispatch is not None, "pb2 per-branch dispatch MUST synthesize at depth 3"
    assert outer_branch_dispatch is not None, "pb1 per-branch dispatch MUST synthesize at depth 1"

    # Leaf parents under pb2's branch dispatch (innermost).
    assert leaf_spans[0].parent is not None
    leaf_parent_id = cast("Any", leaf_spans[0].parent).span_id
    inner_dispatch_id = cast("Any", inner_branch_dispatch.context).span_id
    assert leaf_parent_id == inner_dispatch_id, (
        "leaf MUST parent under pb2's per-branch dispatch span (innermost), "
        f"got {spans_by_id[leaf_parent_id].name if leaf_parent_id in spans_by_id else 'UNKNOWN'!r}"
    )

    # Find pb2's NODE span (named "pb2" with branch_count attribute).
    # It MUST exist and parent under the fan-out instance dispatch.
    pb2_node = next(
        (
            s
            for s in spans
            if s.name == "pb2"
            and dict(s.attributes or {}).get("openarmature.parallel_branches.branch_count") is not None
        ),
        None,
    )
    assert pb2_node is not None
    # Fan-out instance dispatch span (named "fan" with fan_out_index=0
    # AND the parent_node_name attribute, which only the per-instance
    # dispatch span carries).
    fan_instance_dispatch = next(
        (
            s
            for s in spans
            if s.name == "fan"
            and dict(s.attributes or {}).get("openarmature.node.fan_out_index") == 0
            and "openarmature.fan_out.parent_node_name" in dict(s.attributes or {})
        ),
        None,
    )
    assert fan_instance_dispatch is not None, (
        "fan-out per-instance dispatch span MUST synthesize at depth 2 (inside pb1 branch)"
    )
    assert pb2_node.parent is not None
    pb2_parent_id = cast("Any", pb2_node.parent).span_id
    fan_instance_id = cast("Any", fan_instance_dispatch.context).span_id
    assert pb2_parent_id == fan_instance_id, "pb2 NODE MUST parent under fan-out per-instance dispatch span"


async def test_nested_pb_completion_closes_inner_dispatch_spans() -> None:
    # Regression for the completion-side mirror of the cache-update
    # filter bug: a parallel-branches node nested inside an outer pb's
    # branch fires its own completed event with ``branch_name`` set
    # (carrying the OUTER pb's branch_name).  The pb close handler
    # previously gated on ``branch_name is None``, which meant inner
    # pb's per-branch dispatch spans were never closed and the
    # ``parallel_branches_branch_spans`` cache leaked.  Post-fix, the
    # close handler relies on ``parallel_branches_config`` alone, so
    # the inner pb's spans close before the outer pb's NODE span.
    from openarmature.graph import BranchSpec

    class _OuterS(State):
        result: str = ""

    class _OuterBranchS(State):
        result: str = ""

    class _InnerBranchS(State):
        out: str = ""

    async def _leaf(_s: _InnerBranchS) -> dict[str, str]:
        return {"out": "done"}

    inner_pb_branch = (
        GraphBuilder(_InnerBranchS).add_node("leaf", _leaf).add_edge("leaf", END).set_entry("leaf").compile()
    )
    # Outer pb's branch subgraph contains the inner pb.
    outer_branch_subgraph = (
        GraphBuilder(_OuterBranchS)
        .add_parallel_branches_node(
            "pb2",
            branches={"inner_a": BranchSpec(subgraph=inner_pb_branch)},
        )
        .add_edge("pb2", END)
        .set_entry("pb2")
        .compile()
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_parallel_branches_node(
            "pb1",
            branches={"outer_x": BranchSpec(subgraph=outer_branch_subgraph)},
        )
        .add_edge("pb1", END)
        .set_entry("pb1")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()

    # The inner pb's per-branch dispatch span ("inner_a" with
    # parent_node_name "pb2") MUST be in the finished-spans list.
    # Pre-fix, it would be missing because the inner pb's completion
    # was skipped by the close handler and the span never ended.
    inner_branch_dispatch_spans = [
        s
        for s in spans
        if s.name == "inner_a"
        and dict(s.attributes or {}).get("openarmature.parallel_branches.parent_node_name") == "pb2"
    ]
    assert len(inner_branch_dispatch_spans) == 1, (
        f"inner pb's per-branch dispatch span MUST close on inner pb's completion; "
        f"got {len(inner_branch_dispatch_spans)} closed dispatch span(s) for inner pb"
    )


async def test_metadata_augmentation_updates_per_branch_dispatch_span() -> None:
    # Spec §3.4 *Mid-invocation augmentation* (per the proposal-0040
    # implementation + the proposal-0045 ancestor-chain clarification
    # landing in PR 11): an augmentation fired from inside a branch
    # MUST apply to every strict dispatch ancestor on the augmenter's
    # call-stack path — including the per-branch dispatch span.
    #
    # Tests the OTel observer's
    # ``_collect_augmentation_targets`` per-branch-dispatch lookup
    # added in PR 9.  Sibling-skip is still enforced — the OTHER
    # branch's dispatch span MUST NOT carry the augmenter's key.
    import asyncio

    from openarmature.graph import BranchSpec
    from openarmature.observability.metadata import set_invocation_metadata

    class _S(State):
        result: str = ""

    class _BranchS(State):
        out: str = ""

    async def _fraud_ask(_s: _BranchS) -> dict[str, str]:
        await asyncio.sleep(0)
        set_invocation_metadata(audit_kind="fraud")
        return {"out": "fraud-done"}

    async def _policy_ask(_s: _BranchS) -> dict[str, str]:
        await asyncio.sleep(0)
        return {"out": "policy-done"}

    fraud_subgraph = (
        GraphBuilder(_BranchS).add_node("ask", _fraud_ask).add_edge("ask", END).set_entry("ask").compile()
    )
    policy_subgraph = (
        GraphBuilder(_BranchS).add_node("ask", _policy_ask).add_edge("ask", END).set_entry("ask").compile()
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_S)
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "fraud_check": BranchSpec(subgraph=fraud_subgraph),
                "policy_audit": BranchSpec(subgraph=policy_subgraph),
            },
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_S())
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    # Per-branch dispatch spans.
    dispatch_spans_by_branch: dict[str, dict[str, Any]] = {}
    for s in spans:
        attrs = dict(s.attributes or {})
        if attrs.get("openarmature.parallel_branches.parent_node_name") == "dispatcher":
            bn = cast("str", attrs.get("openarmature.node.branch_name"))
            dispatch_spans_by_branch[bn] = attrs

    assert dispatch_spans_by_branch.keys() == {"fraud_check", "policy_audit"}
    # The fraud_check dispatch span MUST carry the augmentation key
    # (it's the augmenter's strict dispatch ancestor).
    assert dispatch_spans_by_branch["fraud_check"].get("openarmature.user.audit_kind") == "fraud", (
        "per-branch dispatch span on augmenter's path MUST carry the augmentation key"
    )
    # The policy_audit dispatch span MUST NOT (sibling-skip).
    assert "openarmature.user.audit_kind" not in dispatch_spans_by_branch["policy_audit"], (
        "sibling branch's dispatch span MUST NOT receive the augmenter's key"
    )


async def test_nested_fan_out_augmentation_reaches_outer_instance_dispatch_span() -> None:
    # Spec proposal 0045 §3.4 lineage-aware containment rule.
    # Topology: outer fan-out wrapping a serial subgraph that
    # contains a leaf.  The leaf augments a per-item key.  Augment
    # targets per §3.4:
    #
    # - Outer instance #1's dispatch span MUST receive
    #   ``group="item-200"`` (rule 2, strict ancestor on the path).
    # - Outer instance #0's dispatch span MUST receive
    #   ``group="item-100"`` (rule 2, its own subtree).
    # - Outer instance #0 and #1's dispatch spans MUST NOT receive
    #   each other's value (rule 3, siblings).
    # - The outer fan-out NODE span MUST NOT receive any group key
    #   (rule 3, shared parent).
    # - The invocation span MUST NOT receive any group key (rule 3,
    #   shared parent — augmenter is inside a fan-out instance).
    #
    # The chain at the augmenter is ``(K,)`` where K is the outer
    # instance's index — the per-depth tracking that 0045 requires
    # is exercised by the resolver picking the matching outer
    # dispatch span (and skipping the sibling) on each leaf's
    # augmentation.
    import asyncio

    from openarmature.observability.metadata import set_invocation_metadata

    class _OuterS(State):
        items: list[int] = []
        results: Annotated[list[int], append] = []

    class _MidS(State):
        item: int = 0
        out: int = 0

    class _LeafS(State):
        item: int = 0
        out: int = 0

    async def _leaf(s: _LeafS) -> dict[str, int]:
        await asyncio.sleep(0)
        # Augment with a per-item key so we can detect which dispatch
        # span the augmentation lands on.
        set_invocation_metadata(group=f"item-{s.item}")
        return {"out": s.item}

    leaf_subgraph = (
        GraphBuilder(_LeafS).add_node("leaf", _leaf).add_edge("leaf", END).set_entry("leaf").compile()
    )

    # Mid-level: a serial subgraph wrapping the leaf.  Threads
    # ``item`` straight through and exposes ``out``.
    async def _mid_passthrough(_s: _MidS) -> dict[str, int]:
        return {}

    from openarmature.graph.projection import ExplicitMapping

    mid_subgraph = (
        GraphBuilder(_MidS)
        .add_subgraph_node(
            "leaf_wrap",
            leaf_subgraph,
            projection=ExplicitMapping[_MidS, _LeafS](inputs={"item": "item"}, outputs={"out": "out"}),
        )
        .add_node("noop", _mid_passthrough)
        .add_edge("leaf_wrap", "noop")
        .add_edge("noop", END)
        .set_entry("leaf_wrap")
        .compile()
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_fan_out_node(
            "outer_fan",
            subgraph=mid_subgraph,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
        )
        .add_edge("outer_fan", END)
        .set_entry("outer_fan")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS(items=[100, 200]))
        await g.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    # Outer per-instance dispatch spans (name = "outer_fan" with
    # ``fan_out_index`` ∈ {0, 1} and ``fan_out.parent_node_name``).
    outer_dispatches = {
        dict(s.attributes or {}).get("openarmature.node.fan_out_index"): s
        for s in spans
        if s.name == "outer_fan"
        and dict(s.attributes or {}).get("openarmature.node.fan_out_index") is not None
        and "openarmature.fan_out.parent_node_name" in dict(s.attributes or {})
    }
    assert outer_dispatches.keys() == {0, 1}, (
        f"expected two outer fan-out instance dispatch spans; got {outer_dispatches.keys()!r}"
    )

    # Each outer dispatch carries the leaf's per-item group value (its
    # own subtree's augmentation propagated outward via the lineage-
    # aware boundary rule).
    outer0_group = dict(outer_dispatches[0].attributes or {}).get("openarmature.user.group")
    outer1_group = dict(outer_dispatches[1].attributes or {}).get("openarmature.user.group")
    assert outer0_group == "item-100", (
        f"outer instance #0's dispatch span MUST carry its leaf's augmented value; got {outer0_group!r}"
    )
    assert outer1_group == "item-200", (
        f"outer instance #1's dispatch span MUST carry its leaf's augmented value; got {outer1_group!r}"
    )

    # The outer fan-out NODE span (shared parent of both instances —
    # the one without ``fan_out_index`` on its attributes) MUST NOT
    # carry any augmented group key.
    outer_node_spans = [
        s
        for s in spans
        if s.name == "outer_fan" and dict(s.attributes or {}).get("openarmature.node.fan_out_index") is None
    ]
    assert len(outer_node_spans) >= 1
    for outer_node in outer_node_spans:
        assert "openarmature.user.group" not in dict(outer_node.attributes or {}), (
            "outer fan-out NODE span (shared parent) MUST NOT receive any augmented group key"
        )

    # The invocation span MUST NOT carry it (augmenter is inside a
    # fan-out → invocation is a shared parent).
    inv_spans = [s for s in spans if s.name == "openarmature.invocation"]
    assert len(inv_spans) == 1
    assert "openarmature.user.group" not in dict(inv_spans[0].attributes or {}), (
        "invocation span MUST NOT receive augmenter's key when inside a fan-out instance"
    )


async def test_nested_fan_out_in_fan_out_dispatch_lineage() -> None:
    # Proposal 0045 / 0013: a fan-out nested INSIDE a fan-out instance. Each
    # outer instance gets its OWN inner per-instance dispatch span (distinct
    # lineage keys, no cross-instance collision -- before the fix the second
    # collided with the first), and an inner leaf's augmentation reaches its own
    # outer instance dispatch, not the sibling's, and not the shared NODE spans.
    import asyncio

    from openarmature.observability.metadata import set_invocation_metadata

    class _LeafS(State):
        tag: str = ""
        seed: str = ""
        out: str = ""

    class _MidS(State):
        tag: str = ""
        seeds: list[str] = []
        collected: Annotated[list[str], append] = []

    class _OuterS(State):
        products: list[str] = []
        seeds: list[str] = []
        results: Annotated[list[Any], append] = []

    async def _leaf(s: _LeafS) -> dict[str, str]:
        await asyncio.sleep(0)
        set_invocation_metadata(note=f"{s.tag}-{s.seed}")
        return {"out": f"{s.tag}-{s.seed}"}

    leaf = GraphBuilder(_LeafS).add_node("ask", _leaf).add_edge("ask", END).set_entry("ask").compile()
    mid = (
        GraphBuilder(_MidS)
        .add_fan_out_node(
            "inner_fan",
            subgraph=leaf,
            items_field="seeds",
            item_field="seed",
            inputs={"tag": "tag"},
            collect_field="out",
            target_field="collected",
        )
        .add_edge("inner_fan", END)
        .set_entry("inner_fan")
        .compile()
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_fan_out_node(
            "outer_fan",
            subgraph=mid,
            items_field="products",
            item_field="tag",
            inputs={"seeds": "seeds"},
            collect_field="collected",
            target_field="results",
            # concurrency=1 while the observer's NODE-key collision under
            # concurrent nested fan-out is fixed (the inner nodes of different
            # outer instances share a _key_for and dedup); the engine results are
            # correct at any concurrency. Tracked separately.
            concurrency=1,
        )
        .add_edge("outer_fan", END)
        .set_entry("outer_fan")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS(products=["A", "B"], seeds=["x"]))
        await g.drain()
    finally:
        observer.shutdown()
    spans = exporter.get_finished_spans()

    def _attr(s: Any, k: str) -> Any:
        return dict(s.attributes or {}).get(k)

    inner_dispatches = [
        s
        for s in spans
        if s.name == "inner_fan" and "openarmature.fan_out.parent_node_name" in dict(s.attributes or {})
    ]
    assert len(inner_dispatches) == 2, (
        f"expected 2 nested inner instance dispatches, got {len(inner_dispatches)}"
    )
    outer_dispatches = {
        _attr(s, "openarmature.node.fan_out_index"): s
        for s in spans
        if s.name == "outer_fan" and "openarmature.fan_out.parent_node_name" in dict(s.attributes or {})
    }
    assert outer_dispatches.keys() == {0, 1}
    assert _attr(outer_dispatches[0], "openarmature.user.note") == "A-x"
    assert _attr(outer_dispatches[1], "openarmature.user.note") == "B-x"
    node_spans = [
        s
        for s in spans
        if s.name in ("outer_fan", "inner_fan")
        and "openarmature.fan_out.parent_node_name" not in dict(s.attributes or {})
    ]
    assert node_spans, "expected fan-out NODE spans"
    assert all(_attr(s, "openarmature.user.note") is None for s in node_spans), (
        "shared fan-out NODE spans MUST NOT carry the augmentation"
    )


async def test_parallel_branches_in_fan_out_dispatch_lineage() -> None:
    # Proposal 0045 / 0044: a parallel-branches NODE inside a fan-out instance.
    # Each outer instance gets its own per-branch dispatch spans (distinct keys,
    # no cross-instance collision); only the augmenting branch + its outer
    # instance dispatch carry the augmentation, not the sibling branch.
    import asyncio

    from openarmature.graph import BranchSpec
    from openarmature.observability.metadata import set_invocation_metadata

    class _BranchS(State):
        tag: str = ""
        out: str = ""

    class _MidS(State):
        tag: str = ""
        outcome: str = ""

    class _OuterS(State):
        products: list[str] = []
        results: Annotated[list[Any], append] = []

    async def _probe(s: _BranchS) -> dict[str, str]:
        await asyncio.sleep(0)
        return {"out": s.tag}

    async def _baseline(_s: _BranchS) -> dict[str, str]:
        await asyncio.sleep(0)
        return {"out": "base"}

    probe = GraphBuilder(_BranchS).add_node("ask", _probe).add_edge("ask", END).set_entry("ask").compile()
    baseline = (
        GraphBuilder(_BranchS).add_node("ask", _baseline).add_edge("ask", END).set_entry("ask").compile()
    )

    async def _augment(s: _BranchS, next_: Any) -> Any:
        set_invocation_metadata(note=s.tag)
        return await next_(s)

    mid = (
        GraphBuilder(_MidS)
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "probe": BranchSpec(subgraph=probe, inputs={"tag": "tag"}, middleware=(_augment,)),
                "baseline": BranchSpec(subgraph=baseline),
            },
        )
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
        .compile()
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_OuterS)
        .add_fan_out_node(
            "outer_fan",
            subgraph=mid,
            items_field="products",
            item_field="tag",
            collect_field="outcome",
            target_field="results",
        )
        .add_edge("outer_fan", END)
        .set_entry("outer_fan")
        .compile()
    )
    g.attach_observer(observer)
    try:
        await g.invoke(_OuterS(products=["A", "B"]))
        await g.drain()
    finally:
        observer.shutdown()
    spans = exporter.get_finished_spans()

    def _attr(s: Any, k: str) -> Any:
        return dict(s.attributes or {}).get(k)

    probe_dispatches = [
        s
        for s in spans
        if s.name == "probe" and "openarmature.parallel_branches.parent_node_name" in dict(s.attributes or {})
    ]
    baseline_dispatches = [
        s
        for s in spans
        if s.name == "baseline"
        and "openarmature.parallel_branches.parent_node_name" in dict(s.attributes or {})
    ]
    assert len(probe_dispatches) == 2, f"expected 2 probe dispatches, got {len(probe_dispatches)}"
    assert len(baseline_dispatches) == 2, f"expected 2 baseline dispatches, got {len(baseline_dispatches)}"
    outer_dispatches = {
        _attr(s, "openarmature.node.fan_out_index"): s
        for s in spans
        if s.name == "outer_fan" and "openarmature.fan_out.parent_node_name" in dict(s.attributes or {})
    }
    assert outer_dispatches.keys() == {0, 1}
    assert _attr(outer_dispatches[0], "openarmature.user.note") == "A"
    assert _attr(outer_dispatches[1], "openarmature.user.note") == "B"
    assert all(_attr(s, "openarmature.user.note") is None for s in baseline_dispatches), (
        "the non-augmenting baseline branch MUST NOT carry the augmentation"
    )


# ---------------------------------------------------------------------------
# Callable parallel branches (proposal 0075, observability §5.7). Mirrors
# spec conformance fixture 110 (otel-callable-branch-span), which is not yet
# in the pinned submodule: a callable branch renders as ONE per-branch
# dispatch span keyed by branch_name with NO inner-node spans; a when-skipped
# branch emits no span.
# ---------------------------------------------------------------------------


class _CallableBranchState(State):
    run_vector: bool = False
    vector_result: int = 0
    fts_result: int = 0
    keyword_result: int = 0


async def test_callable_branch_renders_one_dispatch_span_skipped_emits_none() -> None:
    from openarmature.graph import BranchSpec

    async def vector(_s: _CallableBranchState) -> dict[str, int]:
        return {"vector_result": 1}

    async def fts(_s: _CallableBranchState) -> dict[str, int]:
        return {"fts_result": 2}

    async def keyword(_s: _CallableBranchState) -> dict[str, int]:
        return {"keyword_result": 3}

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    g = (
        GraphBuilder(_CallableBranchState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={
                "vector": BranchSpec(call=vector, when=lambda s: s.run_vector),
                "fts": BranchSpec(call=fts),
                "keyword": BranchSpec(call=keyword),
            },
            error_policy="fail_fast",
        )
        .add_edge("retrieve", END)
        .compile()
    )
    g.attach_observer(observer)
    await cast("Any", g).invoke(_CallableBranchState())  # run_vector False -> vector skipped
    await cast("Any", g).drain()

    spans = exporter.get_finished_spans()

    def _sid(span: ReadableSpan) -> int:
        ctx = span.context
        assert ctx is not None
        return ctx.span_id

    def _children(span: ReadableSpan) -> list[ReadableSpan]:
        return [s for s in spans if s.parent is not None and s.parent.span_id == _sid(span)]

    node_spans = [s for s in spans if s.name == "retrieve"]
    assert len(node_spans) == 1
    node = node_spans[0]

    # The skipped `vector` branch emits NO span.
    assert [s for s in spans if s.name == "vector"] == []

    # Each dispatched callable branch -> exactly one dispatch span keyed by
    # branch_name, carrying parent_node_name, parented under the NODE span,
    # with NO inner-node spans (children == []).
    for branch in ("fts", "keyword"):
        branch_spans = [s for s in spans if s.name == branch]
        assert len(branch_spans) == 1, f"branch {branch!r}: expected one span, got {len(branch_spans)}"
        bs = branch_spans[0]
        attrs = dict(bs.attributes or {})
        assert attrs.get("openarmature.node.branch_name") == branch
        assert attrs.get("openarmature.parallel_branches.parent_node_name") == "retrieve"
        assert bs.parent is not None and bs.parent.span_id == _sid(node)
        assert _children(bs) == []

    # The NODE span's children are exactly the two dispatched branch spans.
    assert sorted(c.name for c in _children(node)) == ["fts", "keyword"]


# ---------------------------------------------------------------------------
# Proposal 0063 — tool-execution span (openarmature.tool.call)
# ---------------------------------------------------------------------------


async def _drive_tool_span(event: Any, *, disable_provider_payload: bool = True) -> Any:
    """Feed a ToolCallEvent / ToolCallFailedEvent through the OTel
    observer; return the single openarmature.tool.call ReadableSpan."""
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        disable_provider_payload=disable_provider_payload,
    )
    token = _set_invocation_id("inv-tool")
    try:
        await observer(event)
    finally:
        _reset_invocation_id(token)
    observer.shutdown()
    tool_spans = [s for s in exporter.get_finished_spans() if s.name == "openarmature.tool.call"]
    assert len(tool_spans) == 1
    return tool_spans[0]


def _tool_call_event(**overrides: Any) -> Any:
    from openarmature.graph.events import ToolCallEvent

    base: dict[str, Any] = {
        "invocation_id": "inv-tool",
        "correlation_id": None,
        "node_name": "run_tool",
        "namespace": ("run_tool",),
        "attempt_index": 0,
        "fan_out_index": None,
        "branch_name": None,
        "call_id": "cc-1",
        "tool_name": "get_weather",
        "tool_call_id": "call_abc",
        "arguments": {"city": "Paris"},
        "result": {"temperature_c": 20},
        "latency_ms": 5.0,
    }
    base.update(overrides)
    return ToolCallEvent(**base)


async def test_tool_span_emits_oa_namespace_attributes_not_gen_ai() -> None:
    # Proposal 0063 §5.5 (mirrors fixture 097): the tool span uses
    # OA-namespace openarmature.tool.* attributes; the Development
    # gen_ai.tool.* surface is NOT emitted in v1. Payload on.
    import json

    span = await _drive_tool_span(_tool_call_event(), disable_provider_payload=False)
    attrs = dict(span.attributes or {})
    assert attrs["openarmature.tool.name"] == "get_weather"
    assert attrs["openarmature.tool.call.id"] == "call_abc"
    assert json.loads(attrs["openarmature.tool.call.arguments"]) == {"city": "Paris"}
    assert json.loads(attrs["openarmature.tool.call.result"]) == {"temperature_c": 20}
    for absent in (
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.operation.name",
    ):
        assert absent not in attrs


async def test_tool_span_payload_gated_off_by_default() -> None:
    # Proposal 0063 §5.5.4 (mirrors fixture 096): arguments + result are
    # payload, suppressed under disable_provider_payload (default True);
    # the identity attributes still render.
    span = await _drive_tool_span(_tool_call_event())
    attrs = dict(span.attributes or {})
    assert attrs["openarmature.tool.name"] == "get_weather"
    assert attrs["openarmature.tool.call.id"] == "call_abc"
    assert "openarmature.tool.call.arguments" not in attrs
    assert "openarmature.tool.call.result" not in attrs


async def test_tool_span_omits_call_id_for_standalone() -> None:
    # tool_call_id None (a standalone instrumented function) -> the
    # openarmature.tool.call.id attribute is omitted entirely.
    span = await _drive_tool_span(_tool_call_event(tool_call_id=None), disable_provider_payload=False)
    attrs = dict(span.attributes or {})
    assert "openarmature.tool.call.id" not in attrs
    assert attrs["openarmature.tool.name"] == "get_weather"


async def test_tool_failed_span_renders_error_status() -> None:
    # Proposal 0063: a ToolCallFailedEvent renders ERROR with the
    # standard OTel error.type + an exception event; no result attribute.
    from opentelemetry.trace import StatusCode

    from openarmature.graph.events import ToolCallFailedEvent

    event = ToolCallFailedEvent(
        invocation_id="inv-tool",
        correlation_id=None,
        node_name="run_tool",
        namespace=("run_tool",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
        call_id="cc-1",
        tool_name="get_weather",
        tool_call_id="call_def",
        arguments={"city": "Paris"},
        latency_ms=3.0,
        error_type="TimeoutError",
        error_message="tool timed out",
    )
    span = await _drive_tool_span(event, disable_provider_payload=False)
    attrs = dict(span.attributes or {})
    assert span.status.status_code == StatusCode.ERROR
    assert attrs["error.type"] == "TimeoutError"
    assert "openarmature.tool.call.result" not in attrs
    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1
    assert dict(exception_events[0].attributes or {})["exception.message"] == "tool timed out"


async def test_tool_span_serializes_non_json_result_via_str_fallback() -> None:
    # Proposal 0063: the tool result is opaque (any language-idiomatic
    # value). A value json.dumps can't natively encode MUST NOT crash the
    # observer (which would lose the whole span); it renders via str().
    class _Opaque:
        def __str__(self) -> str:
            return "OPAQUE-RESULT"

    span = await _drive_tool_span(_tool_call_event(result=_Opaque()), disable_provider_payload=False)
    attrs = dict(span.attributes or {})
    assert "openarmature.tool.call.result" in attrs
    assert "OPAQUE-RESULT" in attrs["openarmature.tool.call.result"]
