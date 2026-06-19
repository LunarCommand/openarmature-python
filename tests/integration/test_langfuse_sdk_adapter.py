"""Integration tests for LangfuseSDKAdapter against the live Langfuse
test account.

Gated by the presence of ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``
env vars. Skipped in CI and local runs that don't have credentials in
scope; runs end-to-end against Langfuse Cloud when invoked from a
shell with credentials (the documented test-account env vars per
[[reference_langfuse_test_account.md]]).

Each test polls the REST API after ``flush()`` with retries to absorb
the eventual-consistency lag between ingestion and the REST projection.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest

# Skip the entire module when credentials aren't sourced. Avoids a
# cryptic ``ImportError`` / ``ValueError`` cascade from the SDK when
# the test environment is bare.
pytestmark = pytest.mark.skipif(
    not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")),
    reason="Requires LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY (live Langfuse account)",
)


def _poll_trace_with_retry(client: Any, hex_id: str, *, attempts: int = 12, sleep_s: float = 5.0) -> Any:
    """Poll Langfuse's REST API until the trace appears or the budget
    runs out. The Langfuse list-view UI updates faster than the REST
    GET projection, so a freshly-flushed trace can 404 for ~30-60s.
    Linear backoff is fine; the API is rate-limited gently and the
    test runs once per CI invocation."""
    from langfuse.api import NotFoundError

    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return client.api.trace.get(hex_id)
        except NotFoundError as exc:
            last_exc = exc
            time.sleep(sleep_s)
    raise AssertionError(
        f"Trace {hex_id} did not appear in REST API after {attempts * sleep_s:.0f}s; last error: {last_exc!r}"
    )


@pytest.mark.integration
async def test_sdk_adapter_emits_trace_input_output_to_live_langfuse() -> None:
    """End-to-end: open a trace via LangfuseSDKAdapter, push input via
    update_trace at the start, push output via update_trace at the end,
    flush, and confirm both fields populate on the live Trace entity."""
    from langfuse import Langfuse

    from openarmature.observability.langfuse.adapter import LangfuseSDKAdapter

    client = Langfuse()
    adapter = LangfuseSDKAdapter(client)

    invocation_id = str(uuid.uuid4())
    expected_input = {"entry_node": "verify_entry", "correlation_id": "test-corr-1"}
    expected_output = {"final_node": "verify_entry", "status": "completed"}

    # Simulate the LangfuseObserver call sequence: trace open →
    # InvocationStartedEvent (update_trace with input) → first node
    # span → node ends → InvocationCompletedEvent (update_trace with
    # output).
    adapter.trace(
        id=invocation_id,
        name="test_sdk_adapter_trace_io",
        metadata={"test_run": "trace_io_emit"},
    )
    adapter.update_trace(id=invocation_id, input=expected_input)
    # Open + close a real observation so the cached pending_input has
    # something to piggyback on.
    span_handle = adapter.span(trace_id=invocation_id, name="verify_entry")
    span_handle.end()
    adapter.update_trace(id=invocation_id, output=expected_output)

    adapter.force_flush()
    # Brief settle for the UI projection; REST poll handles the
    # longer-tail consistency window separately.
    time.sleep(2)

    hex_id = invocation_id.replace("-", "")
    trace = _poll_trace_with_retry(client, hex_id)

    # `trace.input` / `trace.output` are the headline columns proposal
    # 0043 motivates; assert both ingested correctly. These are the
    # spec-compliance signal — they project off OTel attributes on
    # incoming spans and populate on the Trace entity directly.
    assert trace.input == expected_input, f"trace.input mismatch: got {trace.input!r}"
    assert trace.output == expected_output, f"trace.output mismatch: got {trace.output!r}"
    # Note: we deliberately do NOT assert on ``trace.observations``
    # here. Langfuse's REST projection for the observations list
    # lags the Trace's headline fields by an indeterminate window —
    # the two tests in this module hit different consistency points
    # against the same backend. The synthetic ``openarmature.trace_io``
    # observation is verified in the next test (which uses ONLY the
    # synthetic carrier and reliably shows in the list).


@pytest.mark.integration
async def test_sdk_adapter_handles_invocation_with_no_real_observation() -> None:
    """Edge case: invocation fails before any node observation opens
    (resume-path validation failure, etc.). The cached pending_input
    has no real span to piggyback on, so the synthetic output
    observation becomes the sole carrier for BOTH fields."""
    from langfuse import Langfuse

    from openarmature.observability.langfuse.adapter import LangfuseSDKAdapter

    client = Langfuse()
    adapter = LangfuseSDKAdapter(client)

    invocation_id = str(uuid.uuid4())
    expected_input = {"entry_node": "fail_fast", "correlation_id": "test-corr-2"}
    expected_output = {"final_node": "fail_fast", "status": "failed"}

    adapter.trace(id=invocation_id, name="test_sdk_adapter_no_real_span")
    adapter.update_trace(id=invocation_id, input=expected_input)
    # NO real span opens — straight to the output update.
    adapter.update_trace(id=invocation_id, output=expected_output)

    adapter.force_flush()
    time.sleep(2)

    hex_id = invocation_id.replace("-", "")
    trace = _poll_trace_with_retry(client, hex_id)

    # Both input and output land on the Trace even with the synthetic
    # observation as the sole span.
    assert trace.input == expected_input
    assert trace.output == expected_output
    assert len(trace.observations) == 1
    assert trace.observations[0].name == "openarmature.trace_io"


@pytest.mark.integration
async def test_sdk_adapter_generation_timestamps_round_trip_through_langfuse() -> None:
    """End-to-end verification that explicit start_time / end_time on
    the adapter's generation(...) / handle.end(...) calls actually
    land on the Langfuse-side observation. The unit tests cover the
    SDK call-site shape (the back-dated path bypasses the public
    start_observation API and routes through the private
    _otel_tracer.start_span instead, because v4.7's start_observation
    rejects start_time with TypeError); this test closes the loop by
    reading the projected timestamps back from the REST API and
    asserting they reflect the back-dated values.

    Catches the failure mode the Langfuse migration is susceptible to:
    if a future SDK release renames _otel_tracer,
    moves LangfuseGeneration, or otherwise breaks the private-API
    surface the adapter relies on, the back-dating routing fails
    silently — the Langfuse UI shows call-time timestamps instead
    of the back-dated latency_ms-based ones, with no error to
    surface the misconfiguration.
    """
    from datetime import UTC, datetime, timedelta

    from langfuse import Langfuse

    from openarmature.observability.langfuse.adapter import LangfuseSDKAdapter

    client = Langfuse()
    adapter = LangfuseSDKAdapter(client)

    invocation_id = str(uuid.uuid4())
    adapter.trace(id=invocation_id, name="test_sdk_adapter_generation_timestamps")
    # Back-date by 250 ms — a value far enough above the SDK's own
    # call-time jitter that a passthrough failure would show up
    # clearly in the projected start/end timestamps.
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(milliseconds=250)
    handle = adapter.generation(
        trace_id=invocation_id,
        name="openarmature.llm.complete",
        model="test-model",
        start_time=start_time,
    )
    handle.end(end_time=end_time)

    adapter.force_flush()
    time.sleep(2)

    hex_id = invocation_id.replace("-", "")
    try:
        _poll_trace_with_retry(client, hex_id)
        # Pull observations via REST. The observations-list endpoint
        # lags the headline trace fields by an "indeterminate window"
        # per Langfuse's own caveat (mirrored in the trace_io test
        # above), so the retry budget here is wider (180s vs the
        # trace_io test's 60s). Filter server-side by name + type so
        # the response is small + scoped; track seen names across
        # polls so a name-mismatch failure surfaces with diagnostics
        # rather than a generic "not found".
        observation: Any = None
        last_exc: Exception | None = None
        seen_names: set[str] = set()
        for _ in range(36):
            try:
                response = client.api.observations.get_many(
                    trace_id=hex_id,
                    name="openarmature.llm.complete",
                    type="GENERATION",
                )
                if response.data:
                    observation = response.data[0]
                    break
                # Fall back to an unfiltered query so we know whether
                # ANY observations have projected — distinguishes "REST
                # lag" from "observation projected with unexpected name".
                fallback = client.api.observations.get_many(trace_id=hex_id)
                seen_names.update(o.name or "<unnamed>" for o in fallback.data)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(5)
        assert observation is not None, (
            f"openarmature.llm.complete Generation for trace {hex_id} did not appear via REST "
            f"after 180s retry budget. Observations seen under trace (any name): {seen_names or 'none'}. "
            f"Last error: {last_exc!r}"
        )

        # The REST projection's start/end timestamps MUST match the
        # back-dated values within a small tolerance (the SDK rounds
        # to microseconds; Langfuse's REST projection may round
        # further).
        assert observation.start_time is not None
        assert observation.end_time is not None
        start_delta = abs((observation.start_time - start_time).total_seconds())
        end_delta = abs((observation.end_time - end_time).total_seconds())
        assert start_delta < 0.01, (
            f"observation.start_time drift {start_delta * 1000:.3f}ms exceeds 10ms tolerance — "
            f"sent {start_time.isoformat()}, got {observation.start_time.isoformat()}"
        )
        assert end_delta < 0.01, (
            f"observation.end_time drift {end_delta * 1000:.3f}ms exceeds 10ms tolerance — "
            f"sent {end_time.isoformat()}, got {observation.end_time.isoformat()}"
        )
    finally:
        # Match the existing module's clean-exit pattern: shutdown
        # releases the SDK's background OTel exporter + ingestion
        # queues. Without this, a long-running pytest process could
        # accumulate background threads across integration tests.
        client.shutdown()


@pytest.mark.integration
async def test_sdk_adapter_populates_session_and_user_id_on_live_langfuse() -> None:
    """End-to-end (proposal 0064 §8.4.1): trace(session_id=, user_id=)
    populates the live Trace's sessionId / userId grouping fields.

    The observer leaves session_id dormant until the sessions capability
    (proposal 0020) supplies openarmature.session_id, but the adapter
    passes whatever it is given, so this exercises BOTH passthroughs at
    the SDK boundary: when 0020 lands, the session_id rides the same
    propagate_attributes path the userId promotion uses today.
    """
    from langfuse import Langfuse

    from openarmature.observability.langfuse.adapter import LangfuseSDKAdapter

    client = Langfuse()
    adapter = LangfuseSDKAdapter(client)

    invocation_id = str(uuid.uuid4())
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"

    # session_id / user_id ride on the observations under the trace via
    # propagate_attributes (the same carrier as name / metadata), so open
    # one real observation for them to attach to.
    adapter.trace(
        id=invocation_id,
        name="test_sdk_adapter_session_user",
        metadata={"userId": user_id},
        session_id=session_id,
        user_id=user_id,
    )
    span_handle = adapter.span(trace_id=invocation_id, name="verify_entry")
    span_handle.end()

    adapter.force_flush()
    time.sleep(2)

    hex_id = invocation_id.replace("-", "")
    try:
        trace = _poll_trace_with_retry(client, hex_id)
        assert trace.session_id == session_id, f"trace.session_id mismatch: got {trace.session_id!r}"
        assert trace.user_id == user_id, f"trace.user_id mismatch: got {trace.user_id!r}"
    finally:
        client.shutdown()
