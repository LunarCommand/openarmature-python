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
