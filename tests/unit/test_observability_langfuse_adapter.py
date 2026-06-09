"""Unit + integration tests for LangfuseSDKAdapter against langfuse>=4.6.

The unit test instantiates a real ``langfuse.Langfuse`` client with
dummy credentials and verifies the adapter satisfies the
:class:`LangfuseClient` Protocol via runtime ``isinstance`` — no
network calls. Skipped when the ``[langfuse]`` extra isn't installed.

The integration test, gated by ``@pytest.mark.integration`` plus
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` env vars, runs a
small graph end-to-end against real Langfuse Cloud. Use::

    LANGFUSE_PUBLIC_KEY=pk-lf-... \\
    LANGFUSE_SECRET_KEY=sk-lf-... \\
    LANGFUSE_BASE_URL=https://cloud.langfuse.com \\
        uv run pytest tests/unit/test_observability_langfuse_adapter.py \\
        -m integration -v

CI does NOT run integration tests; they're opt-in for local
verification.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import pytest

# Skip the whole module if langfuse isn't installed (extras not present).
pytest.importorskip("langfuse")

from langfuse import Langfuse  # noqa: E402

from openarmature.graph import END, GraphBuilder, State, append  # noqa: E402
from openarmature.observability.langfuse import (  # noqa: E402
    LangfuseClient,
    LangfuseObserver,
    LangfuseSDKAdapter,
)


def _dummy_client() -> Langfuse:
    # langfuse 4.x's Langfuse() constructor accepts credentials via env
    # vars or kwargs. Dummy keys bypass auth_check (which is called
    # opportunistically) — the adapter only needs the methods present
    # on the constructed instance, not a working API connection.
    return Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        host="http://localhost:0",  # unreachable; we don't make calls in unit tests
    )


def test_adapter_satisfies_langfuse_client_protocol() -> None:
    # Structural typing: the adapter MUST satisfy LangfuseClient at
    # runtime so LangfuseObserver accepts it. This is the load-bearing
    # test for the [langfuse] extras pin — if a future SDK release
    # breaks the Protocol's surface, this fails loudly.
    adapter = LangfuseSDKAdapter(_dummy_client())
    assert isinstance(adapter, LangfuseClient)


def test_adapter_observer_construction() -> None:
    # End-to-end: the observer accepts the adapter as its client
    # (Protocol satisfaction proves out at instantiation time under
    # the LangfuseClient annotation).
    adapter = LangfuseSDKAdapter(_dummy_client())
    observer = LangfuseObserver(client=adapter)
    assert observer.client is adapter


def test_adapter_caches_trace_info() -> None:
    # The trace() call doesn't hit the SDK; it caches info that
    # propagate_attributes applies on every observation under that
    # trace_id (not just the first — v4's last-wins display logic
    # would otherwise let later observations clobber the trace name).
    adapter = LangfuseSDKAdapter(_dummy_client())
    adapter.trace(id="trace-1", name="my-trace", metadata={"correlation_id": "c-1"})

    assert "trace-1" in adapter._trace_info  # noqa: SLF001
    cached = adapter._trace_info["trace-1"]  # noqa: SLF001
    assert cached["name"] == "my-trace"
    # "trace-1" is a non-UUID id, so the raw id is also surfaced under
    # metadata.invocation_id (proposal 0039 / §8.4.1).
    assert cached["metadata"] == {"correlation_id": "c-1", "invocation_id": "trace-1"}


def test_adapter_converts_uuid_trace_id_to_otel_hex() -> None:
    # Langfuse v4 expects OTel-format trace IDs (32-char lowercase
    # hex, no dashes). OA's invocation_id is a UUID4 with dashes.
    # The adapter MUST convert before passing to TraceContext, or
    # the SDK fails with ValueError("invalid literal for int() with
    # base 16: 'uuid-with-dashes'") at the OTel-attribute layer
    # — which OA's observer-error-isolation pattern swallows as a
    # warnings.warn, leaving the trace invisibly broken.
    from openarmature.observability.langfuse.adapter import _to_otel_trace_id

    assert _to_otel_trace_id("b24eda93-d06d-4eaa-9891-ca5e56f35722") == "b24eda93d06d4eaa9891ca5e56f35722"
    # Idempotent on already-hex input.
    assert _to_otel_trace_id("b24eda93d06d4eaa9891ca5e56f35722") == "b24eda93d06d4eaa9891ca5e56f35722"


def test_to_otel_trace_id_non_uuid_derivation() -> None:
    import hashlib

    from openarmature.observability.langfuse import langfuse_trace_id
    from openarmature.observability.langfuse.adapter import _to_otel_trace_id

    # Non-UUID -> first 16 bytes of SHA-256 as 32 hex (== Langfuse's
    # create_trace_id(seed)); the public helper is the same mapping.
    expected = hashlib.sha256(b"run_abc123").digest()[:16].hex()
    assert _to_otel_trace_id("run_abc123") == expected
    assert langfuse_trace_id("run_abc123") == expected
    # Spec fixture 036 pins this vector.
    assert expected == "29b50a6c08dabfeaeb1696301f4fabe1"
    # UUID path still strips dashes; the helper agrees.
    assert langfuse_trace_id("b24eda93-d06d-4eaa-9891-ca5e56f35722") == "b24eda93d06d4eaa9891ca5e56f35722"


def test_adapter_trace_surfaces_raw_id_for_non_uuid() -> None:
    adapter = LangfuseSDKAdapter(_dummy_client())
    # Non-UUID id: raw id surfaced under metadata.invocation_id (§8.4.1).
    adapter.trace(id="run_abc123", name="t", metadata=None)
    assert adapter._trace_info["run_abc123"]["metadata"]["invocation_id"] == "run_abc123"  # noqa: SLF001
    # UUID id: no invocation_id injected (its trace.id is reversible).
    uid = "b24eda93-d06d-4eaa-9891-ca5e56f35722"
    adapter.trace(id=uid, name="t", metadata=None)
    assert "invocation_id" not in adapter._trace_info[uid]["metadata"]  # noqa: SLF001


def test_adapter_update_trace_merges_into_cache() -> None:
    # update_trace merges into the cache so subsequent observations
    # under this trace_id pick up the new values via propagate_attributes.
    adapter = LangfuseSDKAdapter(_dummy_client())
    adapter.trace(id="trace-1", name="initial", metadata={"key1": "v1"})
    adapter.update_trace(id="trace-1", name="renamed", metadata={"key2": "v2"})

    cached = adapter._trace_info["trace-1"]  # noqa: SLF001
    assert cached["name"] == "renamed"
    # "trace-1" is non-UUID, so trace() also surfaced metadata.invocation_id.
    assert cached["metadata"] == {"key1": "v1", "key2": "v2", "invocation_id": "trace-1"}


def test_adapter_force_flush_delegates_to_client() -> None:
    # force_flush() must invoke the wrapped SDK's flush() so callers
    # in fast-teardown harnesses get the SDK's internal drain
    # (OTel TracerProvider.force_flush + score/media queue joins).
    # Mock(wraps=...) lets us assert delegation without simulating
    # the SDK's full surface.
    from unittest.mock import Mock

    real_client = _dummy_client()
    wrapped = Mock(wraps=real_client)
    adapter = LangfuseSDKAdapter(wrapped)

    assert adapter.force_flush() is True
    wrapped.flush.assert_called_once_with()

    # timeout_ms is accepted but unused per the documented contract.
    assert adapter.force_flush(timeout_ms=1_000) is True
    assert wrapped.flush.call_count == 2


def test_adapter_generation_routes_back_dated_calls_via_otel_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    # v4.7's public ``Langfuse.start_observation`` does NOT accept
    # ``start_time`` — only the internal ``_otel_tracer.start_span``
    # does. The adapter MUST route back-dated generation() calls via
    # the OTel tracer path (mirroring the SDK's own ``create_event``
    # precedent). This test spies on BOTH paths: ``start_observation``
    # is patched to fail loudly if the back-dated path ever falls
    # through to it (the prior monkeypatch test's gap), and the OTel
    # tracer's ``start_span`` is spied to assert the back-dated
    # nanosecond timestamp lands on the right surface.
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    client = _dummy_client()
    captured_otel_kwargs: dict[str, Any] = {}

    def _otel_spy(**kwargs: Any) -> MagicMock:
        captured_otel_kwargs.update(kwargs)
        # The SDK calls get_span_context(), set_attribute(), etc. on
        # the returned span during LangfuseGeneration construction.
        # Plain MagicMock auto-creates most attrs, but trace_id /
        # span_id MUST be real ints because the SDK formats them as
        # 32 / 16-char hex internally.
        span = MagicMock()
        span.get_span_context.return_value = MagicMock(
            trace_id=int("a" * 32, 16),
            span_id=int("b" * 16, 16),
        )
        return span

    def _start_observation_should_not_be_called(**_kwargs: Any) -> None:
        raise AssertionError(
            "start_observation MUST NOT be called on the back-dated path; "
            "v4 SDK rejects start_time= and the adapter should route via _otel_tracer"
        )

    monkeypatch.setattr(client._otel_tracer, "start_span", _otel_spy)  # noqa: SLF001
    monkeypatch.setattr(client, "start_observation", _start_observation_should_not_be_called)
    adapter = LangfuseSDKAdapter(client)
    adapter.trace(id="trace-ts", name="t")

    start = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
    adapter.generation(trace_id="trace-ts", name="g", model="m", start_time=start)

    expected_ns = int(start.timestamp() * 1_000_000_000)
    assert captured_otel_kwargs.get("start_time") == expected_ns
    assert captured_otel_kwargs.get("name") == "g"


def test_adapter_generation_without_start_time_uses_public_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # Companion to the back-dated test: when ``start_time`` is NOT
    # supplied, the adapter falls back to the v4 SDK's public
    # ``start_observation`` API and does NOT touch the private OTel
    # tracer.
    from unittest.mock import MagicMock

    client = _dummy_client()
    captured_kwargs: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> MagicMock:
        captured_kwargs.update(kwargs)
        return MagicMock(id="obs-spy", end=MagicMock())

    def _otel_tracer_should_not_be_called(**_kwargs: Any) -> None:
        raise AssertionError(
            "_otel_tracer.start_span MUST NOT be called when start_time is None; "
            "the public start_observation API should handle this path"
        )

    monkeypatch.setattr(client, "start_observation", _spy)
    monkeypatch.setattr(client._otel_tracer, "start_span", _otel_tracer_should_not_be_called)  # noqa: SLF001
    adapter = LangfuseSDKAdapter(client)
    adapter.trace(id="trace-ts", name="t")

    adapter.generation(trace_id="trace-ts", name="g", model="m")

    assert captured_kwargs.get("name") == "g"
    assert "start_time" not in captured_kwargs


def test_adapter_generation_handle_end_converts_end_time_to_nanoseconds() -> None:
    # Companion to the start_time test: the handle's end() MUST
    # convert the datetime to int nanoseconds before forwarding to
    # the underlying v4 obs's end(). LangfuseSpan.end is typed
    # ``Optional[int]`` (nanoseconds); passing a datetime through
    # crashes the OTel span_processor's formatter with TypeError on
    # ``end_time / 1e9`` deep in the SDK.
    from datetime import UTC, datetime
    from unittest.mock import MagicMock

    from openarmature.observability.langfuse.adapter import _SpanHandle

    sdk_obs = MagicMock(id="obs-e")
    sdk_obs.end = MagicMock()
    handle = _SpanHandle(sdk_obs)

    end = datetime(2026, 6, 8, 12, 0, 1, tzinfo=UTC)
    handle.end(end_time=end)

    expected_ns = int(end.timestamp() * 1_000_000_000)
    sdk_obs.end.assert_called_once_with(end_time=expected_ns)


def test_adapter_generation_handle_end_omits_end_time_when_unspecified() -> None:
    # When no end_time is supplied, the handle MUST call the SDK obs's
    # end() without the kwarg so the SDK uses its default
    # (call-time). Locks the "default-respecting" branch.
    from unittest.mock import MagicMock

    from openarmature.observability.langfuse.adapter import _SpanHandle

    sdk_obs = MagicMock(id="obs-e")
    sdk_obs.end = MagicMock()
    handle = _SpanHandle(sdk_obs)

    handle.end()

    sdk_obs.end.assert_called_once_with()


# ---------------------------------------------------------------------------
# Integration test against real Langfuse Cloud (opt-in)
# ---------------------------------------------------------------------------


class _S(State):
    trail: Annotated[list[str], append] = []


async def _node(name: str) -> Any:
    return {"trail": [name]}


@pytest.mark.integration
async def test_adapter_against_real_langfuse_cloud() -> None:
    # Validates that the adapter actually exchanges data with Langfuse
    # Cloud — instantiates the real SDK, runs a tiny graph through
    # LangfuseObserver, calls flush(). No assertions on the
    # remote-side ingest (which is async). Manually verify via the
    # Langfuse dashboard that the trace appears with the expected
    # observation tree.
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")

    # Mirror the SDK's precedence: Langfuse() reads LANGFUSE_BASE_URL
    # first, then LANGFUSE_HOST. Resolve the same order here so this
    # explicit host matches what a no-arg Langfuse() would pick up.
    host = (
        os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    )
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    # Fail loudly on bad credentials. Without this, a 401 from the
    # background export thread is just a logged warning and the test
    # passes while traces vanish.
    assert client.auth_check(), (
        "Langfuse auth_check failed — verify LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL"
    )

    observer = LangfuseObserver(client=LangfuseSDKAdapter(client))

    graph = (
        GraphBuilder(_S)
        .add_node("step_a", lambda _s: _node("step_a"))
        .add_node("step_b", lambda _s: _node("step_b"))
        .add_edge("step_a", "step_b")
        .add_edge("step_b", END)
        .set_entry("step_a")
        .compile()
    )
    graph.attach_observer(observer)
    await graph.invoke(_S())
    await graph.drain()
    observer.shutdown()
    # Use ``client.shutdown()`` rather than ``client.flush()`` here:
    # both block on the SDK's internal drain (OTel's force_flush plus
    # the score/media queue joins), but shutdown() also tears down
    # the resource manager so the test process exits cleanly. flush()
    # is the appropriate call from a long-lived process that wants
    # to drain without releasing SDK resources.
    client.shutdown()
    # Manual check: open the trace in the dashboard and confirm
    # "step_a" + "step_b" appear as Span observations under one Trace.
    # The trace_id in the dashboard is the 32-char hex form (no dashes)
    # of OA's UUID4 invocation_id; strip dashes from any logged
    # correlation_id / invocation_id to find it.
