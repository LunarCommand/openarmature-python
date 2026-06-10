"""Unit tests for the caller-supplied invocation surface: metadata
(proposal 0034), the caller-supplied invocation_id (proposal 0039),
and the reserved exact-key-name rejection (proposal 0041).

These tests pin the validation rules, the ContextVar lifecycle, the
mid-invocation augmentation helper, and the per-async-context COW
isolation (fan-out instance augmentation doesn't leak to siblings).
The conformance fixtures (026/027/028/029/030) cover end-to-end
observer emission against the spec's expected shapes; these unit
tests focus on the python-side surface contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from openarmature.graph import END, GraphBuilder, State
from openarmature.observability import (
    current_invocation_metadata,
    get_invocation_metadata,
    set_invocation_metadata,
)
from openarmature.observability.metadata import (
    validate_invocation_metadata,
)

# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def test_validate_accepts_simple_scalars() -> None:
    out = validate_invocation_metadata(
        {
            "tenantId": "acme-corp",
            "seatCount": 42,
            "ratio": 0.75,
            "isCanary": True,
        }
    )
    assert out["tenantId"] == "acme-corp"
    assert out["seatCount"] == 42
    assert out["ratio"] == 0.75
    assert out["isCanary"] is True


def test_validate_accepts_homogeneous_arrays() -> None:
    out = validate_invocation_metadata(
        {
            "labels": ["alpha", "beta"],
            "weights": [1, 2, 3],
            "scores": [0.1, 0.2],
            "flags": [True, False],
        }
    )
    assert out["labels"] == ["alpha", "beta"]
    assert out["weights"] == [1, 2, 3]
    assert out["scores"] == [0.1, 0.2]
    assert out["flags"] == [True, False]


def test_validate_none_returns_empty_mapping() -> None:
    out = validate_invocation_metadata(None)
    assert dict(out) == {}


def test_validate_rejects_openarmature_prefix() -> None:
    with pytest.raises(ValueError, match=r"reserved namespace prefix 'openarmature\.'"):
        validate_invocation_metadata({"openarmature.user.x": "y"})


def test_validate_rejects_gen_ai_prefix() -> None:
    with pytest.raises(ValueError, match=r"reserved namespace prefix 'gen_ai\.'"):
        validate_invocation_metadata({"gen_ai.system": "openai"})


def test_validate_rejects_non_string_key() -> None:
    with pytest.raises(ValueError, match="key must be a string"):
        validate_invocation_metadata({123: "v"})  # pyright: ignore[reportArgumentType]


def test_validate_rejects_none_value() -> None:
    with pytest.raises(ValueError, match="value type NoneType"):
        validate_invocation_metadata({"k": None})  # pyright: ignore[reportArgumentType]


def test_validate_rejects_nested_dict_value() -> None:
    with pytest.raises(ValueError, match="value type dict"):
        validate_invocation_metadata({"k": {"nested": "x"}})  # pyright: ignore[reportArgumentType]


def test_validate_rejects_mixed_type_arrays() -> None:
    with pytest.raises(ValueError, match="MUST be homogeneous"):
        validate_invocation_metadata({"mixed": [1, "two"]})  # pyright: ignore[reportArgumentType]


def test_validate_rejects_array_of_dicts() -> None:
    with pytest.raises(ValueError, match="unsupported type"):
        validate_invocation_metadata({"k": [{"a": 1}]})  # pyright: ignore[reportArgumentType]


def test_validate_accepts_empty_array() -> None:
    out = validate_invocation_metadata({"empty": []})
    assert out["empty"] == []


def test_validate_rejects_non_dict_mapping() -> None:
    with pytest.raises(ValueError, match="must be a dict"):
        validate_invocation_metadata("not a dict")  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# ContextVar reader outside any invocation
# ---------------------------------------------------------------------------


def test_current_invocation_metadata_empty_outside_invocation() -> None:
    # Outside any invocation, the reader returns an empty mapping —
    # not None — so callers can iterate without a guard.
    assert dict(current_invocation_metadata()) == {}


# ---------------------------------------------------------------------------
# set_invocation_metadata augmentation
# ---------------------------------------------------------------------------


def test_set_invocation_metadata_augments_existing() -> None:
    async def _runner() -> dict[str, Any]:
        # Simulate the engine setting initial metadata.
        from openarmature.observability.metadata import (
            _set_invocation_metadata,
            validate_invocation_metadata,
        )

        token = _set_invocation_metadata(validate_invocation_metadata({"tenantId": "acme"}))
        try:
            set_invocation_metadata(productId="p-1", batchId=42)
            return dict(current_invocation_metadata())
        finally:
            from openarmature.observability.metadata import _reset_invocation_metadata

            _reset_invocation_metadata(token)

    result = asyncio.run(_runner())
    # Augmentation merges with the initial mapping; nothing dropped.
    assert result == {"tenantId": "acme", "productId": "p-1", "batchId": 42}


def test_set_invocation_metadata_overwrites_existing_key() -> None:
    async def _runner() -> dict[str, Any]:
        from openarmature.observability.metadata import (
            _reset_invocation_metadata,
            _set_invocation_metadata,
            validate_invocation_metadata,
        )

        token = _set_invocation_metadata(validate_invocation_metadata({"phase": "draft"}))
        try:
            set_invocation_metadata(phase="final")
            return dict(current_invocation_metadata())
        finally:
            _reset_invocation_metadata(token)

    result = asyncio.run(_runner())
    assert result == {"phase": "final"}


def test_set_invocation_metadata_rejects_reserved_namespace() -> None:
    with pytest.raises(ValueError, match="reserved namespace prefix"):
        set_invocation_metadata(**{"openarmature.user.x": "y"})


def test_set_invocation_metadata_no_op_when_empty() -> None:
    # Calling with no entries does nothing (and doesn't error).
    set_invocation_metadata()
    assert dict(current_invocation_metadata()) == {}


# ---------------------------------------------------------------------------
# Engine integration: invoke(metadata=...) + boundary rejection
# ---------------------------------------------------------------------------


class _SimpleState(State):
    counter: int = 0


async def _noop_node(_s: _SimpleState) -> dict[str, Any]:
    return {"counter": 1}


def _build_graph() -> Any:
    return (
        GraphBuilder(_SimpleState)
        .add_node("noop", _noop_node)
        .add_edge("noop", END)
        .set_entry("noop")
        .compile()
    )


async def test_invoke_accepts_metadata() -> None:
    graph = _build_graph()
    await graph.invoke(_SimpleState(), metadata={"tenantId": "acme", "seatCount": 42})


async def test_invoke_rejects_reserved_namespace_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="reserved namespace prefix"):
        await graph.invoke(_SimpleState(), metadata={"openarmature.user.x": "y"})


async def test_invoke_rejects_bad_value_type_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="value type NoneType"):
        # pyright: ignore[reportArgumentType]
        await graph.invoke(_SimpleState(), metadata={"k": None})  # type: ignore[dict-item]


async def test_invoke_resets_metadata_after_return() -> None:
    graph = _build_graph()
    await graph.invoke(_SimpleState(), metadata={"tenantId": "acme"})
    # After the invocation returns, the ContextVar is reset to empty
    # so the next invocation gets a fresh slate.
    assert dict(current_invocation_metadata()) == {}


async def test_metadata_visible_inside_node_body() -> None:
    captured: dict[str, Any] = {}

    async def _capture(_s: _SimpleState) -> dict[str, Any]:
        captured.update(dict(current_invocation_metadata()))
        return {"counter": 1}

    graph = (
        GraphBuilder(_SimpleState)
        .add_node("capture", _capture)
        .add_edge("capture", END)
        .set_entry("capture")
        .compile()
    )
    await graph.invoke(_SimpleState(), metadata={"tenantId": "acme"})
    assert captured == {"tenantId": "acme"}


async def test_otel_observer_emits_user_metadata_on_every_span() -> None:
    # Per observability §5.6: caller-supplied entries appear as
    # `openarmature.user.<key>` cross-cutting attributes on every
    # span (invocation, node, LLM provider if present).
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from openarmature.observability.otel import OTelObserver

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    graph = _build_graph()
    graph.attach_observer(observer)
    try:
        await graph.invoke(_SimpleState(), metadata={"tenantId": "acme", "seatCount": 42})
        await graph.drain()
    finally:
        observer.shutdown()

    spans = exporter.get_finished_spans()
    assert len(spans) >= 2  # invocation span + noop node span
    for span in spans:
        attrs = dict(span.attributes or {})
        assert attrs.get("openarmature.user.tenantId") == "acme", (
            f"span {span.name!r} missing or wrong tenantId: {attrs}"
        )
        assert attrs.get("openarmature.user.seatCount") == 42, (
            f"span {span.name!r} missing or wrong seatCount: {attrs}"
        )


async def test_langfuse_observer_emits_user_metadata_on_trace_and_observations() -> None:
    # Per observability §8.4.1 + §8.4.2: caller-supplied entries
    # appear on `trace.metadata` AND on every `observation.metadata`
    # at the top level.
    from openarmature.observability.langfuse import (
        InMemoryLangfuseClient,
        LangfuseObserver,
    )

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    graph = _build_graph()
    graph.attach_observer(observer)
    await graph.invoke(_SimpleState(), metadata={"tenantId": "acme", "featureFlag": "v2"})
    await graph.drain()

    assert len(client.traces) == 1
    trace = next(iter(client.traces.values()))
    assert trace.metadata.get("tenantId") == "acme"
    assert trace.metadata.get("featureFlag") == "v2"
    # Every observation in the trace must also carry the entries.
    assert len(trace.observations) >= 1
    for obs in trace.observations:
        assert obs.metadata.get("tenantId") == "acme", (
            f"observation {obs.name!r} missing tenantId: {obs.metadata}"
        )
        assert obs.metadata.get("featureFlag") == "v2", (
            f"observation {obs.name!r} missing featureFlag: {obs.metadata}"
        )


async def test_mid_invocation_augmentation_persists_to_next_node() -> None:
    capture_a: dict[str, Any] = {}
    capture_b: dict[str, Any] = {}

    async def _a(_s: _SimpleState) -> dict[str, Any]:
        capture_a.update(dict(current_invocation_metadata()))
        set_invocation_metadata(stage="a-completed")
        return {"counter": 1}

    async def _b(_s: _SimpleState) -> dict[str, Any]:
        capture_b.update(dict(current_invocation_metadata()))
        return {"counter": 2}

    graph = (
        GraphBuilder(_SimpleState)
        .add_node("a", _a)
        .add_node("b", _b)
        .add_edge("a", "b")
        .add_edge("b", END)
        .set_entry("a")
        .compile()
    )
    await graph.invoke(_SimpleState(), metadata={"tenantId": "acme"})
    # Node a sees the initial metadata.
    assert capture_a == {"tenantId": "acme"}
    # Node b sees the initial metadata PLUS node a's augmentation.
    # Sequential nodes share the engine task's Context, so a's
    # set_invocation_metadata persists into b's body.
    assert capture_b == {"tenantId": "acme", "stage": "a-completed"}


# ---------------------------------------------------------------------------
# Reserved exact key names (proposal 0041)
# ---------------------------------------------------------------------------


def test_validate_rejects_reserved_exact_key_name() -> None:
    # An exact match to an OA-emitted top-level key is rejected at the
    # boundary, the same as the namespace-prefix reservation.
    with pytest.raises(ValueError, match="is reserved"):
        validate_invocation_metadata({"namespace": "x"})


def test_validate_rejects_invocation_id_key() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        validate_invocation_metadata({"invocation_id": "abc"})


def test_set_invocation_metadata_rejects_reserved_exact_key_name() -> None:
    # The SAME reservation fires at the mid-invocation boundary, since
    # both paths route through _validate_metadata_key.
    with pytest.raises(ValueError, match="is reserved"):
        set_invocation_metadata(correlation_id="c-2")


async def test_invoke_rejects_reserved_exact_key_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="is reserved"):
        await graph.invoke(_SimpleState(), metadata={"step": 3})


# ---------------------------------------------------------------------------
# Reserved exact key names extension (proposal 0042)
# ---------------------------------------------------------------------------


def test_validate_rejects_reserved_branch_name() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        validate_invocation_metadata({"branch_name": "fraud_check"})


def test_validate_rejects_reserved_detached() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        validate_invocation_metadata({"detached": True})


def test_validate_rejects_reserved_detached_from_invocation_id() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        validate_invocation_metadata({"detached_from_invocation_id": "parent-1"})


def test_set_invocation_metadata_rejects_reserved_branch_name() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        set_invocation_metadata(branch_name="policy_audit")


def test_set_invocation_metadata_rejects_reserved_detached() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        set_invocation_metadata(detached=True)


def test_set_invocation_metadata_rejects_reserved_detached_from_invocation_id() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        set_invocation_metadata(detached_from_invocation_id="parent-1")


async def test_invoke_rejects_reserved_branch_name_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="is reserved"):
        await graph.invoke(_SimpleState(), metadata={"branch_name": "x"})


async def test_invoke_rejects_reserved_detached_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="is reserved"):
        await graph.invoke(_SimpleState(), metadata={"detached": False})


async def test_invoke_rejects_reserved_detached_from_invocation_id_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="is reserved"):
        await graph.invoke(_SimpleState(), metadata={"detached_from_invocation_id": "p"})


# ---------------------------------------------------------------------------
# Reserved exact key names extension (proposal 0052) — implementation
# attribution attributes. The 24-name set grows to 26 with
# implementation_name + implementation_version reserved so a caller
# can't clobber the implementation-emitted Trace metadata / OTel
# attribute values by passing the same key in invoke(metadata=...).
# ---------------------------------------------------------------------------


def test_validate_rejects_reserved_implementation_name() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        validate_invocation_metadata({"implementation_name": "spoof"})


def test_validate_rejects_reserved_implementation_version() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        validate_invocation_metadata({"implementation_version": "9.9.9"})


def test_set_invocation_metadata_rejects_reserved_implementation_name() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        set_invocation_metadata(implementation_name="spoof")


def test_set_invocation_metadata_rejects_reserved_implementation_version() -> None:
    with pytest.raises(ValueError, match="is reserved"):
        set_invocation_metadata(implementation_version="9.9.9")


async def test_invoke_rejects_reserved_implementation_name_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="is reserved"):
        await graph.invoke(_SimpleState(), metadata={"implementation_name": "spoof"})


async def test_invoke_rejects_reserved_implementation_version_at_boundary() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="is reserved"):
        await graph.invoke(_SimpleState(), metadata={"implementation_version": "9.9.9"})


# ---------------------------------------------------------------------------
# Caller-supplied invocation_id (proposal 0039)
# ---------------------------------------------------------------------------


def test_validate_invocation_id_accepts_nanoid_and_uuid() -> None:
    from openarmature.observability.correlation import validate_invocation_id

    assert validate_invocation_id("V1StGXR8_Z5jdHi6B-myT") == "V1StGXR8_Z5jdHi6B-myT"
    uid = "b24eda93-d06d-4eaa-9891-ca5e56f35722"
    assert validate_invocation_id(uid) == uid


def test_validate_invocation_id_rejects_bad() -> None:
    from openarmature.observability.correlation import validate_invocation_id

    with pytest.raises(ValueError, match="non-empty"):
        validate_invocation_id("")
    with pytest.raises(ValueError, match="not URL-safe"):
        validate_invocation_id("has space")


async def test_invoke_uses_caller_invocation_id() -> None:
    from openarmature.observability.correlation import current_invocation_id

    captured: dict[str, Any] = {}

    async def _capture(_s: _SimpleState) -> dict[str, Any]:
        captured["id"] = current_invocation_id()
        return {"counter": 1}

    graph = (
        GraphBuilder(_SimpleState)
        .add_node("capture", _capture)
        .add_edge("capture", END)
        .set_entry("capture")
        .compile()
    )
    await graph.invoke(_SimpleState(), invocation_id="run_abc123")
    assert captured["id"] == "run_abc123"


async def test_invoke_mints_uuid_when_invocation_id_absent() -> None:
    import uuid

    from openarmature.observability.correlation import current_invocation_id

    captured: dict[str, Any] = {}

    async def _capture(_s: _SimpleState) -> dict[str, Any]:
        captured["id"] = current_invocation_id()
        return {"counter": 1}

    graph = (
        GraphBuilder(_SimpleState)
        .add_node("capture", _capture)
        .add_edge("capture", END)
        .set_entry("capture")
        .compile()
    )
    await graph.invoke(_SimpleState())
    # Auto-minted in the absence of a caller id: a parseable UUID.
    uuid.UUID(captured["id"])


async def test_invoke_rejects_non_url_safe_invocation_id() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="not URL-safe"):
        await graph.invoke(_SimpleState(), invocation_id="bad id!")


async def test_invoke_rejects_empty_invocation_id() -> None:
    graph = _build_graph()
    with pytest.raises(ValueError, match="non-empty"):
        await graph.invoke(_SimpleState(), invocation_id="")


# Proposal 0048: ``get_invocation_metadata`` is the canonical
# spec-idiomatic public name for the §3.4 read API, paralleling
# ``set_invocation_metadata`` on the write side. It is the same
# function object as the historical ``current_invocation_metadata``;
# the tests below pin the alias identity and the end-to-end roundtrip
# the spec calls out — boundary baseline + in-node augment, with the
# read returning an immutable mapping containing both.


def test_get_invocation_metadata_is_same_callable_as_current() -> None:
    assert get_invocation_metadata is current_invocation_metadata


def test_get_invocation_metadata_empty_outside_invocation() -> None:
    out = get_invocation_metadata()
    assert dict(out) == {}


def test_get_invocation_metadata_returns_immutable_mapping_outside_invocation() -> None:
    from types import MappingProxyType

    assert isinstance(get_invocation_metadata(), MappingProxyType)


async def test_get_invocation_metadata_roundtrip_baseline_plus_augment() -> None:
    from types import MappingProxyType

    captured: dict[str, Any] = {}
    captured_type: list[type] = []

    async def _read_after_write(_s: _SimpleState) -> dict[str, Any]:
        set_invocation_metadata(audit_kind="fraud")
        read = get_invocation_metadata()
        # Pin the immutable-mapping return type inside an active
        # invocation too — the outside-invocation path returns the
        # module-level ``_EMPTY_METADATA`` sentinel, but the
        # mid-invocation path constructs a fresh MappingProxyType
        # around the merged dict (see ``set_invocation_metadata``).
        captured_type.append(type(read))
        captured.update(dict(read))
        return {"counter": 1}

    graph = (
        GraphBuilder(_SimpleState)
        .add_node("read_after_write", _read_after_write)
        .add_edge("read_after_write", END)
        .set_entry("read_after_write")
        .compile()
    )
    await graph.invoke(_SimpleState(), metadata={"tenantId": "T1"})
    # Caller baseline + in-node write, both visible to the read.
    assert captured == {"tenantId": "T1", "audit_kind": "fraud"}
    assert captured_type == [MappingProxyType]


# Spec observability §3.4 *Per-attempt scoping*: under retry
# middleware, each attempt sees only the metadata in scope at
# retry-entry plus that attempt's own writes; failed-attempt
# writes are discarded along with the attempt itself. The pin
# below mirrors the spec's fixture 045 case shape (attempt 0
# writes + fails, attempt 1 asserts marker absent + writes +
# succeeds, downstream reads successful attempt's marker).
# Companion test verifies the same discard discipline on
# terminal failure (all retries exhausted).


class _RetryTransient(Exception):
    """Carries a transient category so the default classifier
    treats it as retryable. Matches the ``provider_rate_limit``
    category used in ``tests/unit/test_middleware.py``."""

    category = "provider_rate_limit"


async def test_per_attempt_scoping_under_retry_discards_failed_attempt_writes() -> None:
    from openarmature.graph.middleware import RetryConfig, RetryMiddleware

    captured_attempt_1_read: dict[str, Any] = {}
    captured_downstream_read: dict[str, Any] = {}
    attempts: list[int] = []

    async def _retried(_s: _SimpleState) -> dict[str, Any]:
        attempt_n = len(attempts)
        attempts.append(attempt_n)
        if attempt_n == 0:
            # First attempt: write a marker, then raise transient.
            set_invocation_metadata(attempt_marker="first")
            raise _RetryTransient()
        # Second attempt: read first — assert the failed-attempt's
        # marker is NOT visible — then write a new marker and succeed.
        captured_attempt_1_read.update(dict(get_invocation_metadata()))
        set_invocation_metadata(attempt_marker="second")
        return {"counter": 1}

    async def _downstream(_s: _SimpleState) -> dict[str, Any]:
        captured_downstream_read.update(dict(get_invocation_metadata()))
        return {"counter": 2}

    graph = (
        GraphBuilder(_SimpleState)
        .add_node(
            "retried",
            _retried,
            middleware=[RetryMiddleware(RetryConfig(max_attempts=2, backoff=lambda _i: 0.0))],
        )
        .add_node("downstream", _downstream)
        .add_edge("retried", "downstream")
        .add_edge("downstream", END)
        .set_entry("retried")
        .compile()
    )
    await graph.invoke(_SimpleState(), metadata={"tenantId": "T1"})

    assert attempts == [0, 1]
    # Attempt 1's read: baseline only — attempt 0's transient
    # ``attempt_marker=first`` write was discarded on failure.
    assert captured_attempt_1_read == {"tenantId": "T1"}
    # Downstream node: baseline + the successful attempt's write
    # persists past the retry boundary.
    assert captured_downstream_read == {"tenantId": "T1", "attempt_marker": "second"}


async def test_terminal_failure_discards_final_failed_attempt_writes() -> None:
    # Exercises the middleware directly via ``compose_chain`` so the
    # post-retry metadata view is readable in the test scope (the
    # engine's outer invoke() reset would otherwise pop the var back
    # to empty before control returns to the test, masking the
    # middleware's own discard). The contract pinned here is that
    # AFTER the retry middleware re-raises a terminal failure, the
    # metadata ContextVar is back at the pre-attempt baseline — no
    # leak of the final failed attempt's writes.
    from openarmature.graph.middleware import RetryConfig, RetryMiddleware, compose_chain
    from openarmature.observability.metadata import (
        _reset_invocation_metadata,
        _set_invocation_metadata,
        validate_invocation_metadata,
    )

    attempts: list[int] = []

    async def _always_fails(_state: Any) -> Mapping[str, Any]:
        attempts.append(len(attempts))
        set_invocation_metadata(attempt_marker=f"attempt_{len(attempts) - 1}")
        raise _RetryTransient()

    retry = RetryMiddleware(RetryConfig(max_attempts=2, backoff=lambda _i: 0.0))
    chain = compose_chain([retry], _always_fails)

    # Establish a baseline outside the middleware so we can read it
    # back post-failure. Mirrors how the engine sets the baseline
    # at the invoke() boundary.
    baseline_token = _set_invocation_metadata(validate_invocation_metadata({"tenantId": "T1"}))
    try:
        with pytest.raises(_RetryTransient):
            await chain(_SimpleState())
        # Both attempts ran.
        assert attempts == [0, 1]
        # Post-failure view: the pre-attempt baseline, with NO
        # ``attempt_marker`` leaked from the final failed attempt.
        assert dict(get_invocation_metadata()) == {"tenantId": "T1"}
    finally:
        _reset_invocation_metadata(baseline_token)


async def test_cancellation_discards_in_flight_attempt_writes() -> None:
    # Spec §3.4: failed-attempt metadata writes are discarded along
    # with the attempt. When ``CancelledError`` (or any other
    # ``BaseException``) ends the attempt, the same discard discipline
    # applies — cancellation IS a failed attempt from the
    # metadata-scoping perspective. Spec §6.1: cancellation MUST
    # propagate (no retry, no swallow), so the reset must happen IN
    # ADDITION to, not instead of, propagating ``CancelledError``.
    from openarmature.graph.middleware import RetryConfig, RetryMiddleware, compose_chain
    from openarmature.observability.metadata import (
        _reset_invocation_metadata,
        _set_invocation_metadata,
        validate_invocation_metadata,
    )

    attempts: list[int] = []

    async def _writes_then_cancels(_state: Any) -> Mapping[str, Any]:
        attempts.append(len(attempts))
        set_invocation_metadata(attempt_marker="leaked")
        raise asyncio.CancelledError("aborted")

    retry = RetryMiddleware(RetryConfig(max_attempts=3, backoff=lambda _i: 0.0))
    chain = compose_chain([retry], _writes_then_cancels)

    baseline_token = _set_invocation_metadata(validate_invocation_metadata({"tenantId": "T1"}))
    try:
        with pytest.raises(asyncio.CancelledError):
            await chain(_SimpleState())
        # Cancellation propagated — exactly ONE attempt ran (retry
        # MUST NOT swallow ``CancelledError`` per spec §6.1).
        assert attempts == [0]
        # The cancelled attempt's metadata write was discarded per
        # §3.4 — post-failure view is the pre-attempt baseline.
        assert dict(get_invocation_metadata()) == {"tenantId": "T1"}
    finally:
        _reset_invocation_metadata(baseline_token)
