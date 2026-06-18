"""Unit tests for the middleware infrastructure and canonical middleware.

Covers the eight items from the Phase 2 plan:

1. Chain composition + ordering
2. Short-circuit (middleware skips ``next``)
3. Pre/post phase symmetry (single function, two halves)
4. Retry counting + classifier
5. Retry cancellation propagation (``asyncio.CancelledError`` MUST escape)
6. General error recovery (middleware catches + returns partial)
7. Timing on success and failure
8. Subgraph isolation (parent middleware does NOT cross the boundary)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from pydantic import Field

from openarmature.graph import (
    END,
    GraphBuilder,
    Middleware,
    RetryConfig,
    RetryMiddleware,
    State,
    TimingMiddleware,
    TimingRecord,
    append,
    deterministic_backoff,
)
from openarmature.graph.middleware import (
    NextCall,
    compose_chain,
    default_classifier,
)


class TraceState(State):
    trace: Annotated[list[str], append] = Field(default_factory=list)


class _CategorizedTransient(Exception):
    category = "provider_rate_limit"


class _CategorizedFatal(Exception):
    category = "provider_invalid_request"


# ===== 1. Chain composition + ordering =====


async def test_compose_chain_runs_outer_to_inner_and_back() -> None:
    """For chain ``[m1, m2, m3]`` wrapping innermost, the order is
    m1.pre → m2.pre → m3.pre → innermost → m3.post → m2.post → m1.post."""
    log: list[str] = []

    def make_recorder(name: str) -> Middleware:
        async def mw(state: Any, next_: NextCall) -> Mapping[str, Any]:
            log.append(f"{name}.pre")
            partial = await next_(state)
            log.append(f"{name}.post")
            return partial

        return mw

    async def innermost(_state: Any) -> Mapping[str, Any]:
        log.append("innermost")
        return {"trace": ["x"]}

    chain = compose_chain([make_recorder("m1"), make_recorder("m2"), make_recorder("m3")], innermost)
    result = await chain(TraceState())

    assert result == {"trace": ["x"]}
    assert log == ["m1.pre", "m2.pre", "m3.pre", "innermost", "m3.post", "m2.post", "m1.post"]


# ===== 2. Short-circuit =====


async def test_short_circuit_skips_inner_chain() -> None:
    """A middleware that returns without calling ``next`` skips all
    subsequent middleware AND the wrapped innermost."""
    log: list[str] = []

    async def short_circuit(_state: Any, _next: NextCall) -> Mapping[str, Any]:
        log.append("short_circuit")
        return {"trace": ["short"]}

    async def inner_recorder(state: Any, next_: NextCall) -> Mapping[str, Any]:
        log.append("inner_recorder")
        return await next_(state)

    async def innermost(_state: Any) -> Mapping[str, Any]:
        log.append("innermost")
        return {"trace": ["never"]}

    chain = compose_chain([short_circuit, inner_recorder], innermost)
    result = await chain(TraceState())

    assert result == {"trace": ["short"]}
    # Only short_circuit ran. inner_recorder and innermost were skipped.
    assert log == ["short_circuit"]


# ===== 3. Pre/post phase symmetry =====


async def test_pre_post_symmetry() -> None:
    """If ``m1`` is outermost, ``m1.pre`` runs first AND ``m1.post`` runs
    last. Pre/post are tied to the same position in the chain."""
    log: list[str] = []

    async def m1(state: Any, next_: NextCall) -> Mapping[str, Any]:
        log.append("m1.pre")
        partial = await next_(state)
        log.append("m1.post")
        return partial

    async def m2(state: Any, next_: NextCall) -> Mapping[str, Any]:
        log.append("m2.pre")
        partial = await next_(state)
        log.append("m2.post")
        return partial

    async def innermost(_state: Any) -> Mapping[str, Any]:
        return {}

    chain = compose_chain([m1, m2], innermost)
    await chain(TraceState())

    assert log == ["m1.pre", "m2.pre", "m2.post", "m1.post"]


# ===== 4. Retry counting + classifier =====


async def test_retry_succeeds_on_second_attempt() -> None:
    attempts = [0]

    async def innermost(_state: Any) -> Mapping[str, Any]:
        attempts[0] += 1
        if attempts[0] < 2:
            raise _CategorizedTransient()
        return {"trace": ["ok"]}

    retry = RetryMiddleware(RetryConfig(max_attempts=3, backoff=deterministic_backoff(0)))
    chain = compose_chain([retry], innermost)

    result = await chain(TraceState())
    assert result == {"trace": ["ok"]}
    assert attempts[0] == 2


async def test_retry_exhausted_re_raises_last_exception() -> None:
    attempts = [0]

    async def innermost(_state: Any) -> Mapping[str, Any]:
        attempts[0] += 1
        raise _CategorizedTransient()

    retry = RetryMiddleware(RetryConfig(max_attempts=3, backoff=deterministic_backoff(0)))
    chain = compose_chain([retry], innermost)

    with pytest.raises(_CategorizedTransient):
        await chain(TraceState())
    assert attempts[0] == 3


async def test_retry_skips_non_retryable() -> None:
    """The default classifier returns False for non-transient categories
    — the first failure propagates immediately."""
    attempts = [0]

    async def innermost(_state: Any) -> Mapping[str, Any]:
        attempts[0] += 1
        raise _CategorizedFatal()

    retry = RetryMiddleware(RetryConfig(max_attempts=5, backoff=deterministic_backoff(0)))
    chain = compose_chain([retry], innermost)

    with pytest.raises(_CategorizedFatal):
        await chain(TraceState())
    assert attempts[0] == 1


# ===== 5. Retry cancellation propagation =====


async def test_retry_propagates_cancelled_error() -> None:
    """``asyncio.CancelledError`` extends BaseException, not Exception, so
    retry's ``except Exception`` doesn't catch it. Cancellation falls
    straight through, preserving the host's intent to abort."""
    attempts = [0]

    async def innermost(_state: Any) -> Mapping[str, Any]:
        attempts[0] += 1
        raise asyncio.CancelledError("aborted by host")

    retry = RetryMiddleware(RetryConfig(max_attempts=5, backoff=deterministic_backoff(0)))
    chain = compose_chain([retry], innermost)

    with pytest.raises(asyncio.CancelledError):
        await chain(TraceState())
    # Retry MUST NOT swallow CancelledError — exactly one attempt.
    assert attempts[0] == 1


# ===== RetryConfig record =====


def test_retry_config_validates_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        RetryConfig(max_attempts=0)


def test_retry_config_defaults_resolve_at_use() -> None:
    # Optional fields default to None; the consumer (RetryMiddleware, and
    # the upcoming call-level retry) resolves None to the canonical
    # defaults. A bare RetryMiddleware() uses the default RetryConfig().
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.classifier is None
    assert cfg.backoff is None
    assert RetryMiddleware().config == RetryConfig()


def test_retry_middleware_rejects_non_config() -> None:
    with pytest.raises(TypeError, match="expects a RetryConfig"):
        RetryMiddleware(3)  # pyright: ignore[reportArgumentType]


# ===== 6. General error recovery =====


async def test_error_recovery_via_catch_and_return_partial() -> None:
    """Middleware MAY catch an exception and return a partial update.
    The chain returns successfully — no exception
    reaches the engine."""
    inner_called = [0]

    async def recovery(state: Any, next_: NextCall) -> Mapping[str, Any]:
        try:
            return await next_(state)
        except Exception:
            return {"trace": ["recovered"]}

    async def innermost(_state: Any) -> Mapping[str, Any]:
        inner_called[0] += 1
        raise RuntimeError("inner blew up")

    chain = compose_chain([recovery], innermost)
    result = await chain(TraceState())
    assert result == {"trace": ["recovered"]}
    assert inner_called[0] == 1


# ===== 7. Timing on success and failure =====


async def test_timing_records_success_with_duration() -> None:
    records: list[TimingRecord] = []
    counter = [0.0]

    def fake_clock() -> float:
        n = counter[0]
        counter[0] = n + 0.005  # advance 5ms per call
        return n

    async def on_complete(rec: TimingRecord) -> None:
        records.append(rec)

    async def innermost(_state: Any) -> Mapping[str, Any]:
        return {"trace": ["x"]}

    timing = TimingMiddleware(node_name="alpha", on_complete=on_complete, clock=fake_clock)
    chain = compose_chain([timing], innermost)
    await chain(TraceState())

    assert len(records) == 1
    rec = records[0]
    assert rec.node_name == "alpha"
    assert rec.outcome == "success"
    assert rec.exception_category is None
    assert abs(rec.duration_ms - 5.0) < 0.01


async def test_timing_records_failure_with_category() -> None:
    """When ``next_`` raises, timing records the exception's category and
    re-raises."""
    records: list[TimingRecord] = []
    counter = [0.0]

    def fake_clock() -> float:
        n = counter[0]
        counter[0] = n + 0.010
        return n

    async def on_complete(rec: TimingRecord) -> None:
        records.append(rec)

    async def innermost(_state: Any) -> Mapping[str, Any]:
        raise _CategorizedFatal()

    timing = TimingMiddleware(node_name="alpha", on_complete=on_complete, clock=fake_clock)
    chain = compose_chain([timing], innermost)
    with pytest.raises(_CategorizedFatal):
        await chain(TraceState())

    assert len(records) == 1
    rec = records[0]
    assert rec.outcome == "exception"
    assert rec.exception_category == "provider_invalid_request"
    assert abs(rec.duration_ms - 10.0) < 0.01


# ===== Reentrant next: middleware can call next more than once =====


async def test_middleware_can_call_next_repeatedly() -> None:
    """A middleware MAY call ``next`` more than once. Retry
    exercises this with N=2-3 attempts; this test pins the contract
    independently by calling ``next`` 5 times in a loop and asserting the
    inner runs exactly that many times."""
    inner_calls = 0

    async def call_five_times(state: Any, next_: NextCall) -> Mapping[str, Any]:
        last: Mapping[str, Any] = {}
        for _ in range(5):
            last = await next_(state)
        return last

    async def innermost(_state: Any) -> Mapping[str, Any]:
        nonlocal inner_calls
        inner_calls += 1
        return {"trace": [f"call-{inner_calls}"]}

    chain = compose_chain([call_five_times], innermost)
    result = await chain(TraceState())

    assert inner_calls == 5
    # Final result is the partial returned from the LAST call to next.
    assert result == {"trace": ["call-5"]}


# ===== Timing callback failure on the failure path masks the original =====


async def test_timing_callback_failure_replaces_original_exception() -> None:
    """Pins the current behavior: when a node raises and TimingMiddleware's
    ``on_complete`` ALSO raises in the failure path, the callback's
    exception propagates instead of the original. Python's standard
    exception chaining preserves the original on ``__context__``, but the
    active exception observers see is the callback's.

    Callbacks SHOULD be fast and infallible — this test documents what
    happens if a user violates that, so a future change
    that wants to preserve the original (e.g., via explicit ``raise exc
    from cb_exc``) doesn't silently regress this contract.
    """

    async def bad_callback(_record: TimingRecord) -> None:
        raise ValueError("callback bug")

    async def innermost(_state: Any) -> Mapping[str, Any]:
        raise _CategorizedFatal()

    timing = TimingMiddleware(node_name="alpha", on_complete=bad_callback)
    chain = compose_chain([timing], innermost)

    with pytest.raises(ValueError, match="callback bug") as excinfo:
        await chain(TraceState())

    # The original node exception is preserved on __context__, the
    # standard Python exception-chaining link for "exception raised
    # while handling another."
    assert isinstance(excinfo.value.__context__, _CategorizedFatal)


# ===== 8. Subgraph isolation =====


class OuterState(State):
    trace: Annotated[list[str], append] = Field(default_factory=list)


class InnerState(State):
    trace: Annotated[list[str], append] = Field(default_factory=list)


async def test_parent_middleware_does_not_wrap_subgraph_internal_nodes() -> None:
    """Parent's middleware wraps the SubgraphNode dispatch but NOT the
    subgraph's internal nodes. The subgraph's own middleware
    is the only thing wrapping its inner nodes."""
    parent_calls: list[str] = []
    sub_calls: list[str] = []

    async def parent_recorder(state: Any, next_: NextCall) -> Mapping[str, Any]:
        parent_calls.append("dispatch")
        return await next_(state)

    async def sub_recorder(state: Any, next_: NextCall) -> Mapping[str, Any]:
        sub_calls.append("dispatch")
        return await next_(state)

    async def inner_a(_state: InnerState) -> Mapping[str, Any]:
        return {"trace": ["inner_a"]}

    async def inner_b(_state: InnerState) -> Mapping[str, Any]:
        return {"trace": ["inner_b"]}

    sub_builder: GraphBuilder[InnerState] = GraphBuilder(InnerState)
    sub_builder.set_entry("inner_a")
    sub_builder.add_node("inner_a", inner_a)
    sub_builder.add_node("inner_b", inner_b)
    sub_builder.add_edge("inner_a", "inner_b")
    sub_builder.add_edge("inner_b", END)
    sub_builder.add_middleware(sub_recorder)
    sub_compiled = sub_builder.compile()

    async def outer_node(_state: OuterState) -> Mapping[str, Any]:
        return {"trace": ["outer"]}

    parent: GraphBuilder[OuterState] = GraphBuilder(OuterState)
    parent.set_entry("outer")
    parent.add_node("outer", outer_node)
    parent.add_subgraph_node("sub", sub_compiled)
    parent.add_edge("outer", "sub")
    parent.add_edge("sub", END)
    parent.add_middleware(parent_recorder)
    compiled = parent.compile()

    final = await compiled.invoke(OuterState())
    await compiled.drain()

    # Parent middleware fires twice: once for `outer` node, once for
    # `sub` SubgraphNode dispatch. NOT three times — it doesn't see
    # `inner_a` or `inner_b`.
    assert len(parent_calls) == 2

    # Subgraph middleware fires twice: once for each inner node. The
    # parent's middleware is invisible from the subgraph's view.
    assert len(sub_calls) == 2

    # Final state has the full trace.
    assert final.trace == ["outer", "inner_a", "inner_b"]


# ===== Default classifier behavior =====


def test_default_classifier_recognizes_transient_via_direct_category() -> None:
    """The default classifier checks the exception's own ``category``
    attribute against ``TRANSIENT_CATEGORIES``."""
    state: Any = None
    assert default_classifier(_CategorizedTransient(), state) is True


def test_default_classifier_walks_cause_for_node_exception_wrappers() -> None:
    """A node_exception whose ``__cause__`` is a transient category MUST
    be classified as transient."""
    raw = _CategorizedTransient()
    wrapper = RuntimeError("wrapped")
    wrapper.__cause__ = raw
    state: Any = None
    assert default_classifier(wrapper, state) is True


def test_default_classifier_rejects_non_transient() -> None:
    state: Any = None
    assert default_classifier(_CategorizedFatal(), state) is False
