"""Behavioral regression tests for the production-observability example's
queryable observers.

``test_examples_smoke.py`` only proves the example loads and its
``build_graph()`` compiles. This file goes one level deeper into the two
queryable-observer classes the example ships, locking the logic a
happy-path live run never reaches:

- ``LlmUsageAccumulator`` accumulating ``usage.cached_tokens`` and the
  cache-hit ratio the persist node derives from it (a real OpenAI run
  reports zero cached tokens, so the ratio is always 0.0% there).
- ``LlmFailureTracker`` counting failures by category (a successful run
  produces an empty bucket and prints "none").
- Mutual exclusion between the success and failure events: each observer
  ignores the other's event type, so the two never double-count.
- The per-invocation bucket cleanup on ``InvocationCompletedEvent``.
- The OTel formatter surfacing the cache-read span attribute.

The classes live in the example module, so the example is loaded with
``runpy.run_path`` (matching the smoke test). ``runpy`` returns a copy of
the executed namespace; the module's own functions hold the live dict via
``__globals__``, and ``build_graph()`` / ``persist()`` read and write
module-level singletons there, so the fixture hands back that live
namespace rather than the returned copy.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# The example imports opentelemetry-sdk and langfuse record types at module
# top; skip cleanly when the extras aren't installed.
pytest.importorskip("opentelemetry.sdk.trace")
pytest.importorskip("langfuse")

from openarmature.graph import (  # noqa: E402
    InvocationCompletedEvent,
    LlmCompletionEvent,
    LlmFailedEvent,
)
from openarmature.llm import Usage  # noqa: E402
from openarmature.observability.correlation import (  # noqa: E402
    _reset_invocation_id,
    _set_invocation_id,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def example_ns() -> dict[str, Any]:
    """Load the production-observability example and return its live
    module namespace (the dict the module's functions actually read and
    write, not ``runpy``'s returned copy)."""
    main_py = EXAMPLES_DIR / "production-observability" / "main.py"
    # A fresh run_path each call keeps tests isolated: build_graph()
    # mutates module-level singletons, and a fresh namespace per test
    # means those mutations don't leak. Reach the live namespace through
    # a function's __globals__ rather than runpy's returned copy.
    returned = runpy.run_path(str(main_py), run_name="__not_main__")
    return returned["build_graph"].__globals__


def _usage(
    *,
    prompt: int | None = None,
    completion: int | None = None,
    total: int | None = None,
    cached: int | None = None,
) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
    )


def _completion(invocation_id: str, usage: Usage) -> LlmCompletionEvent:
    return LlmCompletionEvent(
        invocation_id=invocation_id,
        correlation_id="corr",
        node_name="respond",
        namespace=("respond",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
        provider="openai",
        model="gpt-4o-mini",
        response_id="resp",
        response_model="gpt-4o-mini-2024-07-18",
        usage=usage,
        latency_ms=12.3,
        finish_reason="stop",
        input_messages=[],
        output_content="ok",
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
        call_id="call",
    )


def _failure(invocation_id: str, category: str) -> LlmFailedEvent:
    return LlmFailedEvent(
        invocation_id=invocation_id,
        correlation_id="corr",
        node_name="respond",
        namespace=("respond",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
        provider="openai",
        model="gpt-4o-mini",
        latency_ms=5.0,
        input_messages=[],
        request_params={},
        request_extras={},
        active_prompt=None,
        active_prompt_group=None,
        call_id="call",
        error_category=category,
        error_message="boom",
    )


def _completed(invocation_id: str) -> InvocationCompletedEvent:
    return InvocationCompletedEvent(
        final_state=None,
        status="completed",
        final_node="persist",
        invocation_id=invocation_id,
        correlation_id="corr",
    )


# --- LlmUsageAccumulator -------------------------------------------------


async def test_usage_accumulator_accumulates_cache_tokens(example_ns: dict[str, Any]) -> None:
    acc = example_ns["LlmUsageAccumulator"]()
    await acc(_completion("inv", _usage(prompt=100, completion=40, total=140, cached=30)))
    await acc(_completion("inv", _usage(prompt=50, completion=20, total=60, cached=10)))
    bucket = acc.get_bucket("inv")
    assert (
        bucket.prompt_tokens,
        bucket.completion_tokens,
        bucket.total_tokens,
        bucket.cached_tokens,
        bucket.call_count,
    ) == (150, 60, 200, 40, 2)


async def test_usage_accumulator_tolerates_null_cache(example_ns: dict[str, Any]) -> None:
    acc = example_ns["LlmUsageAccumulator"]()
    await acc(_completion("inv", _usage(prompt=10, completion=5, total=15, cached=None)))
    assert acc.get_bucket("inv").cached_tokens == 0


async def test_usage_accumulator_ignores_failure_event(example_ns: dict[str, Any]) -> None:
    acc = example_ns["LlmUsageAccumulator"]()
    await acc(_failure("inv", "provider_rate_limit"))
    assert acc.get_bucket("inv") is None


async def test_usage_accumulator_drops_bucket_on_completion(example_ns: dict[str, Any]) -> None:
    acc = example_ns["LlmUsageAccumulator"]()
    await acc(_completion("inv", _usage(prompt=1, completion=1, total=2, cached=0)))
    await acc(_completed("inv"))
    assert acc.get_bucket("inv") is None


# --- LlmFailureTracker ---------------------------------------------------


async def test_failure_tracker_counts_by_category(example_ns: dict[str, Any]) -> None:
    tracker = example_ns["LlmFailureTracker"]()
    await tracker(_failure("inv", "provider_rate_limit"))
    await tracker(_failure("inv", "provider_unavailable"))
    await tracker(_failure("inv", "provider_rate_limit"))
    assert dict(tracker.get_bucket("inv").by_category) == {
        "provider_rate_limit": 2,
        "provider_unavailable": 1,
    }


async def test_failure_tracker_ignores_completion_event(example_ns: dict[str, Any]) -> None:
    tracker = example_ns["LlmFailureTracker"]()
    await tracker(_completion("inv", _usage(prompt=1, completion=1, total=2, cached=0)))
    assert tracker.get_bucket("inv") is None


async def test_failure_tracker_drops_bucket_on_completion(example_ns: dict[str, Any]) -> None:
    tracker = example_ns["LlmFailureTracker"]()
    await tracker(_failure("inv", "provider_rate_limit"))
    await tracker(_completed("inv"))
    assert tracker.get_bucket("inv") is None


# --- persist node output -------------------------------------------------


async def test_persist_reports_cache_ratio_and_failure_breakdown(
    example_ns: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    # build_graph() compiles the graph and installs the two accumulators
    # as module-level singletons that persist() reads. drain_events_for
    # returns an empty summary for an invocation with no active worker,
    # so persist() runs to completion offline without a live invoke().
    example_ns["build_graph"]()
    acc = example_ns["_accumulator"]
    tracker = example_ns["_failure_tracker"]
    inv = "inv-persist"
    await acc(_completion(inv, _usage(prompt=100, completion=40, total=140, cached=30)))
    await tracker(_failure(inv, "provider_rate_limit"))
    await tracker(_failure(inv, "provider_unavailable"))
    await tracker(_failure(inv, "provider_rate_limit"))

    state = example_ns["BriefingState"](question="q")
    token = _set_invocation_id(inv)
    try:
        await example_ns["persist"](state)
    finally:
        _reset_invocation_id(token)

    out = capsys.readouterr().out
    assert (
        "[persist] LLM usage: prompt=100 (cached=30, 30.0% hit), "
        "completion=40, total=140 across 1 call(s)" in out
    )
    assert "[persist] LLM failures: provider_rate_limit=2, provider_unavailable=1" in out


# --- OTel span formatter -------------------------------------------------


def test_otel_formatter_surfaces_cache_read_attribute(example_ns: dict[str, Any]) -> None:
    # Stand-in for a ReadableSpan: the formatter reads name / attributes /
    # start_time / end_time only.
    span = SimpleNamespace(
        name="openarmature.llm.complete",
        attributes={
            "gen_ai.system": "openai",
            "openarmature.llm.cache_read.input_tokens": 12,
        },
        start_time=0,
        end_time=1_000_000,
    )
    rendered = example_ns["_format_otel_spans"]([span])
    assert "openarmature.llm.cache_read.input_tokens=12" in rendered
