"""Unit tests for the typed-event conformance harness helpers.

The 050-056 fixture runners in ``test_observability.py`` depend on a
handful of pure helpers (``_event_fields_match``, ``_assert_relative_order``,
``_assert_observer_expectations``, etc.). The fixtures themselves only
exercise the green path through each helper; these tests fill in
coverage for the edge cases (typo detection, set-equality semantics,
None handling, cause-chain walking) that would otherwise only surface
when a future fixture happens to trip them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from openarmature.graph.events import LlmCompletionEvent
from openarmature.llm.errors import ProviderUnavailable
from openarmature.llm.response import Usage

from .test_observability import (
    _AllEventsCollector,
    _assert_node_completed_event_carries_error,
    _assert_observer_expectations,
    _assert_relative_order,
    _event_fields_match,
    _mock_model_from_first_response,
    _parse_typed_observers,
    _TypedEventCollector,
)

# ---------------------------------------------------------------------------
# Test fixtures (event instances + synthetic NodeEvent)
# ---------------------------------------------------------------------------


def _make_typed_event(**overrides: Any) -> LlmCompletionEvent:
    """Build a `LlmCompletionEvent` with sensible defaults; overrides
    swap individual fields for the test case.
    """
    base: dict[str, Any] = {
        "invocation_id": "inv-1",
        "correlation_id": "corr-1",
        "node_name": "ask",
        "namespace": ("ask",),
        "attempt_index": 0,
        "fan_out_index": None,
        "branch_name": None,
        "provider": "openai",
        "model": "gpt-test",
        "response_id": "req-1",
        "response_model": None,
        "usage": Usage(prompt_tokens=14, completion_tokens=4, total_tokens=18),
        "latency_ms": 42.0,
        "finish_reason": "stop",
        "input_messages": [],
        "output_content": None,
        "request_params": {},
        "request_extras": {},
        "active_prompt": None,
        "active_prompt_group": None,
        "call_id": "cc-1",
        "caller_invocation_metadata": None,
    }
    base.update(overrides)
    return LlmCompletionEvent(**base)


@dataclass(frozen=True)
class NodeEvent:  # noqa: N801 — class-name matches the harness's string discriminator
    """Stand-in for ``graph.events.NodeEvent`` shaped for the tests'
    cause-chain walk. Only the field names the harness reads are
    present; using a dataclass avoids depending on the real
    NodeEvent's full construction surface. Named ``NodeEvent``
    literally so the harness's ``type(event).__name__ == "NodeEvent"``
    discriminator matches.
    """

    node_name: str
    phase: str
    error: Any = None


# ---------------------------------------------------------------------------
# _event_fields_match
# ---------------------------------------------------------------------------


def test_event_fields_match_simple_equal_returns_true() -> None:
    event = _make_typed_event(provider="openai", model="gpt-test")
    assert _event_fields_match(event, {"provider": "openai", "model": "gpt-test"}) is True


def test_event_fields_match_one_mismatch_returns_false() -> None:
    event = _make_typed_event(model="gpt-test")
    assert _event_fields_match(event, {"model": "different"}) is False


def test_event_fields_match_namespace_list_vs_tuple_compares_as_sequence() -> None:
    # The fixture YAML carries lists; the typed event carries tuples.
    event = _make_typed_event(namespace=("outer", "ask"))
    assert _event_fields_match(event, {"namespace": ["outer", "ask"]}) is True
    assert _event_fields_match(event, {"namespace": ["outer", "different"]}) is False


def test_event_fields_match_usage_mapping_compares_field_by_field() -> None:
    # The fixture YAML carries usage as a flat mapping; the typed event
    # carries a Usage record.
    event = _make_typed_event(usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    assert (
        _event_fields_match(
            event,
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        is True
    )


def test_event_fields_match_usage_field_mismatch_returns_false() -> None:
    event = _make_typed_event(usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    assert _event_fields_match(event, {"usage": {"prompt_tokens": 99}}) is False


def test_event_fields_match_usage_none_with_expected_mapping_returns_false() -> None:
    # When the event has usage=None but the fixture expects a mapping,
    # the field-by-field comparison short-circuits and the top-level
    # equality (None == mapping) returns False.
    event = _make_typed_event(usage=None)
    assert _event_fields_match(event, {"usage": {"prompt_tokens": 10}}) is False


def test_event_fields_match_caller_metadata_none_matches_none() -> None:
    event = _make_typed_event(caller_invocation_metadata=None)
    assert _event_fields_match(event, {"caller_invocation_metadata": None}) is True


def test_event_fields_match_missing_attribute_raises_for_fixture_typo() -> None:
    # A fixture-side field-name typo (e.g., ``node_nam`` instead of
    # ``node_name``) must fail loudly rather than silently matching
    # None. Upstream type filtering guarantees the typed event has
    # all canonical fields, so a missing attribute can only be a
    # fixture authoring bug.
    event = _make_typed_event()
    with pytest.raises(AssertionError, match="does not exist on LlmCompletionEvent"):
        _event_fields_match(event, {"node_nam": None})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _assert_observer_expectations — unknown-key detection
# ---------------------------------------------------------------------------


def test_assert_observer_expectations_rejects_unknown_key() -> None:
    collector = _TypedEventCollector(filter_event_type="LlmCompletionEvent")
    collector.events.append(_make_typed_event())
    with pytest.raises(AssertionError, match="unknown assertion key"):
        _assert_observer_expectations(
            "collector",
            collector,
            {"contians_event": {"event_type": "LlmCompletionEvent"}},  # typo
        )


def test_assert_observer_expectations_allows_informational_key() -> None:
    # ``sentinel_node_event_emission_is_impl_defined`` is an informational
    # flag carried by fixture 051; it must not trip the unknown-key guard.
    collector = _TypedEventCollector(filter_event_type=None)
    collector.events.append(_make_typed_event())
    _assert_observer_expectations(
        "collector",
        collector,
        {
            "contains_event_of_type": "LlmCompletionEvent",
            "sentinel_node_event_emission_is_impl_defined": True,
        },
    )


def test_assert_observer_expectations_no_shape_keys_passes_silently() -> None:
    # An empty spec is benign; only typos in shape keys should fail.
    collector = _TypedEventCollector(filter_event_type=None)
    _assert_observer_expectations("collector", collector, {})


def test_every_captured_event_has_raises_on_missing_attribute() -> None:
    # When the fixture asserts a field that doesn't exist on the
    # captured event type, the harness raises a clear error rather
    # than silently matching None.
    collector = _TypedEventCollector(filter_event_type="LlmCompletionEvent")
    collector.events.append(_make_typed_event())
    with pytest.raises(AssertionError, match="does not exist on LlmCompletionEvent"):
        _assert_observer_expectations(
            "collector",
            collector,
            {"every_captured_event_has": {"node_nam": "ask"}},  # typo
        )


# ---------------------------------------------------------------------------
# captured_event_field_values_cover — set-equality semantics
# ---------------------------------------------------------------------------


def test_captured_event_field_values_cover_set_equality_passes() -> None:
    collector = _TypedEventCollector(filter_event_type="LlmCompletionEvent")
    collector.events.extend(
        [
            _make_typed_event(fan_out_index=0),
            _make_typed_event(fan_out_index=1),
        ]
    )
    _assert_observer_expectations(
        "collector",
        collector,
        {"captured_event_field_values_cover": {"field": "fan_out_index", "values": [0, 1]}},
    )


def test_captured_event_field_values_cover_order_independent() -> None:
    # Values in any order must satisfy the set equality.
    collector = _TypedEventCollector(filter_event_type="LlmCompletionEvent")
    collector.events.extend(
        [
            _make_typed_event(branch_name="slow"),
            _make_typed_event(branch_name="fast"),
        ]
    )
    _assert_observer_expectations(
        "collector",
        collector,
        {
            "captured_event_field_values_cover": {
                "field": "branch_name",
                "values": ["fast", "slow"],
            }
        },
    )


def test_captured_event_field_values_cover_missing_value_fails() -> None:
    collector = _TypedEventCollector(filter_event_type="LlmCompletionEvent")
    collector.events.append(_make_typed_event(fan_out_index=0))
    with pytest.raises(AssertionError):
        _assert_observer_expectations(
            "collector",
            collector,
            {
                "captured_event_field_values_cover": {
                    "field": "fan_out_index",
                    "values": [0, 1],
                }
            },
        )


def test_captured_event_field_values_cover_extra_value_fails() -> None:
    # Set equality also fails if captured carries values the fixture
    # didn't expect.
    collector = _TypedEventCollector(filter_event_type="LlmCompletionEvent")
    collector.events.extend(
        [
            _make_typed_event(fan_out_index=0),
            _make_typed_event(fan_out_index=1),
            _make_typed_event(fan_out_index=2),
        ]
    )
    with pytest.raises(AssertionError):
        _assert_observer_expectations(
            "collector",
            collector,
            {
                "captured_event_field_values_cover": {
                    "field": "fan_out_index",
                    "values": [0, 1],
                }
            },
        )


# ---------------------------------------------------------------------------
# _assert_relative_order — filtered subsequence
# ---------------------------------------------------------------------------


def test_assert_relative_order_passes_when_order_matches() -> None:
    events = [
        NodeEvent(node_name="ask", phase="started"),
        _make_typed_event(node_name="ask"),
        NodeEvent(node_name="ask", phase="completed"),
    ]
    _assert_relative_order(
        "collector",
        events,
        {
            "filter": {"node_name": "ask"},
            "expected_order": [
                {"event_type": "NodeEvent", "phase": "started"},
                {"event_type": "LlmCompletionEvent"},
                {"event_type": "NodeEvent", "phase": "completed"},
            ],
        },
    )


def test_assert_relative_order_fails_on_wrong_phase() -> None:
    events = [
        NodeEvent(node_name="ask", phase="completed"),  # wrong phase first
        _make_typed_event(node_name="ask"),
    ]
    with pytest.raises(AssertionError, match="phase"):
        _assert_relative_order(
            "collector",
            events,
            {
                "filter": {"node_name": "ask"},
                "expected_order": [
                    {"event_type": "NodeEvent", "phase": "started"},
                    {"event_type": "LlmCompletionEvent"},
                ],
            },
        )


def test_assert_relative_order_filter_excludes_non_matching_node_name() -> None:
    events = [
        NodeEvent(node_name="other", phase="started"),  # filtered out
        NodeEvent(node_name="ask", phase="started"),
        _make_typed_event(node_name="ask"),
    ]
    _assert_relative_order(
        "collector",
        events,
        {
            "filter": {"node_name": "ask"},
            "expected_order": [
                {"event_type": "NodeEvent", "phase": "started"},
                {"event_type": "LlmCompletionEvent"},
            ],
        },
    )


# ---------------------------------------------------------------------------
# _assert_node_completed_event_carries_error — cause-chain walk
# ---------------------------------------------------------------------------


def test_node_completed_event_error_category_via_direct_attribute() -> None:
    # When the event's error directly carries a category, the walk
    # finds it on the first hop.
    err = ProviderUnavailable("down")
    event = NodeEvent(node_name="ask", phase="completed", error=err)
    _assert_node_completed_event_carries_error(
        [event], {"node_name": "ask", "error_category": "provider_unavailable"}
    )


def test_node_completed_event_error_category_via_cause_chain() -> None:
    # When the engine wraps the provider error in a NodeException-like
    # parent that itself lacks ``category``, the walk follows
    # ``__cause__`` to find the underlying ProviderUnavailable.
    underlying = ProviderUnavailable("down")
    wrapper = Exception("wrapped")
    wrapper.__cause__ = underlying
    event = NodeEvent(node_name="ask", phase="completed", error=wrapper)
    _assert_node_completed_event_carries_error(
        [event], {"node_name": "ask", "error_category": "provider_unavailable"}
    )


def test_node_completed_event_error_category_mismatch_fails() -> None:
    err = ProviderUnavailable("down")
    event = NodeEvent(node_name="ask", phase="completed", error=err)
    with pytest.raises(AssertionError):
        _assert_node_completed_event_carries_error(
            [event], {"node_name": "ask", "error_category": "provider_rate_limit"}
        )


def test_node_completed_event_no_matching_node_fails() -> None:
    err = ProviderUnavailable("down")
    event = NodeEvent(node_name="other", phase="completed", error=err)
    with pytest.raises(AssertionError):
        _assert_node_completed_event_carries_error(
            [event], {"node_name": "ask", "error_category": "provider_unavailable"}
        )


# ---------------------------------------------------------------------------
# _parse_typed_observers — populate_caller_metadata aggregation
# ---------------------------------------------------------------------------


def test_parse_typed_observers_aggregates_populate_caller_metadata_true() -> None:
    case: Mapping[str, Any] = {
        "typed_observers": [
            {"name": "a", "kind": "typed_event_collector"},
            {"name": "b", "kind": "typed_event_collector", "include_caller_metadata": True},
        ]
    }
    collectors, populate = _parse_typed_observers(case)
    assert set(collectors.keys()) == {"a", "b"}
    assert populate is True


def test_parse_typed_observers_no_opt_in_yields_false() -> None:
    case: Mapping[str, Any] = {
        "typed_observers": [
            {"name": "a", "kind": "typed_event_collector"},
            {"name": "b", "kind": "typed_event_collector", "filter_event_type": "LlmCompletionEvent"},
        ]
    }
    _, populate = _parse_typed_observers(case)
    assert populate is False


def test_parse_typed_observers_empty_directive_returns_empty() -> None:
    collectors, populate = _parse_typed_observers({})
    assert collectors == {}
    assert populate is False


def test_parse_typed_observers_rejects_unknown_kind() -> None:
    case: Mapping[str, Any] = {
        "typed_observers": [{"name": "x", "kind": "wat"}],
    }
    with pytest.raises(AssertionError, match="unsupported typed_observer kind"):
        _parse_typed_observers(case)


# ---------------------------------------------------------------------------
# _TypedEventCollector filtering
# ---------------------------------------------------------------------------


async def test_typed_event_collector_filter_drops_non_matching_type() -> None:
    collector = _TypedEventCollector(filter_event_type="LlmCompletionEvent")
    await collector(_make_typed_event())
    await collector(NodeEvent(node_name="ask", phase="started"))
    assert len(collector.events) == 1
    assert type(collector.events[0]).__name__ == "LlmCompletionEvent"


async def test_typed_event_collector_unfiltered_captures_every_event() -> None:
    collector = _TypedEventCollector(filter_event_type=None)
    await collector(_make_typed_event())
    await collector(NodeEvent(node_name="ask", phase="started"))
    await collector(NodeEvent(node_name="ask", phase="completed"))
    assert len(collector.events) == 3


async def test_all_events_collector_preserves_insertion_order() -> None:
    collector = _AllEventsCollector()
    a = NodeEvent(node_name="ask", phase="started")
    b = _make_typed_event()
    c = NodeEvent(node_name="ask", phase="completed")
    await collector(a)
    await collector(b)
    await collector(c)
    assert collector.events == [a, b, c]


# ---------------------------------------------------------------------------
# _mock_model_from_first_response
# ---------------------------------------------------------------------------


def test_mock_model_returns_first_response_body_model() -> None:
    case: Mapping[str, Any] = {
        "mock_llm": [
            {"status": 200, "body": {"model": "gpt-test"}},
            {"status": 200, "body": {"model": "different"}},
        ]
    }
    assert _mock_model_from_first_response(case) == "gpt-test"


def test_mock_model_returns_none_when_no_mock_llm() -> None:
    assert _mock_model_from_first_response({}) is None


def test_mock_model_returns_none_when_body_lacks_model_field() -> None:
    case: Mapping[str, Any] = {"mock_llm": [{"status": 200, "body": {"id": "x"}}]}
    assert _mock_model_from_first_response(case) is None
