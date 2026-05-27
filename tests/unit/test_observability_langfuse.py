"""Focused unit tests for the LangfuseObserver and InMemoryLangfuseClient.

The conformance suite (``tests/conformance/test_observability_langfuse.py``)
exercises the end-to-end Trace + Observation shape against
spec/observability/conformance/022-024. These unit tests fill gaps
those fixtures don't isolate directly: payload-cap validation,
truncation algorithm boundaries, in-memory recorder field handling.
"""

from __future__ import annotations

import pytest

from openarmature.observability.langfuse import (
    InMemoryLangfuseClient,
    LangfuseObserver,
    LangfuseUsage,
)


def test_observer_payload_cap_below_minimum_rejected() -> None:
    # §5.5.5 minimum-cap mirror — 255 sits one byte below the spec
    # minimum and MUST be rejected at construction time.
    client = InMemoryLangfuseClient()
    with pytest.raises(ValueError, match="below the spec §5.5.5 minimum"):
        LangfuseObserver(client=client, payload_byte_cap=255)


def test_observer_payload_cap_at_minimum_accepted() -> None:
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, payload_byte_cap=256)
    assert observer.payload_byte_cap == 256


def test_in_memory_recorder_trace_create_then_update() -> None:
    client = InMemoryLangfuseClient()
    client.trace(id="t1", name="initial", metadata={"correlation_id": "c1"})
    client.update_trace(id="t1", name="renamed", metadata={"extra": "value"})

    trace = client.traces["t1"]
    assert trace.id == "t1"
    assert trace.name == "renamed"
    assert trace.metadata == {"correlation_id": "c1", "extra": "value"}


def test_in_memory_recorder_span_handle_update_and_end() -> None:
    client = InMemoryLangfuseClient()
    client.trace(id="t1")
    handle = client.span(trace_id="t1", name="step", metadata={"k": 1})

    handle.update(metadata={"extra": "v"})
    handle.end(level="ERROR", status_message="failed")

    trace = client.traces["t1"]
    assert len(trace.observations) == 1
    obs = trace.observations[0]
    assert obs.name == "step"
    assert obs.ended is True
    assert obs.level == "ERROR"
    assert obs.status_message == "failed"
    assert obs.metadata == {"k": 1, "extra": "v"}


def test_in_memory_recorder_generation_captures_native_fields() -> None:
    client = InMemoryLangfuseClient()
    client.trace(id="t1")
    handle = client.generation(
        trace_id="t1",
        name="openarmature.llm.complete",
        model="test-model",
        model_parameters={"temperature": 0.7},
        input=[{"role": "user", "content": "hi"}],
        output="hello back",
        usage=LangfuseUsage(input=5, output=2, total=7),
        prompt="lf-prompt-ref-1",
    )
    handle.end(metadata={"finish_reason": "stop"})

    trace = client.traces["t1"]
    assert len(trace.observations) == 1
    obs = trace.observations[0]
    assert obs.type == "generation"
    assert obs.model == "test-model"
    assert obs.model_parameters == {"temperature": 0.7}
    assert obs.input == [{"role": "user", "content": "hi"}]
    assert obs.output == "hello back"
    assert obs.usage is not None
    assert obs.usage.input == 5
    assert obs.usage.output == 2
    assert obs.usage.total == 7
    assert obs.prompt_entity_link == "lf-prompt-ref-1"
    assert obs.metadata == {"finish_reason": "stop"}


def test_in_memory_recorder_observation_id_is_unique_per_recorder() -> None:
    client = InMemoryLangfuseClient()
    client.trace(id="t1")
    a = client.span(trace_id="t1", name="a")
    b = client.span(trace_id="t1", name="b")
    assert a.id != b.id


def test_in_memory_recorder_children_of_walks_parent_links() -> None:
    client = InMemoryLangfuseClient()
    client.trace(id="t1")
    root = client.span(trace_id="t1", name="root")
    child = client.span(trace_id="t1", name="child", parent_observation_id=root.id)
    other = client.span(trace_id="t1", name="other")

    trace = client.traces["t1"]
    top_level = trace.children_of(None)
    assert {o.name for o in top_level} == {"root", "other"}
    root_children = trace.children_of(root.id)
    assert [o.name for o in root_children] == ["child"]
    # Unrelated observation not under root.
    assert child.id != other.id
