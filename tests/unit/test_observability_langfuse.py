"""Focused unit tests for the LangfuseObserver and InMemoryLangfuseClient.

The conformance suite (``tests/conformance/test_observability_langfuse.py``)
exercises the end-to-end Trace + Observation shape against
spec/observability/conformance/022-024. These unit tests fill gaps
those fixtures don't isolate directly: payload-cap validation,
truncation algorithm boundaries, in-memory recorder field handling,
and the synthetic-dispatch-observation paths (subgraph, fan-out
non-detached, detached subgraph, detached fan-out) that no Langfuse
spec fixture exercises today.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

import pytest

from openarmature.graph import END, GraphBuilder, State, append
from openarmature.observability.langfuse import (
    InMemoryLangfuseClient,
    LangfuseObservation,
    LangfuseObserver,
    LangfuseTrace,
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


# ---------------------------------------------------------------------------
# Dispatch synthesis (PR 3.5) — subgraph, fan-out non-detached, detached
# ---------------------------------------------------------------------------
# The Langfuse mapping has no spec fixtures for subgraph dispatch /
# fan-out per-instance / detached-trace mode (spec proposal 0031's
# 022-024 only exercise linear graphs + LLM + prompt linkage). These
# tests pin the synthesis-helper behavior locally so future changes
# don't silently break parenting under composition.


class _S(State):
    trail: Annotated[list[str], append] = []
    worker_results: Annotated[list[str], append] = []


class _WorkerState(State):
    result: str = ""


async def _record(name: str) -> Any:
    return {"trail": [name]}


def _attach(graph: Any) -> tuple[Any, InMemoryLangfuseClient, LangfuseObserver]:
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    graph.attach_observer(observer)
    return graph, client, observer


def _attach_with_detached(
    graph: Any,
    *,
    detached_subgraphs: frozenset[str] = frozenset(),
    detached_fan_outs: frozenset[str] = frozenset(),
) -> tuple[Any, InMemoryLangfuseClient, LangfuseObserver]:
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(
        client=client,
        detached_subgraphs=detached_subgraphs,
        detached_fan_outs=detached_fan_outs,
    )
    graph.attach_observer(observer)
    return graph, client, observer


def _find_observation(trace: LangfuseTrace, name: str) -> LangfuseObservation:
    for obs in trace.observations:
        if obs.name == name:
            return obs
    raise AssertionError(f"observation {name!r} not in trace {trace.id!r}")


async def test_subgraph_dispatch_observation_parents_inner_node() -> None:
    inner = (
        GraphBuilder(_S)
        .add_node("inner_a", lambda _s: _record("inner_a"))
        .add_edge("inner_a", END)
        .set_entry("inner_a")
        .compile()
    )
    parent = GraphBuilder(_S).add_subgraph_node("sub", inner).add_edge("sub", END).set_entry("sub").compile()
    graph, client, _ = _attach(parent)

    await graph.invoke(_S())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    sub_dispatch = _find_observation(trace, "sub")
    inner_node = _find_observation(trace, "inner_a")
    # inner_a must parent under the synthesized subgraph dispatch
    # observation, not directly under the Trace.
    assert inner_node.parent_observation_id == sub_dispatch.id
    # The subgraph dispatch lives at the top level of the Trace.
    assert sub_dispatch.parent_observation_id is None


async def test_fan_out_non_detached_per_instance_dispatch() -> None:
    async def _worker(_s: _WorkerState) -> Any:
        return {"result": "done"}

    inner = (
        GraphBuilder(_WorkerState)
        .add_node("worker", _worker)
        .add_edge("worker", END)
        .set_entry("worker")
        .compile()
    )
    parent = (
        GraphBuilder(_S)
        .add_fan_out_node(
            "fan",
            subgraph=inner,
            count=2,
            collect_field="result",
            target_field="worker_results",
        )
        .add_edge("fan", END)
        .set_entry("fan")
        .compile()
    )
    graph, client, _ = _attach(parent)

    await graph.invoke(_S())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    fan_node = _find_observation(trace, "fan")
    # Per-instance dispatch observations share the fan-out node name.
    dispatches = [o for o in trace.observations if o.name == "fan" and o.parent_observation_id == fan_node.id]
    assert len(dispatches) == 2, f"expected 2 per-instance dispatches, got {len(dispatches)}"
    # Each per-instance dispatch carries the fan_out_index in metadata.
    indices = {d.metadata.get("fan_out_index") for d in dispatches}
    assert indices == {0, 1}
    # Worker observations parent under their per-instance dispatch.
    workers = [o for o in trace.observations if o.name == "worker"]
    assert len(workers) == 2
    worker_parents = {w.parent_observation_id for w in workers}
    dispatch_ids = {d.id for d in dispatches}
    assert worker_parents == dispatch_ids


async def test_detached_subgraph_opens_separate_trace() -> None:
    inner = (
        GraphBuilder(_S)
        .add_node("inner_a", lambda _s: _record("inner_a"))
        .add_edge("inner_a", END)
        .set_entry("inner_a")
        .compile()
    )
    parent = GraphBuilder(_S).add_subgraph_node("sub", inner).add_edge("sub", END).set_entry("sub").compile()
    graph, client, _ = _attach_with_detached(parent, detached_subgraphs=frozenset({"sub"}))

    await graph.invoke(_S())
    await graph.drain()

    # Two Traces: main invocation + detached subgraph.
    assert len(client.traces) == 2
    main = next(t for t in client.traces.values() if "detached_from_invocation_id" not in t.metadata)
    detached = next(t for t in client.traces.values() if "detached_from_invocation_id" in t.metadata)

    # Main Trace has the link observation with detached_child_trace_ids.
    link_obs = _find_observation(main, "sub")
    assert detached.id in link_obs.metadata["detached_child_trace_ids"]
    # Detached Trace has its own dispatch observation + inner_a under it.
    detached_dispatch = _find_observation(detached, "sub")
    assert detached_dispatch.parent_observation_id is None
    inner_node = _find_observation(detached, "inner_a")
    assert inner_node.parent_observation_id == detached_dispatch.id


async def test_detached_fan_out_each_instance_gets_trace() -> None:
    async def _worker(_s: _WorkerState) -> Any:
        return {"result": "done"}

    inner = (
        GraphBuilder(_WorkerState)
        .add_node("worker", _worker)
        .add_edge("worker", END)
        .set_entry("worker")
        .compile()
    )
    parent = (
        GraphBuilder(_S)
        .add_fan_out_node(
            "fan",
            subgraph=inner,
            count=3,
            collect_field="result",
            target_field="worker_results",
        )
        .add_edge("fan", END)
        .set_entry("fan")
        .compile()
    )
    graph, client, _ = _attach_with_detached(parent, detached_fan_outs=frozenset({"fan"}))

    await graph.invoke(_S())
    await graph.drain()

    # Main Trace + one detached Trace per instance.
    assert len(client.traces) == 1 + 3
    main = next(t for t in client.traces.values() if "detached_from_invocation_id" not in t.metadata)
    detached_traces = [t for t in client.traces.values() if "detached_from_invocation_id" in t.metadata]
    assert len(detached_traces) == 3

    fan_node = _find_observation(main, "fan")
    # The fan-out node's metadata accumulates all 3 detached trace ids.
    link_ids = fan_node.metadata.get("detached_child_trace_ids")
    assert isinstance(link_ids, list)
    assert set(cast("list[str]", link_ids)) == {t.id for t in detached_traces}

    # Each detached Trace has its own per-instance dispatch with a
    # worker observation under it.
    for t in detached_traces:
        dispatch = _find_observation(t, "fan")
        worker = _find_observation(t, "worker")
        assert worker.parent_observation_id == dispatch.id
