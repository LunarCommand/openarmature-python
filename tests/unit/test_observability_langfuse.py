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


def test_observer_force_flush_delegates_to_client() -> None:
    # LangfuseObserver.force_flush() calls into the client; the
    # InMemoryLangfuseClient's force_flush is a no-op that returns
    # True, so this just verifies the delegation wires correctly.
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    assert observer.force_flush() is True
    assert observer.force_flush(timeout_ms=1000) is True


def test_in_memory_recorder_force_flush_is_no_op() -> None:
    # In-memory recorder has no outbound buffer; force_flush returns
    # True immediately. The timeout_ms parameter is accepted for
    # Protocol compatibility but unused.
    client = InMemoryLangfuseClient()
    assert client.force_flush() is True
    assert client.force_flush(timeout_ms=5000) is True


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


async def test_entry_node_resolves_to_wrapper_when_entry_is_subgraph() -> None:
    # When the outer entry IS a SubgraphNode, the first event the
    # observer sees comes from inside the subgraph
    # (event.namespace = (wrapper, inner), event.node_name = inner).
    # `entry_node` and trace.name MUST resolve to the wrapper node
    # name (event.namespace[0]), not the inner node name.
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
    assert trace.name == "sub", f"trace name should be the wrapper, got {trace.name!r}"
    assert trace.metadata.get("entry_node") == "sub", (
        f"entry_node should be the wrapper, got {trace.metadata.get('entry_node')!r}"
    )


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


async def test_detached_subgraph_subgraph_name_placement() -> None:
    # Per coord thread `discuss-observability-langfuse-mapping` msg 07
    # and the wrapper-role-migration framing: in detached mode the
    # wrapper role migrates to the detached trace. The parent trace's
    # link observation IS the SubgraphNode span (no wrapper role) and
    # MUST NOT carry `subgraph_name`. The detached trace's dispatch
    # observation IS the migrated wrapper and MUST carry it.
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

    main = next(t for t in client.traces.values() if "detached_from_invocation_id" not in t.metadata)
    detached = next(t for t in client.traces.values() if "detached_from_invocation_id" in t.metadata)

    link_obs = _find_observation(main, "sub")
    assert "subgraph_name" not in link_obs.metadata, (
        f"link observation MUST NOT carry subgraph_name; got {link_obs.metadata!r}"
    )

    detached_dispatch = _find_observation(detached, "sub")
    assert "subgraph_name" in detached_dispatch.metadata, (
        f"detached dispatch MUST carry subgraph_name; got {detached_dispatch.metadata!r}"
    )


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
    assert set(cast(list[str], link_ids)) == {t.id for t in detached_traces}

    # Each detached Trace has its own per-instance dispatch with a
    # worker observation under it.
    for t in detached_traces:
        dispatch = _find_observation(t, "fan")
        worker = _find_observation(t, "worker")
        assert worker.parent_observation_id == dispatch.id


async def test_subgraph_dispatch_observation_ended_on_invocation_close() -> None:
    # Synthetic dispatch observations close on cursor-move; without
    # the close_invocation drain a subgraph at the tail of an
    # invocation would leave its dispatch in-flight forever. Verifies
    # the drain path ends everything.
    inner = (
        GraphBuilder(_S)
        .add_node("inner_a", lambda _s: _record("inner_a"))
        .add_edge("inner_a", END)
        .set_entry("inner_a")
        .compile()
    )
    parent = GraphBuilder(_S).add_subgraph_node("sub", inner).add_edge("sub", END).set_entry("sub").compile()
    graph, client, observer = _attach(parent)

    await graph.invoke(_S())
    await graph.drain()
    # Without explicit close_invocation, the sub dispatch would still
    # be in-flight (ended=False). Call shutdown() to drain.
    observer.shutdown()

    trace = next(iter(client.traces.values()))
    for obs in trace.observations:
        assert obs.ended, f"observation {obs.name!r} not ended after shutdown()"


# ---------------------------------------------------------------------------
# §3.4 mid-invocation augmentation (proposal 0040)
# ---------------------------------------------------------------------------


class _AugmentState(State):
    answer: str = ""


async def test_metadata_augmentation_updates_trace_and_node_for_outermost() -> None:
    # Spec §3.4 MUST + proposal 0040 §6: an outermost-serial
    # ``set_invocation_metadata`` call MUST update both the open Trace
    # (via client.update_trace, surfacing the entries on
    # trace.metadata.<key> for §8.4 top-level filtering) AND the
    # calling node's open observation (via handle.update(metadata=)).
    # Mirrors fixture 034's Langfuse expectations.
    from openarmature.observability.metadata import set_invocation_metadata

    async def node_augments(_s: _AugmentState) -> dict[str, str]:
        set_invocation_metadata(request_id="req-xyz")
        return {"answer": "ok"}

    g = (
        GraphBuilder(_AugmentState)
        .add_node("ask", node_augments)
        .add_edge("ask", END)
        .set_entry("ask")
        .compile()
    )
    graph, client, observer = _attach(g)
    try:
        await graph.invoke(_AugmentState())
        await graph.drain()
    finally:
        observer.shutdown()

    trace = next(iter(client.traces.values()))
    # Trace metadata: augmented key landed on the open Trace.
    assert trace.metadata.get("request_id") == "req-xyz"
    # Calling node's observation: augmented key landed via in-place
    # update before the observation closed.
    ask_obs = _find_observation(trace, "ask")
    assert ask_obs.metadata.get("request_id") == "req-xyz"


async def test_metadata_augmentation_in_fan_out_isolates_per_instance() -> None:
    # Fixture 029-shaped: each fan-out instance augments metadata with
    # its own product_id. The Trace MUST NOT carry any product_id
    # (it's shared across siblings); the per-instance dispatch
    # observation AND the inner ask observation for each instance
    # MUST carry that instance's OWN product_id.
    import asyncio

    from openarmature.observability.correlation import current_fan_out_index
    from openarmature.observability.metadata import set_invocation_metadata

    class _ParentState(State):
        products: list[dict[str, str]] = []
        results: list[str] = []

    class _ChildState(State):
        product: dict[str, str] = {}
        out: str = ""

    async def _ask(s: _ChildState) -> dict[str, str]:
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
        .compile()
    )
    graph, client, observer = _attach(parent)
    try:
        products = [{"id": "prod-A"}, {"id": "prod-B"}, {"id": "prod-C"}]
        await graph.invoke(_ParentState(products=products))
        await graph.drain()
    finally:
        observer.shutdown()

    trace = next(iter(client.traces.values()))
    # Trace metadata MUST NOT carry per-instance product_id (sibling
    # isolation — fixture 029's central invariant).
    assert "product_id" not in trace.metadata, (
        f"per-instance augmentation leaked onto Trace metadata: {trace.metadata}"
    )
    # Each per-instance dispatch observation carries ITS OWN product_id.
    instance_obs = [
        obs for obs in trace.observations if obs.name == "fan" and "fan_out_index" in obs.metadata
    ]
    assert len(instance_obs) == 3
    seen_dispatch: dict[int, str] = {}
    for obs in instance_obs:
        fan_idx_value = obs.metadata.get("fan_out_index")
        product_id_value = obs.metadata.get("product_id")
        assert isinstance(fan_idx_value, int)
        assert isinstance(product_id_value, str)
        seen_dispatch[fan_idx_value] = product_id_value
    assert seen_dispatch == {0: "prod-A", 1: "prod-B", 2: "prod-C"}


async def test_metadata_augmentation_outside_invocation_is_silent() -> None:
    # Plumbing safety: no invocation in scope means no dispatch and no
    # observer event — set_invocation_metadata is a Context-only
    # mutation. The Langfuse handler is never called in this path so
    # no client / no Trace state is created.
    from openarmature.observability.metadata import set_invocation_metadata

    set_invocation_metadata(local_key="local_value")


async def test_metadata_augmentation_no_op_when_no_entries() -> None:
    # Direct-call safety: an augmentation event with empty entries
    # should be a no-op on the observer side.
    from openarmature.graph.events import MetadataAugmentationEvent

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    event = MetadataAugmentationEvent(
        entries={},
        namespace=("ask",),
        attempt_index=0,
        fan_out_index=None,
        branch_name=None,
    )
    observer._handle_metadata_augmentation(event)  # noqa: SLF001
    # No Trace was opened (no invocation in scope) and no exception.
    assert client.traces == {}
