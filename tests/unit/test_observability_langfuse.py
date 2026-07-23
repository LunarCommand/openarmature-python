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


def test_in_memory_trace_records_session_and_user_id() -> None:
    # Proposal 0064 §8.4.1: trace(session_id=, user_id=) populates the two
    # Langfuse cross-trace grouping fields. This exercises the session_id
    # plumbing the observer leaves dormant until 0020 (so the deferred
    # fixture-084 session cases are still covered at the client layer).
    client = InMemoryLangfuseClient()
    client.trace(id="t1", name="a", metadata={"userId": "u-7"}, session_id="sess-9", user_id="u-7")
    trace = client.traces["t1"]
    assert trace.session_id == "sess-9"
    assert trace.user_id == "u-7"
    # Additive: userId also remains in the metadata bag.
    assert trace.metadata["userId"] == "u-7"
    # Both default to None when not supplied.
    client.trace(id="t2", name="b", metadata={})
    assert client.traces["t2"].session_id is None
    assert client.traces["t2"].user_id is None


def test_promoted_user_id_recognizes_userid_key() -> None:
    # Proposal 0064 §8.4.1: the userId promotion reads a recognized key,
    # coerces to str, and is None when absent.
    from openarmature.observability.langfuse.observer import _promoted_user_id

    assert _promoted_user_id({"userId": "u-1"}) == "u-1"
    assert _promoted_user_id({"userId": 42}) == "42"
    assert _promoted_user_id({"tenantId": "acme"}) is None
    assert _promoted_user_id({}) is None


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


# ---------------------------------------------------------------------------
# Trace input/output sourcing (proposal 0043 §8.4.1)
# ---------------------------------------------------------------------------


class _S0043(State):
    msg: str = ""


async def _emit_node(_s: _S0043) -> dict[str, Any]:
    return {"msg": "ok"}


def _build_0043_graph() -> Any:
    return GraphBuilder(_S0043).add_node("a", _emit_node).add_edge("a", END).set_entry("a").compile()


async def test_trace_input_output_default_emits_minimal_stub() -> None:
    # Lever 3 (default). `disable_state_payload` defaults ON; no hooks
    # supplied. trace.input = {entry_node, correlation_id};
    # trace.output = {final_node, status}.
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    graph = _build_0043_graph()
    graph.attach_observer(observer)
    await graph.invoke(_S0043(), correlation_id="corr-1")
    await graph.drain()

    trace = next(iter(client.traces.values()))
    assert trace.input == {"entry_node": "a", "correlation_id": "corr-1"}
    assert trace.output == {"final_node": "a", "status": "completed"}


async def test_trace_input_output_disable_state_payload_off_emits_raw_state() -> None:
    # Lever 2. `disable_state_payload=False`; no hooks. trace.input
    # and trace.output carry the serialized state.
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_state_payload=False)
    graph = _build_0043_graph()
    graph.attach_observer(observer)
    await graph.invoke(_S0043())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    # ``input`` reflects initial_state, ``output`` reflects final state.
    assert trace.input == {"msg": ""}
    assert trace.output == {"msg": "ok"}


async def test_trace_input_output_handles_non_json_native_state_fields() -> None:
    # Regression for the PR #99 copilot finding: pydantic's
    # ``model_dump()`` defaults to Python mode and leaves
    # ``datetime`` / ``UUID`` / ``Decimal`` as Python objects. The
    # downstream truncation path calls ``json.dumps`` without a
    # ``default``, which raises ``TypeError`` on those types. The
    # observer raise is swallowed by the engine's warnings-only
    # observer-isolation contract, leaving trace.input / trace.output
    # silently blank.
    #
    # ``_state_to_jsonable`` MUST call ``model_dump(mode="json")`` so
    # these types serialize to their JSON-compatible string forms
    # before the truncation step.
    import uuid
    from datetime import UTC, datetime
    from decimal import Decimal

    class _DateState(State):
        when: datetime = datetime(2026, 5, 29, 12, 0, 0, tzinfo=UTC)
        request_id: uuid.UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
        amount: Decimal = Decimal("99.99")

    async def _noop(_s: _DateState) -> dict[str, Any]:
        return {}

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_state_payload=False)
    graph = GraphBuilder(_DateState).add_node("a", _noop).add_edge("a", END).set_entry("a").compile()
    graph.attach_observer(observer)
    await graph.invoke(_DateState())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    # The non-JSON-native types serialize to JSON-compatible strings.
    # Both trace.input and trace.output land successfully (the bug
    # would leave them as ``None``).
    assert trace.input is not None, "trace.input should not be blank on State with datetime/UUID/Decimal"
    assert trace.output is not None
    trace_input = cast("dict[str, Any]", trace.input)
    assert trace_input["when"] == "2026-05-29T12:00:00Z"
    assert trace_input["request_id"] == "12345678-1234-5678-1234-567812345678"
    # Decimal serializes to its string form under ``mode="json"``.
    assert trace_input["amount"] == "99.99"


async def test_trace_input_output_caller_hooks_replace_stub() -> None:
    # Lever 1. Caller hooks supplied, returning non-None domain
    # summaries. Hook return values appear on the trace fields verbatim;
    # the stub does NOT appear; `disable_state_payload` is irrelevant.
    client = InMemoryLangfuseClient()

    def input_hook(state: _S0043) -> dict[str, Any]:
        return {"summary": f"received msg={state.msg!r}"}

    def output_hook(state: _S0043) -> dict[str, Any]:
        return {"summary": f"final msg={state.msg!r}"}

    observer = LangfuseObserver(
        client=client,
        trace_input_from_state=input_hook,
        trace_output_from_state=output_hook,
    )
    graph = _build_0043_graph()
    graph.attach_observer(observer)
    await graph.invoke(_S0043())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    assert trace.input == {"summary": "received msg=''"}
    assert trace.output == {"summary": "final msg='ok'"}


async def test_trace_input_output_caller_hooks_return_none_falls_through() -> None:
    # Lever-1 null-fallthrough. Hooks supplied but return None;
    # observer falls through to the next applicable lever — lever 3
    # (stub) when disable_state_payload defaults ON.
    client = InMemoryLangfuseClient()

    def input_hook(_state: _S0043) -> None:
        return None

    def output_hook(_state: _S0043) -> None:
        return None

    observer = LangfuseObserver(
        client=client,
        trace_input_from_state=input_hook,
        trace_output_from_state=output_hook,
    )
    graph = _build_0043_graph()
    graph.attach_observer(observer)
    await graph.invoke(_S0043(), correlation_id="corr-2")
    await graph.drain()

    trace = next(iter(client.traces.values()))
    # Stub applies as if no hook had been supplied.
    assert trace.input == {"entry_node": "a", "correlation_id": "corr-2"}
    assert trace.output == {"final_node": "a", "status": "completed"}


class _FailState(State):
    x: int = 0


async def _raise_node(_s: _FailState) -> dict[str, Any]:
    raise RuntimeError("boom")


async def test_trace_output_status_failed_on_node_raise() -> None:
    # Failure path: `status` enum closed on {completed, failed}. A
    # raise inside the node body fires the InvocationCompletedEvent
    # with status="failed" and final_node set to the raising node.
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)

    graph = (
        GraphBuilder(_FailState)
        .add_node("raises", _raise_node)
        .add_edge("raises", END)
        .set_entry("raises")
        .compile()
    )
    graph.attach_observer(observer)

    # Spec §4: node-raised exceptions surface as NodeException
    # (the runtime category that wraps node body raises).
    from openarmature.graph.errors import NodeException

    with pytest.raises(NodeException, match="raises"):
        await graph.invoke(_FailState())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    assert trace.output == {"final_node": "raises", "status": "failed"}


class _PartialFailState(State):
    a_ran: bool = False
    b_ran: bool = False


async def _node_a_succeeds(_s: _PartialFailState) -> dict[str, Any]:
    return {"a_ran": True}


async def _node_b_raises(_s: _PartialFailState) -> dict[str, Any]:
    raise RuntimeError("node_b boom")


async def test_failure_path_final_state_is_state_at_failure_point() -> None:
    # Spec §8.4.1 *Resume semantics* + the proposal-0043 "partial final
    # state captured at the failure point" clause:  a graph that
    # completes node_a successfully then raises in node_b MUST surface
    # the post-node-a state on the InvocationCompletedEvent so the
    # ``trace_output_from_state`` hook (and the raw-state lever) see
    # the partial state, not the bare initial state.  Pins the engine
    # fix that surfaces ``latest_state_box`` on the failure path.

    captured_output_state: list[_PartialFailState] = []

    def output_hook(state: _PartialFailState) -> dict[str, Any]:
        captured_output_state.append(state)
        return {"a_ran": state.a_ran, "b_ran": state.b_ran}

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, trace_output_from_state=output_hook)
    graph = (
        GraphBuilder(_PartialFailState)
        .add_node("node_a", _node_a_succeeds)
        .add_node("node_b", _node_b_raises)
        .add_edge("node_a", "node_b")
        .add_edge("node_b", END)
        .set_entry("node_a")
        .compile()
    )
    graph.attach_observer(observer)

    from openarmature.graph.errors import NodeException

    with pytest.raises(NodeException, match="node_b"):
        await graph.invoke(_PartialFailState())
    await graph.drain()

    # The output hook fired with the post-node-a state (a_ran=True),
    # not the initial state (a_ran=False).
    assert len(captured_output_state) == 1
    assert captured_output_state[0].a_ran is True
    assert captured_output_state[0].b_ran is False
    trace = next(iter(client.traces.values()))
    assert trace.output == {"a_ran": True, "b_ran": False}


class _OuterFailState(State):
    outer_a_done: bool = False
    sub_done: bool = False


class _InnerFailState(State):
    inner_x_done: bool = False


async def _outer_node_a(_s: _OuterFailState) -> dict[str, Any]:
    return {"outer_a_done": True}


async def _inner_node_x_succeeds(_s: _InnerFailState) -> dict[str, Any]:
    return {"inner_x_done": True}


async def _inner_node_y_raises(_s: _InnerFailState) -> dict[str, Any]:
    raise RuntimeError("inner_node_y boom")


async def test_failure_path_final_state_is_outer_type_when_subgraph_raises() -> None:
    # Engine-bug regression: an inner-subgraph step's success previously
    # overwrote the outermost ``latest_state_box`` (it was shared by
    # reference across subgraph descents), so a subgraph-internal raise
    # would leave the box holding an INNER state at outer ``invoke()``
    # finally time.  The outer ``trace_output_from_state`` hook would
    # then receive an inner-typed state when its signature expects the
    # outer type — a real correctness bug.
    #
    # The box is now per-context: each subgraph descent gets its own
    # fresh ``latest_state_box``, so the outermost level's box holds
    # only outer-state-typed entries.  This test exercises a graph
    # where outer node_a succeeds (outer state = a_done=true), the
    # subgraph step raises inside, and the outer trace.output hook
    # receives the outer state with ``outer_a_done=True``,
    # ``sub_done=False``.
    from openarmature.graph import ExplicitMapping

    inner_graph = (
        GraphBuilder(_InnerFailState)
        .add_node("inner_x", _inner_node_x_succeeds)
        .add_node("inner_y", _inner_node_y_raises)
        .add_edge("inner_x", "inner_y")
        .add_edge("inner_y", END)
        .set_entry("inner_x")
        .compile()
    )

    captured_output_state: list[Any] = []

    def output_hook(state: Any) -> dict[str, Any]:
        captured_output_state.append(state)
        return {"outer_a_done": state.outer_a_done, "sub_done": state.sub_done}

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, trace_output_from_state=output_hook)
    graph = (
        GraphBuilder(_OuterFailState)
        .add_node("outer_a", _outer_node_a)
        .add_subgraph_node(
            "sub",
            inner_graph,
            projection=ExplicitMapping(inputs=None, outputs={"sub_done": "inner_x_done"}),
        )
        .add_edge("outer_a", "sub")
        .add_edge("sub", END)
        .set_entry("outer_a")
        .compile()
    )
    graph.attach_observer(observer)

    from openarmature.graph.errors import NodeException

    with pytest.raises(NodeException):
        await graph.invoke(_OuterFailState())
    await graph.drain()

    # The hook receives the OUTER state (with outer_a_done=True,
    # sub_done=False), not the inner state — confirming the box's
    # per-level isolation worked.
    assert len(captured_output_state) == 1
    assert isinstance(captured_output_state[0], _OuterFailState)
    assert not isinstance(captured_output_state[0], _InnerFailState)
    assert captured_output_state[0].outer_a_done is True
    assert captured_output_state[0].sub_done is False
    trace = next(iter(client.traces.values()))
    assert trace.output == {"outer_a_done": True, "sub_done": False}


# ---------------------------------------------------------------------------
# Per-context box isolation across fan-out + parallel-branches descents
# ---------------------------------------------------------------------------


class _FanOutOuterState(State):
    outer_a_done: bool = False
    items: list[int] = []
    results: Annotated[list[int], append] = []


class _FanOutInnerState(State):
    item: int = 0
    out: int = 0
    inner_done: bool = False


async def _fan_out_inner_succeeds(_s: _FanOutInnerState) -> dict[str, Any]:
    # Successful inner step — writes ``inner_done=true`` to the
    # instance's _invoke ``state`` local AND to the shared
    # ``latest_state_box`` (per-context, so it lands on the instance's
    # OWN box).  Under the original shared-box bug this write would
    # leak into the outer box; under the per-context design it stays
    # isolated to the instance.
    return {"inner_done": True}


async def _fan_out_inner_raises(_s: _FanOutInnerState) -> dict[str, Any]:
    raise RuntimeError("fan_out inner_raise boom")


async def test_failure_path_final_state_is_outer_type_when_fan_out_inner_raises() -> None:
    # Sibling to the subgraph-raise test: pins the per-context
    # ``latest_state_box`` isolation across a fan-out instance descent.
    # Each fan-out instance gets its own ``_InvocationContext``
    # (descend_into_fan_out_instance), so its inner step writes land on
    # the instance's own box, not the outer box.  When the instance
    # raises, the outermost ``invoke()``'s finally-block reads the
    # OUTER box — which holds outer state from ``outer_a``'s successful
    # completion, not the inner instance state.
    #
    # The inner subgraph has TWO inner nodes: ``inner_succeeds`` writes
    # inner state to the instance's box, then ``inner_raises``
    # propagates.  Under the original shared-box bug, the box would
    # end with ``_FanOutInnerState(inner_done=true)`` and the outer
    # hook would receive that inner-typed value.  The two-node shape
    # is load-bearing — a single-node "always raise" subgraph would
    # not exercise the leak because no successful inner step would
    # write to the box.
    inner_graph = (
        GraphBuilder(_FanOutInnerState)
        .add_node("inner_succeeds", _fan_out_inner_succeeds)
        .add_node("inner_raises", _fan_out_inner_raises)
        .add_edge("inner_succeeds", "inner_raises")
        .add_edge("inner_raises", END)
        .set_entry("inner_succeeds")
        .compile()
    )

    async def _outer_a(_s: _FanOutOuterState) -> dict[str, Any]:
        return {"outer_a_done": True}

    captured_output_state: list[Any] = []

    def output_hook(state: Any) -> dict[str, Any]:
        captured_output_state.append(state)
        return {"outer_a_done": state.outer_a_done, "results": list(state.results)}

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, trace_output_from_state=output_hook)
    graph = (
        GraphBuilder(_FanOutOuterState)
        .add_node("outer_a", _outer_a)
        .add_fan_out_node(
            "fan",
            subgraph=inner_graph,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
        )
        .add_edge("outer_a", "fan")
        .add_edge("fan", END)
        .set_entry("outer_a")
        .compile()
    )
    graph.attach_observer(observer)

    from openarmature.graph.errors import RuntimeGraphError

    with pytest.raises(RuntimeGraphError):
        # Three fan-out instances all fail; the engine raises after the
        # fan-out node completes (fail_fast default).
        await graph.invoke(_FanOutOuterState(items=[1, 2, 3]))
    await graph.drain()

    # The hook receives the OUTER state (FanOutOuterState), not an
    # inner FanOutInnerState from the failed instance descent.
    assert len(captured_output_state) == 1
    assert isinstance(captured_output_state[0], _FanOutOuterState)
    assert not isinstance(captured_output_state[0], _FanOutInnerState)
    assert captured_output_state[0].outer_a_done is True
    # No instance succeeded, so results stays empty.
    assert list(captured_output_state[0].results) == []


class _ParBrOuterState(State):
    outer_a_done: bool = False
    branch_x_done: bool = False
    branch_y_done: bool = False


class _ParBrBranchXState(State):
    x_done: bool = False


class _ParBrBranchYState(State):
    y_done: bool = False


async def _par_br_branch_x_succeeds(_s: _ParBrBranchXState) -> dict[str, Any]:
    return {"x_done": True}


async def _par_br_branch_y_raises(_s: _ParBrBranchYState) -> dict[str, Any]:
    raise RuntimeError("parallel_branches branch_y boom")


async def test_failure_path_final_state_is_outer_type_when_parallel_branch_raises() -> None:
    # Sibling to the subgraph + fan-out tests: pins per-context
    # ``latest_state_box`` isolation across a parallel-branches
    # descent.  Each branch's inner _invoke runs in its own
    # ``_InvocationContext`` (descend_into_parallel_branch), so inner
    # writes don't leak to the outer box.  Even when branch_x writes
    # its inner state successfully, the outermost finally-block reads
    # the OUTER box on the branch_y-induced raise.
    from openarmature.graph import BranchSpec

    branch_x_subgraph = (
        GraphBuilder(_ParBrBranchXState)
        .add_node("succeeds", _par_br_branch_x_succeeds)
        .add_edge("succeeds", END)
        .set_entry("succeeds")
        .compile()
    )

    branch_y_subgraph = (
        GraphBuilder(_ParBrBranchYState)
        .add_node("raises", _par_br_branch_y_raises)
        .add_edge("raises", END)
        .set_entry("raises")
        .compile()
    )

    async def _outer_a(_s: _ParBrOuterState) -> dict[str, Any]:
        return {"outer_a_done": True}

    captured_output_state: list[Any] = []

    def output_hook(state: Any) -> dict[str, Any]:
        captured_output_state.append(state)
        return {
            "outer_a_done": state.outer_a_done,
            "branch_x_done": state.branch_x_done,
            "branch_y_done": state.branch_y_done,
        }

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, trace_output_from_state=output_hook)
    graph = (
        GraphBuilder(_ParBrOuterState)
        .add_node("outer_a", _outer_a)
        .add_parallel_branches_node(
            "dispatch",
            branches={
                "branch_x": BranchSpec(
                    subgraph=branch_x_subgraph,
                    outputs={"branch_x_done": "x_done"},
                ),
                "branch_y": BranchSpec(
                    subgraph=branch_y_subgraph,
                    outputs={"branch_y_done": "y_done"},
                ),
            },
        )
        .add_edge("outer_a", "dispatch")
        .add_edge("dispatch", END)
        .set_entry("outer_a")
        .compile()
    )
    graph.attach_observer(observer)

    from openarmature.graph.errors import RuntimeGraphError

    with pytest.raises(RuntimeGraphError):
        await graph.invoke(_ParBrOuterState())
    await graph.drain()

    # The hook receives the OUTER state (ParBrOuterState).  Whether
    # branch_x's success projected back into the outer state by the
    # time of the raise depends on the dispatch's join semantics;
    # what MUST be true is that the captured state is the OUTER
    # type, not branch_x's _ParBrBranchXState or branch_y's
    # _ParBrBranchYState.
    assert len(captured_output_state) == 1
    assert isinstance(captured_output_state[0], _ParBrOuterState)
    assert not isinstance(captured_output_state[0], _ParBrBranchXState)
    assert not isinstance(captured_output_state[0], _ParBrBranchYState)
    assert captured_output_state[0].outer_a_done is True


async def test_parallel_branches_node_renders_no_duplicate_observation() -> None:
    # Regression: a parallel-branches NODE emits its own started/completed
    # pair, so it already has a leaf observation. The observer MUST NOT also
    # synthesize a duplicate subgraph-wrapper observation at the node's
    # namespace (the bug the OTel observer already guards against, now
    # mirrored here). Each callable branch (proposal 0075) renders as a
    # single observation parented under the one NODE observation.
    from openarmature.graph import BranchSpec

    async def vector(_s: _S) -> Any:
        return {"trail": ["vector"]}

    async def keyword(_s: _S) -> Any:
        return {"trail": ["keyword"]}

    graph = (
        GraphBuilder(_S)
        .add_parallel_branches_node(
            "recall",
            branches={
                "vector": BranchSpec(call=vector),
                "keyword": BranchSpec(call=keyword),
            },
        )
        .add_edge("recall", END)
        .set_entry("recall")
        .compile()
    )
    graph, client, _ = _attach(graph)
    await graph.invoke(_S())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    recall_obs = [o for o in trace.observations if o.name == "recall"]
    assert len(recall_obs) == 1, f"expected one 'recall' observation, got {len(recall_obs)}"
    node_id = recall_obs[0].id
    for branch in ("vector", "keyword"):
        branch_obs = [o for o in trace.observations if o.name == branch]
        assert len(branch_obs) == 1, f"branch {branch!r}: expected one observation, got {len(branch_obs)}"
        assert branch_obs[0].parent_observation_id == node_id
        assert (branch_obs[0].metadata or {}).get("branch_name") == branch


async def test_parallel_branches_subgraph_branch_one_dispatch_observation() -> None:
    # A subgraph branch with multiple inner nodes synthesizes exactly ONE
    # per-branch dispatch observation; both inner nodes parent under it (not a
    # fresh dispatch per inner started event). Guards the proposal-0044
    # synthesis idempotency.
    from openarmature.graph import BranchSpec

    class _MultiBranchState(State):
        a: str = ""
        b: str = ""

    class _MultiParentState(State):
        out: str = ""

    async def _node_a(_s: _MultiBranchState) -> dict[str, Any]:
        return {"a": "a"}

    async def _node_b(_s: _MultiBranchState) -> dict[str, Any]:
        return {"b": "b"}

    branch = (
        GraphBuilder(_MultiBranchState)
        .add_node("node_a", _node_a)
        .add_node("node_b", _node_b)
        .add_edge("node_a", "node_b")
        .add_edge("node_b", END)
        .set_entry("node_a")
        .compile()
    )
    graph = (
        GraphBuilder(_MultiParentState)
        .add_parallel_branches_node(
            "dispatch",
            branches={"only": BranchSpec(subgraph=branch, outputs={"out": "b"})},
        )
        .add_edge("dispatch", END)
        .set_entry("dispatch")
        .compile()
    )
    graph, client, _ = _attach(graph)
    await graph.invoke(_MultiParentState())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    branch_obs = [o for o in trace.observations if o.name == "only"]
    assert len(branch_obs) == 1, f"expected one per-branch observation, got {len(branch_obs)}"
    inner = [o for o in trace.observations if o.name in ("node_a", "node_b")]
    assert len(inner) == 2, f"expected two inner observations, got {len(inner)}"
    assert all(o.parent_observation_id == branch_obs[0].id for o in inner)


# Spec §8.4.1 / proposal 0052: implementation attribution rows on
# every Langfuse Trace. The two rows source from the §5.1
# attributes; the always-emit invariant inherits from §5.1 so the
# privacy knobs do not gate them.


async def test_trace_metadata_carries_implementation_attribution_rows() -> None:
    from openarmature import __version__

    graph = (
        GraphBuilder(_S)
        .add_node("entry", lambda _s: _record("entry"))
        .add_edge("entry", END)
        .set_entry("entry")
        .compile()
    )
    graph, client, _ = _attach(graph)

    await graph.invoke(_S())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    assert trace.metadata.get("implementation_name") == "openarmature-python"
    assert trace.metadata.get("implementation_version") == __version__
    # Non-empty-string contract per spec §5.1.
    assert isinstance(trace.metadata["implementation_name"], str)
    assert trace.metadata["implementation_name"]
    assert isinstance(trace.metadata["implementation_version"], str)
    assert trace.metadata["implementation_version"]


async def test_implementation_attribution_rows_emit_with_disable_state_payload_enabled() -> None:
    # Always-emit invariant: regardless of disable_state_payload (the
    # privacy knob that gates state payloads on trace.input /
    # trace.output), the implementation attribution rows MUST appear.
    # They describe runtime identity, not runtime data.
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_state_payload=True)

    graph = (
        GraphBuilder(_S)
        .add_node("entry", lambda _s: _record("entry"))
        .add_edge("entry", END)
        .set_entry("entry")
        .compile()
    )
    graph.attach_observer(observer)

    await graph.invoke(_S())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    assert "implementation_name" in trace.metadata
    assert "implementation_version" in trace.metadata
    assert trace.metadata["implementation_name"] == "openarmature-python"


# Spec §8.4.1 / proposal 0052: every Trace carries the attribution
# rows. An observer reused across multiple invocations on the same
# compiled graph MUST emit the rows on every Trace, not just the
# first. Mirrors the OTel-side test_invocation_span_attribution_
# emits_on_every_invocation contract.
async def test_implementation_attribution_rows_emit_on_every_trace() -> None:
    graph = (
        GraphBuilder(_S)
        .add_node("entry", lambda _s: _record("entry"))
        .add_edge("entry", END)
        .set_entry("entry")
        .compile()
    )
    graph, client, _ = _attach(graph)

    for _ in range(3):
        await graph.invoke(_S())
        await graph.drain()

    # Three invocations → three traces. Every one carries the rows.
    assert len(client.traces) == 3, f"expected three traces, got {len(client.traces)}"
    for trace in client.traces.values():
        assert trace.metadata.get("implementation_name") == "openarmature-python"
        assert isinstance(trace.metadata.get("implementation_version"), str)
        assert trace.metadata["implementation_version"]


# ---------------------------------------------------------------------------
# Typed LlmCompletionEvent handling (proposal 0049 + 0057, PR 3c)
# ---------------------------------------------------------------------------


async def test_typed_llm_event_emits_generation_with_expected_fields() -> None:
    # Happy-path: a single LlmCompletionEvent produces exactly one
    # Generation observation under the typed event's invocation_id
    # Trace, with model / usage / metadata sourced from the event.
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    # disable_provider_payload defaults to True per §8.9; flip it off here
    # so the test can also assert the payload (output) makes it through.
    observer = LangfuseObserver(client=client, disable_provider_payload=False)
    token = _set_invocation_id("inv-typed-1")
    try:
        await observer(
            make_typed_event(
                invocation_id="inv-typed-1",
                model="m-test",
                provider="vllm",
                usage=Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
                finish_reason="stop",
                response_id="resp-abc",
                response_model="m-test-001",
                output_content="hello",
                request_params={"temperature": 0.7},
            )
        )
    finally:
        _reset_invocation_id(token)

    assert "inv-typed-1" in client.traces
    trace = client.traces["inv-typed-1"]
    generations = [o for o in trace.observations if o.type == "generation"]
    assert len(generations) == 1
    obs = generations[0]
    assert obs.model == "m-test"
    assert obs.usage == LangfuseUsage(input=10, output=4, total=14)
    assert obs.model_parameters == {"temperature": 0.7}
    assert obs.output == "hello"
    assert obs.metadata.get("system") == "vllm"
    assert obs.metadata.get("finish_reason") == "stop"
    assert obs.metadata.get("response_id") == "resp-abc"
    assert obs.metadata.get("response_model") == "m-test-001"
    assert obs.ended is True


async def test_structured_output_failure_generation_renders_response_surface() -> None:
    # Proposal 0082: a structured_output_invalid failure renders the response-side
    # surface (output payload-gated, usage, metadata.finish_reason) on the ERROR
    # Generation, alongside level=ERROR + statusMessage=category.
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_provider_payload=False)
    token = _set_invocation_id("inv-soi")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-soi",
                error_category="structured_output_invalid",
                error_type="StructuredOutputInvalid",
                output_content='{"name":"Alice","age":',
                finish_reason="length",
                usage=Usage(prompt_tokens=20, completion_tokens=16, total_tokens=36),
                response_id="cc-xyz",
                response_model="gpt-test-v2",
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-soi"].observations if o.type == "generation")
    assert gen.level == "ERROR"
    assert gen.status_message == "structured_output_invalid"
    assert gen.output == '{"name":"Alice","age":'
    assert gen.usage == LangfuseUsage(input=20, output=16, total=36)
    assert gen.metadata.get("finish_reason") == "length"
    assert gen.metadata.get("response_model") == "gpt-test-v2"
    assert gen.metadata.get("response_id") == "cc-xyz"


async def test_structured_output_failure_generation_redacts_output_when_payload_disabled() -> None:
    # Payload-gated: with disable_provider_payload=True, output is redacted while
    # usage / metadata.finish_reason / ERROR level stay (proposal 0082).
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_provider_payload=True)
    token = _set_invocation_id("inv-soi2")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-soi2",
                error_category="structured_output_invalid",
                output_content='{"name":"Alice","age":',
                finish_reason="length",
                usage=Usage(prompt_tokens=20, completion_tokens=16, total_tokens=36),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-soi2"].observations if o.type == "generation")
    assert gen.level == "ERROR"
    assert gen.output is None
    assert gen.usage == LangfuseUsage(input=20, output=16, total=36)
    assert gen.metadata.get("finish_reason") == "length"


async def test_typed_llm_event_back_dates_generation_using_latency_ms() -> None:
    # Generation observation's start/end timestamps reflect the
    # adapter-boundary latency rather than the typed event's arrival
    # moment. Verify end - start matches latency_ms within tolerance.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    latency_ms = 250.0
    token = _set_invocation_id("inv-typed-dur")
    try:
        await observer(make_typed_event(invocation_id="inv-typed-dur", latency_ms=latency_ms))
    finally:
        _reset_invocation_id(token)

    trace = client.traces["inv-typed-dur"]
    obs = next(o for o in trace.observations if o.type == "generation")
    assert obs.start_time is not None and obs.end_time is not None
    duration_ms = (obs.end_time - obs.start_time).total_seconds() * 1000
    # Float arithmetic tolerance; the back-date should be near-exact
    # apart from microsecond rounding.
    assert abs(duration_ms - latency_ms) < 1.0


async def test_typed_llm_event_zero_duration_when_latency_missing() -> None:
    # latency_ms=None falls back to a zero-duration Generation at
    # end_time. Mirrors the OTel path.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-typed-no-latency")
    try:
        await observer(make_typed_event(invocation_id="inv-typed-no-latency", latency_ms=None))
    finally:
        _reset_invocation_id(token)

    trace = client.traces["inv-typed-no-latency"]
    obs = next(o for o in trace.observations if o.type == "generation")
    assert obs.start_time is not None and obs.end_time is not None
    assert obs.start_time == obs.end_time


async def test_typed_llm_event_drops_silently_outside_invocation() -> None:
    # Without an invocation id ContextVar set, the typed handler
    # MUST early-return without emitting a Generation. Symmetric with
    # the OTel observer.
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    await observer(make_typed_event())
    assert client.traces == {}


async def test_disable_llm_spans_skips_typed_event_path() -> None:
    # disable_llm_spans MUST gate the typed-event handler.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_llm_spans=True)
    token = _set_invocation_id("inv-disabled")
    try:
        await observer(make_typed_event(invocation_id="inv-disabled"))
    finally:
        _reset_invocation_id(token)
    assert client.traces == {}


async def test_llm_error_path_emits_error_generation_from_typed_failed_event() -> None:
    # Per proposal 0058: failures emit a typed LlmFailedEvent. The
    # Langfuse observer drives the Generation observation with ERROR
    # level + error_category as statusMessage.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-err")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-err",
                model="m-test",
                error_category="provider_rate_limit",
                error_type="ProviderRateLimit",
                error_message="429 from upstream",
                call_id="cc-err",
            )
        )
    finally:
        _reset_invocation_id(token)

    trace = client.traces["inv-err"]
    obs = next(o for o in trace.observations if o.type == "generation")
    assert obs.level == "ERROR"
    assert obs.status_message == "provider_rate_limit"


async def test_typed_failed_event_parents_under_branch_calling_node() -> None:
    # Regression cover for the _resolve_llm_parent_observation_id
    # keyword-only signature: when a typed LlmFailedEvent fires
    # inside a parallel-branches branch, the resulting ERROR
    # Generation MUST parent under THAT branch's calling node
    # observation, not under a sibling branch's same-named node.
    # Pre-populates the observer's internal state with two open
    # node observations that differ only by branch_name, then
    # dispatches a typed LlmFailedEvent with the matching
    # branch_name and asserts the parent_observation_id points at
    # the right one.
    #
    # Note: the same _resolve_llm_parent_observation_id call also
    # serves the success-path handler with calling_branch_name =
    # event.branch_name; failure- and success-paths share the
    # resolver so this test transitively covers the success-path
    # branch_name handling.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.observability.langfuse.observer import (
        _InvState,
        _OpenObservation,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    invocation_id = "inv-pb-err"
    token = _set_invocation_id(invocation_id)
    try:
        # Bootstrap the Trace + two branch-distinguished node
        # observations directly. _InvState's open_observations map is
        # keyed by (namespace, attempt_index, fan_out_index,
        # branch_name); the calling node identity on the typed event
        # is (("dispatcher", "ask"), 0, None, "fast").
        client.trace(id=invocation_id, name="dispatcher")
        observer._inv_states[invocation_id] = _InvState(trace_id=invocation_id)  # noqa: SLF001
        inv_state = observer._inv_states[invocation_id]  # noqa: SLF001
        # Open two observations under the trace — one per branch.
        fast_handle = client.generation(trace_id=invocation_id, name="ask", model="m-test")
        slow_handle = client.generation(trace_id=invocation_id, name="ask", model="m-test")
        fast_key = (("dispatcher", "ask"), 0, None, "fast")
        slow_key = (("dispatcher", "ask"), 0, None, "slow")
        inv_state.open_observations[fast_key] = _OpenObservation(handle=fast_handle)
        inv_state.open_observations[slow_key] = _OpenObservation(handle=slow_handle)
        await observer(
            make_failed_event(
                invocation_id=invocation_id,
                node_name="ask",
                namespace=("dispatcher", "ask"),
                attempt_index=0,
                fan_out_index=None,
                branch_name="fast",
                model="m-test",
                error_category="provider_unavailable",
                error_type="ProviderUnavailable",
                error_message="503 from upstream",
                call_id="cc-pb",
            )
        )
    finally:
        _reset_invocation_id(token)

    trace = client.traces[invocation_id]
    # Three observations now: two synthetic "ask" + one error
    # Generation. The error Generation MUST parent under fast_handle,
    # not slow_handle.
    error_gens = [o for o in trace.observations if o.type == "generation" and o.level == "ERROR"]
    assert len(error_gens) == 1
    assert error_gens[0].parent_observation_id == fast_handle.id
    assert error_gens[0].parent_observation_id != slow_handle.id


async def test_llm_event_parents_under_fan_out_instance_dispatch() -> None:
    # Regression cover for _resolve_llm_parent_observation_id fallback #2: when an
    # LLM event fires inside a top-level fan-out instance and the calling node has
    # no open observation (fallback #1 misses), the Generation MUST parent under
    # the per-instance fan-out dispatch observation. The dispatch map is keyed by
    # the lineage-aware _dispatch_key; before the lineage keys this fallback used
    # a flat namespace[:1] + (str(index),) key, which always-misses against the
    # composite map and silently re-parents the Generation under the Trace.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.observability.langfuse.observer import (
        _dispatch_key,
        _InvState,
        _OpenObservation,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    invocation_id = "inv-fanout-llm"
    token = _set_invocation_id(invocation_id)
    try:
        client.trace(id=invocation_id, name="fan")
        observer._inv_states[invocation_id] = _InvState(trace_id=invocation_id)  # noqa: SLF001
        inv_state = observer._inv_states[invocation_id]  # noqa: SLF001
        # Per-instance dispatch for top-level fan-out "fan", instance 0, keyed by
        # the lineage-aware _dispatch_key. No open_observation for the calling
        # node ("fan", "ask"), so the resolver must reach fallback #2.
        dispatch_handle = client.span(trace_id=invocation_id, name="fan")
        instance_key = _dispatch_key(("fan",), (0,), (None,))
        inv_state.fan_out_instance_observations[instance_key] = _OpenObservation(handle=dispatch_handle)
        await observer(
            make_failed_event(
                invocation_id=invocation_id,
                node_name="ask",
                namespace=("fan", "ask"),
                attempt_index=0,
                fan_out_index=0,
                branch_name=None,
                model="m-test",
                error_category="provider_unavailable",
                error_type="ProviderUnavailable",
                error_message="503 from upstream",
                call_id="cc-fan",
            )
        )
    finally:
        _reset_invocation_id(token)

    trace = client.traces[invocation_id]
    error_gens = [o for o in trace.observations if o.type == "generation" and o.level == "ERROR"]
    assert len(error_gens) == 1
    assert error_gens[0].parent_observation_id == dispatch_handle.id, (
        "LLM Generation must parent under the per-instance fan-out dispatch (resolver fallback #2)"
    )


# ---------------------------------------------------------------------------
# Proposal 0063 — tool-execution Tool observation (asType "tool")
# ---------------------------------------------------------------------------


async def test_tool_call_event_renders_dedicated_tool_observation() -> None:
    # A ToolCallEvent renders a dedicated Tool observation (type "tool",
    # NOT generation), DEFAULT level, with input / output populated
    # (payload on) and tool_name / tool_call_id in metadata.
    from openarmature.graph.events import ToolCallEvent
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_provider_payload=False)
    token = _set_invocation_id("inv-tool-1")
    try:
        await observer(
            ToolCallEvent(
                invocation_id="inv-tool-1",
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-1",
                tool_name="get_weather",
                tool_call_id="call_abc123",
                arguments={"city": "Paris"},
                result={"temperature_c": 20},
                latency_ms=5.0,
            )
        )
    finally:
        _reset_invocation_id(token)

    trace = client.traces["inv-tool-1"]
    tools = [o for o in trace.observations if o.type == "tool"]
    assert len(tools) == 1
    assert [o for o in trace.observations if o.type == "generation"] == []
    obs = tools[0]
    assert obs.name == "openarmature.tool.call"
    assert obs.level == "DEFAULT"
    assert obs.input == {"city": "Paris"}
    assert obs.output == {"temperature_c": 20}
    assert obs.metadata.get("openarmature_tool_name") == "get_weather"
    assert obs.metadata.get("openarmature_tool_call_id") == "call_abc123"
    assert obs.ended is True


async def test_tool_call_failed_event_renders_error_level() -> None:
    # A ToolCallFailedEvent renders the Tool observation at ERROR level
    # with error_type / error_message in metadata and as the status
    # message.
    from openarmature.graph.events import ToolCallFailedEvent
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_provider_payload=False)
    token = _set_invocation_id("inv-tool-2")
    try:
        await observer(
            ToolCallFailedEvent(
                invocation_id="inv-tool-2",
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-2",
                tool_name="get_weather",
                tool_call_id="call_def456",
                arguments={"city": "Paris"},
                latency_ms=3.0,
                error_type="TimeoutError",
                error_message="tool timed out",
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-tool-2"].observations if o.type == "tool")
    assert obs.level == "ERROR"
    assert obs.status_message == "tool timed out"
    assert obs.metadata.get("error_type") == "TimeoutError"
    assert obs.metadata.get("error_message") == "tool timed out"
    assert obs.metadata.get("openarmature_tool_name") == "get_weather"


async def test_tool_call_payload_gated_off_by_default() -> None:
    # With disable_provider_payload at its default (True), the Tool
    # observation's input / output are suppressed; metadata still carries
    # the identity.
    from openarmature.graph.events import ToolCallEvent
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-tool-3")
    try:
        await observer(
            ToolCallEvent(
                invocation_id="inv-tool-3",
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-3",
                tool_name="get_weather",
                tool_call_id="call_abc123",
                arguments={"city": "Paris"},
                result={"temperature_c": 20},
                latency_ms=5.0,
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-tool-3"].observations if o.type == "tool")
    assert obs.input is None
    assert obs.output is None
    assert obs.metadata.get("openarmature_tool_name") == "get_weather"


async def test_tool_call_non_json_result_does_not_crash_observer() -> None:
    # Proposal 0063: the tool result is opaque. A value json.dumps can't
    # natively encode MUST NOT crash the observer's serialization (which
    # would lose the Tool observation); the observation is still emitted.
    from openarmature.graph.events import ToolCallEvent
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    class _Opaque:
        def __str__(self) -> str:
            return "OPAQUE-RESULT"

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_provider_payload=False)
    token = _set_invocation_id("inv-tool-opaque")
    try:
        await observer(
            ToolCallEvent(
                invocation_id="inv-tool-opaque",
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-4",
                tool_name="get_weather",
                tool_call_id="call_abc123",
                arguments={"city": "Paris"},
                result=_Opaque(),
                latency_ms=5.0,
            )
        )
    finally:
        _reset_invocation_id(token)

    tools = [o for o in client.traces["inv-tool-opaque"].observations if o.type == "tool"]
    assert len(tools) == 1
