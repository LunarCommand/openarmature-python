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


async def test_a_when_skipped_branch_gets_no_observation_when_a_nested_branch_reuses_its_name() -> None:
    # The Langfuse counterpart of the OTel test of the same shape.  Both
    # observers share one `branch_dispatch_key` now, but they walk their own
    # open-observation / open-span stores, so the end-to-end path is not shared
    # and a key-level test alone does not reach it here.
    #
    # pipeline-utilities 11.10: a `when`-skipped branch is not dispatched and
    # emits no span.  Resolving the OUTER pb from an event inside the inner pb
    # built the outer key from the INNERMOST branch name, so an inner branch
    # reusing an outer branch's name opened a dispatch observation for the
    # skipped outer branch.
    from openarmature.graph.parallel_branches import BranchSpec

    class _S(State):
        n: int = 0

    class _Sub(State):
        n: int = 0

    async def _noop(_s: Any) -> dict[str, Any]:
        return {}

    inner = (
        GraphBuilder(_Sub)
        .add_parallel_branches_node("i", branches={"x": BranchSpec(call=_noop)})
        .add_edge("i", END)
        .set_entry("i")
        .compile()
    )
    graph = (
        GraphBuilder(_S)
        .add_parallel_branches_node(
            "o",
            branches={
                "x": BranchSpec(call=_noop, when=lambda _s: False),
                "y": BranchSpec(subgraph=inner),
            },
        )
        .add_edge("o", END)
        .set_entry("o")
        .compile()
    )
    graph, client, observer = _attach(graph)
    try:
        await graph.invoke(_S())
        await graph.drain()
    finally:
        observer.shutdown()

    trace = next(iter(client.traces.values()))
    dispatch = sorted(
        (obs.name, obs.metadata["parallel_branches_parent_node_name"])
        for obs in trace.observations
        if "parallel_branches_parent_node_name" in (obs.metadata or {})
    )
    # Non-vacuity first: the dispatched branch MUST be present, or the absence
    # assertion below is satisfied by an empty list. It was, on the first draft
    # of this test, because the metadata key is prefixed here and not in OTel.
    assert ("y", "o") in dispatch, f"expected the dispatched outer branch, got {dispatch}"
    assert ("x", "o") not in dispatch, f"`when`-skipped branch acquired a dispatch observation: {dispatch}"


async def test_metadata_augmentation_in_a_callable_branch_stays_off_the_trace() -> None:
    # The parallel-branches counterpart of the fan-out test below, for the shape
    # that has no chain to read.  A callable branch never descends, so its
    # augmenter's branch_name_chain is empty and a chain-only outermost-serial
    # test reads it as pure-serial -- which put BOTH branches' keys on the Trace,
    # the exact sibling leak §3.4's shared-parent boundary forbids.
    from openarmature.graph.parallel_branches import BranchSpec
    from openarmature.observability.metadata import set_invocation_metadata

    class _S(State):
        n: int = 0

    async def _ca(_s: _S) -> dict[str, Any]:
        set_invocation_metadata(from_a="yes")
        return {}

    async def _cb(_s: _S) -> dict[str, Any]:
        set_invocation_metadata(from_b="yes")
        return {}

    graph = (
        GraphBuilder(_S)
        .add_parallel_branches_node("pb", branches={"a": BranchSpec(call=_ca), "b": BranchSpec(call=_cb)})
        .add_edge("pb", END)
        .set_entry("pb")
        .compile()
    )
    graph, client, observer = _attach(graph)
    try:
        await graph.invoke(_S())
        await graph.drain()
    finally:
        observer.shutdown()

    trace = next(iter(client.traces.values()))
    leaked = {k for k in trace.metadata if k in {"from_a", "from_b"}}
    assert leaked == set(), f"callable-branch augmentation leaked onto Trace metadata: {sorted(leaked)}"


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
    from openarmature.observability.metadata import (
        _reset_invocation_metadata,
        _set_invocation_metadata,
        get_invocation_metadata,
        set_invocation_metadata,
    )

    # Reset-guard: snapshot + restore the module ContextVar so local_value does
    # not leak into later tests (conformance 046 expects an empty
    # get_invocation_metadata() outside any invocation).
    token = _set_invocation_metadata(get_invocation_metadata())
    try:
        set_invocation_metadata(local_key="local_value")
        assert get_invocation_metadata().get("local_key") == "local_value"
    finally:
        _reset_invocation_metadata(token)


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


async def test_parallel_branches_node_span_carries_config_attributes() -> None:
    # Proposal 0088 (§8.4.2): the parallel-branches NODE observation carries
    # parallel_branches_branch_count + _error_policy; each per-branch dispatch
    # observation carries parallel_branches_parent_node_name (+ branch_name).
    from openarmature.graph import BranchSpec

    class _BState(State):
        v: str = ""

    class _PState(State):
        x: str = ""
        y: str = ""

    async def _leaf(_s: _BState) -> dict[str, Any]:
        return {"v": "v"}

    branch = GraphBuilder(_BState).add_node("leaf", _leaf).add_edge("leaf", END).set_entry("leaf").compile()
    graph = (
        GraphBuilder(_PState)
        .add_parallel_branches_node(
            "dispatch",
            branches={
                "alpha": BranchSpec(subgraph=branch, outputs={"x": "v"}),
                "beta": BranchSpec(subgraph=branch, outputs={"y": "v"}),
            },
            error_policy="fail_fast",
        )
        .add_edge("dispatch", END)
        .set_entry("dispatch")
        .compile()
    )
    graph, client, _ = _attach(graph)
    await graph.invoke(_PState())
    await graph.drain()

    trace = next(iter(client.traces.values()))
    node_obs = next(o for o in trace.observations if o.name == "dispatch")
    assert node_obs.metadata["parallel_branches_branch_count"] == 2
    assert node_obs.metadata["parallel_branches_error_policy"] == "fail_fast"
    dispatch_obs = [o for o in trace.observations if o.name in ("alpha", "beta")]
    assert len(dispatch_obs) == 2
    assert all(o.metadata["parallel_branches_parent_node_name"] == "dispatch" for o in dispatch_obs)
    assert {o.metadata["branch_name"] for o in dispatch_obs} == {"alpha", "beta"}
    # The config attributes are node-span-only: they MUST NOT leak onto the
    # per-branch dispatch observations.
    assert all(
        "parallel_branches_branch_count" not in o.metadata
        and "parallel_branches_error_policy" not in o.metadata
        for o in dispatch_obs
    )


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


async def test_typed_llm_completion_over_budget_sets_warning_level() -> None:
    # §8.4.3 (proposal 0083): a SUCCESSFUL completion whose prompt_tokens exceed
    # the declared input_max_tokens sets the Generation's advisory WARNING level +
    # a statusMessage naming the breached bound, and maps the declared budget to
    # metadata.token_budget. The call succeeded, so the level is WARNING (not
    # ERROR).
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-tb-warn")
    try:
        await observer(
            make_typed_event(
                invocation_id="inv-tb-warn",
                usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
                token_budget=TokenBudget(input_max_tokens=10),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-tb-warn"].observations if o.type == "generation")
    assert gen.level == "WARNING"
    assert gen.status_message == "token budget exceeded: input 20 > 10"
    assert gen.metadata.get("token_budget") == {"input_max_tokens": 10}
    assert gen.metadata.get("token_budget_exceeded") is True  # 0109: flag on the WARNING path


async def test_typed_llm_completion_under_budget_no_warning_level() -> None:
    # §8.4.3 (proposal 0083): a completion under budget keeps the default level
    # (no advisory WARNING) while still mapping the declared budget to
    # metadata.token_budget.
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-tb-ok")
    try:
        await observer(
            make_typed_event(
                invocation_id="inv-tb-ok",
                usage=Usage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
                token_budget=TokenBudget(input_max_tokens=40),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-tb-ok"].observations if o.type == "generation")
    assert gen.level == "DEFAULT"
    assert gen.status_message is None
    assert gen.metadata.get("token_budget") == {"input_max_tokens": 40}
    assert gen.metadata.get("token_budget_exceeded") is False  # 0109: evaluated bound held


async def test_typed_llm_completion_null_counter_omits_exceeded_flag() -> None:
    # 0109 + 0101: a budget is declared and usage is PRESENT, but the only declared
    # bound's counter is not reported (prompt_tokens is None), so that bound is not
    # evaluated -> the flag is ABSENT (not false), distinct from the
    # absent-via-null-usage case. Guards the per-counter suppression on the
    # Langfuse surface (only a deferred conformance fixture covers it otherwise).
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-tb-nullcount")
    try:
        await observer(
            make_typed_event(
                invocation_id="inv-tb-nullcount",
                usage=Usage(prompt_tokens=None, completion_tokens=5, total_tokens=None),
                token_budget=TokenBudget(input_max_tokens=10),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-tb-nullcount"].observations if o.type == "generation")
    assert gen.metadata.get("token_budget") == {"input_max_tokens": 10}
    assert "token_budget_exceeded" not in gen.metadata


async def test_typed_llm_completion_both_bounds_breach_statusmessage() -> None:
    # §8.4.3 (proposal 0083): a completion breaching BOTH bounds names both in the
    # statusMessage, input then total.
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-tb-both")
    try:
        await observer(
            make_typed_event(
                invocation_id="inv-tb-both",
                usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                token_budget=TokenBudget(input_max_tokens=10, total_max_tokens=15),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-tb-both"].observations if o.type == "generation")
    assert gen.level == "WARNING"
    assert gen.status_message == "token budget exceeded: input 20 > 10, total 30 > 15"
    assert gen.metadata.get("token_budget") == {"input_max_tokens": 10, "total_max_tokens": 15}
    assert gen.metadata.get("token_budget_exceeded") is True  # 0109: flag on the multi-bound path


async def test_typed_llm_failed_generation_carries_token_budget_metadata() -> None:
    # §5.5.15 (proposal 0083): the shared metadata renders token_budget on a
    # FAILED Generation too (it carries the active prompt's budget). ERROR level
    # stands -- the advisory WARNING never displaces a hard failure (§8.4.3).
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-tb-fail")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-tb-fail",
                error_category="provider_unavailable",
                error_type="ProviderUnavailable",
                error_message="503",
                token_budget=TokenBudget(input_max_tokens=10),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-tb-fail"].observations if o.type == "generation")
    assert gen.level == "ERROR"
    assert gen.metadata.get("token_budget") == {"input_max_tokens": 10}
    # 0109: this failure carries no usage, so no bound is evaluable and the flag
    # is ABSENT (not false), mirroring the OTel attribute's null-counter suppression.
    assert "token_budget_exceeded" not in gen.metadata


async def test_structured_over_budget_failure_generation_is_error_not_warning() -> None:
    # §8.4.3 (proposal 0083): a structured_output_invalid failure that ALSO
    # exceeds budget renders the ERROR Generation (a hard ERROR wins over the
    # advisory WARNING) with metadata.token_budget still present -- the failed
    # handler never applies the success-path WARNING.
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.prompts import TokenBudget
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-tb-soi")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-tb-soi",
                error_category="structured_output_invalid",
                error_type="StructuredOutputInvalid",
                error_message="schema mismatch",
                usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
                token_budget=TokenBudget(input_max_tokens=10),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-tb-soi"].observations if o.type == "generation")
    assert gen.level == "ERROR"
    assert gen.status_message == "structured_output_invalid"
    assert "token budget exceeded" not in (gen.status_message or "")
    assert gen.metadata.get("token_budget") == {"input_max_tokens": 10}
    # 0109: the exceeded flag SURVIVES the ERROR-precedence rule -- the level is
    # ERROR + the statusMessage is the category, but the flag is still present and
    # true, giving the Langfuse failure path parity with the OTel span attribute.
    assert gen.metadata.get("token_budget_exceeded") is True


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


async def test_malformed_usage_counter_omitted_from_langfuse_generation_usage() -> None:
    # 0101: a not-reported (null) counter is omitted from the Generation usage.
    # prompt_tokens / total_tokens are null, completion_tokens is sound, so the
    # Generation carries output only. (The SDK adapter drops the None fields from
    # usage_details; in the in-memory double, None IS the not-reported state.)
    from openarmature.llm.response import Usage
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id
    from tests._helpers.typed_event import make_typed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    token = _set_invocation_id("inv-usage-null")
    try:
        await observer(
            make_typed_event(
                invocation_id="inv-usage-null",
                usage=Usage(prompt_tokens=None, completion_tokens=7, total_tokens=None),
            )
        )
    finally:
        _reset_invocation_id(token)

    gen = next(o for o in client.traces["inv-usage-null"].observations if o.type == "generation")
    assert gen.usage is not None
    assert gen.usage.input is None
    assert gen.usage.output == 7
    assert gen.usage.total is None


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
        # keyed by (namespace, attempt_index, fan_out_index, branch_name)
        # plus the proposal-0084 lineage chains; the calling node identity
        # on the (non-nested) typed event has empty chains, so the keys are
        # (("dispatcher", "ask"), 0, None, "fast"/"slow", (), ()). key[3]
        # (branch_name) is the discriminator this test exercises.
        client.trace(id=invocation_id, name="dispatcher")
        observer._inv_states[invocation_id] = _InvState(trace_id=invocation_id)  # noqa: SLF001
        inv_state = observer._inv_states[invocation_id]  # noqa: SLF001
        # Open two observations under the trace — one per branch.
        fast_handle = client.generation(trace_id=invocation_id, name="ask", model="m-test")
        slow_handle = client.generation(trace_id=invocation_id, name="ask", model="m-test")
        fast_key = (("dispatcher", "ask"), 0, None, "fast", (), ())
        slow_key = (("dispatcher", "ask"), 0, None, "slow", (), ())
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
    # Regression cover for the _resolve_llm_parent_observation_id orphan fallback:
    # when an LLM event fires inside a top-level fan-out instance and the calling
    # node has no open observation (fallback #1 misses), the Generation MUST
    # parent under the per-instance fan-out dispatch observation (the nearest
    # enclosing wrapper). Proposal 0084: the fallback resolves the dispatch via
    # the event's lineage chain, so the event carries fan_out_index_chain=
    # (0, None) aligned to namespace ("fan", "ask") -- instance 0 at "fan", none
    # at "ask".
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
                fan_out_index_chain=(0, None),
                branch_name_chain=(None, None),
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


async def test_llm_event_parents_under_parallel_branch_dispatch() -> None:
    # Proposal 0084 branch coverage: the orphan fallback resolves a per-branch
    # dispatch observation as well as a fan-out instance observation, but the
    # 0084 fixtures (132/133/134) are fan-out only. An orphan LLM event fired
    # inside a parallel branch (calling node observation not open) MUST parent
    # under the per-branch dispatch observation -- the nearest enclosing wrapper
    # -- resolved via branch_name_chain. Guards the reference behavior pending a
    # spec fixture (see release-v0.17.0 coord).
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from openarmature.observability.langfuse.observer import (
        _branch_dispatch_key,
        _InvState,
        _OpenObservation,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    invocation_id = "inv-pb-llm"
    token = _set_invocation_id(invocation_id)
    try:
        client.trace(id=invocation_id, name="dispatcher")
        observer._inv_states[invocation_id] = _InvState(trace_id=invocation_id)  # noqa: SLF001
        inv_state = observer._inv_states[invocation_id]  # noqa: SLF001
        # Per-branch dispatch for branch "fast" of pb node "dispatcher". No
        # open_observation for the calling node ("dispatcher", "ask"), so the
        # resolver reaches the per-branch dispatch fallback. The calling node
        # sits in branch "fast": branch_name_chain=(None, "fast") aligned to
        # namespace.
        fi_chain: tuple[int | None, ...] = (None, None)
        bn_chain: tuple[str | None, ...] = (None, "fast")
        dispatch_handle = client.span(trace_id=invocation_id, name="fast")
        branch_key = _branch_dispatch_key(("dispatcher",), fi_chain, bn_chain, "fast")
        inv_state.parallel_branches_branch_spans[branch_key] = _OpenObservation(handle=dispatch_handle)
        await observer(
            make_failed_event(
                invocation_id=invocation_id,
                node_name="ask",
                namespace=("dispatcher", "ask"),
                attempt_index=0,
                fan_out_index=None,
                branch_name="fast",
                fan_out_index_chain=fi_chain,
                branch_name_chain=bn_chain,
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
    error_gens = [o for o in trace.observations if o.type == "generation" and o.level == "ERROR"]
    assert len(error_gens) == 1
    assert error_gens[0].parent_observation_id == dispatch_handle.id, (
        "LLM Generation must parent under the per-branch dispatch observation"
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


async def test_error_message_is_capped_under_this_observer_s_own_cap() -> None:
    # Proposal 0119 (spec v0.116.0). §5.5.5 now governs every payload-classified
    # VALUE rather than only values written as span attributes, and §8.7 gives a
    # failed observation's `error_message` a DIRECT-application arm because it
    # has no span attribute to inherit a cap from.
    #
    # It was written verbatim before 0119: 0118 classified the field for gating
    # without saying it was subject to truncation, so a provider returning a very
    # large exception string rendered it in full.
    #
    # The cap applied is THIS observer's `payload_byte_cap`. An observer MUST NOT
    # take the OTel observer's cap for this value; the two are configured
    # independently, under different names (`payload_max_bytes` there), and a
    # deployment can set one and leave the other at its default. That asymmetry
    # is what conformance fixture 160 exists to catch. This test pins the cap
    # actually used by constructing the observer with `payload_byte_cap=256` and
    # asserting the marker reports the cap applied at that value; it does not
    # construct an OTel observer, so the cross-observer asymmetry itself rides
    # fixture 160 rather than this unit test.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    # Payloads off is the §8.9 default, which withholds the harvested message
    # entirely; that arm is fixture 160's THIRD case. These two are about the
    # cap, so the channel is opened.
    observer = LangfuseObserver(client=client, payload_byte_cap=256, disable_provider_payload=False)
    long_message = "E" * 4000

    token = _set_invocation_id("inv-cap")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-cap",
                model="m-test",
                error_category="provider_rate_limit",
                error_type="ProviderRateLimit",
                error_message=long_message,
                call_id="cc-cap",
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-cap"].observations if o.type == "generation")
    rendered = obs.metadata["error_message"]
    assert rendered != long_message, "the message was written verbatim; the cap was not applied"
    assert len(rendered.encode("utf-8")) <= 256, (
        f"rendered {len(rendered.encode('utf-8'))} bytes against a 256-byte cap"
    )
    # Pin the ALGORITHM, not just the length. Without the next three assertions a
    # marker-less byte chop (`message.encode()[:cap].decode(errors="ignore")`)
    # satisfies everything above while violating §5.5.5 outright. Verified by
    # mutation: that exact chop left the whole suite green before these landed.
    #
    # The marker carries M, the PRE-truncation byte length, so asserting the
    # exact tail also catches an implementation that reports the post-truncation
    # length, or that serializes through JSON first (which would shift M by the
    # two added quote bytes).
    assert rendered.endswith("…[truncated, 4000 bytes total]"), (
        f"missing or malformed §5.5.5 truncation marker; tail was {rendered[-40:]!r}"
    )
    kept = rendered[: -len("…[truncated, 4000 bytes total]")]
    assert long_message.startswith(kept), "the kept bytes are not a prefix of the original message"


async def test_a_below_cap_error_message_is_left_alone() -> None:
    # The control. Without it, an implementation that truncated unconditionally,
    # or wrote a fixed marker, satisfies the cap test above. This is the case
    # conformance fixture 160 adds beyond the three the proposal designed, and it
    # is the one that catches that mistake.
    #
    # It is also why fixtures 150 / 151 are unaffected by the cap landing: their
    # messages are far below any cap, so they still render literally.
    #
    # Provenance, corrected: this control is one of the THREE cases the proposal
    # designed, not an addition beyond them. The two fixture 160 added beyond the
    # proposal are the default-posture arm and the retriever arm.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    # Payloads off is the §8.9 default, which withholds the harvested message
    # entirely; that arm is fixture 160's THIRD case. These two are about the
    # cap, so the channel is opened.
    observer = LangfuseObserver(client=client, payload_byte_cap=256, disable_provider_payload=False)

    token = _set_invocation_id("inv-small")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-small",
                model="m-test",
                error_category="provider_rate_limit",
                error_type="ProviderRateLimit",
                error_message="429 from upstream",
                call_id="cc-small",
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-small"].observations if o.type == "generation")
    assert obs.metadata["error_message"] == "429 from upstream"


async def test_tool_failure_error_message_is_capped() -> None:
    # §8.7's Tool arm, which is normative and has NO conformance fixture: a case
    # would need a `mock_tool` primitive and a `calls_tool` block, and neither is
    # defined in conformance-adapter §5. Spec recorded that in its
    # open-questions and told us to read the arm as binding, so this unit test is
    # the only cover the arm can have until the adapter grows those.
    #
    # The Tool observation is also the one where the message matters most. A
    # failed Tool carries no error CATEGORY, so `error_type` is the only other
    # discriminator on it; whatever the message does here cannot be inferred from
    # the Generation arm, which has a category to fall back on.
    from openarmature.graph.events import ToolCallFailedEvent
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, payload_byte_cap=256, disable_provider_payload=False)
    long_message = "T" * 4000

    token = _set_invocation_id("inv-tool-cap")
    try:
        await observer(
            ToolCallFailedEvent(
                invocation_id="inv-tool-cap",
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-cap",
                tool_name="get_weather",
                tool_call_id="call_cap",
                arguments={"city": "Paris"},
                latency_ms=3.0,
                error_type="TimeoutError",
                error_message=long_message,
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-tool-cap"].observations if o.type == "tool")
    rendered = obs.metadata["error_message"]
    assert rendered != long_message, "the Tool arm wrote the message verbatim"
    assert len(rendered.encode("utf-8")) <= 256
    assert rendered.endswith("…[truncated, 4000 bytes total]"), (
        f"missing or malformed §5.5.5 truncation marker; tail was {rendered[-40:]!r}"
    )
    # `error_type` is NOT payload-gated and NOT capped: it is a classification
    # token, and on a Tool observation it is the only discriminator left.
    assert obs.metadata["error_type"] == "TimeoutError"
    # The status message is the SECOND surface carrying the same harvested
    # string, and it must be capped too. It was not until following the Tool arm
    # turned it up: capping only the metadata copy leaves the whole exception
    # rendered on the observation, which is the outcome the cap exists to stop.
    assert obs.status_message is not None
    assert len(obs.status_message.encode("utf-8")) <= 256, (
        f"status message rendered {len(obs.status_message.encode('utf-8'))} bytes against a 256-byte cap"
    )
    # Identical to the metadata copy, not merely short: the two surfaces render
    # the same value and must not diverge.
    assert obs.status_message == rendered


@pytest.mark.parametrize("arm", ["embedding", "rerank"])
async def test_embedding_and_rerank_error_messages_are_capped(arm: str) -> None:
    # The other two of §8.7's four mapped provider observations. Found by
    # mutation: reverting the cap at each of the four sites individually left
    # these two arms green, because the LLM and Tool tests above cover only their
    # own handlers. A passthrough mutation of the shared helper, which breaks all
    # four sites at once, produced exactly two failures rather than four, which is
    # the same gap seen from the other side.
    from openarmature.graph.events import EmbeddingFailedEvent, RerankFailedEvent
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, payload_byte_cap=256, disable_provider_payload=False)
    long_message = "E" * 4000
    inv = f"inv-{arm}-cap"

    event: Any
    if arm == "embedding":
        event = EmbeddingFailedEvent(
            invocation_id=inv,
            correlation_id=None,
            node_name="embed",
            namespace=("embed",),
            attempt_index=0,
            fan_out_index=None,
            branch_name=None,
            provider="openai",
            model="embed-model",
            latency_ms=1.0,
            input_strings=["q"],
            request_params={},
            request_extras={},
            active_prompt=None,
            active_prompt_group=None,
            call_id=f"cc-{arm}-cap",
            error_category="provider_unavailable",
            error_message=long_message,
        )
    else:
        event = RerankFailedEvent(
            invocation_id=inv,
            correlation_id=None,
            node_name="rerank",
            namespace=("rerank",),
            attempt_index=0,
            fan_out_index=None,
            branch_name=None,
            provider="cohere",
            model="rerank-model",
            latency_ms=1.0,
            query="q",
            documents=["d"],
            document_count=1,
            top_k=1,
            request_params={},
            request_extras={},
            active_prompt=None,
            active_prompt_group=None,
            call_id=f"cc-{arm}-cap",
            error_category="provider_unavailable",
            error_message=long_message,
        )

    token = _set_invocation_id(inv)
    try:
        await observer(event)
    finally:
        _reset_invocation_id(token)

    # Selected by observation TYPE, not by error_message truthiness. Truthiness
    # selection made both parametrizations assert byte-identical things, so
    # misrouting the rerank event into the embedding handler still passed; it
    # also breaks on an empty message, which a bare `raise SomeError()` produces.
    expected_type = "embedding" if arm == "embedding" else "retriever"
    obs = next(o for o in client.traces[inv].observations if o.type == expected_type)
    rendered = obs.metadata["error_message"]
    assert rendered != long_message, f"the {arm} arm wrote the message verbatim; the cap was not applied"
    assert len(rendered.encode("utf-8")) <= 256, (
        f"{arm} rendered {len(rendered.encode('utf-8'))} bytes against a 256-byte cap"
    )
    assert rendered.endswith("…[truncated, 4000 bytes total]"), (
        f"{arm}: missing or malformed §5.5.5 truncation marker; tail was {rendered[-40:]!r}"
    )
    # The status message on these two arms takes the error CATEGORY, a
    # classification token rather than harvested text, so it is correctly
    # uncapped and must survive intact. Only the Tool arm renders the harvested
    # string twice.
    assert obs.status_message == "provider_unavailable"


async def test_a_multibyte_error_message_is_cut_on_a_code_point_boundary() -> None:
    # `_truncate` backtracks off UTF-8 continuation bytes so the cut never lands
    # mid-sequence. Every other cap test uses ASCII filler, where the backtrack
    # is a no-op, so this is the only test that exercises it. Fixture 160's own
    # header calls out the same hazard.
    #
    # A naive `encoded[:target].decode(errors="ignore")` silently drops the
    # partial character and still looks plausible; a strict decode raises. This
    # asserts the strict round-trip so either failure mode is caught.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    # The cap is 257, NOT 256, and the difference is the whole test. The marker
    # for an 8000-byte message is 32 bytes, so a 256-byte cap leaves a 224-byte
    # target, and 224 is an exact multiple of 4: the cut lands cleanly on a code
    # point boundary and the backtracking loop never executes. Verified by
    # mutation: at 256, deleting the loop entirely left the suite green. At 257
    # the target is 225, which is mid-sequence. (256 is also the §5.5.5 floor, so
    # the cap cannot be lowered instead.)
    observer = LangfuseObserver(client=client, payload_byte_cap=257, disable_provider_payload=False)
    long_message = "\U0001f600" * 2000

    token = _set_invocation_id("inv-multibyte")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-multibyte",
                model="m-test",
                error_category="provider_rate_limit",
                error_type="ProviderRateLimit",
                error_message=long_message,
                call_id="cc-multibyte",
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-multibyte"].observations if o.type == "generation")
    rendered = obs.metadata["error_message"]
    assert len(rendered.encode("utf-8")) <= 257
    # M counts BYTES, not code points: 2000 code points at 4 bytes each.
    marker = "…[truncated, 8000 bytes total]"
    assert rendered.endswith(marker)
    # No partial sequence and no dropped character: every kept code point is whole.
    kept = rendered[: -len(marker)]
    assert kept == "\U0001f600" * len(kept), "the cut landed mid-sequence or dropped a partial character"
    assert long_message.startswith(kept)
    assert rendered.encode("utf-8").decode("utf-8", errors="strict") == rendered
    # Prove the backtrack actually ran, rather than the cut happening to land on
    # a boundary. The target is 257 - 32 = 225 bytes; a whole number of 4-byte
    # code points below that is 224, so the kept run MUST be shorter than the
    # target. Without this, a future cap change could silently return the test to
    # the boundary-aligned case where the loop is dead code again.
    target = 257 - len(marker.encode("utf-8"))
    assert len(kept.encode("utf-8")) < target, (
        f"kept {len(kept.encode('utf-8'))} bytes against a {target}-byte target, so the cut "
        "landed on a code point boundary and the backtracking loop was never exercised"
    )


async def test_a_surrogate_in_the_error_message_does_not_destroy_the_observation() -> None:
    # Harvested exception text is the likeliest string in the observer to carry a
    # lone surrogate: a FileNotFoundError naming a surrogateescape-decoded path,
    # or a provider body decoded the same way. `"\udcff".encode("utf-8")` raises
    # UnicodeEncodeError, and the cap is applied BEFORE the client call, so an
    # unguarded encode kills the handler.
    #
    # The engine only `warnings.warn`s an observer exception, so the failed
    # observation would disappear with no log record: the failure path would take
    # out its own reporting. The observation must survive with the field degraded.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, payload_byte_cap=256, disable_provider_payload=False)

    token = _set_invocation_id("inv-surrogate")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-surrogate",
                model="m-test",
                error_category="provider_unavailable",
                error_type="OSError",
                error_message="cannot open /data/\udcff/report.json",
                call_id="cc-surrogate",
            )
        )
    finally:
        _reset_invocation_id(token)

    generations = [o for o in client.traces["inv-surrogate"].observations if o.type == "generation"]
    assert len(generations) == 1, "the observation was lost; the cap raised on the failure path"
    rendered = generations[0].metadata["error_message"]
    # Degraded, not dropped: the surrounding text survives and the value is
    # encodable, so the backend can actually ingest it.
    assert "cannot open" in rendered
    assert "report.json" in rendered
    rendered.encode("utf-8")


async def test_a_long_surrogate_bearing_message_is_still_capped() -> None:
    # The sanitize path must not become a cap bypass: a message that is both
    # malformed AND oversized still has to come back within the cap.
    from openarmature.observability.correlation import (
        _reset_invocation_id,
        _set_invocation_id,
    )
    from tests._helpers.typed_event import make_failed_event

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, payload_byte_cap=256, disable_provider_payload=False)

    token = _set_invocation_id("inv-surrogate-long")
    try:
        await observer(
            make_failed_event(
                invocation_id="inv-surrogate-long",
                model="m-test",
                error_category="provider_unavailable",
                error_type="OSError",
                error_message="\udcff" + "E" * 4000,
                call_id="cc-surrogate-long",
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-surrogate-long"].observations if o.type == "generation")
    rendered = obs.metadata["error_message"]
    assert len(rendered.encode("utf-8")) <= 256, (
        f"the sanitize path bypassed the cap: {len(rendered.encode('utf-8'))} bytes"
    )
    assert "[truncated," in rendered
