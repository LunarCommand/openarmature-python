"""Unit tests for the fan-out runtime.

Covers the edge cases the conformance fixtures exercise only
implicitly:

- items_field projection
- count mode (literal int + state-reading callable)
- inputs mapping per-instance projection
- concurrency limit enforcement
- concurrency callable resolved exactly once at fan-out entry
- fail_fast: first failure cancels siblings; recoverable_state is the
  parent's pre-fan-out snapshot
- collect: per-instance errors recorded; successes merged
- on_empty: raise (default) and noop
- count_field write behavior
- errors_field collection shape
- extra_outputs merge
- instance_middleware chain composition
- fan-in determinism under nondeterministic completion timing
- compile-time errors (count_mode_ambiguous, field_not_list)
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    FailureIsolationMiddleware,
    FanOutCountModeAmbiguous,
    FanOutFieldNotList,
    GraphBuilder,
    NodeException,
    ReducerError,
    RetryConfig,
    RetryMiddleware,
    State,
    append,
    concat_flatten,
    deterministic_backoff,
)

# ---------------------------------------------------------------------------
# Shared state schemas + helper builders
# ---------------------------------------------------------------------------


class WorkerState(State):
    item: int = 0
    extra: int = 0
    result: int = 0
    side: int = 0


def _build_doubler() -> CompiledGraph[WorkerState]:
    """A trivial worker subgraph: result = item * 2."""

    async def compute(state: WorkerState) -> Mapping[str, Any]:
        return {"result": state.item * 2}

    builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    builder.set_entry("compute")
    builder.add_node("compute", compute)
    builder.add_edge("compute", END)
    return builder.compile()


def _build_constant_one() -> CompiledGraph[WorkerState]:
    """Worker that ignores input, returns result=1. Used for count-mode tests."""

    async def compute(_state: WorkerState) -> Mapping[str, Any]:
        return {"result": 1}

    builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    builder.set_entry("compute")
    builder.add_node("compute", compute)
    builder.add_edge("compute", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# items_field projection + basic fan-in
# ---------------------------------------------------------------------------


class ItemsParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])


async def test_items_field_projection_doubles_each() -> None:
    """Each instance receives one item; collected results preserve input
    order."""
    inner = _build_doubler()
    builder: GraphBuilder[ItemsParentState] = GraphBuilder(ItemsParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    final = await compiled.invoke(ItemsParentState(items=[1, 2, 3]))
    await compiled.drain()
    assert final.results == [2, 4, 6]


# ---------------------------------------------------------------------------
# count mode
# ---------------------------------------------------------------------------


class CountParentState(State):
    n: int = 0
    results: Annotated[list[int], append] = Field(default_factory=list[int])


async def test_count_mode_literal_int() -> None:
    inner = _build_constant_one()
    builder: GraphBuilder[CountParentState] = GraphBuilder(CountParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        count=4,
        collect_field="result",
        target_field="results",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    final = await compiled.invoke(CountParentState())
    await compiled.drain()
    assert final.results == [1, 1, 1, 1]


async def test_count_mode_state_reading_callable() -> None:
    """``count`` may be a callable that reads the parent state at entry."""
    inner = _build_constant_one()
    builder: GraphBuilder[CountParentState] = GraphBuilder(CountParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        count=lambda s: int(s.n),
        collect_field="result",
        target_field="results",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    final = await compiled.invoke(CountParentState(n=5))
    await compiled.drain()
    assert final.results == [1, 1, 1, 1, 1]


async def test_count_callable_resolved_exactly_once_at_entry() -> None:
    """The count callable is invoked exactly once at fan-out entry.
    A callable with side effects (counter increment) MUST be observed to
    run exactly once."""
    inner = _build_constant_one()
    invocations = [0]

    def counting_count(s: CountParentState) -> int:
        invocations[0] += 1
        return int(s.n)

    builder: GraphBuilder[CountParentState] = GraphBuilder(CountParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        count=counting_count,
        collect_field="result",
        target_field="results",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()
    await compiled.invoke(CountParentState(n=3))
    await compiled.drain()

    assert invocations[0] == 1


async def test_concurrency_callable_resolved_exactly_once_at_entry() -> None:
    """The concurrency callable, like count, is invoked exactly
    once at fan-out entry — even with many instances (which would
    otherwise be a natural place to call it per-instance by mistake)."""

    class _State(State):
        items: list[int] = Field(default_factory=list[int])
        cap: int = 0
        results: Annotated[list[int], append] = Field(default_factory=list[int])

    invocations = [0]

    def counting_concurrency(s: _State) -> int:
        invocations[0] += 1
        return int(s.cap)

    inner = _build_doubler()
    builder: GraphBuilder[_State] = GraphBuilder(_State)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        concurrency=counting_concurrency,
    )
    builder.add_edge("process", END)
    compiled = builder.compile()
    await compiled.invoke(_State(items=[1, 2, 3, 4, 5], cap=2))
    await compiled.drain()

    assert invocations[0] == 1


# ---------------------------------------------------------------------------
# inputs mapping projection
# ---------------------------------------------------------------------------


class InputsParentState(State):
    items: list[int] = Field(default_factory=list[int])
    boost: int = 0
    results: Annotated[list[int], append] = Field(default_factory=list[int])


async def test_inputs_mapping_projects_parent_fields() -> None:
    """``inputs`` maps parent fields onto the per-instance subgraph
    state at entry, alongside item_field."""

    async def compute(state: WorkerState) -> Mapping[str, Any]:
        return {"result": state.item + state.extra}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", compute)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    builder: GraphBuilder[InputsParentState] = GraphBuilder(InputsParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        inputs={"extra": "boost"},  # subgraph.extra <- parent.boost
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    final = await compiled.invoke(InputsParentState(items=[1, 2, 3], boost=10))
    await compiled.drain()
    assert final.results == [11, 12, 13]


# ---------------------------------------------------------------------------
# nested fan-out (a fan-out inside an outer fan-out instance)
# ---------------------------------------------------------------------------


class _NestedLeafState(State):
    tag: str = ""
    seed: str = ""
    out: str = ""


class _NestedMidState(State):
    tag: str = ""
    seeds: list[str] = Field(default_factory=list[str])
    collected: Annotated[list[str], append] = Field(default_factory=list[str])


class _NestedOuterState(State):
    products: list[str] = Field(default_factory=list[str])
    seeds: list[str] = Field(default_factory=list[str])
    results: Annotated[list[Any], append] = Field(default_factory=list[Any])


async def test_nested_fan_out_distinct_per_outer_instance_under_concurrency() -> None:
    """A fan-out nested inside an outer fan-out instance runs its inner
    subgraph once per (outer, inner) pair and returns the right per-outer
    result, even with the outer instances running concurrently."""
    # Regression: the per-fan-out tracking entry was keyed by (namespace, node
    # name) only, so the inner fan-out's entry collided across outer instances.
    # With concurrent outer instances the second found the first's entry already
    # marked complete and rolled its result forward, so every outer instance
    # returned the first's inner result and the inner subgraph ran only once.
    leaf_calls = 0

    async def leaf(state: _NestedLeafState) -> Mapping[str, Any]:
        nonlocal leaf_calls
        await asyncio.sleep(0)  # yield so the concurrent outer instances interleave
        leaf_calls += 1
        return {"out": f"{state.tag}-{state.seed}"}

    leaf_builder: GraphBuilder[_NestedLeafState] = GraphBuilder(_NestedLeafState)
    leaf_builder.set_entry("ask")
    leaf_builder.add_node("ask", leaf)
    leaf_builder.add_edge("ask", END)
    leaf_graph = leaf_builder.compile()

    mid_builder: GraphBuilder[_NestedMidState] = GraphBuilder(_NestedMidState)
    mid_builder.set_entry("inner_fan")
    mid_builder.add_fan_out_node(
        "inner_fan",
        subgraph=leaf_graph,
        items_field="seeds",
        item_field="seed",
        inputs={"tag": "tag"},
        collect_field="out",
        target_field="collected",
    )
    mid_builder.add_edge("inner_fan", END)
    mid_graph = mid_builder.compile()

    outer_builder: GraphBuilder[_NestedOuterState] = GraphBuilder(_NestedOuterState)
    outer_builder.set_entry("outer_fan")
    outer_builder.add_fan_out_node(
        "outer_fan",
        subgraph=mid_graph,
        items_field="products",
        item_field="tag",
        inputs={"seeds": "seeds"},
        collect_field="collected",
        target_field="results",
    )
    outer_builder.add_edge("outer_fan", END)
    outer_graph = outer_builder.compile()

    final = await outer_graph.invoke(_NestedOuterState(products=["A", "B"], seeds=["x", "y"]))
    await outer_graph.drain()
    # Each outer instance collected its OWN inner results; the collapse bug gave
    # [["A-x", "A-y"], ["A-x", "A-y"]] (the second outer reused the first's).
    got = sorted(tuple(sorted(sub)) for sub in final.results)
    assert got == [("A-x", "A-y"), ("B-x", "B-y")]
    # The inner leaf ran once per (outer, inner) pair, not once total.
    assert leaf_calls == 4


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


class ConcurrencyParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])


async def test_concurrency_limit_caps_in_flight_instances() -> None:
    """``concurrency: 2`` means at most 2 instances run concurrently —
    verified by tracking peak in-flight via a shared counter."""
    in_flight = [0]
    peak = [0]

    async def slow_compute(state: WorkerState) -> Mapping[str, Any]:
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        await asyncio.sleep(0.01)
        in_flight[0] -= 1
        return {"result": state.item}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", slow_compute)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    builder: GraphBuilder[ConcurrencyParentState] = GraphBuilder(ConcurrencyParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        concurrency=2,
    )
    builder.add_edge("process", END)
    compiled = builder.compile()
    await compiled.invoke(ConcurrencyParentState(items=list(range(10))))
    await compiled.drain()

    assert peak[0] <= 2


# ---------------------------------------------------------------------------
# fail_fast / collect
# ---------------------------------------------------------------------------


class FailFastParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])


async def test_fail_fast_propagates_first_failure_with_parent_recoverable_state() -> None:
    """The first failure raises through the fan-out as a
    NodeException whose recoverable_state is the parent's pre-fan-out
    snapshot, NOT the inner instance's state."""

    async def maybe_fail(state: WorkerState) -> Mapping[str, Any]:
        if state.item == 1:
            raise RuntimeError("boom on idx=1")
        return {"result": state.item}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", maybe_fail)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    builder: GraphBuilder[FailFastParentState] = GraphBuilder(FailFastParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    with pytest.raises(NodeException) as excinfo:
        await compiled.invoke(FailFastParentState(items=[0, 1, 2]))
    await compiled.drain()
    assert excinfo.value.node_name == "process"
    assert excinfo.value.recoverable_state.items == [0, 1, 2]
    assert excinfo.value.recoverable_state.results == []


class _StrictReducerParentState(State):
    items: list[int] = Field(default_factory=list[int])
    # concat_flatten requires every collected element to be a list; a degrade
    # that nulls the slot contributes None, which the reducer rejects.
    results: Annotated[list[int], concat_flatten] = Field(default_factory=list[int])


async def test_degrade_null_slot_under_strict_reducer_raises_reducer_error() -> None:
    # Proposal 0069 refinement (2) caveat: an absent collect_field is a
    # graceful null slot and the fan-in does not raise, but under a
    # strict-element reducer (concat_flatten / merge_all) the null contribution
    # still raises ReducerError. The degrade-path .get() null is not suppressed
    # because the reducer runs in the engine merge, downstream of the fan-in. A
    # callable degrade is used because a static degrade omitting collect_field
    # is a compile error (proposal 0066).
    async def always_fail(_state: WorkerState) -> Mapping[str, Any]:
        raise RuntimeError("instance down")

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", always_fail)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    builder: GraphBuilder[_StrictReducerParentState] = GraphBuilder(_StrictReducerParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        instance_middleware=(
            FailureIsolationMiddleware(
                # Callable degrade omitting collect_field -> runtime null slot.
                degraded_update=lambda _state: {},
                event_name="degraded",
            ),
        ),
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    with pytest.raises(ReducerError):
        await compiled.invoke(_StrictReducerParentState(items=[0]))
    await compiled.drain()


class CollectParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])
    errors: Annotated[list[dict[str, str]], append] = Field(default_factory=list[dict[str, str]])


async def test_collect_records_per_instance_errors() -> None:
    """Collect mode runs all instances to completion; failures
    are recorded in errors_field; successes contribute to target_field."""

    async def maybe_fail(state: WorkerState) -> Mapping[str, Any]:
        if state.item == 1:
            raise RuntimeError("boom")
        return {"result": state.item * 10}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", maybe_fail)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    builder: GraphBuilder[CollectParentState] = GraphBuilder(CollectParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        error_policy="collect",
        errors_field="errors",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()
    final = await compiled.invoke(CollectParentState(items=[0, 1, 2]))
    await compiled.drain()

    # Successes preserved in input order; failure (idx=1) omitted.
    assert final.results == [0, 20]
    # Errors carry instance index + category.
    assert len(final.errors) == 1
    assert final.errors[0] == {"fan_out_index": "1", "category": "node_exception"}


# ---------------------------------------------------------------------------
# on_empty
# ---------------------------------------------------------------------------


class EmptyParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])
    processed_count: int = -1


async def test_on_empty_raise_default_raises_fan_out_empty() -> None:
    """Empty fan-out with on_empty='raise' (default) raises
    a NodeException tagged with fan_out_category='fan_out_empty'."""
    inner = _build_doubler()
    builder: GraphBuilder[EmptyParentState] = GraphBuilder(EmptyParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    with pytest.raises(NodeException) as excinfo:
        await compiled.invoke(EmptyParentState(items=[]))
    await compiled.drain()
    assert getattr(excinfo.value, "fan_out_category", None) == "fan_out_empty"


async def test_on_empty_noop_writes_count_field_zero() -> None:
    """on_empty='noop' produces a clean no-op; count_field
    captures the resolved 0."""
    inner = _build_doubler()
    builder: GraphBuilder[EmptyParentState] = GraphBuilder(EmptyParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        on_empty="noop",
        count_field="processed_count",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    final = await compiled.invoke(EmptyParentState(items=[]))
    await compiled.drain()
    assert final.results == []
    assert final.processed_count == 0


# ---------------------------------------------------------------------------
# count_field write behavior
# ---------------------------------------------------------------------------


class CountFieldParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])
    processed: int = -1


async def test_count_field_records_actual_count_on_success() -> None:
    """count_field is written with the resolved instance count after
    fan-in, regardless of whether on_empty fires."""
    inner = _build_doubler()
    builder: GraphBuilder[CountFieldParentState] = GraphBuilder(CountFieldParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        count_field="processed",
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    final = await compiled.invoke(CountFieldParentState(items=[5, 10, 15]))
    await compiled.drain()
    assert final.processed == 3


# ---------------------------------------------------------------------------
# extra_outputs merge
# ---------------------------------------------------------------------------


class ExtraOutputsParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])
    sides: Annotated[list[int], append] = Field(default_factory=list[int])


async def test_extra_outputs_merges_additional_per_instance_fields() -> None:
    """extra_outputs collects additional non-collected fields
    from each instance and merges them via the parent's reducer."""

    async def compute(state: WorkerState) -> Mapping[str, Any]:
        return {"result": state.item, "side": state.item * 100}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", compute)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    builder: GraphBuilder[ExtraOutputsParentState] = GraphBuilder(ExtraOutputsParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        extra_outputs={"sides": "side"},
    )
    builder.add_edge("process", END)
    compiled = builder.compile()
    final = await compiled.invoke(ExtraOutputsParentState(items=[1, 2, 3]))
    await compiled.drain()
    assert final.results == [1, 2, 3]
    assert final.sides == [100, 200, 300]


# ---------------------------------------------------------------------------
# instance_middleware composition
# ---------------------------------------------------------------------------


class InstanceMwParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])


async def test_instance_middleware_retry_recovers_per_instance() -> None:
    """instance_middleware wraps each instance's whole subgraph
    invocation. Retry around an instance retries the WHOLE invocation,
    not the inner node — the chain runs from scratch on each retry."""

    class _Transient(Exception):
        category = "provider_rate_limit"

    instance_attempts: dict[int, int] = {}

    async def maybe_fail(state: WorkerState) -> Mapping[str, Any]:
        n = instance_attempts.get(state.item, 0)
        instance_attempts[state.item] = n + 1
        if n == 0:
            raise _Transient()
        return {"result": state.item}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", maybe_fail)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    retry = RetryMiddleware(RetryConfig(max_attempts=3, backoff=deterministic_backoff(0)))

    builder: GraphBuilder[InstanceMwParentState] = GraphBuilder(InstanceMwParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        instance_middleware=[retry],
    )
    builder.add_edge("process", END)
    compiled = builder.compile()

    final = await compiled.invoke(InstanceMwParentState(items=[7, 9]))
    await compiled.drain()
    assert final.results == [7, 9]
    # Each instance ran twice (1 fail + 1 success).
    assert instance_attempts == {7: 2, 9: 2}


async def test_instance_middleware_sees_fan_out_index() -> None:
    # An instance_middleware that reads current_fan_out_index() / its chain
    # observes the instance's own index: the engine sets the lineage ContextVars
    # around the middleware chain, not only inside node bodies. (Regression --
    # the index was None here when only compiled.py set it, deeper in node
    # execution, so the middleware wrapping the inner subgraph saw nothing.)
    from openarmature.observability.correlation import (
        current_fan_out_index,
        current_fan_out_index_chain,
    )

    seen_index: dict[int, int | None] = {}
    seen_chain: dict[int, tuple[int | None, ...]] = {}

    class _RecordIndexMW:
        async def __call__(self, state: WorkerState, next_: Any, /) -> Any:
            # Key by the item so each instance is identifiable without relying
            # on the index under test.
            seen_index[state.item] = current_fan_out_index()
            seen_chain[state.item] = current_fan_out_index_chain()
            return await next_(state)

    async def compute(state: WorkerState) -> Mapping[str, Any]:
        return {"result": state.item}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", compute)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    parent_builder: GraphBuilder[InstanceMwParentState] = GraphBuilder(InstanceMwParentState)
    parent_builder.set_entry("process")
    parent_builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        instance_middleware=[_RecordIndexMW()],
    )
    parent_builder.add_edge("process", END)
    parent = parent_builder.compile()

    await parent.invoke(InstanceMwParentState(items=[10, 20, 30]))
    await parent.drain()

    # items 10/20/30 are fan-out indices 0/1/2 in order; the chain carries the
    # instance index at the leaf.
    assert seen_index == {10: 0, 20: 1, 30: 2}
    assert seen_chain == {10: (0,), 20: (1,), 30: (2,)}


async def test_instance_middleware_lineage_reset_on_failure() -> None:
    # The lineage ContextVars reset even when an instance fails: the binding's
    # finally runs on the exception path, so a failed instance leaks nothing
    # into the parent scope.
    from openarmature.observability.correlation import current_fan_out_index

    seen: list[int | None] = []

    class _RecordMW:
        async def __call__(self, state: WorkerState, next_: Any, /) -> Any:
            seen.append(current_fan_out_index())
            return await next_(state)

    async def boom(_state: WorkerState) -> Mapping[str, Any]:
        raise RuntimeError("boom")

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("boom")
    inner_builder.add_node("boom", boom)
    inner_builder.add_edge("boom", END)
    inner = inner_builder.compile()

    parent_builder: GraphBuilder[InstanceMwParentState] = GraphBuilder(InstanceMwParentState)
    parent_builder.set_entry("process")
    parent_builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        instance_middleware=[_RecordMW()],
        concurrency=1,
    )
    parent_builder.add_edge("process", END)
    parent = parent_builder.compile()

    with pytest.raises(NodeException):
        await parent.invoke(InstanceMwParentState(items=[1, 2]))
    await parent.drain()

    # The middleware saw the instance index (the bind happened) ...
    assert seen and all(idx is not None for idx in seen)
    # ... and the bind's finally reset it despite the failure.
    assert current_fan_out_index() is None


# ---------------------------------------------------------------------------
# Fan-in determinism under nondeterministic completion order (§9.4)
# ---------------------------------------------------------------------------


class DetParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: Annotated[list[int], append] = Field(default_factory=list[int])


async def _run_with_random_delays(seed: int) -> list[int]:
    """Run a fan-out where each instance sleeps a random duration before
    returning. The collected list MUST preserve input order regardless
    of completion timing."""
    rng = random.Random(seed)

    async def slow(state: WorkerState) -> Mapping[str, Any]:
        await asyncio.sleep(rng.uniform(0, 0.005))
        return {"result": state.item}

    inner_builder: GraphBuilder[WorkerState] = GraphBuilder(WorkerState)
    inner_builder.set_entry("compute")
    inner_builder.add_node("compute", slow)
    inner_builder.add_edge("compute", END)
    inner = inner_builder.compile()

    builder: GraphBuilder[DetParentState] = GraphBuilder(DetParentState)
    builder.set_entry("process")
    builder.add_fan_out_node(
        "process",
        subgraph=inner,
        items_field="items",
        item_field="item",
        collect_field="result",
        target_field="results",
        concurrency=None,  # unbounded — maximum nondeterminism in completion order
    )
    builder.add_edge("process", END)
    compiled = builder.compile()
    final = await compiled.invoke(DetParentState(items=list(range(20))))
    await compiled.drain()
    return list(final.results)


async def test_fan_in_preserves_input_order_under_random_completion_timing() -> None:
    """target_field is in instance-index order, NOT completion
    order. Run the same fan-out N times with different random sleep
    seeds; every run produces the same result list."""
    expected = list(range(20))
    for seed in range(8):
        result = await _run_with_random_delays(seed)
        assert result == expected


# ---------------------------------------------------------------------------
# Compile-time errors
# ---------------------------------------------------------------------------


class _CompileTestState(State):
    items: list[int] = Field(default_factory=list[int])
    not_a_list: int = 0
    results: Annotated[list[int], append] = Field(default_factory=list[int])


def test_compile_error_count_mode_ambiguous_when_both_specified() -> None:
    """Specifying both items_field AND count is a compile
    error with category fan_out_count_mode_ambiguous."""
    inner = _build_doubler()
    builder: GraphBuilder[_CompileTestState] = GraphBuilder(_CompileTestState)
    with pytest.raises(FanOutCountModeAmbiguous):
        builder.add_fan_out_node(
            "process",
            subgraph=inner,
            items_field="items",
            item_field="item",
            count=3,  # invalid — both items_field and count
            collect_field="result",
            target_field="results",
        )


def test_compile_error_count_mode_ambiguous_when_neither_specified() -> None:
    inner = _build_doubler()
    builder: GraphBuilder[_CompileTestState] = GraphBuilder(_CompileTestState)
    with pytest.raises(FanOutCountModeAmbiguous):
        builder.add_fan_out_node(
            "process",
            subgraph=inner,
            collect_field="result",
            target_field="results",
            # no items_field, no count
        )


def test_compile_error_field_not_list() -> None:
    """items_field must reference a list-typed parent field. A non-list
    type is a compile error with category
    fan_out_field_not_list."""
    inner = _build_doubler()
    builder: GraphBuilder[_CompileTestState] = GraphBuilder(_CompileTestState)
    with pytest.raises(FanOutFieldNotList):
        builder.add_fan_out_node(
            "process",
            subgraph=inner,
            items_field="not_a_list",  # int field, not list
            item_field="item",
            collect_field="result",
            target_field="results",
        )


def test_compile_error_inputs_references_undeclared_parent_field() -> None:
    """``inputs`` mapping entries MUST refer to declared fields on both
    sides. An undeclared parent field raises
    ``mapping_references_undeclared_field`` at registration time."""
    from openarmature.graph import MappingReferencesUndeclaredField

    inner = _build_doubler()
    builder: GraphBuilder[_CompileTestState] = GraphBuilder(_CompileTestState)
    with pytest.raises(MappingReferencesUndeclaredField):
        builder.add_fan_out_node(
            "process",
            subgraph=inner,
            items_field="items",
            item_field="item",
            collect_field="result",
            target_field="results",
            inputs={"extra": "no_such_parent_field"},  # parent side undeclared
        )


def test_compile_error_extra_outputs_references_undeclared_subgraph_field() -> None:
    """Same shape as inputs validation, on the extra_outputs side: a
    subgraph field reference that the inner schema doesn't declare
    raises ``mapping_references_undeclared_field``."""
    from openarmature.graph import MappingReferencesUndeclaredField

    inner = _build_doubler()
    builder: GraphBuilder[_CompileTestState] = GraphBuilder(_CompileTestState)
    with pytest.raises(MappingReferencesUndeclaredField):
        builder.add_fan_out_node(
            "process",
            subgraph=inner,
            items_field="items",
            item_field="item",
            collect_field="result",
            target_field="results",
            extra_outputs={"results": "no_such_subgraph_field"},  # subgraph side undeclared
        )
