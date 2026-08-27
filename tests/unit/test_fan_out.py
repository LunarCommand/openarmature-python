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


async def test_sibling_branches_do_not_share_fan_out_state() -> None:
    # Two parallel branches, each a subgraph containing a fan-out node with the
    # SAME node name. Branch names never enter the namespace, and a branch
    # descent contributes only None entries to `fan_out_index_chain` -- which the
    # fan-out lineage comprehension filters out -- so before the branch axis
    # existed both branches built an identical execution-state key. The second
    # branch found the first's instances already `completed` and rolled its
    # results forward, returning results it never computed.
    #
    # This test is BEHAVIOURAL and therefore schedule-dependent. Read the
    # companion test below,
    # `test_sibling_branches_register_under_distinct_keys`, as the load-bearing
    # one: it asserts the invariant directly and no interleaving can disarm it.
    #
    # Properties this test needs, and the honest limit of them:
    #
    # 1. DISTINGUISHABLE per-branch seeds. With identical seeds both branches
    #    produce the same answer and the assertions hold against the defect.
    # 2. NO SUSPENSION POINT between the two branches' registrations. The leaf
    #    body having no `await` is necessary but NOT sufficient: most of that
    #    window is engine code this test does not own, so an await added in
    #    `_step_fan_out_node`, a dispatch hook, or an observer await disarms it
    #    from outside. Measured: revert the branch axis AND add one
    #    `await asyncio.sleep(0)` to the leaf and all three assertions below
    #    pass while the two branches still share one execution-state entry.
    #
    # The execution log is asserted as well as the outputs: the failure mode is
    # that bodies never run, and matching outputs alone cannot see that.
    from openarmature.graph.parallel_branches import BranchSpec

    executed: list[int] = []

    class _Top(State):
        out_a: list[int] = []
        out_b: list[int] = []

    class _BranchA(State):
        seeds: list[int] = [0, 1]
        out: list[int] = []

    class _BranchB(State):
        seeds: list[int] = [7, 8]
        out: list[int] = []

    class _Leaf(State):
        seed: int = 0
        marker: int = 0

    async def _leaf(s: _Leaf) -> dict[str, Any]:
        executed.append(s.seed)
        return {"marker": s.seed + 100}

    def _branch_graph(state_cls: type[State]) -> CompiledGraph[Any]:
        leaf = GraphBuilder(_Leaf).add_node("g", _leaf).add_edge("g", END).set_entry("g").compile()
        return (
            GraphBuilder(state_cls)
            .add_fan_out_node(
                "fo",
                subgraph=leaf,
                items_field="seeds",
                item_field="seed",
                collect_field="marker",
                target_field="out",
            )
            .add_edge("fo", END)
            .set_entry("fo")
            .compile()
        )

    graph = (
        GraphBuilder(_Top)
        .add_parallel_branches_node(
            "pb",
            branches={
                "a": BranchSpec(subgraph=_branch_graph(_BranchA), outputs={"out_a": "out"}),
                "b": BranchSpec(subgraph=_branch_graph(_BranchB), outputs={"out_b": "out"}),
            },
        )
        .add_edge("pb", END)
        .set_entry("pb")
        .compile()
    )
    final = await graph.invoke(_Top())

    assert sorted(executed) == [0, 1, 7, 8], (
        f"every branch's item bodies must run; got {sorted(executed)}. "
        "A short log means one branch rolled the other's results forward."
    )
    assert final.out_a == [100, 101]
    assert final.out_b == [107, 108]


async def test_fan_out_progress_is_popped_after_it_completes() -> None:
    # The cleanup `pop` in the step runner must build the SAME key the fan-out
    # registered under. It passes a default, so a key that does not match
    # silently no-ops: the entry survives and is carried into every later save
    # in the invocation as stale progress.
    #
    # This uses a TOP-LEVEL fan-out deliberately. An earlier version nested it
    # in a parallel branch, to exercise the branch component of the key -- and
    # a later change in this same branch, which stops projecting branch-nested
    # entries onto the record at all, made that version vacuous: the record was
    # empty whether or not the pop worked. It was mutation-verified before that
    # change and not re-verified after, which is exactly how a live test dies
    # quietly. Only a fan-out whose progress actually reaches the record can
    # observe the pop through the record.
    #
    # The branch component of the cleanup key is covered structurally instead,
    # by `test_registration_and_cleanup_build_the_same_progress_key`: all three
    # sites build the key through one shared builder, so they cannot disagree
    # on any component.
    from openarmature.checkpoint import InMemoryCheckpointer

    class _Top(State):
        seeds: list[int] = [0, 1]
        out: list[int] = []
        done: bool = False

    class _Leaf(State):
        seed: int = 0
        marker: int = 0

    async def _leaf(s: _Leaf) -> dict[str, Any]:
        return {"marker": s.seed + 1}

    async def _after(_s: _Top) -> dict[str, Any]:
        return {"done": True}

    leaf = GraphBuilder(_Leaf).add_node("g", _leaf).add_edge("g", END).set_entry("g").compile()
    cp = InMemoryCheckpointer()
    builder = (
        GraphBuilder(_Top)
        .add_fan_out_node(
            "fo",
            subgraph=leaf,
            items_field="seeds",
            item_field="seed",
            collect_field="marker",
            target_field="out",
        )
        .add_node("after", _after)
        .add_edge("fo", "after")
        .add_edge("after", END)
        .set_entry("fo")
    )
    builder.with_checkpointer(cp)
    await builder.compile().invoke(_Top())

    # Public API only. `list()` carries no `fan_out_progress` (a
    # `CheckpointSummary` holds the id, correlation id, timestamp and node
    # count), so it supplies the id and `load()` supplies the record. There is
    # deliberately no `hasattr` guard: a change to the checkpointer surface
    # should break this at the call, not quietly empty the list and leave the
    # behavioural assertion below unreached.
    summaries = list(await cp.list())
    assert summaries, "expected the checkpointer to have saved at least one record"
    last = await cp.load(summaries[-1].invocation_id)
    assert last is not None
    # The last record is written by `after`, long after the fan-out completed
    # and its entry was popped, so it must carry no fan-out progress at all.
    assert last.fan_out_progress == (), (
        f"stale fan-out progress survived into a later save: {last.fan_out_progress}. "
        "The cleanup key no longer matches the registration key."
    )


def test_sibling_branches_register_under_distinct_keys() -> None:
    # The schedule-independent counterpart to the behavioural test above, and
    # the one to trust. It asserts the invariant at the key builder rather than
    # through a race, so no interleaving anywhere in the engine can disarm it.
    #
    # Two contexts differing ONLY in which branch they descended into must
    # produce different execution-state keys. They share a namespace, because a
    # branch name never enters it -- only the parallel-branches node name does.
    from openarmature.graph.observer import _InvocationContext, _QueuedItem

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    root = _InvocationContext(queue=queue, graph_attached=(), invocation_scoped=())

    class _S(State):
        pass

    parent = _S()
    in_a = root.descend_into_parallel_branch("pb", parent, (), branch_name="a")
    in_b = root.descend_into_parallel_branch("pb", parent, (), branch_name="b")

    assert in_a.namespace_prefix == in_b.namespace_prefix, (
        "precondition: branch names do not enter the namespace, so the two "
        "branches must share it -- that is why the key needs the branch axis"
    )

    def _key(ctx: Any, node_name: str) -> tuple[Any, ...]:
        # The identity `FanOutNode.run` registers under.
        return (
            ctx.namespace_prefix,
            node_name,
            tuple(i for i in ctx.fan_out_index_chain if i is not None),
            tuple(b for b in ctx.branch_name_chain if b is not None),
        )

    assert _key(in_a, "fo") != _key(in_b, "fo"), (
        f"sibling branches must not share a fan-out execution-state key; got {_key(in_a, 'fo')} for both"
    )
    # And the branch axis is what separates them, not something incidental.
    assert _key(in_a, "fo")[:3] == _key(in_b, "fo")[:3]


async def test_sibling_branches_with_unequal_item_counts_do_not_raise() -> None:
    # The defect's OTHER observable spelling. With equal item counts it returns
    # silently wrong results; with UNEQUAL counts the second branch finds an
    # entry whose `instance_count` differs from its own resolved count and
    # raises `CheckpointRecordInvalid` -- on a fresh invocation with no
    # checkpointer attached, complaining about a record that does not exist.
    #
    # Measured before the fix:
    #   ParallelBranchesBranchFailed: branch 'b' raised CheckpointRecordInvalid:
    #   ... saved instance_count=2 does not match resolved instance_count=3 on resume
    #
    # It also means `FanOutNode.run`'s comment that the count-drift path can only
    # fire on resume, since the progress dict is empty on a fresh first run, was
    # false for the branch axis until this fix made it true.
    from openarmature.graph.parallel_branches import BranchSpec

    executed: list[int] = []

    class _Top(State):
        oa: list[int] = []
        ob: list[int] = []

    class _A(State):
        seeds: list[int] = [0, 1]
        out: list[int] = []

    class _B(State):
        seeds: list[int] = [7, 8, 9]
        out: list[int] = []

    class _Leaf(State):
        seed: int = 0
        marker: int = 0

    async def _leaf(s: _Leaf) -> dict[str, Any]:
        executed.append(s.seed)
        return {"marker": s.seed + 100}

    def _sub(cls: type[State]) -> CompiledGraph[Any]:
        leaf = GraphBuilder(_Leaf).add_node("g", _leaf).add_edge("g", END).set_entry("g").compile()
        return (
            GraphBuilder(cls)
            .add_fan_out_node(
                "fo",
                subgraph=leaf,
                items_field="seeds",
                item_field="seed",
                collect_field="marker",
                target_field="out",
            )
            .add_edge("fo", END)
            .set_entry("fo")
            .compile()
        )

    graph = (
        GraphBuilder(_Top)
        .add_parallel_branches_node(
            "pb",
            branches={
                "a": BranchSpec(subgraph=_sub(_A), outputs={"oa": "out"}),
                "b": BranchSpec(subgraph=_sub(_B), outputs={"ob": "out"}),
            },
        )
        .add_edge("pb", END)
        .set_entry("pb")
        .compile()
    )
    final = await graph.invoke(_Top())

    assert sorted(executed) == [0, 1, 7, 8, 9]
    assert final.oa == [100, 101]
    assert final.ob == [107, 108, 109]


def test_instance_lookup_finds_a_fan_out_across_an_intervening_branch() -> None:
    # A node inside a parallel branch, inside a fan-out instance, must still
    # resolve the ENCLOSING fan-out's per-instance state.
    #
    # Both lineage chains slice to the same depth, because every descent grows
    # `namespace_prefix`, `fan_out_index_chain` and `branch_name_chain` by
    # exactly one entry. Taking the branch chain unsliced leaked the branch
    # entry into a key built for an OUTER fan-out, so the lookup missed and
    # `completed_inner_positions` silently lost a position.
    #
    # Pinned at the function rather than through a run: branch descent sets
    # `checkpointer=None`, so nothing reaches this from inside a branch today.
    # The key must be right regardless of a policy in another module.
    from openarmature.graph.compiled import _find_innermost_fan_out_instance_state
    from openarmature.graph.observer import (
        _FanOutExecutionState,
        _FanOutInstanceState,
        _InvocationContext,
        _QueuedItem,
    )

    queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
    tracked = _FanOutInstanceState(state="completed", result=7)
    # What `FanOutNode.run` registers for fan-out "F" at the top level.
    registered = _FanOutExecutionState(
        fan_out_node_name="F",
        namespace=(),
        instance_count=2,
        instances=[_FanOutInstanceState(), tracked],
    )
    ctx = _InvocationContext(
        queue=queue,
        graph_attached=(),
        invocation_scoped=(),
        # Inside F's instance 1, then into parallel-branches node "pb", branch "a".
        namespace_prefix=("F", "pb"),
        fan_out_index_chain=(1, None),
        branch_name_chain=(None, "a"),
        fan_out_index=1,
        # F registers from the ROOT context, so both enclosing lineages are
        # empty; the instance index lives on the descendant's context, not on
        # the fan-out node's own registration key.
        fan_out_progress_state={((), "F", (), ()): registered},
    )

    assert _find_innermost_fan_out_instance_state(ctx) is tracked, (
        "the enclosing fan-out's instance state must resolve through an "
        "intervening branch descent; a leaked branch entry makes the key miss"
    )


def test_registration_and_cleanup_build_the_same_progress_key() -> None:
    # The registration in `FanOutNode.run` and the cleanup `pop` in the step
    # runner take their inputs from the SAME context and must agree exactly.
    # They disagreed silently: `pop` passes a default, so a divergence no-ops
    # and leaves stale `completed` progress in the shared dict with no signal.
    #
    # Both now call `fan_out_progress_key`, so the only way to diverge is to
    # stop calling it. This asserts the two sites resolve to that one builder,
    # which is what a hand-rolled copy at either site would break.
    import inspect

    from openarmature.graph import compiled as compiled_mod
    from openarmature.graph import fan_out as fan_out_mod
    from openarmature.graph.observer import fan_out_progress_key

    assert fan_out_mod.fan_out_progress_key is fan_out_progress_key
    assert compiled_mod.fan_out_progress_key is fan_out_progress_key

    # And no site rebuilds the identity by hand alongside calling it. The
    # filtering of None chain entries belongs in the builder only; a second
    # spelling is how the two drifted before.
    hand_rolled = "if i is not None"
    for mod in (fan_out_mod, compiled_mod):
        src = inspect.getsource(mod)
        assert f"context.fan_out_index_chain {hand_rolled}" not in src, (
            f"{mod.__name__} filters a lineage chain by hand; build the key "
            "through `fan_out_progress_key` so the three sites cannot drift"
        )
