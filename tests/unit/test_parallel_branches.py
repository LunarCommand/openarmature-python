"""Unit tests for the parallel-branches runtime (pipeline-utilities §11).

Covers spec corner cases the conformance fixtures exercise only
implicitly:

- compile-time empty-branches rejection
- compile-time empty-branch-name rejection
- compile-time projection validation (inputs/outputs reference declared
  fields on the right side of the projection direction)
- compile-time ``errors_field`` validation
- fail_fast: first failure cancels siblings; recoverable_state is the
  parent's pre-dispatch snapshot
- collect: per-branch errors recorded; successful branches' projections
  merge in branch insertion order
- branch insertion order determines fan-in merge order regardless of
  completion timing
- single-branch ``contributions`` are written through the parent's
  reducer (the ``_MultiContribution`` sentinel only fires for multi-
  branch fields)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Annotated, Any

import pytest

from openarmature.graph import (
    END,
    BranchSpec,
    CompiledGraph,
    GraphBuilder,
    MappingReferencesUndeclaredField,
    ParallelBranchesBranchFailed,
    ParallelBranchesNoBranches,
    State,
    append,
    merge,
)

# ---------------------------------------------------------------------------
# Shared schemas + helpers
# ---------------------------------------------------------------------------


class AlphaState(State):
    a_out: int = 0


class BetaState(State):
    b_out: int = 0


class GammaState(State):
    c_out: int = 0


class ParentState(State):
    alpha_result: int = 0
    beta_result: int = 0
    gamma_result: int = 0


def _build_alpha_succeeds() -> CompiledGraph[AlphaState]:
    async def a(_state: AlphaState) -> Mapping[str, Any]:
        return {"a_out": 1}

    return GraphBuilder(AlphaState).set_entry("a").add_node("a", a).add_edge("a", END).compile()


def _build_beta_succeeds() -> CompiledGraph[BetaState]:
    async def b(_state: BetaState) -> Mapping[str, Any]:
        return {"b_out": 2}

    return GraphBuilder(BetaState).set_entry("b").add_node("b", b).add_edge("b", END).compile()


def _build_beta_raises(message: str) -> CompiledGraph[BetaState]:
    async def b(_state: BetaState) -> Mapping[str, Any]:
        raise RuntimeError(message)

    return GraphBuilder(BetaState).set_entry("b").add_node("b", b).add_edge("b", END).compile()


def _build_gamma_succeeds() -> CompiledGraph[GammaState]:
    async def c(_state: GammaState) -> Mapping[str, Any]:
        return {"c_out": 3}

    return GraphBuilder(GammaState).set_entry("c").add_node("c", c).add_edge("c", END).compile()


# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def test_empty_branches_raises_at_compile_time() -> None:
    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(ParallelBranchesNoBranches) as excinfo:
        builder.add_parallel_branches_node("dispatcher", branches={})
    assert excinfo.value.category == "parallel_branches_no_branches"


def test_empty_branch_name_raises_at_compile_time() -> None:
    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(ValueError) as excinfo:
        builder.add_parallel_branches_node(
            "dispatcher",
            branches={
                "": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
            },
        )
    assert "non-empty" in str(excinfo.value)


def test_outputs_references_undeclared_parent_field() -> None:
    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        builder.add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"missing_parent_field": "a_out"},
                ),
            },
        )
    assert excinfo.value.side == "parent"


def test_outputs_references_undeclared_subgraph_field() -> None:
    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        builder.add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "missing_sub_field"},
                ),
            },
        )
    assert excinfo.value.side == "subgraph"


def test_inputs_references_undeclared_parent_field() -> None:
    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        builder.add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    inputs={"a_out": "missing_parent_field"},
                    outputs={"alpha_result": "a_out"},
                ),
            },
        )
    assert excinfo.value.side == "parent"


def test_errors_field_references_undeclared_parent_field() -> None:
    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(MappingReferencesUndeclaredField) as excinfo:
        builder.add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
            },
            error_policy="collect",
            errors_field="not_declared",
        )
    assert excinfo.value.side == "parent"


# ---------------------------------------------------------------------------
# Runtime — happy path
# ---------------------------------------------------------------------------


async def test_three_heterogeneous_branches_merge_to_parent() -> None:
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
                "beta": BranchSpec(
                    subgraph=_build_beta_succeeds(),
                    outputs={"beta_result": "b_out"},
                ),
                "gamma": BranchSpec(
                    subgraph=_build_gamma_succeeds(),
                    outputs={"gamma_result": "c_out"},
                ),
            },
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    assert final.alpha_result == 1
    assert final.beta_result == 2
    assert final.gamma_result == 3


# ---------------------------------------------------------------------------
# fail_fast policy
# ---------------------------------------------------------------------------


async def test_fail_fast_raises_branch_failed_with_branch_name() -> None:
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
                "beta": BranchSpec(
                    subgraph=_build_beta_raises("boom"),
                    outputs={"beta_result": "b_out"},
                ),
            },
            error_policy="fail_fast",
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    with pytest.raises(ParallelBranchesBranchFailed) as excinfo:
        await compiled.invoke(ParentState())
    await compiled.drain()
    assert excinfo.value.branch_name == "beta"
    # __cause__ chain: ParallelBranchesBranchFailed -> NodeException("b") -> RuntimeError("boom")
    inner = excinfo.value.__cause__
    assert inner is not None
    leaf: BaseException = inner
    while leaf.__cause__ is not None:
        leaf = leaf.__cause__
    assert str(leaf) == "boom"


async def test_fail_fast_recoverable_state_drops_buffered_contributions() -> None:
    # Per spec §11.5: on fail_fast, NO branch contributions are visible
    # in recoverable_state, including the first branch's successful
    # work (the buffer-and-apply semantic).
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                # Slow successful branch — its result must NOT land in
                # recoverable_state even though its inner-node update
                # may complete before fail-fast cancellation propagates.
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
                "beta": BranchSpec(
                    subgraph=_build_beta_raises("boom"),
                    outputs={"beta_result": "b_out"},
                ),
            },
            error_policy="fail_fast",
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    with pytest.raises(ParallelBranchesBranchFailed) as excinfo:
        await compiled.invoke(ParentState())
    await compiled.drain()
    snapshot = excinfo.value.recoverable_state.model_dump()
    # All defaults — alpha's contribution is NOT applied even though
    # its branch may have completed before cancellation landed.
    assert snapshot == {"alpha_result": 0, "beta_result": 0, "gamma_result": 0}


# ---------------------------------------------------------------------------
# collect policy
# ---------------------------------------------------------------------------


class ParentWithErrors(State):
    alpha_result: int = 0
    beta_result: int = 0
    gamma_result: int = 0
    branch_errors: Annotated[list[dict[str, Any]], append] = []


async def test_collect_records_branch_failures_in_errors_field() -> None:
    compiled = (
        GraphBuilder(ParentWithErrors)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
                "beta": BranchSpec(
                    subgraph=_build_beta_raises("boom"),
                    outputs={"beta_result": "b_out"},
                ),
                "gamma": BranchSpec(
                    subgraph=_build_gamma_succeeds(),
                    outputs={"gamma_result": "c_out"},
                ),
            },
            error_policy="collect",
            errors_field="branch_errors",
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(ParentWithErrors())
    await compiled.drain()
    # Successful branches' contributions land.
    assert final.alpha_result == 1
    assert final.gamma_result == 3
    # Failed branch's outputs do NOT fire — beta_result stays at default.
    assert final.beta_result == 0
    # One error record for beta carrying the spec-mandated keys.
    assert len(final.branch_errors) == 1
    rec = final.branch_errors[0]
    assert rec["branch_name"] == "beta"
    assert rec["category"] == "node_exception"


# ---------------------------------------------------------------------------
# Determinism — insertion order is the merge order
# ---------------------------------------------------------------------------


class MergedDictState(State):
    merged: Annotated[dict[str, Any], merge] = {}


def _build_writer(delay_s: float, value: str) -> CompiledGraph[MergedDictState]:
    """Subgraph that sleeps then writes ``{key: value}`` to ``merged``."""

    async def write(_state: MergedDictState) -> Mapping[str, Any]:
        await asyncio.sleep(delay_s)
        return {"merged": {"key": value}}

    return (
        GraphBuilder(MergedDictState)
        .set_entry("write")
        .add_node("write", write)
        .add_edge("write", END)
        .compile()
    )


async def test_branch_fan_in_order_follows_insertion_order_not_completion() -> None:
    # Per spec §11.8: when two branches write the same parent field,
    # the parent's reducer applies them in branch INSERTION order
    # regardless of which branch finishes first. We give alpha (first
    # in insertion order) a deliberately long delay and beta (second)
    # a short one, so completion order is beta-then-alpha. The merge
    # reducer applies alpha first, then beta — beta's value overrides.
    compiled = (
        GraphBuilder(MergedDictState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_writer(0.05, "alpha_value"),
                    outputs={"merged": "merged"},
                ),
                "beta": BranchSpec(
                    subgraph=_build_writer(0.005, "beta_value"),
                    outputs={"merged": "merged"},
                ),
            },
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(MergedDictState())
    await compiled.drain()
    # beta wrote after alpha (per insertion-order fan-in), so beta's
    # value wins the merge for ``key``.
    assert final.merged == {"key": "beta_value"}


# ---------------------------------------------------------------------------
# Single-branch field write (no _MultiContribution sentinel firing)
# ---------------------------------------------------------------------------


async def test_single_branch_field_writes_through_reducer_normally() -> None:
    # When only one branch contributes to a given parent field, the
    # value should flow through the parent's reducer as a plain value,
    # not as a _MultiContribution sentinel — the fan-in code only
    # synthesizes the sentinel for fields touched by multiple branches.
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
            },
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    assert final.alpha_result == 1


# ---------------------------------------------------------------------------
# Fail_fast cancellation drain — second failure absorbed silently
# ---------------------------------------------------------------------------


def _build_alpha_raises(message: str) -> CompiledGraph[AlphaState]:
    async def a(_state: AlphaState) -> Mapping[str, Any]:
        raise RuntimeError(message)

    return GraphBuilder(AlphaState).set_entry("a").add_node("a", a).add_edge("a", END).compile()


async def test_fail_fast_cancellation_drain_absorbs_residual_exceptions() -> None:
    # Per spec §11.5 + Q5 cancellation-drain note: under fail_fast,
    # the raise is committed to the FIRST failure observed; later
    # tasks may race past the cancellation point with their own
    # exceptions, but those are absorbed silently by the drain
    # ``gather(*, return_exceptions=True)``. No second exception
    # surfaces to the caller.
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_raises("first"),
                    outputs={"alpha_result": "a_out"},
                ),
                "beta": BranchSpec(
                    subgraph=_build_beta_raises("second"),
                    outputs={"beta_result": "b_out"},
                ),
            },
            error_policy="fail_fast",
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    with pytest.raises(ParallelBranchesBranchFailed) as excinfo:
        await compiled.invoke(ParentState())
    await compiled.drain()
    # The raise commits to the first observed failure; the dispatcher
    # picks deterministically from the FIRST_EXCEPTION wait — one of
    # the two branches surfaces.
    assert excinfo.value.branch_name in {"alpha", "beta"}
