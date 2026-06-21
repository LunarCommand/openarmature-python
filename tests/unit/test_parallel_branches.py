"""Unit tests for the parallel-branches runtime.

Covers edge cases the conformance fixtures exercise only
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
    FailureIsolatedEvent,
    GraphBuilder,
    MappingReferencesUndeclaredField,
    NodeEvent,
    ObserverEvent,
    ParallelBranchesBranchFailed,
    ParallelBranchesInvalidBranchSpec,
    ParallelBranchesNoBranches,
    State,
    append,
    merge,
)
from openarmature.graph.middleware import (
    FailureIsolationMiddleware,
    RetryConfig,
    RetryMiddleware,
    deterministic_backoff,
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


class ConditionalState(State):
    run_vector: bool = False
    alpha_result: int = 0
    beta_result: int = 0


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
# Branch middleware — state space (§11.7)
# ---------------------------------------------------------------------------


async def test_branch_middleware_degraded_update_projects_through_outputs() -> None:
    # Regression: branch middleware wraps the subgraph invocation (§11.7),
    # so the chain operates in the branch subgraph's state space. A
    # middleware that short-circuits with a subgraph-space partial update —
    # here FailureIsolation's degraded_update writing the subgraph field
    # ``b_out`` — MUST project to the parent through the branch's
    # ``outputs`` mapping, exactly like a real subgraph result. Before the
    # fix the ``outputs`` projection ran INSIDE the middleware chain, so the
    # degraded_update reached the parent as ``b_out`` and tripped
    # extra-field validation (ParentState has no ``b_out``).
    isolation = FailureIsolationMiddleware(
        degraded_update={"b_out": 99},
        event_name="beta_isolated",
    )
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "beta": BranchSpec(
                    subgraph=_build_beta_raises("boom"),
                    outputs={"beta_result": "b_out"},
                    middleware=(isolation,),
                ),
            },
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    # The branch failed; FailureIsolation degraded it in subgraph space
    # (b_out=99); ``outputs`` projected b_out -> parent beta_result.
    assert final.beta_result == 99


async def test_branch_middleware_success_path_projects_subgraph_output() -> None:
    # Guards the other side of the fix: with branch middleware present but
    # the branch SUCCEEDING, the real subgraph output (not the degraded
    # value) must still project through ``outputs``. Confirms moving the
    # projection outside the middleware chain left the success path intact.
    isolation = FailureIsolationMiddleware(
        degraded_update={"a_out": 99},
        event_name="alpha_isolated",
    )
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),  # returns a_out=1
                    outputs={"alpha_result": "a_out"},
                    middleware=(isolation,),
                ),
            },
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    assert final.alpha_result == 1  # real subgraph output, not the degraded 99


async def test_branch_middleware_degraded_update_omitting_field_skips_contribution() -> None:
    # Leniency: a degraded_update that does not cover a projected
    # ``outputs`` sub-field contributes nothing for that field rather than
    # raising. The §11.4 buffer-then-merge model already merges partial
    # contributions, so the parent field keeps its prior value. Here the
    # branch degrades with an EMPTY update, so ``beta_result`` is never
    # contributed and stays at its ParentState default (0). A hard miss
    # would defeat the point of failure isolation (the resilience primitive
    # would itself crash the invocation).
    isolation = FailureIsolationMiddleware(
        degraded_update={},
        event_name="beta_isolated",
    )
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "beta": BranchSpec(
                    subgraph=_build_beta_raises("boom"),
                    outputs={"beta_result": "b_out"},
                    middleware=(isolation,),
                ),
            },
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    assert final.beta_result == 0  # never contributed; parent default retained


async def test_branch_middleware_isolation_wraps_retry_degrades_after_exhaustion() -> None:
    # Fixture-064-Case-1-shaped at a branch: middleware [failure_isolation,
    # retry] (outer-to-inner). The branch's node fails on every attempt;
    # retry exhausts its two attempts and re-raises; failure_isolation
    # catches the exhausted exception and degrades in subgraph space, which
    # projects to the parent. Exercises the state-space fix through a real
    # multi-middleware chain rather than a single frame.
    isolation = FailureIsolationMiddleware(
        degraded_update={"b_out": 99},
        event_name="beta_isolated",
    )
    retry = RetryMiddleware(
        RetryConfig(
            max_attempts=2,
            classifier=lambda _exc, _state: True,
            backoff=deterministic_backoff(0),
        )
    )
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("dispatcher")
        .add_parallel_branches_node(
            "dispatcher",
            branches={
                "beta": BranchSpec(
                    subgraph=_build_beta_raises("boom"),
                    outputs={"beta_result": "b_out"},
                    middleware=(isolation, retry),
                ),
            },
        )
        .add_edge("dispatcher", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    assert final.beta_result == 99


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


# ---------------------------------------------------------------------------
# Inline-callable branches + conditional ``when`` (proposal 0075, §11.1.1 /
# §11.4 / §11.10)
# ---------------------------------------------------------------------------


class _CategorizedError(RuntimeError):
    """A test exception carrying a ``category`` so the cause-chain
    classifier (and a branch FailureIsolationMiddleware's ``catch``)
    resolves it the way the framework resolves engine-categorized errors."""

    def __init__(self, message: str, category: str) -> None:
        super().__init__(message)
        self.category = category


async def _capture_events(events: list[ObserverEvent]) -> Any:
    """Build an observer that appends every delivered event to ``events``."""

    async def observe(event: ObserverEvent) -> None:
        events.append(event)

    return observe


async def test_callable_branches_merge_to_disjoint_parent_fields() -> None:
    # §11.1.1 / §11.4: two inline-callable branches (no subgraph, no
    # projection) run concurrently; each returns a parent-shaped partial
    # that merges into a disjoint parent field via the parent reducer.
    async def vector(_state: ParentState) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    async def fts(_state: ParentState) -> Mapping[str, Any]:
        return {"beta_result": 2}

    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={
                "vector": BranchSpec(call=vector),
                "fts": BranchSpec(call=fts),
            },
        )
        .add_edge("retrieve", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    assert final.alpha_result == 1
    assert final.beta_result == 2


async def test_callable_branch_emits_one_started_completed_pair_keyed_by_branch_name() -> None:
    # graph-engine §6 / observability §5.7: a callable branch has no inner
    # nodes, so it IS the unit — it emits exactly one started/completed pair
    # keyed by its branch_name (node_name == branch_name), not the per-node
    # stream a subgraph branch produces.
    events: list[ObserverEvent] = []

    async def vector(_state: ParentState) -> Mapping[str, Any]:
        return {"alpha_result": 7}

    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node("retrieve", branches={"vector": BranchSpec(call=vector)})
        .add_edge("retrieve", END)
        .compile()
    )
    compiled.attach_observer(await _capture_events(events))
    await compiled.invoke(ParentState())
    await compiled.drain()

    branch_events = [e for e in events if isinstance(e, NodeEvent) and e.branch_name == "vector"]
    assert [e.phase for e in branch_events] == ["started", "completed"]
    assert all(e.node_name == "vector" for e in branch_events)
    # Emitted at the pb NODE's own namespace (not a descended branch
    # namespace), carrying no parallel_branches_config — that shape (a
    # branch_name at the node's namespace, no config) is what identifies a
    # callable branch to the observers, which render it as the branch's
    # single dispatch span with no inner-node span.
    assert all(e.namespace == ("retrieve",) for e in branch_events)
    assert all(e.parallel_branches_config is None for e in branch_events)
    completed = branch_events[1]
    assert completed.error is None
    assert isinstance(completed.post_state, ParentState)
    assert completed.post_state.alpha_result == 7


async def test_node_event_branch_count_excludes_when_skipped_branches() -> None:
    # Proposal 0075: the NODE event's parallel_branches_config.branch_count is
    # the number of branches that DISPATCH (when-skipped branches excluded),
    # while branch_names stays the full declared set. The two answer different
    # questions ("how many ran" vs "what was declared").
    events: list[ObserverEvent] = []

    async def vector(_state: ConditionalState) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    async def fts(_state: ConditionalState) -> Mapping[str, Any]:
        return {"beta_result": 2}

    compiled = (
        GraphBuilder(ConditionalState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={
                "vector": BranchSpec(call=vector, when=lambda s: s.run_vector),
                "fts": BranchSpec(call=fts),
            },
        )
        .add_edge("retrieve", END)
        .compile()
    )
    compiled.attach_observer(await _capture_events(events))
    await compiled.invoke(ConditionalState(run_vector=False))  # vector skipped
    await compiled.drain()

    node_started = next(
        e
        for e in events
        if isinstance(e, NodeEvent)
        and e.node_name == "retrieve"
        and e.phase == "started"
        and e.parallel_branches_config is not None
    )
    config = node_started.parallel_branches_config
    assert config is not None
    assert config.branch_count == 1  # only fts dispatched
    assert config.branch_names == ("vector", "fts")  # full declared set, insertion order


async def test_callable_branch_event_attempt_index_tracks_node_retry() -> None:
    # PR #175 review: under node-level retry the callable branch's
    # started/completed pair carries the NODE's active attempt index (the same
    # value the NODE's own event uses), not a hardcoded 0. A flaky callable
    # fails the first node attempt and succeeds on the retry.
    events: list[ObserverEvent] = []
    calls = {"n": 0}

    async def flaky(_state: ParentState) -> Mapping[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"alpha_result": 1}

    node_retry = RetryMiddleware(
        RetryConfig(
            max_attempts=2,
            classifier=lambda _exc, _state: True,  # retry any failure
            backoff=deterministic_backoff(0.0),
        )
    )
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={"vector": BranchSpec(call=flaky)},
            middleware=[node_retry],
        )
        .add_edge("retrieve", END)
        .compile()
    )
    compiled.attach_observer(await _capture_events(events))
    final = await compiled.invoke(ParentState())
    await compiled.drain()

    assert final.alpha_result == 1  # succeeded on the second node attempt
    branch_started = [
        e for e in events if isinstance(e, NodeEvent) and e.branch_name == "vector" and e.phase == "started"
    ]
    assert [e.attempt_index for e in branch_started] == [0, 1]


async def test_when_false_skips_branch_no_contribution_no_event() -> None:
    # §11.10: a branch whose ``when`` returns false is skipped entirely —
    # not dispatched, no contribution (its field stays at the default), and
    # no observer events. The sibling with no ``when`` runs normally.
    events: list[ObserverEvent] = []

    async def vector(_state: ParentState) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    async def fts(_state: ParentState) -> Mapping[str, Any]:
        return {"beta_result": 2}

    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={
                "vector": BranchSpec(call=vector, when=lambda _s: False),
                "fts": BranchSpec(call=fts),
            },
        )
        .add_edge("retrieve", END)
        .compile()
    )
    compiled.attach_observer(await _capture_events(events))
    final = await compiled.invoke(ParentState())
    await compiled.drain()

    assert final.alpha_result == 0  # skipped -> no contribution -> default
    assert final.beta_result == 2
    assert [e for e in events if isinstance(e, NodeEvent) and e.branch_name == "vector"] == []


async def test_when_true_dispatches_branch() -> None:
    # §11.10: a ``when`` reading dispatch-time parent state; true dispatches.
    async def vector(_state: ConditionalState) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    compiled = (
        GraphBuilder(ConditionalState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={"vector": BranchSpec(call=vector, when=lambda s: s.run_vector)},
        )
        .add_edge("retrieve", END)
        .compile()
    )
    final = await compiled.invoke(ConditionalState(run_vector=True))
    await compiled.drain()
    assert final.alpha_result == 1


async def test_callable_branch_failure_isolation_degrades_and_emits_event() -> None:
    # §11.7: per-leg failure isolation on a callable branch is the existing
    # branch-middleware contract. A callable that raises a categorized error
    # is caught by its FailureIsolationMiddleware, degrades to the configured
    # update, and emits a FailureIsolatedEvent whose category resolves to the
    # originating error; the degraded branch "succeeds", so fail_fast is NOT
    # triggered and the sibling completes.
    events: list[ObserverEvent] = []

    async def vector(_state: ParentState) -> Mapping[str, Any]:
        raise _CategorizedError("vector store down", "provider_unavailable")

    async def fts(_state: ParentState) -> Mapping[str, Any]:
        return {"beta_result": 2}

    isolation = FailureIsolationMiddleware(
        degraded_update={"alpha_result": -1},
        event_name="vector_isolated",
    )
    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={
                "vector": BranchSpec(call=vector, middleware=(isolation,)),
                "fts": BranchSpec(call=fts),
            },
            error_policy="fail_fast",
        )
        .add_edge("retrieve", END)
        .compile()
    )
    compiled.attach_observer(await _capture_events(events))
    final = await compiled.invoke(ParentState())
    await compiled.drain()

    assert final.alpha_result == -1
    assert final.beta_result == 2
    iso = [e for e in events if isinstance(e, FailureIsolatedEvent)]
    assert len(iso) == 1
    assert iso[0].event_name == "vector_isolated"
    assert iso[0].caught_exception.category == "provider_unavailable"


async def test_mixed_subgraph_and_callable_branches() -> None:
    # §11.1.1: a node MAY mix subgraph branches and callable branches freely.
    async def fts(_state: ParentState) -> Mapping[str, Any]:
        return {"beta_result": 2}

    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={
                "alpha": BranchSpec(
                    subgraph=_build_alpha_succeeds(),
                    outputs={"alpha_result": "a_out"},
                ),
                "fts": BranchSpec(call=fts),
            },
        )
        .add_edge("retrieve", END)
        .compile()
    )
    final = await compiled.invoke(ParentState())
    await compiled.drain()
    assert final.alpha_result == 1
    assert final.beta_result == 2


async def test_all_branches_skipped_is_noop() -> None:
    # §11.10: all-skipped is a valid no-op — the node contributes nothing
    # and the parent state is unchanged. Distinct from the compile-time
    # parallel_branches_no_branches (an empty DECLARED mapping).
    async def vector(_state: ParentState) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={"vector": BranchSpec(call=vector, when=lambda _s: False)},
        )
        .add_edge("retrieve", END)
        .compile()
    )
    final = await compiled.invoke(ParentState(alpha_result=99))
    await compiled.drain()
    assert final.alpha_result == 99  # untouched no-op


async def test_all_branches_skipped_collect_sets_empty_errors_field() -> None:
    # §11.10 under collect: an all-skipped node still completes; with an
    # errors_field declared, no branch ran so the field is the empty list.
    async def vector(_state: ParentWithErrors) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    compiled = (
        GraphBuilder(ParentWithErrors)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={"vector": BranchSpec(call=vector, when=lambda _s: False)},
            error_policy="collect",
            errors_field="branch_errors",
        )
        .add_edge("retrieve", END)
        .compile()
    )
    final = await compiled.invoke(ParentWithErrors())
    await compiled.drain()
    assert final.alpha_result == 0
    assert final.branch_errors == []


async def test_callable_branch_unisolated_failure_fail_fast() -> None:
    # A callable branch that raises with no isolating middleware propagates
    # like a subgraph branch: wrapped as ParallelBranchesBranchFailed
    # carrying the branch_name, with the originating error in the chain.
    async def vector(_state: ParentState) -> Mapping[str, Any]:
        raise _CategorizedError("boom", "provider_unavailable")

    compiled = (
        GraphBuilder(ParentState)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={"vector": BranchSpec(call=vector)},
            error_policy="fail_fast",
        )
        .add_edge("retrieve", END)
        .compile()
    )
    with pytest.raises(ParallelBranchesBranchFailed) as excinfo:
        await compiled.invoke(ParentState())
    await compiled.drain()
    assert excinfo.value.branch_name == "vector"


async def test_callable_branch_failure_collect_records_error() -> None:
    # Under collect, a failing callable branch records into errors_field
    # exactly like a subgraph branch (node_exception category over the wrap),
    # and the sibling's contribution still merges.
    async def vector(_state: ParentWithErrors) -> Mapping[str, Any]:
        raise _CategorizedError("boom", "provider_unavailable")

    async def fts(_state: ParentWithErrors) -> Mapping[str, Any]:
        return {"beta_result": 2}

    compiled = (
        GraphBuilder(ParentWithErrors)
        .set_entry("retrieve")
        .add_parallel_branches_node(
            "retrieve",
            branches={
                "vector": BranchSpec(call=vector),
                "fts": BranchSpec(call=fts),
            },
            error_policy="collect",
            errors_field="branch_errors",
        )
        .add_edge("retrieve", END)
        .compile()
    )
    final = await compiled.invoke(ParentWithErrors())
    await compiled.drain()
    assert final.beta_result == 2
    assert len(final.branch_errors) == 1
    assert final.branch_errors[0]["branch_name"] == "vector"


# --- builder validation: exactly one of subgraph / call (§11.1.1) ---


def test_branch_with_both_subgraph_and_call_rejected() -> None:
    async def vector(_state: ParentState) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(ParallelBranchesInvalidBranchSpec) as excinfo:
        builder.add_parallel_branches_node(
            "retrieve",
            branches={"vector": BranchSpec(subgraph=_build_alpha_succeeds(), call=vector)},
        )
    assert excinfo.value.category == "parallel_branches_invalid_branch_spec"


def test_branch_with_neither_subgraph_nor_call_rejected() -> None:
    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(ParallelBranchesInvalidBranchSpec):
        builder.add_parallel_branches_node("retrieve", branches={"vector": BranchSpec()})


def test_callable_branch_with_inputs_or_outputs_rejected() -> None:
    async def vector(_state: ParentState) -> Mapping[str, Any]:
        return {"alpha_result": 1}

    builder: GraphBuilder[ParentState] = GraphBuilder(ParentState)
    with pytest.raises(ParallelBranchesInvalidBranchSpec):
        builder.add_parallel_branches_node(
            "retrieve",
            branches={"vector": BranchSpec(call=vector, outputs={"alpha_result": "x"})},
        )
