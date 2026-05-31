# Spec: realizes pipeline-utilities §11 (parallel branches).

"""Parallel branches: concurrent dispatch of M heterogeneous compiled subgraphs.

Counterpart to :mod:`.fan_out`. Fan-out is data-driven (N items,
one subgraph, instantiated N times); parallel branches is
topology-driven (M heterogeneous compiled subgraphs declared
statically, run concurrently within a single parent invocation).

Each branch's :class:`BranchSpec` carries its own compiled
subgraph (with potentially different state schema, middleware,
topology), its own ``inputs`` / ``outputs`` projection mappings,
and its own optional ``middleware`` wrapping the whole branch
invocation as a unit (§11.7).

Buffer-then-apply semantics per §11.4: contributions are
collected during dispatch and merged deterministically once at
node completion, using the parent's reducer for each output
field. Branch insertion order determines both dispatch order
(§11.8) and merge tie-breaking when two branches write the same
parent field.

Error policies per §11.5:

- ``fail_fast``: first failure cancels still-running branches;
  the buffered contributions are discarded; the parallel-branches
  node raises ``ParallelBranchesBranchFailed`` with the failing
  branch's exception as ``__cause__``. ``recoverable_state``
  equals the parent state at the moment the node entered.
- ``collect``: all branches run to completion; successful
  branches' contributions merge; failed branches' errors land in
  the optional ``errors_field``.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from openarmature.observability.correlation import (
    _reset_branch_name,
    _set_branch_name,
)

from .errors import ParallelBranchesBranchFailed
from .middleware import ChainCall, Middleware, compose_chain
from .state import State

if TYPE_CHECKING:
    from .compiled import CompiledGraph
    from .observer import _InvocationContext

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BranchSpec[ChildT: State]:
    """One entry in a :class:`ParallelBranchesNode`'s branch mapping.

    Branches are heterogeneous: each spec MAY reference a different
    compiled subgraph with a different state schema. ``inputs`` /
    ``outputs`` follow the same shape as subgraph projection
    mappings (proposal 0002).

    Validation lives on the builder side
    (``GraphBuilder.add_parallel_branches_node``):
    ``mapping_references_undeclared_field`` for inputs/outputs
    referencing undeclared fields; ``parallel_branches_no_branches``
    for empty ``branches`` maps; ``ValueError`` for empty-string
    branch names.
    """

    subgraph: CompiledGraph[ChildT]
    inputs: Mapping[str, str] = field(default_factory=dict[str, str])
    outputs: Mapping[str, str] = field(default_factory=dict[str, str])
    middleware: tuple[Middleware, ...] = ()


@dataclass(frozen=True)
class ParallelBranchesNode[ParentT: State]:
    """A node that dispatches M heterogeneous compiled subgraphs
    concurrently per spec §11.

    The Node Protocol contract requires ``name``, ``middleware``,
    and ``run``. Like :class:`FanOutNode`, the engine recognizes
    this type in ``_invoke`` and calls ``run_with_context`` so the
    dispatcher has access to the invocation context for
    observer-attribution + namespace descent. ``run`` exists for
    Protocol conformance only and raises if anyone calls it
    directly.
    """

    name: str
    branches: dict[str, BranchSpec[Any]]
    error_policy: Literal["fail_fast", "collect"] = "fail_fast"
    errors_field: str | None = None
    middleware: tuple[Middleware, ...] = ()

    async def run(self, state: ParentT) -> Mapping[str, Any]:
        """Not implemented at this level. Dispatching parallel branches
        requires the engine's invocation context so each branch gets
        observer attribution and a namespaced descent; the engine
        calls :meth:`run_with_context` instead. This method exists
        only to satisfy the :class:`Node` Protocol and always raises
        :class:`NotImplementedError`."""
        del state
        raise NotImplementedError(
            "ParallelBranchesNode is dispatched by the graph engine; if you're "
            "seeing this, you've likely instantiated it outside an engine "
            "context (e.g., calling node.run(state) directly instead of "
            "compiled.invoke)."
        )

    async def run_with_context(
        self,
        state: ParentT,
        context: _InvocationContext,
    ) -> Mapping[str, Any]:
        """Execute the parallel-branches dispatch and return the
        merged partial update.

        Snapshot parent state, project per-branch initial states,
        dispatch M branches concurrently in insertion order, then
        either fail-fast on first error (cancelling the rest) or
        run to completion and merge per the configured error policy.
        """
        # ``contributions`` is the buffer per §11.4 — keyed by branch
        # name, holds each successful branch's projected outputs
        # (parent_field -> exit_value mapping) until the dispatch
        # completes and the dispatcher applies them in insertion
        # order via the parent's reducers.
        contributions: dict[str, Mapping[str, Any]] = {}
        # ``errors`` collects per-branch failures under ``collect``;
        # under ``fail_fast`` the dispatcher raises before this is
        # consulted.
        errors: list[dict[str, str]] = []

        async def run_branch(branch_name: str, spec: BranchSpec[Any]) -> Mapping[str, Any]:
            # Set the branch_name ContextVar inside the branch's
            # task scope so the OTel observer sees it on every
            # inner-node span. The task copies the current context
            # at spawn time, so this set happens inside the spawned
            # task body — not in the dispatcher loop.
            token = _set_branch_name(branch_name)
            try:
                # Per §11.2 projection in: subgraph fields not in
                # ``inputs`` use the subgraph's declared defaults;
                # named subgraph fields are initialized from the
                # corresponding parent field.
                parent_dump = state.model_dump()
                init: dict[str, Any] = {}
                for sub_field, parent_field in spec.inputs.items():
                    init[sub_field] = parent_dump[parent_field]
                initial = spec.subgraph.state_cls(**init)

                child_context = context.descend_into_parallel_branch(
                    parallel_branches_node_name=self.name,
                    parent_state=state,
                    sub_attached=tuple(spec.subgraph._attached_observers),  # noqa: SLF001
                    branch_name=branch_name,
                )

                async def innermost(s: Any) -> Mapping[str, Any]:
                    final_branch_state = await spec.subgraph._invoke(s, child_context)  # noqa: SLF001
                    # Per §11.4 projection out: only fields named
                    # in ``outputs`` contribute back to parent
                    # state; unnamed subgraph fields are discarded.
                    return {
                        parent_field: getattr(final_branch_state, sub_field)
                        for parent_field, sub_field in spec.outputs.items()
                    }

                chain: ChainCall = compose_chain(spec.middleware, innermost)
                return await chain(initial)
            finally:
                _reset_branch_name(token)

        # Spawn one task per branch, in insertion order. Per §11.8
        # the dispatch order is the branches dict's insertion order;
        # ``started`` events from the inner subgraphs interleave
        # arbitrarily but the branch-level dispatch ordering is
        # deterministic.
        ctx = contextvars.copy_context()
        tasks: list[tuple[str, asyncio.Task[Mapping[str, Any]]]] = []
        for branch_name, spec in self.branches.items():
            task = asyncio.create_task(
                run_branch(branch_name, spec),
                context=ctx.copy(),
            )
            tasks.append((branch_name, task))

        if self.error_policy == "fail_fast":
            return await self._fail_fast(state, tasks, contributions)
        return await self._collect(state, tasks, contributions, errors)

    async def _fail_fast(
        self,
        parent_state: Any,
        tasks: list[tuple[str, asyncio.Task[Mapping[str, Any]]]],
        contributions: dict[str, Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Fail-fast policy per spec §11.5.

        Wait for all branches; on first failure, cancel the rest
        and raise ``ParallelBranchesBranchFailed`` with the failing
        branch's exception as ``__cause__``. Buffered contributions
        are discarded (collect-then-apply means they never reached
        parent state). ``recoverable_state`` equals the parent
        state at the moment the node entered.
        """
        task_map = {t: name for name, t in tasks}
        try:
            done, pending = await asyncio.wait(
                [t for _, t in tasks],
                return_when=asyncio.FIRST_EXCEPTION,
            )
        except BaseException:
            # Defensive: if the dispatcher itself is cancelled,
            # drain the children before propagating.
            for _, t in tasks:
                t.cancel()
            await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
            raise

        # Find the first task that raised; cancel the rest.
        failed_name: str | None = None
        failed_cause: BaseException | None = None
        for t in done:
            if t.exception() is not None:
                failed_name = task_map[t]
                failed_cause = t.exception()
                break

        if failed_cause is None:
            # All tasks finished without raising (the wait can
            # return early when the last task finishes successfully).
            # Buffer the contributions in branch insertion order.
            for name, t in tasks:
                contributions[name] = t.result()
            return self._merge_contributions(contributions)

        # Cancel remaining + any pending; drain to absorb
        # CancelledError so it doesn't propagate as unhandled.
        for t in pending:
            t.cancel()
        # Subtle case worth flagging: a second task may race past
        # the cancellation point with its own raised exception (a
        # near-simultaneous failure). ``return_exceptions=True``
        # absorbs both the CancelledErrors AND that second exception
        # into the gather's return list. We discard them silently —
        # the raise is committed to the FIRST failure observed
        # above; logging stragglers at DEBUG helps post-mortem
        # analysis without muddying the raise contract.
        drained = await asyncio.gather(
            *(t for _, t in tasks if not t.done()),
            return_exceptions=True,
        )
        for residual in drained:
            if isinstance(residual, BaseException) and not isinstance(residual, asyncio.CancelledError):
                _log.debug(
                    "parallel-branches node %r: post-cancellation residual exception "
                    "(discarded; raise is committed to %r): %r",
                    self.name,
                    failed_name,
                    residual,
                )

        raise ParallelBranchesBranchFailed(
            self.name,
            failed_cause,
            parent_state,
            branch_name=failed_name or "<unknown>",
        ) from failed_cause

    async def _collect(
        self,
        parent_state: Any,
        tasks: list[tuple[str, asyncio.Task[Mapping[str, Any]]]],
        contributions: dict[str, Mapping[str, Any]],
        errors: list[dict[str, str]],
    ) -> Mapping[str, Any]:
        """Collect policy per spec §11.5.

        All branches run to completion regardless of individual
        failures. Successful branches' contributions go to the
        buffer; failed branches' errors land in ``errors_field``
        (when configured). The node returns normally.
        """
        del parent_state
        results = await asyncio.gather(
            *(t for _, t in tasks),
            return_exceptions=True,
        )
        for (name, _task), result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(
                    {
                        "branch_name": name,
                        "category": getattr(result, "category", type(result).__name__),
                        "message": str(result),
                        "cause_type": type(
                            result.__cause__ if result.__cause__ is not None else result
                        ).__name__,
                    }
                )
            else:
                contributions[name] = result
        partial = dict(self._merge_contributions(contributions))
        if self.errors_field is not None:
            partial[self.errors_field] = errors
        return partial

    def _merge_contributions(
        self,
        contributions: dict[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Flatten per-branch contributions into a single partial.

        Per §11.4 + §11.8: contributions apply in branch insertion
        order, using each parent field's reducer. The actual reducer
        application happens at ``_merge_partial`` in compiled.py
        when the engine merges this partial into parent state. Here
        we just flatten the per-branch contributions into a dict
        of ``parent_field -> [values in branch insertion order]``
        when multiple branches write the same field, OR
        ``parent_field -> value`` when only one branch writes it.

        Returning multi-value lists lets ``_merge_partial`` route
        each value through the parent's reducer in order.
        """
        # First pass: detect fields written by multiple branches.
        field_contributors: dict[str, list[Any]] = {}
        for branch_name in self.branches.keys():
            if branch_name not in contributions:
                continue
            for parent_field, value in contributions[branch_name].items():
                field_contributors.setdefault(parent_field, []).append(value)

        partial: dict[str, Any] = {}
        for parent_field, values in field_contributors.items():
            if len(values) == 1:
                partial[parent_field] = values[0]
            else:
                # Multi-branch contributions to the same field: the
                # parent reducer applies in branch insertion order.
                # The engine's _merge_partial sees a list and routes
                # each entry through the parent's reducer; we lift
                # the multi-write case via a sentinel marker the
                # engine recognizes.
                partial[parent_field] = _MultiContribution(values=tuple(values))
        return partial


@dataclass(frozen=True)
class _MultiContribution:
    """Sentinel for ``_merge_partial`` indicating that multiple
    branches contributed to the same parent field. The engine
    applies the parent's reducer to each value in sequence,
    preserving branch insertion order per §11.8.
    """

    values: tuple[Any, ...]


__all__ = [
    "BranchSpec",
    "ParallelBranchesNode",
]
