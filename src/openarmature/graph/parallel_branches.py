# Spec: realizes pipeline-utilities §11 (parallel branches).

"""Parallel branches: concurrent dispatch of M heterogeneous branches.

Counterpart to :mod:`.fan_out`. Fan-out is data-driven (N items,
one subgraph, instantiated N times); parallel branches is
topology-driven (M heterogeneous branches declared statically, run
concurrently within a single parent invocation).

Each branch's :class:`BranchSpec` gives its work as exactly one of a
compiled ``subgraph`` (with its own state schema, middleware, topology,
and ``inputs`` / ``outputs`` projection) or an inline ``call`` (an async
function over the parent state returning a parent-shaped partial update,
no subgraph / projection — the lightweight form). A branch may also carry
its own optional ``middleware`` wrapping the whole branch as a unit, and
an optional ``when`` predicate that skips the branch at dispatch when it
returns false.

Buffer-then-apply semantics: contributions are collected during
dispatch and merged deterministically once at node completion,
using the parent's reducer for each output field. Branch insertion
order determines both dispatch order and merge tie-breaking when
two branches write the same parent field.

Error policies:

- ``fail_fast``: first failure cancels still-running branches;
  the buffered contributions are discarded; the parallel-branches
  node raises ``ParallelBranchesBranchFailed`` with the failing
  branch's exception as ``__cause__``. ``recoverable_state``
  equals the parent state at the moment the node entered.
- ``collect``: all branches run to completion; successful
  branches' contributions merge; failed branches' errors land in
  the optional ``errors_field``.
"""
# Spec pipeline-utilities §11 (parallel branches): §11.4 buffer-then-
# apply, §11.5 error policies, §11.7 branch middleware, §11.8 order.

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from openarmature.observability.correlation import (
    _reset_branch_name,
    _set_branch_name,
)

from .errors import NodeException, ParallelBranchesBranchFailed
from .middleware import ChainCall, Middleware, compose_chain
from .state import State

if TYPE_CHECKING:
    from .compiled import CompiledGraph
    from .observer import _InvocationContext

_log = logging.getLogger(__name__)


# A callable branch's work: an async function over the PARENT state
# returning a parent-shaped partial update directly (no subgraph, no
# state schema, no inputs/outputs projection).
BranchCallable = Callable[[Any], Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class BranchSpec[ChildT: State]:
    """One entry in a :class:`ParallelBranchesNode`'s branch mapping.

    A branch's work is given by exactly one of:

    - ``subgraph`` — a compiled subgraph (the heterogeneous-subgraph
      form): each branch may reference a different compiled subgraph
      with a different state schema, with ``inputs`` / ``outputs``
      following the same shape as subgraph projection mappings.
    - ``call`` — an inline async function over the parent state
      returning a parent-shaped partial update (the lightweight form):
      no subgraph, no state schema, no ``inputs`` / ``outputs``. The
      function reads the parent state directly and returns parent fields.

    An optional ``when`` predicate over the parent state, evaluated once
    at dispatch, skips the branch entirely when it returns false.

    Validation lives on the builder side
    (``GraphBuilder.add_parallel_branches_node``): exactly one of
    ``subgraph`` / ``call`` and no ``inputs`` / ``outputs`` on a
    callable branch (``parallel_branches_invalid_branch_spec``);
    ``mapping_references_undeclared_field`` for subgraph-branch
    inputs/outputs referencing undeclared fields;
    ``parallel_branches_no_branches`` for empty ``branches`` maps;
    ``ValueError`` for empty-string branch names.
    """

    subgraph: CompiledGraph[ChildT] | None = None
    call: BranchCallable | None = None
    when: Callable[[Any], bool] | None = None
    inputs: Mapping[str, str] = field(default_factory=dict[str, str])
    outputs: Mapping[str, str] = field(default_factory=dict[str, str])
    middleware: tuple[Middleware, ...] = ()


@dataclass(frozen=True)
class ParallelBranchesNode[ParentT: State]:
    """A node that dispatches M heterogeneous compiled subgraphs
    concurrently.

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

    def dispatched_branches(self, state: Any) -> list[tuple[str, BranchSpec[Any]]]:
        """Return the branches that dispatch for ``state``, in insertion
        order: every declared branch whose ``when`` predicate is absent or
        returns true (§11.10).

        ``when`` MUST be a pure function of the dispatch-time parent state,
        so this is deterministic and safe to evaluate both here (the
        dispatch set) and at the NODE event's ``branch_count`` (the count of
        branches that dispatch, which excludes ``when``-skipped branches)."""
        return [
            (branch_name, spec)
            for branch_name, spec in self.branches.items()
            if spec.when is None or spec.when(state)
        ]

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
                # A callable branch (§11.1.1) IS the unit of work: run the
                # inline function (wrapped in its branch middleware) and
                # return its parent-shaped partial directly — no subgraph,
                # no inputs/outputs projection. The builder guarantees
                # exactly one of ``call`` / ``subgraph`` per branch.
                if spec.call is not None:
                    return await self._dispatch_callable_branch(
                        branch_name, spec.call, spec.middleware, state, context
                    )
                assert spec.subgraph is not None  # builder: exactly one of call/subgraph
                subgraph = spec.subgraph
                # Per §11.2 projection in: subgraph fields not in
                # ``inputs`` use the subgraph's declared defaults;
                # named subgraph fields are initialized from the
                # corresponding parent field.
                parent_dump = state.model_dump()
                init: dict[str, Any] = {}
                for sub_field, parent_field in spec.inputs.items():
                    init[sub_field] = parent_dump[parent_field]
                initial = subgraph.state_cls(**init)

                child_context = context.descend_into_parallel_branch(
                    parallel_branches_node_name=self.name,
                    parent_state=state,
                    sub_attached=tuple(subgraph._attached_observers),  # noqa: SLF001
                    branch_name=branch_name,
                )

                async def innermost(s: Any) -> Mapping[str, Any]:
                    final_branch_state = await subgraph._invoke(s, child_context)  # noqa: SLF001
                    # Branch middleware wraps the subgraph invocation
                    # (§11.7), so the chain operates in the branch
                    # subgraph's state space. Surface the ``outputs``
                    # source fields keyed by their subgraph names (via
                    # getattr, preserving field-value identity) so a
                    # middleware that short-circuits with a subgraph-space
                    # partial update — FailureIsolation's degraded_update —
                    # composes in the same space. The §11.4 projection to
                    # parent fields runs below, OUTSIDE the chain.
                    return {
                        sub_field: getattr(final_branch_state, sub_field)
                        for sub_field in spec.outputs.values()
                    }

                chain: ChainCall = compose_chain(spec.middleware, innermost)
                branch_partial = await chain(initial)
                # Per §11.4 projection out: map each ``outputs`` sub-field
                # (read from the subgraph-space partial the chain produced
                # — the real subgraph result on success, or a
                # degraded_update on isolation) to its parent field.
                # Unnamed subgraph fields are discarded.
                #
                # Skip a sub-field the partial doesn't carry: a branch
                # contributes only the parent fields it supplies and the
                # §11.4 buffer-then-merge model already merges heterogeneous
                # partial contributions, so an omitted field leaves the
                # parent to its prior / sibling-branch value. On the success
                # path ``innermost`` always supplies every sub-field; the
                # subset case is a degraded_update that doesn't cover a
                # projected field, where a hard miss would defeat the point
                # of failure isolation.
                return {
                    parent_field: branch_partial[sub_field]
                    for parent_field, sub_field in spec.outputs.items()
                    if sub_field in branch_partial
                }
            finally:
                _reset_branch_name(token)

        # Conditional branches (§11.10): the branches that dispatch are
        # those whose ``when`` admits the parent state the node received;
        # a skipped branch runs no work, contributes nothing, and emits no
        # observer events / span.
        dispatching = self.dispatched_branches(state)

        # All branches skipped (§11.10): a valid no-op — the node
        # contributes nothing. Distinct from the compile-time
        # ``parallel_branches_no_branches`` (an empty DECLARED mapping).
        # Return early; ``asyncio.wait([])`` would raise.
        if not dispatching:
            if self.error_policy == "collect" and self.errors_field is not None:
                return {self.errors_field: []}
            return {}

        # Spawn one task per dispatching branch, in insertion order. Per
        # §11.8 the dispatch order is the branches dict's insertion order
        # (over the branches that dispatch); ``started`` events from the
        # inner subgraphs interleave arbitrarily but the branch-level
        # dispatch ordering is deterministic.
        ctx = contextvars.copy_context()
        tasks: list[tuple[str, asyncio.Task[Mapping[str, Any]]]] = []
        for branch_name, spec in dispatching:
            task = asyncio.create_task(
                run_branch(branch_name, spec),
                context=ctx.copy(),
            )
            tasks.append((branch_name, task))

        if self.error_policy == "fail_fast":
            return await self._fail_fast(state, tasks, contributions)
        return await self._collect(state, tasks, contributions, errors)

    async def _dispatch_callable_branch(
        self,
        branch_name: str,
        call: BranchCallable,
        middleware: tuple[Middleware, ...],
        state: Any,
        context: _InvocationContext,
    ) -> Mapping[str, Any]:
        """Run one inline-callable branch and return its contribution.

        The callable reads the parent state and returns a parent-shaped
        partial update directly — no subgraph, no ``inputs`` / ``outputs``
        projection (§11.4). Branch ``middleware`` (e.g. a per-leg
        ``FailureIsolationMiddleware``, §11.7) wraps the callable.

        A callable branch has no inner nodes, so it IS the unit of work:
        it emits one ``started`` / ``completed`` observer pair keyed by its
        ``branch_name`` (graph-engine §6), which the observers render as the
        branch's per-branch dispatch span with NO inner-node span beneath it
        (observability §5.7). To render that way the pair is emitted at the
        parallel-branches NODE's own namespace (not a descended branch
        namespace), tagged with ``branch_name`` (set on the ContextVar in
        ``run_branch``): an event at a pb-node's own namespace carrying a
        ``branch_name`` and no ``parallel_branches_config`` is unambiguously
        a callable branch, since a subgraph branch's inner-node events are
        always one level deeper. A ``when``-skipped branch never reaches
        here, so it emits nothing.
        """
        # Reuse the engine's NodeEvent construction (the static dispatch
        # helpers) so a callable branch's §6 events carry the same lineage
        # as the NODE's own events. Function-scope import keeps the textual
        # cycle off the module graph (compiled.py imports parallel_branches
        # at function scope too, so neither loads the other at import time).
        from .compiled import CompiledGraph

        node_namespace = context.namespace_prefix + (self.name,)
        step = context.take_step()
        CompiledGraph._dispatch_started(  # noqa: SLF001
            context, branch_name, node_namespace, step, state, attempt_index=0
        )
        try:
            # ``call`` is the chain's innermost. Its public type returns the
            # broad ``Awaitable``; ``compose_chain`` wants the coroutine-
            # returning ``ChainCall`` (the NextCall protocol shape). The two
            # are await-compatible at runtime, so cast across the gap.
            chain: ChainCall = compose_chain(middleware, cast("ChainCall", call))
            partial = await chain(state)
        except Exception as exc:
            # The callable (or its middleware) raised unrecovered. Wrap as
            # a NodeException (mirroring the subgraph form, where the inner
            # ``_invoke`` wraps node raises) so the completed event carries
            # a RuntimeGraphError and ``collect`` classifies it the same as
            # a subgraph branch. The node's error policy then wraps this in
            # ParallelBranchesBranchFailed (fail_fast) or records it
            # (collect) — exactly as for a subgraph branch.
            wrapped = NodeException(node_name=branch_name, cause=exc, recoverable_state=state)
            CompiledGraph._dispatch_completed(  # noqa: SLF001
                context, branch_name, node_namespace, step, state, error=wrapped, attempt_index=0
            )
            raise wrapped from exc
        # Success path — including a degraded update from a branch
        # FailureIsolationMiddleware (§11.7), which "succeeds" from the
        # node's view. The contribution is the returned partial directly
        # (parent-shaped, no projection). ``post_state`` shows this
        # branch's local effect on its input; the authoritative reducer
        # merge across siblings is the NODE's completed event.
        post_state = state.model_copy(update=dict(partial))
        CompiledGraph._dispatch_completed(  # noqa: SLF001
            context, branch_name, node_namespace, step, state, post_state=post_state, attempt_index=0
        )
        return partial

    async def _fail_fast(
        self,
        parent_state: Any,
        tasks: list[tuple[str, asyncio.Task[Mapping[str, Any]]]],
        contributions: dict[str, Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Fail-fast policy.

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
        """Collect policy.

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

        Contributions apply in branch insertion order, using each
        parent field's reducer. The actual reducer
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
    preserving branch insertion order.
    """

    values: tuple[Any, ...]


__all__ = [
    "BranchSpec",
    "ParallelBranchesNode",
]
