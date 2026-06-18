# Spec: realizes pipeline-utilities §9 (fan-out node).

"""Fan-out node: parallel per-item / per-count subgraph dispatch.

A fan-out node executes a compiled subgraph (or async callable) once
per item in a designated parent state field, with instances running
concurrently up to a configurable
bound, and collects per-instance results back into a parent collection
field.

This is the single place in the engine where multiple subgraph
executions overlap in time within a single invocation; everywhere else
execution is single-threaded.

The module contains:

- :class:`FanOutConfig`: frozen configuration dataclass.
- :class:`FanOutNode`: a node compatible with the engine's Node
  Protocol; ``run`` resolves count + concurrency, builds per-instance
  states, runs them concurrently with the configured error policy, and
  fan-ins results back as a partial update.

Performance note (mirroring the compose_chain note in middleware/_core.py):
under heavy fan-out workloads the inner-instance middleware chains are
built once per instance, so for N instances × M instance_middlewares
that's N×M closure constructions per fan-out dispatch. Fine for
typical workloads (50 instances, 3 middlewares = 150 closures); worth
measuring once large-N workloads exist.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from .errors import (
    FanOutEmpty,
    FanOutInvalidConcurrency,
    FanOutInvalidCount,
    NodeException,
)
from .middleware import ChainCall, Middleware, compose_chain
from .observer import _FanOutExecutionState, _FanOutInstanceState
from .state import State

if TYPE_CHECKING:
    from .compiled import CompiledGraph
    from .observer import _InvocationContext


# Type aliases for the count / concurrency callables: per spec §9 each
# accepts the parent's pre-fan-out state snapshot and returns an int.
CountResolver = Callable[[Any], int]
ConcurrencyResolver = Callable[[Any], int | None]


@dataclass(frozen=True)
class FanOutConfig:
    """Frozen configuration for a :class:`FanOutNode`.

    Validation happens at builder compile time (see
    ``GraphBuilder.add_fan_out_node``); construction here is
    unchecked beyond the obvious type-level constraints.
    """

    subgraph: CompiledGraph[Any]
    collect_field: str
    target_field: str
    items_field: str | None = None
    item_field: str | None = None
    count: int | CountResolver | None = None
    concurrency: int | ConcurrencyResolver | None = 10
    error_policy: Literal["fail_fast", "collect"] = "fail_fast"
    on_empty: Literal["raise", "noop"] = "raise"
    count_field: str | None = None
    inputs: Mapping[str, str] = field(default_factory=dict[str, str])
    extra_outputs: Mapping[str, str] = field(default_factory=dict[str, str])
    instance_middleware: tuple[Middleware, ...] = ()
    errors_field: str | None = None
    # The identity of the compiled inner subgraph (the key under
    # which the subgraph is declared in a ``subgraphs:`` registry).
    # Threaded onto every per-instance event so observers can emit
    # ``observation.metadata.subgraph_name`` on each per-instance
    # dispatch observation (Langfuse) /
    # ``openarmature.subgraph.name`` on the corresponding span
    # (OTel). Optional and BC-preserving — direct callers that don't
    # supply it get the empty-string fallback per observability §5.3.
    subgraph_identity: str | None = None


@dataclass(frozen=True)
class FanOutNode[ParentT: State, ChildT: State]:
    """A node that fans out into N concurrent subgraph instances.

    The Node Protocol contract requires ``name``, ``middleware``, and
    ``run``; ``run`` here is unusual because it needs the engine
    invocation context to descend properly. The engine recognizes this
    type in ``_invoke`` and calls ``run_with_context`` (see
    ``compiled.CompiledGraph._step_fan_out_node``) rather than the
    plain ``run(state)`` shape. ``run`` exists for Protocol conformance
    only and raises if anyone calls it without context.
    """

    name: str
    config: FanOutConfig
    middleware: tuple[Middleware, ...] = ()

    async def run(self, state: ParentT) -> Mapping[str, Any]:
        """Not implemented at this level. The fan-out node requires the
        engine's invocation context to resolve count and concurrency
        and dispatch instances; the engine calls
        :meth:`run_with_context` instead. This method exists only to
        satisfy the :class:`Node` Protocol and always raises
        :class:`NotImplementedError`."""
        del state
        raise NotImplementedError(
            "FanOutNode is dispatched by the graph engine; if you're seeing "
            "this, you've likely instantiated it outside an engine context "
            "(e.g., calling node.run(state) directly instead of compiled.invoke)."
        )

    async def run_with_context(
        self,
        state: ParentT,
        context: _InvocationContext,
        *,
        pre_resolved_count: int | None = None,
        pre_resolved_concurrency: tuple[int | None] | None = None,
    ) -> Mapping[str, Any]:
        """Execute the fan-out and return the merged partial update.

        Snapshot, resolve count + concurrency, build per-instance
        states, run concurrently with the configured error policy,
        fan-in collected/extra fields, write count_field and
        errors_field if configured.

        Per the per-instance resume contract: this method registers a
        per-fan-out tracking entry on the shared
        ``context.fan_out_progress_state`` dict before dispatching,
        flips each instance's state through
        ``not_started -> in_flight -> completed`` as the instance
        progresses, and fires an explicit "instance completed" save
        after the per-instance contribution has been recorded into
        the accumulator. The atomicity contract is
        observed: the per-instance state mutation precedes the save,
        so a crash after mutation but before save leaves the saved
        record showing ``in_flight`` (resume re-runs the instance).

        ``pre_resolved_count`` / ``pre_resolved_concurrency`` are
        hooks: when the engine has already
        resolved the config eagerly to populate
        ``NodeEvent.fan_out_config`` for the fan-out node's events,
        it passes the resolved values in so callable resolvers
        aren't invoked twice. ``pre_resolved_concurrency`` is wrapped
        in a 1-tuple to disambiguate "caller passed ``None``
        (unbounded)" from "caller didn't pass anything."
        """
        cfg = self.config
        instance_states = _build_instance_states(self.name, cfg, state, pre_resolved_count=pre_resolved_count)
        instance_count = len(instance_states)

        if instance_count == 0:
            if cfg.on_empty == "raise":
                raise FanOutEmpty(node_name=self.name, recoverable_state=state)
            # noop — write count_field=0, leave target_field unchanged
            # by contributing an empty list (which the parent's append
            # reducer absorbs as a no-op).
            return _empty_fan_out_partial(cfg)

        if pre_resolved_concurrency is not None:
            max_concurrency = pre_resolved_concurrency[0]
        else:
            max_concurrency = _resolve_concurrency(self.name, cfg, state)

        # Register / reuse the per-fan-out tracking entry on the
        # shared dict. Resume threads a pre-restored entry through
        # ``context.fan_out_progress_state``; first-run constructs a
        # fresh one with all instances ``not_started``.
        key = (context.namespace_prefix, self.name)
        exec_state = context.fan_out_progress_state.get(key)
        if exec_state is None:
            exec_state = _FanOutExecutionState(
                fan_out_node_name=self.name,
                namespace=context.namespace_prefix,
                instance_count=instance_count,
                instances=[_FanOutInstanceState() for _ in range(instance_count)],
            )
            context.fan_out_progress_state[key] = exec_state
        elif exec_state.instance_count != instance_count:
            # Per spec §10.11 + §10.10 (proposal 0029): a saved
            # ``instance_count`` that differs from the resumed run's
            # resolved count MUST raise ``checkpoint_record_invalid``
            # before any fan-out instance work runs on this path. The
            # pre-0029 pad/truncate behavior would silently drop
            # ``completed`` contributions on shrink (breaking §10.11.1's
            # exactly-once guarantee) and dispatch unsaved work on grow
            # (violating §10.5's idempotency framing). The strict raise
            # surfaces the divergence to the user; they cohere inputs
            # or restart cleanly.
            # Local import to avoid an engine ↔ checkpoint package cycle
            # at module load (mirrors the existing
            # ``CheckpointSaveFailed`` imports below in this file).
            from openarmature.checkpoint.errors import CheckpointRecordInvalid  # noqa: PLC0415

            # ``context.resume_invocation`` identifies the SAVED record
            # being validated (per spec §10.4 step 3); ``context.invocation_id``
            # is freshly minted for the resumed run (step 4). The fresh-
            # run fallback is defensive only — the count-drift path can
            # only fire on resume since fan_out_progress_state is empty
            # on a fresh first run.
            raise CheckpointRecordInvalid(
                context.resume_invocation or context.invocation_id,
                f"fan_out {self.name!r} at namespace {context.namespace_prefix!r}: "
                f"saved instance_count={exec_state.instance_count} does not match "
                f"resolved instance_count={instance_count} on resume",
            )

        # Shared cancel signal for the fail_fast path. Defined here (not
        # inside the fail_fast branch below) so ``run_instance`` can
        # check it AFTER semaphore acquisition but BEFORE mutating
        # tracked state — closes a race where a semaphore-blocked
        # sibling would flip its tracked state to ``in_flight`` after a
        # sibling failed and set the signal. In collect mode the signal
        # is never set, so the check inside ``run_instance`` is a
        # no-op there.
        cancel_signal = asyncio.Event()

        # Per-instance task: build the instance_middleware chain, run
        # the subgraph against it, and return the per-instance partial
        # (collect_field + extra_outputs).
        #
        # Resume gating: an instance whose tracked state is
        # ``completed`` is skipped (its result rolls forward from the
        # accumulator entry). Instances tracked as ``in_flight`` or
        # ``not_started`` dispatch normally with fresh per-instance
        # state — per §10.7 the inner subgraph re-enters at its
        # declared entry, not at any of the ``completed_inner_positions``
        # captured in the prior run.
        async def run_instance(idx: int, instance_state: ChildT) -> Mapping[str, Any]:
            tracked = exec_state.instances[idx]
            if tracked.state == "completed":
                if tracked.result_is_error:
                    # Per §10.11.2 collect-mode resume: an error
                    # contribution rolls forward through the
                    # ``errors_field`` bucket, not ``target_field``.
                    # Raise a categorized exception so the outer
                    # gather captures it and ``_fan_in_collect``
                    # routes it through error_records (with the same
                    # ``category`` the original failure carried).
                    raise _RolledForwardError(category=_extract_error_category(tracked.result))
                # Roll the success contribution forward verbatim.
                return _rolled_forward_partial(cfg, tracked)

            # Cancel-signal check AFTER the resume rollforward branch
            # but BEFORE the first tracked-state mutation. Covers the
            # race where a sibling failed (setting the signal) while
            # this task was blocked on the bounded-concurrency
            # semaphore inside ``gated_run``; without this check, the
            # task would acquire the semaphore and flip ``tracked.state``
            # to ``in_flight``, contradicting §10.11.2's
            # "not-yet-dispatched siblings end up not_started" contract.
            if cancel_signal.is_set():
                raise asyncio.CancelledError()

            # Flip to in_flight BEFORE dispatching so a sibling-
            # triggered save during this instance's execution observes
            # the correct state. Reset completed_inner_positions to
            # ensure resume re-execution doesn't accumulate against
            # the prior run's prefix.
            tracked.state = "in_flight"
            tracked.completed_inner_positions.clear()
            tracked.result = None
            tracked.extra_outputs = {}

            child_context = context.descend_into_fan_out_instance(
                fan_out_node_name=self.name,
                parent_state=state,
                sub_attached=tuple(cfg.subgraph._attached_observers),  # noqa: SLF001
                fan_out_index=idx,
                subgraph_identity=cfg.subgraph_identity,
            )

            async def innermost(s: ChildT) -> Mapping[str, Any]:
                # Run the inner subgraph end-to-end. The inner _invoke
                # uses the context we descended into, so its events
                # carry namespace + parent_states + fan_out_index.
                final_inst_state = await cfg.subgraph._invoke(s, child_context)  # noqa: SLF001
                return _extract_instance_partial(cfg, final_inst_state)

            chain: ChainCall = compose_chain(cfg.instance_middleware, innermost)
            try:
                partial = await chain(instance_state)
            except Exception as exc:
                if cfg.error_policy == "collect":
                    # Per §10.11.2 collect mode: the failure becomes a
                    # ``completed`` contribution with the error record
                    # as ``result``. Mutate state BEFORE saving so the
                    # save durably reflects the completion (atomicity
                    # contract per §10.11). The re-raise hands the
                    # exception back to the outer gather so the
                    # ``_fan_in_collect`` path builds the parent
                    # ``errors_field`` from raw_results.
                    error_record: dict[str, str] = {
                        "fan_out_index": str(idx),
                        "category": getattr(exc, "category", type(exc).__name__),
                    }
                    tracked.result = error_record
                    tracked.result_is_error = True
                    tracked.extra_outputs = {}
                    tracked.state = "completed"
                    await _save_instance_completed(state, context)
                    raise
                # Per §10.11 in_flight observability under fail_fast:
                # if no sibling completion fired a save during this
                # instance's execution (the serial-execution +
                # first-node-fails case), the saved record would not
                # otherwise reflect this instance's in_flight
                # transition. Fire an explicit "instance failed" save
                # so the per-instance in_flight observation reaches
                # the saved record. Tracked state stays ``in_flight``
                # (no accumulator write happens on failure under
                # fail_fast) per §10.11.2. Re-raise after the save so
                # the fail_fast cancellation path stays intact.
                await _save_instance_in_flight(state, context)
                raise

            # Atomicity contract (§10.11): produce contribution -> record
            # into accumulator -> save. The accumulator update below
            # happens BEFORE the explicit "instance completed" save so a
            # crash between accumulator write and save leaves the saved
            # record showing ``in_flight`` and resume re-runs the
            # instance. The ``append`` reducer's no-double-merge guarantee
            # (§10.11.1) depends on this ordering.
            tracked.result = partial.get(cfg.collect_field)
            tracked.result_is_error = False
            # ``partial`` is subgraph-space (success or degrade); read each
            # extra_outputs value by its subgraph field name and store the
            # accumulator entry under the parent field name.
            tracked.extra_outputs = {
                parent_field: partial[sub_field]
                for parent_field, sub_field in cfg.extra_outputs.items()
                if sub_field in partial
            }
            tracked.state = "completed"

            # Fire an explicit "instance completed" save so the saved
            # record durably reflects the completed state. Without this
            # save, only the terminal inner node's intrinsic save fires
            # (which executed BEFORE the accumulator mutation above and
            # therefore showed the instance as ``in_flight``). The
            # explicit save closes the atomicity gap. Routed through
            # the fan-out-internal batching seam per §10.11.4.
            await _save_instance_completed(state, context)

            return partial

        gated_run = _bounded_runner(run_instance, max_concurrency)

        if cfg.error_policy == "fail_fast":
            # ``cancel_signal`` is defined above (before ``run_instance``)
            # so the in-instance check can read it. The wrapper below
            # adds a fast-path check before semaphore acquisition for
            # tasks that haven't entered ``run_instance`` yet; the
            # in-instance check covers the race after semaphore
            # acquisition. The explicit signal closes the
            # bounded-concurrency-semaphore cancellation gap that
            # ``asyncio.gather`` / ``asyncio.wait`` don't enforce
            # strongly enough on their own.

            async def signaled_run(idx: int, st: Any) -> Mapping[str, Any]:
                # Check before any work — if a sibling already failed,
                # exit immediately so this instance's tracked state
                # stays at its default not_started.
                if cancel_signal.is_set():
                    raise asyncio.CancelledError()
                try:
                    return await gated_run(idx, st)
                except Exception:
                    # Set the signal so siblings about to run see it
                    # before they enter run_instance and mutate
                    # tracked state. The first task to raise wins.
                    cancel_signal.set()
                    raise

            tasks: list[tuple[int, asyncio.Task[Mapping[str, Any]]]] = [
                (idx, asyncio.create_task(signaled_run(idx, st))) for idx, st in enumerate(instance_states)
            ]
            try:
                await asyncio.wait(
                    [t for _, t in tasks],
                    return_when=asyncio.FIRST_EXCEPTION,
                )
            except BaseException:
                for _, t in tasks:
                    t.cancel()
                await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
                raise

            # Iterate all completed-not-cancelled tasks to retrieve
            # each exception via ``t.exception()`` (otherwise asyncio
            # warns "Task exception was never retrieved" on GC for any
            # task that failed before fail_fast cancelled its
            # siblings). ``failed_cause`` still captures only the
            # FIRST real exception — NodeException's ``cause`` should
            # be the originating instance's error, not a later
            # sibling's. CancelledErrors are siblings we cancelled, so
            # ignore them.
            failed_cause: BaseException | None = None
            for _, t in tasks:
                if t.done() and not t.cancelled():
                    exc = t.exception()
                    if exc is not None and not isinstance(exc, asyncio.CancelledError):
                        if failed_cause is None:
                            failed_cause = exc

            # Cancel any still-pending tasks; drain to absorb
            # CancelledError so it doesn't propagate as unhandled.
            for _, t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*(t for _, t in tasks if not t.done()), return_exceptions=True)

            if failed_cause is None:
                # All tasks finished without raising. Collect results
                # in instance-index order and fan-in.
                results = [t.result() for _, t in tasks]
                return _fan_in_fail_fast(cfg, results)

            # Per spec §9.5: the propagated exception is the offending
            # instance's, wrapped in a node_exception with
            # recoverable_state set to the parent's pre-fan-out
            # snapshot. Per §10.11.2 the failed instance's tracked
            # state is ``in_flight`` (no accumulator entry was
            # recorded because the contribution -> mutation -> save
            # sequence raised before the mutation; resume re-runs).
            raise NodeException(
                node_name=self.name,
                cause=failed_cause,
                recoverable_state=state,
            ) from failed_cause

        # collect — run all instances; capture per-instance exceptions
        # rather than propagate. Per §10.11.2 a collect-mode failure
        # produces a ``completed`` instance whose ``result`` is the
        # error record contributed to ``errors_field``. Per-instance
        # promotion happens inside ``run_instance`` so the
        # ``completed`` save fires before sibling instances dispatch
        # (the §10.11 atomicity contract still holds and the abort_-
        # after_instance harness directive sees the right state).
        collect_tasks = [gated_run(idx, st) for idx, st in enumerate(instance_states)]
        raw_results = await asyncio.gather(*collect_tasks, return_exceptions=True)
        return _fan_in_collect(cfg, raw_results, instance_count)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_instance_states(
    node_name: str,
    cfg: FanOutConfig,
    parent_state: Any,
    *,
    pre_resolved_count: int | None = None,
) -> list[Any]:
    """Project parent state to per-instance subgraph states.

    By mode:
    - items_field mode: one instance per item, item_field gets the item
    - count mode: ``count`` instances, item_field absent
    - both modes: inputs map parent fields onto subgraph state fields

    ``pre_resolved_count``: if the engine has
    already resolved ``cfg.count`` to populate
    ``NodeEvent.fan_out_config.item_count``, the resolved value is
    passed in here so the callable resolver isn't invoked twice.
    Only consulted in ``count`` mode; ``items_field`` mode reads
    from the state field unchanged.
    """
    sub_state_cls = cfg.subgraph.state_cls
    parent_dump = parent_state.model_dump()

    if cfg.items_field is not None:
        items_raw = parent_dump.get(cfg.items_field, [])
        if not isinstance(items_raw, list):
            # Compile-time validation already caught the non-list case;
            # this is a defensive runtime check only.
            raise NodeException(
                node_name=node_name,
                cause=TypeError(f"items_field {cfg.items_field!r} is not a list at runtime"),
                recoverable_state=parent_state,
            )
        items: list[Any] = list(cast("list[Any]", items_raw))
        sub_field_names = set(sub_state_cls.model_fields.keys())
        instances: list[Any] = []
        for item in items:
            init: dict[str, Any] = {}
            # Per spec §9 + fixture 023: item_field may name a field the
            # subgraph doesn't declare (placeholder pattern). Skip the
            # assignment if so — pydantic strict would otherwise reject
            # the unknown field, and the fixture's contract is that the
            # subgraph isn't reading the item anyway.
            if cfg.item_field is not None and cfg.item_field in sub_field_names:
                init[cfg.item_field] = item
            for sub_field, parent_field in cfg.inputs.items():
                init[sub_field] = parent_dump[parent_field]
            instances.append(sub_state_cls(**init))
        return instances

    # count mode
    if pre_resolved_count is not None:
        resolved_count = pre_resolved_count
    else:
        resolved_count = _resolve_count(node_name, cfg, parent_state)
    instances = []
    for _idx in range(resolved_count):
        init = {}
        for sub_field, parent_field in cfg.inputs.items():
            init[sub_field] = parent_dump[parent_field]
        instances.append(sub_state_cls(**init))
    return instances


def _resolve_count(node_name: str, cfg: FanOutConfig, parent_state: Any) -> int:
    """Resolve the ``count`` config to an int."""
    raw = cfg.count
    if callable(raw):
        resolved = raw(parent_state)
    elif isinstance(raw, int):
        resolved = raw
    else:
        # Builder validation should prevent this; defensive.
        raise NodeException(
            node_name=node_name,
            cause=TypeError(f"count must be int or callable, got {type(raw).__name__}"),
            recoverable_state=parent_state,
        )
    if resolved < 0:
        raise FanOutInvalidCount(node_name=node_name, returned=resolved, recoverable_state=parent_state)
    return resolved


def _resolve_concurrency(node_name: str, cfg: FanOutConfig, parent_state: Any) -> int | None:
    """Resolve the ``concurrency`` config."""
    raw = cfg.concurrency
    if callable(raw):
        resolved = raw(parent_state)
    else:
        resolved = raw
    if resolved is None:
        return None
    if resolved <= 0:
        raise FanOutInvalidConcurrency(node_name=node_name, returned=resolved, recoverable_state=parent_state)
    return resolved


def _bounded_runner(
    run_one: Callable[[int, Any], Any],
    concurrency: int | None,
) -> Callable[[int, Any], Any]:
    """Wrap ``run_one(idx, state)`` so at most ``concurrency`` instances
    run concurrently. ``None`` = unbounded."""
    if concurrency is None:
        return run_one
    sem = asyncio.Semaphore(concurrency)

    async def gated(idx: int, st: Any) -> Any:
        async with sem:
            return await run_one(idx, st)

    return gated


def _extract_instance_partial(cfg: FanOutConfig, final_state: Any) -> Mapping[str, Any]:
    """Extract collect_field + extra_outputs values from a finished
    instance's state. Returned as the per-instance partial that flows
    up the instance_middleware chain."""
    # Per §9.3 the per-instance partial is subgraph-space: collect_field
    # and every extra_outputs SOURCE field are keyed by their subgraph
    # field name (the same shape a degrade's degraded_update carries), so
    # the success and degrade paths compose through one fan-in. The §9.4
    # projection to parent field names happens in the fan-in.
    partial: dict[str, Any] = {
        cfg.collect_field: getattr(final_state, cfg.collect_field),
    }
    for sub_field in cfg.extra_outputs.values():
        partial[sub_field] = getattr(final_state, sub_field)
    return partial


class _RolledForwardError(Exception):
    """Exception raised by ``run_instance`` to signal that a
    collect-mode resume is rolling forward a recorded error
    contribution. Carries the original failure's ``category`` so
    the resumed fan-in path can record an error entry with the
    same category that the prior run produced. Internal — never
    propagates out of the fan-out's run_with_context.
    """

    def __init__(self, *, category: str) -> None:
        super().__init__(f"rolled-forward error ({category})")
        self.category = category


def _extract_error_category(error_record: Any) -> str:
    """Pull the ``category`` field from an error_record dict the engine
    stored as a tracked instance's ``result``. Falls back to
    ``node_exception`` when the field isn't present (defensive — the
    engine always sets ``category`` per ``_fan_in_collect``)."""
    if isinstance(error_record, dict):
        result_dict = cast("dict[str, Any]", error_record)
        category = result_dict.get("category", "node_exception")
        if isinstance(category, str):
            return category
    return "node_exception"


def _rolled_forward_partial(cfg: FanOutConfig, tracked: _FanOutInstanceState) -> Mapping[str, Any]:
    """Reconstruct the per-instance partial for a ``completed`` instance
    being skipped on resume. The accumulator entry rolls forward
    verbatim — same shape as :func:`_extract_instance_partial` would
    have produced on the original run, sourced from the per-instance
    tracked state instead of a freshly-computed inner state."""
    # Reconstruct the subgraph-space partial: collect_field plus each
    # extra_outputs SOURCE field keyed by its subgraph name, sourced from
    # the parent-keyed accumulator entry.
    partial: dict[str, Any] = {cfg.collect_field: tracked.result}
    for parent_field, sub_field in cfg.extra_outputs.items():
        if parent_field in tracked.extra_outputs:
            partial[sub_field] = tracked.extra_outputs[parent_field]
    return partial


async def _save_instance_in_flight(
    parent_state: Any,
    context: _InvocationContext,
) -> None:
    """Fire an explicit save when an instance fails before any sibling
    triggered a save during its execution. Without this save, the
    instance's in_flight transition would not be observable on the
    saved record under serial execution: no sibling completion fires
    during a serial instance's run, and the instance's own inner-node
    save only fires on successful merge (failure path skips it).

    Routes through the checkpointer's ``save_fan_out_in_flight_failure``
    seam (when present). Batching backends typically
    buffer this save WITHOUT triggering a flush — the "crash" the
    failure represents would lose the buffer, including this save,
    in a real-world scenario. Non-batching backends route it through
    the synchronous ``save`` path so the in_flight observability of
    fixture 048 holds.
    """
    from openarmature.checkpoint.errors import CheckpointSaveFailed  # noqa: PLC0415
    from openarmature.checkpoint.protocol import CheckpointRecord  # noqa: PLC0415

    from .compiled import (  # noqa: PLC0415
        _project_fan_out_progress,
        _save_fan_out_in_flight_failure,
    )

    checkpointer = context.checkpointer
    if checkpointer is None:
        return
    fan_out_progress = _project_fan_out_progress(context.fan_out_progress_state)
    # Per spec §10.2 (proposal 0028): ``schema_version`` sourced from the
    # declared graph state class via ``context.state_cls`` so every save
    # site in the invocation reports the same value. See the matching
    # comment in ``_save_instance_completed`` below for the full
    # rationale.
    schema_version = cast("str", getattr(context.state_cls, "schema_version", ""))
    record = CheckpointRecord(
        invocation_id=context.invocation_id,
        correlation_id=context.correlation_id,
        state=parent_state,
        completed_positions=tuple(context.completed_positions),
        parent_states=context.parent_states_prefix,
        last_saved_at=time.time(),
        schema_version=schema_version,
        fan_out_progress=fan_out_progress,
    )
    try:
        await _save_fan_out_in_flight_failure(checkpointer, context.invocation_id, record)
    except Exception as exc:
        raise CheckpointSaveFailed(context.invocation_id, exc) from exc


async def _save_instance_completed(
    parent_state: Any,
    context: _InvocationContext,
) -> None:
    """Fire the explicit "instance completed" save closing the
    atomicity gap. The per-instance state has already been flipped to
    ``completed`` with ``result`` populated; this save durably records
    that transition so resume can skip the instance.

    Routed through the fan-out-internal batching seam —
    backends opting into batching may buffer the save; non-batching
    backends call ``save`` directly. On crash with buffered-but-
    unflushed saves, the instance reverts to ``in_flight`` /
    ``not_started`` on resume and re-runs (contributing for the first
    time, no double-merge).
    """
    # Lazy imports: ``compiled`` and ``checkpoint.protocol`` would
    # create textual cycles at module-load. Function-scope keeps the
    # import cheap (cached after first call) and the cycle off the
    # static analyzer's graph.
    from openarmature.checkpoint.errors import CheckpointSaveFailed  # noqa: PLC0415
    from openarmature.checkpoint.protocol import CheckpointRecord  # noqa: PLC0415

    from .compiled import (  # noqa: PLC0415
        _project_fan_out_progress,
        _save_fan_out_internal,
    )

    checkpointer = context.checkpointer
    if checkpointer is None:
        return
    # The "instance completed" save records the post-merge outer state
    # via ``parent_state`` (the snapshot of outer state at fan-out
    # dispatch time). ``parent_states`` carries any enclosing subgraph
    # chain. This save shape mirrors a top-level "outer node completed"
    # save: ``state`` = outer; ``parent_states`` = enclosing chain
    # (empty for outermost fan-outs). The inner-node saves fired during
    # the instance's execution have a different shape (state = inner,
    # parent_states includes outer) — both shapes are valid checkpoint
    # records and the resume path handles either based on
    # ``parent_states`` length.
    fan_out_progress = _project_fan_out_progress(context.fan_out_progress_state)
    # Per spec §10.2 (proposal 0028): ``schema_version`` is sourced from
    # the declared graph state class on the outermost ``CompiledGraph``,
    # not from ``type(state)`` at save time. Threaded as
    # ``context.state_cls`` from the outermost ``invoke``; the rule
    # matters when the user passes a State subclass that shadows
    # ``schema_version`` (instance class would yield a different value;
    # the declared class is the only value §10.12 migration lookups
    # know about). Mirrors ``_maybe_save_checkpoint``'s
    # ``self.state_cls.schema_version`` read in ``compiled.py`` so every
    # save site within an invocation reports the same value.
    schema_version = cast("str", getattr(context.state_cls, "schema_version", ""))
    record = CheckpointRecord(
        invocation_id=context.invocation_id,
        correlation_id=context.correlation_id,
        state=parent_state,
        completed_positions=tuple(context.completed_positions),
        parent_states=context.parent_states_prefix,
        last_saved_at=time.time(),
        schema_version=schema_version,
        fan_out_progress=fan_out_progress,
    )
    try:
        await _save_fan_out_internal(checkpointer, context.invocation_id, record)
    except Exception as exc:
        raise CheckpointSaveFailed(context.invocation_id, exc) from exc
    # Per §10.8: the explicit "instance completed" save is a save like
    # any other and SHOULD emit a ``checkpoint_saved`` observer event.
    # However the engine's primary save call site
    # (``_maybe_save_checkpoint``) already dispatches the event for
    # every save it owns, and the explicit save here is conceptually
    # part of the same save-point: the inner node's intrinsic save
    # already fired ``checkpoint_saved`` for this fan-out instance's
    # progress. Adding a second event would double-count for backends
    # that surface them as spans. Suppress to keep the event stream
    # node-aligned.


def _fan_in_fail_fast(
    cfg: FanOutConfig,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge per-instance partials into a single fan-out partial under
    the fail_fast policy. All ``results`` succeeded (otherwise gather
    would have raised), so the count is just ``len(results)``;
    instance-index order."""
    # §9.4 projection: read each instance's subgraph-space partial by
    # subgraph field name and collect into the parent field. ``.get`` keeps
    # an omitted collect_field (a callable degrade that doesn't set it, §9.3)
    # a graceful null slot rather than a raise.
    partial: dict[str, Any] = {
        cfg.target_field: [r.get(cfg.collect_field) for r in results],
    }
    for parent_field, sub_field in cfg.extra_outputs.items():
        partial[parent_field] = [r.get(sub_field) for r in results]
    if cfg.count_field is not None:
        partial[cfg.count_field] = len(results)
    return partial


def _fan_in_collect(
    cfg: FanOutConfig,
    raw_results: Sequence[Any],
    instance_count: int,
) -> dict[str, Any]:
    """Merge per-instance results under the collect policy. Failures
    contribute nothing to target_field; if errors_field is configured,
    failed instances' exceptions are recorded there."""
    successes: list[Mapping[str, Any]] = []
    error_records: list[dict[str, str]] = []
    for idx, r in enumerate(raw_results):
        if isinstance(r, Exception):
            # Record per-instance failures as a small struct keyed by
            # fan_out_index + category. Per spec §9.5: "Per-instance
            # errors are recorded in a parent state field named by an
            # additional config field `errors_field`." The exact shape
            # is implementation-defined; we follow fixture 019's choice
            # of stringified fields so errors_field can be declared as
            # ``list[dict[str, str]]`` on the parent state schema.
            # Consumers that want int-typed indices should declare
            # errors_field with a richer type and provide a custom
            # error-recording layer downstream.
            error_records.append(
                {
                    "fan_out_index": str(idx),
                    "category": getattr(r, "category", type(r).__name__),
                }
            )
        else:
            successes.append(r)

    partial: dict[str, Any] = {
        cfg.target_field: [s.get(cfg.collect_field) for s in successes],
    }
    for parent_field, sub_field in cfg.extra_outputs.items():
        partial[parent_field] = [s.get(sub_field) for s in successes]
    if cfg.errors_field is not None:
        partial[cfg.errors_field] = error_records
    if cfg.count_field is not None:
        partial[cfg.count_field] = instance_count
    return partial


def _empty_fan_out_partial(cfg: FanOutConfig) -> dict[str, Any]:
    """Build the partial update for a noop empty fan-out (zero instances)."""
    partial: dict[str, Any] = {cfg.target_field: []}
    for parent_field in cfg.extra_outputs:
        partial[parent_field] = []
    if cfg.errors_field is not None:
        partial[cfg.errors_field] = []
    if cfg.count_field is not None:
        partial[cfg.count_field] = 0
    return partial


__all__ = [
    "ConcurrencyResolver",
    "CountResolver",
    "FanOutConfig",
    "FanOutNode",
]
