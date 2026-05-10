"""Fan-out node — parallel per-item / per-count subgraph dispatch.

Per spec pipeline-utilities §9: a fan-out node executes a compiled
subgraph (or async callable) once per item in a designated parent
state field, with instances running concurrently up to a configurable
bound, and collects per-instance results back into a parent collection
field.

This is the single place in the engine where multiple subgraph
executions overlap in time within a single invocation; everywhere else
(graph-engine §3) execution is single-threaded.

The module contains:

- :class:`FanOutConfig` — frozen configuration dataclass.
- :class:`FanOutNode` — a node compatible with the engine's Node
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

    See spec §9 for field semantics. Validation happens at builder
    compile time (see ``GraphBuilder.add_fan_out_node``); construction
    here is unchecked beyond the obvious type-level constraints.
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
        del state
        raise RuntimeError(
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

        Per spec §9.1–§9.5: snapshot, resolve count + concurrency,
        build per-instance states, run concurrently with the configured
        error policy, fan-in collected/extra fields, write count_field
        and errors_field if configured.

        ``pre_resolved_count`` / ``pre_resolved_concurrency`` are the
        proposal-0013 v0.10.0 hooks: when the engine has already
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

        # Per-instance task: build the instance_middleware chain, run
        # the subgraph against it, and return the per-instance partial
        # (collect_field + extra_outputs).
        async def run_instance(idx: int, instance_state: ChildT) -> Mapping[str, Any]:
            child_context = context.descend_into_fan_out_instance(
                fan_out_node_name=self.name,
                parent_state=state,
                sub_attached=tuple(cfg.subgraph._attached_observers),  # noqa: SLF001
                fan_out_index=idx,
            )

            async def innermost(s: ChildT) -> Mapping[str, Any]:
                # Run the inner subgraph end-to-end. The inner _invoke
                # uses the context we descended into, so its events
                # carry namespace + parent_states + fan_out_index.
                final_inst_state = await cfg.subgraph._invoke(s, child_context)  # noqa: SLF001
                return _extract_instance_partial(cfg, final_inst_state)

            chain: ChainCall = compose_chain(cfg.instance_middleware, innermost)
            return await chain(instance_state)

        gated_run = _bounded_runner(run_instance, max_concurrency)

        if cfg.error_policy == "fail_fast":
            tasks = [gated_run(idx, st) for idx, st in enumerate(instance_states)]
            try:
                results = await asyncio.gather(*tasks)
            except Exception as exc:
                # Per spec §9.5: the propagated exception is the
                # offending instance's, wrapped in a node_exception
                # with recoverable_state set to the parent's pre-fan-out
                # snapshot. Sibling cancellations are infrastructure
                # (asyncio.gather already cancelled them) and don't
                # produce additional node_exception per cancelled
                # instance.
                raise NodeException(
                    node_name=self.name,
                    cause=exc,
                    recoverable_state=state,
                ) from exc
            return _fan_in_fail_fast(cfg, results)

        # collect — run all instances; capture per-instance exceptions
        # rather than propagate.
        tasks = [gated_run(idx, st) for idx, st in enumerate(instance_states)]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        return _fan_in_collect(cfg, raw, instance_count)


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

    Per spec §9.1:
    - items_field mode: one instance per item, item_field gets the item
    - count mode: ``count`` instances, item_field absent
    - both modes: inputs map parent fields onto subgraph state fields

    ``pre_resolved_count`` (proposal-0013 hook): if the engine has
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
    """Resolve the ``count`` config to an int. Spec §9.1."""
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
    """Resolve the ``concurrency`` config. Spec §9.2."""
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
    partial: dict[str, Any] = {
        cfg.collect_field: getattr(final_state, cfg.collect_field),
    }
    for parent_field, sub_field in cfg.extra_outputs.items():
        partial[parent_field] = getattr(final_state, sub_field)
    return partial


def _fan_in_fail_fast(
    cfg: FanOutConfig,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge per-instance partials into a single fan-out partial under
    the fail_fast policy. All ``results`` succeeded (otherwise gather
    would have raised), so the count is just ``len(results)``. Spec
    §9.3 + §9.4: instance-index order."""
    partial: dict[str, Any] = {
        cfg.target_field: [r[cfg.collect_field] for r in results],
    }
    for parent_field in cfg.extra_outputs:
        partial[parent_field] = [r[parent_field] for r in results]
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
    failed instances' exceptions are recorded there. Spec §9.5."""
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
        cfg.target_field: [s[cfg.collect_field] for s in successes],
    }
    for parent_field in cfg.extra_outputs:
        partial[parent_field] = [s[parent_field] for s in successes]
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
