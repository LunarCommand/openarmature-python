"""Compiled graph + execute loop.

Execution begins at the entry node; each step runs a node, merges
its partial update via per-field reducers, then evaluates the
outgoing edge against the post-update state to choose the next node
(or END to halt).

Node, edge, reducer, and routing errors carry recoverable state;
state validation errors do not.

Each node attempt produces a started/completed event PAIR. The
engine dispatches the started event before invoking the wrapped node
function and the completed event after the reducer merge succeeds
(with ``post_state`` populated) or after the node, reducer, or state
validation fails (with ``error`` populated). Routing errors do NOT
produce their own event pair; they land on the preceding node's
``completed`` event with ``error`` populated.

``CompiledGraph[StateT]`` and ``_merge_partial[StateT]`` carry the
concrete state subclass through to ``invoke()``'s return type, so
consumers don't need ``cast(MyState, ...)`` at the call site.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    # ``FanOutNode`` lives in ``.fan_out`` which has a TYPE_CHECKING
    # back-reference to ``CompiledGraph`` here. Importing at module
    # top would create a textual cycle CodeQL's
    # ``py/cyclic-import`` rule flags (no runtime issue —
    # ``fan_out``'s ``compiled`` import is itself TYPE_CHECKING-gated
    # — but the static analyzer doesn't see that). Type annotations
    # use the string form via ``from __future__ import annotations``;
    # runtime use (the ``isinstance`` check in ``_invoke``) imports
    # lazily inside the function.
    from .fan_out import FanOutNode

from pydantic import ValidationError

from openarmature.checkpoint.errors import (
    CheckpointError,
    CheckpointNotFound,
    CheckpointRecordInvalid,
    CheckpointSaveFailed,
    CheckpointStateMigrationFailed,
    CheckpointStateMigrationMissing,
)
from openarmature.checkpoint.migration import MigrationRegistry, StateMigration
from openarmature.checkpoint.protocol import (
    Checkpointer,
    CheckpointRecord,
    FanOutInstanceProgress,
    FanOutProgress,
    NodePosition,
)
from openarmature.observability.correlation import (
    _reset_active_dispatch,
    _reset_active_observers,
    _reset_branch_name_chain,
    _reset_correlation_id,
    _reset_fan_out_index,
    _reset_fan_out_index_chain,
    _reset_invocation_id,
    _reset_namespace_prefix,
    _set_active_dispatch,
    _set_active_observer_span,
    _set_active_observers,
    _set_branch_name_chain,
    _set_correlation_id,
    _set_fan_out_index,
    _set_fan_out_index_chain,
    _set_invocation_id,
    _set_namespace_prefix,
    current_active_observer_span,
    current_attempt_index,
    validate_invocation_id,
)
from openarmature.observability.metadata import (
    _reset_invocation_metadata,
    _set_invocation_metadata,
    current_invocation_metadata,
    validate_invocation_metadata,
)

from .edges import END, ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    EdgeException,
    NodeException,
    ReducerError,
    RoutingError,
    RuntimeGraphError,
    StateValidationError,
)
from .events import (
    FanOutEventConfig,
    InvocationCompletedEvent,
    InvocationStartedEvent,
    NodeEvent,
    ParallelBranchesEventConfig,
)
from .middleware import ChainCall, Middleware, compose_chain
from .nodes import Node
from .observer import (
    _DRAIN_SENTINEL,
    DrainSummary,
    Observer,
    RemoveHandle,
    SubscribedObserver,
    _coerce_subscribed,
    _dispatch,
    _FanOutExecutionState,
    _FanOutInstanceState,
    _InvocationContext,
    _QueuedItem,
    deliver_loop,
)
from .reducers import Reducer
from .state import State
from .subgraph import SubgraphNode

# Try-import OpenTelemetry attach primitives so the engine can splice an
# observer-published span into the OTel context for the duration of a
# node body. The engine treats the span value opaquely (writes by an
# observer's ``prepare_sync``, reads via ``current_active_observer_span``)
# and only touches OTel when both: (a) the extras are installed, and
# (b) an observer actually published a span. Installs without ``[otel]``
# get a no-op attach/detach pair; the observer ContextVar stays
# ``None`` and nothing changes.
#
# The names are bound to ``None`` in the except branch so pyright
# narrows correctly at call sites (``if _otel_attach is None: ...``)
# rather than flagging "possibly unbound."
try:
    from opentelemetry.context import attach as _otel_attach
    from opentelemetry.context import detach as _otel_detach
    from opentelemetry.trace.propagation import set_span_in_context as _otel_set_span_in_context
except ImportError:  # pragma: no cover — exercised only in non-otel installs
    _otel_attach = None  # type: ignore[assignment]
    _otel_detach = None  # type: ignore[assignment]
    _otel_set_span_in_context = None  # type: ignore[assignment]


def _attach_active_observer_span() -> object | None:
    """Read ``current_active_observer_span``; if an observer published
    one and OTel is installed, attach the span into the OTel context
    so that any logs emitted from the next user-code scope (a node
    body) pick up the right ``trace_id``/``span_id`` via OTel's
    ``LoggingHandler``.

    Returns the OTel context token to hand back to
    :func:`_detach_active_observer_span` in ``finally``, or ``None``
    if no attach happened (no observer, no OTel, or both).
    """
    if _otel_attach is None or _otel_set_span_in_context is None:
        return None
    span = current_active_observer_span()
    if span is None:
        return None
    return _otel_attach(_otel_set_span_in_context(cast("Any", span)))


def _detach_active_observer_span(token: object | None) -> None:
    """Pair to :func:`_attach_active_observer_span`. No-op when no
    attach was performed (token is ``None``)."""
    if token is None or _otel_detach is None:
        return
    _otel_detach(cast("Any", token))


def _merge_partial[StateT: State](
    prior: StateT,
    partial: Mapping[str, Any],
    reducers: Mapping[str, Reducer],
    producing_node: str,
) -> StateT:
    """Apply per-field reducers to merge a node's partial update into prior state.

    Re-validates the resulting state against the schema (validation
    happens at node boundaries). Wraps reducer failures as
    ``ReducerError`` and schema failures as ``StateValidationError``.
    """

    # Lazy import to avoid a textual cycle (parallel_branches has a
    # TYPE_CHECKING back-reference to this module). _MultiContribution
    # is the sentinel ParallelBranchesNode uses when multiple branches
    # write the same parent field — each value flows through the
    # parent's reducer in branch insertion order per spec §11.4 +
    # §11.8.
    from .parallel_branches import _MultiContribution  # noqa: PLC0415

    new_values = prior.model_dump()
    for field_name, partial_value in partial.items():
        reducer = reducers.get(field_name)
        if reducer is None:
            # Unknown field — surface as a schema validation failure below.
            new_values[field_name] = partial_value
            continue
        try:
            if isinstance(partial_value, _MultiContribution):
                # Per pipeline-utilities §11.4: multi-branch
                # contributions to one parent field apply in branch
                # insertion order via the parent's reducer. Fold
                # each value in sequence.
                acc = new_values[field_name]
                for v in partial_value.values:
                    acc = reducer(acc, v)
                new_values[field_name] = acc
            else:
                new_values[field_name] = reducer(new_values[field_name], partial_value)
        except Exception as e:
            raise ReducerError(
                field_name=field_name,
                reducer_name=reducer.name,
                producing_node=producing_node,
                cause=e,
                recoverable_state=prior,
            ) from e

    try:
        # type(prior) narrows to `type[StateT]`; model_validate returns StateT.
        return type(prior).model_validate(new_values)
    except ValidationError as e:
        offending = sorted({str(err["loc"][0]) for err in e.errors() if err["loc"]})
        raise StateValidationError(
            f"state validation failed after node {producing_node!r}: {e}",
            fields=offending,
            cause=e,
        ) from e


@dataclass(frozen=True)
class _StepResult[StateT: State]:
    """Return shape of the per-step dispatchers
    (``_step_function_node`` / ``_step_subgraph_node`` /
    ``_step_fan_out_node``).

    The ``completed`` event for the just-completed node fires AFTER
    edge evaluation completes — so that edge-resolution failures
    (``routing_error``, ``edge_exception``) land on the preceding
    node's completed event with ``error`` populated, sharing the
    started/completed pair rather than producing a separate event
    pair.

    The step dispatchers can't call ``_dispatch_completed`` for
    the success path themselves anymore, because the outcome
    isn't knowable until edge eval (which lives in ``_invoke``)
    runs. Failure-path dispatches (``node_exception`` /
    ``reducer_error`` / ``state_validation_error``) still fire
    inline inside ``innermost`` — those errors short-circuit
    before edge eval can run, and the step function raises out.

    For the success path, the step dispatcher returns the
    finalized state plus a closure ``finalize_completed`` that
    ``_invoke`` calls AFTER edge eval, passing either ``None``
    (edge eval succeeded → dispatch completed with
    ``post_state``) or the edge error (dispatch completed with
    ``error`` populated).

    For ``_step_subgraph_node``, the wrapper is transparent per
    fixture 013 (no started/completed pair); ``finalize_completed``
    is a no-op closure so edge errors after a subgraph wrapper
    propagate silently — the "preceding unit's pair" framing applied
    to a unit that never had one. Same for middleware that short-
    circuits without invoking ``next``.
    """

    state: StateT
    finalize_completed: Callable[[RuntimeGraphError | None], None]


def _no_op_finalize(_edge_error: RuntimeGraphError | None) -> None:
    """Default ``finalize_completed`` for cases where the step
    didn't dispatch a started/completed pair — subgraph wrappers
    (transparent per fixture 013) and middleware that short-
    circuits without invoking ``next``. Edge errors propagate
    silently per fixture 013."""


# Helpers for the proposal 0009 per-instance fan-out resume contract.
# The shared mutable ``fan_out_progress_state`` dict on
# _InvocationContext is keyed by ``(namespace, fan_out_node_name)``;
# these helpers locate / project / mutate it consistently.


def _find_innermost_fan_out_instance_state(
    context: _InvocationContext,
) -> _FanOutInstanceState | None:
    """Locate the per-instance state for the innermost active fan-out
    relative to ``context``.

    A node running inside fan-out instance ``i`` of fan-out ``F``
    sees ``context.namespace_prefix`` ending with ``F``'s own name
    and ``context.fan_out_index == i``. Walk the namespace prefix
    back to find the longest matching key in ``fan_out_progress_state``
    so nested fan-outs route to the right level.

    Returns ``None`` when no match is found — defensive against an
    inner node firing outside any registered fan-out (shouldn't
    happen if ``FanOutNode.run_with_context`` correctly registers
    each fan-out before descending). Callers that expect a hit
    surface the missing-state case as a no-op rather than a crash.
    """
    if context.fan_out_index is None:
        return None
    prefix = context.namespace_prefix
    state_dict = context.fan_out_progress_state
    # Walk the prefix from longest to shortest. The innermost
    # fan-out's full key is (namespace_before_fan_out, fan_out_name)
    # where namespace_before_fan_out + (fan_out_name,) == prefix.
    for split in range(len(prefix), 0, -1):
        key = (prefix[: split - 1], prefix[split - 1])
        if key in state_dict:
            exec_state = state_dict[key]
            idx = context.fan_out_index
            if 0 <= idx < len(exec_state.instances):
                return exec_state.instances[idx]
    return None


def _project_fan_out_progress(
    state_dict: Mapping[tuple[tuple[str, ...], str], _FanOutExecutionState],
) -> tuple[FanOutProgress, ...]:
    """Project the engine-internal mutable per-fan-out state into the
    frozen :class:`FanOutProgress` shape on a saved record.

    Per the snapshot semantics, a save fires with ALL concurrent
    fan-out instances' states captured at the moment of the save —
    not just the one whose ``completed`` event triggered the save.
    This projection enumerates the whole dict; the engine save site
    calls it once per save regardless of which fan-out's inner node
    fired the event.

    Deterministic ordering: sort by (namespace, fan_out_node_name).
    Two saves carrying the same logical state then serialize
    byte-identically, which matters for backends that hash records.
    """
    out: list[FanOutProgress] = []
    for (namespace, name), exec_state in sorted(state_dict.items()):
        instances = tuple(
            FanOutInstanceProgress(
                state=inst.state,
                result=inst.result,
                result_is_error=inst.result_is_error,
                completed_inner_positions=tuple(inst.completed_inner_positions),
            )
            for inst in exec_state.instances
        )
        out.append(
            FanOutProgress(
                fan_out_node_name=name,
                namespace=namespace,
                instance_count=exec_state.instance_count,
                instances=instances,
            )
        )
    return tuple(out)


def _restore_fan_out_progress_state(
    saved: Sequence[FanOutProgress],
) -> dict[tuple[tuple[str, ...], str], _FanOutExecutionState]:
    """Inverse projection of :func:`_project_fan_out_progress`. On resume
    the loaded record's frozen ``fan_out_progress`` tuple gets unpacked
    into the mutable per-fan-out tracking dict that ``FanOutNode``
    consults to decide which instances to skip vs re-run.

    Extra-output state isn't preserved across resume — the spec models
    ``result`` as a single accumulator entry and is silent on
    ``extra_outputs``. Reconstructing them would require either
    serializing them on the record (a spec change) or recomputing them
    (defeating the point of skip-on-resume). Fixtures don't exercise
    ``extra_outputs`` on the resume path; if a future workload needs
    them, surface as a follow-on.

    ``result_is_error`` is read verbatim from the saved record's
    explicit field. The earlier structural-pattern heuristic is gone
    — the spec mandates the
    explicit field as the authoritative discriminator because the
    user's state schema can legitimately contain values that match
    the engine's canonical error-record shape, and a heuristic would
    misclassify them.
    """
    out: dict[tuple[tuple[str, ...], str], _FanOutExecutionState] = {}
    for fp in saved:
        instances: list[_FanOutInstanceState] = []
        for inst in fp.instances:
            instances.append(
                _FanOutInstanceState(
                    state=inst.state,
                    result=inst.result,
                    result_is_error=inst.result_is_error,
                    extra_outputs={},
                    completed_inner_positions=list(inst.completed_inner_positions),
                )
            )
        key = (fp.namespace, fp.fan_out_node_name)
        out[key] = _FanOutExecutionState(
            fan_out_node_name=fp.fan_out_node_name,
            namespace=fp.namespace,
            instance_count=fp.instance_count,
            instances=instances,
        )
    return out


async def _save_fan_out_internal(
    checkpointer: Any,
    invocation_id: str,
    record: CheckpointRecord,
) -> None:
    """Route a fan-out-internal save through the checkpointer's
    optional batching seam.

    Checkpointer backends MAY support batching scoped to fan-out
    internal saves. When the backend exposes a
    ``save_fan_out_internal`` coroutine, route there so it can buffer
    or flush per its configuration. Otherwise, fall back to the
    standard ``save`` — non-batching backends see no behavioral change.
    """
    saver = getattr(checkpointer, "save_fan_out_internal", None)
    if saver is None:
        await checkpointer.save(invocation_id, record)
        return
    await saver(invocation_id, record)


async def _save_fan_out_in_flight_failure(  # pyright: ignore[reportUnusedFunction]
    checkpointer: Any,
    invocation_id: str,
    record: CheckpointRecord,
) -> None:
    """Route an "instance failed mid-execution" save through the
    checkpointer's failure-save seam (closing the in_flight
    observability gap).

    Backends that expose ``save_fan_out_in_flight_failure`` get the
    save directly; under batching, the typical implementation
    buffers without triggering the flush count (preserving the
    "buffered saves lost on crash" model). Backends that don't
    expose the hook fall back to ``save`` so non-batching backends
    keep the failure save durable.
    """
    saver = getattr(checkpointer, "save_fan_out_in_flight_failure", None)
    if saver is None:
        await checkpointer.save(invocation_id, record)
        return
    await saver(invocation_id, record)


@dataclass(frozen=True)
class _MigrationSummary:
    """Per-resume migration-chain metadata threaded out of
    ``_migrate_record`` so the engine can dispatch an
    ``openarmature.checkpoint.migrate`` observer event after the
    invocation context is built. Carried on the synthetic
    ``NodeEvent.pre_state``
    payload for ``phase="checkpoint_migrated"``; the OTel observer
    reads it to emit the span.
    """

    from_version: str
    to_version: str
    chain_length: int


def _apply_migration_step(
    migration: StateMigration,
    value: Any,
    label: str,
) -> Any:
    """Apply one migration step to one value (outer state or one
    parent-state entry). Wraps the user-supplied migration function's
    raise as ``CheckpointStateMigrationFailed``. The original
    exception rides ``__cause__``.
    """
    try:
        return migration.migrate(value)
    except CheckpointError:
        # Preserve canonical category — if a migration raises a
        # CheckpointError subclass itself (rare; migrations are
        # spec-mandated pure per §10.12.2), propagate the original
        # category rather than wrapping it as
        # CheckpointStateMigrationFailed.
        raise
    except Exception as exc:
        # Concise wrap-message intentionally. ``raise ... from exc``
        # preserves the original exception on ``__cause__``;
        # Python's traceback formatter surfaces it, so embedding the
        # underlying ``type/str`` in this message would just
        # duplicate information (and confuse the output when the
        # underlying ``__str__`` is multi-line).
        raise CheckpointStateMigrationFailed(
            f"migration {migration.from_version!r}→{migration.to_version!r} raised while migrating {label}",
            from_version=migration.from_version,
            to_version=migration.to_version,
        ) from exc


@dataclass(frozen=True)
class CompiledGraph[StateT: State]:
    """An immutable, executable graph produced by `GraphBuilder.compile()`.

    The compile-time topology (state class, entry, nodes, edges, reducers) is
    immutable. Two mutable lists ride alongside for observer plumbing
    (`_attached_observers` and `_active_workers`), neither of which affect the
    compiled topology and both of which are scoped to the same instance.
    """

    state_cls: type[StateT]
    entry: str
    nodes: Mapping[str, Node[StateT]]
    edges: Mapping[str, StaticEdge | ConditionalEdge[StateT]]
    reducers: Mapping[str, Reducer]
    # Per-graph middleware in registration order (outer-to-inner). Composes
    # OUTSIDE per-node middleware at runtime per pipeline-utilities §3.
    middleware: tuple[Middleware, ...] = ()
    # Observer plumbing — see attach_observer/drain. Mutable on a frozen
    # dataclass: the list reference is fixed but its contents change.
    # Parameterized factories so pyright infers the element types.
    _attached_observers: list[SubscribedObserver] = field(default_factory=list[SubscribedObserver])
    # Per-task `add_done_callback` auto-removes completed workers — long-
    # running services that never call drain() don't accumulate completed
    # Task references indefinitely. Values are the per-invocation
    # `_InvocationContext` so `drain()` can read each worker's
    # `drain_counters` to compute the undelivered-event count at timeout.
    _active_workers: dict[asyncio.Task[None], _InvocationContext] = field(
        default_factory=dict[asyncio.Task[None], _InvocationContext]
    )
    # Single-element list so the frozen-dataclass binding is stable but
    # the user can swap the registered Checkpointer via
    # ``attach_checkpointer``. ``None`` when no backend is registered.
    _checkpointer_slot: list[Checkpointer | None] = field(default_factory=lambda: [None])
    # State-migration registry (pipeline-utilities §10.12 / proposal
    # 0014). Populated by ``GraphBuilder.with_state_migration(s)``;
    # consulted on resume when the loaded record's ``schema_version``
    # does not match the current state class's ``schema_version``.
    migration_registry: MigrationRegistry = field(default_factory=MigrationRegistry)

    # ------------------------------------------------------------------
    # Observer registration (spec v0.6.0 §6)
    # ------------------------------------------------------------------

    def attach_observer(
        self,
        observer: Observer,
        *,
        phases: Iterable[str] | None = None,
    ) -> RemoveHandle:
        """Register a graph-attached observer.

        Graph-attached observers fire on every invocation of this
        graph until removed; including when this graph runs as a
        subgraph inside a parent. Returns a ``RemoveHandle`` whose
        ``.remove()`` method detaches the observer; idempotent.

        ``phases`` selects the phase strings (``"started"``,
        ``"completed"``) the observer subscribes to; default is both.
        An empty ``phases`` set raises ``ValueError`` at registration
        time.

        Changes to the registered set during a graph run do NOT take
        effect until the next invocation. The set of observers
        delivering events for an in-flight invocation is fixed at
        the point the invocation begins.
        """
        subscribed = _coerce_subscribed(observer, phases=phases)
        self._attached_observers.append(subscribed)
        return RemoveHandle(_observers=self._attached_observers, _observer=subscribed)

    # ------------------------------------------------------------------
    # Checkpointer registration
    # ------------------------------------------------------------------

    def attach_checkpointer(self, checkpointer: Checkpointer | None) -> None:
        """Register a Checkpointer for this graph.

        Pass ``None`` to clear a previously-registered backend.
        Without a registered Checkpointer the engine never calls
        ``save()`` and ``invoke(resume_invocation=...)`` raises
        ``checkpoint_not_found``.

        At most one Checkpointer per graph. Calling
        ``attach_checkpointer`` again replaces the previously-
        registered one; multi-backend fan-out is the user's
        responsibility (wrap two underlying Checkpointers behind a
        custom protocol-conforming implementation if needed).
        """
        self._checkpointer_slot[0] = checkpointer

    @property
    def checkpointer(self) -> Checkpointer | None:
        """Currently-registered Checkpointer, or ``None``."""
        return self._checkpointer_slot[0]

    # ------------------------------------------------------------------
    # State migration (pipeline-utilities §10.12 / proposal 0014)
    # ------------------------------------------------------------------

    async def _migrate_record(
        self,
        record: CheckpointRecord,
        checkpointer: Checkpointer,
        invocation_id: str,
        current_schema_version: str,
    ) -> tuple[CheckpointRecord, _MigrationSummary]:
        """Resolve a migration chain for ``record`` and apply it.

        Returns ``(migrated_record, summary)``. ``migrated_record``
        has ``state`` + ``parent_states`` mapped through the chain.
        ``summary`` carries the chain's metadata so the caller can
        dispatch a ``checkpoint_migrated`` observer event after the
        invocation context exists.

        Caller is responsible for the post-migration deserialization
        step: if the migrated state cannot deserialize against the
        current state class, the resulting failure surfaces as
        ``CheckpointRecordInvalid``.

        Parent states MUST be treated as carrying the same
        ``schema_version`` as the outer record, so we apply the same
        chain to every entry in ``parent_states`` lockstep with the
        outer state. Future per-parent versioning would need a
        follow-on.
        """
        # Eligibility check first per §10.12.1: backends that hold
        # typed in-memory state or class-bound serialization cannot
        # expose the class-independent intermediate the registry
        # consumes. Mismatch + no eligibility → CheckpointRecordInvalid.
        if not getattr(checkpointer, "supports_state_migration", False):
            raise CheckpointRecordInvalid(
                invocation_id,
                f"persisted schema_version={record.schema_version!r} does not "
                f"match current {current_schema_version!r}, and the active "
                f"checkpointer ({type(checkpointer).__name__}) does not "
                f"support state migration",
            )

        # resolve_chain raises CheckpointStateMigrationChainAmbiguous
        # directly on multi-shortest-path detection per spec §10.10
        # / §10.12.2 (proposal 0018, spec v0.16.0). No except-wrap
        # needed here — the canonical category propagates straight
        # through and the registry's exception contract is one type
        # regardless of when ambiguity surfaces (register vs resolve).
        chain = self.migration_registry.resolve_chain(
            record.schema_version,
            current_schema_version,
        )

        if chain is None:
            raise CheckpointStateMigrationMissing(
                f"no migration chain from {record.schema_version!r} to {current_schema_version!r}",
                from_version=record.schema_version,
                to_version=current_schema_version,
                registered_migrations_count=len(self.migration_registry),
                registry_description=self.migration_registry.describe(),
            )

        migrated_state: Any = record.state
        migrated_parents: list[Any] = list(record.parent_states)
        for migration in chain:
            migrated_state = _apply_migration_step(migration, migrated_state, "state")
            for i, parent in enumerate(migrated_parents):
                migrated_parents[i] = _apply_migration_step(migration, parent, f"parent_states[{i}]")

        # Per spec §6 cross-ref, the caller dispatches a synthetic
        # ``checkpoint_migrated`` observer event using the summary
        # below as soon as the invocation context exists. We can't
        # dispatch from here because the context isn't built yet.
        summary = _MigrationSummary(
            from_version=record.schema_version,
            to_version=current_schema_version,
            chain_length=len(chain),
        )
        migrated = dataclass_replace(
            record,
            state=migrated_state,
            parent_states=tuple(migrated_parents),
        )
        return migrated, summary

    async def drain(self, timeout: float | None = None) -> DrainSummary:
        """Await delivery of every observer event produced by prior
        invocations of this graph, optionally bounded by ``timeout``.

        Callers running in short-lived processes (scripts, serverless
        functions, CLIs) MUST use drain to avoid losing observer events
        that were dispatched but not yet delivered.

        Only events dispatched before this call are awaited; events
        from invocations started concurrently with drain may or may
        not be included. Subgraph events from active invocations are
        part of the parent invocation's worker and are covered
        automatically.

        ``timeout`` is a non-negative duration in seconds. If omitted
        or ``None``, drain waits indefinitely — a slow, hung, or
        misbehaving observer can therefore hold drain (and the calling
        process) indefinitely. If supplied, drain returns no later
        than ``timeout`` seconds after the call begins; any observer
        events still queued or in-flight at that point are considered
        undelivered. Workers are cancelled via ``Task.cancel()`` so
        the compiled graph remains usable for subsequent invocations
        — partial delivery state from one drain does NOT leak into
        the next invocation.

        Returns a :class:`DrainSummary` with ``undelivered_count`` and
        ``timeout_reached`` fields. The shape is the same whether or
        not a timeout was supplied; on the no-timeout / timeout-not-
        fired path both fields are zero / false.

        Observers SHOULD be written to be cancellation-safe
        (idempotent writes, try/finally cleanup) so that interruption
        by drain timeout does not leave partial side effects in an
        inconsistent state.

        Raises ``ValueError`` if ``timeout`` is negative or NaN.
        Non-numeric input raises ``TypeError`` from the comparison.
        """
        # ``not (timeout >= 0)`` is the right check: catches negative
        # values, catches NaN (all comparisons with NaN return False),
        # and lets non-numeric input raise ``TypeError`` from the
        # comparison operator itself. Silently treating a negative
        # timeout as "immediate cancel" would be a user-hostile failure
        # mode — the spec contract is non-negative seconds.
        if timeout is not None and not (timeout >= 0):
            raise ValueError(f"drain timeout must be non-negative, got {timeout!r}")
        if not self._active_workers:
            return DrainSummary(undelivered_count=0, timeout_reached=False)
        # Snapshot the dict: each worker's done-callback removes its
        # entry from `_active_workers`, so iterating directly while
        # `asyncio.wait` awaits would mutate during iteration.
        snapshot = dict(self._active_workers)
        workers = list(snapshot.keys())

        _done, pending = await asyncio.wait(
            workers,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        if pending:
            undelivered = sum(
                snapshot[w].drain_counters.dispatched - snapshot[w].drain_counters.delivered for w in pending
            )
            timeout_reached = True
            for w in pending:
                w.cancel()
        else:
            undelivered = 0
            timeout_reached = False

        # Gather ALL workers (done + pending) so any exception that
        # escaped a delivery worker surfaces here instead of leaking
        # as a "Task exception was never retrieved" warning. The
        # ``return_exceptions=True`` absorbs both the synthetic
        # ``CancelledError`` from cancelled workers and any genuine
        # bug-escape from a ``deliver_loop`` that ever raised past
        # its inner ``warnings.warn`` isolation. Also load-bearing
        # for the cross-invocation cleanliness contract — done-
        # callbacks fire on cancellation, so ``_active_workers`` is
        # empty by the time we return.
        await asyncio.gather(*workers, return_exceptions=True)

        return DrainSummary(undelivered_count=undelivered, timeout_reached=timeout_reached)

    # Spec graph-engine §6 *Per-invocation drain* (proposal 0054).
    # Symmetric with the process-wide ``drain`` method on the same
    # class but scoped to one in-flight invocation, with one
    # spec-mandated divergence: the per-invocation primitive MUST
    # NOT cancel the deliver worker on timeout (drain is shutdown
    # semantics; this is in-flight synchronization). The snapshot
    # semantic — events dispatched after the call begins do not
    # extend the target — is what keeps an in-node call (e.g., a
    # terminal node draining its own invocation before reading a
    # queryable observer accumulator) from deadlocking on its own
    # ``completed`` event.
    async def drain_events_for(
        self,
        invocation_id: str,
        *,
        timeout: float | None = 5.0,
    ) -> DrainSummary:
        """Await delivery of every observer event tagged with
        ``invocation_id`` that was dispatched as of this call's entry,
        optionally bounded by ``timeout``.

        Use this from a terminal node body to synchronize on the
        observer event stream before reading derived observer state
        (a queryable accumulator's per-invocation bucket, a latency
        rollup, a token-usage record). The drain blocks until every
        event dispatched up to the moment of the call has reached
        every attached observer, then returns.

        Snapshot semantic: the drain awaits the events dispatched as
        of call time. Events emitted after the call begins (notably
        the calling node's own ``completed`` event, which fires only
        after the node body returns) are out of scope. This is what
        allows an in-node call to avoid deadlocking on its own
        completed event. The calling node's ``started`` event, by
        contrast, fires immediately BEFORE the body runs and IS in
        the snapshot — the drain awaits its delivery normally.

        ``timeout`` is a non-negative duration in seconds (default
        ``5.0``). ``None`` waits indefinitely. ``timeout=0.0`` is a
        non-blocking check: returns immediately whether the snapshot
        target was met. Raises :class:`ValueError` on negative or
        ``NaN`` input.

        On timeout the deliver worker is left running. The compiled
        graph stays available to serve other invocations after a
        per-invocation drain times out; the deliver loop continues
        processing the queue, including the events the timed-out
        caller failed to await. This is the load-bearing difference
        from :meth:`drain`, which cancels its workers.

        Returns a :class:`DrainSummary` with ``undelivered_count`` and
        ``timeout_reached``. On the clean path both are zero / false;
        on timeout ``undelivered_count`` is the snapshot target minus
        the deliver loop's current ``delivered`` count for this
        invocation. Unknown ``invocation_id`` (no active worker, or
        the invocation has already drained and the worker has exited)
        returns an empty summary — not an error.

        Interaction with :meth:`drain`: if process-wide ``drain`` is
        called while a per-invocation drain is pending, ``drain``'s
        shutdown semantics take precedence. The deliver worker is
        cancelled, its remaining events are not delivered, and the
        per-invocation waker's target may never be reached. The
        per-invocation call then blocks until its own ``timeout``
        fires and returns ``timeout_reached=True``. Mixing the two
        primitives in the same shutdown path is unusual; use
        ``drain`` for lifespan / shutdown coordination and
        ``drain_events_for`` for in-flight synchronization.
        """
        if timeout is not None and not (timeout >= 0):
            raise ValueError(f"drain_events_for timeout must be non-negative, got {timeout!r}")

        target_context: _InvocationContext | None = None
        for context in self._active_workers.values():
            if context.invocation_id == invocation_id:
                target_context = context
                break
        if target_context is None:
            return DrainSummary(undelivered_count=0, timeout_reached=False)

        counters = target_context.drain_counters
        snapshot_target = counters.dispatched
        if counters.delivered >= snapshot_target:
            return DrainSummary(undelivered_count=0, timeout_reached=False)

        waker: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        counters.drain_wakers.append((snapshot_target, waker))
        try:
            await asyncio.wait_for(waker, timeout=timeout)
        except TimeoutError:
            counters.drain_wakers = [(t, f) for t, f in counters.drain_wakers if f is not waker]
            undelivered = max(0, snapshot_target - counters.delivered)
            return DrainSummary(undelivered_count=undelivered, timeout_reached=True)
        return DrainSummary(undelivered_count=0, timeout_reached=False)

    # ------------------------------------------------------------------
    # Public invocation
    # ------------------------------------------------------------------

    async def invoke(
        self,
        initial_state: StateT,
        observers: Iterable[Observer | SubscribedObserver] | None = None,
        *,
        correlation_id: str | None = None,
        invocation_id: str | None = None,
        resume_invocation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StateT:
        """Run the graph from ``initial_state`` to END and return the
        final state.

        Optional ``observers`` are invocation-scoped; they fire only
        for this run, after all graph-attached observers (including
        subgraph-attached ones for events originating in subgraphs).

        Each entry in ``observers`` may be either a bare ``Observer``
        callable (subscribes to both phases) or a ``SubscribedObserver``
        wrapping an observer with an explicit ``phases`` set.

        This method returns as soon as the graph execution loop
        completes, regardless of whether the observer delivery queue
        has finished processing every dispatched event. Use
        ``await compiled.drain()`` if you need delivery-completion
        guarantees.

        **Checkpointing.**

        - ``correlation_id`` is the per-invocation cross-backend join
          key. Caller-supplied or auto-generated UUIDv4 when absent.
          Preserved unchanged across ``resume_invocation``.
        - ``invocation_id`` is the per-attempt id.
          Caller-supplied or auto-generated UUIDv4 when absent; a
          caller value MAY be any non-empty URL-safe string. Applies
          to the fresh-invocation path only — a ``resume_invocation``
          mints a fresh id regardless (each attempt is its own
          invocation).
        - ``resume_invocation`` names a prior ``invocation_id`` to
          resume from. Requires a registered Checkpointer; raises
          ``CheckpointNotFound`` when the backend has no record for
          the supplied id, ``CheckpointRecordInvalid`` when the
          loaded record's schema is incompatible. Resume mints a NEW
          ``invocation_id``; each attempt is its own invocation in
          the observability sense; the ``correlation_id`` is the
          cross-attempt join key.
        - **Save-failure policy.** This implementation raises
          ``CheckpointSaveFailed`` to the caller of ``invoke()``
          immediately when ``Checkpointer.save`` raises; saves are
          NOT retried by the engine. Wrap the Checkpointer in your
          own retry logic if transient backend failures should be
          reattempted.

        **Caller-supplied invocation metadata.**

        - ``metadata`` is an optional mapping of arbitrary
          ``key → value`` entries the framework propagates to every
          observability backend. Values MUST be OTel-attribute-
          compatible scalars (``str`` / ``int`` / ``float`` / ``bool``)
          or homogeneous arrays of those types. Keys MUST NOT use
          the ``openarmature.*`` or ``gen_ai.*`` reserved namespaces.
          Validation runs synchronously at the API boundary; rule
          violations raise ``ValueError`` BEFORE any work begins.
        - The OTel observer emits each entry as an
          ``openarmature.user.<key>`` cross-cutting span attribute on
          every span and OTel log record. The Langfuse observer
          merges each entry into ``trace.metadata`` AND every
          ``observation.metadata`` (top level, sibling to
          ``correlation_id``).
        - Mid-invocation augmentation via
          :func:`openarmature.observability.set_invocation_metadata`
          merges into the same ContextVar with the same validation
          rules; affects spans emitted AFTER the call returns.

        Raises one of the runtime error categories on failure.
        """
        # Validate caller-supplied metadata at the API boundary so any
        # rule violation surfaces synchronously before the worker task
        # is created or any node body runs.
        validated_metadata = validate_invocation_metadata(metadata)

        invocation_scoped = tuple(_coerce_subscribed(o) for o in (observers or ()))
        queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()

        # Resolve the resume path BEFORE building the context so we can
        # restore the correlation_id from the saved record (per §10.4
        # step 3) and pre-populate the skip-set + completed_positions.
        starting_state: StateT = initial_state
        resolved_correlation_id = correlation_id or str(uuid.uuid4())
        # Caller-supplied invocation_id (proposal 0039) applies to the
        # fresh-invocation path only; a resume mints a fresh id
        # regardless (each attempt is its own invocation, §5.1).
        if invocation_id is not None and resume_invocation is None:
            invocation_id = validate_invocation_id(invocation_id)
        else:
            invocation_id = str(uuid.uuid4())
        resume_skip_set: frozenset[tuple[str, ...]] = frozenset()
        completed_positions: list[NodePosition] = []
        pending_resume_states: dict[int, Any] = {}
        # Populated by ``_migrate_record`` during a version-mismatched
        # resume; left ``None`` for the no-resume + versions-match
        # paths. Dispatched as a synthetic ``checkpoint_migrated``
        # observer event after the invocation context is built so
        # the OTel observer can emit an
        # ``openarmature.checkpoint.migrate`` span per spec §6.
        migration_summary: _MigrationSummary | None = None
        if resume_invocation is not None:
            checkpointer = self._checkpointer_slot[0]
            if checkpointer is None:
                # §10.1.1: resume against an unregistered backend
                # surfaces as ``checkpoint_not_found`` — the user has
                # misconfigured the run.
                raise CheckpointNotFound(resume_invocation)
            record = await checkpointer.load(resume_invocation)
            if record is None:
                raise CheckpointNotFound(resume_invocation)
            # Per spec §10.12 (proposal 0014): version-mismatch resume.
            # Routing precedence (per §10.10 + §10.12.1):
            #   1. unsupported backend → CheckpointRecordInvalid.
            #      Backends that hold typed in-memory state or
            #      class-bound serialization can't expose the
            #      class-independent intermediate the migration
            #      registry needs.
            #   2. no chain in the registry → CheckpointStateMigrationMissing.
            #      Actionable: register a migration.
            #   3. chain found but a migration raises →
            #      CheckpointStateMigrationFailed.
            #   4. post-migration state fails to deserialize →
            #      CheckpointRecordInvalid (the §10.12.4 boundary).
            # Order matters — do NOT swap eligibility and registry-lookup.
            current_schema_version = self.state_cls.schema_version
            if record.schema_version != current_schema_version:
                record, migration_summary = await self._migrate_record(
                    record,
                    checkpointer,
                    resume_invocation,
                    current_schema_version,
                )
            # The saved record's ``state`` is post-merge state at the
            # saving node's level (depth = len(parent_states)). For
            # outer-level saves, parent_states is empty and ``state``
            # IS the outermost state. For inner-node saves
            # (parent_states populated), the OUTERMOST state lives in
            # ``parent_states[0]`` and the deeper levels are
            # parent_states[1:] + (state,) at depths 1..N. The descent
            # path consumes the depth-keyed map to skip projection
            # when re-entering an in-flight subgraph.
            parent_states_chain: tuple[Any, ...] = record.parent_states
            if parent_states_chain:
                outer_raw = parent_states_chain[0]
                # Inner depths 1..N: parent_states[1:] then state at depth N.
                deeper_states = list(parent_states_chain[1:]) + [record.state]
                for depth, st in enumerate(deeper_states, start=1):
                    pending_resume_states[depth] = st
            else:
                outer_raw = record.state
            # State coercion: if the record carries a Pydantic instance
            # (in-memory backend), use it directly; if it's a dict (JSON-
            # mode SQLite), re-validate against the declared state class.
            # A validation failure means the persisted record is
            # incompatible with the current graph (state-shape mismatch
            # or missing required fields), which §10.10 names as
            # ``checkpoint_record_invalid`` — wrap the ValidationError
            # so callers see the canonical category, not the raw
            # pydantic exception.
            if isinstance(outer_raw, dict):
                try:
                    starting_state = self.state_cls.model_validate(outer_raw)
                except ValidationError as exc:
                    raise CheckpointRecordInvalid(
                        resume_invocation,
                        f"saved outer state does not validate against {self.state_cls.__name__}: {exc}",
                    ) from exc
            else:
                starting_state = cast("StateT", outer_raw)
            # §10.4 step 3: keep the original correlation_id verbatim.
            # Per spec resume MUST preserve the cross-backend join key.
            resolved_correlation_id = record.correlation_id
            completed_positions = list(record.completed_positions)
            # Skip-set keys are the FULL identity tuple of a node:
            # NodePosition.namespace + (NodePosition.node_name,). This
            # matches what the engine looks up at run time
            # (``context.namespace_prefix + (current,)``).
            resume_skip_set = frozenset(p.namespace + (p.node_name,) for p in completed_positions)
            # Per spec §10.7 / §10.11: restore per-fan-out per-instance
            # state from the loaded record. ``FanOutNode.run_with_context``
            # consults this on re-dispatch — completed instances skip,
            # in_flight / not_started instances re-execute. Empty tuple
            # when no fan-outs were in flight at save time.
            fan_out_progress_state = _restore_fan_out_progress_state(record.fan_out_progress)
        else:
            fan_out_progress_state = {}

        context = _InvocationContext(
            queue=queue,
            graph_attached=tuple(self._attached_observers),
            invocation_scoped=invocation_scoped,
            invocation_id=invocation_id,
            correlation_id=resolved_correlation_id,
            checkpointer=self._checkpointer_slot[0],
            completed_positions=completed_positions,
            resume_skip_set=resume_skip_set,
            pending_resume_states=pending_resume_states,
            resume_invocation=resume_invocation,
            fan_out_progress_state=fan_out_progress_state,
            # Per spec §10.2 (proposal 0028): the canonical source for
            # ``schema_version`` on every save during this invocation.
            # Threaded unchanged through every descent.
            state_cls=self.state_cls,
        )
        # Spec observability §3.1: the correlation_id MUST be readable
        # from anywhere within the invocation's async call tree via the
        # language's idiomatic context primitive. Set the ContextVar
        # BEFORE creating the delivery worker so the worker's captured
        # context sees the correlation_id (asyncio.create_task snapshots
        # the current Context at creation time). Reset on return so
        # subsequent invocations get a fresh slate. Nested ``invoke()``
        # calls (subgraph-as-node uses ``_invoke`` directly, not the
        # public ``invoke``, so they don't re-set; see §3.1's
        # "per-invocation is OUTERMOST invoke" wording).
        correlation_token = _set_correlation_id(resolved_correlation_id)
        invocation_token = _set_invocation_id(invocation_id)
        metadata_token = _set_invocation_metadata(validated_metadata)
        worker = asyncio.create_task(deliver_loop(queue, context.drain_counters))
        self._active_workers[worker] = context
        # Auto-prune: when the worker completes (after the sentinel is
        # processed, or after cancellation by drain() on timeout), remove
        # it from the active set so long-running services don't leak Task
        # references between drain() calls. ``pop(key, None)`` is the
        # idempotent form — if a concurrent drain() removed the entry
        # already (it shouldn't with the current design, but the no-arg
        # form would raise KeyError), this is a safe no-op.
        worker.add_done_callback(lambda t: self._active_workers.pop(t, None))
        # Per spec §6 cross-ref in proposal 0014: dispatch the
        # ``checkpoint_migrated`` event as soon as the delivery
        # worker is alive but before any node runs, so the OTel
        # observer can emit an
        # ``openarmature.checkpoint.migrate`` span ahead of the
        # invocation's node spans. The synthetic event carries the
        # ``_MigrationSummary`` on ``pre_state`` mirroring the
        # ``checkpoint_saved`` convention (state-on-pre, post=None).
        if migration_summary is not None:
            _dispatch(
                context,
                NodeEvent(
                    node_name="openarmature.checkpoint.migrate",
                    namespace=("openarmature.checkpoint.migrate",),
                    step=-1,
                    phase="checkpoint_migrated",
                    pre_state=migration_summary,
                    post_state=None,
                    error=None,
                    parent_states=(),
                    caller_invocation_metadata=current_invocation_metadata(),
                ),
            )
        # Proposal 0043: invocation-boundary event for trace.input
        # sourcing. Carries the engine-constructed initial_state plus
        # the §3 / §5.1 ids and the outermost-graph entry node name
        # so Trace-level observers (Langfuse) can populate
        # ``trace.input`` via the §8.4.1 three-lever decision tree.
        # Dispatched AFTER the checkpoint-migrated event (when there
        # is one) so the migration span is observable before the
        # invocation-input event.
        _dispatch(
            context,
            InvocationStartedEvent(
                initial_state=starting_state,
                invocation_id=invocation_id,
                correlation_id=resolved_correlation_id,
                entry_node=self.entry,
            ),
        )
        final_state: StateT | None = None
        status: Literal["completed", "failed"] = "failed"
        try:
            final_state = await self._invoke(starting_state, context)
            status = "completed"
            return final_state
        finally:
            # Proposal 0043: invocation-boundary event for trace.output
            # sourcing. Fires on both the success path
            # (status="completed") and the failure path
            # (status="failed"). ``final_node`` comes from the shared
            # box the engine populates as nodes enter; on the failure
            # path that's the inner-most node that raised, on the
            # success path that's the last node before the END-routing
            # edge. ``final_state`` precedence: the engine's returned
            # state on success → the most recent successful step's
            # post-merge state on a mid-graph raise (per §8.4.1
            # *Resume semantics* "partial final state captured at the
            # failure point") → ``starting_state`` only when no step
            # ever completed.
            if context.final_node_box:
                final_node = context.final_node_box[0]
            else:
                # Defensive: invocation raised before any node fired
                # (e.g., resume-path validation). Fall back to the
                # declared entry node.
                final_node = self.entry
            # ``latest_state_box`` is typed ``list[Any]`` on
            # _InvocationContext (the context isn't parameterized on
            # StateT), but at the outermost level (where this code
            # runs) it always holds an outer ``StateT`` from a
            # successful step's post-merge state.  Cast for type
            # narrowing; the per-context box-isolation pinned by
            # ``test_failure_path_final_state_is_outer_type_*`` keeps
            # this invariant honest.
            event_final_state: StateT
            if final_state is not None:
                event_final_state = final_state
            elif context.latest_state_box:
                event_final_state = cast("StateT", context.latest_state_box[0])
            else:
                event_final_state = starting_state
            _dispatch(
                context,
                InvocationCompletedEvent(
                    final_state=event_final_state,
                    status=status,
                    final_node=final_node,
                    invocation_id=invocation_id,
                    correlation_id=resolved_correlation_id,
                ),
            )
            _reset_invocation_metadata(metadata_token)
            _reset_invocation_id(invocation_token)
            _reset_correlation_id(correlation_token)
            # Sentinel terminates the worker after it processes events
            # already on the queue (including any error event we just
            # dispatched on the failure path). Drain semantics live on
            # `.drain()` — we do NOT await the worker here, per spec.
            queue.put_nowait(_DRAIN_SENTINEL)

    # ------------------------------------------------------------------
    # Internal invocation (used by SubgraphNode for nested execution)
    # ------------------------------------------------------------------

    async def _invoke(
        self,
        initial_state: StateT,
        context: _InvocationContext,
    ) -> StateT:
        """Execution loop that dispatches events through the supplied context.

        Public `invoke()` builds a fresh root context. Subgraph-as-node
        execution calls `_invoke` directly with a context derived from the
        parent's, so the queue, step counter, and observer chain thread
        through the boundary.
        """

        state = initial_state
        current = self.entry

        while True:
            node = self.nodes[current]

            # Resume gate (spec §10.4 step 5). When resume_invocation
            # populated ``resume_skip_set``, any node whose namespace
            # tuple matches a saved completed_position is skipped —
            # the loaded ``state`` already reflects that node's
            # contribution, so we just advance to its outgoing edge
            # without re-running it. The skip applies uniformly to
            # function nodes, subgraph wrappers, and fan-out nodes:
            # a subgraph that fully completed in the prior run does
            # not re-enter; a fan-out that fully completed does not
            # re-fan-out. Partially-completed subgraphs have their
            # wrapper-level position absent (the wrapper's save
            # didn't fire), so the engine descends and the inner
            # _invoke filters its own inner positions against the
            # same skip-set.
            current_namespace = context.namespace_prefix + (current,)
            if current_namespace in context.resume_skip_set:
                # Advance edge selection from loaded state.
                edge = self.edges[current]
                skip_target: str | EndSentinel
                if isinstance(edge, StaticEdge):
                    skip_target = edge.target
                else:
                    try:
                        skip_target = edge.fn(state)
                    except Exception as e:
                        raise EdgeException(source_node=current, cause=e, recoverable_state=state) from e
                if skip_target is END:
                    return state
                if not isinstance(skip_target, str) or skip_target not in self.nodes:
                    raise RoutingError(source_node=current, returned=skip_target, recoverable_state=state)
                current = skip_target
                continue

            # Lazy import: keeps the textual cycle off the module
            # graph (``fan_out`` has a TYPE_CHECKING back-reference
            # to this module). Function-scope import is cheap once
            # cached; this branch fires once per fan-out step.
            from .fan_out import FanOutNode  # noqa: PLC0415
            from .parallel_branches import ParallelBranchesNode  # noqa: PLC0415

            # Proposal 0043: track the most recent node about to run
            # so the outermost ``invoke()`` can populate
            # ``InvocationCompletedEvent.final_node`` on both the
            # END-reached success path (last node before the
            # END-routing edge) and the failure path (the node that
            # raised). Subgraph descents reuse the same shared box
            # via ``descend_into_subgraph``, so a failure deep in a
            # subgraph leaves the innermost node's name in the box —
            # the actual culprit, not the wrapper.
            context.final_node_box[:] = [current]

            if isinstance(node, FanOutNode):
                # Fan-out nodes are recognized as a distinct node type
                # per pipeline-utilities §9. Dispatched through
                # ``_step_fan_out_node`` which wraps the whole fan-out
                # as one parent dispatch (per §9.6) — instance-level
                # concurrency lives inside the FanOutNode itself.
                fn_node = cast("FanOutNode[StateT, State]", node)
                step_result = await self._step_fan_out_node(fn_node, current, state, context)
            elif isinstance(node, ParallelBranchesNode):
                # Parallel-branches nodes are recognized as a distinct
                # node type per pipeline-utilities §11. Dispatched
                # through ``_step_parallel_branches_node`` which wraps
                # the whole dispatch as one parent unit (per §11.6) —
                # M heterogeneous subgraphs run concurrently inside.
                step_result = await self._step_parallel_branches_node(node, current, state, context)
            elif isinstance(node, SubgraphNode):
                # Subgraph wrappers are transparent to the observer protocol
                # (per fixture 013): no event is dispatched for the wrapper
                # itself, the step counter does not advance for it, and any
                # `RuntimeGraphError` bubbling up from the subgraph's
                # _invoke is already wrapped with the inner node's identity
                # — pass it through. Other exceptions (projection errors,
                # subgraph state-class init errors) escape the spec §4
                # categories, so we wrap them as NodeException tagged with
                # the wrapper's name.
                #
                # Per pipeline-utilities §4: the parent's middleware wraps
                # the subgraph dispatch as a single atomic call. Subgraph-
                # internal nodes have their own middleware (from the
                # subgraph's own CompiledGraph.middleware tuple) and do
                # NOT see the parent's middleware. Cast erases ChildT
                # because the dispatcher only needs to invoke `node.run`
                # and pass the parent's chain — the inner state class
                # lives on the subgraph's own CompiledGraph.
                sub = cast("SubgraphNode[StateT, State]", node)
                step_result = await self._step_subgraph_node(sub, current, state, context)
            else:
                step_result = await self._step_function_node(node, current, state, context)
            state = step_result.state
            # Proposal 0043 (post-PR-99 review): surface the most
            # recent successful step's post-merge state so the
            # outermost ``invoke()`` can populate
            # ``InvocationCompletedEvent.final_state`` on the failure
            # path with the partial state, not the bare initial state.
            # Updated AFTER ``state = step_result.state`` so an
            # exception inside the step bypasses this assignment and
            # the previous value (or the empty box) survives.
            context.latest_state_box[:] = [state]

            # Proposal 0043 (post-PR-99 review): restore the outer
            # ``current`` to the shared box after a successful step.
            # Descended `_step_*` calls (subgraph, fan-out, parallel-
            # branches) write inner-node names into the box; without
            # this restore, the wrapper's name leaks out of the box
            # when the wrapper is the last node before the END-routing
            # edge — and for parallel-branches the box would end with
            # whichever branch's inner finished last (nondeterministic).
            # On the failure path, the raise above bypasses this line,
            # so the inner-most node that raised stays in the box as
            # the failure-path ``final_node`` (matching spec §4
            # attribution).
            context.final_node_box[:] = [current]

            # Per spec graph-engine §3 step 3 (revised in proposal
            # 0012 / v0.9.0): the engine MUST dispatch the
            # ``completed`` event AFTER edge evaluation completes.
            # Edge-resolution failures (``routing_error`` /
            # ``edge_exception``) populate the ``error`` field of
            # the just-completed node's ``completed`` event,
            # sharing the started/completed pair rather than
            # producing a separate one (§6 revised). The step
            # function deferred its success-case dispatch via
            # ``finalize_completed``; we call it below with the
            # edge outcome.
            edge = self.edges[current]
            edge_error: RuntimeGraphError | None = None
            target: str | EndSentinel | None = None
            if isinstance(edge, StaticEdge):
                target = edge.target
            else:
                try:
                    target = edge.fn(state)
                except Exception as e:
                    edge_error = EdgeException(source_node=current, cause=e, recoverable_state=state)
            if edge_error is None:
                # Validate the conditional edge's return — undeclared
                # target is a ``routing_error``.
                if target is not END and not (isinstance(target, str) and target in self.nodes):
                    edge_error = RoutingError(source_node=current, returned=target, recoverable_state=state)

            # Dispatch the deferred completed event with the edge
            # outcome. For function and fan-out nodes this is the
            # success/failure dispatch the proposal pinned to
            # post-edge-eval timing. For subgraph wrappers (no
            # event pair) this is a no-op closure per
            # ``_step_subgraph_node``'s `_no_op_finalize` —
            # silent propagation per proposal 0012 + fixture 013.
            step_result.finalize_completed(edge_error)
            if edge_error is not None:
                raise edge_error

            if target is END:
                return state
            # Non-END targets are validated above; mypy/pyright
            # don't narrow through the ``edge_error`` path, so
            # cast for the assignment.
            current = cast("str", target)

    async def _step_function_node(
        self,
        node: Node[StateT],
        current: str,
        state: StateT,
        context: _InvocationContext,
    ) -> _StepResult[StateT]:
        """Run one function-node step through the middleware chain.

        The runtime chain composes:

            [per_graph...] -> [per_node...] -> innermost

        where ``innermost`` is the per-attempt dispatch wrapper around
        ``node.run`` + reducer merge + observer event dispatch. Each call
        to ``innermost`` is one attempt; middleware that calls ``next``
        repeatedly (e.g., retry) produces multiple attempts and therefore
        multiple started/completed event pairs from the engine, each
        tagged with an incrementing ``attempt_index``.

        The success-case ``completed`` event for the FINAL successful
        attempt fires AFTER edge eval, not
        inside ``innermost``. Failure-case dispatches
        (``node_exception`` / ``reducer_error`` /
        ``state_validation_error``) stay inline in ``innermost`` —
        those errors short-circuit before edge eval can run, so the
        spec's "before the failure propagates" MUST is preserved by
        the inline dispatch.

        Returns a :class:`_StepResult` carrying the merged state +
        a ``finalize_completed`` closure that ``_invoke`` invokes
        after edge eval, passing either ``None`` (edge succeeded) or
        the edge error (``RoutingError`` / ``EdgeException``). The
        closure dispatches the deferred completed event with the
        right shape: ``post_state=merged`` on success, ``error``
        populated on edge-resolution failure.
        """
        step = context.take_step()
        namespace = context.namespace_prefix + (current,)

        # Mutable single-element list so innermost (a closure) can
        # increment the counter while the outer function still reads
        # the final value after ``chain`` returns — needed to record
        # the final successful attempt_index in the checkpoint save.
        attempt_counter: list[int] = [0]

        # Cell holding the FINAL successful attempt's
        # (attempt_index, pre_state, merged) — populated by
        # ``innermost`` on each successful invocation, overwritten
        # if retry middleware re-enters. Stays ``None`` if the chain
        # never reached a successful attempt (e.g., middleware
        # short-circuited without invoking ``next``, or every
        # attempt failed and the chain raised).
        deferred_info: list[tuple[int, StateT, StateT] | None] = [None]

        async def innermost(s: Any) -> Mapping[str, Any]:
            # Per pipeline-utilities §5 + graph-engine §6: per-attempt
            # events use the wrapped §4 error type (NodeException etc.)
            # for the observer's `error` field, but the RAW exception
            # propagates up the chain so middleware classifiers can read
            # the original `category` attribute (timing's
            # exception_category, retry's classifier). The engine wraps
            # any exception that escapes the chain, OUTSIDE this layer.
            attempt_counter[0] += 1

            # Per graph-engine §6 (clarified in v0.16.1): event
            # emission reads ``attempt_index`` from the ContextVar set
            # by any enclosing retry middleware — direct (per-node
            # MW) or transitive (instance / branch MW on a subgraph
            # the retry re-invokes). The engine itself no longer
            # writes the var; innermost-wins precedence falls out of
            # Python's ContextVar token-stack semantics.
            attempt_index = current_attempt_index()

            self._dispatch_started(context, current, namespace, step, s, attempt_index=attempt_index)

            # Splice the observer-published span (if any) into the
            # OTel context so logs emitted from the FIRST line of
            # the node body — before any ``await`` — pick up the
            # right trace_id/span_id via OTel's LoggingHandler.
            # Detach in ``finally`` so retries / merge / completed
            # dispatch don't run with the span still active, and
            # clear ``current_active_observer_span`` to ``None`` so
            # the next dispatch that raises or early-returns from
            # ``prepare_sync`` can't reveal this node's span as a
            # stale value to the engine's read.
            otel_token = _attach_active_observer_span()
            try:
                try:
                    partial = await node.run(s)
                except Exception as e:
                    wrapped = NodeException(node_name=current, cause=e, recoverable_state=s)
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        s,
                        error=wrapped,
                        attempt_index=attempt_index,
                    )
                    raise
            finally:
                _detach_active_observer_span(otel_token)
                _set_active_observer_span(None)

            try:
                merged = _merge_partial(s, partial, self.reducers, current)
            except (ReducerError, StateValidationError) as e:
                self._dispatch_completed(
                    context,
                    current,
                    namespace,
                    step,
                    s,
                    error=e,
                    attempt_index=attempt_index,
                )
                raise

            # Defer the success-case completed dispatch to
            # ``finalize_completed`` per proposal-0012; just
            # record the info for the outer scope.
            deferred_info[0] = (attempt_index, cast("StateT", s), cast("StateT", merged))
            # Return the partial (not the merged state) so middleware sees
            # the partial-update shape per pipeline-utilities §2. The
            # engine's canonical merge against the original state happens
            # below, after the chain returns.
            return partial

        chain: ChainCall = compose_chain(
            list(self.middleware) + list(node.middleware),
            innermost,
        )

        # Spec observability §3 / Phase 6 LLM-span hook: capability
        # backends emitting from inside a node body (the
        # llm-provider span instrumentation in OpenAIProvider) need
        # to find the observers active for THIS invocation, which
        # node is calling, and which fan-out instance (if any) the
        # call belongs to. ``namespace_prefix`` and ``fan_out_index``
        # are set in this outer scope (per-node, not per-attempt);
        # ``attempt_index`` is set inside ``innermost`` per attempt.
        # All four reset in ``try/finally`` so an exception escaping
        # the chain still restores the prior values.
        observers_token = _set_active_observers(context.full_observers())
        dispatch_token = _set_active_dispatch(lambda event: _dispatch(context, event))
        namespace_token = _set_namespace_prefix(namespace)
        fan_out_token = _set_fan_out_index(context.fan_out_index)
        # Per proposal 0045 (v0.37.0): drive the per-depth chain
        # ContextVars from the context so ``set_invocation_metadata``
        # sees the full lineage chain at augmentation time.
        fan_out_chain_token = _set_fan_out_index_chain(context.fan_out_index_chain)
        branch_chain_token = _set_branch_name_chain(context.branch_name_chain)
        try:
            try:
                final_partial = await chain(state)
            except RuntimeGraphError:
                raise
            except CheckpointError:
                # CheckpointError categories (CheckpointRecordInvalid,
                # CheckpointStateMigrationMissing, CheckpointSaveFailed,
                # …) are sibling-typed to RuntimeGraphError but carry
                # their own canonical category strings per spec §10.10
                # / §10.12. They MUST propagate to the invoke() caller
                # unwrapped — wrapping as NodeException would mask the
                # checkpoint-category surface the user is meant to
                # branch on. Notably proposal 0029's count-drift raise
                # surfaces here when it fires from inside FanOutNode.
                raise
            except Exception as e:
                # A raw exception (node-raised or middleware-raised) escaped
                # the chain unrecovered. Wrap as NodeException per §4.
                raise NodeException(node_name=current, cause=e, recoverable_state=state) from e
        finally:
            _reset_branch_name_chain(branch_chain_token)
            _reset_fan_out_index_chain(fan_out_chain_token)
            _reset_fan_out_index(fan_out_token)
            _reset_namespace_prefix(namespace_token)
            _reset_active_dispatch(dispatch_token)
            _reset_active_observers(observers_token)
        # Engine's canonical merge uses the ORIGINAL state per §2: "the
        # transformed state is passed to ``next``, NOT to the engine's
        # merge step." If middleware transformed state mid-chain, the
        # per-attempt completed events showed the transformed merge for
        # observability, but the state advancing the graph loop is built
        # from the original.
        merged_outer = _merge_partial(state, final_partial, self.reducers, current)
        # Spec §10.3: save fires once the canonical merge succeeds —
        # the LAST attempt's index is what gets recorded (retries
        # don't multiply saves). Per graph-engine §6 v0.16.1, the
        # recorded value is the wrapping retry MW's attempt counter
        # (which the inner-node events also reflected via the
        # ContextVar). ``deferred_info[0]`` captures that value at
        # the moment of the successful merge, sourced from
        # ``current_attempt_index()``. When middleware short-
        # circuited without invoking ``next()``, ``deferred_info[0]``
        # is None and the save records attempt_index=0.
        info = deferred_info[0]
        saved_attempt = info[0] if info is not None else 0
        await self._maybe_save_checkpoint(
            context,
            node_name=current,
            namespace=namespace,
            step=step,
            attempt_index=saved_attempt,
            post_state=merged_outer,
        )

        # Build the deferred-dispatch closure for the success-case
        # completed event. ``_invoke`` calls this after edge eval.
        if info is None:
            # Middleware short-circuited without invoking ``next`` —
            # no started/completed pair fired. Edge errors after this
            # node propagate silently per proposal-0012 + fixture-013
            # framing (preceding unit emitted no pair to share).
            return _StepResult(state=merged_outer, finalize_completed=_no_op_finalize)
        final_attempt_index, final_pre_state, final_merged = info

        def finalize_completed(edge_error: RuntimeGraphError | None) -> None:
            if edge_error is None:
                self._dispatch_completed(
                    context,
                    current,
                    namespace,
                    step,
                    final_pre_state,
                    post_state=final_merged,
                    attempt_index=final_attempt_index,
                )
            else:
                self._dispatch_completed(
                    context,
                    current,
                    namespace,
                    step,
                    final_pre_state,
                    error=edge_error,
                    attempt_index=final_attempt_index,
                )

        return _StepResult(state=merged_outer, finalize_completed=finalize_completed)

    async def _step_subgraph_node(
        self,
        node: SubgraphNode[StateT, State],
        current: str,
        state: StateT,
        context: _InvocationContext,
    ) -> _StepResult[StateT]:
        """Run one subgraph-as-node step through the parent's middleware chain.

        The parent's per-graph middleware plus
        any per-node middleware on the SubgraphNode wraps the subgraph
        dispatch as a single atomic call. The subgraph's INTERNAL nodes
        get their own middleware via the subgraph's own CompiledGraph;
        parent middleware does NOT cross the boundary.

        No started/completed events fire for the wrapper itself; the
        events come from the subgraph's internal node executions (per
        fixture 013).

        Edge errors AFTER a transparent subgraph wrapper propagate to
        the caller as ``RuntimeGraphError`` WITHOUT an associated
        completed event — the wrapper has no started/completed pair
        to share, and the "preceding node's pair" MUST is vacuous
        (not violated) when the preceding unit emitted
        no pair. The :class:`_StepResult` returned here uses
        :func:`_no_op_finalize` so the outer ``_invoke`` call to
        ``finalize_completed(edge_error)`` is a no-op.
        """

        async def innermost(s: Any) -> Mapping[str, Any]:
            try:
                return await node.run(s, context=context)
            except RuntimeGraphError:
                raise
            except Exception as e:
                raise NodeException(node_name=current, cause=e, recoverable_state=s) from e

        chain: ChainCall = compose_chain(
            list(self.middleware) + list(node.middleware),
            innermost,
        )
        # Same active-observers + calling-node scope as
        # ``_step_function_node`` — parent middleware running before
        # the descent should see the wrapper node's namespace +
        # fan_out_index for any LLM-provider hook emissions.
        # ``attempt_index`` defaults to 0 from the ContextVar; the
        # subgraph wrapper has no engine-managed attempt counter
        # (inner ``_step_function_node`` calls own their own).
        namespace = context.namespace_prefix + (current,)
        observers_token = _set_active_observers(context.full_observers())
        dispatch_token = _set_active_dispatch(lambda event: _dispatch(context, event))
        namespace_token = _set_namespace_prefix(namespace)
        fan_out_token = _set_fan_out_index(context.fan_out_index)
        # Per proposal 0045: drive per-depth chain ContextVars.
        fan_out_chain_token = _set_fan_out_index_chain(context.fan_out_index_chain)
        branch_chain_token = _set_branch_name_chain(context.branch_name_chain)

        try:
            try:
                final_partial = await chain(state)
            except RuntimeGraphError:
                raise
            except Exception as e:
                # Same wrap as _step_function_node: a raw exception escaping
                # the parent's middleware chain (e.g., a middleware bug or a
                # projection error) becomes NodeException tagged with the
                # SubgraphNode's wrapper name so §4 recoverable_state is
                # preserved.
                raise NodeException(node_name=current, cause=e, recoverable_state=state) from e
        finally:
            _reset_branch_name_chain(branch_chain_token)
            _reset_fan_out_index_chain(fan_out_chain_token)
            _reset_fan_out_index(fan_out_token)
            _reset_namespace_prefix(namespace_token)
            _reset_active_dispatch(dispatch_token)
            _reset_active_observers(observers_token)
        merged = _merge_partial(state, final_partial, self.reducers, current)
        return _StepResult(state=merged, finalize_completed=_no_op_finalize)

    async def _step_fan_out_node(
        self,
        node: FanOutNode[StateT, State],
        current: str,
        state: StateT,
        context: _InvocationContext,
    ) -> _StepResult[StateT]:
        """Run one fan-out-as-node step through the parent's middleware chain.

        The parent's per-graph + per-node
        middleware wraps the fan-out as a SINGLE dispatch — one started
        event before the fan-out begins, one completed event after all
        instances complete and fan-in is done. Per-instance events
        come from the inner subgraph executions; their pre_state /
        post_state shape is the inner subgraph's state, and they carry
        ``fan_out_index`` populated.

        Raw exceptions escaping the chain become NodeException.

        The fan-out's success-case completed event fires AFTER edge
        eval (mirrors
        ``_step_function_node``). Failure-path dispatches stay
        inline; the success-case is deferred via the returned
        :class:`_StepResult`.
        """
        step = context.take_step()
        namespace = context.namespace_prefix + (current,)
        # Same pattern as ``_step_function_node``: a mutable counter the
        # innermost closure reads-and-increments per attempt so retry
        # middleware wrapped at the parent level (per fixture 020)
        # produces correctly-indexed per-attempt events, and the save
        # records the final successful attempt's index rather than a
        # hardcoded 0.
        attempt_counter: list[int] = [0]

        # Resolve the fan-out config eagerly so the resolved values
        # ride on every fan-out node event (per spec proposal 0013,
        # v0.10.0: ``fan_out_config`` is populated on fan-out node
        # events including retried attempts). For ``items_field``
        # mode the count is ``len(parent_state.<items_field>)``; for
        # ``count`` mode it's ``_resolve_count``. ``_resolve_concurrency``
        # is pure regardless. Repeating these inside
        # ``FanOutNode.run_with_context`` is cheap and matches the
        # values surfaced here.
        # Lazy import: function-scope to avoid a module-top
        # textual cycle CodeQL flags. ``fan_out`` has a
        # TYPE_CHECKING back-reference to this module, so the
        # static-analyzer view of an importable cycle goes away
        # when the engine doesn't reach into ``fan_out`` at module
        # load time. Fires once per fan-out step.
        from .fan_out import _resolve_concurrency, _resolve_count  # noqa: PLC0415

        # Resolver failures (callable count/concurrency raising,
        # ``getattr`` on a malformed state, etc.) used to land inside
        # ``innermost``'s ``except Exception → NodeException`` block
        # below and produce a started/completed event pair via the
        # surrounding dispatches. Hoisting resolution out of
        # ``run_with_context`` for the eager ``FanOutEventConfig``
        # build moved them past that scope, so re-establish the
        # contract here: surface a started/completed pair with
        # ``fan_out_config=None`` (we never built one) and raise as
        # ``NodeException``.
        try:
            if node.config.items_field is not None:
                items_attr: Any = getattr(state, node.config.items_field, [])
                if not isinstance(items_attr, list):
                    raise NodeException(
                        node_name=current,
                        cause=TypeError(f"items_field {node.config.items_field!r} is not a list at runtime"),
                        recoverable_state=state,
                    )
                item_count = len(cast("list[Any]", items_attr))
            else:
                item_count = _resolve_count(current, node.config, state)
            concurrency_resolved: int | None = _resolve_concurrency(current, node.config, state)
            fan_out_event_config = FanOutEventConfig(
                item_count=item_count,
                concurrency=concurrency_resolved,
                error_policy=node.config.error_policy,
                parent_node_name=current,
            )
        except NodeException as resolution_error:
            self._dispatch_started(
                context,
                current,
                namespace,
                step,
                state,
                attempt_index=0,
                fan_out_config=None,
            )
            self._dispatch_completed(
                context,
                current,
                namespace,
                step,
                state,
                error=resolution_error,
                attempt_index=0,
                fan_out_config=None,
            )
            raise
        except Exception as resolution_error:
            wrapped = NodeException(
                node_name=current,
                cause=resolution_error,
                recoverable_state=state,
            )
            self._dispatch_started(
                context,
                current,
                namespace,
                step,
                state,
                attempt_index=0,
                fan_out_config=None,
            )
            self._dispatch_completed(
                context,
                current,
                namespace,
                step,
                state,
                error=wrapped,
                attempt_index=0,
                fan_out_config=None,
            )
            raise wrapped from resolution_error

        # Cell holding the FINAL successful attempt's
        # (attempt_index, pre_state, merged); see same comment in
        # ``_step_function_node``.
        deferred_info: list[tuple[int, StateT, StateT] | None] = [None]

        async def innermost(s: Any) -> Mapping[str, Any]:
            attempt_counter[0] += 1
            # Read from ContextVar — see ``_step_function_node``'s
            # ``innermost`` comment on the v0.16.1 attempt-index
            # propagation rule.
            attempt_index = current_attempt_index()

            self._dispatch_started(
                context,
                current,
                namespace,
                step,
                s,
                attempt_index=attempt_index,
                fan_out_config=fan_out_event_config,
            )
            # Same OTel attach pattern as ``_step_function_node``'s
            # ``innermost`` — splice the observer-published span
            # into the OTel context so logs emitted from inside
            # the fan-out node's own scope (middleware bodies,
            # the dispatch machinery) carry the right
            # trace_id/span_id. Per-instance bodies get their own
            # attach inside their ``_step_function_node``
            # innermost when the recursive invocation hits leaf
            # nodes. ``finally`` clears the ContextVar so a later
            # dispatch whose ``prepare_sync`` raises or early-
            # returns can't reveal this fan-out's span as a stale
            # value to the engine's read.
            otel_token = _attach_active_observer_span()
            try:
                try:
                    partial = await node.run_with_context(
                        s,
                        context,
                        pre_resolved_count=item_count,
                        pre_resolved_concurrency=(concurrency_resolved,),
                    )
                except RuntimeGraphError as e:
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        s,
                        error=e,
                        attempt_index=attempt_index,
                        fan_out_config=fan_out_event_config,
                    )
                    raise
                except CheckpointError as e:
                    # Spec proposal 0012's pairing contract requires
                    # every started event have a paired completed
                    # event. CheckpointError categories (notably
                    # proposal 0029's count-drift raise) are sibling-
                    # typed to RuntimeGraphError and propagate to the
                    # invoke() caller unwrapped so callers can branch
                    # on ``e.category``. To preserve pairing while
                    # keeping ``NodeEvent.error`` typed as
                    # ``RuntimeGraphError | None`` per spec §6, the
                    # completed event carries a ``NodeException``
                    # wrapper whose ``__cause__`` is the original
                    # CheckpointError. The bare ``raise`` re-raises
                    # the active exception (the CheckpointError, not
                    # the wrapper) so the caller still sees the
                    # checkpoint category. Mirrors the ``except
                    # Exception`` branch below structurally; the
                    # difference is what gets re-raised.
                    wrapped = NodeException(node_name=current, cause=e, recoverable_state=s)
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        s,
                        error=wrapped,
                        attempt_index=attempt_index,
                        fan_out_config=fan_out_event_config,
                    )
                    raise
                except Exception as e:
                    wrapped = NodeException(node_name=current, cause=e, recoverable_state=s)
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        s,
                        error=wrapped,
                        attempt_index=attempt_index,
                        fan_out_config=fan_out_event_config,
                    )
                    raise wrapped from e
            finally:
                _detach_active_observer_span(otel_token)
                _set_active_observer_span(None)

            try:
                merged = _merge_partial(s, partial, self.reducers, current)
            except (ReducerError, StateValidationError) as e:
                self._dispatch_completed(
                    context,
                    current,
                    namespace,
                    step,
                    s,
                    error=e,
                    attempt_index=attempt_index,
                    fan_out_config=fan_out_event_config,
                )
                raise

            # Defer the success-case completed dispatch per
            # proposal-0012; record the info for the outer scope.
            deferred_info[0] = (attempt_index, cast("StateT", s), cast("StateT", merged))
            return partial

        chain: ChainCall = compose_chain(
            list(self.middleware) + list(node.middleware),
            innermost,
        )

        # Same observability §3 / LLM-span hook contract as
        # _step_function_node: set the active observer set, calling
        # node identity, and dispatch scope around the chain
        # invocation so capability backends emitting from inside the
        # fan-out's parent dispatch (or any code running on its call
        # stack) can find them. ``fan_out_index`` here is the parent
        # context's view (the fan-out node from outside); per-instance
        # values get set when the inner subgraph descends with the
        # instance's index in its own context.
        observers_token = _set_active_observers(context.full_observers())
        dispatch_token = _set_active_dispatch(lambda event: _dispatch(context, event))
        namespace_token = _set_namespace_prefix(namespace)
        fan_out_token = _set_fan_out_index(context.fan_out_index)
        # Per proposal 0045: drive per-depth chain ContextVars.
        fan_out_chain_token = _set_fan_out_index_chain(context.fan_out_index_chain)
        branch_chain_token = _set_branch_name_chain(context.branch_name_chain)
        # Per spec §10.11 the ``fan_out_progress`` entry is "in-flight
        # only"; the fan-out's own completion save below is the last
        # point where the entry is needed (proposal 0009: that save
        # "also finalizes fan_out_progress to mark all instances
        # complete"). Pop the entry after the save fires, regardless of
        # whether the fan-out completed normally, short-circuited, or
        # raised, so subsequent saves in this invocation do not carry
        # stale fan-out progress and a retry middleware on the fan-out
        # node sees a fresh tracked state on the second attempt.
        fan_out_progress_key = (context.namespace_prefix, current)
        try:
            try:
                try:
                    final_partial = await chain(state)
                except RuntimeGraphError:
                    raise
                except CheckpointError:
                    # See the matching branch in ``_step_function_node``:
                    # checkpoint-category errors propagate unwrapped so
                    # callers can branch on ``e.category``. The
                    # proposal-0029 count-drift raise from
                    # ``FanOutNode.run_with_context`` surfaces through
                    # here on the resume path.
                    raise
                except Exception as e:
                    raise NodeException(node_name=current, cause=e, recoverable_state=state) from e
            finally:
                _reset_branch_name_chain(branch_chain_token)
                _reset_fan_out_index_chain(fan_out_chain_token)
                _reset_fan_out_index(fan_out_token)
                _reset_namespace_prefix(namespace_token)
                _reset_active_dispatch(dispatch_token)
                _reset_active_observers(observers_token)
            merged_outer = _merge_partial(state, final_partial, self.reducers, current)
            # Spec §10.3 + §10.7 + proposal 0009 §10.11: the fan-out's
            # own completion DOES save — one record once the fan-out as
            # a whole has finished and results have merged back. The
            # save also finalizes ``fan_out_progress`` (the projection
            # at the save site captures every tracked instance's
            # terminal state before the outer ``finally`` pops the
            # entry). Per graph-engine §6 v0.16.1: the saved
            # attempt_index reflects the wrapping retry MW's counter
            # (sourced from ``deferred_info[0]`` which captured
            # ``current_attempt_index()`` at the moment of the
            # successful merge). Short-circuit case (middleware
            # returned without invoking ``next``) records
            # attempt_index=0.
            info = deferred_info[0]
            saved_attempt = info[0] if info is not None else 0
            await self._maybe_save_checkpoint(
                context,
                node_name=current,
                namespace=namespace,
                step=step,
                attempt_index=saved_attempt,
                post_state=merged_outer,
            )

            if info is None:
                return _StepResult(state=merged_outer, finalize_completed=_no_op_finalize)
            final_attempt_index, final_pre_state, final_merged = info

            def finalize_completed(edge_error: RuntimeGraphError | None) -> None:
                if edge_error is None:
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        final_pre_state,
                        post_state=final_merged,
                        attempt_index=final_attempt_index,
                        fan_out_config=fan_out_event_config,
                    )
                else:
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        final_pre_state,
                        error=edge_error,
                        attempt_index=final_attempt_index,
                        fan_out_config=fan_out_event_config,
                    )

            return _StepResult(state=merged_outer, finalize_completed=finalize_completed)
        finally:
            context.fan_out_progress_state.pop(fan_out_progress_key, None)

    async def _step_parallel_branches_node(
        self,
        node: Any,  # ParallelBranchesNode[StateT] — lazy import keeps the
        # textual cycle off the module graph (``parallel_branches`` has a
        # TYPE_CHECKING back-reference to this module).
        current: str,
        state: StateT,
        context: _InvocationContext,
    ) -> _StepResult[StateT]:
        """Run one parallel-branches-as-node step through the parent's
        middleware chain.

        The parent's per-graph +
        per-node middleware wraps the parallel-branches dispatch
        as a SINGLE unit — one started event before dispatch
        begins, one completed event after all branches complete
        and fan-in is done. Per-branch internal events come from
        the branches' subgraph executions and carry ``branch_name``.

        Mirrors ``_step_fan_out_node`` minus the eager
        count/concurrency resolution (parallel branches has no
        callable resolvers — the branch set is static at compile
        time).
        """
        step = context.take_step()
        namespace = context.namespace_prefix + (current,)
        attempt_counter: list[int] = [0]
        deferred_info: list[tuple[int, StateT, StateT] | None] = [None]

        # Per proposal 0044 (observability §5.7, v0.36.0): the
        # resolved parallel-branches configuration is static at
        # compile time (no count / concurrency resolvers like fan-out
        # has), so build the event config once here and ship it on
        # both started + completed events. ``branch_names`` mirrors
        # the dispatch order ParallelBranchesNode uses internally
        # (insertion order of the ``branches`` dict per pipeline-
        # utilities §11.1).
        #
        # Python dicts preserve insertion order (PEP 468; guaranteed
        # since 3.7), and YAML / direct-dict-literal ``branches``
        # construction at the call site preserves the source order
        # through into the dict's keys().  Spec §11.1 ties branch
        # declaration order to dispatch order, so this tuple IS the
        # declaration order observers should see.
        branch_names: tuple[str, ...] = tuple(node.branches.keys())
        parallel_branches_event_config = ParallelBranchesEventConfig(
            branch_names=branch_names,
            branch_count=len(branch_names),
            error_policy=node.error_policy,
            parent_node_name=current,
        )

        async def innermost(s: Any) -> Mapping[str, Any]:
            attempt_counter[0] += 1
            # Read from ContextVar — see ``_step_function_node``'s
            # ``innermost`` for the v0.16.1 propagation rule.
            attempt_index = current_attempt_index()

            self._dispatch_started(
                context,
                current,
                namespace,
                step,
                s,
                attempt_index=attempt_index,
                parallel_branches_config=parallel_branches_event_config,
            )
            otel_token = _attach_active_observer_span()
            try:
                try:
                    partial = await node.run_with_context(s, context)
                except RuntimeGraphError as e:
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        s,
                        error=e,
                        attempt_index=attempt_index,
                        parallel_branches_config=parallel_branches_event_config,
                    )
                    raise
                except Exception as e:
                    wrapped = NodeException(node_name=current, cause=e, recoverable_state=s)
                    self._dispatch_completed(
                        context,
                        current,
                        namespace,
                        step,
                        s,
                        error=wrapped,
                        attempt_index=attempt_index,
                        parallel_branches_config=parallel_branches_event_config,
                    )
                    raise wrapped from e
            finally:
                _detach_active_observer_span(otel_token)
                _set_active_observer_span(None)

            try:
                merged = _merge_partial(s, partial, self.reducers, current)
            except (ReducerError, StateValidationError) as e:
                self._dispatch_completed(
                    context,
                    current,
                    namespace,
                    step,
                    s,
                    error=e,
                    attempt_index=attempt_index,
                    parallel_branches_config=parallel_branches_event_config,
                )
                raise

            deferred_info[0] = (attempt_index, cast("StateT", s), cast("StateT", merged))
            return partial

        chain: ChainCall = compose_chain(
            list(self.middleware) + list(node.middleware),
            innermost,
        )

        observers_token = _set_active_observers(context.full_observers())
        dispatch_token = _set_active_dispatch(lambda event: _dispatch(context, event))
        namespace_token = _set_namespace_prefix(namespace)
        fan_out_token = _set_fan_out_index(context.fan_out_index)
        # Per proposal 0045: drive per-depth chain ContextVars.
        fan_out_chain_token = _set_fan_out_index_chain(context.fan_out_index_chain)
        branch_chain_token = _set_branch_name_chain(context.branch_name_chain)
        try:
            try:
                final_partial = await chain(state)
            except RuntimeGraphError:
                raise
            except Exception as e:
                raise NodeException(node_name=current, cause=e, recoverable_state=state) from e
        finally:
            _reset_branch_name_chain(branch_chain_token)
            _reset_fan_out_index_chain(fan_out_chain_token)
            _reset_fan_out_index(fan_out_token)
            _reset_namespace_prefix(namespace_token)
            _reset_active_dispatch(dispatch_token)
            _reset_active_observers(observers_token)
        merged_outer = _merge_partial(state, final_partial, self.reducers, current)
        info = deferred_info[0]
        saved_attempt = info[0] if info is not None else 0
        await self._maybe_save_checkpoint(
            context,
            node_name=current,
            namespace=namespace,
            step=step,
            attempt_index=saved_attempt,
            post_state=merged_outer,
        )

        if info is None:
            return _StepResult(state=merged_outer, finalize_completed=_no_op_finalize)
        final_attempt_index, final_pre_state, final_merged = info

        def finalize_completed(edge_error: RuntimeGraphError | None) -> None:
            if edge_error is None:
                self._dispatch_completed(
                    context,
                    current,
                    namespace,
                    step,
                    final_pre_state,
                    post_state=final_merged,
                    attempt_index=final_attempt_index,
                    parallel_branches_config=parallel_branches_event_config,
                )
            else:
                self._dispatch_completed(
                    context,
                    current,
                    namespace,
                    step,
                    final_pre_state,
                    error=edge_error,
                    attempt_index=final_attempt_index,
                    parallel_branches_config=parallel_branches_event_config,
                )

        return _StepResult(state=merged_outer, finalize_completed=finalize_completed)

    @staticmethod
    def _dispatch_started(
        context: _InvocationContext,
        current: str,
        namespace: tuple[str, ...],
        step: int,
        pre_state: State,
        *,
        attempt_index: int = 0,
        fan_out_config: FanOutEventConfig | None = None,
        parallel_branches_config: ParallelBranchesEventConfig | None = None,
    ) -> None:
        # Per graph-engine §6 + pipeline-utilities §11: read the
        # active branch_name (set by ParallelBranchesNode inside
        # each branch's task ``copy_context``) and stamp it on
        # every event emitted from inside the branch. Outside any
        # branch, current_branch_name() returns None.
        from openarmature.observability.correlation import current_branch_name  # noqa: PLC0415

        _dispatch(
            context,
            NodeEvent(
                node_name=current,
                namespace=namespace,
                step=step,
                phase="started",
                pre_state=pre_state,
                post_state=None,
                error=None,
                parent_states=context.parent_states_prefix,
                attempt_index=attempt_index,
                fan_out_index=context.fan_out_index,
                fan_out_config=fan_out_config,
                parallel_branches_config=parallel_branches_config,
                branch_name=current_branch_name(),
                # Per proposal 0045: per-depth lineage chains so
                # observers can identify the augmenter's call-stack
                # ancestor path under nested dispatch.
                fan_out_index_chain=context.fan_out_index_chain,
                branch_name_chain=context.branch_name_chain,
                subgraph_identities=context.subgraph_identities,
                caller_invocation_metadata=current_invocation_metadata(),
            ),
        )

    @staticmethod
    def _dispatch_completed(
        context: _InvocationContext,
        current: str,
        namespace: tuple[str, ...],
        step: int,
        pre_state: State,
        *,
        post_state: State | None = None,
        error: RuntimeGraphError | None = None,
        attempt_index: int = 0,
        fan_out_config: FanOutEventConfig | None = None,
        parallel_branches_config: ParallelBranchesEventConfig | None = None,
    ) -> None:
        from openarmature.observability.correlation import current_branch_name  # noqa: PLC0415

        _dispatch(
            context,
            NodeEvent(
                node_name=current,
                namespace=namespace,
                step=step,
                phase="completed",
                pre_state=pre_state,
                post_state=post_state,
                error=error,
                parent_states=context.parent_states_prefix,
                attempt_index=attempt_index,
                fan_out_index=context.fan_out_index,
                fan_out_config=fan_out_config,
                parallel_branches_config=parallel_branches_config,
                branch_name=current_branch_name(),
                # Per proposal 0045: per-depth lineage chains.
                fan_out_index_chain=context.fan_out_index_chain,
                branch_name_chain=context.branch_name_chain,
                subgraph_identities=context.subgraph_identities,
                caller_invocation_metadata=current_invocation_metadata(),
            ),
        )

    # Instance method (not @staticmethod) so the save-time
    # schema_version read goes through ``self.state_cls`` — matches
    # the resume-side check, per spec §10.2's "framework reads
    # schema_version from the state definition at save time"
    # wording. Reading from ``type(post_state)`` would let a State
    # subclass instance shadow the declared graph schema and
    # trigger spurious migrations on resume.
    async def _maybe_save_checkpoint(
        self,
        context: _InvocationContext,
        *,
        node_name: str,
        namespace: tuple[str, ...],
        step: int,
        attempt_index: int,
        post_state: Any,
    ) -> None:
        """Fire a checkpoint save for the just-completed node, if a
        backend is registered.

        Save policy:

        - Save fires for outermost-graph nodes, subgraph-internal
          nodes, fan-out instance internal nodes, AND the fan-out
          node's own completion (the parent dispatch).
        - When the save fires from inside a fan-out instance
          (``context.fan_out_index is not None``), the inner node's
          position is recorded against the per-instance state on the
          shared ``fan_out_progress_state`` rather than the outer
          ``completed_positions`` list. The saved record's
          ``fan_out_progress`` field projects this shared dict so
          all concurrent instances' snapshots are captured atomically.

        Atomicity contract: the save-call site below
        completes the "produce contribution + record into accumulator
        + save" sequence the spec mandates. ``FanOutNode.run_with_context``
        flips an instance's state to ``completed`` and stashes its
        ``result`` BEFORE invoking the save that durably records the
        transition. A crash between that state mutation and the save
        below leaves the in-memory dict updated but the persisted
        record showing ``in_flight``, so resume re-runs the instance
        and the append/last_write_wins/merge reducer's exactly-once
        guarantee holds.

        Save also enumerates ALL concurrent fan-out instances when
        building ``fan_out_progress`` (not just the one whose
        ``completed`` event triggered this save) — the per-instance
        snapshot is consistent across siblings, captured when a
        sibling instance's ``completed`` event triggers a save during
        this instance's execution.

        After ``Checkpointer.save`` returns, dispatch a
        ``checkpoint_saved`` observer event so observability backends
        can surface saves as spans.

        Save failures raise ``CheckpointSaveFailed`` to the caller of
        ``invoke()`` immediately; saves are NOT retried by the engine.
        """
        checkpointer = context.checkpointer
        if checkpointer is None:
            return
        # Per spec §10.2: NodePosition.namespace is the containing-
        # graph chain (outermost first), NOT including the node's
        # own name — distinct from NodeEvent.namespace which
        # includes it. The two are related by
        # NodeEvent.namespace == NodePosition.namespace +
        # (NodePosition.node_name,).
        #
        # Inner-position scoping (per §10.11.1, in-flight observability
        # rules): a position from inside a fan-out instance is scoped
        # to that instance's inner subgraph execution, NOT the outer
        # graph. It accumulates on the per-instance state's
        # ``completed_inner_positions`` list rather than the outer
        # ``completed_positions`` list. The outer list keeps the outer
        # graph's positions plus the fan-out node's own completion
        # position (added by ``_step_fan_out_node`` after fan-in).
        position = NodePosition(
            namespace=context.namespace_prefix,
            node_name=node_name,
            step=step,
            attempt_index=attempt_index,
            fan_out_index=context.fan_out_index,
        )
        if context.fan_out_index is not None:
            # Locate the per-instance state for the innermost active
            # fan-out (the one this node is running inside). The
            # innermost fan-out's key has the longest namespace; the
            # context's namespace_prefix at this depth is exactly that
            # fan-out's full namespace prefix (namespace + name), so
            # we walk the prefix back to find the matching key.
            instance_state = _find_innermost_fan_out_instance_state(context)
            if instance_state is not None:
                instance_state.completed_inner_positions.append(position)
        else:
            context.completed_positions.append(position)
        # Project the shared mutable per-fan-out tracking dict into the
        # frozen ``FanOutProgress`` shape on the record. Per §10.11:
        # enumerate every fan-out entry the engine has registered, not
        # just the innermost one — concurrent fan-outs (nested or
        # parallel) all contribute their state to the same save.
        # Deterministic order: sort by (namespace, name) so two saves
        # with identical state serialize identically (relevant for
        # backends that hash records).
        fan_out_progress = _project_fan_out_progress(context.fan_out_progress_state)
        record = CheckpointRecord(
            invocation_id=context.invocation_id,
            correlation_id=context.correlation_id,
            state=post_state,
            completed_positions=tuple(context.completed_positions),
            parent_states=context.parent_states_prefix,
            # ``time.time()`` is wall-clock seconds, not strictly
            # monotonic (NTP adjustments can regress it). Per spec
            # §10.2 ``last_saved_at`` is "implementation-defined
            # precision; SHOULD be monotonic per invocation" — we
            # accept the wall-clock trade-off because save records
            # are typically inspected hours/days later, where the
            # absolute timestamp is more useful than a monotonic
            # delta. Two saves within the same μs would tie; the
            # ``step`` field on each NodePosition is the canonical
            # within-invocation order.
            last_saved_at=time.time(),
            # Per spec §10.2 (proposal 0028, supersedes proposal 0014's
            # ``type(state)`` framing): ``schema_version`` is sourced
            # from the OUTERMOST declared state class threaded as
            # ``context.state_cls``, NOT from ``self.state_cls`` on
            # the current graph. The distinction matters at subgraph
            # save sites: subgraphs have their own ``state_cls`` (the
            # subgraph's state class), but the saved record represents
            # the WHOLE invocation tree and the canonical version is
            # the outer graph's. Reading from ``context.state_cls``
            # gives every save site within an invocation the same
            # value, aligning subgraph-internal saves with outer
            # dispatch saves and fan-out instance internal saves.
            # Empty-string sentinel when the user hasn't declared one
            # — those records are not migration-eligible until they
            # declare a non-empty version (per §10.2).
            schema_version=cast("str", getattr(context.state_cls, "schema_version", "")),
            fan_out_progress=fan_out_progress,
        )
        # Per §10.11.4: batching applies ONLY to fan-out instance
        # internal saves. Outer-graph + subgraph-internal +
        # fan-out-node-completion saves remain synchronous.
        # ``checkpointer.save`` is invoked via the batching helper
        # which falls back to direct ``save`` for non-fan-out-internal
        # events even on a batching-enabled backend.
        try:
            if context.fan_out_index is not None:
                await _save_fan_out_internal(checkpointer, context.invocation_id, record)
            else:
                await checkpointer.save(context.invocation_id, record)
        except Exception as exc:
            raise CheckpointSaveFailed(context.invocation_id, exc) from exc
        # §10.8: dispatch a ``checkpoint_saved`` observer event so
        # observability mappings can surface saves as spans. Default
        # observer subscriptions don't include this phase, so legacy
        # observers don't see it without explicit opt-in.
        #
        # Convention for ``checkpoint_saved`` events: ``pre_state``
        # carries the SAVED state (the post-merge state at the moment
        # the save fired). ``post_state`` is None — there's no
        # before/after distinction for a save like there is for a
        # node attempt. The field is repurposed because a save
        # event represents "the state was persisted" rather than
        # "the state transitioned." Phase 6 OTel mapping reads
        # ``pre_state`` as the save's state.
        _dispatch(
            context,
            NodeEvent(
                node_name=node_name,
                namespace=namespace,
                step=step,
                phase="checkpoint_saved",
                pre_state=post_state,
                post_state=None,
                error=None,
                parent_states=context.parent_states_prefix,
                attempt_index=attempt_index,
                fan_out_index=None,
                subgraph_identities=context.subgraph_identities,
                caller_invocation_metadata=current_invocation_metadata(),
            ),
        )
