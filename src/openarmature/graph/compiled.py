"""Compiled graph + execute loop.

Per spec §3 Execution model: execution begins at the entry node; each step
runs a node, merges its partial update via per-field reducers, then evaluates
the outgoing edge against the post-update state to choose the next node (or
END to halt).

Per spec §4 Error semantics: node, edge, reducer, and routing errors carry
recoverable state; state validation errors do not.

Per spec v0.6.0 §6 Observer hooks: each node attempt produces a
started/completed event PAIR. The engine dispatches the started event
before invoking the wrapped node function and the completed event after
the reducer merge succeeds (with `post_state` populated) or after the
node, reducer, or state validation fails (with `error` populated).
Routing errors do NOT produce their own event pair — they arise after
the preceding node's completed event has already been dispatched.

`CompiledGraph[StateT]` and `_merge_partial[StateT]` carry the concrete state
subclass through to `invoke()`'s return type, so consumers don't need
`cast(MyState, ...)` at the call site.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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
    CheckpointNotFound,
    CheckpointRecordInvalid,
    CheckpointSaveFailed,
)
from openarmature.checkpoint.protocol import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpointer,
    CheckpointRecord,
    NodePosition,
)
from openarmature.observability.correlation import (
    _reset_active_dispatch,
    _reset_active_observers,
    _reset_attempt_index,
    _reset_correlation_id,
    _reset_fan_out_index,
    _reset_invocation_id,
    _reset_namespace_prefix,
    _set_active_dispatch,
    _set_active_observers,
    _set_attempt_index,
    _set_correlation_id,
    _set_fan_out_index,
    _set_invocation_id,
    _set_namespace_prefix,
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
from .events import FanOutEventConfig, NodeEvent
from .middleware import ChainCall, Middleware, compose_chain
from .nodes import Node
from .observer import (
    _DRAIN_SENTINEL,
    Observer,
    RemoveHandle,
    SubscribedObserver,
    _coerce_subscribed,
    _dispatch,
    _InvocationContext,
    _QueuedItem,
    deliver_loop,
)
from .reducers import Reducer
from .state import State
from .subgraph import SubgraphNode


def _merge_partial[StateT: State](
    prior: StateT,
    partial: Mapping[str, Any],
    reducers: Mapping[str, Reducer],
    producing_node: str,
) -> StateT:
    """Apply per-field reducers to merge a node's partial update into prior state.

    Re-validates the resulting state against the schema (per spec §2 SHOULD
    validate at node boundaries). Wraps reducer failures as `ReducerError` and
    schema failures as `StateValidationError`.
    """

    new_values = prior.model_dump()
    for field_name, partial_value in partial.items():
        reducer = reducers.get(field_name)
        if reducer is None:
            # Unknown field — surface as a schema validation failure below.
            new_values[field_name] = partial_value
            continue
        try:
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
    ``_step_fan_out_node``) under the proposal-0012 v0.9.0 swap.

    Spec graph-engine §3 step 3 (revised) requires the
    ``completed`` event for the just-completed node to fire AFTER
    edge evaluation completes — so that edge-resolution failures
    (``routing_error``, ``edge_exception``) land on the preceding
    node's completed event with ``error`` populated, sharing the
    started/completed pair rather than producing a separate event
    pair (§6 revised).

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
    propagate silently per proposal 0012's "preceding unit's
    pair" framing applied to a unit that never had one. Same for
    middleware that short-circuits without invoking ``next``.
    """

    state: StateT
    finalize_completed: Callable[[RuntimeGraphError | None], None]


def _no_op_finalize(_edge_error: RuntimeGraphError | None) -> None:
    """Default ``finalize_completed`` for cases where the step
    didn't dispatch a started/completed pair — subgraph wrappers
    (transparent per fixture 013) and middleware that short-
    circuits without invoking ``next``. Edge errors propagate
    silently per proposal 0012 + fixture 013."""


@dataclass(frozen=True)
class CompiledGraph[StateT: State]:
    """An immutable, executable graph produced by `GraphBuilder.compile()`.

    The compile-time topology (state class, entry, nodes, edges, reducers) is
    immutable. Two mutable lists ride alongside for observer plumbing —
    `_attached_observers` and `_active_workers` — neither of which affect the
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
    # `set` (not list) so a per-task `add_done_callback(self._active_workers.discard)`
    # auto-removes completed workers — long-running services that never call
    # drain() don't accumulate completed Task references indefinitely.
    _active_workers: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])
    # Single-element list so the frozen-dataclass binding is stable but
    # the user can swap the registered Checkpointer via
    # ``attach_checkpointer``. ``None`` when no backend is registered.
    _checkpointer_slot: list[Checkpointer | None] = field(default_factory=lambda: [None])

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

        Per spec v0.6.0 §6: graph-attached observers fire on every invocation
        of this graph until removed — including when this graph runs as a
        subgraph inside a parent. Returns a `RemoveHandle` whose `.remove()`
        method detaches the observer; idempotent.

        `phases` selects the phase strings (`"started"`, `"completed"`) the
        observer subscribes to; default is both. An empty `phases` set
        raises `ValueError` at registration time.

        Per spec: changes to the registered set during a graph run do NOT
        take effect until the next invocation. The set of observers
        delivering events for an in-flight invocation is fixed at the point
        the invocation begins.
        """
        subscribed = _coerce_subscribed(observer, phases=phases)
        self._attached_observers.append(subscribed)
        return RemoveHandle(_observers=self._attached_observers, _observer=subscribed)

    # ------------------------------------------------------------------
    # Checkpointer registration (spec pipeline-utilities §10.1.1)
    # ------------------------------------------------------------------

    def attach_checkpointer(self, checkpointer: Checkpointer | None) -> None:
        """Register a Checkpointer for this graph (spec §10.1.1).

        Pass ``None`` to clear a previously-registered backend. Without
        a registered Checkpointer the engine never calls ``save()`` and
        ``invoke(resume_invocation=...)`` raises
        ``checkpoint_not_found`` — the default-off behavior matches the
        broader OA pattern of "the contract is normative; the
        activation is an explicit choice."

        At most one Checkpointer per graph (§10.1.1). Calling
        ``attach_checkpointer`` again replaces the previously-registered
        one; multi-backend fan-out is the user's responsibility (wrap
        two underlying Checkpointers behind a custom protocol-conforming
        implementation if needed).
        """
        self._checkpointer_slot[0] = checkpointer

    @property
    def checkpointer(self) -> Checkpointer | None:
        """Currently-registered Checkpointer, or ``None``."""
        return self._checkpointer_slot[0]

    async def drain(self) -> None:
        """Await delivery of every observer event produced by prior
        invocations of this graph.

        Per spec v0.6.0 §6: callers running in short-lived processes (scripts,
        serverless functions, CLIs) MUST use drain to avoid losing observer
        events that were dispatched but not yet delivered.

        Only events dispatched before this call are awaited; events from
        invocations started concurrently with drain may or may not be
        included. Subgraph events from active invocations are part of the
        parent invocation's worker and are covered automatically.

        **Unbounded by design.** Drain blocks until every queued event has
        been delivered to every subscribed observer. A slow, hung, or
        misbehaving observer can therefore hold drain — and the calling
        process — indefinitely. If you need a bounded wait, wrap the call
        in `asyncio.wait_for` and accept that events still queued when the
        deadline elapses will not be delivered::

            await asyncio.wait_for(compiled.drain(), timeout=5.0)
        """
        if not self._active_workers:
            return
        # Snapshot the set: each worker's done-callback removes itself
        # from `_active_workers`, so iterating it directly while gather
        # awaits would mutate during iteration.
        await asyncio.gather(*list(self._active_workers), return_exceptions=True)

    # ------------------------------------------------------------------
    # Public invocation
    # ------------------------------------------------------------------

    async def invoke(
        self,
        initial_state: StateT,
        observers: Iterable[Observer | SubscribedObserver] | None = None,
        *,
        correlation_id: str | None = None,
        resume_invocation: str | None = None,
    ) -> StateT:
        """Run the graph from `initial_state` to END and return the final state.

        Optional `observers` are invocation-scoped — they fire only for this
        run, after all graph-attached observers (including subgraph-attached
        ones for events originating in subgraphs) per spec v0.6.0 §6.

        Each entry in `observers` may be either a bare `Observer` callable
        (subscribes to both phases) or a `SubscribedObserver` wrapping an
        observer with an explicit `phases` set.

        Per spec v0.6.0 §6: this method returns as soon as the graph
        execution loop completes, regardless of whether the observer
        delivery queue has finished processing every dispatched event. Use
        `await compiled.drain()` if you need delivery-completion guarantees.

        **Checkpointing (pipeline-utilities §10):**

        - ``correlation_id`` is the per-invocation cross-backend join
          key (see observability §3 in spec v0.7+). Caller-supplied or
          auto-generated UUIDv4 when absent. Preserved unchanged across
          ``resume_invocation``.
        - ``resume_invocation`` names a prior invocation_id to resume
          from. Requires a registered Checkpointer; raises
          ``CheckpointNotFound`` when the backend has no record for
          the supplied id, ``CheckpointRecordInvalid`` when the
          loaded record's schema is incompatible. Resume mints a NEW
          ``invocation_id`` per §10.4 — each attempt is its own
          invocation in the observability sense; the
          ``correlation_id`` is the cross-attempt join key.
        - **Save-failure policy.** This implementation raises
          ``CheckpointSaveFailed`` to the caller of ``invoke()``
          immediately when ``Checkpointer.save`` raises; saves are
          NOT retried by the engine. Wrap the Checkpointer in your
          own retry logic if transient backend failures should be
          reattempted.

        Raises one of the runtime error categories from spec §4 on failure.
        """

        invocation_scoped = tuple(_coerce_subscribed(o) for o in (observers or ()))
        queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()

        # Resolve the resume path BEFORE building the context so we can
        # restore the correlation_id from the saved record (per §10.4
        # step 3) and pre-populate the skip-set + completed_positions.
        starting_state: StateT = initial_state
        resolved_correlation_id = correlation_id or str(uuid.uuid4())
        invocation_id = str(uuid.uuid4())
        resume_skip_set: frozenset[tuple[str, ...]] = frozenset()
        completed_positions: list[NodePosition] = []
        pending_resume_states: dict[int, Any] = {}
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
            if record.schema_version != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointRecordInvalid(
                    resume_invocation,
                    f"persisted schema_version={record.schema_version!r} "
                    f"does not match current {CHECKPOINT_SCHEMA_VERSION!r}",
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
        worker = asyncio.create_task(deliver_loop(queue))
        self._active_workers.add(worker)
        # Auto-prune: when the worker completes (after the sentinel is
        # processed), remove it from the active set so long-running
        # services don't leak Task references between drain() calls.
        worker.add_done_callback(self._active_workers.discard)
        try:
            return await self._invoke(starting_state, context)
        finally:
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

            if isinstance(node, FanOutNode):
                # Fan-out nodes are recognized as a distinct node type
                # per pipeline-utilities §9. Dispatched through
                # ``_step_fan_out_node`` which wraps the whole fan-out
                # as one parent dispatch (per §9.6) — instance-level
                # concurrency lives inside the FanOutNode itself.
                fn_node = cast("FanOutNode[StateT, State]", node)
                step_result = await self._step_fan_out_node(fn_node, current, state, context)
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

        Per pipeline-utilities §3, the runtime chain composes:

            [per_graph...] -> [per_node...] -> innermost

        where ``innermost`` is the per-attempt dispatch wrapper around
        ``node.run`` + reducer merge + observer event dispatch. Each call
        to ``innermost`` is one attempt; middleware that calls ``next``
        repeatedly (e.g., retry) produces multiple attempts and therefore
        multiple started/completed event pairs from the engine, each
        tagged with an incrementing ``attempt_index`` (graph-engine §6).

        Per proposal-0012 v0.9.0: the success-case ``completed`` event
        for the FINAL successful attempt fires AFTER edge eval, not
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
            attempt_index = attempt_counter[0]
            attempt_counter[0] += 1

            # Calling-node identity for capability backends emitting
            # from inside this attempt's scope (e.g., LLM provider's
            # span hook). Per-attempt scope so retry middleware that
            # re-enters innermost bumps the visible attempt_index.
            attempt_token = _set_attempt_index(attempt_index)
            try:
                self._dispatch_started(context, current, namespace, step, s, attempt_index=attempt_index)

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
            finally:
                _reset_attempt_index(attempt_token)

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
        try:
            try:
                final_partial = await chain(state)
            except RuntimeGraphError:
                raise
            except Exception as e:
                # A raw exception (node-raised or middleware-raised) escaped
                # the chain unrecovered. Wrap as NodeException per §4.
                raise NodeException(node_name=current, cause=e, recoverable_state=state) from e
        finally:
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
        # don't multiply saves). attempt_counter[0] is one past the
        # final attempt; ``max(0, ... - 1)`` covers the
        # short-circuit case where middleware returns a partial
        # without ever invoking ``next()`` (counter stays at 0,
        # subtracting 1 would yield an invalid -1).
        await self._maybe_save_checkpoint(
            context,
            node_name=current,
            namespace=namespace,
            step=step,
            attempt_index=max(0, attempt_counter[0] - 1),
            post_state=merged_outer,
        )

        # Build the deferred-dispatch closure for the success-case
        # completed event. ``_invoke`` calls this after edge eval.
        info = deferred_info[0]
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

        Per pipeline-utilities §4: the parent's per-graph middleware plus
        any per-node middleware on the SubgraphNode wraps the subgraph
        dispatch as a single atomic call. The subgraph's INTERNAL nodes
        get their own middleware via the subgraph's own CompiledGraph;
        parent middleware does NOT cross the boundary.

        No started/completed events fire for the wrapper itself; the
        events come from the subgraph's internal node executions (per
        fixture 013).

        Per proposal-0012 v0.9.0 + spec coordination: edge errors
        AFTER a transparent subgraph wrapper propagate to the caller
        as ``RuntimeGraphError`` per §4 WITHOUT an associated
        completed event — the wrapper has no started/completed pair
        to share, and proposal 0012's "preceding node's pair" MUST
        is vacuous (not violated) when the preceding unit emitted
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

        Per pipeline-utilities §9.6: the parent's per-graph + per-node
        middleware wraps the fan-out as a SINGLE dispatch — one started
        event before the fan-out begins, one completed event after all
        instances complete and fan-in is done. Per-instance events
        come from the inner subgraph executions; their pre_state /
        post_state shape is the inner subgraph's state, and they carry
        ``fan_out_index`` populated.

        Raw exceptions escaping the chain become NodeException per §4.

        Per proposal-0012 v0.9.0: the fan-out's success-case
        completed event fires AFTER edge eval (mirrors
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

        if node.config.items_field is not None:
            items_attr: Any = getattr(state, node.config.items_field, [])
            item_count = len(cast("list[Any]", items_attr)) if isinstance(items_attr, list) else 0
        else:
            item_count = _resolve_count(current, node.config, state)
        concurrency_resolved: int | None = _resolve_concurrency(current, node.config, state)
        fan_out_event_config = FanOutEventConfig(
            item_count=item_count,
            concurrency=concurrency_resolved,
            error_policy=node.config.error_policy,
            parent_node_name=current,
        )

        # Cell holding the FINAL successful attempt's
        # (attempt_index, pre_state, merged); see same comment in
        # ``_step_function_node``.
        deferred_info: list[tuple[int, StateT, StateT] | None] = [None]

        async def innermost(s: Any) -> Mapping[str, Any]:
            attempt_index = attempt_counter[0]
            attempt_counter[0] += 1
            attempt_token = _set_attempt_index(attempt_index)
            try:
                self._dispatch_started(
                    context,
                    current,
                    namespace,
                    step,
                    s,
                    attempt_index=attempt_index,
                    fan_out_config=fan_out_event_config,
                )
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
            finally:
                _reset_attempt_index(attempt_token)

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
        try:
            try:
                final_partial = await chain(state)
            except RuntimeGraphError:
                raise
            except Exception as e:
                raise NodeException(node_name=current, cause=e, recoverable_state=state) from e
        finally:
            _reset_fan_out_index(fan_out_token)
            _reset_namespace_prefix(namespace_token)
            _reset_active_dispatch(dispatch_token)
            _reset_active_observers(observers_token)
        merged_outer = _merge_partial(state, final_partial, self.reducers, current)
        # Spec §10.3 + §10.7: the fan-out's own completion DOES save —
        # one record once the fan-out as a whole has finished and
        # results have merged back. Per-instance internal saves are
        # gated off by the fan-out instance descent setting
        # ``checkpointer=None`` on the inner context. ``max(0, ...)``
        # guards against the short-circuit case (middleware returns a
        # partial without ever invoking ``next()``).
        await self._maybe_save_checkpoint(
            context,
            node_name=current,
            namespace=namespace,
            step=step,
            attempt_index=max(0, attempt_counter[0] - 1),
            post_state=merged_outer,
        )

        info = deferred_info[0]
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
    ) -> None:
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
    ) -> None:
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
            ),
        )

    @staticmethod
    async def _maybe_save_checkpoint(
        context: _InvocationContext,
        *,
        node_name: str,
        namespace: tuple[str, ...],
        step: int,
        attempt_index: int,
        post_state: Any,
    ) -> None:
        """Fire a checkpoint save for the just-completed node, if a
        backend is registered and we're not inside a fan-out instance.

        Per spec pipeline-utilities §10.3:

        - Save fires for outermost-graph nodes, subgraph-internal
          nodes, AND the fan-out node's own completion (the parent
          dispatch). All three have ``fan_out_index is None`` from
          the context's perspective.
        - Save does NOT fire for events from inside a fan-out
          instance. The atomic-restart contract (§10.7) means
          per-instance progress isn't recoverable in v1, so saving
          inner-instance state is dead weight.

        After ``Checkpointer.save`` returns, dispatch a
        ``checkpoint_saved`` observer event (per §10.8 SHOULD-level
        guidance) so observability backends — wired in Phase 6 — can
        surface saves as spans.

        Save failures raise ``CheckpointSaveFailed`` to the caller of
        ``invoke()`` immediately; saves are NOT retried by the engine.
        """
        checkpointer = context.checkpointer
        if checkpointer is None:
            return
        if context.fan_out_index is not None:
            return
        # Per spec §10.2: NodePosition.namespace is the containing-
        # graph chain (outermost first), NOT including the node's
        # own name — distinct from NodeEvent.namespace which
        # includes it. The two are related by
        # NodeEvent.namespace == NodePosition.namespace +
        # (NodePosition.node_name,).
        position = NodePosition(
            namespace=context.namespace_prefix,
            node_name=node_name,
            step=step,
            attempt_index=attempt_index,
            fan_out_index=None,
        )
        context.completed_positions.append(position)
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
            schema_version=CHECKPOINT_SCHEMA_VERSION,
        )
        try:
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
            ),
        )
