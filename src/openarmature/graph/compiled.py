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

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .edges import END, ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    EdgeException,
    NodeException,
    ReducerError,
    RoutingError,
    RuntimeGraphError,
    StateValidationError,
)
from .events import NodeEvent
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
    # Observer plumbing — see attach_observer/drain. Mutable on a frozen
    # dataclass: the list reference is fixed but its contents change.
    # Parameterized factories so pyright infers the element types.
    _attached_observers: list[SubscribedObserver] = field(default_factory=list[SubscribedObserver])
    # `set` (not list) so a per-task `add_done_callback(self._active_workers.discard)`
    # auto-removes completed workers — long-running services that never call
    # drain() don't accumulate completed Task references indefinitely.
    _active_workers: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])

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

    async def drain(self) -> None:
        """Await delivery of every observer event produced by prior
        invocations of this graph.

        Per spec v0.3.0 §6: callers running in short-lived processes (scripts,
        serverless functions, CLIs) MUST use drain to avoid losing observer
        events that were dispatched but not yet delivered.

        Only events dispatched before this call are awaited; events from
        invocations started concurrently with drain may or may not be
        included. Subgraph events from active invocations are part of the
        parent invocation's worker and are covered automatically.
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

        Raises one of the runtime error categories from spec §4 on failure.
        """

        invocation_scoped = tuple(_coerce_subscribed(o) for o in (observers or ()))
        queue: asyncio.Queue[_QueuedItem | None] = asyncio.Queue()
        context = _InvocationContext(
            queue=queue,
            graph_attached=tuple(self._attached_observers),
            invocation_scoped=invocation_scoped,
        )
        worker = asyncio.create_task(deliver_loop(queue))
        self._active_workers.add(worker)
        # Auto-prune: when the worker completes (after the sentinel is
        # processed), remove it from the active set so long-running
        # services don't leak Task references between drain() calls.
        worker.add_done_callback(self._active_workers.discard)
        try:
            return await self._invoke(initial_state, context)
        finally:
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

            if isinstance(node, SubgraphNode):
                # Subgraph wrappers are transparent to the observer protocol
                # (per fixture 013): no event is dispatched for the wrapper
                # itself, the step counter does not advance for it, and any
                # `RuntimeGraphError` bubbling up from the subgraph's
                # _invoke is already wrapped with the inner node's identity
                # — pass it through. Other exceptions (projection errors,
                # subgraph state-class init errors) escape the spec §4
                # categories, so we wrap them as NodeException tagged with
                # the wrapper's name.
                try:
                    partial = await node.run(state, context=context)
                except RuntimeGraphError:
                    raise
                except Exception as e:
                    raise NodeException(node_name=current, cause=e, recoverable_state=state) from e
                state = _merge_partial(state, partial, self.reducers, current)
            else:
                state = await self._step_function_node(node, current, state, context)

            edge = self.edges[current]
            if isinstance(edge, StaticEdge):
                target: str | EndSentinel = edge.target
            else:
                try:
                    target = edge.fn(state)
                except Exception as e:
                    raise EdgeException(source_node=current, cause=e, recoverable_state=state) from e

            if target is END:
                return state

            if not isinstance(target, str) or target not in self.nodes:
                raise RoutingError(source_node=current, returned=target, recoverable_state=state)

            current = target

    async def _step_function_node(
        self,
        node: Node[StateT],
        current: str,
        state: StateT,
        context: _InvocationContext,
    ) -> StateT:
        """Run one function-node step: take a step, dispatch started, run,
        merge, dispatch completed.

        Per spec v0.6.0 §6: each attempt produces a started/completed pair.
        Both events share the same `step`. The completed event carries
        `post_state` on success, or `error` on failure (one of run, reducer,
        or state-validation). The completed event is dispatched before the
        failure propagates.
        """
        step = context.take_step()
        namespace = context.namespace_prefix + (current,)
        pre_state = state

        self._dispatch_started(context, current, namespace, step, pre_state)

        try:
            partial = await node.run(state)
        except Exception as e:
            wrapped = NodeException(node_name=current, cause=e, recoverable_state=state)
            self._dispatch_completed(context, current, namespace, step, pre_state, error=wrapped)
            raise wrapped from e

        try:
            new_state = _merge_partial(state, partial, self.reducers, current)
        except (ReducerError, StateValidationError) as e:
            self._dispatch_completed(context, current, namespace, step, pre_state, error=e)
            raise

        self._dispatch_completed(context, current, namespace, step, pre_state, post_state=new_state)
        return new_state

    @staticmethod
    def _dispatch_started(
        context: _InvocationContext,
        current: str,
        namespace: tuple[str, ...],
        step: int,
        pre_state: State,
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
            ),
        )
