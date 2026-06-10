# Spec: canonical failure-isolation middleware per pipeline-utilities
# §6.3 (proposal 0050). Packages the §2 third-MAY-bullet
# catch-and-recover pattern as a named primitive alongside §6.1 retry
# and §6.2 timing.

"""Failure-isolation middleware (canonical).

Wraps a node's chain so an exception escaping the inner chain becomes a
configured degraded partial update instead of propagating. The
companion to ``RetryMiddleware`` for the "retry transients, give up
gracefully on exhaustion" pattern.

Catches ``Exception`` by default; ``BaseException``
(``asyncio.CancelledError``, ``KeyboardInterrupt``) propagates so
cancellation works as expected — the same rule as ``RetryMiddleware``.

On a caught exception the middleware first resolves ``degraded_update``
(a static mapping, or a callable taking the pre-call state; invoked
once, at catch time, which is also what populates the dispatched
event's ``post_state``), then in order:

1. Dispatches a ``FailureIsolatedEvent`` onto the engine's serial
   observer-delivery queue (a framework-emitted event; the bundled
   OTel and Langfuse observers render the catch). The default emission
   path is the observer event, with no logging-library dependency.
2. Awaits the optional ``on_caught`` hook.
3. Returns the resolved degraded update as the node's partial update.

Composition with ``RetryMiddleware``: failure isolation MUST be the
OUTER middleware (it only sees what escapes retry); retry MUST be INNER
(it sees raw transients first and retries them). Reversing the order
lets the inner isolation swallow transients before retry can see them,
defeating retry entirely.
"""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from openarmature.observability.correlation import (
    current_attempt_index,
    current_branch_name,
    current_dispatch,
    current_fan_out_index,
    current_namespace_prefix,
)

from ._core import NextCall

# A degraded update is either a static partial-update mapping or a
# callable resolving one from the pre-call state. Resolved at catch
# time; the callable form covers input-state-dependent degraded shapes.
DegradedUpdate = Mapping[str, Any] | Callable[[Any], Mapping[str, Any]]


class FailureIsolationMiddleware:
    """Catch exceptions escaping the inner chain; return a degraded
    partial update.

    Configuration:

    - ``degraded_update`` (required): the partial update returned on a
      caught exception, OR a callable ``state -> partial_update`` for
      input-state-dependent degraded shapes.
    - ``event_name`` (required): a stable identifier for this catch
      site; surfaces on the ``FailureIsolatedEvent``. No default —
      useful values are node-specific, and a generic default would make
      downstream telemetry strictly worse.
    - ``predicate`` (optional): ``Exception -> bool``. When supplied,
      only exceptions where ``predicate(exc)`` is true are caught;
      others propagate. Defaults to catching every ``Exception``.
    - ``on_caught`` (optional): an async ``Exception -> Awaitable[None]``
      hook fired on a caught exception, for caller-specific telemetry
      beyond the framework event. It runs inline before the degraded
      update is returned, so a slow hook delays the node's return; an
      exception raised by the hook is isolated (logged via
      ``warnings.warn``, not propagated) so a telemetry bug cannot turn
      a recovered node back into a failure.
    """

    def __init__(
        self,
        *,
        degraded_update: DegradedUpdate,
        event_name: str,
        predicate: Callable[[Exception], bool] | None = None,
        on_caught: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        self.degraded_update = degraded_update
        self.event_name = event_name
        self.predicate = predicate
        self.on_caught = on_caught

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        try:
            return await next_(state)
        except Exception as exc:
            # BaseException (cancellation) never enters here — it
            # extends BaseException, not Exception. Same rule as
            # RetryMiddleware: cancellation MUST propagate.
            if self.predicate is not None and not self.predicate(exc):
                raise
            # Resolve the degraded update once, at catch time, and reuse
            # it for the event's post_state and the node return so a
            # callable degraded_update is invoked exactly once. The
            # observable order the spec prescribes — emit the event, then
            # on_caught, then return the update — is preserved below;
            # resolving here first only populates post_state.
            degraded = self._resolve_degraded(state)
            self._emit_event(state, exc, degraded)
            if self.on_caught is not None:
                try:
                    await self.on_caught(exc)
                except Exception as hook_error:  # noqa: BLE001
                    # on_caught is caller telemetry; a bug in it MUST NOT
                    # turn a recovered node back into a crash. Isolate it
                    # the way the observer-delivery contract isolates
                    # observer exceptions (warn, don't propagate).
                    # BaseException (cancellation) still propagates by not
                    # being caught here.
                    warnings.warn(
                        f"FailureIsolationMiddleware on_caught raised "
                        f"{type(hook_error).__name__}: {hook_error}",
                        stacklevel=2,
                    )
            return degraded

    def _resolve_degraded(self, state: Any) -> Mapping[str, Any]:
        if callable(self.degraded_update):
            return self.degraded_update(state)
        return self.degraded_update

    def _emit_event(self, state: Any, exc: Exception, degraded: Mapping[str, Any]) -> None:
        dispatch = current_dispatch()
        # current_dispatch() is None outside an invocation (no observers
        # in scope, e.g. unit-testing the middleware directly) — the
        # degraded return still happens; there is just no delivery queue
        # to enqueue onto.
        if dispatch is None:
            return
        # Local import mirrors set_invocation_metadata's 0040 emit: it
        # keeps the event-type import off the middleware module-load
        # path and defers it until the first catch.
        from openarmature.graph.events import CaughtException, FailureIsolatedEvent

        # A categorized exception (e.g. an llm-provider error) carries a
        # string ``category``. When the engine has wrapped the original
        # in a graph-engine error before it reached the middleware, the
        # category rides on ``__cause__`` — walk it the same way the
        # default retry classifier does so the caught failure's category
        # survives the wrapping. A bare exception yields ``None``.
        category = getattr(exc, "category", None)
        if not isinstance(category, str):
            cause = getattr(exc, "__cause__", None)
            cause_category = getattr(cause, "category", None) if cause is not None else None
            category = cause_category if isinstance(cause_category, str) else None
        # ``attempt_index`` here is deliberately the NODE-level baseline,
        # not a per-attempt wire index: failure isolation is a node-level
        # concern ("the node, across its retries, was isolated"). When
        # this middleware is OUTER of RetryMiddleware, retry has already
        # reset the attempt ContextVar to that baseline (0) in its
        # ``finally`` by the time the terminal exception reaches this
        # catch, which is the frame we want (spec-confirmed). Parenting is
        # unaffected: the node's attempt spans are already closed by
        # delivery time (their completed event precedes this one on the
        # serial queue), so observers parent the marker under the
        # invocation span and correlate by ``namespace`` + node name.
        dispatch(
            FailureIsolatedEvent(
                event_name=self.event_name,
                namespace=current_namespace_prefix(),
                attempt_index=current_attempt_index(),
                fan_out_index=current_fan_out_index(),
                branch_name=current_branch_name(),
                pre_state=state,
                post_state=degraded,
                caught_exception=CaughtException(category=category, message=str(exc)),
            )
        )


__all__ = [
    "DegradedUpdate",
    "FailureIsolationMiddleware",
]
