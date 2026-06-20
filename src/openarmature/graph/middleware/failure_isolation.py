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
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any

from openarmature.observability.correlation import (
    _current_terminal_attempt_index,
    _reset_terminal_attempt_index,
    _set_terminal_attempt_index,
    current_attempt_index,
    current_branch_name,
    current_dispatch,
    current_fan_out_index,
    current_namespace_prefix,
)

from ._core import NextCall

if TYPE_CHECKING:
    # Annotation-only import; classify_cause_chain is imported lazily on the
    # catch path to keep cause_chain (and its events / errors imports) off the
    # middleware module-load path.
    from openarmature.graph.cause_chain import CaughtException

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
    - ``catch`` (optional): a set of error categories. When supplied, an
      exception is caught only if the derived category of its cause chain
      (the outermost non-carrier link's category, resolving through
      ``node_exception`` carriers, the value reported as
      ``caught_exception.category``) is in the set. Composes with
      ``predicate`` as a conjunction; both default permissive (both unset
      catches every ``Exception``). The recommended gate for
      category-scoped degradation.
    - ``predicate`` (optional): ``Exception -> bool`` over the SURFACE
      (caught) exception. When supplied, only exceptions where
      ``predicate(exc)`` is true are caught; others propagate. Defaults to
      always-true. A predicate inspecting the exception directly sees the
      ``node_exception`` carrier at a wrapping placement, not the
      originating failure; use ``catch`` for category gating, or classify
      the chain via ``classify_cause_chain``.
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
        catch: Collection[str] | None = None,
        predicate: Callable[[Exception], bool] | None = None,
        on_caught: Callable[[Exception], Awaitable[None]] | None = None,
    ) -> None:
        self.degraded_update = degraded_update
        self.event_name = event_name
        self.catch = catch
        self.predicate = predicate
        self.on_caught = on_caught

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        # Establish a clean terminal-attempt scope: an inner
        # RetryMiddleware records its final / exhausting attempt here on
        # give-up, and _emit_event reports it (proposal 0050 §6.3). The
        # ``None`` on entry shadows any stale ambient value so this call
        # reads correctly; the finally restores the prior value (token
        # semantics), and the next isolation call shadows again on entry.
        terminal_token = _set_terminal_attempt_index(None)
        try:
            try:
                return await next_(state)
            except Exception as exc:
                # BaseException (cancellation) never enters here — it
                # extends BaseException, not Exception. Same rule as
                # RetryMiddleware: cancellation MUST propagate.
                #
                # Classify once (proposal 0074 / §6.4): the ``catch`` gate and
                # the emitted event both need the cause-chain derivation, so
                # compute it here and thread it through. The local import keeps
                # cause_chain (and its events / errors imports) off the
                # middleware module-load path.
                from openarmature.graph.cause_chain import classify_cause_chain

                classification = classify_cause_chain(exc)
                # Catch gate (§6.3): caught iff the derived category is in
                # ``catch`` (or ``catch`` is unset) AND ``predicate`` admits the
                # surface exception, a conjunction with both gates permissive by
                # default (both unset stays catch-all). ``catch`` is checked first
                # and short-circuits, matching the spec's ``catch_gate(exc) and
                # predicate(exc)``, so ``predicate`` is not invoked once ``catch``
                # has rejected. ``catch`` classifies THROUGH carriers (the derived
                # category, == caught_exception.category); a null derived category
                # never matches a non-empty set, so a bare uncategorized error
                # propagates. ``predicate`` sees the surface exception.
                if self.catch is not None and classification.category not in self.catch:
                    raise
                if self.predicate is not None and not self.predicate(exc):
                    raise
                # Resolve the degraded update once, at catch time, and
                # reuse it for the event's post_state and the node return
                # so a callable degraded_update is invoked exactly once.
                # The observable order the spec prescribes — emit the
                # event, then on_caught, then return the update — is
                # preserved below; resolving here first only populates
                # post_state.
                degraded = self._resolve_degraded(state)
                self._emit_event(state, classification, degraded)
                if self.on_caught is not None:
                    try:
                        await self.on_caught(exc)
                    except Exception as hook_error:  # noqa: BLE001
                        # on_caught is caller telemetry; a bug in it MUST
                        # NOT turn a recovered node back into a crash.
                        # Isolate it the way the observer-delivery contract
                        # isolates observer exceptions (warn, don't
                        # propagate). BaseException (cancellation) still
                        # propagates by not being caught here.
                        warnings.warn(
                            f"FailureIsolationMiddleware on_caught raised "
                            f"{type(hook_error).__name__}: {hook_error}",
                            stacklevel=2,
                        )
                return degraded
        finally:
            _reset_terminal_attempt_index(terminal_token)

    def _resolve_degraded(self, state: Any) -> Mapping[str, Any]:
        if callable(self.degraded_update):
            return self.degraded_update(state)
        return self.degraded_update

    def _emit_event(self, state: Any, classification: CaughtException, degraded: Mapping[str, Any]) -> None:
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
        from openarmature.graph.events import FailureIsolatedEvent

        # The §6.4 classification (computed once at catch, in __call__) is a
        # CaughtException: the full cause chain (carriers flagged) plus the
        # derived single category / message -- the outermost non-carrier link's,
        # NOT the masking node_exception. It is the event's caught_exception
        # record directly, so the reported category equals the value the
        # ``catch`` gate matched on.
        # ``attempt_index`` is the wrapped node's final / exhausting
        # attempt (proposal 0050 §6.3: "the same lineage tuple NodeEvent
        # carries, for correlation with the wrapped node's other events").
        # When this middleware is OUTER of RetryMiddleware, retry records
        # that index in the terminal-attempt scope on give-up — its own
        # ``finally`` has reset the live attempt-index var to the baseline
        # by the time the exception reaches this catch, so we read the
        # recorded terminal index instead. With no retry, nothing is
        # recorded and we fall back to the live attempt index (0 at a node
        # body). Parenting is unaffected: the node's attempt spans are
        # already closed by delivery time (their completed event precedes
        # this one on the serial queue), so observers parent the marker
        # under the invocation span and correlate by ``namespace`` + name.
        terminal_attempt = _current_terminal_attempt_index()
        attempt_index = terminal_attempt if terminal_attempt is not None else current_attempt_index()
        dispatch(
            FailureIsolatedEvent(
                event_name=self.event_name,
                namespace=current_namespace_prefix(),
                attempt_index=attempt_index,
                fan_out_index=current_fan_out_index(),
                branch_name=current_branch_name(),
                pre_state=state,
                post_state=degraded,
                caught_exception=classification,
            )
        )


__all__ = [
    "DegradedUpdate",
    "FailureIsolationMiddleware",
]
