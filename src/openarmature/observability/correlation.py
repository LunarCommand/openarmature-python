"""Cross-backend correlation primitives (spec observability §3).

Two ``ContextVar``-backed primitives that any observability backend
mapping (OTel here, Langfuse / Datadog / custom in the future) consumes
through a uniform user-readable surface:

- :data:`current_correlation_id` — the per-invocation cross-backend
  join key. Set on every outermost ``invoke()`` call (caller-supplied
  or auto-generated UUIDv4 per spec §3.1) and reset on return. User
  code in node bodies, middleware, and observers reads it via
  :func:`current_correlation_id`.
- :data:`_active_observers` — the observer set in scope for any code
  running INSIDE a node body. Read by capability backends that need
  to emit observer events from outside the engine's per-step machinery
  (e.g., the llm-provider span hook puts a NodeEvent-shaped record on
  the engine's delivery queue, then those observers receive it). The
  engine sets this around each ``chain(state)`` invocation via
  ``try/finally`` so reset is guaranteed even on exception.

These primitives live in the core package — no OpenTelemetry
dependency — because the spec §3.1 contract ("MUST propagate via the
language's idiomatic context primitive — Python ``ContextVar``") is
backend-agnostic. The OTel-specific surfacing lives under
``openarmature.observability.otel`` and is gated behind the
``[otel]`` extras.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openarmature.graph.events import NodeEvent
    from openarmature.graph.observer import SubscribedObserver


# ---------------------------------------------------------------------------
# Correlation ID — spec observability §3.1
# ---------------------------------------------------------------------------


_correlation_id_var: ContextVar[str | None] = ContextVar("openarmature.correlation_id", default=None)


def current_correlation_id() -> str | None:
    """Return the correlation ID for the current invocation, or
    ``None`` if no openarmature invocation is in scope.

    Per spec §3.1 the correlation ID MUST be readable from anywhere
    within an invocation's async call tree — node bodies, middleware,
    observers — without explicit threading through function arguments.
    This is the public reader.

    Returns ``None`` outside an invocation (e.g., at module import
    time, inside a test that runs without going through ``invoke()``).
    Callers MUST handle the ``None`` case rather than asserting a
    string is always present.
    """
    return _correlation_id_var.get()


def _set_correlation_id(value: str) -> Token[str | None]:
    """Set the correlation ID for the current invocation. Internal —
    callers OUTSIDE the engine should not touch this; the engine
    paves the lifecycle in ``CompiledGraph.invoke``.

    Returns the ``Token`` the caller MUST hand back to
    :func:`_reset_correlation_id` so the prior value is restored
    cleanly under nesting. Use ``try/finally``."""
    return _correlation_id_var.set(value)


def _reset_correlation_id(token: Token[str | None]) -> None:
    _correlation_id_var.reset(token)


# ---------------------------------------------------------------------------
# Active observer set — for capability backends emitting from outside the
# engine's per-step path (llm-provider span hook in Phase 6, future
# Langfuse/Datadog backends, user-written instrumented capabilities).
# ---------------------------------------------------------------------------


_active_observers_var: ContextVar[tuple[SubscribedObserver, ...]] = ContextVar(
    "openarmature.active_observers", default=()
)


def current_active_observers() -> tuple[SubscribedObserver, ...]:
    """Return the observer tuple in scope for the current node body
    (or empty tuple outside any invocation).

    Capability code that needs to emit observer events from outside
    the engine's per-step machinery (e.g., the llm-provider span hook
    inside ``OpenAIProvider.complete``) reads this to find which
    observers should receive the event. Combined with the engine's
    delivery queue, this preserves spec §6's strict serial ordering
    across all event sources within an invocation.

    Returns an empty tuple when no invocation is active, by design —
    callers can iterate without a None check.
    """
    return _active_observers_var.get()


def _set_active_observers(
    observers: tuple[SubscribedObserver, ...],
) -> Token[tuple[SubscribedObserver, ...]]:
    """Set the observer tuple in scope. Internal — engine-only.
    Returns a Token to hand to :func:`_reset_active_observers`."""
    return _active_observers_var.set(observers)


def _reset_active_observers(token: Token[tuple[SubscribedObserver, ...]]) -> None:
    _active_observers_var.reset(token)


# ---------------------------------------------------------------------------
# Active dispatch hook — queue-mediated event emission from outside the
# engine's per-step path. The engine sets this ContextVar to a closure
# over the current invocation's delivery queue + observer chain;
# capability backends (the LLM provider span hook in Phase 6, future
# Langfuse/Datadog instrumentations) call ``current_dispatch()(event)``
# to enqueue an event for the same delivery worker the engine uses.
#
# Spec §6 mandates strictly serial per-invocation event delivery. By
# routing capability events through the same queue (rather than calling
# observers directly), the engine's ordering guarantees extend
# automatically to LLM events, future backend events, etc. without each
# capability re-deriving the locking story.
# ---------------------------------------------------------------------------


_active_dispatch_var: ContextVar[Callable[[NodeEvent], None] | None] = ContextVar(
    "openarmature.active_dispatch", default=None
)


def current_dispatch() -> Callable[[NodeEvent], None] | None:
    """Return the engine's dispatch callable for the current invocation,
    or ``None`` outside any invocation.

    Capability code emitting observer events from inside a node body
    calls this to put a ``NodeEvent``-shaped record on the engine's
    delivery queue. The queue's serial worker preserves spec §6's
    per-invocation event ordering across all event sources (engine,
    checkpoint, LLM provider, future backends).
    """
    return _active_dispatch_var.get()


def _set_active_dispatch(
    dispatch: Callable[[NodeEvent], None],
) -> Token[Callable[[NodeEvent], None] | None]:
    """Set the engine's dispatch callable in scope. Internal —
    engine-only."""
    return _active_dispatch_var.set(dispatch)


def _reset_active_dispatch(
    token: Token[Callable[[NodeEvent], None] | None],
) -> None:
    _active_dispatch_var.reset(token)


__all__ = [
    # Public surface — readable from anywhere within an invocation.
    "current_active_observers",
    "current_correlation_id",
    "current_dispatch",
    # Engine-internal lifecycle helpers — exported so the engine in
    # ``openarmature.graph.compiled`` can drive set/reset without
    # pyright's strict ``reportUnusedFunction`` flagging them as
    # dead. Underscore-prefixed; not part of the user-facing API.
    "_reset_active_dispatch",
    "_reset_active_observers",
    "_reset_correlation_id",
    "_set_active_dispatch",
    "_set_active_observers",
    "_set_correlation_id",
]
