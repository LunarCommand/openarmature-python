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
# Invocation ID — spec observability §5.1
#
# The framework-generated UUIDv4 that ties spans of one invocation
# together within a single backend. Distinct from ``correlation_id``
# (which is the cross-backend join key, caller-supplied or auto-
# generated). Engine sets this ContextVar in ``invoke()`` BEFORE the
# delivery worker is created so the worker's captured context sees
# the right value.
# ---------------------------------------------------------------------------


_invocation_id_var: ContextVar[str | None] = ContextVar("openarmature.invocation_id", default=None)


def current_invocation_id() -> str | None:
    """Return the engine-minted invocation ID for the current
    invocation, or ``None`` if no openarmature invocation is in
    scope.

    Per spec observability §5.1 every invocation produces a unique
    UUIDv4 ``invocation_id``, framework-generated, surfaced as the
    ``openarmature.invocation_id`` attribute on the invocation span
    + on every per-backend record. This is the public reader for
    backend mappings (OTel, future Langfuse) that need to populate
    that attribute.
    """
    return _invocation_id_var.get()


def _set_invocation_id(value: str) -> Token[str | None]:
    """Set the invocation ID for the current invocation. Internal —
    engine-only."""
    return _invocation_id_var.set(value)


def _reset_invocation_id(token: Token[str | None]) -> None:
    _invocation_id_var.reset(token)


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


# ---------------------------------------------------------------------------
# Calling-node identity — for the OTel observer's §5.5 LLM-span parent
# attribution under concurrent fan-out + retry. The engine sets these
# ContextVars around node-body execution in ``_step_*_node``; capability
# code emitting ``NodeEvent``s from inside a node body (the LLM provider
# span hook) reads them to record which node the event originated from.
#
# Without these, the OTel observer falls back to ``opentelemetry.trace``'s
# current-span context to resolve the parent, which under concurrent
# fan-out can yield a sibling instance's span rather than the actual
# calling node. The §5.5 contract states the *outcome* (LLM span parents
# under the calling node); these ContextVars provide the *mechanism*.
#
# Defaults are baked into ContextVar construction so readers outside any
# node body (e.g., LLM ``complete`` called from a top-level harness)
# return the sentinel values directly without engine-side initialization.
# ---------------------------------------------------------------------------


_namespace_prefix_var: ContextVar[tuple[str, ...]] = ContextVar("openarmature.namespace_prefix", default=())


def current_namespace_prefix() -> tuple[str, ...]:
    """Return the namespace prefix of the node currently executing,
    or the empty tuple outside any node body.

    The empty-tuple default makes top-level (outside-invocation) and
    between-nodes (e.g., middleware bodies) calls fall back to
    invocation-level parenting cleanly.
    """
    return _namespace_prefix_var.get()


def _set_namespace_prefix(value: tuple[str, ...]) -> Token[tuple[str, ...]]:
    """Set the calling node's namespace prefix. Internal —
    engine-only; called inside ``_step_*_node`` around node-body
    execution."""
    return _namespace_prefix_var.set(value)


def _reset_namespace_prefix(token: Token[tuple[str, ...]]) -> None:
    _namespace_prefix_var.reset(token)


_fan_out_index_var: ContextVar[int | None] = ContextVar("openarmature.fan_out_index", default=None)


def current_fan_out_index() -> int | None:
    """Return the fan_out_index of the node currently executing, or
    ``None`` outside any fan-out instance body (top-level nodes,
    subgraph dispatch, between nodes).
    """
    return _fan_out_index_var.get()


def _set_fan_out_index(value: int | None) -> Token[int | None]:
    """Set the calling node's fan_out_index. Internal — engine-only."""
    return _fan_out_index_var.set(value)


def _reset_fan_out_index(token: Token[int | None]) -> None:
    _fan_out_index_var.reset(token)


_attempt_index_var: ContextVar[int] = ContextVar("openarmature.attempt_index", default=0)


def current_attempt_index() -> int:
    """Return the attempt_index of the node currently executing, or
    ``0`` outside any node body. Retry middleware bumps this per
    attempt; the OTel observer uses it to disambiguate per-attempt
    spans when an LLM call happens inside a retried node body.
    """
    return _attempt_index_var.get()


def _set_attempt_index(value: int) -> Token[int]:
    """Set the calling node's attempt_index. Internal — engine-only."""
    return _attempt_index_var.set(value)


def _reset_attempt_index(token: Token[int]) -> None:
    _attempt_index_var.reset(token)


# ---------------------------------------------------------------------------
# Active observer span — for engine-side OTel context attach inside
# ``innermost``. Populated synchronously by an observer's ``prepare_sync``
# hook BEFORE the engine queues the started event; read by ``innermost``
# AFTER ``_dispatch_started`` returns to attach the span into the OTel
# context for the duration of the node-body chain.
#
# Inverted directionality vs. the engine→observer ContextVars above:
# this one flows observer→engine. The producer (an opt-in observer's
# ``prepare_sync``) and the consumer (``innermost``) both run in the
# engine task, so the same-task ContextVar contract holds — last-writer-
# wins is fine in practice (charter §6 says "one OTelObserver per
# private provider," so multi-OTelObserver attach is rare).
#
# Typed as ``object | None`` rather than ``Span | None`` so the base
# package stays free of an OpenTelemetry import. The OTel observer
# writes ``Span`` instances; the engine treats the value opaquely and
# delegates the actual attach to a try-imported OTel helper.
# ---------------------------------------------------------------------------


_active_observer_span_var: ContextVar[object | None] = ContextVar(
    "openarmature.active_observer_span", default=None
)


def current_active_observer_span() -> object | None:
    """Return the active observer-side span for the current node body,
    or ``None`` if no observer published one (no opt-in observer with
    ``prepare_sync`` is attached, or this is being called outside a
    node-body scope).

    Engine-readable handle to the span an opt-in observer's
    ``prepare_sync`` created synchronously during dispatch. The engine's
    ``innermost`` reads this AFTER ``_dispatch_started`` returns and
    attaches the span into the OTel context (via a try-imported OTel
    helper) so that any logs emitted FROM INSIDE the node body — even
    on the first line, before any ``await`` — pick up the span's
    trace_id/span_id via OTel's ``LoggingHandler``.

    Backend coupling note: typed as ``object | None`` so this primitive
    works in installs without the ``[otel]`` extras. OTel observers
    write OpenTelemetry ``Span`` instances; the engine treats the
    value opaquely.
    """
    return _active_observer_span_var.get()


def _set_active_observer_span(value: object | None) -> Token[object | None]:
    """Set the active observer span. Internal — observers' ``prepare_sync``
    implementations call this synchronously before returning so the
    engine's ``innermost`` reads the right value when it attaches."""
    return _active_observer_span_var.set(value)


def _reset_active_observer_span(token: Token[object | None]) -> None:
    _active_observer_span_var.reset(token)


__all__ = [
    # Public surface — readable from anywhere within an invocation.
    "current_active_observer_span",
    "current_active_observers",
    "current_attempt_index",
    "current_correlation_id",
    "current_dispatch",
    "current_fan_out_index",
    "current_invocation_id",
    "current_namespace_prefix",
    # Engine-internal lifecycle helpers — exported so the engine in
    # ``openarmature.graph.compiled`` can drive set/reset without
    # pyright's strict ``reportUnusedFunction`` flagging them as
    # dead. Underscore-prefixed; not part of the user-facing API.
    "_reset_active_dispatch",
    "_reset_active_observer_span",
    "_reset_active_observers",
    "_reset_attempt_index",
    "_reset_correlation_id",
    "_reset_fan_out_index",
    "_reset_invocation_id",
    "_reset_namespace_prefix",
    "_set_active_dispatch",
    "_set_active_observer_span",
    "_set_active_observers",
    "_set_attempt_index",
    "_set_correlation_id",
    "_set_fan_out_index",
    "_set_invocation_id",
    "_set_namespace_prefix",
]
