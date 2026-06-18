"""Test-only middleware classes used by pipeline-utilities conformance fixtures.

These mirror the test seams documented in fixture 001's header and live
in tests/ rather than src/openarmature so they aren't shipped as part of
the public API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from openarmature.graph.middleware import NextCall


@dataclass
class TraceRecord:
    """A single record from a TraceRecorderMiddleware run."""

    state_in: dict[str, Any]
    partial_update_returned: dict[str, Any] | None = None
    exception_caught: bool = False
    # Pre/post phase flags — pre is set when the middleware enters its
    # body (before next is called); post is set when next returns
    # successfully. If next raises, pre_seen=True, post_seen=False.
    pre_seen: bool = False
    post_seen: bool = False


class TraceRecorderMiddleware:
    """Records `(state_in, partial_update_returned)` per dispatch.

    Optionally appends a pre/post marker to a state field — used by
    fixture 002 (composition ordering) where each recorder writes a
    distinct marker so the trace's ordering verifies chain composition.

    `pre_marker` is appended to `state.<marker_field>` BEFORE calling
    next (the partial_update returned by this middleware after the chain
    returns includes the marker). `post_marker` is appended AFTER.
    """

    def __init__(
        self,
        *,
        sink: list[TraceRecord],
        pre_marker: str | None = None,
        post_marker: str | None = None,
        marker_field: str = "trace",
    ) -> None:
        self.sink = sink
        self.pre_marker = pre_marker
        self.post_marker = post_marker
        self.marker_field = marker_field

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        record = TraceRecord(state_in=state.model_dump(), pre_seen=True)
        # Append on entry so an exception path still leaves the
        # pre_seen=True record visible in the sink.
        self.sink.append(record)

        try:
            inner_partial = await next_(state)
        except Exception:
            record.exception_caught = True
            raise

        record.partial_update_returned = dict(inner_partial)
        record.post_seen = True

        merged: dict[str, Any] = dict(inner_partial)
        if self.pre_marker is not None or self.post_marker is not None:
            existing = list(merged.get(self.marker_field, []))
            pre_part = [self.pre_marker] if self.pre_marker is not None else []
            post_part = [self.post_marker] if self.post_marker is not None else []
            merged[self.marker_field] = pre_part + existing + post_part

        return merged


class ShortCircuitMiddleware:
    """Returns the configured partial without calling `next`.

    The rest of the chain — subsequent middleware and the wrapped node
    — does not execute. The short-circuiting middleware's
    own post-phase is also skipped (because there's no `await next`
    return point to pass through).
    """

    def __init__(self, *, partial_update: Mapping[str, Any]) -> None:
        self.partial_update = dict(partial_update)

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        del state, next_
        return self.partial_update


class ErrorRecoveryMiddleware:
    """Catches any Exception from `next`; returns the configured partial.

    Middleware MAY catch an exception and return a partial update
    instead of re-raising. The engine treats the dispatch as a
    success (post_state populated, no error in the completed event).
    """

    def __init__(self, *, partial_update: Mapping[str, Any]) -> None:
        self.partial_update = dict(partial_update)

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        try:
            return await next_(state)
        except Exception:
            return self.partial_update


class ErrorRaiserMiddleware:
    """Raises a configured exception in the pre-phase.

    Verifies that middleware-raised exceptions surface as
    ``node_exception``.
    """

    def __init__(self, *, message: str) -> None:
        self.message = message

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        del state, next_
        raise RuntimeError(self.message)


class StateInspectorMiddleware:
    """Captures the input state on entry, then verifies after `next`
    returns that the same state instance is unchanged (immutability)."""

    def __init__(self, *, sink: list[bool]) -> None:
        self.sink = sink

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        snapshot = state.model_dump()
        partial = await next_(state)
        # State instances are frozen pydantic models; this assertion
        # mostly guards against a future regression where the engine or
        # middleware accidentally mutates a state field.
        self.sink.append(state.model_dump() == snapshot)
        return partial


__all__ = [
    "ErrorRaiserMiddleware",
    "ErrorRecoveryMiddleware",
    "ShortCircuitMiddleware",
    "StateInspectorMiddleware",
    "TraceRecord",
    "TraceRecorderMiddleware",
]


# Force `field` to be referenced so the unused-import linter doesn't
# strip it; dataclass fixtures may need default factories later.
_ = field
