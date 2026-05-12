# Spec: canonical timing middleware per pipeline-utilities §6.2.

"""Timing middleware (canonical).

Records wall-clock duration of the wrapped chain (including any inner
middleware time, e.g., retries) and dispatches the result to a
user-supplied async callback. Uses ``time.monotonic`` (monotonic
across NTP corrections and DST transitions, where wall-clock would
produce negative durations that corrupt downstream metric pipelines).

The middleware is constructed with an explicit ``node_name`` because
the ``(state, next)`` middleware shape doesn't expose node identity
at call time. Per-instance clock injection (defaulting to
``time.monotonic``) lets test fixtures supply a deterministic stub
without globally patching ``time.monotonic``, which would also
affect asyncio's scheduling layer.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ._core import NextCall


@dataclass(frozen=True)
class TimingRecord:
    """A single timing measurement produced by ``TimingMiddleware``.

    - ``node_name``: the node this middleware was attached to
      (captured at registration; users supply it explicitly for
      per-node use).
    - ``duration_ms``: milliseconds from middleware entry to chain
      return-or-raise, measured with a monotonic clock.
    - ``outcome``: one of ``"success"`` or ``"exception"``.
    - ``exception_category``: when ``outcome == "exception"`` and the
      exception carries a ``category`` attribute, that string;
      otherwise ``None``.
    """

    node_name: str
    duration_ms: float
    outcome: str
    exception_category: str | None


OnCompleteCallback = Callable[[TimingRecord], Awaitable[None]]


class TimingMiddleware:
    """Canonical timing middleware.

    Records wall-clock duration of the wrapped chain via the host
    language's monotonic clock (Python's ``time.monotonic``). The
    callback fires inline before the chain's result returns to the
    caller — slow callbacks add to the apparent node duration, so
    users SHOULD keep them fast (queue work, defer I/O).

    Errors raised by ``on_complete`` propagate to the engine as a
    ``node_exception``.
    """

    def __init__(
        self,
        *,
        node_name: str,
        on_complete: OnCompleteCallback,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.node_name = node_name
        self.on_complete = on_complete
        # Per-instance clock so test fixtures can supply a deterministic
        # stub without globally patching ``time.monotonic`` (which would
        # also affect asyncio's scheduling layer). Defaults to the spec-
        # mandated ``time.monotonic`` for production use.
        self._clock: Callable[[], float] = clock or time.monotonic

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        started_at = self._clock()
        try:
            partial = await next_(state)
        except Exception as exc:
            duration_ms = (self._clock() - started_at) * 1000.0
            category = getattr(exc, "category", None)
            await self.on_complete(
                TimingRecord(
                    node_name=self.node_name,
                    duration_ms=duration_ms,
                    outcome="exception",
                    exception_category=category if isinstance(category, str) else None,
                )
            )
            raise

        duration_ms = (self._clock() - started_at) * 1000.0
        await self.on_complete(
            TimingRecord(
                node_name=self.node_name,
                duration_ms=duration_ms,
                outcome="success",
                exception_category=None,
            )
        )
        return partial


__all__ = [
    "OnCompleteCallback",
    "TimingMiddleware",
    "TimingRecord",
]
