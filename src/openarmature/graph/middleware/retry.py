"""Retry middleware (canonical, spec pipeline-utilities §6.1).

Wraps a node's chain with retry-on-transient-error logic. Each retry
attempt produces its own ``started``/``completed`` event pair from the
engine — retry middleware does NOT dispatch directly. With N attempts
the engine emits 2N events tagged with ``attempt_index`` 0..N-1, the
first 2(N-1) ending in ``error`` and the final pair ending in either
``post_state`` (success) or ``error`` (terminal failure).

Cancellation MUST propagate. Python's ``asyncio.CancelledError`` extends
``BaseException`` (not ``Exception``), so the ``except Exception`` here
does not catch it — cancellation falls straight through to the caller,
preserving the host's intent to abort.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ._core import NextCall

# Default transient set per pipeline-utilities §6.1: matches against an
# exception's ``category`` attribute (or the cause's, if the exception is
# a NodeException wrapper). The canonical strings are spec-defined and
# loose-coupled to llm-provider §7 — implementers don't have to import
# the llm-provider module to wire up retry.
TRANSIENT_CATEGORIES: frozenset[str] = frozenset(
    {
        "provider_rate_limit",
        "provider_unavailable",
        "provider_model_not_loaded",
    }
)


def default_classifier(exc: Exception, state: Any) -> bool:
    """Spec §6.1 default classifier — purely category-based, ignores state.

    Returns True if either the exception itself or its ``__cause__``
    carries a ``category`` attribute matching ``TRANSIENT_CATEGORIES``.
    The cause-walking covers the common case of a graph-engine
    ``NodeException`` wrapping an llm-provider transient — per the spec:
    "a `node_exception` whose `__cause__` is a transient category MUST
    be classified as transient."
    """
    # Suppress the unused-arg warning while keeping the signature stable
    # for user-supplied state-aware classifiers.
    del state
    direct = getattr(exc, "category", None)
    if isinstance(direct, str) and direct in TRANSIENT_CATEGORIES:
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        cause_cat = getattr(cause, "category", None)
        if isinstance(cause_cat, str) and cause_cat in TRANSIENT_CATEGORIES:
            return True
    return False


def exponential_jitter_backoff(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 30.0,
) -> float:
    """Default backoff: ``random.uniform(0, min(cap, base * 2**attempt))``.

    Per spec §6.1: jitter is mandatory — fixed exponential backoff
    causes synchronized retries from many concurrent callers, amplifying
    rate-limit storms. ``base`` and ``cap`` are configurable; the
    spec-mandated defaults are 1.0 and 30.0 seconds.
    """
    return random.uniform(0, min(cap, base * (2**attempt)))


def deterministic_backoff(seconds: float) -> Callable[[int], float]:
    """Constant-N seconds backoff factory — for deterministic testing.

    The conformance fixtures use this form via ``backoff: {type:
    deterministic, seconds: N}`` so retry timing is reproducible across
    runs.
    """

    def fn(_attempt: int) -> float:
        return seconds

    return fn


# Type aliases for the configuration shapes — exposed at module scope
# so users can write callbacks against them without re-deriving.
Classifier = Callable[[Exception, Any], bool]
BackoffStrategy = Callable[[int], float]
OnRetryCallback = Callable[[Exception, int], Awaitable[None]]


class RetryMiddleware:
    """Spec §6.1 canonical retry middleware.

    Configuration:

    - ``max_attempts``: total attempts including the first call. ``1``
      disables retry. Default ``3``.
    - ``classifier``: predicate ``(exception, state) -> bool``. Default
      :func:`default_classifier` (matches ``category`` against
      ``TRANSIENT_CATEGORIES``).
    - ``backoff``: callable ``(attempt_index) -> seconds``. Default
      :func:`exponential_jitter_backoff` (base 1s, cap 30s, full jitter).
    - ``on_retry``: optional async callback ``(exception, attempt_index)
      -> None``. Fires before each sleep.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        classifier: Classifier | None = None,
        backoff: BackoffStrategy | None = None,
        on_retry: OnRetryCallback | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.classifier: Classifier = classifier or default_classifier
        self.backoff: BackoffStrategy = backoff or exponential_jitter_backoff
        self.on_retry: OnRetryCallback | None = on_retry

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        attempt = 0
        while True:
            try:
                return await next_(state)
            except Exception as exc:
                # Spec §6.1: cancellation propagates by virtue of
                # `CancelledError` extending `BaseException`, not
                # `Exception` — it never enters this branch in Python.
                if attempt + 1 >= self.max_attempts or not self.classifier(exc, state):
                    raise
                if self.on_retry is not None:
                    await self.on_retry(exc, attempt)
                await asyncio.sleep(self.backoff(attempt))
                attempt += 1


__all__ = [
    "BackoffStrategy",
    "Classifier",
    "OnRetryCallback",
    "RetryMiddleware",
    "TRANSIENT_CATEGORIES",
    "default_classifier",
    "deterministic_backoff",
    "exponential_jitter_backoff",
]
