# Spec: canonical retry middleware per pipeline-utilities §6.1.

"""Retry middleware (canonical).

Wraps a node's chain with retry-on-transient-error logic. Each retry
attempt produces its own ``started``/``completed`` event pair from the
engine; retry middleware does NOT dispatch directly. With N attempts
the engine emits 2N events tagged with ``attempt_index`` 0..N-1, the
first 2(N-1) ending in ``error`` and the final pair ending in either
``post_state`` (success) or ``error`` (terminal failure).

Cancellation MUST propagate. Python's ``asyncio.CancelledError`` extends
``BaseException`` (not ``Exception``), so the ``except Exception`` here
does not catch it; cancellation falls straight through to the caller,
preserving the host's intent to abort.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from openarmature.llm.errors import TRANSIENT_CATEGORIES
from openarmature.observability.correlation import (
    _record_terminal_attempt_index,
    _reset_attempt_index,
    _set_attempt_index,
)
from openarmature.observability.metadata import (
    _invocation_metadata_var,
    _reset_invocation_metadata,
    _set_invocation_metadata,
)

from ._core import NextCall

# ``TRANSIENT_CATEGORIES`` is re-exported via this module's __all__
# below. Canonical source of truth lives in ``openarmature.llm.errors``
# (per llm-provider §7) — the retry middleware is just a consumer.


def default_classifier(exc: Exception, _state: Any) -> bool:
    """Default classifier; purely category-based, ignores state.

    Returns True if either the exception itself or its ``__cause__``
    carries a ``category`` attribute matching ``TRANSIENT_CATEGORIES``.
    The cause-walking covers the common case of a graph-engine
    ``NodeException`` wrapping an llm-provider transient: a
    ``node_exception`` whose ``__cause__`` is a transient category
    classifies as transient.

    The ``_state`` parameter is ignored by the default; the leading
    underscore is the canonical Python convention for "intentionally
    unused" while keeping the signature stable for user-supplied
    state-aware classifiers.
    """
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

    Jitter is mandatory; fixed exponential backoff causes
    synchronized retries from many concurrent callers, amplifying
    rate-limit storms. ``base`` and ``cap`` are configurable; the
    defaults are 1.0 and 30.0 seconds.
    """
    return random.uniform(0, min(cap, base * (2**attempt)))


def deterministic_backoff(seconds: float) -> Callable[[int], float]:
    """Constant-N seconds backoff factory, for deterministic testing.

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


@dataclass(frozen=True)
class RetryConfig:
    """Canonical retry configuration record consumed by
    :class:`RetryMiddleware`.

    - ``max_attempts``: total attempts including the first call. ``1``
      disables retry. Default ``3``.
    - ``classifier``: predicate ``(exception, state) -> bool`` deciding
      whether a failure is retry-eligible. ``None`` (the default)
      selects :func:`default_classifier` (matches ``category`` against
      ``TRANSIENT_CATEGORIES``).
    - ``backoff``: callable ``(attempt_index) -> seconds``. ``None``
      (the default) selects :func:`exponential_jitter_backoff` (base
      1s, cap 30s, full jitter).
    - ``on_retry``: optional async callback ``(exception, attempt_index)
      -> None`` fired before each backoff sleep.
    """

    max_attempts: int = 3
    classifier: Classifier | None = None
    backoff: BackoffStrategy | None = None
    on_retry: OnRetryCallback | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


class RetryMiddleware:
    """Canonical retry middleware.

    Configured with a :class:`RetryConfig` (or the default
    ``RetryConfig()`` when omitted). Construct as
    ``RetryMiddleware(RetryConfig(max_attempts=...))``.
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        if config is None:
            config = RetryConfig()
        # Defensive guard for untyped callers: the static type already
        # rules a non-RetryConfig out (pyright flags this as redundant),
        # but an eager TypeError beats a cryptic AttributeError when a
        # mistyped value (e.g. ``RetryMiddleware(3)``) reaches ``.config``.
        if not isinstance(config, RetryConfig):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"RetryMiddleware expects a RetryConfig (or None); got "
                f"{type(config).__name__}. Construct as "
                f"RetryMiddleware(RetryConfig(max_attempts=...))."
            )
        self.config = config

    async def __call__(self, state: Any, next_: NextCall) -> Mapping[str, Any]:
        attempt = 0
        # ``None`` config fields select the canonical defaults; resolve
        # once here so the loop works against concrete callables.
        classifier = self.config.classifier or default_classifier
        backoff = self.config.backoff or exponential_jitter_backoff
        # Spec observability §3.4 per-attempt scoping: each retry
        # attempt sees only the metadata in scope at retry-loop entry
        # ("pre-attempt baseline") plus that attempt's own writes;
        # writes from a prior attempt that subsequently failed do NOT
        # carry over. Captured once outside the loop because the
        # baseline is the metadata view at retry-middleware entry, not
        # at each iteration.
        pre_attempt_baseline = _invocation_metadata_var.get()
        while True:
            # Spec graph-engine §6 (clarified in v0.16.1): the wrapping
            # retry's attempt counter MUST propagate to events emitted
            # from any inner node the retry re-invokes — including
            # nodes inside subgraph / branch / fan-out-instance
            # invocations the retry wraps transitively. Set on entry,
            # reset on exit; Python's ContextVar token stack gives
            # innermost-wins precedence for free when retry middlewares
            # nest.
            attempt_token = _set_attempt_index(attempt)
            # Reset the metadata ContextVar to the pre-attempt baseline.
            # The token captures the var's state at the moment of the
            # set call — on failure we reset against this token to
            # discard any writes the attempt's node body issued.
            metadata_token = _set_invocation_metadata(pre_attempt_baseline)
            try:
                try:
                    result = await next_(state)
                    # Success path: keep the successful attempt's
                    # metadata writes in scope so downstream nodes see
                    # them. Do NOT reset metadata_token here — the
                    # engine's outer reset (around the whole invoke)
                    # pops the stack at invocation exit.
                    return result
                except Exception as exc:
                    # Spec §6.1: cancellation propagates by virtue of
                    # `CancelledError` extending `BaseException`, not
                    # `Exception` — it never enters this branch in Python.
                    # Failure path (retry-eligible OR terminal):
                    # discard the failed attempt's metadata writes per
                    # §3.4. Reset BEFORE the re-raise so the caller's
                    # error-handling path (e.g., observer hooks reading
                    # metadata for the error span) sees the baseline,
                    # not the failed attempt's transient state.
                    _reset_invocation_metadata(metadata_token)
                    if attempt + 1 >= self.config.max_attempts or not classifier(exc, state):
                        # Record the final / exhausting attempt so an OUTER
                        # FailureIsolationMiddleware reports it rather than
                        # the post-reset baseline (proposal 0050 §6.3). The
                        # enclosing isolation scope owns the cleanup.
                        _record_terminal_attempt_index(attempt)
                        raise
                    if self.config.on_retry is not None:
                        await self.config.on_retry(exc, attempt)
                    await asyncio.sleep(backoff(attempt))
                    attempt += 1
                except BaseException:
                    # Cancellation path. `CancelledError` (or other
                    # `BaseException`) ends the attempt without retry —
                    # spec §6.1 cancellation MUST propagate, never get
                    # swallowed or retried. But spec §3.4 per-attempt
                    # scoping still applies: cancellation IS a failed
                    # attempt from the metadata-scoping perspective, so
                    # its writes must be discarded too. Reset the token,
                    # then re-raise. NO on_retry, NO sleep — straight
                    # propagation.
                    _reset_invocation_metadata(metadata_token)
                    raise
            finally:
                _reset_attempt_index(attempt_token)


__all__ = [
    "BackoffStrategy",
    "Classifier",
    "OnRetryCallback",
    "RetryConfig",
    "RetryMiddleware",
    "TRANSIENT_CATEGORIES",
    "default_classifier",
    "deterministic_backoff",
    "exponential_jitter_backoff",
]
