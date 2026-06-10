"""Middleware subpackage: protocol + canonical implementations.

The ``Middleware`` Protocol and chain-composition machinery live in
:mod:`._core`; the canonical middleware (Retry, Timing) live in their
own modules. The subpackage's public surface is
re-exported here so callers can write::

    from openarmature.graph.middleware import (
        Middleware,
        RetryMiddleware,
        TimingMiddleware,
    )

without reaching into the internal module layout. The top-level
``openarmature.graph`` namespace also re-exports the most-used names
(``RetryMiddleware``, ``TimingMiddleware``, etc.) for ergonomic
imports at the package boundary.
"""

from ._core import ChainCall, Middleware, NextCall, compose_chain
from .failure_isolation import DegradedUpdate, FailureIsolationMiddleware
from .retry import (
    TRANSIENT_CATEGORIES,
    BackoffStrategy,
    Classifier,
    OnRetryCallback,
    RetryMiddleware,
    default_classifier,
    deterministic_backoff,
    exponential_jitter_backoff,
)
from .timing import OnCompleteCallback, TimingMiddleware, TimingRecord

__all__ = [
    "BackoffStrategy",
    "ChainCall",
    "Classifier",
    "DegradedUpdate",
    "FailureIsolationMiddleware",
    "Middleware",
    "NextCall",
    "OnCompleteCallback",
    "OnRetryCallback",
    "RetryMiddleware",
    "TRANSIENT_CATEGORIES",
    "TimingMiddleware",
    "TimingRecord",
    "compose_chain",
    "default_classifier",
    "deterministic_backoff",
    "exponential_jitter_backoff",
]
