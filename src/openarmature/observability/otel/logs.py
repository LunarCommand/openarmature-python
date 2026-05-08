"""OTel Logs Bridge integration (spec observability §7).

Provides :func:`install_log_bridge` — an opt-in helper that wires
the stdlib :mod:`logging` root logger through the OTel Logs SDK so
every log record emitted within an invocation carries the active
``trace_id``/``span_id`` plus ``openarmature.correlation_id``.

Opt-in by design: users may have their own logging configuration we
shouldn't override silently. Calling ``install_log_bridge(provider)``
explicitly attaches an OTel ``LoggingHandler`` to the root logger
and registers a filter that injects the correlation_id from the
ContextVar.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider


class _CorrelationIdFilter(logging.Filter):
    """Logging filter that reads the openarmature correlation_id
    ContextVar and attaches it to every log record as the
    ``openarmature.correlation_id`` attribute. Per spec §7 the
    attribute MUST appear on every log record emitted during an
    invocation."""

    def filter(self, record: logging.LogRecord) -> bool:
        from openarmature.observability.correlation import current_correlation_id

        cid = current_correlation_id()
        if cid is not None:
            # Stored on the log record so any formatter/handler that
            # reads ``record.__dict__`` (including the OTel
            # LoggingHandler) sees it.
            setattr(record, "openarmature.correlation_id", cid)
        return True


def install_log_bridge(
    provider: LoggerProvider,
    *,
    level: int = logging.NOTSET,
) -> None:
    """Wire the stdlib root logger to the supplied OTel
    :class:`LoggerProvider`. Adds a
    :class:`opentelemetry.sdk._logs.LoggingHandler` for OTel-native
    ``trace_id`` / ``span_id`` bridging, AND attaches an
    :class:`_CorrelationIdFilter` directly to the ROOT LOGGER (not
    the handler) so the ``openarmature.correlation_id`` attribute
    lands on every log record emitted during an invocation —
    including records consumed by pre-existing stdout / file /
    third-party handlers the user already had configured.

    Filter-on-the-root-logger placement matters per spec §7:
    "log records emitted from anywhere within an invocation MUST
    carry ``openarmature.correlation_id``." A handler-level filter
    would only modify records flowing through THAT handler, so a
    user's existing stdout handler would see records without the
    attribute. The root-logger filter applies to every record,
    regardless of which handler eventually processes it.

    Idempotent: re-calling is a no-op (we check for the existing
    OA-tagged handler AND for an existing filter instance on the
    root logger).

    The user retains responsibility for providing the
    :class:`LoggerProvider` (typically built with their preferred
    exporter — :class:`InMemoryLogExporter` for tests,
    :class:`OTLPLogExporter` for production).
    """
    from opentelemetry.sdk._logs import LoggingHandler  # type: ignore[attr-defined]

    root = logging.getLogger()
    # Idempotency #1: don't double-add the OTel LoggingHandler.
    handler_already_installed = any(
        isinstance(h, LoggingHandler) and getattr(h, "_openarmature_installed", False) for h in root.handlers
    )
    if not handler_already_installed:
        handler = LoggingHandler(level=level, logger_provider=provider)
        # Direct assignment isn't typed on LoggingHandler; route
        # through ``object.__setattr__`` to avoid pyright's strict
        # attribute-access check without losing the idempotency-
        # marker behavior.
        object.__setattr__(handler, "_openarmature_installed", True)
        root.addHandler(handler)
    # Idempotency #2: don't double-add the correlation_id filter to
    # the root logger.
    filter_already_installed = any(isinstance(f, _CorrelationIdFilter) for f in root.filters)
    if not filter_already_installed:
        root.addFilter(_CorrelationIdFilter())


__all__ = [
    "install_log_bridge",
]
