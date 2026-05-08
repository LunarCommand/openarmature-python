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
    :class:`opentelemetry.sdk._logs.LoggingHandler` and an
    :class:`_CorrelationIdFilter` so every log record emitted from
    anywhere within an invocation carries the active
    ``trace_id``/``span_id`` + ``openarmature.correlation_id``.

    Idempotent: re-calling with a previously-installed provider is
    a no-op (we check for an existing OA handler before adding).

    The user retains responsibility for providing the
    :class:`LoggerProvider` (typically built with their preferred
    exporter — :class:`InMemoryLogExporter` for tests,
    :class:`OTLPLogExporter` for production).
    """
    from opentelemetry.sdk._logs import LoggingHandler  # type: ignore[attr-defined]

    root = logging.getLogger()
    # Idempotency check — don't double-install on repeated calls.
    for handler in root.handlers:
        if isinstance(handler, LoggingHandler) and getattr(handler, "_openarmature_installed", False):
            return
    handler = LoggingHandler(level=level, logger_provider=provider)
    # Direct assignment isn't typed on LoggingHandler; route through
    # ``setattr`` to avoid pyright's strict attribute-access check
    # without losing the idempotency-marker behavior.
    object.__setattr__(handler, "_openarmature_installed", True)
    handler.addFilter(_CorrelationIdFilter())
    root.addHandler(handler)


__all__ = [
    "install_log_bridge",
]
