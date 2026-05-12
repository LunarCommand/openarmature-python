# Spec: realizes observability §7 (logs correlation contract).

"""OTel Logs Bridge integration.

Provides :func:`install_log_bridge` — an opt-in helper that wires
the stdlib :mod:`logging` root logger through the OTel Logs SDK so
every log record emitted within an invocation carries the active
``trace_id`` / ``span_id`` plus ``openarmature.correlation_id``.

Opt-in by design: users may have their own logging configuration we
shouldn't override silently. Calling ``install_log_bridge(provider)``
explicitly attaches an OTel ``LoggingHandler`` to the root logger
and installs a process-global ``LogRecord`` factory that injects
the correlation_id from the ContextVar.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider


# Marker attribute used to detect "this is the OA-installed
# LogRecord factory" so re-calling ``install_log_bridge`` doesn't
# stack a second wrapper on top of the already-installed one.
_FACTORY_MARKER = "_openarmature_correlation_factory"


def _install_correlation_id_factory() -> None:
    """Install a process-global :class:`logging.LogRecord` factory
    that reads the openarmature correlation_id ContextVar and
    attaches it to every constructed record as the
    ``openarmature.correlation_id`` attribute.

    Why a factory instead of a logger filter: filters added to the
    ROOT logger only fire for records originating directly on the
    root logger — Python's logging propagation walks ancestors'
    HANDLERS but not their filters. A filter on root therefore
    misses every record from a child logger (the normal case; every
    reasonable user does ``logger = logging.getLogger("module")``).
    The attribute MUST appear on records emitted from anywhere
    within an invocation — the factory hooks at record construction,
    fires uniformly for every emit regardless of which logger
    originated the record, and chains over any user-installed
    factory rather than replacing it.

    Idempotent: re-calling skips installation if the current factory
    is already the OA-installed one.
    """
    from openarmature.observability.correlation import current_correlation_id

    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, _FACTORY_MARKER, False):
        # Already installed — re-calling is a no-op.
        return

    prior_factory = current_factory

    def _correlation_id_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = prior_factory(*args, **kwargs)
        cid = current_correlation_id()
        if cid is not None:
            # Stored on the log record so any formatter/handler that
            # reads ``record.__dict__`` (including the OTel
            # LoggingHandler) sees it.
            setattr(record, "openarmature.correlation_id", cid)
        return record

    setattr(_correlation_id_factory, _FACTORY_MARKER, True)
    logging.setLogRecordFactory(_correlation_id_factory)


def install_log_bridge(
    provider: LoggerProvider,
    *,
    level: int = logging.NOTSET,
) -> None:
    """Wire the stdlib root logger to the supplied OTel
    :class:`LoggerProvider`. Adds a
    :class:`opentelemetry.instrumentation.logging.handler.LoggingHandler`
    for OTel-native ``trace_id`` / ``span_id`` bridging, AND
    installs a process-global :class:`logging.LogRecord` factory
    that injects ``openarmature.correlation_id`` on every record.

    The factory placement matters: log records emitted from
    anywhere within an invocation MUST carry
    ``openarmature.correlation_id``. Filters added to the root
    logger fire only for records originating on root — Python's
    propagation walks ancestor handlers but not ancestor filters —
    so a root-logger filter misses every child-logger record. The
    factory hook fires at record construction time, before any
    logger or handler dispatch, so every record gets the attribute
    regardless of which logger originated it.

    Idempotent: re-calling is a no-op (we check for the existing
    OA-tagged handler on the root logger AND for the OA-installed
    factory marker on the current global factory).

    The user retains responsibility for providing the
    :class:`LoggerProvider` (typically built with their preferred
    exporter — :class:`InMemoryLogRecordExporter` for tests,
    :class:`OTLPLogExporter` for production).
    """
    from opentelemetry.instrumentation.logging.handler import LoggingHandler

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
    # Idempotency #2: don't stack the LogRecord factory.
    _install_correlation_id_factory()


__all__ = [
    "install_log_bridge",
]
