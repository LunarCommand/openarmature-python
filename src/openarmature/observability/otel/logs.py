# Spec: realizes observability §7 (logs correlation contract).

"""OTel Logs Bridge integration.

Provides :func:`install_log_bridge`; an opt-in helper that wires
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
    logger fire only for records originating on root (Python's
    propagation walks ancestor handlers but not ancestor filters),
    so a root-logger filter misses every child-logger record. The
    factory hook fires at record construction time, before any
    logger or handler dispatch, so every record gets the attribute
    regardless of which logger originated it.

    Idempotent across both OTel-Logs handler classes. Two different
    classes both named ``LoggingHandler`` exist in the OTel Python
    ecosystem and both bridge stdlib records to the Logs SDK:

    - :class:`opentelemetry.sdk._logs.LoggingHandler` (the SDK class,
      what an application's own logging setup typically installs).
    - :class:`opentelemetry.instrumentation.logging.handler.LoggingHandler`
      (the instrumentation class, what this helper installs).

    Different classes, same OTel-Logs export path. If an application
    has already attached the SDK class against the same
    :class:`LoggerProvider`, calling this helper would attach the
    instrumentation class on top and every record would emit to OTLP
    twice. The check below detects EITHER class against the same
    provider and skips the ``addHandler`` step accordingly; the
    correlation_id factory still installs. Re-calling with no prior
    OA-installed handler is also a no-op via the OA marker check.

    The user retains responsibility for providing the
    :class:`LoggerProvider` (typically built with their preferred
    exporter; :class:`InMemoryLogRecordExporter` for tests,
    :class:`OTLPLogExporter` for production).
    """
    from opentelemetry.instrumentation.logging.handler import (
        LoggingHandler as _InstrLoggingHandler,
    )

    root = logging.getLogger()
    if not _otel_logs_handler_already_bridges(root, provider):
        handler = _InstrLoggingHandler(level=level, logger_provider=provider)
        # Direct assignment isn't typed on LoggingHandler; route
        # through ``object.__setattr__`` to avoid pyright's strict
        # attribute-access check without losing the idempotency-
        # marker behavior.
        object.__setattr__(handler, "_openarmature_installed", True)
        root.addHandler(handler)
    # Idempotency #2: don't stack the LogRecord factory.
    _install_correlation_id_factory()


def _otel_logs_handler_already_bridges(root: logging.Logger, provider: LoggerProvider) -> bool:
    """True iff the root logger already has an OTel-Logs
    ``LoggingHandler`` (SDK class OR instrumentation class) wired to
    ``provider`` — meaning every record will already reach the OTLP
    export path and a second ``addHandler`` here would duplicate.

    Handler-class isinstance covers the case where an application
    attached the SDK handler in its own logging setup; the
    ``_openarmature_installed`` marker covers the case where this
    helper was already called previously. ``_logger_provider`` is
    OTel-private on both handler classes today — if a future SDK
    rename hides it, ``getattr`` returns ``None`` and we conclude
    "doesn't bridge", falling back to adding our own handler. Worst
    case is the pre-fix behavior (potential dup); we never crash.
    """
    from opentelemetry.instrumentation.logging.handler import (
        LoggingHandler as _InstrLoggingHandler,
    )
    from opentelemetry.sdk._logs import LoggingHandler as _SDKLoggingHandler

    handler_classes = (_SDKLoggingHandler, _InstrLoggingHandler)
    for handler in root.handlers:
        if not isinstance(handler, handler_classes):
            continue
        if getattr(handler, "_openarmature_installed", False):
            return True
        if getattr(handler, "_logger_provider", None) is provider:
            return True
    return False


__all__ = [
    "install_log_bridge",
]
