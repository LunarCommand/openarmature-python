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

import inspect
import logging
from typing import TYPE_CHECKING, Any, cast

from ..diagnostics import EVENT_NAME_ATTR as _EVENT_NAME_ATTR

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
    if _otel_logs_handler_already_bridges(root, provider):
        # An application that wired its own OTel handler gets no second one, but
        # it still needs the event-name lift: without this the field is never
        # populated in the setup this module documents as typical, and the name
        # survives only as an attribute.
        _retrofit_event_name_lift(root)
    else:
        handler_cls = _event_name_handler_class(_InstrLoggingHandler)
        handler = handler_cls(level=level, logger_provider=provider)
        # Direct assignment isn't typed on LoggingHandler; route
        # through ``object.__setattr__`` to avoid pyright's strict
        # attribute-access check without losing the idempotency-
        # marker behavior.
        object.__setattr__(handler, "_openarmature_installed", True)
        root.addHandler(handler)
    # Idempotency #2: don't stack the LogRecord factory.
    _install_correlation_id_factory()


def _accepts_one_record(method: Any) -> bool:
    """True iff ``method`` still takes just ``self`` and the record."""
    # A name check alone is not enough. The override hard-codes
    # `super()._translate(record)`, so a seam that grew a parameter raises
    # TypeError out of every `logging` call in the process, this handler being
    # on the root logger. Shape-check it and decline the subclass instead.
    #
    # Two positionals, because the lookup is on the CLASS and so includes
    # `self`. A keyword-only required parameter fails the check for the same
    # reason a second positional does: the hard-coded call cannot supply it.
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return False
    required = [p for p in params if p.default is inspect.Parameter.empty]
    positional = [p for p in required if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(required) == len(positional) == 2


_logger = logging.getLogger("openarmature.observability")


# Set on a generated handler class so a repeat call recognises its own work.
# `hasattr(base, "_translate")` cannot: a lifted class has one, being ours.
_LIFT_MARKER = "_openarmature_event_name_lift"


def _otel_logs_handler_classes() -> tuple[type[Any], ...]:
    """The OTel logs handler classes a root-logger handler may be one of."""
    # Two classes named LoggingHandler exist in the OTel Python tree, the SDK's
    # and the instrumentation package's, and an application may have attached
    # either.
    from opentelemetry.instrumentation.logging.handler import (
        LoggingHandler as _InstrLoggingHandler,
    )
    from opentelemetry.sdk._logs import LoggingHandler as _SDKLoggingHandler

    return (_SDKLoggingHandler, _InstrLoggingHandler)


def _retrofit_event_name_lift(root: logging.Logger) -> None:
    """Give an already-attached OTel logs handler the event-name lift."""
    # Re-classing rather than replacing: the handler is the application's, with
    # its own level, filters and formatter, and swapping it would discard them.
    #
    # This modifies an object the caller constructed, so it says so. The
    # alternatives are worse: adding a second handler double-exports every log
    # record to the same pipeline, and doing nothing leaves §7's field unset in
    # the setup this module documents as typical, with no signal at all.
    for handler in list(root.handlers):
        if not isinstance(handler, _otel_logs_handler_classes()):
            continue
        original = type(handler)
        lifted = _event_name_handler_class(original)
        if lifted is original:
            # The seam is gone or has changed shape. Nothing is modified, and
            # the names ride as attributes only.
            _logger.warning(
                "%s on the root logger cannot carry openarmature's diagnostic event names: "
                "its record-translation hook is missing or has changed shape. The names are "
                "still set as log-record attributes, but not on the OTel LogRecord's "
                "EventName field",
                original.__name__,
            )
            continue
        handler.__class__ = lifted
        _logger.warning(
            "openarmature re-classed the %s you attached to the root logger, so its records "
            "carry openarmature's diagnostic event names on the OTel LogRecord's EventName "
            "field. Your handler's level, filters and formatter are unchanged; only its class "
            "is, and `type()` on it now reports %s. To avoid this, call install_log_bridge "
            "before attaching your own handler, or attach yours to a different LoggerProvider",
            original.__name__,
            lifted.__name__,
        )


def _event_name_handler_class(base: type[Any]) -> type[Any]:
    """The handler class to bridge with, lifting a record's event name.

    Returns ``base`` unchanged where the upstream handler no longer exposes the
    seam this needs.
    """
    # §7 wants a diagnostic's event name on the OTel LogRecord's `event_name`
    # FIELD. Neither OTel logging handler populates it: both map every stdlib
    # record attribute into `attributes` and leave the field unset, so a name
    # passed via `extra=` arrives as an attribute and the field stays empty.
    #
    # `_translate` is the handlers' own private surface, so subclassing it
    # reaches past the public API. Guarded on the method still existing, because
    # `super()._translate(...)` inside an override raises AttributeError once it
    # does not, which would break logging rather than degrade. Without the
    # subclass the name rides as an attribute only.
    if getattr(base, _LIFT_MARKER, False):
        # Already lifted. Subclassing again would work, since the outer override
        # finds the field set and skips, but each pass adds an MRO entry that
        # never goes away.
        return base
    inherited = getattr(base, "_translate", None)
    if inherited is None or not _accepts_one_record(inherited):
        return base

    class _EventNameHandler(base):  # type: ignore[misc, valid-type]
        def _translate(self, record: logging.LogRecord) -> Any:
            # `base` is `type[Any]` so the checker cannot see through `super()`;
            # the seam is checked for both name and shape above.
            raw = super()._translate(record)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            translated = cast("Any", raw)
            name = getattr(record, _EVENT_NAME_ATTR, None)
            if isinstance(name, str) and getattr(translated, "event_name", None) is None:
                try:
                    translated.event_name = name
                except AttributeError:
                    # A LogRecord with no such field: the name still rides as an
                    # attribute, so nothing that was there before is lost.
                    pass
            return translated

    setattr(_EventNameHandler, _LIFT_MARKER, True)
    return _EventNameHandler


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
    handler_classes = _otel_logs_handler_classes()
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
