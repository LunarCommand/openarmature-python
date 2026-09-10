# Spec: realizes observability §7 (diagnostic event names, proposal 0121).

"""Stable event names for the log records openarmature emits.

A diagnostic's human-readable message is free to change; its event name is
not. Consumers filter and alert on the name, so treat these as part of the
public surface.
"""

from __future__ import annotations

import logging
from typing import Any

# §7: a log record signalling a condition openarmature specifies carries an
# event name in the `openarmature.` namespace. The names are stable
# identifiers and MUST NOT be reworded once shipped.
#
# The obligation is on the RECORD, which this does not change: where a record
# is emitted it carries its name, and a diagnostic openarmature chooses not to
# emit needs none. `openarmature.langfuse.supplied_client_shared_provider`
# (MAY) has no emitter here, because a caller-supplied client is never
# inspected: §6 mode (a) leaves its provider the caller's responsibility.
LANGFUSE_SHARED_PROVIDER_ACCEPTED = "openarmature.langfuse.shared_provider_accepted"
LANGFUSE_PAYLOAD_SUPPRESSED = "openarmature.langfuse.payload_suppressed"
TOKEN_BUDGET_EXCEEDED = "openarmature.token_budget.exceeded"

# The stdlib LogRecord attribute the name rides on. `install_log_bridge` lifts
# it onto the OTel LogRecord's own `event_name` field, which is where §7 wants
# it; neither OTel logging handler populates that field on its own.
EVENT_NAME_ATTR = "event_name"


def diagnostic(event_name: str) -> dict[str, Any]:
    """Build the ``extra=`` mapping that tags a log record with its event name.

    Use at every call site that emits one of the named diagnostics::

        _logger.warning("...", extra=diagnostic(LANGFUSE_PAYLOAD_SUPPRESSED))
    """
    return {EVENT_NAME_ATTR: event_name}


def event_name_of(record: logging.LogRecord) -> str | None:
    """Return the event name a record carries, or ``None`` for an untagged one."""
    value = getattr(record, EVENT_NAME_ATTR, None)
    return value if isinstance(value, str) else None


__all__ = [
    "EVENT_NAME_ATTR",
    "LANGFUSE_PAYLOAD_SUPPRESSED",
    "LANGFUSE_SHARED_PROVIDER_ACCEPTED",
    "TOKEN_BUDGET_EXCEEDED",
    "diagnostic",
    "event_name_of",
]
