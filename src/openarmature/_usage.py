# Spec: the figure-level rule "a malformed figure is not a reported figure",
# shared by retrieval-provider (proposal 0100) and llm-provider (proposal 0101).
# A usage figure present on the wire but not a non-negative integer is nulled --
# never raised on, coerced, clamped, or repaired (a repaired counter is
# indistinguishable from a reported one). Sits at the package root because the
# rule spans both capabilities and the mechanics are identical.

"""Shared usage-figure parsing for the wire mappings."""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def nonneg_int(value: Any, *, field: str, source: str = "provider") -> int | None:
    """Return ``value`` when it is a non-negative int, else ``None``.

    ``None`` (the figure was not reported) returns quietly; any other
    non-conforming value returns ``None`` after a WARNING log. ``field`` names
    the figure and ``source`` names the misbehaving component in that log line.
    """
    # bool is an int subclass, so exclude it explicitly.
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if value is not None:
        _log.warning(
            "%s reported a %s that is not a non-negative int (got %s %.80r); recording usage as unknown",
            source,
            field,
            type(value).__name__,
            value,
        )
    return None
