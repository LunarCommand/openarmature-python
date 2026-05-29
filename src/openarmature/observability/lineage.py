# Spec: cross-cutting helpers for the observer-side lineage match
# introduced by proposal 0040 (metadata-augmentation event open-span
# update, observability §3.4 + §6).

"""Tuple-prefix predicates used by the OTel + Langfuse observers to
match an augmentation event's namespace against the namespaces of
open spans / observations. Shared so both observers express the
ancestor-or-equal rule identically.
"""

from __future__ import annotations

__all__ = ["is_prefix_or_equal", "is_strict_prefix"]


def is_strict_prefix(prefix: tuple[str, ...], full: tuple[str, ...]) -> bool:
    """True iff ``prefix`` is a strict prefix of ``full`` (NOT equal)."""
    return len(prefix) < len(full) and full[: len(prefix)] == prefix


def is_prefix_or_equal(prefix: tuple[str, ...], full: tuple[str, ...]) -> bool:
    """True iff ``prefix`` is a prefix of (or equal to) ``full``."""
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix
