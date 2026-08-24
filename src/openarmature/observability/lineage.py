# Spec: cross-cutting helpers for the observer-side lineage match
# introduced by proposal 0040 (metadata-augmentation event open-span
# update, observability §3.4 + §6).

"""Tuple-prefix predicates used by the OTel + Langfuse observers to
match an augmentation event's namespace against the namespaces of
open spans / observations. Shared so both observers express the
ancestor-or-equal rule identically.
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["AugmentationLineage", "is_outermost_serial", "is_prefix_or_equal", "is_strict_prefix"]


def is_strict_prefix(prefix: tuple[str, ...], full: tuple[str, ...]) -> bool:
    """True iff ``prefix`` is a strict prefix of ``full`` (NOT equal)."""
    return len(prefix) < len(full) and full[: len(prefix)] == prefix


def is_prefix_or_equal(prefix: tuple[str, ...], full: tuple[str, ...]) -> bool:
    """True iff ``prefix`` is a prefix of (or equal to) ``full``."""
    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


class AugmentationLineage(Protocol):
    """The lineage axes an augmentation event carries, on both the
    per-depth chains and the augmenter's own scalars.
    """

    @property
    def fan_out_index(self) -> int | None: ...

    @property
    def branch_name(self) -> str | None: ...

    @property
    def fan_out_index_chain(self) -> tuple[int | None, ...]: ...

    @property
    def branch_name_chain(self) -> tuple[str | None, ...]: ...


def is_outermost_serial(event: AugmentationLineage) -> bool:
    """True iff no fan-out instance and no parallel-branches branch sits
    on the augmenter's call-stack path.
    """
    # Per observability §3.4 this decides whether the invocation span is a
    # shared parent (MUST NOT update) or an ancestor on the augmenter's path
    # (MUST update).  Both observers ask it, so it lives here: they each held
    # their own copy, and the copy read only the CHAINS.  A branch's identity
    # reaches the chain only when the branch DESCENDS -- a callable branch
    # never does, so its augmenter's chain is empty and the chain-only test
    # called it pure-serial, putting every callable branch's metadata on the
    # shared invocation span.  The scalars are part of the predicate, not a
    # shortcut for the chains.
    return (
        all(fi is None for fi in event.fan_out_index_chain)
        and all(bn is None for bn in event.branch_name_chain)
        and event.fan_out_index is None
        and event.branch_name is None
    )
