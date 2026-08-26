# Spec: cross-cutting helpers for the observer-side lineage match.
# Prefix predicates + the outermost-serial predicate come from proposal
# 0040 (metadata-augmentation open-span update, observability §3.4 +
# §6); the dispatch-span identity keys realize §4.3 / §5.7 per
# proposals 0044 / 0045 / 0084.

"""Lineage helpers shared by the OTel and Langfuse observers.

Two halves, both here so the two observers cannot express them
differently:

- namespace prefix predicates, and the outermost-serial predicate, used
  to decide which open spans an augmentation event reaches;
- the lineage-aware identity keys for fan-out instance and per-branch
  dispatch spans.

The dispatch keys live here rather than in either observer because each
held its own copy and the same defect had to be fixed twice by hand in
three consecutive changes.
"""

from __future__ import annotations

from typing import Protocol

__all__ = [
    "AugmentationLineage",
    "BranchDispatchKey",
    "DispatchKey",
    "branch_dispatch_key",
    "dispatch_key",
    "is_outermost_serial",
    "is_prefix_or_equal",
    "is_strict_prefix",
]


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


# The fan-out / parallel-branches NODE namespace prefix, plus the fan-out
# instance and branch chains sliced along the path to it.
DispatchKey = tuple[tuple[str, ...], tuple[int | None, ...], tuple[str | None, ...]]

# As above, plus the branch's own name at that position.
BranchDispatchKey = tuple[tuple[str, ...], tuple[int | None, ...], tuple[str | None, ...], str]


def branch_dispatch_key(
    prefix: tuple[str, ...],
    fan_out_index_chain: tuple[int | None, ...],
    branch_name_chain: tuple[str | None, ...],
    branch_name: str,
) -> BranchDispatchKey:
    """Lineage-aware identity key for a per-branch dispatch span at namespace
    ``prefix``.
    """
    # Shared by both observers.  They each held an identical copy, and the same
    # defect had to be fixed twice by hand three times running.
    n = len(prefix)
    # Chains are normalized to the prefix DEPTH in both directions: truncated
    # when longer, padded with None when shorter.  Truncating alone was a
    # defect: an orphan provider call issued from branch middleware carries
    # empty chains and built `(prefix, (), (), branch)` where the span had been
    # registered under `(prefix, (None,), (), branch)`.  Those denote the same
    # lineage, "no enclosing fan-out at that depth", and differed only as tuple
    # keys, so the lookup missed and the orphan fell through to the root.
    fan_out = tuple(fan_out_index_chain[:n]) + (None,) * max(0, n - len(fan_out_index_chain))
    branches = tuple(branch_name_chain[: max(0, n - 1)]) + (None,) * max(0, (n - 1) - len(branch_name_chain))
    # The branch's identity at THIS key's position, which is not always the
    # innermost one.  Reading the scalar unconditionally was correct only when
    # the key's position was the innermost, so every single-level shape worked
    # and nesting collided: resolving an OUTER pb from an event inside an inner
    # branch built the outer key with the INNER branch's name.  Where the two
    # names coincide across depths that key names a different branch entirely,
    # including one that `when` skipped and that must hold no span at all
    # (pipeline-utilities 11.10).
    #
    # The scalar remains the fallback for a branch that does not DESCEND, whose
    # chain never reaches this position: a callable branch, or a wrapper-issued
    # call. `n >= 1` guards an empty prefix, where `n - 1` would index the last
    # chain entry rather than nothing.
    own = branch_name_chain[n - 1] if n >= 1 and len(branch_name_chain) >= n else None
    # `is not None`, not truthiness: an empty-string entry is a name the chain
    # actually carries, and falling back to the scalar there would silently
    # resolve a different branch.
    return (prefix, fan_out, branches, branch_name if own is None else own)


def dispatch_key(
    prefix: tuple[str, ...],
    fan_out_index_chain: tuple[int | None, ...],
    branch_name_chain: tuple[str | None, ...],
) -> DispatchKey:
    """Lineage-aware identity key for a fan-out instance dispatch span at
    namespace ``prefix``.
    """
    # Shared by both observers, alongside `branch_dispatch_key`.
    #
    # Two dispatches at the same namespace but in different enclosing fan-out
    # instances / branches get distinct keys, so a fan-out or pb nested inside
    # an outer fan-out instance does not collide across outer instances.  For a
    # top-level or serial-nested dispatch the enclosing entries are all None, so
    # the key is a stable function of the namespace plus the dispatch's own axis.
    #
    # Unlike `branch_dispatch_key` this carries no separate identity argument,
    # and the reason is a binding difference in the engine rather than descent.
    # `graph/fan_out.py:_bind_instance_lineage` binds fan_out_index, both
    # chains, and branch_name_chain onto the CHILD context for instance
    # middleware, so an instance's index reaches the chain even for a call
    # issued from middleware that never descends.  `graph/parallel_branches.py`
    # binds only the scalar via `_set_branch_name`, leaving both chains at the
    # parent's, which is why the branch helper needs the explicit fallback.
    #
    # NOTE this truncates where `branch_dispatch_key` pads.  That is safe only
    # because every lookup is gated on the fan-out axis at the lookup depth:
    # each call site computes `fi_axis` as the chain entry for that depth and
    # skips the lookup when it is None, which implies the chain already reaches
    # `n`.  Grep `fi_axis` for the sites -- line numbers were tried here and
    # went stale within the same commit that wrote them.  An ungated lookup with
    # a short chain would build a short tuple and miss the padded registration
    # key, which is the orphan-lookup miss fixture 152 exists for.
    n = len(prefix)
    return (prefix, tuple(fan_out_index_chain[:n]), tuple(branch_name_chain[:n]))
