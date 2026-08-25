"""Unit tests for the shared observer-side lineage predicates.

These helpers are consumed identically by the OTel and Langfuse
observers, so they are pinned here at their own level rather than only
through whichever observer happens to exercise them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from openarmature.observability.lineage import is_outermost_serial


@dataclass(frozen=True)
class _Lineage:
    fan_out_index: int | None = None
    branch_name: str | None = None
    fan_out_index_chain: tuple[int | None, ...] = ()
    branch_name_chain: tuple[str | None, ...] = ()


# The four shapes an augmenter can occupy, and where each one's identity
# actually lives.  A dispatch that DESCENDS writes its identity into the
# per-depth chain; one that does not leaves the chain empty and carries the
# identity only on the scalar.  Reading the chains alone therefore answers
# correctly for the descending shapes and wrongly for the non-descending
# ones -- observability §3.4's "at least one fan-out or parallel-branches
# dispatch is on the augmenter's call-stack path" covers both.
@pytest.mark.parametrize(
    ("shape", "lineage", "expected"),
    [
        ("pure serial", _Lineage(), True),
        (
            "serial through subgraph wrappers",
            _Lineage(fan_out_index_chain=(None, None), branch_name_chain=(None, None)),
            True,
        ),
        (
            "subgraph branch (identity in the chain)",
            _Lineage(branch_name="b", branch_name_chain=("b",), fan_out_index_chain=(None,)),
            False,
        ),
        (
            "fan-out instance (identity in the chain)",
            _Lineage(fan_out_index=1, fan_out_index_chain=(1,), branch_name_chain=(None,)),
            False,
        ),
        (
            "callable branch (identity only on the scalar)",
            _Lineage(branch_name="a"),
            False,
        ),
    ],
)
def test_outermost_serial_covers_every_shape(shape: str, lineage: _Lineage, expected: bool) -> None:
    assert is_outermost_serial(lineage) is expected, shape


def test_a_scalar_alone_is_enough_to_disqualify() -> None:
    # The regression this predicate exists to prevent: empty chains plus a
    # set scalar is a callable branch, not the outermost serial context.
    assert is_outermost_serial(_Lineage(branch_name="a")) is False
    assert is_outermost_serial(_Lineage(fan_out_index=0)) is False


# Both copies of the lineage key builder, parametrized by MODULE NAME rather than
# by imported function. Calling `pytest.importorskip` while building the
# parametrize argument runs it at module import, and its skip is module-scoped:
# without the langfuse extra the whole ~5,300-line OTel module would collapse to
# a single skip, taking ~95 unrelated tests with it. Importing inside the test
# body keeps the skip to the tests that actually need the extra.
_BRANCH_KEY_MODULES = [
    ("otel", "openarmature.observability.otel.observer"),
    ("langfuse", "openarmature.observability.langfuse.observer"),
]


def _branch_key(module_name: str) -> Any:
    module = pytest.importorskip(module_name)
    return module._branch_dispatch_key  # noqa: SLF001


@pytest.mark.parametrize(("label", "module_name"), _BRANCH_KEY_MODULES)
def test_branch_dispatch_key_pads_chains_shallower_than_the_prefix(label: str, module_name: str) -> None:
    key = _branch_key(module_name)
    # An orphan provider call issued from branch middleware carries EMPTY
    # lineage chains, while the dispatch span was registered from an inner node
    # event whose chains are padded to the namespace depth. Both denote "no
    # enclosing fan-out at that depth", so they MUST produce the same key; when
    # they did not, the lookup missed and the orphan span fell through to the
    # invocation root (conformance fixture 152).
    prefix = ("dispatcher",)
    registered = key(prefix, (None,), (), "branch_a")
    from_orphan = key(prefix, (), (), "branch_a")
    assert from_orphan == registered, f"{label}: shallow chains must normalize to the registered key"


@pytest.mark.parametrize(("label", "module_name"), _BRANCH_KEY_MODULES)
def test_branch_dispatch_key_pads_branch_chain_shallower_than_the_prefix(
    label: str, module_name: str
) -> None:
    key = _branch_key(module_name)
    # The branch-name half of the same normalization, which the fan-out test
    # above does not reach: it uses a depth-1 prefix, where the branch slice is
    # `chain[:0]` and is empty whether padded or not. A depth-2 prefix slices
    # `chain[:1]`, so a caller whose branch chain is shorter than `n - 1` builds
    # an unpadded key while the span was registered with a padded one.
    #
    # Without this, deleting the `branches` padding line from BOTH copies leaves
    # the entire suite green; deleting it from one is caught only incidentally,
    # by the agreement test noticing the copies diverged.
    prefix = ("outer", "dispatcher")
    registered = key(prefix, (None, None), (None,), "branch_a")
    from_orphan = key(prefix, (None, None), (), "branch_a")
    assert from_orphan == registered, (
        f"{label}: a branch chain shallower than the prefix must normalize to the registered key"
    )


@pytest.mark.parametrize(("label", "module_name"), _BRANCH_KEY_MODULES)
def test_branch_dispatch_key_still_discriminates_real_lineages(label: str, module_name: str) -> None:
    key = _branch_key(module_name)
    # The padding must not collapse genuinely different enclosing lineages: a pb
    # node inside outer fan-out instance 0 and the same node inside instance 1
    # are different dispatch spans and must not share a key.
    prefix = ("outer", "dispatcher")
    assert key(prefix, (0, None), (), "b") != key(prefix, (1, None), (), "b"), (
        f"{label}: distinct enclosing fan-out instances must not collide"
    )
    assert key(prefix, (None, None), ("x",), "b") != key(prefix, (None, None), ("y",), "b"), (
        f"{label}: distinct enclosing branch chains must not collide"
    )
    assert key(prefix, (None, None), (), "a") != key(prefix, (None, None), (), "b"), (
        f"{label}: distinct branch names must not collide"
    )


@pytest.mark.parametrize(("label", "module_name"), _BRANCH_KEY_MODULES)
def test_branch_dispatch_key_reads_its_own_depth_not_the_innermost_scalar(
    label: str, module_name: str
) -> None:
    key = _branch_key(module_name)
    # The scalar `branch_name` is always the INNERMOST branch on the event.  When
    # the key's position is an OUTER pb, the identity has to come from that
    # position in the chain instead.  Reading the scalar unconditionally was
    # right only when the two coincided, so every single-level shape worked and
    # nesting collided.
    #
    # An event inside inner branch "x", resolving the OUTER pb at ("o",): the
    # branch actually descended through is "y", carried at chain position 0.
    outer = key(("o",), (None,), ("y",), "x")
    assert outer[3] == "y", f"{label}: outer key must name the branch at its own depth, got {outer[3]!r}"
    # And it must equal the key built while inside that outer branch directly,
    # or the span registered on the way down is not the one found on the way up.
    assert outer == key(("o",), (None,), ("y",), "y"), (
        f"{label}: outer key must not depend on the inner scalar"
    )
    # The inner pb still reads its own scalar, because the callable branch never
    # descends so the chain does not reach position 1.
    inner = key(("o", "i"), (None, None), ("y",), "x")
    assert inner[3] == "x", f"{label}: innermost position falls back to the scalar, got {inner[3]!r}"


@pytest.mark.parametrize(("label", "module_name"), _BRANCH_KEY_MODULES)
def test_branch_dispatch_key_repeated_name_across_depths_does_not_collide(
    label: str, module_name: str
) -> None:
    key = _branch_key(module_name)
    # The shape that made a `when`-skipped outer branch acquire a span: an inner
    # branch reusing an ancestor's name.  Outer branch "x" (skipped) and the
    # inner "x" reached via outer "y" must be different keys.
    skipped_outer = key(("o",), (None,), (), "x")
    inner_reached_via_y = key(("o",), (None,), ("y",), "x")
    assert skipped_outer != inner_reached_via_y, (
        f"{label}: a repeated branch name across depths must not collapse to one key"
    )


def test_both_observers_share_one_branch_dispatch_key_implementation() -> None:
    # This guarded two duplicated copies against drift.  They are no longer
    # duplicated: both observers import one definition from
    # `observability.lineage`, so drift is now structurally impossible rather
    # than merely asserted.  What is worth guarding is that it stays that way.
    #
    # Three consecutive PRs fixed the same defect twice by hand because the copy
    # existed: the chain padding, the outermost-serial predicate, and the
    # depth-keyed read.  A reintroduced local copy would restore that, and a
    # diff-scoped reviewer would see one side of it.
    from openarmature.observability import lineage

    shared = {"_branch_dispatch_key": lineage.branch_dispatch_key, "_dispatch_key": lineage.dispatch_key}
    modules = [(label, pytest.importorskip(name)) for label, name in _BRANCH_KEY_MODULES]
    assert len(modules) == 2, "expected both backends' observer modules to be importable"
    for label, module in modules:
        for attr, canonical in shared.items():
            assert getattr(module, attr) is canonical, (
                f"{label}.{attr} no longer uses the shared `lineage` implementation; "
                "a local copy has been reintroduced and can drift from the other observer"
            )


def test_branch_dispatch_key_shape_over_the_edge_cases() -> None:
    # The behavioural spec of the key, kept from the copies-agree test above.
    # The empty prefix is the one that matters: `n` is 0 there, and reading the
    # branch identity at `n - 1` without a guard indexes the LAST chain entry
    # instead of nothing, which raises on an empty chain.  That is exactly what
    # the first draft of the depth-keyed fix did.
    from openarmature.observability.lineage import branch_dispatch_key as key

    assert key((), (), (), "a") == ((), (), (), "a")
    # A non-empty chain at n == 0: `chain[: n - 1]` is `chain[:-1]`, which drops
    # the LAST entry instead of yielding nothing. The empty-chain row above is
    # the one input where the buggy and correct expressions coincide, so it
    # cannot see this.
    assert key((), (), ("x", "y"), "a") == ((), (), (), "a")
    # Shallower chains pad to the prefix depth rather than staying short.
    assert key(("dispatcher",), (), (), "a") == key(("dispatcher",), (None,), (), "a")
    # Enclosing positions are carried; the own position is not duplicated there.
    assert key(("outer", "dispatcher"), (0, 1), ("x",), "a") == (
        ("outer", "dispatcher"),
        (0, 1),
        ("x",),
        "a",
    )
