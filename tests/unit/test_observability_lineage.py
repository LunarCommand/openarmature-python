"""Unit tests for the shared observer-side lineage predicates.

These helpers are consumed identically by the OTel and Langfuse
observers, so they are pinned here at their own level rather than only
through whichever observer happens to exercise them.
"""

from __future__ import annotations

from dataclasses import dataclass

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
