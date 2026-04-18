"""Edges and the END sentinel.

Per spec §2 Concepts (Edge, END): edges are static or conditional; each node
has exactly one outgoing edge. END is a distinct engine sentinel (not a
reserved node name) used as a routing target to halt execution.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final


class _EndSentinel:
    """Engine-provided sentinel routing target. Use the module-level `END`."""

    def __repr__(self) -> str:
        return "END"


END: Final[_EndSentinel] = _EndSentinel()


@dataclass(frozen=True)
class StaticEdge:
    """Always routes from `source` to `target`."""

    source: str
    target: str | _EndSentinel


@dataclass(frozen=True)
class ConditionalEdge:
    """Routes from `source` to whichever node `fn(state)` returns. The function
    MUST return either a declared node name or `END`; any other value raises
    `RoutingError` at runtime.
    """

    source: str
    fn: Callable[[Any], str | _EndSentinel]
