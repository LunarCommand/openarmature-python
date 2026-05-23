# Spec: realizes graph-engine §2 (Edge, END concepts). END is a
# distinct engine sentinel (not a reserved node name) — using the
# literal string ``"END"`` as a target fails ``DanglingEdge`` at compile.

"""Edges and the END sentinel.

Edges are static or conditional; each node has exactly one outgoing
edge. `END` is a distinct engine sentinel used as a routing target
to halt execution.

`ConditionalEdge` is generic on the outer graph's state type so the
routing function's parameter is typed against the user's `State`
subclass, not `Any`, at type-check time.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from .state import State


class EndSentinel:
    """Engine-provided sentinel routing target. Use the module-level `END`."""

    def __repr__(self) -> str:
        return "END"


END: Final[EndSentinel] = EndSentinel()


@dataclass(frozen=True)
class StaticEdge:
    """Always routes from `source` to `target`."""

    source: str
    target: str | EndSentinel


@dataclass(frozen=True)
class ConditionalEdge[StateT: State]:
    """Routes from `source` to whichever node `fn(state)` returns. The function
    MUST return either a declared node name or `END`; any other value raises
    `RoutingError` at runtime.
    """

    source: str
    fn: Callable[[StateT], str | EndSentinel]
