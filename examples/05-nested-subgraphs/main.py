"""openarmature demo: two-level subgraph nesting and the depth invariants.

**Use case:** Show what observer events look like when subgraphs are nested
TWO levels deep. The first four examples cover composition basics; this one
zooms in on what happens at depth > 1, where the engine's namespace chain,
parent-state stack, and step counter all extend across multiple subgraph
boundaries.

**Demonstrates:** Spec graph-engine §6 depth invariants —

- `len(parent_states) == len(namespace) - 1` at every depth, including 3.
- `step` is monotonic across both subgraph boundaries (no resets).
- `parent_states[k]` is the k-th containing graph's state at the moment
  that graph entered the subgraph leading to this event. Snapshots are
  stable for the duration of the inner run.
- Subgraph state shape is the SUBGRAPH's schema (not the parent's),
  so `pre_state`/`post_state` carry the inner graph's fields.

The bodies are trivial integer updates so the observer output is short
and predictable. Read the printed output top-to-bottom — depth is shown
both numerically and via indentation, and the parent-states list grows
as the engine descends and shrinks as it returns.

Run with:
    uv run python main.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    NodeEvent,
    State,
    append,
)

# ---------------------------------------------------------------------------
# State schemas — same shape at every level so default field-name matching
# projects values across boundaries.
# ---------------------------------------------------------------------------


class InnerState(State):
    v: int = 0
    trace: Annotated[list[str], append] = Field(default_factory=list)


class MiddleState(State):
    v: int = 0
    trace: Annotated[list[str], append] = Field(default_factory=list)


class OuterState(State):
    v: int = 0
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Node bodies — trivial updates. The numbers are chosen so it's easy to see
# at a glance which level a value came from in the final state:
#   1, 999     → outer
#   10, 20     → middle
#   100, 200   → inner
# ---------------------------------------------------------------------------


async def outer_a(_: OuterState) -> Mapping[str, Any]:
    return {"v": 1, "trace": ["outer_a"]}


async def outer_b(_: OuterState) -> Mapping[str, Any]:
    return {"v": 999, "trace": ["outer_b"]}


async def mid_x(_: MiddleState) -> Mapping[str, Any]:
    return {"v": 10, "trace": ["mid_x"]}


async def mid_y(_: MiddleState) -> Mapping[str, Any]:
    return {"v": 20, "trace": ["mid_y"]}


async def inner_p(_: InnerState) -> Mapping[str, Any]:
    return {"v": 100, "trace": ["inner_p"]}


async def inner_q(_: InnerState) -> Mapping[str, Any]:
    return {"v": 200, "trace": ["inner_q"]}


# ---------------------------------------------------------------------------
# Graph builders — innermost first, since each outer graph references the
# inner one as a SubgraphNode and needs it compiled at build time.
# ---------------------------------------------------------------------------


def build_inner() -> CompiledGraph[InnerState]:
    builder: GraphBuilder[InnerState] = GraphBuilder(InnerState)
    builder.set_entry("inner_p")
    builder.add_node("inner_p", inner_p)
    builder.add_node("inner_q", inner_q)
    builder.add_edge("inner_p", "inner_q")
    builder.add_edge("inner_q", END)
    return builder.compile()


def build_middle(inner: CompiledGraph[InnerState]) -> CompiledGraph[MiddleState]:
    builder: GraphBuilder[MiddleState] = GraphBuilder(MiddleState)
    builder.set_entry("mid_x")
    builder.add_node("mid_x", mid_x)
    builder.add_subgraph_node("mid_inner", inner)
    builder.add_node("mid_y", mid_y)
    builder.add_edge("mid_x", "mid_inner")
    builder.add_edge("mid_inner", "mid_y")
    builder.add_edge("mid_y", END)
    return builder.compile()


def build_outer(middle: CompiledGraph[MiddleState]) -> CompiledGraph[OuterState]:
    builder: GraphBuilder[OuterState] = GraphBuilder(OuterState)
    builder.set_entry("outer_a")
    builder.add_node("outer_a", outer_a)
    builder.add_subgraph_node("outer_mid", middle)
    builder.add_node("outer_b", outer_b)
    builder.add_edge("outer_a", "outer_mid")
    builder.add_edge("outer_mid", "outer_b")
    builder.add_edge("outer_b", END)
    return builder.compile()


def build_graph() -> CompiledGraph[OuterState]:
    """Top-level graph factory — the convention every example exposes for
    CI smoke validation. Builds inner first, then middle (which references
    inner), then outer (which references middle).
    """
    return build_outer(build_middle(build_inner()))


# ---------------------------------------------------------------------------
# Observer — formats events so depth invariants are visually obvious.
# ---------------------------------------------------------------------------


def _fmt(state: Any) -> str:
    """Compact one-line state dump."""
    if state is None:
        return "—"
    return f"v={state.v} trace={state.trace}"


async def depth_observer(event: NodeEvent) -> None:
    """Print every event with depth-aware indentation.

    The leading spaces visualize how deep into the nested subgraphs this
    event came from. Number of `parent_states` entries always equals
    `len(namespace) - 1` per the §6 invariant.
    """
    depth = len(event.namespace)
    indent = "  " * (depth - 1)
    ns = " > ".join(event.namespace)
    parents_summary = " | ".join(_fmt(p) for p in event.parent_states) if event.parent_states else "(none)"

    if event.phase == "started":
        line = (
            f"{indent}[step {event.step}] depth={depth}  ns=[{ns}]\n"
            f"{indent}    started   pre={_fmt(event.pre_state)}\n"
            f"{indent}    parents:  {parents_summary}"
        )
    else:  # completed
        if event.error is not None:
            line = (
                f"{indent}    completed pre={_fmt(event.pre_state)}  "
                f"ERROR={type(event.error).__name__}: {event.error}"
            )
        else:
            line = f"{indent}    completed pre={_fmt(event.pre_state)}  post={_fmt(event.post_state)}"
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    outer = build_graph()

    # Observer attached to the OUTER graph fires for every node executed
    # during this invocation, including subgraph-internal nodes at depth
    # 2 and 3. (See example 04 for graph-attached vs invocation-scoped
    # subtleties; this example uses one observer to keep the output clean.)
    outer.attach_observer(depth_observer)

    print("=" * 72)
    print("Two-level subgraph nesting — observer events at depths 1, 2, 3")
    print("=" * 72)
    print()
    print("Read top-to-bottom. Indentation = depth. Watch the parents list")
    print("grow as the engine descends and shrink as it returns.")
    print()

    try:
        final = await outer.invoke(OuterState())
    finally:
        # Drain MUST be awaited in short-lived processes per spec §6 — without
        # it, this script could exit before the delivery worker finishes
        # processing the queue, losing late events. In a `finally` so failure
        # events flush even if invoke() raises.
        await outer.drain()

    print()
    print("=" * 72)
    print("Final outer state (after both subgraphs project back via")
    print("default field-name matching):")
    print(f"  v     = {final.v}")
    print(f"  trace = {final.trace}")
    print("=" * 72)
    print()
    print("Notice in the events above:")
    print("  - step counter never resets across subgraph boundaries (0..5)")
    print("  - namespace length matches depth at every event")
    print("  - parent_states length = depth - 1, always")
    print("  - inner_p and inner_q share the same parents list because")
    print("    neither outer nor middle is stepping while inner runs")


if __name__ == "__main__":
    asyncio.run(main())
