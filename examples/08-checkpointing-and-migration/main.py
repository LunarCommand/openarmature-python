"""openarmature demo: a lunar-mission planning pipeline that checkpoints
its progress, then resumes under an upgraded state schema.

**Use case:** A multi-step planning pipeline drafts a lunar mission plan
(objective, crew size, timeline). It writes a checkpoint after every
step so a crash or restart can pick up where it left off. Some time
later, you add a new analysis step (risk assessment) and a new state
field (``risk_assessment``) to support it. Resuming an old checkpoint
shouldn't require re-running the work that already finished — and
shouldn't fail because the saved state has the old shape.

That's exactly what state migration is for. The pipeline runs once
against the v1 schema, the checkpoint persists, and the v2 schema
declares a migration from v1 that backfills the new field. The v2
graph resumes from the v1 checkpoint, the migration runs once on the
loaded state, and execution picks up at the new node.

**What's interesting in the implementation:**

- ``SQLiteCheckpointer(path, serialization="json")`` writes records to
  a SQLite file in JSON mode. JSON is the migration-eligible
  serialization — it lets the engine load the saved state as a plain
  dict, apply migrations, and re-deserialize against the current
  state class. ``pickle`` mode is faster but can't bridge schemas.
- ``GraphBuilder.with_checkpointer(...)`` wires the checkpointer to
  the graph. The engine then fires a save at every ``completed``
  event for outermost and subgraph-internal nodes.
- ``State.schema_version`` is a ``ClassVar[str]`` declared on the
  state class. Empty string is the "no migration support" sentinel;
  any non-empty value opts the class into the migration registry.
- ``GraphBuilder.with_state_migration(from_version, to_version,
  migrate)`` registers one edge of the migration chain. The
  ``migrate`` callable receives the saved state as a dict and returns
  the dict at the new schema. Pure function; no I/O, no side effects.
- ``compiled.invoke(state, resume_invocation=<id>)`` resumes from a
  saved record. The engine reads the record, applies any registered
  migration chain that bridges the saved ``schema_version`` to the
  current state class's, and continues execution from the first node
  whose ``completed`` event isn't in the record.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/08-checkpointing-and-migration
    LLM_API_KEY=sk-... uv run python main.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import Field

from openarmature.checkpoint import CheckpointFilter, SQLiteCheckpointer
from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    State,
    append,
)
from openarmature.llm import OpenAIProvider, SystemMessage, UserMessage

_provider_instance: OpenAIProvider | None = None


def _get_provider() -> OpenAIProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OpenAIProvider(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("LLM_API_KEY") or None,
        )
    return _provider_instance


async def _chat(system: str, user: str) -> str:
    response = await _get_provider().complete(
        [SystemMessage(content=system), UserMessage(content=user)],
    )
    return (response.message.content or "").strip()


# ---------------------------------------------------------------------------
# Phase 1: v1 schema + v1 graph
# ---------------------------------------------------------------------------
# The v1 schema doesn't have ``risk_assessment``. The v1 graph is
# objective → crew_size → timeline → END. A real codebase would have
# this as ``state.py`` and ``main.py`` until v2 came along; here both
# generations live in the same file so the demo can replay both.


class MissionPlanStateV1(State):
    schema_version: ClassVar[str] = "v1"

    destination: str = ""
    objective: str = ""
    crew_size: int = 0
    timeline: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


async def define_objective_v1(s: MissionPlanStateV1) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "You are a mission planner. Given a lunar destination, state the "
            "single primary objective of a notional crewed mission there in one "
            "tight sentence. No preamble."
        ),
        user=s.destination,
    )
    return {"objective": content, "trace": ["define_objective"]}


async def size_crew_v1(s: MissionPlanStateV1) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "You are a mission planner. Given the objective below, reply with "
            "the optimal crew size as a single integer between 2 and 8. No "
            "other text."
        ),
        user=s.objective,
    )
    digits = "".join(c for c in content if c.isdigit())
    n = int(digits) if digits else 4
    return {"crew_size": max(2, min(8, n)), "trace": ["size_crew"]}


async def draft_timeline_v1(s: MissionPlanStateV1) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "You are a mission planner. Given the objective and crew size, "
            "draft a high-level timeline as a single sentence covering launch, "
            "lunar transit, surface operations, and return. No preamble."
        ),
        user=f"Objective: {s.objective}\nCrew size: {s.crew_size}",
    )
    return {"timeline": content, "trace": ["draft_timeline"]}


def build_graph_v1(checkpointer: SQLiteCheckpointer) -> CompiledGraph[MissionPlanStateV1]:
    return (
        GraphBuilder(MissionPlanStateV1)
        .add_node("define_objective", define_objective_v1)
        .add_node("size_crew", size_crew_v1)
        .add_node("draft_timeline", draft_timeline_v1)
        .add_edge("define_objective", "size_crew")
        .add_edge("size_crew", "draft_timeline")
        .add_edge("draft_timeline", END)
        .set_entry("define_objective")
        .with_checkpointer(checkpointer)
        .compile()
    )


# ---------------------------------------------------------------------------
# Phase 2: v2 schema + migration + v2 graph
# ---------------------------------------------------------------------------
# v2 adds a ``risk_assessment`` field and a new ``assess_risks`` node at
# the end of the pipeline. The migration backfills ``risk_assessment``
# with an empty string for v1 records — the new node will fill it in
# when resume executes.


class MissionPlanStateV2(State):
    schema_version: ClassVar[str] = "v2"

    destination: str = ""
    objective: str = ""
    crew_size: int = 0
    timeline: str = ""
    risk_assessment: str = ""  # NEW in v2
    trace: Annotated[list[str], append] = Field(default_factory=list)


def migrate_v1_to_v2(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Backfill the new ``risk_assessment`` field with an empty string.

    Pure function: takes the saved state as a dict, returns the dict at
    the new schema. The engine reads the v1 record, applies this
    function, and re-deserializes against MissionPlanStateV2.

    Multi-version chains: a third schema (v3) would add a second
    migration function (``migrate_v2_to_v3``) and a second
    ``builder.with_state_migration("v2", "v3", migrate_v2_to_v3)``
    call. The framework's MigrationRegistry runs a BFS over the
    registered edges to find the shortest chain from the saved
    record's ``schema_version`` to the current state class's. A v1
    record loaded under a v3 graph would run v1->v2 then v2->v3
    automatically; no caller-side composition required. If two
    distinct edges with the same ``(from, to)`` pair exist, or two
    distinct shortest paths exist for one resolution, the registry
    raises ``CheckpointStateMigrationChainAmbiguous`` at registration
    or resume time.
    """
    return {**state_dict, "risk_assessment": ""}


async def define_objective_v2(s: MissionPlanStateV2) -> Mapping[str, Any]:
    # Same body as v1; included so v2 builds a complete graph. When
    # resuming a saved record whose ``define_objective`` already
    # completed, the engine skips this node and starts from the first
    # un-completed step.
    return await define_objective_v1(s)  # type: ignore[arg-type]


async def size_crew_v2(s: MissionPlanStateV2) -> Mapping[str, Any]:
    return await size_crew_v1(s)  # type: ignore[arg-type]


async def draft_timeline_v2(s: MissionPlanStateV2) -> Mapping[str, Any]:
    return await draft_timeline_v1(s)  # type: ignore[arg-type]


async def assess_risks_v2(s: MissionPlanStateV2) -> Mapping[str, Any]:
    """The new step v2 introduces — names the top risk for the plan."""
    content = await _chat(
        system=(
            "You are a mission planner. Given the timeline below, identify "
            "the single highest-priority risk in one short sentence. No "
            "preamble."
        ),
        user=s.timeline,
    )
    return {"risk_assessment": content, "trace": ["assess_risks"]}


def build_graph(checkpointer: SQLiteCheckpointer | None = None) -> CompiledGraph[MissionPlanStateV2]:
    """Build the v2 graph with checkpointing and migration registered.

    The smoke test calls this with no checkpointer; main() passes a real
    one. Either path produces a compilable graph.
    """
    builder = (
        GraphBuilder(MissionPlanStateV2)
        .add_node("define_objective", define_objective_v2)
        .add_node("size_crew", size_crew_v2)
        .add_node("draft_timeline", draft_timeline_v2)
        .add_node("assess_risks", assess_risks_v2)
        .add_edge("define_objective", "size_crew")
        .add_edge("size_crew", "draft_timeline")
        .add_edge("draft_timeline", "assess_risks")
        .add_edge("assess_risks", END)
        .set_entry("define_objective")
        .with_state_migration("v1", "v2", migrate_v1_to_v2)
    )
    if checkpointer is not None:
        builder = builder.with_checkpointer(checkpointer)
    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    destination = "Lunar South Pole"

    # SQLite checkpointer in JSON mode (the migration-eligible
    # serialization). A real app would point at a persistent path; the
    # demo uses TemporaryDirectory so the DB file + folder get cleaned
    # up on exit (happy path or exception) without leaving cruft in /tmp.
    with tempfile.TemporaryDirectory(prefix="oa-checkpoint-demo-") as db_dir:
        db_path = Path(db_dir) / "checkpoints.sqlite"
        checkpointer = SQLiteCheckpointer(path=db_path, serialization="json")

        try:
            print("=" * 72)
            print("Phase 1 — invoke v1 graph; checkpoints save after every node")
            print("=" * 72)
            print()
            print(f"  destination:       {destination}")
            print(f"  checkpoint db:     {db_path}")
            print()

            # Pass a deterministic correlation_id so phase 2 can find the
            # invocation's saved records via the checkpoint filter. Without a
            # caller-supplied correlation_id, invoke() generates a UUIDv4.
            run_id = "demo-mission-plan-1"

            graph_v1 = build_graph_v1(checkpointer)
            initial_v1 = MissionPlanStateV1(destination=destination)
            final_v1 = await graph_v1.invoke(initial_v1, correlation_id=run_id)
            await graph_v1.drain()

            # Look up the saved record's invocation_id by correlation_id. The
            # invocation_id is generated by invoke() and isn't exposed on the
            # returned state; finding it through the checkpointer's list API
            # is the canonical lookup path.
            summaries = list(await checkpointer.list(CheckpointFilter(correlation_id=run_id)))
            assert summaries, "expected at least one saved checkpoint"
            invocation_id = summaries[-1].invocation_id

            print("v1 result:")
            print(f"  objective:  {final_v1.objective}")
            print(f"  crew_size:  {final_v1.crew_size}")
            print(f"  timeline:   {final_v1.timeline}")
            print()
            print(f"  v1 invocation_id: {invocation_id}")
            print()

            print("=" * 72)
            print("Phase 2 — invoke v2 graph with resume; v1->v2 migration runs")
            print("=" * 72)
            print()
            print("  v2 adds:    risk_assessment field + assess_risks node")
            print("  migration:  backfills risk_assessment='' for v1 records")
            print()

            graph_v2 = build_graph(checkpointer)
            # Resume from the v1 invocation. The engine reads the saved record,
            # applies migrate_v1_to_v2, re-deserializes against
            # MissionPlanStateV2, and continues at the first uncompleted node
            # (assess_risks — the v1 pipeline's three nodes are all in
            # completed_positions, the new v2 node is not).
            final_v2 = await graph_v2.invoke(
                MissionPlanStateV2(destination=destination),
                resume_invocation=invocation_id,
            )
            await graph_v2.drain()

            print("v2 result after resume:")
            print(f"  objective:        {final_v2.objective}")
            print(f"  crew_size:        {final_v2.crew_size}")
            print(f"  timeline:         {final_v2.timeline}")
            print(f"  risk_assessment:  {final_v2.risk_assessment}")
            print()
            print(f"  trace: {final_v2.trace}")
            print()
            print(
                "The v1 nodes appear once each in v1's trace and NOT in v2's "
                "trace — they were skipped on resume because completed_positions "
                "already covered them. Only assess_risks ran in phase 2."
            )
        finally:
            if _provider_instance is not None:
                await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
