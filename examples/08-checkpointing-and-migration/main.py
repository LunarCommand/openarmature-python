"""openarmature demo: a lunar-mission planning pipeline that survives a
mid-pipeline crash and later resumes under an upgraded state schema.

**Use case:** A multi-step planning pipeline drafts a lunar mission plan
(objective, crew size, timeline). Two production-grade reliability
scenarios stack on top of each other:

1. **Crash and resume.** The pipeline writes a checkpoint after every
   step. Mid-run, ``size_crew`` raises a simulated transient failure
   (OOM kill, pod preemption, network blip during the LLM call); the
   first invoke() bubbles a ``NodeException`` at its boundary. The
   define_objective record is durable on disk. A second invoke() with
   ``resume_invocation=<id>`` reads the saved record, skips the
   already-completed node, retries size_crew (which now succeeds), and
   runs through to END.

2. **State migration on resume.** Some time later, you add a new
   analysis step (risk assessment) and a new state field
   (``risk_assessment``) to support it. The v2 schema declares a
   migration from v1 that backfills the new field. The v2 graph
   resumes from the (crash-survived) v1 invocation, the migration runs
   once on the loaded state, and execution picks up at the new node.

**What's interesting in the implementation:**

- ``SQLiteCheckpointer(path, serialization="json")`` writes records to
  a SQLite file in JSON mode. JSON is the migration-eligible
  serialization; it lets the engine load the saved state as a plain
  dict, apply migrations, and re-deserialize against the current
  state class. ``pickle`` mode is faster but can't bridge schemas.
- ``GraphBuilder.with_checkpointer(...)`` wires the checkpointer to
  the graph. The engine fires a save synchronously at every
  ``completed`` event for outermost and subgraph-internal nodes;
  the save returns before the next node starts, so a crash mid-next-
  node can't lose the previous node's record.
- ``NodeException`` reaches the caller from ``invoke()`` when a node
  raises. ``exc.__cause__`` is the original exception; ``exc.node_name``
  identifies the failing node; ``exc.recoverable_state`` is the state
  as it was just before the failing node ran.
- ``compiled.invoke(state, resume_invocation=<id>)`` resumes a saved
  invocation. The engine reads the record, skips nodes whose
  ``completed`` event is already in ``completed_positions``, and
  continues execution from the first uncompleted node.
- ``State.schema_version`` is a ``ClassVar[str]`` declared on the
  state class. Empty string is the "no migration support" sentinel;
  any non-empty value opts the class into the migration registry.
- ``GraphBuilder.with_state_migration(from_version, to_version,
  migrate)`` registers one edge of the migration chain. The
  ``migrate`` callable receives the saved state as a dict and returns
  the dict at the new schema. Pure function; no I/O, no side effects.
  Migration runs once on resume when the saved record's
  ``schema_version`` doesn't match the current state class's.

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
    NodeException,
    State,
    append,
)
from openarmature.llm import OpenAIProvider, SystemMessage, UserMessage

_provider_instance: OpenAIProvider | None = None

# Crash-and-resume drama: the first call to ``size_crew_v1`` raises a
# transient RuntimeError to simulate a mid-pipeline infrastructure
# failure (OOM kill, pod preemption, network blip during the LLM
# round-trip). The second call (during the resumed invocation) runs
# normally so the pipeline can complete.
#
# A real process restart would reset this counter when the OS rebooted
# the worker; the demo keeps the value across phases because both
# phases run in the same Python process. The engine-side invariant on
# display is unchanged: define_objective's saved checkpoint is durable
# across the failure, and the engine skips it on resume.
_size_crew_attempt_count = 0


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
    # Demo crash-trigger; see the _size_crew_attempt_count comment at
    # the top of the module for the framing.
    global _size_crew_attempt_count
    _size_crew_attempt_count += 1
    if _size_crew_attempt_count == 1:
        raise RuntimeError("simulated transient mid-pipeline crash before size_crew completed")
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
# with an empty string for v1 records; the new node will fill it in
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
    """The new step v2 introduces; names the top risk for the plan."""
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
            print("Phase 1 - invoke v1 graph; size_crew crashes; resume picks up")
            print("=" * 72)
            print()
            print(f"  destination:       {destination}")
            print(f"  checkpoint db:     {db_path}")
            print()

            # Pass a deterministic correlation_id so we can look up the
            # invocation's saved records via the checkpoint filter
            # between attempts. Without a caller-supplied correlation_id,
            # invoke() generates a UUIDv4.
            run_id = "demo-mission-plan-1"

            graph_v1 = build_graph_v1(checkpointer)
            initial_v1 = MissionPlanStateV1(destination=destination)

            # First attempt: define_objective completes (its checkpoint
            # saves synchronously before size_crew runs); size_crew
            # raises the simulated transient failure; NodeException
            # bubbles to the invoke() boundary.
            print("  first attempt:")
            try:
                await graph_v1.invoke(initial_v1, correlation_id=run_id)
            except NodeException as exc:
                cause = exc.__cause__
                cause_msg = str(cause) if cause is not None else "<no cause>"
                print(f"    NodeException at node {exc.node_name!r}: {cause_msg}")
            await graph_v1.drain()

            # Look up the saved record's invocation_id by correlation_id.
            # The invocation_id is generated by invoke() and isn't
            # exposed on the returned state (the call raised), so the
            # checkpointer's list API is the canonical lookup path.
            # The list API returns lightweight CheckpointSummary records;
            # load() returns the full CheckpointRecord with
            # completed_positions for the inspection below.
            summaries = list(await checkpointer.list(CheckpointFilter(correlation_id=run_id)))
            assert summaries, "expected at least one saved checkpoint"
            invocation_id = summaries[-1].invocation_id
            record = await checkpointer.load(invocation_id)
            assert record is not None, "expected the saved record to load"
            completed_node_names = sorted(p.node_name for p in record.completed_positions)
            print(f"    saved invocation_id:    {invocation_id}")
            print(f"    completed nodes:        {completed_node_names}")
            print()

            # Second attempt: resume from the pre-crash record. The
            # engine reads the saved record, skips define_objective (its
            # position is in completed_positions), retries size_crew
            # (now succeeds because _size_crew_attempt_count is past 1),
            # then runs draft_timeline to END. The user-supplied state
            # argument is a placeholder; the engine ignores it on resume
            # and starts from the saved record's state instead.
            #
            # Important: each invoke() mints its OWN invocation_id, even
            # on resume. The pre-crash record stays under the original
            # id; the resumed attempt's new checkpoints (size_crew +
            # draft_timeline completions) save under a fresh id. After
            # the resume we re-query to capture that new id, which is
            # the one phase 2 needs as its resume target.
            print("  second attempt (resume from saved invocation):")
            final_v1 = await graph_v1.invoke(
                MissionPlanStateV1(destination=destination),
                resume_invocation=invocation_id,
            )
            await graph_v1.drain()

            resume_summaries = list(await checkpointer.list(CheckpointFilter(correlation_id=run_id)))
            resumed_invocation_id = resume_summaries[-1].invocation_id

            print(f"    objective:              {final_v1.objective}")
            print(f"    crew_size:              {final_v1.crew_size}")
            print(f"    timeline:               {final_v1.timeline}")
            print(f"    trace:                  {final_v1.trace}")
            print(f"    resumed invocation_id:  {resumed_invocation_id}")
            print()
            print(
                "  Each node name appears exactly once across two invoke() "
                "calls. define_objective is in trace from the first attempt "
                "(its append survived the crash via the synchronous "
                "checkpoint); size_crew + draft_timeline are from the "
                "resumed attempt. size_crew has no duplicate entry because "
                "its first call raised before returning a state update."
            )
            print()

            print("=" * 72)
            print("Phase 2 - invoke v2 graph with resume; v1->v2 migration runs")
            print("=" * 72)
            print()
            print("  v2 adds:    risk_assessment field + assess_risks node")
            print("  migration:  backfills risk_assessment='' for v1 records")
            print()

            graph_v2 = build_graph(checkpointer)
            # Resume from the post-crash, post-resume completed record
            # (resumed_invocation_id), NOT the pre-crash partial record
            # (invocation_id). The pre-crash record only has
            # define_objective in completed_positions; resuming from it
            # would re-run size_crew + draft_timeline, defeating the
            # "completed v1 then migrate" narrative. The engine reads
            # the resumed-id record, applies migrate_v1_to_v2,
            # re-deserializes against MissionPlanStateV2, and continues
            # at the first uncompleted node (assess_risks - the v1
            # pipeline's three nodes are all in completed_positions on
            # this record, the new v2 node is not).
            final_v2 = await graph_v2.invoke(
                MissionPlanStateV2(destination=destination),
                resume_invocation=resumed_invocation_id,
            )
            await graph_v2.drain()

            print("  v2 result after resume:")
            print(f"    objective:        {final_v2.objective}")
            print(f"    crew_size:        {final_v2.crew_size}")
            print(f"    timeline:         {final_v2.timeline}")
            print(f"    risk_assessment:  {final_v2.risk_assessment}")
            print(f"    trace:            {final_v2.trace}")
            print()
            print(
                "  v2's trace appends 'assess_risks' to the v1 entries the "
                "migration preserved. Each v1 node appears exactly once "
                "(no duplicates from the v2 graph re-running them) because "
                "completed_positions skipped them. Only assess_risks was "
                "new work in phase 2."
            )
        finally:
            if _provider_instance is not None:
                await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
