# 08 - Checkpointing and migration

A lunar-mission planning pipeline that writes a SQLite checkpoint
after every step, then resumes the saved invocation under an
upgraded state schema with a v1->v2 migration backfilling new
fields.

## Overview

A planning pipeline drafts a lunar mission plan in three steps:

1. `define_objective` - state the primary objective in one
   sentence.
2. `size_crew` - pick a crew size between 2 and 8.
3. `draft_timeline` - draft a one-sentence timeline.

The graph is wired with a `SQLiteCheckpointer` in JSON mode, so the
engine writes a record after every `completed` event. The whole v1
pipeline runs once, and the saved record persists on disk in a
temporary directory.

Then "some time later" - in the same script for demo purposes - a
v2 schema lands. It adds a `risk_assessment` field and a new
`assess_risks` node at the end. A migration function backfills
`risk_assessment=""` for v1 records. The v2 graph resumes the
saved v1 invocation, the migration runs once on load, and
execution continues at `assess_risks`. The original three nodes do
not re-execute.

## What it teaches

- [`SQLiteCheckpointer(path, serialization="json")`](../concepts/checkpointing.md)
  writing records to a SQLite file. JSON is the
  migration-eligible serialization; `pickle` mode is faster but
  can't bridge schemas.
- [`with_checkpointer`](../concepts/checkpointing.md) wiring the
  checkpointer to the graph. The engine fires a save at every
  `completed` event for outermost and subgraph-internal nodes.
- [`State.schema_version`](../concepts/checkpointing.md) as a
  `ClassVar[str]` declaration. Empty string opts the class out of
  migration support; any non-empty value opts it in.
- [`with_state_migration(from_version, to_version, migrate)`](../concepts/checkpointing.md)
  registering one edge of the migration chain. The `migrate`
  callable is pure (dict in, dict out, no I/O).
- [`invoke(state, resume_invocation=<id>)`](../concepts/checkpointing.md)
  resuming from a saved record. The engine reads the record,
  applies the migration chain, re-deserializes against the current
  state class, and continues at the first uncompleted node.
- The migration registry's BFS resolution: with a v3 schema and
  two migration edges (`v1→v2`, `v2→v3`), the registry walks the
  shortest chain automatically. A v1 record loaded under a v3
  graph runs `v1→v2` then `v2→v3` without caller-side composition.

## How to run

```bash
uv sync --group examples
LLM_API_KEY=sk-... uv run python examples/08-checkpointing-and-migration/main.py
```

The SQLite database is created in a `TemporaryDirectory` that's
cleaned up automatically. The demo runs both phases in one
invocation so you can see the resume behavior end-to-end without
manual orchestration.

## The graph

V1 graph:

```mermaid
flowchart LR
  start([start])
  define_objective[define_objective]
  size_crew[size_crew]
  draft_timeline[draft_timeline]
  stop([end])

  start --> define_objective --> size_crew --> draft_timeline --> stop
```

V2 graph (adds `assess_risks` at the end):

```mermaid
flowchart LR
  start([start])
  define_objective[define_objective]
  size_crew[size_crew]
  draft_timeline[draft_timeline]
  assess_risks[assess_risks]
  stop([end])

  start --> define_objective --> size_crew --> draft_timeline --> assess_risks --> stop
```

The v2 graph also registers `with_state_migration("v1", "v2",
migrate_v1_to_v2)`. The migration function takes the saved state
as a plain dict and returns a dict at the new schema (here, just
`{**state_dict, "risk_assessment": ""}`).

## Reading the output

```
========================================================================
Phase 1 - invoke v1 graph; checkpoints save after every node
========================================================================

  destination:       Lunar South Pole
  checkpoint db:     /tmp/oa-checkpoint-demo-.../checkpoints.sqlite

v1 result:
  objective:  <objective sentence>
  crew_size:  4
  timeline:   <timeline sentence>

  v1 invocation_id: <uuid>

========================================================================
Phase 2 - invoke v2 graph with resume; v1->v2 migration runs
========================================================================

  v2 adds:    risk_assessment field + assess_risks node
  migration:  backfills risk_assessment='' for v1 records

v2 result after resume:
  objective:        <same objective sentence from v1>
  crew_size:        4
  timeline:         <same timeline sentence from v1>
  risk_assessment:  <new sentence from assess_risks>

  trace: ['assess_risks']

The v1 nodes appear once each in v1's trace and NOT in v2's
trace - they were skipped on resume because completed_positions
already covered them. Only assess_risks ran in phase 2.
```

- **`v1 invocation_id`** is the saved record's correlation key. We
  passed a deterministic `correlation_id` to `invoke()` so the
  checkpoint filter can find the right record; in production, the
  caller usually owns the correlation_id and persists it alongside
  the request that produced the run.
- **`trace: ['assess_risks']`** on the v2 result is the key signal.
  The v1 nodes (`define_objective`, `size_crew`, `draft_timeline`)
  did not re-execute on resume. Their `completed_positions` entries
  in the saved record told the engine they were already done; the
  v2 pipeline began at the first uncompleted position, which is
  `assess_risks`.
- **`crew_size: 4`** and the other v1 fields are present on the v2
  result because the migration preserved them via `{**state_dict,
  ...}`. A migration that *changed* an existing field (e.g.,
  splitting `name` into `first_name` + `last_name`) would
  transform the dict more thoroughly.
- **JSON serialization** is what made this possible. With
  `serialization="pickle"`, the saved record would be a pickled
  v1 instance that couldn't be re-deserialized against
  `MissionPlanStateV2`; JSON makes the saved state a plain dict
  that the migration function can rewrite freely.
