# State migration on resume

**Problem.** A long-running pipeline has saved checkpoints
mid-flight. You add a field to the state schema and rename another.
How do older checkpoints resume against the new schema without
each node body having to handle both shapes?

## Approach

Tag the state class with a `schema_version` and register migration
callables at compile time via `GraphBuilder.with_state_migration`.
On resume, the engine inspects the loaded record's `schema_version`,
walks the registered chain (v1 → v2 → v3 → …), and hands node
bodies a fully-migrated state object. Node code stays single-shape;
all version-aware logic lives in the migration functions.

The migration callable's typed signature is `Callable[[Any], Any]`.
For JSON-backed checkpointers (the only kind that supports
migration; see [Checkpointing](../concepts/checkpointing.md)),
that resolves to `(state_dict: dict) -> dict`: the callable
receives the deserialized record and returns the new shape. The
`from_version` and `to_version` are registered alongside the
callable on `with_state_migration`; the callable itself stays
signature-light because migrations MUST be pure (no implicit
version-dispatch logic inside the function body). The engine
dispatches a `checkpoint_migrated` observer event after each
migration step so OTel / Langfuse spans can correlate the migration
with the resume.

## Snippet

```python
from typing import ClassVar

from openarmature.checkpoint import SQLiteCheckpointer
from openarmature.graph import END, GraphBuilder, State


# v2 schema: renamed `step_count` -> `steps_completed` and added
# `last_node`. Old v1 checkpoints carry `step_count` and lack
# `last_node` entirely.
class PipelineState(State):
    schema_version: ClassVar[str] = "2"

    query: str = ""
    steps_completed: int = 0
    last_node: str | None = None


def _migrate_v1_to_v2(state_dict: dict) -> dict:
    # Rename: step_count -> steps_completed. Default missing
    # last_node to None (the v2 schema allows it).
    state_dict["steps_completed"] = state_dict.pop("step_count", 0)
    state_dict.setdefault("last_node", None)
    return state_dict


async def _step(s: PipelineState) -> dict:
    return {"steps_completed": s.steps_completed + 1, "last_node": "step"}


# ``serialization="json"`` is required for migration to operate on a
# dict; the default ``"pickle"`` mode round-trips through class
# identity and can't migrate across schemas.
compiled = (
    GraphBuilder(PipelineState)
    .add_node("step", _step)
    .add_edge("step", END)
    .set_entry("step")
    .with_checkpointer(SQLiteCheckpointer("ck.db", serialization="json"))
    .with_state_migration("1", "2", _migrate_v1_to_v2)
    .compile()
)

# Later, on resume:
# final = await compiled.invoke(
#     PipelineState(),  # overwritten by the loaded checkpoint
#     resume_invocation=prior_invocation_id,
# )
```

When the chain spans multiple versions (v1 → v2 → v3), register
each step separately with repeated `with_state_migration` calls;
the engine walks them in version order. Gaps fail loudly: if v1→v2
and v3→v4 are registered but a record loads at v2 needing v3, the
engine raises `CheckpointStateMigrationMissing` at resume time
rather than silently using a partial schema.

## When this is the right pattern

- A schema change lands while in-flight checkpoints exist. Without
  migrations, those resume attempts would fail validation at the
  state-merge boundary.
- The change is shape-altering (rename, type change, field
  add/remove) rather than purely additive with a safe default. A
  bare field add with a Pydantic default doesn't need migration;
  Pydantic fills it in on load.
- You want resume to be transparent to node bodies. Migrations let
  each node body assume the current schema unconditionally.

## When it isn't

- Adding a field with a safe default and NOT bumping
  `schema_version`. Pydantic's default handling resolves the missing
  field at load. Bumping `schema_version` without a corresponding
  migration is fail-loud: the engine raises
  `CheckpointStateMigrationMissing` at resume rather than silently
  skipping. If you bump the version, register an identity migration
  (a callable that returns the dict unchanged) to make the additive
  intent explicit.
- Migrations need to call the LLM or do other slow / fallible work.
  The migration runs synchronously during resume; long-running work
  belongs in a dedicated `recompute` node guarded by
  [bypass-if-output-exists](bypass-if-output-exists.md), not in a
  migration callable.
- Schema changes are happening on every release. Migration
  callables accumulate fast; if the cadence is high enough that
  v1→v2→v3→…→v9 starts to feel like a chain, consider whether the
  schema would benefit from being more open at the seams (e.g. a
  `metadata: dict[str, Any]` field for evolving auxiliary data
  instead of dedicated columns).

## Cross-references

- [Checkpointing concept page](../concepts/checkpointing.md):
  checkpointer backends and the resume contract.
- [`session-as-checkpoint-resume`](session-as-checkpoint-resume.md):
  multi-turn agent state via the same checkpointer machinery.
- Spec: [pipeline-utilities](https://openarmature.org/capabilities/pipeline-utilities/),
  the state-migration contract and `checkpoint_migrated` event.
