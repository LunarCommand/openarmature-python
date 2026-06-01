# Checkpointing

Save state at every node boundary; resume a crashed run from the last
saved point on a subsequent `invoke()`. Without a checkpointer, the
engine holds no state across invocations; a crash means start-from-entry.

## Wiring a checkpointer

Register at build time via `with_checkpointer`:

```python
from openarmature.checkpoint import SQLiteCheckpointer

checkpointer = SQLiteCheckpointer(path="./checkpoints.db")

graph = (
    GraphBuilder(MyState)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge("step_a", "step_b")
    .add_edge("step_b", END)
    .set_entry("step_a")
    .with_checkpointer(checkpointer)
    .compile()
)
```

The engine writes a record at every `completed` event for outermost-
graph nodes, subgraph-internal nodes, and fan-out instance internal
nodes. **Per-instance fan-out resume** is the contract: on resume the
engine re-runs only the instances that did not complete-and-record
their contribution into the fan-out accumulator in the prior run;
completed instances skip and their contributions roll forward to the
fan-in step.

## Saves are synchronous-by-contract

The engine **awaits** every `Checkpointer.save` before continuing to
the next node. This is the load-bearing property that makes
checkpointing useful at all: a crash immediately
after a `completed` event cannot have lost the corresponding save,
because the save resolves before the next node runs.

The corollary: slow backends throttle execution. Wrapping a high-
latency persistence layer in a checkpointer makes the whole graph
run at its latency. Plan accordingly: async writes inside the
backend (e.g., `asyncio.to_thread` around a sync driver) are fine;
fire-and-forget patterns that return before durability is established
violate the contract.

## Resuming

Pass `resume_invocation` to `invoke()`:

```python
final = await graph.invoke(initial_state, resume_invocation="<id>")
```

- If a record exists for that `invocation_id`, the engine restores
  state from `record.state` (or `parent_states` chain for subgraph-
  internal resumes), reconstructs the completed-node set from
  `record.completed_positions`, and continues from the first not-yet-
  completed node.
- **If no record exists, the engine raises `CheckpointNotFound`.**
  It does NOT silently start a fresh run; the user must explicitly
  handle the not-found case (typically: drop the
  `resume_invocation=` and re-invoke without it for a fresh start).

`CheckpointRecordInvalid` surfaces when a record's `schema_version`
doesn't match the current `CHECKPOINT_SCHEMA_VERSION`, or when its
persisted state can't be re-validated against the current state class
(state-shape mismatch after a refactor).

## What a CheckpointRecord carries

```python
@dataclass(frozen=True)
class CheckpointRecord:
    invocation_id: str
    correlation_id: str
    state: Any
    completed_positions: tuple[NodePosition, ...]
    parent_states: tuple[Any, ...]
    last_saved_at: float
    schema_version: str = ""
    fan_out_progress: tuple[FanOutProgress, ...] = field(default=())
```

Field framing worth getting right:

- **`completed_positions` is history, not "next."** It's the list of
  `NodePosition`s that have already completed. Resume works by
  *replaying that list to derive the next node*, not by reading a
  pointer to the next node. This is why the field is plural and why
  the framing matters: every saved node contributes a position, and
  resume walks the graph skipping every position that's already
  there.
- **`correlation_id` ≠ `invocation_id`.** `invocation_id` identifies
  *this* graph run uniquely. `correlation_id` is a cross-system
  identifier propagated via ContextVar; multiple invocations
  related by a higher-level request can share one `correlation_id`
  while each having its own `invocation_id`. See
  [Observability](observability.md) for how `correlation_id`
  threads through logs and spans.
- **`parent_states` is the chain of containing-graph snapshots.**
  Outermost first; empty for an outer-level save. Inner-node saves
  populate it so resume can re-enter a subgraph from the right
  depth without re-projecting.
- **`fan_out_progress` carries per-fan-out-node progress** when one
  or more fan-outs are in flight at save time. Each `FanOutProgress`
  entry records the fan-out's name, namespace, instance count, and a
  per-instance state machine (`not_started` / `in_flight` /
  `completed`) plus the recorded contribution for finalized
  instances. On resume the engine consults this field to decide
  which instances skip (their contributions roll forward) vs re-run
  (re-execute from the inner subgraph's declared entry node). Each
  per-instance entry carries an explicit `result_is_error` boolean
  that discriminates success contributions (roll forward into
  `target_field`) from `collect`-mode error contributions (roll
  forward into `errors_field`). The engine reads the explicit field
  on resume rather than inferring routing from the shape of `result`.
  Empty tuple when no fan-outs are in flight. See
  [Resume semantics](fan-out.md#resume-semantics) on the fan-out
  page for the full per-instance contract including reducer
  composition, error_policy semantics, and the optional
  fan-out-internal save batching.

  **Count drift between runs raises.** If the resumed run's resolved
  instance count differs from the saved entry's `instance_count`
  (e.g., the user shrunk or grew the `items_field` list between
  crash and resume), the engine raises `CheckpointRecordInvalid`
  before any fan-out instance work runs. The strict raise prevents
  silent contribution loss (under shrink) or dispatching unsaved
  work (under grow); the user either coheres the inputs or restarts
  cleanly.

## The Checkpointer Protocol

Four methods:

```python
class Checkpointer(Protocol):
    async def save(self, invocation_id: str, record: CheckpointRecord) -> None: ...
    async def load(self, invocation_id: str) -> CheckpointRecord | None: ...
    async def list(self, filter: CheckpointFilter | None = None) -> Iterable[CheckpointSummary]: ...
    async def delete(self, invocation_id: str) -> None: ...
```

- **`save`**: persist the record under `invocation_id`. Durable for
  any backend that documents durability. Synchronous-by-contract per
  the section above.
- **`load`**: return the *most recent* record for `invocation_id`,
  or `None`. Round-trip-stable with what `save` wrote.
- **`list`**: enumerate saved invocations, optionally filtered by
  `CheckpointFilter` (currently a single `correlation_id` field; v1
  ships intentionally narrow).
- **`delete`**: remove all records for `invocation_id`. No-op if the
  invocation has no record (no error).

Backends MUST be safe to share across concurrent invocations; the
engine doesn't serialize access. For backends with sync I/O, the
standard pattern is `asyncio.to_thread` around the actual driver
call.

## Two built-in backends

- **`InMemoryCheckpointer`**: backed by a dict in process memory.
  Loses everything on process exit. Useful for tests and short-lived
  contexts that want the API surface without disk overhead.
- **`SQLiteCheckpointer`**: backed by a SQLite database file.
  Survives process exit. Reasonable default for any non-trivial use.

Custom backends just implement the four-method Protocol. Targets that
make sense: Redis (ephemeral, network-shared), Postgres (durable,
multi-process), S3 (cross-region durability). For event-sourced
runtimes (Temporal, DBOS, Restate, Inngest) the Protocol is the
adapter layer.

## State migrations

When a checkpoint was saved against an earlier version of your state
schema and the code has since evolved, the engine consults a
**migration registry** to bridge the saved record into the current
shape. Without migrations, a schema change invalidates every prior
checkpoint; with one short registration per change, you keep your
saved records working across releases.

The wire-up is two pieces: declare a version on your state class,
and register one migration per version bump.

```python
from typing import ClassVar
from openarmature.graph import State, GraphBuilder
from openarmature.checkpoint import SQLiteCheckpointer


class MyState(State):
    schema_version: ClassVar[str] = "v2"
    x: int = 0
    new_field: str = "default"      # added in v2


def add_new_field_default(state: dict) -> dict:
    return {**state, "new_field": "default"}


graph = (
    GraphBuilder(MyState)
    .add_node(...)
    .with_checkpointer(SQLiteCheckpointer("ck.db", serialization="json"))
    .with_state_migration("v1", "v2", add_new_field_default)
    .compile()
)
```

On resume, the engine reads the saved record's `schema_version`. If
it equals `MyState.schema_version`, the record loads via the §10.4
fast path (no migration consulted). If it differs, the engine
resolves a chain through the registry (BFS for the shortest path),
applies each migration in order to the record's state, then
deserializes the result into your current state class.

**Canonical source for `schema_version`.** The framework reads
`schema_version` from the state class declared at graph construction
time: the class passed to `GraphBuilder(...)`. If you pass a State
subclass instance at runtime whose `schema_version` shadows the
declared class's value, the saved record still carries the declared
class's value. This rule keeps every save site within an invocation
consistent (outer dispatch saves, subgraph-internal saves, fan-out
instance internal saves all report the same version) and aligns
with how the migration registry is keyed.

### Chain resolution

Registered migrations form a directed graph. Each
`with_state_migration(a, b, fn)` is an edge from `a` to `b`. Chain
resolution finds the shortest path between the saved version and the
current version. Branching is fine: a v1 record can have one
migration leading to v2 and another leading to v2-experimental;
chain resolution picks the path that ends at the current declared
version.

Two ambiguity cases are configuration errors. Both surface as
`CheckpointStateMigrationChainAmbiguous`:

- **Duplicate edges.** Registering two migrations with the same
  `(from_version, to_version)` pair raises at registration time so
  the configuration error surfaces before any resume attempt.
  Either delete one or pick distinct version identifiers.
- **Multiple shortest paths.** A diamond like
  `v1 → v2 → v4` and `v1 → v3 → v4` is ambiguous: both paths have
  length 2. The engine raises during resume so the user can
  register fewer migrations or pick a single canonical route.

### The three new error categories

- **`CheckpointStateMigrationChainAmbiguous`**: the registered
  migration graph is ambiguous (duplicate `(from, to)` pair at
  registration time, OR multiple distinct shortest paths between
  the saved and current versions at resume time). Surfaces before
  any migration function runs. Carries `from_version` and
  `to_version` when known.
- **`CheckpointStateMigrationMissing`**: the saved version doesn't
  match the current version, and no chain bridges them. Carries
  `from_version`, `to_version`, a count of registered migrations,
  and a human-readable `registry_description` so operators see what
  IS available.
- **`CheckpointStateMigrationFailed`**: a user-supplied migration
  function raised. Subsequent migrations in the chain don't run;
  the resume fails. The migration's exception rides `__cause__`.

Routing precedence on resume: chain-ambiguous → missing → failed →
record-invalid.

A third category, `CheckpointRecordInvalid`, continues to cover the
**post**-migration case: a migration ran cleanly but produced
output that the current state class can't deserialize (missing a
required field, wrong type, etc.). The three categories are
mutually exclusive on any given resume.

### Backend support

Not every backend can migrate. Migration needs the backend to expose
a **structural intermediate form** of the loaded state (a plain
dict, JSON tree, or similar) that's independent of the current
state class.

- **`SQLiteCheckpointer(serialization="json")`** can. JSON-encoded
  state loads to a dict; the migration function operates on the
  dict directly.
- **`SQLiteCheckpointer(serialization="pickle")`** can NOT. Pickle
  holds class identity and round-trips back to typed instances.
- **`InMemoryCheckpointer`** can NOT. It holds live typed-state
  references by reference; there's no serialization step.

On version mismatch against a non-migration-eligible backend, the
engine raises `CheckpointRecordInvalid` (not
`CheckpointStateMigrationMissing`): the registry has no chance to
bridge.

### Parent-state migration

Subgraph saves carry a `parent_states` chain of the outer-graph
state captured at the moment of the inner save. On resume, the same
migration chain applies to each entry in `parent_states` in lockstep
with the outer state. The spec treats `parent_states` as carrying
the same `schema_version` as the outer record (no per-parent
version metadata in v1).

### Migrations MUST be pure

A migration function MUST be deterministic, with no I/O, no implicit
state, no random or wall-clock-derived output. The framework
doesn't enforce purity, but violating it breaks determinism
guarantees for resume.

## When NOT to use checkpointing

- **Pure pipelines that complete in seconds.** Restart-from-entry is
  cheap; checkpoints are pure overhead.
- **Pipelines whose external side effects can't safely be re-played.**
  If node A sends an email, resuming from after A means the email
  has already sent; fine if your downstream is idempotent, surprising
  if it isn't. Reason explicitly about replay semantics before turning
  on resume.

## What checkpointing is NOT

- **Not a database.** It's a serialization/deserialization seam for
  state, not a query layer. Don't drive analytics off saved records;
  emit observability events instead.
- **Not human-in-the-loop.** Pausing for human input is a separate
  capability; checkpointing is just "save and resume," not "pause and
  wait."
- **Not a workflow orchestrator.** Long-running, multi-process,
  cross-system orchestration belongs at a higher layer (Temporal,
  Airflow). Checkpointing is for crash-recovery and resumability
  within one logical run.
