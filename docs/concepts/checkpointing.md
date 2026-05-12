# Checkpointing

Save state at every node boundary; resume a crashed run from the last
saved point on a subsequent `invoke()`. Without a checkpointer, the
engine holds no state across invocations — a crash means start-from-entry.

## Wiring a checkpointer

Register at build time via `with_checkpointer`:

```python
from openarmature.checkpoint import SQLiteCheckpointer

checkpointer = SQLiteCheckpointer(db_path="./checkpoints.db")

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
graph nodes and subgraph-internal nodes. **Fan-out instance internal
events do NOT save** in the shipping version (per spec
pipeline-utilities §10.3 + §10.7 atomic-restart).

## Saves are synchronous-by-contract

The engine **awaits** every `Checkpointer.save` before continuing to
the next node. This is mandatory per spec §10.3 and the load-bearing
property that makes checkpointing useful at all: a crash immediately
after a `completed` event cannot have lost the corresponding save,
because the save resolves before the next node runs.

The corollary: slow backends throttle execution. Wrapping a high-
latency persistence layer in a checkpointer makes the whole graph
run at its latency. Plan accordingly — async writes inside the
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
- **If no record exists, the engine raises `CheckpointNotFound`** —
  per spec §10.4 step 1. It does NOT silently start a fresh run; the
  user must explicitly handle the not-found case (typically: drop the
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
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    fan_out_progress: None = field(default=None)
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
  identifier propagated via ContextVar — multiple invocations
  related by a higher-level request can share one `correlation_id`
  while each having its own `invocation_id`. See
  [Observability](observability.md) for how `correlation_id`
  threads through logs and spans.
- **`parent_states` is the chain of containing-graph snapshots.**
  Outermost first; empty for an outer-level save. Inner-node saves
  populate it so resume can re-enter a subgraph from the right
  depth without re-projecting.
- **`fan_out_progress: None` is reserved** for the v2 per-instance
  fan-out resume mode (spec proposal 0009, currently Draft). In v1
  it's always `None`.

## The Checkpointer Protocol

Four methods:

```python
class Checkpointer(Protocol):
    async def save(self, invocation_id: str, record: CheckpointRecord) -> None: ...
    async def load(self, invocation_id: str) -> CheckpointRecord | None: ...
    async def list(self, filter: CheckpointFilter | None = None) -> Iterable[CheckpointSummary]: ...
    async def delete(self, invocation_id: str) -> None: ...
```

- **`save`** — persist the record under `invocation_id`. Durable for
  any backend that documents durability. Synchronous-by-contract per
  the section above.
- **`load`** — return the *most recent* record for `invocation_id`,
  or `None`. Round-trip-stable with what `save` wrote.
- **`list`** — enumerate saved invocations, optionally filtered by
  `CheckpointFilter` (currently a single `correlation_id` field; v1
  ships intentionally narrow).
- **`delete`** — remove all records for `invocation_id`. No-op if the
  invocation has no record (no error).

Backends MUST be safe to share across concurrent invocations; the
engine doesn't serialize access. For backends with sync I/O, the
standard pattern is `asyncio.to_thread` around the actual driver
call.

## Two built-in backends

- **`InMemoryCheckpointer`** — backed by a dict in process memory.
  Loses everything on process exit. Useful for tests and short-lived
  contexts that want the API surface without disk overhead.
- **`SQLiteCheckpointer`** — backed by a SQLite database file.
  Survives process exit. Reasonable default for any non-trivial use.

Custom backends just implement the four-method Protocol. Targets that
make sense: Redis (ephemeral, network-shared), Postgres (durable,
multi-process), S3 (cross-region durability). For event-sourced
runtimes (Temporal, DBOS, Restate, Inngest) the Protocol is the
adapter layer.

## When NOT to use checkpointing

- **Pure pipelines that complete in seconds.** Restart-from-entry is
  cheap; checkpoints are pure overhead.
- **Pipelines whose external side effects can't safely be re-played.**
  If node A sends an email, resuming from after A means the email
  has already sent — fine if your downstream is idempotent, surprising
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
