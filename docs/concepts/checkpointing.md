# Checkpointing

Save state at each node boundary, resume from a prior point on the next
invocation. Useful for long-running graphs where a crash partway
through is expensive to re-run from scratch.

Checkpointing is opt-in. Without a checkpointer, the engine holds no
state across invocations and a crash means restart-from-entry.

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

The engine fires a save at every `completed` event for outermost-graph
nodes and subgraph-internal nodes. Saves run async; the engine doesn't
block on disk IO.

## Resuming

Pass `resume_invocation` to `invoke()`:

```python
final = await graph.invoke(initial_state, resume_invocation="<id>")
```

The engine loads the last saved record for that invocation ID, restores
state, and continues from the next node. If no record exists for the
ID, the engine starts fresh (still passing your `initial_state` to the
entry node) and records the ID for future resumes.

## What's in a CheckpointRecord

A record carries:

- The full `state` at the checkpoint point (the state after a `completed`
  event's merge — i.e., what the next node would receive).
- The `NodePosition` — the next node to run, including namespace
  context for subgraph-internal positions.
- An invocation identifier.
- A schema version (for record format migrations).

The records are *the state plus enough context to resume the engine
loop* — no more.

## Two built-in backends

- **`InMemoryCheckpointer`** — backed by a dict in process memory.
  Loses everything on process exit. Useful for tests and for
  short-lived contexts where you want the checkpoint API surface
  without disk overhead.
- **`SQLiteCheckpointer`** — backed by a SQLite database file. Survives
  process exit. The default choice for any non-trivial use.

Both implement the `Checkpointer` Protocol:

```python
class Checkpointer(Protocol):
    async def save(self, record: CheckpointRecord) -> None: ...
    async def load(
        self,
        invocation_id: str,
        *,
        filter: CheckpointFilter | None = None,
    ) -> CheckpointRecord | None: ...
    async def list_invocations(self) -> list[CheckpointSummary]: ...
```

Custom backends just implement these three methods. Targets that make
sense: Redis (for ephemeral, network-shared state), Postgres (for
durable, multi-process state), S3 (for cross-region durability).

## When NOT to use checkpointing

- **Pure pipelines that complete in seconds.** Restart-from-entry is
  cheap; checkpoints are pure overhead.
- **Pipelines whose external side effects can't be replayed.** If
  node A sends an email, resuming from after A means the email already
  sent — which may or may not be what you want. Reason explicitly
  about replay semantics before turning on resume.

## Fan-out resume

Fan-out semantics with checkpointing: per spec proposal 0009
(currently Draft), individual fan-out instances save at their internal
`completed` events. On resume, only the instances that didn't already
merge into outer state re-run; finished instances are skipped.

The Draft hasn't shipped yet — the current implementation atomically
restarts the whole fan-out on resume. Watch the spec repo for proposal
0009 to ship.

## What checkpointing is NOT

- **Not a database.** It's a serialization/deserialization seam for
  state, not a query layer. Don't try to query checkpoint records
  for analytics; emit observability events instead.
- **Not human-in-the-loop.** Pausing for human input is a separate
  capability; checkpointing is just "save and resume," not "pause and
  wait."
- **Not a workflow orchestrator.** Long-running, multi-process
  orchestration belongs at a higher layer (Temporal, Airflow). Checkpointing
  is for crash-recovery and resumability within one logical run.
