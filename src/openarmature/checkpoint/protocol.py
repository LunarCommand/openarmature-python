# Spec mapping (pipeline-utilities):
# - Module realizes §10.1 (Checkpointer Protocol) + §10.2 (record types:
#   CheckpointRecord, NodePosition, CheckpointSummary).
# - Save fires per §10.3 (outermost-graph + subgraph-internal + fan-out
#   completed events).
# - Fan-out instance-internal events do NOT save in v1 per §10.7
#   (atomic-restart contract); proposal 0009 specifies the future
#   per-instance variant.

"""Checkpointer Protocol + record types.

The :class:`Checkpointer` Protocol is the persistence seam between the
engine's save/resume machinery and a concrete backend (in-memory,
SQLite, Temporal, DBOS, etc.). Backends receive structured records,
not state-class instances directly — this lets them serialize, batch,
or hand off to a durable backend without taking a hard dependency on
the user's Pydantic state schema.

A :class:`CheckpointRecord` is a frozen, hashable snapshot of one
invocation's progress at one save point. The engine produces one
record per ``completed`` event for outermost-graph nodes and
subgraph-internal nodes. Fan-out instance internal events do NOT
produce records in the shipping version (atomic-restart contract).

``CheckpointRecord.schema_version`` carries the user-facing
state-schema identifier per spec §10.2 (proposal 0014 repurposes
the field from the original backend-internal record-shape role).
The framework reads ``type(state).schema_version`` at save time;
on load, version mismatches route through the migration registry
(per §10.12) rather than a strict equality check.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


# Spec: realizes pipeline-utilities §10.2 NodePosition. Field semantics
# tied to graph-engine §6 NodeEvent (step shared; namespace here omits
# the node's own name where NodeEvent.namespace includes it).
# ``fan_out_index`` is part of the shape so a future v2 per-instance
# fan-out resume mode can populate it without a record-shape migration
# (proposal 0009).
@dataclass(frozen=True)
class NodePosition:
    """A single completed-node coordinate in the resume map.

    Frozen + automatically hashable (no mutable fields), so positions
    can live in sets and dict keys — the engine's resume-entry
    derivation relies on ``set`` membership to skip nodes that have
    already completed.

    Fields:

    - ``namespace`` — chain of containing-graph node names from
      outermost down to (but **not including**) this node. Empty for
      outermost-graph nodes; one entry for subgraph-internal nodes;
      two entries when nested two deep, and so on. Distinct from
      ``NodeEvent.namespace`` which includes the node's own name —
      ``NodeEvent.namespace == NodePosition.namespace +
      (NodePosition.node_name,)``.
    - ``node_name`` — the node's local name in its containing graph.
    - ``step`` — the monotonic step counter at the time the node
      completed (shared with ``NodeEvent.step``).
    - ``attempt_index`` — 0-based retry attempt index. The final
      successful attempt's index is what gets recorded.
    - ``fan_out_index`` — populated only for events from inside a
      fan-out instance. Those events do NOT produce records in the
      shipping version; the field is part of the position shape so a
      future per-instance fan-out resume can populate it without a
      record-shape migration.
    """

    namespace: tuple[str, ...]
    node_name: str
    step: int
    attempt_index: int = 0
    fan_out_index: int | None = None


# Spec: realizes pipeline-utilities §10.2 CheckpointRecord.
# ``fan_out_progress`` is reserved for proposal 0009 (per-instance
# fan-out resume); always ``None`` in the shipping version.
@dataclass(frozen=True)
class CheckpointRecord:
    """One invocation's progress at one save point.

    Frozen — backends MUST treat the record as immutable; the engine
    builds a fresh record per ``completed`` event rather than mutating
    a shared one. The ``fan_out_progress`` field is reserved for a
    future per-instance fan-out resume mode; in the shipping version
    it is always ``None``.
    """

    invocation_id: str
    correlation_id: str
    state: Any
    completed_positions: tuple[NodePosition, ...]
    parent_states: tuple[Any, ...]
    last_saved_at: float
    # Per spec §10.2 (proposal 0014): the user's state-schema
    # version, read off ``type(state).schema_version`` at save time.
    # Empty-string sentinel for state classes that don't declare a
    # version; non-empty declares migration-eligibility.
    schema_version: str = ""
    fan_out_progress: None = field(default=None)


# Spec: realizes pipeline-utilities §10.1 CheckpointSummary. The four
# declared fields are the spec-mandated minimum; implementations MAY
# add backend-specific fields beyond these.
@dataclass(frozen=True)
class CheckpointSummary:
    """Lightweight record-level metadata returned by
    :meth:`Checkpointer.list`.

    Implementations MAY add backend-specific fields; the four declared
    here are the cross-backend portable subset callers can rely on.
    """

    invocation_id: str
    correlation_id: str
    last_saved_at: float
    completed_node_count: int


@dataclass(frozen=True)
class CheckpointFilter:
    """Predicate for :meth:`Checkpointer.list`. v1 ships two narrow
    fields; richer query DSLs are deferred to follow-on work.

    - ``correlation_id`` — match records whose ``correlation_id``
      equals the supplied value. ``None`` matches every record
      (the "list all" case).
    """

    correlation_id: str | None = None


# Spec: realizes pipeline-utilities §10.1 Checkpointer Protocol. Save
# semantics are synchronous-by-contract per §10.3; resume on missing
# record raises ``CheckpointNotFound`` per §10.4 step 1.
class Checkpointer(Protocol):
    """Persistence seam for graph invocations.

    Implementations MUST be safe to share across concurrent
    invocations of the same graph (the engine does not serialize
    access). Each operation MUST be thread-safe (Python) /
    task-coroutine-safe (asyncio); backends with synchronous I/O
    typically wrap their work in ``asyncio.to_thread`` or equivalent.

    ``supports_state_migration`` marks whether the backend can
    expose a structural intermediate form of the loaded state (a
    plain dict, JSON tree, or similar) that is independent of the
    current state class. JSON-encoded backends naturally satisfy
    this; backends that store live typed state instances or use
    class-bound serialization (pickle) cannot. Per spec §10.12.1,
    backends that cannot expose the intermediate MUST raise
    ``CheckpointRecordInvalid`` on version mismatch even when
    migrations are registered — the registry has no chance to bridge.
    """

    # Declared as an instance attribute (not ``ClassVar``) so backends
    # can compute it at construction time when the answer depends on
    # constructor args. SQLiteCheckpointer is the concrete case:
    # JSON-mode supports migration, pickle-mode doesn't, and the mode
    # is a per-instance constructor arg. Backends with a static answer
    # (InMemoryCheckpointer is always False) override at the class
    # level with ``ClassVar[bool] = False``; pyright is happy with
    # either shape because Protocol attribute conformance ignores the
    # ClassVar marker on subclasses.
    supports_state_migration: bool = False

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        """Persist ``record`` for ``invocation_id``. After return the
        record MUST be durable across process crashes for backends
        that document durability (in-memory backends are not durable
        and MUST document this). Synchronous-by-contract: the engine
        awaits this call before continuing to the next node so a
        crash immediately after a ``completed`` event cannot have
        lost the corresponding save."""
        ...

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        """Return the most recent record for ``invocation_id`` or
        ``None`` if no record exists. The returned record MUST be
        structurally identical to what ``save`` last wrote for this
        ``invocation_id`` (round-trip integrity)."""
        ...

    async def list(self, filter: CheckpointFilter | None = None) -> Iterable[CheckpointSummary]:
        """Enumerate saved invocations. The ``filter`` shape is
        backend-defined; this implementation ships ``list_all`` and
        ``list_by_correlation_id`` predicates."""
        ...

    async def delete(self, invocation_id: str) -> None:
        """Remove all records for ``invocation_id``. MUST be a no-op
        when the invocation_id has no record (no error)."""
        ...


__all__ = [
    "CheckpointFilter",
    "CheckpointRecord",
    "CheckpointSummary",
    "Checkpointer",
    "NodePosition",
]
