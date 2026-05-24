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
not state-class instances directly; this lets them serialize, batch,
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
from typing import Any, Literal, Protocol


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
    can live in sets and dict keys; the engine's resume-entry
    derivation relies on ``set`` membership to skip nodes that have
    already completed.

    Fields:

    - ``namespace``: chain of containing-graph node names from
      outermost down to (but **not including**) this node. Empty for
      outermost-graph nodes; one entry for subgraph-internal nodes;
      two entries when nested two deep, and so on. Distinct from
      ``NodeEvent.namespace`` which includes the node's own name;
      ``NodeEvent.namespace == NodePosition.namespace +
      (NodePosition.node_name,)``.
    - ``node_name``: the node's local name in its containing graph.
    - ``step``: the monotonic step counter at the time the node
      completed (shared with ``NodeEvent.step``).
    - ``attempt_index``: 0-based retry attempt index. The final
      successful attempt's index is what gets recorded.
    - ``fan_out_index``: populated only for events from inside a
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


# Spec: realizes pipeline-utilities §10.11 per-instance fan-out
# progress. Promoted from a None placeholder under proposal 0008 to
# populated structures under proposal 0009: each fan-out node that is
# in flight at save time contributes one FanOutProgress entry with
# per-instance state, and saves now fire at every fan-out instance
# internal completed event (not only at fan-out node completion).
@dataclass(frozen=True)
class FanOutInstanceProgress:
    """Per-instance progress entry inside a fan-out's
    :attr:`FanOutProgress.instances` sequence.

    Fields:

    - ``state``: one of ``"completed"``, ``"in_flight"``,
      ``"not_started"``. The ``completed`` state is the load-bearing
      correctness contract: an instance marked ``completed`` MUST have
      its contribution recorded into the accumulator AND that
      contribution MUST be reflected in ``result``. Reducer composition
      rules (§10.11.1) depend on this exactly-once guarantee.
    - ``result``: for ``completed`` instances, the durable contribution
      to the fan-out accumulator (a success value for the
      ``target_field`` bucket, or under ``collect`` error policy an
      error entry for the ``errors_field`` bucket). Typed per the
      parent state schema's ``target_field`` / ``errors_field``
      (representation is implementation-defined per §10.11; Python
      stores as ``Any`` since dynamic typing absorbs the variance).
      Unused for ``in_flight`` and ``not_started``.
    - ``completed_inner_positions``: for ``in_flight`` instances, a
      tuple of ``NodePosition`` entries captured at save time. Same
      shape as :attr:`CheckpointRecord.completed_positions` but scoped
      to this instance's inner subgraph rather than the outer graph.
      Empty when the instance fired its first ``started`` event but
      no inner ``completed`` event yet. Observational only:
      ``in_flight`` instances re-enter at the subgraph entry node on
      resume, not at any of these positions. Unused for ``completed``
      and ``not_started``.
    """

    state: Literal["completed", "in_flight", "not_started"]
    result: Any = None
    completed_inner_positions: tuple[NodePosition, ...] = ()


@dataclass(frozen=True)
class FanOutProgress:
    """Per-fan-out-node progress entry inside a
    :attr:`CheckpointRecord.fan_out_progress` sequence.

    Fields:

    - ``fan_out_node_name``: the fan-out node's name in its containing
      graph.
    - ``namespace``: the chain of outer subgraph-node names enclosing
      the fan-out (empty for outermost-graph fan-outs). Disambiguates
      fan-outs of the same name in different nested-subgraph contexts.
    - ``instance_count``: the resolved instance count for this fan-out
      (per pipeline-utilities §9 count or items_field mode).
    - ``instances``: a tuple of per-instance entries indexed by
      ``fan_out_index`` (``instances[i]`` is the entry for
      ``fan_out_index=i``). Length equals ``instance_count``.
    """

    fan_out_node_name: str
    namespace: tuple[str, ...]
    instance_count: int
    instances: tuple[FanOutInstanceProgress, ...]


# Spec: realizes pipeline-utilities §10.2 CheckpointRecord.
@dataclass(frozen=True)
class CheckpointRecord:
    """One invocation's progress at one save point.

    Frozen: backends MUST treat the record as immutable. The engine
    builds a fresh record per ``completed`` event rather than mutating
    a shared one. The ``fan_out_progress`` field (per §10.11) carries
    per-fan-out-node entries when one or more fan-outs are in flight
    at save time; an empty tuple means no fan-out progress to record.
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
    fan_out_progress: tuple[FanOutProgress, ...] = field(default=())


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

    - ``correlation_id``: match records whose ``correlation_id``
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
    migrations are registered; the registry has no chance to bridge.

    **Attribute-presence contract.** The class-body ``= False``
    below is a typing-level signal, not a runtime guarantee:
    ``typing.Protocol`` does not create an instance attribute on
    a conforming class that doesn't declare it itself. Concrete
    backends SHOULD declare ``supports_state_migration`` (either
    at the class level like ``InMemoryCheckpointer`` does, or as
    an ``__init__``-set instance attribute like
    ``SQLiteCheckpointer`` does for the mode-dependent case) so
    Pyright accepts the structural conformance and ``getattr``
    sees the value. The engine's resume path reads the attribute
    via ``getattr(checkpointer, "supports_state_migration",
    False)``, so a third-party backend that omits the attribute
    entirely is treated as non-migration-eligible without
    raising; that's the runtime default the engine guarantees.
    """

    # Declared as an instance attribute (not ``ClassVar``) so backends
    # can compute it at construction time when the answer depends on
    # constructor args. SQLiteCheckpointer is the concrete case:
    # JSON-mode supports migration, pickle-mode doesn't, and the mode
    # is a per-instance constructor arg. Backends with a static answer
    # (InMemoryCheckpointer is always False) override at the class
    # level. Pyright accepts either shape because Protocol attribute
    # conformance ignores the ClassVar marker on subclasses. The
    # class-body default below is for typing only; see the
    # docstring's "Attribute-presence contract" section for the
    # runtime ``getattr``-based safety net.
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
    "FanOutInstanceProgress",
    "FanOutProgress",
    "NodePosition",
]
