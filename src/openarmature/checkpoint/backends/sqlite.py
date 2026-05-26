# Spec: one of the reference backends listed in pipeline-utilities §10.11.

"""SQLite-backed Checkpointer.

Persists records to a SQLite database with WAL mode enabled. Durable
across process crashes within a single host. One row per
``invocation_id`` (upsert retention; overwritten on every save).

**Serialization knobs:**

- ``"pickle"`` (default): accepts any pickleable state shape.
  Python-only on the read side; a TypeScript reimplementation cannot
  decode pickle blobs.
- ``"json"``: accepts only JSON-native state shapes (Pydantic
  ``model_dump(mode="json")`` output). Cross-language portable; if
  the user wants to read python-written records from a TypeScript
  consumer (or vice versa), this is the choice.

Choose deliberately at construction time; the same database file
MUST be read with the same serialization mode it was written with;
mismatches surface as :class:`CheckpointRecordInvalid` on
:meth:`load`.

I/O runs on the asyncio default thread pool via
``asyncio.to_thread``: the SQLite library is synchronous, and
running it inline on the event loop would block other tasks during
disk I/O.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from ..errors import CheckpointRecordInvalid
from ..protocol import (
    CheckpointFilter,
    CheckpointRecord,
    CheckpointSummary,
    FanOutInstanceProgress,
    FanOutProgress,
    NodePosition,
)

SerializationMode = Literal["pickle", "json"]


# Proposal 0009 / spec v0.18.0 sqlite serialization choice (Q3 in the
# impl plan): the new ``fan_out_progress`` field on CheckpointRecord
# gets a dedicated BLOB column (Plan B). Plan A from the impl plan
# was JSON-blob expansion of an existing blob, but each existing blob
# encodes one specific field (state, positions, parent_states); adding
# fan_out_progress as a new column keeps the field-to-blob mapping
# obvious and avoids smearing two semantically distinct fields into
# one. The column is added via ALTER TABLE for backward compatibility
# with databases written before this proposal landed.
_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    invocation_id     TEXT PRIMARY KEY,
    correlation_id    TEXT NOT NULL,
    state_blob        BLOB NOT NULL,
    positions_blob    BLOB NOT NULL,
    parent_states_blob BLOB NOT NULL,
    last_saved_at     REAL NOT NULL,
    schema_version    TEXT NOT NULL,
    serialization     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_correlation_id
    ON checkpoints (correlation_id);
"""

# Idempotent column add for the fan_out_progress blob. Older databases
# created before proposal 0009 lack this column; SQLite has no
# ``ADD COLUMN IF NOT EXISTS``, so we attempt the ADD and swallow the
# duplicate-column error.
_FAN_OUT_PROGRESS_COLUMN_DDL = "ALTER TABLE checkpoints ADD COLUMN fan_out_progress_blob BLOB"


def _fan_out_progress_to_dict(fp: FanOutProgress) -> dict[str, Any]:
    """Serialize a frozen :class:`FanOutProgress` entry to a dict shape
    the configured serialization mode round-trips cleanly.

    JSON mode walks tuples as lists already; pickle mode round-trips
    dicts identically. The shape mirrors the dataclass fields one for
    one with namespace/positions flattened to lists.
    """
    return {
        "fan_out_node_name": fp.fan_out_node_name,
        "namespace": list(fp.namespace),
        "instance_count": fp.instance_count,
        "instances": [
            {
                "state": inst.state,
                "result": inst.result,
                # Per proposal 0027: explicit discriminator on the saved
                # record. Backward-compat path on load: pre-0027 records
                # omit the key, which `dict.get(...)` returns ``False``
                # for, matching the ``FanOutInstanceProgress`` default
                # for ``in_flight`` / ``not_started`` entries.
                "result_is_error": inst.result_is_error,
                "completed_inner_positions": [asdict(p) for p in inst.completed_inner_positions],
            }
            for inst in fp.instances
        ],
    }


def _fan_out_progress_from_dict(d: dict[str, Any], invocation_id: str) -> FanOutProgress:
    """Inverse of :func:`_fan_out_progress_to_dict` — rebuild a frozen
    :class:`FanOutProgress` from its dict shape, restoring positions
    as :class:`NodePosition` instances.

    ``invocation_id`` is threaded in so ``CheckpointRecordInvalid``
    raises (e.g., a non-bool ``result_is_error`` field) can point at
    the correct persisted record.
    """
    instances: list[FanOutInstanceProgress] = []
    for inst in cast("list[dict[str, Any]]", d["instances"]):
        inner_positions = tuple(
            NodePosition(
                namespace=tuple(p["namespace"]),
                node_name=p["node_name"],
                step=p["step"],
                attempt_index=p.get("attempt_index", 0),
                fan_out_index=p.get("fan_out_index"),
            )
            for p in cast("list[dict[str, Any]]", inst.get("completed_inner_positions", []))
        )
        # Per proposal 0027: distinguish key-absent (pre-0027 records,
        # default ``False`` for backward compat) from key-present
        # (strictly validate it's a bool). The naive
        # ``bool(inst.get("result_is_error", False))`` would coerce
        # truthy non-bools (e.g. the string ``"false"``) to ``True``
        # and misclassify resume routing — the JSON deserializer can
        # in principle land a non-bool here from a corrupted record,
        # and silently coercing it would route a success contribution
        # through the errors_field bucket on resume.
        if "result_is_error" in inst:
            raw_rie = inst["result_is_error"]
            if not isinstance(raw_rie, bool):
                raise CheckpointRecordInvalid(
                    invocation_id,
                    f"fan_out_progress instance result_is_error must be bool, "
                    f"got {type(raw_rie).__name__}: {raw_rie!r}",
                )
            result_is_error = raw_rie
        else:
            result_is_error = False
        instances.append(
            FanOutInstanceProgress(
                state=inst["state"],
                result=inst.get("result"),
                result_is_error=result_is_error,
                completed_inner_positions=inner_positions,
            )
        )
    return FanOutProgress(
        fan_out_node_name=d["fan_out_node_name"],
        namespace=tuple(d["namespace"]),
        instance_count=d["instance_count"],
        instances=tuple(instances),
    )


def _to_json_native(obj: Any) -> Any:
    """Walk ``obj`` converting Pydantic ``BaseModel`` instances to
    JSON-native dicts via ``model_dump(mode="json")``. Lists and
    tuples recurse; dicts and scalars pass through unchanged. Used by
    the JSON-mode encoder so callers don't have to pre-convert their
    state objects."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, list | tuple):
        seq: list[Any] = list(cast("list[Any]", obj))
        out: list[Any] = []
        for item in seq:
            out.append(_to_json_native(item))
        return out
    return obj


class SQLiteCheckpointer:
    """SQLite Checkpointer with WAL-mode durability.

    **Retention:** upsert; one row per ``invocation_id``, overwritten
    on every save. Saved records are NOT historical: only the most
    recent save for any given ``invocation_id`` is retained.

    **Cross-language portability:** depends on the ``serialization``
    constructor argument. ``"pickle"`` is Python-only; ``"json"``
    works across languages but is restricted to JSON-native state
    shapes (the engine's Pydantic state must successfully
    ``model_dump(mode="json")``).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        serialization: SerializationMode = "pickle",
    ) -> None:
        self._path = str(path)
        self._serialization: SerializationMode = serialization
        self._lock = asyncio.Lock()
        self._initialized = False
        # Per spec §10.12.1, a backend supports state migration only
        # when it can expose a structural intermediate form of the
        # loaded state that is independent of the current state
        # class. JSON serialization satisfies this (loads to dicts);
        # pickle holds class identity and round-trips to typed
        # instances, so it cannot bridge a schema-version mismatch.
        self.supports_state_migration: bool = serialization == "json"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _initialize_sync(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_DDL)
            # Add the fan_out_progress_blob column for databases written
            # before proposal 0009. Idempotent: subsequent runs against
            # an already-migrated database hit "duplicate column" and
            # swallow it. New databases pick the column up via the
            # initial table create + this ALTER, equivalent end state.
            try:
                conn.execute(_FAN_OUT_PROGRESS_COLUMN_DDL)
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _encode(self, obj: Any) -> bytes:
        if self._serialization == "pickle":
            return pickle.dumps(obj)
        # JSON mode: the engine hands us live Pydantic State instances
        # in record.state and record.parent_states. Walk the value
        # tree converting every BaseModel via ``model_dump(mode="json")``
        # so the result is JSON-native before passing to ``json.dumps``.
        # Lists/tuples (parent_states is a tuple of states) recurse;
        # dicts and scalars pass through unchanged.
        return json.dumps(_to_json_native(obj)).encode("utf-8")

    def _decode(self, blob: bytes, recorded_mode: str, invocation_id: str) -> Any:
        if recorded_mode != self._serialization:
            raise CheckpointRecordInvalid(
                invocation_id,
                f"record was written with serialization={recorded_mode!r} "
                f"but this checkpointer was constructed with "
                f"serialization={self._serialization!r}",
            )
        if recorded_mode == "pickle":
            return pickle.loads(blob)
        return json.loads(blob.decode("utf-8"))

    # ------------------------------------------------------------------
    # Protocol operations
    # ------------------------------------------------------------------

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        """Upsert ``record`` under ``invocation_id``. The state,
        completed positions, parent-state stack, and (per proposal 0009)
        per-fan-out-node progress are serialized via the configured
        :class:`SerializationMode` and written in a single statement.
        Writes are durable on return (WAL mode, per-write fsync at the
        SQLite layer)."""
        await self._ensure_initialized()
        state_blob = self._encode(record.state)
        positions_blob = self._encode([asdict(p) for p in record.completed_positions])
        parent_states_blob = self._encode(list(record.parent_states))
        # Per pipeline-utilities §10.11: serialize the per-fan-out-node
        # progress sequence. Empty tuple is the common case (no fan-outs
        # in flight at save time) and round-trips as an empty list.
        fan_out_progress_blob = self._encode(
            [_fan_out_progress_to_dict(fp) for fp in record.fan_out_progress]
        )
        serialization_mode = self._serialization

        def _do() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO checkpoints
                        (invocation_id, correlation_id, state_blob,
                         positions_blob, parent_states_blob, last_saved_at,
                         schema_version, serialization, fan_out_progress_blob)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(invocation_id) DO UPDATE SET
                        correlation_id        = excluded.correlation_id,
                        state_blob            = excluded.state_blob,
                        positions_blob        = excluded.positions_blob,
                        parent_states_blob    = excluded.parent_states_blob,
                        last_saved_at         = excluded.last_saved_at,
                        schema_version        = excluded.schema_version,
                        serialization         = excluded.serialization,
                        fan_out_progress_blob = excluded.fan_out_progress_blob
                    """,
                    (
                        invocation_id,
                        record.correlation_id,
                        state_blob,
                        positions_blob,
                        parent_states_blob,
                        record.last_saved_at,
                        record.schema_version,
                        serialization_mode,
                        fan_out_progress_blob,
                    ),
                )

        async with self._lock:
            await asyncio.to_thread(_do)

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        """Return the saved record for ``invocation_id`` or ``None``
        when no row exists. The serialization mode stored with the
        row is used to decode the blobs back, so a database written
        with one mode can still be loaded after the backend has been
        reconfigured."""
        await self._ensure_initialized()

        def _do() -> tuple[Any, ...] | None:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    SELECT correlation_id, state_blob, positions_blob,
                           parent_states_blob, last_saved_at,
                           schema_version, serialization, fan_out_progress_blob
                    FROM checkpoints
                    WHERE invocation_id = ?
                    """,
                    (invocation_id,),
                )
                row = cur.fetchone()
                return cast("tuple[Any, ...] | None", row)

        row = await asyncio.to_thread(_do)
        if row is None:
            return None
        (
            correlation_id,
            state_blob,
            positions_blob,
            parent_states_blob,
            last_saved_at,
            schema_version,
            recorded_serialization,
            fan_out_progress_blob,
        ) = row
        # Note: per spec §10.12 (proposal 0014), version mismatches
        # are no longer rejected at the backend boundary. The engine
        # routes mismatches through the migration registry on resume
        # (CheckpointStateMigrationMissing if no chain, else applies
        # the chain). The backend just round-trips the version
        # identifier as opaque data.
        state = self._decode(state_blob, recorded_serialization, invocation_id)
        position_dicts = self._decode(positions_blob, recorded_serialization, invocation_id)
        parent_states = self._decode(parent_states_blob, recorded_serialization, invocation_id)
        positions = tuple(
            NodePosition(
                namespace=tuple(p["namespace"]),
                node_name=p["node_name"],
                step=p["step"],
                attempt_index=p.get("attempt_index", 0),
                fan_out_index=p.get("fan_out_index"),
            )
            for p in position_dicts
        )
        # fan_out_progress_blob may be NULL on rows written before
        # proposal 0009 (the column was added via ALTER TABLE and
        # back-fills as NULL on pre-existing rows). Treat NULL as
        # "no fan-outs in flight at save time" — the empty-tuple
        # default on CheckpointRecord.
        if fan_out_progress_blob is None:
            fan_out_progress: tuple[FanOutProgress, ...] = ()
        else:
            fan_out_progress_dicts = self._decode(
                fan_out_progress_blob,
                recorded_serialization,
                invocation_id,
            )
            fan_out_progress = tuple(
                _fan_out_progress_from_dict(fp, invocation_id) for fp in fan_out_progress_dicts
            )
        return CheckpointRecord(
            invocation_id=invocation_id,
            correlation_id=correlation_id,
            state=state,
            completed_positions=positions,
            parent_states=tuple(parent_states),
            last_saved_at=last_saved_at,
            schema_version=schema_version,
            fan_out_progress=fan_out_progress,
        )

    async def list(self, filter: CheckpointFilter | None = None) -> Iterable[CheckpointSummary]:
        """Enumerate saved invocations as :class:`CheckpointSummary`
        rows, ordered by ``last_saved_at`` ascending. With
        ``filter.correlation_id`` set the SQL query is constrained at
        the database (indexed lookup); without a filter the full
        table is returned."""
        await self._ensure_initialized()

        def _do() -> list[tuple[Any, ...]]:
            with self._connect() as conn:
                if filter is not None and filter.correlation_id is not None:
                    cur = conn.execute(
                        """
                        SELECT invocation_id, correlation_id, last_saved_at,
                               positions_blob, serialization
                        FROM checkpoints
                        WHERE correlation_id = ?
                        ORDER BY last_saved_at
                        """,
                        (filter.correlation_id,),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT invocation_id, correlation_id, last_saved_at,
                               positions_blob, serialization
                        FROM checkpoints
                        ORDER BY last_saved_at
                        """
                    )
                return cast("list[tuple[Any, ...]]", cur.fetchall())

        rows = await asyncio.to_thread(_do)
        summaries: list[CheckpointSummary] = []
        for row in rows:
            invocation_id, correlation_id, last_saved_at, positions_blob, recorded_mode = row
            position_count = len(self._decode(positions_blob, recorded_mode, invocation_id))
            summaries.append(
                CheckpointSummary(
                    invocation_id=invocation_id,
                    correlation_id=correlation_id,
                    last_saved_at=last_saved_at,
                    completed_node_count=position_count,
                )
            )
        return summaries

    async def delete(self, invocation_id: str) -> None:
        """Remove the row for ``invocation_id``. No-op when no row
        exists (no error). The delete is durable on return."""
        await self._ensure_initialized()

        def _do() -> None:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM checkpoints WHERE invocation_id = ?",
                    (invocation_id,),
                )

        async with self._lock:
            await asyncio.to_thread(_do)


__all__ = ["SQLiteCheckpointer", "SerializationMode"]
