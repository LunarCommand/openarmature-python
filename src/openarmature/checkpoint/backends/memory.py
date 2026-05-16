# Spec: one of the reference backends listed in pipeline-utilities §10.11.

"""In-memory Checkpointer.

Keeps records in a Python ``dict`` keyed by ``invocation_id``. NOT
durable across process crashes — useful for tests, short-lived runs,
and development. Accepts any state shape (the dict holds the
:class:`CheckpointRecord` directly; nothing is serialized).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from ..protocol import CheckpointFilter, CheckpointRecord, CheckpointSummary


class InMemoryCheckpointer:
    """Dict-backed Checkpointer.

    **Durability:** none. Records live for the lifetime of this
    instance only; restarting the process loses everything.
    Appropriate for unit tests, the dev loop, and short-lived
    in-process pipelines that don't need crash recovery.

    **State shape:** any. The record is held by reference, so the
    Pydantic state instance the engine produces is what comes back
    from :meth:`load` — no serialization round-trip. (This is the
    feature: tests can assert on the saved state's identity.)

    **State-migration eligibility:** none. Per spec §10.12.1, a
    backend supports migration only when it can expose a structural
    intermediate form of the loaded state independent of the current
    state class. This backend holds live typed instances by
    reference, so a version mismatch on resume raises
    ``CheckpointRecordInvalid`` rather than consulting the
    migration registry.
    """

    # Per spec §10.12.1: in-memory storage holds live typed-state
    # references, so there's no class-independent intermediate form
    # the migration registry could consume. Declared at the class
    # level (not as a per-instance attribute) since the answer is
    # constructor-independent; the Protocol declaration in
    # ``protocol.py`` types this as ``bool`` (not ``ClassVar[bool]``)
    # so Pyright accepts a class-attribute override here.
    supports_state_migration: bool = False

    def __init__(self) -> None:
        self._records: dict[str, CheckpointRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        async with self._lock:
            self._records[invocation_id] = record

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        async with self._lock:
            return self._records.get(invocation_id)

    async def list(self, filter: CheckpointFilter | None = None) -> Iterable[CheckpointSummary]:
        async with self._lock:
            records = list(self._records.values())
        summaries = [
            CheckpointSummary(
                invocation_id=r.invocation_id,
                correlation_id=r.correlation_id,
                last_saved_at=r.last_saved_at,
                completed_node_count=len(r.completed_positions),
            )
            for r in records
        ]
        if filter is None or filter.correlation_id is None:
            return summaries
        return [s for s in summaries if s.correlation_id == filter.correlation_id]

    async def delete(self, invocation_id: str) -> None:
        async with self._lock:
            self._records.pop(invocation_id, None)


__all__ = ["InMemoryCheckpointer"]
