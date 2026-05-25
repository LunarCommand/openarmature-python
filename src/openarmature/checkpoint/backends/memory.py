# Spec: one of the reference backends listed in pipeline-utilities §10.11.

"""In-memory Checkpointer.

Keeps records in a Python ``dict`` keyed by ``invocation_id``. NOT
durable across process crashes; useful for tests, short-lived runs,
and development. Accepts any state shape (the dict holds the
:class:`CheckpointRecord` directly; nothing is serialized).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

from ..protocol import CheckpointFilter, CheckpointRecord, CheckpointSummary


@dataclass(frozen=True)
class FanOutInternalSaveBatching:
    """Per-Checkpointer-instance configuration for §10.11.4 fan-out
    internal save batching.

    Applies ONLY to fan-out instance internal saves. Outermost-graph,
    subgraph-internal, and fan-out node completion saves remain
    synchronous per §10.3.

    - ``flush_every``: flush the buffer every N buffered saves. ``0``
      / negative means batching is disabled (every save flushes
      immediately). The buffered save count resets at each flush.

    Buffered-but-unflushed saves are LOST on crash per §10.11.4;
    on resume, instances whose completed state was buffered-only
    revert to ``in_flight`` / ``not_started`` and re-run. The §10.11.1
    reducer correctness holds because their contributions hadn't
    durably committed.
    """

    flush_every: int = 0


class InMemoryCheckpointer:
    """Dict-backed Checkpointer.

    **Durability:** none. Records live for the lifetime of this
    instance only; restarting the process loses everything.
    Appropriate for unit tests, the dev loop, and short-lived
    in-process pipelines that don't need crash recovery.

    **State shape:** any. The record is held by reference, so the
    Pydantic state instance the engine produces is what comes back
    from :meth:`load`; no serialization round-trip. (This is the
    feature: tests can assert on the saved state's identity.)

    **State-migration eligibility:** none. Per spec §10.12.1, a
    backend supports migration only when it can expose a structural
    intermediate form of the loaded state independent of the current
    state class. This backend holds live typed instances by
    reference, so a version mismatch on resume raises
    ``CheckpointRecordInvalid`` rather than consulting the
    migration registry.

    **Fan-out internal save batching** (per spec §10.11.4): optional
    via the ``fan_out_internal_save_batching`` constructor parameter.
    Default is no batching (every save flushes immediately). When
    enabled, fan-out instance internal saves buffer in memory and
    flush every ``flush_every`` saves. Outermost-graph,
    subgraph-internal, and fan-out node completion saves bypass the
    buffer entirely (they remain synchronous). On crash, buffered
    saves are lost — by design, per §10.11.4's documented cost
    trade-off.
    """

    # Per spec §10.12.1: in-memory storage holds live typed-state
    # references, so there's no class-independent intermediate form
    # the migration registry could consume. Declared at the class
    # level (not as a per-instance attribute) since the answer is
    # constructor-independent; the Protocol declaration in
    # ``protocol.py`` types this as ``bool`` (not ``ClassVar[bool]``)
    # so Pyright accepts a class-attribute override here.
    supports_state_migration: bool = False

    def __init__(
        self,
        *,
        fan_out_internal_save_batching: FanOutInternalSaveBatching | None = None,
    ) -> None:
        self._records: dict[str, CheckpointRecord] = {}
        self._lock = asyncio.Lock()
        self._fan_out_batching = fan_out_internal_save_batching
        # Buffered fan-out internal saves keyed by invocation_id. Each
        # entry holds the latest buffered record for that invocation;
        # subsequent buffered saves overwrite (the most recent state
        # is what would have flushed). Per-invocation counts of
        # buffered saves decide when to flush per ``flush_every``;
        # keeping counts per-invocation isolates concurrent
        # invocations that share the same checkpointer.
        self._fan_out_buffer: dict[str, CheckpointRecord] = {}
        self._fan_out_buffer_counts: dict[str, int] = {}

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        """Store ``record`` under ``invocation_id``, replacing any
        previous record for the same id. Not durable across process
        restarts.

        Per §10.11.4: outermost-graph, subgraph-internal, and
        fan-out node completion saves are synchronous regardless of
        the batching configuration. The engine routes fan-out
        instance internal saves through :meth:`save_fan_out_internal`
        instead; this method bypasses the buffer.
        """
        async with self._lock:
            # Flush any buffered fan-out internal saves for this
            # invocation before recording the (synchronous) save —
            # otherwise a fan-out node completion save could land in
            # the persistent slot while a more-recent buffered
            # in-flight save sits in the buffer, inverting the
            # save order.
            self._flush_invocation_buffer_locked(invocation_id)
            self._records[invocation_id] = record

    async def save_fan_out_internal(self, invocation_id: str, record: CheckpointRecord) -> None:
        """Buffer a fan-out instance internal save under the §10.11.4
        batching policy. When batching is disabled (default), behaves
        identically to :meth:`save` — every save is synchronously
        durable. When ``flush_every`` is positive, the save is
        buffered; the buffer flushes when the count reaches the
        configured threshold.
        """
        if self._fan_out_batching is None or self._fan_out_batching.flush_every <= 0:
            await self.save(invocation_id, record)
            return
        async with self._lock:
            self._fan_out_buffer[invocation_id] = record
            self._fan_out_buffer_counts[invocation_id] = self._fan_out_buffer_counts.get(invocation_id, 0) + 1
            if self._fan_out_buffer_counts[invocation_id] >= self._fan_out_batching.flush_every:
                self._flush_invocation_buffer_locked(invocation_id)

    async def save_fan_out_in_flight_failure(
        self,
        invocation_id: str,
        record: CheckpointRecord,
    ) -> None:
        """Buffer an "instance failed mid-execution" save under §10.11.4
        batching. The failure save records the in_flight state of an
        instance whose terminal inner node raised; this save closes the
        in_flight observability gap (per §10.11) for instances whose
        subgraphs have no sibling-completed save to piggyback on.

        Under batching, this save buffers BUT does NOT count toward
        the flush threshold. The rationale: this save logically
        represents "the moment of crash" — a real crash wouldn't
        complete an extra save first; the buffered records (and this
        one) would simply be lost. The batching count-trigger mechanism
        is meant for steady-state save flow, not the abort path.

        Backends without batching route this to a synchronous
        :meth:`save` — the failure save is durable in the non-batching
        case (fixture 048's in_flight observability requirement).
        """
        if self._fan_out_batching is None or self._fan_out_batching.flush_every <= 0:
            await self.save(invocation_id, record)
            return
        async with self._lock:
            # Overwrite the buffer slot (the most-recent state is
            # what the next flush would capture if one fires later)
            # but DO NOT increment the count or trigger a flush.
            # On crash, this record is lost along with the rest of
            # the buffer — by design per §10.11.4.
            self._fan_out_buffer[invocation_id] = record

    def _flush_invocation_buffer_locked(self, invocation_id: str) -> None:
        """Caller-holds-lock helper: flush this invocation's buffered
        fan-out internal save (if any) into the persistent records
        dict. Resets only this invocation's buffer count, leaving
        other invocations' accounting untouched so concurrent
        invocations sharing the checkpointer don't interfere with
        each other's flush thresholds."""
        buffered = self._fan_out_buffer.pop(invocation_id, None)
        if buffered is not None:
            self._records[invocation_id] = buffered
        self._fan_out_buffer_counts.pop(invocation_id, None)

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        """Return the saved record for ``invocation_id`` or ``None``
        if nothing has been saved under that id. Per §10.11.4:
        buffered-but-unflushed fan-out internal saves are NOT visible
        to ``load`` — that's the crash-loses-buffered contract. To
        simulate a crash before the buffer flushes, drop the
        Checkpointer reference; the buffer is in-memory only.
        """
        async with self._lock:
            return self._records.get(invocation_id)

    async def list(self, filter: CheckpointFilter | None = None) -> Iterable[CheckpointSummary]:
        """Enumerate stored invocations as :class:`CheckpointSummary`
        rows. With ``filter.correlation_id`` set, restricts the
        results to invocations carrying that correlation id;
        otherwise returns all rows."""
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
        """Remove the record for ``invocation_id``. No-op when nothing
        is saved under that id (no error)."""
        async with self._lock:
            self._records.pop(invocation_id, None)


__all__ = ["FanOutInternalSaveBatching", "InMemoryCheckpointer"]
