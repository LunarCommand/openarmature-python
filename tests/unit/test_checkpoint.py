"""Focused tests for the checkpoint module.

The conformance suite (``tests/conformance/test_checkpoint.py``)
covers the spec's behavioral surface end-to-end against the
fixtures. These unit tests fill gaps the conformance suite doesn't
exercise directly: backend round-trip + durability, the canonical
category-string contract, schema_version mismatch handling, the
fan-out save gate, the §10.1.1 default-off behavior, and the
subgraph-resume parent_states preservation that fixture 029 covers
in conformance but is awaiting spec namespace clarification (see
test_checkpoint.py's _DEFERRED_FIXTURES note).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from openarmature.checkpoint import (
    Checkpointer,
    CheckpointFilter,
    CheckpointNotFound,
    CheckpointRecord,
    CheckpointRecordInvalid,
    CheckpointSaveFailed,
    InMemoryCheckpointer,
    NodePosition,
    SQLiteCheckpointer,
)
from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    NodeException,
    State,
)

# ---------------------------------------------------------------------------
# Error category contract
# ---------------------------------------------------------------------------


def test_checkpoint_not_found_category() -> None:
    err = CheckpointNotFound("abc")
    assert err.category == "checkpoint_not_found"
    assert err.invocation_id == "abc"


def test_checkpoint_save_failed_category_and_cause() -> None:
    underlying = RuntimeError("disk full")
    err = CheckpointSaveFailed("xyz", underlying)
    assert err.category == "checkpoint_save_failed"
    assert err.invocation_id == "xyz"
    assert err.__cause__ is underlying


def test_checkpoint_record_invalid_category() -> None:
    err = CheckpointRecordInvalid("xyz", "schema mismatch")
    assert err.category == "checkpoint_record_invalid"
    assert err.invocation_id == "xyz"


# ---------------------------------------------------------------------------
# NodePosition + CheckpointRecord shape
# ---------------------------------------------------------------------------


def test_node_position_is_hashable() -> None:
    p1 = NodePosition(namespace=("a",), node_name="x", step=0)
    p2 = NodePosition(namespace=("a",), node_name="x", step=0)
    s = {p1, p2}
    # Equal positions collapse in a set.
    assert len(s) == 1


def test_node_position_distinct_attempts_unequal() -> None:
    p1 = NodePosition(namespace=(), node_name="x", step=0, attempt_index=0)
    p2 = NodePosition(namespace=(), node_name="x", step=0, attempt_index=1)
    assert p1 != p2


def test_checkpoint_record_default_schema_version() -> None:
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={},
        completed_positions=(),
        parent_states=(),
        last_saved_at=0.0,
    )
    # Default schema_version is the empty-string sentinel per spec
    # §10.2 (proposal 0014): records carry the user's state-schema
    # version, which is "" until the state class declares one.
    assert record.schema_version == ""
    assert record.fan_out_progress == ()


# ---------------------------------------------------------------------------
# InMemoryCheckpointer round-trip
# ---------------------------------------------------------------------------


async def test_in_memory_round_trip() -> None:
    cp = InMemoryCheckpointer()
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 1},
        completed_positions=(NodePosition(namespace=(), node_name="a", step=0),),
        parent_states=(),
        last_saved_at=42.0,
    )
    await cp.save("i", record)
    loaded = await cp.load("i")
    assert loaded == record


async def test_in_memory_load_missing_returns_none() -> None:
    cp = InMemoryCheckpointer()
    assert await cp.load("ghost") is None


async def test_in_memory_delete_missing_is_noop() -> None:
    cp = InMemoryCheckpointer()
    # No exception.
    await cp.delete("ghost")


async def test_in_memory_list_filter_by_correlation_id() -> None:
    cp = InMemoryCheckpointer()
    rec1 = CheckpointRecord(
        invocation_id="i1",
        correlation_id="A",
        state={},
        completed_positions=(),
        parent_states=(),
        last_saved_at=1.0,
    )
    rec2 = CheckpointRecord(
        invocation_id="i2",
        correlation_id="B",
        state={},
        completed_positions=(),
        parent_states=(),
        last_saved_at=2.0,
    )
    await cp.save("i1", rec1)
    await cp.save("i2", rec2)
    all_summaries = list(await cp.list())
    assert len(all_summaries) == 2
    just_a = list(await cp.list(CheckpointFilter(correlation_id="A")))
    assert len(just_a) == 1
    assert just_a[0].invocation_id == "i1"


# ---------------------------------------------------------------------------
# SQLiteCheckpointer round-trip + durability + serialization knob
# ---------------------------------------------------------------------------


async def test_sqlite_pickle_round_trip(tmp_path: Path) -> None:
    cp = SQLiteCheckpointer(tmp_path / "ck.db", serialization="pickle")
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 1, "tag": "hi"},
        completed_positions=(NodePosition(namespace=(), node_name="a", step=0),),
        parent_states=(),
        last_saved_at=42.0,
    )
    await cp.save("i", record)
    loaded = await cp.load("i")
    assert loaded is not None
    assert loaded.invocation_id == "i"
    assert loaded.correlation_id == "c"
    assert loaded.state == {"x": 1, "tag": "hi"}
    assert loaded.completed_positions == record.completed_positions


async def test_sqlite_json_round_trip_with_pydantic_state(tmp_path: Path) -> None:
    """Spec §10.11: JSON mode accepts Pydantic State instances. The
    backend's encoder MUST walk the value tree converting BaseModel
    instances via ``model_dump(mode="json")`` before ``json.dumps`` —
    otherwise live State instances handed in by the engine raise
    TypeError. Round-trip a Pydantic State subclass plus a tuple of
    parent states (also Pydantic) to lock the behavior in."""

    class _SaveState(State):
        x: int = 0
        tag: str = ""

    class _OuterState(State):
        outer_flag: bool = False

    cp = SQLiteCheckpointer(tmp_path / "ck.db", serialization="json")
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state=_SaveState(x=1, tag="hi"),
        completed_positions=(NodePosition(namespace=("dispatch",), node_name="step", step=0),),
        parent_states=(_OuterState(outer_flag=True),),
        last_saved_at=42.0,
    )
    # Save MUST NOT raise (the bug being fixed: live Pydantic in
    # record.state -> json.dumps TypeError).
    await cp.save("i", record)
    loaded = await cp.load("i")
    assert loaded is not None
    # JSON mode loads dicts (no Pydantic types preserved). The resume
    # path in CompiledGraph.invoke is responsible for re-validating
    # against the declared state class — verified separately.
    assert loaded.state == {"x": 1, "tag": "hi"}
    assert list(loaded.parent_states) == [{"outer_flag": True}]
    assert loaded.completed_positions == record.completed_positions


async def test_sqlite_durability_across_reopen(tmp_path: Path) -> None:
    """A SQLite save MUST be durable: re-open the same file, load the
    record, and confirm it matches what was written."""
    db_path = tmp_path / "ck.db"
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 1},
        completed_positions=(NodePosition(namespace=(), node_name="a", step=0),),
        parent_states=(),
        last_saved_at=99.0,
    )
    cp1 = SQLiteCheckpointer(db_path)
    await cp1.save("i", record)
    # New instance simulates a process restart.
    cp2 = SQLiteCheckpointer(db_path)
    loaded = await cp2.load("i")
    assert loaded is not None
    assert loaded.invocation_id == "i"
    assert loaded.state == {"x": 1}


async def test_sqlite_upsert_retention(tmp_path: Path) -> None:
    """Spec §10.3.1: upsert — one row per invocation_id, overwritten on
    every save. After two saves with the same id, only the second
    record is retrievable."""
    cp = SQLiteCheckpointer(tmp_path / "ck.db")
    rec1 = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 1},
        completed_positions=(),
        parent_states=(),
        last_saved_at=1.0,
    )
    rec2 = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 2},
        completed_positions=(),
        parent_states=(),
        last_saved_at=2.0,
    )
    await cp.save("i", rec1)
    await cp.save("i", rec2)
    loaded = await cp.load("i")
    assert loaded is not None
    assert loaded.state == {"x": 2}
    summaries = list(await cp.list())
    assert len(summaries) == 1


async def test_sqlite_serialization_mismatch_raises(tmp_path: Path) -> None:
    """A record written with serialization=pickle MUST NOT be loadable
    by a checkpointer constructed with serialization=json (and vice
    versa). The mismatch raises CheckpointRecordInvalid per §10.10."""
    db_path = tmp_path / "ck.db"
    cp_pickle = SQLiteCheckpointer(db_path, serialization="pickle")
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 1},
        completed_positions=(),
        parent_states=(),
        last_saved_at=1.0,
    )
    await cp_pickle.save("i", record)
    cp_json = SQLiteCheckpointer(db_path, serialization="json")
    with pytest.raises(CheckpointRecordInvalid):
        await cp_json.load("i")


# ---------------------------------------------------------------------------
# Schema version round-trips and rejects mismatches
# ---------------------------------------------------------------------------


async def test_schema_version_round_trips(tmp_path: Path) -> None:
    cp = SQLiteCheckpointer(tmp_path / "ck.db")
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 1},
        completed_positions=(),
        parent_states=(),
        last_saved_at=1.0,
    )
    await cp.save("i", record)
    loaded = await cp.load("i")
    assert loaded is not None
    # Records round-trip the user-facing schema_version verbatim per
    # spec §10.2. With no version declared on the saved state, the
    # default sentinel is "".
    assert loaded.schema_version == ""


async def test_schema_version_round_trips_through_sqlite_unchanged(tmp_path: Path) -> None:
    """Per spec §10.12 (proposal 0014), the SQLite backend no longer
    rejects records with non-default ``schema_version`` values — that
    routing is now an engine concern at resume time. The backend
    just round-trips the version identifier as opaque data so the
    engine's migration registry has the chance to bridge it."""
    cp = SQLiteCheckpointer(tmp_path / "ck.db")
    record = CheckpointRecord(
        invocation_id="i",
        correlation_id="c",
        state={"x": 1},
        completed_positions=(),
        parent_states=(),
        last_saved_at=1.0,
        schema_version="999",  # an arbitrary user-facing identifier
    )
    await cp.save("i", record)
    loaded = await cp.load("i")
    assert loaded is not None
    assert loaded.schema_version == "999"


# ---------------------------------------------------------------------------
# Engine integration: §10.1.1 default-off behavior
# ---------------------------------------------------------------------------


class _SimpleState(State):
    a: int = 0
    b: int = 0


async def _node_a(_s: _SimpleState) -> dict[str, int]:
    return {"a": 1}


async def _node_b(_s: _SimpleState) -> dict[str, int]:
    return {"b": 2}


def _build_simple_graph(checkpointer: Checkpointer | None = None) -> CompiledGraph[_SimpleState]:
    builder = (
        GraphBuilder(_SimpleState)
        .add_node("node_a", _node_a)
        .add_node("node_b", _node_b)
        .add_edge("node_a", "node_b")
        .add_edge("node_b", END)
        .set_entry("node_a")
    )
    if checkpointer is not None:
        builder.with_checkpointer(checkpointer)
    return builder.compile()


async def test_no_checkpointer_means_no_saves() -> None:
    """§10.1.1: without a registered Checkpointer the engine never
    calls ``save()`` — no record is produced."""
    compiled = _build_simple_graph(None)
    final = await compiled.invoke(_SimpleState())
    assert final.a == 1
    assert final.b == 2


async def test_no_checkpointer_resume_raises_not_found() -> None:
    """§10.1.1: ``invoke(resume_invocation=X)`` against an unregistered
    backend raises checkpoint_not_found — the user has misconfigured
    the run."""
    compiled = _build_simple_graph(None)
    with pytest.raises(CheckpointNotFound):
        await compiled.invoke(_SimpleState(), resume_invocation="ghost")


async def test_resume_against_empty_checkpointer_raises_not_found() -> None:
    """§10.10: load() returning None surfaces as
    checkpoint_not_found."""
    cp = InMemoryCheckpointer()
    compiled = _build_simple_graph(cp)
    with pytest.raises(CheckpointNotFound):
        await compiled.invoke(_SimpleState(), resume_invocation="ghost")


async def test_resume_with_invalid_saved_state_raises_record_invalid() -> None:
    """§10.10: a saved record whose state-shape doesn't validate
    against the current graph's state class MUST surface as
    ``checkpoint_record_invalid``, not a raw pydantic ValidationError.
    Models the JSON-serialized backend path: the load returns a
    dict that the engine re-validates against ``state_cls``; an
    incompatible dict trips ``model_validate`` which the engine
    wraps."""
    cp = InMemoryCheckpointer()
    # Hand-craft a record whose state dict can't validate against
    # _SimpleState (extra="forbid" on State + missing required fields
    # → ValidationError on model_validate).
    bad_record = CheckpointRecord(
        invocation_id="bogus",
        correlation_id="c",
        # _SimpleState has int fields; passing a string for `a` will
        # trip the type-coerce guard.
        state={"a": "not-an-int", "b": 0},
        completed_positions=(),
        parent_states=(),
        last_saved_at=1.0,
    )
    await cp.save("bogus", bad_record)
    compiled = _build_simple_graph(cp)
    with pytest.raises(CheckpointRecordInvalid):
        await compiled.invoke(_SimpleState(), resume_invocation="bogus")


# ---------------------------------------------------------------------------
# Save-failure policy
# ---------------------------------------------------------------------------


class _AlwaysFailingCheckpointer:
    """Backend whose ``save`` always raises. Engine wraps the failure
    as :class:`CheckpointSaveFailed` and raises immediately to the
    caller of ``invoke()`` per the documented save-failure policy."""

    supports_state_migration: bool = False

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        raise RuntimeError("simulated backend failure")

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        return None

    async def list(self, filter: Any = None) -> Any:
        return []

    async def delete(self, invocation_id: str) -> None:
        return None


async def test_save_failure_raises_to_invoke_caller() -> None:
    compiled = _build_simple_graph(_AlwaysFailingCheckpointer())
    with pytest.raises(CheckpointSaveFailed):
        await compiled.invoke(_SimpleState())


# ---------------------------------------------------------------------------
# Per-instance fan-out resume contract (proposal 0009 / spec v0.18.0)
# ---------------------------------------------------------------------------


class _CapturingCheckpointer:
    supports_state_migration: bool = False

    def __init__(self) -> None:
        self.saves: list[CheckpointRecord] = []
        self._records: dict[str, CheckpointRecord] = {}

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        self.saves.append(record)
        self._records[invocation_id] = record

    async def save_fan_out_internal(self, invocation_id: str, record: CheckpointRecord) -> None:
        await self.save(invocation_id, record)

    async def save_fan_out_in_flight_failure(self, invocation_id: str, record: CheckpointRecord) -> None:
        await self.save(invocation_id, record)

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        return self._records.get(invocation_id)

    async def list(self, filter: Any = None) -> Any:
        return []

    async def delete(self, invocation_id: str) -> None:
        return None


class _ItemState(State):
    item: int = 0
    out: int = 0


class _ParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: list[int] = Field(default_factory=list[int])


async def _scorer(s: _ItemState) -> dict[str, int]:
    return {"out": s.item + 100}


async def test_fan_out_internal_saves_fire_per_instance() -> None:
    """Per spec §10.3 (revised by proposal 0009 / v0.18.0): fan-out
    instance internal nodes DO produce saves. Each per-instance
    completion emits at least one save with ``fan_out_index``
    populated on the inner-node position, plus an explicit "instance
    completed" save that flips the instance's ``fan_out_progress``
    state to ``completed``."""
    inner = (
        GraphBuilder(_ItemState)
        .add_node("scorer", _scorer)
        .add_edge("scorer", END)
        .set_entry("scorer")
        .compile()
    )
    cp = _CapturingCheckpointer()
    parent = (
        GraphBuilder(_ParentState)
        .add_fan_out_node(
            "fan",
            subgraph=inner,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
            concurrency=2,
        )
        .add_edge("fan", END)
        .set_entry("fan")
        .with_checkpointer(cp)
        .compile()
    )
    await parent.invoke(_ParentState(items=[1, 2, 3]))
    # Some inner-node saves fire (one per instance per inner node)
    # AND the fan-out node's own completion save. Total >= instances +
    # 1 (instance count + fan-out completion); the explicit
    # ``_save_instance_completed`` adds another save per instance.
    assert len(cp.saves) >= 3
    # At least one save carries an inner position with fan_out_index
    # populated — that's the inner-node save inside an instance,
    # recorded against the per-instance ``completed_inner_positions``
    # field on ``fan_out_progress`` (per spec §10.11).
    saves_with_inner_positions = [
        s
        for s in cp.saves
        for fp in s.fan_out_progress
        for inst in fp.instances
        if inst.completed_inner_positions
    ]
    assert saves_with_inner_positions, "expected at least one save with per-instance inner positions"
    # The terminal save (fan-out node's own completion) carries the
    # outer "fan" position with fan_out_index=None.
    last_save = cp.saves[-1]
    fan_positions = [p for p in last_save.completed_positions if p.node_name == "fan"]
    assert len(fan_positions) == 1
    assert fan_positions[0].fan_out_index is None


# ---------------------------------------------------------------------------
# Q4 from the spec impl review: focused unit test on fail_fast fast-cancel
# ensuring the failed instance lands as in_flight (no result) on the
# saved record after cancellation completes.
# ---------------------------------------------------------------------------


class _FailingItemState(State):
    item: int = 0
    out: int = 0


class _FailingParentState(State):
    items: list[int] = Field(default_factory=list[int])
    results: list[int] = Field(default_factory=list[int])


_failing_instance_counter = [0]


async def _failing_scorer(s: _FailingItemState) -> dict[str, int]:
    # Fail when item == 999 (sentinel). All others succeed and
    # contribute ``out = item``. The sentinel is positioned in the
    # items list to trigger fail_fast cancellation of siblings.
    if s.item == 999:
        raise RuntimeError(f"intentional failure for item {s.item}")
    return {"out": s.item}


async def test_fail_fast_cancellation_leaves_failed_instance_in_flight() -> None:
    """Per §10.11.2 fail_fast cancellation contract: the failed
    instance's ``fan_out_progress`` state on the saved record is
    ``in_flight`` (no ``result`` recorded), and cancelled siblings
    are also ``in_flight`` or ``not_started`` — never ``completed``
    for the failed slot. Closes the spec impl-review Q4 follow-on."""
    inner = (
        GraphBuilder(_FailingItemState)
        .add_node("scorer", _failing_scorer)
        .add_edge("scorer", END)
        .set_entry("scorer")
        .compile()
    )
    cp = _CapturingCheckpointer()
    parent = (
        GraphBuilder(_FailingParentState)
        .add_fan_out_node(
            "fan",
            subgraph=inner,
            collect_field="out",
            target_field="results",
            items_field="items",
            item_field="item",
            concurrency=1,  # serial so the failure ordering is deterministic
            error_policy="fail_fast",
        )
        .add_edge("fan", END)
        .set_entry("fan")
        .with_checkpointer(cp)
        .compile()
    )
    # Items: [10, 20, 999, 40] — instance 2 (item 999) fails. The
    # engine wraps the raw RuntimeError as ``NodeException``.
    with pytest.raises(NodeException):
        await parent.invoke(_FailingParentState(items=[10, 20, 999, 40]))
    # Locate the latest save's fan_out_progress for the "fan" node.
    assert cp.saves, "expected at least one save to fire"
    latest = cp.saves[-1]
    fan_progress = next(
        (fp for fp in latest.fan_out_progress if fp.fan_out_node_name == "fan"),
        None,
    )
    assert fan_progress is not None, "expected fan_out_progress entry for the 'fan' node"
    # Per §10.11.2: failed instance (idx 2) state is ``in_flight``
    # (no ``result`` recorded). Successful preceding instances
    # (0, 1) are ``completed``; cancelled siblings (3) are
    # ``in_flight`` or ``not_started``.
    assert fan_progress.instances[0].state == "completed"
    assert fan_progress.instances[1].state == "completed"
    assert fan_progress.instances[2].state == "in_flight", (
        f"failed instance state should be in_flight, got {fan_progress.instances[2].state!r}"
    )
    assert fan_progress.instances[2].result is None, (
        f"failed instance result should be None, got {fan_progress.instances[2].result!r}"
    )
    assert fan_progress.instances[3].state in {"in_flight", "not_started"}, (
        f"cancelled sibling state should be in_flight or not_started, got {fan_progress.instances[3].state!r}"
    )


# ---------------------------------------------------------------------------
# Resume re-entry into subgraph: parent_states populated on inner-node saves
# ---------------------------------------------------------------------------


class _OuterState(State):
    flag: bool = False
    result: int = 0


class _InnerState(State):
    inner_flag: bool = False
    result: int = 0


async def _inner_step(_s: _InnerState) -> dict[str, Any]:
    return {"inner_flag": True, "result": 42}


async def test_inner_node_save_carries_parent_states() -> None:
    """Spec §10.2: a save from inside a subgraph populates
    ``parent_states`` with the chain of containing-graph states.
    This is the contract that fixture 029 verifies in conformance —
    here we isolate the parent_states logic without depending on
    the namespace-convention question."""
    inner = (
        GraphBuilder(_InnerState)
        .add_node("step", _inner_step)
        .add_edge("step", END)
        .set_entry("step")
        .compile()
    )
    cp = _CapturingCheckpointer()
    outer = (
        GraphBuilder(_OuterState)
        .add_subgraph_node("dispatch", inner)
        .add_edge("dispatch", END)
        .set_entry("dispatch")
        .with_checkpointer(cp)
        .compile()
    )
    await outer.invoke(_OuterState())
    # The subgraph wrapper has no started/completed events of its own
    # (transparent per fixture 013), so no save fires for it. Only the
    # inner "step" node saves. Per spec §10.2, that save's
    # ``parent_states`` carries the chain of containing-graph states.
    inner_save = next(
        (s for s in cp.saves if s.completed_positions[-1].node_name == "step"),
        None,
    )
    assert inner_save is not None, (
        f"expected a save from the inner node; got {[s.completed_positions[-1].node_name for s in cp.saves]}"
    )
    assert len(inner_save.parent_states) == 1, (
        f"inner save should carry one parent state; got {len(inner_save.parent_states)}"
    )
    # And no save fires for the wrapper itself.
    assert all(s.completed_positions[-1].node_name != "dispatch" for s in cp.saves), (
        "no save should fire for the subgraph wrapper (no completed event per graph-engine §6 + fixture 013)"
    )


# ---------------------------------------------------------------------------
# Resume mints new invocation_id but preserves correlation_id
# ---------------------------------------------------------------------------


_first_run_should_fail = [True]


async def _flaky_node(_s: _SimpleState) -> dict[str, int]:
    if _first_run_should_fail[0]:
        _first_run_should_fail[0] = False
        raise RuntimeError("first-run abort")
    return {"a": 1}


async def test_resume_preserves_correlation_id_and_mints_new_invocation_id() -> None:
    """Spec §10.4 steps 3+4: resume MUST keep the original
    correlation_id verbatim (cross-backend join key) AND mint a new
    invocation_id (each attempt is its own invocation)."""
    _first_run_should_fail[0] = True
    cp = InMemoryCheckpointer()
    # First run aborts; we install a capturing wrapper on save() to
    # snag the engine-minted invocation_id and correlation_id from
    # the most recent persisted record (the latest save fires from
    # node_a, before flaky fails).
    saved_invocation_ids: list[str] = []
    saved_correlation_ids: list[str] = []
    original_save = cp.save

    async def capture_save(invocation_id: str, record: CheckpointRecord) -> None:
        saved_invocation_ids.append(invocation_id)
        saved_correlation_ids.append(record.correlation_id)
        await original_save(invocation_id, record)

    cp.save = capture_save  # type: ignore[method-assign]

    # Drive the failing path. The flaky node fails before any save
    # fires, so we need a successful first-run state. Use a graph
    # where node_a saves before flaky.
    saved_invocation_ids.clear()
    saved_correlation_ids.clear()

    builder2 = (
        GraphBuilder(_SimpleState)
        .add_node("node_a", _node_a)
        .add_node("flaky", _flaky_node)
        .add_edge("node_a", "flaky")
        .add_edge("flaky", END)
        .set_entry("node_a")
        .with_checkpointer(cp)
    )
    compiled2 = builder2.compile()
    _first_run_should_fail[0] = True
    with pytest.raises(NodeException):
        await compiled2.invoke(_SimpleState(), correlation_id="my-correlation")
    assert saved_invocation_ids, "expected at least one save before the abort"
    first_invocation_id = saved_invocation_ids[-1]
    first_correlation_id = saved_correlation_ids[-1]
    assert first_correlation_id == "my-correlation"

    # Resume.
    saved_invocation_ids.clear()
    saved_correlation_ids.clear()
    await compiled2.invoke(_SimpleState(), resume_invocation=first_invocation_id)
    assert saved_invocation_ids, "expected saves on the resumed run"
    resumed_invocation_id = saved_invocation_ids[-1]
    resumed_correlation_id = saved_correlation_ids[-1]
    assert resumed_invocation_id != first_invocation_id, (
        "resume MUST mint a new invocation_id per §10.4 step 4"
    )
    assert resumed_correlation_id == "my-correlation", (
        "resume MUST preserve the original correlation_id per §10.4 step 3"
    )
