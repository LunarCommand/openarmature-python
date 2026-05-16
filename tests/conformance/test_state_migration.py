"""Run every spec state-migration conformance fixture (039-046) end-to-end.

The fixtures live under
``spec/pipeline-utilities/conformance/`` as ``cases`` shapes; each
case defines a state schema (with a ``schema_version``), an entry
node and edges, a ``seeded_record:`` describing a checkpoint that
was saved at a prior schema version, a ``migrations:`` list (each
naming one of the harness mock functions), and a ``resume`` block
specifying either an ``expected`` happy-path or an
``expected_error`` raise.

The driver:

1. Builds a State subclass via ``adapter.build_state_cls``, then
   patches the generated class's ``schema_version`` ClassVar so
   it matches the fixture's declared version.
2. Builds a minimal graph (entry node + edge to END) via the
   existing adapter primitives.
3. Resolves each ``migrations[i].migrate`` name against the mock
   library, wrapping every mock so the harness can count
   invocations + record the ``v1->v2`` ordered list (for the
   ``migrations_run`` / ``migration_count`` /
   ``migration_order_matches_chain`` assertions).
4. Seeds a ``CheckpointRecord`` via the configured backend
   (SQLite in JSON mode) using a stable seeded ``invocation_id``.
5. Calls ``invoke(resume_invocation=<seeded id>)`` and asserts
   against ``resume.expected`` (final state, migrations_run,
   invariants) OR ``resume.expected_error`` (category, carries,
   cause).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.checkpoint import (
    CheckpointError,
    CheckpointRecord,
    CheckpointRecordInvalid,
    CheckpointStateMigrationFailed,
    CheckpointStateMigrationMissing,
    SQLiteCheckpointer,
)
from openarmature.graph import END, GraphBuilder, State

from .adapter import build_state_cls

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "pipeline-utilities" / "conformance"
)

_FIXTURE_RANGE = range(39, 47)

# ---------------------------------------------------------------------------
# Migration mock library (harness-side; fixtures refer by name)
# ---------------------------------------------------------------------------

# Wrapped mocks all receive a dict (JSON-mode SQLite hands the engine the
# structural intermediate form). Each returns a dict suitable as input
# for the next migration in the chain (or for final deserialization).


def _add_new_field_default(state: Any) -> Any:
    out = dict(state)
    out["new_field"] = "v2_default"
    return out


def _add_v2_field(state: Any) -> Any:
    out = dict(state)
    out["v2_field"] = "v2_default"
    return out


def _add_v3_field(state: Any) -> Any:
    out = dict(state)
    out["v3_field"] = "v3_default"
    return out


def _identity_passthrough(state: Any) -> Any:
    # Used by 044 to verify post-migration deserialization-failure
    # routing: the migration runs cleanly but produces output that
    # the v2 state class can't deserialize (missing a required field).
    return dict(state)


def _raises_keyerror(_state: Any) -> Any:
    # Used by 046 to verify CheckpointStateMigrationFailed routing.
    raise KeyError("simulated buggy migration")


def _should_not_run(_state: Any) -> Any:
    # Used by 042 (versions-match no-op) to verify the engine does
    # NOT consult the migration registry when versions match.
    raise AssertionError("fixture 042 invariant violated: should_not_run was called despite version match")


def _irrelevant(state: Any) -> Any:
    # Used by 045 — migration is registered but the engine doesn't
    # find a path to it. Returns input unchanged.
    return dict(state)


_MOCK_LIBRARY: dict[str, Any] = {
    "add_new_field_default": _add_new_field_default,
    "add_v2_field": _add_v2_field,
    "add_v3_field": _add_v3_field,
    "identity_passthrough": _identity_passthrough,
    "raises_keyerror": _raises_keyerror,
    "should_not_run": _should_not_run,
    "irrelevant": _irrelevant,
}


class _MigrationTrace:
    """Captures the order migrations were invoked in, for the
    ``migrations_run`` / ``migration_count`` /
    ``migration_order_matches_chain`` fixture assertions."""

    def __init__(self) -> None:
        self.order: list[str] = []

    def wrap(self, name: str, fn: Any, from_v: str, to_v: str) -> Any:
        label = f"{from_v}->{to_v}"

        def _traced(state: Any) -> Any:
            self.order.append(label)
            return fn(state)

        _traced.__name__ = f"_traced_{name}"
        return _traced


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------


def _fixture_paths() -> list[Path]:
    out: list[Path] = []
    for p in sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")):
        try:
            number = int(p.stem.split("-", 1)[0])
        except ValueError:
            continue
        if number in _FIXTURE_RANGE:
            out.append(p)
    return out


def _fixture_id(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_state_cls(state_spec: dict[str, Any], model_name: str) -> type[State]:
    """Build a Pydantic State subclass from the fixture spec and stamp
    its ``schema_version`` ClassVar with the declared version."""
    fields_spec = state_spec.get("fields", {})
    state_cls = build_state_cls(model_name, fields_spec)
    # Stamp the per-fixture schema_version. The adapter's build_state_cls
    # produces a fresh subclass via pydantic.create_model so it's safe
    # to set the ClassVar after construction.
    state_cls.schema_version = state_spec.get("schema_version", "")
    return state_cls


async def _seed_record(
    checkpointer: SQLiteCheckpointer,
    invocation_id: str,
    seeded: dict[str, Any],
) -> None:
    """Persist a checkpoint record matching the fixture's
    ``seeded_record:`` block. ``state`` and ``parent_states`` go
    through as plain dicts (JSON-mode round-trip)."""
    raw_positions: list[dict[str, Any]] = seeded.get("completed_positions", [])
    from openarmature.checkpoint import NodePosition

    positions = tuple(
        NodePosition(
            namespace=tuple(p.get("namespace", [])),
            node_name=p["node_name"],
            step=p["step"],
            attempt_index=p.get("attempt_index", 0),
            fan_out_index=p.get("fan_out_index"),
        )
        for p in raw_positions
    )
    record = CheckpointRecord(
        invocation_id=invocation_id,
        correlation_id=seeded.get("correlation_id", "seeded-corr"),
        state=seeded["state"],
        completed_positions=positions,
        parent_states=tuple(seeded.get("parent_states", [])),
        last_saved_at=0.0,
        schema_version=seeded.get("schema_version", ""),
    )
    await checkpointer.save(invocation_id, record)


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------


async def _run_one_case(case: dict[str, Any], tmp_path: Path) -> None:
    """Run one fixture case end-to-end: build, seed, resume, assert."""
    state_cls = _build_state_cls(case["state"], model_name=f"Case_{case['name']}")

    # Minimal node body: apply ``update_pure`` to the state.
    nodes_spec = case["nodes"]
    edges_spec = case["edges"]
    entry = case["entry"]

    builder = GraphBuilder(state_cls)
    for node_name, node_spec in nodes_spec.items():
        update_pure = cast("dict[str, Any]", node_spec.get("update_pure", {}))

        async def _node_body(_s: State, _u: dict[str, Any] = update_pure) -> dict[str, Any]:
            return _u

        builder.add_node(node_name, _node_body)
    for edge in edges_spec:
        target_raw = edge["to"]
        target = END if target_raw == "END" else target_raw
        builder.add_edge(edge["from"], target)
    builder.set_entry(entry)

    # Register migrations + wrap each in the trace recorder.
    trace = _MigrationTrace()
    for m in case.get("migrations", []):
        mock = _MOCK_LIBRARY[m["migrate"]]
        wrapped = trace.wrap(m["migrate"], mock, m["from_version"], m["to_version"])
        builder.with_state_migration(m["from_version"], m["to_version"], wrapped)

    # Configure the SQLite backend in JSON mode (the migration-eligible
    # backend per spec §10.12.1). One database file per case, isolated
    # under tmp_path.
    db_path = tmp_path / f"{case['name']}.db"
    checkpointer = SQLiteCheckpointer(db_path, serialization="json")
    builder.with_checkpointer(checkpointer)
    compiled = builder.compile()

    # Seed the prior record.
    invocation_id = f"seeded-{case['name']}"
    if "seeded_record" in case:
        await _seed_record(checkpointer, invocation_id, case["seeded_record"])

    # Resume + assert.
    resume = case["resume"]
    # For resume-from-seeded cases the engine loads state from the
    # checkpoint and never reads ``initial_state``; using
    # ``model_construct`` skips Pydantic validation so fixtures with
    # required fields (e.g., 044) can construct a placeholder without
    # tripping the validator before the resume even starts.
    initial_state = state_cls.model_construct()
    raised: BaseException | None = None
    final_state: Any = None
    try:
        if resume.get("from_seeded_record"):
            final_state = await compiled.invoke(initial_state, resume_invocation=invocation_id)
        else:
            final_state = await compiled.invoke(initial_state)
    except CheckpointError as exc:
        raised = exc

    if "expected_error" in resume:
        _assert_error(resume["expected_error"], resume.get("invariants", {}), raised)
    elif "expected" in resume:
        assert raised is None, f"expected success, got {raised!r}"
        _assert_success(resume["expected"], resume.get("invariants", {}), final_state, trace)


def _assert_error(
    expected_error: dict[str, Any],
    invariants: dict[str, Any],
    raised: BaseException | None,
) -> None:
    assert raised is not None, (
        f"expected raise of category {expected_error.get('category')!r}, got no exception"
    )
    actual_category = getattr(raised, "category", None)
    assert actual_category == expected_error["category"], (
        f"expected category {expected_error['category']!r}, got {actual_category!r} ({raised!r})"
    )
    for key, expected_value in expected_error.get("carries", {}).items():
        actual_attr = getattr(raised, key, None)
        assert actual_attr == expected_value, f"expected {key}={expected_value!r}, got {actual_attr!r}"
    cause_spec = expected_error.get("cause")
    if cause_spec is not None:
        cause = raised.__cause__
        assert cause is not None, "expected __cause__ to be populated"
        assert type(cause).__name__ == cause_spec["exception_type"], (
            f"expected __cause__ type {cause_spec['exception_type']!r}, got {type(cause).__name__!r}"
        )
    forbidden_categories = invariants.get("error_category_not", [])
    for forbidden in forbidden_categories:
        assert actual_category != forbidden, (
            f"invariant violated: error category {forbidden!r} forbidden but raised"
        )


def _assert_success(
    expected: dict[str, Any],
    invariants: dict[str, Any],
    final_state: Any,
    trace: _MigrationTrace,
) -> None:
    expected_final_state = expected.get("final_state")
    if expected_final_state is not None:
        actual = final_state.model_dump()
        for key, value in expected_final_state.items():
            assert actual.get(key) == value, f"final_state.{key}: expected {value!r}, got {actual.get(key)!r}"

    migrations_run = expected.get("migrations_run")
    if migrations_run is not None:
        # Per the spec's lockstep ordering, each migration step runs
        # once for the outer state. The fixtures only seed parent_states
        # on fixture 043; for the simpler fixtures, the order list
        # tracks a single application of each step. fixture 043's
        # lockstep semantics produce 2 invocations per migration step
        # (once for outer, once for parent) — collapse the consecutive
        # duplicates for the comparison so the order assertion stays
        # "each step ran in chain order."
        dedup_consecutive: list[str] = []
        for label in trace.order:
            if not dedup_consecutive or dedup_consecutive[-1] != label:
                dedup_consecutive.append(label)
        assert dedup_consecutive == migrations_run, (
            f"migrations_run: expected {migrations_run!r}, got {dedup_consecutive!r} "
            f"(raw invocation order: {trace.order!r})"
        )

    expected_count = invariants.get("migration_count")
    if expected_count is not None:
        # ``migration_count`` counts distinct migration steps in the
        # chain (each step applied to outer + parents counts once per
        # step). Collapse consecutive duplicates the same way.
        dedup_consecutive_for_count: list[str] = []
        for label in trace.order:
            if not dedup_consecutive_for_count or dedup_consecutive_for_count[-1] != label:
                dedup_consecutive_for_count.append(label)
        assert len(dedup_consecutive_for_count) == expected_count, (
            f"migration_count: expected {expected_count}, got "
            f"{len(dedup_consecutive_for_count)} ({dedup_consecutive_for_count!r})"
        )

    if invariants.get("single_migration_invocation"):
        # Each migration step ran once per state-or-parent-state entry.
        # For fixture 039 there's no parent_states, so the dedup count
        # equals the raw count and equals 1 (one migration step).
        assert len(trace.order) == 1, (
            f"single_migration_invocation: expected 1 invocation, got {len(trace.order)} ({trace.order!r})"
        )

    if invariants.get("migration_order_matches_chain"):
        # The chain ordering is captured implicitly by the migrations_run
        # comparison above; this is a redundant invariant the fixture
        # surfaces explicitly. No-op here since migrations_run carries
        # the assertion.
        pass


# ---------------------------------------------------------------------------
# Parametrized driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_state_migration_fixture(fixture_path: Path, tmp_path: Path) -> None:
    """Parametrize across 039-046; each case in a fixture's ``cases``
    list runs as a sub-case under one parametrize id (matching how
    test_checkpoint.py handles the cases-shape fixtures).
    """
    with fixture_path.open() as f:
        spec = cast("dict[str, Any]", yaml.safe_load(f))
    cases = spec.get("cases", [])
    for case in cases:
        case_name = case.get("name", "<unnamed>")
        try:
            await _run_one_case(case, tmp_path)
        except AssertionError as exc:
            raise AssertionError(f"case {case_name!r}: {exc}") from exc


# Make sure the imports for the error categories are reachable from
# the test module (pyright would flag them as unused otherwise; we
# reference them via the spec-error-category strings inside the
# fixtures, not by class identity, so the imports are load-bearing
# for the package-public API surface verification).
_ = CheckpointRecordInvalid, CheckpointStateMigrationFailed, CheckpointStateMigrationMissing
