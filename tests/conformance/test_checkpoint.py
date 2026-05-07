"""Run every spec checkpoint conformance fixture (024-031) against the engine.

Phase 5 scope: pipeline-utilities §10 (proposal 0008). Drives the real
:class:`InMemoryCheckpointer` through the engine's save+resume path
end-to-end, asserting against the fixture's ``expected.checkpoint_saves``
+ ``invariants`` + resume expectations.

Fixture-by-fixture status (Phase 5):

- 024 save-on-every-completed-event — supported.
- 025 resume-from-completed-position — supported.
- 026 record-shape — supported.
- 027 attempt-index-resets-on-resume — needs a resume-aware
  ``flaky_resume_aware`` test seam in the adapter; deferred.
- 028 fan-out-atomic-restart — needs a resume-aware
  ``flaky_per_index`` test seam; deferred.
- 029 subgraph-resume — supported (uses plain ``flaky``).
- 030 checkpoint-not-found — supported.
- 031 correlation-id-preserved-across-resume — record-level
  assertions supported here; the OTel span/log assertions are
  gated until Phase 6 lands the observability mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.checkpoint import (
    Checkpointer,
    CheckpointError,
    CheckpointNotFound,
    CheckpointRecord,
    InMemoryCheckpointer,
)
from openarmature.graph import (
    RuntimeGraphError,
    State,
)

from .adapter import build_graph

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "pipeline-utilities" / "conformance"
)

# Phase 5 fixture range: 024-031 are the proposal-0008 conformance set.
_CHECKPOINT_FIXTURE_RANGE = range(24, 32)

# Fixtures that need resume-aware test seams the conformance adapter
# doesn't yet translate. Skipped here with a clear reason — the engine
# plumbing they'd verify (retry-budget reset on resume, fan-out
# atomic restart) is independently covered by unit tests in
# tests/unit/test_checkpoint.py.
_DEFERRED_FIXTURES = frozenset(
    {
        "027-checkpoint-attempt-index-resets-on-resume",
        "028-checkpoint-fan-out-atomic-restart",
    }
)


def _fixture_paths() -> list[Path]:
    out: list[Path] = []
    for p in sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")):
        try:
            number = int(p.stem.split("-", 1)[0])
        except ValueError:
            continue
        if number in _CHECKPOINT_FIXTURE_RANGE:
            out.append(p)
    return out


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Capturing wrapper around the in-memory backend
# ---------------------------------------------------------------------------


class _CapturingCheckpointer:
    """Wraps an :class:`InMemoryCheckpointer` and records every save
    in order so the harness can assert against the fixture's
    ``expected.checkpoint_saves`` block. Implements the
    :class:`Checkpointer` Protocol shape."""

    def __init__(self) -> None:
        self._inner = InMemoryCheckpointer()
        self.saves: list[CheckpointRecord] = []

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        self.saves.append(record)
        await self._inner.save(invocation_id, record)

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        return await self._inner.load(invocation_id)

    async def list(self, filter: Any = None) -> Any:
        return await self._inner.list(filter)

    async def delete(self, invocation_id: str) -> None:
        await self._inner.delete(invocation_id)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_checkpoint_fixture(fixture_path: Path) -> None:
    fixture_id = fixture_path.stem
    if fixture_id in _DEFERRED_FIXTURES:
        pytest.skip(
            f"{fixture_id}: needs resume-aware test seam in adapter; "
            "engine path independently covered in tests/unit/test_checkpoint.py"
        )
    spec = _load(fixture_path)
    if "cases" in spec:
        for case in cast("list[dict[str, Any]]", spec["cases"]):
            case_name = case.get("name", "<unnamed>")
            try:
                await _run_one_case(case)
            except AssertionError as e:
                raise AssertionError(f"case {case_name!r}: {e}") from e
        return
    await _run_one_case(spec)


async def _run_one_case(spec: Mapping[str, Any]) -> None:
    """Run one fixture or one case from a cases-shape fixture."""
    capturing = _CapturingCheckpointer()
    subgraphs = _build_subgraphs(spec)

    trace: list[str] = []
    built = build_graph(spec, subgraphs=subgraphs, trace=trace)
    builder = built.builder
    builder.with_checkpointer(cast("Checkpointer", capturing))
    compiled = builder.compile()
    initial_state = built.initial_state(spec.get("initial_state", {}))

    # Run #1 — first invocation. May succeed or fail per fixture.
    first_run_expected_error = spec.get("first_run_expected_error")
    invocation_id_first_run: str | None = None
    final_first_run: State | None = None
    try:
        final_first_run = await compiled.invoke(
            initial_state,
            correlation_id=spec.get("correlation_id"),
        )
        if first_run_expected_error is not None:
            raise AssertionError(
                f"expected first run to fail with category "
                f"{first_run_expected_error!r} but it returned successfully"
            )
    except RuntimeGraphError as e:
        if first_run_expected_error is None:
            raise
        expected_category = first_run_expected_error["category"]
        assert e.category == expected_category, (
            f"first-run error category mismatch: actual={e.category!r}, expected={expected_category!r}"
        )

    # Capture invocation_id from the latest save (we attached a
    # capturing checkpointer; every save record carries the engine's
    # invocation_id verbatim).
    if capturing.saves:
        invocation_id_first_run = capturing.saves[-1].invocation_id

    # ----- Saved record assertions (fixture 029) -----
    if "saved_record_assertions" in spec:
        _assert_saved_record(cast("Mapping[str, Any]", spec["saved_record_assertions"]), capturing)

    # ----- Single-run expected assertions -----
    expected = cast("Mapping[str, Any]", spec.get("expected") or {})
    if "checkpoint_saves" in expected:
        _assert_checkpoint_saves(
            cast("list[Mapping[str, Any]]", expected["checkpoint_saves"]),
            capturing.saves,
        )
    if "final_state" in expected and final_first_run is not None:
        _assert_state_matches(final_first_run, cast("Mapping[str, Any]", expected["final_state"]))
    if "invariants" in expected:
        _assert_invariants(cast("Mapping[str, Any]", expected["invariants"]), capturing.saves)

    # ----- checkpoint_not_found expected (fixture 030) -----
    if expected.get("expected_error") == "checkpoint_not_found":
        ghost = cast("str", expected.get("resume_invocation_id", "ghost"))
        with pytest.raises(CheckpointNotFound):
            await compiled.invoke(initial_state, resume_invocation=ghost)
        return

    # ----- Resume path (fixtures 025, 029, 031) -----
    resume_block = spec.get("resume")
    if resume_block is None or not resume_block.get("from_first_run"):
        return
    if invocation_id_first_run is None:
        raise AssertionError("resume requested but no invocation_id captured (no saves fired)")
    saves_before_resume = list(capturing.saves)
    capturing.saves.clear()
    try:
        final_resume = await compiled.invoke(
            initial_state,
            resume_invocation=invocation_id_first_run,
        )
    except CheckpointError:
        raise
    resume_expected = cast("Mapping[str, Any]", resume_block.get("expected") or {})
    if "final_state" in resume_expected:
        _assert_state_matches(final_resume, cast("Mapping[str, Any]", resume_expected["final_state"]))

    # Fixture 031: assert correlation_id preserved + invocation_id
    # changed. Span/log assertions deferred to Phase 6 — observability
    # isn't wired yet. Skip those cleanly here.
    if "correlation_id_assertions" in resume_expected:
        cid_block = cast("Mapping[str, Any]", resume_expected["correlation_id_assertions"])
        if cid_block.get("requires_observability"):
            pytest.skip("correlation_id span/log assertions require observability — Phase 6")
    if saves_before_resume and capturing.saves:
        original_id = saves_before_resume[-1].invocation_id
        resumed_id = capturing.saves[-1].invocation_id
        if resume_block.get("invariants", {}).get("invocation_id_differs"):
            assert original_id != resumed_id, (
                f"invocation_id should differ on resume; got original={original_id!r}, resumed={resumed_id!r}"
            )
        if resume_block.get("invariants", {}).get("correlation_id_preserved"):
            original_corr = saves_before_resume[-1].correlation_id
            resumed_corr = capturing.saves[-1].correlation_id
            assert original_corr == resumed_corr, (
                f"correlation_id should be preserved across resume; "
                f"got original={original_corr!r}, resumed={resumed_corr!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_subgraphs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build any subgraphs (`subgraph:` or `subgraphs:`) the fixture
    declares. Returns a registry the adapter consumes by name."""
    subgraph_specs: dict[str, Any] = {}
    if "subgraph" in spec:
        single = cast("Mapping[str, Any]", spec["subgraph"])
        name = single.get("name") or "subgraph"
        subgraph_specs[name] = single
    if "subgraphs" in spec:
        for k, v in cast("dict[str, Any]", spec["subgraphs"]).items():
            subgraph_specs[k] = v
    compiled_subgraphs: dict[str, Any] = {}
    for name, sub_spec in subgraph_specs.items():
        sub_trace: list[str] = []
        sub_built = build_graph(sub_spec, trace=sub_trace)
        compiled_subgraphs[name] = sub_built.builder.compile()
    return compiled_subgraphs


def _assert_saved_record(
    block: Mapping[str, Any],
    capturing: _CapturingCheckpointer,
) -> None:
    if not capturing.saves:
        raise AssertionError("saved_record_assertions: no saves were recorded")
    record = capturing.saves[-1]
    if "completed_positions" in block:
        expected_positions = cast("list[Mapping[str, Any]]", block["completed_positions"])
        actual = [
            {
                "namespace": list(p.namespace),
                "node_name": p.node_name,
                "step": p.step,
                "attempt_index": p.attempt_index,
            }
            for p in record.completed_positions
        ]
        assert actual == [dict(p) for p in expected_positions], (
            f"completed_positions mismatch: actual={actual}, expected={[dict(p) for p in expected_positions]}"
        )
    if block.get("parent_states_present"):
        assert record.parent_states, "expected parent_states to be populated; got empty tuple"
    if block.get("parent_states_outermost_first"):
        assert record.parent_states[0] is not None


def _assert_checkpoint_saves(
    expected: list[Mapping[str, Any]],
    actual: list[CheckpointRecord],
) -> None:
    assert len(actual) == len(expected), (
        f"save count mismatch: actual={len(actual)}, expected={len(expected)}"
    )
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        if "state" in e:
            _assert_state_matches(a.state, e["state"])
        if "completed_positions" in e:
            expected_positions = cast("list[Mapping[str, Any]]", e["completed_positions"])
            actual_positions = [
                {
                    "namespace": list(p.namespace),
                    "node_name": p.node_name,
                    "step": p.step,
                    "attempt_index": p.attempt_index,
                }
                for p in a.completed_positions
            ]
            assert actual_positions == [dict(p) for p in expected_positions], (
                f"save #{i} completed_positions mismatch: "
                f"actual={actual_positions}, "
                f"expected={[dict(p) for p in expected_positions]}"
            )


def _assert_state_matches(actual: Any, expected: Mapping[str, Any]) -> None:
    if isinstance(actual, State):
        actual_dict = actual.model_dump()
    elif isinstance(actual, dict):
        actual_dict = cast("dict[str, Any]", actual)
    else:
        raise AssertionError(f"unexpected actual state type {type(actual).__name__}")
    for k, v in expected.items():
        assert actual_dict.get(k) == v, f"state field {k!r}: actual={actual_dict.get(k)!r}, expected={v!r}"


def _assert_invariants(
    invariants: Mapping[str, Any],
    saves: list[CheckpointRecord],
) -> None:
    if "save_count" in invariants:
        assert len(saves) == invariants["save_count"], (
            f"save_count: actual={len(saves)}, expected={invariants['save_count']}"
        )
    if invariants.get("save_order_matches_completed_event_order"):
        last_steps = [s.completed_positions[-1].step for s in saves]
        assert last_steps == sorted(last_steps), f"save order does not match step order: {last_steps}"
    if invariants.get("saves_for_fan_out_internals") == 0:
        for s in saves:
            for p in s.completed_positions:
                assert p.fan_out_index is None, f"unexpected fan-out internal save: {p}"
