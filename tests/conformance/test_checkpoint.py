"""Run every checkpoint conformance fixture against the engine.

Drives the real :class:`InMemoryCheckpointer` (with optional fan-out
internal save batching) through the engine's save+resume path
end-to-end,
asserting against the fixture's ``saved_record_assertions`` (including
``fan_out_progress`` matchers), ``expected.checkpoint_saves``,
``invariants``, and resume expectations (including per-instance
``instances_executed_during_resume`` / ``instances_skipped_during_resume``
and per-instance attempt-count assertions from the per-instance resume
fixtures).

Fixture-by-fixture status:

- 024 save-on-every-completed-event — supported.
- 025 resume-from-completed-position — supported.
- 026 record-shape — supported.
- 027 attempt-index-resets-on-resume — needs a resume-aware
  ``flaky_resume_aware`` test seam in the adapter; deferred.
- 029 subgraph-resume — supported (uses plain ``flaky``).
- 030 checkpoint-not-found — supported.
- 031 correlation-id-preserved-across-resume — record-level
  predicates checked here; its span-level claims run in the OTel
  runner.
- 048-054 per-instance fan-out resume contract — supported.
- 055 schema-version-declared-class — supported.
- 056 fan-out-count-drift — supported.
- 057 resume-mints-fresh-invocation-id — supported.
- 067 crash-injection-fan-out-resume — supported.
- 069 fan-out-degrade-refinements — its crash and resume case runs
  here; the plain degrade cases run in ``test_pipeline_utilities.py``.
- 070 crash-injection-after-node-resume — supported.
- 076 nested-fan-out-resume-lineage — supported, through the
  seeded-record path.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest

from openarmature.checkpoint import (
    Checkpointer,
    CheckpointError,
    CheckpointNotFound,
    CheckpointRecord,
    CheckpointRecordInvalid,
    EnclosingFanOutInstance,
    FanOutInstanceProgress,
    FanOutInternalSaveBatching,
    FanOutProgress,
    InMemoryCheckpointer,
    NodePosition,
)
from openarmature.graph import (
    FailureIsolationMiddleware,
    RuntimeGraphError,
    State,
)
from openarmature.llm import TRANSIENT_CATEGORIES

from .adapter import ObserverFixture, build_graph, make_observer_fn, raised_from_node
from .harness import read_trace
from .harness.subgraph_placement import resolve_subgraphs

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "pipeline-utilities" / "conformance"
)

# Proposal sources: 0008 for 024-031, 0009 for 048-054, 0028 for 055,
# 0029 for 056, 0039 for 057, 0070 for 067 and 070, 0069 for 069, and
# 0085 for 076. 028 is retired; 048-054 cover its contract.
_CHECKPOINT_FIXTURE_NUMBERS: frozenset[int] = frozenset(
    (set(range(24, 32)) - {28}) | set(range(48, 58)) | {67, 69, 70, 76}
)

# Fixtures that need resume-aware test seams the conformance adapter
# doesn't yet translate. Skipped here with a clear reason — the engine
# plumbing they'd verify is independently covered by unit tests.
_DEFERRED_FIXTURES = frozenset(
    {
        "027-checkpoint-attempt-index-resets-on-resume",
    }
)


def _fixture_paths() -> list[Path]:
    out: list[Path] = []
    for p in sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml")):
        try:
            number = int(p.stem.split("-", 1)[0])
        except ValueError:
            continue
        if number in _CHECKPOINT_FIXTURE_NUMBERS:
            out.append(p)
    return out


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    return read_trace.load(path)


# ---------------------------------------------------------------------------
# Capturing wrapper around the in-memory backend
# ---------------------------------------------------------------------------


class _AbortAfterInstance(Exception):  # noqa: N818
    """Sentinel exception raised by the capturing wrapper to simulate a
    crash after the configured instance's "instance completed" save
    has fired.

    Under collect mode, this exception fires from inside a
    per-instance save and gets captured by
    ``asyncio.gather(..., return_exceptions=True)``. The test driver
    sees the captured-but-not-surfaced abort by inspecting the
    wrapper's ``_aborted`` flag after the invoke returns.
    """


class _CapturingCheckpointer:
    """Wraps an :class:`InMemoryCheckpointer` and records every save
    in order so the harness can assert against the fixture's
    ``expected.checkpoint_saves`` block. Implements the
    :class:`Checkpointer` Protocol shape AND the optional
    ``save_fan_out_internal`` hook (batching) so the
    engine routes inner-instance saves here.

    ``abort_after_instance``: when set, the wrapper raises
    :class:`_AbortAfterInstance` AFTER the save that just transitioned
    the named instance index from ``not_started`` / ``in_flight`` to
    ``completed``. Simulates a crash at that exact point — used by
    fixture 052 to test collect-mode error-record rollforward, and by
    the ``crash_injection: {after_fan_out_instance}`` directive.
    ``abort_after_node``: the same simulated crash AFTER the save
    that records the named node in ``completed_positions`` — the
    ``crash_injection: {after_node}`` boundary.
    """

    def __init__(
        self,
        *,
        fan_out_internal_save_batching: FanOutInternalSaveBatching | None = None,
        abort_after_instance: int | None = None,
        abort_after_instance_node: str | None = None,
        abort_after_node: str | None = None,
    ) -> None:
        self._inner = InMemoryCheckpointer(
            fan_out_internal_save_batching=fan_out_internal_save_batching,
        )
        self.saves: list[CheckpointRecord] = []
        # Parallel to ``saves``: the method each save arrived through.
        self.save_routes: list[str] = []
        self._abort_after_instance = abort_after_instance
        # The fan-out node ``abort_after_instance`` targets. ``None`` (the
        # legacy fan_out.abort_after_instance path) matches any fan-out;
        # crash_injection.after_fan_out_instance sets it to scope the abort to
        # the named node in a multi-fan-out graph.
        self._abort_after_instance_node = abort_after_instance_node
        self._abort_after_node = abort_after_node
        self._aborted = False
        # Per proposal 0029 (fixture 056): mutating the saved record's
        # outer state on ``load`` simulates "user shrank/grew the input
        # set between runs." The engine restores from this mutated
        # state, the fan-out node re-resolves count from the mutated
        # ``items``, and the count-drift check raises
        # ``checkpoint_record_invalid`` because the saved
        # ``fan_out_progress`` entry's ``instance_count`` doesn't match.
        # Keys are field names on the outer state; values replace
        # those fields when the record is returned to the engine on
        # resume.
        self.load_state_overrides: dict[str, Any] = {}

    async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
        self._raise_if_post_abort()
        self.saves.append(record)
        self.save_routes.append("save")
        await self._inner.save(invocation_id, record)
        self._maybe_abort(record)

    async def save_fan_out_internal(self, invocation_id: str, record: CheckpointRecord) -> None:
        self._raise_if_post_abort()
        self.saves.append(record)
        self.save_routes.append("save_fan_out_internal")
        await self._inner.save_fan_out_internal(invocation_id, record)
        self._maybe_abort(record)

    async def save_fan_out_in_flight_failure(self, invocation_id: str, record: CheckpointRecord) -> None:
        self._raise_if_post_abort()
        self.saves.append(record)
        self.save_routes.append("save_fan_out_in_flight_failure")
        await self._inner.save_fan_out_in_flight_failure(invocation_id, record)
        self._maybe_abort(record)

    def _raise_if_post_abort(self) -> None:
        """Raise on every save once the abort has fired, modelling a process
        that crashed at the boundary."""
        # Without this, gather would keep dispatching sibling instances whose
        # saves complete normally and pollute the loaded record.
        if self._aborted:
            raise _AbortAfterInstance("post-abort save call")

    # Every save route calls this after its inner save, so the record is
    # durably persisted before the simulated crash.
    def _maybe_abort(self, record: CheckpointRecord) -> None:
        """Raise the sentinel if ``record``'s save is the configured crash
        boundary. ``abort_after_instance`` fires on the save transitioning
        that instance index to ``completed``; ``abort_after_node`` fires on
        the save recording that node in ``completed_positions``."""
        if self._aborted:
            return
        if self._abort_after_instance is not None:
            target_idx = self._abort_after_instance
            for fp in record.fan_out_progress:
                # Scope to the targeted fan-out node when one is named
                # (crash_injection.after_fan_out_instance); the legacy path
                # leaves it None and matches any fan-out.
                if (
                    self._abort_after_instance_node is not None
                    and fp.fan_out_node_name != self._abort_after_instance_node
                ):
                    continue
                if target_idx < len(fp.instances) and fp.instances[target_idx].state == "completed":
                    # Subsequent instances must NOT be completed — otherwise
                    # we'd abort after a later instance's save instead.
                    if all(inst.state != "completed" for inst in fp.instances[target_idx + 1 :]):
                        self._aborted = True
                        raise _AbortAfterInstance(
                            f"simulated crash after instance {target_idx} completed save"
                        )
        if self._abort_after_node is not None and any(
            p.node_name == self._abort_after_node for p in record.completed_positions
        ):
            self._aborted = True
            raise _AbortAfterInstance(f"simulated crash after node {self._abort_after_node} save")

    async def load(self, invocation_id: str) -> CheckpointRecord | None:
        record = await self._inner.load(invocation_id)
        if record is None or not self.load_state_overrides:
            return record
        # Apply overrides to the outer state. For outer-level saves the
        # outer state is ``record.state``; for inner saves (fan-out
        # instance, subgraph) it's ``record.parent_states[0]``. Mutate
        # whichever shape is present so the test driver doesn't need
        # to care which save site landed last.
        from dataclasses import replace as dataclass_replace  # noqa: PLC0415

        if record.parent_states:
            outer = record.parent_states[0]
            outer_updates = {**outer.model_dump(), **self.load_state_overrides}
            new_outer = type(outer)(**outer_updates)
            new_parents = (new_outer,) + record.parent_states[1:]
            return dataclass_replace(record, parent_states=new_parents)
        outer = record.state
        outer_updates = {**outer.model_dump(), **self.load_state_overrides}
        new_outer = type(outer)(**outer_updates)
        return dataclass_replace(record, state=new_outer)

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
        cases_run = 0
        for case in cast("list[dict[str, Any]]", spec["cases"]):
            case_name = case.get("name", "<unnamed>")
            # This runner drives the checkpoint cases. A mixed fixture (069)
            # interleaves plain FI-degrade cases owned by
            # test_pipeline_utilities.py; skip a case with no checkpoint
            # concern. The marker is checkpointer / resume / crash_injection —
            # NOT resume alone: fixtures like 024 / 026 / 030 / 055 assert
            # checkpoint behavior (saves, record shape, not-found,
            # schema_version) with a checkpointer but no resume.
            if not any(k in case for k in ("checkpointer", "resume", "crash_injection")):
                continue
            cases_run += 1
            try:
                await _run_one_case(case, top_level=spec)
            except AssertionError as e:
                raise AssertionError(f"case {case_name!r}: {e}") from e
        # A cases-shaped fixture in this runner's set that drives zero cases
        # (all skipped as non-checkpoint) would pass vacuously; fail loudly
        # instead so a routing mistake surfaces.
        assert cases_run > 0, (
            f"{fixture_id}: cases-shaped fixture drove zero cases in this runner "
            f"(all skipped as non-checkpoint). Fix the routing or remove it from "
            f"_CHECKPOINT_FIXTURE_NUMBERS."
        )
        return
    await _run_one_case(spec, top_level=spec)


def _build_capturing(spec: Mapping[str, Any]) -> _CapturingCheckpointer:
    """Build the capturing checkpointer for a case, honoring the
    optional batching / abort directives from the fixture.

    The fixture's ``checkpointer`` field accepts two shapes:
    - ``"in_memory"``: default no-batching backend.
    - ``{kind: in_memory_batched, fan_out_internal_save_batching: {flush_every: N}}``:
      the batched backend with N-save flush interval.

    The fixture's fan-out node may also carry ``abort_after_instance: N``
    — a harness-level directive that simulates a crash after the named
    instance's "instance completed" save fires. Surface that here so
    the capturing wrapper can raise the sentinel.
    """
    checkpointer_cfg = spec.get("checkpointer")
    batching: FanOutInternalSaveBatching | None = None
    if isinstance(checkpointer_cfg, dict):
        cfg_dict = cast("dict[str, Any]", checkpointer_cfg)
        kind = cfg_dict.get("kind")
        if kind == "in_memory_batched":
            batching_cfg = cast(
                "Mapping[str, Any]",
                cfg_dict.get("fan_out_internal_save_batching") or {},
            )
            flush_every = int(batching_cfg.get("flush_every", 0))
            batching = FanOutInternalSaveBatching(flush_every=flush_every)
    ci_instance, ci_instance_node, ci_node = _find_crash_injection(spec)
    if ci_instance is not None or ci_node is not None:
        # crash_injection defines the crash boundary exclusively; the legacy
        # fan_out.abort_after_instance directive is ignored when it is set, so
        # an instance-boundary and a node-boundary abort can't both be active.
        abort_after = ci_instance
    else:
        abort_after = _find_abort_after_instance(spec)
    return _CapturingCheckpointer(
        fan_out_internal_save_batching=batching,
        abort_after_instance=abort_after,
        abort_after_instance_node=ci_instance_node,
        abort_after_node=ci_node,
    )


def _find_abort_after_instance(spec: Mapping[str, Any]) -> int | None:
    """Locate the ``abort_after_instance`` directive (if any) on a
    fan-out node config inside the case spec. Returns the int idx or
    None if no fan-out node declares the directive. Used by fixture 052.
    """
    for node_spec in cast("dict[str, dict[str, Any]]", spec.get("nodes", {})).values():
        if "fan_out" in node_spec:
            fan_out = cast("Mapping[str, Any]", node_spec["fan_out"])
            if "abort_after_instance" in fan_out:
                return int(fan_out["abort_after_instance"])
    return None


# Conformance-adapter §5.6 ``crash_injection`` (proposal 0070): a simulated
# crash at a checkpoint boundary, independent of an instance failure.
def _find_crash_injection(spec: Mapping[str, Any]) -> tuple[int | None, str | None, str | None]:
    """Parse the top-level ``crash_injection`` directive. Returns
    ``(after_fan_out_instance_index, after_fan_out_instance_node,
    after_node_name)``: the index + node identify the
    ``after_fan_out_instance`` boundary, ``after_node_name`` the ``after_node``
    boundary; at most one boundary is set. Pairs with ``resume:`` the way
    ``first_run_expected_error`` does, but the first run has no asserted
    outcome (it "crashed")."""
    ci = spec.get("crash_injection")
    if not isinstance(ci, dict):
        return None, None, None
    ci_dict = cast("Mapping[str, Any]", ci)
    after_instance = ci_dict.get("after_fan_out_instance")
    if isinstance(after_instance, dict):
        ai = cast("Mapping[str, Any]", after_instance)
        node = ai.get("node")
        return int(ai["index"]), (str(node) if node is not None else None), None
    after_node = ci_dict.get("after_node")
    if after_node is not None:
        return None, None, str(after_node)
    return None, None, None


def _translate_fi_instance_middleware(
    spec: Mapping[str, Any],
) -> dict[str, list[FailureIsolationMiddleware]]:
    """Translate each fan-out node's ``instance_middleware: [failure_isolation]``
    entries into FailureIsolationMiddleware instances keyed by node name, for
    build_graph's ``fan_out_instance_middleware``.

    Supports only the static ``degraded_update`` mapping form, and raises
    ``ValueError`` for an entry that omits it or gives another form.
    """
    # The static form is the only one the checkpoint fixtures use (fixture 069
    # case 3, degrade survives resume). test_pipeline_utilities.py owns the
    # callable forms and drives the plain FI-degrade cases.
    out: dict[str, list[FailureIsolationMiddleware]] = {}
    nodes = cast("dict[str, dict[str, Any]]", spec.get("nodes") or {})
    for node_name, node_spec in nodes.items():
        fan_out = node_spec.get("fan_out")
        if not isinstance(fan_out, dict):
            continue
        entries = cast(
            "list[dict[str, Any]]",
            cast("Mapping[str, Any]", fan_out).get("instance_middleware") or [],
        )
        mws: list[FailureIsolationMiddleware] = []
        for entry in entries:
            # Only failure_isolation is translated here. Other instance
            # middleware, such as fixture 053's retry, stays unwired: those
            # fixtures drive their behavior through flaky_per_index seams.
            if entry.get("type") != "failure_isolation":
                continue
            if "degraded_update" not in entry:
                raise ValueError(
                    f"fan-out node {node_name!r}: failure_isolation instance middleware "
                    f"entry is missing the required 'degraded_update'"
                )
            degraded = entry["degraded_update"]
            if not isinstance(degraded, dict):
                raise ValueError(
                    f"fan-out node {node_name!r}: checkpoint runner supports only the static "
                    f"degraded_update form for instance middleware"
                )
            mws.append(
                FailureIsolationMiddleware(
                    degraded_update=dict(cast("Mapping[str, Any]", degraded)),
                    event_name=entry.get("event_name", "degraded"),
                )
            )
        if mws:
            out[node_name] = mws
    return out


def _abort_after_instance_node(spec: Mapping[str, Any]) -> str | None:
    """The fan-out node that carries ``abort_after_instance``, where the
    harness simulates the first run's failure."""
    for node_name, node_spec in cast("dict[str, dict[str, Any]]", spec.get("nodes", {})).items():
        if "abort_after_instance" in (node_spec.get("fan_out") or {}):
            return node_name
    return None


def _assert_first_run_raised_from(expected: Mapping[str, Any], node: str | None) -> None:
    if "raised_from" in expected:
        assert node == expected["raised_from"], (
            f"first-run raised_from: actual={node!r}, expected={expected['raised_from']!r}"
        )


def _strip_abort_directive(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the spec with ``abort_after_instance`` removed from every
    fan-out node, or the spec itself when no node carries it."""
    # The engine does not recognize the directive. The capturing checkpointer
    # interprets it on the harness side, so build_graph must not see it.
    nodes_raw = spec.get("nodes")
    if not isinstance(nodes_raw, dict):
        return spec
    nodes = cast("dict[str, dict[str, Any]]", nodes_raw)
    new_nodes: dict[str, dict[str, Any]] = {}
    changed = False
    for node_name, node_spec in nodes.items():
        if "fan_out" in node_spec:
            fan_out = cast("dict[str, Any]", node_spec["fan_out"])
            if "abort_after_instance" in fan_out:
                new_fan_out = {k: v for k, v in fan_out.items() if k != "abort_after_instance"}
                new_nodes[node_name] = {**node_spec, "fan_out": new_fan_out}
                changed = True
                continue
        new_nodes[node_name] = node_spec
    if not changed:
        return spec
    return {**spec, "nodes": new_nodes}


def _build_seeded_fan_out_progress(fp: Mapping[str, Any]) -> FanOutProgress:
    """Build one FanOutProgress entry, with its
    ``enclosing_fan_out_lineage``, from a fixture ``seeded_record`` block."""
    instances = tuple(
        FanOutInstanceProgress(
            state=inst["state"],
            result=inst.get("result"),
            result_is_error=inst.get("result_is_error", False),
        )
        for inst in cast("list[Mapping[str, Any]]", fp.get("instances", []))
    )
    # Proposal 0085: lineage-qualified fan_out_progress.
    lineage = tuple(
        EnclosingFanOutInstance(
            namespace=tuple(e.get("namespace", [])),
            fan_out_node_name=e["fan_out_node_name"],
            fan_out_index=e["fan_out_index"],
        )
        for e in cast("list[Mapping[str, Any]]", fp.get("enclosing_fan_out_lineage", []))
    )
    return FanOutProgress(
        fan_out_node_name=fp["fan_out_node_name"],
        namespace=tuple(fp.get("namespace", [])),
        instance_count=fp["instance_count"],
        instances=instances,
        enclosing_fan_out_lineage=lineage,
    )


# test_state_migration.py seeds the same ``seeded_record`` shape. A
# fan-out's runtime namespace_prefix equals its node-name path, so a seeded
# namespace matches the re-entering execution's key with no mapping step.
def _build_seeded_record(seeded: Mapping[str, Any], invocation_id: str) -> CheckpointRecord:
    """Construct a CheckpointRecord from a fixture ``seeded_record`` block,
    using its ``namespace`` node-name paths verbatim."""
    positions = tuple(
        NodePosition(
            namespace=tuple(p.get("namespace", [])),
            node_name=p["node_name"],
            step=p["step"],
            attempt_index=p.get("attempt_index", 0),
            fan_out_index=p.get("fan_out_index"),
        )
        for p in cast("list[Mapping[str, Any]]", seeded.get("completed_positions", []))
    )
    fan_out_progress = tuple(
        _build_seeded_fan_out_progress(fp)
        for fp in cast("list[Mapping[str, Any]]", seeded.get("fan_out_progress", []))
    )
    return CheckpointRecord(
        invocation_id=invocation_id,
        correlation_id=seeded.get("correlation_id", "seeded-corr"),
        state=dict(seeded.get("state", {})),
        completed_positions=positions,
        parent_states=tuple(seeded.get("parent_states", [])),
        last_saved_at=0.0,
        schema_version=seeded.get("schema_version", ""),
        fan_out_progress=fan_out_progress,
    )


# Fixture 076's fixture-specific predicates (proposal 0085), which
# conformance-adapter §5.9 says an adapter MUST implement.
_KNOWN_NESTED_RESUME_INVARIANTS = frozenset(
    {
        "inner_fan_out_progress_matched_per_enclosing_lineage",
        "each_outer_instance_skips_only_its_own_completed_inner_instances",
        "no_inner_instance_skipped_across_enclosing_instances",
        "legacy_unlineaged_entry_not_applied_to_nested_fan_out",
        "all_inner_instances_re_run_in_every_outer_instance",
    }
)


def _assert_nested_resume_invariants(
    seeded: Mapping[str, Any],
    recorded_values: Sequence[Any],
    invariants: Mapping[str, Any],
) -> None:
    """Assert fixture 076's nested-fan-out resume invariants against the set
    of inner-leaf source values that actually RE-RAN on resume.

    Expectations are derived from the seeded record: an inner entry with a
    matching lineage skips its ``completed`` positions and re-runs the rest;
    a legacy (no-lineage) entry matches nothing, so every inner instance
    re-runs. Fails loud on an unmapped invariant name."""
    unknown = set(invariants) - _KNOWN_NESTED_RESUME_INVARIANTS
    assert not unknown, f"unhandled nested-resume invariant(s): {sorted(unknown)}"

    # The leaf's source values are globally unique, so the recorded set tells
    # a correct per-lineage skip from a full re-run. Final state cannot: both
    # yield the same accumulator.
    recorded = {v for v in recorded_values}
    outer_items = cast("list[list[int]]", seeded["state"]["outer_items"])
    all_inner = {v for sub in outer_items for v in sub}
    nested_entries = [
        fp for fp in cast("list[Mapping[str, Any]]", seeded["fan_out_progress"]) if fp.get("namespace")
    ]
    lineaged = [fp for fp in nested_entries if fp.get("enclosing_fan_out_lineage")]

    # Per-lineage expected re-run / skip: for each lineage-qualified inner
    # entry, its enclosing instance index selects the inner items list; a
    # ``completed`` position is skipped, everything else re-runs.
    expected_rerun: set[Any] = set()
    expected_skipped: set[Any] = set()
    for fp in lineaged:
        outer_idx = int(fp["enclosing_fan_out_lineage"][-1]["fan_out_index"])
        items: list[int] = outer_items[outer_idx]
        for i, inst in enumerate(cast("list[Mapping[str, Any]]", fp["instances"])):
            target = expected_skipped if inst["state"] == "completed" else expected_rerun
            target.add(items[i])

    if invariants.get("inner_fan_out_progress_matched_per_enclosing_lineage"):
        keys = {(tuple(fp["namespace"]), fp["fan_out_node_name"]) for fp in lineaged}
        assert len(lineaged) >= 2 and len(keys) < len(lineaged), (
            "expected >= 2 inner entries sharing (namespace, fan_out_node_name) with distinct lineage"
        )
        assert recorded == expected_rerun, (
            f"inner entries not matched per lineage: re-ran {sorted(recorded)}, "
            f"expected {sorted(expected_rerun)} (a collapsed/mismatched entry would differ)"
        )
    if invariants.get("each_outer_instance_skips_only_its_own_completed_inner_instances"):
        assert recorded.isdisjoint(expected_skipped), (
            f"an inner instance completed under its own lineage was re-run: "
            f"re-ran {sorted(recorded)}, should have skipped {sorted(expected_skipped)}"
        )
    if invariants.get("no_inner_instance_skipped_across_enclosing_instances"):
        assert expected_rerun <= recorded, (
            f"an inner instance was mis-skipped across enclosing instances: "
            f"expected re-run {sorted(expected_rerun)}, actually re-ran {sorted(recorded)}"
        )
    if invariants.get("legacy_unlineaged_entry_not_applied_to_nested_fan_out"):
        legacy_completed = {
            inst["result"]
            for fp in nested_entries
            if not fp.get("enclosing_fan_out_lineage")
            for inst in cast("list[Mapping[str, Any]]", fp["instances"])
            if inst["state"] == "completed"
        }
        assert legacy_completed <= recorded, (
            f"a legacy (no-lineage) entry's completed instances were applied (skipped) to a nested "
            f"fan-out: its completed results {sorted(legacy_completed)} were not all re-run "
            f"(re-ran {sorted(recorded)})"
        )
    if invariants.get("all_inner_instances_re_run_in_every_outer_instance"):
        assert recorded == all_inner, (
            f"expected a full re-run of every inner instance {sorted(all_inner)}, "
            f"but re-ran {sorted(recorded)}"
        )


async def _run_seeded_resume_case(spec: Mapping[str, Any], *, top_level: Mapping[str, Any]) -> None:
    """Drive a seeded-record resume case (fixture 076).

    Seeds a precise mid-flight CheckpointRecord (carrying lineage-qualified
    fan_out_progress entries) rather than crash-producing one, then resumes
    and asserts the nested-fan-out consume-side contract: lineage-matched
    skip, no mis-skip across enclosing instances, exactly-once accumulator."""
    expected_block = cast("Mapping[str, Any]", spec.get("expected") or {})
    assert not spec.get("invariants") and not expected_block.get("invariants"), (
        "a seeded-record case checks only resume invariants"
    )
    leaf_values: list[Any] = []
    subgraphs = _build_subgraphs_for(spec, top_level, leaf_value_recorder=leaf_values)
    built = build_graph(spec, subgraphs=subgraphs, trace=[], leaf_value_recorder=leaf_values)
    checkpointer = InMemoryCheckpointer()
    built.builder.with_checkpointer(cast(Checkpointer, checkpointer))
    compiled = built.builder.compile()

    seeded_block = cast("Mapping[str, Any]", spec["seeded_record"])
    invocation_id = f"seeded-{spec.get('name', 'case')}"
    await checkpointer.save(invocation_id, _build_seeded_record(seeded_block, invocation_id))

    resume_block = cast("Mapping[str, Any]", spec.get("resume") or {})
    if not resume_block.get("from_seeded_record"):
        raise AssertionError("seeded_record present but resume.from_seeded_record not set")

    initial_state = built.initial_state(spec.get("initial_state", {}))
    # No first run: leaf_values accrues only the resume run's re-run values.
    leaf_values.clear()
    final = await compiled.invoke(initial_state, resume_invocation=invocation_id)

    resume_expected = cast("Mapping[str, Any]", resume_block.get("expected") or {})
    if "final_state" in resume_expected:
        _assert_state_matches(final, cast("Mapping[str, Any]", resume_expected["final_state"]))

    invariants_block: dict[str, Any] = {}
    invariants_block.update(cast("dict[str, Any]", resume_block.get("invariants") or {}))
    invariants_block.update(cast("dict[str, Any]", resume_expected.get("invariants") or {}))
    if invariants_block:
        _assert_nested_resume_invariants(seeded_block, leaf_values, invariants_block)


async def _run_one_case(spec: Mapping[str, Any], *, top_level: Mapping[str, Any]) -> None:
    """Run one fixture or one case from a cases-shape fixture."""
    # A seeded-record case (fixture 076, proposal 0085 nested-fan-out resume)
    # does not crash-produce its record: it seeds a precise mid-flight record
    # and drives the RESUME consume-side. Route it to the dedicated runner.
    if "seeded_record" in spec:
        await _run_seeded_resume_case(spec, top_level=top_level)
        return
    case_invariants = cast("Mapping[str, Any]", spec.get("invariants") or {})
    resume_declared = cast("Mapping[str, Any]", spec.get("resume") or {})
    resume_expected_declared = cast("Mapping[str, Any]", resume_declared.get("expected") or {})
    # Fixtures put resume invariants under `resume.invariants` or under
    # `resume.expected.invariants`.
    resume_invariants: dict[str, Any] = {
        **cast("Mapping[str, Any]", resume_declared.get("invariants") or {}),
        **cast("Mapping[str, Any]", resume_expected_declared.get("invariants") or {}),
    }
    if case_invariants:
        _assert_run_invariant_names(case_invariants, "case-level")
    if resume_invariants:
        _assert_run_invariant_names(resume_invariants, "resume")
    if case_invariants or resume_invariants:
        # Both blocks are checked once the resume completes, so a case that
        # never completes one would skip them all.
        assert resume_declared.get("from_first_run") and "expected_error" not in resume_declared, (
            "case-level and resume invariants need a resume that completes"
        )
    capturing = _build_capturing(spec)
    # Shared recorders so flaky_per_index nodes inside subgraphs feed
    # the same per-instance attempt table the resume assertions consult.
    # Subgraphs and the outer graph both contribute keyed by node name.
    flaky_per_index_recorders: dict[str, dict[int, list[int]]] = {}
    # General per-instance execution recorder — populated for plain-node
    # fan-outs (e.g. the crash_injection fixture 067) where no flaky_per_index
    # body records which instances ran. Consulted as a fallback for the
    # instances_executed / skipped assertions when no flaky_per_index node did.
    instance_execution_recorders: dict[str, dict[int, list[int]]] = {}
    trace: list[str] = []
    subgraphs = _build_subgraphs_for(
        spec,
        top_level,
        flaky_per_index_recorders=flaky_per_index_recorders,
        instance_execution_recorders=instance_execution_recorders,
        trace=trace,
    )

    sanitized_spec = _strip_abort_directive(spec)
    built = build_graph(
        sanitized_spec,
        subgraphs=subgraphs,
        trace=trace,
        flaky_per_index_attempt_recorders=flaky_per_index_recorders,
        instance_execution_recorders=instance_execution_recorders,
        fan_out_instance_middleware=_translate_fi_instance_middleware(sanitized_spec),
    )
    builder = built.builder

    # Per proposal 0028 (fixture 055): the fixture's ``state.schema_version``
    # directive declares the graph state class's schema_version, and the
    # optional ``runtime_state_subclass.schema_version`` directive
    # creates a subclass shadowing it. The harness applies both directly
    # to the constructed state class (build_state_cls in adapter.py
    # ignores schema_version today — supporting it via class-level
    # attribute writes here keeps the adapter signature stable).
    state_block = cast("Mapping[str, Any]", spec.get("state") or {})
    declared_schema_version = state_block.get("schema_version")
    if declared_schema_version is not None:
        built.state_cls.schema_version = str(declared_schema_version)

    builder.with_checkpointer(cast(Checkpointer, capturing))
    compiled = builder.compile()

    # Per proposal 0028: ``runtime_state_subclass`` constructs a Python
    # subclass with the overridden ``schema_version`` and passes an
    # instance of THAT subclass to ``invoke()``. The test verifies the
    # engine ignores the subclass's value and writes saves using the
    # declared class's value — proving §10.2's "declared class is
    # canonical" rule.
    runtime_subclass_directive = cast(
        "Mapping[str, Any] | None",
        spec.get("runtime_state_subclass"),
    )
    if runtime_subclass_directive is not None:
        override_version = str(runtime_subclass_directive["schema_version"])
        # Subclass with ClassVar override at the class level. The
        # subclass IS-A built.state_cls (Pydantic structural-conformance
        # holds), so ``compiled.invoke(subclass_instance, ...)`` accepts
        # it without complaint.
        runtime_subclass = type(
            f"{built.state_cls.__name__}Runtime",
            (built.state_cls,),
            {"schema_version": override_version},
        )
        initial_state = cast("State", runtime_subclass(**spec.get("initial_state", {})))
    else:
        initial_state = built.initial_state(spec.get("initial_state", {}))

    # ----- checkpoint_not_found expected (fixture 030) -----
    #
    # Read as a SIBLING of `expected:` first. conformance-adapter section 5.8's
    # preamble says the `expected_*` directives sit inside that block, but the
    # corpus puts `expected_error` at case level nearly everywhere; spec has
    # confirmed the corpus is right and the preamble wrong. The nested read stays
    # as the compatibility form for the fixtures that follow the preamble, and
    # goes when they move.
    #
    # It is a MAPPING, not a bare category string. Reading it as a string meant
    # this arm never fired: fixture 030 ran, asserted nothing, and stayed green
    # with the engine's `raise CheckpointNotFound` removed entirely.
    # Both positions are resolved here rather than from the `expected` block,
    # which is built further down: this arm runs before the first invoke,
    # because its premise is about what the store contains.
    nested_expected = cast("Mapping[str, Any]", spec.get("expected") or {})
    raw_not_found = spec.get("expected_error") or nested_expected.get("expected_error")
    declared = cast("Mapping[str, Any]", raw_not_found or {})
    if declared.get("category") == "checkpoint_not_found":
        assert not nested_expected.get("invariants"), (
            "a checkpoint_not_found case never checks expected.invariants"
        )
        invoke_with = cast("Mapping[str, Any]", spec.get("invoke_with") or {})
        ghost = cast("str", invoke_with.get("resume_invocation", "ghost"))
        # Without this the store is empty and case 2 becomes case 1. Its name
        # says `when_other_records_exist`: the point is that a fabricated id
        # misses even when the checkpointer HAS records, so not-found is a
        # lookup miss rather than an empty-store artefact.
        populate = cast("int", spec.get("populate_checkpointer_via_runs", 0))
        for _ in range(populate):
            await compiled.invoke(initial_state)
        # Non-vacuity on the setup: the assertion below is identical whether the
        # store holds records or not, so a populate loop that wrote nothing
        # would leave this case silently testing the empty-store path again.
        assert len(await capturing.list()) == populate, (
            f"populate_checkpointer_via_runs={populate} left "
            f"{len(await capturing.list())} records; the setup did not take"
        )
        with pytest.raises(CheckpointNotFound) as excinfo:
            await compiled.invoke(initial_state, resume_invocation=ghost)
        assert excinfo.value.category == declared["category"], (
            f"expected category {declared['category']!r}, got {excinfo.value.category!r}"
        )
        if "transient" in declared:
            # Assertable rather than documentary: retry classifies on category
            # membership, so `transient: false` says this category must not be
            # in the set a retry would re-run on.
            actual_transient = excinfo.value.category in TRANSIENT_CATEGORIES
            assert actual_transient == declared["transient"], (
                f"{excinfo.value.category!r} transient={actual_transient}, "
                f"fixture declares {declared['transient']}"
            )
        return

    # Run #1 — first invocation. May succeed or fail per fixture.
    first_run_expected_error = spec.get("first_run_expected_error")
    # crash_injection (proposal 0070): a simulated crash at a checkpoint
    # boundary with NO asserted first-run outcome (it "crashed"). When set,
    # the abort is expected and swallowed without a first_run_expected_error.
    # Coerced to None when not a mapping, matching _find_crash_injection, so a
    # malformed directive parses to no boundary rather than swallowing aborts
    # or tripping the "configured but no crash fired" assertion.
    crash_injection: Any = spec.get("crash_injection")
    if not isinstance(crash_injection, dict):
        crash_injection = None
    invocation_id_first_run: str | None = None
    final_first_run: State | None = None
    trace.clear()
    first_run_started = time.time()
    try:
        final_first_run = await compiled.invoke(
            initial_state,
            correlation_id=spec.get("caller_correlation_id"),
            invocation_id=spec.get("caller_invocation_id"),
        )
        # Under collect mode, the abort_after_instance sentinel fires
        # from inside a per-instance save and is captured by gather's
        # return_exceptions=True. The invoke returns "successfully"
        # from gather's perspective. Detect the simulated crash by
        # the wrapper's ``_aborted`` flag and treat it like a
        # node_exception per the fixture's first_run_expected_error
        # contract.
        if capturing._aborted:  # noqa: SLF001 — test driver intentional
            if crash_injection is not None:
                # The simulated crash fired; no first-run outcome is asserted.
                pass
            elif first_run_expected_error is None:
                raise AssertionError("abort_after_instance fired but no first_run_expected_error declared")
            else:
                expected_category = first_run_expected_error["category"]
                assert expected_category == "node_exception", (
                    f"abort_after_instance simulates node_exception; fixture asserts {expected_category!r}"
                )
                _assert_first_run_raised_from(first_run_expected_error, _abort_after_instance_node(spec))
        elif crash_injection is not None:
            raise AssertionError("crash_injection configured but no crash fired during the first run")
        elif first_run_expected_error is not None:
            raise AssertionError(
                f"expected first run to fail with category "
                f"{first_run_expected_error!r} but it returned successfully"
            )
    except _AbortAfterInstance:
        # Simulated crash sentinel propagated out of the engine (serial /
        # fail_fast flows). For crash_injection it is the expected crash with
        # no asserted outcome; otherwise it pairs with the fixture's
        # ``first_run_expected_error: node_exception`` shape.
        if crash_injection is None:
            if first_run_expected_error is None:
                raise
            expected_category = first_run_expected_error["category"]
            assert expected_category == "node_exception", (
                f"abort_after_instance simulates node_exception; fixture asserts {expected_category!r}"
            )
            _assert_first_run_raised_from(first_run_expected_error, _abort_after_instance_node(spec))
    except CheckpointError:
        # When the abort fires during a subsequent post-abort save
        # (instance dispatched after the target's save), the engine wraps
        # the abort sentinel as ``CheckpointSaveFailed`` and propagates it
        # out. Treat the wrapped abort like a direct sentinel propagation:
        # expected-and-swallowed under crash_injection, else paired with a
        # ``node_exception`` first-run failure.
        if crash_injection is not None and capturing._aborted:  # noqa: SLF001
            pass
        elif first_run_expected_error is None or not capturing._aborted:  # noqa: SLF001
            raise
        else:
            expected_category = first_run_expected_error["category"]
            assert expected_category == "node_exception", (
                f"abort_after_instance simulates node_exception; fixture asserts {expected_category!r}"
            )
            _assert_first_run_raised_from(first_run_expected_error, _abort_after_instance_node(spec))
    except RuntimeGraphError as e:
        # crash_injection's simulated crash can surface here too: under
        # serial fail_fast the abort sentinel is wrapped as
        # ``CheckpointSaveFailed`` and re-wrapped as a ``NodeException``.
        # When the abort fired (``_aborted``), swallow it as the expected
        # crash with no asserted outcome.
        if crash_injection is not None and capturing._aborted:  # noqa: SLF001
            pass
        elif first_run_expected_error is None:
            raise
        else:
            expected_category = first_run_expected_error["category"]
            assert e.category == expected_category, (
                f"first-run error category mismatch: actual={e.category!r}, expected={expected_category!r}"
            )
            _assert_first_run_raised_from(first_run_expected_error, raised_from_node(e))
    first_run_ended = time.time()

    # Capture invocation_id from the latest save (we attached a
    # capturing checkpointer; every save record carries the engine's
    # invocation_id verbatim).
    if capturing.saves:
        invocation_id_first_run = capturing.saves[-1].invocation_id

    # Track per-instance attempts observed in the first run (used by
    # proposal-0009 resume-side assertions). Snapshot before the
    # resume run clears the recorder.
    first_run_attempts = _snapshot_attempt_recorders(flaky_per_index_recorders)
    _ = first_run_attempts  # reserved for cross-run assertions; not used directly yet

    # ----- Saved record assertions -----
    # Source the assertion against the LOADED record, not the
    # last-recorded save call. For batching backends (fixture 054)
    # the two differ: the in-memory ``saves`` list captures every
    # call including buffered-not-flushed ones, but ``load`` only
    # returns durably-flushed state. Per §10.11.4, the spec's
    # ``saved record`` is the loaded record.
    if "saved_record_assertions" in spec and invocation_id_first_run is not None:
        loaded_record = await capturing.load(invocation_id_first_run)
        if loaded_record is None:
            raise AssertionError(f"saved_record_assertions: load({invocation_id_first_run!r}) returned None")
        _assert_saved_record_from(cast("Mapping[str, Any]", spec["saved_record_assertions"]), loaded_record)

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
    if "latest_record_assertions" in expected:
        assert invocation_id_first_run is not None, "latest_record_assertions: no save fired"
        latest = await capturing.load(invocation_id_first_run)
        assert latest is not None, f"latest_record_assertions: no record for {invocation_id_first_run!r}"
        _assert_latest_record(
            cast("Mapping[str, Any]", expected["latest_record_assertions"]),
            latest,
            built.state_cls.schema_version,
            (first_run_started, first_run_ended),
        )

    # Per proposal 0028 (fixture 055): ``every_save_assertions`` is a
    # cross-save invariant block — every captured save during the
    # invocation MUST match every key in this block. Catches
    # implementations that read ``schema_version`` from
    # ``type(state)`` (the runtime subclass) at any intermediate save
    # site instead of from the declared graph state class. Distinct
    # from ``invariants`` above which asserts properties of the SET of
    # saves (e.g., "at least one save fired"); this asserts the same
    # property holds on EVERY save.
    every_save_block = cast(
        "Mapping[str, Any] | None",
        spec.get("every_save_assertions"),
    )
    if every_save_block is not None:
        assert capturing.saves, (
            "every_save_assertions declared but no saves were captured during the invocation"
        )
        for save_idx, saved_record in enumerate(capturing.saves):
            # Read by name rather than through `.items()`, which the nested
            # read check cannot see.
            for key in every_save_block:
                expected_value = every_save_block[key]
                actual_value = getattr(saved_record, key, None)
                assert actual_value == expected_value, (
                    f"every_save_assertions: save[{save_idx}].{key} mismatch — "
                    f"actual={actual_value!r}, expected={expected_value!r}"
                )

    # ----- Resume path (fixtures 025, 029, 031, 048-054, 056) -----
    resume_block = spec.get("resume")
    if resume_block is None or not resume_block.get("from_first_run"):
        return
    if invocation_id_first_run is None:
        raise AssertionError("resume requested but no invocation_id captured (no saves fired)")
    saves_before_resume = list(capturing.saves)
    routes_before_resume = list(capturing.save_routes)
    capturing.saves.clear()
    capturing.save_routes.clear()
    first_record = await capturing.load(invocation_id_first_run)
    # Clear per-instance attempt recorders so the resume run's
    # entries are isolated for ``instance_N_attempt_index_on_resume``
    # and ``instance_N_resume_attempt_count`` assertions.
    for recorder in flaky_per_index_recorders.values():
        recorder.clear()
    for recorder in instance_execution_recorders.values():
        recorder.clear()
    # Reset the abort gate so the resume run completes normally.
    # ``_aborted`` being False disables the ``_raise_if_post_abort``
    # pre-flight check; clearing the abort targets ensures
    # ``_maybe_abort`` is also a no-op on the resume path.
    capturing._aborted = False  # noqa: SLF001 — test driver intentional
    capturing._abort_after_instance = None  # noqa: SLF001
    capturing._abort_after_instance_node = None  # noqa: SLF001
    capturing._abort_after_node = None  # noqa: SLF001
    # Clear the trace so post-resume execution capture is isolated.
    trace.clear()

    # Per proposal 0029 (fixture 056): ``resume_with_modified_items``
    # simulates "user changed the input set between runs." The engine
    # restores state from the saved record on resume (the
    # ``initial_state`` parameter to ``invoke`` is ignored on the
    # resume path); to actually mutate the resumed run's state we
    # install overrides on the capturing checkpointer's ``load``
    # path, which patches the outer state when the engine reads back
    # the saved record. The fan-out node then re-resolves its count
    # from the mutated state and the count-drift check raises.
    modified_items_directive = cast(
        "Mapping[str, Any] | None",
        resume_block.get("resume_with_modified_items"),
    )
    if modified_items_directive is not None:
        capturing.load_state_overrides = dict(modified_items_directive)

    # Per proposal 0029: a resume that hits count drift MUST raise
    # ``checkpoint_record_invalid``. ``resume.expected_error`` carries
    # the assertion (sibling to ``resume.expected``); when present, the
    # invoke MUST raise the named category before final_state can be
    # checked.
    resume_expected_error = cast(
        "Mapping[str, Any] | None",
        resume_block.get("expected_error"),
    )
    if resume_expected_error is not None:
        # CheckpointRecordInvalid (the proposal-0029 count-drift category)
        # is a CheckpointError, NOT a RuntimeGraphError — they're sibling
        # categorized error hierarchies. Catch the broader Exception and
        # assert ``category`` on the value to match both paths.
        with pytest.raises((CheckpointError, RuntimeGraphError)) as excinfo:
            await compiled.invoke(
                initial_state,
                resume_invocation=invocation_id_first_run,
            )
        expected_cat = resume_expected_error["category"]
        actual_cat = cast("str", getattr(excinfo.value, "category", ""))
        assert actual_cat == expected_cat, (
            f"resume expected_error category mismatch: actual={actual_cat!r}, expected={expected_cat!r}"
        )
        return

    resume_events = ObserverFixture(name="resume", attach="invocation", target="outer", behavior="record")
    try:
        final_resume = await compiled.invoke(
            initial_state,
            [make_observer_fn(resume_events, [])],
            resume_invocation=invocation_id_first_run,
            invocation_id=resume_block.get("caller_invocation_id"),
        )
    except CheckpointError:
        raise
    await compiled.drain()
    resume_expected = cast("Mapping[str, Any]", resume_block.get("expected") or {})
    if "final_state" in resume_expected:
        _assert_state_matches(final_resume, cast("Mapping[str, Any]", resume_expected["final_state"]))

    if "nodes_skipped_during_resume" in resume_expected:
        # Every name below passes against an observer that received nothing.
        # Killed by dropping the invocation's observers in the engine.
        assert resume_events.events, "nodes_skipped_during_resume: the resume observer received no events"
        emitted = {ev["node_name"] for ev in resume_events.events}
        for node_name in cast(Iterable[str], resume_expected["nodes_skipped_during_resume"]):
            assert node_name not in emitted, (
                f"node {node_name!r} expected to be skipped on resume but emitted events: "
                f"{[ev for ev in resume_events.events if ev['node_name'] == node_name]}"
            )
            # Catches a body that re-ran without dispatching events. Killed by
            # running the node at the engine's resume skip gate.
            assert node_name not in trace, (
                f"node {node_name!r} expected to be skipped on resume but its body ran: trace={trace}"
            )
    # Read from the resume invocation's events: the trace records a `flaky`
    # node only on its first attempt.
    if "nodes_executed_during_resume" in resume_expected:
        started = [ev["node_name"] for ev in resume_events.events if ev["phase"] == "started"]
        expected_nodes = list(cast(Iterable[str], resume_expected["nodes_executed_during_resume"]))
        assert started == expected_nodes, (
            f"nodes_executed_during_resume mismatch: actual={started}, expected={expected_nodes}"
        )

    # proposal-0009 instances_executed_during_resume /
    # instances_skipped_during_resume — assert against the
    # per-instance attempt recorders (each instance whose body ran
    # appears in the recorder).
    # flaky_per_index recorders capture execution for retry-resume fixtures;
    # for a plain-node fan-out (the crash_injection fixture 067) no
    # flaky_per_index body records, so fall back to the general execution
    # recorder. The flaky_per_index path stays primary, so existing fixtures
    # are unchanged.
    executed_set = set(_flatten_executed_instances(flaky_per_index_recorders))
    if not executed_set:
        executed_set = set(_flatten_executed_instances(instance_execution_recorders))
    if "instances_executed_during_resume" in resume_expected:
        expected_executed = sorted(
            int(i) for i in cast(Iterable[Any], resume_expected["instances_executed_during_resume"])
        )
        actual_executed = sorted(executed_set)
        assert actual_executed == expected_executed, (
            f"instances_executed_during_resume mismatch: "
            f"actual={actual_executed}, expected={expected_executed}"
        )
    if "instances_skipped_during_resume" in resume_expected:
        expected_skipped = sorted(
            int(i) for i in cast(Iterable[Any], resume_expected["instances_skipped_during_resume"])
        )
        # An instance is "skipped" if its body did NOT run during resume.
        for skipped_idx in expected_skipped:
            assert skipped_idx not in executed_set, (
                f"instance {skipped_idx} expected to be skipped on resume "
                f"but its body ran (recorded attempts: {executed_set})"
            )

    evidence = _RunEvidence(
        spec=spec,
        first_saves=saves_before_resume,
        resume_saves=list(capturing.saves),
        saves=[*saves_before_resume, *capturing.saves],
        save_routes=[*routes_before_resume, *capturing.save_routes],
        first_record=first_record,
        resume_events=resume_events.events,
        resume_trace=list(trace),
        final_resume=final_resume,
        executed_instances=frozenset(executed_set),
        recorders=flaky_per_index_recorders,
        resume_caller_invocation_id=cast("str | None", resume_block.get("caller_invocation_id")),
    )
    if resume_invariants:
        _assert_run_invariants(resume_invariants, evidence)
    if case_invariants:
        _assert_run_invariants(case_invariants, evidence)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_subgraphs_for(
    spec: Mapping[str, Any],
    top_level: Mapping[str, Any],
    *,
    flaky_per_index_recorders: dict[str, dict[int, list[int]]] | None = None,
    instance_execution_recorders: dict[str, dict[int, list[int]]] | None = None,
    leaf_value_recorder: list[Any] | None = None,
    trace: list[str] | None = None,
) -> dict[str, Any]:
    """Build subgraphs from either the case's own ``subgraph`` /
    ``subgraphs`` block or the cases-fixture's top-level shared
    ``subgraph`` block. Each case may declare local subgraphs OR
    inherit from the top level.

    ``flaky_per_index_recorders`` and ``instance_execution_recorders``
    (when supplied) thread through to inner-subgraph build so per-instance
    bodies inside subgraphs populate the recorder maps the resume
    assertions read. ``leaf_value_recorder`` (fixture 076) records the
    source values that inner-leaf ``update_from_field`` bodies read, so the
    resume driver can tell which inner instances re-ran vs. skipped.
    ``trace`` (when supplied) is shared by every subgraph's bodies, so the
    caller sees inner nodes run.
    """
    # Section 5.4 ranks per NAME. The merge below replaces the document's whole
    # block when the case declares any, dropping a name the document declares
    # and the case omits. The singular form is left in the merge: `_build_subgraphs`
    # reads it separately, so folding it into the mapping would build the same
    # subgraph twice.
    merged = {**dict(top_level), **dict(spec)}
    ranked = resolve_subgraphs(top_level, spec)
    if ranked:
        merged["subgraphs"] = ranked
    return _build_subgraphs(
        merged,
        flaky_per_index_recorders=flaky_per_index_recorders,
        instance_execution_recorders=instance_execution_recorders,
        leaf_value_recorder=leaf_value_recorder,
        trace=trace,
    )


def _build_subgraphs(
    spec: Mapping[str, Any],
    *,
    flaky_per_index_recorders: dict[str, dict[int, list[int]]] | None = None,
    instance_execution_recorders: dict[str, dict[int, list[int]]] | None = None,
    leaf_value_recorder: list[Any] | None = None,
    trace: list[str] | None = None,
) -> dict[str, Any]:
    """Build any subgraphs (`subgraph:` or `subgraphs:`) the fixture
    declares. Returns a registry the adapter consumes by name.

    Inner subgraphs may declare flaky_per_index nodes (fixture 048+:
    the failing/succeeding scorer node lives in the inner subgraph,
    not the outer graph) or plain nodes (fixture 067's crash_injection
    scorer). Thread both recorder maps through so those bodies populate
    the per-instance attempt / execution tables the resume assertions read.
    """
    subgraph_specs: dict[str, Any] = {}
    if "subgraph" in spec:
        single = cast("Mapping[str, Any]", spec["subgraph"])
        name = single.get("name") or "subgraph"
        subgraph_specs[name] = single
    if "subgraphs" in spec:
        for k, v in cast("dict[str, Any]", spec["subgraphs"]).items():
            subgraph_specs[k] = v
    compiled_subgraphs: dict[str, Any] = {}
    # Build in declaration order, threading the registry built so far so a
    # subgraph may reference an earlier one (fixture 076: inner_runner's
    # fan-out targets the leaf_scorer subgraph). YAML preserves insertion
    # order, so an inner subgraph declared before its user resolves.
    for name, sub_spec in subgraph_specs.items():
        sub_built = build_graph(
            sub_spec,
            subgraphs=compiled_subgraphs,
            trace=trace,
            flaky_per_index_attempt_recorders=flaky_per_index_recorders,
            instance_execution_recorders=instance_execution_recorders,
            leaf_value_recorder=leaf_value_recorder,
        )
        compiled_subgraphs[name] = sub_built.builder.compile()
    return compiled_subgraphs


def _assert_saved_record_from(
    block: Mapping[str, Any],
    record: CheckpointRecord,
) -> None:
    """Assert ``block`` against ``record``. Same semantics as
    :func:`_assert_saved_record` but the caller supplies the record
    directly (used for fixtures where the assertion targets the
    loaded record rather than the last in-memory save call —
    e.g., the batching case where buffered saves are invisible to
    ``load``)."""
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
    if "fan_out_progress" in block:
        _assert_fan_out_progress(
            cast("Mapping[str, Any]", block["fan_out_progress"]),
            record.fan_out_progress,
        )
    if "fan_out_node_in_completed_positions" in block:
        expected_present = bool(block["fan_out_node_in_completed_positions"])
        actual_present = any(
            p.node_name in {fp.fan_out_node_name for fp in record.fan_out_progress}
            for p in record.completed_positions
        )
        assert actual_present == expected_present, (
            f"fan_out_node_in_completed_positions mismatch: "
            f"actual={actual_present}, expected={expected_present}"
        )


def _assert_latest_record(
    block: Mapping[str, Any],
    record: CheckpointRecord,
    declared_schema_version: str,
    saved_during: tuple[float, float],
) -> None:
    """Assert a ``latest_record_assertions`` block against the latest
    loaded record, which was saved between the two ``saved_during``
    wall-clock times."""
    handled = {
        "invocation_id",
        "correlation_id",
        "state",
        "completed_positions",
        "parent_states",
        "fan_out_progress",
        "last_saved_at",
        "schema_version",
    }
    unhandled = set(block) - handled
    assert not unhandled, f"latest_record_assertions keys {sorted(unhandled)} are not handled"
    for key in ("invocation_id", "correlation_id"):
        if key not in block:
            continue
        actual_id = cast("str", getattr(record, key))
        want_id = block[key]
        if want_id == "<uuid>":
            # conformance-adapter section 5.10: a canonical UUIDv4.
            assert _is_canonical_uuid4(actual_id), f"latest record {key}={actual_id!r} is not a UUIDv4"
        else:
            assert actual_id == want_id, f"latest record {key}: actual={actual_id!r}, expected={want_id!r}"
    if "state" in block:
        actual_state = record.state.model_dump()
        assert actual_state == block["state"], (
            f"latest record state: actual={actual_state}, expected={block['state']}"
        )
    if "completed_positions" in block:
        _assert_saved_record_from({"completed_positions": block["completed_positions"]}, record)
    if "parent_states" in block:
        actual_parents = [ps.model_dump() for ps in record.parent_states]
        assert actual_parents == block["parent_states"], (
            f"latest record parent_states: actual={actual_parents}, expected={block['parent_states']}"
        )
    if "fan_out_progress" in block:
        if block["fan_out_progress"] is None:
            # Null reads as absent, which this record holds as an empty tuple.
            assert record.fan_out_progress == (), (
                f"latest record fan_out_progress expected absent, got {record.fan_out_progress}"
            )
        else:
            expected_progress = cast("Mapping[str, Any]", block["fan_out_progress"])
            _assert_fan_out_progress(expected_progress, record.fan_out_progress)
    if "last_saved_at" in block:
        # `<timestamp>` is not a section 5.10 matcher; read as a wall-clock time
        # taken while the saving run was in flight. Killed by time.monotonic()
        # at the engine's record build.
        want_saved_at = block["last_saved_at"]
        assert want_saved_at == "<timestamp>", f"unsupported last_saved_at {want_saved_at!r}"
        started, ended = saved_during
        saved_at = record.last_saved_at
        assert isinstance(saved_at, float) and started <= saved_at <= ended, (
            f"latest record last_saved_at={saved_at!r} is outside the saving run [{started}, {ended}]"
        )
    if "schema_version" in block:
        # pipeline-utilities section 10.2: the declared state class's value, or
        # the empty-string sentinel when it declares none. `<any-string>` would
        # reject the sentinel (conformance-adapter section 5.10), so the token
        # reads as the declared value.
        want = block["schema_version"]
        if want == "<any-string>":
            want = declared_schema_version
        assert record.schema_version == want, (
            f"latest record schema_version: actual={record.schema_version!r}, expected={want!r}"
        )


def _is_canonical_uuid4(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _assert_fan_out_progress(
    expected: Mapping[str, Any],
    actual: tuple[FanOutProgress, ...],
) -> None:
    """Assert against a ``fan_out_progress`` block in the fixture.

    Block shape:

        fan_out_progress:
          <node_name>:
            instance_count: int
            instances:
              - state: completed | in_flight | not_started
                result: <any>                 # optional, scalar matches
                result_kind: error            # optional, asserts result is an error dict
                state_one_of: [in_flight, not_started]  # optional alternation
                completed_inner_positions:    # optional list-of-dicts matchers
                  - {node_name: step_a, attempt_index: 0}
    """
    by_name = {fp.fan_out_node_name: fp for fp in actual}
    for node_name, fp_expected in expected.items():
        fp_expected_dict = cast("Mapping[str, Any]", fp_expected)
        if node_name not in by_name:
            raise AssertionError(
                f"fan_out_progress: no entry for fan-out node {node_name!r}; "
                f"actual entries: {sorted(by_name)}"
            )
        fp = by_name[node_name]
        if "instance_count" in fp_expected_dict:
            assert fp.instance_count == fp_expected_dict["instance_count"], (
                f"fan_out_progress[{node_name!r}].instance_count: "
                f"actual={fp.instance_count}, expected={fp_expected_dict['instance_count']}"
            )
        if "instances" in fp_expected_dict:
            instances_expected = cast("list[Mapping[str, Any]]", fp_expected_dict["instances"])
            assert len(fp.instances) == len(instances_expected), (
                f"fan_out_progress[{node_name!r}].instances length: "
                f"actual={len(fp.instances)}, expected={len(instances_expected)}"
            )
            for idx, (inst_expected, inst_actual) in enumerate(
                zip(instances_expected, fp.instances, strict=True)
            ):
                _assert_fan_out_instance(node_name, idx, inst_expected, inst_actual)


def _assert_fan_out_instance(
    node_name: str,
    idx: int,
    expected: Mapping[str, Any],
    actual: FanOutInstanceProgress,
) -> None:
    """Assert one entry inside a fan_out_progress.instances list."""
    if "state" in expected:
        assert actual.state == expected["state"], (
            f"fan_out_progress[{node_name!r}].instances[{idx}].state: "
            f"actual={actual.state!r}, expected={expected['state']!r}"
        )
    if "state_one_of" in expected:
        allowed = set(cast("Iterable[str]", expected["state_one_of"]))
        assert actual.state in allowed, (
            f"fan_out_progress[{node_name!r}].instances[{idx}].state: "
            f"actual={actual.state!r}, expected one of {allowed!r}"
        )
    if "result" in expected:
        assert actual.result == expected["result"], (
            f"fan_out_progress[{node_name!r}].instances[{idx}].result: "
            f"actual={actual.result!r}, expected={expected['result']!r}"
        )
    if "result_is_error" in expected:
        # Spec §10.11 (proposal 0027): explicit boolean discriminator
        # on the per-instance entry. Replaced the pre-0027
        # ``result_kind: error`` shape heuristic.
        assert actual.result_is_error == expected["result_is_error"], (
            f"fan_out_progress[{node_name!r}].instances[{idx}].result_is_error: "
            f"actual={actual.result_is_error!r}, expected={expected['result_is_error']!r}"
        )
    if "result_present" in expected:
        # Spec §10.11 (proposal 0027): assert the captured
        # contribution is a non-None value — the fixture's way of
        # saying "an entry was recorded" without constraining its
        # shape (the value remains impl-defined per §9.5). Pair with
        # ``result_is_error: true`` to assert "an error contribution
        # was captured" portably across implementations whose
        # error_record formats differ.
        #
        # Note on the semantic: ``FanOutInstanceProgress.result`` is
        # a dataclass field that always exists as an attribute with
        # a None default; "presence" here means "result is a
        # non-None value." A hypothetical legitimate ``result =
        # None`` on a completed-success entry would be treated as
        # "not present" by this check, but spec frames ``result`` as
        # "the durable contribution" and no fixture exercises a
        # None contribution on a completed entry.
        result_has_value = actual.result is not None
        assert result_has_value == expected["result_present"], (
            f"fan_out_progress[{node_name!r}].instances[{idx}].result_present: "
            f"actual={result_has_value!r}, expected={expected['result_present']!r}"
        )
    if "completed_inner_positions" in expected:
        positions_expected = cast("list[Mapping[str, Any]]", expected["completed_inner_positions"])
        # Compare by node_name + attempt_index per the spec; namespace
        # and step are engine-internal details fixture authors don't
        # always include.
        actual_min = [
            {"node_name": p.node_name, "attempt_index": p.attempt_index}
            for p in actual.completed_inner_positions
        ]
        expected_min = [
            {"node_name": p["node_name"], "attempt_index": p.get("attempt_index", 0)}
            for p in positions_expected
        ]
        assert actual_min == expected_min, (
            f"fan_out_progress[{node_name!r}].instances[{idx}].completed_inner_positions: "
            f"actual={actual_min}, expected={expected_min}"
        )


class _RunEvidence(NamedTuple):
    """What the runner recorded across a case's first run and its resume."""

    spec: Mapping[str, Any]
    first_saves: Sequence[CheckpointRecord]
    resume_saves: Sequence[CheckpointRecord]
    # Both runs' saves, in order.
    saves: Sequence[CheckpointRecord]
    # Parallel to ``saves``: the checkpointer method each save arrived through.
    save_routes: Sequence[str]
    first_record: CheckpointRecord | None
    resume_events: Sequence[Mapping[str, Any]]
    resume_trace: Sequence[str]
    final_resume: State | None
    executed_instances: frozenset[int]
    recorders: Mapping[str, dict[int, list[int]]]
    resume_caller_invocation_id: str | None


def _one(values: Iterable[str], what: str) -> str:
    distinct = set(values)
    assert len(distinct) == 1, f"{what}: expected exactly one value, got {sorted(distinct)}"
    return next(iter(distinct))


def _first_correlation_id(ev: _RunEvidence) -> str:
    return _one((s.correlation_id for s in ev.first_saves), "first-run correlation_id")


def _resumed_correlation_id(ev: _RunEvidence) -> str:
    return _one((s.correlation_id for s in ev.resume_saves), "resumed-run correlation_id")


def _first_invocation_id(ev: _RunEvidence) -> str:
    return _one((s.invocation_id for s in ev.first_saves), "first-run invocation_id")


def _resumed_invocation_id(ev: _RunEvidence) -> str:
    return _one((s.invocation_id for s in ev.resume_saves), "resumed-run invocation_id")


def _saved_record(ev: _RunEvidence) -> CheckpointRecord:
    assert ev.first_record is not None, "the first run left no saved record"
    return ev.first_record


def _ran_on_resume(ev: _RunEvidence, position: NodePosition) -> bool:
    # Without events every node would read as skipped.
    assert ev.resume_events, "the resume observer received no events"
    # An event's namespace ends with the node itself; a position's stops at its parent.
    path = [*position.namespace, position.node_name]
    started = any(e["phase"] == "started" and e["namespace"] == path for e in ev.resume_events)
    return started or position.node_name in ev.resume_trace


def _final_list(field_name: str, ev: _RunEvidence) -> list[Any]:
    assert ev.final_resume is not None, "the resume returned no final state"
    value = ev.final_resume.model_dump().get(field_name)
    assert isinstance(value, list), f"final state {field_name!r} is not a list ({value!r})"
    return cast("list[Any]", value)


def _check_first_correlation_id(want: Any, ev: _RunEvidence) -> None:
    actual = _first_correlation_id(ev)
    assert actual == want, f"first-run correlation_id: actual={actual!r}, expected={want!r}"


def _check_resumed_correlation_id(want: Any, ev: _RunEvidence) -> None:
    actual = _resumed_correlation_id(ev)
    assert actual == want, f"resumed-run correlation_id: actual={actual!r}, expected={want!r}"


def _check_saved_record_correlation_id(want: Any, ev: _RunEvidence) -> None:
    actual = _saved_record(ev).correlation_id
    assert actual == want, f"saved record correlation_id: actual={actual!r}, expected={want!r}"


def _check_one_correlation_id(_: Any, ev: _RunEvidence) -> None:
    assert ev.first_saves and ev.resume_saves, "both runs must save"
    _one((s.correlation_id for s in ev.saves), "correlation_id across both runs")


def _check_correlation_id_carried(_: Any, ev: _RunEvidence) -> None:
    first, resumed = _first_correlation_id(ev), _resumed_correlation_id(ev)
    assert resumed == first, f"the resume carries correlation_id {resumed!r}; the first run had {first!r}"


def _check_first_correlation_id_uuid4(_: Any, ev: _RunEvidence) -> None:
    actual = _first_correlation_id(ev)
    assert _is_canonical_uuid4(actual), f"first-run correlation_id {actual!r} is not a UUIDv4"


def _check_first_invocation_id(want: Any, ev: _RunEvidence) -> None:
    actual = _first_invocation_id(ev)
    assert actual == want, f"first-run invocation_id: actual={actual!r}, expected={want!r}"


def _check_invocation_ids_differ(_: Any, ev: _RunEvidence) -> None:
    first = _first_invocation_id(ev)
    assert _resumed_invocation_id(ev) != first, f"the resume reused the first run's invocation_id {first!r}"


def _check_resumed_invocation_id_fresh(_: Any, ev: _RunEvidence) -> None:
    resumed = _resumed_invocation_id(ev)
    assert _is_canonical_uuid4(resumed), f"resumed-run invocation_id {resumed!r} is not a UUIDv4"
    assert resumed != _first_invocation_id(ev), "the resume reused the first run's invocation_id"


def _check_resumed_invocation_id_not_callers(_: Any, ev: _RunEvidence) -> None:
    supplied = ev.resume_caller_invocation_id
    assert supplied is not None, "the resume supplied no invocation_id, so there was none to ignore"
    assert _resumed_invocation_id(ev) != supplied, (
        f"the resume adopted the caller's invocation_id {supplied!r}"
    )


def _check_inner_first_node_not_re_run(_: Any, ev: _RunEvidence) -> None:
    inner = sorted((p for p in _saved_record(ev).completed_positions if p.namespace), key=lambda p: p.step)
    assert inner, "the saved record holds no completed position inside a subgraph"
    assert not _ran_on_resume(ev, inner[0]), f"inner node {inner[0].node_name!r} re-ran on resume"


def _check_subgraph_re_entry_parent_states(_: Any, ev: _RunEvidence) -> None:
    saved = [ps.model_dump() for ps in _saved_record(ev).parent_states]
    assert saved, "the saved record carries no parent_states"
    re_entered = next(
        (e for e in ev.resume_events if e["phase"] == "started" and len(e["namespace"]) > 1),
        None,
    )
    assert re_entered is not None, "no node inside a subgraph started on resume"
    # Killed by changing the outer state restored at re-entry. Restoring the
    # state class's defaults instead is invisible here: 029 saves its outer
    # state at the defaults.
    assert re_entered["parent_states"] == saved, (
        f"re-entered {re_entered['node_name']!r} got parent_states {re_entered['parent_states']}; "
        f"the record saved {saved}"
    )


def _check_degraded_slot_rolled_forward(_: Any, ev: _RunEvidence) -> None:
    assert ev.final_resume is not None, "the resume returned no final state"
    final = ev.final_resume.model_dump()
    nodes = cast("Mapping[str, Mapping[str, Any]]", ev.spec.get("nodes") or {})
    checked = 0
    for fp in _saved_record(ev).fan_out_progress:
        target = cast("str", nodes[fp.fan_out_node_name]["fan_out"]["target_field"])
        for index, inst in enumerate(fp.instances):
            if inst.state != "completed":
                continue
            assert index not in ev.executed_instances, f"{fp.fan_out_node_name}[{index}] re-ran on resume"
            # Indexing the target by instance holds while the target starts
            # empty, as 069's does.
            assert final[target][index] == inst.result, (
                f"{fp.fan_out_node_name}[{index}]: final {target}[{index}]={final[target][index]!r}, "
                f"saved {inst.result!r}"
            )
            checked += 1
    assert checked, "the saved record holds no completed fan-out instance"


def _check_after_node_rolled_forward(_: Any, ev: _RunEvidence) -> None:
    node = cast("Mapping[str, Any]", ev.spec.get("crash_injection") or {}).get("after_node")
    assert isinstance(node, str), "after_node_crash_rolls_node_forward needs crash_injection.after_node"
    saved = [p for p in _saved_record(ev).completed_positions if p.node_name == node]
    assert saved, f"the record saved at the crash boundary does not list {node!r}"
    assert not _ran_on_resume(ev, saved[0]), f"{node!r} re-ran on resume"


def _check_batching_scope(_: Any, ev: _RunEvidence) -> None:
    routed = list(zip(ev.saves, ev.save_routes, strict=True))
    assert any(route == "save_fan_out_internal" for _, route in routed), (
        "no save went through the batched path"
    )
    fan_outs = {fp.fan_out_node_name for record, _ in routed for fp in record.fan_out_progress}
    completions = [
        route
        for record, route in routed
        if any(p.node_name in fan_outs and p.fan_out_index is None for p in record.completed_positions)
    ]
    assert completions, "no save recorded a fan-out node's own completion"
    # Killed by sending the fan-out node's completion save down the batched path.
    assert completions[0] == "save", f"the fan-out node's completion save went through {completions[0]!r}"


def _check_no_duplicates(field_name: str) -> Callable[[Any, _RunEvidence], None]:
    def check(_: Any, ev: _RunEvidence) -> None:
        values = _final_list(field_name, ev)
        hashes = [_hashable(v) for v in values]
        assert len(set(hashes)) == len(hashes), f"final {field_name} has duplicate entries: {values}"

    return check


def _check_list_length(field_name: str) -> Callable[[Any, _RunEvidence], None]:
    def check(want: Any, ev: _RunEvidence) -> None:
        values = _final_list(field_name, ev)
        assert len(values) == want, f"final {field_name} length: actual={len(values)}, expected={want}"

    return check


def _check_instance_invariant(index: int, kind: str, node: str | None, value: Any, ev: _RunEvidence) -> None:
    if node is not None:
        assert value is True, f"instance_{index}_executes_{node}_on_resume: only `true` is supported"
        assert any(
            e["phase"] == "started" and e["node_name"] == node and e.get("fan_out_index") == index
            for e in ev.resume_events
        ), f"instance {index} did not run {node!r} on resume"
        return
    seen = [recorder[index] for recorder in ev.recorders.values() if index in recorder]
    if kind == "resume_attempt_count" and value == 0:
        assert not seen, f"instance {index} ran on resume: {seen}"
        return
    assert seen and all(seen), f"no recorder saw instance {index} attempt on resume"
    for attempts in seen:
        if kind == "resume_attempt_count":
            assert len(attempts) == value, (
                f"instance {index} resume attempts: actual={len(attempts)}, expected={value}"
            )
        else:
            assert attempts[0] == value, (
                f"instance {index} first resume attempt_index: actual={attempts[0]}, expected={value}"
            )


_INSTANCE_INVARIANT = re.compile(
    r"instance_(\d+)_(attempt_index_on_resume|resume_attempt_count|executes_(\w+)_on_resume)"
)

# Every invariant name this runner checks once a resume has run, at case level
# or under `resume`. The unknown-name guard is this table plus the
# `instance_<N>_...` family above.
_RUN_INVARIANTS: dict[str, Callable[[Any, _RunEvidence], None]] = {
    "first_run_correlation_id": _check_first_correlation_id,
    "resumed_run_correlation_id": _check_resumed_correlation_id,
    "saved_record_correlation_id": _check_saved_record_correlation_id,
    "correlation_id_uniform_across_both_runs": _check_one_correlation_id,
    "resumed_run_correlation_id_matches_first": _check_correlation_id_carried,
    "resumed_correlation_id_matches_original": _check_correlation_id_carried,
    "no_new_correlation_id_generated_on_resume": _check_correlation_id_carried,
    "first_run_correlation_id_is_uuidv4": _check_first_correlation_id_uuid4,
    "first_run_invocation_id": _check_first_invocation_id,
    "first_and_resumed_invocation_ids_differ": _check_invocation_ids_differ,
    "resumed_invocation_id_differs_from_original": _check_invocation_ids_differ,
    "resumed_run_invocation_id_is_fresh_uuidv4": _check_resumed_invocation_id_fresh,
    "resumed_run_invocation_id_not_caller_supplied": _check_resumed_invocation_id_not_callers,
    "inner_first_node_not_re_run": _check_inner_first_node_not_re_run,
    "subgraph_re_entry_uses_parent_states": _check_subgraph_re_entry_parent_states,
    "degrade_slot_rolled_forward_unchanged": _check_degraded_slot_rolled_forward,
    "after_node_crash_rolls_node_forward": _check_after_node_rolled_forward,
    "batching_scoped_to_fan_out_internal_saves_only": _check_batching_scope,
    "no_duplicate_results": _check_no_duplicates("results"),
    "no_duplicate_error_entries": _check_no_duplicates("errors"),
    "results_list_length": _check_list_length("results"),
    "errors_list_length": _check_list_length("errors"),
}

# Names whose value is the expectation. Every other name accepts only `true`,
# so a fixture that flips one to `false` fails rather than switching it off.
_VALUED_RUN_INVARIANTS = frozenset(
    {
        "first_run_correlation_id",
        "resumed_run_correlation_id",
        "saved_record_correlation_id",
        "first_run_invocation_id",
        "results_list_length",
        "errors_list_length",
    }
)


def _assert_run_invariant_names(block: Mapping[str, Any], where: str) -> None:
    unknown = sorted(n for n in block if n not in _RUN_INVARIANTS and not _INSTANCE_INVARIANT.fullmatch(n))
    assert not unknown, f"unhandled {where} invariant(s): {unknown}"


def _assert_run_invariants(block: Mapping[str, Any], ev: _RunEvidence) -> None:
    """Check each invariant in ``block`` against what the first run and the
    resume recorded."""
    for name, value in block.items():
        if (match := _INSTANCE_INVARIANT.fullmatch(name)) is not None:
            _check_instance_invariant(int(match.group(1)), match.group(2), match.group(3), value, ev)
            continue
        if name not in _VALUED_RUN_INVARIANTS:
            assert value is True, f"invariant {name}: only `true` is supported; got {value!r}"
        _RUN_INVARIANTS[name](value, ev)


def _hashable(value: Any) -> Any:
    """Make a value hashable for set-based duplicate detection. Lists
    and dicts get rendered as tuples of (key, value) pairs."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in cast("dict[str, Any]", value).items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in cast("list[Any]", value))
    return value


def _snapshot_attempt_recorders(
    recorders: Mapping[str, dict[int, list[int]]],
) -> dict[str, dict[int, list[int]]]:
    """Deep-copy the per-flaky-node attempt recorder map."""
    out: dict[str, dict[int, list[int]]] = {}
    for node_name, idx_map in recorders.items():
        out[node_name] = {idx: list(attempts) for idx, attempts in idx_map.items()}
    return out


def _flatten_executed_instances(
    recorders: Mapping[str, dict[int, list[int]]],
) -> list[int]:
    """Union of instance indices observed across every flaky_per_index
    recorder. An instance whose body fired at least once during this
    run appears."""
    seen: set[int] = set()
    for idx_map in recorders.values():
        seen.update(idx for idx, attempts in idx_map.items() if attempts)
    return sorted(seen)


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
        # Magic-key length assertions can appear inside ``final_state``
        # alongside literal field assertions (e.g. fixture 052 has
        # ``errors_list_length: 1`` as a sibling of the literal
        # ``results`` and ``items`` fields). Route the magic keys to
        # length checks against the corresponding list field; literal
        # keys take the standard equality path.
        if k == "errors_list_length":
            errors = actual_dict.get("errors")
            assert isinstance(errors, list), f"errors_list_length: errors is not a list ({errors!r})"
            errors_list = cast("list[Any]", errors)
            assert len(errors_list) == v, f"errors_list_length: actual={len(errors_list)}, expected={v}"
            continue
        if k == "results_list_length":
            results = actual_dict.get("results")
            assert isinstance(results, list), f"results_list_length: results is not a list ({results!r})"
            results_list = cast("list[Any]", results)
            assert len(results_list) == v, f"results_list_length: actual={len(results_list)}, expected={v}"
            continue
        assert actual_dict.get(k) == v, f"state field {k!r}: actual={actual_dict.get(k)!r}, expected={v!r}"


_FIRST_RUN_INVARIANTS = frozenset(
    {
        "save_count",
        "save_order_matches_completed_event_order",
        "saves_for_fan_out_internals",
        "invocation_id_is_uuidv4",
        "last_saved_at_monotonic_across_saves",
        "completed_positions_step_monotonic",
    }
)


def _assert_invariants(
    invariants: Mapping[str, Any],
    saves: list[CheckpointRecord],
) -> None:
    unknown = sorted(set(invariants) - _FIRST_RUN_INVARIANTS)
    assert not unknown, f"unhandled expected invariant(s): {unknown}"
    for name, value in invariants.items():
        if name not in {"save_count", "saves_for_fan_out_internals"}:
            assert value is True, f"invariant {name}: only `true` is supported; got {value!r}"
    if "save_count" in invariants:
        assert len(saves) == invariants["save_count"], (
            f"save_count: actual={len(saves)}, expected={invariants['save_count']}"
        )
    if "save_order_matches_completed_event_order" in invariants:
        assert saves, "save_order_matches_completed_event_order: no save fired"
        last_steps = [s.completed_positions[-1].step for s in saves]
        assert last_steps == sorted(last_steps), f"save order does not match step order: {last_steps}"
    if "saves_for_fan_out_internals" in invariants:
        assert invariants["saves_for_fan_out_internals"] == 0, (
            "saves_for_fan_out_internals: only 0 is supported"
        )
        for s in saves:
            for p in s.completed_positions:
                assert p.fan_out_index is None, f"unexpected fan-out internal save: {p}"
    if "invocation_id_is_uuidv4" in invariants:
        assert saves, "invocation_id_is_uuidv4: no save fired"
        not_uuid = sorted({s.invocation_id for s in saves if not _is_canonical_uuid4(s.invocation_id)})
        assert not not_uuid, f"saved invocation_id is not a UUIDv4: {not_uuid}"
    if "last_saved_at_monotonic_across_saves" in invariants:
        stamps = [s.last_saved_at for s in saves]
        assert len(stamps) >= 2, (
            f"last_saved_at_monotonic_across_saves needs two or more saves; got {len(stamps)}"
        )
        # Non-decreasing: two saves can land in the same clock tick.
        assert stamps == sorted(stamps), f"last_saved_at went backwards across saves: {stamps}"
    if "completed_positions_step_monotonic" in invariants:
        assert any(len(s.completed_positions) >= 2 for s in saves), (
            "completed_positions_step_monotonic: no save lists two or more positions"
        )
        for s in saves:
            steps = [p.step for p in s.completed_positions]
            assert all(a < b for a, b in zip(steps, steps[1:], strict=False)), (
                f"completed_positions steps do not increase: {steps}"
            )


# ---------------------------------------------------------------------------
# crash_injection: after_node boundary (proposal 0070)
# ---------------------------------------------------------------------------


async def test_capturing_checkpointer_aborts_after_node() -> None:
    # The ``crash_injection: {after_node}`` boundary fires the simulated-crash
    # sentinel after the save whose record records the named node in
    # ``completed_positions``. Fixture 070 drives the boundary end to end, but
    # its target is the entry node, so the first save already records it: a
    # double that aborted after ANY node passes 070. The second half here is
    # the only check that a save recording a different node does not abort.
    cp = _CapturingCheckpointer(abort_after_node="target")
    record = CheckpointRecord(
        invocation_id="inv",
        correlation_id="c",
        state={},
        completed_positions=(NodePosition(namespace=(), node_name="target", step=0),),
        parent_states=(),
        last_saved_at=0.0,
    )
    with pytest.raises(_AbortAfterInstance):
        await cp.save("inv", record)
    assert cp._aborted is True  # noqa: SLF001 — test driver intentional

    # A save whose record does not record the target node does not abort.
    cp_other = _CapturingCheckpointer(abort_after_node="target")
    other_record = CheckpointRecord(
        invocation_id="inv",
        correlation_id="c",
        state={},
        completed_positions=(NodePosition(namespace=(), node_name="other", step=0),),
        parent_states=(),
        last_saved_at=0.0,
    )
    await cp_other.save("inv", other_record)
    assert cp_other._aborted is False  # noqa: SLF001 — test driver intentional


# ---------------------------------------------------------------------------
# Proposal 0085 nested-resume coverage the fixture-076 shape leaves out:
# per-lineage count-drift and collect-mode error-entry roll-forward. Surfaced
# by the adversarial review of 0085 (both are correct on inspection + verified
# here; the fixture's symmetric fail_fast shape never exercises these branches).
# ---------------------------------------------------------------------------


def _nested_subgraphs(*, inner_error_policy: str) -> dict[str, Any]:
    """A 076-shaped two-level fan-out subgraph set, with the inner fan-out's
    error_policy configurable so the collect-mode error path can be driven."""
    inner_state: dict[str, Any] = {
        "inner_items": {"type": "list<int>", "default": []},
        "inner_results": {"type": "list<int>", "reducer": "append", "default": []},
    }
    inner_fan_out: dict[str, Any] = {
        "subgraph": "leaf_scorer",
        "items_field": "inner_items",
        "item_field": "input",
        "collect_field": "out",
        "target_field": "inner_results",
        "error_policy": inner_error_policy,
        "concurrent_mode": "serial",
    }
    if inner_error_policy == "collect":
        inner_state["inner_errors"] = {"type": "list<error_entry>", "default": []}
        inner_fan_out["errors_field"] = "inner_errors"
    return {
        "leaf_scorer": {
            "state": {
                "fields": {"input": {"type": "int", "default": 0}, "out": {"type": "int", "default": 0}}
            },
            "entry": "score",
            "nodes": {"score": {"update_from_field": {"out": "input"}}},
            "edges": [{"from": "score", "to": "END"}],
        },
        "inner_runner": {
            "state": {"fields": inner_state},
            "entry": "inner_process",
            "nodes": {"inner_process": {"fan_out": inner_fan_out}},
            "edges": [{"from": "inner_process", "to": "END"}],
        },
    }


def _nested_outer_case(outer_items: list[list[int]]) -> dict[str, Any]:
    return {
        "state": {
            "fields": {
                "outer_items": {"type": "list<list<int>>", "default": outer_items},
                "all_results": {"type": "list<list<int>>", "reducer": "append", "default": []},
            }
        },
        "entry": "outer_process",
        "nodes": {
            "outer_process": {
                "fan_out": {
                    "subgraph": "inner_runner",
                    "items_field": "outer_items",
                    "item_field": "inner_items",
                    "collect_field": "inner_results",
                    "target_field": "all_results",
                    "error_policy": "fail_fast",
                    "concurrent_mode": "concurrent",
                }
            }
        },
        "edges": [{"from": "outer_process", "to": "END"}],
        "initial_state": {},
    }


def _outer_in_flight(n: int) -> dict[str, Any]:
    return {
        "namespace": [],
        "fan_out_node_name": "outer_process",
        "instance_count": n,
        "enclosing_fan_out_lineage": [],
        "instances": [{"state": "in_flight"} for _ in range(n)],
    }


def _inner_entry(outer_idx: int, count: int, instances: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "namespace": ["outer_process"],
        "fan_out_node_name": "inner_process",
        "instance_count": count,
        "enclosing_fan_out_lineage": [
            {"namespace": [], "fan_out_node_name": "outer_process", "fan_out_index": outer_idx}
        ],
        "instances": instances,
    }


def _cause_chain(exc: BaseException | None) -> list[BaseException]:
    out: list[BaseException] = []
    seen = exc
    while seen is not None and seen not in out:
        out.append(seen)
        seen = seen.__cause__ or seen.__context__
    return out


async def _seed_and_resume(
    outer_case: Mapping[str, Any],
    subgraphs_block: Mapping[str, Any],
    seeded: Mapping[str, Any],
) -> tuple[Any, list[Any]]:
    """Build the nested graph, seed the mid-flight record, resume, and return
    (final_state, inner-leaf source values re-run on resume)."""
    top_level = {"subgraphs": subgraphs_block}
    leaf_values: list[Any] = []
    subgraphs = _build_subgraphs_for(outer_case, top_level, leaf_value_recorder=leaf_values)
    built = build_graph(outer_case, subgraphs=subgraphs, trace=[], leaf_value_recorder=leaf_values)
    cp = InMemoryCheckpointer()
    built.builder.with_checkpointer(cast(Checkpointer, cp))
    compiled = built.builder.compile()
    inv = "seeded-cov"
    await cp.save(inv, _build_seeded_record(seeded, inv))
    leaf_values.clear()
    final = await compiled.invoke(built.initial_state({}), resume_invocation=inv)
    return final, leaf_values


async def test_nested_resume_asymmetric_inner_counts_skip_per_lineage() -> None:
    # Fixture 076 uses symmetric inner counts (both length 3). Here outer-0 has
    # 2 inner items and outer-1 has 3: each lineage-qualified entry resolves its
    # OWN instance_count, and resume skips only that entry's completed instances.
    outer_items = [[1, 2], [4, 5, 6]]
    seeded = {
        "state": {"outer_items": outer_items, "all_results": []},
        "fan_out_progress": [
            _outer_in_flight(2),
            _inner_entry(0, 2, [{"state": "completed", "result": 1}, {"state": "not_started"}]),
            _inner_entry(
                1,
                3,
                [{"state": "completed", "result": 4}, {"state": "not_started"}, {"state": "not_started"}],
            ),
        ],
    }
    final, leaf_values = await _seed_and_resume(
        _nested_outer_case(outer_items), _nested_subgraphs(inner_error_policy="fail_fast"), seeded
    )
    assert [sorted(sub) for sub in final.all_results] == [[1, 2], [4, 5, 6]]
    # outer-0 skips inner-0 (=1), re-runs inner-1 (=2); outer-1 skips inner-0 (=4),
    # re-runs inner-1 (=5) + inner-2 (=6). No cross-skip across the asymmetric counts.
    assert sorted(leaf_values) == [2, 5, 6]


async def test_nested_resume_count_drift_on_matched_entry_raises() -> None:
    # A POSITIVELY matched inner entry whose saved instance_count no longer equals
    # the per-enclosing-instance re-resolved count MUST raise
    # checkpoint_record_invalid (count-drift re-resolved per lineage-qualified entry).
    outer_items = [[1, 2], [4, 5, 6]]  # outer-0 now resolves to 2 inner items
    seeded = {
        "state": {"outer_items": outer_items, "all_results": []},
        "fan_out_progress": [
            _outer_in_flight(2),
            # Stale: saved instance_count=3 but outer-0's items resolve to 2.
            _inner_entry(
                0,
                3,
                [
                    {"state": "completed", "result": 1},
                    {"state": "completed", "result": 2},
                    {"state": "not_started"},
                ],
            ),
            _inner_entry(
                1,
                3,
                [{"state": "completed", "result": 4}, {"state": "not_started"}, {"state": "not_started"}],
            ),
        ],
    }
    with pytest.raises((CheckpointError, RuntimeGraphError)) as excinfo:
        await _seed_and_resume(
            _nested_outer_case(outer_items), _nested_subgraphs(inner_error_policy="fail_fast"), seeded
        )
    # The inner-fan-out drift surfaces wrapped through the outer node; the
    # count-drift CheckpointRecordInvalid MUST be in the cause chain.
    assert any(isinstance(e, CheckpointRecordInvalid) for e in _cause_chain(excinfo.value)), (
        f"expected CheckpointRecordInvalid in the cause chain; got {excinfo.value!r}"
    )


async def test_nested_resume_error_entry_rolls_forward_per_lineage() -> None:
    # Collect-mode resume (§10.11.2) under a nested lineage: a completed instance
    # whose contribution is an ERROR (result_is_error=True) is skipped and rolled
    # forward under ITS OWN enclosing instance only, never crossing to a sibling.
    outer_items = [[10, 20], [30, 40]]
    seeded = {
        "state": {"outer_items": outer_items, "all_results": []},
        "fan_out_progress": [
            _outer_in_flight(2),
            # outer-0: nothing done -> both inner instances re-run (10, 20).
            _inner_entry(0, 2, [{"state": "not_started"}, {"state": "not_started"}]),
            # outer-1: inner-0 completed as an ERROR (value 30 -> errors bucket),
            # inner-1 re-runs (40).
            _inner_entry(
                1,
                2,
                [
                    {"state": "completed", "result": {"category": "node_exception"}, "result_is_error": True},
                    {"state": "not_started"},
                ],
            ),
        ],
    }
    final, leaf_values = await _seed_and_resume(
        _nested_outer_case(outer_items), _nested_subgraphs(inner_error_policy="collect"), seeded
    )
    # The errored inner-0 under outer-1 (value 30) is SKIPPED (rolled forward as an
    # error), never re-run, and never crosses into outer-0. outer-0 re-runs 10, 20;
    # outer-1 re-runs only its non-errored inner-1 (40).
    assert sorted(leaf_values) == [10, 20, 40]
    assert 30 not in leaf_values
    # outer-0 collects [10, 20]; outer-1 collects only the success value [40].
    assert [sorted(sub) for sub in final.all_results] == [[10, 20], [40]]
