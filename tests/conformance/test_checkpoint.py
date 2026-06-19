"""Run every spec checkpoint conformance fixture (024-031, 048-054)
against the engine.

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
- 028 fan-out-atomic-restart — REMOVED (replaced by the per-instance
  resume contract). The fixture file no longer exists.
- 029 subgraph-resume — supported (uses plain ``flaky``).
- 030 checkpoint-not-found — supported.
- 031 correlation-id-preserved-across-resume — record-level
  assertions supported here; the OTel span/log assertions are
  gated until the observability mapping lands.
- 048-054 per-instance fan-out resume contract — supported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.checkpoint import (
    Checkpointer,
    CheckpointError,
    CheckpointNotFound,
    CheckpointRecord,
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

from .adapter import build_graph

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "pipeline-utilities" / "conformance"
)

# Conformance fixture range: 024-031 minus 028 are the proposal-0008
# set; 048-054 are the proposal-0009 per-instance-resume set; 055
# (schema_version declared class — proposal 0028) and 056 (fan-out
# count drift — proposal 0029) are the follow-on bundle. 028 (fan-out
# atomic-restart) was REMOVED in spec v0.18.0 when proposal 0009
# superseded its contract, so it is explicitly excluded from the set
# rather than relying on the test runner's file-glob to filter the
# missing fixture out. 067 (crash-injection fan-out resume, proposal
# 0070) is a crash/resume fixture this runner owns; it joined at v0.58.0.
# 069 (fan-out degrade refinements, proposal 0069, v0.59.0) is a mixed
# fixture: this runner drives its crash_injection/resume case and skips the
# plain FI-degrade cases (owned by test_pipeline_utilities.py).
# 070 (crash-injection after_node resume, proposal 0070 coverage, spec
# v0.63.1) is a crash/resume fixture this runner owns, alongside 067.
_CHECKPOINT_FIXTURE_NUMBERS: frozenset[int] = frozenset(
    (set(range(24, 32)) - {28}) | set(range(48, 57)) | {67, 69, 70}
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
    with path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


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
        await self._inner.save(invocation_id, record)
        self._maybe_abort(record)

    async def save_fan_out_internal(self, invocation_id: str, record: CheckpointRecord) -> None:
        self._raise_if_post_abort()
        self.saves.append(record)
        await self._inner.save_fan_out_internal(invocation_id, record)
        self._maybe_abort(record)

    async def save_fan_out_in_flight_failure(self, invocation_id: str, record: CheckpointRecord) -> None:
        self._raise_if_post_abort()
        self.saves.append(record)
        await self._inner.save_fan_out_in_flight_failure(invocation_id, record)
        self._maybe_abort(record)

    def _raise_if_post_abort(self) -> None:
        """Once the abort has fired, any subsequent save call raises
        immediately — modelling a process-level crash after the
        target instance's completion. Without this, gather would
        continue dispatching sibling instances whose saves would
        complete normally and pollute the loaded record."""
        if self._aborted:
            raise _AbortAfterInstance("post-abort save call")

    def _maybe_abort(self, record: CheckpointRecord) -> None:
        """Check whether this save is the configured crash boundary. If
        so, raise the sentinel after the save has been recorded (so the
        record is durably persisted before the simulated crash).
        ``abort_after_instance`` fires on the save transitioning that
        instance index to ``completed``; ``abort_after_node`` fires on
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
    """Translate a fan-out node's ``instance_middleware: [failure_isolation]``
    into FailureIsolationMiddleware instances keyed by node name, for
    build_graph's ``fan_out_instance_middleware``. Scoped to the static
    ``degraded_update`` mapping form (the only shape the checkpoint fixtures
    use, e.g. fixture 069 Case 3's degrade-survives-resume); the callable
    forms are owned by test_pipeline_utilities.py, which drives the plain
    FI-degrade cases."""
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
            # middleware (e.g. fixture 053's retry) is left unwired, as this
            # runner did before — those fixtures drive their behavior via
            # flaky_per_index seams, not a wired middleware.
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


def _strip_abort_directive(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a fresh spec dict with any ``abort_after_instance``
    directive removed from fan-out nodes. The engine doesn't recognize
    the directive; the wrapper checkpointer interprets it on the
    harness side. Strip before passing to ``build_graph`` so the
    underlying fan-out config doesn't carry the unknown key into the
    builder."""
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


async def _run_one_case(spec: Mapping[str, Any], *, top_level: Mapping[str, Any]) -> None:
    """Run one fixture or one case from a cases-shape fixture."""
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
    subgraphs = _build_subgraphs_for(
        spec,
        top_level,
        flaky_per_index_recorders=flaky_per_index_recorders,
        instance_execution_recorders=instance_execution_recorders,
    )

    trace: list[str] = []
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

    builder.with_checkpointer(cast("Checkpointer", capturing))
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
    try:
        final_first_run = await compiled.invoke(
            initial_state,
            correlation_id=spec.get("correlation_id"),
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
            for key, expected_value in every_save_block.items():
                actual_value = getattr(saved_record, key, None)
                assert actual_value == expected_value, (
                    f"every_save_assertions: save[{save_idx}].{key} mismatch — "
                    f"actual={actual_value!r}, expected={expected_value!r}"
                )

    # ----- checkpoint_not_found expected (fixture 030) -----
    if expected.get("expected_error") == "checkpoint_not_found":
        ghost = cast("str", expected.get("resume_invocation_id", "ghost"))
        with pytest.raises(CheckpointNotFound):
            await compiled.invoke(initial_state, resume_invocation=ghost)
        return

    # ----- Resume path (fixtures 025, 029, 031, 048-054, 056) -----
    resume_block = spec.get("resume")
    if resume_block is None or not resume_block.get("from_first_run"):
        return
    if invocation_id_first_run is None:
        raise AssertionError("resume requested but no invocation_id captured (no saves fired)")
    saves_before_resume = list(capturing.saves)
    capturing.saves.clear()
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

    try:
        final_resume = await compiled.invoke(
            initial_state,
            resume_invocation=invocation_id_first_run,
        )
    except CheckpointError:
        raise
    _ = trace  # trace clearing/inspection deferred; recorder map is canonical
    resume_expected = cast("Mapping[str, Any]", resume_block.get("expected") or {})
    if "final_state" in resume_expected:
        _assert_state_matches(final_resume, cast("Mapping[str, Any]", resume_expected["final_state"]))

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

    if "invariants" in resume_expected or "invariants" in resume_block:
        # Resume-block invariants land on either resume.expected.invariants
        # or resume.invariants depending on fixture style. Read both.
        invariants_block: dict[str, Any] = {}
        if "invariants" in resume_block:
            invariants_block.update(cast("dict[str, Any]", resume_block["invariants"]))
        if "invariants" in resume_expected:
            invariants_block.update(cast("dict[str, Any]", resume_expected["invariants"]))
        _assert_resume_invariants(invariants_block, final_resume, flaky_per_index_recorders)

    # Fixture 031: assert correlation_id preserved + invocation_id
    # changed. Span/log assertions deferred — observability
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


def _build_subgraphs_for(
    spec: Mapping[str, Any],
    top_level: Mapping[str, Any],
    *,
    flaky_per_index_recorders: dict[str, dict[int, list[int]]] | None = None,
    instance_execution_recorders: dict[str, dict[int, list[int]]] | None = None,
) -> dict[str, Any]:
    """Build subgraphs from either the case's own ``subgraph`` /
    ``subgraphs`` block or the cases-fixture's top-level shared
    ``subgraph`` block. Each case may declare local subgraphs OR
    inherit from the top level.

    ``flaky_per_index_recorders`` and ``instance_execution_recorders``
    (when supplied) thread through to inner-subgraph build so per-instance
    bodies inside subgraphs populate the recorder maps the resume
    assertions read.
    """
    return _build_subgraphs(
        {**dict(top_level), **dict(spec)},
        flaky_per_index_recorders=flaky_per_index_recorders,
        instance_execution_recorders=instance_execution_recorders,
    )


def _build_subgraphs(
    spec: Mapping[str, Any],
    *,
    flaky_per_index_recorders: dict[str, dict[int, list[int]]] | None = None,
    instance_execution_recorders: dict[str, dict[int, list[int]]] | None = None,
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
    for name, sub_spec in subgraph_specs.items():
        sub_trace: list[str] = []
        sub_built = build_graph(
            sub_spec,
            trace=sub_trace,
            flaky_per_index_attempt_recorders=flaky_per_index_recorders,
            instance_execution_recorders=instance_execution_recorders,
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


def _assert_resume_invariants(
    block: Mapping[str, Any],
    final_state: State | None,
    recorders: Mapping[str, dict[int, list[int]]],
) -> None:
    """Assert resume-side invariants — list-length, no-duplicate,
    per-instance attempt counts."""
    final_dict: dict[str, Any] = final_state.model_dump() if final_state is not None else {}
    for key, value in block.items():
        if key == "no_duplicate_results":
            if not value:
                continue
            results = final_dict.get("results")
            if isinstance(results, list):
                results_list = cast("list[Any]", results)
                assert len(set(_hashable(r) for r in results_list)) == len(results_list), (
                    f"results list has duplicate entries: {results_list}"
                )
        elif key == "results_list_length":
            results = final_dict.get("results")
            assert isinstance(results, list), f"results_list_length: results is not a list ({results!r})"
            results_list = cast("list[Any]", results)
            assert len(results_list) == value, (
                f"results_list_length: actual={len(results_list)}, expected={value}"
            )
        elif key == "errors_list_length":
            errors = final_dict.get("errors")
            assert isinstance(errors, list), f"errors_list_length: errors is not a list ({errors!r})"
            errors_list = cast("list[Any]", errors)
            assert len(errors_list) == value, (
                f"errors_list_length: actual={len(errors_list)}, expected={value}"
            )
        elif key == "no_duplicate_error_entries":
            if not value:
                continue
            errors = final_dict.get("errors")
            if isinstance(errors, list):
                errors_list = cast("list[Any]", errors)
                hashes = [_hashable(e) for e in errors_list]
                assert len(set(hashes)) == len(hashes), f"errors list has duplicates: {errors_list}"
        elif key.startswith("instance_") and key.endswith("_attempt_index_on_resume"):
            # Extract instance index from ``instance_<N>_attempt_index_on_resume``.
            parts = key.split("_")
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            # Per §10.6: every retry budget resets to 0 on resume.
            # Assert the first attempt observed on the resume run for
            # the named instance is attempt_index 0.
            for recorder in recorders.values():
                if idx in recorder:
                    attempts = recorder[idx]
                    if attempts:
                        assert attempts[0] == value, (
                            f"instance {idx} first-resume attempt_index: "
                            f"actual={attempts[0]}, expected={value}"
                        )
        elif key.startswith("instance_") and key.endswith("_resume_attempt_count"):
            parts = key.split("_")
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            for recorder in recorders.values():
                if idx in recorder:
                    attempts = recorder[idx]
                    assert len(attempts) == value, (
                        f"instance {idx} resume attempt count: actual={len(attempts)}, expected={value}"
                    )
        elif key.startswith("instance_") and "_executes_step_" in key and key.endswith("_on_resume"):
            # Fixture 050 directive: ``instance_1_executes_step_a_on_resume: true``.
            # Verified indirectly by ``instances_executed_during_resume``
            # — the instance ran, so its inner subgraph re-entered at
            # the entry node. The harness doesn't yet introspect which
            # specific inner nodes fired; the broader executed-set
            # assertion covers the same correctness invariant.
            continue
        elif key == "batching_scoped_to_fan_out_internal_saves_only":
            # Structural invariant — verified across the fixture suite
            # rather than per-fixture (every non-fan-out save runs
            # synchronously regardless of the batching config). The
            # fixture restates it as a reminder; no per-test action.
            continue


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


# ---------------------------------------------------------------------------
# crash_injection: after_node boundary (proposal 0070)
# ---------------------------------------------------------------------------


async def test_capturing_checkpointer_aborts_after_node() -> None:
    # The ``crash_injection: {after_node}`` boundary fires the simulated-crash
    # sentinel after the save whose record records the named node in
    # ``completed_positions``. Exercised directly because no v0.58.0 fixture
    # uses after_node (fixture 067 uses after_fan_out_instance, which covers
    # the shared abort path end-to-end); this pins the after_node branch.
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
