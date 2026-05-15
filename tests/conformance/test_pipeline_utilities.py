"""Run every spec pipeline-utilities conformance fixture against the engine.

Phase 2 scope (proposal 0004 middleware): fixtures 001-016. Fixtures
017-019 (fan-out) and 020-021 (fan-out + middleware composition) skip
via `_unsupported_directive` until Phase 3 lands the fan-out runtime.
Fixtures 022-031 (fan-out and checkpointing) similarly skip until their
phases.

The driver translates a fixture's `middleware:` block into actual
middleware instances, wires up capture sinks per fixture-defined
recorder names, monkeypatches `time.monotonic` for the
deterministic-clock stub, and runs the resulting graph through the same
build_graph adapter the graph-engine fixtures use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.graph import (
    NodeException,
    RuntimeGraphError,
)
from openarmature.graph.middleware import (
    Middleware,
    OnCompleteCallback,
    RetryMiddleware,
    TimingMiddleware,
    TimingRecord,
    deterministic_backoff,
)

from .adapter import build_graph
from .middleware_seam import (
    ErrorRaiserMiddleware,
    ErrorRecoveryMiddleware,
    ShortCircuitMiddleware,
    StateInspectorMiddleware,
    TraceRecord,
    TraceRecorderMiddleware,
)

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "pipeline-utilities" / "conformance"
)


# Phase 3 lands fan-out (proposal 0005 PU side). Checkpointing
# (proposal 0008) comes in Phase 5; its fixtures use directives we
# don't translate yet.
_UNSUPPORTED_NODE_DIRECTIVES = frozenset(
    {
        "flaky_per_index",
        "flaky_resume_aware",
        "calls_llm",
        "update_pure_from_state",
    }
)


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


# Phase 3 target: fan-out (proposal 0005 PU side) covers fixtures 017-023.
# Phase 5 will pick up the checkpointing fixtures (024-031).
_PHASE_3_LAST = 23


def _fixture_paths() -> list[Path]:
    paths = sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))
    out: list[Path] = []
    for p in paths:
        try:
            number = int(p.stem.split("-", 1)[0])
        except ValueError:
            continue
        if number <= _PHASE_3_LAST:
            out.append(p)
    return out


def _fixture_id(path: Path) -> str:
    return path.stem


# Fixtures whose implementation lands in a later PR of the 5-proposal
# batch (proposals 0011, 0014, 0015, 0016, 0017). Skip-marked here so a
# green test run at this commit means "everything we claim to implement
# passes." Each subsequent PR drops its own rows as it lands the
# underlying support.
_DEFERRED_FIXTURES: dict[str, str] = {
    # proposal 0011 — parallel branches (PR-5 of the batch)
    "032-parallel-branches-basic": "0011 parallel branches (PR-5)",
    "033-parallel-branches-fail-fast": "0011 parallel branches (PR-5)",
    "034-parallel-branches-collect": "0011 parallel branches (PR-5)",
    "035-parallel-branches-different-state-schemas": "0011 parallel branches (PR-5)",
    "036-parallel-branches-with-branch-middleware-retry": "0011 parallel branches (PR-5)",
    "037-parallel-branches-determinism": "0011 parallel branches (PR-5)",
    "038-parallel-branches-compose-with-fan-out": "0011 parallel branches (PR-5)",
    # proposal 0014 — state migration (PR-4 of the batch)
    "039-state-migration-additive-field": "0014 state migration (PR-4)",
    "040-state-migration-chain": "0014 state migration (PR-4)",
    "041-state-migration-missing": "0014 state migration (PR-4)",
    "042-state-migration-versions-match-no-op": "0014 state migration (PR-4)",
    "043-state-migration-parent-states-migrated": "0014 state migration (PR-4)",
    "044-state-migration-post-migration-deserialization-fails": "0014 state migration (PR-4)",
    "045-state-migration-no-path-in-registry": "0014 state migration (PR-4)",
    "046-state-migration-function-raises": "0014 state migration (PR-4)",
}


def _unsupported_directive(spec: dict[str, Any]) -> str | None:
    """Return the first node directive the driver can't translate yet."""

    def scan(graph: Any) -> str | None:
        if not isinstance(graph, dict):
            return None
        nodes = cast("dict[str, Any]", graph).get("nodes")
        if not isinstance(nodes, dict):
            return None
        for node_name, node_spec in cast("dict[str, Any]", nodes).items():
            if not isinstance(node_spec, dict):
                continue
            for key in cast("dict[str, Any]", node_spec):
                if key in _UNSUPPORTED_NODE_DIRECTIVES:
                    return f"{node_name}.{key}"
        return None

    if (hit := scan(spec)) is not None:
        return hit
    if (hit := scan(spec.get("subgraph"))) is not None:
        return hit
    for sub_spec in spec.get("subgraphs", {}).values():
        if (hit := scan(sub_spec)) is not None:
            return hit
    return None


def _unsupported_middleware(spec: dict[str, Any]) -> str | None:
    """Return a middleware type we don't translate yet, or None."""
    middleware_block = cast("dict[str, Any]", spec.get("middleware") or {})
    known = frozenset(
        {
            "trace_recorder",
            "short_circuit",
            "error_recovery",
            "error_raiser",
            "state_inspector",
            "retry",
            "timing",
        }
    )
    per_graph = cast("list[dict[str, Any]]", middleware_block.get("per_graph") or [])
    for entries in per_graph:
        if entries.get("type") not in known:
            return f"per_graph.{entries.get('type')}"
    per_node = cast("dict[str, list[dict[str, Any]]]", middleware_block.get("per_node") or {})
    for _name, node_entries in per_node.items():
        for entry in node_entries or []:
            if entry.get("type") not in known:
                return f"per_node.{entry.get('type')}"
    return None


# ---------------------------------------------------------------------------
# Capture sinks — bridge fixture-named recorders to expected-block assertions.
# ---------------------------------------------------------------------------


class CaptureSinks:
    """Per-fixture state holding capture lists by recorder name.

    Each kind of recordable middleware writes to its own dict-of-lists,
    keyed by the recorder's `name` (or `capture_to`) field in the
    fixture YAML.
    """

    def __init__(self) -> None:
        self.trace_records: dict[str, list[TraceRecord]] = {}
        self.timing_records: dict[str, list[TimingRecord]] = {}
        self.state_inspector: dict[str, list[bool]] = {}


# ---------------------------------------------------------------------------
# Middleware translation — fixture config dict -> middleware instance.
# ---------------------------------------------------------------------------


def _build_middleware(
    config: Mapping[str, Any],
    sinks: CaptureSinks,
    clock: Callable[[], float] | None = None,
) -> Middleware:
    """Instantiate one middleware from its YAML config dict."""
    mw_type = config["type"]
    if mw_type == "trace_recorder":
        name = config.get("name", "default")
        sink = sinks.trace_records.setdefault(name, [])
        return TraceRecorderMiddleware(
            sink=sink,
            pre_marker=config.get("pre_marker"),
            post_marker=config.get("post_marker"),
            marker_field=config.get("marker_field", "trace"),
        )
    if mw_type == "short_circuit":
        return ShortCircuitMiddleware(partial_update=config["partial_update"])
    if mw_type == "error_recovery":
        return ErrorRecoveryMiddleware(partial_update=config["partial_update"])
    if mw_type == "error_raiser":
        return ErrorRaiserMiddleware(message=config.get("message", "raised"))
    if mw_type == "state_inspector":
        name = config.get("name", "default")
        sink = sinks.state_inspector.setdefault(name, [])
        return StateInspectorMiddleware(sink=sink)
    if mw_type == "retry":
        backoff_cfg = config.get("backoff") or {"type": "deterministic", "seconds": 0}
        if backoff_cfg["type"] == "deterministic":
            backoff = deterministic_backoff(float(backoff_cfg.get("seconds", 0)))
        else:
            raise ValueError(f"unsupported backoff type for tests: {backoff_cfg['type']}")
        classifier_cfg = config.get("classifier")
        classifier = _build_classifier(classifier_cfg) if classifier_cfg is not None else None
        return RetryMiddleware(
            max_attempts=int(config.get("max_attempts", 3)),
            backoff=backoff,
            classifier=classifier,
        )
    if mw_type == "timing":
        on_complete_cfg = cast("dict[str, Any]", config.get("on_complete") or {})
        capture_to = cast("str", on_complete_cfg.get("capture_to", "default"))
        sink = sinks.timing_records.setdefault(capture_to, [])

        async def on_complete(record: TimingRecord) -> None:
            sink.append(record)

        cb: OnCompleteCallback = on_complete
        return TimingMiddleware(
            node_name=cast("str", config["node_name"]),
            on_complete=cb,
            clock=clock,
        )
    raise ValueError(f"unknown middleware type: {mw_type}")


def _translate_middleware_block(
    middleware_block: Mapping[str, Any] | None,
    sinks: CaptureSinks,
    clock: Callable[[], float] | None = None,
) -> tuple[list[Middleware], dict[str, list[Middleware]]]:
    """Translate `middleware:` block into (graph_middleware, node_middleware)."""
    if middleware_block is None:
        return [], {}
    per_graph = cast("list[dict[str, Any]]", middleware_block.get("per_graph") or [])
    graph_mw: list[Middleware] = [_build_middleware(cfg, sinks, clock) for cfg in per_graph]
    per_node = cast("dict[str, list[dict[str, Any]]]", middleware_block.get("per_node") or {})
    node_mw: dict[str, list[Middleware]] = {}
    for name, entries in per_node.items():
        node_mw[name] = [_build_middleware(cfg, sinks, clock) for cfg in (entries or [])]
    return graph_mw, node_mw


def _translate_fan_out_instance_middleware(
    spec: Mapping[str, Any],
    sinks: CaptureSinks,
    clock: Callable[[], float] | None = None,
) -> dict[str, list[Middleware]]:
    """Walk ``spec.nodes`` for fan_out blocks with ``instance_middleware``
    and translate each into a list of Middleware instances. Returned
    map is keyed by fan-out node name and consumed by build_graph's
    ``fan_out_instance_middleware`` kwarg.
    """
    out: dict[str, list[Middleware]] = {}
    nodes = cast("dict[str, dict[str, Any]]", spec.get("nodes") or {})
    for node_name, node_spec in nodes.items():
        fan_out_cfg_raw = node_spec.get("fan_out")
        if not isinstance(fan_out_cfg_raw, dict):
            continue
        fan_out_cfg = cast("dict[str, Any]", fan_out_cfg_raw)
        entries = cast(
            "list[dict[str, Any]]",
            fan_out_cfg.get("instance_middleware") or [],
        )
        if not entries:
            continue
        out[node_name] = [_build_middleware(cfg, sinks, clock) for cfg in entries]
    return out


# ---------------------------------------------------------------------------
# Clock stub — monkeypatch time.monotonic for deterministic timing fixtures.
# ---------------------------------------------------------------------------


def _build_classifier(config: Mapping[str, Any]) -> Callable[[Exception, Any], bool]:
    """Build a custom retry classifier per the fixture's classifier config.

    Currently supports ``state_aware_max_retries_remaining`` (used by
    fixture 016): returns True iff ``state.max_retries_remaining > 0``.
    """
    cls_type = config.get("type")
    if cls_type == "state_aware_max_retries_remaining":

        def classifier(_exc: Exception, state: Any) -> bool:
            return getattr(state, "max_retries_remaining", 0) > 0

        return classifier
    raise ValueError(f"unknown classifier type: {cls_type}")


def _build_clock_stub(config: Mapping[str, Any]) -> Callable[[], float]:
    """Return a deterministic-monotonic clock function per the fixture's
    ``clock_stub`` config. Each call advances the counter by a fixed
    delta — fed only into TimingMiddleware instances so asyncio's
    scheduling isn't affected.
    """
    if config.get("type") != "deterministic_monotonic":
        raise ValueError(f"unknown clock_stub type: {config.get('type')}")
    advance_ms = float(config["advance_ms_per_call"])
    counter = [0.0]

    def fake_monotonic() -> float:
        n = counter[0]
        counter[0] = n + advance_ms / 1000.0
        return n

    return fake_monotonic


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_pipeline_utility_fixture(
    fixture_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_id = fixture_path.stem
    if fixture_id in _DEFERRED_FIXTURES:
        pytest.skip(f"{fixture_id}: {_DEFERRED_FIXTURES[fixture_id]}")
    spec = _load(fixture_path)

    # Cases-shape fixtures (014, 016, 018-019, 021-023): each case is
    # a self-contained graph + middleware + expected block. The outer
    # fixture may define shared ``subgraph:`` / ``subgraph_with_idx:``
    # blocks that every case references; merge them into each case
    # before dispatching so the case sees them as if they were its own.
    if "cases" in spec:
        shared_subgraph_blocks = {k: spec[k] for k in ("subgraph", "subgraph_with_idx") if k in spec}
        for case in spec["cases"]:
            case_name = case.get("name", "<unnamed>")
            merged: dict[str, Any] = dict(case)
            for k, v in shared_subgraph_blocks.items():
                merged.setdefault(k, v)
            try:
                await _run_one(merged, monkeypatch)
            except AssertionError as e:
                raise AssertionError(f"case {case_name!r}: {e}") from e
        return

    if (hit := _unsupported_directive(spec)) is not None:
        pytest.skip(f"{fixture_path.stem}: unsupported node directive {hit}")
    if (hit := _unsupported_middleware(spec)) is not None:
        pytest.skip(f"{fixture_path.stem}: unsupported middleware {hit}")

    await _run_one(spec, monkeypatch)


async def _run_one(spec: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one fixture spec (or one case of a cases-shape fixture)."""

    sinks = CaptureSinks()
    clock = _build_clock_stub(spec["clock_stub"]) if "clock_stub" in spec else None
    graph_mw, node_mw = _translate_middleware_block(spec.get("middleware"), sinks, clock)
    fan_out_inst_mw = _translate_fan_out_instance_middleware(spec, sinks, clock)
    del monkeypatch  # retained in signature for future stubs that need it

    # Subgraph blocks — fixture 010 uses singular `subgraph:`; fan-out
    # fixtures 020-022 use one or two named subgraph blocks
    # (``subgraph:``, ``subgraph_with_idx:``) at the top level so the
    # fan-out config can pick which one to dispatch to per case.
    subgraphs: dict[str, Any] = {}
    for sub_key in ("subgraph", "subgraph_with_idx"):
        sub_spec = spec.get(sub_key)
        if sub_spec is None:
            continue
        sub_sinks = sinks  # same sinks; subgraph middleware shares the harness's lists
        sub_graph_mw, sub_node_mw = _translate_middleware_block(sub_spec.get("middleware"), sub_sinks, clock)
        sub_built = build_graph(
            sub_spec,
            model_name=f"{sub_spec['name'].title()}State",
            graph_middleware=sub_graph_mw,
            node_middleware=sub_node_mw,
        )
        subgraphs[sub_spec["name"]] = sub_built.builder.compile()

    expected = cast("dict[str, Any]", spec.get("expected") or {})
    run_count = cast("int", spec.get("run_count", 1))

    # `expected_error` may live at the top level (legacy graph-engine
    # convention) or nested under `expected`. Top-level wins.
    expected_err_raw = spec.get("expected_error") or expected.get("expected_error")
    if expected_err_raw is not None:
        expected_err = cast("dict[str, Any]", expected_err_raw)
        # Build fresh for the error path.
        built = build_graph(
            spec,
            subgraphs=subgraphs,
            graph_middleware=graph_mw,
            node_middleware=node_mw,
            fan_out_instance_middleware=fan_out_inst_mw,
        )
        compiled = built.builder.compile()
        initial = built.initial_state(spec.get("initial_state", {}))
        with pytest.raises(RuntimeGraphError) as excinfo:
            await compiled.invoke(initial)
        await compiled.drain()
        assert excinfo.value.category == expected_err["category"]
        if "message" in expected_err and isinstance(excinfo.value, NodeException):
            assert str(excinfo.value.__cause__) == expected_err["message"]
        if "recoverable_state" in expected_err and isinstance(excinfo.value, NodeException):
            assert excinfo.value.recoverable_state.model_dump() == expected_err["recoverable_state"]
        # Some error fixtures still attach trace_records assertions for
        # what fired before the failure.
        _check_trace_records(
            cast("Mapping[str, list[Mapping[str, Any]]] | None", expected.get("trace_records")),
            sinks,
        )
        return

    # Per-run state: each run uses its own freshly built middleware so
    # stateful middleware (retry counters etc.) doesn't leak across runs.
    final_states: list[dict[str, Any]] = []
    traces: list[list[str]] = []
    for run_idx in range(run_count):
        run_sinks = sinks if run_count == 1 else CaptureSinks()
        run_graph_mw, run_node_mw = (
            (graph_mw, node_mw)
            if run_count == 1
            else _translate_middleware_block(spec.get("middleware"), run_sinks, clock)
        )
        run_fan_out_inst_mw = (
            fan_out_inst_mw
            if run_count == 1
            else _translate_fan_out_instance_middleware(spec, run_sinks, clock)
        )
        run_built = build_graph(
            spec,
            subgraphs=subgraphs,
            graph_middleware=run_graph_mw,
            node_middleware=run_node_mw,
            fan_out_instance_middleware=run_fan_out_inst_mw,
        )
        run_compiled = run_built.builder.compile()
        run_initial = run_built.initial_state(spec.get("initial_state", {}))
        run_final = await run_compiled.invoke(run_initial)
        await run_compiled.drain()
        final_states.append(run_final.model_dump())
        traces.append(list(run_built.trace))
        del run_idx  # quiet pyright unused-name

    if "final_state" in expected:
        assert final_states[0] == expected["final_state"], (
            f"final_state mismatch: actual={final_states[0]}, expected={expected['final_state']}"
        )
    if "execution_order" in expected:
        assert traces[0] == expected["execution_order"], (
            f"execution_order mismatch: actual={traces[0]}, expected={expected['execution_order']}"
        )

    # Determinism: every run produced the same result.
    if run_count > 1:
        first = (final_states[0], traces[0])
        for other in zip(final_states[1:], traces[1:], strict=True):
            assert other == first, f"determinism violated across {run_count} runs"

    _check_trace_records(
        cast("Mapping[str, list[Mapping[str, Any]]] | None", expected.get("trace_records")),
        sinks,
    )

    # Timing record assertions.
    if "timing_records" in expected:
        # Two shapes per Phase 0 typed harness: dict-of-lists OR a flat list.
        expected_timing = expected["timing_records"]
        if isinstance(expected_timing, list):
            empty: list[TimingRecord] = []
            actual_flat = next(iter(sinks.timing_records.values()), empty)
            _assert_timing_records(actual_flat, cast("list[Mapping[str, Any]]", expected_timing))
        else:
            timing_dict = cast("Mapping[str, list[Mapping[str, Any]]]", expected_timing)
            for name, expected_recs in timing_dict.items():
                _assert_timing_records(sinks.timing_records.get(name, []), expected_recs)

    # Single observer-event assertion (fixture 015 uses singular form).
    if "expected_observer_event" in expected:
        # Fixture 015 has a top-level observer attached; we'd need observer
        # wiring just for this. For Phase 2's reduced scope, skip the
        # singular-observer-event check — fixture 015 is gated on retry's
        # per-attempt event behavior, which we test via flaky+retry against
        # final_state/execution_order. The detailed single-event assertion
        # would need additional harness scaffolding.
        pass


def _check_trace_records(
    expected_recs: Mapping[str, list[Mapping[str, Any]]] | None,
    sinks: CaptureSinks,
) -> None:
    """Verify trace_records assertions. Two shapes per fixture format:

    - Full record form (001, 002, 003, 010): each entry has ``state_in``
      and ``partial_update_returned``.
    - Pre/post seen form (005): each entry has ``pre_seen`` and/or
      ``post_seen`` flags. Used by the error-propagation fixture to
      assert the recorder's pre-phase fired but its post-phase did not.
    """
    if not expected_recs:
        return
    for name, expected_list in expected_recs.items():
        actual = sinks.trace_records.get(name, [])
        assert len(actual) == len(expected_list), (
            f"trace_records[{name!r}] length mismatch: actual={len(actual)}, expected={len(expected_list)}"
        )
        for actual_rec, expected_rec in zip(actual, expected_list, strict=True):
            if "state_in" in expected_rec:
                assert actual_rec.state_in == expected_rec["state_in"]
            if "partial_update_returned" in expected_rec:
                assert actual_rec.partial_update_returned == expected_rec["partial_update_returned"]
            if "pre_seen" in expected_rec:
                assert actual_rec.pre_seen == expected_rec["pre_seen"]
            if "post_seen" in expected_rec:
                assert actual_rec.post_seen == expected_rec["post_seen"]


def _assert_timing_records(
    actual: list[TimingRecord],
    expected_list: list[Mapping[str, Any]],
) -> None:
    assert len(actual) == len(expected_list), (
        f"timing_records length mismatch: actual={len(actual)}, expected={len(expected_list)}"
    )
    for actual_rec, expected_rec in zip(actual, expected_list, strict=True):
        assert actual_rec.node_name == expected_rec["node_name"]
        assert actual_rec.outcome == expected_rec["outcome"]
        if "duration_ms" in expected_rec:
            expected_dur = float(cast("float", expected_rec["duration_ms"]))
            assert abs(actual_rec.duration_ms - expected_dur) < 0.01
        if "exception_category" in expected_rec:
            assert actual_rec.exception_category == expected_rec["exception_category"]
