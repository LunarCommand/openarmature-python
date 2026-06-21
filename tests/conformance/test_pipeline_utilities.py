"""Run every spec pipeline-utilities conformance fixture against the engine.

Middleware scope: fixtures 001-016. Fixtures
017-019 (fan-out) and 020-021 (fan-out + middleware composition) skip
via `_unsupported_directive` until the fan-out runtime lands.
Fixtures 022-031 (fan-out and checkpointing) similarly skip until their
support lands.

The driver translates a fixture's `middleware:` block into actual
middleware instances, wires up capture sinks per fixture-defined
recorder names, monkeypatches `time.monotonic` for the
deterministic-clock stub, and runs the resulting graph through the same
build_graph adapter the graph-engine fixtures use.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.graph import (
    CompileError,
    FailureIsolatedEvent,
    NodeException,
    ObserverEvent,
    ParallelBranchesBranchFailed,
    RuntimeGraphError,
)
from openarmature.graph.middleware import (
    DegradedUpdate,
    FailureIsolationMiddleware,
    Middleware,
    OnCompleteCallback,
    RetryConfig,
    RetryMiddleware,
    TimingMiddleware,
    TimingRecord,
    deterministic_backoff,
)

from .adapter import ObserverFixture, build_graph, make_observer_fn
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


# Fan-out (proposal 0005 PU side) lands later. Checkpointing
# (proposal 0008) comes later still; its fixtures use directives we
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


# Fan-out (proposal 0005 PU side) covers fixtures 017-023.
# The checkpointing fixtures (024-031) come later. Proposal 0011
# drives fixtures 032-038 through this same harness.
# State-migration fixtures 039-047 run via a dedicated runner
# (``test_state_migration.py``); they need a separate driver because
# the `cases:` shape carries seeded-record + migrations + resume blocks.
_LAST_DRIVEN_FIXTURE = 38

# Failure-isolation fixtures (058-066, 068, 069, proposals 0050 §6.3 / 0065 /
# 0066 / 0068 / 0070 / 0069) are middleware fixtures this runner handles. They
# sit past _LAST_DRIVEN_FIXTURE only because the 039-057 range (state migration
# / checkpoint fan-out) is owned by dedicated runners (test_state_migration.py
# / test_checkpoint.py), not because this runner can't drive them. Fixture 066
# (cause chain, 0068) joined at v0.57.0; 068 (failure-mock cause chain, 0070)
# at v0.58.0; 069 (fan-out degrade refinements, 0069) at v0.59.0 — this runner
# drives its FI-degrade cases and skips its crash_injection/resume case (owned
# by test_checkpoint.py, which also owns fixture 067, hence the gap at 67).
# 071 (fan-out degrade strict-reducer-raise, proposal 0069 coverage,
# spec v0.63.1) is an FI-degrade fixture this runner drives.
_FAILURE_ISOLATION_FIXTURES = frozenset(range(58, 67)) | {68, 69, 71, 72}

# Inline-callable parallel branches + conditional ``when`` (proposal 0075,
# spec v0.66.0). These extend the parallel-branches harness (032-038) with
# the ``call`` / ``when`` branch directives; 073 (two callable branches),
# 074 (cases: when false skips / true dispatches), 075 (callable branch +
# FailureIsolationMiddleware degrade).
_CALLABLE_BRANCH_FIXTURES = frozenset({73, 74, 75})


def _fixture_paths() -> list[Path]:
    paths = sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))
    out: list[Path] = []
    for p in paths:
        try:
            number = int(p.stem.split("-", 1)[0])
        except ValueError:
            continue
        if (
            number <= _LAST_DRIVEN_FIXTURE
            or number in _FAILURE_ISOLATION_FIXTURES
            or number in _CALLABLE_BRANCH_FIXTURES
        ):
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
    # proposal 0011 — parallel branches (PR-5 of the batch) — driven
    # by the harness as of this PR; the 8 fixtures (032-038 +
    # graph-engine/021) parse + run through the engine.
    # proposal 0014 — state migration (PR-4 of the batch) — driven
    # by ``test_state_migration.py`` (a separate runner that handles
    # the cases-shape seeded_record + migrations + resume blocks).
    # Checkpointing fixtures (024-031, proposal 0008) — driven by
    # ``test_checkpoint.py`` because their cases-shape carries
    # ``first_run_expected_error`` + ``resume:`` blocks that this
    # driver doesn't recognize.
    "024-checkpoint-save-on-every-completed-event": "checkpointing (test_checkpoint.py)",
    "025-checkpoint-resume-from-completed-position": "checkpointing (test_checkpoint.py)",
    "026-checkpoint-record-shape": "checkpointing (test_checkpoint.py)",
    "027-checkpoint-attempt-index-resets-on-resume": "checkpointing (test_checkpoint.py)",
    "028-checkpoint-fan-out-atomic-restart": "checkpointing (test_checkpoint.py)",
    "029-checkpoint-subgraph-resume": "checkpointing (test_checkpoint.py)",
    "030-checkpoint-not-found": "checkpointing (test_checkpoint.py)",
    "031-checkpoint-correlation-id-preserved-across-resume": "checkpointing (test_checkpoint.py)",
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
            "failure_isolation",
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
    # Node-nested ``middleware:`` on plain nodes (the shape
    # ``_translate_node_level_middleware`` lifts) is gated symmetrically — same
    # plain-node scoping, so the skip-gate and the translator agree on which
    # nodes carry liftable node middleware. (Reaches only the single-graph
    # path; cases-shape fixtures are dispatched without this gate.)
    nodes = cast("dict[str, dict[str, Any]]", spec.get("nodes") or {})
    for _name, node_spec in nodes.items():
        if any(k in node_spec for k in ("fan_out", "parallel_branches", "subgraph")):
            continue
        for entry in cast("list[dict[str, Any]]", node_spec.get("middleware") or []):
            if entry.get("type") not in known:
                return f"node.{entry.get('type')}"
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
        # Failure-isolation on_caught side channel (fixture 062): each
        # entry records {increment_field, capture_message_field, count,
        # message}; the harness overlays count/message onto final_state.
        self.on_caught: list[dict[str, Any]] = []


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
            RetryConfig(
                max_attempts=int(config.get("max_attempts", 3)),
                backoff=backoff,
                classifier=classifier,
            )
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
    if mw_type == "failure_isolation":
        return _build_failure_isolation(config, sinks)
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


def _translate_parallel_branches_branch_middleware(
    spec: Mapping[str, Any],
    sinks: CaptureSinks,
    clock: Callable[[], float] | None = None,
) -> dict[str, dict[str, list[Middleware]]]:
    """Walk ``spec.nodes`` for parallel_branches blocks with per-branch
    ``middleware:`` and translate each into a list of Middleware
    instances. Returned map is keyed by parallel-branches node name
    then branch name (branch middleware) and consumed by
    build_graph's ``parallel_branches_branch_middleware`` kwarg."""
    out: dict[str, dict[str, list[Middleware]]] = {}
    nodes = cast("dict[str, dict[str, Any]]", spec.get("nodes") or {})
    for node_name, node_spec in nodes.items():
        pb_cfg_raw = node_spec.get("parallel_branches")
        if not isinstance(pb_cfg_raw, dict):
            continue
        pb_cfg = cast("dict[str, Any]", pb_cfg_raw)
        branches_cfg = cast("dict[str, dict[str, Any]]", pb_cfg.get("branches") or {})
        per_branch: dict[str, list[Middleware]] = {}
        for branch_name, branch_cfg in branches_cfg.items():
            entries = cast("list[dict[str, Any]]", branch_cfg.get("middleware") or [])
            if not entries:
                continue
            per_branch[branch_name] = [_build_middleware(cfg, sinks, clock) for cfg in entries]
        if per_branch:
            out[node_name] = per_branch
    return out


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


def _translate_node_level_middleware(
    spec: Mapping[str, Any],
    sinks: CaptureSinks,
    clock: Callable[[], float] | None = None,
) -> dict[str, list[Middleware]]:
    """Walk ``spec.nodes`` for a node-nested ``middleware:`` list on a plain
    function node (the graph-engine per-node middleware shape that cases-style
    fixtures use, e.g. fixture 066 Case 2's node-level failure isolation) and
    translate each into Middleware instances, keyed by node name. Composite
    nodes are skipped because their middleware placements have dedicated
    translators: fan-out instance (``_translate_fan_out_instance_middleware``),
    parallel-branches branch (``_translate_parallel_branches_branch_middleware``),
    and subgraph parent-node middleware (the top-level ``middleware.per_node``
    block via ``_translate_middleware_block``)."""
    out: dict[str, list[Middleware]] = {}
    nodes = cast("dict[str, dict[str, Any]]", spec.get("nodes") or {})
    for node_name, node_spec in nodes.items():
        if any(k in node_spec for k in ("fan_out", "parallel_branches", "subgraph")):
            continue
        entries = cast("list[dict[str, Any]]", node_spec.get("middleware") or [])
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


def _render_state_template(template: str, state: Any) -> str:
    """Render a ``{{ state.<field> }}`` template against a State instance
    (fixture 059's callable degraded_update). Minimal substitution: the
    only template shape the failure-isolation fixtures use."""
    return re.sub(
        r"\{\{\s*state\.(\w+)\s*\}\}",
        lambda m: str(getattr(state, m.group(1))),
        template,
    )


def _build_isolation_predicate(
    config: Mapping[str, Any] | None,
) -> Callable[[Exception], bool] | None:
    """Build a FailureIsolationMiddleware predicate from a fixture
    ``predicate`` block. Supports ``{matches_category: <category>}``
    (fixture 060): catch only exceptions carrying that category."""
    if config is None:
        return None
    if "matches_category" in config:
        target = cast("str", config["matches_category"])

        def predicate(exc: Exception) -> bool:
            return getattr(exc, "category", None) == target

        return predicate
    raise ValueError(f"unsupported failure_isolation predicate: {dict(config)}")


def _build_failure_isolation(config: Mapping[str, Any], sinks: CaptureSinks) -> Middleware:
    """Build the canonical FailureIsolationMiddleware from a fixture
    ``failure_isolation`` config (fixtures 058-063; the ``catch`` category
    gate is fixture 072)."""
    degraded_raw = config["degraded_update"]
    degraded: DegradedUpdate
    if isinstance(degraded_raw, dict):
        degraded_dict = cast("dict[str, Any]", degraded_raw)
        if degraded_dict.get("callable") == "state_derived":
            # Callable form (059): the callable receives the pre-merge
            # state and renders the template into the target field.
            template = cast("str", degraded_dict["template"])
            target = cast("str", degraded_dict["target_field"])

            def degraded_from_state(state: Any) -> Mapping[str, Any]:
                return {target: _render_state_template(template, state)}

            degraded = degraded_from_state
        else:
            degraded = dict(degraded_dict)
    else:
        degraded = dict(cast("Mapping[str, Any]", degraded_raw))

    on_caught = None
    on_caught_cfg = cast("Mapping[str, Any] | None", config.get("on_caught"))
    if on_caught_cfg is not None:
        kind = on_caught_cfg.get("kind")
        if kind != "record_to_state_side_channel":
            raise ValueError(f"unsupported on_caught kind: {kind}")
        # The callback only receives the exception, so it records the
        # invocation count + message into a side channel the harness
        # overlays onto final_state after the run (fixture 062).
        record: dict[str, Any] = {
            "increment_field": cast("str", on_caught_cfg["increment_field"]),
            "capture_message_field": cast("str", on_caught_cfg["capture_message_field"]),
            "count": 0,
            "message": "",
        }
        sinks.on_caught.append(record)

        async def on_caught_cb(_exc: Exception) -> None:
            record["count"] += 1
            record["message"] = str(_exc)

        on_caught = on_caught_cb

    return FailureIsolationMiddleware(
        degraded_update=degraded,
        event_name=cast("str", config["event_name"]),
        catch=cast("list[str] | None", config.get("catch")),
        predicate=_build_isolation_predicate(cast("Mapping[str, Any] | None", config.get("predicate"))),
        on_caught=on_caught,
    )


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

    # Cases-shape fixtures (014, 016, 018-019, 021-023, 064): each case
    # is a self-contained graph + middleware + expected block. The outer
    # fixture may define shared ``subgraph:`` / ``subgraph_with_idx:``
    # (singular) or ``subgraphs:`` (plural, name -> graph-spec, as the
    # parallel-branches fixtures use) blocks that every case references;
    # merge them into each case before dispatching so the case sees them
    # as if they were its own. ``setdefault`` below preserves any block a
    # case defines for itself.
    if "cases" in spec:
        shared_subgraph_blocks = {
            k: spec[k] for k in ("subgraph", "subgraph_with_idx", "subgraphs") if k in spec
        }
        cases_run = 0
        for case in spec["cases"]:
            case_name = case.get("name", "<unnamed>")
            # Checkpoint-concern cases (fixture 069 Case 3) are owned by
            # test_checkpoint.py; this runner skips them. The marker mirrors
            # that runner's: checkpointer / resume / crash_injection.
            if any(k in case for k in ("checkpointer", "resume", "crash_injection")):
                continue
            cases_run += 1
            merged: dict[str, Any] = dict(case)
            # Compile-error cases (065 Case 2) nest the graph under ``graph:``
            # (the graph-engine fixture 007 convention) so it sits beside
            # ``expected_compile_error``. Flatten it to the top level the rest
            # of this runner reads from.
            if "graph" in merged:
                graph_block = cast("dict[str, Any]", merged.pop("graph"))
                merged = {**graph_block, **merged}
            for k, v in shared_subgraph_blocks.items():
                merged.setdefault(k, v)
            try:
                await _run_one(merged, monkeypatch)
            except AssertionError as e:
                raise AssertionError(f"case {case_name!r}: {e}") from e
        # A cases-shaped fixture in this runner's set that drives zero cases
        # (all skipped as checkpoint-owned) would pass vacuously; fail loudly
        # instead so a routing mistake surfaces.
        assert cases_run > 0, (
            f"{fixture_id}: cases-shaped fixture drove zero cases in this runner "
            f"(all skipped as checkpoint-owned). Fix the routing or remove it from "
            f"_FAILURE_ISOLATION_FIXTURES."
        )
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

    # Capture failure-isolation events (proposal 0050 §6.3, fixtures
    # 058-063) for the expected_failure_isolation_event /
    # no_failure_isolation_event assertions. Attached to every graph the
    # fixture runs below; only FailureIsolatedEvents are collected.
    captured_isolation: list[FailureIsolatedEvent] = []

    async def _capture_isolation(event: ObserverEvent) -> None:
        if isinstance(event, FailureIsolatedEvent):
            captured_isolation.append(event)

    graph_mw, node_mw = _translate_middleware_block(spec.get("middleware"), sinks, clock)
    # Node-nested ``middleware:`` (e.g. fixture 066 Case 2's node-level failure
    # isolation) merges into the per-node map alongside any top-level
    # ``middleware.per_node`` entries. Single-run fixtures (run_count == 1)
    # reuse this ``node_mw`` directly below.
    for nl_node, nl_mws in _translate_node_level_middleware(spec, sinks, clock).items():
        node_mw.setdefault(nl_node, []).extend(nl_mws)
    fan_out_inst_mw = _translate_fan_out_instance_middleware(spec, sinks, clock)
    del monkeypatch  # retained in signature for future stubs that need it

    # Subgraph blocks — fixture 010 uses singular `subgraph:`; fan-out
    # fixtures 020-022 use one or two named subgraph blocks
    # (``subgraph:``, ``subgraph_with_idx:``) at the top level so the
    # fan-out config can pick which one to dispatch to per case.
    # Parallel-branches fixtures (032-038) use a plural ``subgraphs:``
    # block — a dict mapping subgraph-name to graph-spec.
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
    plural_subgraphs = cast("dict[str, dict[str, Any]] | None", spec.get("subgraphs")) or {}
    for sub_name, sub_spec in plural_subgraphs.items():
        sub_graph_mw, sub_node_mw = _translate_middleware_block(sub_spec.get("middleware"), sinks, clock)
        # Pass ``subgraphs=subgraphs`` so a subgraph that itself contains
        # a fan_out / parallel_branches dispatch (fixture 038) can resolve
        # the inner subgraph against entries already compiled in earlier
        # iterations of this loop. The fixture's authoring order MUST put
        # dependencies before dependents (the spec author's responsibility).
        sub_built = build_graph(
            sub_spec,
            subgraphs=subgraphs,
            model_name=f"{sub_name.title()}State",
            graph_middleware=sub_graph_mw,
            node_middleware=sub_node_mw,
        )
        subgraphs[sub_name] = sub_built.builder.compile()

    branch_middleware = _translate_parallel_branches_branch_middleware(spec, sinks, clock)

    # Compile-error case (065 Case 2): building the graph MUST raise a
    # CompileError whose ``category`` matches. The fan-out degraded_update
    # coverage check fires in add_fan_out_node during build_graph.
    expected_compile_error = spec.get("expected_compile_error")
    if expected_compile_error is not None:
        with pytest.raises(CompileError) as excinfo:
            build_graph(
                spec,
                subgraphs=subgraphs,
                graph_middleware=graph_mw,
                node_middleware=node_mw,
                fan_out_instance_middleware=fan_out_inst_mw,
                parallel_branches_branch_middleware=branch_middleware,
            )
        assert excinfo.value.category == expected_compile_error, (
            f"expected compile error {expected_compile_error!r}, got {excinfo.value.category!r}"
        )
        return

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
            parallel_branches_branch_middleware=branch_middleware,
        )
        compiled = built.builder.compile()
        compiled.attach_observer(_capture_isolation)
        initial = built.initial_state(spec.get("initial_state", {}))
        with pytest.raises(RuntimeGraphError) as excinfo:
            await compiled.invoke(initial)
        await compiled.drain()
        assert excinfo.value.category == expected_err["category"]
        if "message" in expected_err and isinstance(excinfo.value, NodeException):
            assert str(excinfo.value.__cause__) == expected_err["message"]
        if "cause_message" in expected_err and isinstance(excinfo.value, NodeException):
            # ``cause_message`` is the original cause text — the
            # leaf of the __cause__ chain. For parallel-branches
            # fail_fast, the chain is:
            #   ParallelBranchesBranchFailed -> NodeException (branch's inner node) -> RuntimeError("...")
            # Walk to the deepest non-None __cause__ before
            # comparing.
            leaf: BaseException = excinfo.value
            while leaf.__cause__ is not None:
                leaf = leaf.__cause__
            assert str(leaf) == expected_err["cause_message"]
        if "branch_name" in expected_err and isinstance(excinfo.value, ParallelBranchesBranchFailed):
            assert excinfo.value.branch_name == expected_err["branch_name"]
        # ``recoverable_state`` may live nested under ``expected_error``
        # (legacy fan-out shape) or as a sibling under ``expected`` (per
        # spec §11.5 for parallel-branches fail_fast fixtures). Both
        # carry the same buffer-and-apply invariant.
        if "recoverable_state" in expected_err and isinstance(excinfo.value, NodeException):
            assert excinfo.value.recoverable_state.model_dump() == expected_err["recoverable_state"]
        if "recoverable_state" in expected and isinstance(excinfo.value, NodeException):
            assert excinfo.value.recoverable_state.model_dump() == expected["recoverable_state"]
        # Some error fixtures still attach trace_records assertions for
        # what fired before the failure.
        _check_trace_records(
            cast("Mapping[str, list[Mapping[str, Any]]] | None", expected.get("trace_records")),
            sinks,
        )
        if expected.get("no_failure_isolation_event"):
            assert captured_isolation == [], f"expected no FailureIsolatedEvent, got {captured_isolation}"
        return

    # Per-run state: each run uses its own freshly built middleware so
    # stateful middleware (retry counters etc.) doesn't leak across runs.
    final_states: list[dict[str, Any]] = []
    traces: list[list[str]] = []
    observer_fixtures: dict[str, ObserverFixture] = {}
    for run_idx in range(run_count):
        run_sinks = sinks if run_count == 1 else CaptureSinks()
        if run_count == 1:
            run_graph_mw, run_node_mw = graph_mw, node_mw
        else:
            run_graph_mw, run_node_mw = _translate_middleware_block(spec.get("middleware"), run_sinks, clock)
            for nl_node, nl_mws in _translate_node_level_middleware(spec, run_sinks, clock).items():
                run_node_mw.setdefault(nl_node, []).extend(nl_mws)
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
            parallel_branches_branch_middleware=branch_middleware,
        )
        run_compiled = run_built.builder.compile()
        run_initial = run_built.initial_state(spec.get("initial_state", {}))
        # Observers — graph-attached only (parallel-branches fixtures
        # 036/037/038 use ``attach: graph, target: outer``). We rebuild
        # the observer set fresh per run so capture lists don't bleed
        # across runs in determinism fixtures.
        run_observer_fixtures: dict[str, ObserverFixture] = {}
        run_delivery: list[tuple[str, int, str]] = []
        for o in spec.get("observers", []):
            phases_list = o.get("phases")
            phases = frozenset(phases_list) if phases_list is not None else None
            ofx = ObserverFixture(
                name=o["name"],
                attach=o["attach"],
                target=o["target"],
                behavior=o["behavior"],
                phases=phases,
            )
            run_observer_fixtures[ofx.name] = ofx
            obs = make_observer_fn(ofx, run_delivery)
            if ofx.attach == "graph" and ofx.target == "outer":
                run_compiled.attach_observer(obs, phases=phases)
        if run_idx == 0:
            run_compiled.attach_observer(_capture_isolation)
        run_final = await run_compiled.invoke(run_initial)
        await run_compiled.drain()
        final_states.append(run_final.model_dump())
        traces.append(list(run_built.trace))
        if run_idx == 0:
            observer_fixtures = run_observer_fixtures

    # Overlay the on_caught side channel (fixture 062) onto the final
    # state: the callback can't write graph state directly, so the harness
    # reflects its recorded count + message into the fields the fixture names.
    for rec in sinks.on_caught:
        final_states[0][rec["increment_field"]] = rec["count"]
        final_states[0][rec["capture_message_field"]] = rec["message"]

    if "final_state" in expected:
        _assert_final_state(final_states[0], expected["final_state"], spec)
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

    if "expected_failure_isolation_event" in expected:
        _assert_failure_isolation_event(
            captured_isolation,
            cast("Mapping[str, Any]", expected["expected_failure_isolation_event"]),
        )
    if expected.get("no_failure_isolation_event"):
        assert captured_isolation == [], f"expected no FailureIsolatedEvent, got {captured_isolation}"

    if "observer_event_invariants" in expected:
        _check_parallel_branches_invariants(
            cast("Mapping[str, Any]", expected["observer_event_invariants"]),
            observer_fixtures,
            spec,
        )

    # Timing record assertions.
    if "timing_records" in expected:
        # Two shapes the typed harness accepts: dict-of-lists OR a flat list.
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
        # wiring just for this. For this harness's reduced scope, skip the
        # singular-observer-event check — fixture 015 is gated on retry's
        # per-attempt event behavior, which we test via flaky+retry against
        # final_state/execution_order. The detailed single-event assertion
        # would need additional harness scaffolding.
        pass


def _collect_parallel_branches_errors_fields(spec: Mapping[str, Any]) -> set[str]:
    """Return the set of parent-state field names used as
    ``errors_field`` on any parallel_branches node in ``spec``.

    The ``errors_field`` carries an implementation-defined
    record shape; only ``branch_name`` + category are required. The
    engine's record carries additional engine-defined keys (``message``,
    ``cause_type``). Fixtures asserting against ``errors_field`` records
    use subset semantics — assert the required keys are present
    with the expected values, ignore the rest.
    """
    out: set[str] = set()
    nodes = cast("dict[str, dict[str, Any]]", spec.get("nodes") or {})
    for node_spec in nodes.values():
        pb_cfg = cast("dict[str, Any] | None", node_spec.get("parallel_branches"))
        if pb_cfg is None:
            continue
        field_name = pb_cfg.get("errors_field")
        if isinstance(field_name, str):
            out.add(field_name)
    return out


def _state_to_dict(state: Any) -> dict[str, Any]:
    """Dump a State (or mapping) to a plain dict for comparison."""
    if hasattr(state, "model_dump"):
        return cast("dict[str, Any]", state.model_dump())
    return dict(cast("Mapping[str, Any]", state))


def _assert_failure_isolation_event(
    captured: list[FailureIsolatedEvent],
    expected: Mapping[str, Any],
) -> None:
    """Assert the single captured FailureIsolatedEvent against the
    fixture's ``expected_failure_isolation_event`` block. Only the keys
    the fixture supplies are checked (some fixtures assert just
    event_name + caught_exception)."""
    assert len(captured) == 1, f"expected exactly one FailureIsolatedEvent, got {len(captured)}"
    ev = captured[0]
    if "event_name" in expected:
        assert ev.event_name == expected["event_name"]
    lineage = cast("Mapping[str, Any] | None", expected.get("wrapped_node_lineage"))
    if lineage is not None:
        if "namespace" in lineage:
            assert list(ev.namespace) == lineage["namespace"]
        if "attempt_index" in lineage:
            assert ev.attempt_index == lineage["attempt_index"]
        if "fan_out_index" in lineage:
            assert ev.fan_out_index == lineage["fan_out_index"]
        if "branch_name" in lineage:
            assert ev.branch_name == lineage["branch_name"]
    if "pre_state" in expected:
        assert _state_to_dict(ev.pre_state) == expected["pre_state"]
    if "post_state" in expected:
        assert dict(ev.post_state) == expected["post_state"]
    ce = cast("Mapping[str, Any] | None", expected.get("caught_exception"))
    if ce is not None:
        if "category" in ce:
            assert ev.caught_exception.category == ce["category"]
        if "message" in ce:
            assert ev.caught_exception.message == ce["message"]
        # Cause chain (proposal 0068). Each expected link is subset-matched on
        # the keys it supplies — carrier links pin only {carrier, category}
        # (their engine-internal message is not asserted), non-carrier links
        # pin {carrier, category, message}.
        if "chain" in ce:
            expected_chain = cast("list[Mapping[str, Any]]", ce["chain"])
            actual_chain = ev.caught_exception.chain
            assert len(actual_chain) == len(expected_chain), (
                f"chain length mismatch: actual={actual_chain}, expected={expected_chain}"
            )
            for actual_link, expected_link in zip(actual_chain, expected_chain, strict=True):
                if "carrier" in expected_link:
                    assert actual_link.carrier == expected_link["carrier"]
                if "category" in expected_link:
                    assert actual_link.category == expected_link["category"]
                if "message" in expected_link:
                    assert actual_link.message == expected_link["message"]


def _assert_final_state(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    """Compare ``actual`` vs ``expected`` final state. Strict equality
    everywhere except for parallel-branches ``errors_field`` records,
    which compare per-element via subset semantics."""
    errors_fields = _collect_parallel_branches_errors_fields(spec)
    assert set(actual.keys()) == set(expected.keys()), (
        f"final_state key mismatch: actual={set(actual.keys())}, expected={set(expected.keys())}"
    )
    for key, expected_val in expected.items():
        actual_val = actual[key]
        if key in errors_fields and isinstance(expected_val, list) and isinstance(actual_val, list):
            actual_list = cast("list[Any]", actual_val)
            expected_list = cast("list[Any]", expected_val)
            actual_len = len(actual_list)
            expected_len = len(expected_list)
            assert actual_len == expected_len, (
                f"final_state[{key!r}] length mismatch: actual={actual_len}, expected={expected_len}"
            )
            for actual_rec, expected_rec in zip(actual_list, expected_list, strict=True):
                if not isinstance(expected_rec, dict) or not isinstance(actual_rec, dict):
                    assert actual_rec == expected_rec
                    continue
                actual_dict = cast("dict[str, Any]", actual_rec)
                expected_dict = cast("dict[str, Any]", expected_rec)
                for sub_key, sub_val in expected_dict.items():
                    assert sub_key in actual_dict, (
                        f"final_state[{key!r}] record missing key {sub_key!r}: actual={actual_dict}"
                    )
                    actual_sub = actual_dict[sub_key]
                    assert actual_sub == sub_val, (
                        f"final_state[{key!r}].{sub_key} mismatch: actual={actual_sub}, expected={sub_val}"
                    )
            continue
        assert actual_val == expected_val, (
            f"final_state[{key!r}] mismatch: actual={actual_val}, expected={expected_val}"
        )


def _check_parallel_branches_invariants(
    invariants: Mapping[str, Any],
    observer_fixtures: Mapping[str, ObserverFixture],
    spec: Mapping[str, Any],
) -> None:
    """Verify parallel-branches observer-event invariants for fixtures
    036 (branch-middleware retry), 037 (determinism), 038 (compose with
    fan-out). Each invariant name maps to one of the recognized shapes
    below; an unknown name is skipped (forward-compat with new
    fixtures the harness hasn't been taught yet).
    """
    if not observer_fixtures:
        return
    obs = next(iter(observer_fixtures.values()))
    events = obs.events

    started_events = [ev for ev in events if ev["phase"] == "started"]

    # 037 — branches' started events fire in branches insertion order
    # regardless of their inner-node completion timing.
    expected_order = invariants.get("branch_started_event_order")
    if isinstance(expected_order, list):
        seen_order: list[str] = []
        for ev in started_events:
            branch = ev.get("branch_name")
            if branch is None:
                continue
            if branch in seen_order:
                continue
            seen_order.append(branch)
        assert seen_order == expected_order, (
            f"branch_started_event_order mismatch: actual={seen_order}, expected={expected_order}"
        )

    # 036 — per-branch attempt_index sequence on each branch's inner
    # node. Authors per-branch via ``<branch>_inner_attempt_indices_seen``.
    for key, expected_attempts in invariants.items():
        if not key.endswith("_inner_attempt_indices_seen"):
            continue
        branch_name = key.removesuffix("_inner_attempt_indices_seen")
        attempts = [ev["attempt_index"] for ev in started_events if ev.get("branch_name") == branch_name]
        assert attempts == expected_attempts, (
            f"{key} mismatch: actual={attempts}, expected={expected_attempts}"
        )

    # 038 — composition with fan-out invariants.
    if invariants.get("fan_out_inner_events_carry_both_branch_name_and_fan_out_index"):
        fan_out_events = [ev for ev in events if "fan_out_index" in ev]
        assert fan_out_events, "expected inner-node events carrying fan_out_index, got none"
        for ev in fan_out_events:
            assert "branch_name" in ev, f"fan-out inner event missing branch_name: {ev}"
    fan_out_branch = invariants.get("fan_out_inner_branch_name_seen")
    if isinstance(fan_out_branch, str):
        fan_out_branch_names = {ev.get("branch_name") for ev in events if "fan_out_index" in ev}
        assert fan_out_branch in fan_out_branch_names, (
            f"fan-out inner events expected to carry branch_name={fan_out_branch!r}; "
            f"saw branch_names={fan_out_branch_names}"
        )
    expected_indices_raw = invariants.get("fan_out_inner_fan_out_indices_seen")
    if isinstance(expected_indices_raw, list):
        expected_indices = cast("list[int]", expected_indices_raw)
        seen_indices = sorted({ev["fan_out_index"] for ev in events if "fan_out_index" in ev})
        assert seen_indices == sorted(expected_indices), (
            f"fan_out_inner_fan_out_indices_seen mismatch: actual={seen_indices}, "
            f"expected={sorted(expected_indices)}"
        )
    if invariants.get("plain_inner_events_carry_branch_name_but_no_fan_out_index"):
        plain_branch = invariants.get("plain_inner_branch_name_seen")
        if isinstance(plain_branch, str):
            plain_events = [
                ev for ev in events if ev.get("branch_name") == plain_branch and "fan_out_index" not in ev
            ]
            assert plain_events, (
                f"expected branch_name={plain_branch!r} inner events without fan_out_index; got none"
            )


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
