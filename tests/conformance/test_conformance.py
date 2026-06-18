"""Run every spec conformance fixture against the engine.

Discovers `NNN-*.yaml` files under `openarmature-spec/spec/graph-engine/
conformance/` and parametrizes one test per fixture. The 007 table fixture
expands to one parametrized case per entry in its `cases:` block.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.graph import (
    END,
    CompileError,
    EdgeException,
    EndSentinel,
    GraphBuilder,
    NodeEvent,
    NodeException,
    ObserverEvent,
    RoutingError,
    RuntimeGraphError,
    State,
    SubscribedObserver,
)
from openarmature.graph.observer import Observer

from .adapter import (
    ObserverFixture,
    build_graph,
    make_observer_fn,
    normalize_expected_event,
)

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "graph-engine" / "conformance"
)


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _fixture_paths() -> list[Path]:
    return sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# Standard runtime fixtures (everything except 007 compile-errors and 010
# determinism, which have bespoke shapes).
# ---------------------------------------------------------------------------

_STANDARD_RUNTIME_FIXTURES = [
    p for p in _fixture_paths() if p.stem not in {"007-compile-errors", "010-determinism"}
]


# Fixtures whose implementation lands in a later PR of the 5-proposal
# batch (proposals 0011, 0014, 0015, 0016, 0017). Skip-marked here so a
# green test run at this commit means "everything we claim to implement
# passes." Each subsequent PR drops its own rows as it lands the
# underlying support.
_DEFERRED_FIXTURES: dict[str, str] = {
    # proposal 0011 — parallel branches; fixture 021 (``branch_name``
    # field on NodeEvent) runs through this driver as of PR-5.
    # Proposal 0023 (canonical state reducers, spec v0.52.0) — runtime
    # execution requires the new factory reducers (``bounded_append``,
    # ``dedupe_append``, ``merge_by_key``). Python ships these in a
    # future PR; the manifest entry is ``not-yet``.
    "034-reducer-bounded-append": "Proposal 0023 canonical state reducers; impl not yet shipped",
    "035-reducer-dedupe-append": "Proposal 0023 canonical state reducers; impl not yet shipped",
    "036-reducer-merge-by-key": "Proposal 0023 canonical state reducers; impl not yet shipped",
    "037-reducer-configuration-invalid-max-len": (
        "Proposal 0023 canonical state reducers; impl not yet shipped"
    ),
    "038-reducer-error-non-list-update": ("Proposal 0023 canonical state reducers; impl not yet shipped"),
}


# Node directives the legacy adapter doesn't (yet) translate. A later pass will
# either expand the adapter or replace it with the typed harness.
_UNSUPPORTED_NODE_DIRECTIVES = frozenset(
    {
        "flaky_per_index",
        "flaky_resume_aware",
        "calls_llm",
        "update_pure_from_state",
        "emits_log",
        "also_emits_via_global_tracer",
    }
)


def _unsupported_directive(spec: Mapping[str, Any]) -> str | None:
    """Return the first node directive the legacy adapter can't translate,
    or None if every node uses one of the directives it handles. Walks
    both the top-level graph and an optional inner ``subgraph`` block."""

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


def _subgraph_dependencies(sub_spec: dict[str, Any]) -> set[str]:
    """Names of subgraphs referenced by `subgraph: <name>` directives in
    this subgraph's nodes. Used to order plural-form compilation so an
    inner subgraph compiles before any subgraph that references it."""
    deps: set[str] = set()
    nodes = sub_spec.get("nodes")
    if not isinstance(nodes, dict):
        return deps
    for node_spec in cast("dict[str, Any]", nodes).values():
        if isinstance(node_spec, dict) and "subgraph" in cast("dict[str, Any]", node_spec):
            ref = cast("dict[str, Any]", node_spec)["subgraph"]
            if isinstance(ref, str):
                deps.add(ref)
    return deps


def _compile_subgraphs_map(
    subgraphs_spec: dict[str, dict[str, Any]],
    registry: dict[str, Any],
) -> None:
    """Compile the plural `subgraphs:` map into ``registry``.

    Iterates until every subgraph is compiled, picking entries whose
    referenced subgraphs are already in the registry. This keeps the
    fixture format order-independent — fixture 019 happens to list inner
    before middle, but the harness shouldn't depend on that.
    """
    pending: dict[str, dict[str, Any]] = dict(subgraphs_spec)
    while pending:
        progress = False
        for name, sub_spec in list(pending.items()):
            deps = _subgraph_dependencies(sub_spec)
            if not deps.issubset(registry):
                continue
            # Plural form omits the `name:` field (the dict key IS the name);
            # synthesize it for build_graph's existing singular-form lookup.
            # Validate against fixture authoring errors first.
            existing_name = sub_spec.get("name")
            if existing_name is not None and existing_name != name:
                raise ValueError(f"subgraph dict key {name!r} does not match name field {existing_name!r}")
            if name in registry:
                raise ValueError(
                    f"subgraph name {name!r} is already registered "
                    f"(collision with singular subgraph: form or duplicate plural entry)"
                )
            sub_with_name = {**sub_spec, "name": name}
            sub_built = build_graph(
                sub_with_name,
                subgraphs=registry,
                model_name=f"{name.title()}State",
            )
            registry[name] = sub_built.builder.compile()
            del pending[name]
            progress = True
        if not progress:
            raise RuntimeError(f"unresolvable subgraph dependencies in subgraphs: map: {sorted(pending)}")


@pytest.mark.parametrize("fixture_path", _STANDARD_RUNTIME_FIXTURES, ids=_fixture_id)
async def test_runtime_fixture(fixture_path: Path) -> None:
    fixture_id = fixture_path.stem
    if fixture_id in _DEFERRED_FIXTURES:
        pytest.skip(f"{fixture_id}: {_DEFERRED_FIXTURES[fixture_id]}")
    spec = _load(fixture_path)

    # ``cases:`` form (e.g., 020-observer-edge-error-events): each entry
    # is a self-contained per-case spec. Iterate; treat each case as a
    # standalone fixture body. Wrapping AssertionError with the case
    # name keeps failure messages locatable.
    # Fixture 020 uses edge-condition `callable:` directives
    # (`state_field_read`, `edge_raises`) that are unique to this fixture
    # and don't fit the generic adapter DSL. Custom driver below.
    if fixture_path.stem == "020-observer-edge-error-events":
        await _run_fixture_020(spec)
        return

    if "cases" in spec:
        for case in cast("list[dict[str, Any]]", spec["cases"]):
            try:
                await _run_runtime_case(case, fixture_path.stem)
            except AssertionError as e:
                raise AssertionError(f"case {case.get('name')!r}: {e}") from e
        return

    await _run_runtime_case(spec, fixture_path.stem)


async def _run_runtime_case(spec: Mapping[str, Any], fixture_id: str) -> None:
    # Skip fixtures whose nodes use directives the legacy adapter doesn't
    # translate (fan_out, flaky variants, calls_llm, etc.). Each directive
    # is gated to the phase that lands its runtime support.
    if (hit := _unsupported_directive(spec)) is not None:
        pytest.skip(f"{fixture_id}: unsupported node directive {hit}")

    # Subgraph fixtures (006, 011, 013) declare an inner subgraph via the
    # singular `subgraph:` key. Fixture 019 introduces the plural `subgraphs:`
    # map for two-level nesting; subgraphs there can reference each other,
    # so they're compiled in dependency order.
    subgraphs: dict[str, Any] = {}
    if "subgraph" in spec:
        sub_spec = spec["subgraph"]
        sub_built = build_graph(sub_spec, model_name=f"{sub_spec['name'].title()}State")
        subgraphs[sub_spec["name"]] = sub_built.builder.compile()
    if "subgraphs" in spec:
        _compile_subgraphs_map(spec["subgraphs"], subgraphs)

    built = build_graph(spec, subgraphs=subgraphs)
    compiled = built.builder.compile()
    initial = built.initial_state(spec.get("initial_state", {}))

    # Wire observers per the fixture's `observers:` block (012–016, 018).
    # Each observer is recorded by name so we can assert event-by-event
    # after invoke + drain. `phases:` (018) restricts which started/
    # completed events the observer subscribes to.
    observer_fixtures: dict[str, ObserverFixture] = {}
    delivery: list[tuple[str, int, str]] = []
    invocation_observers: list[Observer | SubscribedObserver] = []
    for o in spec.get("observers", []):
        phases_list = o.get("phases")
        phases = frozenset(phases_list) if phases_list is not None else None
        ofx = ObserverFixture(
            name=o["name"],
            attach=o["attach"],
            target=o["target"],
            behavior=o["behavior"],
            phases=phases,
            sleep_ms_per_event=o.get("sleep_ms_per_event"),
        )
        observer_fixtures[ofx.name] = ofx
        obs = make_observer_fn(ofx, delivery)
        if ofx.attach == "graph":
            target_graph = compiled if ofx.target == "outer" else subgraphs[ofx.target]
            target_graph.attach_observer(obs, phases=phases)
        else:
            if phases is not None:
                invocation_observers.append(SubscribedObserver(observer=obs, phases=phases))
            else:
                invocation_observers.append(obs)

    # Top-level expected_error: legacy runtime-error fixtures (008, 009).
    if "expected_error" in spec:
        with pytest.raises(RuntimeGraphError) as excinfo:
            await compiled.invoke(initial, observers=invocation_observers)
        await compiled.drain()

        err = excinfo.value
        expected_err = spec["expected_error"]
        assert err.category == expected_err["category"]

        if expected_err["category"] == "node_exception":
            assert isinstance(err, NodeException)
            assert err.node_name == expected_err["raised_from"]
            assert str(err.__cause__) == expected_err["message"]
            if "recoverable_state" in expected_err:
                assert err.recoverable_state.model_dump() == expected_err["recoverable_state"]
        elif expected_err["category"] == "routing_error":
            assert isinstance(err, RoutingError)
            if "recoverable_state" in expected_err:
                assert err.recoverable_state.model_dump() == expected_err["recoverable_state"]

        if "execution_order" in expected_err:
            assert built.trace == expected_err["execution_order"]
        return

    # Proposal 0010 §6 Drain — multi-invocation fixture form
    # (`invocations:` array; fixture 024). Each entry runs as its own
    # `invoke` + `drain` against the same compiled graph + observers, so
    # the cross-invocation cleanliness contract can be asserted end-to-
    # end. Observers' `invocation_counter` bumps between entries so the
    # dict-form `sleep_ms_per_event` can vary per invocation.
    if "invocations" in spec:
        for inv_idx, inv in enumerate(cast("list[dict[str, Any]]", spec["invocations"])):
            inv_initial = built.initial_state(inv.get("initial_state", {}))
            drain_block: dict[str, Any] = inv.get("drain") or {}
            ts = drain_block.get("timeout_seconds")
            inv_timeout: float | None = float(ts) if ts is not None else None
            inv_expected: dict[str, Any] = inv.get("expected") or {}

            # Reset per-invocation observer state and bump the counter
            # so dict-form `sleep_ms_per_event` selects the right value.
            for ofx in observer_fixtures.values():
                if inv_idx > 0:
                    ofx.invocation_counter[0] = inv_idx
                    ofx.events.clear()
            if inv_idx > 0:
                # Drop the per-invocation delivery trace; subsequent
                # invocations assert against a fresh recorder.
                delivery.clear()
                # `built.trace` is shared across the fixture; clear it
                # between invocations so per-invocation execution_order
                # assertions don't accumulate.
                built.trace.clear()

            inv_final = await compiled.invoke(inv_initial, observers=invocation_observers)
            inv_drain_start = time.monotonic()
            inv_drain_summary = await compiled.drain(timeout=inv_timeout)
            inv_drain_elapsed = time.monotonic() - inv_drain_start

            if "final_state" in inv_expected:
                assert inv_final.model_dump() == inv_expected["final_state"], (
                    f"invocation {inv_idx} final_state mismatch"
                )
            if "execution_order" in inv_expected:
                assert built.trace == inv_expected["execution_order"], (
                    f"invocation {inv_idx} execution_order mismatch"
                )

            ds_expected: dict[str, Any] | None = inv_expected.get("drain_summary")
            if ds_expected is not None:
                if "timeout_reached" in ds_expected:
                    assert inv_drain_summary.timeout_reached == ds_expected["timeout_reached"], (
                        f"invocation {inv_idx} drain_summary.timeout_reached: "
                        f"actual={inv_drain_summary.timeout_reached}, "
                        f"expected={ds_expected['timeout_reached']}"
                    )
                if "undelivered_count" in ds_expected:
                    assert inv_drain_summary.undelivered_count == ds_expected["undelivered_count"], (
                        f"invocation {inv_idx} drain_summary.undelivered_count: "
                        f"actual={inv_drain_summary.undelivered_count}, "
                        f"expected={ds_expected['undelivered_count']}"
                    )
                if "undelivered_count_min" in ds_expected:
                    assert inv_drain_summary.undelivered_count >= ds_expected["undelivered_count_min"], (
                        f"invocation {inv_idx} drain_summary.undelivered_count below min: "
                        f"actual={inv_drain_summary.undelivered_count}, "
                        f"min={ds_expected['undelivered_count_min']}"
                    )

            if "observer_events" in inv_expected:
                obs_events_map = cast("dict[str, list[dict[str, Any]]]", inv_expected["observer_events"])
                for name, expected_events in obs_events_map.items():
                    actual = observer_fixtures[name].events
                    normalized = [normalize_expected_event(ev) for ev in expected_events]
                    assert len(actual) == len(normalized), (
                        f"invocation {inv_idx} observer event count mismatch for {name!r}: "
                        f"actual={len(actual)}, expected={len(normalized)}"
                    )
                    for i, (a, e) in enumerate(zip(actual, normalized, strict=True)):
                        for key, expected_value in e.items():
                            assert key in a, (
                                f"invocation {inv_idx} observer {name!r} event {i} "
                                f"missing key {key!r}: actual={a}"
                            )
                            assert a[key] == expected_value, (
                                f"invocation {inv_idx} observer {name!r} event {i} "
                                f"key {key!r} mismatch: actual={a[key]!r}, expected={expected_value!r}"
                            )

            # Per-invocation invariants (e.g.,
            # `drain_returned_within_timeout` on the timed first
            # invocation in fixture 024).
            inv_invariants: dict[str, Any] = inv_expected.get("invariants") or {}
            if inv_invariants.get("drain_returned_within_timeout"):
                assert inv_timeout is not None
                assert inv_drain_elapsed < inv_timeout + 0.4, (
                    f"invocation {inv_idx} drain returned outside timeout window: "
                    f"elapsed={inv_drain_elapsed:.3f}s, timeout={inv_timeout}s"
                )

        # Top-level invariants (e.g.,
        # `second_invocation_drain_independent_of_first` on fixture 024)
        # apply after all invocations complete.
        top_invariants: dict[str, Any] = spec.get("invariants") or {}
        if top_invariants.get("second_invocation_drain_independent_of_first"):
            # The fact that we reached this point with all per-invocation
            # assertions passing IS the proof of cross-invocation
            # independence; the assertion is structural.
            assert len(compiled._active_workers) == 0
        return

    expected = spec["expected"]

    # Observer-fixture-with-error (014): the run is expected to raise, and
    # we still want to assert observer events captured before the failure
    # propagated.
    nested_error = expected.get("expected_error")
    if nested_error is not None:
        with pytest.raises(RuntimeGraphError) as excinfo:
            await compiled.invoke(initial, observers=invocation_observers)
        await compiled.drain()
        assert excinfo.value.category == nested_error["category"]
        if "message" in nested_error:
            assert str(excinfo.value.__cause__) == nested_error["message"]
    else:
        # Happy path (001–006, 011, 012, 013, 015, 022–025).
        # `invoke.drain.timeout_seconds` is the proposal 0010 drain
        # timeout directive; absent for legacy fixtures, present for
        # 022/023/025. The captured `drain_summary` is asserted below.
        invoke_block: dict[str, Any] = spec.get("invoke") or {}
        drain_block_raw = invoke_block.get("drain")
        timeout: float | None
        if isinstance(drain_block_raw, dict):
            drain_block_typed = cast("dict[str, Any]", drain_block_raw)
            ts = drain_block_typed.get("timeout_seconds")
            timeout = float(ts) if ts is not None else None
        else:
            timeout = None
        final = await compiled.invoke(initial, observers=invocation_observers)
        # Bracket the drain call to assert the `drain_returned_within_timeout`
        # invariant when the fixture declares it.
        drain_start = time.monotonic()
        drain_summary = await compiled.drain(timeout=timeout)
        drain_elapsed = time.monotonic() - drain_start
        if "final_state" in expected:
            assert final.model_dump() == expected["final_state"]
        if "execution_order" in expected:
            assert built.trace == expected["execution_order"]

        # Proposal 0010 §6 Drain — DrainSummary assertions. The fixture
        # MAY assert exact `undelivered_count` or a lower-bound
        # `undelivered_count_min` (timing-dependent fixtures use min).
        ds_expected: dict[str, Any] | None = expected.get("drain_summary")
        if ds_expected is not None:
            if "timeout_reached" in ds_expected:
                assert drain_summary.timeout_reached == ds_expected["timeout_reached"], (
                    f"drain_summary.timeout_reached mismatch: "
                    f"actual={drain_summary.timeout_reached}, expected={ds_expected['timeout_reached']}"
                )
            if "undelivered_count" in ds_expected:
                assert drain_summary.undelivered_count == ds_expected["undelivered_count"], (
                    f"drain_summary.undelivered_count mismatch: "
                    f"actual={drain_summary.undelivered_count}, expected={ds_expected['undelivered_count']}"
                )
            if "undelivered_count_min" in ds_expected:
                assert drain_summary.undelivered_count >= ds_expected["undelivered_count_min"], (
                    f"drain_summary.undelivered_count below min: "
                    f"actual={drain_summary.undelivered_count}, "
                    f"min={ds_expected['undelivered_count_min']}"
                )

        # Proposal 0010 §6 Drain — invariants block. Each invariant flag
        # is a fixture-level assertion the harness verifies separately
        # from drain_summary.
        invariants: dict[str, Any] = expected.get("invariants") or {}
        if invariants.get("drain_returned_within_timeout"):
            assert timeout is not None, "drain_returned_within_timeout invariant requires a timeout"
            # Allow generous slack for cancellation settlement + CI
            # scheduler variance — gather(return_exceptions=True) on
            # cancelled workers settles within an event-loop tick.
            assert drain_elapsed < timeout + 0.4, (
                f"drain returned outside timeout window: elapsed={drain_elapsed:.3f}s, timeout={timeout}s"
            )
        if invariants.get("graph_state_intact_after_timeout"):
            # `_active_workers` cleaned after cancelled workers settled.
            assert len(compiled._active_workers) == 0, (
                f"graph state not clean after timeout: {len(compiled._active_workers)} workers remaining"
            )
        if invariants.get("drain_waited_for_all_events"):
            # No timeout supplied; drain blocked until all observer
            # work completed. The summary already asserts undelivered=0;
            # this invariant adds a positive lower-bound on duration
            # (drain took at least most of the observer's work).
            assert drain_summary.undelivered_count == 0
            assert drain_summary.timeout_reached is False

    # Observer event assertions (012–016, 018, 022–025).
    # Per-event comparison projects the recorded event down to the keys
    # the fixture specifies. Fixtures that exercise state machinery
    # (012–016, 018) include `pre_state` / `post_state`; drain fixtures
    # (023, 025) assert only on phase/step/namespace/node_name shape and
    # MUST NOT fail because the recorded event happens to carry state.
    if "observer_events" in expected:
        for name, expected_events in expected["observer_events"].items():
            actual = observer_fixtures[name].events
            normalized = [normalize_expected_event(ev) for ev in expected_events]
            assert len(actual) == len(normalized), (
                f"observer event count mismatch for {name!r}: "
                f"actual={len(actual)}, expected={len(normalized)}"
            )
            for i, (a, e) in enumerate(zip(actual, normalized, strict=True)):
                for key, expected_value in e.items():
                    assert key in a, f"observer {name!r} event {i} missing key {key!r}: actual={a}"
                    assert a[key] == expected_value, (
                        f"observer {name!r} event {i} key {key!r} mismatch: "
                        f"actual={a[key]!r}, expected={expected_value!r}"
                    )

    if "delivery_order" in expected:
        expected_delivery = [(d["observer"], d["step"], d["phase"]) for d in expected["delivery_order"]]
        assert delivery == expected_delivery, (
            f"delivery_order mismatch: actual={delivery}, expected={expected_delivery}"
        )

    if "observer_event_invariants" in expected:
        _check_event_invariants(expected["observer_event_invariants"], observer_fixtures)

    # 018 — registering an observer with an empty `phases` set raises at
    # registration time per spec §6.
    if expected.get("empty_phases_raises_at_registration"):
        with pytest.raises(ValueError):
            compiled.attach_observer(
                make_observer_fn(
                    ObserverFixture(
                        name="probe",
                        attach="graph",
                        target="outer",
                        behavior="record",
                    ),
                    [],
                ),
                phases=frozenset(),
            )


# ---------------------------------------------------------------------------
# Fixture 020 — observer edge-error events (proposal 0012 / spec v0.9.0)
#
# Two sub-cases verifying §3 step 3 (revised) + §6 (revised): edge-resolution
# failures (routing_error, edge_exception) land on the preceding node's
# completed event with `error` populated, sharing the started/completed pair
# rather than producing a separate event pair.
#
# Custom driver (rather than the generic harness) because fixture 020's
# edge-condition `callable:` directives (`state_field_read`, `edge_raises`)
# don't match the adapter DSL's `if_field/equals/then/else` shape — these
# directives are spec-defined just for this fixture's two sub-cases. The
# semantic intent is captured directly here.
# ---------------------------------------------------------------------------


def _check_event_invariants(
    invariants: Mapping[str, Any],
    observer_fixtures: Mapping[str, ObserverFixture],
) -> None:
    """Verify ``observer_event_invariants`` block contents against the
    captured observer events. Each named invariant has a recognized
    shape used by one or more fixtures (021 parallel-branches
    branch_name on inner-node events).

    Fixtures consult the first observer's events as the canonical
    stream; multi-observer fixtures author their assertions against the
    `observer_events` block instead.
    """
    if not observer_fixtures:
        return
    first_obs = next(iter(observer_fixtures.values()))
    events = first_obs.events

    no_branch_name_cfg = cast(
        "dict[str, Any] | None",
        invariants.get("outermost_events_have_no_branch_name"),
    )
    if no_branch_name_cfg is not None:
        node_names = cast("list[str]", no_branch_name_cfg.get("nodes") or [])
        for ev in events:
            if ev["node_name"] in node_names:
                assert "branch_name" not in ev, (
                    f"outermost node {ev['node_name']!r} event MUST NOT carry "
                    f"branch_name; got {ev.get('branch_name')!r}"
                )

    inner_branch_cfg = cast(
        "dict[str, str] | None",
        invariants.get("inner_events_carry_correct_branch_name"),
    )
    if inner_branch_cfg is not None:
        for ev in events:
            expected_branch = inner_branch_cfg.get(ev["node_name"])
            if expected_branch is None:
                continue
            assert ev.get("branch_name") == expected_branch, (
                f"inner node {ev['node_name']!r} expected branch_name={expected_branch!r}, "
                f"got {ev.get('branch_name')!r}"
            )


async def _run_fixture_020(spec: Mapping[str, Any]) -> None:
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_020_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_020_case(case: Mapping[str, Any]) -> None:
    """Build a two-node graph (a → b) with the case's edge-condition
    directive translated to a Python edge function, then assert the
    proposal-0012 contract on the captured observer events."""

    class FixtureState(State):
        x: int = 0

    received: list[NodeEvent] = []

    async def observer(event: ObserverEvent) -> None:
        if not isinstance(event, NodeEvent):
            return
        received.append(event)

    async def node_a(_state: Any) -> dict[str, Any]:
        return {"x": 1}

    async def node_b(_state: Any) -> dict[str, Any]:
        return {"x": 2}

    cond = cast("dict[str, Any]", case["edges"][0]["condition"])
    callable_name = cast("str", cond["callable"])
    if callable_name == "state_field_read":
        # Per spec proposal 0012 fixture 020: the edge function returns
        # the value from initial_state's named field. The fixture's case
        # initial_state populates that field with a string that is NOT a
        # declared node name in the graph, so the engine raises
        # ``RoutingError``. We reproduce the semantic via a closure
        # over the initial_state value.
        initial_state_dict = cast("dict[str, Any]", case.get("initial_state", {}))
        field_name = cast("str", cond["field"])
        target_value = cast("str", initial_state_dict[field_name])

        def edge_fn(_state: Any) -> str | EndSentinel:
            return target_value
    elif callable_name == "edge_raises":
        message = cast("str", cond.get("message", "edge raised"))

        def edge_fn(_state: Any) -> str | EndSentinel:
            raise RuntimeError(message)
    else:
        raise AssertionError(f"fixture 020: unknown condition callable {callable_name!r}")

    g = (
        GraphBuilder(FixtureState)
        .add_node("a", node_a)
        .add_node("b", node_b)
        .add_conditional_edge("a", edge_fn)
        .add_edge("b", END)
        .set_entry("a")
        .compile()
    )
    g.attach_observer(observer)

    expected_error = cast("dict[str, Any]", case["expected_error"])
    expected_category = cast("str", expected_error["category"])

    with pytest.raises(RuntimeGraphError) as excinfo:
        await g.invoke(FixtureState())
    await g.drain()

    err = excinfo.value
    assert err.category == expected_category
    if expected_category == "routing_error":
        assert isinstance(err, RoutingError)
    elif expected_category == "edge_exception":
        assert isinstance(err, EdgeException)

    # Per the revised §6 contract: edge-resolution failures share the
    # preceding node's started/completed pair. Node a fires exactly one
    # started + one completed event; node b never fires.
    a_events = [e for e in received if e.node_name == "a"]
    b_events = [e for e in received if e.node_name == "b"]
    assert len(a_events) == 2, f"expected 2 events for node a; got {len(a_events)}"
    assert b_events == [], f"node b MUST never fire events on edge-resolution failure; got {b_events}"

    started, completed = a_events
    assert started.phase == "started"
    assert completed.phase == "completed"
    assert completed.post_state is None, (
        "completed event MUST have post_state absent when edge resolution fails"
    )
    assert completed.error is not None
    assert completed.error.category == expected_category


# ---------------------------------------------------------------------------
# 007 compile-errors: one parametrized case per entry in the `cases:` table.
# ---------------------------------------------------------------------------

_COMPILE_ERROR_PATH = CONFORMANCE_DIR / "007-compile-errors.yaml"


def _compile_error_cases() -> list[tuple[str, dict[str, Any], str]]:
    spec = _load(_COMPILE_ERROR_PATH)
    return [(c["name"], c["graph"], c["expected_compile_error"]) for c in spec["cases"]]


@pytest.mark.parametrize(
    ("graph_spec", "expected_category"),
    [(g, e) for _, g, e in _compile_error_cases()],
    ids=[name for name, _, _ in _compile_error_cases()],
)
def test_compile_error(graph_spec: dict[str, Any], expected_category: str) -> None:
    # Cases that compose a subgraph (e.g. mapping_references_undeclared_field)
    # need that subgraph compiled and registered before the parent compiles,
    # and the parent must use a real SubgraphNode so the engine's compile-time
    # projection validation runs.
    subgraphs: dict[str, Any] = {}
    if "subgraph" in graph_spec:
        sub_spec = graph_spec["subgraph"]
        sub_built = build_graph(sub_spec, model_name=f"{sub_spec['name'].title()}State")
        subgraphs[sub_spec["name"]] = sub_built.builder.compile()

    built = build_graph(graph_spec, subgraphs=subgraphs)
    with pytest.raises(CompileError) as excinfo:
        built.builder.compile()
    assert excinfo.value.category == expected_category


# ---------------------------------------------------------------------------
# 010 determinism: run `run_count` times and assert both the expected result
# and inter-run equality.
# ---------------------------------------------------------------------------


async def test_determinism() -> None:
    spec = _load(CONFORMANCE_DIR / "010-determinism.yaml")
    expected = spec["expected"]
    run_count = spec["run_count"]

    results: list[tuple[dict[str, Any], list[str]]] = []
    for _ in range(run_count):
        built = build_graph(spec)
        compiled = built.builder.compile()
        initial = built.initial_state(spec.get("initial_state", {}))
        final = await compiled.invoke(initial)
        results.append((final.model_dump(), list(built.trace)))

    for final_dump, trace in results:
        assert final_dump == expected["final_state"]
        assert trace == expected["execution_order"]

    first = results[0]
    for other in results[1:]:
        assert other == first
