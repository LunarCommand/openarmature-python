"""Run every spec conformance fixture against the engine.

Discovers `NNN-*.yaml` files under `openarmature-spec/spec/graph-engine/
conformance/` and parametrizes one test per fixture. The 007 table fixture
expands to one parametrized case per entry in its `cases:` block.
"""

from __future__ import annotations

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
    NodeException,
    RoutingError,
    RuntimeGraphError,
    State,
    SubscribedObserver,
)
from openarmature.graph.events import NodeEvent
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


# Node directives the legacy adapter doesn't (yet) translate. Phase 1+ will
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
        # Happy path (001–006, 011, 012, 013, 015).
        final = await compiled.invoke(initial, observers=invocation_observers)
        await compiled.drain()
        if "final_state" in expected:
            assert final.model_dump() == expected["final_state"]
        if "execution_order" in expected:
            assert built.trace == expected["execution_order"]

    # Observer event assertions (012–016, 018).
    if "observer_events" in expected:
        for name, expected_events in expected["observer_events"].items():
            actual = observer_fixtures[name].events
            normalized = [normalize_expected_event(ev) for ev in expected_events]
            assert actual == normalized, (
                f"observer events mismatch for {name!r}: actual={actual}, expected={normalized}"
            )

    if "delivery_order" in expected:
        expected_delivery = [(d["observer"], d["step"], d["phase"]) for d in expected["delivery_order"]]
        assert delivery == expected_delivery, (
            f"delivery_order mismatch: actual={delivery}, expected={expected_delivery}"
        )

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

    async def observer(event: NodeEvent) -> None:
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
