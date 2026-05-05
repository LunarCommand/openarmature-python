"""Run every spec conformance fixture against the engine.

Discovers `NNN-*.yaml` files under `openarmature-spec/spec/graph-engine/
conformance/` and parametrizes one test per fixture. The 007 table fixture
expands to one parametrized case per entry in its `cases:` block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.graph import (
    CompileError,
    NodeException,
    RoutingError,
    RuntimeGraphError,
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


# Node directives the legacy adapter doesn't (yet) translate. Phase 1+ will
# either expand the adapter or replace it with the typed harness.
_UNSUPPORTED_NODE_DIRECTIVES = frozenset(
    {
        "fan_out",
        "flaky",
        "flaky_by_index",
        "flaky_per_index",
        "flaky_instance_only",
        "flaky_resume_aware",
        "calls_llm",
        "update_pure",
        "update_pure_from_state",
        "update_from_field",
        "emits_log",
        "also_emits_via_global_tracer",
    }
)


def _unsupported_directive(spec: dict[str, Any]) -> str | None:
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
    return None


@pytest.mark.parametrize("fixture_path", _STANDARD_RUNTIME_FIXTURES, ids=_fixture_id)
async def test_runtime_fixture(fixture_path: Path) -> None:
    spec = _load(fixture_path)

    # Skip fixtures whose nodes use directives the legacy adapter doesn't
    # translate (fan_out, flaky variants, calls_llm, etc.). Each directive
    # is gated to the phase that lands its runtime support.
    if (hit := _unsupported_directive(spec)) is not None:
        pytest.skip(f"{fixture_path.stem}: unsupported node directive {hit}")

    # Subgraph fixtures (006, 011, 013) declare an inner subgraph that the
    # outer graph references by name.
    subgraphs: dict[str, Any] = {}
    if "subgraph" in spec:
        sub_spec = spec["subgraph"]
        sub_built = build_graph(sub_spec, model_name=f"{sub_spec['name'].title()}State")
        subgraphs[sub_spec["name"]] = sub_built.builder.compile()

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
