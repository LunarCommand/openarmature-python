# Spec: conformance-adapter section 5.4 *Subgraph declaration placement* and
# section 4.2 *The `graph:` container* (proposal 0123, spec v0.114.0).

"""The conformance-adapter capability's own fixtures.

Unlike the other five capabilities, these test the FIXTURE FORMAT rather than a
runtime behaviour: section 5.4's rules bind an adapter, and every assertion turns
on which declaration this harness resolved while building the graph, a choice
made before the engine runs.

Reported as adapter conformance, distinct from the five runtime capabilities.
Spec ruled the directory in scope on coord thread
`proposal-0120-0123-adapter-obligations` msg 02.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from ._deferral import skip_if_deferred
from .adapter import build_graph
from .harness.subgraph_placement import graph_spec_for, resolve_subgraphs

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "conformance-adapter" / "conformance"
)

_DEFERRED_FIXTURES: dict[str, str] = {}


def _fixture_paths() -> list[Path]:
    return sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(path.read_text()))


async def _run_case(case: dict[str, Any], document: dict[str, Any]) -> None:
    # `graph_spec_for` picks the container or the case body; `resolve_subgraphs`
    # decides which declaration of each name governs. Those two calls are the
    # whole of what this fixture tests.
    spec = graph_spec_for(case)
    declared = resolve_subgraphs(document, case)

    subgraphs = {name: build_graph(body).builder.compile() for name, body in declared.items()}
    built = build_graph(spec, subgraphs=subgraphs)
    compiled = built.builder.compile()
    final = await compiled.invoke(built.initial_state(case.get("initial_state", {})))

    expected = cast("dict[str, Any]", case.get("expected") or {})
    final_state = cast("dict[str, Any]", expected.get("final_state") or {})
    assert final_state, f"case {case.get('name')!r} asserts no final_state"
    for field, want in final_state.items():
        got = getattr(final, field)
        assert got == want, f"case {case.get('name')!r}: {field}={got!r}, expected {want!r}"


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_conformance_adapter_fixture(fixture_path: Path) -> None:
    fixture_id = _fixture_id(fixture_path)
    skip_if_deferred(fixture_id, _DEFERRED_FIXTURES)
    document = _load(fixture_path)
    cases = cast("list[dict[str, Any]]", document.get("cases") or [])
    assert cases, f"{fixture_id} declares no cases"
    for case in cases:
        try:
            await _run_case(case, document)
        except AssertionError as e:
            raise AssertionError(f"case {case.get('name')!r}: {e}") from e
