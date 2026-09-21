# Spec: conformance-adapter section 5.4 *Subgraph declaration placement*
# (proposal 0123, spec v0.114.0).

"""The placement rules the corpus cannot exercise.

Fixture 001 drives the site ranking and the same-site tie-break, and it is the
authority for both. What it does not cover is the case that made the key-level
folds in three runners wrong: a name the DOCUMENT declares and the case omits.
No shipped fixture declares that shape, measured at spec v0.118.2, so without
these the fix for it is asserted by nothing.
"""

from __future__ import annotations

from typing import Any, cast

from .harness.subgraph_placement import graph_spec_for, resolve_subgraphs

_DOC: dict[str, Any] = {"subgraphs": {"shared": {"marker": "doc-shared"}, "pick": {"marker": "doc-pick"}}}
_CASE: dict[str, Any] = {"subgraphs": {"pick": {"marker": "case-pick"}}}


def test_a_document_name_the_case_omits_stays_in_scope() -> None:
    # The whole reason the runners' key-level folds were wrong. Replacing the
    # document's block wholesale when a case declares any loses `shared`, and
    # nothing in the corpus would have said so.
    resolved = resolve_subgraphs(_DOC, _CASE)
    assert resolved["shared"]["marker"] == "doc-shared"
    assert resolved["pick"]["marker"] == "case-pick"


def test_the_graph_block_is_the_innermost_site() -> None:
    case = {**_CASE, "graph": {"subgraphs": {"pick": {"marker": "graph-pick"}}}}
    resolved = resolve_subgraphs(_DOC, case)
    assert resolved["pick"]["marker"] == "graph-pick"
    # Shadowed at the inner site, still in scope from the outer one.
    assert resolved["shared"]["marker"] == "doc-shared"


def test_the_full_resolver_applies_the_same_site_tie_break() -> None:
    # Section 5.4: where both forms bind one name at one site, the `subgraphs:`
    # mapping governs. Written with the inline form LAST, so a resolver that
    # merely takes the last declaration seen would answer "inline".
    site = {
        "subgraphs": {"pick": {"marker": "mapping"}},
        "subgraph": {"name": "pick", "marker": "inline"},
    }
    assert resolve_subgraphs(site)["pick"]["marker"] == "mapping"


def test_site_ranks_before_form() -> None:
    # An inline declaration at an inner site beats a mapping at an outer one.
    # The tie-break applies only where two forms meet at ONE site.
    document = {"subgraphs": {"pick": {"marker": "doc-mapping"}}}
    case = {"subgraph": {"name": "pick", "marker": "case-inline"}}
    assert resolve_subgraphs(document, case)["pick"]["marker"] == "case-inline"


def test_the_graph_container_replaces_only_the_graph_half() -> None:
    # Section 4.2: the container holds the graph specification and nothing else,
    # and every other case key stays its sibling.
    case: dict[str, Any] = {
        "name": "c",
        "expected": {"final_state": {}},
        "graph": {"entry": "a", "nodes": {}},
    }
    spec = graph_spec_for(case)
    assert spec["entry"] == "a"
    assert "expected" not in spec, "the container must not absorb its siblings"
    # A case carrying the specification directly reads back as itself.
    direct: dict[str, Any] = {"name": "c", "entry": "b", "nodes": {}}
    assert graph_spec_for(direct)["entry"] == "b"


_INNER: dict[str, Any] = {
    "state": {"fields": {"which": {"type": "string", "default": ""}}},
    "entry": "mark",
    "nodes": {"mark": {"update": {"which": "inner"}}},
    "edges": [{"from": "mark", "to": "END"}],
}


def test_the_module_exposes_one_resolution_entry_point() -> None:
    # Pinned because the split was tried and was wrong. A narrower resolver over
    # the plural form alone leaves each caller its own inline declaration, and
    # nothing then ranks across the two forms: a document-level mapping beats a
    # case-level inline declaration, inverting section 5.4's site-before-form.
    #
    # One winning declaration per name, resolved in one place, compiled once.
    from .harness import subgraph_placement

    public = {n for n in dir(subgraph_placement) if n.startswith("resolve")}
    assert public == {"resolve_subgraphs"}, (
        f"expected one resolution entry point, found {sorted(public)}. Ranking one form "
        "separately cannot express site-before-form."
    )


async def test_a_container_case_declaring_subgraphs_runs_through_the_dispatcher() -> None:
    # The seam neither half covered: the unit tests above drive the resolver and
    # the fixtures drive the runners, so a container case going THROUGH a routed
    # runner was exercised by nothing. It raised a duplicate-name ValueError
    # before the case could run, because the ranked map was hoisted alongside the
    # container's own declarations rather than replacing them, and the runner
    # collects from both when a container is present.
    #
    # No graph-engine fixture declares subgraphs inside a container today, which
    # is why the corpus stayed green. It is site 3 of the three section 5.4
    # sanctions, so it is precisely what this adoption exists to enable.
    from . import test_conformance as runtime_runner

    case: dict[str, Any] = {
        "name": "container_with_subgraphs",
        "graph": {
            "state": {"fields": {"resolved": {"type": "string", "default": ""}}},
            "entry": "run",
            "nodes": {"run": {"subgraph": "pick", "outputs": {"resolved": "which"}}},
            "edges": [{"from": "run", "to": "END"}],
            "subgraphs": {"pick": _INNER},
        },
        "initial_state": {},
        "expected": {"final_state": {"resolved": "inner"}},
    }
    merged = dict(case)
    ranked = resolve_subgraphs({"cases": [case]}, case)
    assert ranked, "the container's declaration must reach the ranked map"
    merged["subgraphs"] = ranked
    container = cast("dict[str, Any]", merged["graph"])
    merged["graph"] = {k: v for k, v in container.items() if k != "subgraphs"}

    await runtime_runner._run_runtime_case(merged, "synthetic")  # noqa: SLF001


async def test_an_inner_singular_declaration_beats_an_outer_mapping_through_a_runner() -> None:
    # Section 5.4's site-before-form rule, driven through a routed runner rather
    # than against the resolver alone. No fixture declares one name in both forms
    # at different sites outside conformance-adapter/001, so nothing in the
    # corpus exercises this path.
    #
    # It is the shape that a split resolver cannot answer: rank the plural form
    # alone and the caller keeps its own inline declaration, so nothing compares
    # the two and the outer mapping wins. Which inverts the rule.
    from . import test_conformance as runtime_runner

    shell: dict[str, Any] = {
        "state": {"fields": {"which": {"type": "string", "default": ""}}},
        "entry": "mark",
        "edges": [{"from": "mark", "to": "END"}],
    }
    document: dict[str, Any] = {
        "subgraphs": {"pick": {**shell, "nodes": {"mark": {"update": {"which": "outer-mapping"}}}}}
    }
    case: dict[str, Any] = {
        "name": "cross_form_cross_site",
        "subgraph": {
            "name": "pick",
            **shell,
            "nodes": {"mark": {"update": {"which": "inner-inline"}}},
        },
        "state": {"fields": {"resolved": {"type": "string", "default": ""}}},
        "entry": "run",
        "nodes": {"run": {"subgraph": "pick", "outputs": {"resolved": "which"}}},
        "edges": [{"from": "run", "to": "END"}],
        "initial_state": {},
        "expected": {"final_state": {"resolved": "inner-inline"}},
    }
    merged: dict[str, Any] = {**document, **case}
    ranked = resolve_subgraphs(document, case)
    assert ranked["pick"]["nodes"]["mark"]["update"]["which"] == "inner-inline"
    merged["subgraphs"] = ranked

    await runtime_runner._run_runtime_case(merged, "synthetic")  # noqa: SLF001
