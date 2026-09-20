# Spec: conformance-adapter section 5.4 *Subgraph declaration placement* and
# section 4.2 *The `graph:` container* (proposal 0123, spec v0.114.0).

"""Where a subgraph declaration may sit, and which one wins.

A declaration is scoped to the graph specification it accompanies, so it may sit
at three sites. They rank outermost to innermost:

1. the fixture document's top level, in scope for every case in the file;
2. an individual case, visible to that case alone;
3. a case's ``graph:`` block, for a table fixture whose cases each carry a
   complete graph specification.

Two rules decide a name declared more than once. **Site ranks first**: the
innermost declaration in scope for the running case governs. Where both
declaration *forms* meet at one site and bind the same name, the site ranking
cannot separate them, so the ``subgraphs:`` mapping entry governs because it
names the binding explicitly.

Section 5.4 leaves a declaration nested inside another subgraph body neither
sanctioned nor forbidden, so this resolves the three sites it names and stops.
"""

from __future__ import annotations

from typing import Any, cast

# Outermost to innermost. The order is the rule, so it lives in one place rather
# than being reproduced by each caller's merge order.
SITES = ("document", "case", "graph")


def _bodies_at(container: Any) -> dict[str, dict[str, Any]]:
    """The declarations one site makes, as ``name -> body``.

    Applies the same-site tie-break: the plural mapping is read after the
    singular form, so where both bind one name at this site the mapping wins.
    That ordering IS the rule, not an artefact of dict update order -- the
    singular form is read first precisely so the mapping can overwrite it.
    """
    if not isinstance(container, dict):
        return {}
    site = cast("dict[str, Any]", container)
    out: dict[str, dict[str, Any]] = {}

    for singular in ("subgraph", "subgraph_with_idx"):
        body = site.get(singular)
        if isinstance(body, dict):
            named = cast("dict[str, Any]", body)
            # The singular form carries its own name; a body without one binds
            # nothing resolvable and is left to the caller that reads it
            # positionally.
            name = named.get("name")
            if isinstance(name, str):
                out[name] = named

    plural = site.get("subgraphs")
    if isinstance(plural, dict):
        for name, body in cast("dict[str, Any]", plural).items():
            if isinstance(body, dict):
                out[name] = cast("dict[str, Any]", body)
    return out


def resolve_subgraphs(document: Any, case: Any = None) -> dict[str, dict[str, Any]]:
    """Every subgraph name in scope for ``case``, bound to the body that governs.

    Later sites overwrite earlier ones, which is innermost-wins expressed as
    application order rather than as a search. A name declared only at an outer
    site survives, since a top-level declaration stays in scope and is merely
    shadowed where an inner site rebinds it.
    """
    resolved: dict[str, dict[str, Any]] = {}
    resolved.update(_bodies_at(document))
    if case is not None:
        resolved.update(_bodies_at(case))
        if isinstance(case, dict):
            resolved.update(_bodies_at(cast("dict[str, Any]", case).get("graph")))
    return resolved


def resolve_subgraph_mappings(document: Any, case: Any = None) -> dict[str, dict[str, Any]]:
    """The ``subgraphs:`` mappings in scope for ``case``, ranked by site.

    The same site ranking as :func:`resolve_subgraphs`, over the plural form
    only. For a caller that consumes the singular ``subgraph:`` itself: folding
    that body into the returned mapping would hand it back a second time under
    its declared name, and it would build the same subgraph twice.

    The same-site tie-break is not expressible here and does not need to be. It
    decides between the two forms at one site, and a caller using this function
    is resolving one form; no site in the corpus declares both anyway.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for site in (
        document,
        case,
        cast("dict[str, Any]", case).get("graph") if isinstance(case, dict) else None,
    ):
        if not isinstance(site, dict):
            continue
        plural = cast("dict[str, Any]", site).get("subgraphs")
        if isinstance(plural, dict):
            for name, body in cast("dict[str, Any]", plural).items():
                if isinstance(body, dict):
                    resolved[name] = cast("dict[str, Any]", body)
    return resolved


def graph_spec_for(case: Any) -> dict[str, Any]:
    """A case's graph specification, from the container or from the case body.

    Section 4.2 makes the two forms equivalent and requires a container case
    asserting a runtime outcome to execute, so a caller reads the graph through
    this rather than off the case directly. Every non-graph key stays a sibling
    of ``graph:``, so the container is not merged over the case: it replaces the
    graph-shaped half and nothing else.
    """
    if not isinstance(case, dict):
        return {}
    body = cast("dict[str, Any]", case)
    container = body.get("graph")
    if isinstance(container, dict):
        return cast("dict[str, Any]", container)
    return body
