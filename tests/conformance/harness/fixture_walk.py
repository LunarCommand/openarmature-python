"""The containers and nested blocks a conformance fixture declares directives in."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from .vocabulary import CASE_MAPPINGS, NODE_MAPPING_ALIASES, NODE_MAPPINGS

CALL_LISTS = ("calls", "secondary_calls", "tertiary_calls")
_SINGULAR_BODIES = ("subgraph", "subgraph_with_idx", "graph")
_PLURAL_BODIES = ("subgraphs", "inner_subgraphs")


def as_dict(value: Any) -> dict[str, Any] | None:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [cast("dict[str, Any]", v) for v in cast("list[Any]", value) if isinstance(v, dict)]


# The root counts: a fixture with no `cases:` list carries every directive
# there, and the document-root models that would forbid an unknown key are
# applied only by `test_fixture_parsing`, which parses `loader.CAPABILITIES`
# alone and skips its own deferrals.
def containers(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The document root and each of its cases, as directive-bearing containers."""
    out = [doc]
    cases = doc.get("cases")
    if isinstance(cases, list):
        out.extend(cast("dict[str, Any]", c) for c in cast("list[Any]", cases) if isinstance(c, dict))
    return out


def _node_specs(body: dict[str, Any], path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """`(path, spec)` for every node in a body and the subgraph bodies below it."""
    nodes = as_dict(body.get("nodes")) or {}
    for name, spec in nodes.items():
        if (node := as_dict(spec)) is not None:
            yield f"{path}nodes.{name}", node
    for key in _SINGULAR_BODIES:
        if (inner := as_dict(body.get(key))) is not None:
            yield from _node_specs(inner, f"{path}{key}.")
    for key in _PLURAL_BODIES:
        for name, inner in (as_dict(body.get(key)) or {}).items():
            if (one := as_dict(inner)) is not None:
                yield from _node_specs(one, f"{path}{key}.{name}.")


def _node_position(path: str) -> str | None:
    """The node position `path` names, directly or through an alias."""
    position = NODE_MAPPING_ALIASES.get(path, path)
    return position if position == "nodes.*" or position in NODE_MAPPINGS else None


def _node_mapping_blocks(
    position: str, where: str, block: dict[str, Any]
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """`(position, where, block)` for each closed node mapping inside `block`, at any depth."""
    for key, value in block.items():
        path = f"{position}.{key}"
        children: list[tuple[str, str, dict[str, Any]]] = []
        if (child := _node_position(path)) is not None and (mapping := as_dict(value)) is not None:
            children = [(child, f"{where}.{key}", mapping)]
        elif (child := _node_position(f"{path}[]")) is not None:
            children = [(child, f"{where}.{key}[{i}]", entry) for i, entry in enumerate(as_dicts(value))]
        elif (child := _node_position(f"{path}.*")) is not None:
            children = [
                (child, f"{where}.{key}.{name}", entry)
                for name, named in (as_dict(value) or {}).items()
                if (entry := as_dict(named)) is not None
            ]
        for child, child_where, child_block in children:
            yield child, child_where, child_block
            yield from _node_mapping_blocks(child, child_where, child_block)


def blocks(doc: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """`(position, where, block)` for every nested block a fixture carries."""
    for container in containers(doc):
        where = "<root>" if container is doc else str(container.get("name", "<unnamed>"))
        if (expected := as_dict(container.get("expected"))) is not None:
            yield "expected", where, expected
        if (resume := as_dict(container.get("resume"))) is not None:
            yield "resume", f"{where}.resume", resume
            if (resumed := as_dict(resume.get("expected"))) is not None:
                yield "expected", f"{where}.resume", resumed
        for i, invocation in enumerate(as_dicts(container.get("invocations"))):
            if (expected := as_dict(invocation.get("expected"))) is not None:
                yield "expected", f"{where}.invocations[{i}]", expected
        for list_key in CALL_LISTS:
            for i, call in enumerate(as_dicts(container.get(list_key))):
                yield "calls[]", f"{where}.{list_key}[{i}]", call
                if (expected := as_dict(call.get("expected"))) is not None:
                    yield "expected", f"{where}.{list_key}[{i}]", expected
        for path, spec in _node_specs(container):
            yield "nodes.*", f"{where}.{path}", spec
            yield from _node_mapping_blocks("nodes.*", f"{where}.{path}", spec)
        for key, value in container.items():
            if key in CASE_MAPPINGS and (block := as_dict(value)) is not None:
                yield key, f"{where}.{key}", block
            elif f"{key}[]" in CASE_MAPPINGS:
                for i, entry in enumerate(as_dicts(value)):
                    yield f"{key}[]", f"{where}.{key}[{i}]", entry
