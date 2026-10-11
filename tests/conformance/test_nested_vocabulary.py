# Spec: conformance-adapter section 8.2 + section 5 *Definition homes* (proposal
# 0120), applied one level below the case: inside `expected`, `resume`, `calls[]`
# and node specs.

"""Every nested key is recognized, and every one a running fixture declares is read.

The case-level checks in `test_case_vocabulary` stop at the case. A sub-key of
an `expected` block, a `resume` block, a `calls[]` entry or a node spec is
accepted and ignored by every runner that reads raw YAML, so a fixture declaring
it passes with the assertion never made.

Two things differ from the case level, both deliberately:

* modelled keys are NOT exempt from the read check. Inside these blocks a
  modelled key no runner reads is exactly the gap, since nothing applies the
  models at run time;
* keys already known to be unread sit in a ledger that may only shrink, so the
  check holds the line now and each wiring PR removes its entries.
"""

from __future__ import annotations

import ast
import importlib
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, cast

from pydantic import BaseModel

from . import test_case_vocabulary as case_level
from .harness.vocabulary import KNOWN_UNREAD_NESTED, NESTED_EXTRAS, NESTED_MODELS

_CALL_LISTS = ("calls", "secondary_calls", "tertiary_calls")
_SINGULAR_BODIES = ("subgraph", "subgraph_with_idx", "graph")
_PLURAL_BODIES = ("subgraphs", "inner_subgraphs")

# Runners that validate their fixtures into a strict model and read the parsed
# result by attribute, paired with that model's name. Only in these runners does
# an attribute access count as a read: elsewhere `client.traces` would vouch for
# observability's `traces` assertion without reading it.
_MODEL_PARSED_RUNNERS: dict[str, tuple[str, str]] = {
    "prompt-management": ("tests/conformance/test_prompt_management.py", "PromptManagementFixture"),
}

# `(capability, fixture, position, where, key)`
Declared = tuple[str, str, str, str, str]


def _as_dict(value: Any) -> dict[str, Any] | None:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [cast("dict[str, Any]", v) for v in cast("list[Any]", value) if isinstance(v, dict)]


def _node_specs(body: dict[str, Any], path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """`(path, spec)` for every node in a body and the subgraph bodies below it."""
    nodes = _as_dict(body.get("nodes")) or {}
    for name, spec in nodes.items():
        if (node := _as_dict(spec)) is not None:
            yield f"{path}nodes.{name}", node
    for key in _SINGULAR_BODIES:
        if (inner := _as_dict(body.get(key))) is not None:
            yield from _node_specs(inner, f"{path}{key}.")
    for key in _PLURAL_BODIES:
        for name, inner in (_as_dict(body.get(key)) or {}).items():
            if (one := _as_dict(inner)) is not None:
                yield from _node_specs(one, f"{path}{key}.{name}.")


def _captures(doc: dict[str, Any]) -> set[str]:
    """Every `capture_as` name the fixture declares, in any call list."""
    names: set[str] = set()
    for container in case_level._containers(doc):  # noqa: SLF001
        for list_key in _CALL_LISTS:
            for call in _as_dicts(container.get(list_key)):
                if isinstance(call.get("capture_as"), str):
                    names.add(call["capture_as"])
    return names


def _blocks(doc: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """`(position, where, block)` for every nested block a fixture carries."""
    for container in case_level._containers(doc):  # noqa: SLF001
        where = "<root>" if container is doc else str(container.get("name", "<unnamed>"))
        if (expected := _as_dict(container.get("expected"))) is not None:
            yield "expected", where, expected
        if (resume := _as_dict(container.get("resume"))) is not None:
            yield "resume", f"{where}.resume", resume
            if (resumed := _as_dict(resume.get("expected"))) is not None:
                yield "expected", f"{where}.resume", resumed
        for i, invocation in enumerate(_as_dicts(container.get("invocations"))):
            if (expected := _as_dict(invocation.get("expected"))) is not None:
                yield "expected", f"{where}.invocations[{i}]", expected
        for list_key in _CALL_LISTS:
            for i, call in enumerate(_as_dicts(container.get(list_key))):
                yield "calls[]", f"{where}.{list_key}[{i}]", call
                if (expected := _as_dict(call.get("expected"))) is not None:
                    yield "expected", f"{where}.{list_key}[{i}]", expected
        for path, spec in _node_specs(container):
            yield "nodes.*", f"{where}.{path}", spec


def _declared() -> list[Declared]:
    out: list[Declared] = []
    for capability, fixture, doc in case_level._fixture_docs():  # noqa: SLF001
        # prompt-management keys its top-level `expected` by capture name as
        # well as by assertion, and a capture name is data, not vocabulary.
        data_keys: set[str] = _captures(doc) if capability == "prompt-management" else set()
        for position, where, block in _blocks(doc):
            for key in block:
                if position == "expected" and key in data_keys:
                    continue
                out.append((capability, fixture, position, where, str(key)))
    return out


def _vocabulary(position: str) -> frozenset[str]:
    fields: set[str] = set()
    for target in NESTED_MODELS[position]:
        module, _, name = target.rpartition(".")
        model = cast("type[BaseModel]", getattr(importlib.import_module(module), name))
        fields |= set(model.model_fields)
    return frozenset(fields) | NESTED_EXTRAS[position]


def _attribute_reads() -> dict[str, set[str]]:
    """Attribute names each model-parsed runner reads, by capability."""
    sources = case_level._runner_sources()  # noqa: SLF001
    out: dict[str, set[str]] = {}
    for capability, (path, model) in _MODEL_PARSED_RUNNERS.items():
        src = sources[path]
        assert f"{model}.model_validate(" in src, f"{path} no longer validates into {model}"
        out[capability] = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    return out


def _unread() -> dict[tuple[str, str], set[str]]:
    """`(position, key)` to the running fixtures declaring it that no owning runner reads."""
    positions = case_level._read_positions()  # noqa: SLF001
    attributes = _attribute_reads()
    owners = case_level._runner_owners()  # noqa: SLF001
    executed = case_level._executed_fixture_ids()  # noqa: SLF001
    shared = {"harness/", "adapter.py", "middleware_seam.py"}
    read_via = set(case_level.READ_VIA)
    unread: dict[tuple[str, str], set[str]] = defaultdict(set)
    for capability, fixture, position, _, key in _declared():
        if fixture not in executed or key in read_via or key in attributes.get(capability, set()):
            continue
        seen = positions.get(key, set())
        if seen & owners[capability] or any(any(m in src for m in shared) for src in seen):
            continue
        unread[(position, key)].add(fixture)
    return unread


# Each container shape the walk descends into, with a floor of fixtures it must
# reach. Per shape rather than per position: losing the subgraph-body recursion,
# or `expected` under `resume`, barely moves a per-position count because those
# fixtures also carry top-level blocks.
def _shape(position: str, where: str) -> str:
    _, _, path = where.partition(".")
    if position == "nodes.*":
        return "nodes" if path.startswith("nodes.") else "nodes in subgraph bodies"
    if position == "expected":
        if not path:
            return "expected at case or root"
        if path == "resume":
            return "expected under resume"
        return "expected per invocation" if path.startswith("invocations[") else "expected per call"
    return position


_SHAPE_FLOORS: dict[str, int] = {
    "expected at case or root": 330,
    "expected under resume": 20,
    "expected per invocation": 3,
    "expected per call": 60,
    "resume": 25,
    "calls[]": 75,
    "nodes": 325,
    "nodes in subgraph bodies": 70,
}


def test_the_walk_reaches_every_container_shape() -> None:
    # Non-vacuity on the INPUT. A walk that stops reaching a shape reports a
    # clean result from no data.
    reached: dict[str, set[str]] = defaultdict(set)
    for _, fixture, position, where, _ in _declared():
        reached[_shape(position, where)].add(fixture)
    assert set(reached) == set(_SHAPE_FLOORS), f"shapes reached {sorted(reached)} != {sorted(_SHAPE_FLOORS)}"
    short = {
        shape: len(reached[shape]) for shape, floor in _SHAPE_FLOORS.items() if len(reached[shape]) < floor
    }
    assert not short, f"the walk reached too few fixtures for {short}; a container shape is being skipped"
    walked = {position for _, _, position, _, _ in _declared()}
    assert walked == set(NESTED_MODELS), (
        f"walked positions {sorted(walked)} but the registry declares {sorted(NESTED_MODELS)}"
    )


def test_the_registry_names_every_position_once() -> None:
    assert set(NESTED_MODELS) == set(NESTED_EXTRAS), (
        f"NESTED_MODELS {sorted(NESTED_MODELS)} and NESTED_EXTRAS {sorted(NESTED_EXTRAS)} disagree"
    )
    # An extra the models already declare is a stale entry: it reads as a gap
    # in the model when there is none.
    for position in NESTED_MODELS:
        modelled = _vocabulary(position) - NESTED_EXTRAS[position]
        redundant = sorted(NESTED_EXTRAS[position] & modelled)
        assert not redundant, f"{position}: {redundant} are model fields; drop them from NESTED_EXTRAS"


def test_every_nested_key_is_recognized() -> None:
    # Section 8.2 one level down. Dormant fixtures count: recognition is about
    # the corpus, not about what runs.
    vocabularies = {p: _vocabulary(p) for p in NESTED_MODELS}
    unrecognized = sorted(
        (cap, fixture, where, key)
        for cap, fixture, position, where, key in _declared()
        if key not in vocabularies[position]
    )
    assert not unrecognized, (
        "nested key(s) outside the recognized vocabulary:\n"
        + "\n".join(f"  {c}/{f} at {w} carries {k!r}" for c, f, w, k in unrecognized)
        + "\nWire the key and add it to NESTED_EXTRAS, or defer the fixture."
    )


def test_every_nested_key_on_a_running_fixture_is_read() -> None:
    unread = _unread()
    unledgered = sorted(
        (position, key, fixture)
        for (position, key), fixtures in unread.items()
        for fixture in fixtures - set(KNOWN_UNREAD_NESTED.get((position, key), ()))
    )
    assert not unledgered, (
        "nested key(s) declared by a running fixture that no owning runner reads:\n"
        + "\n".join(f"  {f} declares {k!r} at {p}" for p, k, f in unledgered)
        + "\nWire the key in that capability's runner."
    )


def test_the_unread_ledger_only_names_keys_that_are_still_unread() -> None:
    # The ratchet. An entry outliving its gap -- the key got wired, or the
    # fixture stopped running -- would let a later regression of the same key
    # land in silence.
    unread = _unread()
    stale = sorted(
        (position, key, fixture)
        for (position, key), fixtures in KNOWN_UNREAD_NESTED.items()
        for fixture in fixtures
        if fixture not in unread.get((position, key), set())
    )
    assert not stale, (
        "KNOWN_UNREAD_NESTED names gaps that are gone:\n"
        + "\n".join(f"  {f} at {p} for {k!r}" for p, k, f in stale)
        + "\nRemove the entries."
    )
