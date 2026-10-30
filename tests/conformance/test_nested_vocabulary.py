# Spec: conformance-adapter section 8.2 + section 5 *Definition homes* (proposal
# 0120), applied one level below the case: inside `expected`, `resume`, `calls[]`,
# node specs, and every other directive whose value is a mapping.

"""Every nested key is recognized, and every one a running fixture declares is read.

The case-level checks in `test_case_vocabulary` stop at the case. A sub-key of
an `expected` block, a `resume` block, a `calls[]` entry, a node spec or any
other directive mapping is accepted and ignored by every runner that reads raw
YAML, so a fixture declaring it passes with the assertion never made.

Two things differ from the case level, both deliberately:

* modelled keys are NOT exempt from the read check. Inside these blocks a
  modelled key no runner reads is exactly the gap, since nothing applies the
  models at run time;
* keys already known to be unread sit in a ledger that may only shrink, so the
  check holds the line now and each wiring PR removes its entries.

The read check judges what the runners actually read in this session. Each one
loads its fixtures through `harness.read_trace`, which records every key looked
up in a nested block against that block's position, so the check runs after
every runner. On a partial run it skips, and under CI it fails instead.
"""

from __future__ import annotations

import ast
import importlib
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from typing import Any, cast, get_args

import pytest
from pydantic import BaseModel, ValidationError

from . import test_case_vocabulary as case_level
from .harness import read_trace
from .harness.fixture_walk import CALL_LISTS, as_dict, as_dicts, blocks, containers
from .harness.vocabulary import (
    CASE_MAPPINGS,
    FORWARDING_BUILDERS,
    KNOWN_UNREAD_NESTED,
    NESTED_EXTRAS,
    NESTED_MODELS,
    NODE_FORWARDED_MAPPINGS,
    NODE_MAPPING_ALIASES,
    NODE_MAPPINGS,
    NODE_OPEN_MAPPINGS,
    OPEN_MAPPINGS,
    OUT_OF_SCOPE_MAPPINGS,
    WALKED_MAPPINGS,
)

# Runners that validate their fixtures into a strict model and read the parsed
# result by attribute, paired with that model's name. Only in these runners does
# an attribute access count as a read, and only off a value typed as one of the
# capability's fixture models: elsewhere `client.traces` would vouch for
# observability's `traces` assertion without reading it.
_MODEL_PARSED_RUNNERS: dict[str, tuple[str, str]] = {
    "prompt-management": ("tests/conformance/test_prompt_management.py", "PromptManagementFixture"),
}
_FIXTURE_MODELS: dict[str, str] = {
    "prompt-management": "tests.conformance.harness.prompt_management",
}

# `(capability, fixture, position, where, key)`
Declared = tuple[str, str, str, str, str]

# Every position's models and extra keys: the nested positions and each closed
# case-level and node-level mapping.
_POSITIONS: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {
    **{position: (NESTED_MODELS[position], NESTED_EXTRAS[position]) for position in NESTED_MODELS},
    **{position: (mapping.models, mapping.extras) for position, mapping in CASE_MAPPINGS.items()},
    **{position: (mapping.models, mapping.extras) for position, mapping in NODE_MAPPINGS.items()},
}


def _captures(doc: dict[str, Any]) -> set[str]:
    """Every `capture_as` name the fixture declares, in any call list."""
    names: set[str] = set()
    for container in containers(doc):
        for list_key in CALL_LISTS:
            for call in as_dicts(container.get(list_key)):
                if isinstance(call.get("capture_as"), str):
                    names.add(call["capture_as"])
    return names


def _path_of(position: str) -> str:
    return position.removesuffix("[]").removesuffix(".*")


def _declared() -> list[Declared]:
    out: list[Declared] = []
    for capability, fixture, doc in case_level._fixture_docs():  # noqa: SLF001
        # prompt-management keys its top-level `expected` by capture name as
        # well as by assertion, and a capture name is data, not vocabulary.
        data_keys: set[str] = _captures(doc) if capability == "prompt-management" else set()
        for position, where, block in blocks(doc):
            for key in block:
                if position == "expected" and key in data_keys:
                    continue
                out.append((capability, fixture, position, where, str(key)))
    return out


def _model_fields(position: str) -> frozenset[str]:
    fields: set[str] = set()
    for target in _POSITIONS[position][0]:
        module, _, name = target.rpartition(".")
        model = cast(type[BaseModel], getattr(importlib.import_module(module), name))
        # The YAML spelling: `EdgeSpec.from_` is declared in a fixture as `from`.
        fields |= {field.alias or field_name for field_name, field in model.model_fields.items()}
    return frozenset(fields)


def _vocabulary(position: str) -> frozenset[str]:
    return _model_fields(position) | _POSITIONS[position][1]


def _model_in(annotation: Any) -> type[BaseModel] | None:
    """The one pydantic model an annotation holds, looking through Optional and containers."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    found = {m for arg in get_args(annotation) if (m := _model_in(arg)) is not None}
    return found.pop() if len(found) == 1 else None


class _ModelValues:
    """Which names in one function hold a fixture model value, and which model."""

    def __init__(
        self, fn: ast.FunctionDef | ast.AsyncFunctionDef, models: dict[str, type[BaseModel]]
    ) -> None:
        self.models = models
        self.env: dict[str, type[BaseModel]] = {}
        # A list of tuples, such as `[(fixture.manager, fixture.calls), ...]`,
        # unpacked by a `for a, b in ...`: the model at each tuple position.
        self.rows: dict[str, list[type[BaseModel] | None]] = {}
        args = fn.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            self.bind(arg.arg, self.annotated(arg.annotation))
        # Bindings can feed each other in any order, so settle them first.
        for _ in range(4):
            for node in ast.walk(fn):
                self.learn(node)

    def annotated(self, node: ast.expr | None) -> type[BaseModel] | None:
        if node is None:
            return None
        named = {self.models[n.id] for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in self.models}
        return named.pop() if len(named) == 1 else None

    def bind(self, target: ast.expr | str, model: type[BaseModel] | None) -> None:
        name = target if isinstance(target, str) else target.id if isinstance(target, ast.Name) else None
        if name is not None and model is not None:
            self.env[name] = model

    def type_of(self, expr: ast.expr) -> Any:
        if isinstance(expr, ast.Name):
            return self.env.get(expr.id)
        if isinstance(expr, ast.Attribute) and (owner := _model_in(self.type_of(expr.value))) is not None:
            field = owner.model_fields.get(expr.attr)
            return field.annotation if field is not None else None
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "model_validate"
            and isinstance(expr.func.value, ast.Name)
        ):
            return self.models.get(expr.func.value.id)
        return None

    def row_of(self, value: ast.expr) -> list[type[BaseModel] | None] | None:
        if not isinstance(value, ast.List) or not value.elts:
            return None
        tuples = [e.elts for e in value.elts if isinstance(e, ast.Tuple)]
        if len(tuples) != len(value.elts) or len({len(t) for t in tuples}) != 1:
            return None
        columns = zip(*([_model_in(self.type_of(e)) for e in t] for t in tuples), strict=True)
        return [col[0] if len(set(col)) == 1 else None for col in columns]

    def learn(self, node: ast.AST) -> None:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and (row := self.row_of(node.value)) is not None:
                self.rows[target.id] = row
            self.bind(target, _model_in(self.type_of(node.value)))
        elif isinstance(node, ast.AnnAssign):
            self.bind(node.target, self.annotated(node.annotation))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            target = node.target
            if isinstance(target, ast.Tuple) and isinstance(node.iter, ast.Name):
                row = self.rows.get(node.iter.id, [])
                if len(row) == len(target.elts):
                    for element, model in zip(target.elts, row, strict=True):
                        self.bind(element, model)
            self.bind(target, _model_in(self.type_of(node.iter)))


def _field_reads(src: str, models: dict[str, type[BaseModel]]) -> set[tuple[str, str]]:
    """`(model, field)` for each field the source reads off a fixture model value.

    A value is a fixture model when it comes from a parameter or variable
    annotated with one, a `<Model>.model_validate(...)` call, iteration over a
    model-typed value, or an attribute of one. Any other receiver is ignored:
    `backend.name` says nothing about `FixtureCall.name`.
    """
    reads: set[tuple[str, str]] = set()
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        values = _ModelValues(fn, models)
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Attribute)
                and (owner := _model_in(values.type_of(node.value))) is not None
            ):
                if node.attr in owner.model_fields:
                    reads.add((owner.__name__, node.attr))
    return reads


def _attribute_reads() -> dict[str, set[tuple[str, str]]]:
    """`(model, field)` reads each model-parsed runner makes, by capability."""
    sources = case_level._runner_sources()  # noqa: SLF001
    out: dict[str, set[tuple[str, str]]] = {}
    for capability, (path, model) in _MODEL_PARSED_RUNNERS.items():
        src = sources[path]
        assert f"{model}.model_validate(" in src, f"{path} no longer validates into {model}"
        module = importlib.import_module(_FIXTURE_MODELS[capability])
        models = {
            name: obj
            for name, obj in vars(module).items()
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj.__module__ == module.__name__
        }
        out[capability] = _field_reads(src, models)
    return out


def _read_as_attribute(reads: set[tuple[str, str]], position: str, key: str) -> bool:
    return any((target.rpartition(".")[2], key) in reads for target in _POSITIONS[position][0])


Traced = Mapping[tuple[str, str, str], frozenset[tuple[str, str]]]
Reads = dict[str, set[tuple[str, str]]]
Unread = dict[tuple[str, str], set[str]]


def _owned(traced: Traced) -> tuple[Reads, Reads, set[tuple[str, str]]]:
    """The traced `(position, key)` reads that count, by capability and by fixture,
    and the `(capability, fixture)` pairs a runner owning the capability loaded.

    A runner that owns a capability vouches for every fixture in it, and one
    that loads a single named fixture for that fixture alone. A read by any
    other runner counts for nothing.
    """
    owners = case_level._runner_owners()  # noqa: SLF001
    fixture_owners = case_level._fixture_owners()  # noqa: SLF001
    by_capability: Reads = defaultdict(set)
    by_fixture: Reads = defaultdict(set)
    loaded: set[tuple[str, str]] = set()
    for (capability, fixture, runner), pairs in traced.items():
        if runner in owners.get(capability, frozenset()):
            by_capability[capability] |= pairs
            loaded.add((capability, fixture))
        elif runner in fixture_owners.get(fixture, frozenset()):
            by_fixture[fixture] |= pairs
    return by_capability, by_fixture, loaded


def _unread(traced: Traced) -> Unread:
    """The fixtures declaring each `(position, key)` no owning runner read, on running fixtures."""
    by_capability, by_fixture, _ = _owned(traced)
    attributes = _attribute_reads()
    executed = case_level._executed_fixture_ids()  # noqa: SLF001
    read_via = set(case_level.READ_VIA)
    unread: Unread = defaultdict(set)
    for capability, fixture, position, _, key in _declared():
        if fixture not in executed or key in read_via:
            continue
        # A model-parsed runner reads its fixture as model fields rather than
        # through the trace, so its reads are found in its source instead.
        if capability in attributes:
            if not _read_as_attribute(attributes[capability], position, key):
                unread[(position, key)].add(fixture)
            continue
        if (position, key) not in by_capability[capability] and (position, key) not in by_fixture[fixture]:
            unread[(position, key)].add(fixture)
    return unread


@pytest.fixture(scope="session")
def unread_nested() -> Unread:
    """The nested keys no owning runner read this session, by `(position, key)`.

    The conformance conftest runs every test using this after the rest of the
    session.
    """
    traced = read_trace.reads()
    _, _, loaded = _owned(traced)
    executed = case_level._executed_fixture_ids()  # noqa: SLF001
    missing = sorted(
        fixture
        for capability, fixture, _ in case_level._fixture_docs()  # noqa: SLF001
        if fixture in executed
        and capability not in _MODEL_PARSED_RUNNERS
        and (capability, fixture) not in loaded
    )
    if missing:
        reason = (
            f"{len(missing)} running fixture(s) were not loaded by an owning runner this session, "
            f"{missing[0]} first; the nested read check needs the whole suite"
        )
        # CI always runs the whole suite, so there a missing load means a runner
        # stopped loading through the trace rather than a partial run.
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)
    return _unread(traced)


# Each container shape the walk descends into, with a floor of fixtures it must
# reach. Per shape rather than per position: losing the subgraph-body recursion,
# or `expected` under `resume`, barely moves a per-position count because those
# fixtures also carry top-level blocks.
def _shape(position: str, where: str) -> str:
    _, _, path = where.partition(".")
    if position in CASE_MAPPINGS:
        return "case-level mappings"
    if position in NODE_MAPPINGS:
        return "node-level mappings" if position.count(".") == 2 else "nested node mappings"
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
    "case-level mappings": 445,
    "node-level mappings": 260,
    "nested node mappings": 120,
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
    assert walked == set(_POSITIONS), (
        f"walked positions {sorted(walked)} but the registry declares {sorted(_POSITIONS)}"
    )


def test_the_registry_names_every_position_once() -> None:
    assert set(NESTED_MODELS) == set(NESTED_EXTRAS), (
        f"NESTED_MODELS {sorted(NESTED_MODELS)} and NESTED_EXTRAS {sorted(NESTED_EXTRAS)} disagree"
    )
    registries = {
        "nested": set(NESTED_MODELS),
        "case-level": set(CASE_MAPPINGS),
        "node-level": set(NODE_MAPPINGS),
    }
    named = Counter(position for positions in registries.values() for position in positions)
    twice = sorted(position for position, count in named.items() if count > 1)
    assert not twice, f"{twice} are registered as more than one of {sorted(registries)}"
    # An alias is a path of its own, onto a node position that exists.
    for alias, target in NODE_MAPPING_ALIASES.items():
        assert alias not in NODE_MAPPINGS, f"{alias} is both a node position and an alias"
        assert target == "nodes.*" or target in NODE_MAPPINGS, f"alias {alias} names no position: {target}"
    # An extra the models already declare is a stale entry: it reads as a gap
    # in the model when there is none.
    for position, (_, extras) in _POSITIONS.items():
        redundant = sorted(extras & _model_fields(position))
        assert not redundant, f"{position}: {redundant} are model fields; drop them from its extras"


def _holds_mappings(value: Any) -> bool:
    entries = as_dicts(value)
    return as_dict(value) is not None or bool(entries and len(entries) == len(value))


def _assert_placed(level: str, used: set[str], groups: dict[str, frozenset[str]], hint: str) -> None:
    placed = Counter(name for names in groups.values() for name in names)
    twice = sorted(name for name, count in placed.items() if count > 1)
    assert not twice, f"{level} mapping(s) placed in more than one group: {twice}"
    unplaced = sorted(used - set(placed))
    assert not unplaced, f"{level} mapping(s) in no group: {unplaced}\n{hint}"
    stale = sorted(set(placed) - used)
    assert not stale, f"no fixture declares these as a {level} mapping any more; remove them: {stale}"


def test_every_case_level_mapping_is_placed() -> None:
    # The classification is a decision per directive. A mapping nothing places
    # would have its keys go unchecked with no record that anyone chose that.
    used = {
        str(key)
        for _, _, doc in case_level._fixture_docs()  # noqa: SLF001
        for container in containers(doc)
        for key, value in container.items()
        if _holds_mappings(value)
    }
    groups = {
        "walked": WALKED_MAPPINGS,
        "open": frozenset(OPEN_MAPPINGS),
        "out of scope": frozenset(OUT_OF_SCOPE_MAPPINGS),
        "closed": frozenset(position.removesuffix("[]") for position in CASE_MAPPINGS),
    }
    _assert_placed(
        "case-level",
        used,
        groups,
        "Place each in harness/vocabulary.py: CASE_MAPPINGS with its keys, or OPEN_MAPPINGS with the reason.",
    )


def test_every_node_level_mapping_is_placed() -> None:
    # At every depth: inside a node spec, and inside each closed node mapping
    # below it, since the walk descends only through closed ones.
    used = {
        f"{position}.{key}"
        for _, _, doc in case_level._fixture_docs()  # noqa: SLF001
        for position, _, block in blocks(doc)
        if position == "nodes.*" or position in NODE_MAPPINGS
        for key, value in block.items()
        if _holds_mappings(value)
    }
    groups = {
        "open": frozenset(NODE_OPEN_MAPPINGS),
        "forwarded": frozenset(NODE_FORWARDED_MAPPINGS),
        "closed": frozenset(_path_of(position) for position in [*NODE_MAPPINGS, *NODE_MAPPING_ALIASES]),
    }
    _assert_placed(
        "node-level",
        used,
        groups,
        "Place each in harness/vocabulary.py: NODE_MAPPINGS with its keys, "
        "or NODE_OPEN_MAPPINGS with the reason.",
    )


def _forwarded_blocks() -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    """`(capability, fixture, path, mapping)` for each forwarded node mapping."""
    for capability, fixture, doc in case_level._fixture_docs():  # noqa: SLF001
        for position, _, block in blocks(doc):
            if position != "nodes.*" and position not in NODE_MAPPINGS:
                continue
            for key, value in block.items():
                path = f"{position}.{key}"
                if path in NODE_FORWARDED_MAPPINGS and (mapping := as_dict(value)) is not None:
                    yield capability, fixture, path, mapping


def _forwarded_type(path: str) -> type[BaseModel]:
    module, _, name = NODE_FORWARDED_MAPPINGS[path].rpartition(".")
    return cast(type[BaseModel], getattr(importlib.import_module(module), name))


def test_every_forwarded_key_is_a_field_of_its_type() -> None:
    # The type rejects a field it does not declare, so its fields are the
    # vocabulary. Dormant fixtures count, as for every recognized key.
    reached: set[str] = set()
    unknown: list[tuple[str, str, str]] = []
    for _, fixture, path, mapping in _forwarded_blocks():
        reached.add(path)
        model = _forwarded_type(path)
        assert model.model_config.get("extra") == "forbid", (
            f"{model.__name__} accepts fields it does not declare"
        )
        fields = {field.alias or name for name, field in model.model_fields.items()}
        unknown.extend((fixture, path, str(key)) for key in mapping if key not in fields)
    assert reached == set(NODE_FORWARDED_MAPPINGS), (
        f"no fixture forwards {sorted(set(NODE_FORWARDED_MAPPINGS) - reached)}"
    )
    assert not unknown, "forwarded key(s) the type does not declare:\n" + "\n".join(
        f"  {fixture} at {path} carries {key!r}" for fixture, path, key in sorted(unknown)
    )


def _calls_unpacking_into(fn: ast.AST, type_name: str) -> bool:
    """Whether `fn` calls `type_name` with a `**` argument."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and any(keyword.arg is None for keyword in node.keywords):
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called == type_name:
                return True
    return False


def _builders_in(runner: str, src: str, type_name: str) -> set[str]:
    """`module.function` for each function in `src` that builds `type_name` from a mapping."""
    module = runner.removesuffix(".py").replace("/", ".")
    found: set[str] = set()
    for node in ast.parse(src).body:
        defs: list[tuple[str, ast.AST]] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            defs.extend(
                (f"{node.name}.{member.name}", member)
                for member in node.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        found.update(f"{module}.{name}" for name, fn in defs if _calls_unpacking_into(fn, type_name))
    return found


def test_every_builder_of_a_forwarded_type_is_declared() -> None:
    # Building the type from a mapping is where a runner hands the mapping over,
    # so every function in an owning runner that does it must be declared, and
    # the next test checks each one passes the mapping whole. The type is matched
    # by its exact name: `EmbeddingRuntimeConfig(**...)` builds no RuntimeConfig.
    sources = case_level._runner_sources()  # noqa: SLF001
    owners = case_level._runner_owners()  # noqa: SLF001
    fixture_owners = case_level._fixture_owners()  # noqa: SLF001
    executed = case_level._executed_fixture_ids()  # noqa: SLF001
    runners: dict[str, set[str]] = defaultdict(set)
    for capability, fixture, path, _ in _forwarded_blocks():
        if fixture in executed:
            runners[path] |= owners[capability] | fixture_owners.get(fixture, frozenset())
    assert set(runners) == set(NODE_FORWARDED_MAPPINGS), (
        f"no running fixture forwards {sorted(set(NODE_FORWARDED_MAPPINGS) - set(runners))}"
    )
    assert set(FORWARDING_BUILDERS) == set(NODE_FORWARDED_MAPPINGS), (
        f"FORWARDING_BUILDERS names {sorted(FORWARDING_BUILDERS)}, the forwarded mappings are "
        f"{sorted(NODE_FORWARDED_MAPPINGS)}"
    )
    for path in sorted(NODE_FORWARDED_MAPPINGS):
        type_name = _forwarded_type(path).__name__
        found = set[str]().union(*(_builders_in(r, sources[r], type_name) for r in runners[path]))
        assert found == set(FORWARDING_BUILDERS[path]), (
            f"{path}: owning runners build {type_name} from a mapping in {sorted(found)}, "
            f"FORWARDING_BUILDERS declares {sorted(FORWARDING_BUILDERS[path])}"
        )


def test_every_forwarding_builder_passes_the_whole_mapping() -> None:
    # A builder that copies only the keys it knows drops an unknown one in
    # silence; one that passes the mapping whole lets the type reject it. Killed
    # by the LLM config builders this replaced, which copied seven named keys.
    dropped: list[str] = []
    for builders in FORWARDING_BUILDERS.values():
        for dotted in builders:
            module, _, name = dotted.rpartition(".")
            builder = getattr(importlib.import_module(module), name)
            try:
                builder({"_probe_unknown_key": 1})
            except ValidationError:
                continue
            dropped.append(dotted)
    assert not dropped, f"builder(s) that accept a key their type does not declare: {dropped}"


def test_every_nested_key_is_recognized() -> None:
    # Section 8.2 one level down. Dormant fixtures count: recognition is about
    # the corpus, not about what runs.
    vocabularies = {p: _vocabulary(p) for p in _POSITIONS}
    unrecognized = sorted(
        (cap, fixture, where, key)
        for cap, fixture, position, where, key in _declared()
        if key not in vocabularies[position]
    )
    assert not unrecognized, (
        "nested key(s) outside the recognized vocabulary:\n"
        + "\n".join(f"  {c}/{f} at {w} carries {k!r}" for c, f, w, k in unrecognized)
        + "\nWire the key and add it to NESTED_EXTRAS or CASE_MAPPINGS, or defer the fixture."
    )


def test_every_nested_key_on_a_running_fixture_is_read(unread_nested: Unread) -> None:
    unledgered = sorted(
        (position, key, fixture)
        for (position, key), fixtures in unread_nested.items()
        for fixture in fixtures - set(KNOWN_UNREAD_NESTED.get((position, key), ()))
    )
    assert not unledgered, (
        "nested key(s) declared by a running fixture that no owning runner read:\n"
        + "\n".join(f"  {f} declares {k!r} at {p}" for p, k, f in unledgered)
        + "\nWire the key in that capability's runner. A read from a copy made with dict(...) "
        "or {**...} is not recorded; copy the block with .copy() instead."
    )


def test_the_unread_ledger_only_names_keys_that_are_still_unread(unread_nested: Unread) -> None:
    # The ratchet. An entry outliving its gap -- the key got wired, or the
    # fixture stopped running -- would let a later regression of the same key
    # land in silence.
    stale = sorted(
        (position, key, fixture)
        for (position, key), fixtures in KNOWN_UNREAD_NESTED.items()
        for fixture in fixtures
        if fixture not in unread_nested.get((position, key), set())
    )
    assert not stale, (
        "KNOWN_UNREAD_NESTED names gaps that are gone:\n"
        + "\n".join(f"  {f} at {p} for {k!r}" for p, k, f in stale)
        + "\nRemove the entries."
    )


def test_a_block_with_unread_invariants_carries_a_read_directive(unread_nested: Unread) -> None:
    # The `invariants` ledger entry holds only while each block it covers also
    # carries a directive the runner reads. Predicates alone assert nothing, so
    # a block of them and nothing else would pass as coverage under the entry.
    ledgered = set(KNOWN_UNREAD_NESTED.get(("expected", "invariants"), ()))
    walked: set[str] = set()
    bare: list[tuple[str, str]] = []
    for _, fixture, doc in case_level._fixture_docs():  # noqa: SLF001
        if fixture not in ledgered:
            continue
        for position, where, block in blocks(doc):
            if position != "expected" or "invariants" not in block:
                continue
            walked.add(fixture)
            siblings = [str(key) for key in block if key != "invariants"]
            if all(fixture in unread_nested.get((position, key), set()) for key in siblings):
                bare.append((fixture, where))
    # Non-vacuity on the INPUT: a ledgered fixture whose block the walk misses
    # would pass here unchecked.
    assert walked == ledgered, f"found no `expected.invariants` block in {sorted(ledgered - walked)}"
    assert not bare, (
        "block(s) whose `invariants` are ledgered as unread, with no directive the runner reads:\n"
        + "\n".join(f"  {f} at {w}" for f, w in sorted(bare))
        + "\nAdd a directive the runner reads, or wire the predicates and drop the ledger entry."
    )


def test_every_owning_runner_loads_its_fixtures_through_the_trace() -> None:
    # A runner that parses its fixtures any other way records nothing, so each
    # nested key only it reads would show as unread. prompt-management reads its
    # fixtures as a model and is checked from its source instead.
    sources = case_level._runner_sources()  # noqa: SLF001
    owners = case_level._runner_owners()  # noqa: SLF001
    fixture_owners = case_level._fixture_owners()  # noqa: SLF001
    owning = {
        path
        for capability, paths in owners.items()
        if capability not in _MODEL_PARSED_RUNNERS
        for path in paths
    } | {path for paths in fixture_owners.values() for path in paths}
    assert len(owning) >= 8, f"found only {sorted(owning)} as owning runners; the ownership scan is broken"
    untraced: list[str] = []
    for path in sorted(owning):
        called = {
            (node.func.value.id, node.func.attr)
            for node in ast.walk(ast.parse(sources[path]))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }
        parses = sorted(
            f"yaml.{attr}" for module, attr in called if module == "yaml" and attr.endswith("load")
        )
        if parses:
            untraced.append(f"{path} parses with {', '.join(parses)}")
        elif ("read_trace", "load") not in called:
            untraced.append(f"{path} never calls read_trace.load")
    assert not untraced, (
        "owning runner(s) that load fixtures outside the read trace:\n"
        + "\n".join(f"  {line}" for line in untraced)
        + "\nLoad each fixture with read_trace.load."
    )
