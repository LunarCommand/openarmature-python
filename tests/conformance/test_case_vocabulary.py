# Spec: conformance-adapter section 8.2 + section 5 *Definition homes* (proposal
# 0120, spec v0.113.0). Section 9 requires the adapter to raise rather than skip,
# and to surface the offending directive and its location; the token itself is
# not observable surface: no fixture can see which category an adapter raises.

"""Every directive is recognized, and every one outside the models is read.

Both halves are narrower than they sound, and the second is narrower than the
first. Recognition covers every key the corpus declares. The read check covers
only directives the fixture models do NOT declare: a field on `CaseSpec` or on a
root model is skipped, and that exclusion is not the harmless one it looks like.
`expected_wire_request`, `checkpointer` and `mock_llm` are modelled AND
behavioural, so deleting every read of one from its owning runner passes here.
Closing that needs each modelled field classified as read through a model, read
by name, inert by construction, or genuinely dark, not a wider exclusion list.

`extra="forbid"` cannot carry this. It applies at the fixture's document root,
while `CaseSpec` and `SubgraphDefinition` both allow extras, and the runners
read raw YAML rather than the typed model at all -- so the model's config does
not reach the behaviour section 8.2 is about. These are repository checks over
the corpus instead, which no runner can forget to call.

Three things each check has to get right, and only the first is about the
assertion:

* the ASSERTION fires when its claim is false;
* the INPUT is complete, since a walk that silently skips a container reports a
  clean result from partial data;
* the EVIDENCE means what it claims -- "this key is read" must not be satisfied
  by a literal in a comment, by a different capability's runner, or by a key
  that merely appears in a file.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from .harness import fixtures as fixture_models
from .harness.fixtures import CaseSpec, SubgraphDefinition
from .harness.vocabulary import (
    PENDING_ADOPTION,
    READ_VIA,
    RECOGNIZED_DIRECTIVES,
    UNAPPLIED_PENDING_CASE_DEFERRAL,
    UNAPPLIED_PENDING_DEFERRAL,
    UNIMPLEMENTED_CAPABILITIES,
)

_SPEC_ROOT = Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec"
_RUNNER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Capability directories this implementation runs fixtures from.
#
# Wider than `loader.CAPABILITIES`, which enumerates only what the shared
# `discover_fixtures` yields: several capabilities are driven by a module with
# its own conformance directory and glob, and some llm-provider fixtures are
# driven from `tests/unit`.
_RUN_DIRS: tuple[str, ...] = (
    "graph-engine",
    "llm-provider",
    "pipeline-utilities",
    "observability",
    "prompt-management",
    "retrieval-provider",
    "conformance-adapter",
)

# Modules whose own `_fixture_paths()` is the authority on what they collect,
# so "runs" is computed from the collection logic rather than pattern-matched
# on a registry's variable name -- a fixture can be held back by a deferral
# dict, by a different module's registry, or by a numeric cutoff, and only the
# module itself knows which. `(module, attributes that hold a fixture back,
# optional positive gate)`.
#
# The attributes are named rather than matched by pattern, because a name does
# not say which way a registry points: some hold fixtures back and others gate
# them in. Every name is resolved with `getattr`, so a rename fails loudly
# rather than shrinking the held-back set in silence.
_COLLECTORS: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    ("test_conformance", ("_DEFERRED_FIXTURES",), None),
    ("test_llm_provider", ("_DEFERRED_FIXTURES",), None),
    ("test_pipeline_utilities", ("_DEFERRED_FIXTURES",), None),
    ("test_prompt_management", ("_DEFERRED_FIXTURES",), None),
    (
        "test_observability",
        (
            "_DEFERRED_FIXTURES",
            "_UNIT_TESTED_FIXTURES",
            "_CONVENTION_ONLY_FIXTURES",
            "_LANGFUSE_HARNESS_FIXTURES",
        ),
        "_SUPPORTED_FIXTURES",
    ),
    ("test_observability_langfuse", (), None),
    ("test_retrieval_provider", ("_DEFERRED_FIXTURES",), None),
    ("test_checkpoint", ("_DEFERRED_FIXTURES",), None),
    ("test_state_migration", (), None),
    ("test_conformance_adapter", ("_DEFERRED_FIXTURES",), None),
)

# Fixtures a module outside `tests/conformance` drives, by the function whose
# `parametrize` names them.
#
# `test_llm_provider` defers fixtures to these functions by name, so subtracting
# its deferrals without adding these back marks RUNNING fixtures dormant -- and
# a dormant fixture is skipped by the read check, which is the direction that
# hides a vacuous case. Read off the parametrize decorators so the list cannot
# drift from what actually runs.
_UNIT_DRIVEN: tuple[tuple[str, str], ...] = (
    ("tests.unit.test_observability_otel", "test_call_level_retry_fixture_per_attempt_spans"),
    ("tests.unit.test_observability_otel", "test_call_level_reask_retry_fixture"),
)

# This module and the registry it reads: scanning them would let a key count as
# read because it is registered.
_SELF = frozenset({"test_case_vocabulary.py", "vocabulary.py"})

_BODY_KEYS = ("subgraph", "subgraph_with_idx", "subgraphs", "inner_subgraphs")


def _runner_sources() -> dict[str, str]:
    """Every runner module's source, keyed by path relative to the repo root."""
    # `rglob`, not `glob`: `harness/runtime/` exists and its README designates it
    # the future home of the fixture-executing code, so a directive reader landing
    # there would be invisible to the read check and to the collection scan. The
    # unit-side llm-provider runner is included for the same reason: it drives
    # the fixtures `test_llm_provider` defers to it by name.
    paths = sorted(_RUNNER_DIR.rglob("*.py")) + [_REPO_ROOT / "tests/unit/test_observability_otel.py"]
    kept = [p for p in paths if p.name not in _SELF]
    sources = {str(p.relative_to(_REPO_ROOT)): p.read_text() for p in kept}
    assert len(sources) == len(kept), (
        f"walked {len(kept)} runner files and kept {len(sources)} sources; a path collided"
    )
    return sources


# Derived from execution, not from mention. A hand-listed map misses owners as
# runners are added, and a scan for the capability name anywhere in a source
# goes wrong the other way, silently: it makes owners of `loader.py`'s registry
# tuple and of a gate-wiring test that builds a synthetic `tmp_path` tree and
# never opens the submodule. A too-broad owner vouches for a read that the real
# runner never performs.
def _runner_owners() -> dict[str, frozenset[str]]:
    """Which runner sources execute each capability's fixtures.

    A source owns a capability when it is a collector whose `_fixture_paths()`
    returns that capability's fixtures, or when it holds a path expression rooted
    at the capability's directory in the submodule. A source that loads one named
    fixture owns that fixture alone, and `_fixture_owners` reports it instead.
    """
    owners: dict[str, set[str]] = {cap: set() for cap in _RUN_DIRS}
    by_module = {name: f"tests/conformance/{name}.py" for name, _, _ in _COLLECTORS}

    for name, _, _ in _COLLECTORS:
        module = importlib.import_module(f".{name}", __package__)
        for path in cast("list[Path]", module._fixture_paths()):  # noqa: SLF001
            cap = path.parent.parent.name
            if cap in owners:
                owners[cap].add(by_module[name])

    for path, src in _runner_sources().items():
        for cap, fixture in _submodule_loads(src):
            if fixture is None and cap in owners:
                owners[cap].add(path)

    missing = sorted(cap for cap, paths in owners.items() if not paths)
    assert not missing, f"no runner executes {missing}; the ownership derivation is broken"
    return {cap: frozenset(paths) for cap, paths in owners.items()}


def _fixture_owners() -> dict[str, frozenset[str]]:
    """Runner sources that load one named fixture, keyed by fixture id."""
    owners: dict[str, set[str]] = defaultdict(set)
    for path, src in _runner_sources().items():
        for _, fixture in _submodule_loads(src):
            if fixture is not None:
                owners[fixture].add(path)
    return {fixture: frozenset(paths) for fixture, paths in owners.items()}


def _fixture_stem(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str) and expr.value.endswith(".yaml"):
        return expr.value.removesuffix(".yaml")
    return None


def _submodule_loads(src: str) -> list[tuple[str, str | None]]:
    """`(capability, fixture)` for each way a source reaches a capability's
    conformance directory: the fixture's id when it joins one fixture filename
    onto the directory, `None` when it reaches the directory any other way."""
    tree = ast.parse(src)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    loads: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        # Any `/`-joined path expression, not only an assignment: a load can be
        # written inline, and scanning assignments alone is the same
        # partial-input mistake this module exists to catch.
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        if isinstance(parents.get(node), ast.BinOp):
            continue
        literals = {
            sub.value
            for sub in ast.walk(node)
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
        }
        # Rooted at the submodule. A synthetic tree under `tmp_path` carries the
        # same capability names and the same "conformance" segment, and is not a
        # load -- `test_capability_gate_wiring` builds exactly that.
        if "openarmature-spec" not in literals or "conformance" not in literals:
            continue
        caps = literals & set(_RUN_DIRS)
        binding = parents.get(node)
        if isinstance(binding, ast.Assign):
            names = {t.id for t in binding.targets if isinstance(t, ast.Name)}
        elif isinstance(binding, ast.AnnAssign) and isinstance(binding.target, ast.Name):
            names = {binding.target.id}
        else:
            loads.extend((cap, _fixture_stem(node.right)) for cap in caps)
            continue
        # A directory bound to a name owns the whole capability unless every use
        # joins one fixture filename onto it. Owning the whole directory for a
        # one-fixture load would let that runner's string reads vouch for every
        # other fixture in it.
        for use in ast.walk(tree):
            if not (isinstance(use, ast.Name) and use.id in names and isinstance(use.ctx, ast.Load)):
                continue
            joined = parents.get(use)
            fixture = None
            if isinstance(joined, ast.BinOp) and isinstance(joined.op, ast.Div) and joined.left is use:
                fixture = _fixture_stem(joined.right)
            loads.extend((cap, fixture) for cap in caps)
    return loads


# The root counts: a fixture with no `cases:` list carries every directive
# there, and the document-root models that would forbid an unknown key are
# applied only by `test_fixture_parsing`, which parses `loader.CAPABILITIES`
# alone and skips its own deferrals.
def _containers(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The document root and each of its cases, as directive-bearing containers."""
    out = [doc]
    cases = doc.get("cases")
    if isinstance(cases, list):
        out.extend(cast("dict[str, Any]", c) for c in cast("list[Any]", cases) if isinstance(c, dict))
    return out


def _fixture_docs() -> list[tuple[str, str, dict[str, Any]]]:
    """`(capability, fixture, document)` for every fixture in a run directory."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for name in _RUN_DIRS:
        for path in sorted((_SPEC_ROOT / name / "conformance").glob("[0-9][0-9][0-9]-*.yaml")):
            doc: Any = yaml.safe_load(path.read_text())
            if isinstance(doc, dict):
                out.append((name, path.stem, cast("dict[str, Any]", doc)))
    return out


def _root_model_fields() -> set[str]:
    """Every field the document-root fixture models declare, as one set.

    A root container is governed by whichever root variant the discriminator
    picks, so the union is the right comparison: narrowing it per variant would
    re-implement the discriminator here and get it wrong.
    """
    from pydantic import BaseModel  # noqa: PLC0415

    models = [
        cast("type[BaseModel]", getattr(fixture_models, name))
        for name in dir(fixture_models)
        if isinstance(getattr(fixture_models, name), type)
        and issubclass(cast("type", getattr(fixture_models, name)), BaseModel)
        and name.endswith("Fixture")
    ]
    assert len(models) >= 4, f"found only {len(models)} root fixture models; the scan is wrong"
    fields: set[str] = set()
    for model in models:
        fields |= set(model.model_fields)
    return fields


def _declared_keys() -> list[tuple[str, str, str, str]]:
    """`(capability, fixture, where, key)` for every directive the corpus declares."""
    out: list[tuple[str, str, str, str]] = []
    for capability, fixture, doc in _fixture_docs():
        for container in _containers(doc):
            where = "<root>" if container is doc else str(container.get("name", "<unnamed>"))
            out.extend((capability, fixture, where, key) for key in container)
    return out


# `(source path, module-level name)`: a function or class with everything nested
# in it, or a module-level assignment's value. A read outside all of them runs
# at import rather than against a fixture, so it is recorded under `<module>`:
# it counts for its own file's runner, and no other runner reaches it.
_Unit = tuple[str, str]


def _runner_trees() -> dict[str, ast.Module]:
    return {path: ast.parse(src) for path, src in _runner_sources().items()}


def _unit_names(stmt: ast.stmt) -> tuple[str, ...]:
    """The module-level names one top-level statement defines."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (stmt.name,)
    if isinstance(stmt, ast.Assign):
        return tuple(target.id for target in stmt.targets if isinstance(target, ast.Name))
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return (stmt.target.id,)
    return ()


def _read_keys(node: ast.AST) -> list[str]:
    """The string keys one node reads: a subscript, a ``.get`` / ``.pop``, or a
    membership test."""
    hits: list[Any] = []
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        hits.append(node.slice.value)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get", "pop"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ):
        hits.append(node.args[0].value)
    elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant):
        hits.append(node.left.value)
    return [hit for hit in hits if isinstance(hit, str)]


# Matched on the syntax tree, because a substring match cannot tell a read from
# a mention: a key named in a comment, or as some other key's value, would count
# as read.
def _read_positions() -> dict[str, set[_Unit]]:
    """Where each directive name is used to READ a mapping: the source and the
    module-level definition holding each read.

    Only a subscript, a ``.get`` / ``.pop``, or a membership test counts. A key
    reached any other way is declared in `READ_VIA` instead.
    """
    found: dict[str, set[_Unit]] = defaultdict(set)
    for path, tree in _runner_trees().items():
        for stmt in tree.body:
            units = _unit_names(stmt) or ("<module>",)
            for node in ast.walk(stmt):
                for key in _read_keys(node):
                    found[key].update((path, unit) for unit in units)
    return found


def _module_name(path: str) -> str:
    parts = path.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: str, tree: ast.Module) -> dict[str, tuple[str, str | None]]:
    """Each name a source binds by import, in any scope, as `(module, attribute)`.

    The attribute is ``None`` when the name binds the module itself.
    """
    module = _module_name(path)
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    bound: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.partition(".")[0]
                bound[name] = (alias.name if alias.asname else name, None)
        elif isinstance(node, ast.ImportFrom):
            base = package
            for _ in range(node.level - 1):
                base = base.rpartition(".")[0]
            source = ".".join(part for part in (base if node.level else "", node.module or "") if part)
            for alias in node.names:
                bound[alias.asname or alias.name] = (source, alias.name)
    return bound


class _Namespace:
    """Resolves a name in a runner source to the module-level definition it
    refers to, following imports between runner sources."""

    def __init__(self, trees: dict[str, ast.Module]) -> None:
        self.paths = {_module_name(path): path for path in trees}
        self.units = {
            path: {n for stmt in tree.body for n in _unit_names(stmt)} for path, tree in trees.items()
        }
        self.imports = {path: _imports(path, tree) for path, tree in trees.items()}

    def lookup(self, module: str, name: str, depth: int = 0) -> _Unit | str | None:
        """`name` as an attribute of `module`: a definition, a submodule's
        dotted name, or ``None`` outside the runner sources."""
        if f"{module}.{name}" in self.paths:
            return f"{module}.{name}"
        path = self.paths.get(module)
        # A re-export chain longer than this is a cycle.
        if path is None or depth > 8:
            return None
        if name in self.units[path]:
            return (path, name)
        if name in self.imports[path]:
            return self.imported(path, name, depth + 1)
        return None

    def imported(self, path: str, name: str, depth: int = 0) -> _Unit | str | None:
        module, attr = self.imports[path][name]
        if attr is None:
            return module if module in self.paths else None
        return self.lookup(module, attr, depth)

    def resolve(self, path: str, expr: ast.expr) -> _Unit | str | None:
        if isinstance(expr, ast.Name):
            if expr.id in self.units[path]:
                return (path, expr.id)
            if expr.id in self.imports[path]:
                return self.imported(path, expr.id)
            return None
        if isinstance(expr, ast.Attribute) and isinstance(base := self.resolve(path, expr.value), str):
            return self.lookup(base, expr.attr)
        return None


def _annotation_ids(tree: ast.AST) -> set[int]:
    """Every node inside a type annotation."""
    roots: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            roots.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            roots.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            roots.append(node.annotation)
    return {id(sub) for root in roots for sub in ast.walk(root)}


def _runner_reach() -> dict[str, frozenset[_Unit]]:
    """The module-level definitions each runner source reaches by name, directly
    or through a definition it reaches."""
    trees = _runner_trees()
    namespace = _Namespace(trees)
    statements: dict[_Unit, list[ast.stmt]] = defaultdict(list)
    for path, tree in trees.items():
        for stmt in tree.body:
            for name in _unit_names(stmt):
                statements[(path, name)].append(stmt)
    # An annotation names a type without running any of it.
    annotations = {path: _annotation_ids(tree) for path, tree in trees.items()}

    def referenced(path: str, nodes: Sequence[ast.AST]) -> set[_Unit]:
        found: set[_Unit] = set()
        for node in nodes:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, (ast.Name, ast.Attribute))
                    and isinstance(sub.ctx, ast.Load)
                    and id(sub) not in annotations[path]
                    and isinstance(unit := namespace.resolve(path, sub), tuple)
                ):
                    found.add(unit)
        return found

    edges = {unit: referenced(unit[0], stmts) for unit, stmts in statements.items()}
    reach: dict[str, frozenset[_Unit]] = {}
    for path, tree in trees.items():
        seen = referenced(path, [tree])
        todo = list(seen)
        while todo:
            for unit in edges[todo.pop()] - seen:
                seen.add(unit)
                todo.append(unit)
        reach[path] = frozenset(seen)
    return reach


# A shared helper's reads count only for the runners that reach it. `adapter.py`
# builds the graph for most capabilities, and its reads of `from` and `to` say
# nothing about retrieval-provider, whose runner drives the entry node alone and
# never builds one.
def _owner_reads() -> Callable[[str, frozenset[str]], bool]:
    """Whether a directive name is read by one of a set of runner sources: in
    the source itself, or in a definition the source reaches."""
    positions = _read_positions()
    reach = _runner_reach()
    reached_by: dict[frozenset[str], frozenset[_Unit]] = {}

    def read(key: str, owning: frozenset[str]) -> bool:
        if owning not in reached_by:
            reached_by[owning] = frozenset().union(*(reach[path] for path in owning))
        reached = reached_by[owning]
        return any(site[0] in owning or site in reached for site in positions.get(key, ()))

    return read


# Computed from each module's own `_fixture_paths()` rather than pattern-matched
# on registry names, because "in a `_DEFERRED` dict" does not mean "does not
# run": observability defers fixtures with reasons that name the Langfuse runner,
# which executes them; llm-provider defers fixtures to `tests/unit`;
# `test_fixture_parsing`'s registry defers parsing only; and
# `test_pipeline_utilities` also holds fixtures back with a numeric cutoff that
# no registry records.
def _executed_fixture_ids() -> set[str]:
    """Every fixture id some runner actually collects and does not hold back,
    including those a unit-side runner drives."""
    runs: set[str] = set()
    for name, held_back_attrs, gate_attr in _COLLECTORS:
        module = importlib.import_module(f".{name}", __package__)
        collected = {p.stem for p in cast("list[Path]", module._fixture_paths())}  # noqa: SLF001
        held_back: set[str] = set()
        for attr in held_back_attrs:
            # No default: a renamed registry raises AttributeError here rather
            # than quietly shrinking the held-back set. Empty is legitimate: a
            # runner may defer nothing.
            held_back |= set(getattr(module, attr))
        if gate_attr is not None:
            gate = getattr(module, gate_attr)
            assert gate, f"{name}.{gate_attr} is empty; the positive gate is not what it was"
            collected &= set(gate)
        runs |= collected - held_back
    runs |= _unit_driven_fixture_ids()
    assert len(runs) > 200, f"computed only {len(runs)} executed fixtures; the collection scan is broken"
    return runs


def _unit_driven_fixture_ids() -> set[str]:
    """Fixture ids named by a `parametrize` on a unit-side conformance runner."""
    ids: set[str] = set()
    for module_path, func_name in _UNIT_DRIVEN:
        module = importlib.import_module(module_path)
        tree = ast.parse(inspect.getsource(getattr(module, func_name)))
        named = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        found = {n for n in named if n[:3].isdigit() and n[3:4] == "-"}
        assert found, f"{module_path}.{func_name} names no fixture ids; the scan is broken"
        ids |= found
    return ids


def test_every_declared_key_is_recognized() -> None:
    # Section 8.2: a directive outside the recognized vocabulary must be
    # rejected. The failure names the key and the fixture and case carrying it,
    # which is what section 9 asks of the raise.
    case_fields = set(CaseSpec.model_fields)
    root_fields = _root_model_fields()
    declared = _declared_keys()
    assert len({f for _, f, _, _ in declared}) > 400, (
        f"walked only {len({f for _, f, _, _ in declared})} fixtures; the corpus walk is incomplete"
    )
    unrecognized = [
        (cap, fixture, where, key)
        for cap, fixture, where, key in declared
        if key not in RECOGNIZED_DIRECTIVES and key not in (root_fields if where == "<root>" else case_fields)
    ]
    assert not unrecognized, (
        "directive(s) outside the recognized vocabulary:\n"
        + "\n".join(f"  {c}/{f} at {w} carries {k!r}" for c, f, w, k in sorted(unrecognized))
        + "\nEither wire the directive and add it to RECOGNIZED_DIRECTIVES, or defer the fixture."
    )


def test_read_via_declarations_resolve() -> None:
    # READ_VIA is the escape hatch for a directive reached by attribute access or
    # through a key list, so it has to be verified rather than believed: a field
    # that stops existing, or a key that leaves the collection, must fail here
    # instead of quietly widening what counts as read.
    for key, claim in sorted(READ_VIA.items()):
        kind, _, target = claim.partition(":")
        module_path, _, attr = target.rpartition(".")
        if kind == "model":
            model_mod, _, model_name = module_path.rpartition(".")
            model = getattr(importlib.import_module(model_mod), model_name)
            assert attr in model.model_fields, f"{key!r}: {model_name} has no field {attr!r}"
        elif kind == "keylist":
            collection = getattr(importlib.import_module(module_path), attr)
            assert key in collection, f"{key!r}: not present in {attr}"
        else:  # pragma: no cover - a typo in the claim form
            pytest.fail(f"{key!r}: unknown READ_VIA form {kind!r}")


def test_every_declared_key_is_read_by_a_runner_that_owns_it() -> None:
    # The state a corpus-derived allowlist cannot see: a directive declared by a
    # running fixture that nothing reads, so the case passes with the knob at its
    # default while counting as coverage.
    #
    # Scoped per capability. A flat pool over every runner lets a key read by one
    # capability's runner count for a fixture whose own runner ignores it, which
    # is the same mis-routing defect one level up.
    read = _owner_reads()
    owners = _runner_owners()
    fixture_owners = _fixture_owners()
    exempt = set(UNAPPLIED_PENDING_DEFERRAL) | set(READ_VIA)
    # Matched on the whole triple, not the key. `UNAPPLIED_PENDING_CASE_DEFERRAL`
    # is keyed by `(fixture, case)` exactly so the exemption stays narrow;
    # collapsing it to the key would let one skipped case excuse every other
    # declaration of that directive, including from cases that run.
    case_exempt = {
        (fixture, case, key)
        for key, pairs in UNAPPLIED_PENDING_CASE_DEFERRAL.items()
        for fixture, case in pairs
    }
    # Fields the fixture models declare are out of scope, and the reason is weaker
    # than "they are read through the model": only `test_prompt_management` parses
    # into a fixture model at all, so most modelled fields are not read that way
    # either. The exclusion is a real hole rather than a technicality --
    # `expected_wire_request`, `checkpointer` and `mock_llm` are modelled and
    # behavioural, and deleting every read of one would pass here. Some others
    # are genuinely inert: `description` is prose, `initial_state` means nothing
    # to a runner driving a provider with no engine. Separating the four
    # categories is the tracked task; a half-classification here fails the inert
    # ones and still misses the live ones.
    modelled = set(CaseSpec.model_fields) | _root_model_fields()
    executed = _executed_fixture_ids()
    unread: list[str] = []
    for cap, fixture, where, key in _declared_keys():
        # A dormant fixture declaring a directive its own runner ignores is not a
        # vacuous pass, because nothing passes. Only a RUNNING declarer can hide
        # one, and that is the whole claim.
        if (
            key in exempt
            or (fixture, where, key) in case_exempt
            or key in modelled
            or fixture not in executed
        ):
            continue
        if read(key, owners[cap] | fixture_owners.get(fixture, frozenset())):
            continue
        unread.append(f"  {cap}/{fixture} at {where} declares {key!r}; no owning runner reads it")
    assert not unread, (
        "directive(s) declared by a fixture whose own runner never reads them:\n"
        + "\n".join(sorted(set(unread)))
        + "\nWire the directive in that capability's runner, or record it against a deferral."
    )


def test_a_helper_read_counts_only_for_the_runners_that_reach_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reach is resolved by name through imports, so the way it goes wrong in
    # silence is a runner counted as reaching a helper it never calls. Every
    # runner below imports from the same helper module; only the ones that
    # reach the reading definition may count.
    helper = "tests/conformance/harness/helper.py"
    sources = {
        helper: (
            "class Graph:\n"
            "    def edges(self, spec):\n"
            "        return spec['edges']\n"
            "def build(spec):\n"
            "    return spec['nodes']\n"
            "def unrelated():\n"
            "    return None\n"
        ),
        "tests/conformance/harness/__init__.py": "from .helper import build as build\n",
        "tests/conformance/test_annotation.py": (
            "from .harness.helper import Graph\ndef test(graph: Graph):\n    return graph\n"
        ),
    }
    # Each runner's import, and the one line its test runs.
    runners = {
        "test_direct": ("from .harness.helper import build", "build(spec)"),
        "test_module": ("from .harness import helper", "helper.build(spec)"),
        "test_reexport": ("from .harness import build", "build(spec)"),
        "test_other_name": ("from .harness.helper import unrelated", "unrelated()"),
        "test_class": ("from .harness.helper import Graph", "Graph().edges(spec)"),
    }
    for name, (imports, body) in runners.items():
        sources[f"tests/conformance/{name}.py"] = f"{imports}\ndef test(spec):\n    {body}\n"
    monkeypatch.setattr(f"{__name__}._runner_sources", lambda: sources)
    read = _owner_reads()
    for runner in ("test_direct", "test_module", "test_reexport"):
        assert read("nodes", frozenset({f"tests/conformance/{runner}.py"})), runner
    assert not read("nodes", frozenset({"tests/conformance/test_other_name.py"}))
    assert not read("edges", frozenset({"tests/conformance/test_annotation.py"}))
    assert read("edges", frozenset({"tests/conformance/test_class.py"}))
    assert read("nodes", frozenset({helper}))


def test_exempt_keys_are_declared_only_by_fixtures_that_do_not_run() -> None:
    # The tie, derived rather than hand-listed. Naming a subset of a key's
    # declarers exempts the key globally while leaving the unnamed declarers free
    # to run, so the dict entry is documentation and the corpus is the input.
    executed = _executed_fixture_ids()
    declarers: dict[str, set[str]] = defaultdict(set)
    for _, fixture, _, key in _declared_keys():
        declarers[key].add(fixture)
    live = [
        (key, fixture)
        for key in UNAPPLIED_PENDING_DEFERRAL
        for fixture in sorted(declarers.get(key, set()) & executed)
    ]
    assert not live, (
        "key(s) exempted as unread while a RUNNING fixture declares them:\n"
        + "\n".join(f"  {k!r} declared by {f}, which executes" for k, f in sorted(live))
        + "\nWire the directive, or the exemption is hiding a vacuous case."
    )


@pytest.mark.parametrize("key", sorted(UNAPPLIED_PENDING_DEFERRAL))
def test_named_deferrals_are_real_declarers(key: str) -> None:
    # The other half: an entry naming a fixture that does not declare the key is
    # a stale citation, and it is the citation that makes the exemption readable.
    declarers = {f for _, f, _, k in _declared_keys() if k == key}
    named = set(UNAPPLIED_PENDING_DEFERRAL[key])
    assert named <= declarers, f"{key!r} cites {sorted(named - declarers)}, which do not declare it"


def test_case_deferral_exemptions_name_cases_that_are_actually_skipped() -> None:
    # The per-case counterpart: `session_id` rides one case of a fixture that
    # otherwise runs, so a fixture-level check cannot see it.
    from . import test_observability  # noqa: PLC0415

    for key, pairs in sorted(UNAPPLIED_PENDING_CASE_DEFERRAL.items()):
        for fixture, case in pairs:
            driver = "_run_fixture_" + fixture.split("-")[0]
            tree = ast.parse(inspect.getsource(getattr(test_observability, driver)))
            skipped = {
                sub.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign | ast.AnnAssign)
                and any(
                    isinstance(t, ast.Name) and t.id == "_deferred_cases"
                    for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                )
                for sub in ast.walk(node)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            }
            assert case in skipped, (
                f"{key!r} is exempted because {fixture}::{case} does not run, but {driver} "
                f"skips only {sorted(skipped)}. Wire the directive or drop the exemption."
            )


# Recursive: bodies also sit under a case's `inner_subgraphs` and inside its
# `graph:` wrapper. `inner_subgraphs` is the same shape by construction, since
# the Langfuse runner renames it onto `subgraphs` before use.
def _subgraph_bodies() -> list[tuple[str, str]]:
    """`(fixture, key)` for each key of every subgraph body, at any depth, in
    every fixture in a run directory."""
    out: list[tuple[str, str]] = []

    def visit(fixture: str, node: Any) -> None:
        if isinstance(node, list):
            for item in cast("list[Any]", node):
                visit(fixture, item)
            return
        if not isinstance(node, dict):
            return
        mapping = cast("dict[str, Any]", node)
        for key, value in mapping.items():
            if key in _BODY_KEYS and isinstance(value, dict):
                body = cast("dict[str, Any]", value)
                # Singular forms ARE a body; plural forms map name -> body.
                bodies = (
                    [body]
                    if key in {"subgraph", "subgraph_with_idx"}
                    else [cast("dict[str, Any]", b) for b in body.values() if isinstance(b, dict)]
                )
                for one in bodies:
                    out.extend((fixture, k) for k in one)
            visit(fixture, value)

    for _, fixture, doc in _fixture_docs():
        visit(fixture, doc)
    return out


def test_every_subgraph_level_key_is_modelled() -> None:
    # `SubgraphDefinition` allows extras like `CaseSpec` does, so the same
    # silent-accept path exists one level down. This finds nothing today: no
    # fixture declares a subgraph key outside the model. It is here so the first
    # one that does is rejected rather than dropped.
    observed = _subgraph_bodies()
    # Non-vacuity on the INPUT, with the floor close to what the walk reaches
    # rather than loose: a walk of two enumerated levels misses every body under
    # `inner_subgraphs` and `graph:`, and only a close floor catches it.
    assert len({f for f, _ in observed}) >= 71, (
        f"walked {len(observed)} subgraph keys in {len({f for f, _ in observed})} fixtures; "
        "the traversal is not reaching them all"
    )
    modelled = set(SubgraphDefinition.model_fields)
    unknown = sorted({(f, k) for f, k in observed if k not in modelled})
    assert not unknown, (
        "subgraph-level key(s) outside SubgraphDefinition:\n"
        + "\n".join(f"  {f} declares {k!r}" for f, k in unknown)
        + "\nModel the field, or defer the fixture."
    )


def test_every_conformance_directory_is_run_or_declared_unimplemented() -> None:
    # The coarse counterpart: a whole capability can go unread the same way a
    # directive can. This is what makes a new capability directory arriving at a
    # pin bump an error rather than a silence.
    with_fixtures = {
        d.name
        for d in sorted(_SPEC_ROOT.iterdir())
        if d.is_dir() and any((d / "conformance").glob("[0-9][0-9][0-9]-*.yaml"))
    }
    # The two registries assert opposite things -- "does not exist here and
    # nothing is coming" against "the fixtures are real and adoption is in
    # flight" -- so naming a capability in both is a contradiction rather than
    # redundancy. They are unioned below, which would absorb the pair in silence
    # and fold back together the distinction they were split to keep.
    both = sorted(set(UNIMPLEMENTED_CAPABILITIES) & set(PENDING_ADOPTION))
    assert not both, (
        f"{both} is declared both unimplemented and pending adoption, which cannot both "
        "be true. Keep the one that describes the capability."
    )
    declared = set(UNIMPLEMENTED_CAPABILITIES) | set(PENDING_ADOPTION)
    unaccounted = sorted(with_fixtures - set(_RUN_DIRS) - declared)
    assert not unaccounted, (
        f"capability directory/ies ship fixtures but are neither run nor declared: "
        f"{unaccounted}. Wire a runner, or record why not in UNIMPLEMENTED_CAPABILITIES "
        "(the capability does not exist here) or PENDING_ADOPTION (its adoption is in flight)."
    )
    # Both declarations are claims about NOT running something, so both go stale
    # the moment a runner lands and both have to say so.
    stale = sorted(declared & set(_RUN_DIRS))
    assert not stale, f"{stale} is declared as not running while a runner executes it. Drop the entry."
    # And both go stale the other way, which is the direction a subtraction
    # cannot see: an entry naming a directory that ships no fixtures removes a
    # name that was never in the set, so a misspelling declares nothing while
    # reading as coverage. `PENDING_ADOPTION` is the more exposed of the two,
    # being temporary by design.
    orphaned = sorted(declared - with_fixtures)
    assert not orphaned, (
        f"{orphaned} is declared as not running, but no such capability directory ships "
        "fixtures. Correct the spelling, or drop an entry whose capability is gone."
    )


@pytest.mark.parametrize("capability", sorted(set(UNIMPLEMENTED_CAPABILITIES) | set(PENDING_ADOPTION)))
def test_capabilities_declared_as_not_running_are_not_referenced_by_any_runner(
    capability: str,
) -> None:
    # Declaring a capability as not running drops its whole directory out of every
    # walk above, so the claim has to be checked rather than taken. Without this,
    # moving a live capability into either dict silently removes its fixtures from
    # the corpus while every assertion stays green.
    #
    # Over BOTH registries, because the other two staleness checks have conditions
    # a runner can land without meeting: `stale` needs the name in `_RUN_DIRS`, and
    # `orphaned` needs the directory to ship no fixtures. Runner code arriving
    # before its `_RUN_DIRS` entry satisfies neither, and this is the only guard
    # that sees it. `PENDING_ADOPTION` is the one that will hit this, since its
    # whole purpose is to name a capability whose runner is being written.
    referencing = sorted(path for path, src in _runner_sources().items() if f'"{capability}"' in src)
    # Name the registry the entry is actually in: this runs over both, and
    # telling someone their pending-adoption entry is "declared unimplemented"
    # sends them to the wrong dict.
    registry = (
        "UNIMPLEMENTED_CAPABILITIES" if capability in UNIMPLEMENTED_CAPABILITIES else "PENDING_ADOPTION"
    )
    assert not referencing, (
        f"{capability!r} is declared in {registry} as not running, but is named in {referencing}. "
        "Either it has a runner and belongs in _RUN_DIRS, or the reference is stale."
    )
