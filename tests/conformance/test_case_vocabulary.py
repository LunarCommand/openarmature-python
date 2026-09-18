# Spec: conformance-adapter section 8.2 + section 5 *Definition homes* (proposal
# 0120, spec v0.113.0). Section 9 requires the adapter to raise rather than skip,
# and to surface the offending directive and its location; the token itself is
# not observable surface, per the ruling in coord thread
# `proposal-0120-0123-adapter-obligations`.

"""Every directive is recognized, and every recognized one is actually read.

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
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from .harness import fixtures as fixture_models
from .harness.fixtures import CaseSpec, SubgraphDefinition
from .harness.vocabulary import (
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
)

# Modules whose own `_fixture_paths()` is the authority on what they collect.
# Paired with the capability each drives, so "runs" is computed from the
# collection logic rather than pattern-matched on a registry's variable name --
# a fixture can be held back by a deferral dict, by a different module's
# registry, or by a numeric cutoff, and only the module itself knows which.
# `(module, attributes that hold a fixture back, optional positive gate)`.
#
# Named explicitly, and every name is resolved with `getattr` so a rename fails
# loudly rather than shrinking the held-back set in silence. A prefix guess is
# what the previous version used, and it missed `_CONVENTION_ONLY_FIXTURES`
# outright; `test_observability` alone skips through four registries and a
# membership gate.
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
)

# Fixtures a module outside `tests/conformance` drives, by the function whose
# `parametrize` names them.
#
# `test_llm_provider` defers 056-058 and 061-066 with reasons pointing here, so
# subtracting its deferrals without adding these back marks nine RUNNING fixtures
# dormant -- and a dormant fixture is skipped by the read check, which is the
# direction that hides a vacuous case. Read off the parametrize decorators so the
# list cannot drift from what actually runs.
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
    # unit-side llm-provider runner is included for the same reason -- it drives
    # fixtures 056-058, which `test_llm_provider` defers to it by name.
    paths = sorted(_RUNNER_DIR.rglob("*.py")) + [_REPO_ROOT / "tests/unit/test_observability_otel.py"]
    kept = [p for p in paths if p.name not in _SELF]
    sources = {str(p.relative_to(_REPO_ROOT)): p.read_text() for p in kept}
    assert len(sources) == len(kept), (
        f"walked {len(kept)} runner files and kept {len(sources)} sources; a path collided"
    )
    return sources


def _runner_owners() -> dict[str, frozenset[str]]:
    """Which runner sources execute each capability's fixtures.

    Derived rather than hand-listed. A map maintained by hand claimed to
    enumerate every execution path and was wrong three times in the same
    direction -- `retrieval-provider` has its own glob outside
    `loader.CAPABILITIES`, `tests/unit` drives llm-provider fixtures, and
    `test_observability` reaches across into pipeline-utilities for fixture 031.
    A module that names a capability directory is executing from it, so the
    source is the map.
    """
    owners: dict[str, set[str]] = {cap: set() for cap in _RUN_DIRS}
    for path, src in _runner_sources().items():
        for cap in _RUN_DIRS:
            if f'"{cap}"' in src:
                owners[cap].add(path)
    missing = sorted(cap for cap, paths in owners.items() if not paths)
    assert not missing, f"no runner source names {missing}; the ownership scan is broken"
    return {cap: frozenset(paths) for cap, paths in owners.items()}


def _containers(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The document root and each of its cases, as directive-bearing containers.

    The root counts. 140 of the 448 fixtures in `_RUN_DIRS` carry their
    directives there with no `cases:` list at all, and the document-root models
    that would forbid an unknown key are applied only by `test_fixture_parsing`,
    which skips retrieval-provider and its own deferrals.
    """
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


def _read_positions() -> dict[str, set[str]]:
    """Where each directive name is used to READ a mapping, by source path.

    A bare substring match cannot tell a read from a mention. `manager` occurs
    twice in runner source and neither is a read of the case-level block -- both
    concern a call's `target:` value -- so the scan it satisfies proves nothing.
    Only subscript, ``.get`` / ``.pop``, and membership count here; anything
    reached another way is declared in `READ_VIA` and verified there.
    """
    found: dict[str, set[str]] = defaultdict(set)
    for path, src in _runner_sources().items():
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover - a runner that does not parse fails elsewhere
            continue
        for node in ast.walk(tree):
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
            for hit in hits:
                if isinstance(hit, str):
                    found[hit].add(path)
    return found


def _executed_fixture_ids() -> set[str]:
    """Every fixture id some runner actually collects and does not defer.

    Computed from each module's own `_fixture_paths()` rather than pattern-
    matched on registry names. "In a `_DEFERRED` dict" does not mean "does not
    run": four observability fixtures are deferred with reasons that name the
    Langfuse runner, which executes them; three llm-provider fixtures are
    deferred to `tests/unit`; `test_fixture_parsing`'s registry defers PARSING
    only; and `test_pipeline_utilities` holds fixtures back with a numeric
    cutoff and no registry at all.
    """
    runs: set[str] = set()
    for name, held_back_attrs, gate_attr in _COLLECTORS:
        module = importlib.import_module(f".{name}", __package__)
        collected = {p.stem for p in cast("list[Path]", module._fixture_paths())}  # noqa: SLF001
        held_back: set[str] = set()
        for attr in held_back_attrs:
            # No default: a renamed registry raises AttributeError here rather
            # than quietly shrinking the held-back set. Empty is legitimate --
            # retrieval-provider defers nothing at the moment.
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
    positions = _read_positions()
    owners = _runner_owners()
    shared = {"harness/", "adapter.py", "middleware_seam.py"}
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
    # Fields the fixture models declare are out of scope here, and the reason is
    # weaker than "they are read through the model". Only `test_prompt_management`
    # parses into a fixture model at all; every other runner takes the raw mapping
    # from `yaml.safe_load`, so most modelled fields are not read through a model
    # either. Some are genuinely inert -- `description` is prose, `initial_state`
    # means nothing to a runner that drives a provider with no engine -- and some
    # are live gaps. Telling those apart is the task this exclusion is tracked
    # under; a half-classification here would grant the wrong ones a pass.
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
        seen = positions.get(key, set())
        if seen & owners[cap] or any(any(m in src for m in shared) for src in seen):
            continue
        unread.append(f"  {cap}/{fixture} at {where} declares {key!r}; no owning runner reads it")
    assert not unread, (
        "directive(s) declared by a fixture whose own runner never reads them:\n"
        + "\n".join(sorted(set(unread)))
        + "\nWire the directive in that capability's runner, or record it against a deferral."
    )


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


def _subgraph_bodies() -> list[tuple[str, str]]:
    """`(fixture, key)` for every subgraph body anywhere in a running fixture.

    Recursive rather than two enumerated container levels. Bodies also sit under
    a case's `inner_subgraphs` and inside its `graph:` wrapper, and
    `inner_subgraphs` is the same shape by construction -- the Langfuse runner
    renames it onto `subgraphs` before use.
    """
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
    # Non-vacuity on the INPUT, at the post-fix count rather than a loose floor.
    # A guard set well below what the walk reaches tolerates losing a whole
    # container shape, which is how the two-level version passed while missing
    # `inner_subgraphs` and `graph:` entirely.
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
    unaccounted = sorted(with_fixtures - set(_RUN_DIRS) - set(UNIMPLEMENTED_CAPABILITIES))
    assert not unaccounted, (
        f"capability directory/ies ship fixtures but are neither run nor declared "
        f"unimplemented: {unaccounted}. Wire a runner, or record why not in "
        "UNIMPLEMENTED_CAPABILITIES."
    )
    stale = sorted(set(UNIMPLEMENTED_CAPABILITIES) & set(_RUN_DIRS))
    assert not stale, f"UNIMPLEMENTED_CAPABILITIES still names {stale}, which has a runner. Drop it."


@pytest.mark.parametrize("capability", sorted(UNIMPLEMENTED_CAPABILITIES))
def test_unimplemented_capabilities_are_not_referenced_by_any_runner(capability: str) -> None:
    # Declaring a capability unimplemented drops its whole directory out of every
    # walk above, so the claim has to be checked rather than taken. Without this,
    # moving a live capability into that dict silently removes its fixtures from
    # the corpus while every assertion stays green.
    referencing = sorted(path for path, src in _runner_sources().items() if f'"{capability}"' in src)
    assert not referencing, (
        f"{capability!r} is declared unimplemented but is named in {referencing}. "
        "Either it has a runner and belongs in _RUNNER_OWNERS, or the reference is stale."
    )
