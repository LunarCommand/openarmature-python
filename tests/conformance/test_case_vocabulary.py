# Spec: conformance-adapter §8.2 + §5 *Definition homes* (proposal 0120, spec
# v0.113.0). §9 requires the adapter to raise rather than skip, and to surface
# the offending directive and its location; the token itself is not observable
# surface, per the ruling in coord thread
# `proposal-0120-0123-adapter-obligations`.

"""Every case-level directive is recognized, and every recognized one is read.

`extra="forbid"` cannot carry this. It applies at the fixture's document root,
while `CaseSpec` and `SubgraphDefinition` both allow extras, and the runners
read raw YAML rather than the typed model at all -- so the model's config does
not reach the behaviour §8.2 is about. These are repository checks over the
corpus instead, which no runner can forget to call.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from .harness.fixtures import CaseSpec, SubgraphDefinition
from .harness.vocabulary import (
    RECOGNIZED_CASE_KEYS,
    UNAPPLIED_PENDING_CASE_DEFERRAL,
    UNAPPLIED_PENDING_DEFERRAL,
    UNIMPLEMENTED_CAPABILITIES,
)

_SPEC_ROOT = Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec"

# Directories with a runner. Wider than `loader.CAPABILITIES`, which enumerates
# only what the shared `discover_fixtures` yields: retrieval-provider has its own
# `CONFORMANCE_DIR` and glob, so its 53 fixtures execute without appearing there.
_RUN_DIRS = (
    "graph-engine",
    "llm-provider",
    "pipeline-utilities",
    "observability",
    "prompt-management",
    "retrieval-provider",
)

_RUNNER_DIR = Path(__file__).resolve().parent


def _case_keys() -> list[tuple[str, str, str]]:
    """Every `(fixture, case, key)` a running fixture declares at case level."""
    out: list[tuple[str, str, str]] = []
    for name in _RUN_DIRS:
        for path in sorted((_SPEC_ROOT / name / "conformance").glob("[0-9][0-9][0-9]-*.yaml")):
            doc: Any = yaml.safe_load(path.read_text())
            if not isinstance(doc, dict):
                continue
            cases: Any = cast("dict[str, Any]", doc).get("cases")
            if not isinstance(cases, list):
                continue
            for raw in cast("list[Any]", cases):
                if not isinstance(raw, dict):
                    continue
                case = cast("dict[str, Any]", raw)
                case_name = str(case.get("name", "<unnamed>"))
                for key in case:
                    out.append((path.stem, case_name, key))
    return out


def _runner_sources() -> dict[str, str]:
    return {
        p.name: p.read_text()
        for p in [*sorted(_RUNNER_DIR.glob("*.py")), *sorted((_RUNNER_DIR / "harness").glob("*.py"))]
        if p.name not in {"test_case_vocabulary.py", "vocabulary.py"}
    }


def test_every_case_level_key_is_recognized() -> None:
    # §8.2: a directive outside the recognized vocabulary must be rejected. The
    # failure names the key and the fixture and case carrying it, which is what
    # §9 asks of the raise.
    modelled = set(CaseSpec.model_fields)
    unrecognized = [
        (fixture, case, key)
        for fixture, case, key in _case_keys()
        if key not in modelled and key not in RECOGNIZED_CASE_KEYS
    ]
    assert not unrecognized, (
        "case-level directive(s) outside the recognized vocabulary:\n"
        + "\n".join(f"  {f}::{c} carries {k!r}" for f, c, k in sorted(unrecognized))
        + "\nEither wire the directive and add it to RECOGNIZED_CASE_KEYS, or defer the fixture."
    )


def test_every_recognized_key_is_read_by_some_runner() -> None:
    # The state a corpus-derived allowlist cannot see: recognized, declared by a
    # running fixture, and read by nothing, so the case passes with the knob at
    # its default while counting as coverage.
    sources = _runner_sources()
    exempt = set(UNAPPLIED_PENDING_DEFERRAL) | set(UNAPPLIED_PENDING_CASE_DEFERRAL)
    unread = sorted(
        key
        for key in RECOGNIZED_CASE_KEYS
        if key not in exempt and not any(f'"{key}"' in src or f"'{key}'" in src for src in sources.values())
    )
    assert not unread, (
        f"recognized case-level directive(s) no runner reads: {unread}. "
        "A fixture declaring one passes with the knob at its default. Wire it, or record "
        "it in UNAPPLIED_PENDING_DEFERRAL against the deferral that justifies it."
    )


# Module-level registries that hold back a fixture. Matched by name, and the
# prefixes are listed rather than guessed at: a runner that grows a registry
# under some other name makes this check report its fixtures as running, which
# fails an exemption loudly instead of quietly accepting one.
_DEFERRAL_REGISTRY_PREFIXES = ("_DEFERRED", "_CONVENTION_ONLY")


def _deferred_fixture_ids() -> set[str]:
    """Every fixture id some runner module holds back, read off its registries."""
    # Read from the modules rather than re-listed here, so an un-deferral cannot
    # leave a stale exemption standing.
    ids: set[str] = set()
    for src in _runner_sources().values():
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(t, ast.Name) and t.id.startswith(_DEFERRAL_REGISTRY_PREFIXES) for t in targets
            ):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    ids.add(sub.value)
    return ids


def test_deferral_exemptions_name_fixtures_that_are_actually_deferred() -> None:
    # What ties an exemption to its justification. Without this the dict is a
    # list of keys someone once decided to ignore, and un-deferring a fixture
    # silently turns its directive dark again.
    deferred = _deferred_fixture_ids()
    stale = [
        (key, fixture)
        for key, fixtures in UNAPPLIED_PENDING_DEFERRAL.items()
        for fixture in fixtures
        if fixture not in deferred
    ]
    assert not stale, (
        "UNAPPLIED_PENDING_DEFERRAL names fixture(s) that are no longer deferred:\n"
        + "\n".join(f"  {k!r} cites {f}" for k, f in sorted(stale))
        + "\nThose fixtures run now, so the directive must be wired or the entry removed."
    )


def _case_deferrals_in(function_name: str) -> set[str]:
    """The case names a driver skips, read off its `_deferred_cases` literal."""
    from . import test_observability  # noqa: PLC0415

    tree = ast.parse(inspect.getsource(getattr(test_observability, function_name)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "_deferred_cases" for t in targets):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                names.add(sub.value)
    return names


@pytest.mark.parametrize(
    ("key", "fixture", "case"),
    [
        (key, fixture, case)
        for key, pairs in UNAPPLIED_PENDING_CASE_DEFERRAL.items()
        for fixture, case in pairs
    ],
)
def test_case_deferral_exemptions_name_cases_that_are_actually_skipped(
    key: str, fixture: str, case: str
) -> None:
    # The per-case counterpart. `session_id` is carried by one case of a fixture
    # that otherwise runs, so a fixture-level deferral check would not see it.
    driver = "_run_fixture_" + fixture.split("-")[0]
    skipped = _case_deferrals_in(driver)
    assert case in skipped, (
        f"{key!r} is exempted because {fixture}::{case} does not run, but {driver} "
        f"skips only {sorted(skipped)}. Wire the directive or drop the exemption."
    )


def test_every_conformance_directory_is_run_or_declared_unimplemented() -> None:
    # The coarse counterpart to the directive checks above: a whole capability
    # can go unread the same way a directive can. This is what makes a new
    # capability directory arriving at a pin bump an error rather than a silence.
    with_fixtures = {
        d.name
        for d in sorted(_SPEC_ROOT.iterdir())
        if d.is_dir() and any((d / "conformance").glob("[0-9][0-9][0-9]-*.yaml"))
    }
    accounted = set(_RUN_DIRS) | set(UNIMPLEMENTED_CAPABILITIES)
    unaccounted = sorted(with_fixtures - accounted)
    assert not unaccounted, (
        f"capability directory/ies ship fixtures but are neither run nor declared "
        f"unimplemented: {unaccounted}. Wire a runner, or record why not in "
        "UNIMPLEMENTED_CAPABILITIES."
    )
    # The other direction: a declaration that has outlived its capability. Once a
    # runner lands, the entry has to go or it understates what we claim.
    stale = sorted(set(UNIMPLEMENTED_CAPABILITIES) & set(_RUN_DIRS))
    assert not stale, (
        f"UNIMPLEMENTED_CAPABILITIES still names {stale}, which now has a runner. Drop the entry."
    )


def _subgraph_definitions() -> list[tuple[str, str]]:
    """Every `(fixture, key)` a running fixture declares inside a subgraph body."""
    out: list[tuple[str, str]] = []
    for name in _RUN_DIRS:
        for path in sorted((_SPEC_ROOT / name / "conformance").glob("[0-9][0-9][0-9]-*.yaml")):
            doc: Any = yaml.safe_load(path.read_text())
            if not isinstance(doc, dict):
                continue
            root = cast("dict[str, Any]", doc)
            cases: Any = root.get("cases")
            containers = [root] + [
                cast("dict[str, Any]", c) for c in cast("list[Any]", cases or []) if isinstance(c, dict)
            ]
            for container in containers:
                bodies: list[dict[str, Any]] = []
                for singular in ("subgraph", "subgraph_with_idx"):
                    body: Any = container.get(singular)
                    if isinstance(body, dict):
                        bodies.append(cast("dict[str, Any]", body))
                plural: Any = container.get("subgraphs")
                if isinstance(plural, dict):
                    bodies.extend(
                        cast("dict[str, Any]", b)
                        for b in cast("dict[str, Any]", plural).values()
                        if isinstance(b, dict)
                    )
                for body in bodies:
                    out.extend((path.stem, key) for key in body)
    return out


def test_every_subgraph_level_key_is_modelled() -> None:
    # `SubgraphDefinition` allows extras like `CaseSpec` does, so the same
    # silent-accept path exists one level down. Unlike the case-level check this
    # one finds nothing today: no fixture declares a subgraph key outside the
    # model. It is here so the first one that does is rejected rather than
    # dropped, which is the whole failure mode a day-one-green guard prevents.
    modelled = set(SubgraphDefinition.model_fields)
    observed = _subgraph_definitions()
    # Non-vacuity on the INPUT. This check passes today with nothing to report,
    # so a walk that silently visited no subgraph body would be indistinguishable
    # from a clean result. 65 fixtures carry one.
    assert len({f for f, _ in observed}) > 50, (
        f"expected subgraph bodies across the corpus, walked {len(observed)} keys in "
        f"{len({f for f, _ in observed})} fixtures -- the traversal is not reaching them"
    )
    unknown = sorted({(f, k) for f, k in observed if k not in modelled})
    assert not unknown, (
        "subgraph-level key(s) outside SubgraphDefinition:\n"
        + "\n".join(f"  {f} declares {k!r}" for f, k in unknown)
        + "\nModel the field, or defer the fixture."
    )
