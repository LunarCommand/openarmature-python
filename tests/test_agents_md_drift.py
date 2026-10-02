"""Drift check for the generated agent-docs artifacts.

Two artifacts are regenerated and diffed against their committed
forms:

- ``src/openarmature/AGENTS.md`` — the bundled agent-facing
  reference shipped at the package root.
- ``src/openarmature/_patterns/`` — per-pattern markdown files
  consumed by the programmatic ``openarmature.patterns`` API.

Drift in either guards against:

- A spec submodule pin bump that should refresh capability summaries
  but didn't.
- An edit to ``docs/patterns/*.md``, ``docs/agent/tldr.md``, or
  ``docs/agent/non-obvious-shapes.md`` that should propagate into
  the bundle / patterns data but didn't.
- A new example added to ``examples/`` that should appear in the
  index but doesn't.

Sits alongside ``tests/test_smoke.py``'s version-sync checks. If
this test fails, regenerate both artifacts:

    uv run python scripts/build_agents_md.py
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "src" / "openarmature" / "AGENTS.md"
PATTERNS_DIR = REPO_ROOT / "src" / "openarmature" / "_patterns"
GENERATOR = REPO_ROOT / "scripts" / "build_agents_md.py"

REGEN_HINT = "Regenerate with: uv run python scripts/build_agents_md.py"


def _load_generator() -> Any:
    """Import the generator module by path.

    ``scripts/`` isn't a Python package, so the standard
    ``import scripts.build_agents_md`` doesn't work. ``importlib``
    handles the path-based load.
    """
    spec = importlib.util.spec_from_file_location("build_agents_md", GENERATOR)
    assert spec is not None and spec.loader is not None, GENERATOR
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec_submodule_is_queryable(repo_root: Path = REPO_ROOT) -> bool:
    """Whether the generator can resolve the spec submodule's pinned tag here.

    Takes the tree root so the condition itself is testable against a synthetic
    one; the drift marker calls it with no argument.
    """
    # Keyed on the submodule the generator interrogates rather than on
    # repository discovery, which ASCENDS. A probe rooted anywhere beneath a
    # checkout answers for the parent, so an sdist unpacked in a packaging
    # feedstock, a vendored copy, or a CI workspace that is itself a checkout
    # would all read as queryable while the submodule is absent, and the drift
    # test would run and hard-fail there instead of skipping.
    spec_root = repo_root / "openarmature-spec"
    if shutil.which("git") is None:
        return False
    if not (spec_root / ".git").exists():
        return False
    probe = subprocess.run(
        ["git", "-C", str(spec_root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return False
    # An ascent resolves to the enclosing repository rather than the submodule.
    return Path(probe.stdout.strip()).resolve() == spec_root.resolve()


# The bundle records which spec tag it was built from, so the generator asks git
# for the submodule's pinned tag and refuses to build off an untagged commit. An
# sdist ships the source without the repository, so there is nothing to ask and
# the drift check cannot run at all rather than running and passing.
#
# Deliberately narrow: keyed on the absence of a repository, not on the
# generator failing. Skipping whenever the generator raised would let a broken
# generator read as a clean run in CI, where the repository is always present
# and this never skips.
@pytest.mark.skipif(
    not _spec_submodule_is_queryable(),
    reason=(
        "the spec submodule is not a queryable git checkout here, so the generator "
        "cannot resolve its pinned tag; drift is checked in the repository"
    ),
)
def test_agents_md_matches_generator_output() -> None:
    generator = _load_generator()
    expected = generator.build()
    actual = OUTPUT.read_text()
    assert actual == expected, f"src/openarmature/AGENTS.md is out of date with its sources.\n{REGEN_HINT}"


def test_the_drift_check_is_not_skipping_in_this_repository() -> None:
    # A skip condition cannot be caught by the test it guards: widen it and the
    # drift check quietly stops running while the suite still reports green.
    # That is the null result this repository treats as evidence of nothing, one
    # level up from a vacuous assertion.
    #
    # Asserted against the RESOLVED MARKER rather than the helper, so widening
    # the decorator itself is caught too. A guard that only re-called the helper
    # would stay green under `skipif(not helper() or something_else, ...)`.
    #
    # Gated on the two repository markers directly, which is what makes this
    # non-circular: both are filesystem checks, while the part of the condition
    # that matters is the ascent comparison the helper runs. Without the gate
    # this test fails from an unpacked sdist, where skipping is correct.
    if not (REPO_ROOT / ".git").exists() or not (REPO_ROOT / "openarmature-spec" / ".git").exists():
        pytest.skip("not a complete repository checkout, so skipping is the right answer here")
    # pytest attaches `pytestmark` dynamically, so the type checker cannot see it.
    applied = cast("list[Any]", getattr(test_agents_md_matches_generator_output, "pytestmark", []))
    marks = [m for m in applied if m.name == "skipif"]
    assert len(marks) == 1, f"expected exactly one skipif marker, got {marks}"
    condition = marks[0].args[0]
    assert condition is False, (
        "the drift check is skipping inside its own repository, so it is not "
        "running at all. The condition should be true only where the spec "
        "submodule cannot be queried, such as an unpacked sdist."
    )


def test_the_drift_check_skips_where_the_submodule_is_absent(tmp_path: Path) -> None:
    # The other direction, and the one that catches a guard which never skips:
    # that puts a hard failure in front of every downstream packager running the
    # shipped suite.
    #
    # The shape is an unpacked sdist NESTED INSIDE A REPOSITORY, which is the
    # ordinary case: a packaging feedstock, a vendored copy, a CI workspace that
    # is itself a checkout. The sdist has an `openarmature-spec/` directory
    # (9933608 ships the fixtures there) but no repository of its own.
    #
    # Not vacuous, and the nesting is what makes it so. Git repository discovery
    # ASCENDS, so a condition keyed on `rev-parse` from the tree root answers for
    # the enclosing repository and returns True here, which is the defect this
    # replaces. Without an enclosing repository the mutation survives, because
    # the probe simply fails and the condition is False for the wrong reason.
    if shutil.which("git") is None:
        pytest.skip("git is absent, so the enclosing-repository shape cannot be built")

    enclosing = tmp_path / "feedstock"
    enclosing.mkdir()
    init = subprocess.run(["git", "init", "-q", str(enclosing)], capture_output=True, text=True)
    assert init.returncode == 0, init.stderr
    assert (enclosing / ".git").exists(), "the enclosing repository did not initialize"

    sdist_root = enclosing / "openarmature-0.0.0"
    (sdist_root / "openarmature-spec" / "spec").mkdir(parents=True)

    # Confirm the ascent is real in this fixture before relying on it, so the
    # assertion below cannot pass because the setup failed to reproduce it.
    ascent = subprocess.run(
        ["git", "-C", str(sdist_root), "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
    )
    assert ascent.returncode == 0, (
        "the nested sdist should resolve to the enclosing repository; if it does "
        "not, this fixture is not reproducing the case it exists for"
    )

    assert not _spec_submodule_is_queryable(sdist_root), (
        "an unpacked sdist has no queryable spec submodule, so the drift check "
        "must skip there rather than running the generator and failing. A "
        "condition keyed on repository discovery answers for the enclosing "
        "repository instead."
    )


def test_patterns_dir_matches_generator_output() -> None:
    generator = _load_generator()
    expected_payload: dict[str, str] = generator.build_patterns_data()
    expected_init: str = generator._PATTERNS_INIT_CONTENT

    # All committed ``.md`` files match the regenerated payload.
    for filename, expected_content in expected_payload.items():
        committed = PATTERNS_DIR / filename
        assert committed.is_file(), f"src/openarmature/_patterns/{filename} is missing.\n{REGEN_HINT}"
        assert committed.read_text() == expected_content, (
            f"src/openarmature/_patterns/{filename} is out of date.\n{REGEN_HINT}"
        )

    # No stale ``.md`` files left from a prior generation (e.g.,
    # a pattern was renamed or removed but the old file persists).
    committed_md = {p.name for p in PATTERNS_DIR.iterdir() if p.suffix == ".md"}
    assert committed_md == set(expected_payload.keys()), (
        f"src/openarmature/_patterns/ contains stale or extra .md files.\n"
        f"  committed: {sorted(committed_md)}\n"
        f"  expected:  {sorted(expected_payload.keys())}\n"
        f"{REGEN_HINT}"
    )

    # The package marker ``__init__.py`` matches the generator's
    # canonical content (the docstring describes the directory's
    # purpose; rewritten on every generate).
    init_path = PATTERNS_DIR / "__init__.py"
    assert init_path.is_file(), f"src/openarmature/_patterns/__init__.py is missing.\n{REGEN_HINT}"
    assert init_path.read_text() == expected_init, (
        f"src/openarmature/_patterns/__init__.py is out of date.\n{REGEN_HINT}"
    )
