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
from typing import Any

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


def _in_a_git_checkout() -> bool:
    """Whether this tree is a git repository the generator can interrogate."""
    if shutil.which("git") is None:
        return False
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


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
    not _in_a_git_checkout(),
    reason=(
        "not a git checkout, so the generator cannot resolve the spec submodule's "
        "pinned tag; drift is checked in the repository, not from an sdist"
    ),
)
def test_agents_md_matches_generator_output() -> None:
    generator = _load_generator()
    expected = generator.build()
    actual = OUTPUT.read_text()
    assert actual == expected, f"src/openarmature/AGENTS.md is out of date with its sources.\n{REGEN_HINT}"


def test_the_drift_check_is_not_skipping_in_this_repository() -> None:
    # A skip condition cannot be caught by the test it guards: widen it and the
    # drift check quietly stops running while the suite still reports green. That
    # is the failure this repository treats as the null result, one level up.
    #
    # So the condition is asserted separately. Here the repository is present by
    # definition (this file is in it), so the guard must evaluate False, and a
    # widened condition fails here instead of going unnoticed.
    # Both preconditions are checked by a different mechanism than the helper
    # uses, so this is a cross-check rather than a restatement: the helper shells
    # out to git, these read the filesystem and PATH. Where they disagree, the
    # helper is wrong.
    if shutil.which("git") is None:
        pytest.skip("git is absent, so the drift check could not run either way")
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("no repository here, as in an unpacked sdist")
    assert _in_a_git_checkout(), (
        "the drift check's skip condition is true inside the repository, so the "
        "check is not running. Narrow the condition: it should only skip where "
        "there is genuinely no git repository, such as an unpacked sdist."
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
