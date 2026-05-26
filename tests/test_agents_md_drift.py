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
from pathlib import Path
from typing import Any

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


def test_agents_md_matches_generator_output() -> None:
    generator = _load_generator()
    expected = generator.build()
    actual = OUTPUT.read_text()
    assert actual == expected, f"src/openarmature/AGENTS.md is out of date with its sources.\n{REGEN_HINT}"


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
