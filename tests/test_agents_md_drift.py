"""Drift check for the bundled ``src/openarmature/AGENTS.md``.

Regenerates the bundle in-memory and diffs against the committed
file. Fails if the committed file is stale — guards against:

- A spec submodule pin bump that should refresh capability summaries
  but didn't.
- An edit to ``docs/patterns/*.md``, ``docs/agent/tldr.md``, or
  ``docs/agent/non-obvious-shapes.md`` that should propagate into
  the bundle but didn't.
- A new example added to ``examples/`` that should appear in the
  index but doesn't.

Sits alongside ``tests/test_smoke.py``'s version-sync checks. If
this test fails, regenerate the bundle:

    uv run python scripts/build_agents_md.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "src" / "openarmature" / "AGENTS.md"
GENERATOR = REPO_ROOT / "scripts" / "build_agents_md.py"


def _load_generator() -> object:
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
    expected = generator.build()  # type: ignore[attr-defined]
    actual = OUTPUT.read_text()
    assert actual == expected, (
        "src/openarmature/AGENTS.md is out of date with its sources.\n"
        "Regenerate with: uv run python scripts/build_agents_md.py"
    )
