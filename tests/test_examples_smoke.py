"""Smoke test: load each example's ``main.py`` to verify it imports
cleanly.

We don't actually run the demos — they hit a local OpenAI-compatible
LLM endpoint that isn't available in CI. But loading the module
catches:

- syntax errors,
- accidental breakage in openarmature's public API that would only
  otherwise surface when a user runs the demo,
- missing imports (e.g. a renamed symbol that the demo still references).

``runpy.run_path`` with ``run_name`` set to a sentinel skips the
example's ``if __name__ == "__main__":`` block, so we get the
module-level import side-effects without firing any LLM calls.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

DEMOS = [
    "01-linear-pipeline",
    "02-routing-and-subgraphs",
    "03-explicit-subgraph-mapping",
    "04-observer-hooks",
    "05-nested-subgraphs",
]


@pytest.mark.parametrize("demo", DEMOS)
def test_example_loads(demo: str) -> None:
    main_py = EXAMPLES_DIR / demo / "main.py"
    assert main_py.exists(), f"missing: {main_py}"
    runpy.run_path(str(main_py), run_name="__not_main__")
