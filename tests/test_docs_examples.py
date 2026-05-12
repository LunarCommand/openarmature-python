"""Run every Python code block in ``docs/`` as a test.

Uses pytest-examples to find each ```python ... ``` block under ``docs/``
and execute it. Catches docs drift — if a refactor breaks a snippet's
imports or API call, this test fails before the docs site can mislead
a reader.

Non-Python blocks (``bash``, ``toml``, etc.) are skipped — the
language filter happens at parametrize time so the test count stays
honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

DOCS_DIR = Path(__file__).parent.parent / "docs"

PYTHON_EXAMPLES = [example for example in find_examples(DOCS_DIR) if example.prefix in ("python", "py")]


@pytest.mark.parametrize("example", PYTHON_EXAMPLES, ids=str)
def test_docs_example(example: CodeExample, eval_example: EvalExample) -> None:
    # Only snippets under sections holding full runnable programs are
    # executed (``getting-started/`` Quickstart, ``model-providers/``
    # Provider Protocol + skeleton). Concept-page snippets are
    # illustrative — they reference names defined out-of-band (a
    # builder local, a not-yet-defined class) and aren't meant to
    # stand alone. Drift detection still covers the core API via the
    # executed snippets plus the auto-generated mkdocstrings reference.
    runnable_sections = ("getting-started", "model-providers")
    if not any(section in str(example.path) for section in runnable_sections):
        pytest.skip("illustrative snippet, not a standalone program")
    eval_example.run(example)
