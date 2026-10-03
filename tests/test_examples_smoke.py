"""Smoke test: load each example's ``main.py`` and invoke its
``build_graph()`` factory.

We don't run the demo end-to-end — that would hit a local OpenAI-
compatible LLM endpoint that isn't available in CI. But loading the
module and compiling its graph catches:

- syntax errors,
- accidental breakage in openarmature's public API that would only
  otherwise surface when a user runs the demo,
- missing imports (e.g. a renamed symbol that the demo still
  references),
- graph-compile failures in the demo's structure (dangling edges,
  unreachable nodes, conflicting reducers, missing entry, etc.).

``runpy.run_path`` with ``run_name`` set to a sentinel skips the
example's ``if __name__ == "__main__":`` block, so we get the
module-level import side-effects without firing any LLM calls.
``build_graph()`` is a convention every demo exposes — invoking it
exercises the same compile path the demo's ``main()`` does.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

DEMOS = [
    "hello-world",
    "routing-and-subgraphs",
    "explicit-subgraph-mapping",
    "nested-subgraphs",
    "fan-out-with-retry",
    "parallel-branches",
    "multimodal-prompt",
    "chat-with-multimodal",
    "tool-use",
    "checkpointing-and-migration",
    "observer-hooks",
    "langfuse-observability",
    "production-observability",
    "retrieval-rag",
    "structured-output-reask",
    "provider-extras",
]


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that survives mkdocs-material's python tags.

    `mkdocs.yml` carries `!!python/name:` values for the emoji extension, which
    SafeLoader refuses and the unsafe loaders would execute. Ignoring unknown
    tags parses the file without either.
    """


def _ignore_unknown_tag(loader: Any, suffix: str, node: Any) -> None:
    return None


for _prefix in (
    "tag:yaml.org,2002:python/name:",
    "tag:yaml.org,2002:python/object/apply:",
):
    # pyyaml ships no annotations for this classmethod, so reading it is
    # unknown-typed under strict mode regardless of what the result is cast to.
    _TolerantLoader.add_multi_constructor(_prefix, _ignore_unknown_tag)  # pyright: ignore[reportUnknownMemberType]


def _nav_paths(nav: Any) -> list[str]:
    """Every document path the nav references, at any depth.

    A nav is nested lists of either a bare path or a one-key `{title: entry}`
    mapping, where the entry is a path or another list. Only the leaf strings
    are paths.
    """
    if isinstance(nav, str):
        return [nav]
    if isinstance(nav, list):
        return [path for item in cast("list[Any]", nav) for path in _nav_paths(item)]
    if isinstance(nav, dict):
        return [path for value in cast("dict[str, Any]", nav).values() for path in _nav_paths(value)]
    return []


def test_every_example_directory_is_listed() -> None:
    # `DEMOS` is an explicit enumeration, so a new example is covered by nothing
    # until someone remembers to add it here. Derive the directory set and
    # compare, so forgetting fails loudly rather than passing quietly.
    on_disk = {
        d.name
        for d in EXAMPLES_DIR.iterdir()
        if d.is_dir() and (d / "main.py").exists() and not d.name.startswith(".")
    }
    listed = set(DEMOS)
    assert on_disk == listed, (
        f"DEMOS and examples/ disagree. Only on disk: {sorted(on_disk - listed)}. "
        f"Only listed: {sorted(listed - on_disk)}"
    )


def test_every_example_has_a_published_docs_page() -> None:
    # The examples ship three ways: in the sdist, in `examples/README.md`, and
    # as a page on the docs site. The first two are derived from the directory
    # listing and stay in step on their own. The site is hand-maintained in two
    # places, so an example can ship with no page and nothing notices, which is
    # what happened to three of them across two releases.
    #
    # Checked against the nav as well as the file, because a page absent from
    # `mkdocs.yml` is unreachable even when it exists, and `mkdocs build` reports
    # that as INFO rather than failing.
    #
    # The nav half reads the PARSED nav rather than the file's text. A substring
    # search over mkdocs.yml is satisfied by a commented-out entry, by a mention
    # in a comment, and by a path under `plugins:`, so it passes while the page
    # is gone from the navigation. Which is the defect this guard exists to
    # catch, one level up.
    docs_dir = EXAMPLES_DIR.parent / "docs" / "examples"
    config = yaml.load((EXAMPLES_DIR.parent / "mkdocs.yml").read_text(), Loader=_TolerantLoader)

    missing_page = sorted(name for name in DEMOS if not (docs_dir / f"{name}.md").is_file())
    assert not missing_page, (
        f"these examples have no docs page: {missing_page}. Add docs/examples/<name>.md for each."
    )

    in_nav = set(_nav_paths(config.get("nav")))
    assert in_nav, "parsed no nav entries at all, so the check below cannot mean anything"
    missing_nav = sorted(name for name in DEMOS if f"examples/{name}.md" not in in_nav)
    assert not missing_nav, (
        f"these examples have a docs page that the nav does not reference: "
        f"{missing_nav}. Add each to the Examples section of mkdocs.yml."
    )


@pytest.mark.parametrize("demo", DEMOS)
def test_example_loads(demo: str) -> None:
    main_py = EXAMPLES_DIR / demo / "main.py"
    assert main_py.exists(), f"missing: {main_py}"
    module_globals = runpy.run_path(str(main_py), run_name="__not_main__")
    build_graph = module_globals.get("build_graph")
    assert callable(build_graph), f"{demo}/main.py missing build_graph() factory"
    build_graph()
