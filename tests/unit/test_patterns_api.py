"""Unit tests for the ``openarmature.patterns`` programmatic API.

Covers the two-function surface (``list`` + ``get``) and its
contract with the generated payload in
``src/openarmature/_patterns/``. The drift catch lives in
``tests/test_agents_md_drift.py``; these tests verify the
runtime-facing behavior independently of that.
"""

from __future__ import annotations

import pytest

import openarmature.patterns as patterns


def test_list_returns_known_pattern_slugs() -> None:
    names = patterns.list()
    # Exact set of seed patterns shipped from ``docs/patterns/``.
    # If a new pattern lands, update this list deliberately —
    # silent additions mask scope expansion.
    assert names == [
        "bypass-if-output-exists",
        "parameterized-entry-point",
        "session-as-checkpoint-resume",
        "tool-dispatch-as-node",
    ]


def test_list_is_sorted() -> None:
    names = patterns.list()
    assert names == sorted(names)


def test_get_returns_markdown_starting_with_h1() -> None:
    content = patterns.get("bypass-if-output-exists")
    # Programmatic-transformed patterns keep the original H1
    # (heading demotion is bundle-only).
    assert content.startswith("# ")
    # Strip the first line to check it looks like a pattern title.
    first_line = content.splitlines()[0]
    assert "bypass" in first_line.lower() or "output" in first_line.lower()


def test_get_rewrites_relative_links_to_absolute_urls() -> None:
    """Bundle uses anchors for intra-pattern links; the programmatic
    transform rewrites them to absolute ``openarmature.ai`` URLs so
    each pattern stands alone.
    """
    # ``bypass-if-output-exists`` references the middleware concept
    # page via a relative ``../concepts/middleware.md`` link in
    # source. The transform turns it into an absolute URL.
    content = patterns.get("bypass-if-output-exists")
    assert "../concepts/" not in content
    assert "../examples/" not in content
    # At least one openarmature.ai URL should be present (any of
    # the rewritten doc-tree links).
    assert "openarmature.ai" in content


def test_get_unknown_pattern_raises_key_error_with_known_names() -> None:
    with pytest.raises(KeyError) as exc_info:
        patterns.get("does-not-exist")
    msg = str(exc_info.value)
    # Error includes the unknown name (quoted) and the known names
    # so callers don't have to call ``list()`` to recover.
    assert "does-not-exist" in msg
    assert "bypass-if-output-exists" in msg


def test_get_returns_distinct_content_per_pattern() -> None:
    """Sanity check: the four patterns aren't accidentally aliasing
    to the same payload (e.g., a generator bug that wrote one file's
    content under all four slugs).
    """
    contents = {name: patterns.get(name) for name in patterns.list()}
    # All four contents are unique.
    assert len(set(contents.values())) == len(contents)


def test_module_exposes_only_list_and_get() -> None:
    # ``__all__`` defines the public surface. Keep it minimal —
    # implementation helpers (``_PATTERNS_PACKAGE``, ``builtin_list``)
    # are intentional implementation details.
    assert sorted(patterns.__all__) == ["get", "list"]
