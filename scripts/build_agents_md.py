"""Generator for the bundled ``src/openarmature/AGENTS.md`` agent docs.

Pulls from canonical sources (pinned spec submodule, patterns docs,
hand-curated agent docs, example program docstrings) and concatenates
into a single agent-discoverable file shipped in the wheel.

Sources, in order of bundle layout:

1. Self-reference header — version-stamped, pointers out to the docs
   site and the spec capabilities page.
2. ``docs/agent/tldr.md`` — hand-written 3-5 sentence orientation.
3. Capability summaries — §1 (Purpose) + §2 (Concepts) of each
   capability spec, read from the pinned ``openarmature-spec``
   submodule via ``git show <sha>:spec/...`` rather than the
   working tree.
4. ``docs/patterns/*.md`` — concatenation of the patterns docs
   (excluding ``index.md``), with bundle-side transforms applied
   in ``_transform_pattern_content``: ATX headings demoted by two
   levels (so a pattern's ``# Title`` H1 renders as ``### Title``
   H3 under the bundle's ``## Patterns`` H2) and relative
   ``../concepts/...md`` / ``../examples/...md`` links rewritten
   to absolute ``openarmature.ai`` URLs (the relative paths
   resolve in the MkDocs source tree but not in the installed
   wheel).
5. ``docs/agent/non-obvious-shapes.md`` — hand-written opinionated
   recipes.
6. Example index — one-line description + path for each
   ``examples/*/main.py`` program.
7. Discovery footer — pointer back out to docs / spec / host
   project conventions.

Build-time invariants (matches proposal-0028 follow-on review's
submodule-pin discipline):

- Submodule HEAD MUST be AT a ``v*`` tag (``git tag --points-at HEAD``).
  The build refuses to read draft (untagged) spec text — or text from
  a commit between two release tags — into a release bundle.
- Spec text is read from the pinned commit via ``git show``, NOT
  from the submodule working tree. Closes the "submodule HEAD
  moved but bundle still reads stale tree" failure mode.

Drift between the committed bundle and the regenerated output is
caught by ``tests/test_agents_md_drift.py``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Make ``openarmature`` importable without requiring an editable install
# pass through ``uv`` — the build script runs locally and on CI.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import openarmature  # noqa: E402

SPEC_ROOT = REPO_ROOT / "openarmature-spec"
DOCS = REPO_ROOT / "docs"
EXAMPLES = REPO_ROOT / "examples"
OUTPUT = REPO_ROOT / "src" / "openarmature" / "AGENTS.md"
# Directory holding per-pattern transformed markdown for the
# programmatic API (``openarmature.patterns``). Each ``<slug>.md``
# file inside is a generated artifact; the directory is a package
# (``__init__.py`` exists) so ``importlib.resources.files()`` can
# locate it through the standard import mechanism. Sandboxed
# environments that can ``import openarmature`` can also resolve
# its package resources.
PATTERNS_DIR_OUTPUT = REPO_ROOT / "src" / "openarmature" / "_patterns"

# Spec capability directory names under ``openarmature-spec/spec/``,
# in the order they appear in the bundle's "Capability contracts"
# section. The order matches the order capabilities were introduced
# (graph-engine first, prompt-management most recent) so an agent
# reading top-down sees the foundational layer before the layers
# built on top.
CAPABILITIES = (
    "graph-engine",
    "pipeline-utilities",
    "llm-provider",
    "observability",
    "prompt-management",
)


def _git_in_spec(*args: str) -> str:
    """Run ``git -C openarmature-spec <args>`` and return stdout stripped."""
    return subprocess.run(
        ["git", "-C", str(SPEC_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _assert_pin_at_tag() -> str:
    """Confirm the submodule HEAD is at a ``v*`` tag.

    Returns the tag name (e.g., ``v0.22.1``). Raises ``RuntimeError``
    on a non-tag pin so a release can't accidentally ship a bundle
    pinned to a draft spec commit.

    Prefers the highest semver tag when multiple ``v*`` tags point at
    the same SHA (the v0.19.0 / v0.20.0 / v0.20.1 retag during the
    fixture 052 backport produced this shape). Uses git's native
    ``--sort=-version:refname`` rather than Python lexicographic sort,
    which mis-orders multi-digit versions (``v0.9.0`` lex-sorts after
    ``v0.10.0``).
    """
    sha = _git_in_spec("rev-parse", "HEAD")
    tags_out = _git_in_spec(
        "tag",
        "--sort=-version:refname",
        "--points-at",
        sha,
        "--list",
        "v*",
    )
    if not tags_out:
        raise RuntimeError(
            f"submodule HEAD {sha[:8]} is not at a v* tag; "
            f"bundle build refuses to read draft (untagged) spec text. "
            f"Pin the submodule to a published tag before regenerating."
        )
    # Git's version-aware descending sort puts the highest semver tag first.
    return tags_out.splitlines()[0]


def _read_pinned_spec(path_in_spec: str) -> str:
    """Read a file from the pinned spec commit via ``git show``.

    Distinct from reading the working tree: a stale checkout would
    silently produce stale bundle content. ``git show HEAD:<path>``
    always reads from the recorded commit.
    """
    return subprocess.run(
        ["git", "-C", str(SPEC_ROOT), "show", f"HEAD:{path_in_spec}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _header(version: str, spec_tag: str) -> str:
    return (
        f"# OpenArmature — Agent documentation\n"
        f"\n"
        f"*This is the agent guide bundled with the openarmature Python package, "
        f"version {version} (spec {spec_tag}). For the full docs site see "
        f"[openarmature.ai](https://openarmature.ai). For the canonical spec text see "
        f"[openarmature.org/capabilities](https://openarmature.org/capabilities/). "
        f"For project-specific conventions for the code you're editing, see the host "
        f"project's `AGENTS.md` or `CLAUDE.md`.*"
    )


def _tldr() -> str:
    body = (DOCS / "agent" / "tldr.md").read_text().strip()
    return f"## TL;DR\n\n{body}"


def _extract_sections_1_2(spec_text: str) -> str:
    """Extract content between ``## 1.`` and ``## 3.`` (inclusive of §1+§2).

    Demotes ATX headings by two levels so the bundled markdown's
    hierarchy stays consistent: the wrapping ``### Capability: ...``
    H3 sits above the extracted ``## 1. Purpose`` rendered as
    ``#### 1. Purpose`` (H4). Any deeper nested headings inside §1+§2
    (e.g., ``### State``) preserve their relative depth one step
    deeper. Without this demotion, the spec's H2 headings would
    appear higher in the document than the H3 they sit under,
    breaking TOC rendering and navigation.
    """
    out: list[str] = []
    in_target = False
    for line in spec_text.splitlines():
        if line.startswith("## 1."):
            in_target = True
        elif line.startswith("## 3."):
            break
        if in_target:
            if line.startswith("#"):
                # Demote ATX heading by two levels.
                line = "##" + line
            out.append(line)
    if not out:
        raise RuntimeError(
            "spec heading-extraction failed: no `## 1.` heading found. "
            "Spec capability may have renumbered; revisit the build script."
        )
    return "\n".join(out).rstrip()


def _capability_summaries(spec_tag: str) -> str:
    # Long-string entries use explicit ``+`` concat (not Python's
    # implicit adjacent-string-literal concat) so CodeQL / static
    # analyzers don't flag the pattern as a possibly-missing comma
    # inside the list literal.
    sections = [
        "## Capability contracts",
        "",
        (
            f"_Sourced from openarmature-spec {spec_tag}. Each entry below "
            + "reproduces §1 (Purpose) and §2 (Concepts) of the capability's "
            + "`spec.md`. For the full spec text (execution model, error semantics, "
            + "determinism, observer hooks, etc.) see the linked docs site._"
        ),
    ]
    for cap in CAPABILITIES:
        text = _read_pinned_spec(f"spec/{cap}/spec.md")
        sections.append("")
        sections.append(f"### Capability: `{cap}`")
        sections.append("")
        sections.append(_extract_sections_1_2(text))
    return "\n".join(sections)


_PATTERN_LINK_RE = re.compile(r"\(\.\./(concepts|examples)/([^)]+?)\.md\)")

# Matches bare-name ``.md`` references in the patterns markdown
# (pattern-to-pattern links like ``(bypass-if-output-exists.md)``).
# The negative lookahead skips ``../`` parent-relative paths
# (handled by ``_PATTERN_LINK_RE``), ``http(s)://`` absolute URLs,
# and ``#`` in-document anchors.
_PATTERN_INTRA_LINK_RE = re.compile(r"\((?!\.\.|https?://|#)([a-z0-9-]+)\.md\)")


def _demote_headings(text: str) -> str:
    """Demote ATX headings by two levels by prepending ``##``.

    Bundle-only transform. Pattern files open with ``# Title`` (H1);
    inlined verbatim under the bundle's ``## Patterns`` H2, those
    H1s would create multiple top-level headings in the same
    document. Prepending ``##`` to every ``#``-prefixed line puts
    pattern titles at H3 (under ``## Patterns``) and preserves the
    relative depth of any deeper nested headings.
    """
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            line = "##" + line
        out.append(line)
    return "\n".join(out)


def _rewrite_doc_tree_links(text: str) -> str:
    """Rewrite relative ``../concepts/...md`` / ``../examples/...md``
    references to absolute ``openarmature.ai`` URLs.

    Shared between the bundle and the programmatic patterns API —
    relative paths resolve in the MkDocs source tree but break
    everywhere else (the installed wheel, programmatic `import`
    consumers). The MkDocs site strips ``.md`` and serves at
    ``/<section>/<name>/``; ``../<section>/index.md`` collapses to
    the section root.
    """

    def _rewrite(m: re.Match[str]) -> str:
        section, name = m.group(1), m.group(2)
        if name == "index":
            return f"(https://openarmature.ai/{section}/)"
        return f"(https://openarmature.ai/{section}/{name}/)"

    return _PATTERN_LINK_RE.sub(_rewrite, text)


def _rewrite_intra_pattern_to_anchor(text: str) -> str:
    """Bundle-only: rewrite pattern-to-pattern bare-name ``.md``
    references to in-document anchors.

    In the bundled single-file ``AGENTS.md`` all patterns appear
    inline; the demoted H3 heading slug matches the filename slug,
    so ``(bypass-if-output-exists.md)`` → ``(#bypass-if-output-exists)``
    resolves to the in-document section.
    """
    return _PATTERN_INTRA_LINK_RE.sub(lambda m: f"(#{m.group(1)})", text)


def _rewrite_intra_pattern_to_url(text: str) -> str:
    """Programmatic-only: rewrite pattern-to-pattern bare-name ``.md``
    references to absolute docs-site URLs.

    The programmatic API returns one pattern at a time; in-document
    anchors would be dead links because the other patterns aren't
    in the same string. Absolute URLs to the MkDocs site let the
    consumer follow the cross-reference if they want to.
    """

    def _rewrite(m: re.Match[str]) -> str:
        name = m.group(1)
        return f"(https://openarmature.ai/patterns/{name}/)"

    return _PATTERN_INTRA_LINK_RE.sub(_rewrite, text)


def _transform_pattern_content_for_bundle(text: str) -> str:
    """Apply bundle-side transforms to a pattern doc's markdown.

    Composes the heading-demotion + doc-tree-link + intra-anchor
    rewrites. The source files in ``docs/patterns/`` stay unchanged
    — they're MkDocs source where relative links work correctly;
    only the bundled copy gets these rewrites.
    """
    out = _demote_headings(text)
    out = _rewrite_doc_tree_links(out)
    out = _rewrite_intra_pattern_to_anchor(out)
    return out


def _transform_pattern_content_for_programmatic(text: str) -> str:
    """Apply programmatic-API transforms to a pattern doc's markdown.

    Doc-tree-link rewrites + intra-pattern → absolute URL. No
    heading demotion: each pattern accessed via
    ``openarmature.patterns.get(name)`` is a standalone document; its
    ``# Title`` H1 is the right level.
    """
    out = _rewrite_doc_tree_links(text)
    out = _rewrite_intra_pattern_to_url(out)
    return out


def _patterns() -> str:
    # See ``_capability_summaries`` for the explicit-concat rationale.
    sections = [
        "## Patterns",
        "",
        (
            "_Recipes that compose the primitives. Not framework contracts — "
            + "these are how to do common things idiomatically._"
        ),
    ]
    pattern_files = sorted(p for p in (DOCS / "patterns").glob("*.md") if p.name != "index.md")
    for pf in pattern_files:
        sections.append("")
        sections.append(_transform_pattern_content_for_bundle(pf.read_text()).rstrip())
    return "\n".join(sections)


def _non_obvious_shapes() -> str:
    # The file's own top-level heading is `## Non-obvious shapes`;
    # inlined verbatim with the heading intact.
    return (DOCS / "agent" / "non-obvious-shapes.md").read_text().rstrip()


def _extract_first_docstring_paragraph(source: str) -> str:
    """Extract the first paragraph of a Python module docstring.

    Module docstrings open with a triple-quoted string at line 0.
    The first "paragraph" is the text from the opening quotes to
    the first blank line within the docstring (or to the closing
    quotes if the docstring is one paragraph).
    """
    lines = source.splitlines()
    if not lines or not lines[0].startswith('"""'):
        return ""
    # First line after the opening triple-quote
    first_text = lines[0][3:].rstrip()
    if first_text.endswith('"""'):
        return first_text[:-3].rstrip()
    para = [first_text] if first_text else []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "" or stripped.startswith('"""') or stripped.endswith('"""'):
            break
        para.append(stripped)
    return " ".join(p for p in para if p)


def _example_index() -> str:
    # See ``_capability_summaries`` for the explicit-concat rationale.
    sections = [
        "## Example index",
        "",
        (
            "_Runnable example programs shipped in the source tree at `examples/`. "
            + "The full code is not bundled here (each example is 300+ lines); read "
            + "the file at the listed path to see the canonical shape for that use case._"
        ),
        "",
    ]
    for ex in sorted(EXAMPLES.glob("*/main.py")):
        first_paragraph = _extract_first_docstring_paragraph(ex.read_text())
        rel = ex.relative_to(REPO_ROOT)
        sections.append(f"- **`{rel}`** — {first_paragraph}")
    return "\n".join(sections)


def _discovery_footer() -> str:
    return (
        "## Discovery cross-references\n"
        "\n"
        "If your question isn't covered above, look here:\n"
        "\n"
        "- **Full docs site:** [openarmature.ai](https://openarmature.ai)\n"
        "- **Spec text:** [openarmature.org/capabilities](https://openarmature.org/capabilities/)\n"
        "- **API reference:** [openarmature.ai/reference](https://openarmature.ai/reference/)\n"
        "- **Host project conventions:** the project's own `AGENTS.md` / `CLAUDE.md`\n"
    )


def build() -> str:
    spec_tag = _assert_pin_at_tag()
    version = openarmature.__version__
    sections = [
        _header(version, spec_tag),
        _tldr(),
        _capability_summaries(spec_tag),
        _patterns(),
        _non_obvious_shapes(),
        _example_index(),
        _discovery_footer(),
    ]
    # ``_discovery_footer`` already ends with ``\n``; strip any
    # trailing whitespace from the joined output, then add exactly
    # one final newline. Avoids the pre-commit end-of-file-fixer /
    # editor "strip trailing blank line" normalization producing a
    # different byte sequence than the committed file.
    return "\n\n".join(sections).rstrip() + "\n"


_PATTERNS_INIT_CONTENT = (
    '"""Auto-generated package holding the programmatic patterns API\'s\n'
    "transformed markdown payload.\n"
    "\n"
    "``openarmature.patterns.list()`` / ``get(name)`` resolve the\n"
    "per-pattern ``<slug>.md`` files in this package via\n"
    "``importlib.resources``. The files are generated artifacts —\n"
    "regenerate with ``uv run python scripts/build_agents_md.py``.\n"
    "\n"
    "Source: ``docs/patterns/*.md`` (excluding ``index.md``) with\n"
    "the programmatic-API transforms applied — relative\n"
    "``../concepts/...md`` / ``../examples/...md`` links rewritten\n"
    "to absolute ``openarmature.ai`` URLs, intra-pattern bare-name\n"
    "``.md`` links rewritten to absolute\n"
    "``openarmature.ai/patterns/...`` URLs (see\n"
    "``_transform_pattern_content_for_programmatic`` in\n"
    "``scripts/build_agents_md.py``). No heading demotion: each\n"
    "pattern stands alone when read via the programmatic API.\n"
    '"""\n'
)


def build_patterns_data() -> dict[str, str]:
    """Build the per-pattern transformed markdown payload.

    Returns a dict mapping ``<slug>.md`` filename → transformed
    content. Caller writes each entry to
    ``src/openarmature/_patterns/<slug>.md``. Consumed by the
    programmatic patterns API (``openarmature.patterns.list()`` /
    ``get(name)``) via ``importlib.resources``.

    Uses the programmatic transform set (no heading demotion,
    intra-pattern → absolute URLs) so each pattern stands alone
    when read individually.
    """
    pattern_files = sorted(p for p in (DOCS / "patterns").glob("*.md") if p.name != "index.md")
    out: dict[str, str] = {}
    for pf in pattern_files:
        content = _transform_pattern_content_for_programmatic(pf.read_text()).rstrip() + "\n"
        out[f"{pf.stem}.md"] = content
    return out


def _write_patterns_data(payload: dict[str, str]) -> tuple[int, int]:
    """Write per-pattern files into ``_patterns/`` + the package
    ``__init__.py``. Returns ``(file_count, total_bytes)``.

    Files that exist but aren't in the payload (e.g., a pattern
    removed upstream) are deleted so the directory stays in lockstep
    with the source. The ``__init__.py`` is rewritten unconditionally
    to keep its docstring current.
    """
    PATTERNS_DIR_OUTPUT.mkdir(exist_ok=True)
    init_path = PATTERNS_DIR_OUTPUT / "__init__.py"
    init_path.write_text(_PATTERNS_INIT_CONTENT)
    expected = set(payload.keys())
    expected.add("__init__.py")
    # Clean up stray .md files from a prior generation that aren't in
    # the current payload (e.g., a pattern was renamed or removed).
    for existing in PATTERNS_DIR_OUTPUT.iterdir():
        if existing.name not in expected and existing.suffix == ".md":
            existing.unlink()
    total_bytes = len(_PATTERNS_INIT_CONTENT.encode("utf-8"))
    for filename, content in payload.items():
        (PATTERNS_DIR_OUTPUT / filename).write_text(content)
        total_bytes += len(content.encode("utf-8"))
    return (len(payload), total_bytes)


def main() -> None:
    content = build()
    OUTPUT.write_text(content)
    line_count = content.count("\n")
    byte_count = len(content.encode("utf-8"))
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}: {line_count} lines, {byte_count:,} bytes")

    patterns_payload = build_patterns_data()
    file_count, total_bytes = _write_patterns_data(patterns_payload)
    print(
        f"wrote {PATTERNS_DIR_OUTPUT.relative_to(REPO_ROOT)}/: "
        f"{file_count} pattern files + __init__.py, {total_bytes:,} bytes total"
    )


if __name__ == "__main__":
    main()
