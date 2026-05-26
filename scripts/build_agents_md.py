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
4. ``docs/patterns/*.md`` — verbatim concatenation of the patterns
   docs (excluding ``index.md``).
5. ``docs/agent/non-obvious-shapes.md`` — hand-written opinionated
   recipes.
6. Example index — one-line description + path for each
   ``examples/*/main.py`` program.
7. Discovery footer — pointer back out to docs / spec / host
   project conventions.

Build-time invariants (matches proposal-0028 follow-on review's
submodule-pin discipline):

- Submodule HEAD MUST be reachable from a ``v*`` tag. The build
  refuses to read draft (untagged) spec text into a release bundle.
- Spec text is read from the pinned commit via ``git show``, NOT
  from the submodule working tree. Closes the "submodule HEAD
  moved but bundle still reads stale tree" failure mode.

Drift between the committed bundle and the regenerated output is
caught by ``tests/test_agents_md_drift.py``.
"""

from __future__ import annotations

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
    """
    sha = _git_in_spec("rev-parse", "HEAD")
    tags_out = _git_in_spec("tag", "--points-at", sha, "--list", "v*")
    if not tags_out:
        raise RuntimeError(
            f"submodule HEAD {sha[:8]} is not at a v* tag; "
            f"bundle build refuses to read draft (untagged) spec text. "
            f"Pin the submodule to a published tag before regenerating."
        )
    # Prefer the highest version tag if multiple point at the same SHA
    # (e.g., during a re-tag) — sort by version-string descending.
    tags = sorted(tags_out.splitlines(), reverse=True)
    return tags[0]


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
        f"[openarmature.ai/capabilities](https://openarmature.ai/capabilities/). "
        f"For project-specific conventions for the code you're editing, see the host "
        f"project's `AGENTS.md` or `CLAUDE.md`.*"
    )


def _tldr() -> str:
    body = (DOCS / "agent" / "tldr.md").read_text().strip()
    return f"## TL;DR\n\n{body}"


def _extract_sections_1_2(spec_text: str) -> str:
    """Extract content between ``## 1.`` and ``## 3.`` (inclusive of §1+§2)."""
    out: list[str] = []
    in_target = False
    for line in spec_text.splitlines():
        if line.startswith("## 1."):
            in_target = True
        elif line.startswith("## 3."):
            break
        if in_target:
            out.append(line)
    if not out:
        raise RuntimeError(
            "spec heading-extraction failed: no `## 1.` heading found. "
            "Spec capability may have renumbered; revisit the build script."
        )
    return "\n".join(out).rstrip()


def _capability_summaries(spec_tag: str) -> str:
    sections = [
        "## Capability contracts",
        "",
        f"_Sourced from openarmature-spec {spec_tag}. Each entry below "
        f"reproduces §1 (Purpose) and §2 (Concepts) of the capability's "
        f"`spec.md`. For the full spec text (execution model, error semantics, "
        f"determinism, observer hooks, etc.) see the linked docs site._",
    ]
    for cap in CAPABILITIES:
        text = _read_pinned_spec(f"spec/{cap}/spec.md")
        sections.append("")
        sections.append(f"### Capability: `{cap}`")
        sections.append("")
        sections.append(_extract_sections_1_2(text))
    return "\n".join(sections)


def _patterns() -> str:
    sections = [
        "## Patterns",
        "",
        "_Recipes that compose the primitives. Not framework contracts — "
        "these are how to do common things idiomatically._",
    ]
    pattern_files = sorted(p for p in (DOCS / "patterns").glob("*.md") if p.name != "index.md")
    for pf in pattern_files:
        sections.append("")
        sections.append(pf.read_text().rstrip())
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
    sections = [
        "## Example index",
        "",
        "_Runnable example programs shipped in the source tree at `examples/`. "
        "The full code is not bundled here (each example is 300+ lines); read "
        "the file at the listed path to see the canonical shape for that use case._",
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
        "- **Spec text:** [openarmature.ai/capabilities](https://openarmature.ai/capabilities/)\n"
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
    return "\n\n".join(sections) + "\n"


def main() -> None:
    content = build()
    OUTPUT.write_text(content)
    line_count = content.count("\n")
    byte_count = len(content.encode("utf-8"))
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}: {line_count} lines, {byte_count:,} bytes")


if __name__ == "__main__":
    main()
