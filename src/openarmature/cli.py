"""Command-line entry point for the ``openarmature`` distribution.

Two subcommands:

- ``openarmature init`` — write the discovery pointer block into
  the host project's ``AGENTS.md`` and ``CLAUDE.md`` so future
  agent sessions opening the project find the bundled OpenArmature
  agent docs.
- ``openarmature docs`` — print the absolute path to the bundled
  ``AGENTS.md`` shipped at the installed package root.

The dispatch is plain :mod:`argparse` — no Click / Typer
dependency. Same surface is reachable as ``python -m openarmature``
via :mod:`openarmature.__main__`, so environments where the
``[project.scripts]`` entry point doesn't land cleanly (some
``pip install --target`` layouts, path-shadowed venvs, etc.) still
work as long as the package is importable.
"""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path

# Comment marker that ``openarmature init`` writes into managed
# AGENTS.md / CLAUDE.md sections. Used to detect prior
# installations on re-run so we don't append duplicate blocks.
# Chosen over a heading-text match so renaming the visible
# heading (e.g., ``## Framework: OpenArmature``) doesn't fool
# the idempotency check. Kept as a module-level constant so tests
# and downstream tooling can reference the canonical literal
# rather than scraping it out of the pointer block content.
INIT_MARKER = "<!-- openarmature-init -->"

# Files ``init`` manages, in the order it processes them.
_MANAGED_FILES = ("AGENTS.md", "CLAUDE.md")


def _pointer_block() -> str:
    """Return the canonical pointer block ``init`` writes.

    Sourced from ``openarmature/_pointer_block.md`` shipped in the
    package data so the block has one canonical home rather than
    being duplicated in a Python string literal. The file is the
    single source of truth — edit it (and re-run the CLI tests) to
    change what ``openarmature init`` writes.

    The returned string ends with a trailing newline; callers handle
    leading-whitespace trimming based on whether they're creating a
    new file or appending to an existing one.
    """
    return files("openarmature").joinpath("_pointer_block.md").read_text(encoding="utf-8")


def _bundled_agents_md_path() -> Path:
    """Return the absolute path to the bundled ``AGENTS.md``.

    Resolved via :mod:`importlib.resources` so it works whether the
    package is installed as a wheel, editable, or zipped — same
    mechanism the README discovery one-liner relies on.
    """
    resource = files("openarmature").joinpath("AGENTS.md")
    # ``files()`` returns a Traversable; the bundle is a regular
    # file shipped at the package root, so ``str()`` is the on-disk
    # path. ``Path()`` normalizes platform separators.
    return Path(str(resource))


def _apply_init_to_file(target: Path, *, force: bool, dry_run: bool) -> tuple[str, str]:
    """Apply the pointer block to a single file.

    Returns ``(action, detail)`` where ``action`` is one of:

    - ``"create"`` — target didn't exist; would create with just
      the pointer block.
    - ``"append"`` — target exists; would append the pointer
      block.
    - ``"skip"`` — target exists and already contains the marker;
      no change.
    - ``"force-append"`` — target exists, already contains the
      marker, but ``--force`` re-appends anyway.

    ``detail`` is a short human-readable note (e.g., the target
    path, why it was skipped).

    With ``dry_run=True``, no file is written; the action describes
    what *would* happen.
    """
    block = _pointer_block()
    if not target.exists():
        # Fresh file gets the block verbatim: no leading blank line,
        # trailing newline preserved.
        if not dry_run:
            target.write_text(block)
        return ("create", str(target))

    existing = target.read_text()
    if INIT_MARKER in existing and not force:
        return ("skip", f"{target} already contains {INIT_MARKER}")

    # Append onto an existing file: normalize a blank-line separator
    # between prior content and the new section so the file reads as
    # ``<existing trimmed>\n\n## OpenArmature\n...``.
    appended = existing.rstrip() + "\n\n" + block
    if not dry_run:
        target.write_text(appended)
    action = "force-append" if force and INIT_MARKER in existing else "append"
    return (action, str(target))


def cmd_init(args: argparse.Namespace) -> int:
    """Handle ``openarmature init``."""
    base = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    if not base.is_dir():
        print(f"error: --cwd path is not a directory: {base}", file=sys.stderr)
        return 2

    prefix = "[dry-run] " if args.dry_run else ""
    for name in _MANAGED_FILES:
        action, detail = _apply_init_to_file(base / name, force=args.force, dry_run=args.dry_run)
        print(f"{prefix}{action}: {detail}")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    """Handle ``openarmature docs``."""
    del args
    print(_bundled_agents_md_path())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser.

    Factored out from :func:`main` so the parser is importable for
    tests and shell-completion tooling without invoking the CLI.
    """
    parser = argparse.ArgumentParser(
        prog="openarmature",
        description=(
            "OpenArmature CLI. Wires agent-discovery pointers into a "
            "project's AGENTS.md / CLAUDE.md and prints the path to "
            "the bundled agent docs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser(
        "init",
        help="Write the OpenArmature discovery pointer block into AGENTS.md / CLAUDE.md.",
        description=(
            "Append an OpenArmature pointer section to AGENTS.md and CLAUDE.md "
            "in the current directory (or --cwd). Skips files that already "
            "contain the marker unless --force is set."
        ),
    )
    init_p.add_argument(
        "--force",
        action="store_true",
        help="Append the pointer block even if the marker is already present.",
    )
    init_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any files.",
    )
    init_p.add_argument(
        "--cwd",
        metavar="PATH",
        help="Operate against PATH/AGENTS.md and PATH/CLAUDE.md instead of the current directory.",
    )
    init_p.set_defaults(func=cmd_init)

    docs_p = sub.add_parser(
        "docs",
        help="Print the absolute path to the bundled AGENTS.md.",
        description=(
            "Print the absolute path to the bundled openarmature/AGENTS.md "
            "shipped with this installation. Equivalent to "
            "`python -c \"import openarmature; print(openarmature.__path__[0] + '/AGENTS.md')\"`."
        ),
    )
    docs_p.set_defaults(func=cmd_docs)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``openarmature`` and ``python -m openarmature``.

    Returns the process exit code. Raises no exceptions on normal
    flow — argparse handles ``--help`` and unknown subcommands by
    printing usage and calling :func:`sys.exit` directly.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
