"""Programmatic access to the openarmature patterns catalog.

Exposes the same patterns content shipped in the bundled
``AGENTS.md`` (at ``openarmature/AGENTS.md`` in the installed
wheel) via an ``import``-accessible API. Useful for agents in
sandboxed environments that can ``import openarmature`` but can't
freely read arbitrary package paths — the patterns content is
resolved through ``importlib.resources``, which uses the same
import mechanism as ``import openarmature.patterns`` itself.

Two functions:

- :func:`list` — returns sorted pattern names (e.g.,
  ``["bypass-if-output-exists", "parameterized-entry-point", ...]``).
- :func:`get` — returns the canonical recipe content as a
  markdown string.

Each pattern stands alone when read via :func:`get`: the markdown
opens at H1 (``# Title``) and relative doc-tree links are rewritten
to absolute ``openarmature.ai`` URLs at build time so cross-
references resolve outside the source tree.

Example::

    import openarmature.patterns as patterns

    for name in patterns.list():
        print(name)
        print(patterns.get(name))
        print("---")

The module-level ``list`` function shadows the builtin within this
namespace. Users call it qualified (``patterns.list()``) so the
shadow is contained; the openarmature.patterns module doesn't use
``list`` as a constructor internally.
"""

from __future__ import annotations

# ``list`` is a module-level function in this namespace per the
# A3 API contract (``openarmature.patterns.list()``). That shadows
# the builtin in lexical scope, so internal references and type
# annotations need ``builtin_list`` to refer to the underlying
# type. Users call the API qualified (``patterns.list()``); the
# shadow is contained to this module.
from builtins import list as builtin_list
from importlib.resources import files

# The ``_patterns`` sub-package is the auto-generated payload (see
# ``scripts/build_agents_md.py``). Each ``<slug>.md`` file is one
# pattern's transformed markdown content. Resolved via
# ``importlib.resources``, which works as long as the package is
# importable — same mechanism as the patterns module itself, so
# sandboxed environments that allow ``import openarmature`` also
# resolve these resources.
_PATTERNS_PACKAGE = "openarmature._patterns"


def list() -> builtin_list[str]:  # noqa: A001 — name matches the A3 API contract
    """Return pattern slugs sorted alphabetically.

    Each slug matches the canonical filename of the pattern docs in
    ``docs/patterns/<slug>.md`` (e.g., ``bypass-if-output-exists``).
    Use the slug with :func:`get` to retrieve the recipe content.
    """
    resource_root = files(_PATTERNS_PACKAGE)
    slugs: builtin_list[str] = []
    for entry in resource_root.iterdir():
        # ``importlib.resources`` returns Traversable entries; the
        # ``name`` attribute is the filename (including extension).
        if entry.name.endswith(".md"):
            slugs.append(entry.name[: -len(".md")])
    slugs.sort()
    return slugs


def get(name: str) -> str:
    """Return the markdown content of the named pattern.

    Raises :class:`KeyError` when ``name`` doesn't match any pattern.
    The error message lists the known names so callers don't need
    to call :func:`list` separately to recover.
    """
    resource = files(_PATTERNS_PACKAGE).joinpath(f"{name}.md")
    if not resource.is_file():
        known = ", ".join(list())
        raise KeyError(f"unknown pattern {name!r}; known patterns: {known}")
    return resource.read_text(encoding="utf-8")


__all__ = ["get", "list"]
