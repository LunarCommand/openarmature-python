"""Fail when a function re-imports something its module already imports.

A function-local ``import asyncio`` in a module that already has
``import asyncio`` at the top does nothing: the name is bound either way.
CodeQL reports it as ``py/repeated-import`` and ruff has no rule for it, so
this check runs where ruff does, at commit time and in CI.

Function-local imports are not flagged in general; many are deliberate. Only
an import whose binding and target both match a module-level runtime import is
a repeat. That is broader than CodeQL's rule, which considers plain
``import X`` only and only when it can resolve ``X`` to a module, so it misses
``from X import Y`` repeats and some stdlib modules entirely.

Usage:
    python scripts/check_repeated_imports.py [FILE ...]

With no arguments, checks every Python file under the source, test, example and
script trees. Exits 1 and lists each repeat when any are found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOTS = ("src", "tests", "examples", "scripts")

# (bound name, what it is bound to). `import a.b` binds `a` but loads `a.b`, so
# the target is the full dotted path; a `from` import's target is the module,
# its relative level, and the member.
Binding = tuple[str, tuple[object, ...]]


def _bindings(node: ast.Import | ast.ImportFrom) -> list[Binding]:
    if isinstance(node, ast.Import):
        return [(alias.asname or alias.name.split(".")[0], ("import", alias.name)) for alias in node.names]
    return [
        (alias.asname or alias.name, ("from", node.level, node.module, alias.name))
        for alias in node.names
        if alias.name != "*"
    ]


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _module_bindings(body: list[ast.stmt]) -> set[Binding]:
    # Runtime bindings only. An import under `if TYPE_CHECKING:` binds nothing
    # at runtime, and a function re-importing the same name is the standard way
    # to use it there without a circular import, so it is not a repeat.
    found: set[Binding] = set()
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            found.update(_bindings(stmt))
        elif isinstance(stmt, ast.If) and not _is_type_checking_guard(stmt):
            found |= _module_bindings(stmt.body) | _module_bindings(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                found |= _module_bindings(block)
            for handler in stmt.handlers:
                found |= _module_bindings(handler.body)
    return found


def check(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return []  # not ours to report; the interpreter and ruff will
    module_level = _module_bindings(tree.body)
    if not module_level:
        return []
    # Keyed on the import node, so an import inside a nested function is
    # reported once against its innermost function rather than once per
    # enclosing one. ast.walk is breadth-first, so the innermost function is
    # the last to claim each node.
    owner: dict[ast.AST, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    owner[node] = fn.name
    problems = []
    for node, fn_name in sorted(owner.items(), key=lambda item: item[0].lineno):
        assert isinstance(node, (ast.Import, ast.ImportFrom))
        for binding in _bindings(node):
            if binding in module_level:
                problems.append(
                    f"{path}:{node.lineno}: `{binding[0]}` is already imported "
                    f"at module level; this import in `{fn_name}` does nothing"
                )
    return problems


def main(argv: list[str]) -> int:
    if argv:
        files = [Path(a) for a in argv if a.endswith(".py")]
    else:
        files = sorted(p for root in ROOTS if Path(root).is_dir() for p in Path(root).rglob("*.py"))
    problems = [line for f in files for line in check(f)]
    for line in problems:
        print(line)
    if problems:
        print(f"\n{len(problems)} repeated import(s). Delete the function-local import.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
