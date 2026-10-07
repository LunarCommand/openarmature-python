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
from collections.abc import Iterator
from pathlib import Path

ROOTS = ("src", "tests", "examples", "scripts")

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

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
    # Only bindings every runtime path makes, so deleting a function-local
    # import the guard flags can never leave the name undefined. An import
    # under `if TYPE_CHECKING:` binds nothing at runtime, and a function
    # re-importing the same name is the standard way to use it there without
    # a circular import, so it is not a repeat.
    found: set[Binding] = set()
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            found.update(_bindings(stmt))
        elif isinstance(stmt, ast.If) and _is_type_checking_guard(stmt):
            found |= _module_bindings(stmt.orelse)
        elif isinstance(stmt, ast.If):
            found |= _module_bindings(stmt.body) & _module_bindings(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            paths = [_module_bindings(stmt.body + stmt.orelse)]
            paths += [_module_bindings(handler.body) for handler in stmt.handlers]
            found |= paths[0].intersection(*paths[1:]) | _module_bindings(stmt.finalbody)
    return found


def _own_scope(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    # Stops at nested scopes: an import in a nested class body sets a class
    # attribute, and nested functions are checked as functions of their own.
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _scope_names(fn: ast.FunctionDef | ast.AsyncFunctionDef, skip: set[Binding]) -> set[str]:
    # Every name the function's own scope binds, except through imports whose
    # binding is in `skip`. Comprehension targets are included; that only
    # makes the guard report less.
    args = fn.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    names = {a.arg for a in params if a is not None}
    for node in _own_scope(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(name for name, target in _bindings(node) if (name, target) not in skip)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def check(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return []  # not ours to report; the interpreter and ruff will
    module_level = _module_bindings(tree.body)
    if not module_level:
        return []
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    problems: list[tuple[int, str]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # A name bound anywhere else in this function, or in an enclosing
        # function, would not fall through to the module import if this
        # import were deleted. Class scopes are skipped by name resolution.
        shadowed = _scope_names(fn, module_level)
        outer = parents.get(fn)
        while outer is not None:
            if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
                shadowed |= _scope_names(outer, set())
            outer = parents.get(outer)
        for node in _own_scope(fn):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for binding in _bindings(node):
                if binding in module_level and binding[0] not in shadowed:
                    problems.append(
                        (
                            node.lineno,
                            f"{path}:{node.lineno}: `{binding[0]}` is already imported "
                            f"at module level; this import in `{fn.name}` does nothing",
                        )
                    )
    return [message for _, message in sorted(problems)]


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
