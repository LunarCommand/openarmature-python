"""Load conformance fixtures so the keys a runner reads from them are recorded."""

# The evidence behind the nested read check in `test_nested_vocabulary`
# (conformance-adapter section 8.2, one level below the case). A key's name
# cannot say which mapping it was read from, since the same key occurs at several
# positions, so each nested block comes back tagged with its positions and every
# key looked up in it by name is recorded against them.

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any, cast

import yaml

from .fixture_walk import as_dict, blocks

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC_ROOT = _REPO_ROOT / "openarmature-spec" / "spec"

# `(capability, fixture, runner)` to the `(position, key)` pairs read so far.
_READS: defaultdict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)


class _Traced(dict[str, Any]):
    """A nested fixture block that records each key looked up in it by name.

    `copy()` and `copy.deepcopy` give blocks that keep recording. Iteration,
    `dict(...)`, `{**...}`, `==` and model validation read in bulk and record
    nothing, so a key consumed only that way stays unread.
    """

    __slots__ = ("_positions", "_sink")
    _positions: tuple[str, ...]
    _sink: set[tuple[str, str]]

    def __init__(self, positions: tuple[str, ...], sink: set[tuple[str, str]]) -> None:
        super().__init__()
        self._positions = positions
        self._sink = sink

    def _record(self, key: object) -> None:
        if isinstance(key, str):
            self._sink.update((position, key) for position in self._positions)

    def __getitem__(self, key: str, /) -> Any:
        self._record(key)
        return super().__getitem__(key)

    def __contains__(self, key: object, /) -> bool:
        self._record(key)
        return super().__contains__(key)

    def get(self, key: str, default: Any = None, /) -> Any:
        self._record(key)
        return super().get(key, default)

    def pop(self, key: str, /, *default: Any) -> Any:
        self._record(key)
        return super().pop(key, *default)

    def setdefault(self, key: str, default: Any = None, /) -> Any:
        self._record(key)
        return super().setdefault(key, default)

    def copy(self) -> _Traced:
        twin = _Traced(self._positions, self._sink)
        twin.update(self)
        return twin

    def __copy__(self) -> _Traced:
        return self.copy()

    # The default deep copy would copy the sink too, and the copy's reads would
    # land in a set nothing looks at.
    def __deepcopy__(self, memo: dict[int, Any]) -> _Traced:
        twin = _Traced(self._positions, self._sink)
        memo[id(self)] = twin
        for key, value in self.items():
            twin[key] = copy.deepcopy(value, memo)
        return twin


# Cached, so a fixture is parsed and walked once however many tests load it. The
# cache keeps the document alive, which keeps the ids stable.
@cache
def _parsed(path: Path) -> tuple[Any, dict[int, tuple[str, ...]]]:
    """A fixture's document, and the positions of each nested block in it by `id`."""
    doc: Any = yaml.safe_load(path.read_text())
    positions: dict[int, tuple[str, ...]] = {}
    if (mapping := as_dict(doc)) is not None:
        for position, _, block in blocks(mapping):
            held = positions.get(id(block), ())
            if position not in held:
                positions[id(block)] = (*held, position)
    return doc, positions


def _copy(
    value: Any, positions: dict[int, tuple[str, ...]], sink: set[tuple[str, str]], memo: dict[int, Any]
) -> Any:
    """A copy of `value` whose nested blocks record their reads into `sink`."""
    if id(value) in memo:
        return memo[id(value)]
    if (mapping := as_dict(value)) is not None:
        held = positions.get(id(mapping))
        twin: dict[str, Any] = _Traced(held, sink) if held else {}
        memo[id(mapping)] = twin
        for key, item in mapping.items():
            twin[key] = _copy(item, positions, sink, memo)
        return twin
    if isinstance(value, list):
        source = cast("list[Any]", value)
        items: list[Any] = []
        memo[id(source)] = items
        items.extend(_copy(item, positions, sink, memo) for item in source)
        return items
    return value


def load(path: Path) -> dict[str, Any]:
    """Parse a fixture file, recording what the calling runner reads from its nested blocks.

    The reads are attributed to the file whose code calls this. Each call returns
    a fresh document, so a runner may change what it gets back.
    """
    resolved = path.resolve()
    doc, positions = _parsed(resolved)
    if not resolved.is_relative_to(_SPEC_ROOT):
        return cast("dict[str, Any]", copy.deepcopy(doc))
    caller = Path(sys._getframe(1).f_code.co_filename).resolve()
    runner = str(caller.relative_to(_REPO_ROOT)) if caller.is_relative_to(_REPO_ROOT) else str(caller)
    sink = _READS[(resolved.parent.parent.name, resolved.stem, runner)]
    return cast("dict[str, Any]", _copy(doc, positions, sink, {}))


def reads() -> dict[tuple[str, str, str], frozenset[tuple[str, str]]]:
    """`(capability, fixture, runner)` to the `(position, key)` pairs that runner
    read from that fixture, for every fixture loaded through here so far."""
    return {loaded: frozenset(pairs) for loaded, pairs in _READS.items()}
