"""The read trace records each key looked up in a nested block, at the block's position.

These run on a small stand-in fixture rather than the spec corpus, so the
mechanism the nested read check rests on is checked wherever the tests run.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import pytest

from .harness import read_trace

_FIXTURE = """\
cases:
  - name: one
    expected:
      a: 1
      b: 2
    calls:
      - operation: complete
        expected: {c: 3}
"""

_RUNNER = "tests/conformance/test_read_trace.py"


@pytest.fixture
def path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fixture file laid out like the corpus, under a stand-in spec root,
    with its reads kept out of the session's record."""
    fixture = tmp_path / "stand-in" / "conformance" / "001-stand-in.yaml"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(_FIXTURE)
    monkeypatch.setattr(read_trace, "_SPEC_ROOT", tmp_path.resolve())
    monkeypatch.setattr(read_trace, "_READS", defaultdict[tuple[str, str, str], set[tuple[str, str]]](set))
    return fixture


def _read(path: Path) -> frozenset[tuple[str, str]]:
    return read_trace.reads().get(("stand-in", path.stem, _RUNNER), frozenset())


def test_each_lookup_by_name_records_the_key_at_its_blocks_position(path: Path) -> None:
    expected = read_trace.load(path)["cases"][0]["expected"]
    _ = expected["a"]
    expected.get("b")
    expected.pop("c", None)
    expected.setdefault("d", 4)
    _ = "e" in expected
    assert _read(path) == {("expected", key) for key in "abcde"}


def test_a_call_and_its_expected_record_at_their_own_positions(path: Path) -> None:
    call = read_trace.load(path)["cases"][0]["calls"][0]
    _ = call["operation"]
    _ = call["expected"]["c"]
    assert _read(path) == {("calls[]", "operation"), ("calls[]", "expected"), ("expected", "c")}


def test_a_copy_keeps_recording(path: Path) -> None:
    expected = read_trace.load(path)["cases"][0]["expected"]
    _ = expected.copy()["a"]
    _ = copy.copy(expected)["b"]
    _ = copy.deepcopy(expected).get("c")
    assert _read(path) == {("expected", "a"), ("expected", "b"), ("expected", "c")}


def test_reading_in_bulk_records_nothing(path: Path) -> None:
    # The loud direction: a key a runner consumes only in bulk shows as unread
    # rather than read, so a copy that drops the trace cannot pass in silence.
    expected = read_trace.load(path)["cases"][0]["expected"]
    _ = dict(expected)["a"]
    _ = {**expected}["a"]
    _ = list(expected.items())
    _ = list(expected)
    assert expected == {"a": 1, "b": 2}
    # One lookup by name, so an empty record cannot pass for a dead one.
    _ = expected["b"]
    assert _read(path) == {("expected", "b")}


def test_each_load_returns_a_fresh_document(path: Path) -> None:
    read_trace.load(path)["cases"][0]["expected"]["a"] = 99
    assert read_trace.load(path)["cases"][0]["expected"]["a"] == 1
