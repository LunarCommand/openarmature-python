# Spec: conformance-adapter section 5.15 `message_repeat` (proposal 0119,
# spec v0.116.0), and its llm-provider counterpart on `mock_llm`'s `raises`.

"""The exception message a mock `raises` entry specifies.

A `raises` entry carries either a literal ``message`` or a ``message_repeat``
that synthesizes an oversized one, so a fixture can induce a message longer than
any cap without carrying it inline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


# Section 9: the category an adapter MUST raise for a malformed directive, or
# for a violated co-occurrence constraint between two known ones.
class FixtureSchemaInvalid(ValueError):
    """A fixture directive that is malformed, or that contradicts another."""


# Section 5.15: whole repetitions rather than a byte slice is the normative
# half. These directives feed the section 5.5.5 truncation contract, where the
# value under test is whether an implementation backtracks to a code-point
# boundary; a synthesizer that itself cut mid-sequence would hand the assertion
# an invalid string and hide the defect `utf8_valid` exists to catch.
def message_for(raises: Mapping[str, Any]) -> str:
    """The message a ``raises`` entry specifies, literal or synthesized.

    A ``message_repeat`` of ``{char, bytes}`` repeats ``char`` the largest whole
    number of times whose UTF-8 encoding is at most ``bytes``: never longer than
    ``bytes``, possibly shorter when ``char`` is multi-byte, and always valid
    UTF-8. A caller needing an exact byte count uses a single-byte ``char``.

    Raises:
        FixtureSchemaInvalid: the entry carries both ``message`` and
            ``message_repeat``, or its ``message_repeat`` is malformed.
    """
    # Keyed on presence, not on value: a bare `message_repeat:` in YAML reads
    # back as None, which is the directive supplied rather than omitted.
    if "message_repeat" not in raises:
        return str(raises.get("message", ""))
    if "message" in raises:
        # Section 5.15: an adapter MUST reject an entry carrying both, since the
        # intended message would be ambiguous.
        raise FixtureSchemaInvalid(
            "fixture_schema_invalid: a `raises` entry carries both `message` and "
            "`message_repeat`, which are mutually exclusive"
        )
    repeat = raises["message_repeat"]
    spec = cast("Mapping[str, Any]", repeat) if isinstance(repeat, Mapping) else None
    # Section 9: a malformed value for a known directive is rejected, never
    # defaulted. An empty `char` would otherwise divide by zero below.
    if spec is None or not spec.get("char") or "bytes" not in spec:
        raise FixtureSchemaInvalid(
            "fixture_schema_invalid: `message_repeat` requires a non-empty `char` and `bytes`"
        )
    char = cast("str", spec["char"])
    budget = int(cast("int", spec["bytes"]))
    width = len(char.encode("utf-8"))
    return char * (budget // width)
