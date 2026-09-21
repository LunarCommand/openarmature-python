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


class FixtureSchemaInvalid(ValueError):
    """A fixture directive the spec requires an adapter to reject (section 9)."""


def message_for(raises: Mapping[str, Any]) -> str:
    """The message a ``raises`` entry specifies, literal or synthesized.

    Section 5.15: repeat ``char`` to the largest whole number of repetitions
    whose UTF-8 encoding is at most ``bytes``. Never longer than ``bytes``,
    possibly shorter when ``char`` is multi-byte, and always valid UTF-8.

    Whole repetitions rather than a byte slice is the normative half. These
    directives feed the section 5.5.5 truncation contract, where the value under
    test is whether an implementation backtracks to a code-point boundary; a
    synthesizer that itself cut mid-sequence would hand the assertion an invalid
    string and hide the defect `utf8_valid` exists to catch.
    """
    repeat = raises.get("message_repeat")
    if repeat is None:
        return str(raises.get("message", ""))
    if "message" in raises:
        # Section 5.15: an adapter MUST reject an entry carrying both, since the
        # intended message would be ambiguous.
        raise FixtureSchemaInvalid(
            "fixture_schema_invalid: a `raises` entry carries both `message` and "
            "`message_repeat`, which are mutually exclusive"
        )
    spec = cast("Mapping[str, Any]", repeat)
    char = cast("str", spec["char"])
    budget = int(cast("int", spec["bytes"]))
    width = len(char.encode("utf-8"))
    return char * (budget // width)
