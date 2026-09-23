# Spec: conformance-adapter section 5.15 `message_repeat` and observability
# section 5.5.5's per-value byte cap (proposals 0119 / 0125).

"""Assertions for machinery whose only fixture is deferred.

Fixture 160 is the corpus consumer of `message_repeat`, `metadata_truncation`
and a non-default `langfuse_observer.payload_byte_cap`, and it cannot run until
its driver exists. Without these, all three ship asserted by nothing: verified
once by hand at the time of writing and never again.

Each was measured before this file existed. Removing `payload_byte_cap` from the
observer kwargs entirely left the whole suite green, including fixture 023,
which declares the directive.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from .harness.mock_messages import FixtureSchemaInvalid, message_for
from .test_observability import _assert_metadata_truncation, _langfuse_observer_kwargs

_MARKER = r"…\[truncated, [0-9]+ bytes total\]$"


def test_message_repeat_fills_the_budget_exactly_when_the_width_divides_it() -> None:
    # Fixture 160's own arithmetic: 102400 / 4 = 25600 whole repetitions.
    message = message_for({"message_repeat": {"char": "\U0001f600", "bytes": 102400}})
    assert len(message) == 25600
    assert len(message.encode("utf-8")) == 102400


def test_message_repeat_rounds_down_to_whole_repetitions() -> None:
    # Section 5.15: the largest whole number of repetitions at most `bytes`,
    # never longer, possibly shorter for a multi-byte char. A synthesizer that
    # sliced to the budget would emit 10 bytes ending mid-sequence, handing
    # `utf8_valid` an invalid string and hiding the defect it exists to catch.
    message = message_for({"message_repeat": {"char": "\U0001f600", "bytes": 10}})
    assert len(message.encode("utf-8")) == 8
    assert message.encode("utf-8").decode("utf-8") == message


def test_message_repeat_and_message_are_mutually_exclusive() -> None:
    # Section 5.15: an adapter MUST reject an entry carrying both, since the
    # intended message would be ambiguous.
    with pytest.raises(FixtureSchemaInvalid):
        message_for({"message": "a", "message_repeat": {"char": "b", "bytes": 4}})


def test_a_null_valued_message_repeat_still_collides_with_a_literal_message() -> None:
    # The shape a value-keyed check waves through. YAML writes a bare
    # `message_repeat:` as None, so reading the VALUE makes the directive
    # indistinguishable from an omitted one and the mutual-exclusion raise above
    # never fires. Both keys are supplied; section 5.15 rejects that.
    with pytest.raises(FixtureSchemaInvalid):
        message_for({"message": "a", "message_repeat": None})


def test_a_malformed_message_repeat_is_rejected_rather_than_defaulted() -> None:
    # Section 9: `fixture_schema_invalid` covers a malformed type for a known
    # directive, and the adapter MUST raise rather than infer a default. A
    # null-valued directive alone read back as the empty message before this,
    # which is that inference.
    with pytest.raises(FixtureSchemaInvalid):
        message_for({"message_repeat": None})
    with pytest.raises(FixtureSchemaInvalid):
        message_for({"message_repeat": {"char": "a"}})
    # A scalar where the directive wants a mapping. `message_repeat: boom` is
    # the shape a fixture author writes when reaching for the literal form, and
    # it is the only one the mapping check alone still answers: a null value is
    # already caught downstream.
    with pytest.raises(FixtureSchemaInvalid):
        message_for({"message_repeat": "boom"})


def test_an_empty_char_is_rejected_as_a_schema_error_not_a_zero_division() -> None:
    # `char: ""` measures zero bytes wide, so it reaches `budget // width` and
    # dies on the division. The raise has to carry section 9's category for a
    # fixture author to read it as their error rather than ours.
    with pytest.raises(FixtureSchemaInvalid):
        message_for({"message_repeat": {"char": "", "bytes": 4}})


def test_a_literal_message_passes_through() -> None:
    assert message_for({"message": "boom"}) == "boom"
    assert message_for({}) == ""


def _observation(value: Any) -> SimpleNamespace:
    return SimpleNamespace(name="openarmature.llm.complete", metadata={"error_message": value})


def test_the_byte_cap_claim_catches_an_oversized_value() -> None:
    over = "x" * 100 + "…[truncated, 9999 bytes total]"
    with pytest.raises(AssertionError, match="over the"):
        _assert_metadata_truncation(_observation(over), {"error_message": {"max_bytes": 32}})


def test_the_marker_claim_catches_a_value_that_was_never_truncated() -> None:
    # A byte cap alone passes here: "HTTP 503" is comfortably under any cap, and
    # a short message is indistinguishable from a truncated one without this.
    with pytest.raises(AssertionError, match="no truncation marker"):
        _assert_metadata_truncation(_observation("HTTP 503"), {"error_message": {"marker_pattern": _MARKER}})


def test_the_utf8_claim_catches_a_cut_through_a_multi_byte_sequence() -> None:
    # Section 5.5.5 step 4. The cap and the marker both pass on this value; only
    # a code-point-boundary check rejects it.
    broken = ("\U0001f600" * 3).encode("utf-8")[:10].decode("utf-8", errors="surrogateescape")
    with pytest.raises(AssertionError, match="not valid UTF-8"):
        _assert_metadata_truncation(
            _observation(broken + "…[truncated, 12 bytes total]"),
            {"error_message": {"utf8_valid": True}},
        )


def test_a_claim_this_harness_cannot_answer_raises() -> None:
    # `prefix_of_full_serialization` needs the pre-truncation message, which this
    # comparator does not receive. It raises rather than asserting the weaker
    # check its name would imply: a placeholder carrying a marker passed that.
    with pytest.raises(AssertionError, match="cannot evaluate"):
        _assert_metadata_truncation(
            _observation("placeholder…[truncated, 102400 bytes total]"),
            {"error_message": {"prefix_of_full_serialization": True}},
        )


def test_a_well_formed_truncation_satisfies_every_answerable_claim() -> None:
    # The positive control. Without it the four rejections above are satisfied by
    # a comparator that rejects everything.
    value = "\U0001f600" * 3 + "…[truncated, 102400 bytes total]"
    _assert_metadata_truncation(
        _observation(value),
        {"error_message": {"max_bytes": 1024, "marker_pattern": _MARKER, "utf8_valid": True}},
    )


def test_the_observer_directive_applies_the_declared_cap() -> None:
    # Removing this application left the whole suite green, fixture 023 included,
    # which is why the assertion lives here rather than resting on a fixture.
    assert _langfuse_observer_kwargs({"langfuse_observer": {"payload_byte_cap": 1024}}) == {
        "payload_byte_cap": 1024
    }
    assert _langfuse_observer_kwargs({"langfuse_observer": {"disable_provider_payload": False}}) == {
        "disable_provider_payload": False
    }
    assert _langfuse_observer_kwargs({}) == {}


@pytest.mark.parametrize("cap", [-5, 0, "big", True])
def test_a_cap_that_cannot_truncate_is_rejected(cap: Any) -> None:
    # A cap at or below zero produces no truncation at all, so a case declaring
    # one passes or fails for reasons unrelated to its subject. `True` is listed
    # because it is an `int` in Python, so a bare isinstance check accepts it as
    # a one-byte cap.
    with pytest.raises(AssertionError, match="positive integer"):
        _langfuse_observer_kwargs({"langfuse_observer": {"payload_byte_cap": cap}})


@pytest.mark.parametrize("raw", ["disable_provider_payload", ["a"], 7])
def test_a_non_mapping_directive_is_rejected_as_a_shape_problem(raw: Any) -> None:
    # A string would otherwise reach the unknown-key check, which iterates it
    # into CHARACTERS and reports them as sub-keys.
    with pytest.raises(AssertionError, match="must be a mapping"):
        _langfuse_observer_kwargs({"langfuse_observer": raw})


def test_an_unapplied_sub_key_is_rejected() -> None:
    with pytest.raises(AssertionError, match="does not apply"):
        _langfuse_observer_kwargs({"langfuse_observer": {"disable_llm_spans": True}})
