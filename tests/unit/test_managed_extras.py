"""The shared managed-field collision resolver (proposals 0105 + 0108)."""

# Proposal 0105 + 0108 (llm-provider §6, inherited by retrieval §10). The
# resolver is capability-neutral; the per-mapping wiring is exercised in the
# provider test suites. Behavior ships ahead of the v0.88.0 pin (unit-tested);
# fixtures ride the pin bump.

from typing import Any

import pytest

from openarmature._managed_extras import apply_managed_extras
from openarmature.llm.errors import ProviderInvalidRequest


def test_unmanaged_key_rides_through_untouched() -> None:
    body: dict[str, Any] = {"model": "m"}
    apply_managed_extras(body, {"guided_decoding": {"grammar": "x"}}, {"model": "reject"})
    assert body == {"model": "m", "guided_decoding": {"grammar": "x"}}


def test_produced_body_key_absent_from_managed_fails_loud() -> None:
    # Contract guard: if a mapping produces a body key but forgets to declare it
    # in `managed`, a colliding extra must fail loud rather than silently drop it
    # (0105 forbids the silent drop) or silently override the produced value.
    # This is a mapping-author under-declaration, so it is a ValueError, not the
    # caller-facing ProviderInvalidRequest.
    body: dict[str, Any] = {"produced_key": "mapping_value"}
    with pytest.raises(ValueError, match="did not declare it in `managed`"):
        apply_managed_extras(body, {"produced_key": "caller_value"}, {})


def test_reject_arm_conflicting_scalar_raises() -> None:
    # A conflicting non-additive collision is rejected pre-send, not dropped and
    # not silently overriding the managed value.
    body: dict[str, Any] = {"temperature": 0.7}
    with pytest.raises(ProviderInvalidRequest, match="temperature"):
        apply_managed_extras(body, {"temperature": 0.2}, {"temperature": "reject"})


def test_reject_arm_matching_scalar_is_a_noop() -> None:
    # A caller value EQUAL to the managed value is a redundant no-op: no raise,
    # managed value untouched.
    body: dict[str, Any] = {"temperature": 0.7}
    apply_managed_extras(body, {"temperature": 0.7}, {"temperature": "reject"})
    assert body == {"temperature": 0.7}


def test_reject_arm_object_field() -> None:
    # The reject arm covers objects the mapping constructs wholesale, not only
    # scalars (response_format / stream_options).
    managed_rf = {"type": "json_schema", "json_schema": {"name": "P"}}
    body: dict[str, Any] = {"response_format": managed_rf}
    with pytest.raises(ProviderInvalidRequest):
        apply_managed_extras(body, {"response_format": {"type": "text"}}, {"response_format": "reject"})
    # An identical object is a no-op.
    body2: dict[str, Any] = {"response_format": dict(managed_rf)}
    apply_managed_extras(body2, {"response_format": dict(managed_rf)}, {"response_format": "reject"})
    assert body2 == {"response_format": managed_rf}


def test_merge_arm_appends_after_managed_deduped() -> None:
    # stop: declared value first, caller's appended, de-duplicated.
    body: dict[str, Any] = {"stop": ["A"]}
    apply_managed_extras(body, {"stop": ["B", "A"]}, {"stop": "merge"})
    assert body["stop"] == ["A", "B"]


def test_merge_arm_coerces_scalar_extra_to_list() -> None:
    # The wire scalar-or-array shape (OpenAI stop): a scalar-string extra is
    # coerced to a one-element list before merging.
    body: dict[str, Any] = {"stop": ["A"]}
    apply_managed_extras(body, {"stop": "C"}, {"stop": "merge"})
    assert body["stop"] == ["A", "C"]


def test_merge_arm_matching_value_collapses() -> None:
    # The merge arm's form of the matching no-op: a caller value already present
    # collapses via the dedup.
    body: dict[str, Any] = {"stop": ["A"]}
    apply_managed_extras(body, {"stop": ["A"]}, {"stop": "merge"})
    assert body["stop"] == ["A"]


def test_merge_arm_onto_absent_managed_value() -> None:
    # A merge field the mapping did not set (managed value None) still folds the
    # caller's value in as a fresh list.
    body: dict[str, Any] = {}
    apply_managed_extras(body, {"stop": ["A", "A", "B"]}, {"stop": "merge"})
    assert body["stop"] == ["A", "B"]


def test_merge_arm_malformed_list_element_is_treated_as_absent() -> None:
    # 0113: a merge extra with a non-string element is structurally malformed and
    # treated as absent -- all-or-nothing, the well-formed "END" is NOT salvaged
    # and no error is raised. The managed value stands alone.
    body: dict[str, Any] = {"stop": ["STOP"]}
    apply_managed_extras(body, {"stop": ["END", 123]}, {"stop": "merge"})
    assert body["stop"] == ["STOP"]


def test_merge_arm_malformed_scalar_is_treated_as_absent() -> None:
    # 0113: a non-string scalar has no string-or-array-of-strings shape, so it is
    # malformed and treated as absent rather than coerced onto the wire.
    body: dict[str, Any] = {"stop": ["STOP"]}
    apply_managed_extras(body, {"stop": 5}, {"stop": "merge"})
    assert body["stop"] == ["STOP"]


def test_merge_arm_empty_string_element_is_well_formed_and_merges() -> None:
    # 0113 judges malformation STRUCTURALLY: an empty string is a well-typed
    # string, so ["A", ""] is a list of strings -- well-formed -- and merges. The
    # mapping does not semantically validate element values, so a "" stop token
    # reaches the wire and the provider decides. Pins the structural-only rule
    # against a future narrowing to a semantic (non-empty) check.
    body: dict[str, Any] = {"stop": ["STOP"]}
    apply_managed_extras(body, {"stop": ["A", ""]}, {"stop": "merge"})
    assert body["stop"] == ["STOP", "A", ""]


def test_merge_arm_empty_list_is_well_formed_no_op() -> None:
    # An empty list is well-formed (every element -- none -- is a string) and
    # merges as a no-op, leaving the managed value untouched.
    body: dict[str, Any] = {"stop": ["STOP"]}
    apply_managed_extras(body, {"stop": []}, {"stop": "merge"})
    assert body["stop"] == ["STOP"]


def test_conditionally_managed_field_when_not_produced_rides_untouched() -> None:
    # The escape hatch: a field the mapping is NOT currently producing is absent
    # from `managed`, so its extra rides untouched (e.g. Jina task with no
    # input_type). Here `task` is unmanaged this call.
    body: dict[str, Any] = {"model": "m"}
    apply_managed_extras(body, {"task": "classification"}, {"model": "reject"})
    assert body == {"model": "m", "task": "classification"}


def test_extras_mapping_is_not_mutated() -> None:
    body: dict[str, Any] = {"stop": ["A"]}
    extras = {"stop": ["B"]}
    apply_managed_extras(body, extras, {"stop": "merge"})
    assert extras == {"stop": ["B"]}


def test_multiple_keys_mixed_arms() -> None:
    body: dict[str, Any] = {"model": "m", "stop": ["A"], "temperature": 0.5}
    apply_managed_extras(
        body,
        {"stop": ["B"], "user_meta": "keep"},
        {"model": "reject", "stop": "merge", "temperature": "reject"},
    )
    assert body["stop"] == ["A", "B"]
    assert body["user_meta"] == "keep"
    assert body["model"] == "m"
