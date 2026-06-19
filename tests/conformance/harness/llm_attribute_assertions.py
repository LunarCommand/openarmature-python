# Assertion helpers for the v0.17.0 LLM span attribute fixtures
# (012-021). Check the attribute shapes mandated by observability
# §5.5.1-§5.5.5 against the OTel spans the fixture run produced.
# Deliberately small + pure so fixture drivers can assemble their own
# assertions from this toolkit.
#
# Supported assertion-key shapes inside a span entry's expected block:
#   attributes_absent: [name, …]
#   attribute_parses_as_messages: {attr: expected_message_list}
#   attribute_parses_as_object: {attr: expected_object}
#   attribute_does_not_contain: {attr: {forbidden_substring_kind: …}}
#   attribute_truncation: {attr: {max_bytes, marker_pattern,
#                                 utf8_valid,
#                                 prefix_of_full_serialization}}
"""Assertion helpers for the LLM span attribute fixtures."""

from __future__ import annotations

import json
import re
from typing import Any

# Records the deterministic base64 prefixes emitted by the
# ``base64_data_synthetic`` directive across a fixture run. The
# ``attribute_does_not_contain`` assertion (with
# ``forbidden_substring_kind=synthetic_base64_prefix``) reads from here
# to know what substring to verify is absent from the emitted attribute.
# Reset per fixture run.
SYNTHESIZED_BASE64_PREFIXES: list[str] = []


def reset_synthesized_base64_prefixes() -> None:
    """Clear the synthesized-base64 record. Call at the start of each
    fixture run."""
    SYNTHESIZED_BASE64_PREFIXES.clear()


def record_synthesized_base64_prefix(prefix: str) -> None:
    """Record a base64 prefix the harness emitted via the
    ``base64_data_synthetic`` directive. The
    ``attribute_does_not_contain`` assertion uses these to verify
    image bytes don't leak into emitted attributes."""
    SYNTHESIZED_BASE64_PREFIXES.append(prefix)


def assert_attributes_absent(attrs: dict[str, Any], absent: list[str]) -> None:
    """Assert no name in ``absent`` appears in ``attrs``."""
    for name in absent:
        assert name not in attrs, f"attribute {name!r} MUST NOT be present; found value {attrs[name]!r}"


# Used by fixtures 013 (payload enabled) and 015 (inline-image
# redaction) to assert the parsed message structure without depending
# on bytewise JSON output.
def assert_attribute_parses_as_messages(
    attrs: dict[str, Any],
    expected_by_attr: dict[str, list[dict[str, Any]]],
) -> None:
    """Assert each ``attrs[name]`` is a JSON-encoded message array
    whose parse equals the expected list."""
    for attr_name, expected_messages in expected_by_attr.items():
        raw = attrs.get(attr_name)
        assert isinstance(raw, str), (
            f"attribute {attr_name!r} MUST be present as a string; got {type(raw).__name__}"
        )
        parsed = json.loads(raw)
        assert parsed == expected_messages, (
            f"attribute {attr_name!r} parsed-shape mismatch.\n"
            f"expected: {expected_messages!r}\ngot: {parsed!r}"
        )


# Used by fixture 018 (request.extras).
def assert_attribute_parses_as_object(
    attrs: dict[str, Any],
    expected_by_attr: dict[str, dict[str, Any]],
) -> None:
    """Assert each ``attrs[name]`` is a JSON-encoded object whose parse
    equals the expected dict."""
    for attr_name, expected_obj in expected_by_attr.items():
        raw = attrs.get(attr_name)
        assert isinstance(raw, str), (
            f"attribute {attr_name!r} MUST be present as a string; got {type(raw).__name__}"
        )
        parsed = json.loads(raw)
        assert parsed == expected_obj, (
            f"attribute {attr_name!r} parsed-shape mismatch.\nexpected: {expected_obj!r}\ngot: {parsed!r}"
        )


# The only ``forbidden_substring_kind`` currently defined is
# ``synthetic_base64_prefix`` — used by fixture 015 to verify inline
# image bytes don't leak through redaction. The kind names the most-
# recently synthesized base64 prefix recorded by the harness when it
# processed a ``base64_data_synthetic`` directive.
def assert_attribute_does_not_contain(
    attrs: dict[str, Any],
    forbidden_by_attr: dict[str, dict[str, Any]],
) -> None:
    """Assert each ``attrs[name]`` does not contain the forbidden
    substring identified by ``forbidden_substring_kind``."""
    for attr_name, spec in forbidden_by_attr.items():
        kind = spec.get("forbidden_substring_kind")
        if kind != "synthetic_base64_prefix":
            raise AssertionError(f"unknown forbidden_substring_kind: {kind!r}")
        if not SYNTHESIZED_BASE64_PREFIXES:
            raise AssertionError(
                "synthetic_base64_prefix assertion requested but no base64 was "
                "synthesized during this fixture run"
            )
        raw = attrs.get(attr_name)
        assert isinstance(raw, str), (
            f"attribute {attr_name!r} MUST be present as a string; got {type(raw).__name__}"
        )
        for forbidden in SYNTHESIZED_BASE64_PREFIXES:
            # 64-char prefix is plenty of signal that bytes leaked; the
            # full blob may have been truncated by the cap so substring
            # at the front is the realistic check.
            probe = forbidden[: min(64, len(forbidden))]
            assert probe not in raw, (
                f"attribute {attr_name!r} contains forbidden synthesized base64 "
                f"prefix (first 64 chars present); image bytes leaked through redaction"
            )


# truncation_by_attr entries carry:
#   max_bytes: configured cap (emitted byte length must be ≤ this)
#   marker_pattern: regex the attribute ends with (captures the
#     "…[truncated, M bytes total]" shape)
#   utf8_valid: when True, emitted attribute decodes as valid UTF-8
#     (catches mid-multi-byte-sequence cuts)
#   prefix_of_full_serialization: when True, the bytes preceding the
#     marker are a literal prefix of the full pre-truncation
#     serialization (supplied via full_serialization_by_attr[name])
def assert_attribute_truncation(
    attrs: dict[str, Any],
    truncation_by_attr: dict[str, dict[str, Any]],
    full_serialization_by_attr: dict[str, str] | None = None,
) -> None:
    """Verify the truncation contract on a payload attribute."""
    for attr_name, spec in truncation_by_attr.items():
        raw = attrs.get(attr_name)
        assert isinstance(raw, str), (
            f"attribute {attr_name!r} MUST be present as a string; got {type(raw).__name__}"
        )
        encoded = raw.encode("utf-8")
        max_bytes = int(spec["max_bytes"])
        assert len(encoded) <= max_bytes, (
            f"attribute {attr_name!r} byte length {len(encoded)} exceeds configured cap {max_bytes}"
        )
        marker_pattern = str(spec["marker_pattern"])
        match = re.search(marker_pattern, raw)
        assert match is not None, (
            f"attribute {attr_name!r} MUST end with marker matching /{marker_pattern}/; "
            f"got suffix {raw[-80:]!r}"
        )
        if bool(spec.get("utf8_valid", True)):
            try:
                encoded.decode("utf-8")
            except UnicodeDecodeError as e:
                raise AssertionError(
                    f"attribute {attr_name!r} contains invalid UTF-8 — "
                    f"truncation may have split a code point: {e}"
                ) from e
        if bool(spec.get("prefix_of_full_serialization", False)):
            if full_serialization_by_attr is None or attr_name not in full_serialization_by_attr:
                raise AssertionError(
                    f"prefix_of_full_serialization=True requires the harness to supply "
                    f"the full pre-truncation serialization for {attr_name!r}"
                )
            full = full_serialization_by_attr[attr_name]
            preceding = raw[: match.start()]
            assert full.startswith(preceding), (
                f"attribute {attr_name!r} prefix {preceding[:60]!r}… is not a prefix of "
                f"the full serialization {full[:60]!r}…"
            )


__all__ = [
    "SYNTHESIZED_BASE64_PREFIXES",
    "assert_attribute_does_not_contain",
    "assert_attribute_parses_as_messages",
    "assert_attribute_parses_as_object",
    "assert_attribute_truncation",
    "assert_attributes_absent",
    "record_synthesized_base64_prefix",
    "reset_synthesized_base64_prefixes",
]
