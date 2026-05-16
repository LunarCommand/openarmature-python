"""Content-derived hash helpers for prompt-management identity."""

from __future__ import annotations

import hashlib
import json

from openarmature.llm.messages import Message

# All hashes carry a ``sha256:`` prefix so future algorithm changes are
# self-describing. Spec §3 / §4 mark the hash function as SHOULD
# (cryptographic) and the canonical serialization as MUST be
# deterministic.
_HASH_PREFIX = "sha256:"


def compute_template_hash(template_source: str) -> str:
    """SHA-256 over the UTF-8 bytes of the raw template source."""
    digest = hashlib.sha256(template_source.encode("utf-8")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


def compute_rendered_hash(messages: list[Message]) -> str:
    """SHA-256 over a canonical JSON serialization of ``messages``.

    Preserves message boundaries, roles, content (including
    content-block structure per llm-provider §3.1), and tool_calls.
    ``json.dumps(sort_keys=True, separators=(",", ":"))`` over the
    per-message ``model_dump(mode="json")`` is deterministic across
    runs; datetimes serialize as ISO-8601 strings.
    """
    canonical = json.dumps(
        [m.model_dump(mode="json") for m in messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"
