"""Trace-id derivation helpers, SDK-independent.

Pure functions for mapping an OA ``invocation_id`` to the 32-char hex
Langfuse ``trace.id``. Imports only stdlib (``uuid``, ``hashlib``), so
this module is importable without the ``[langfuse]`` extras —
operators and tooling can compute trace ids without pulling in the
SDK.
"""

from __future__ import annotations

import hashlib
import uuid as _uuid


def _is_uuid(value: str) -> bool:
    try:
        _uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


# Per observability §8.4.1: Langfuse v4 trace ids are 128-bit values
# rendered as 32 lowercase hex. A UUID invocation_id maps to its hex
# form (dashes stripped). A non-UUID maps to the first 16 bytes of
# SHA-256(invocation_id) as 32 hex (the same derivation as Langfuse's
# create_trace_id(seed), so a consumer can reproduce it); the raw id is
# also written to trace.metadata.invocation_id (see the adapter's
# `trace`) for lookup.
def _to_otel_trace_id(trace_id: str) -> str:
    """Return the 32-char hex Langfuse trace id for an OA invocation_id."""
    if _is_uuid(trace_id):
        return _uuid.UUID(trace_id).hex
    return hashlib.sha256(trace_id.encode("utf-8")).digest()[:16].hex()


def langfuse_trace_id(invocation_id: str) -> str:
    """Return the Langfuse ``trace.id`` for an OA ``invocation_id``.

    Public helper for mapping a logged ``invocation_id`` (a dashed UUID
    or a caller-supplied non-UUID string) to the 32-char hex
    ``trace.id`` Langfuse stores, e.g. to build a direct trace URL.
    """
    return _to_otel_trace_id(invocation_id)
