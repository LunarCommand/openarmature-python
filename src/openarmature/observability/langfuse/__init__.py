# Spec mapping: realizes observability §8 (Langfuse backend mapping).
# Sibling to the OTel mapping in ``observability.otel``; both are
# self-contained consumers of the §6 event stream.
#
# Unlike the OTel subpackage, this one does NOT extras-gate: the
# observer is decoupled from any concrete SDK via the
# :class:`LangfuseClient` Protocol, and the bundled
# :class:`InMemoryLangfuseClient` recorder satisfies it without
# requiring the real ``langfuse`` package. Production users pass a
# real ``langfuse.Langfuse()`` instance (Protocol-compatible with the
# methods the observer calls) or write a thin adapter; the
# ``langfuse`` SDK install is on them.

"""Langfuse backend mapping for openarmature observability.

Public surface:

- :class:`LangfuseObserver` — observer-driven Langfuse Trace +
  Observation emission per spec observability §8.
- :class:`LangfuseClient` — Protocol the observer calls. Satisfied by
  the bundled :class:`InMemoryLangfuseClient` and (structurally) by
  the real ``langfuse.Langfuse`` SDK class.
- :class:`InMemoryLangfuseClient` — in-process recorder used by the
  conformance harness and useful for unit tests.
- :class:`LangfuseTrace` / :class:`LangfuseObservation` /
  :class:`LangfuseUsage` — captured-data records returned by the
  recorder.
"""

from __future__ import annotations

from .client import (
    InMemoryLangfuseClient,
    LangfuseClient,
    LangfuseGenerationHandle,
    LangfuseObservation,
    LangfuseSpanHandle,
    LangfuseTrace,
    LangfuseUsage,
    ObservationLevel,
    ObservationType,
)
from .observer import LangfuseObserver

__all__ = [
    "InMemoryLangfuseClient",
    "LangfuseClient",
    "LangfuseGenerationHandle",
    "LangfuseObservation",
    "LangfuseObserver",
    "LangfuseSpanHandle",
    "LangfuseTrace",
    "LangfuseUsage",
    "ObservationLevel",
    "ObservationType",
]
