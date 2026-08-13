"""Provider-faithful Langfuse client fakes for the conformance harness."""

# Spec basis: conformance-adapter §6.4 (proposal 0115, extended by 0116 / 0117 /
# 0118). The bundled InMemoryLangfuseClient records observation content but has no
# provider at all, so a leak onto a shared TracerProvider was not expressible and
# the 0114 isolation obligations shipped unfixtured.
#
# These fakes record content as the in-memory client does AND emit each
# observation as an OTel span through their bound TracerProvider, the way a real
# Langfuse v4 client does. An exporter installed on that provider therefore
# observes any observation that reaches it, which is what the leak assertions
# read: content assertions read the recorded side, leak assertions read the
# provider side.
#
# The emitted spans carry the real `langfuse.*` attribute namespace rather than a
# harness-invented summary. That is deliberate. A double that stamps its own
# verdict, read by an assertion, tests the double's opinion instead of the
# property -- any channel the double's predicate missed would be invisible, and
# the assertion could not be cross-checked against a real client.
#
# SCOPE. This module serves `langfuse_client: {mode: supplied}` only, where the
# CALLER hands openarmature an object satisfying its LangfuseClient protocol.
#
# `mode: credentials` is NOT faked. There openarmature constructs the client
# itself and wraps it in LangfuseSDKAdapter, which drives the real SDK's private
# surface (start_observation, _otel_tracer, _create_remote_parent_span,
# _get_otel_trace_id, and the real langfuse._client.span classes). Faking that is
# chasing a private API, and the provider observations would not land in a
# recorded side anyway, since the adapter builds real SDK span objects for them.
# That mode uses a REAL client with its egress neutralised -- see
# `langfuse_real_client.py`.

from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace as otel_trace

from openarmature.observability.langfuse.client import (
    InMemoryLangfuseClient,
    LangfuseObservation,
)

# Attribute names a Langfuse v4 client writes on the spans it exports, verified
# against langfuse/_client/attributes.py. The observation NAME is the OTel span
# name, not an attribute. The leak assertions select openarmature's Langfuse
# observations out of an exporter by the `langfuse.observation.` prefix, mirroring
# how the OTel-side check selects openarmature spans.
OBSERVATION_TYPE_ATTR = "langfuse.observation.type"
OBSERVATION_INPUT_ATTR = "langfuse.observation.input"
OBSERVATION_OUTPUT_ATTR = "langfuse.observation.output"
OBSERVATION_METADATA_ATTR = "langfuse.observation.metadata"
OBSERVATION_LEVEL_ATTR = "langfuse.observation.level"
OBSERVATION_STATUS_MESSAGE_ATTR = "langfuse.observation.status_message"
TRACE_INPUT_ATTR = "langfuse.trace.input"
TRACE_OUTPUT_ATTR = "langfuse.trace.output"

_OBSERVATION_PREFIX = "langfuse.observation."


def _encode(value: Any) -> str:
    """Render a payload the way the SDK does, as a JSON string attribute."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(value)


class _ProviderBinding:
    """The shared provider side: which TracerProvider, and how a span is emitted.

    Composed rather than inherited so it can sit under both client shapes without
    entangling with the dataclass base one of them already has.
    """

    # `tracer_provider=None` means NONE SUPPLIED, which the SDK resolves to the
    # globally-registered provider; that is the priming construction §6.4
    # describes and is a different state from tracing being off. Collapsing the
    # two onto one argument would make a primed client both unable to leak and
    # classified isolated, so 158's raise arm could never fire.
    def __init__(self, tracer_provider: Any = None, *, tracing_enabled: bool = True) -> None:
        resolved = None
        if tracing_enabled:
            resolved = tracer_provider if tracer_provider is not None else otel_trace.get_tracer_provider()
        self.tracer_provider = resolved
        self.tracing_enabled = tracing_enabled
        self.tracer = resolved.get_tracer("langfuse") if resolved is not None else None
        # Trace-level input / output ride the trace's first exported span, as the
        # SDK writes them onto the span that opens the trace.
        self.pending_trace_attrs: dict[str, dict[str, str]] = {}

    @property
    def active(self) -> bool:
        return self.tracer is not None

    def note_trace_payload(self, trace_id: str, *, input: Any = None, output: Any = None) -> None:
        pending = self.pending_trace_attrs.setdefault(trace_id, {})
        if input is not None:
            pending[TRACE_INPUT_ATTR] = _encode(input)
        if output is not None:
            pending[TRACE_OUTPUT_ATTR] = _encode(output)

    def emit(self, trace_id: str, observation: LangfuseObservation) -> None:
        """Emit one recorded observation as an OTel span on the bound provider."""
        if self.tracer is None:
            return
        span = self.tracer.start_span(observation.name or f"langfuse.{observation.type}")
        span.set_attribute(OBSERVATION_TYPE_ATTR, observation.type)
        if observation.input is not None:
            span.set_attribute(OBSERVATION_INPUT_ATTR, _encode(observation.input))
        if observation.output is not None:
            span.set_attribute(OBSERVATION_OUTPUT_ATTR, _encode(observation.output))
        if observation.metadata:
            span.set_attribute(OBSERVATION_METADATA_ATTR, _encode(observation.metadata))
        span.set_attribute(OBSERVATION_LEVEL_ATTR, observation.level)
        if observation.status_message is not None:
            span.set_attribute(OBSERVATION_STATUS_MESSAGE_ATTR, observation.status_message)
        for name, value in self.pending_trace_attrs.pop(trace_id, {}).items():
            span.set_attribute(name, value)
        span.end()

    def drain_orphan_trace_payloads(self) -> None:
        """Emit trace payload that never rode an observation.

        A state-channel leak on an invocation that produced no observations would
        otherwise never reach the provider, and the assertion would read a clean
        exporter.
        """
        for trace_id, attrs in list(self.pending_trace_attrs.items()):
            if attrs and self.tracer is not None:
                span = self.tracer.start_span("langfuse.trace")
                # Carries the observation-namespace type too, or the leak selector
                # (which matches on `langfuse.observation.`) drops the very span
                # this method exists to surface, and a state-channel leak on an
                # invocation with no observations reads as a clean exporter.
                span.set_attribute(OBSERVATION_TYPE_ATTR, "trace")
                for name, value in attrs.items():
                    span.set_attribute(name, value)
                span.end()
            self.pending_trace_attrs.pop(trace_id, None)

    def flush_provider(self, timeout_ms: int) -> bool:
        provider: Any = self.tracer_provider
        if provider is None or not hasattr(provider, "force_flush"):
            return True
        return bool(provider.force_flush(timeout_ms))


class _Resources:
    """Stands in for the SDK's resource manager, exposing the one attribute
    openarmature reads to establish a client's provider binding."""

    def __init__(self, tracer_provider: Any) -> None:
        self.tracer_provider = tracer_provider


# ---------------------------------------------------------------------------
# Protocol shape -- what a caller supplies (mode: supplied)
# ---------------------------------------------------------------------------


class _ExportingHandle:
    """Wraps a recorded-observation handle and exports the span at `end()`."""

    # The real client keeps its span open for the observation's lifetime and
    # exports at end, so anything written by a later `update()` / `end(...)` is on
    # the exported span. Exporting at CREATION instead would classify a payload
    # that arrives late as absent -- `status_message`, for one, is set from
    # harvested exception text after the fact.

    def __init__(self, inner: Any, binding: _ProviderBinding, trace_id: str) -> None:
        self._inner = inner
        self._binding = binding
        self._trace_id = trace_id

    @property
    def id(self) -> str:
        return str(self._inner.id)

    @property
    def observation(self) -> LangfuseObservation:
        obs: LangfuseObservation = self._inner.observation
        return obs

    def update(self, **fields: Any) -> None:
        self._inner.update(**fields)

    def end(self, **kwargs: Any) -> None:
        self._inner.end(**kwargs)
        self._binding.emit(self._trace_id, self.observation)


class ProviderFaithfulLangfuseClient(InMemoryLangfuseClient):
    """A protocol-shaped client that also exports through a TracerProvider."""

    def __init__(self, tracer_provider: Any = None, *, tracing_enabled: bool = True) -> None:
        super().__init__()
        self._binding = _ProviderBinding(tracer_provider, tracing_enabled=tracing_enabled)
        self.tracer_provider = self._binding.tracer_provider
        self._tracing_enabled = tracing_enabled
        self._resources = _Resources(self._binding.tracer_provider)

    def _wrap(self, handle: Any, trace_id: str) -> Any:
        return _ExportingHandle(handle, self._binding, trace_id) if self._binding.active else handle

    def span(self, **kwargs: Any) -> Any:
        return self._wrap(super().span(**kwargs), kwargs["trace_id"])

    def generation(self, **kwargs: Any) -> Any:
        return self._wrap(super().generation(**kwargs), kwargs["trace_id"])

    def tool(self, **kwargs: Any) -> Any:
        return self._wrap(super().tool(**kwargs), kwargs["trace_id"])

    def embedding(self, **kwargs: Any) -> Any:
        return self._wrap(super().embedding(**kwargs), kwargs["trace_id"])

    def retriever(self, **kwargs: Any) -> Any:
        return self._wrap(super().retriever(**kwargs), kwargs["trace_id"])

    def update_trace(self, **kwargs: Any) -> Any:
        result = super().update_trace(**kwargs)
        self._binding.note_trace_payload(kwargs["id"], input=kwargs.get("input"), output=kwargs.get("output"))
        return result

    def force_flush(self, timeout_ms: int = 5000) -> bool:
        """Flush the recorded side AND the bound provider."""
        # The inherited implementation returns True without flushing anything,
        # documented as "no outbound buffer" -- false for this subclass. A harness
        # on a BatchSpanProcessor would otherwise read an empty exporter and every
        # leak assertion would pass having observed nothing.
        recorded = bool(super().force_flush(timeout_ms))
        self._binding.drain_orphan_trace_payloads()
        for trace in self.traces.values():
            for observation in trace.observations:
                if not observation.ended:
                    self._binding.emit(trace.id, observation)
                    observation.ended = True
        return recorded and self._binding.flush_provider(timeout_ms)


def langfuse_observation_spans(spans: Any) -> list[Any]:
    """The openarmature Langfuse observations among an exporter's spans."""
    selected: list[Any] = []
    for span in spans:
        attributes: dict[str, Any] = dict(span.attributes or {})
        if any(name.startswith(_OBSERVATION_PREFIX) for name in attributes):
            selected.append(span)
    return selected


__all__ = [
    "OBSERVATION_INPUT_ATTR",
    "OBSERVATION_METADATA_ATTR",
    "OBSERVATION_OUTPUT_ATTR",
    "OBSERVATION_TYPE_ATTR",
    "TRACE_INPUT_ATTR",
    "TRACE_OUTPUT_ATTR",
    "ProviderFaithfulLangfuseClient",
    "langfuse_observation_spans",
]
