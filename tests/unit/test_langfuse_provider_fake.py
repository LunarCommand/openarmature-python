"""The provider-faithful Langfuse fake, driven against a real TracerProvider."""

# Spec basis: conformance-adapter §6.4 (proposals 0115 / 0116 / 0117 / 0118).
#
# This file exists because the previous version of the fake shipped importing
# nowhere: every claim it made about the Langfuse v4 SDK went unexecuted, and an
# adversarial review found nine defects in it that no test run could have caught.
# So each of its obligations is driven here against a real SDK TracerProvider and
# in-memory exporter, and the observations it records are cross-checked against
# the spans it exports, so the two sides cannot silently disagree.

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.conformance.harness.langfuse_provider_fake import (
    OBSERVATION_INPUT_ATTR,
    OBSERVATION_METADATA_ATTR,
    OBSERVATION_OUTPUT_ATTR,
    OBSERVATION_TYPE_ATTR,
    TRACE_INPUT_ATTR,
    ProviderFaithfulLangfuseClient,
    langfuse_observation_spans,
)


def _provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _attrs(span: Any) -> dict[str, Any]:
    return dict(span.attributes or {})


# -- emission ---------------------------------------------------------------


def test_an_observation_reaches_the_bound_provider() -> None:
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.generation(trace_id="t", name="openarmature.llm.complete").end()

    spans = langfuse_observation_spans(exporter.get_finished_spans())
    assert len(spans) == 1
    assert spans[0].name == "openarmature.llm.complete"
    assert _attrs(spans[0])[OBSERVATION_TYPE_ATTR] == "generation"


def test_payload_written_at_end_is_on_the_exported_span() -> None:
    # The load-bearing lifecycle property. A fake exporting at CREATION would
    # stamp the span before this output exists and report it payload-free, so a
    # real leak arriving via `end()` would read as no leak at all.
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    handle = client.generation(trace_id="t", name="gen")
    handle.end(output="LEAKED MODEL OUTPUT")

    span = langfuse_observation_spans(exporter.get_finished_spans())[0]
    assert _attrs(span)[OBSERVATION_OUTPUT_ATTR] == "LEAKED MODEL OUTPUT"


def test_payload_written_by_update_is_on_the_exported_span() -> None:
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    handle = client.span(trace_id="t", name="node")
    handle.update(input={"role": "user"}, output="done")
    handle.end()

    attrs = _attrs(langfuse_observation_spans(exporter.get_finished_spans())[0])
    assert json.loads(attrs[OBSERVATION_INPUT_ATTR]) == {"role": "user"}
    assert attrs[OBSERVATION_OUTPUT_ATTR] == "done"


def test_error_message_metadata_reaches_the_span() -> None:
    # The third harvested channel (0117): a failed observation's error_message
    # rides observation.metadata, so it has to survive onto the exported span or
    # a leak through it would be invisible to the provider-side assertions.
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.tool(trace_id="t", name="run_tool").end(
        metadata={"error_type": "ValueError", "error_message": "LEAKED EXCEPTION TEXT"}
    )

    attrs = _attrs(langfuse_observation_spans(exporter.get_finished_spans())[0])
    assert "LEAKED EXCEPTION TEXT" in attrs[OBSERVATION_METADATA_ATTR]


def test_trace_payload_reaches_the_provider() -> None:
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.update_trace(id="t", input={"raw": "STATE PAYLOAD"})
    client.span(trace_id="t", name="node").end()

    joined = " ".join(json.dumps(_attrs(s)) for s in exporter.get_finished_spans())
    assert "STATE PAYLOAD" in joined
    assert TRACE_INPUT_ATTR in joined


def test_trace_payload_reaches_the_provider_even_with_no_observations() -> None:
    # A state-channel leak on an invocation that produced no observations would
    # otherwise never be exported and the assertion would see a clean provider.
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.update_trace(id="t", output={"raw": "STATE PAYLOAD"})
    client.force_flush()

    joined = " ".join(json.dumps(_attrs(s)) for s in exporter.get_finished_spans())
    assert "STATE PAYLOAD" in joined


def test_an_unended_observation_still_exports_on_flush() -> None:
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.generation(trace_id="t", name="never-ended", output="LEAKED")
    assert langfuse_observation_spans(exporter.get_finished_spans()) == []

    client.force_flush()
    spans = langfuse_observation_spans(exporter.get_finished_spans())
    assert [s.name for s in spans] == ["never-ended"]


def test_force_flush_flushes_the_provider_not_only_the_recorder() -> None:
    # Under a BatchSpanProcessor the exporter stays empty until the PROVIDER is
    # flushed. The inherited force_flush returns True having flushed nothing, so
    # a harness reading the exporter after it would find no spans and every leak
    # assertion would pass having observed nothing.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.generation(trace_id="t", name="gen", output="payload").end()

    assert exporter.get_finished_spans() == ()
    assert client.force_flush() is True
    assert len(langfuse_observation_spans(exporter.get_finished_spans())) == 1


# -- binding ----------------------------------------------------------------


def test_no_provider_supplied_binds_the_global_one() -> None:
    # §6.4: "a plain priming construction with no provider supplied binds the
    # global provider". Treating that as tracing-off instead would make the primed
    # client both unable to leak and classified isolated, so 158's raise arm could
    # never fire.
    provider, exporter = _provider()
    previous = otel_trace._TRACER_PROVIDER  # type: ignore[attr-defined] # pyright: ignore[reportPrivateUsage]
    otel_trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined] # pyright: ignore[reportPrivateUsage]
    try:
        client = ProviderFaithfulLangfuseClient()
        assert client.tracer_provider is provider
        assert client._resources.tracer_provider is provider
        client.trace(id="t")
        client.span(trace_id="t", name="node").end()
    finally:
        otel_trace._TRACER_PROVIDER = previous  # type: ignore[attr-defined] # pyright: ignore[reportPrivateUsage]
    assert len(langfuse_observation_spans(exporter.get_finished_spans())) == 1


def test_tracing_disabled_is_a_separate_state_from_no_provider() -> None:
    client = ProviderFaithfulLangfuseClient(tracing_enabled=False)
    assert client.tracer_provider is None
    assert client._resources.tracer_provider is None
    assert client._tracing_enabled is False
    client.trace(id="t")
    client.span(trace_id="t", name="node").end()  # records, exports nowhere
    assert client.traces["t"].observations[0].name == "node"


def test_the_bound_provider_is_readable_where_openarmature_looks() -> None:
    # adapter._classify_isolation walks client._resources.tracer_provider. If the
    # fake does not expose it there, every fixture client classifies undetectable
    # and the arms 157 / 158 exist to pin are unreachable.
    provider, _ = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    resources = getattr(client, "_resources", None)
    assert getattr(resources, "tracer_provider", None) is provider


# -- per-credential singleton ----------------------------------------------


# -- span selection ---------------------------------------------------------


def test_selection_ignores_spans_that_are_not_langfuse_observations() -> None:
    # The leak assertions filter an exporter that also carries openarmature's own
    # OTel spans, so the selector must not sweep those up.
    provider, exporter = _provider()
    provider.get_tracer("openarmature").start_span("openarmature.node").end()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.span(trace_id="t", name="node").end()

    finished = exporter.get_finished_spans()
    assert len(finished) == 2
    assert [s.name for s in langfuse_observation_spans(finished)] == ["node"]


@pytest.mark.parametrize("tracing_enabled", [True, False])
def test_recording_is_unaffected_by_the_provider_binding(tracing_enabled: bool) -> None:
    # The recorded side is what the content assertions read, and it must behave
    # identically whichever provider the client is bound to.
    provider, _ = _provider()
    client = ProviderFaithfulLangfuseClient(
        provider if tracing_enabled else None, tracing_enabled=tracing_enabled
    )
    client.trace(id="t")
    client.generation(trace_id="t", name="gen", output="hello").end()
    observation = client.traces["t"].observations[0]
    assert observation.name == "gen"
    assert observation.output == "hello"
