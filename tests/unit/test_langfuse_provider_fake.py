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
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.conformance.harness.langfuse_provider_fake import (
    _TRACE_STUB_KEYS,
    OBSERVATION_INPUT_ATTR,
    OBSERVATION_METADATA_ATTR,
    OBSERVATION_OUTPUT_ATTR,
    OBSERVATION_STATUS_MESSAGE_ATTR,
    OBSERVATION_TYPE_ATTR,
    TRACE_INPUT_ATTR,
    ProviderFaithfulLangfuseClient,
    _is_trace_stub,
    langfuse_observation_spans,
    span_is_payload_bearing,
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
    # Flattened per key, as a real v4 client renders a dict. The bare
    # OBSERVATION_METADATA_ATTR name carries nothing for dict metadata.
    assert attrs[f"{OBSERVATION_METADATA_ATTR}.error_message"] == "LEAKED EXCEPTION TEXT"
    assert OBSERVATION_METADATA_ATTR not in attrs


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


# -- payload-bearing classification (§6.4, narrowed by 0118) ----------------


def _one_span(build: Callable[[ProviderFaithfulLangfuseClient], None]) -> Any:
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    build(client)
    client.force_flush()
    return langfuse_observation_spans(exporter.get_finished_spans())[0]


def test_provider_payload_is_payload_bearing() -> None:
    span = _one_span(lambda c: c.generation(trace_id="t", name="gen").end(output="LEAKED"))
    assert span_is_payload_bearing(span) is True


def test_error_message_is_payload_bearing() -> None:
    # The third harvested channel (0117): a failed observation's error_message.
    span = _one_span(
        lambda c: c.tool(trace_id="t", name="run_tool").end(
            metadata={"error_type": "ValueError", "error_message": "LEAKED EXCEPTION TEXT"}
        )
    )
    assert span_is_payload_bearing(span) is True


def test_error_type_alone_is_payload_free() -> None:
    # 0118 narrowed the rule to the MESSAGE. error_type is a classification
    # token, so a failed observation carrying only classifications is
    # payload-free; treating it as bearing would make 158's assertions
    # unsatisfiable for any failed observation.
    span = _one_span(
        lambda c: c.tool(trace_id="t", name="run_tool").end(metadata={"error_type": "ValueError"})
    )
    assert span_is_payload_bearing(span) is False


@pytest.mark.parametrize(
    "stub",
    [
        {"entry_node": "ask"},
        {"entry_node": "ask", "correlation_id": "c-1"},
        {"final_node": "ask", "status": "completed"},
    ],
)
def test_the_minimal_trace_stub_is_payload_free(stub: dict[str, Any]) -> None:
    # The load-bearing one. openarmature writes a stub whenever the isolation
    # floor suppresses the state channel, and _resolve_trace_input / _output
    # never return None, so classifying any non-None trace payload as present
    # would be a constant true and 158 could never pass.
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.update_trace(id="t", input=stub)
    client.span(trace_id="t", name="node").end()
    span = langfuse_observation_spans(exporter.get_finished_spans())[0]
    assert span_is_payload_bearing(span) is False


def test_raw_state_on_the_trace_is_payload_bearing() -> None:
    # Non-vacuity for the stub cases: a real state payload through the same
    # channel IS bearing, so the stub result comes from its shape rather than
    # from the trace channel being ignored.
    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    client.trace(id="t")
    client.update_trace(id="t", input={"raw_user_text": "LEAKED STATE"})
    client.span(trace_id="t", name="node").end()
    span = langfuse_observation_spans(exporter.get_finished_spans())[0]
    assert span_is_payload_bearing(span) is True


def _stub_resolving_observer(*, blocked: bool) -> Any:
    from openarmature.observability.langfuse import InMemoryLangfuseClient
    from openarmature.observability.langfuse.client import ISOLATION_LEAKED
    from openarmature.observability.langfuse.observer import LangfuseObserver

    client = InMemoryLangfuseClient()
    if blocked:
        # The isolation floor, which closes the state channel whatever the knobs
        # say. Set dynamically because the observer reads it with getattr, so it
        # is not a declared attribute on the client.
        cast("Any", client)._isolation_status = ISOLATION_LEAKED  # noqa: SLF001
    # disable_state_payload drives the OTHER path to the same stub (lever 3), so
    # both spellings get walked below rather than only the isolation one.
    return LangfuseObserver(client=client, disable_state_payload=True)


@pytest.mark.parametrize("blocked", [True, False])
@pytest.mark.parametrize("correlation_id", [None, "c-1"])
def test_every_stub_the_observer_can_emit_is_a_known_stub_shape(
    blocked: bool, correlation_id: str | None
) -> None:
    # `_is_trace_stub` hardcodes the key sets, so it is correct only while those
    # are what openarmature actually emits. Drive the resolvers and read the keys
    # off the RESULT: a stub gaining, losing, or renaming a key fails here, which
    # a source-substring match could not detect. That matters because the
    # conformance cases that would catch the fallout are capability skips for this
    # adapter, so the classifier could be silently wrong with a green suite.
    observer = _stub_resolving_observer(blocked=blocked)
    started = SimpleNamespace(entry_node="ask", correlation_id=correlation_id, initial_state=None)
    completed = SimpleNamespace(final_node="ask", status="completed", final_state=None)

    for resolved in (
        observer._resolve_trace_input(cast("Any", started)),  # noqa: SLF001
        observer._resolve_trace_output(cast("Any", completed)),  # noqa: SLF001
    ):
        assert isinstance(resolved, dict)
        keys = set(cast("dict[str, Any]", resolved))
        assert keys in _TRACE_STUB_KEYS, (
            f"observer emitted stub keys {sorted(keys)}, which _TRACE_STUB_KEYS does not "
            f"recognise; every suppressed trace would reclassify as payload-bearing"
        )
        assert _is_trace_stub(json.dumps(resolved, sort_keys=True)) is True


_TOOL_ERROR = "CANARY-tool-error-message-4b7c"


async def _emit_failed_tool(observer: Any) -> None:
    from openarmature.graph.events import InvocationStartedEvent, ToolCallFailedEvent
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id

    invocation_id = "inv-status-message"
    token = _set_invocation_id(invocation_id)
    try:
        await observer(
            InvocationStartedEvent(
                initial_state={},
                invocation_id=invocation_id,
                correlation_id=None,
                entry_node="run_tool",
            )
        )
        await observer(
            ToolCallFailedEvent(
                invocation_id=invocation_id,
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-tool",
                tool_name="lookup",
                tool_call_id="call_1",
                arguments={},
                latency_ms=1.0,
                error_type="ValueError",
                error_message=_TOOL_ERROR,
            )
        )
    finally:
        _reset_invocation_id(token)


def test_the_classifier_agrees_with_a_real_langfuse_client() -> None:
    # The decisive one. Every fixture-158 case is `mode: credentials`, i.e. a REAL
    # client, and the fake is the only client that ever wrote metadata as a single
    # blob. Classifying on that blob made the error-message limb a constant False
    # on the path the fixture actually drives, while this file's other tests -- all
    # driving the fake -- passed. A double must not be the only witness for the
    # behaviour it exists to model, so drive the real SDK here.
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from tests.conformance.harness.langfuse_real_client import (
        CONFORMANCE_HOST,
        CONFORMANCE_PUBLIC_KEY,
        CONFORMANCE_SECRET_KEY,
        langfuse_sdk_without_egress,
    )

    with langfuse_sdk_without_egress():
        from langfuse import Langfuse

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        client = Langfuse(
            public_key=CONFORMANCE_PUBLIC_KEY,
            secret_key=CONFORMANCE_SECRET_KEY,
            host=CONFORMANCE_HOST,
            tracer_provider=provider,
        )
        # Exactly the shape observer.py renders for a failed observation.
        failed = client.start_observation(
            name="run_tool",
            as_type="span",
            metadata={"error_type": "ValueError", "error_message": "LEAKED EXCEPTION TEXT"},
        )
        failed.update(status_message="LEAKED EXCEPTION TEXT", level="ERROR")
        failed.end()
        # And a category-only failure, which must stay payload-free.
        classified = client.start_observation(
            name="classified", as_type="span", metadata={"error_type": "ValueError"}
        )
        classified.update(status_message="provider_unavailable", level="ERROR")
        classified.end()
        generation = client.start_observation(
            name="gen", as_type="generation", input="PROMPT", output="COMPLETION"
        )
        generation.end()
        provider.force_flush()

    by_name = {s.name: s for s in langfuse_observation_spans(exporter.get_finished_spans())}
    assert set(by_name) == {"run_tool", "classified", "gen"}
    assert span_is_payload_bearing(by_name["run_tool"]) is True
    assert span_is_payload_bearing(by_name["classified"]) is False
    assert span_is_payload_bearing(by_name["gen"]) is True


@pytest.mark.parametrize("disable_provider_payload", [False, True])
@pytest.mark.asyncio
async def test_harvested_status_message_never_travels_without_its_metadata_row(
    disable_provider_payload: bool,
) -> None:
    # `span_is_payload_bearing` classifies on `metadata.error_message` and treats
    # `status_message` as payload-free, because openarmature writes an error
    # CATEGORY there on most failure paths and classifying the attribute itself
    # would misread every category-only failure as a leak. That narrower rule is
    # safe only while the harvested message reaches status_message ONLY alongside
    # the metadata row the classifier does read. Drive a real failed observation
    # both ways and assert the coupling, so routing the message to status_message
    # alone fails here instead of going quiet.
    from openarmature.observability.langfuse.observer import LangfuseObserver

    provider, exporter = _provider()
    client = ProviderFaithfulLangfuseClient(provider)
    observer = LangfuseObserver(client=client, disable_provider_payload=disable_provider_payload)
    await _emit_failed_tool(observer)
    client.force_flush()

    spans = [
        s
        for s in langfuse_observation_spans(exporter.get_finished_spans())
        if _attrs(s).get(OBSERVATION_TYPE_ATTR) == "tool"
    ]
    assert spans, "the failed tool observation never reached the provider"
    attrs = _attrs(spans[0])
    status_message = attrs.get(OBSERVATION_STATUS_MESSAGE_ATTR)
    metadata_row = attrs.get(f"{OBSERVATION_METADATA_ATTR}.error_message")

    if status_message == _TOOL_ERROR:
        assert metadata_row == _TOOL_ERROR, (
            "the harvested message reached status_message without the "
            "metadata.error_message row span_is_payload_bearing reads, so the leak "
            "would classify payload-free"
        )
        assert span_is_payload_bearing(spans[0]) is True
    else:
        # Suppressed: status_message carries only the classification token.
        assert metadata_row is None
        assert span_is_payload_bearing(spans[0]) is False
