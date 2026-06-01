"""Integration test for OTel span export against a live HyperDX endpoint.

Gated by the presence of ``HYPERDX_API_KEY`` + ``HYPERDX_OTLP_ENDPOINT``
env vars. Skipped in CI and local runs that don't have credentials in
scope; runs end-to-end against HyperDX Cloud (or any other OTLP-HTTP
collector) when invoked from a shell with both env vars sourced.

``HYPERDX_OTLP_ENDPOINT`` MUST be the full traces-collector URL
including the ``/v1/traces`` path suffix, e.g.::

    HYPERDX_OTLP_ENDPOINT=https://in-otel.hyperdx.io/v1/traces

``OTLPSpanExporter`` uses the ``endpoint`` kwarg verbatim and does
not append the path itself (that auto-append only happens for the
``OTEL_EXPORTER_OTLP_ENDPOINT`` host-only convention this test does
not use). A host-only URL POSTs to ``/`` and HyperDX 404s.

The test verifies the production export path the documentation
recommends (``BatchSpanProcessor`` + ``OTLPSpanExporter``) drains
cleanly from the local pipeline. The assertion is local-side: the
BatchSpanProcessor's ``force_flush`` succeeded within the deadline.
HyperDX-side acceptance (auth, payload accepted, span visible in the
UI) is verified by checking the HyperDX UI for a span named ``ping``
under service ``openarmature-hyperdx-integration``; the OTel SDK
swallows exporter errors silently, so a local-side success does not
prove the collector received the spans.
"""

from __future__ import annotations

import os

import pytest

# Skip the entire module when credentials / endpoint aren't sourced.
# Avoids an ImportError cascade from the OTLP exporter if its env-var
# fallback also can't find a target.
pytestmark = pytest.mark.skipif(
    not (os.environ.get("HYPERDX_API_KEY") and os.environ.get("HYPERDX_OTLP_ENDPOINT")),
    reason=(
        "Requires HYPERDX_API_KEY + HYPERDX_OTLP_ENDPOINT (live HyperDX endpoint); "
        "endpoint MUST include the /v1/traces path suffix"
    ),
)


@pytest.mark.integration
async def test_otel_observer_pipeline_drains_with_hyperdx_exporter() -> None:
    """End-to-end: invoke a tiny graph under an OTelObserver wired to
    the OTLPSpanExporter pointing at the configured HyperDX endpoint,
    flush, and assert the local pipeline drained within the deadline.
    """
    # Imports inside the function so the heavy OTLP-protobuf
    # dependencies don't load when the module is collected and skipped
    # under the default ``-m "not integration"`` pytest filter.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from openarmature.graph import END, GraphBuilder, State
    from openarmature.observability.otel import OTelObserver

    # HyperDX accepts the API key as a bare ``authorization`` header
    # value (no ``Bearer`` prefix). Other OTLP collectors that expect
    # ``Bearer <token>`` will need the caller to format the header
    # themselves; this is the documented HyperDX shape.
    exporter = OTLPSpanExporter(
        endpoint=os.environ["HYPERDX_OTLP_ENDPOINT"],
        headers={"authorization": os.environ["HYPERDX_API_KEY"]},
    )

    observer = OTelObserver(
        span_processor=BatchSpanProcessor(exporter),
        resource=Resource.create({"service.name": "openarmature-hyperdx-integration"}),
    )

    class _PingState(State):
        ping: bool = False

    async def _node(_s: _PingState) -> dict[str, bool]:
        return {"ping": True}

    graph = GraphBuilder(_PingState).add_node("ping", _node).add_edge("ping", END).set_entry("ping").compile()
    graph.attach_observer(observer)

    try:
        final = await graph.invoke(_PingState())
        assert final.ping is True

        # Local-side assertion. ``BatchSpanProcessor.force_flush``
        # returns True when every registered processor finishes
        # flushing within the timeout, False when any one times out.
        # The OTel SDK swallows exporter-side errors (401s, schema
        # rejections) silently, so a True here proves the pipeline
        # drained but not that HyperDX accepted the payload; that
        # confirmation is in the HyperDX UI.
        flushed = observer.force_flush(timeout_ms=15_000)
        assert flushed, "BatchSpanProcessor did not finish flushing within 15s"
    finally:
        # Releases the BatchSpanProcessor's background export thread;
        # ``OTelObserver.shutdown`` is idempotent and calls
        # ``_provider.shutdown`` under the hood.
        observer.shutdown()
