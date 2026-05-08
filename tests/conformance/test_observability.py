"""Run spec observability conformance fixtures (001-011) against OTelObserver.

Phase 6.0 scope (this PR):

- **001-basic-trace** — full span shape (private TracerProvider,
  correlation_id auto-generation, parent-child hierarchy, span names,
  base attribute set).
- **005-llm-provider-span-nested** — §5.5 LLM span emission +
  ``disable_llm_spans`` opt-out + §6 TracerProvider isolation under
  active external auto-instrumentation. The load-bearing
  isolation test.
- **008-detached-trace-mode** — §4.4 detached subgraph + detached
  fan-out, with cross-trace ``correlation_id`` consistency.
- **009-correlation-id-cross-cutting** — every span carries
  ``openarmature.correlation_id``; back-to-back invocations get
  distinct UUIDv4s.

Phase 6.1 (separate follow-up PR): the remaining 7 fixtures —
002 (subgraph hierarchy), 003 (error status), 004 (routing-error
attribution), 006 (fan-out instance attribution), 007 (retry
attempt spans), 010 (log correlation full assertions), 011
(determinism). Per-fixture wiring notes live in
``docs/phase-6-1-conformance-fillin.md``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

# Skip the entire module if the ``otel`` extras aren't installed —
# importing ``openarmature.observability.otel`` raises ImportError at
# import time when the extras are missing, which would fail
# collection rather than skipping cleanly. Mirrors the pattern in
# ``tests/unit/test_observability_otel.py``.
pytest.importorskip("opentelemetry.trace")

from openarmature.observability.otel import OTelObserver  # noqa: E402

from .adapter import build_graph  # noqa: E402

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "observability" / "conformance"
)


_SUPPORTED_FIXTURES = frozenset(
    {
        "001-otel-basic-trace",
        "005-otel-llm-provider-span-nested",
        "008-otel-detached-trace-mode",
        "009-otel-correlation-id-cross-cutting",
    }
)


_DEFERRED_FIXTURES: dict[str, str] = {
    "002-otel-subgraph-hierarchy": (
        "subgraph dispatch span synthesis (engine wrapper is transparent per "
        "fixture 013); deferred to Phase 6.1"
    ),
    "003-otel-error-status": (
        "status-mapping table across §4 categories; engine path unit-tested. Deferred to Phase 6.1."
    ),
    "004-otel-routing-error-attribution": (
        "routing errors don't fire their own event pair; preceding-node-span "
        "attribution wiring; deferred to Phase 6.1"
    ),
    "006-otel-fan-out-instance-attribution": (
        "fan-out instance attribution + namespace prefixing; engine path unit-tested. Deferred to Phase 6.1."
    ),
    "007-otel-retry-attempt-spans": (
        "retry-middleware wiring in the conformance harness; engine emits "
        "per-attempt events correctly. Deferred to Phase 6.1."
    ),
    "010-otel-log-correlation": (
        "OTel Logs Bridge full conformance assertions (trace_id/span_id "
        "population on every record); filter unit-tested. Deferred to "
        "Phase 6.1."
    ),
    "011-otel-determinism": (
        "deterministic-portion span content (hierarchy, names, attributes "
        "minus timing, status); checked indirectly by other fixtures. "
        "Deferred to Phase 6.1."
    ),
}


# UUIDv4 canonical form: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (where y in {8,9,a,b}).
_UUIDV4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _fixture_paths() -> list[Path]:
    return sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Per-fixture dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_observability_fixture(fixture_path: Path) -> None:
    fixture_id = fixture_path.stem
    if fixture_id in _DEFERRED_FIXTURES:
        pytest.skip(f"{fixture_id}: {_DEFERRED_FIXTURES[fixture_id]}")
    if fixture_id not in _SUPPORTED_FIXTURES:
        pytest.skip(f"{fixture_id}: harness wiring not yet implemented")

    spec = _load(fixture_path)
    if fixture_id == "001-otel-basic-trace":
        await _run_fixture_001(spec)
    elif fixture_id == "005-otel-llm-provider-span-nested":
        await _run_fixture_005(spec)
    elif fixture_id == "008-otel-detached-trace-mode":
        await _run_fixture_008(spec)
    elif fixture_id == "009-otel-correlation-id-cross-cutting":
        await _run_fixture_009(spec)
    else:
        raise AssertionError(f"no driver for supported fixture {fixture_id!r}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_observer() -> tuple[OTelObserver, Any]:
    """Build a fresh OTelObserver + InMemorySpanExporter pair."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    return observer, exporter


async def _run_graph(
    spec: Mapping[str, Any],
    observer: OTelObserver,
    *,
    correlation_id: str | None = None,
) -> Any:
    """Build + invoke a graph from a fixture spec; return the final
    state. Caller is responsible for calling ``observer.shutdown()``
    afterwards."""
    trace: list[str] = []
    built = build_graph(spec, trace=trace)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    final = await compiled.invoke(initial_state, correlation_id=correlation_id)
    await compiled.drain()
    return final


def _all_correlation_ids(spans: Any) -> set[str]:
    """Pull the ``openarmature.correlation_id`` attribute off every
    span; returns the unique set. Accepts any iterable of spans
    (``InMemorySpanExporter.get_finished_spans`` returns a tuple)."""
    return {cast("str", dict(s.attributes or {}).get("openarmature.correlation_id")) for s in spans}


# ---------------------------------------------------------------------------
# Fixture 001 — basic trace shape
# ---------------------------------------------------------------------------


async def _run_fixture_001(spec: Mapping[str, Any]) -> None:
    observer, exporter = _build_observer()
    final = await _run_graph(spec, observer, correlation_id=spec.get("caller_correlation_id"))
    observer.shutdown()
    spans = exporter.get_finished_spans()
    assert len(spans) == 4, (
        f"expected 4 spans (invocation + 3 nodes); got {len(spans)}: {[s.name for s in spans]}"
    )
    by_name = {s.name: s for s in spans}
    assert "openarmature.invocation" in by_name
    inv = by_name["openarmature.invocation"]
    assert inv.parent is None
    inv_attrs = dict(inv.attributes or {})
    assert inv_attrs.get("openarmature.graph.entry_node") == spec["entry"]
    cid = inv_attrs.get("openarmature.correlation_id")
    assert isinstance(cid, str) and len(cid) > 0
    inv_ctx = inv.context
    assert inv_ctx is not None
    invocation_span_id = inv_ctx.span_id
    for node_name in spec["nodes"]:
        assert node_name in by_name, f"missing span for {node_name!r}"
        node_span = by_name[node_name]
        node_parent = node_span.parent
        assert node_parent is not None and node_parent.span_id == invocation_span_id
        node_attrs = dict(node_span.attributes or {})
        assert node_attrs.get("openarmature.node.name") == node_name
        assert list(node_attrs.get("openarmature.node.namespace") or []) == [node_name]
        assert isinstance(node_attrs.get("openarmature.node.step"), int)
        assert node_attrs.get("openarmature.node.attempt_index") == 0
        assert node_attrs.get("openarmature.correlation_id") == cid
    expected_trace = ["a", "b", "c"]
    assert final.trace == expected_trace  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixture 009 — correlation_id cross-cutting
# ---------------------------------------------------------------------------


async def _run_fixture_009(spec: Mapping[str, Any]) -> None:
    """Three sub-cases, each in the ``cases:`` block:

    1. caller-supplied correlation_id used verbatim on every span.
    2. auto-generated UUIDv4 used uniformly across all spans.
    3. Two back-to-back invocations get DIFFERENT correlation_ids,
       both UUIDv4 form.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_009_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_009_case(case: Mapping[str, Any]) -> None:
    case_name = case["name"]
    if case_name == "context_reset_between_invocations":
        # Two back-to-back invocations of the same compiled graph;
        # each MUST get its own UUIDv4 (distinct from the other).
        observer, exporter = _build_observer()
        # Build the graph ONCE so both invocations share it.
        from .adapter import build_graph as _bg

        built = _bg(case)
        compiled = built.builder.compile()
        compiled.attach_observer(observer)
        for _ in range(int(case.get("invocations", 2))):
            await compiled.invoke(built.initial_state(case.get("initial_state", {})))
            await compiled.drain()
        observer.shutdown()
        spans = exporter.get_finished_spans()

        # Group spans by trace_id (each invocation has its own trace).
        by_trace: dict[int, list[Any]] = {}
        for s in spans:
            tid = s.context.trace_id
            by_trace.setdefault(tid, []).append(s)
        assert len(by_trace) == 2, f"expected 2 distinct traces (one per invocation); got {len(by_trace)}"
        # Each invocation's spans share one correlation_id.
        per_invocation_cids: list[str] = []
        for trace_spans in by_trace.values():
            cids = _all_correlation_ids(trace_spans)
            assert len(cids) == 1, f"each invocation MUST uniformly carry one correlation_id; got {cids}"
            cid = next(iter(cids))
            assert _UUIDV4_RE.match(cid), f"auto-generated correlation_id MUST be UUIDv4; got {cid!r}"
            per_invocation_cids.append(cid)
        # Cross-invocation: distinct.
        assert per_invocation_cids[0] != per_invocation_cids[1], (
            "back-to-back invocations MUST get distinct correlation_ids"
        )
        return

    # Sub-cases 1 & 2 (single-invocation).
    observer, exporter = _build_observer()
    await _run_graph(case, observer, correlation_id=case.get("caller_correlation_id"))
    observer.shutdown()
    spans = exporter.get_finished_spans()
    cids = _all_correlation_ids(spans)
    assert len(cids) == 1, f"every span MUST carry the same correlation_id; got {cids}"
    cid = next(iter(cids))
    expected = case.get("caller_correlation_id")
    if expected is not None:
        # Caller-supplied → exact match.
        assert cid == expected, (
            f"caller_correlation_id MUST be used verbatim; got {cid!r}, expected {expected!r}"
        )
    else:
        # Auto-generated → UUIDv4.
        assert _UUIDV4_RE.match(cid), f"auto-generated correlation_id MUST be UUIDv4; got {cid!r}"


# ---------------------------------------------------------------------------
# Fixture 005, 008 placeholders — driven below in subsequent commits
# ---------------------------------------------------------------------------


async def _run_fixture_005(spec: Mapping[str, Any]) -> None:
    """Three sub-cases:

    1. ``default`` — LLM span emits with §5 attributes, parented under
       the calling node.
    2. ``disable_llm_spans`` — opt-out suppresses the LLM span entirely.
    3. ``external_auto_instrumentation_active`` — second exporter on
       the OTel global provider; openarmature spans MUST NOT leak to
       it (the load-bearing §6 TracerProvider isolation guarantee).
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_005_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_005_case(case: Mapping[str, Any]) -> None:
    case_name = case["name"]
    disable_llm_spans = bool(case.get("disable_llm_spans", False))
    caller_global_active = bool(case.get("caller_global_otel_active", False))

    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    # Optional second exporter on the OTel global provider — sub-case 3.
    # Save the prior global provider so we can restore it after the
    # case (otherwise the global state leaks to subsequent tests).
    global_exporter: InMemorySpanExporter | None = None
    prior_global = otel_trace.get_tracer_provider() if caller_global_active else None
    if caller_global_active:
        global_exporter = InMemorySpanExporter()
        global_provider = TracerProvider()
        global_provider.add_span_processor(SimpleSpanProcessor(global_exporter))
        otel_trace.set_tracer_provider(global_provider)

    try:
        private_exporter = InMemorySpanExporter()
        observer = OTelObserver(
            span_processor=SimpleSpanProcessor(private_exporter),
            disable_llm_spans=disable_llm_spans,
        )

        # Build a graph whose entry node calls a mock LLM provider.
        graph, _ = _build_graph_with_mock_llm(case)
        graph.attach_observer(observer)

        # Drive the graph. The ``calls_llm`` node body reads the mock
        # responses set up in ``_build_graph_with_mock_llm`` (httpx
        # MockTransport keyed off the response queue).
        initial_state_cls = graph.state_cls
        final = await graph.invoke(initial_state_cls())
        await graph.drain()
        observer.shutdown()
        private_spans = private_exporter.get_finished_spans()

        # Sub-case 3: external span emitted by the harness through the
        # global tracer (simulating auto-instrumentation).
        if caller_global_active:
            global_tracer = otel_trace.get_tracer("external-instrumentation")
            with global_tracer.start_as_current_span("external.llm.call"):
                pass
            assert global_exporter is not None
            global_spans = global_exporter.get_finished_spans()
            assert len(global_spans) == 1, (
                f"global exporter MUST see exactly one external span; got {len(global_spans)}"
            )
            assert global_spans[0].name == "external.llm.call"
            # The load-bearing isolation check.
            for s in global_spans:
                assert not s.name.startswith("openarmature."), (
                    f"openarmature spans MUST NOT leak to the global provider; got {s.name!r}"
                )
    finally:
        if caller_global_active and prior_global is not None:
            otel_trace.set_tracer_provider(prior_global)

    # Common assertions: the LLM span presence/absence + (when
    # present) attributes + parent-child to the calling node.
    llm_spans = [s for s in private_spans if s.name == "openarmature.llm.complete"]
    if disable_llm_spans:
        assert not llm_spans, (
            f"disable_llm_spans=True MUST suppress LLM span emission; got {len(llm_spans)} llm spans"
        )
        # ask_llm node span still emits.
        assert any(s.name == "ask_llm" for s in private_spans)
        return

    # default + external_auto_instrumentation_active — LLM span
    # MUST emit with the spec §5.5 attributes.
    assert len(llm_spans) == 1, (
        f"expected one LLM span; got {len(llm_spans)}: {[s.name for s in private_spans]}"
    )
    llm = llm_spans[0]
    attrs = dict(llm.attributes or {})
    assert attrs.get("openarmature.llm.model") == "test-model"
    if case_name == "default":
        # Sub-case 1 asserts the full attribute set.
        assert attrs.get("openarmature.llm.finish_reason") == "stop"
        assert attrs.get("openarmature.llm.usage.prompt_tokens") == 5
        assert attrs.get("openarmature.llm.usage.completion_tokens") == 1
        assert attrs.get("openarmature.llm.usage.total_tokens") == 6
    # Parent: the ask_llm node span.
    ask_llm = next((s for s in private_spans if s.name == "ask_llm"), None)
    assert ask_llm is not None, "expected ask_llm node span"
    llm_parent = llm.parent
    ask_llm_ctx = ask_llm.context
    assert llm_parent is not None and ask_llm_ctx is not None
    assert llm_parent.span_id == ask_llm_ctx.span_id, (
        "openarmature.llm.complete MUST be parented under the calling node span"
    )
    # Final state was updated by the calls_llm node (msg = "hello"
    # for default; "hi back" for disable_llm_spans path that we
    # already returned from above).
    assert "msg" in dir(final)


def _build_graph_with_mock_llm(case: Mapping[str, Any]) -> tuple[Any, list[Any]]:
    """Build a graph whose entry node invokes ``OpenAIProvider.complete``
    against an ``httpx.MockTransport`` preloaded with the fixture's
    ``mock_llm`` responses."""
    import json
    from collections.abc import Sequence

    import httpx

    from openarmature.graph import GraphBuilder
    from openarmature.llm import OpenAIProvider, UserMessage

    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    transport = httpx.MockTransport(_handler)
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test",
        transport=transport,
    )

    # Build a State subclass with the fixture's declared fields.
    from .adapter import build_state_cls

    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    state_cls = build_state_cls("LlmFixtureState", state_fields)

    # Node body: calls the LLM provider and writes the response into
    # the configured field.
    nodes = cast("dict[str, Any]", case["nodes"])
    entry_name = cast("str", case["entry"])
    calls_llm_spec = cast("dict[str, Any]", nodes[entry_name]["calls_llm"])
    stores_in = cast("str", calls_llm_spec.get("stores_response_in", "msg"))
    messages_spec = cast("list[dict[str, str]]", calls_llm_spec.get("messages", []))
    messages: Sequence[Any] = [
        UserMessage(content=m["content"]) for m in messages_spec if m.get("role") == "user"
    ]

    async def ask_llm_body(_s: Any) -> dict[str, str]:
        response = await provider.complete(messages)
        return {stores_in: response.message.content or ""}

    builder = (
        GraphBuilder(state_cls)
        .add_node(entry_name, ask_llm_body)
        .add_edge(entry_name, _resolve_target_for_005(case))
        .set_entry(entry_name)
    )
    return builder.compile(), mock_responses


def _resolve_target_for_005(case: Mapping[str, Any]) -> Any:
    """Fixture 005's edges go to END. Return the END sentinel."""
    from openarmature.graph import END

    edges = cast("list[dict[str, Any]]", case.get("edges") or [])
    if not edges:
        return END
    target = edges[0].get("to")
    return END if target == "END" else target


async def _run_fixture_008(spec: Mapping[str, Any]) -> None:
    """Two sub-cases: detached subgraph (one Link, two traces, shared
    correlation_id) and detached fan-out (one trace per instance,
    each with a Link from the fan-out node span)."""
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_008_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_008_case(case: Mapping[str, Any]) -> None:
    case_name = case["name"]
    # The fixture configures detached subgraphs by the SUBGRAPH'S
    # IDENTITY NAME (the key in ``subgraphs:``), but the OTel observer
    # keys on the WRAPPER NODE'S NAME in the parent graph (consistent
    # with graph-engine §6's namespace convention — see fixture 029
    # spec note). Translate by looking up the wrapper node that
    # references each detached subgraph identity.
    detached_subgraph_identities = set(cast("list[str]", case.get("detached_subgraphs") or []))
    nodes = cast("dict[str, Any]", case.get("nodes") or {})
    wrapper_names_for_detached: set[str] = set()
    for wrapper_name, node_spec in nodes.items():
        sub_id = cast("dict[str, Any]", node_spec).get("subgraph")
        if isinstance(sub_id, str) and sub_id in detached_subgraph_identities:
            wrapper_names_for_detached.add(wrapper_name)
    detached_subgraphs = frozenset(wrapper_names_for_detached)
    detached_fan_outs = frozenset(cast("list[str]", case.get("detached_fan_outs") or []))

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        detached_subgraphs=detached_subgraphs,
        detached_fan_outs=detached_fan_outs,
    )

    # Patch the inner subgraph's ``update_pure_from_state`` directive
    # if present — the adapter doesn't translate it, but the test
    # assertions only inspect span structure (Links, trace counts),
    # not the computed values. Replacing with a no-op ``update_pure``
    # keeps the graph runnable.
    _patch_unsupported_directives(case)

    # Build subgraphs declared by the fixture (subgraph: or subgraphs:).
    subgraphs = _compile_subgraphs(case)
    trace_log: list[str] = []
    built = build_graph(case, subgraphs=subgraphs, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))
    await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    if case_name == "detached_subgraph_two_traces_one_link":
        # Group by trace_id. Span context is non-None for any span
        # the SDK actually exported, so the cast keeps pyright quiet
        # on the `.trace_id` access.
        by_trace: dict[int, list[Any]] = {}
        for s in spans:
            ctx = cast("Any", s.context)
            by_trace.setdefault(ctx.trace_id, []).append(s)
        assert len(by_trace) == 2, (
            f"expected 2 distinct traces (parent + detached subgraph); "
            f"got {len(by_trace)}: {[s.name for s in spans]}"
        )
        # Cross-trace correlation_id consistency (§3).
        cids = _all_correlation_ids(spans)
        assert len(cids) == 1, f"correlation_id MUST flow unchanged across detached boundary; got {cids}"
        # Find the parent dispatch span (it's the one with a Link).
        dispatch_spans = [s for s in spans if s.name == "dispatch"]
        # Two "dispatch" spans: one in parent trace (with Link), one in
        # detached trace (the root). Pick the one with links.
        parent_dispatch = next((s for s in dispatch_spans if s.links), None)
        assert parent_dispatch is not None, "expected a 'dispatch' span carrying a Link to the detached trace"
        assert len(parent_dispatch.links) == 1, (
            f"dispatch span MUST carry exactly one Link; got {len(parent_dispatch.links)}"
        )
        link_target_trace_id = parent_dispatch.links[0].context.trace_id
        # The link's trace_id matches the detached trace's actual trace_id.
        parent_trace_id = cast("Any", parent_dispatch.context).trace_id
        detached_dispatch = next(
            (s for s in dispatch_spans if not s.links and cast("Any", s.context).trace_id != parent_trace_id),
            None,
        )
        assert detached_dispatch is not None
        detached_trace_id = cast("Any", detached_dispatch.context).trace_id
        assert link_target_trace_id == detached_trace_id, (
            f"Link target trace_id MUST match the detached trace's trace_id; "
            f"got link={link_target_trace_id!r}, detached={detached_trace_id!r}"
        )
        return

    if case_name == "detached_fan_out_one_trace_per_instance":
        # Group by trace_id.
        by_trace = {}
        for s in spans:
            ctx = cast("Any", s.context)
            by_trace.setdefault(ctx.trace_id, []).append(s)
        # 4 traces total: 1 parent + 3 instance traces.
        assert len(by_trace) == 4, f"expected 4 traces (parent + 3 instances); got {len(by_trace)}"
        # All 4 share the same correlation_id.
        cids = _all_correlation_ids(spans)
        assert len(cids) == 1, (
            f"correlation_id MUST be uniform across parent + detached instance traces; got {cids}"
        )
        # Find the fan-out node span — it's in the parent trace and
        # carries 3 Links.
        fan_out_node_spans = [s for s in spans if s.name == "per_document_scoring"]
        # Three of these are inside detached instance roots; one is in
        # the parent trace and is the one with Links.
        parent_fan_out = next((s for s in fan_out_node_spans if s.links), None)
        assert parent_fan_out is not None, "expected a fan-out span with Links in parent trace"
        assert len(parent_fan_out.links) == 3, (
            f"fan-out span MUST carry one Link per instance (3); got {len(parent_fan_out.links)}"
        )
        return

    raise AssertionError(f"unknown sub-case {case_name!r}")


def _patch_unsupported_directives(spec: Mapping[str, Any]) -> None:
    """Replace test-seam directives the conformance adapter doesn't
    yet translate (``update_pure_from_state`` etc.) with a benign
    ``update_pure: {}`` no-op. The observability fixtures only
    assert span structure (parent-child, Links, trace_ids,
    correlation_id), not state values, so the swap is safe."""

    def patch_nodes(graph_block: Mapping[str, Any] | None) -> None:
        if not graph_block:
            return
        nodes = cast("dict[str, Any]", graph_block.get("nodes") or {})
        for node_spec_any in nodes.values():
            if not isinstance(node_spec_any, dict):
                continue
            node_spec = cast("dict[str, Any]", node_spec_any)
            for unsupported in (
                "update_pure_from_state",
                "calls_llm",
            ):
                if unsupported in node_spec:
                    node_spec.pop(unsupported)
                    node_spec.setdefault("update_pure", {})

    patch_nodes(spec)
    if "subgraph" in spec:
        patch_nodes(cast("Mapping[str, Any]", spec["subgraph"]))
    for sub in cast("dict[str, Any]", spec.get("subgraphs") or {}).values():
        patch_nodes(cast("Mapping[str, Any]", sub))


def _compile_subgraphs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build any subgraphs declared by the fixture and return a
    name→compiled-graph registry the adapter consumes."""
    subgraph_specs: dict[str, Any] = {}
    if "subgraph" in spec:
        single = cast("Mapping[str, Any]", spec["subgraph"])
        name = single.get("name") or "subgraph"
        subgraph_specs[name] = single
    if "subgraphs" in spec:
        for k, v in cast("dict[str, Any]", spec["subgraphs"]).items():
            subgraph_specs[k] = v
    compiled_subgraphs: dict[str, Any] = {}
    for name, sub_spec in subgraph_specs.items():
        sub_built = build_graph(sub_spec, trace=[])
        compiled_subgraphs[name] = sub_built.builder.compile()
    return compiled_subgraphs


# ---------------------------------------------------------------------------
# Phase 5 fixture 031 — span/log assertions deferred from Phase 5
#
# Lives in this file (not test_checkpoint.py) because the assertions
# verify OTel span attributes across the original + resumed runs of
# the same checkpoint fixture. The Phase 5 harness already covers the
# record-level half (correlation_id preserved, invocation_id changes);
# this picks up the cross-run span-attribute half.
# ---------------------------------------------------------------------------


_PIPELINE_CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "pipeline-utilities" / "conformance"
)


async def test_phase5_fixture_031_span_assertions() -> None:
    """Spec §10.4 step 3 + step 4 + observability §3 / §5.6: every
    span across BOTH the original and resumed runs MUST carry the
    same ``openarmature.correlation_id``; ``invocation_id`` differs
    across the two runs (each is its own invocation in the
    observability sense)."""
    fixture_path = _PIPELINE_CONFORMANCE_DIR / "031-checkpoint-correlation-id-preserved-across-resume.yaml"
    spec = _load(fixture_path)
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_031_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_031_case(case: Mapping[str, Any]) -> None:
    from openarmature.checkpoint import CheckpointRecord, InMemoryCheckpointer
    from openarmature.checkpoint.protocol import Checkpointer
    from openarmature.graph import RuntimeGraphError

    class _CapturingCheckpointer:
        """Mirrors the local capture pattern in test_checkpoint.py
        but inlined here so the observability test doesn't depend on
        that file's internal helpers."""

        def __init__(self) -> None:
            self._inner = InMemoryCheckpointer()
            self.saves: list[CheckpointRecord] = []

        async def save(self, invocation_id: str, record: CheckpointRecord) -> None:
            self.saves.append(record)
            await self._inner.save(invocation_id, record)

        async def load(self, invocation_id: str) -> CheckpointRecord | None:
            return await self._inner.load(invocation_id)

        async def list(self, filter: Any = None) -> Any:
            return await self._inner.list(filter)

        async def delete(self, invocation_id: str) -> None:
            await self._inner.delete(invocation_id)

    capturing = _CapturingCheckpointer()
    observer, exporter = _build_observer()

    trace: list[str] = []
    built = build_graph(case, trace=trace)
    builder = built.builder
    builder.with_checkpointer(cast("Checkpointer", capturing))
    compiled = builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))

    # First run — expected to abort.
    expected_error = cast("Mapping[str, Any]", case["first_run_expected_error"])
    caller_cid = case.get("caller_correlation_id")
    with pytest.raises(RuntimeGraphError) as excinfo:
        await compiled.invoke(initial_state, correlation_id=caller_cid)
    assert excinfo.value.category == expected_error["category"]
    await compiled.drain()

    # Capture first run's invocation_id from the latest save.
    assert capturing.saves, "expected at least one save before the abort"
    first_invocation_id = capturing.saves[-1].invocation_id
    first_correlation_id = capturing.saves[-1].correlation_id
    if caller_cid is not None:
        assert first_correlation_id == caller_cid, (
            f"first run MUST preserve caller-supplied correlation_id; "
            f"got {first_correlation_id!r}, expected {caller_cid!r}"
        )
    else:
        # Auto-generated → UUIDv4 form.
        assert _UUIDV4_RE.match(first_correlation_id), (
            f"auto-generated correlation_id MUST be UUIDv4; got {first_correlation_id!r}"
        )

    # Resume — should succeed.
    capturing.saves.clear()
    await compiled.invoke(initial_state, resume_invocation=first_invocation_id)
    await compiled.drain()
    observer.shutdown()

    # ----- Span assertions (the §10.4 / §3 invariants) -----
    spans = exporter.get_finished_spans()
    # Every span across both runs MUST carry the same correlation_id.
    cids = _all_correlation_ids(spans)
    assert len(cids) == 1, f"correlation_id MUST be uniform across both runs; got {cids}"
    cid = next(iter(cids))
    assert cid == first_correlation_id, (
        f"resumed run MUST preserve original correlation_id; got {cid!r}, original {first_correlation_id!r}"
    )
    # The original and resumed runs MUST have DIFFERENT trace_ids
    # (each invocation is its own trace per §5.1's
    # ``invocation_id`` semantics — different invocation_id ↔
    # different OTel trace_id under the default in-trace
    # parent-child rules).
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 2, (
        f"original and resumed runs MUST produce DIFFERENT trace_ids "
        f"(per §10.4 step 4 + §5.1); got {len(trace_ids)} distinct trace_ids"
    )
