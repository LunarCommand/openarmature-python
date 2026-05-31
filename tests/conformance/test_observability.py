"""Run spec observability conformance fixtures (001-011) against OTelObserver.

Driven fixtures:

- **001-basic-trace** (Phase 6.0) — full span shape.
- **002-subgraph-hierarchy** (PR-C) — synthetic dispatch span +
  inner-node parenting per §4.5.
- **003-error-status** (PR-C) — §4.2 ERROR status mapping for the
  ``node_exception`` case.
- **005-llm-provider-span-nested** (Phase 6.0) — §5.5 LLM span +
  ``disable_llm_spans`` opt-out + §6 TracerProvider isolation.
- **007-retry-attempt-spans** (PR-C) — sibling attempt spans with
  per-attempt ``attempt_index`` under retry middleware.
- **008-detached-trace-mode** (Phase 6.0) — §4.4 detached subgraph
  + detached fan-out + cross-trace ``correlation_id``.
- **009-correlation-id-cross-cutting** (Phase 6.0) — every span
  carries ``openarmature.correlation_id``; back-to-back
  invocations get distinct UUIDv4s.
- **010-log-correlation** (PR-C.3) — log records emitted from
  inside node bodies pick up the active node span's
  ``trace_id``/``span_id`` via the engine-side
  ``prepare_sync`` → OTel context attach pipeline; both nested
  and detached-trace cases.
- **011-determinism** (PR-C) — deterministic span content
  (hierarchy, names, status, attributes minus the canonical
  non-deterministic-by-design list) is identical across runs.

Per-fixture wiring notes live in
``docs/phase-6-1-conformance-fillin.md``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

# Skip the entire module if the ``otel`` extras aren't installed —
# importing ``openarmature.observability.otel`` raises ImportError at
# import time when the extras are missing, which would fail
# collection rather than skipping cleanly. Mirrors the pattern in
# ``tests/unit/test_observability_otel.py``.
pytest.importorskip("opentelemetry.sdk.trace")

from openarmature.observability.otel import OTelObserver  # noqa: E402

from .adapter import build_graph  # noqa: E402


# OTel SDK 1.x makes ``set_tracer_provider`` one-shot: once a non-default
# provider is set, subsequent calls are no-ops (the SDK logs a warning
# and returns). The set is guarded by a ``Once`` primitive at
# ``opentelemetry.trace._TRACER_PROVIDER_SET_ONCE``, not just by the
# value of ``_TRACER_PROVIDER``. Restoring via the public API silently
# fails after a prior set, leaking the test's global provider into
# subsequent tests that also touch the OTel global. This helper resets
# BOTH the value and the Once via the SDK's private API so a sibling
# test running after this one starts from a clean global state.
def _reset_otel_global_tracer_provider(restore_to: object) -> None:
    from opentelemetry import trace as otel_trace

    once = otel_trace._TRACER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
    with once._lock:  # pyright: ignore[reportPrivateUsage]
        if isinstance(restore_to, otel_trace.ProxyTracerProvider):
            otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
            once._done = False  # pyright: ignore[reportPrivateUsage]
        else:
            otel_trace._TRACER_PROVIDER = restore_to  # type: ignore[attr-defined]
            once._done = True  # pyright: ignore[reportPrivateUsage]


CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "observability" / "conformance"
)


_SUPPORTED_FIXTURES = frozenset(
    {
        # v0.36.0 — proposal 0044 (parallel-branches OTel dispatch
        # span). Asserts the per-branch dispatch span synthesis +
        # §5.7 attribute surface end-to-end against a two-branch
        # parallel-branches graph with calls_llm in each branch.
        "038-otel-parallel-branches-dispatch-span",
        "001-otel-basic-trace",
        "002-otel-subgraph-hierarchy",
        "003-otel-error-status",
        "004-otel-routing-error-attribution",
        "005-otel-llm-provider-span-nested",
        "006-otel-fan-out-instance-attribution",
        "007-otel-retry-attempt-spans",
        "008-otel-detached-trace-mode",
        "009-otel-correlation-id-cross-cutting",
        "010-otel-log-correlation",
        "011-otel-determinism",
        # v0.17.0 — proposal 0024 (friction-roundup #1, #2, #6).
        "012-otel-llm-payload-default-off",
        "013-otel-llm-payload-enabled",
        "014-otel-llm-payload-truncation",
        "015-otel-llm-payload-image-redaction",
        "016-otel-llm-request-params",
        "017-otel-llm-request-params-partial",
        "018-otel-llm-request-extras",
        "019-otel-llm-genai-semconv",
        "020-otel-llm-genai-system-override",
        "021-otel-llm-disable-genai-semconv",
        # v0.24.0 — proposal 0032 (three new declared RuntimeConfig
        # fields surfaced as gen_ai.request.* attributes).
        "025-otel-llm-request-params-extended",
        # v0.10.0 — proposal 0034 (caller-supplied invocation metadata
        # cross-cutting on every span). 026 verifies the
        # ``openarmature.user.*`` attribute family lands on the
        # invocation span, every node span, and the LLM provider span.
        "026-otel-caller-supplied-metadata",
        # 028 — proposal 0034 API-boundary rejection: caller-supplied
        # metadata keys under reserved namespaces (openarmature.*,
        # gen_ai.*) MUST raise at the ``invoke()`` boundary before
        # any work begins. Two cases (one per reserved prefix).
        "028-caller-metadata-namespace-rejection",
    }
)


_DEFERRED_FIXTURES: dict[str, str] = {
    # Proposal 0045 (nested-lineage augmentation, v0.37.0) — engine
    # + observer work lands in PR 11.
    "039-nested-lineage-augmentation": ("Proposal 0045 not yet implemented (PR 11)"),
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
    elif fixture_id == "002-otel-subgraph-hierarchy":
        await _run_fixture_002(spec)
    elif fixture_id == "003-otel-error-status":
        await _run_fixture_003(spec)
    elif fixture_id == "004-otel-routing-error-attribution":
        await _run_fixture_004(spec)
    elif fixture_id == "005-otel-llm-provider-span-nested":
        await _run_fixture_005(spec)
    elif fixture_id == "006-otel-fan-out-instance-attribution":
        await _run_fixture_006(spec)
    elif fixture_id == "007-otel-retry-attempt-spans":
        await _run_fixture_007(spec)
    elif fixture_id == "008-otel-detached-trace-mode":
        await _run_fixture_008(spec)
    elif fixture_id == "009-otel-correlation-id-cross-cutting":
        await _run_fixture_009(spec)
    elif fixture_id == "010-otel-log-correlation":
        await _run_fixture_010(spec)
    elif fixture_id == "011-otel-determinism":
        await _run_fixture_011(spec)
    elif fixture_id == "028-caller-metadata-namespace-rejection":
        await _run_fixture_028(spec)
    elif fixture_id == "038-otel-parallel-branches-dispatch-span":
        await _run_fixture_038(spec)
    elif fixture_id in {
        "012-otel-llm-payload-default-off",
        "013-otel-llm-payload-enabled",
        "014-otel-llm-payload-truncation",
        "015-otel-llm-payload-image-redaction",
        "016-otel-llm-request-params",
        "017-otel-llm-request-params-partial",
        "018-otel-llm-request-extras",
        "019-otel-llm-genai-semconv",
        "020-otel-llm-genai-system-override",
        "021-otel-llm-disable-genai-semconv",
        "025-otel-llm-request-params-extended",
        "026-otel-caller-supplied-metadata",
    }:
        await _run_llm_payload_fixture(spec)
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
# Fixture 002 — subgraph hierarchy
# ---------------------------------------------------------------------------


async def _run_fixture_002(spec: Mapping[str, Any]) -> None:
    """Spec §4.5: the subgraph wrapper synthesizes a dispatch span;
    inner-node spans parent under it; the dispatch span parents
    under the invocation."""
    observer, exporter = _build_observer()
    subgraphs = _compile_subgraphs(spec)
    trace_log: list[str] = []
    built = build_graph(spec, subgraphs=subgraphs, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    by_name: dict[str, list[Any]] = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)

    # Invocation span at the root.
    inv_list = by_name.get("openarmature.invocation") or []
    assert len(inv_list) == 1, f"expected 1 invocation span; got {len(inv_list)}"
    inv = inv_list[0]
    assert inv.parent is None
    assert inv.context is not None
    invocation_span_id = inv.context.span_id

    # Top-level outer nodes parent under invocation.
    for outer_node in ("outer_in", "outer_out"):
        outer_spans = by_name.get(outer_node) or []
        assert len(outer_spans) == 1, f"expected 1 span for {outer_node!r}; got {len(outer_spans)}"
        node = outer_spans[0]
        assert node.parent is not None and node.parent.span_id == invocation_span_id, (
            f"{outer_node!r} MUST parent under invocation span"
        )

    # The subgraph wrapper synthesizes a dispatch span at namespace
    # ("outer_sub",); its parent is the invocation span.
    sub_dispatch_spans = by_name.get("outer_sub") or []
    assert len(sub_dispatch_spans) == 1, (
        f"expected 1 synthetic subgraph dispatch span for outer_sub; got {len(sub_dispatch_spans)}"
    )
    sub_dispatch = sub_dispatch_spans[0]
    assert sub_dispatch.parent is not None and sub_dispatch.parent.span_id == invocation_span_id, (
        "subgraph dispatch span MUST parent under the invocation span per §4.5"
    )
    assert sub_dispatch.context is not None
    sub_dispatch_id = sub_dispatch.context.span_id
    sub_dispatch_attrs = dict(sub_dispatch.attributes or {})
    # Per observability §5.3 + coord thread `clarify-subgraph-name-
    # semantics` Option A: `openarmature.subgraph.name` carries the
    # compiled subgraph's identity, NOT the wrapper node name. The
    # conformance adapter sets ``subgraph_identity = "inner"`` when
    # compiling the fixture's ``subgraph: { name: inner }`` block.
    assert sub_dispatch_attrs.get("openarmature.subgraph.name") == "inner"

    # Inner-node spans parent under the subgraph dispatch span and
    # carry the nested namespace.
    for inner_node in ("inner_x", "inner_y"):
        inner_spans = by_name.get(inner_node) or []
        assert len(inner_spans) == 1, f"expected 1 span for {inner_node!r}; got {len(inner_spans)}"
        inner = inner_spans[0]
        assert inner.parent is not None and inner.parent.span_id == sub_dispatch_id, (
            f"{inner_node!r} MUST parent under the subgraph dispatch span per §4.5"
        )
        inner_attrs = dict(inner.attributes or {})
        assert list(inner_attrs.get("openarmature.node.namespace") or []) == ["outer_sub", inner_node], (
            f"{inner_node!r} namespace MUST be ['outer_sub', '{inner_node}']; got "
            f"{inner_attrs.get('openarmature.node.namespace')!r}"
        )


# ---------------------------------------------------------------------------
# Fixture 003 — error status mapping (node_exception case)
# ---------------------------------------------------------------------------


async def _run_fixture_003(spec: Mapping[str, Any]) -> None:
    """Spec §4.2: a node-exception failure produces an ERROR span
    with the canonical category in the description, an exception
    event recorded, and the ``openarmature.error.category``
    attribute. Sibling spans before the failure stay OK; the
    invocation span ends ERROR (OTel doesn't auto-propagate child
    status to parents, so the OTelObserver explicitly sets ERROR
    on the invocation span when any child errors per
    ``_handle_completed``)."""
    from opentelemetry.trace import StatusCode

    from openarmature.graph import RuntimeGraphError

    observer, exporter = _build_observer()
    trace_log: list[str] = []
    built = build_graph(spec, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    with pytest.raises(RuntimeGraphError):
        await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    by_name = {s.name: s for s in spans}

    ok_node = by_name.get("ok_node")
    assert ok_node is not None
    assert ok_node.status.status_code == StatusCode.OK, (
        f"ok_node status MUST be OK; got {ok_node.status.status_code}"
    )

    fail_node = by_name.get("fail_node")
    assert fail_node is not None
    assert fail_node.status.status_code == StatusCode.ERROR, (
        f"fail_node status MUST be ERROR; got {fail_node.status.status_code}"
    )
    assert fail_node.status.description == "node_exception", (
        f"fail_node status_description MUST be 'node_exception'; got {fail_node.status.description!r}"
    )
    fail_attrs = dict(fail_node.attributes or {})
    assert fail_attrs.get("openarmature.error.category") == "node_exception"
    # Exception event recorded on the span via record_exception.
    exception_events = [e for e in fail_node.events if e.name == "exception"]
    event_names = [e.name for e in fail_node.events]
    assert len(exception_events) >= 1, (
        f"fail_node MUST have at least one 'exception' event recorded; got {event_names}"
    )

    # Invocation span ends ERROR when any child errors per spec
    # §4.2 / fixture 003. The OTelObserver sets ERROR explicitly in
    # ``_handle_completed`` (OTel doesn't auto-propagate child status
    # to parents).
    inv = by_name.get("openarmature.invocation")
    assert inv is not None
    assert inv.status.status_code == StatusCode.ERROR, (
        f"invocation span status MUST be ERROR when a child errored; got {inv.status.status_code}"
    )


# ---------------------------------------------------------------------------
# Fixture 004 — routing-error attribution (proposal 0012 / spec v0.9.0)
# ---------------------------------------------------------------------------


async def _run_fixture_004(spec: Mapping[str, Any]) -> None:
    """Spec §4.2 + spec v0.9.0 / proposal 0012: routing errors land on
    the preceding node's ``completed`` event with ``error`` populated
    (sharing the started/completed pair rather than producing a
    separate one). The OTel observer's existing
    ``_handle_completed`` ERROR-mapping path picks this up
    automatically — no observer-side change needed for the swap.

    Driver verifies: the ``pick`` node's span ends ERROR with
    ``status_description == "routing_error"``, an ``exception``
    event recorded, and the ``openarmature.error.category``
    attribute. No span for the edge function (no ``edge_spans``)
    per §4.2's "edge logic folded into the preceding node span"
    framing."""
    from opentelemetry.trace import StatusCode

    from openarmature.graph import RuntimeGraphError

    observer, exporter = _build_observer()
    trace_log: list[str] = []
    built = build_graph(spec, trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(spec.get("initial_state", {}))
    with pytest.raises(RuntimeGraphError) as excinfo:
        await compiled.invoke(initial_state)
    assert excinfo.value.category == "routing_error"
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    by_name = {s.name: s for s in spans}

    pick = by_name.get("pick")
    assert pick is not None
    assert pick.status.status_code == StatusCode.ERROR, (
        f"preceding node 'pick' span MUST be ERROR; got {pick.status.status_code}"
    )
    assert pick.status.description == "routing_error", (
        f"preceding node 'pick' span status_description MUST be 'routing_error'; "
        f"got {pick.status.description!r}"
    )
    pick_attrs = dict(pick.attributes or {})
    assert pick_attrs.get("openarmature.error.category") == "routing_error"
    # Exception event recorded on the span via record_exception.
    exception_events = [e for e in pick.events if e.name == "exception"]
    event_names = [e.name for e in pick.events]
    assert len(exception_events) >= 1, (
        f"'pick' MUST have at least one 'exception' event recorded; got {event_names}"
    )

    # Per fixture 004's "no_edge_spans: true" — the edge function
    # itself does not produce a separate span; the routing error is
    # folded into the preceding node's span.
    edge_span_names = {"edge", "openarmature.edge", "edge_function"}
    edge_spans = [s for s in spans if s.name in edge_span_names]
    assert edge_spans == [], (
        f"there MUST be no separate edge-function spans per §4.2; got {[s.name for s in edge_spans]}"
    )

    # Unreachable nodes never fire spans (they were unreached).
    for unreachable in ("unreachable_a", "unreachable_b"):
        assert unreachable not in by_name, f"{unreachable!r} MUST not produce a span — never reached"

    # Invocation span ends ERROR per the §4.2 invocation-status
    # propagation contract (PR-C review fix).
    inv = by_name.get("openarmature.invocation")
    assert inv is not None
    assert inv.status.status_code == StatusCode.ERROR, (
        f"invocation span MUST end ERROR when a child errors; got {inv.status.status_code}"
    )


# ---------------------------------------------------------------------------
# Fixture 007 — retry attempt spans
# ---------------------------------------------------------------------------


async def _run_fixture_006(spec: Mapping[str, Any]) -> None:
    """Spec §5.4 + proposal 0013 (v0.10.0): non-detached fan-out
    instances synthesize per-instance dispatch spans nested between
    the fan-out node span and the inner-node spans. The fan-out node
    span carries ``item_count`` / ``concurrency`` / ``error_policy``
    from ``NodeEvent.fan_out_config``; per-instance spans carry
    ``fan_out_index`` and ``parent_node_name``."""
    # Subgraphs declared at the spec level (outside ``cases:``) — the
    # ``worker`` subgraph used by every case in this fixture lives
    # there. Compile once and reuse across cases.
    _patch_unsupported_directives(spec)
    subgraphs = _compile_subgraphs(spec)
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_006_case(case, subgraphs)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_006_case(case: Mapping[str, Any], subgraphs: Mapping[str, Any]) -> None:
    _patch_unsupported_directives(case)

    observer, exporter = _build_observer()
    trace_log: list[str] = []
    built = build_graph(case, subgraphs=dict(subgraphs), trace=trace_log)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))
    await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    # Span-tree shape per fixture:
    #   invocation
    #   └─ process (fan-out NODE) — item_count=3, concurrency=2, error_policy="collect"
    #      ├─ process (instance, fan_out_index=0) — parent_node_name="process"
    #      │  └─ compute
    #      ├─ process (instance, fan_out_index=1) — parent_node_name="process"
    #      │  └─ compute
    #      └─ process (instance, fan_out_index=2) — parent_node_name="process"
    #         └─ compute
    process_spans = [s for s in spans if s.name == "process"]
    # Expect 4: 1 fan-out node + 3 per-instance dispatch spans.
    assert len(process_spans) == 4, (
        f"expected 4 'process' spans (1 fan-out node + 3 per-instance dispatch); got {len(process_spans)}"
    )
    # Fan-out node span carries the three §5.4 attributes.
    fan_out_node_spans = [
        s
        for s in process_spans
        if dict(s.attributes or {}).get("openarmature.fan_out.item_count") is not None
    ]
    assert len(fan_out_node_spans) == 1, (
        f"expected exactly 1 fan-out NODE span (with item_count attribute); got {len(fan_out_node_spans)}"
    )
    fan_out_node_span = fan_out_node_spans[0]
    fan_out_attrs = dict(fan_out_node_span.attributes or {})
    assert fan_out_attrs.get("openarmature.fan_out.item_count") == 3
    assert fan_out_attrs.get("openarmature.fan_out.concurrency") == 2
    assert fan_out_attrs.get("openarmature.fan_out.error_policy") == "collect"

    # Per-instance dispatch spans: 3 of them, each with
    # fan_out_index 0..2 and parent_node_name "process".
    per_instance_spans = [s for s in process_spans if s != fan_out_node_span]
    assert len(per_instance_spans) == 3
    fan_out_indices: set[int] = set()
    for s in per_instance_spans:
        attrs = dict(s.attributes or {})
        idx = attrs.get("openarmature.node.fan_out_index")
        assert isinstance(idx, int), f"per-instance span MUST carry fan_out_index; got attrs={attrs}"
        fan_out_indices.add(idx)
        assert attrs.get("openarmature.fan_out.parent_node_name") == "process", (
            f"per-instance span MUST carry parent_node_name='process'; got {attrs}"
        )
        # Each per-instance dispatch span parents under the fan-out
        # node span (proposal 0013 + §5.4 nesting).
        assert s.parent is not None and s.parent.span_id == fan_out_node_span.context.span_id, (
            "per-instance dispatch span MUST parent under the fan-out node span"
        )
    assert fan_out_indices == {0, 1, 2}, (
        f"per-instance fan_out_index range MUST be 0..2; got {sorted(fan_out_indices)}"
    )

    # Each per-instance dispatch span has a 'compute' child (the
    # inner-node work).
    per_instance_ids = {s.context.span_id for s in per_instance_spans}
    compute_spans = [s for s in spans if s.name == "compute"]
    assert len(compute_spans) == 3, f"expected 3 compute spans; got {len(compute_spans)}"
    for cs in compute_spans:
        assert cs.parent is not None and cs.parent.span_id in per_instance_ids, (
            "compute span MUST parent under a per-instance dispatch span"
        )


async def _run_fixture_007(spec: Mapping[str, Any]) -> None:
    """Two sub-cases:

    1. ``three_attempts_third_succeeds`` — retry succeeds on
       attempt 2; expect 3 sibling attempt spans (ERROR, ERROR, OK).
    2. ``retry_exhausts_all_three_spans_error`` — retry exhausts;
       expect 3 sibling attempt spans (all ERROR); invoke raises.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_007_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_007_case(case: Mapping[str, Any]) -> None:
    from opentelemetry.trace import StatusCode

    from openarmature.graph import RuntimeGraphError
    from openarmature.graph.middleware import RetryMiddleware
    from openarmature.graph.middleware.retry import deterministic_backoff

    observer, exporter = _build_observer()
    # The fixture's flaky directive uses ``fail_count: N`` shape
    # (fail attempts 0..N-1, succeed on attempt N) which the
    # adapter doesn't translate; rewrite to the adapter's
    # ``failure_sequence`` shape before building.
    flaky_node_name = cast("str", case["entry"])
    nodes = cast("dict[str, Any]", case["nodes"])
    flaky_node = cast("dict[str, Any]", nodes[flaky_node_name])
    flaky_directive = cast("dict[str, Any]", flaky_node["flaky"])
    fail_count = int(flaky_directive["fail_count"])
    fail_category = cast("str", flaky_directive.get("category", "provider_unavailable"))
    on_success = cast("dict[str, Any]", flaky_directive.get("on_success", {}))
    flaky_node["flaky"] = {
        "failure_sequence": [
            {"category": fail_category, "message": f"flaky attempt {i}"} for i in range(fail_count)
        ],
        "success_update": on_success,
    }
    # Translate the per-node retry middleware. The adapter accepts
    # ``node_middleware`` mapping; the YAML's
    # ``nodes.flaky.middleware: [{type: retry, ...}]`` maps in.
    middleware_specs = cast("list[dict[str, Any]]", flaky_node.pop("middleware", []) or [])
    node_middleware: dict[str, list[Any]] = {}
    for mw_spec in middleware_specs:
        if mw_spec["type"] != "retry":
            raise AssertionError(f"fixture 007: unexpected middleware type {mw_spec['type']!r}")
        backoff_cfg = cast(
            "dict[str, Any]", mw_spec.get("backoff") or {"type": "deterministic", "seconds": 0}
        )
        if backoff_cfg["type"] != "deterministic":
            raise AssertionError(f"fixture 007: unsupported backoff type {backoff_cfg['type']!r}")
        backoff = deterministic_backoff(float(backoff_cfg.get("seconds", 0)))
        classifier_cfg = cast("dict[str, Any] | None", mw_spec.get("classifier"))
        if classifier_cfg is not None:
            transient = frozenset(cast("list[str]", classifier_cfg.get("transient_categories", [])))

            def _classifier(exc: Exception, _state: Any, _transient: frozenset[str] = transient) -> bool:
                return getattr(exc, "category", None) in _transient

            classifier_fn: Any = _classifier
        else:
            classifier_fn = None
        node_middleware.setdefault(flaky_node_name, []).append(
            RetryMiddleware(
                max_attempts=int(mw_spec.get("max_attempts", 3)),
                backoff=backoff,
                classifier=classifier_fn,
            )
        )

    trace_log: list[str] = []
    built = build_graph(case, trace=trace_log, node_middleware=node_middleware)
    compiled = built.builder.compile()
    compiled.attach_observer(observer)
    initial_state = built.initial_state(case.get("initial_state", {}))
    expected_error = case.get("expected_error")
    if expected_error is not None:
        with pytest.raises(RuntimeGraphError):
            await compiled.invoke(initial_state)
    else:
        await compiled.invoke(initial_state)
    await compiled.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    attempt_spans = [s for s in spans if s.name == flaky_node_name]
    assert len(attempt_spans) == 3, (
        f"expected 3 sibling attempt spans for {flaky_node_name!r}; got {len(attempt_spans)}"
    )
    # Each attempt span has a distinct attempt_index in 0..2 and
    # they all share the invocation as parent (siblings).
    attempt_indices: list[int] = []
    parent_span_ids: set[int] = set()
    for span in attempt_spans:
        attrs = dict(span.attributes or {})
        idx = attrs.get("openarmature.node.attempt_index")
        assert isinstance(idx, int)
        attempt_indices.append(idx)
        assert span.parent is not None
        parent_span_ids.add(span.parent.span_id)
    assert sorted(attempt_indices) == [0, 1, 2], (
        f"attempt_index values MUST be 0..2; got {sorted(attempt_indices)}"
    )
    assert len(parent_span_ids) == 1, (
        "all attempt spans MUST share the same parent (sibling-level under the invocation); "
        f"got {len(parent_span_ids)} distinct parents"
    )

    # Status assertions.
    by_attempt = {
        cast("int", dict(s.attributes or {})["openarmature.node.attempt_index"]): s for s in attempt_spans
    }
    if expected_error is not None:
        # All three attempts ERROR.
        for idx in (0, 1, 2):
            assert by_attempt[idx].status.status_code == StatusCode.ERROR, (
                f"attempt {idx} status MUST be ERROR (retry exhausted); "
                f"got {by_attempt[idx].status.status_code}"
            )
    else:
        # Attempts 0 + 1 ERROR, attempt 2 OK.
        for idx in (0, 1):
            assert by_attempt[idx].status.status_code == StatusCode.ERROR, (
                f"attempt {idx} status MUST be ERROR (failed before retry succeeded); "
                f"got {by_attempt[idx].status.status_code}"
            )
        assert by_attempt[2].status.status_code == StatusCode.OK, (
            f"attempt 2 status MUST be OK (success on third attempt); got {by_attempt[2].status.status_code}"
        )


# ---------------------------------------------------------------------------
# Fixture 011 — determinism
# ---------------------------------------------------------------------------


# Spec-canonical attributes that are non-deterministic by design and
# MUST be excluded from determinism-comparison runs. Per spec
# coordination on the fixture (08-spec-prep-sync-confirmed reaffirms
# the §3.2 + §5.1 distinction): caller-supplied correlation_id is
# deterministic; auto-generated UUIDv4 is not. Fixture 011 omits
# ``caller_correlation_id``, so the auto-generated correlation_id IS
# in the ignore set for this fixture.
_DETERMINISM_IGNORED_ATTRS: frozenset[str] = frozenset(
    {
        "openarmature.invocation_id",
        "openarmature.correlation_id",
    }
)


async def _run_fixture_011(spec: Mapping[str, Any]) -> None:
    """Spec §8: deterministic span content is identical across two
    invocations of the same graph with the same input. The
    signature compared per-span:
    ``(name, status_code, parent_name, attrs ∖ ignored_set)``.
    Parent linkage is encoded as the parent span's NAME rather
    than its span_id (span_ids are non-deterministic per OTel SDK's
    default RandomIdGenerator); a hierarchy regression where a
    node reparented to a different ancestor surfaces as a
    parent_name divergence."""
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_011_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_011_case(case: Mapping[str, Any]) -> None:
    # Translate the fixture's ``when:`` conditional-edge syntax
    # (``when: {field: counter, gt: 0}``) into the adapter's
    # ``condition: {if_field, equals, then, else}`` shape. The
    # adapter doesn't have a ``gt`` builder, but the deterministic
    # input means ``counter == 1`` always — so ``gt: 0`` is
    # functionally equivalent to ``equals: 1`` for this fixture's
    # flow. The determinism comparison itself doesn't depend on
    # which adapter construct represents the edge; the same
    # branch always fires under identical inputs. (Generic
    # ``gt``/``lt``/etc. edge-condition support is tracked under
    # the Harness backlog in
    # ``openarmature-coord/docs/phase-6-1-conformance-fillin.md``.)
    case_for_build = _translate_011_when_edges(case)

    invocations = int(case.get("invocations", 2))
    assert invocations == 2, f"fixture 011: expected invocations=2; got {invocations}"

    runs: list[list[Any]] = []
    for _ in range(invocations):
        observer, exporter = _build_observer()
        trace_log: list[str] = []
        built = build_graph(case_for_build, trace=trace_log)
        compiled = built.builder.compile()
        compiled.attach_observer(observer)
        initial_state = built.initial_state(case.get("initial_state", {}))
        await compiled.invoke(initial_state)
        await compiled.drain()
        observer.shutdown()
        runs.append(list(exporter.get_finished_spans()))

    assert len(runs[0]) == len(runs[1]), (
        f"deterministic input MUST produce equal span counts; got {len(runs[0])} vs {len(runs[1])}"
    )

    # Compare each span's structural signature across runs. Span
    # span_ids are non-deterministic, so we encode the parent
    # linkage by looking up parent.span_id in the same run's
    # by-id map and including the parent's NAME in the signature.
    # That way a hierarchy regression (e.g., a node reparented
    # from invocation to a sibling) shows up as a signature
    # difference even though both spans' own attributes are
    # unchanged.
    def _signature(
        span: Any, by_id: Mapping[int, Any]
    ) -> tuple[str, str, str | None, tuple[tuple[str, Any], ...]]:
        attrs = dict(span.attributes or {})
        deterministic_items = sorted(
            (k, _normalize_attr_value(v)) for k, v in attrs.items() if k not in _DETERMINISM_IGNORED_ATTRS
        )
        parent_name: str | None = None
        if span.parent is not None:
            parent_span = by_id.get(span.parent.span_id)
            if parent_span is not None:
                parent_name = cast("str", parent_span.name)
        return (
            cast("str", span.name),
            str(span.status.status_code),
            parent_name,
            tuple(deterministic_items),
        )

    by_id_run_0: dict[int, Any] = {}
    for s in runs[0]:
        if s.context is not None:
            by_id_run_0[s.context.span_id] = s
    by_id_run_1: dict[int, Any] = {}
    for s in runs[1]:
        if s.context is not None:
            by_id_run_1[s.context.span_id] = s
    sig_run_0 = sorted(_signature(s, by_id_run_0) for s in runs[0])
    sig_run_1 = sorted(_signature(s, by_id_run_1) for s in runs[1])
    assert sig_run_0 == sig_run_1, (
        f"deterministic span content MUST match across runs; "
        f"first divergence: run_0={sig_run_0!r} vs run_1={sig_run_1!r}"
    )


async def _run_fixture_028(spec: Mapping[str, Any]) -> None:
    """Proposal 0034 §3.4: caller-supplied metadata keys under
    reserved namespaces (``openarmature.*``, ``gen_ai.*``) MUST
    raise at the ``invoke()`` boundary before any work begins.
    The harness asserts:

    - The invocation raises ``ValueError`` synchronously.
    - No OTel spans are emitted (the OTel observer attached
      to the graph never saw a single event).
    - No Langfuse observations are emitted (the Langfuse
      observer attached likewise saw nothing).
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    from openarmature.graph import END, GraphBuilder  # noqa: PLC0415
    from openarmature.observability.langfuse import (  # noqa: PLC0415
        InMemoryLangfuseClient,
        LangfuseObserver,
    )

    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            # Build a minimal graph from the case's nodes/edges. The
            # fixture's node is a noop update — we never expect it to
            # run since the boundary rejects before any worker spins
            # up.
            from .adapter import build_state_cls  # noqa: PLC0415

            state_cls = build_state_cls("RejectionFixtureState", case["state"]["fields"])
            builder = GraphBuilder(state_cls)
            nodes_spec = cast("dict[str, Any]", case["nodes"])
            for node_name, node_spec in nodes_spec.items():
                node_dict = cast("dict[str, Any]", node_spec)
                update_block = cast("dict[str, Any]", node_dict["update"])
                augment_block = cast("dict[str, Any] | None", node_dict.get("augment_metadata"))

                def _make_body(
                    payload: dict[str, Any],
                    augment: dict[str, Any] | None,
                ) -> Any:
                    # Per spec §3.4 + proposal 0040: the augment_metadata
                    # primitive injects a ``set_invocation_metadata(**augment)``
                    # call at the top of the node body. Used by 028's
                    # mid-invocation-rejection case (reserved name `step`)
                    # and by 034 for the open-span update demonstration.
                    from openarmature.observability.metadata import (  # noqa: PLC0415
                        set_invocation_metadata,
                    )

                    async def _body(_s: Any) -> dict[str, Any]:
                        if augment is not None:
                            set_invocation_metadata(**augment)
                        return dict(payload)

                    return _body

                builder.add_node(node_name, _make_body(update_block, augment_block))
            for edge in cast("list[dict[str, str]]", case["edges"]):
                target_raw = edge["to"]
                target = END if target_raw == "END" else target_raw
                builder.add_edge(edge["from"], target)
            builder.set_entry(cast("str", case["entry"]))
            graph = builder.compile()

            exporter = InMemorySpanExporter()
            otel_observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
            langfuse_client = InMemoryLangfuseClient()
            langfuse_observer = LangfuseObserver(client=langfuse_client)
            graph.attach_observer(otel_observer)
            graph.attach_observer(langfuse_observer)

            caller_metadata = cast("dict[str, Any]", case["caller_metadata"])
            expected = cast("dict[str, Any]", case["expected"])
            expects_boundary_rejection = expected.get("invoke_rejects_at_api_boundary", False)
            expects_call_site_rejection = expected.get("augment_rejects_at_call_site", False)
            try:
                if expects_boundary_rejection:
                    # Boundary-rejection path: invoke()'s caller_metadata
                    # validator rejects before any work begins. Covers
                    # both the prefix-namespace rejection (openarmature.*
                    # / gen_ai.*, from 0034) and the exact-key-name
                    # rejection (0041's §8.4 reserved set). Both error
                    # messages contain "reserved".
                    with pytest.raises(ValueError, match="reserved"):
                        await graph.invoke(state_cls(), metadata=caller_metadata)
                elif expects_call_site_rejection:
                    # Mid-invocation rejection path: caller_metadata
                    # passes the boundary; the node body's
                    # ``set_invocation_metadata(**augment)`` raises a
                    # ValueError at the call site. The engine wraps the
                    # node-body raise in NodeException whose
                    # ``__cause__`` is the ValueError. The §3.4 contract
                    # is that the helper raises at the call site — the
                    # reserved key MUST NOT reach any emission, hence
                    # no spans / no Langfuse observations afterward.
                    from openarmature.graph import NodeException  # noqa: PLC0415

                    with pytest.raises(NodeException) as exc_info:
                        await graph.invoke(state_cls(), metadata=caller_metadata)
                    cause = exc_info.value.__cause__
                    assert isinstance(cause, ValueError), (
                        f"expected NodeException.__cause__ to be ValueError; got {type(cause).__name__}"
                    )
                    assert "reserved" in str(cause), f"expected 'reserved' in cause message; got {cause!s}"
                else:
                    raise AssertionError(
                        "case has neither invoke_rejects_at_api_boundary nor augment_rejects_at_call_site set"
                    )
                await graph.drain()
            finally:
                otel_observer.shutdown()

            if expected.get("no_spans_emitted"):
                spans = exporter.get_finished_spans()
                assert len(spans) == 0, f"expected zero spans, got {[s.name for s in spans]}"
            if expected.get("no_langfuse_observations_emitted"):
                # Trace MAY exist (lazy-open on first event); the
                # invariant is "no observations are emitted." Since
                # invoke rejects before any event fires, neither
                # trace nor observations should be created.
                assert len(langfuse_client.traces) == 0, (
                    f"expected zero Langfuse traces, got {sorted(langfuse_client.traces.keys())}"
                )
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


def _normalize_attr_value(value: Any) -> Any:
    """OTel attribute values can be tuple or list shapes for sequence
    types depending on how they were set; normalize for comparison."""
    if isinstance(value, list):
        return tuple(cast("list[Any]", value))
    if isinstance(value, tuple):
        return cast("tuple[Any, ...]", value)
    return value


def _translate_011_when_edges(case: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite fixture 011's ``when: {field: counter, gt: 0}``
    edges into the adapter's ``condition: {if_field, equals,
    then, else}`` shape. The deterministic input always satisfies
    the branch, so the comparison can be ``equals: 1``."""
    new_case = cast("dict[str, Any]", copy.deepcopy(case))
    new_edges: list[Any] = []
    branch_when_edge: dict[str, Any] | None = None
    branch_default_edge: dict[str, Any] | None = None
    for edge in cast("list[dict[str, Any]]", new_case.get("edges", [])):
        if "when" in edge:
            branch_when_edge = edge
        elif edge.get("from") == "branch" and "when" not in edge:
            branch_default_edge = edge
        else:
            new_edges.append(edge)
    if branch_when_edge is not None and branch_default_edge is not None:
        when = cast("dict[str, Any]", branch_when_edge["when"])
        if_field = cast("str", when["field"])
        # gt: 0 with the deterministic input (counter == 1) →
        # equals: 1 is equivalent for this fixture's flow.
        new_edges.append(
            {
                "from": "branch",
                "condition": {
                    "if_field": if_field,
                    "equals": 1,
                    "then": branch_when_edge["to"],
                    "else": branch_default_edge["to"],
                },
            }
        )
    elif branch_when_edge is not None or branch_default_edge is not None:
        raise AssertionError("fixture 011: expected paired when/default edges from 'branch'")
    new_case["edges"] = new_edges
    return new_case


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
        # OTel SDK 1.x's ``set_tracer_provider`` is guarded by a
        # ``_TRACER_PROVIDER_SET_ONCE`` primitive — once a non-default
        # provider is set, subsequent calls are silent no-ops (with a
        # WARNING log "Overriding of current TracerProvider is not
        # allowed"). If a prior test in the suite-run order left a
        # non-default provider behind, the call below would no-op and
        # this case's ``global_exporter`` would receive 0 spans. Reset
        # both the value AND the Once explicitly so this case's set
        # always wins. The finally block below restores ``prior_global``
        # via the same direct reset so the next test starts clean.
        once = otel_trace._TRACER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
        with once._lock:  # pyright: ignore[reportPrivateUsage]
            otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
            once._done = False  # pyright: ignore[reportPrivateUsage]
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
            # OTel SDK 1.x makes set_tracer_provider one-shot: once a
            # non-default provider is set, subsequent calls are no-ops.
            # Restore by resetting the private Once + state directly so
            # the global doesn't leak into subsequent tests.
            _reset_otel_global_tracer_provider(prior_global)

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


async def _run_fixture_038(spec: Mapping[str, Any]) -> None:
    """Single-case proposal-0044 fixture: a two-branch parallel-branches
    graph where each branch's inner ``ask`` node makes an LLM call.

    The OTel observer MUST synthesize a per-branch dispatch span between
    the parallel-branches NODE span and each branch's inner-node spans;
    the §5.7 attribute surface (``branch_count`` + ``error_policy`` on
    the NODE span, ``branch_name`` + ``parent_node_name`` on each
    dispatch span, ``branch_name`` on inner-branch leaf spans) MUST
    appear; per-branch dispatch spans MUST close before the NODE span
    in branch-declaration order.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    assert len(cases) == 1, f"fixture 038 expects exactly one case; got {len(cases)}"
    case = cases[0]
    case_name = cast("str", case["name"])
    try:
        await _run_fixture_038_case(case)
    except AssertionError as e:
        raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_038_case(case: Mapping[str, Any]) -> None:
    import json

    import httpx
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from openarmature.graph import END, BranchSpec, GraphBuilder
    from openarmature.llm import OpenAIProvider, UserMessage

    from .adapter import build_state_cls

    # ---- Build a queue-backed mock LLM transport.  The fixture's
    # per-branch ``ask`` nodes share a single OpenAIProvider keyed off
    # an httpx MockTransport.  Both branches dispatch concurrently and
    # this fixture asserts span topology + §5.7 attributes only (not
    # response-content routing), so a FIFO mock returning any queued
    # response per call is correct.  Build the queue from the
    # fixture-declared ``calls_llm.response`` values; the per-user-msg
    # bookkeeping below is a side effect of parsing the fixture, not
    # used by the handler.
    branches_spec = cast("dict[str, Any]", case["nodes"]["dispatcher"]["parallel_branches"]["branches"])
    branch_response_by_user_msg: dict[str, str] = {}
    for branch_name, branch_spec in branches_spec.items():
        sub_id = cast("str", branch_spec["subgraph"])
        sub = cast("dict[str, Any]", case["subgraphs"][sub_id])
        ask_calls_llm = cast("dict[str, Any]", sub["nodes"]["ask"]["calls_llm"])
        user_msg = "answer the question"
        if "messages" in ask_calls_llm:
            messages = cast("list[dict[str, str]]", ask_calls_llm["messages"])
            user_msg = next((m["content"] for m in messages if m.get("role") == "user"), user_msg)
        branch_response_by_user_msg[user_msg + f"::{branch_name}"] = cast(
            "str", ask_calls_llm.get("response", f"{branch_name} response")
        )

    fallback_responses = list(branch_response_by_user_msg.values())

    def _handler(_request: httpx.Request) -> httpx.Response:
        # Both branches dispatch concurrently; response-to-branch
        # mapping is non-deterministic by design.  The fixture asserts
        # span topology + §5.7 attributes, NOT response-content
        # routing — so a FIFO mock returning ANY of the queued
        # responses is correct.  Don't add response-content assertions
        # without first switching the mock to a content-routed shape.
        if not fallback_responses:
            raise AssertionError("mock_llm queue exhausted")
        next_response = fallback_responses.pop(0)
        body = {
            "id": "test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": next_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(
            200, content=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )

    transport = httpx.MockTransport(_handler)
    provider = OpenAIProvider(
        base_url="http://mock-llm.test", model="test-model", api_key="test", transport=transport
    )

    # ---- Build the inner subgraphs (one per branch).  Each inner
    # subgraph has a single ``ask`` node that calls the mock provider.
    subgraphs: dict[str, Any] = {}
    for sub_id, sub_spec in cast("dict[str, Any]", case["subgraphs"]).items():
        inner_fields = cast("dict[str, dict[str, Any]]", sub_spec["state"]["fields"])
        inner_state_cls = build_state_cls(f"Inner_{sub_id}", inner_fields)
        ask_calls_llm = cast("dict[str, Any]", sub_spec["nodes"]["ask"]["calls_llm"])
        stores_in = cast("str", ask_calls_llm.get("stores_response_in", "msg"))
        messages_in: tuple[Any, ...] = tuple(
            UserMessage(content=m["content"])
            for m in cast("list[dict[str, str]]", ask_calls_llm.get("messages", []))
            if m.get("role") == "user"
        ) or (UserMessage(content="answer the question"),)

        # Need to bind ``stores_in`` and the messages into the closure
        # for each subgraph independently — default-arg binding is the
        # idiomatic late-binding sidestep for loop-scope closures.
        async def _ask_body(
            _s: Any,
            stores: str = stores_in,
            messages: tuple[Any, ...] = messages_in,
        ) -> dict[str, str]:
            response = await provider.complete(list(messages))
            return {stores: response.message.content or ""}

        subgraphs[sub_id] = (
            GraphBuilder(inner_state_cls)
            .add_node("ask", _ask_body)
            .add_edge("ask", END)
            .set_entry("ask")
            .compile()
        )

    # ---- Build the outer graph with the parallel-branches node.
    outer_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    outer_state_cls = build_state_cls("Outer_038", outer_fields)
    error_policy = cast(
        "str", case["nodes"]["dispatcher"]["parallel_branches"].get("error_policy", "fail_fast")
    )
    branches = {
        branch_name: BranchSpec(subgraph=subgraphs[cast("str", branch_spec["subgraph"])])
        for branch_name, branch_spec in branches_spec.items()
    }
    builder = (
        GraphBuilder(outer_state_cls)
        .add_parallel_branches_node("dispatcher", branches=branches, error_policy=error_policy)  # type: ignore[arg-type]
        .add_edge("dispatcher", END)
        .set_entry("dispatcher")
    )
    graph = builder.compile()

    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    graph.attach_observer(observer)
    try:
        await graph.invoke(outer_state_cls())
        await graph.drain()
    finally:
        observer.shutdown()
        await provider.aclose()

    spans = exporter.get_finished_spans()

    # ---- span_tree assertions
    expected_tree = cast("list[dict[str, Any]]", case["expected"]["span_tree"])
    # Find the invocation root.
    inv_root = next(
        (s for s in spans if s.name == "openarmature.invocation" and cast("Any", s.parent) is None), None
    )
    assert inv_root is not None, f"invocation root span missing; got {[s.name for s in spans]}"
    _assert_span_tree_matches(spans, [inv_root], expected_tree)

    # ---- Invariants
    invariants = cast("dict[str, Any]", case["expected"].get("invariants") or {})
    dispatch_spans = [
        s
        for s in spans
        if (s.attributes or {}).get("openarmature.parallel_branches.parent_node_name") is not None
    ]
    node_span = next(
        (
            s
            for s in spans
            if s.name == "dispatcher"
            and (s.attributes or {}).get("openarmature.parallel_branches.branch_count") is not None
        ),
        None,
    )
    if invariants.get("same_named_inner_spans_disambiguated_by_dispatch_parent"):
        ask_spans = [s for s in spans if s.name == "ask"]
        assert len(ask_spans) == 2, f"expected 2 inner ask spans; got {len(ask_spans)}"
        dispatch_span_ids = {cast("Any", d.context).span_id for d in dispatch_spans}
        ask_parents = {cast("Any", s.parent).span_id for s in ask_spans if s.parent is not None}
        assert ask_parents.issubset(dispatch_span_ids), (
            "same-named ask spans MUST parent under distinct per-branch dispatch spans"
        )
        assert len(ask_parents) == 2, "each ask span MUST parent under a DIFFERENT dispatch span"
    if invariants.get("dispatch_spans_close_before_node_span"):
        assert node_span is not None
        node_end = node_span.end_time
        for d in dispatch_spans:
            assert d.end_time is not None and node_end is not None and d.end_time <= node_end, (
                f"dispatch span {d.name!r} MUST close before parallel-branches NODE span"
            )
    declaration_order = cast("list[str] | None", invariants.get("dispatch_spans_close_in_declaration_order"))
    if declaration_order is not None:
        dispatch_by_name = {d.name: d for d in dispatch_spans}
        ends = [
            (name, dispatch_by_name[name].end_time) for name in declaration_order if name in dispatch_by_name
        ]
        found = [n for n, _ in ends]
        assert len(ends) == len(declaration_order), (
            f"declaration_order references {declaration_order!r} but only {found} dispatch spans found"
        )
        for (prev_name, prev_end), (name, end) in zip(ends, ends[1:], strict=False):
            assert prev_end is not None and end is not None and prev_end <= end, (
                f"dispatch span {prev_name!r} (end={prev_end}) MUST close before {name!r} (end={end})"
            )


def _assert_span_tree_matches(
    all_spans: Sequence[Any], actual_roots: Sequence[Any], expected_nodes: Sequence[Mapping[str, Any]]
) -> None:
    """Recursive structural match: for every expected node, find a
    matching actual span by name + attribute-subset; recurse on its
    children.  Children are matched as a SET (order independent —
    parallel-branches dispatch order isn't span-emission order)."""
    actual_by_name: dict[str, list[Any]] = {}
    for root in actual_roots:
        actual_by_name.setdefault(root.name, []).append(root)

    for expected in expected_nodes:
        expected_name = cast("str", expected["name"])
        candidates = actual_by_name.get(expected_name, [])
        expected_attrs = cast("dict[str, Any]", expected.get("attributes") or {})

        def _matches(span: Any, eattrs: dict[str, Any] = expected_attrs) -> bool:
            attrs = dict(span.attributes or {})
            return all(attrs.get(k) == v for k, v in eattrs.items())

        matching = [c for c in candidates if _matches(c)]
        assert len(matching) >= 1, (
            f"no span found matching expected node name={expected_name!r} attrs={expected_attrs!r}; "
            f"candidates: {[c.name for c in candidates]}"
        )
        # Ambiguous-match guard: when multiple candidates pass the
        # name + attribute-subset filter, fail loudly rather than
        # silently picking the first one.  Future fixtures with
        # multiple same-named siblings at the same tree level MUST
        # provide a disambiguator attribute in the expected node's
        # ``attributes`` block.
        assert len(matching) == 1, (
            f"ambiguous match for expected node name={expected_name!r} "
            f"attrs={expected_attrs!r}: {len(matching)} candidates pass "
            f"the name + attribute-subset filter.  Add a disambiguating "
            f"attribute to the expected node's ``attributes:`` block."
        )
        matched = matching[0]
        actual_by_name[expected_name] = [c for c in candidates if c is not matched]
        expected_status = cast("str | None", expected.get("status"))
        if expected_status is not None:
            actual_status_name = matched.status.status_code.name
            # OTel's StatusCode default is UNSET — observers set OK
            # explicitly on success ends, but the spec's expected
            # ``status: OK`` semantic is "non-error", so accept
            # UNSET as OK.  ERROR vs OK is the load-bearing
            # distinction.
            if expected_status == "OK" and actual_status_name in {"OK", "UNSET"}:
                pass
            else:
                assert actual_status_name == expected_status, (
                    f"{expected_name!r} status: expected {expected_status!r}, got {actual_status_name!r}"
                )
        expected_children = cast("list[dict[str, Any]] | None", expected.get("children"))
        if expected_children is not None:
            matched_span_id = matched.context.span_id
            actual_children = [
                s for s in all_spans if s.parent is not None and s.parent.span_id == matched_span_id
            ]
            _assert_span_tree_matches(all_spans, actual_children, expected_children)


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


# ---------------------------------------------------------------------------
# Fixture 010 — log correlation (PR-C.3)
#
# Two sub-cases. Both build the graph by hand rather than going through the
# adapter — fixture 010's ``emits_log:`` directive isn't an adapter primitive
# (the adapter recognizes ``update_pure``, ``subgraph``, etc., and silently
# ignores anything else), and the sub-cases are small enough that hand-built
# python is clearer than threading a new directive through the adapter.
# ---------------------------------------------------------------------------


def _setup_isolated_log_bridge() -> tuple[Any, Any, Any]:
    """Spin up an OTel ``LoggerProvider`` + ``InMemoryLogRecordExporter`` and
    install the log bridge against the root logger, snapshotting the prior
    log state so the caller can restore it in ``finally`` (the bridge mutates
    process-global ``logging`` state — handlers, factory).

    Returns ``(exporter, provider, restore_state)`` where ``restore_state``
    is a snapshot to pass to :func:`_restore_log_state`.
    """
    import logging as _logging  # noqa: PLC0415

    from opentelemetry.sdk._logs import LoggerProvider  # noqa: PLC0415
    from opentelemetry.sdk._logs.export import (  # noqa: PLC0415
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )

    from openarmature.observability.otel import install_log_bridge  # noqa: PLC0415

    root = _logging.getLogger()
    snapshot = (list(root.handlers), list(root.filters), _logging.getLogRecordFactory())

    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    install_log_bridge(provider)
    return exporter, provider, snapshot


def _restore_log_state(snapshot: Any) -> None:
    """Pair to :func:`_setup_isolated_log_bridge` — restores the root logger's
    handler list, filters, and ``LogRecord`` factory to the snapshot taken
    before ``install_log_bridge`` ran."""
    import logging as _logging  # noqa: PLC0415

    handlers, filters, factory = snapshot
    root = _logging.getLogger()
    root.handlers[:] = handlers
    root.filters[:] = filters
    _logging.setLogRecordFactory(factory)


def _enable_test_logger_at_info() -> tuple[Any, int]:
    """Bring the fixture-010 test logger up to ``INFO`` so YAML's
    ``level: INFO`` records actually flow through Python's logger-level
    filter to the bridge handler. Returns ``(logger, prior_level)`` to
    pair with a restore in ``finally``."""
    import logging as _logging  # noqa: PLC0415

    test_logger = _logging.getLogger("openarmature.test.fixture_010")
    prior_level = test_logger.level
    test_logger.setLevel(_logging.INFO)
    return test_logger, prior_level


async def _run_fixture_010(spec: Mapping[str, Any]) -> None:
    """Two sub-cases: nested-trace log correlation (single graph, all logs
    share the parent trace_id) and detached-subgraph log correlation
    (logs across the detached boundary carry distinct trace_ids but the
    same correlation_id)."""
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        case_name = cast("str", case["name"])
        try:
            await _run_fixture_010_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case_name!r}: {e}") from e


async def _run_fixture_010_case(case: Mapping[str, Any]) -> None:
    case_name = cast("str", case["name"])
    if case_name == "log_records_carry_trace_span_correlation_ids":
        await _run_fixture_010_nested_trace(case)
    elif case_name == "detached_subgraph_log_uses_detached_trace_id_keeps_correlation_id":
        await _run_fixture_010_detached(case)
    else:
        raise AssertionError(f"unknown fixture 010 sub-case: {case_name!r}")


async def _run_fixture_010_nested_trace(case: Mapping[str, Any]) -> None:
    """Sub-case 1: 2 nodes ``a`` → ``b``, both emit logs from the FIRST line
    of their body. The log bridge MUST report all logs in the parent
    trace_id, with each log's span_id matching the active node span at
    emission, and all carrying the invocation's correlation_id."""
    from openarmature.graph import END, GraphBuilder, State  # noqa: PLC0415

    nodes_spec = cast("dict[str, Any]", case["nodes"])
    correlation_id = cast("str", case["caller_correlation_id"])
    # Spec YAML is the single source of truth for the log bodies; derive
    # them up front rather than hard-coding so a fixture rename doesn't
    # silently break the driver's record filtering.
    node_emit_messages: dict[str, str] = {
        name: cast("str", cast("dict[str, Any]", nodes_spec[name])["emits_log"]["message"])
        for name in nodes_spec
    }

    class _S(State):
        x: int = 0

    test_logger, prior_level = _enable_test_logger_at_info()

    def _make_body(node_name: str) -> Any:
        spec = cast("dict[str, Any]", nodes_spec[node_name])
        emit_msg = cast("str", spec["emits_log"]["message"])
        update = cast("dict[str, Any]", spec["update_pure"])

        async def body(_s: _S) -> dict[str, Any]:
            # FIRST line, before any await — the load-bearing case
            # the engine attach via ``prepare_sync`` exists to cover.
            test_logger.info(emit_msg)
            return dict(update)

        return body

    builder = GraphBuilder(_S)
    for node_name in nodes_spec:
        builder.add_node(node_name, _make_body(node_name))
    for edge in cast("list[dict[str, Any]]", case["edges"]):
        from_node = cast("str", edge["from"])
        to = edge["to"]
        builder.add_edge(from_node, END if to == "END" else cast("str", to))
    builder.set_entry(cast("str", case["entry"]))
    compiled = builder.compile()

    observer, span_exporter = _build_observer()
    log_exporter, log_provider, snapshot = _setup_isolated_log_bridge()
    try:
        compiled.attach_observer(observer)
        await compiled.invoke(_S(), correlation_id=correlation_id)
        await compiled.drain()
        observer.shutdown()
        log_provider.force_flush()

        records = log_exporter.get_finished_logs()
        # Filter to OUR test loggers so concurrent test setup noise
        # doesn't contaminate the assertions. Expected message set
        # comes from the spec YAML, not hard-coded strings.
        expected_messages = set(node_emit_messages.values())
        ours = [r for r in records if str(r.log_record.body) in expected_messages]
        assert len(ours) == 2, (
            f"expected 2 log records (one per node body); got {len(ours)}: "
            f"{[str(r.log_record.body) for r in ours]}"
        )

        # Group by body for predictable lookup, indexing by the spec's
        # emit-message values.
        by_body = {str(r.log_record.body): r for r in ours}
        a_log = by_body[node_emit_messages["a"]]
        b_log = by_body[node_emit_messages["b"]]

        # Invariant: all_logs_same_trace_id.
        trace_ids = {a_log.log_record.trace_id, b_log.log_record.trace_id}
        assert len(trace_ids) == 1, f"all logs MUST share a trace_id (single nested trace); got {trace_ids}"

        # Invariant: log_span_ids_match_active_span_at_emission.
        spans = span_exporter.get_finished_spans()
        node_span_ids: dict[str, int] = {}
        for s in spans:
            if s.name in {"a", "b"}:
                node_span_ids[s.name] = s.context.span_id
        assert a_log.log_record.span_id == node_span_ids["a"], (
            f"node-a log MUST carry node-a span's span_id; "
            f"got log span_id={a_log.log_record.span_id}, span={node_span_ids['a']}"
        )
        assert b_log.log_record.span_id == node_span_ids["b"], (
            f"node-b log MUST carry node-b span's span_id; "
            f"got log span_id={b_log.log_record.span_id}, span={node_span_ids['b']}"
        )

        # Invariant: all_logs_carry_correlation_id.
        for r in ours:
            attrs = dict(r.log_record.attributes or {})
            assert attrs.get("openarmature.correlation_id") == correlation_id, (
                f"every log MUST carry openarmature.correlation_id={correlation_id!r}; "
                f"got {attrs.get('openarmature.correlation_id')!r}"
            )
    finally:
        test_logger.setLevel(prior_level)
        _restore_log_state(snapshot)


async def _run_fixture_010_detached(case: Mapping[str, Any]) -> None:
    """Sub-case 2: outer invocation has a detached subgraph. Logs emitted
    inside the detached subgraph carry the DETACHED trace's trace_id —
    NOT the parent's — while the correlation_id flows unchanged across
    the boundary."""
    from openarmature.graph import END, GraphBuilder, State  # noqa: PLC0415

    correlation_id = cast("str", case["caller_correlation_id"])
    sub_specs = cast("dict[str, Any]", case["subgraphs"])
    inner_spec = cast("dict[str, Any]", sub_specs["detached_inner"])
    outer_nodes = cast("dict[str, Any]", case["nodes"])

    # Detached subgraph identity → wrapper-node-name translation, same
    # convention as fixture 008. The fixture YAML lists subgraph identities
    # in ``detached_subgraphs:``; OTelObserver keys on the wrapper node's
    # name in the parent graph.
    detached_identities = set(cast("list[str]", case.get("detached_subgraphs") or []))
    wrapper_names: set[str] = set()
    for wrapper_name, node_spec in outer_nodes.items():
        sub_id = cast("dict[str, Any]", node_spec).get("subgraph")
        if isinstance(sub_id, str) and sub_id in detached_identities:
            wrapper_names.add(wrapper_name)
    detached_subgraphs = frozenset(wrapper_names)

    test_logger, prior_level = _enable_test_logger_at_info()

    # Inner subgraph (detached_inner): 1 node ``inner`` with
    # ``update_pure: {y: 1}`` + ``emits_log: "inside detached subgraph"``.
    class _Inner(State):
        y: int = 0

    inner_node_spec = cast("dict[str, Any]", inner_spec["nodes"]["inner"])
    inner_emit = cast("str", inner_node_spec["emits_log"]["message"])
    inner_update = cast("dict[str, Any]", inner_node_spec["update_pure"])

    async def _inner_body(_s: _Inner) -> dict[str, Any]:
        test_logger.info(inner_emit)
        return dict(inner_update)

    inner_compiled = (
        GraphBuilder(_Inner)
        .add_node("inner", _inner_body)
        .add_edge("inner", END)
        .set_entry("inner")
        .compile()
    )

    # Outer graph: ``outer_dispatch`` is a SubgraphNode wrapper around
    # ``inner_compiled`` AND emits a log "before subgraph dispatch".
    # SubgraphNode wrappers don't get ``prepare_sync`` per spec — the
    # outer log is emitted via per-node middleware that fires inside
    # the wrapper's chain. Without an attached span at wrapper scope,
    # the outer log's trace_id is OTel's "no active span" sentinel
    # (0); the inner log's trace_id is the detached trace's. The
    # invariant ``log_trace_ids_differ_when_detached`` holds either
    # way.
    class _Outer(State):
        z: int = 0

    outer_node_spec = cast("dict[str, Any]", outer_nodes["outer_dispatch"])
    outer_emit = cast("str", outer_node_spec["emits_log"]["message"])

    async def _outer_log_middleware(s: Any, next_call: Any) -> Mapping[str, Any]:
        test_logger.info(outer_emit)
        return cast("Mapping[str, Any]", await next_call(s))

    outer_compiled = (
        GraphBuilder(_Outer)
        .add_subgraph_node("outer_dispatch", inner_compiled, middleware=[_outer_log_middleware])
        .add_edge("outer_dispatch", END)
        .set_entry("outer_dispatch")
        .compile()
    )

    observer, _span_exporter = _build_observer_with_detached(detached_subgraphs)
    log_exporter, log_provider, snapshot = _setup_isolated_log_bridge()
    try:
        outer_compiled.attach_observer(observer)
        await outer_compiled.invoke(_Outer(), correlation_id=correlation_id)
        await outer_compiled.drain()
        observer.shutdown()
        log_provider.force_flush()

        records = log_exporter.get_finished_logs()
        ours = [r for r in records if str(r.log_record.body) in {outer_emit, inner_emit}]
        assert len(ours) == 2, (
            f"expected 2 log records (outer + inner); got {len(ours)}: "
            f"{[str(r.log_record.body) for r in ours]}"
        )

        by_body = {str(r.log_record.body): r for r in ours}
        outer_log = by_body[outer_emit]
        inner_log = by_body[inner_emit]

        # Invariant: log_trace_ids_differ_when_detached.
        assert outer_log.log_record.trace_id != inner_log.log_record.trace_id, (
            f"detached-subgraph log MUST carry the detached trace's trace_id, "
            f"DIFFERENT from the parent log; both got {outer_log.log_record.trace_id}"
        )

        # Invariant: all_logs_carry_correlation_id.
        for r in ours:
            attrs = dict(r.log_record.attributes or {})
            assert attrs.get("openarmature.correlation_id") == correlation_id, (
                f"every log MUST carry openarmature.correlation_id={correlation_id!r}; "
                f"got {attrs.get('openarmature.correlation_id')!r}"
            )
    finally:
        test_logger.setLevel(prior_level)
        _restore_log_state(snapshot)


def _build_observer_with_detached(detached_subgraphs: frozenset[str]) -> tuple[OTelObserver, Any]:
    """Variant of :func:`_build_observer` that takes a detached_subgraphs
    set — needed for fixture 010 sub-case 2."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    observer = OTelObserver(
        span_processor=SimpleSpanProcessor(exporter),
        detached_subgraphs=detached_subgraphs,
    )
    return observer, exporter


# ---------------------------------------------------------------------------
# v0.17.0 LLM-payload + GenAI-semconv fixtures (012-021)
# ---------------------------------------------------------------------------


async def _run_llm_payload_fixture(spec: Mapping[str, Any]) -> None:
    """Generic driver for the ten v0.17.0 LLM-attribute fixtures.

    Each fixture is single-case (GraphFixture shape) with a top-level
    ``cases:`` list of one entry; the case carries the graph + the
    ``calls_llm`` config + the optional observer/provider flags.
    """
    cases = cast("list[dict[str, Any]]", spec["cases"])
    for case in cases:
        try:
            await _run_llm_payload_case(case)
        except AssertionError as e:
            raise AssertionError(f"case {case.get('name')!r}: {e}") from e


async def _run_llm_payload_case(case: Mapping[str, Any]) -> None:
    """Build + invoke the graph, then walk the expected span tree
    asserting via the LLM-attribute helpers (parse-shape, truncation,
    redaction-substring-absence)."""
    import json

    import httpx
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    from openarmature.graph import END, GraphBuilder
    from openarmature.llm import OpenAIProvider
    from openarmature.llm.response import RuntimeConfig

    from .adapter import build_state_cls
    from .harness.llm_attribute_assertions import (
        assert_attribute_does_not_contain,
        assert_attribute_parses_as_messages,
        assert_attribute_parses_as_object,
        assert_attribute_truncation,
        assert_attributes_absent,
        record_synthesized_base64_prefix,
        reset_synthesized_base64_prefixes,
    )

    reset_synthesized_base64_prefixes()

    # ---- Resolve harness primitives (content_repeat, base64_data_synthetic)
    nodes_spec = cast("dict[str, Any]", case["nodes"])
    entry_name = cast("str", case["entry"])
    # Most LLM-payload fixtures are single-node (the entry IS the
    # calls_llm node); fixture 026 has a non-LLM ``prep`` step
    # before the LLM call. Find whichever node carries ``calls_llm``
    # and treat the others as plain ``update`` nodes.
    llm_node_name = next(
        (name for name, spec in nodes_spec.items() if isinstance(spec, dict) and "calls_llm" in spec),
        entry_name,
    )
    calls_llm_spec = cast("dict[str, Any]", nodes_spec[llm_node_name]["calls_llm"])
    raw_messages = cast("list[dict[str, Any]]", calls_llm_spec.get("messages", []))
    materialized_messages, full_input_serialization = _materialize_messages(
        raw_messages,
        record_base64_prefix=record_synthesized_base64_prefix,
    )

    # ---- RuntimeConfig from the calls_llm.config block
    config_spec = cast("dict[str, Any] | None", calls_llm_spec.get("config"))
    runtime_config: RuntimeConfig | None = None
    if config_spec:
        extras = cast("dict[str, Any]", config_spec.get("extras") or {})
        runtime_config_kwargs: dict[str, Any] = {
            k: v
            for k, v in config_spec.items()
            if k
            in {
                "temperature",
                "max_tokens",
                "top_p",
                "seed",
                "frequency_penalty",
                "presence_penalty",
                "stop_sequences",
            }
        }
        runtime_config_kwargs.update(extras)
        runtime_config = RuntimeConfig(**runtime_config_kwargs)

    # ---- Provider knobs (provider.genai_system override)
    provider_spec = cast("dict[str, Any] | None", case.get("provider"))
    genai_system = "openai"
    if provider_spec and isinstance(provider_spec.get("genai_system"), str):
        genai_system = cast("str", provider_spec["genai_system"])

    # ---- Mock LLM transport
    mock_responses = list(cast("list[dict[str, Any]]", case.get("mock_llm") or []))

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not mock_responses:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = mock_responses.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model="test-model",
        api_key="test",
        transport=httpx.MockTransport(_handler),
        genai_system=genai_system,
    )

    # ---- State + node body
    state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
    state_cls = build_state_cls("LlmPayloadFixtureState", state_fields)
    stores_in = cast("str", calls_llm_spec.get("stores_response_in", "msg"))

    async def ask_llm_body(_s: Any) -> dict[str, str]:
        response = await provider.complete(
            cast("Sequence[Any]", materialized_messages),
            config=runtime_config,
        )
        return {stores_in: response.message.content or ""}

    # Build the graph: the calls_llm node uses ``ask_llm_body``; any
    # other node carries an ``update:`` block translated to a simple
    # async function that returns it verbatim. Edges come from the
    # fixture's ``edges:`` list when present (multi-node case); the
    # single-node case falls back to ``entry → END``.
    builder = GraphBuilder(state_cls)
    for node_name, node_spec in nodes_spec.items():
        if node_name == llm_node_name:
            builder.add_node(node_name, ask_llm_body)
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        update_block = cast("dict[str, Any] | None", node_dict.get("update"))
        if update_block is None:
            raise AssertionError(
                f"non-LLM node {node_name!r} in LLM fixture has neither "
                f"`calls_llm` nor `update`; harness needs an extension"
            )

        def _make_update_body(payload: dict[str, Any]) -> Any:
            async def _body(_s: Any) -> dict[str, Any]:
                return dict(payload)

            return _body

        builder.add_node(node_name, _make_update_body(update_block))
    edges_spec = cast("list[dict[str, str]] | None", case.get("edges"))
    if edges_spec is None:
        builder.add_edge(llm_node_name, END)
    else:
        for edge in edges_spec:
            target_raw = edge["to"]
            target = END if target_raw == "END" else target_raw
            builder.add_edge(edge["from"], target)
    builder.set_entry(entry_name)
    graph = builder.compile()

    # ---- Observer
    exporter = InMemorySpanExporter()
    observer_kwargs: dict[str, Any] = {"span_processor": SimpleSpanProcessor(exporter)}
    if "disable_llm_payload" in case:
        observer_kwargs["disable_llm_payload"] = bool(case["disable_llm_payload"])
    if "disable_genai_semconv" in case:
        observer_kwargs["disable_genai_semconv"] = bool(case["disable_genai_semconv"])
    if "disable_llm_spans" in case:
        observer_kwargs["disable_llm_spans"] = bool(case["disable_llm_spans"])
    observer = OTelObserver(**observer_kwargs)
    graph.attach_observer(observer)

    # ---- Run + collect spans
    initial_state_cls = graph.state_cls
    invoke_kwargs: dict[str, Any] = {}
    caller_metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
    if caller_metadata is not None:
        invoke_kwargs["metadata"] = caller_metadata
    await graph.invoke(initial_state_cls(), **invoke_kwargs)
    await graph.drain()
    observer.shutdown()
    spans = exporter.get_finished_spans()

    # ---- Walk expected.span_tree and check per-span assertions
    expected = cast("dict[str, Any]", case["expected"])
    expected_tree = cast("list[dict[str, Any]]", expected.get("span_tree") or [])
    _check_payload_span_tree(
        spans,
        expected_tree,
        full_input_serialization=full_input_serialization,
        assert_attributes_absent=assert_attributes_absent,
        assert_attribute_parses_as_messages=assert_attribute_parses_as_messages,
        assert_attribute_parses_as_object=assert_attribute_parses_as_object,
        assert_attribute_does_not_contain=assert_attribute_does_not_contain,
        assert_attribute_truncation=assert_attribute_truncation,
    )


def _materialize_messages(
    raw_messages: list[dict[str, Any]],
    *,
    record_base64_prefix: Any,
) -> tuple[list[Any], str | None]:
    """Resolve harness directives (``content_repeat``,
    ``base64_data_synthetic``) into real ``Message`` instances.

    Returns the message list AND the canonical full-serialization
    string for the materialized payload — the truncation fixture
    needs the latter for its ``prefix_of_full_serialization`` check.
    """
    from openarmature.llm.messages import UserMessage

    out: list[Any] = []
    full_serial_target: str | None = None
    for msg in raw_messages:
        role = msg.get("role")
        # ``content_repeat`` may live at the message level (fixture 014:
        # ``{role: user, content_repeat: {char, bytes}}``) — no ``content``
        # key in that case; synthesize a string of N repeated chars.
        content: Any
        if "content_repeat" in msg:
            repeat = cast("dict[str, Any]", msg["content_repeat"])
            content = cast("str", repeat["char"]) * int(repeat["bytes"])
        else:
            content = msg.get("content")
        if role == "user":
            materialized = _materialize_user_content(
                content,
                record_base64_prefix=record_base64_prefix,
            )
            out.append(UserMessage(content=materialized))
        elif role == "system":
            from openarmature.llm.messages import SystemMessage

            out.append(SystemMessage(content=cast("str", content)))
        else:
            raise AssertionError(f"unsupported role in payload fixture: {role!r}")

    # Compute the full serialization (what the observer would emit
    # before truncation). The provider's _serialize_messages_for_payload
    # is the canonical encoder; mirror its shape via the same import.
    from openarmature.llm.providers.openai import _serialize_messages_for_payload

    plain = _serialize_messages_for_payload(out)
    import json

    full_serial_target = json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return out, full_serial_target


def _materialize_user_content(content: Any, *, record_base64_prefix: Any) -> Any:
    """Resolve the user message's content. Strings pass through; lists
    of blocks materialize the harness directives in each block.

    ``content_repeat: {char, bytes}`` on a string-only message synthesizes
    a repeated-character string of N bytes. ``base64_data_synthetic:
    {bytes}`` on an inline image source synthesizes a deterministic
    base64 blob; the prefix is recorded via the supplied callable so
    the ``attribute_does_not_contain`` assertion can verify absence.
    """
    from openarmature.llm.messages import (
        ImageBlock,
        ImageSourceInline,
        ImageSourceURL,
        TextBlock,
    )

    # Compact form: ``content`` is a dict with ``content_repeat`` —
    # synthesize a string of N repeated chars.
    if isinstance(content, dict) and "content_repeat" in content:
        repeat = cast("dict[str, Any]", content["content_repeat"])
        char = cast("str", repeat["char"])
        nbytes = int(repeat["bytes"])
        return char * nbytes
    if isinstance(content, str):
        return content
    # List of content blocks.
    blocks: list[Any] = []
    for block in cast("list[dict[str, Any]]", content):
        btype = block.get("type")
        if btype == "text":
            blocks.append(TextBlock(text=cast("str", block["text"])))
        elif btype == "image":
            source_spec = cast("dict[str, Any]", block["source"])
            stype = source_spec.get("type")
            if stype == "inline":
                synth = cast("dict[str, Any] | None", source_spec.get("base64_data_synthetic"))
                if synth is not None:
                    nbytes = int(synth["bytes"])
                    blob = _synth_base64(nbytes)
                    record_base64_prefix(blob)
                    source = ImageSourceInline(base64_data=blob)
                else:
                    source = ImageSourceInline(base64_data=cast("str", source_spec["base64_data"]))
            elif stype == "url":
                source = ImageSourceURL(url=cast("str", source_spec["url"]))
            else:
                raise AssertionError(f"unsupported image source type: {stype!r}")
            blocks.append(
                ImageBlock(
                    source=source,
                    media_type=cast("str | None", block.get("media_type")),
                    detail=cast("Any", block.get("detail")),
                )
            )
        else:
            raise AssertionError(f"unsupported content block type: {btype!r}")
    # Compact form: a single ``content_repeat`` entry inside a list.
    return blocks


def _synth_base64(nbytes: int) -> str:
    """Synthesize a deterministic base64 blob of exactly ``nbytes`` bytes.

    Fixture 015 uses 4096 bytes; deterministic so the synthesized prefix
    can be recorded once and the ``attribute_does_not_contain`` helper
    verifies the same prefix is absent from the redacted attribute.
    """
    # Repeated-letter base64 — valid base64 chars, deterministic, length
    # exactly nbytes.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    # Use a single character so the prefix-check signal is strong; the
    # bytes are not a real PNG (the redaction rule is about SHAPE).
    return alphabet[0] * nbytes


def _check_payload_span_tree(
    spans: Any,
    expected_tree: list[dict[str, Any]],
    *,
    full_input_serialization: str | None,
    assert_attributes_absent: Any,
    assert_attribute_parses_as_messages: Any,
    assert_attribute_parses_as_object: Any,
    assert_attribute_does_not_contain: Any,
    assert_attribute_truncation: Any,
) -> None:
    """Walk ``expected_tree`` and verify each expected span's attribute
    block matches the spans in ``spans``."""
    spans_by_name: dict[str, list[Any]] = {}
    for s in spans:
        spans_by_name.setdefault(s.name, []).append(s)

    def _walk(expected_entries: list[dict[str, Any]]) -> None:
        for entry in expected_entries:
            name = cast("str", entry["name"])
            candidates = spans_by_name.get(name, [])
            assert candidates, f"expected a span named {name!r}; got {sorted(spans_by_name.keys())}"
            # The fixtures we cover have unique span names in each tree.
            span = candidates[0]
            attrs = dict(span.attributes or {})
            # ``attributes:`` block — exact match per key.
            for k, v in cast("dict[str, Any]", entry.get("attributes") or {}).items():
                actual: Any = attrs.get(k)
                # OTel attribute arrays come back as tuples; normalize.
                if isinstance(v, list) and isinstance(actual, tuple):
                    actual = list(cast("tuple[Any, ...]", actual))
                assert actual == v, f"span {name!r} attribute {k!r} mismatch: expected {v!r}, got {actual!r}"
            # ``attributes_absent:`` list of names that MUST NOT appear.
            absent = entry.get("attributes_absent")
            if absent:
                assert_attributes_absent(attrs, cast("list[str]", absent))
            # ``attribute_parses_as_messages:`` shape assertion.
            parses_as_messages = entry.get("attribute_parses_as_messages")
            if parses_as_messages:
                assert_attribute_parses_as_messages(attrs, cast("dict[str, Any]", parses_as_messages))
            # ``attribute_parses_as_object:`` shape assertion.
            parses_as_object = entry.get("attribute_parses_as_object")
            if parses_as_object:
                assert_attribute_parses_as_object(attrs, cast("dict[str, Any]", parses_as_object))
            # ``attribute_does_not_contain:`` substring absence.
            does_not_contain = entry.get("attribute_does_not_contain")
            if does_not_contain:
                assert_attribute_does_not_contain(attrs, cast("dict[str, Any]", does_not_contain))
            # ``attribute_truncation:`` §5.5.5 contract.
            truncation = entry.get("attribute_truncation")
            if truncation:
                full_map: dict[str, str] = {}
                # The fixture is single-attribute; supply the full
                # serialization under the same key for the
                # prefix_of_full_serialization clause.
                if full_input_serialization is not None:
                    for attr_name in cast("dict[str, Any]", truncation):
                        full_map[attr_name] = full_input_serialization
                assert_attribute_truncation(attrs, cast("dict[str, Any]", truncation), full_map)
            # Recurse into children.
            children = cast("list[dict[str, Any]] | None", entry.get("children"))
            if children:
                _walk(children)

    _walk(expected_tree)
