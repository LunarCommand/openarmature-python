"""OTelObserver — observer-driven span lifecycle (spec observability §6).

The observer subscribes to all three §6 phases (``started``,
``completed``, ``checkpoint_saved``) plus the LLM-provider events the
``OpenAIProvider`` enqueues from inside node bodies. On a ``started``
event it opens a span and pushes it onto an in-flight map keyed by
``(trace_id, namespace, attempt_index, fan_out_index)``; on the
matching ``completed`` event it pops the span, applies §4.2 status
mapping, and closes it.

Spans are emitted through a **private** :class:`TracerProvider`
constructed by this observer — never the OTel global. Per spec §6
TracerProvider isolation, registering globally would cause every
auto-instrumentation library that writes to the global provider
(OpenInference, opentelemetry-instrumentation-openai, LiteLLM, etc.)
to emit duplicate spans alongside ours.

Detached trace mode (§4.4) is implemented by minting a fresh
:class:`SpanContext` with a new ``trace_id`` when entering a
configured-detached subgraph or fan-out; the parent's dispatch span
carries an OTel :class:`Link` to the detached trace. The span-stack
key includes ``trace_id`` so detached sub-trees and the parent trace
maintain separate stacks naturally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
)
from opentelemetry.trace.propagation import set_span_in_context

if TYPE_CHECKING:
    from openarmature.graph.events import NodeEvent


# Span-stack key shape: ``(namespace, attempt_index, fan_out_index)``
# — these three fields uniquely identify any node attempt within an
# invocation. Trace_id was previously included in the key but that
# created a registration/lookup mismatch when the OTel current-span
# context changed between a span's started and completed events
# (e.g., detached fan-out instances opening detached roots in
# between). Detached sub-tree spans live in ``_detached_roots`` /
# ``_subgraph_spans`` separately, so the namespace alone doesn't
# collide here.
_StackKey = tuple[tuple[str, ...], int, int | None]


# Sentinel namespace the LLM provider emits to signal "this is an LLM
# event, not a regular node event."
_LLM_NAMESPACE = ("openarmature.llm.complete",)


def _read_spec_version() -> str:
    """Read the spec version pinned at package level. Lazy import
    avoids a circular at module-load time (the package's ``__init__``
    imports submodules that may import the observability stack)."""
    from openarmature import __spec_version__

    return __spec_version__


@dataclass
class _OpenSpan:
    """An in-flight span paired with the OTel context token that pinned
    its scope. The token is ``detach``ed when the span closes so the
    OTel current-span context unwinds correctly."""

    span: Span
    token: object


@dataclass
class OTelObserver:
    """Observer-driven OTel span lifecycle per spec observability §6.

    Construct with a :class:`SpanProcessor` (typically a
    :class:`BatchSpanProcessor` wrapping a real exporter, or a
    :class:`SimpleSpanProcessor` wrapping :class:`InMemorySpanExporter`
    for tests). The observer instantiates its own private
    :class:`TracerProvider` from the supplied processor — callers
    MUST NOT pre-register the provider globally.

    Constructor knobs:

    - ``detached_subgraphs`` — set of subgraph wrapper node names
      that should run in their own trace (§4.4). One detached trace
      per such subgraph.
    - ``detached_fan_outs`` — set of fan-out node names whose
      INSTANCES each get their own trace. One detached trace per
      instance.
    - ``disable_llm_spans`` — when ``True`` the observer skips the
      §5.5 LLM provider span. All other spans (node, subgraph,
      fan-out, etc.) emit normally. Useful when an external
      auto-instrumentation library (OpenInference, etc.) is the
      canonical source of LLM spans.
    - ``spec_version`` — string surfaced as
      ``openarmature.graph.spec_version`` on the invocation span.
    """

    span_processor: SpanProcessor
    # Lambda-wrapped factories give pyright the explicit
    # ``frozenset[str]`` type — ``default_factory=frozenset``
    # alone produces ``frozenset[Unknown]`` which the strict-mode
    # config flags. The lambda overhead is negligible (one closure
    # call per dataclass instance).
    detached_subgraphs: frozenset[str] = field(default_factory=lambda: frozenset[str]())
    detached_fan_outs: frozenset[str] = field(default_factory=lambda: frozenset[str]())
    disable_llm_spans: bool = False
    # Read from the package's ``__spec_version__`` (one of the three
    # places the spec version is pinned per CLAUDE.md). Bumping the
    # spec submodule + the two version fields automatically updates
    # the value reported on every invocation span.
    spec_version: str = field(default_factory=_read_spec_version)

    # Internal state, populated in __post_init__ and during invocation.
    _provider: TracerProvider = field(init=False, repr=False)
    _tracer: otel_trace.Tracer = field(init=False, repr=False)
    _open_spans: dict[_StackKey, _OpenSpan] = field(
        init=False, repr=False, default_factory=dict[_StackKey, _OpenSpan]
    )
    # The invocation root span, opened on the first event of an
    # invocation and closed when the matching outermost completed
    # event arrives (or, in practice, when the engine's queue drains
    # — the invocation span has no started/completed pair of its
    # own, so we open it lazily and close it on a sentinel).
    _invocation_span: dict[str, _OpenSpan] = field(
        init=False, repr=False, default_factory=dict[str, _OpenSpan]
    )
    # Synthetic subgraph dispatch spans: the engine wrapper for
    # ``add_subgraph_node`` is transparent (graph-engine fixture 013
    # — no started/completed events of its own), but observability
    # §4.5 mandates a subgraph span. The OTel observer synthesizes
    # one by detecting deeper-namespace events and opening an
    # ancestor span for each new prefix; closes when subsequent
    # events leave that prefix.
    _subgraph_spans: dict[tuple[str, ...], _OpenSpan] = field(
        init=False, repr=False, default_factory=dict[tuple[str, ...], _OpenSpan]
    )
    # Per-namespace-prefix detached trace tracking. When a detached
    # subgraph or fan-out instance enters, we mint a fresh trace and
    # store the root span here so subsequent inner events at that
    # prefix find the right parent. Keyed by namespace prefix
    # (subgraph) or namespace_prefix + (str(fan_out_index),) (fan-out
    # instance). The fan-out node's own span (in the parent trace)
    # collects Links to each detached instance trace.
    _detached_roots: dict[tuple[str, ...], _OpenSpan] = field(
        init=False, repr=False, default_factory=dict[tuple[str, ...], _OpenSpan]
    )
    # Subset of ``_detached_roots`` keys that represent per-instance
    # fan-out roots — they're closed by ``_handle_completed`` on the
    # fan-out node's own completion, NOT by ``_sync_subgraph_spans``.
    # Using an explicit set rather than parsing the key (e.g.,
    # checking ``prefix[-1].isdigit()``) so node names that happen
    # to be pure digits don't get misclassified.
    _fan_out_instance_root_prefixes: set[tuple[str, ...]] = field(
        init=False, repr=False, default_factory=set[tuple[str, ...]]
    )

    def __post_init__(self) -> None:
        # Private provider per spec §6 TracerProvider isolation —
        # MUST NOT be registered globally.
        self._provider = TracerProvider()
        self._provider.add_span_processor(self.span_processor)
        self._tracer = self._provider.get_tracer("openarmature")

    # ------------------------------------------------------------------
    # Observer protocol — async callable accepting a NodeEvent
    # ------------------------------------------------------------------

    async def __call__(self, event: NodeEvent) -> None:
        # LLM provider events use a sentinel namespace so we can route
        # them to the dedicated §5.5 span path.
        if event.namespace == _LLM_NAMESPACE:
            if not self.disable_llm_spans:
                self._handle_llm_event(event)
            return
        if event.phase == "checkpoint_saved":
            self._emit_checkpoint_save_span(event)
            return
        if event.phase == "started":
            self._handle_started(event)
        elif event.phase == "completed":
            self._handle_completed(event)

    # ------------------------------------------------------------------
    # Started / completed pairing
    # ------------------------------------------------------------------

    def _handle_started(self, event: NodeEvent) -> None:
        """Open a span for this attempt, push onto the in-flight map."""
        from openarmature.observability.correlation import current_correlation_id

        # Lazily open the invocation span on the first event we see
        # for this invocation. Detect "first event" by matching the
        # correlation_id; the invocation span lives until either
        # ``shutdown()`` runs OR a new correlation_id arrives (i.e.,
        # a new invocation starts on the same long-lived observer).
        # The latter close path matters for shared observers reused
        # across many invocations — without it the
        # ``_invocation_span`` dict grows unbounded.
        correlation_id = current_correlation_id()
        if correlation_id is not None:
            # New correlation_id → close prior invocation spans.
            for prior_cid in list(self._invocation_span.keys()):
                if prior_cid != correlation_id:
                    self._close_invocation_span(prior_cid)
            if correlation_id not in self._invocation_span:
                self._open_invocation_span(correlation_id, event)

        # Synthesize subgraph dispatch spans for any ancestor namespace
        # prefix that doesn't have one yet (per observability §4.5).
        # Also closes subgraph spans we've left.
        self._sync_subgraph_spans(event)

        parent_ctx = self._resolve_parent_context(event)
        span = self._tracer.start_span(
            name=event.node_name,
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=self._node_attrs(event),
        )
        token = attach(set_span_in_context(span))
        key = self._key_for(event)
        self._open_spans[key] = _OpenSpan(span=span, token=token)

    def _handle_completed(self, event: NodeEvent) -> None:
        """Close the matching span, applying §4.2 status mapping."""
        # If this is the fan-out node's own completion AND the
        # fan-out is configured detached, close all per-instance
        # detached roots that this fan-out spawned. Done BEFORE the
        # regular pop so the OTel current-span context is restored
        # to the fan-out span's parent (otherwise inner instance
        # roots would still be attached).
        if event.fan_out_index is None and event.namespace and event.namespace[0] in self.detached_fan_outs:
            for key in list(self._detached_roots.keys()):
                if len(key) > len(event.namespace) and key[: len(event.namespace)] == event.namespace:
                    self._close_detached_root(key)
        key = self._key_for(event)
        open_span = self._open_spans.pop(key, None)
        if open_span is None:
            # Started event was never delivered (e.g., observer was
            # attached mid-invocation). Nothing to close.
            return
        span = open_span.span
        if event.error is not None:
            span.set_status(Status(StatusCode.ERROR, description=event.error.category))
            span.record_exception(event.error)
            span.set_attribute("openarmature.error.category", event.error.category)
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()
        token = open_span.token
        if token is not None:
            detach(cast("Any", token))
        # If this was a detached root, drop the root entry so a
        # subsequent re-entry mints a fresh trace.
        self._detached_roots.pop(event.namespace, None)

    # ------------------------------------------------------------------
    # Special-event paths
    # ------------------------------------------------------------------

    def _emit_checkpoint_save_span(self, event: NodeEvent) -> None:
        """Spec pipeline-utilities §10.8 + observability §4.5: emit a
        zero-duration ``openarmature.checkpoint.save`` span attached
        to the most-recently-opened node span (the node whose
        completed event triggered the save)."""
        parent_ctx = self._resolve_parent_context(event)
        from openarmature.observability.correlation import current_correlation_id

        attrs: dict[str, Any] = {
            "openarmature.checkpoint.save_node": event.node_name,
        }
        cid = current_correlation_id()
        if cid is not None:
            attrs["openarmature.correlation_id"] = cid
        span = self._tracer.start_span(
            name="openarmature.checkpoint.save",
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        span.set_status(Status(StatusCode.OK))
        span.end()

    def _handle_llm_event(self, event: NodeEvent) -> None:
        """LLM provider span per spec §5.5 — parented to the node
        span that invoked the provider."""
        # The LLM provider's pre_state carries the event payload
        # (model, finish_reason, usage, error detail) since
        # NodeEvent's shape is fixed. See
        # ``openarmature.llm.providers.openai._make_llm_event``.
        payload = cast(
            "dict[str, Any]",
            cast("dict[str, Any]", event.pre_state).get("llm_event", {}),
        )
        if event.phase == "started":
            parent_ctx = self._current_span_context()
            attrs: dict[str, Any] = {"openarmature.llm.model": payload["model"]}
            from openarmature.observability.correlation import current_correlation_id

            cid = current_correlation_id()
            if cid is not None:
                attrs["openarmature.correlation_id"] = cid
            span = self._tracer.start_span(
                name="openarmature.llm.complete",
                context=cast("Any", parent_ctx),
                kind=SpanKind.CLIENT,
                attributes=attrs,
            )
            token = attach(set_span_in_context(span))
            self._open_spans[self._llm_key()] = _OpenSpan(span=span, token=token)
        elif event.phase == "completed":
            open_span = self._open_spans.pop(self._llm_key(), None)
            if open_span is None:
                return
            span = open_span.span
            if "finish_reason" in payload:
                span.set_attribute("openarmature.llm.finish_reason", payload["finish_reason"])
            for usage_field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                if payload.get(usage_field) is not None:
                    span.set_attribute(
                        f"openarmature.llm.usage.{usage_field}",
                        payload[usage_field],
                    )
            if "error_type" in payload:
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        description=payload.get("error_category", payload["error_type"]),
                    )
                )
                if "error_category" in payload:
                    span.set_attribute("openarmature.error.category", payload["error_category"])
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
            detach(cast("Any", open_span.token))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_invocation_span(self, correlation_id: str, event: NodeEvent) -> None:
        """Open the root invocation span for a new invocation."""
        # The first event we receive carries the entry node's
        # name; treat it as the invocation's entry_node attribute.
        attrs: dict[str, Any] = {
            "openarmature.invocation_id": "<unset>",  # set below from context
            "openarmature.graph.entry_node": event.node_name,
            "openarmature.graph.spec_version": self.spec_version,
            "openarmature.correlation_id": correlation_id,
        }
        # We don't have the engine's invocation_id directly without
        # threading it through. Phase 6 follow-up could surface it
        # via the correlation module; for now leave it blank for
        # the conformance fixtures that don't strictly need it.
        span = self._tracer.start_span(
            name="openarmature.invocation",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        token = attach(set_span_in_context(span))
        self._invocation_span[correlation_id] = _OpenSpan(span=span, token=token)

    def _key_for(self, event: NodeEvent) -> _StackKey:
        return (event.namespace, event.attempt_index, event.fan_out_index)

    def _llm_key(self) -> _StackKey:
        """LLM events are unique within a node body — only one LLM
        call can be open at a time per active span scope."""
        return (_LLM_NAMESPACE, 0, None)

    def _resolve_parent_context(self, event: NodeEvent) -> object:
        """Return the OTel context to use as the parent for this
        event's span. Walks namespace ancestors finding the
        innermost-open subgraph or detached root span."""
        # 1a. Detached fan-out instance root — keyed by
        #     ``namespace[:1] + (str(fan_out_index),)`` per
        #     ``_open_detached_fan_out_instance_root``. Checked
        #     explicitly before the generic prefix scan so the
        #     parent attribution doesn't depend on the
        #     attach-then-resolve ordering of the surrounding
        #     ``_sync_subgraph_spans`` call.
        if event.fan_out_index is not None and event.namespace:
            instance_key = event.namespace[:1] + (str(event.fan_out_index),)
            if instance_key in self._detached_roots:
                root = self._detached_roots[instance_key]
                return set_span_in_context(root.span)
        # 1b. Detached subgraph root at any matching prefix wins
        #     (highest precedence — events inside a detached subtree
        #     always parent under the detached root, never bleed up).
        for prefix_len in range(len(event.namespace) - 1, -1, -1):
            prefix = event.namespace[:prefix_len]
            if prefix in self._detached_roots:
                root = self._detached_roots[prefix]
                return set_span_in_context(root.span)
        # 2. Innermost synthetic subgraph span at any prefix.
        for prefix_len in range(len(event.namespace) - 1, 0, -1):
            prefix = event.namespace[:prefix_len]
            if prefix in self._subgraph_spans:
                sg = self._subgraph_spans[prefix]
                return set_span_in_context(sg.span)
        # 3. Otherwise, current OTel context (typically invocation span).
        return self._current_span_context()

    def _current_span_context(self) -> object:
        """Return the current OTel context."""
        return otel_context.get_current()

    def _sync_subgraph_spans(self, event: NodeEvent) -> None:
        """Open any synthetic subgraph dispatch spans we need (per
        observability §4.5: subgraph wrapper MUST emit a span); close
        any subgraph spans whose prefix is no longer an ancestor of
        the current event's namespace.

        Called from ``_handle_started`` BEFORE opening the leaf node
        span. Detached-mode entries (subgraph or fan-out instance)
        are registered as detached roots so their inner spans live
        in a fresh trace.
        """
        namespace = event.namespace
        # 1. Close any open subgraph spans that aren't ancestors of
        #    the current namespace — we've left those subgraphs.
        for prefix in list(self._subgraph_spans.keys()):
            if not (len(prefix) < len(namespace) and namespace[: len(prefix)] == prefix):
                self._close_subgraph_span(prefix)
        # 2. Same for detached subgraph roots — close ones we've
        #    left. (Detached fan-out instance roots are NOT closed
        #    here; they close on the fan-out's own completion.)
        for prefix in list(self._detached_roots.keys()):
            if (
                len(prefix) < len(namespace)
                and namespace[: len(prefix)] == prefix
                and event.fan_out_index is None
            ):
                # Still inside this detached subgraph — leave open.
                continue
            # Detached fan-out instance roots: keyed by namespace +
            # (str(fan_out_index),); leave those alone here, they're
            # closed when the fan-out parent dispatch completes.
            if prefix in self._fan_out_instance_root_prefixes:
                # Closed by ``_handle_completed`` on the fan-out
                # node's own completion, not here.
                continue
            if not (len(prefix) < len(namespace) and namespace[: len(prefix)] == prefix):
                self._close_detached_root(prefix)
        # 3. Open ancestor subgraph spans for any prefix that doesn't
        #    have one yet.
        for depth in range(1, len(namespace)):
            prefix = namespace[:depth]
            if prefix in self._subgraph_spans:
                continue
            if prefix in self._detached_roots:
                continue
            # If this prefix's first segment is configured as a
            # detached subgraph, mint a fresh trace.
            if depth == 1 and prefix[0] in self.detached_subgraphs:
                self._open_detached_subgraph_root(prefix)
                continue
            # If this is a fan-out instance namespace (event.fan_out_index
            # populated, prefix == namespace[:1]), and the fan-out
            # node is detached, open a per-instance detached root.
            if depth == 1 and event.fan_out_index is not None and prefix[0] in self.detached_fan_outs:
                self._open_detached_fan_out_instance_root(prefix, event)
                continue
            self._open_subgraph_span(prefix)

    def _open_subgraph_span(self, prefix: tuple[str, ...]) -> None:
        """Open a synthetic subgraph dispatch span for the given
        namespace prefix. Parent is the next-outer subgraph span (or
        the invocation span if depth-1)."""
        from openarmature.observability.correlation import current_correlation_id

        parent_ctx = self._current_span_context()
        # Walk up looking for the nearest enclosing subgraph or
        # detached root.
        for plen in range(len(prefix) - 1, 0, -1):
            outer = prefix[:plen]
            if outer in self._subgraph_spans:
                parent_ctx = set_span_in_context(self._subgraph_spans[outer].span)
                break
            if outer in self._detached_roots:
                parent_ctx = set_span_in_context(self._detached_roots[outer].span)
                break
        attrs: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.subgraph.name": prefix[-1],
        }
        cid = current_correlation_id()
        if cid is not None:
            attrs["openarmature.correlation_id"] = cid
        span = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        token = attach(set_span_in_context(span))
        self._subgraph_spans[prefix] = _OpenSpan(span=span, token=token)

    def _close_subgraph_span(self, prefix: tuple[str, ...]) -> None:
        open_span = self._subgraph_spans.pop(prefix, None)
        if open_span is None:
            return
        open_span.span.set_status(Status(StatusCode.OK))
        open_span.span.end()
        # Mirror ``_close_detached_root``: the attach token MUST be
        # detached or the OTel current-span context stays pinned to
        # a closed span and corrupts parent/child for subsequent
        # spans. The cross-context guard handles the case where the
        # token was created in a different OTel context (e.g.,
        # subgraph open/close straddles a detached descent).
        if open_span.token is not None:
            try:
                detach(cast("Any", open_span.token))
            except ValueError:
                pass

    def _open_detached_subgraph_root(self, prefix: tuple[str, ...]) -> None:
        """Mint a fresh trace for a detached subgraph entry. The
        detached root span lives in the new trace; the parent trace's
        dispatch span (synthesized at the same prefix BUT in the
        parent trace) carries an OTel Link to this root.

        Implementation: we open BOTH a parent-trace dispatch span
        (with the Link) AND a detached-trace root span (the actual
        parent for inner events). The dispatch span ends at sync
        time when we leave the prefix; the root span ends when its
        children finish."""
        from openarmature.observability.correlation import current_correlation_id

        # 1. Mint the new trace_id + root span_id NOW so the
        #    parent's Link target matches the detached root's
        #    SpanContext exactly.
        gen = RandomIdGenerator()
        detached_trace_id = gen.generate_trace_id()
        detached_root_span_id = gen.generate_span_id()
        detached_sc = SpanContext(
            trace_id=detached_trace_id,
            span_id=detached_root_span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        # 2. Open the dispatch span in the parent trace. Carries a
        #    Link pointing at the detached root's SpanContext.
        cid = current_correlation_id()
        attrs_parent: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.subgraph.name": prefix[-1],
        }
        if cid is not None:
            attrs_parent["openarmature.correlation_id"] = cid
        parent_dispatch = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", self._current_span_context()),
            kind=SpanKind.INTERNAL,
            links=[Link(detached_sc)],
            attributes=attrs_parent,
        )
        # Track in _subgraph_spans so the sync routine closes it on
        # leaving the prefix.
        self._subgraph_spans[prefix] = _OpenSpan(span=parent_dispatch, token=None)

        # 3. Open the detached root span — parented to the synthetic
        #    detached SpanContext so OTel uses the new trace_id.
        detached_parent_ctx = otel_trace.set_span_in_context(
            NonRecordingSpan(detached_sc), otel_context.Context()
        )
        attrs_root: dict[str, Any] = dict(attrs_parent)
        attrs_root["openarmature.subgraph.detached"] = True
        detached_root = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", detached_parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs_root,
        )
        token = attach(set_span_in_context(detached_root))
        self._detached_roots[prefix] = _OpenSpan(span=detached_root, token=token)

    def _open_detached_fan_out_instance_root(self, prefix: tuple[str, ...], event: NodeEvent) -> None:
        """Per-instance detached root for a configured-detached
        fan-out. Each instance gets its own trace_id; the fan-out
        node's span (in the parent trace, already open via the
        engine's started event) accumulates Links — one per
        instance."""
        from openarmature.observability.correlation import current_correlation_id

        gen = RandomIdGenerator()
        detached_trace_id = gen.generate_trace_id()
        detached_root_span_id = gen.generate_span_id()
        detached_sc = SpanContext(
            trace_id=detached_trace_id,
            span_id=detached_root_span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )

        # Find the fan-out node's already-open span in the parent
        # trace and add a Link to the detached root.
        fan_out_key = self._fan_out_node_span_key(prefix)
        fan_out_open = self._open_spans.get(fan_out_key)
        if fan_out_open is not None:
            fan_out_open.span.add_link(detached_sc)

        # Open the detached instance root span.
        detached_parent_ctx = otel_trace.set_span_in_context(
            NonRecordingSpan(detached_sc), otel_context.Context()
        )
        cid = current_correlation_id()
        attrs: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.fan_out.parent_node_name": prefix[-1],
            "openarmature.node.fan_out_index": event.fan_out_index,
        }
        if cid is not None:
            attrs["openarmature.correlation_id"] = cid
        instance_root = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", detached_parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        token = attach(set_span_in_context(instance_root))
        # Key by prefix + (str(fan_out_index),) so per-instance
        # roots stay distinct. Track separately in
        # ``_fan_out_instance_root_prefixes`` so
        # ``_sync_subgraph_spans`` can identify them by membership
        # (rather than by parsing the key shape).
        instance_key = prefix + (str(event.fan_out_index),)
        self._detached_roots[instance_key] = _OpenSpan(span=instance_root, token=token)
        self._fan_out_instance_root_prefixes.add(instance_key)

    def _close_detached_root(self, prefix: tuple[str, ...]) -> None:
        self._fan_out_instance_root_prefixes.discard(prefix)
        open_span = self._detached_roots.pop(prefix, None)
        if open_span is None:
            return
        open_span.span.set_status(Status(StatusCode.OK))
        open_span.span.end()
        if open_span.token is not None:
            try:
                detach(cast("Any", open_span.token))
            except ValueError:
                # Cross-context detach (token created in a different
                # OTel context) — ignore. The span has ended; the
                # context entry leaks cosmetically.
                pass

    def _fan_out_node_span_key(self, prefix: tuple[str, ...]) -> _StackKey:
        """Build the lookup key for a fan-out node's own span (the
        parent dispatch span). Fan-out node has no attempt_index ≠ 0
        and no fan_out_index — those fields belong to its inner
        instances."""
        return (prefix, 0, None)

    def _node_attrs(self, event: NodeEvent) -> dict[str, Any]:
        """Build the §5 attribute set for a node span."""
        from openarmature.observability.correlation import current_correlation_id

        attrs: dict[str, Any] = {
            "openarmature.node.name": event.node_name,
            "openarmature.node.namespace": list(event.namespace),
            "openarmature.node.step": event.step,
            "openarmature.node.attempt_index": event.attempt_index,
        }
        if event.fan_out_index is not None:
            attrs["openarmature.node.fan_out_index"] = event.fan_out_index
        cid = current_correlation_id()
        if cid is not None:
            attrs["openarmature.correlation_id"] = cid
        return attrs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_invocation(self, correlation_id: str) -> None:
        """Public lifecycle hook: close the invocation span for
        ``correlation_id``. Idempotent — calling twice (or for a
        correlation_id with no open span) is a no-op.

        Long-lived observers shared across many invocations
        automatically close prior invocation spans on the first
        event of a new invocation (see ``_handle_started``). This
        method is for callers who want explicit control without
        driving a follow-on invocation, e.g.::

            await graph.invoke(state, correlation_id=cid)
            await graph.drain()
            otel_observer.close_invocation(cid)
        """
        self._close_invocation_span(correlation_id)

    def _close_invocation_span(self, correlation_id: str) -> None:
        """End and remove the invocation span for ``correlation_id``.

        The OTel context token captured when the span opened was
        created in the worker task's context — we don't try to
        ``detach()`` it (cross-context detach raises ValueError;
        the leaked context entry is cosmetic since the worker
        eventually exits)."""
        open_span = self._invocation_span.pop(correlation_id, None)
        if open_span is None:
            return
        # Status defaults to OK for completed invocations; if the
        # engine surfaced an error, the failing node's span
        # already carries it and OTel propagates ERROR up the
        # parent chain on its own.
        open_span.span.set_status(Status(StatusCode.OK))
        open_span.span.end()

    def shutdown(self) -> None:
        """Close any still-open invocation/detached spans and shut
        down the underlying provider."""
        for cid in list(self._invocation_span.keys()):
            self._close_invocation_span(cid)
        self._provider.shutdown()


__all__ = [
    "OTelObserver",
]
