"""OTelObserver — observer-driven span lifecycle (spec observability §6).

The observer subscribes to all three §6 phases (``started``,
``completed``, ``checkpoint_saved``) plus the LLM-provider events the
``OpenAIProvider`` enqueues from inside node bodies. On a ``started``
event it opens a leaf span and pushes it onto an in-flight map keyed
by ``(namespace, attempt_index, fan_out_index)``; on the matching
``completed`` event it pops the span, applies §4.2 status mapping,
and closes it.

**Per-invocation state isolation.** All internal span maps are
outer-keyed by ``invocation_id`` (per spec §5.1: each invocation has
a fresh framework-minted UUIDv4). A single observer can be safely
shared across concurrent invocations (e.g., an ASGI service running
``asyncio.gather([invoke(), invoke()])`` on one observer); each
invocation's spans live in their own sub-dict, lazy-allocated on
first event. The ``correlation_id`` is the cross-run join key (spec
§3.1) and is set as the ``openarmature.correlation_id`` attribute on
every span — it is *not* the state-scoping key, because resume runs
preserve the correlation_id and would (incorrectly) cause the
resumed run's spans to inherit the prior invocation's trace.

**No cross-event OTel context tokens.** Parent spans are resolved from
the observer's own internal maps within a single event handler's
scope — never from ``opentelemetry.context.get_current()``. Spans are
opened with ``context=set_span_in_context(parent_span)`` directly
rather than ``attach()``-ing tokens that would have to be ``detach()``
-ed on the matching completed event. This eliminates LIFO-violation
hazards under interleaved fan-out events and makes the observer
robust to dispatch ordering.

Subtree isolation lives in dedicated dicts rather than the leaf-span
key:

- ``subgraph_spans`` — synthetic subgraph dispatch spans (the engine
  wrapper is transparent per fixture 013, but observability §4.5
  mandates a span). Keyed by namespace prefix. Open lazily on the
  first deeper-namespace event, close when subsequent events leave
  the prefix.
- ``detached_roots`` — root spans for detached subgraphs (§4.4) and
  per-instance detached fan-out roots. Each lives in its own fresh
  ``trace_id``; the parent's dispatch span carries an OTel
  :class:`Link` to the detached trace.
- ``_invocation_span`` — root invocation span keyed by
  ``invocation_id``. Closed via :meth:`close_invocation` /
  :meth:`shutdown`.

Spans are emitted through a **private** :class:`TracerProvider`
constructed by this observer — never the OTel global. Per spec §6
TracerProvider isolation, registering globally would cause every
auto-instrumentation library that writes to the global provider
(OpenInference, opentelemetry-instrumentation-openai, LiteLLM, etc.)
to emit duplicate spans alongside ours.

Detached trace mode (§4.4) is implemented by minting a fresh
:class:`SpanContext` with a new ``trace_id`` when entering a
configured-detached subgraph or fan-out; the parent's dispatch span
carries an OTel :class:`Link` to the detached trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
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
# invocation.
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


def _empty_str_frozenset() -> frozenset[str]:
    """Typed empty frozenset factory for ``detached_subgraphs`` /
    ``detached_fan_outs`` defaults."""
    return frozenset()


@dataclass
class _OpenSpan:
    """An in-flight span. No OTel context token: the new architecture
    resolves parents from the observer's internal maps within a
    single event handler's scope, so no token needs to live across
    events."""

    span: Span


@dataclass
class _InvState:
    """Per-invocation span state. One instance per concurrent
    invocation — the outer ``OTelObserver`` keys these by
    ``invocation_id`` so concurrent invocations (and resumed runs of
    the same correlation_id) don't collide."""

    open_spans: dict[_StackKey, _OpenSpan] = field(default_factory=dict[_StackKey, _OpenSpan])
    open_llm_spans: dict[str, _OpenSpan] = field(default_factory=dict[str, _OpenSpan])
    subgraph_spans: dict[tuple[str, ...], _OpenSpan] = field(default_factory=dict[tuple[str, ...], _OpenSpan])
    detached_roots: dict[tuple[str, ...], _OpenSpan] = field(default_factory=dict[tuple[str, ...], _OpenSpan])
    fan_out_instance_root_prefixes: set[tuple[str, ...]] = field(default_factory=set[tuple[str, ...]])


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

    Safe to share across concurrent invocations and across
    resumes of the same correlation_id — every internal span map is
    outer-keyed by ``invocation_id``, and parent resolution stays
    within a single event handler's scope.
    """

    span_processor: SpanProcessor
    detached_subgraphs: frozenset[str] = field(default_factory=_empty_str_frozenset)
    detached_fan_outs: frozenset[str] = field(default_factory=_empty_str_frozenset)
    disable_llm_spans: bool = False
    # Read from the package's ``__spec_version__`` (one of the three
    # places the spec version is pinned per CLAUDE.md). Bumping the
    # spec submodule + the two version fields automatically updates
    # the value reported on every invocation span.
    spec_version: str = field(default_factory=_read_spec_version)

    # Internal state, populated in __post_init__ and during invocation.
    _provider: TracerProvider = field(init=False, repr=False)
    _tracer: otel_trace.Tracer = field(init=False, repr=False)
    # Per-invocation_id span state — concurrent invocations on a
    # shared observer each get their own ``_InvState`` so internal
    # maps never collide.
    _inv_states: dict[str, _InvState] = field(init=False, repr=False, default_factory=dict[str, _InvState])
    # Root invocation spans, keyed by invocation_id. Opened lazily on
    # the first event for a new invocation_id; closed via
    # ``close_invocation`` / ``shutdown``.
    _invocation_span: dict[str, _OpenSpan] = field(
        init=False, repr=False, default_factory=dict[str, _OpenSpan]
    )

    def __post_init__(self) -> None:
        # Private provider per spec §6 TracerProvider isolation —
        # MUST NOT be registered globally.
        self._provider = TracerProvider()
        self._provider.add_span_processor(self.span_processor)
        self._tracer = self._provider.get_tracer("openarmature")

    # ------------------------------------------------------------------
    # Per-invocation state lookup
    # ------------------------------------------------------------------

    def _inv_state_for(self, invocation_id: str) -> _InvState:
        """Get-or-create the state container for an invocation_id."""
        state = self._inv_states.get(invocation_id)
        if state is None:
            state = _InvState()
            self._inv_states[invocation_id] = state
        return state

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
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()
        inv_state = self._inv_state_for(invocation_id)

        # Lazily open the invocation span on the first event we see
        # for this invocation_id. Per-invocation_id scoping means
        # resumed runs of the same correlation_id (each with a fresh
        # invocation_id per §5.1) get their own invocation span and
        # therefore their own trace_id.
        if invocation_id not in self._invocation_span:
            self._open_invocation_span(invocation_id, correlation_id, event)

        # Synthesize subgraph dispatch spans for any ancestor namespace
        # prefix that doesn't have one yet (per observability §4.5).
        # Also closes subgraph spans we've left.
        self._sync_subgraph_spans(inv_state, invocation_id, correlation_id, event)

        parent_ctx = self._resolve_parent_context(inv_state, invocation_id, event)
        span = self._tracer.start_span(
            name=event.node_name,
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=self._node_attrs(event, correlation_id),
        )
        inv_state.open_spans[self._key_for(event)] = _OpenSpan(span=span)

    def _handle_completed(self, event: NodeEvent) -> None:
        """Close the matching span, applying §4.2 status mapping."""
        from openarmature.observability.correlation import current_invocation_id

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_states.get(invocation_id)
        if inv_state is None:
            return

        # If this is the fan-out node's own completion AND the
        # fan-out is configured detached, close all per-instance
        # detached roots that this fan-out spawned. Done BEFORE the
        # regular pop so the close ordering is parents-after-children.
        if event.fan_out_index is None and event.namespace and event.namespace[0] in self.detached_fan_outs:
            for key in list(inv_state.detached_roots.keys()):
                if len(key) > len(event.namespace) and key[: len(event.namespace)] == event.namespace:
                    self._close_detached_root(inv_state, key)
        key = self._key_for(event)
        open_span = inv_state.open_spans.pop(key, None)
        if open_span is None:
            # Started event was never delivered (e.g., observer was
            # attached mid-invocation). Nothing to close.
            return
        span = open_span.span
        if event.error is not None:
            span.set_status(Status(StatusCode.ERROR, description=event.error.category))
            span.record_exception(event.error)
            span.set_attribute("openarmature.error.category", event.error.category)
            # Per spec §4.2 / fixture 003: the invocation span MUST
            # end with ERROR status when any child node errors. OTel
            # doesn't auto-propagate child status to parents — we set
            # it explicitly here. The OTel SDK's status-precedence
            # rule preserves ERROR through any subsequent
            # ``set_status(OK)`` calls (only UNSET → OK transitions
            # are honoured), so the close path's UNSET-leave still
            # works for clean invocations.
            inv_open = self._invocation_span.get(invocation_id)
            if inv_open is not None:
                inv_open.span.set_status(Status(StatusCode.ERROR, description=event.error.category))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()
        # If this was a detached root prefix, drop the root entry so a
        # subsequent re-entry mints a fresh trace.
        inv_state.detached_roots.pop(event.namespace, None)

    # ------------------------------------------------------------------
    # Special-event paths
    # ------------------------------------------------------------------

    def _emit_checkpoint_save_span(self, event: NodeEvent) -> None:
        """Spec pipeline-utilities §10.8 + observability §4.5: emit a
        zero-duration ``openarmature.checkpoint.save`` span attached
        to the most-recently-opened node span (the node whose
        completed event triggered the save)."""
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_states.get(invocation_id)
        if inv_state is None:
            return
        parent_ctx = self._resolve_parent_context(inv_state, invocation_id, event)
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
        """LLM provider span per spec §5.5 — parented to the calling
        node's span via the calling-node identity carried on the
        ``_LlmEventState`` payload (namespace_prefix + attempt_index
        + fan_out_index). Lookup hits the per-invocation_id
        ``open_spans`` so concurrent fan-out instances each find
        their own calling node, not a sibling's."""
        from openarmature.llm.providers.openai import _LlmEventState
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        if not isinstance(event.pre_state, _LlmEventState):
            # Defensive — callers other than the OpenAIProvider hook
            # shouldn't dispatch through the LLM_NAMESPACE sentinel.
            return
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_state_for(invocation_id)
        payload = event.pre_state
        if event.phase == "started":
            parent_ctx = self._resolve_llm_parent(inv_state, invocation_id, payload)
            attrs: dict[str, Any] = {"openarmature.llm.model": payload.model}
            cid = current_correlation_id()
            if cid is not None:
                attrs["openarmature.correlation_id"] = cid
            span = self._tracer.start_span(
                name="openarmature.llm.complete",
                context=cast("Any", parent_ctx),
                kind=SpanKind.CLIENT,
                attributes=attrs,
            )
            inv_state.open_llm_spans[payload.call_id] = _OpenSpan(span=span)
        elif event.phase == "completed":
            open_span = inv_state.open_llm_spans.pop(payload.call_id, None)
            if open_span is None:
                return
            span = open_span.span
            if payload.finish_reason is not None:
                span.set_attribute("openarmature.llm.finish_reason", payload.finish_reason)
            if payload.prompt_tokens is not None:
                span.set_attribute("openarmature.llm.usage.prompt_tokens", payload.prompt_tokens)
            if payload.completion_tokens is not None:
                span.set_attribute("openarmature.llm.usage.completion_tokens", payload.completion_tokens)
            if payload.total_tokens is not None:
                span.set_attribute("openarmature.llm.usage.total_tokens", payload.total_tokens)
            if payload.error_type is not None:
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        description=payload.error_category or payload.error_type,
                    )
                )
                if payload.error_category is not None:
                    span.set_attribute("openarmature.error.category", payload.error_category)
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()

    def _resolve_llm_parent(
        self,
        inv_state: _InvState,
        invocation_id: str,
        payload: Any,
    ) -> object:
        """Look up the calling node's span using the calling-node
        identity carried on the LLM event payload, fall back through
        subgraph dispatch / invocation span."""
        # 1. Direct match on the calling node's ``_StackKey``.
        calling_key: _StackKey = (
            payload.calling_namespace_prefix,
            payload.calling_attempt_index,
            payload.calling_fan_out_index,
        )
        calling = inv_state.open_spans.get(calling_key)
        if calling is not None:
            return set_span_in_context(calling.span)
        # 2. Walk up the calling namespace prefix for a synthetic
        #    subgraph dispatch span at any ancestor — covers LLM
        #    calls from inside subgraph wrapper middleware.
        prefix = payload.calling_namespace_prefix
        for plen in range(len(prefix), 0, -1):
            ancestor = prefix[:plen]
            sg = inv_state.subgraph_spans.get(ancestor)
            if sg is not None:
                return set_span_in_context(sg.span)
            dr = inv_state.detached_roots.get(ancestor)
            if dr is not None:
                return set_span_in_context(dr.span)
        # 3. Invocation span — ``complete()`` called outside any
        #    node body but inside an ``invoke()``.
        inv = self._invocation_span.get(invocation_id)
        if inv is not None:
            return set_span_in_context(inv.span)
        # 4. No invocation in scope — return a fresh empty Context.
        #    The span will live in its own trace.
        return otel_context.Context()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_invocation_span(
        self,
        invocation_id: str,
        correlation_id: str | None,
        event: NodeEvent,
    ) -> None:
        """Open the root invocation span for a new invocation."""
        attrs: dict[str, Any] = {
            "openarmature.graph.entry_node": event.node_name,
            "openarmature.graph.spec_version": self.spec_version,
            "openarmature.invocation_id": invocation_id,
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        span = self._tracer.start_span(
            name="openarmature.invocation",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        self._invocation_span[invocation_id] = _OpenSpan(span=span)

    def _key_for(self, event: NodeEvent) -> _StackKey:
        return (event.namespace, event.attempt_index, event.fan_out_index)

    def _resolve_parent_context(
        self,
        inv_state: _InvState,
        invocation_id: str,
        event: NodeEvent,
    ) -> object:
        """Return the OTel context to use as the parent for this
        event's span. Walks namespace ancestors finding the
        innermost-open subgraph or detached root span; falls back to
        the invocation span."""
        # 1a. Detached fan-out instance root — keyed by
        #     ``namespace[:1] + (str(fan_out_index),)`` per
        #     ``_open_detached_fan_out_instance_root``. Checked
        #     explicitly before the generic prefix scan.
        if event.fan_out_index is not None and event.namespace:
            instance_key = event.namespace[:1] + (str(event.fan_out_index),)
            root = inv_state.detached_roots.get(instance_key)
            if root is not None:
                return set_span_in_context(root.span)
        # 1b. Detached subgraph root at any matching prefix wins
        #     (highest precedence — events inside a detached subtree
        #     always parent under the detached root, never bleed up).
        for prefix_len in range(len(event.namespace) - 1, -1, -1):
            prefix = event.namespace[:prefix_len]
            root = inv_state.detached_roots.get(prefix)
            if root is not None:
                return set_span_in_context(root.span)
        # 2. Innermost synthetic subgraph span at any prefix.
        for prefix_len in range(len(event.namespace) - 1, 0, -1):
            prefix = event.namespace[:prefix_len]
            sg = inv_state.subgraph_spans.get(prefix)
            if sg is not None:
                return set_span_in_context(sg.span)
        # 3. Otherwise, parent under the invocation span.
        inv = self._invocation_span.get(invocation_id)
        if inv is not None:
            return set_span_in_context(inv.span)
        # 4. No invocation in scope — fresh empty Context.
        return otel_context.Context()

    def _sync_subgraph_spans(
        self,
        inv_state: _InvState,
        invocation_id: str,
        correlation_id: str | None,
        event: NodeEvent,
    ) -> None:
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
        for prefix in list(inv_state.subgraph_spans.keys()):
            if not (len(prefix) < len(namespace) and namespace[: len(prefix)] == prefix):
                self._close_subgraph_span(inv_state, prefix)
        # 2. Same for detached subgraph roots — close ones we've
        #    left. (Detached fan-out instance roots are NOT closed
        #    here; they close on the fan-out's own completion.)
        for prefix in list(inv_state.detached_roots.keys()):
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
            if prefix in inv_state.fan_out_instance_root_prefixes:
                continue
            if not (len(prefix) < len(namespace) and namespace[: len(prefix)] == prefix):
                self._close_detached_root(inv_state, prefix)
        # 3. Open ancestor subgraph spans for any prefix that doesn't
        #    have one yet.
        for depth in range(1, len(namespace)):
            prefix = namespace[:depth]
            if prefix in inv_state.subgraph_spans:
                continue
            if prefix in inv_state.detached_roots:
                continue
            # If this prefix's first segment is configured as a
            # detached subgraph, mint a fresh trace.
            if depth == 1 and prefix[0] in self.detached_subgraphs:
                self._open_detached_subgraph_root(inv_state, invocation_id, correlation_id, prefix)
                continue
            # If this is a fan-out instance namespace (event.fan_out_index
            # populated, prefix == namespace[:1]), and the fan-out
            # node is detached, open a per-instance detached root.
            if depth == 1 and event.fan_out_index is not None and prefix[0] in self.detached_fan_outs:
                self._open_detached_fan_out_instance_root(inv_state, correlation_id, prefix, event)
                continue
            self._open_subgraph_span(inv_state, invocation_id, correlation_id, prefix)

    def _open_subgraph_span(
        self,
        inv_state: _InvState,
        invocation_id: str,
        correlation_id: str | None,
        prefix: tuple[str, ...],
    ) -> None:
        """Open a synthetic subgraph dispatch span for the given
        namespace prefix. Parent is the next-outer subgraph span (or
        the invocation span if depth-1)."""
        # Walk up looking for the nearest enclosing subgraph or
        # detached root.
        parent_ctx: object = otel_context.Context()
        for plen in range(len(prefix) - 1, 0, -1):
            outer = prefix[:plen]
            sg = inv_state.subgraph_spans.get(outer)
            if sg is not None:
                parent_ctx = set_span_in_context(sg.span)
                break
            dr = inv_state.detached_roots.get(outer)
            if dr is not None:
                parent_ctx = set_span_in_context(dr.span)
                break
        else:
            inv = self._invocation_span.get(invocation_id)
            if inv is not None:
                parent_ctx = set_span_in_context(inv.span)
        attrs: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.subgraph.name": prefix[-1],
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        span = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        inv_state.subgraph_spans[prefix] = _OpenSpan(span=span)

    def _close_subgraph_span(self, inv_state: _InvState, prefix: tuple[str, ...]) -> None:
        open_span = inv_state.subgraph_spans.pop(prefix, None)
        if open_span is None:
            return
        open_span.span.set_status(Status(StatusCode.OK))
        open_span.span.end()

    def _open_detached_subgraph_root(
        self,
        inv_state: _InvState,
        invocation_id: str,
        correlation_id: str | None,
        prefix: tuple[str, ...],
    ) -> None:
        """Mint a fresh trace for a detached subgraph entry. The
        detached root span lives in the new trace; the parent trace's
        dispatch span (synthesized at the same prefix BUT in the
        parent trace) carries an OTel Link to this root."""
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

        # 2. Open the dispatch span in the parent trace. Parent of
        #    the dispatch span is the invocation span (or whatever
        #    was already in scope) per the per-invocation map.
        parent_ctx_for_dispatch: object = otel_context.Context()
        inv = self._invocation_span.get(invocation_id)
        if inv is not None:
            parent_ctx_for_dispatch = set_span_in_context(inv.span)
        attrs_parent: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.subgraph.name": prefix[-1],
        }
        if correlation_id is not None:
            attrs_parent["openarmature.correlation_id"] = correlation_id
        parent_dispatch = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", parent_ctx_for_dispatch),
            kind=SpanKind.INTERNAL,
            links=[Link(detached_sc)],
            attributes=attrs_parent,
        )
        inv_state.subgraph_spans[prefix] = _OpenSpan(span=parent_dispatch)

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
        inv_state.detached_roots[prefix] = _OpenSpan(span=detached_root)

    def _open_detached_fan_out_instance_root(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        """Per-instance detached root for a configured-detached
        fan-out. Each instance gets its own trace_id; the fan-out
        node's span (in the parent trace, already open via the
        engine's started event) accumulates Links — one per
        instance."""
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
        # trace and add a Link to the detached root. Retry middleware
        # wrapping the fan-out bumps its attempt_index, so the span
        # sits at ``(prefix, N, None)`` for the in-flight attempt N
        # — scan for any entry at ``prefix`` with
        # ``fan_out_index is None`` rather than hardcoding the key.
        # Only one such entry is open at a time (retry opens and
        # closes within a single attempt's lifecycle).
        fan_out_open = self._find_fan_out_node_span(inv_state, prefix)
        if fan_out_open is not None:
            fan_out_open.span.add_link(detached_sc)

        # Open the detached instance root span.
        detached_parent_ctx = otel_trace.set_span_in_context(
            NonRecordingSpan(detached_sc), otel_context.Context()
        )
        attrs: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.fan_out.parent_node_name": prefix[-1],
            "openarmature.node.fan_out_index": event.fan_out_index,
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        instance_root = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", detached_parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        # Key by prefix + (str(fan_out_index),) so per-instance
        # roots stay distinct.
        instance_key = prefix + (str(event.fan_out_index),)
        inv_state.detached_roots[instance_key] = _OpenSpan(span=instance_root)
        inv_state.fan_out_instance_root_prefixes.add(instance_key)

    def _close_detached_root(self, inv_state: _InvState, prefix: tuple[str, ...]) -> None:
        inv_state.fan_out_instance_root_prefixes.discard(prefix)
        open_span = inv_state.detached_roots.pop(prefix, None)
        if open_span is None:
            return
        open_span.span.set_status(Status(StatusCode.OK))
        open_span.span.end()

    @staticmethod
    def _drain_open_span(open_span: _OpenSpan) -> None:
        """Close an open span as an orphan during shutdown: OK
        status, end. No paired completed event will arrive, so we
        don't have an error category to record."""
        open_span.span.set_status(Status(StatusCode.OK))
        open_span.span.end()

    def _find_fan_out_node_span(self, inv_state: _InvState, prefix: tuple[str, ...]) -> _OpenSpan | None:
        """Find the currently-open fan-out node's parent dispatch
        span at ``prefix`` regardless of ``attempt_index``. Under
        retry middleware wrapping the fan-out, the in-flight
        attempt's span lives at ``(prefix, attempt_index, None)``;
        only one such entry is open at a time (retry opens and
        closes within each attempt's lifecycle), so a scan finds it
        unambiguously."""
        for key, open_span in inv_state.open_spans.items():
            ns, _attempt, fan_idx = key
            if ns == prefix and fan_idx is None:
                return open_span
        return None

    def _node_attrs(self, event: NodeEvent, correlation_id: str | None) -> dict[str, Any]:
        """Build the §5 attribute set for a node span."""
        attrs: dict[str, Any] = {
            "openarmature.node.name": event.node_name,
            "openarmature.node.namespace": list(event.namespace),
            "openarmature.node.step": event.step,
            "openarmature.node.attempt_index": event.attempt_index,
        }
        if event.fan_out_index is not None:
            attrs["openarmature.node.fan_out_index"] = event.fan_out_index
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        return attrs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_invocation(self, invocation_id: str) -> None:
        """Close the invocation span for ``invocation_id`` and drain
        the per-invocation state. Idempotent — calling twice (or for
        an invocation_id with no open span) is a no-op.

        Drains any still-open spans in the per-invocation state in
        child→parent order (LLM spans → leaf spans → detached roots
        → subgraph dispatch → invocation).

        **Sourcing the invocation_id.** ``CompiledGraph.invoke()``
        does not currently return the invocation_id, and the
        ``current_invocation_id`` ContextVar is reset before control
        returns to the caller. The practical use case for this
        method is test code that captures the invocation_id from
        inside a node body (or middleware / observer callback),
        debugging scenarios, and integration code that has the id
        from a checkpoint record's ``invocation_id`` field.

        For typical production lifecycle on long-lived observers,
        prefer :meth:`shutdown` — it drains every in-flight
        invocation in one call without needing to track ids
        externally. A first-class engine-level signal that lets
        observers auto-drain per-invocation state on completion is
        tracked as Phase 6.1+ follow-up work in
        ``openarmature-coord/docs/phase-6-1-conformance-fillin.md``.
        """
        inv_state = self._inv_states.pop(invocation_id, None)
        if inv_state is not None:
            self._drain_inv_state(inv_state)
        self._close_invocation_span(invocation_id)

    def _drain_inv_state(self, inv_state: _InvState) -> None:
        """Close any still-open spans in a per-invocation state
        container in child→parent order. LLM spans (deepest leaves)
        → leaf node spans (sorted deepest-first by namespace) →
        detached roots → subgraph dispatch spans. Matches the
        ordering used in ``shutdown``."""
        for call_id in list(inv_state.open_llm_spans.keys()):
            open_span = inv_state.open_llm_spans.pop(call_id, None)
            if open_span is not None:
                self._drain_open_span(open_span)
        for key in sorted(inv_state.open_spans.keys(), key=lambda k: -len(k[0])):
            open_span = inv_state.open_spans.pop(key, None)
            if open_span is not None:
                self._drain_open_span(open_span)
        for prefix in sorted(inv_state.detached_roots.keys(), key=lambda k: -len(k)):
            self._close_detached_root(inv_state, prefix)
        for prefix in sorted(inv_state.subgraph_spans.keys(), key=lambda k: -len(k)):
            self._close_subgraph_span(inv_state, prefix)

    def _close_invocation_span(self, invocation_id: str) -> None:
        """End and remove the invocation span for ``invocation_id``."""
        open_span = self._invocation_span.pop(invocation_id, None)
        if open_span is None:
            return
        # Don't unconditionally call ``set_status(OK)`` here. OTel
        # doesn't auto-propagate child span status to parents, so
        # the spec §4.2 / fixture 003 contract ("invocation span
        # ends ERROR when a child errored") is satisfied by
        # ``_handle_completed`` setting ERROR on this span when an
        # error event fires. Calling ``set_status(OK)`` here would
        # be a no-op when ERROR was already set (OTel SDK
        # status-precedence preserves ERROR), but it's clearer to
        # leave the status UNSET in the clean-completion path —
        # exporters map UNSET to OK by convention, and the explicit
        # ERROR-set in ``_handle_completed`` handles the failure
        # path.
        open_span.span.end()

    def shutdown(self) -> None:
        """Close any still-open spans across all in-flight
        invocations and shut down the underlying provider. Each
        per-invocation state is drained in child→parent order (LLM
        spans → leaf spans → detached roots → subgraph dispatch);
        invocation spans drain last. Idempotent."""
        for invocation_id in list(self._inv_states.keys()):
            inv_state = self._inv_states.pop(invocation_id)
            self._drain_inv_state(inv_state)
        for invocation_id in list(self._invocation_span.keys()):
            self._close_invocation_span(invocation_id)
        self._provider.shutdown()


__all__ = [
    "OTelObserver",
]
