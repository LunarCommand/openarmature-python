# Spec mapping (observability):
# - Observer-driven span lifecycle realizes §6 (RECOMMENDED path).
# - Span status comes from §4.2 error-category mapping.
# - Internal span maps are keyed by ``invocation_id``, which is also
#   surfaced as the ``openarmature.invocation_id`` span attribute per
#   §5.1.
# - Subgraph dispatch span names come from §4.5 (parent graph's
#   SubgraphNode name); hierarchy from §4.1/§4.3; detached trace mode
#   from §4.4.
# - ``correlation_id`` cross-run join key surfaces as the
#   ``openarmature.correlation_id`` span attribute (§3.1).
# - Private TracerProvider per §6 isolation requirement (prevents
#   global-provider auto-instrumentation libraries from emitting
#   duplicate spans alongside ours).

"""OTelObserver: observer-driven span lifecycle.

The observer subscribes to all three node-event phases (``started``,
``completed``, ``checkpoint_saved``) plus the LLM-provider events the
``OpenAIProvider`` enqueues from inside node bodies. On a ``started``
event it opens a leaf span and pushes it onto an in-flight map keyed
by ``(namespace, attempt_index, fan_out_index)``; on the matching
``completed`` event it pops the span, applies the status mapping, and
closes it.

**Per-invocation state isolation.** All internal span maps are
outer-keyed by ``invocation_id`` (each invocation has a fresh
framework-minted UUIDv4). A single observer can be safely shared
across concurrent invocations (e.g., an ASGI service running
``asyncio.gather([invoke(), invoke()])`` on one observer); each
invocation's spans live in their own sub-dict, lazy-allocated on
first event. The ``correlation_id`` is the cross-run join key set as
the ``openarmature.correlation_id`` attribute on every span; it is
*not* the state-scoping key, because resume runs preserve the
correlation_id and would (incorrectly) cause the resumed run's spans
to inherit the prior invocation's trace.

**No cross-event OTel context tokens.** Parent spans are resolved
from the observer's own internal maps within a single event
handler's scope; never from ``opentelemetry.context.get_current()``.
Spans are opened with ``context=set_span_in_context(parent_span)``
directly rather than ``attach()``-ing tokens that would have to be
``detach()``-ed on the matching completed event. This eliminates
LIFO-violation hazards under interleaved fan-out events and makes
the observer robust to dispatch ordering.

Subtree isolation lives in dedicated dicts rather than the leaf-span
key:

- ``subgraph_spans``: synthetic subgraph dispatch spans (the engine
  wrapper is transparent but the observer mints a span anyway).
  Keyed by namespace prefix. Open lazily on the first deeper-
  namespace event, close when subsequent events leave the prefix.
- ``detached_roots``: root spans for detached subgraphs and per-
  instance detached fan-out roots. Each lives in its own fresh
  ``trace_id``; the parent's dispatch span carries an OTel
  :class:`Link` to the detached trace.
- ``_invocation_span``: root invocation span keyed by
  ``invocation_id``. Closed via :meth:`close_invocation` /
  :meth:`shutdown`.

Spans are emitted through a **private** :class:`TracerProvider`
constructed by this observer; never the OTel global. Registering
globally would cause every auto-instrumentation library that writes
to the global provider (OpenInference,
opentelemetry-instrumentation-openai, LiteLLM, etc.) to emit
duplicate spans alongside ours.

Detached trace mode is implemented by minting a fresh
:class:`SpanContext` with a new ``trace_id`` when entering a
configured-detached subgraph or fan-out; the parent's dispatch span
carries an OTel :class:`Link` to the detached trace.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
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

from openarmature.graph.events import (
    FailureIsolatedEvent,
    InvocationCompletedEvent,
    InvocationStartedEvent,
    LlmCompletionEvent,
    LlmFailedEvent,
    LlmRetryAttemptEvent,
    MetadataAugmentationEvent,
    NodeEvent,
)
from openarmature.observability.lineage import is_strict_prefix
from openarmature.observability.llm_event import serialize_tool_calls

# Span-stack key shape:
# ``(namespace, attempt_index, fan_out_index, branch_name)`` — these
# four fields jointly identify any node attempt within an invocation.
# ``branch_name`` discriminates concurrent same-named inner nodes
# across sibling parallel-branches branches (pipeline-utilities §11);
# without it the two inner ``ask`` nodes of two branches with the
# same namespace + fan_out_index would collide on the same key.
_StackKey = tuple[tuple[str, ...], int, int | None, str | None]


# §5.5.5 truncation marker. The leading character is U+2026 HORIZONTAL
# ELLIPSIS (3 bytes UTF-8); the marker is a fixed UTF-8 string and is
# appended as a whole unit so no boundary backtracking is needed past
# the prefix cut.
_TRUNCATION_MARKER_TEMPLATE = "…[truncated, {m} bytes total]"


# §5.5.5 minimum-cap rule: any payload byte cap configuration below
# 256 bytes is rejected at observer construction time. Rationale (spec
# verbatim): "256 bytes leaves room for the worst-case marker (~36
# bytes) plus a diagnostically useful payload preview; caps below this
# would produce attributes that are almost entirely marker with little
# or no preview value."
_PAYLOAD_MIN_BYTES = 256


# §5.5 default truncation cap — 64 KiB per the spec's preferred
# default. Implementations MAY configure; ours sits on a constructor
# field.
_PAYLOAD_DEFAULT_BYTES = 65536


def _read_spec_version() -> str:
    """Read the spec version pinned at package level. Lazy import
    avoids a circular at module-load time (the package's ``__init__``
    imports submodules that may import the observability stack)."""
    from openarmature import __spec_version__

    return __spec_version__


# Proposal 0052: implementation attribution attributes sourced from
# the package's identity constants. Same lazy-import discipline as
# ``_read_spec_version`` to avoid a load-time cycle.
def _read_implementation_name() -> str:
    from openarmature import __implementation_name__

    return __implementation_name__


def _read_implementation_version() -> str:
    from openarmature import __version__

    return __version__


def _apply_caller_metadata(attrs: dict[str, Any], metadata: Mapping[str, Any]) -> None:
    """Merge caller-supplied invocation metadata into a span's
    attribute dict as ``openarmature.user.<key>`` entries.

    Called at every span-emission site so the metadata family is
    cross-cutting (invocation span, every node span, subgraph
    dispatch, fan-out instance dispatch, LLM provider span,
    detached roots). Source values may come from
    ``NodeEvent.caller_invocation_metadata`` for graph events or
    from the typed LLM events' (LlmCompletionEvent / LlmFailedEvent)
    caller_invocation_metadata field for LLM events; both are
    dispatch-time snapshots.
    """
    for key, value in metadata.items():
        attrs[f"openarmature.user.{key}"] = value


def _subgraph_identity_at(event: NodeEvent, depth: int) -> str:
    """Return the compiled-subgraph identity for the wrapper at the
    given 1-based namespace depth, or the empty string when no
    identity is tracked at that depth.

    The empty-string fallback is the "no identity tracked" case, for
    callers using ``SubgraphNode(name=..., compiled=...)`` without
    supplying ``subgraph_identity``.
    """
    # Spec observability §5.3 (coord thread
    # clarify-subgraph-name-semantics).
    idx = depth - 1
    if 0 <= idx < len(event.subgraph_identities):
        identity = event.subgraph_identities[idx]
        if identity is not None:
            return identity
    return ""


def _empty_str_frozenset() -> frozenset[str]:
    """Typed empty frozenset factory for ``detached_subgraphs`` /
    ``detached_fan_outs`` defaults."""
    return frozenset()


@dataclass
class _OpenSpan:
    """An in-flight span. No OTel context token: the new architecture
    resolves parents from the observer's internal maps within a
    single event handler's scope, so no token needs to live across
    events.

    Carries the span's own ``fan_out_index_chain`` and
    ``branch_name_chain`` so the augmentation walk can apply the
    lineage-aware boundary rule without re-deriving the chain from
    successive events."""

    span: Span
    fan_out_index_chain: tuple[int | None, ...] = ()
    branch_name_chain: tuple[str | None, ...] = ()


def _span_chain_on_path(
    open_span: _OpenSpan,
    aug_fi_chain: tuple[int | None, ...],
    aug_bn_chain: tuple[str | None, ...],
) -> bool:
    """Return True iff ``open_span``'s chain is a prefix-match of the
    augmenter's chain — i.e., the span sits on the augmenter's
    call-stack ancestor path:

    - A span shorter than the augmenter (chain prefix-matches) is an
      ancestor on the path.
    - A span at the same depth (chain exact-matches) is the augmenter
      itself (or a descendant sharing the mutated mapping).
    - A span deeper than the augmenter, OR with a position-mismatch
      anywhere, is a sibling and MUST NOT be updated.
    """
    # Spec observability §3.4 (proposal 0045): lineage-aware boundary.
    span_fi = open_span.fan_out_index_chain
    span_bn = open_span.branch_name_chain
    if len(span_fi) > len(aug_fi_chain):
        return False
    if len(span_bn) > len(aug_bn_chain):
        return False
    for i in range(len(span_fi)):
        if span_fi[i] != aug_fi_chain[i]:
            return False
    for i in range(len(span_bn)):
        if span_bn[i] != aug_bn_chain[i]:
            return False
    return True


# Sorted object keys, no insignificant whitespace, UTF-8 output (per
# observability §5.5.1 / §5.5.6). Within-impl determinism for identical
# inputs is required; cross-impl bytewise stability is NOT required by
# v0.17.0 — conformance fixtures use parse-shape assertions, not
# bytewise equality.
def _serialize_for_attribute(value: Any) -> str:
    """JSON-encode ``value`` for emission as an OTel string attribute."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# §5.5.5 truncation algorithm:
#   1. Compute M, the pre-truncation byte length.
#   2. Format the marker with M substituted; compute L_marker.
#   3. Compute target prefix size N = cap_bytes - L_marker.
#   4. Backtrack to a UTF-8 code-point boundary ≤ N (avoid splitting
#      multi-byte sequences — CJK, emoji, combining marks).
#   5. Emit first N' bytes + marker.
# The resulting string is at most cap_bytes UTF-8 bytes (may be strictly
# less due to step-4 backtracking). The marker leading char is U+2026
# HORIZONTAL ELLIPSIS (3 bytes UTF-8); appended as a whole unit so no
# further boundary concerns beyond step 4.
def _truncate_for_attribute(serialized: str, cap_bytes: int) -> str:
    """Truncate ``serialized`` to fit within ``cap_bytes`` UTF-8 bytes,
    returning the original string unchanged if it already fits."""
    encoded = serialized.encode("utf-8")
    full_length = len(encoded)
    if full_length <= cap_bytes:
        return serialized
    marker = _TRUNCATION_MARKER_TEMPLATE.format(m=full_length)
    marker_bytes = marker.encode("utf-8")
    target = cap_bytes - len(marker_bytes)
    if target <= 0:
        # Cap is smaller than the marker itself — the __post_init__
        # validation guards against this (256-byte minimum allows a
        # ~36-byte marker plus preview), but be defensive.
        return marker
    # UTF-8 lead-byte detection: a byte is a continuation byte when its
    # top two bits are 10. Backtrack from ``target`` until we land on a
    # lead byte (or hit 0). This is the cheapest correct way to find
    # the largest code-point boundary ≤ target without round-tripping
    # through ``str``.
    boundary = target
    while boundary > 0 and (encoded[boundary] & 0b1100_0000) == 0b1000_0000:
        boundary -= 1
    return encoded[:boundary].decode("utf-8", errors="strict") + marker


@dataclass
class _InvState:
    """Per-invocation span state. One instance per concurrent
    invocation — the outer ``OTelObserver`` keys these by
    ``invocation_id`` so concurrent invocations (and resumed runs of
    the same correlation_id) don't collide."""

    open_spans: dict[_StackKey, _OpenSpan] = field(default_factory=dict[_StackKey, _OpenSpan])
    subgraph_spans: dict[tuple[str, ...], _OpenSpan] = field(default_factory=dict[tuple[str, ...], _OpenSpan])
    detached_roots: dict[tuple[str, ...], _OpenSpan] = field(default_factory=dict[tuple[str, ...], _OpenSpan])
    # Proposal 0061 (observability §4.4): a detached trace roots in its
    # own ``openarmature.invocation`` span carrying the parent's
    # invocation_id; the detached subgraph / fan-out-instance span in
    # ``detached_roots`` nests under it. Keyed identically to
    # ``detached_roots`` (the prefix for a detached subgraph, ``prefix +
    # (str(fan_out_index),)`` for a detached fan-out instance) so the
    # two close together as a coterminous parent/child pair.
    detached_invocation_spans: dict[tuple[str, ...], _OpenSpan] = field(
        default_factory=dict[tuple[str, ...], _OpenSpan]
    )
    # Proposal 0061 §4.2: keys (matching ``detached_roots`` /
    # ``detached_invocation_spans``, and the detached prefix in
    # ``subgraph_spans``) whose spans the detached-error propagation
    # marked ERROR. The synthetic close paths consult this to SKIP their
    # default ``set_status(OK)``: the OTel SDK treats OK as final and
    # lets it OVERRIDE a prior ERROR (only a set to UNSET is ignored), so
    # an unconditional OK at close would erase the detached-trace ERROR.
    errored_detached_keys: set[tuple[str, ...]] = field(default_factory=set[tuple[str, ...]])
    fan_out_instance_root_prefixes: set[tuple[str, ...]] = field(default_factory=set[tuple[str, ...]])
    # Per spec §5.4 + proposal 0013 (v0.10.0): non-detached fan-outs
    # synthesize per-instance dispatch spans nested between the fan-out
    # node span and the inner-node spans. Keyed by ``prefix +
    # (str(fan_out_index),)`` mirroring the detached path's keying in
    # ``detached_roots``. Lives in the parent trace (not a fresh
    # trace_id, unlike detached). Closed when the fan-out node's
    # ``completed`` event fires.
    fan_out_instance_spans: dict[tuple[str, ...], _OpenSpan] = field(
        default_factory=dict[tuple[str, ...], _OpenSpan]
    )
    # ``parent_node_name`` cache (per spec proposal 0013 correction #4):
    # the fan-out node's name surfaces on its own ``started`` event via
    # ``NodeEvent.fan_out_config.parent_node_name``. The observer caches
    # it keyed by the fan-out node's namespace prefix so when
    # ``_open_fan_out_instance_dispatch_span`` synthesizes a per-instance
    # dispatch span, it can attach
    # ``openarmature.fan_out.parent_node_name`` even though the inner
    # event itself doesn't carry ``fan_out_config``.
    fan_out_parent_node_name: dict[tuple[str, ...], str] = field(default_factory=dict[tuple[str, ...], str])
    # Per proposal 0044 (observability §5.7, v0.36.0): synthesize
    # per-branch dispatch spans nested between the parallel-branches
    # node span and the inner-branch spans (mirroring fan-out's
    # per-instance synthesis above).  Keyed by
    # ``prefix + (branch_name,)``.  Closed when the parallel-branches
    # node's ``completed`` event fires.
    parallel_branches_branch_spans: dict[tuple[str, ...], _OpenSpan] = field(
        default_factory=dict[tuple[str, ...], _OpenSpan]
    )
    # ``parent_node_name`` cache for parallel-branches (mirrors the
    # ``fan_out_parent_node_name`` cache above).  Surfaces from the
    # parallel-branches NODE's ``started`` event via
    # ``NodeEvent.parallel_branches_config.parent_node_name``; cached
    # per-namespace-prefix so the per-branch dispatch synthesizer can
    # attach ``openarmature.parallel_branches.parent_node_name``
    # without re-reading the inner event (which only carries
    # ``branch_name``).
    parallel_branches_parent_node_name: dict[tuple[str, ...], str] = field(
        default_factory=dict[tuple[str, ...], str]
    )

    # Per proposal 0044 (v0.36.0): cached branch-name set per
    # parallel-branches NODE namespace.  Lets the per-branch dispatch
    # synthesizer reject events whose ``branch_name`` belongs to a
    # DIFFERENT parallel-branches node (nested case: an inner pb's
    # branch event walks past outer pbs in the synthesis loop; outer
    # pbs MUST NOT synthesize a phantom dispatch span for a branch
    # name they don't actually own).
    parallel_branches_branch_names: dict[tuple[str, ...], frozenset[str]] = field(
        default_factory=dict[tuple[str, ...], frozenset[str]]
    )


@dataclass
class OTelObserver:
    """Observer-driven OTel span lifecycle.

    Construct with a :class:`SpanProcessor` (typically a
    :class:`BatchSpanProcessor` wrapping a real exporter, or a
    :class:`SimpleSpanProcessor` wrapping :class:`InMemorySpanExporter`
    for tests). The observer instantiates its own private
    :class:`TracerProvider` from the supplied processor; callers
    MUST NOT pre-register the provider globally.

    Constructor knobs:

    - ``span_processor``: a single :class:`SpanProcessor` or a sequence
      of them. Every processor is registered on the private
      :class:`TracerProvider`; spans flow to each.
    - ``resource``: optional :class:`Resource` passed to the private
      :class:`TracerProvider`. Sets ``service.name`` / ``service.version``
      / etc. without relying on environment variables.
    - ``detached_subgraphs``: set of subgraph wrapper node names that
      run in their own trace. One detached trace per such subgraph.
    - ``detached_fan_outs``: set of fan-out node names whose instances
      each get their own trace. One detached trace per instance.
    - ``disable_llm_spans``: when ``True`` the observer skips the LLM
      provider span; all other spans emit normally.
    - ``disable_provider_payload``: default ``True``. Gates the LLM input/
      output payload attributes (``openarmature.llm.input.messages``,
      ``openarmature.llm.output.content``,
      ``openarmature.llm.request.extras``). The name carries the broadened
      provider-payload scope; LLM completion is the only provider-call
      payload OA emits today.
    - ``disable_genai_semconv``: default ``False``. Gates the
      ``gen_ai.*`` attribute set on the LLM span.
    - ``payload_max_bytes``: per-attribute byte cap for the LLM payload
      attributes. Default 64 KiB; minimum 256 bytes (rejected at
      construction time below that).
    - ``attribute_enrichers``: optional sequence of callables run just
      before the observer ends each span. Each receives the live
      :class:`Span` plus the :class:`NodeEvent` or
      :class:`LlmCompletionEvent` that triggered the close (or
      ``None`` on synthetic close sites). Exceptions are caught and
      warned; never propagated.
    - ``spec_version``: string surfaced as
      ``openarmature.graph.spec_version`` on the invocation span.
    - ``implementation_name``: string surfaced as
      ``openarmature.implementation.name`` on the invocation span.
      Defaults to the package's ``__implementation_name__``
      (``"openarmature-python"``). Configurable for test
      parameterization.
    - ``implementation_version``: string surfaced as
      ``openarmature.implementation.version`` on the invocation span.
      Defaults to ``openarmature.__version__``. Always-emit invariant:
      not gated by any privacy knob.

    Safe to share across concurrent invocations and across resumes of
    the same correlation_id; every internal span map is outer-keyed by
    ``invocation_id``, and parent resolution stays within a single
    event handler's scope.
    """

    # Spec observability §6 (observer-driven span lifecycle).
    # span_processor accepts a single processor or a sequence per
    # observability friction-roundup #5. The dataclass field type is
    # the union; ``__post_init__`` normalizes to a tuple internally.
    span_processor: SpanProcessor | Sequence[SpanProcessor]
    # Optional Resource per friction-roundup #4. Default behavior
    # (resource=None) falls through to OTel's default Resource (reads
    # OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES env vars at
    # construction time).
    resource: Resource | None = None
    detached_subgraphs: frozenset[str] = field(default_factory=_empty_str_frozenset)
    detached_fan_outs: frozenset[str] = field(default_factory=_empty_str_frozenset)
    disable_llm_spans: bool = False
    # disable_provider_payload defaults to True per observability §5.5.4.
    # Default-off because the payload may contain PII the user hasn't
    # audited — opting in is a deliberate second choice. Naming inverts
    # the natural reading ("default-off via True") to keep symmetry
    # with the existing disable_llm_spans parameter family.
    disable_provider_payload: bool = True
    # disable_genai_semconv defaults to False (emit) per §5.5.4. The
    # value proposition of installing the OTel observer is that
    # LLM-aware backends (Langfuse, Phoenix, Honeycomb's LLM lens)
    # render correctly out of the box, which keys off gen_ai.*.
    disable_genai_semconv: bool = False
    # Per-attribute byte cap for the §5.5.1 payload attributes. Default
    # 64 KiB; minimum 256 bytes (§5.5.5), validated in __post_init__.
    payload_max_bytes: int = _PAYLOAD_DEFAULT_BYTES
    # attribute_enrichers per friction-roundup #7p2 — runs before every
    # span.end() the observer issues. NodeEvent is None on synthetic
    # close sites (subgraph dispatch, detached root, fan-out instance,
    # invocation span, shutdown drain).
    attribute_enrichers: Sequence[
        Callable[[Span, NodeEvent | LlmCompletionEvent | LlmFailedEvent | LlmRetryAttemptEvent | None], None]
    ] = ()
    # Read from the package's ``__spec_version__`` (one of the three
    # places the spec version is pinned per CLAUDE.md). Bumping the
    # spec submodule + the two version fields automatically updates
    # the value reported on every invocation span.
    spec_version: str = field(default_factory=_read_spec_version)
    # Proposal 0052 (spec v0.44.0): implementation identity emitted on
    # every invocation span. ``implementation_name`` is the package
    # registry name (``openarmature-python``);
    # ``implementation_version`` is ``openarmature.__version__``.
    # Configurable for test parameterization but defaults to the
    # package-pinned values; the always-emit invariant means neither
    # ``disable_state_payload``, ``disable_provider_payload``, nor any
    # other privacy knob gates them.
    implementation_name: str = field(default_factory=_read_implementation_name)
    implementation_version: str = field(default_factory=_read_implementation_version)

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
        # §5.5.5 minimum-cap validation. Reject misconfigurations at
        # construction time rather than emitting silently broken
        # attributes.
        if self.payload_max_bytes < _PAYLOAD_MIN_BYTES:
            raise ValueError(
                f"payload_max_bytes={self.payload_max_bytes} below the spec §5.5.5 "
                f"minimum cap of {_PAYLOAD_MIN_BYTES} bytes"
            )
        # Private provider per spec §6 TracerProvider isolation —
        # MUST NOT be registered globally. Resource set on the
        # provider when supplied; otherwise OTel's default Resource
        # (which reads OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES
        # env vars at construction time) applies.
        if self.resource is not None:
            self._provider = TracerProvider(resource=self.resource)
        else:
            self._provider = TracerProvider()
        # Multi-processor: a sequence registers every entry; a single
        # processor wraps in a 1-tuple. ``SpanProcessor`` is itself a
        # class so we can't use isinstance against ``Sequence`` first
        # (Sequence matches strings too); compare against the explicit
        # union arms.
        if isinstance(self.span_processor, SpanProcessor):
            processors: Sequence[SpanProcessor] = (self.span_processor,)
        else:
            processors = tuple(self.span_processor)
        for proc in processors:
            self._provider.add_span_processor(proc)
        self._tracer = self._provider.get_tracer("openarmature")

    # ------------------------------------------------------------------
    # Enricher invocation (friction-roundup #7p2)
    # ------------------------------------------------------------------

    # Exception isolation mirrors the observer-error-isolation contract
    # in ``openarmature.graph.observer`` — enricher raises are caught +
    # warned, never propagated to the dispatch worker.
    # ``event`` is None on synthetic close sites (subgraph dispatch,
    # detached root, fan-out instance, invocation span, orphan drain).
    def _run_enrichers(
        self, span: Span, event: NodeEvent | LlmCompletionEvent | LlmFailedEvent | LlmRetryAttemptEvent | None
    ) -> None:
        """Invoke configured enrichers against ``span`` before
        ``span.end()`` is called."""
        if not self.attribute_enrichers:
            return
        import warnings

        for enricher in self.attribute_enrichers:
            try:
                enricher(span, event)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"attribute_enricher raised {type(e).__name__}: {e}",
                    stacklevel=2,
                )

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
    # Observer protocol — async callable accepting node events + the
    # proposal-0040 metadata-augmentation event variant + the
    # proposal-0043 invocation-boundary events (no-op on the OTel
    # mapping; OTel has no Trace-level input/output concept per the
    # proposal's Out-of-Scope section).
    # ------------------------------------------------------------------

    async def __call__(
        self,
        event: (
            NodeEvent
            | MetadataAugmentationEvent
            | InvocationStartedEvent
            | InvocationCompletedEvent
            | LlmCompletionEvent
            | LlmFailedEvent
            | LlmRetryAttemptEvent
            | FailureIsolatedEvent
        ),
    ) -> None:
        # Proposal 0043 invocation-boundary events: OTel has no
        # Trace-level input/output payload concept (a trace is a
        # collection of Spans sharing a trace_id; no Trace-level
        # payload field). No-op gates here; isinstance early-return
        # before any node-specific logic runs.
        if isinstance(event, InvocationStartedEvent | InvocationCompletedEvent):
            return
        # Proposal 0050 per-attempt LLM span surface: the
        # openarmature.llm.complete span(s) render from the per-attempt
        # LlmRetryAttemptEvent — one span per attempt under call-level
        # retry (attempt_index 0..N-1), one for a no-retry call.
        if isinstance(event, LlmRetryAttemptEvent):
            if not self.disable_llm_spans:
                self._handle_typed_llm_retry_attempt(event)
            return
        # The terminal LlmCompletionEvent / LlmFailedEvent no longer
        # drive the OTel span (the per-attempt event does); they stay on
        # the queue for the Langfuse mapping and payload/latency
        # consumers, so the OTel observer ignores them here.
        if isinstance(event, LlmCompletionEvent | LlmFailedEvent):
            return
        # Proposal 0050 §6.3 framework-emitted failure-isolation event.
        if isinstance(event, FailureIsolatedEvent):
            self._handle_failure_isolated(event)
            return
        if isinstance(event, MetadataAugmentationEvent):
            self._handle_metadata_augmentation(event)
            return
        if event.phase == "checkpoint_saved":
            self._emit_checkpoint_save_span(event)
            return
        if event.phase == "checkpoint_migrated":
            self._emit_checkpoint_migrate_span(event)
            return
        if event.phase == "started":
            # Idempotent — short-circuits inside ``_open_started_span``
            # if ``prepare_sync`` already opened the span synchronously
            # in the engine task. Falls through (and opens the span)
            # for observers attached after the engine started, or for
            # test paths that bypass ``prepare_sync``.
            self._open_started_span(event)
        elif event.phase == "completed":
            self._handle_completed(event)

    # ------------------------------------------------------------------
    # Started / completed pairing
    # ------------------------------------------------------------------

    def prepare_sync(self, event: NodeEvent) -> None:
        """Synchronous engine-task entry point: open the span for this
        attempt AND publish it via ``current_active_observer_span`` so
        the engine's ``innermost`` can attach it into the OTel context
        before the node body runs.

        Called by ``_dispatch`` BEFORE ``queue.put_nowait`` for
        ``"started"``-phase events. The async ``__call__`` later sees
        the span already in ``inv_state.open_spans`` and short-circuits.

        Skipped for non-``"started"`` phases and for the LLM sentinel
        namespace; only graph-node started events participate in the
        engine-side attach. Errors don't leak: ``_dispatch`` wraps this
        call in try/except + ``warnings.warn`` matching the async path.
        """
        if event.phase != "started":
            return
        from openarmature.observability.correlation import (
            _set_active_observer_span,
            current_invocation_id,
        )

        self._open_started_span(event)
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_states.get(invocation_id)
        if inv_state is None:
            return
        open_span = inv_state.open_spans.get(self._key_for(event))
        if open_span is None:
            # Proposal 0075: a callable parallel-branch has no leaf — its span
            # is the per-branch dispatch span opened in ``_open_started_span``.
            # Publish that so a provider call inside the callable nests under
            # the branch span.
            if event.branch_name is not None and event.parallel_branches_config is None:
                dispatch = inv_state.parallel_branches_branch_spans.get(
                    event.namespace + (event.branch_name,)
                )
                if dispatch is not None:
                    _set_active_observer_span(dispatch.span)
            return
        # Publish the span to the engine via the ContextVar. Discard
        # the Token — last-writer-wins is the documented contract
        # (next ``prepare_sync`` overwrites; task-local context dies
        # with the invocation task).
        _set_active_observer_span(open_span.span)

    def _open_started_span(self, event: NodeEvent) -> None:
        """Sync core: create the span + mutate ``inv_state.open_spans``.

        Idempotent — short-circuits if a span already exists for this
        event's ``_StackKey``. That covers the common case where
        ``prepare_sync`` opened the span synchronously in the engine
        task and the async ``__call__`` later re-fires for the same
        event; the second call becomes a true no-op rather than
        opening a duplicate span.
        """
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_state_for(invocation_id)
        # Idempotency: a span already exists for this attempt — likely
        # opened by ``prepare_sync`` in the engine task. No-op to avoid
        # duplicates.
        if self._key_for(event) in inv_state.open_spans:
            return
        correlation_id = current_correlation_id()

        # Lazily open the invocation span on the first event we see
        # for this invocation_id. Per-invocation_id scoping means
        # resumed runs of the same correlation_id (each with a fresh
        # invocation_id per §5.1) get their own invocation span and
        # therefore their own trace_id.
        if invocation_id not in self._invocation_span:
            self._open_invocation_span(invocation_id, correlation_id, event)

        # Per spec proposal 0013 (v0.10.0): cache the fan-out node's
        # ``parent_node_name`` from its ``started`` event so
        # ``_open_fan_out_instance_dispatch_span`` can attach
        # ``openarmature.fan_out.parent_node_name`` to per-instance
        # spans. Inner events from inside the fan-out's instances
        # don't carry ``fan_out_config`` themselves; the cache bridges.
        # ``fan_out_config`` is only populated on the NODE's OWN
        # events, so the presence check is sufficient — we don't
        # additionally filter on ``fan_out_index is None`` because
        # when a fan-out is nested inside another fan-out or a pb
        # branch, the NODE's own event carries the OUTER axis values.
        if event.fan_out_config is not None:
            inv_state.fan_out_parent_node_name[event.namespace] = event.fan_out_config.parent_node_name

        # Per proposal 0044 (v0.36.0): mirror cache for the parallel-
        # branches NODE.  Same logic as the fan-out cache: rely on
        # ``parallel_branches_config`` being populated only on the
        # NODE's own events; don't additionally filter on
        # ``branch_name is None`` (which would skip caching for a pb
        # nested inside another pb's branch).
        if event.parallel_branches_config is not None:
            inv_state.parallel_branches_parent_node_name[event.namespace] = (
                event.parallel_branches_config.parent_node_name
            )
            inv_state.parallel_branches_branch_names[event.namespace] = frozenset(
                event.parallel_branches_config.branch_names
            )

        # Synthesize subgraph dispatch spans for any ancestor namespace
        # prefix that doesn't have one yet (per observability §4.5).
        # Also closes subgraph spans we've left.
        self._sync_subgraph_spans(inv_state, invocation_id, correlation_id, event)

        # Proposal 0075 (observability §5.7): a callable parallel-branch emits
        # its started/completed pair at the pb NODE's own namespace, tagged
        # with branch_name and (unlike the NODE's own events) no
        # parallel_branches_config. It IS the unit, so render it as the
        # branch's per-branch dispatch span (keyed by branch_name, parented
        # under the NODE span) with NO inner-node leaf. A subgraph branch's
        # inner-node events are always one level deeper, so this never
        # misfires for them. The dispatch span is closed when the NODE's own
        # completed event fires (children-before-parents), like any branch
        # dispatch span; this branch's completed event is then a no-op pop.
        if (
            event.branch_name is not None
            and event.parallel_branches_config is None
            and event.namespace in inv_state.parallel_branches_parent_node_name
        ):
            branch_key = event.namespace + (event.branch_name,)
            if branch_key not in inv_state.parallel_branches_branch_spans:
                self._open_parallel_branches_branch_dispatch_span(
                    inv_state, correlation_id, event.namespace, event
                )
            return

        parent_ctx = self._resolve_parent_context(inv_state, invocation_id, event)
        span = self._tracer.start_span(
            name=event.node_name,
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=self._node_attrs(event, correlation_id),
        )
        inv_state.open_spans[self._key_for(event)] = _OpenSpan(
            span=span,
            fan_out_index_chain=event.fan_out_index_chain,
            branch_name_chain=event.branch_name_chain,
        )

    def _handle_completed(self, event: NodeEvent) -> None:
        """Close the matching span, applying the status mapping."""
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
        # Per spec proposal 0013 (v0.10.0): if this is the fan-out
        # node's own completion (non-detached path), close all
        # per-instance dispatch spans synthesized for this fan-out.
        # Children-before-parents close ordering: per-instance
        # dispatch spans before the fan-out node's own span.
        # ``fan_out_config`` is only populated on the NODE's OWN
        # events, so the presence check is sufficient — we don't
        # additionally filter on ``fan_out_index is None`` because
        # when a fan-out is nested inside another fan-out's instance
        # or a pb's branch, the NODE's own completion event carries
        # the OUTER axis values.
        if event.fan_out_config is not None:
            for key in list(inv_state.fan_out_instance_spans.keys()):
                if len(key) > len(event.namespace) and key[: len(event.namespace)] == event.namespace:
                    self._close_fan_out_instance_dispatch_span(inv_state, key)
            inv_state.fan_out_parent_node_name.pop(event.namespace, None)

        # Per proposal 0044 (v0.36.0): if this is the parallel-branches
        # node's own completion, close all per-branch dispatch spans
        # synthesized for it.  Children-before-parents close ordering
        # (per-branch dispatch spans before the parallel-branches node
        # span itself).  Same logic as the fan-out close above: rely
        # on ``parallel_branches_config`` being populated only on the
        # NODE's own events; don't additionally filter on
        # ``branch_name is None`` (which would skip close for a pb
        # nested inside another pb's branch).
        if event.parallel_branches_config is not None:
            for key in list(inv_state.parallel_branches_branch_spans.keys()):
                if len(key) > len(event.namespace) and key[: len(event.namespace)] == event.namespace:
                    self._close_parallel_branches_branch_dispatch_span(inv_state, key)
            inv_state.parallel_branches_parent_node_name.pop(event.namespace, None)
            inv_state.parallel_branches_branch_names.pop(event.namespace, None)
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
            # it explicitly here. This ERROR survives to export because
            # ``_close_invocation_span`` deliberately never calls
            # ``set_status(OK)`` on the clean-completion path (it leaves
            # the status UNSET, which exporters map to OK). That matters
            # because the OTel SDK treats OK as FINAL and lets a later
            # OK OVERRIDE a prior ERROR (only a set to UNSET is ignored)
            # — an unconditional OK at a close site would erase this,
            # the same hazard the §4.2 detached-error propagation guards
            # against via ``errored_detached_keys``.
            inv_open = self._invocation_span.get(invocation_id)
            if inv_open is not None:
                inv_open.span.set_status(Status(StatusCode.ERROR, description=event.error.category))
            # Proposal 0061 §4.2: when the erroring node is inside a
            # DETACHED subtree, that trace needs its OWN error carriers
            # — the invocation span set just above belongs to the PARENT
            # trace. Surface ERROR on the detached trace's spans too.
            self._propagate_error_to_detached_spans(inv_state, event)
        else:
            span.set_status(Status(StatusCode.OK))
        self._run_enrichers(span, event)
        span.end()
        # If this was a detached root prefix, drop the root entry so a
        # subsequent re-entry mints a fresh trace.
        inv_state.detached_roots.pop(event.namespace, None)

    def _propagate_error_to_detached_spans(self, inv_state: _InvState, event: NodeEvent) -> None:
        # Proposal 0061 §4.2 (Detached invocation span status): a node
        # raising inside a detached subtree surfaces ERROR on that
        # trace's OWN carriers, not just the parent trace's. For each
        # enclosing detached prefix:
        #   - the detached invocation span (the detached trace's root /
        #     authoritative carrier) and the parent-trace dispatch span
        #     (the §4.4 Link carrier) each get the FULL treatment —
        #     ERROR status + an OTel exception event + the §4 category
        #     attribute, mirroring the parent invocation span;
        #   - the detached subgraph / instance span between them gets
        #     ERROR status only (the invocation span above it carries
        #     the exception event for that trace).
        # Set while the spans are still open; the synthetic close paths
        # SKIP their default ``set_status(OK)`` for keys recorded in
        # ``errored_detached_keys`` (OTel treats OK as final and lets it
        # override a prior ERROR), so the ERROR survives to export. Keys
        # cover both the detached-subgraph (prefix) and detached-fan-out-
        # instance (prefix + index) schemes.
        if event.error is None:
            return
        err = event.error
        category = err.category
        for prefix_len in range(len(event.namespace), 0, -1):
            base = event.namespace[:prefix_len]
            keys = [base]
            if event.fan_out_index is not None:
                keys.append(base + (str(event.fan_out_index),))
            for key in keys:
                inv_span = inv_state.detached_invocation_spans.get(key)
                if inv_span is None:
                    continue
                inv_state.errored_detached_keys.add(key)
                inv_span.span.set_status(Status(StatusCode.ERROR, description=category))
                inv_span.span.record_exception(err)
                inv_span.span.set_attribute("openarmature.error.category", category)
                dispatch = inv_state.subgraph_spans.get(key)
                if dispatch is not None:
                    dispatch.span.set_status(Status(StatusCode.ERROR, description=category))
                    dispatch.span.record_exception(err)
                    dispatch.span.set_attribute("openarmature.error.category", category)
                root = inv_state.detached_roots.get(key)
                if root is not None:
                    root.span.set_status(Status(StatusCode.ERROR, description=category))

    # ------------------------------------------------------------------
    # Metadata augmentation (proposal 0040 §3.4 + §6)
    # ------------------------------------------------------------------

    def _handle_metadata_augmentation(self, event: MetadataAugmentationEvent) -> None:
        # Spec proposal 0040: spans whose lineage ancestor-or-equals the
        # augmenting context (within the same fan-out instance /
        # parallel-branch boundary) get ``openarmature.user.<key>``
        # applied in place. Sibling instances / branches and ancestors
        # ABOVE the boundary are skipped.
        #
        # Match rule (using the augmentation event's lineage tuple
        # ``(NS, AI, FI, BN)``):
        # - Invocation span: included iff ``FI is None and BN is None``
        #   (outermost-serial context). The shared fan-out node span and
        #   the invocation span are explicitly out of scope when
        #   augmenting from inside a fan-out instance or branch.
        # - Subgraph wrapper spans: included on the outermost-serial
        #   path when their namespace is a strict prefix of NS.
        # - Fan-out instance dispatch spans: included iff the dispatch
        #   span's FI suffix matches ``str(FI)`` and the anchor namespace
        #   is a strict prefix of NS.
        # - Per-attempt node spans (``open_spans``): included iff the
        #   span's FI equals the augmenter's FI and its namespace is a
        #   prefix of (or equal to) NS.
        from openarmature.observability.correlation import current_invocation_id

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        if not event.entries:
            return
        targets = self._collect_augmentation_targets(invocation_id, event)
        for span in targets:
            for key, value in event.entries.items():
                # OTel forbids None as an attribute value; the metadata
                # validator at the engine boundary rejects None already,
                # so we can pass through directly.
                span.set_attribute(f"openarmature.user.{key}", value)

    def _collect_augmentation_targets(
        self, invocation_id: str, event: MetadataAugmentationEvent
    ) -> list[Span]:
        """Collect open spans on the augmenter's call-stack ancestor
        chain.  Three-step boundary decision tree per open span:

        1. Same context as augmenter (or descendant sharing the
           mutated mapping) — update.
        2. Strict dispatch ancestor on the augmenter's call-stack path
           (each outer fan-out instance dispatch, each outer parallel-
           branches branch dispatch, each outer serial-subgraph wrapper
           on the path) — update.
        3. Sibling at any depth, OR shared parent at any depth (the
           fan-out NODE itself, the parallel-branches NODE itself, the
           invocation span when augmenter is non-root) — do not update.

        Chain match: for an open span to be on the augmenter's path,
        the span's own per-depth lineage chain MUST prefix-match the
        augmenter's chain at every position the span occupies.  Where
        positions disagree, the span is a sibling.
        """
        targets: list[Span] = []
        aug_ns = event.namespace
        aug_fi_chain = event.fan_out_index_chain
        aug_bn_chain = event.branch_name_chain
        inv_state = self._inv_states.get(invocation_id)

        # Invocation span: included only when the augmenter is in
        # OUTERMOST SERIAL context — no fan-out instance and no
        # parallel-branches branch on its call-stack path.  Subgraph
        # wrappers (chain entries with None at both axes) don't
        # introduce the shared-parent semantics; only fan-out and
        # parallel-branches do.  Mirrors fixture 034 and matches the
        # 0045 §3.4 statement that "existing single-level fixtures
        # remain unchanged."
        outermost_serial = all(fi is None for fi in aug_fi_chain) and all(bn is None for bn in aug_bn_chain)
        if outermost_serial:
            inv_open = self._invocation_span.get(invocation_id)
            if inv_open is not None:
                targets.append(inv_open.span)

        if inv_state is None:
            return targets

        # Subgraph wrapper spans on the path: prefix-of-aug-namespace +
        # chain prefix-matches aug's chain.
        for prefix, open_span in inv_state.subgraph_spans.items():
            if not is_strict_prefix(prefix, aug_ns):
                continue
            if _span_chain_on_path(open_span, aug_fi_chain, aug_bn_chain):
                targets.append(open_span.span)

        # Fan-out instance dispatch spans: keyed by anchor_ns +
        # (str(fi),).  The dispatch represents the descent at namespace
        # position len(anchor_ns)-1 — i.e., chain position
        # ``len(anchor_ns)-1`` should match the dispatch's fi.
        for key, open_span in inv_state.fan_out_instance_spans.items():
            if not key:
                continue
            anchor_ns = key[:-1]
            fi_str = key[-1]
            if not (is_strict_prefix(anchor_ns, aug_ns) or anchor_ns == aug_ns):
                continue
            chain_pos = len(anchor_ns) - 1
            if chain_pos < 0 or chain_pos >= len(aug_fi_chain):
                continue
            aug_fi_at_pos = aug_fi_chain[chain_pos]
            if aug_fi_at_pos is None or str(aug_fi_at_pos) != fi_str:
                continue
            targets.append(open_span.span)

        # Per-branch dispatch spans: keyed by anchor_ns + (branch_name,).
        # Mirror logic to fan-out above, against the branch chain.
        for key, open_span in inv_state.parallel_branches_branch_spans.items():
            if not key:
                continue
            anchor_ns = key[:-1]
            bn_str = key[-1]
            if not (is_strict_prefix(anchor_ns, aug_ns) or anchor_ns == aug_ns):
                continue
            chain_pos = len(anchor_ns) - 1
            if chain_pos < 0 or chain_pos >= len(aug_bn_chain):
                continue
            if aug_bn_chain[chain_pos] != bn_str:
                continue
            targets.append(open_span.span)

        # Open NODE spans: same context (aug's own attempt span), or
        # strict ancestor on the augmenter's path.  Skip:
        # - sibling NODE spans (chain mismatch at some position)
        # - shared-parent NODE spans (fan-out NODE / pb NODE
        #   identified structurally by their presence in the
        #   parent_node_name caches)
        for key, open_span in inv_state.open_spans.items():
            ns, _ai, _fi, _bn = key
            if ns == aug_ns:
                # Same context — must have matching chain to be the
                # augmenter's own attempt rather than a sibling
                # instance's same-named node.
                if _span_chain_on_path(open_span, aug_fi_chain, aug_bn_chain):
                    targets.append(open_span.span)
                continue
            if not is_strict_prefix(ns, aug_ns):
                continue
            # Shared-parent check: if this NODE is a fan-out node or
            # a parallel-branches node (dispatcher), it's a shared
            # parent and MUST NOT be updated regardless of cardinality
            # (§3.4 — the structural classification governs, not the
            # live sibling count).
            if ns in inv_state.fan_out_parent_node_name or ns in inv_state.parallel_branches_parent_node_name:
                continue
            if _span_chain_on_path(open_span, aug_fi_chain, aug_bn_chain):
                targets.append(open_span.span)

        return targets

    # ------------------------------------------------------------------
    # Special-event paths
    # ------------------------------------------------------------------

    def _emit_checkpoint_migrate_span(self, event: NodeEvent) -> None:
        """Emit a zero-duration ``openarmature.checkpoint.migrate``
        span when a versioned resume's migration chain runs. The
        synthetic event carries ``_MigrationSummary`` on ``pre_state``;
        this handler reads ``from_version`` / ``to_version`` /
        ``chain_length`` from the summary onto the span.

        Emitted under the invocation's root span (no parent-node
        context — the migration runs before any node fires), so
        trace UIs surface it as the first child of the invocation.
        """
        # Spec pipeline-utilities §6 cross-ref (proposal 0014).
        from openarmature.graph.compiled import _MigrationSummary
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        summary = event.pre_state
        if not isinstance(summary, _MigrationSummary):
            # Defensive — the engine only sets pre_state to a
            # _MigrationSummary on this phase. Skip if something
            # else dispatched a checkpoint_migrated event.
            return
        # Open (or reuse) the invocation's root span and parent the
        # migrate span under it. The migration runs before any node
        # fires, so the root is the natural parent — no node span
        # exists yet at this point in the invocation.
        if invocation_id not in self._invocation_span:
            cid = current_correlation_id() or ""
            self._open_invocation_span(invocation_id, cid, event)
        root_open = self._invocation_span.get(invocation_id)
        parent_ctx: Any = None
        if root_open is not None:
            parent_ctx = set_span_in_context(root_open.span)
        attrs: dict[str, Any] = {
            "openarmature.checkpoint.migrate.from_version": summary.from_version,
            "openarmature.checkpoint.migrate.to_version": summary.to_version,
            "openarmature.checkpoint.migrate.chain_length": summary.chain_length,
        }
        cid = current_correlation_id()
        if cid is not None:
            attrs["openarmature.correlation_id"] = cid
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        span = self._tracer.start_span(
            name="openarmature.checkpoint.migrate",
            context=parent_ctx,
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        span.set_status(Status(StatusCode.OK))
        self._run_enrichers(span, event)
        span.end()

    def _emit_checkpoint_save_span(self, event: NodeEvent) -> None:
        """Emit a zero-duration ``openarmature.checkpoint.save`` span
        attached to the most-recently-opened node span (the node whose
        completed event triggered the save)."""
        # Spec pipeline-utilities §10.8 + observability §4.5.
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
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        span = self._tracer.start_span(
            name="openarmature.checkpoint.save",
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        span.set_status(Status(StatusCode.OK))
        self._run_enrichers(span, event)
        span.end()

    # LLM provider span per observability §5.5 — parented to the
    # calling node's span via the calling-node identity carried on the
    # event (namespace + attempt_index + fan_out_index + branch_name).
    # Lookup hits the per-invocation_id open_spans so concurrent fan-out
    # instances each find their own calling node, not a sibling's.
    #
    # v0.13.0 (proposal 0049 + 0057): the success-path span lifecycle is
    # driven by the typed LlmCompletionEvent — opened and closed in one
    # shot at typed-event arrival, with start_time back-dated by
    # latency_ms so the span duration matches the adapter-boundary
    # measurement. The error-path span is still driven by the sentinel
    # NodeEvent (LlmCompletionEvent is success-only per proposal 0049
    # §3 alternative 3). Success-side sentinel emission was dropped
    # from the provider in v0.13.0; the failure-side sentinel emission
    # stays until the spec extends LlmCompletionEvent with error
    # semantics (coord-thread tracked).
    #
    # v0.17.0 attribute set (proposal 0024) preserved unchanged:
    #   - Baseline openarmature.llm.* attributes
    #   - §5.5.1 payload (input.messages, output.content,
    #     request.extras) gated by disable_provider_payload
    #   - §5.5.2 gen_ai.request.* request params
    #   - §5.5.3 gen_ai.* response semconv set
    #   - §5.5.4 opt-out flags
    #   - §5.5.5 truncation contract on payload attributes
    #
    # Prompt-identity attributes come from the active_prompt /
    # active_prompt_group snapshots taken at dispatch time — NOT the
    # ContextVar. The dispatch worker's task-local Context doesn't see
    # node-body ContextVar writes.
    def _handle_typed_llm_retry_attempt(self, event: LlmRetryAttemptEvent) -> None:
        """Open + close one ``openarmature.llm.complete`` span from a
        per-attempt LlmRetryAttemptEvent.

        N call-level retry attempts produce N spans, all parented under
        the calling node span, each carrying
        ``openarmature.llm.attempt_index`` 0..N-1. A successful attempt
        (``error_category is None``) carries the full response surface
        with OK status; a failed attempt carries ERROR status + the
        error category and no response attributes.
        """
        # Mid-call metadata augmentation can't reach this span: the
        # typed event arrives only after complete() returns, and the
        # span is back-dated past any augmentation event that fired
        # while the call was in flight. Since complete() is awaited,
        # node bodies can't actually run augmentation mid-call, so
        # this is theoretical only — but it does mean the snapshot
        # on the event is what the span reflects, not a later view.
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_state_for(invocation_id)
        # Back-date start_time using latency_ms so the span's duration
        # reflects the actual adapter-boundary measurement rather than
        # dispatcher queue delay. When latency is missing, fall back to
        # a zero-duration span at end_time.
        end_time_ns = time.time_ns()
        if event.latency_ms is not None:
            start_time_ns = end_time_ns - int(event.latency_ms * 1_000_000)
        else:
            start_time_ns = end_time_ns
        parent_ctx = self._resolve_llm_parent(
            inv_state,
            invocation_id,
            calling_namespace_prefix=event.namespace,
            calling_attempt_index=event.attempt_index,
            calling_fan_out_index=event.fan_out_index,
            calling_branch_name=event.branch_name,
        )
        attrs: dict[str, Any] = {
            "openarmature.llm.model": event.model,
            "openarmature.llm.attempt_index": event.llm_attempt_index,
        }
        cid = current_correlation_id()
        if cid is not None:
            attrs["openarmature.correlation_id"] = cid
        if event.caller_invocation_metadata is not None:
            _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        active_prompt = event.active_prompt
        if active_prompt is not None:
            attrs["openarmature.prompt.name"] = active_prompt.name
            attrs["openarmature.prompt.version"] = active_prompt.version
            attrs["openarmature.prompt.label"] = active_prompt.label
            attrs["openarmature.prompt.template_hash"] = active_prompt.template_hash
            attrs["openarmature.prompt.rendered_hash"] = active_prompt.rendered_hash
        active_group = event.active_prompt_group
        if active_group is not None:
            attrs["openarmature.prompt.group_name"] = active_group.group_name
        if not self.disable_genai_semconv:
            attrs["gen_ai.system"] = event.provider
            attrs["gen_ai.request.model"] = event.model
            request_params = event.request_params or {}
            if "temperature" in request_params:
                attrs["gen_ai.request.temperature"] = request_params["temperature"]
            if "max_tokens" in request_params:
                attrs["gen_ai.request.max_tokens"] = request_params["max_tokens"]
            if "top_p" in request_params:
                attrs["gen_ai.request.top_p"] = request_params["top_p"]
            if "seed" in request_params:
                attrs["gen_ai.request.seed"] = request_params["seed"]
            if "frequency_penalty" in request_params:
                attrs["gen_ai.request.frequency_penalty"] = request_params["frequency_penalty"]
            if "presence_penalty" in request_params:
                attrs["gen_ai.request.presence_penalty"] = request_params["presence_penalty"]
            if "stop_sequences" in request_params:
                attrs["gen_ai.request.stop_sequences"] = request_params["stop_sequences"]
        if not self.disable_provider_payload:
            if event.input_messages:
                serialized = _serialize_for_attribute(event.input_messages)
                attrs["openarmature.llm.input.messages"] = _truncate_for_attribute(
                    serialized, self.payload_max_bytes
                )
            if event.request_extras:
                serialized_extras = _serialize_for_attribute(event.request_extras)
                attrs["openarmature.llm.request.extras"] = _truncate_for_attribute(
                    serialized_extras, self.payload_max_bytes
                )
        span = self._tracer.start_span(
            name="openarmature.llm.complete",
            context=cast("Any", parent_ctx),
            kind=SpanKind.CLIENT,
            attributes=attrs,
            start_time=start_time_ns,
        )
        if event.error_category is not None:
            # Failed attempt: ERROR + the §4 category, no response
            # attributes (no response was received).
            span.set_status(Status(StatusCode.ERROR, description=event.error_category))
            span.set_attribute("openarmature.error.category", event.error_category)
            self._run_enrichers(span, event)
            span.end(end_time=end_time_ns)
            return
        usage = event.usage
        if event.finish_reason is not None:
            span.set_attribute("openarmature.llm.finish_reason", event.finish_reason)
        if usage is not None:
            if usage.prompt_tokens is not None:
                span.set_attribute("openarmature.llm.usage.prompt_tokens", usage.prompt_tokens)
            if usage.completion_tokens is not None:
                span.set_attribute("openarmature.llm.usage.completion_tokens", usage.completion_tokens)
            if usage.total_tokens is not None:
                span.set_attribute("openarmature.llm.usage.total_tokens", usage.total_tokens)
            # Proposal 0047 §5.5.3.1 cache attributes. Absent (None)
            # means the provider didn't report; 0 is "reported miss"
            # and distinct from absent.
            if usage.cached_tokens is not None:
                span.set_attribute("openarmature.llm.cache_read.input_tokens", usage.cached_tokens)
            if usage.cache_creation_tokens is not None:
                span.set_attribute(
                    "openarmature.llm.cache_creation.input_tokens",
                    usage.cache_creation_tokens,
                )
        if not self.disable_genai_semconv:
            if usage is not None:
                if usage.prompt_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                if usage.completion_tokens is not None:
                    span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            if event.finish_reason is not None:
                span.set_attribute("gen_ai.response.finish_reasons", [event.finish_reason])
            if event.response_id is not None:
                span.set_attribute("gen_ai.response.id", event.response_id)
            if event.response_model is not None:
                span.set_attribute("gen_ai.response.model", event.response_model)
        # §5.5.1 output payload. Assistant messages with empty content
        # (tool-call-only responses) MUST NOT emit this attribute per
        # spec — ``output_content`` is already None in that case (see
        # provider.py).
        if not self.disable_provider_payload and event.output_content:
            attrs_out = _truncate_for_attribute(event.output_content, self.payload_max_bytes)
            span.set_attribute("openarmature.llm.output.content", attrs_out)
        # §5.5.10 ungated tool-call identity + §5.5.1 gated full
        # serialization (proposal 0076). The identity projections
        # (count / names / ids) are identifiers, not payload, so they
        # render regardless of disable_provider_payload; the full
        # [{id, name, arguments}] serialization carries the arguments and
        # is gated. The whole family emits only on a tool-calling
        # completion (>= 1 call) — absence means "no tools requested",
        # per the §5.5 omit-when-empty convention.
        output_tool_calls = event.output_tool_calls
        if output_tool_calls:
            # .count / .names / .ids are identity, NOT payload, so they
            # are deliberately untruncated: truncating would break the
            # count == len(.names) invariant and the .names/.ids index-
            # alignment, or sever a .id from its downstream tool execution.
            # The backstop for a pathological call count is the OTel SDK's
            # own SpanLimits, applied uniformly across all attributes.
            span.set_attribute("openarmature.llm.output.tool_calls.count", len(output_tool_calls))
            span.set_attribute(
                "openarmature.llm.output.tool_calls.names",
                [tc.name for tc in output_tool_calls],
            )
            span.set_attribute(
                "openarmature.llm.output.tool_calls.ids",
                [tc.id for tc in output_tool_calls],
            )
            if not self.disable_provider_payload:
                serialized_calls = _serialize_for_attribute(serialize_tool_calls(output_tool_calls))
                span.set_attribute(
                    "openarmature.llm.output.tool_calls",
                    _truncate_for_attribute(serialized_calls, self.payload_max_bytes),
                )
        span.set_status(Status(StatusCode.OK))
        self._run_enrichers(span, event)
        span.end(end_time=end_time_ns)

    def _handle_failure_isolated(self, event: FailureIsolatedEvent) -> None:
        """Emit a zero-duration ``openarmature.failure_isolated`` span for
        a FailureIsolationMiddleware catch.

        Parented under the calling node when its span is still open;
        ``_resolve_llm_parent`` falls back to the invocation span
        otherwise. The wrapped node's span is typically already
        closed-with-error by the time this event is delivered — the
        node-body raise dispatches the node's completed event before the
        middleware recovers — so the marker most often parents directly
        under the invocation span. The wrapped node's name rides on the
        ``openarmature.failure_isolation.node`` attribute for
        correlation regardless of parenting."""
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_state_for(invocation_id)
        parent_ctx = self._resolve_llm_parent(
            inv_state,
            invocation_id,
            calling_namespace_prefix=event.namespace,
            calling_attempt_index=event.attempt_index,
            calling_fan_out_index=event.fan_out_index,
            calling_branch_name=event.branch_name,
        )
        attrs: dict[str, Any] = {
            "openarmature.failure_isolation.event_name": event.event_name,
            "openarmature.failure_isolation.message": event.caught_exception.message,
        }
        if event.namespace:
            attrs["openarmature.failure_isolation.node"] = event.namespace[-1]
        if event.caught_exception.category is not None:
            attrs["openarmature.error.category"] = event.caught_exception.category
        cid = current_correlation_id()
        if cid is not None:
            attrs["openarmature.correlation_id"] = cid
        span = self._tracer.start_span(
            name="openarmature.failure_isolated",
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        # The failure was caught and the node degraded gracefully, so the
        # marker span itself is OK; the caught failure surfaces via the
        # attributes (event_name / category / message), queryable without
        # painting the span red.
        span.set_status(Status(StatusCode.OK))
        self._run_enrichers(span, None)
        span.end()

    def _resolve_llm_parent(
        self,
        inv_state: _InvState,
        invocation_id: str,
        *,
        calling_namespace_prefix: tuple[str, ...],
        calling_attempt_index: int,
        calling_fan_out_index: int | None,
        calling_branch_name: str | None,
    ) -> object:
        """Look up the calling node's span using the calling-node
        identity, fall back through subgraph dispatch / invocation
        span."""
        # 1. Direct match on the calling node's ``_StackKey``.
        calling_key: _StackKey = (
            calling_namespace_prefix,
            calling_attempt_index,
            calling_fan_out_index,
            calling_branch_name,
        )
        calling = inv_state.open_spans.get(calling_key)
        if calling is not None:
            return set_span_in_context(calling.span)
        # 2. Walk up the calling namespace prefix for a synthetic
        #    subgraph dispatch span at any ancestor — covers LLM
        #    calls from inside subgraph wrapper middleware.
        for plen in range(len(calling_namespace_prefix), 0, -1):
            ancestor = calling_namespace_prefix[:plen]
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
            # Proposal 0052 §5.1: implementation attribution attributes.
            # Always-emit on every invocation span; not cross-cutting
            # (§5.6) so inner-node spans don't carry them.
            "openarmature.implementation.name": self.implementation_name,
            "openarmature.implementation.version": self.implementation_version,
            "openarmature.invocation_id": invocation_id,
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        span = self._tracer.start_span(
            name="openarmature.invocation",
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        self._invocation_span[invocation_id] = _OpenSpan(span=span)

    def _key_for(self, event: NodeEvent) -> _StackKey:
        return (event.namespace, event.attempt_index, event.fan_out_index, event.branch_name)

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
        # 1. Walk prefix lengths longest-to-shortest.  The INNERMOST
        #    matching synthetic dispatch span wins.  Three keying
        #    schemes live alongside each other at each prefix:
        #      - per-branch dispatch (proposal 0044, v0.36.0): keyed by
        #        ``prefix + (branch_name,)`` in
        #        ``parallel_branches_branch_spans``
        #      - detached fan-out instance root: keyed by
        #        ``prefix + (str(fan_out_index),)`` in
        #        ``detached_roots``
        #      - non-detached fan-out instance dispatch (proposal 0013,
        #        v0.10.0): keyed by ``prefix + (str(fan_out_index),)``
        #        in ``fan_out_instance_spans``
        #    Walking longest-to-shortest gives the right answer for
        #    arbitrary composition (parallel-branches inside fan-out
        #    instance and vice versa) — the dispatch span at the
        #    deepest matching depth is the most-immediate parent.
        for prefix_len in range(len(event.namespace), 0, -1):
            prefix = event.namespace[:prefix_len]
            if event.branch_name is not None:
                branch_dispatch = inv_state.parallel_branches_branch_spans.get(prefix + (event.branch_name,))
                if branch_dispatch is not None:
                    return set_span_in_context(branch_dispatch.span)
            if event.fan_out_index is not None:
                instance_key = prefix + (str(event.fan_out_index),)
                root = inv_state.detached_roots.get(instance_key)
                if root is not None:
                    return set_span_in_context(root.span)
                instance_dispatch = inv_state.fan_out_instance_spans.get(instance_key)
                if instance_dispatch is not None:
                    return set_span_in_context(instance_dispatch.span)
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
        """Open any synthetic subgraph dispatch spans we need (the
        subgraph wrapper MUST emit a span); close any subgraph spans
        whose prefix is no longer an ancestor of the current event's
        namespace.

        Called from ``_open_started_span`` BEFORE opening the leaf
        node span. Detached-mode entries (subgraph or fan-out instance)
        are registered as detached roots so their inner spans live
        in a fresh trace.
        """
        # Spec observability §4.5: the subgraph wrapper emits a span.
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
            # Per spec proposal 0013 (v0.10.0): for non-detached
            # fan-out instances, the per-instance dispatch span lives
            # at ``prefix + (str(fan_out_index),)`` in
            # ``fan_out_instance_spans``. If the per-instance dispatch
            # span for THIS event's instance is already open, the
            # fan-out node span at ``prefix`` is already open too
            # (synthesized as part of the fan-out node's started
            # event), so we don't open a new subgraph span at
            # ``prefix``.  This dedup runs at any depth — the fan-out
            # node may sit inside a subgraph wrapper, another fan-out,
            # or a parallel-branches branch.
            if (
                event.fan_out_index is not None
                and (prefix + (str(event.fan_out_index),)) in inv_state.fan_out_instance_spans
            ):
                continue
            # If this prefix's first segment is configured as a
            # detached subgraph, mint a fresh trace.
            # Detached subgraph wrapper at any depth: ``detached_subgraphs``
            # holds bare node names, so we match on ``prefix[-1]`` (the
            # node-name segment) at all depths — at depth 1 this
            # coincides with ``prefix[0]`` so the depth-1 behavior is
            # unchanged.
            if prefix[-1] in self.detached_subgraphs:
                self._open_detached_subgraph_root(inv_state, invocation_id, correlation_id, prefix, event)
                continue
            # Per-instance detached root for a configured-detached
            # fan-out (event.fan_out_index populated, fan-out NODE
            # name at ``prefix[-1]`` in the configured set).
            if event.fan_out_index is not None and prefix[-1] in self.detached_fan_outs:
                self._open_detached_fan_out_instance_root(
                    inv_state, invocation_id, correlation_id, prefix, event
                )
                continue
            # Per spec §5.4 + proposal 0013: non-detached fan-out
            # instances get a synthetic per-instance dispatch span
            # under the fan-out node span. Triggered by an event
            # from inside a fan-out instance (event.fan_out_index
            # populated) at the depth where the parent_node_name
            # cache has an entry — i.e., the fan-out node's started
            # event has been seen.  The cache match self-gates to
            # the fan-out node's actual namespace, so this works at
            # any depth (including when the node sits inside a
            # subgraph wrapper or a parallel-branches branch).  The
            # detached check uses ``prefix[-1]`` (the fan-out node
            # name) so it remains correct at depth > 1.
            if (
                event.fan_out_index is not None
                and prefix[-1] not in self.detached_fan_outs
                and prefix in inv_state.fan_out_parent_node_name
            ):
                self._open_fan_out_instance_dispatch_span(inv_state, correlation_id, prefix, event)
                continue
            # Per proposal 0044 (v0.36.0): parallel-branches per-branch
            # dispatch synthesis.  Triggered by an inner-branch event
            # (event.branch_name populated) at the depth where the
            # parent_node_name cache has an entry (i.e., the
            # parallel-branches NODE's started event has been seen).
            # The cache match self-gates to the parallel-branches
            # node's actual namespace, so this works at any depth —
            # including when the node sits inside a subgraph wrapper
            # or a fan-out instance.  If a dispatch span for THIS
            # branch is already open, skip — only one dispatch span
            # per branch per parallel-branches NODE execution.
            if (
                event.branch_name is not None
                and prefix in inv_state.parallel_branches_parent_node_name
                and event.branch_name in inv_state.parallel_branches_branch_names.get(prefix, frozenset())
            ):
                branch_key = prefix + (event.branch_name,)
                if branch_key not in inv_state.parallel_branches_branch_spans:
                    self._open_parallel_branches_branch_dispatch_span(
                        inv_state, correlation_id, prefix, event
                    )
                continue
            # If ``prefix`` names a parallel-branches or fan-out NODE
            # (detected by an entry in the respective parent_node_name
            # cache), don't open a synthetic subgraph wrapper span —
            # the NODE has its own span via ``_open_started_span``.
            # This catches the case where the event walks past a pb /
            # fan-out NODE depth without triggering per-branch /
            # per-instance synthesis at THIS depth (e.g., an inner
            # pb's branch event traversing an outer pb depth where
            # the inner branch_name is not declared).
            if (
                prefix in inv_state.parallel_branches_parent_node_name
                or prefix in inv_state.fan_out_parent_node_name
            ):
                continue
            self._open_subgraph_span(inv_state, invocation_id, correlation_id, prefix, event)

    def _open_subgraph_span(
        self,
        inv_state: _InvState,
        invocation_id: str,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
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
            "openarmature.subgraph.name": _subgraph_identity_at(event, len(prefix)),
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        span = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        # Per proposal 0045: the subgraph wrapper sits at namespace
        # depth ``len(prefix)``; its chain is the slice of the
        # inner event's chain up to that position.  Both chain
        # entries at the wrapper's position are ``None`` (subgraph
        # wrappers don't introduce a fan-out or branch axis).
        inv_state.subgraph_spans[prefix] = _OpenSpan(
            span=span,
            fan_out_index_chain=event.fan_out_index_chain[: len(prefix)],
            branch_name_chain=event.branch_name_chain[: len(prefix)],
        )

    def _close_subgraph_span(self, inv_state: _InvState, prefix: tuple[str, ...]) -> None:
        open_span = inv_state.subgraph_spans.pop(prefix, None)
        if open_span is None:
            return
        # Skip the default OK for a detached dispatch span the §4.2 error
        # path marked ERROR (proposal 0061) — OTel lets a later OK
        # override ERROR (see errored_detached_keys).
        if prefix not in inv_state.errored_detached_keys:
            open_span.span.set_status(Status(StatusCode.OK))
        self._run_enrichers(open_span.span, None)
        open_span.span.end()

    def _detached_invocation_attrs(
        self,
        invocation_id: str,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> dict[str, Any]:
        # Proposal 0061 §5.1 attribute set for a detached invocation
        # span. Mirrors ``_open_invocation_span`` but carries the SAME
        # invocation_id as the parent (detached mode shares the run
        # identity per §4.3 — distinct from checkpoint-resume, which
        # mints a fresh id) and the detached unit's OWN entry node: the
        # namespace segment immediately after the detached prefix, which
        # is the entry node of the outermost graph OF THE DETACHED TRACE
        # (the detached subgraph, or the fan-out instance subgraph).
        # Callers always pass a strict-ancestor prefix (the ancestor walk
        # in ``_sync_subgraph_spans`` only fires at depths below the
        # event's namespace length), so ``len(prefix) < len(namespace)``
        # and the index is always in range.
        attrs: dict[str, Any] = {
            "openarmature.graph.entry_node": event.namespace[len(prefix)],
            "openarmature.graph.spec_version": self.spec_version,
            "openarmature.implementation.name": self.implementation_name,
            "openarmature.implementation.version": self.implementation_version,
            "openarmature.invocation_id": invocation_id,
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        return attrs

    def _open_detached_subgraph_root(
        self,
        inv_state: _InvState,
        invocation_id: str,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
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

        # 2. Open the dispatch span in the parent trace. At depth 1
        #    the parent is the invocation span; at depth > 1 it's the
        #    nearest enclosing synthetic subgraph or detached root
        #    span (mirroring ``_open_subgraph_span``'s walk).
        parent_ctx_for_dispatch: object = otel_context.Context()
        for plen in range(len(prefix) - 1, 0, -1):
            outer = prefix[:plen]
            outer_sg = inv_state.subgraph_spans.get(outer)
            if outer_sg is not None:
                parent_ctx_for_dispatch = set_span_in_context(outer_sg.span)
                break
            outer_dr = inv_state.detached_roots.get(outer)
            if outer_dr is not None:
                parent_ctx_for_dispatch = set_span_in_context(outer_dr.span)
                break
        else:
            inv = self._invocation_span.get(invocation_id)
            if inv is not None:
                parent_ctx_for_dispatch = set_span_in_context(inv.span)
        attrs_parent: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.subgraph.name": _subgraph_identity_at(event, len(prefix)),
        }
        if correlation_id is not None:
            attrs_parent["openarmature.correlation_id"] = correlation_id
        _apply_caller_metadata(attrs_parent, event.caller_invocation_metadata)
        parent_dispatch = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", parent_ctx_for_dispatch),
            kind=SpanKind.INTERNAL,
            links=[Link(detached_sc)],
            attributes=attrs_parent,
        )
        inv_state.subgraph_spans[prefix] = _OpenSpan(span=parent_dispatch)

        # 3. Proposal 0061: the detached trace roots in its OWN
        #    ``openarmature.invocation`` span (parented to the synthetic
        #    detached SpanContext so OTel uses the new trace_id). It
        #    carries the §5.1 invocation-span attribute set with the
        #    SAME invocation_id as the parent — detached mode is
        #    observer-side trace rendering, not a new run (§4.3) — so
        #    the §5.1 always-emit attribution invariant lands on the
        #    detached trace's root with no per-context caveat.
        detached_parent_ctx = otel_trace.set_span_in_context(
            NonRecordingSpan(detached_sc), otel_context.Context()
        )
        detached_invocation = self._tracer.start_span(
            name="openarmature.invocation",
            context=cast("Any", detached_parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=self._detached_invocation_attrs(invocation_id, correlation_id, prefix, event),
        )
        # A fresh detached invocation span at this prefix starts clean:
        # discard any ERROR marker a prior generation left here (cyclic
        # / fire-and-forget re-entry at the same prefix). Keys are only
        # ever added while such a span is open, so this open is the one
        # place stale state could otherwise persist; clearing it keeps
        # the synthetic close paths reflecting only this generation.
        inv_state.errored_detached_keys.discard(prefix)
        inv_state.detached_invocation_spans[prefix] = _OpenSpan(span=detached_invocation)

        # 4. Open the detached subgraph span as a child of the detached
        #    invocation span (normal §4.3 nesting within the detached
        #    trace). Inner-node spans continue to parent under THIS span
        #    via ``detached_roots`` — the invocation span sits above it.
        attrs_root: dict[str, Any] = dict(attrs_parent)
        attrs_root["openarmature.subgraph.detached"] = True
        detached_root = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", set_span_in_context(detached_invocation)),
            kind=SpanKind.INTERNAL,
            attributes=attrs_root,
        )
        inv_state.detached_roots[prefix] = _OpenSpan(span=detached_root)

    def _open_detached_fan_out_instance_root(
        self,
        inv_state: _InvState,
        invocation_id: str,
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

        # Proposal 0061: each detached instance trace roots in its OWN
        # ``openarmature.invocation`` span (shared parent invocation_id,
        # the instance subgraph's entry node), with the fan-out instance
        # span nested under it. Keyed by prefix + (str(fan_out_index),)
        # so per-instance roots stay distinct.
        instance_key = prefix + (str(event.fan_out_index),)
        detached_parent_ctx = otel_trace.set_span_in_context(
            NonRecordingSpan(detached_sc), otel_context.Context()
        )
        detached_invocation = self._tracer.start_span(
            name="openarmature.invocation",
            context=cast("Any", detached_parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=self._detached_invocation_attrs(invocation_id, correlation_id, prefix, event),
        )
        # Clear any stale ERROR marker for this instance key before the
        # fresh span opens (see the detached-subgraph open path) so a
        # re-run of the same instance starts clean.
        inv_state.errored_detached_keys.discard(instance_key)
        inv_state.detached_invocation_spans[instance_key] = _OpenSpan(span=detached_invocation)

        # Open the detached instance root span as a child of the
        # per-instance invocation span.
        attrs: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.fan_out.parent_node_name": prefix[-1],
            "openarmature.node.fan_out_index": event.fan_out_index,
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        instance_root = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", set_span_in_context(detached_invocation)),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        inv_state.detached_roots[instance_key] = _OpenSpan(span=instance_root)
        inv_state.fan_out_instance_root_prefixes.add(instance_key)

    def _open_fan_out_instance_dispatch_span(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        """Per-instance dispatch span for a non-detached fan-out.
        Mirror of ``_open_detached_fan_out_instance_root`` but lives in
        the parent trace (no fresh trace_id).

        Parents under the fan-out node span at ``prefix``. Span name
        is the fan-out node's name; attributes are
        ``openarmature.fan_out.parent_node_name`` (looked up from the
        cache populated when the fan-out node's ``started`` event
        landed), ``openarmature.node.fan_out_index`` (from
        ``event.fan_out_index``), and the correlation_id if set.

        Stored in ``inv_state.fan_out_instance_spans`` keyed by
        ``prefix + (str(fan_out_index),)``. Closed when the fan-out
        node's own ``completed`` event fires (children-before-parents
        ordering).
        """
        # Find the fan-out node's open span (latest attempt under
        # retry) to use as parent.
        fan_out_open = self._find_fan_out_node_span(inv_state, prefix)
        parent_ctx: object
        if fan_out_open is not None:
            parent_ctx = set_span_in_context(fan_out_open.span)
        else:
            parent_ctx = otel_context.Context()

        parent_node_name = inv_state.fan_out_parent_node_name.get(prefix, prefix[-1])
        attrs: dict[str, Any] = {
            "openarmature.node.name": prefix[-1],
            "openarmature.fan_out.parent_node_name": parent_node_name,
            "openarmature.node.fan_out_index": event.fan_out_index,
            "openarmature.subgraph.name": _subgraph_identity_at(event, len(prefix)),
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        instance_span = self._tracer.start_span(
            name=prefix[-1],
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        instance_key = prefix + (str(event.fan_out_index),)
        # Per proposal 0045: this dispatch span sits AT the descent
        # boundary into the fan-out instance.  Its chain is the slice
        # of the inner event's chain up to and including the
        # boundary's own position.
        chain_len = len(prefix)
        inv_state.fan_out_instance_spans[instance_key] = _OpenSpan(
            span=instance_span,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )

    def _close_fan_out_instance_dispatch_span(self, inv_state: _InvState, key: tuple[str, ...]) -> None:
        open_span = inv_state.fan_out_instance_spans.pop(key, None)
        if open_span is None:
            return
        open_span.span.set_status(Status(StatusCode.OK))
        self._run_enrichers(open_span.span, None)
        open_span.span.end()

    def _open_parallel_branches_branch_dispatch_span(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        """Per-branch dispatch span for a parallel-branches NODE.
        Mirror of ``_open_fan_out_instance_dispatch_span``.

        Parents under the parallel-branches node span at ``prefix``.
        Span name is the branch's identifier (``event.branch_name``).
        Attributes are ``openarmature.node.branch_name``,
        ``openarmature.parallel_branches.parent_node_name`` (looked up
        from the cache populated when the parallel-branches NODE's
        ``started`` event landed), and the correlation_id if set.

        Stored in ``inv_state.parallel_branches_branch_spans`` keyed by
        ``prefix + (branch_name,)``.  Closed when the parallel-branches
        NODE's own ``completed`` event fires (children-before-parents
        ordering).
        """
        assert event.branch_name is not None, (
            "parallel-branches branch dispatch synthesis requires event.branch_name"
        )
        # Find the parallel-branches NODE's open span (latest attempt
        # under retry) to use as parent.  Scan ``open_spans`` by
        # namespace only — the NODE may carry an outer
        # ``fan_out_index`` (if the parallel-branches node sits inside
        # a fan-out instance) or ``branch_name`` (if it sits inside
        # another parallel-branches branch).  Only one entry per
        # namespace is open at a time, so the scan is unambiguous.
        node_open: _OpenSpan | None = None
        for key, open_span in inv_state.open_spans.items():
            ns, _attempt, _fan_idx, _bn = key
            if ns == prefix:
                node_open = open_span
                break
        parent_ctx: object
        if node_open is not None:
            parent_ctx = set_span_in_context(node_open.span)
        else:
            parent_ctx = otel_context.Context()

        parent_node_name = inv_state.parallel_branches_parent_node_name.get(prefix, prefix[-1])
        attrs: dict[str, Any] = {
            "openarmature.node.name": event.branch_name,
            "openarmature.node.branch_name": event.branch_name,
            "openarmature.parallel_branches.parent_node_name": parent_node_name,
            "openarmature.subgraph.name": _subgraph_identity_at(event, len(prefix)),
        }
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        branch_span = self._tracer.start_span(
            name=event.branch_name,
            context=cast("Any", parent_ctx),
            kind=SpanKind.INTERNAL,
            attributes=attrs,
        )
        branch_key = prefix + (event.branch_name,)
        # Per proposal 0045: this dispatch span sits at the descent
        # boundary into this branch.  Its chain is the slice up to
        # and including the boundary's own position.
        chain_len = len(prefix)
        inv_state.parallel_branches_branch_spans[branch_key] = _OpenSpan(
            span=branch_span,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )

    def _close_parallel_branches_branch_dispatch_span(
        self, inv_state: _InvState, key: tuple[str, ...]
    ) -> None:
        open_span = inv_state.parallel_branches_branch_spans.pop(key, None)
        if open_span is None:
            return
        open_span.span.set_status(Status(StatusCode.OK))
        self._run_enrichers(open_span.span, None)
        open_span.span.end()

    def _close_detached_root(self, inv_state: _InvState, prefix: tuple[str, ...]) -> None:
        inv_state.fan_out_instance_root_prefixes.discard(prefix)
        open_span = inv_state.detached_roots.pop(prefix, None)
        if open_span is not None:
            if prefix not in inv_state.errored_detached_keys:
                open_span.span.set_status(Status(StatusCode.OK))
            self._run_enrichers(open_span.span, None)
            open_span.span.end()
        # Proposal 0061: close the paired detached invocation span (the
        # parent of the detached root within the detached trace) AFTER
        # the root — children before parents.
        self._close_detached_invocation_span(inv_state, prefix)

    def _close_detached_invocation_span(self, inv_state: _InvState, prefix: tuple[str, ...]) -> None:
        open_span = inv_state.detached_invocation_spans.pop(prefix, None)
        if open_span is None:
            return
        # Skip the default OK when the §4.2 error path marked this key
        # ERROR (proposal 0061) — OTel lets a later OK override ERROR
        # (see errored_detached_keys); an UNSET close maps to OK by
        # exporter convention.
        if prefix not in inv_state.errored_detached_keys:
            open_span.span.set_status(Status(StatusCode.OK))
        self._run_enrichers(open_span.span, None)
        open_span.span.end()

    def _drain_open_span(self, open_span: _OpenSpan) -> None:
        """Close an open span as an orphan during shutdown: OK
        status, end. No paired completed event will arrive, so we
        don't have an error category to record. Enrichers run with
        ``event=None`` — they can no-op when event context matters."""
        open_span.span.set_status(Status(StatusCode.OK))
        self._run_enrichers(open_span.span, None)
        open_span.span.end()

    def _find_fan_out_node_span(self, inv_state: _InvState, prefix: tuple[str, ...]) -> _OpenSpan | None:
        """Find the currently-open fan-out NODE span at ``prefix``.
        Scans by namespace only — the NODE may carry an outer
        ``fan_out_index`` or ``branch_name`` if the fan-out is itself
        nested inside another fan-out instance or a parallel-branches
        branch.  Only one entry per namespace is open at a time
        (retry middleware opens and closes attempts serially), so the
        scan is unambiguous."""
        for key, open_span in inv_state.open_spans.items():
            ns, _attempt, _fan_idx, _bn = key
            if ns == prefix:
                return open_span
        return None

    def _node_attrs(self, event: NodeEvent, correlation_id: str | None) -> dict[str, Any]:
        """Build the attribute set for a node span."""
        attrs: dict[str, Any] = {
            "openarmature.node.name": event.node_name,
            "openarmature.node.namespace": list(event.namespace),
            "openarmature.node.step": event.step,
            "openarmature.node.attempt_index": event.attempt_index,
        }
        if event.fan_out_index is not None:
            attrs["openarmature.node.fan_out_index"] = event.fan_out_index
        # Per observability §5.7 + proposal 0044 (v0.36.0): the
        # ``openarmature.node.branch_name`` attribute MUST appear on
        # every inner-node span within a parallel-branches branch
        # (parallels ``openarmature.node.fan_out_index`` on fan-out
        # instance inner spans).  Independent of ``fan_out_index`` —
        # both MAY be present when a branch's subgraph contains a
        # fan-out, and §6 treats them as independent identification
        # slots.  Pre-0044 the python observer emitted this as
        # ``openarmature.branch_name``; renamed to match the spec
        # attribute namespace.
        if event.branch_name is not None:
            attrs["openarmature.node.branch_name"] = event.branch_name
        if correlation_id is not None:
            attrs["openarmature.correlation_id"] = correlation_id
        # Per spec §5.4 + proposal 0013 (v0.10.0): fan-out node spans
        # carry item_count / concurrency / error_policy. ``concurrency``
        # is ``int | None`` on the event payload (spec §9.2 canonical
        # type); the §5.4 attribute is a bare int with ``0`` as the
        # "unbounded" sentinel (OTel attribute primitives can't carry
        # null cleanly), so this is the OTel-attribute-layer
        # translation.
        if event.fan_out_config is not None:
            cfg = event.fan_out_config
            attrs["openarmature.fan_out.item_count"] = cfg.item_count
            attrs["openarmature.fan_out.concurrency"] = 0 if cfg.concurrency is None else cfg.concurrency
            attrs["openarmature.fan_out.error_policy"] = cfg.error_policy
        # Per spec §5.7 + proposal 0044 (v0.36.0): the parallel-
        # branches NODE span carries branch_count + error_policy.
        # Both come straight off ``parallel_branches_config`` which
        # the engine attaches to the NODE's started/completed events
        # only (analogous to fan_out_config).
        if event.parallel_branches_config is not None:
            pcfg = event.parallel_branches_config
            attrs["openarmature.parallel_branches.branch_count"] = pcfg.branch_count
            attrs["openarmature.parallel_branches.error_policy"] = pcfg.error_policy
        _apply_caller_metadata(attrs, event.caller_invocation_metadata)
        return attrs

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_invocation(self, invocation_id: str) -> None:
        """Close the invocation span for ``invocation_id`` and drain
        the per-invocation state. Idempotent; calling twice (or for
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
        prefer :meth:`shutdown`; it drains every in-flight
        invocation in one call without needing to track ids
        externally. A first-class engine-level signal that lets
        observers auto-drain per-invocation state on completion is
        tracked as follow-up work in
        ``openarmature-coord/docs/phase-6-1-conformance-fillin.md``.
        """
        inv_state = self._inv_states.pop(invocation_id, None)
        if inv_state is not None:
            self._drain_inv_state(inv_state)
        self._close_invocation_span(invocation_id)

    def _drain_inv_state(self, inv_state: _InvState) -> None:
        """Close any still-open spans in a per-invocation state
        container in child→parent order. Leaf node spans (sorted
        deepest-first by namespace) → non-detached fan-out per-instance
        dispatch spans → detached roots → subgraph dispatch spans.
        Matches the ordering used in ``shutdown``. LLM spans don't
        appear here — both the success and error paths open + close
        the span in one shot at handler-time, so there are no in-flight
        LLM spans to drain."""
        # Inner-node spans (depth >= 2) drain first — these include
        # the inner-node bodies inside fan-out instances, which are
        # children of the per-instance dispatch spans.
        for key in sorted(
            (k for k in inv_state.open_spans if len(k[0]) >= 2),
            key=lambda k: -len(k[0]),
        ):
            open_span = inv_state.open_spans.pop(key, None)
            if open_span is not None:
                self._drain_open_span(open_span)
        # Per-instance dispatch spans (children of the fan-out NODE
        # span) drain BEFORE depth-1 entries of ``open_spans`` —
        # otherwise the fan-out NODE span (depth 1) would end
        # before its per-instance children, violating
        # children-before-parents close ordering.
        for key in sorted(inv_state.fan_out_instance_spans.keys(), key=lambda k: -len(k)):
            self._close_fan_out_instance_dispatch_span(inv_state, key)
        # Remaining depth-1 entries in ``open_spans`` drain last —
        # these are the fan-out NODE span itself + any sibling
        # depth-1 leaf nodes still open.
        for key in list(inv_state.open_spans.keys()):
            open_span = inv_state.open_spans.pop(key, None)
            if open_span is not None:
                self._drain_open_span(open_span)
        for prefix in sorted(inv_state.detached_roots.keys(), key=lambda k: -len(k)):
            self._close_detached_root(inv_state, prefix)
        # Defensive: _close_detached_root closes the paired invocation
        # span, but a re-entry pop (or a partial open) could orphan one.
        for prefix in sorted(inv_state.detached_invocation_spans.keys(), key=lambda k: -len(k)):
            self._close_detached_invocation_span(inv_state, prefix)
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
        self._run_enrichers(open_span.span, None)
        open_span.span.end()

    def force_flush(self, timeout_ms: int = 30_000) -> bool:
        """Flush any pending spans through every registered span processor.

        Returns ``True`` when all processors finish flushing within the
        deadline, ``False`` otherwise. Wraps the underlying OTel
        :class:`TracerProvider`'s ``force_flush`` so callers don't have
        to reach into the private ``_provider`` attribute.

        **When to call.** Distinct from :meth:`drain` on
        :class:`~openarmature.graph.compiled.CompiledGraph` (which covers
        the engine's observer-event queue): this method covers the
        outbound span-export buffer of each registered
        :class:`SpanProcessor`. Under fast or unusual teardown
        orderings (FastAPI ``TestClient`` teardown, CLI one-shots,
        serverless functions) the :class:`BatchSpanProcessor`'s export
        thread can be cut off before its buffer drains; calling
        ``force_flush()`` from a ``finally`` block right before process
        exit is the canonical hardening.

        The default 30 s ``timeout_ms`` matches the OTel SDK's own
        default. Pass a smaller value when running under a hard
        deadline (a serverless function's max execution time, an
        ASGI lifespan timeout).
        """
        return self._provider.force_flush(timeout_millis=timeout_ms)

    def shutdown(self) -> None:
        """Close any still-open spans across all in-flight invocations
        and shut down the underlying provider. Each per-invocation
        state is drained in child→parent order (LLM spans → leaf spans
        → detached roots → subgraph dispatch); invocation spans drain
        last. Idempotent.

        **BatchSpanProcessor flush note.** ``self._provider.shutdown()``
        flushes every registered processor. Under fast or unusual
        teardown orderings (e.g., FastAPI ``TestClient`` teardown that
        closes the event loop before the BatchSpanProcessor's export
        thread finishes), the flush may not complete in time and spans
        can appear dropped. Workarounds:

        - Call :meth:`force_flush` explicitly before this method.
        - Use :class:`SimpleSpanProcessor` instead of
          :class:`BatchSpanProcessor` in tests; it exports synchronously
          and is unaffected by teardown timing.
        """
        for invocation_id in list(self._inv_states.keys()):
            inv_state = self._inv_states.pop(invocation_id)
            self._drain_inv_state(inv_state)
        for invocation_id in list(self._invocation_span.keys()):
            self._close_invocation_span(invocation_id)
        self._provider.shutdown()


__all__ = [
    "OTelObserver",
]
