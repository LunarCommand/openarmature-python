# Spec mapping (observability §8):
# - Consumes the §6 observer event stream as a sibling to the OTel
#   observer (§8.9 composition).
# - Maps invocation → Trace, node/subgraph/fan-out → Span observation,
#   LLM provider → Generation observation (§8.3 table).
# - Sets the Trace `id` equal to the OA `invocation_id` so cross-system
#   lookup by invocation_id finds the Langfuse Trace verbatim (§8.4.1).
# - Routes correlation_id to both `trace.metadata.correlation_id` and
#   every `observation.metadata.correlation_id` per §8.5.
# - Sources Trace name from the entry-node name (§8.6 fallback). The
#   caller-supplied invocation-label path lands in proposal 0034 (PR 4
#   of the v0.10.0 batch).
# - Generation rendering follows §8.7: input/output/request_extras
#   appear only when `disable_provider_payload=False`; the truncation
#   marker is preserved verbatim as a raw string when the §5.5.5
#   truncation makes the JSON unparseable.
# - Prompt linkage follows §8.4.4: reads
#   `Prompt.observability_entities["langfuse_prompt"]` to establish a
#   native Prompt-entity link when present; metadata-only otherwise.

"""LangfuseObserver: maps OA events to Langfuse Traces + Observations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from openarmature.graph.events import (
    EmbeddingEvent,
    EmbeddingFailedEvent,
    FailureIsolatedEvent,
    InvocationCompletedEvent,
    InvocationStartedEvent,
    LlmCompletionEvent,
    LlmFailedEvent,
    LlmRetryAttemptEvent,
    MetadataAugmentationEvent,
    NodeEvent,
    RerankEvent,
    RerankFailedEvent,
    ToolCallEvent,
    ToolCallFailedEvent,
)
from openarmature.graph.observer import ObserverEvent
from openarmature.observability.lineage import is_strict_prefix

from .client import (
    LangfuseClient,
    LangfuseGenerationHandle,
    LangfuseSpanHandle,
    LangfuseUsage,
    ObservationLevel,
)

# §5.5.5 / §8.7 truncation: when the serialized payload exceeds the
# configured cap, the marker below is appended and the unparseable
# JSON serves as the "this was truncated" signal in Langfuse's input
# / output / metadata.request_extras fields.
_TRUNCATION_MARKER_TEMPLATE = "…[truncated, {m} bytes total]"

# §5.5.5 minimum-cap rule mirrors the OTel observer's bound. 256 bytes
# is the smallest value that fits the worst-case marker (~36 bytes)
# plus a diagnostically useful preview.
_PAYLOAD_MIN_BYTES = 256


def _read_spec_version() -> str:
    """Lazy spec-version read; mirrors the OTel observer's lookup so
    Langfuse-side spec_version metadata stays in lockstep."""
    from openarmature import __spec_version__

    return __spec_version__


# Proposal 0052: implementation attribution attributes. Sourced from
# the package identity constants via the same lazy-import discipline
# as ``_read_spec_version``.
def _read_implementation_name() -> str:
    from openarmature import __implementation_name__

    return __implementation_name__


def _read_implementation_version() -> str:
    from openarmature import __version__

    return __version__


# In-flight Span observation handle, keyed by the standard span-stack
# key (namespace, attempt_index, fan_out_index, branch_name).
# ``branch_name`` discriminates concurrent same-named inner nodes
# across sibling parallel-branches branches (pipeline-utilities §11);
# without it the two inner ``ask`` nodes of two branches with the
# same namespace + fan_out_index would collide on the same key.
# Mirrors the OTel observer's ``_StackKey`` shape but holds a
# Langfuse handle instead of an OTel Span.
_StackKey = tuple[tuple[str, ...], int, int | None, str | None]

# Lineage-aware dispatch keys (proposal 0045): the fan-out / pb NODE namespace
# prefix plus the fan-out instance index / branch name chain slices along the
# path to it. _BranchDispatchKey carries the explicit branch name (a callable
# branch sets it without extending the chain) instead of a trailing chain entry.
_DispatchKey = tuple[tuple[str, ...], tuple[int | None, ...], tuple[str | None, ...]]
_BranchDispatchKey = tuple[tuple[str, ...], tuple[int | None, ...], tuple[str | None, ...], str]


@dataclass
class _OpenObservation:
    """An in-flight Langfuse observation pinned in the observer's state.

    Carries the observation's own ``fan_out_index_chain`` and
    ``branch_name_chain`` so the augmentation walk can apply the
    lineage-aware boundary rule (mirror of the OTel observer's
    ``_OpenSpan``)."""

    handle: LangfuseSpanHandle | LangfuseGenerationHandle
    fan_out_index_chain: tuple[int | None, ...] = ()
    branch_name_chain: tuple[str | None, ...] = ()


def _observation_chain_on_path(
    open_obs: _OpenObservation,
    aug_fi_chain: tuple[int | None, ...],
    aug_bn_chain: tuple[str | None, ...],
) -> bool:
    """Mirror of the OTel observer's ``_span_chain_on_path`` for
    Langfuse observations.  Returns True iff the observation's chain
    is a prefix-match of the augmenter's chain."""
    obs_fi = open_obs.fan_out_index_chain
    obs_bn = open_obs.branch_name_chain
    if len(obs_fi) > len(aug_fi_chain):
        return False
    if len(obs_bn) > len(aug_bn_chain):
        return False
    for i in range(len(obs_fi)):
        if obs_fi[i] != aug_fi_chain[i]:
            return False
    for i in range(len(obs_bn)):
        if obs_bn[i] != aug_bn_chain[i]:
            return False
    return True


def _dispatch_key(
    prefix: tuple[str, ...],
    fan_out_index_chain: tuple[int | None, ...],
    branch_name_chain: tuple[str | None, ...],
) -> _DispatchKey:
    """Lineage-aware identity key for a fan-out instance / per-branch dispatch at
    namespace ``prefix``. Encodes the fan-out/pb NODE namespace plus the full
    chain of fan-out instance indices / branch names along the path to it (sliced
    to ``len(prefix)``). Two dispatches at the same namespace but in different
    enclosing fan-out instances / branches therefore get distinct keys -- the
    enclosing chain entries differ -- which is what lets a fan-out / pb nested
    inside an outer fan-out instance avoid colliding across outer instances. For
    a top-level or serial-nested dispatch (no enclosing fan-out/branch) the
    enclosing chain entries are all None, so the key is a stable function of the
    namespace plus the dispatch's own axis."""
    n = len(prefix)
    return (prefix, tuple(fan_out_index_chain[:n]), tuple(branch_name_chain[:n]))


def _branch_dispatch_key(
    prefix: tuple[str, ...],
    fan_out_index_chain: tuple[int | None, ...],
    branch_name_chain: tuple[str | None, ...],
    branch_name: str,
) -> _BranchDispatchKey:
    """Lineage-aware identity key for a per-branch dispatch at namespace
    ``prefix``. The branch IDENTITY comes from ``branch_name`` explicitly (not
    ``branch_name_chain[-1]``): a callable branch carries its name on the event
    but never extends ``branch_name_chain`` (no subgraph descent). The key still
    carries the ENCLOSING fan-out instance / branch chain (positions above this
    pb node) so a pb nested inside an outer fan-out instance doesn't collide
    across outer instances."""
    n = len(prefix)
    return (prefix, tuple(fan_out_index_chain[:n]), tuple(branch_name_chain[: n - 1]), branch_name)


def _empty_str_frozenset() -> frozenset[str]:
    """Typed empty frozenset factory for ``detached_subgraphs`` /
    ``detached_fan_outs`` defaults."""
    return frozenset()


def _apply_caller_metadata(metadata: dict[str, Any], caller_metadata: Mapping[str, Any]) -> None:
    """Merge caller-supplied invocation metadata into a Trace's or
    Observation's metadata bag at top level.

    Top-level placement lets the Langfuse UI filter on
    ``metadata.<key>`` directly, so caller-supplied entries become
    siblings to ``correlation_id`` / ``entry_node`` rather than
    nested under a ``user`` sub-object.

    Reserved-key collision with the OA-emitted keys
    (``correlation_id``, ``entry_node``, ``spec_version``,
    ``namespace``, etc.) is not currently checked here: the rejection
    may happen at either boundary, and the ``invoke()`` API-boundary
    validation already rejects ``openarmature.*`` / ``gen_ai.*``
    prefixed keys. Per-Langfuse-backend collision rejection is queued
    as a follow-up.
    """
    # Spec observability §8.4.1 / §8.4.2 (proposal 0034): top-level
    # placement of caller-supplied metadata on the Trace / Observation.
    for key, value in caller_metadata.items():
        metadata[key] = value


def _promoted_user_id(metadata: Mapping[str, Any]) -> str | None:
    # Proposal 0064 §8.4.1: a recognized ``userId`` caller-metadata key
    # promotes to Langfuse's first-class trace.userId (recognized, not
    # reserved; automatic, not opt-in). Read from the already-merged trace
    # metadata, so the promotion is additive -- the key also remains at
    # trace.metadata.userId. Absent key -> None (trace.userId unset).
    value = metadata.get("userId")
    return str(value) if value is not None else None


def _subgraph_identity_at(event: NodeEvent, depth: int) -> str:
    """Return the compiled-subgraph identity for the wrapper at the
    given 1-based namespace depth, or the empty string when no
    identity is tracked at that depth.

    The empty-string fallback is the "no identity tracked" case, for
    implementations / direct ``SubgraphNode(...)`` callers that don't
    wire an identity through.
    Conformance fixtures 031/032/033 lock identity as the required
    value; the empty-string path keeps direct callers conformant but
    failing those fixtures.
    """
    # Spec observability §5.3 (coord thread
    # clarify-subgraph-name-semantics): empty-string fallback is
    # conformant for callers that don't track a subgraph identity.
    idx = depth - 1
    if 0 <= idx < len(event.subgraph_identities):
        identity = event.subgraph_identities[idx]
        if identity is not None:
            return identity
    return ""


@dataclass
class _InvState:
    """Per-invocation state, isolated by invocation_id.

    A single LangfuseObserver is safe to share across concurrent
    invocations; each invocation's in-flight observations live under
    its own _InvState so they never collide.
    """

    trace_id: str
    open_observations: dict[_StackKey, _OpenObservation] = field(
        default_factory=dict[_StackKey, _OpenObservation]
    )
    # Synthetic subgraph dispatch Span observations, keyed by namespace
    # prefix. Per spec §8.3 each subgraph wrapper produces a Span
    # observation in its parent's Trace; descendant node observations
    # parent under it. For a detached subgraph, this dictionary holds
    # the dispatch Span observation that lives in the DETACHED Trace
    # (so descendants in that subtree parent under it via the detached
    # Trace's observation tree); the main Trace carries a separate
    # link observation surfacing metadata.detached_child_trace_ids
    # that's opened and closed in one shot, not tracked here.
    subgraph_observations: dict[tuple[str, ...], _OpenObservation] = field(
        default_factory=dict[tuple[str, ...], _OpenObservation]
    )
    # Per-instance fan-out dispatch Span observations (non-detached),
    # keyed by ``prefix + (str(fan_out_index),)``. Parents under the
    # fan-out node's own Span observation; inner-node observations
    # parent under this dispatch instead of the shared fan-out node
    # span. Closed when the fan-out node's completed event fires.
    fan_out_instance_observations: dict[_DispatchKey, _OpenObservation] = field(
        default_factory=dict[_DispatchKey, _OpenObservation]
    )
    # Maps a namespace prefix to the detached Langfuse trace_id when
    # that subtree is configured detached (per the observer's
    # ``detached_subgraphs`` / ``detached_fan_outs`` knobs). The
    # presence of a prefix here switches descendant observations onto
    # the detached Trace.
    detached_traces: dict[tuple[str, ...], str] = field(default_factory=dict[tuple[str, ...], str])
    # Set of detached fan-out instance prefixes
    # (``prefix + (str(fan_out_index),)``) — distinguished from
    # detached subgraph prefixes because they're closed when the
    # fan-out node's completed event fires, not when the namespace
    # cursor leaves the subtree.
    fan_out_instance_root_prefixes: set[_DispatchKey] = field(default_factory=set[_DispatchKey])
    # ``parent_node_name`` cache for per-instance attribution
    # (spec proposal 0013 v0.10.0 — inner events from inside a
    # non-detached fan-out instance don't carry fan_out_config
    # themselves; the cache bridges the lookup so the synthetic
    # per-instance dispatch observation can attach
    # metadata.fan_out_parent_node_name).
    fan_out_parent_node_name: dict[tuple[str, ...], str] = field(default_factory=dict[tuple[str, ...], str])
    # Per proposal 0045: structural identification of parallel-
    # branches NODE namespaces.  Populated on a pb NODE's started
    # event (whichever events carry ``parallel_branches_config``);
    # consulted by the augmentation walk to skip the pb NODE itself
    # as a shared parent (§3.4's structural classification).
    parallel_branches_parent_node_name: dict[tuple[str, ...], str] = field(
        default_factory=dict[tuple[str, ...], str]
    )
    # Per proposal 0044: per-branch dispatch-span observations synthesized from
    # the first inner event of each branch, keyed ``prefix + (branch_name,)``
    # (prefix = the parallel-branches NODE namespace). Inner branch-node
    # observations parent under this dispatch instead of the shared pb NODE
    # span; closed when the pb NODE's completed event fires.
    parallel_branches_branch_spans: dict[_BranchDispatchKey, _OpenObservation] = field(
        default_factory=dict[_BranchDispatchKey, _OpenObservation]
    )
    # Declared branch-name set per pb NODE namespace (from the NODE's
    # parallel_branches_config), so an inner branch event matches only the node
    # that actually declares its branch.
    parallel_branches_branch_names: dict[tuple[str, ...], frozenset[str]] = field(
        default_factory=dict[tuple[str, ...], frozenset[str]]
    )
    # Side-cache: accumulator for `metadata.detached_child_trace_ids`
    # on dispatch observations that spawn detached children. Keyed by
    # the dispatch observation's prefix (the fan-out node's namespace,
    # or the detached-subgraph parent's prefix). Each new detached
    # child append-then-snapshot lets us preserve §8.5's string-array
    # shape across multiple instances without re-reading metadata
    # from the client (the Protocol doesn't expose a read accessor).
    detached_child_trace_ids: dict[tuple[str, ...], list[str]] = field(
        default_factory=dict[tuple[str, ...], list[str]]
    )


@dataclass
class LangfuseObserver:
    """Observer-driven Langfuse mapping.

    Construct with a :class:`LangfuseClient` — the bundled
    :class:`InMemoryLangfuseClient` for tests, or a real
    ``langfuse.Langfuse()`` instance for production. The observer
    handles the event stream and emits Trace + Observation entities
    through the client.

    Constructor knobs:

    - ``client``: the Langfuse sink (Protocol-typed).
    - ``disable_llm_spans``: when ``True`` the observer skips
      Generation observations on LLM provider events.
    - ``disable_provider_payload``: default ``True`` for a symmetric
      privacy posture with the OTel observer. Gates
      ``generation.input`` / ``output`` / ``metadata.request_extras``
      emission. The name carries the broadened provider-payload scope;
      LLM completion is OA's only provider-call payload today.
    - ``payload_byte_cap``: per-attribute byte cap on the source
      payload string before parse-back. Mirrors the OTel observer's
      ``payload_max_bytes`` semantic — emission preserves the raw
      truncated string when the truncation marker is present. Default
      64 KiB; same minimum (256 bytes) applies.
    - ``detached_subgraphs``: set of subgraph wrapper node names that
      run in their own Langfuse Trace. Each such subgraph gets a fresh
      trace_id; the main Trace's dispatch observation surfaces the link
      via ``metadata.detached_child_trace_ids``.
    - ``detached_fan_outs``: set of fan-out node names whose instances
      each get their own Langfuse Trace. Same link mechanism on the
      fan-out node observation: each per-instance detached trace_id
      lands in the array.
    - ``disable_state_payload``: default ``True`` (Trace input/output
      sourcing). When ``True`` the observer does NOT serialize
      ``initial_state`` / final state directly onto ``trace.input`` /
      ``trace.output``; the minimal stub applies unless
      ``trace_input_from_state`` / ``trace_output_from_state``
      overrides. When ``False`` the raw state object is serialized to
      the Trace fields, subject to ``payload_byte_cap`` truncation.
      Independent of ``disable_provider_payload`` — the two payloads
      carry distinct threat models (LLM-call transcript vs.
      application state).
    - ``trace_input_from_state``: optional caller hook returning the
      value to use as ``trace.input``. Called once per invocation at
      the ``InvocationStartedEvent``. Returning ``None`` falls
      through to the next lever (raw state when
      ``disable_state_payload=False``, minimal stub otherwise).
    - ``trace_output_from_state``: same shape for ``trace.output``,
      called once per invocation at the ``InvocationCompletedEvent``.
    - ``implementation_name``: string surfaced as
      ``trace.metadata.implementation_name`` on every Trace. Defaults
      to the package's ``__implementation_name__``
      (``"openarmature-python"``). Configurable for test
      parameterization.
    - ``implementation_version``: string surfaced as
      ``trace.metadata.implementation_version`` on every Trace.
      Defaults to ``openarmature.__version__``. Always emitted —
      not gated by ``disable_state_payload``,
      ``disable_provider_payload``, or any other privacy knob.

    The observer reads the spec version from the package at
    construction time. Safe to share across concurrent invocations
    and across resumes of the same correlation_id; per-invocation
    state isolation keys all internal maps by invocation_id.
    """

    # Spec observability §8 (Langfuse backend mapping). Knob spec
    # basis: §8.9 privacy posture; §8.4.1 Trace input/output sourcing
    # (proposal 0043); §8.5 detached traces; §5.1 always-emit
    # attribution invariant.

    client: LangfuseClient
    disable_llm_spans: bool = False
    disable_provider_payload: bool = True
    payload_byte_cap: int = 65536
    detached_subgraphs: frozenset[str] = field(default_factory=_empty_str_frozenset)
    detached_fan_outs: frozenset[str] = field(default_factory=_empty_str_frozenset)
    spec_version: str = field(default_factory=_read_spec_version)
    # Proposal 0052 §8.4.1: implementation attribution rows on every
    # Trace. Configurable for test parameterization; defaults to the
    # package identity. Always-emit invariant inherited from §5.1 —
    # ``disable_state_payload`` and the other privacy knobs do not
    # gate these rows because they describe runtime identity, not
    # runtime data.
    implementation_name: str = field(default_factory=_read_implementation_name)
    implementation_version: str = field(default_factory=_read_implementation_version)
    # Proposal 0043 §8.4.1 *Trace input/output sourcing*.
    disable_state_payload: bool = True
    trace_input_from_state: Callable[[Any], Any] | None = None
    trace_output_from_state: Callable[[Any], Any] | None = None

    # Internal state populated during invocation.
    _inv_states: dict[str, _InvState] = field(init=False, repr=False, default_factory=dict[str, _InvState])

    def __post_init__(self) -> None:
        # §5.5.5 minimum-cap validation mirrors the OTel observer's bound.
        # Reject misconfigurations at construction time rather than
        # surfacing them as a Langfuse-ingest error later.
        if self.payload_byte_cap < _PAYLOAD_MIN_BYTES:
            raise ValueError(
                f"payload_byte_cap={self.payload_byte_cap} below the spec §5.5.5 "
                f"minimum of {_PAYLOAD_MIN_BYTES} bytes"
            )

    async def __call__(
        self,
        event: ObserverEvent,
    ) -> None:
        if isinstance(event, InvocationStartedEvent):
            self._handle_invocation_started(event)
            return
        if isinstance(event, InvocationCompletedEvent):
            self._handle_invocation_completed(event)
            return
        # Proposal 0050 per-attempt LLM events are OTel-span-only: the
        # Langfuse mapping renders one Generation per call from the
        # terminal LlmCompletionEvent / LlmFailedEvent, so the
        # per-attempt event is ignored here.
        if isinstance(event, LlmRetryAttemptEvent):
            return
        # Proposal 0049 typed LlmCompletionEvent (success path). Drives
        # the §5.5 Generation observation lifecycle for successful
        # provider calls.
        if isinstance(event, LlmCompletionEvent):
            if not self.disable_llm_spans:
                self._handle_typed_llm_completion(event)
            return
        # Proposal 0058 typed LlmFailedEvent (failure path). Drives
        # the same Generation observation lifecycle with ERROR level +
        # error_category as statusMessage.
        if isinstance(event, LlmFailedEvent):
            if not self.disable_llm_spans:
                self._handle_typed_llm_failed(event)
            return
        # Proposal 0063 tool-execution observability: render the dedicated
        # Langfuse Tool observation (asType "tool") under the calling
        # node's Span observation.
        if isinstance(event, ToolCallEvent | ToolCallFailedEvent):
            self._handle_tool_call(event)
            return
        # Proposal 0050 §6.3 framework-emitted failure-isolation event.
        if isinstance(event, FailureIsolatedEvent):
            self._handle_failure_isolated(event)
            return
        if isinstance(event, MetadataAugmentationEvent):
            self._handle_metadata_augmentation(event)
            return
        # Proposal 0059 embedding observability (observability §8.4.5): render
        # the dedicated Langfuse Embedding observation (asType "embedding")
        # under the calling node's Span observation. NOT gated by
        # disable_llm_spans (scoped to LLM completion per §5.5.8); the
        # input / output payload is gated by disable_provider_payload inside
        # the handler.
        if isinstance(event, EmbeddingEvent | EmbeddingFailedEvent):
            self._handle_embedding(event)
            return
        # Proposal 0060 rerank observability (observability §8.4.7): render the
        # dedicated Langfuse Retriever observation (asType "retriever") under
        # the calling node's Span observation. NOT gated by disable_llm_spans
        # (scoped to LLM completion per §5.5.13); the input / output payload is
        # gated by disable_provider_payload inside the handler.
        if isinstance(event, RerankEvent | RerankFailedEvent):
            self._handle_rerank(event)
            return
        if event.phase == "started":
            self._open_started_observation(event)
        elif event.phase == "completed":
            self._handle_completed(event)
        # checkpoint_saved and checkpoint_migrated are OTel-mapping-
        # specific synthetic phases per §5.5 / §10.8; the Langfuse
        # mapping doesn't surface checkpoint events as observations
        # in v0.23.0 (§8.10's deferral envelope).

    # ------------------------------------------------------------------
    # Span observation lifecycle (node / subgraph / fan-out)
    # ------------------------------------------------------------------

    def _open_started_observation(self, event: NodeEvent) -> None:
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()

        # Lazy Trace open on the first event for this invocation_id.
        # The Trace ID equals the invocation_id verbatim per §8.4.1 so
        # cross-system lookup is a direct hit.
        if invocation_id not in self._inv_states:
            self._open_trace(invocation_id, correlation_id, event)

        inv_state = self._inv_states[invocation_id]
        # Cache the fan-out node's parent_node_name from its own
        # started event so synthetic per-instance dispatch observations
        # can attach metadata.fan_out_parent_node_name (the inner
        # events from inside the fan-out don't carry fan_out_config
        # themselves; this cache bridges). fan_out_config is set only on
        # the NODE's own events, so it alone identifies them -- NOT
        # ``fan_out_index is None``, which would miss a fan-out node nested
        # inside an outer fan-out instance (its own event carries the OUTER
        # instance index), leaving the inner dispatch unsynthesized.
        if event.fan_out_config is not None:
            inv_state.fan_out_parent_node_name[event.namespace] = event.fan_out_config.parent_node_name

        # Per proposal 0045: mirror cache for parallel-branches NODE
        # identification (used by the augmentation shared-parent
        # check).  No additional ``branch_name is None`` filter — the
        # ``*_config`` field is itself only populated on a NODE's own
        # events.
        if event.parallel_branches_config is not None:
            inv_state.parallel_branches_parent_node_name[event.namespace] = (
                event.parallel_branches_config.parent_node_name
            )
            inv_state.parallel_branches_branch_names[event.namespace] = frozenset(
                event.parallel_branches_config.branch_names
            )

        key = self._key_for(event)
        if key in inv_state.open_observations:
            # Idempotent: a second started for the same (namespace,
            # attempt_index, fan_out_index) tuple is a no-op (matches
            # the OTel observer's behavior under retry-replay).
            return

        # Synthesize any subgraph dispatch / fan-out per-instance
        # dispatch observations the leaf needs as ancestors. Also
        # closes dispatch observations whose subtree we've left.
        self._sync_subgraph_observations(inv_state, correlation_id, event)

        parent_observation_id = self._resolve_parent_observation_id(inv_state, event)
        metadata = self._observation_metadata(event, correlation_id)
        target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
        handle = self.client.span(
            trace_id=target_trace_id,
            name=event.node_name,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        inv_state.open_observations[key] = _OpenObservation(
            handle=handle,
            fan_out_index_chain=event.fan_out_index_chain,
            branch_name_chain=event.branch_name_chain,
        )

    def _handle_completed(self, event: NodeEvent) -> None:
        from openarmature.observability.correlation import current_invocation_id

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        inv_state = self._inv_states.get(invocation_id)
        if inv_state is None:
            return

        # If this is the fan-out node's own completion (event.fan_out_index
        # is None) AND the fan-out is configured detached, close any
        # detached per-instance Trace dispatch observations the fan-out
        # spawned. Done BEFORE the regular pop so the close ordering is
        # children-before-parents.
        if event.fan_out_index is None and event.namespace and event.namespace[0] in self.detached_fan_outs:
            ns = event.namespace
            for key in list(inv_state.fan_out_instance_root_prefixes):
                anchor_ns = key[0]
                if len(anchor_ns) >= len(ns) and anchor_ns[: len(ns)] == ns:
                    # Detached per-instance dispatches live in
                    # fan_out_instance_observations (same map as
                    # non-detached); close via the matching helper.
                    self._close_fan_out_instance_dispatch_observation(inv_state, key)
                    inv_state.fan_out_instance_root_prefixes.discard(key)
                    # detached_traces uses the top-level routing key shape; derive
                    # it from the lineage key's own instance index (last entry).
                    fi_chain = key[1]
                    inv_state.detached_traces.pop(anchor_ns + (str(fi_chain[-1]),), None)
        # Per spec proposal 0013 (v0.10.0): when the fan-out node's
        # own completion fires, close all per-instance dispatch
        # observations synthesized for it. Children-before-parents.
        if event.fan_out_index is None and event.fan_out_config is not None:
            ns = event.namespace
            for key in list(inv_state.fan_out_instance_observations.keys()):
                anchor_ns = key[0]
                # The dispatch key is now (anchor_ns, fi_chain, bn_chain); match
                # on the NODE namespace (anchor_ns) being in this completing
                # node's subtree.
                if len(anchor_ns) >= len(ns) and anchor_ns[: len(ns)] == ns:
                    self._close_fan_out_instance_dispatch_observation(inv_state, key)
            inv_state.fan_out_parent_node_name.pop(event.namespace, None)
            # Clear the detached-child-trace-ids accumulator for this
            # fan-out node — cyclic execution that re-enters the same
            # fan-out starts the next iteration with a fresh list
            # rather than appending to the previous iteration's
            # accumulator and overwriting the prior link metadata.
            inv_state.detached_child_trace_ids.pop(event.namespace, None)
        # Per proposals 0044/0045: on a pb NODE's own completion, close the
        # per-branch dispatch observations synthesized for it (children-before-
        # parents) and clear the pb caches. Same shape as the fan-out cleanup.
        if event.parallel_branches_config is not None:
            ns = event.namespace
            for key in list(inv_state.parallel_branches_branch_spans.keys()):
                anchor_ns = key[0]
                if len(anchor_ns) >= len(ns) and anchor_ns[: len(ns)] == ns:
                    self._close_parallel_branches_branch_dispatch_observation(inv_state, key)
            inv_state.parallel_branches_parent_node_name.pop(event.namespace, None)
            inv_state.parallel_branches_branch_names.pop(event.namespace, None)

        key = self._key_for(event)
        observation = inv_state.open_observations.pop(key, None)
        if observation is None:
            return
        # Error-category mapping per §8.4.2: error.category → level=ERROR
        # + statusMessage=<category>.
        if event.error is not None and getattr(event.error, "category", None) is not None:
            observation.handle.end(level="ERROR", status_message=event.error.category)
        else:
            observation.handle.end()
        # If this was a detached subgraph root prefix, drop the
        # detached_traces entry so a subsequent re-entry mints fresh.
        inv_state.detached_traces.pop(event.namespace, None)

    # ------------------------------------------------------------------
    # Metadata augmentation (proposal 0040 §3.4 + §6)
    # ------------------------------------------------------------------

    def _handle_metadata_augmentation(self, event: MetadataAugmentationEvent) -> None:
        # Spec proposal 0040 §3.4 MUST: open observations whose lineage
        # ancestor-or-equals the augmenting context get the entries
        # applied in place via the Langfuse handle's
        # ``update(metadata=...)`` method. Sibling instances / branches
        # and ancestors above the containment are skipped (same scoping
        # rule as the OTel mapping — see
        # ``OTelObserver._handle_metadata_augmentation`` for the algebra).
        #
        # For an outermost-serial augmenter (FI=None, BN=None), the
        # invocation's Trace itself is updated via
        # ``client.update_trace`` so the augmented keys land on
        # ``trace.metadata.<key>`` for §8.4-style top-level filtering.
        # Inside a fan-out instance / parallel-branches branch the
        # Trace is OUT of scope (it's shared with siblings); only the
        # innermost containment + the augmenter's own subtree update.
        #
        # Per-instance / per-branch isolation:
        # ``set_invocation_metadata`` runs in the calling node's task
        # whose Context already carries the per-async-context COW
        # mapping (proposal 0034 §3.4). The augmentation event's
        # ``entries`` are that delta only — applying them to matching
        # open observations preserves the per-async-context isolation
        # 029 / 030 encode.
        from openarmature.observability.correlation import current_invocation_id

        invocation_id = current_invocation_id()
        if invocation_id is None or not event.entries:
            return
        inv_state = self._inv_states.get(invocation_id)
        aug_ns = event.namespace
        aug_fi_chain = event.fan_out_index_chain
        aug_bn_chain = event.branch_name_chain
        metadata_delta = dict(event.entries)

        # Trace.metadata: only when the augmenter sits in OUTERMOST
        # SERIAL context (no fan-out instance and no parallel-branches
        # branch on its call-stack path).  Per §3.4 the Trace is a
        # shared parent inside any dispatch boundary — siblings would
        # leak — so only the no-dispatch-on-path case writes.
        outermost_serial = all(fi is None for fi in aug_fi_chain) and all(bn is None for bn in aug_bn_chain)
        if outermost_serial:
            self.client.update_trace(id=invocation_id, metadata=metadata_delta)

        if inv_state is None:
            return

        # Per proposal 0045: parallel walk of the OTel observer's
        # _collect_augmentation_targets — subgraph wrappers on the
        # call-stack path (chain prefix-matches), fan-out instance
        # dispatch observations whose dispatch position matches the
        # augmenter's chain, and open NODE observations on the path
        # (skipping fan-out / pb shared-parent NODEs).

        # Subgraph wrapper observations on the path.
        for prefix, observation in inv_state.subgraph_observations.items():
            if not is_strict_prefix(prefix, aug_ns):
                continue
            if _observation_chain_on_path(observation, aug_fi_chain, aug_bn_chain):
                observation.handle.update(metadata=metadata_delta)

        # Fan-out instance dispatch observations: on the augmenter's path iff the
        # dispatch NODE namespace (key[0]) is an ancestor-or-equal of the
        # augmenter AND its full lineage chain (carried on the observation) is a
        # prefix of the augmenter's -- so a SIBLING outer instance's dispatch,
        # whose chain differs at the enclosing position, is excluded.
        for key, observation in inv_state.fan_out_instance_observations.items():
            anchor_ns = key[0]
            if not (is_strict_prefix(anchor_ns, aug_ns) or anchor_ns == aug_ns):
                continue
            if _observation_chain_on_path(observation, aug_fi_chain, aug_bn_chain):
                observation.handle.update(metadata=metadata_delta)

        # Open NODE observations.  Same as augmenter or strict
        # ancestor on the path; skip shared-parent NODE observations
        # (fan-out NODE / pb NODE) identified by presence in the
        # parent_node_name caches.
        for key, observation in inv_state.open_observations.items():
            ns, _ai, _fi, _bn = key
            if ns != aug_ns and not is_strict_prefix(ns, aug_ns):
                continue
            # A fan-out / pb NODE is a shared parent and MUST NOT carry an
            # instance's / branch's augmentation (proposal 0045 §3.4). This skip
            # applies whether the NODE sits strictly above the augmenter OR at
            # the augmenter's own namespace: an instance/branch executes AT the
            # fan-out/pb node's namespace, so ns == aug_ns also matches the shared
            # NODE (its per-instance dispatch is the one updated, separately above).
            if ns in inv_state.fan_out_parent_node_name or ns in inv_state.parallel_branches_parent_node_name:
                continue
            if _observation_chain_on_path(observation, aug_fi_chain, aug_bn_chain):
                observation.handle.update(metadata=metadata_delta)

    # ------------------------------------------------------------------
    # Failure-isolation event (proposal 0050 §6.3)
    # ------------------------------------------------------------------

    def _handle_failure_isolated(self, event: FailureIsolatedEvent) -> None:
        # Render the FailureIsolationMiddleware catch as a marker
        # observation. Parented under the wrapped node's observation when
        # it is still open; otherwise trace-level (the node observation
        # is typically already closed-with-error by delivery time, since
        # the node-body raise fires the node's completed event before the
        # middleware recovers). The wrapped node's name rides on
        # ``metadata.failure_isolation_node`` for correlation regardless.
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
        key: _StackKey = (event.namespace, event.attempt_index, event.fan_out_index, event.branch_name)
        parent = inv_state.open_observations.get(key)
        parent_observation_id = parent.handle.id if parent is not None else None
        metadata: dict[str, Any] = {
            "failure_isolation_event_name": event.event_name,
            "error_message": event.caught_exception.message,
        }
        if event.namespace:
            metadata["failure_isolation_node"] = event.namespace[-1]
        if event.caught_exception.category is not None:
            metadata["error_category"] = event.caught_exception.category
        correlation_id = current_correlation_id()
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        handle = self.client.span(
            trace_id=inv_state.trace_id,
            name="openarmature.failure_isolated",
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        handle.end()

    # ------------------------------------------------------------------
    # Invocation-boundary events (proposal 0043 §8.4.1 sourcing)
    # ------------------------------------------------------------------

    def _handle_invocation_started(self, event: InvocationStartedEvent) -> None:
        # Spec proposal 0043 §8.4.1 *Trace input/output sourcing*.
        # Lazy-open the Trace if this is the first signal for the
        # invocation_id (no node event has fired yet), then resolve
        # ``trace.input`` via the three-lever decision tree:
        #   1. Hook supplied AND returns non-None → hook value.
        #   2. ``disable_state_payload`` is False → raw initial_state
        #      serialized (subject to payload_byte_cap truncation).
        #   3. Otherwise → minimal stub:
        #        {entry_node, correlation_id}.
        # The stub carries no application payload — both fields are
        # already in ``trace.metadata``; surfacing them on
        # ``trace.input`` makes the Langfuse Traces list view
        # scannable without revealing state shape.
        if event.invocation_id not in self._inv_states:
            self._open_trace_lazy(event.invocation_id, event.correlation_id, event.entry_node)
        input_value = self._resolve_trace_input(event)
        self.client.update_trace(id=event.invocation_id, input=input_value)

    def _handle_invocation_completed(self, event: InvocationCompletedEvent) -> None:
        # Spec proposal 0043 §8.4.1. Resolve ``trace.output`` via the
        # same three-lever decision tree as input, with the minimal
        # stub carrying {final_node, status}.
        if event.invocation_id not in self._inv_states:
            # Defensive: a fast-failure invocation may complete before
            # any node event fired (e.g., resume-path validation
            # rejected). Lazy-open the Trace so the stub still lands.
            entry_node = event.final_node  # best-effort fallback
            self._open_trace_lazy(event.invocation_id, event.correlation_id, entry_node)
        output_value = self._resolve_trace_output(event)
        self.client.update_trace(id=event.invocation_id, output=output_value)

    def _resolve_trace_input(self, event: InvocationStartedEvent) -> Any:
        # Lever 1: caller hook.
        if self.trace_input_from_state is not None:
            try:
                hook_value = self.trace_input_from_state(event.initial_state)
            except Exception:
                # Hook raise: skip emission (defensive — caller code
                # should not break observability). Fall through to the
                # next lever rather than crash the observer.
                hook_value = None
            if hook_value is not None:
                return self._maybe_truncate_for_extras(hook_value)
        # Lever 2: raw state when knob is OFF.
        if not self.disable_state_payload:
            serialized = self._state_to_jsonable(event.initial_state)
            return self._maybe_truncate_for_extras(serialized)
        # Lever 3: minimal stub.
        stub: dict[str, Any] = {"entry_node": event.entry_node}
        if event.correlation_id is not None:
            stub["correlation_id"] = event.correlation_id
        return stub

    def _resolve_trace_output(self, event: InvocationCompletedEvent) -> Any:
        # Lever 1: caller hook.
        if self.trace_output_from_state is not None:
            try:
                hook_value = self.trace_output_from_state(event.final_state)
            except Exception:
                hook_value = None
            if hook_value is not None:
                return self._maybe_truncate_for_extras(hook_value)
        # Lever 2: raw state when knob is OFF.
        if not self.disable_state_payload:
            serialized = self._state_to_jsonable(event.final_state)
            return self._maybe_truncate_for_extras(serialized)
        # Lever 3: minimal stub.
        return {"final_node": event.final_node, "status": event.status}

    @staticmethod
    def _state_to_jsonable(state: Any) -> Any:
        # Best-effort conversion of a State instance to a JSON-able
        # shape. Pydantic models expose ``model_dump`` directly; other
        # objects fall through to a str representation. The serialized
        # form is what ends up on the Langfuse Trace's
        # ``input`` / ``output`` field.
        #
        # ``mode="json"`` (rather than the default Python mode) coerces
        # non-JSON-native types — ``datetime``, ``UUID``, ``Decimal``,
        # etc. — into JSON-compatible strings BEFORE the dict reaches
        # the downstream ``json.dumps`` truncation path. Without it the
        # truncation path raises ``TypeError`` and the observer's
        # ``__call__`` raise is swallowed by the engine's warnings-only
        # observer-isolation contract, leaving ``trace.input`` /
        # ``trace.output`` silently blank on states containing those
        # types.
        dumper = getattr(state, "model_dump", None)
        if callable(dumper):
            try:
                return dumper(mode="json")
            except Exception:
                return str(state)
        return str(state)

    def _client_trace(self, *, id: str, name: str | None, metadata: dict[str, Any]) -> None:
        # Proposal 0064 §8.4.1: every Trace open routes through here so the
        # sessionId / userId promotions apply uniformly across the main,
        # lazy, and detached trace-open sites.
        #   - trace.userId: promoted from the recognized ``userId`` caller
        #     key (already merged into ``metadata`` by _apply_caller_metadata).
        #   - trace.sessionId: sourced from openarmature.session_id (sessions
        #     capability, observability §5.6 / proposal 0020). python has no
        #     session_id source until 0020 lands, so it is unset (None) today;
        #     this is the single hook 0020 wires the source into.
        self.client.trace(
            id=id,
            name=name,
            metadata=metadata,
            session_id=None,
            user_id=_promoted_user_id(metadata),
        )

    def _open_trace_lazy(
        self,
        invocation_id: str,
        correlation_id: str | None,
        entry_node: str,
    ) -> None:
        # Open the Trace from a non-NodeEvent path (the proposal 0043
        # invocation-boundary events). The existing ``_open_trace``
        # entry point reads ``entry_node`` and caller metadata from a
        # NodeEvent; this lazy path doesn't have one. Caller metadata
        # is still readable via ``current_invocation_metadata`` —
        # ``_apply_caller_metadata`` mirrors the existing path.
        from openarmature.observability.metadata import current_invocation_metadata

        metadata: dict[str, Any] = {
            "entry_node": entry_node,
            "spec_version": self.spec_version,
            # Proposal 0052 §8.4.1: implementation attribution rows.
            "implementation_name": self.implementation_name,
            "implementation_version": self.implementation_version,
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(metadata, current_invocation_metadata())
        self._client_trace(id=invocation_id, name=entry_node, metadata=metadata)
        self._inv_states[invocation_id] = _InvState(trace_id=invocation_id)

    def _open_trace(self, invocation_id: str, correlation_id: str | None, event: NodeEvent) -> None:
        # ``entry_node`` and the trace name MUST identify the outer-graph
        # entry, not whichever node fired first. Subgraph wrappers do not
        # emit their own events — when the outer entry is a SubgraphNode
        # the first event the observer sees comes from inside the
        # subgraph (with ``event.namespace = (wrapper, inner)`` and
        # ``event.node_name = inner``). Using ``event.namespace[0]``
        # walks back to the outermost prefix component, which IS the
        # outer entry by construction (the graph engine fires inner
        # events under the wrapper's namespace).
        entry_node = event.namespace[0] if event.namespace else event.node_name
        metadata: dict[str, Any] = {
            "entry_node": entry_node,
            "spec_version": self.spec_version,
            # Proposal 0052 §8.4.1: implementation attribution rows.
            "implementation_name": self.implementation_name,
            "implementation_version": self.implementation_version,
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        # §8.6 trace name: caller-supplied invocation label takes
        # precedence; entry-node name is the spec-recommended fallback.
        # The caller-supplied path lands in proposal 0034 (PR 4) — for
        # now only the fallback is wired.
        trace_name = entry_node
        self._client_trace(id=invocation_id, name=trace_name, metadata=metadata)
        self._inv_states[invocation_id] = _InvState(trace_id=invocation_id)

    def _key_for(self, event: NodeEvent) -> _StackKey:
        return (event.namespace, event.attempt_index, event.fan_out_index, event.branch_name)

    def _resolve_parent_observation_id(self, inv_state: _InvState, event: NodeEvent) -> str | None:
        # Parent precedence (innermost wins):
        #   1. Per-instance fan-out dispatch observation at
        #      namespace[:1] + (str(fan_out_index),) — both detached
        #      (where the dispatch observation lives in the detached
        #      Trace) and non-detached (where it lives in the main
        #      Trace) cases route here when event is inside a fan-out
        #      instance.
        #   2. Subgraph dispatch observation at any matching ancestor
        #      prefix, walked longest-first.
        #   3. Leaf node observation at any matching ancestor prefix,
        #      walked longest-first.
        #   4. None — the Trace itself becomes the implicit parent.
        # Per proposals 0044 / 0013 / 0045: an inner node parents under the
        # INNERMOST dispatch on its lineage -- a per-branch dispatch
        # (parallel_branches_branch_spans) or a per-instance fan-out dispatch
        # (fan_out_instance_observations), both keyed by the lineage-aware
        # _dispatch_key. Walk prefixes longest-first so the innermost wins; the
        # lineage key carries the enclosing fan-out instance / branch chain, so
        # this resolves arbitrary nesting (fan-out in fan-out, parallel-branches
        # in fan-out, ...) to the RIGHT outer instance. Mirrors OTel
        # _resolve_parent_context.
        for prefix_len in range(len(event.namespace), 0, -1):
            prefix = event.namespace[:prefix_len]
            fi_axis = (
                event.fan_out_index_chain[prefix_len - 1]
                if prefix_len - 1 < len(event.fan_out_index_chain)
                else None
            )
            if event.branch_name is not None:
                branch_dispatch = inv_state.parallel_branches_branch_spans.get(
                    _branch_dispatch_key(
                        prefix, event.fan_out_index_chain, event.branch_name_chain, event.branch_name
                    )
                )
                if branch_dispatch is not None:
                    return branch_dispatch.handle.id
            if fi_axis is not None:
                dispatch = inv_state.fan_out_instance_observations.get(
                    _dispatch_key(prefix, event.fan_out_index_chain, event.branch_name_chain)
                )
                if dispatch is not None:
                    return dispatch.handle.id
        for prefix_len in range(len(event.namespace) - 1, 0, -1):
            prefix = event.namespace[:prefix_len]
            sg = inv_state.subgraph_observations.get(prefix)
            if sg is not None:
                return sg.handle.id
        # Open leaf-node observation fallback. The outer loop already
        # walks longest-first; the inner scan picks the first matching
        # open observation, which is fine for the cases dispatch
        # synthesis didn't cover (no subgraph wrapping the namespace).
        for prefix_len in range(len(event.namespace) - 1, 0, -1):
            prefix = event.namespace[:prefix_len]
            for key, observation in inv_state.open_observations.items():
                if key[0] == prefix:
                    return observation.handle.id
        # Proposal 0075: a callable parallel-branch's event sits at the pb
        # NODE's own namespace (branch_name set, no parallel_branches_config),
        # so it IS the unit — render it as a single observation parented under
        # the NODE observation. The strict-ancestor fallback above misses the
        # same-namespace NODE, so resolve it explicitly here.
        if (
            event.branch_name is not None
            and event.parallel_branches_config is None
            and event.namespace in inv_state.parallel_branches_parent_node_name
        ):
            for key, observation in inv_state.open_observations.items():
                if key[0] == event.namespace and key[3] is None:
                    return observation.handle.id
        return None

    def _trace_id_for(
        self,
        inv_state: _InvState,
        namespace: tuple[str, ...],
        fan_out_index: int | None,
    ) -> str:
        # Walk ancestor prefixes longest-first to find the innermost
        # detached Trace mapping; fall back to the main invocation
        # Trace. Detached fan-out instance Traces are keyed by
        # ``namespace[:1] + (str(fan_out_index),)`` so check that
        # specific composite first.
        if fan_out_index is not None and namespace:
            instance_key = namespace[:1] + (str(fan_out_index),)
            if instance_key in inv_state.detached_traces:
                return inv_state.detached_traces[instance_key]
        for prefix_len in range(len(namespace), 0, -1):
            prefix = namespace[:prefix_len]
            if prefix in inv_state.detached_traces:
                return inv_state.detached_traces[prefix]
        return inv_state.trace_id

    def _sync_subgraph_observations(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        event: NodeEvent,
    ) -> None:
        # Open synthetic subgraph dispatch / fan-out per-instance
        # dispatch observations for any ancestor prefix of this
        # event's namespace that doesn't have one yet. Also closes
        # subgraph dispatch observations whose subtree we've left.
        #
        # Called BEFORE opening the leaf observation, so descendants
        # find the right parent via _resolve_parent_observation_id.
        namespace = event.namespace
        # 1. Close subgraph dispatch observations whose prefix is no
        #    longer an ancestor of the current namespace.
        for prefix in list(inv_state.subgraph_observations.keys()):
            if prefix in inv_state.fan_out_instance_root_prefixes:
                # Detached fan-out instance dispatches close with the
                # fan-out's completed event, not on namespace moves.
                continue
            if not (len(prefix) < len(namespace) and namespace[: len(prefix)] == prefix):
                self._close_subgraph_observation(inv_state, prefix)
                inv_state.detached_traces.pop(prefix, None)
        # 2. Open ancestor dispatch observations for prefixes that
        #    don't have one yet.
        for depth in range(1, len(namespace)):
            prefix = namespace[:depth]
            if prefix in inv_state.subgraph_observations:
                continue
            # The fan-out instance axis at THIS depth -- the chain entry for the
            # dispatch boundary into prefix -- NOT the innermost event.fan_out_index
            # (which differs for an OUTER fan-out in a nested stack). Branches use
            # event.branch_name directly (callable branches don't extend the chain).
            fi_axis = (
                event.fan_out_index_chain[depth - 1] if depth - 1 < len(event.fan_out_index_chain) else None
            )
            # The per-instance dispatch for this prefix's instance is opened
            # below; skip the regular subgraph path so we don't double-open. Keyed
            # by the full lineage so nested instances don't collide across outer
            # ones. Runs at ANY depth (nested fan-out / subgraph wrapper).
            if (
                fi_axis is not None
                and _dispatch_key(prefix, event.fan_out_index_chain, event.branch_name_chain)
                in inv_state.fan_out_instance_observations
            ):
                continue
            # Detached subgraph: kept top-level (depth == 1). _trace_id_for routes
            # detached events by namespace[:1], so a nested detached unit would
            # partially detach (its dispatch in the new Trace, inner nodes in the
            # main one). Nested-detached support rides with the deferred
            # generalization of _trace_id_for (out of scope here).
            if depth == 1 and prefix[0] in self.detached_subgraphs:
                self._open_detached_subgraph_trace(inv_state, correlation_id, prefix, event)
                continue
            # Detached fan-out: the fan-out instance gets its own Trace per spec
            # §8.5. The fan-out node's Span observation in the parent Trace
            # already exists; the detached dispatch goes into the new Trace. Kept
            # top-level for the same reason as detached subgraphs above.
            if depth == 1 and event.fan_out_index is not None and prefix[0] in self.detached_fan_outs:
                self._open_detached_fan_out_instance_trace(inv_state, correlation_id, prefix, event)
                continue
            # Non-detached fan-out: synthesize the per-instance dispatch
            # observation under the fan-out node observation (proposal 0013
            # v0.10.0). Gated on the fan-out NODE at prefix and a fan-out axis at
            # this depth, so it runs at any depth (nested fan-out, or a fan-out
            # inside a subgraph wrapper / branch).
            if (
                fi_axis is not None
                and prefix[-1] not in self.detached_fan_outs
                and prefix in inv_state.fan_out_parent_node_name
            ):
                self._open_fan_out_instance_dispatch_observation(inv_state, correlation_id, prefix, event)
                continue
            # Per proposal 0044: synthesize a per-branch dispatch observation
            # under the pb NODE for an inner branch event, so inner branch
            # nodes parent under it rather than the shared pb NODE span. Mirror
            # of the fan-out per-instance arm above; gated on the branch at this
            # depth belonging to the pb node declared at this prefix.
            if (
                event.branch_name is not None
                and prefix in inv_state.parallel_branches_parent_node_name
                and event.branch_name in inv_state.parallel_branches_branch_names.get(prefix, frozenset())
            ):
                # Synthesize once per branch: _sync runs on every inner node's
                # started event, so guard against re-opening (a second open would
                # orphan the first observation and split the branch's nodes).
                # event.branch_name (not bn_axis) so callable branches -- which
                # don't extend branch_name_chain -- still synthesize.
                if (
                    _branch_dispatch_key(
                        prefix, event.fan_out_index_chain, event.branch_name_chain, event.branch_name
                    )
                    not in inv_state.parallel_branches_branch_spans
                ):
                    self._open_parallel_branches_branch_dispatch_observation(
                        inv_state, correlation_id, prefix, event
                    )
                continue
            # A parallel-branches or fan-out NODE prefix already has its own
            # leaf observation (from the NODE's own started event), unlike a
            # transparent subgraph wrapper. Don't synthesize a duplicate
            # subgraph wrapper observation over it; inner branch / instance
            # events parent under the NODE observation via the
            # _resolve_parent_observation_id leaf fallback. Mirrors the OTel
            # observer's same guard (it skips the synthetic subgraph span at a
            # pb / fan-out NODE depth for the same reason).
            if (
                prefix in inv_state.parallel_branches_parent_node_name
                or prefix in inv_state.fan_out_parent_node_name
            ):
                continue
            # Plain non-detached subgraph dispatch.
            self._open_subgraph_observation(inv_state, correlation_id, prefix, event)

    def _open_subgraph_observation(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        # Parent is the nearest enclosing subgraph dispatch (if any),
        # else None (the Trace is the implicit parent for top-level
        # subgraphs).
        parent_observation_id: str | None = None
        for plen in range(len(prefix) - 1, 0, -1):
            outer = prefix[:plen]
            sg = inv_state.subgraph_observations.get(outer)
            if sg is not None:
                parent_observation_id = sg.handle.id
                break
        # Subgraph wrappers don't dispatch their own events, so the
        # synthetic wrapper observation inherits its scalar metadata
        # from the FIRST inner event that triggered the synthesis.
        # ``attempt_index`` is hardcoded to 0: the wrapper has no
        # engine-managed retry counter of its own (inner nodes own
        # their own attempt_index independently).
        metadata: dict[str, Any] = {
            "namespace": list(prefix),
            "step": event.step,
            "attempt_index": 0,
            "subgraph_name": _subgraph_identity_at(event, len(prefix)),
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        handle = self.client.span(
            trace_id=inv_state.trace_id,
            name=prefix[-1],
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        # Per proposal 0045: chain sliced to wrapper depth.
        chain_len = len(prefix)
        inv_state.subgraph_observations[prefix] = _OpenObservation(
            handle=handle,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )

    def _open_fan_out_instance_dispatch_observation(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        # Non-detached per-instance dispatch lives in the parent
        # Trace under the fan-out node's own Span observation.
        fan_out_open = self._find_node_observation(inv_state, prefix, event)
        parent_observation_id = fan_out_open.handle.id if fan_out_open is not None else None
        parent_node_name = inv_state.fan_out_parent_node_name.get(prefix, prefix[-1])
        # Per-instance dispatch is synthesized from the first inner
        # event inside the instance subtree; inherit scalar metadata
        # from that event (same pattern as ``_open_subgraph_observation``).
        # The dispatch's OWN instance index is the chain entry at this depth, not
        # event.fan_out_index (the innermost index of the synthesizing inner
        # event) -- they differ when this is an OUTER fan-out in a nested stack.
        chain_len = len(prefix)
        fan_out_index = (
            event.fan_out_index_chain[chain_len - 1]
            if chain_len - 1 < len(event.fan_out_index_chain)
            else event.fan_out_index
        )
        metadata: dict[str, Any] = {
            "namespace": list(prefix),
            "step": event.step,
            "attempt_index": 0,
            "fan_out_parent_node_name": parent_node_name,
            "fan_out_index": fan_out_index,
            "subgraph_name": _subgraph_identity_at(event, len(prefix)),
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        handle = self.client.span(
            trace_id=inv_state.trace_id,
            name=prefix[-1],
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        # Lineage-aware key (proposal 0045): the namespace plus the full instance
        # / branch chain, so nested instances don't collide across outer ones.
        instance_key = _dispatch_key(prefix, event.fan_out_index_chain, event.branch_name_chain)
        inv_state.fan_out_instance_observations[instance_key] = _OpenObservation(
            handle=handle,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )

    def _open_parallel_branches_branch_dispatch_observation(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        # Per-branch dispatch lives under the parallel-branches NODE's own Span
        # observation (mirror of the fan-out per-instance dispatch).
        pb_open = self._find_node_observation(inv_state, prefix, event)
        parent_observation_id = pb_open.handle.id if pb_open is not None else None
        parent_node_name = inv_state.parallel_branches_parent_node_name.get(prefix, prefix[-1])
        # Synthesized from the first inner event in the branch subtree; inherit
        # scalar metadata from it. branch_name is the OA-emitted §8.4.2 row (from
        # event.branch_name, which a callable branch sets without extending
        # branch_name_chain); the caller's branchName augmentation rides in via
        # the caller metadata.
        chain_len = len(prefix)
        branch_name = cast("str", event.branch_name)
        metadata: dict[str, Any] = {
            "namespace": list(prefix),
            "step": event.step,
            "attempt_index": 0,
            "parallel_branches_parent_node_name": parent_node_name,
            "branch_name": branch_name,
            "subgraph_name": _subgraph_identity_at(event, len(prefix)),
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        handle = self.client.span(
            trace_id=inv_state.trace_id,
            name=branch_name,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        # Lineage-aware key (proposal 0045): the enclosing fan-out instance /
        # branch chain plus the explicit branch name, so a branch nested inside
        # an outer fan-out instance doesn't collide across outer instances.
        branch_key = _branch_dispatch_key(
            prefix, event.fan_out_index_chain, event.branch_name_chain, branch_name
        )
        inv_state.parallel_branches_branch_spans[branch_key] = _OpenObservation(
            handle=handle,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )

    def _open_detached_subgraph_trace(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        # Mint a fresh Trace for the detached subtree. The main Trace's
        # dispatch observation surfaces the link via
        # metadata.detached_child_trace_ids; the detached Trace gets
        # its own dispatch observation that descendants parent under.
        #
        # Asymmetry note vs. _open_detached_fan_out_instance_trace:
        # subgraphs are namespace-prefix-only constructs with no
        # per-subgraph node event of their own. The observer never
        # opens a leaf Span observation for the subgraph itself, only
        # synthesized dispatch observations. To carry the cross-Trace
        # link in the main Trace's shape, this helper opens an extra
        # "link" Span observation in the main Trace — a small
        # observation whose subtree is empty but whose
        # detached_child_trace_ids metadata points at the new Trace.
        # Dashboard users see two observations named ``prefix[-1]``:
        # one in the main Trace (link with link metadata, no subtree)
        # and one in the detached Trace (the real dispatch with the
        # subgraph subtree under it).
        #
        # Detached fan-out instances, by contrast, already have a
        # parent observation in the main Trace (the fan-out node's
        # leaf observation opened on its own started event). The
        # link metadata accumulates on that pre-existing observation
        # instead of synthesizing a separate link observation.
        detached_trace_id = str(uuid.uuid4())
        # Open the link observation in the main Trace and update its
        # metadata immediately — the array-form preserves §8.5's
        # "string array, one entry per detached child" shape so
        # later detached siblings under the same parent can append.
        #
        # `detached: True` per §8.4.2 (proposal 0042) — the
        # parent-side dispatching observation marks itself when it
        # fires a detached child.
        #
        # Note: `subgraph_name` is intentionally NOT on this link
        # observation. Per §5.3 + §8.5, in detached mode the wrapper
        # role migrates to the detached trace's dispatch observation;
        # the main trace's link observation IS the SubgraphNode span
        # (no wrapper role) and so does not carry `subgraph_name`.
        link_metadata: dict[str, Any] = {
            "detached_child_trace_ids": [detached_trace_id],
            "detached": True,
        }
        if correlation_id is not None:
            link_metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(link_metadata, event.caller_invocation_metadata)
        parent_observation_id: str | None = None
        for plen in range(len(prefix) - 1, 0, -1):
            outer = prefix[:plen]
            sg = inv_state.subgraph_observations.get(outer)
            if sg is not None:
                parent_observation_id = sg.handle.id
                break
        # Zero-duration link observation in the main Trace — it
        # exists only to surface the cross-Trace reference via
        # metadata.detached_child_trace_ids; close it immediately so
        # nothing perceives it as in-flight. Mirrors the OTel
        # observer's synthetic-event zero-duration spans.
        link_handle = self.client.span(
            trace_id=inv_state.trace_id,
            name=prefix[-1],
            metadata=link_metadata,
            parent_observation_id=parent_observation_id,
        )
        link_handle.end()
        # Open the detached Trace + the dispatch observation that
        # subtree descendants parent under.
        detached_metadata: dict[str, Any] = {"detached_from_invocation_id": inv_state.trace_id}
        if correlation_id is not None:
            detached_metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(detached_metadata, event.caller_invocation_metadata)
        identity = _subgraph_identity_at(event, len(prefix))
        # The detached trace's wrapper observation IS the migrated
        # SubgraphNode wrapper. Per the resolution in coord thread
        # ``clarify-subgraph-name-semantics`` and fixture 033's
        # expected shape, the observation name uses the compiled-
        # subgraph identity (e.g., ``"long_running_workflow"``); its
        # ``metadata.subgraph_name`` carries the same identity.
        #
        # When the identity is empty (BC path — ``SubgraphNode``
        # constructed without ``subgraph_identity``), the two
        # diverge intentionally: the observation NAME falls back to
        # the wrapper node name (an empty observation name is worse
        # UX than a wrapper-named one), but ``metadata.subgraph_name``
        # stays empty per §5.3's "empty string when no identity is
        # tracked" contract. Filtering on
        # ``metadata.subgraph_name == "X"`` then matches only
        # wrappers explicitly registered with
        # ``subgraph_identity = "X"``, not every wrapper that
        # happens to be named ``X``.
        wrapper_obs_name = identity or prefix[-1]
        self._client_trace(id=detached_trace_id, name=wrapper_obs_name, metadata=detached_metadata)
        # §8.4.2 (proposal 0042): `detached: true` lives on the
        # PARENT-side dispatching observation (the link observation
        # above), not on the dispatch observation IN the detached
        # trace. The detached-side observation is the migrated
        # SubgraphNode wrapper and carries `subgraph_name` only.
        dispatch_metadata: dict[str, Any] = {
            "subgraph_name": identity,
        }
        if correlation_id is not None:
            dispatch_metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(dispatch_metadata, event.caller_invocation_metadata)
        handle = self.client.span(
            trace_id=detached_trace_id,
            name=wrapper_obs_name,
            metadata=dispatch_metadata,
            parent_observation_id=None,
        )
        # Per proposal 0045: detached subgraph wrapper sits in its own
        # trace; chain still mirrors the parent-trace path so the
        # augmentation lookup is consistent with non-detached.
        chain_len = len(prefix)
        inv_state.subgraph_observations[prefix] = _OpenObservation(
            handle=handle,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )
        inv_state.detached_traces[prefix] = detached_trace_id

    def _open_detached_fan_out_instance_trace(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
        event: NodeEvent,
    ) -> None:
        # Mint a fresh Trace per instance. The fan-out node's own
        # Span observation in the parent Trace accumulates the
        # detached_child_trace_ids array (one entry per instance);
        # each detached Trace gets its own per-instance dispatch
        # observation that inner-node observations parent under.
        #
        # See _open_detached_subgraph_trace's docstring for why the
        # detached-fan-out path doesn't synthesize a separate "link"
        # observation in the main Trace: the fan-out node already
        # has a leaf observation there (opened on its started event),
        # so the link metadata accumulates on that existing
        # observation rather than on a parallel link observation.
        detached_trace_id = str(uuid.uuid4())
        # Accumulate the per-fan-out link-ids list via the side cache
        # so each new instance appends to the array on the fan-out
        # node's observation rather than overwriting the previous
        # instance's entry.
        ids_list = inv_state.detached_child_trace_ids.setdefault(prefix, [])
        ids_list.append(detached_trace_id)
        fan_out_open = self._find_node_observation(inv_state, prefix, event)
        if fan_out_open is not None:
            # `detached: True` per §8.4.2 (proposal 0042) — the
            # parent-side fan-out node observation marks itself when
            # its instances are detached. Re-sent on every instance
            # update; the Langfuse client merges metadata, so this is
            # idempotent.
            link_metadata: dict[str, Any] = {
                "detached_child_trace_ids": list(ids_list),
                "detached": True,
            }
            if correlation_id is not None:
                link_metadata["correlation_id"] = correlation_id
            _apply_caller_metadata(link_metadata, event.caller_invocation_metadata)
            fan_out_open.handle.update(metadata=link_metadata)
        # Open the detached Trace + per-instance dispatch observation.
        detached_metadata: dict[str, Any] = {
            "detached_from_invocation_id": inv_state.trace_id,
            "fan_out_index": event.fan_out_index,
        }
        if correlation_id is not None:
            detached_metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(detached_metadata, event.caller_invocation_metadata)
        self._client_trace(
            id=detached_trace_id,
            name=prefix[-1],
            metadata=detached_metadata,
        )
        # §8.4.2 (proposal 0042): `detached: true` lives on the
        # PARENT-side fan-out node observation (link_metadata above),
        # not on the per-instance dispatch observation IN the detached
        # trace. The detached-side per-instance observation carries
        # only `fan_out_parent_node_name` + `fan_out_index`.
        parent_node_name = inv_state.fan_out_parent_node_name.get(prefix, prefix[-1])
        dispatch_metadata: dict[str, Any] = {
            "fan_out_parent_node_name": parent_node_name,
            "fan_out_index": event.fan_out_index,
        }
        if correlation_id is not None:
            dispatch_metadata["correlation_id"] = correlation_id
        _apply_caller_metadata(dispatch_metadata, event.caller_invocation_metadata)
        handle = self.client.span(
            trace_id=detached_trace_id,
            name=prefix[-1],
            metadata=dispatch_metadata,
            parent_observation_id=None,
        )
        # Shared fan_out_instance_observations / root_prefixes use the
        # lineage-aware key (consistent with resolution / close). detached_traces
        # keeps the top-level routing key shape that _trace_id_for reconstructs
        # (detached fan-outs are top-level; generalizing _trace_id_for is out of
        # scope for this fix).
        instance_key = _dispatch_key(prefix, event.fan_out_index_chain, event.branch_name_chain)
        chain_len = len(prefix)
        inv_state.fan_out_instance_observations[instance_key] = _OpenObservation(
            handle=handle,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )
        inv_state.detached_traces[prefix + (str(event.fan_out_index),)] = detached_trace_id
        inv_state.fan_out_instance_root_prefixes.add(instance_key)

    def _close_subgraph_observation(self, inv_state: _InvState, prefix: tuple[str, ...]) -> None:
        observation = inv_state.subgraph_observations.pop(prefix, None)
        if observation is None:
            return
        observation.handle.end()

    def _close_fan_out_instance_dispatch_observation(self, inv_state: _InvState, key: _DispatchKey) -> None:
        observation = inv_state.fan_out_instance_observations.pop(key, None)
        if observation is None:
            return
        observation.handle.end()

    def _close_parallel_branches_branch_dispatch_observation(
        self, inv_state: _InvState, key: _BranchDispatchKey
    ) -> None:
        observation = inv_state.parallel_branches_branch_spans.pop(key, None)
        if observation is None:
            return
        observation.handle.end()

    def _find_node_observation(
        self, inv_state: _InvState, prefix: tuple[str, ...], event: NodeEvent
    ) -> _OpenObservation | None:
        # Find a NODE's own open leaf observation at ``prefix`` (the fan-out or
        # parallel-branches NODE, whose per-instance / per-branch dispatches
        # parent under it). Match the ENCLOSING lineage, not just the namespace:
        # when the NODE is itself nested inside an outer fan-out instance /
        # branch, several instances of the same NODE namespace are open at once
        # under concurrency, so a namespace-only scan would bind the wrong one.
        # The NODE's own event carries the instance / branch it sits in as its
        # fan_out_index / branch_name (key[2] / key[3]); that equals the
        # augmenting/leaf event's chain entry at the level above this NODE.
        n = len(prefix)
        enclosing_fi = (
            event.fan_out_index_chain[n - 2] if n >= 2 and n - 2 < len(event.fan_out_index_chain) else None
        )
        enclosing_bn = (
            event.branch_name_chain[n - 2] if n >= 2 and n - 2 < len(event.branch_name_chain) else None
        )
        for key, observation in inv_state.open_observations.items():
            if key[0] == prefix and key[2] == enclosing_fi and key[3] == enclosing_bn:
                return observation
        return None

    # ------------------------------------------------------------------
    # Lifecycle: close_invocation / shutdown
    # ------------------------------------------------------------------

    def close_invocation(self, invocation_id: str) -> None:
        """Drain still-open observations for ``invocation_id``.

        Synthetic dispatch observations only close on cursor-move when
        a subsequent event arrives with a different namespace prefix.
        For a subgraph or fan-out that's the last subtree of an
        invocation, no follow-up event triggers the close — this
        method walks the per-invocation state and ends anything left
        in child→parent order so the Langfuse-side observations don't
        stay perpetually in-flight.

        Idempotent: calling twice (or for an invocation_id with no
        open state) is a no-op.
        """
        inv_state = self._inv_states.pop(invocation_id, None)
        if inv_state is None:
            return
        # Order: deepest leaves first so parents see all children
        # closed before they end. Leaf node observations (sorted
        # deepest-first by namespace length) → per-instance fan-out
        # dispatches → subgraph dispatches. LLM observations don't
        # appear here — both the success and error paths open + close
        # the Generation in one shot at handler-time, so there are no
        # in-flight LLM observations to drain.
        for key in sorted(
            inv_state.open_observations.keys(),
            key=lambda k: -len(k[0]),
        ):
            obs = inv_state.open_observations.pop(key, None)
            if obs is not None:
                obs.handle.end()
        for prefix in list(inv_state.fan_out_instance_observations.keys()):
            self._close_fan_out_instance_dispatch_observation(inv_state, prefix)
        for prefix in sorted(
            inv_state.subgraph_observations.keys(),
            key=lambda p: -len(p),
        ):
            self._close_subgraph_observation(inv_state, prefix)

    def force_flush(self, timeout_ms: int = 30_000) -> bool:
        """Flush pending observations through the underlying client.

        Returns ``True`` when the client's flush completes within the
        deadline, ``False`` otherwise. Mirrors the OTel observer's
        ``force_flush`` surface — distinct from
        :meth:`~openarmature.graph.compiled.CompiledGraph.drain` (which
        covers the engine's observer-event queue): this method covers
        the outbound buffer of the Langfuse client (the SDK's OTel
        BatchSpanProcessor when wrapped via :class:`LangfuseSDKAdapter`).

        Useful in fast-teardown harnesses (CLI one-shots, serverless
        functions, ASGI lifespan shutdown) where the SDK's
        BatchSpanProcessor export thread would otherwise be cut off
        before its buffer drains.
        """
        return self.client.force_flush(timeout_ms=timeout_ms)

    def shutdown(self) -> None:
        """Drain every in-flight invocation. Use for long-lived
        observers shared across requests; CLI / one-shot processes
        typically call this from a ``finally`` block alongside
        ``compiled.drain()``.
        """
        for invocation_id in list(self._inv_states.keys()):
            self.close_invocation(invocation_id)

    def _observation_metadata(self, event: NodeEvent, correlation_id: str | None) -> dict[str, Any]:
        # §8.4.2 observation-level mapping. Fields below mirror the
        # OTel observer's _node_attrs() output, renamed for Langfuse's
        # flat metadata shape (no `openarmature.` namespace prefix —
        # Langfuse's metadata bag is per-observation).
        metadata: dict[str, Any] = {
            "namespace": list(event.namespace),
            "step": event.step,
            "attempt_index": event.attempt_index,
        }
        if event.fan_out_index is not None:
            metadata["fan_out_index"] = event.fan_out_index
        if event.branch_name is not None:
            metadata["branch_name"] = event.branch_name
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        if event.fan_out_config is not None:
            cfg = event.fan_out_config
            metadata["fan_out_item_count"] = cfg.item_count
            metadata["fan_out_concurrency"] = 0 if cfg.concurrency is None else cfg.concurrency
            metadata["fan_out_error_policy"] = cfg.error_policy
        _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        return metadata

    # ------------------------------------------------------------------
    # Generation observation lifecycle (LLM provider events)
    # ------------------------------------------------------------------

    # v0.13.0 (proposals 0049 + 0057 + 0058): both Generation
    # observation lifecycles are driven by typed events — success path
    # from LlmCompletionEvent, failure path from LlmFailedEvent. Both
    # handlers open + close in one shot at typed-event arrival, with
    # start_time back-dated by latency_ms so duration reflects the
    # adapter-boundary measurement rather than dispatcher queue delay.
    # The provider dropped sentinel-namespace NodeEvent emission for
    # LLM events entirely in this release.
    def _handle_typed_llm_completion(self, event: LlmCompletionEvent) -> None:
        """Open + close the Generation observation from the typed
        LlmCompletionEvent (success path)."""
        # Mid-call metadata augmentation can't reach this observation:
        # the typed event arrives only after complete() returns, and
        # the observation is back-dated past any augmentation event
        # that fired while the call was in flight. Since complete()
        # is awaited, node bodies can't actually run augmentation
        # mid-call, so this is theoretical only.
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()
        # The Trace MAY not exist yet if the LLM call fires before any
        # node `started` event has hit this observer. Create-on-demand
        # mirrors the sentinel-pair handler's behavior.
        if invocation_id not in self._inv_states:
            self._open_trace_for_typed_event(invocation_id, correlation_id, event)
        inv_state = self._inv_states[invocation_id]
        # Back-date start_time using latency_ms; fall back to end_time
        # for both when latency is missing (zero-duration observation,
        # mirroring the OTel path).
        end_time = datetime.now(UTC)
        if event.latency_ms is not None:
            start_time = end_time - timedelta(milliseconds=event.latency_ms)
        else:
            start_time = end_time
        parent_observation_id = self._resolve_llm_parent_observation_id(
            inv_state,
            calling_namespace_prefix=event.namespace,
            calling_attempt_index=event.attempt_index,
            calling_fan_out_index=event.fan_out_index,
            calling_branch_name=event.branch_name,
        )
        metadata = self._typed_event_metadata(event, correlation_id)
        model_parameters: dict[str, Any] = dict(event.request_params or {})
        input_value: Any = None
        output_value: Any = None
        if not self.disable_provider_payload:
            if event.input_messages:
                input_value = self._maybe_truncate_for_input(event.input_messages)
            if event.output_content is not None:
                output_value = self._maybe_truncate_for_output(event.output_content)
            if event.request_extras:
                metadata["request_extras"] = self._maybe_truncate_for_extras(dict(event.request_extras))
        target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
        handle = self.client.generation(
            trace_id=target_trace_id,
            name="openarmature.llm.complete",
            model=event.model,
            model_parameters=model_parameters,
            input=input_value,
            output=output_value,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            prompt=self._resolve_prompt_link_from_typed_event(event),
            start_time=start_time,
        )
        usage = self._usage_from_typed_event(event)
        end_kwargs: dict[str, Any] = {}
        if usage is not None:
            end_kwargs["usage"] = usage
        handle.end(end_time=end_time, **end_kwargs)

    def _handle_typed_llm_failed(self, event: LlmFailedEvent) -> None:
        """Open + close an ERROR-level Generation observation from the
        typed LlmFailedEvent (failure path): ERROR level + error_category
        as the Generation's statusMessage. A structured_output_invalid
        failure additionally carries the response-side surface (payload-gated
        output, usage, metadata.finish_reason) like a completion, since its
        wire response was intact."""
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()
        if invocation_id not in self._inv_states:
            self._open_trace_for_typed_event(invocation_id, correlation_id, event)
        inv_state = self._inv_states[invocation_id]
        # Back-date timestamps using latency_ms (mirrors the success
        # path); for failures the duration reflects time until the §7
        # exception was raised.
        end_time = datetime.now(UTC)
        if event.latency_ms is not None:
            start_time = end_time - timedelta(milliseconds=event.latency_ms)
        else:
            start_time = end_time
        parent_observation_id = self._resolve_llm_parent_observation_id(
            inv_state,
            calling_namespace_prefix=event.namespace,
            calling_attempt_index=event.attempt_index,
            calling_fan_out_index=event.fan_out_index,
            calling_branch_name=event.branch_name,
        )
        metadata = self._typed_event_metadata(event, correlation_id)
        # Failure-specific metadata rows: surface error_type + error_
        # message as well as the category-as-statusMessage on the
        # observation. error_type is null when no impl-side type was
        # available; the metadata key is omitted in that case so the
        # absence-is-meaningful semantic is preserved.
        if event.error_type is not None:
            metadata["error_type"] = event.error_type
        metadata["error_message"] = event.error_message
        model_parameters: dict[str, Any] = dict(event.request_params or {})
        input_value: Any = None
        output_value: Any = None
        end_kwargs: dict[str, Any] = {}
        if not self.disable_provider_payload:
            if event.input_messages:
                input_value = self._maybe_truncate_for_input(event.input_messages)
            if event.request_extras:
                metadata["request_extras"] = self._maybe_truncate_for_extras(dict(event.request_extras))
        # Proposal 0082: a structured_output_invalid failure has an intact wire
        # response, so the failed Generation carries the response-side surface
        # (output payload-gated, usage) like a completion, alongside the ERROR
        # level + category. metadata.finish_reason is added for this category by
        # _typed_event_metadata.
        if event.error_category == "structured_output_invalid":
            if not self.disable_provider_payload and event.output_content is not None:
                output_value = self._maybe_truncate_for_output(event.output_content)
            usage = self._usage_from_typed_event(event)
            if usage is not None:
                end_kwargs["usage"] = usage
        target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
        handle = self.client.generation(
            trace_id=target_trace_id,
            name="openarmature.llm.complete",
            model=event.model,
            model_parameters=model_parameters,
            input=input_value,
            output=output_value,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            prompt=self._resolve_prompt_link_from_typed_event(event),
            start_time=start_time,
        )
        # Error-category mapping: §8.4.2 + §8.4.3.
        handle.end(
            end_time=end_time,
            level="ERROR",
            status_message=event.error_category,
            **end_kwargs,
        )

    # Spec proposal 0063: dedicated Langfuse Tool observation (asType="tool").
    def _handle_tool_call(self, event: ToolCallEvent | ToolCallFailedEvent) -> None:
        """Open + close a dedicated Tool observation (Langfuse
        ``asType="tool"``) under the calling node's Span
        observation. DEFAULT level on a ToolCallEvent; ERROR (with
        ``error_type`` / ``error_message`` in metadata and as the status
        message) on a ToolCallFailedEvent. ``input`` (arguments) /
        ``output`` (result) are payload-gated per ``disable_provider_payload``.
        """
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()
        if invocation_id not in self._inv_states:
            self._open_trace_for_typed_event(invocation_id, correlation_id, event)
        inv_state = self._inv_states[invocation_id]
        end_time = datetime.now(UTC)
        if event.latency_ms is not None:
            start_time = end_time - timedelta(milliseconds=event.latency_ms)
        else:
            start_time = end_time
        parent_observation_id = self._resolve_llm_parent_observation_id(
            inv_state,
            calling_namespace_prefix=event.namespace,
            calling_attempt_index=event.attempt_index,
            calling_fan_out_index=event.fan_out_index,
            calling_branch_name=event.branch_name,
        )
        # §8.4.6 metadata: tool name always, tool_call_id when present.
        metadata: dict[str, Any] = {"openarmature_tool_name": event.tool_name}
        if event.tool_call_id is not None:
            metadata["openarmature_tool_call_id"] = event.tool_call_id
        input_value: Any = None
        output_value: Any = None
        if not self.disable_provider_payload:
            if event.arguments is not None:
                input_value = self._maybe_truncate_for_input(event.arguments)
            if isinstance(event, ToolCallEvent):
                # The tool result is a structured value (not a plain
                # string like output_content), so reuse the structured
                # truncator: native value when it fits, marker string
                # otherwise.
                output_value = self._maybe_truncate_for_input(event.result)
        level: ObservationLevel = "DEFAULT"
        status_message: str | None = None
        if isinstance(event, ToolCallFailedEvent):
            level = "ERROR"
            if event.error_type is not None:
                metadata["error_type"] = event.error_type
            metadata["error_message"] = event.error_message
            status_message = event.error_message
        target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
        handle = self.client.tool(
            trace_id=target_trace_id,
            name="openarmature.tool.call",
            input=input_value,
            output=output_value,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            level=level,
            status_message=status_message,
            start_time=start_time,
        )
        handle.end(end_time=end_time)

    # Spec: observability §8.4.5 (proposal 0059) Embedding observation
    # (Langfuse asType="embedding"). The failure path is the generic §4.2 /
    # §8.4.2 ERROR mapping with the §7 error category, mirroring the tool
    # failure.
    def _handle_embedding(self, event: EmbeddingEvent | EmbeddingFailedEvent) -> None:
        """Open + close a dedicated Embedding observation (Langfuse
        ``asType="embedding"``) under the calling node's Span observation.

        Success (``EmbeddingEvent``): DEFAULT level, ``model`` =
        ``response_model`` (falling back to the requested ``model``),
        ``usage`` from the embedding token record, the OA identity metadata
        (``openarmature_input_count`` / ``openarmature_dimensions`` /
        ``openarmature_response_id``), and the payload-gated ``input``
        strings + ``output`` vectors.

        Failure (``EmbeddingFailedEvent``): ERROR level with the
        ``error_category`` as the status message and ``error_type`` /
        ``error_message`` in metadata, mirroring the tool failure. The
        request-side ``input`` strings are still payload-gated; there is NO
        ``output`` (no response received).
        """
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()
        if invocation_id not in self._inv_states:
            self._open_trace_for_typed_event(invocation_id, correlation_id, event)
        inv_state = self._inv_states[invocation_id]
        end_time = datetime.now(UTC)
        if event.latency_ms is not None:
            start_time = end_time - timedelta(milliseconds=event.latency_ms)
        else:
            start_time = end_time
        parent_observation_id = self._resolve_llm_parent_observation_id(
            inv_state,
            calling_namespace_prefix=event.namespace,
            calling_attempt_index=event.attempt_index,
            calling_fan_out_index=event.fan_out_index,
            calling_branch_name=event.branch_name,
        )
        # §8.4.5 metadata: input_count always; dimensions / response_id are
        # response-derived (success-only). correlation_id + caller metadata
        # mirror the other observation handlers' scoping rows.
        metadata: dict[str, Any] = {}
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        if event.fan_out_index is not None:
            metadata["fan_out_index"] = event.fan_out_index
        if event.branch_name is not None:
            metadata["branch_name"] = event.branch_name
        if event.caller_invocation_metadata is not None:
            _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        input_value: Any = None
        if not self.disable_provider_payload and event.input_strings:
            input_value = self._maybe_truncate_for_input(event.input_strings)
        if isinstance(event, EmbeddingEvent):
            metadata["openarmature_input_count"] = event.input_count
            if event.dimensions is not None:
                metadata["openarmature_dimensions"] = event.dimensions
            if event.response_id is not None:
                metadata["openarmature_response_id"] = event.response_id
            output_value: Any = None
            if not self.disable_provider_payload and event.output_vectors:
                output_value = self._maybe_truncate_for_input(event.output_vectors)
            usage = LangfuseUsage(input=event.usage.input_tokens) if event.usage is not None else None
            target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
            handle = self.client.embedding(
                trace_id=target_trace_id,
                name="openarmature.embedding.complete",
                model=event.response_model or event.model,
                usage=usage,
                input=input_value,
                output=output_value,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
                start_time=start_time,
            )
            handle.end(end_time=end_time)
            return
        # Failure path: request-side input_count survives; the response-derived
        # rows do not. No output. ERROR level + category-as-statusMessage.
        metadata["openarmature_input_count"] = len(event.input_strings)
        if event.error_type is not None:
            metadata["error_type"] = event.error_type
        metadata["error_message"] = event.error_message
        target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
        handle = self.client.embedding(
            trace_id=target_trace_id,
            name="openarmature.embedding.complete",
            model=event.model,
            input=input_value,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            level="ERROR",
            status_message=event.error_category,
            start_time=start_time,
        )
        handle.end(end_time=end_time)

    # Spec: observability §8.4.7 (proposal 0060) Retriever observation
    # (Langfuse asType="retriever"). The failure path is the generic §4.2 /
    # §8.4.2 ERROR mapping with the §7 error category, mirroring the tool /
    # embedding failure. The output results are sourced from
    # RerankEvent.output_results per proposal 0089; the searchUnits usageDetails
    # key follows the §8.4.7 OA convention.
    def _handle_rerank(self, event: RerankEvent | RerankFailedEvent) -> None:
        """Open + close a dedicated Retriever observation (Langfuse
        ``asType="retriever"``) under the calling node's Span observation.

        Success (``RerankEvent``): DEFAULT level, ``model`` = ``response_model``
        (falling back to the requested ``model``), ``usage`` carrying the input-
        token + search-unit figures, the OA identity metadata
        (``openarmature_query_length`` / ``openarmature_document_count`` /
        ``openarmature_top_k`` / ``openarmature_result_count`` /
        ``openarmature_response_id``), and the payload-gated ``input``
        (``{query, documents}``) + ``output`` (scored results).

        Failure (``RerankFailedEvent``): ERROR level with the ``error_category``
        as the status message and ``error_type`` / ``error_message`` in
        metadata, mirroring the tool / embedding failure. The request-side
        ``input`` is still payload-gated; there is NO ``output`` (no response
        received).
        """
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()
        if invocation_id not in self._inv_states:
            self._open_trace_for_typed_event(invocation_id, correlation_id, event)
        inv_state = self._inv_states[invocation_id]
        end_time = datetime.now(UTC)
        if event.latency_ms is not None:
            start_time = end_time - timedelta(milliseconds=event.latency_ms)
        else:
            start_time = end_time
        parent_observation_id = self._resolve_llm_parent_observation_id(
            inv_state,
            calling_namespace_prefix=event.namespace,
            calling_attempt_index=event.attempt_index,
            calling_fan_out_index=event.fan_out_index,
            calling_branch_name=event.branch_name,
        )
        # §8.4.7 request-side identity metadata (present on both variants):
        # query_length (UTF-8 byte length), document_count, top_k (when
        # supplied). correlation_id + caller metadata mirror the other
        # observation handlers' scoping rows.
        metadata: dict[str, Any] = {}
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        if event.fan_out_index is not None:
            metadata["fan_out_index"] = event.fan_out_index
        if event.branch_name is not None:
            metadata["branch_name"] = event.branch_name
        if event.caller_invocation_metadata is not None:
            _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        metadata["openarmature_query_length"] = len(event.query.encode("utf-8"))
        metadata["openarmature_document_count"] = event.document_count
        if event.top_k is not None:
            metadata["openarmature_top_k"] = event.top_k
        input_value: Any = None
        if not self.disable_provider_payload:
            input_value = self._maybe_truncate_for_input(
                {"query": event.query, "documents": list(event.documents)}
            )
        if isinstance(event, RerankEvent):
            # §8.4.7 response-derived metadata (success-only).
            metadata["openarmature_result_count"] = event.result_count
            if event.response_id is not None:
                metadata["openarmature_response_id"] = event.response_id
            output_value: Any = None
            if not self.disable_provider_payload and event.output_results:
                output_value = self._maybe_truncate_for_input(
                    [result.model_dump(exclude_none=True) for result in event.output_results]
                )
            # usageDetails omitted entirely when no usage record; when present,
            # only the non-null figures render (input / searchUnits) per §8.4.7.
            usage = None
            if event.usage is not None:
                usage = LangfuseUsage(
                    input=event.usage.input_tokens,
                    search_units=event.usage.search_units,
                )
            target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
            handle = self.client.retriever(
                trace_id=target_trace_id,
                name="openarmature.rerank.complete",
                model=event.response_model or event.model,
                usage=usage,
                input=input_value,
                output=output_value,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
                start_time=start_time,
            )
            handle.end(end_time=end_time)
            return
        # Failure path: the request-side metadata survives; the response-derived
        # rows do not. No output. ERROR level + category-as-statusMessage.
        if event.error_type is not None:
            metadata["error_type"] = event.error_type
        metadata["error_message"] = event.error_message
        target_trace_id = self._trace_id_for(inv_state, event.namespace, event.fan_out_index)
        handle = self.client.retriever(
            trace_id=target_trace_id,
            name="openarmature.rerank.complete",
            model=event.model,
            input=input_value,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            level="ERROR",
            status_message=event.error_category,
            start_time=start_time,
        )
        handle.end(end_time=end_time)

    def _resolve_llm_parent_observation_id(
        self,
        inv_state: _InvState,
        *,
        calling_namespace_prefix: tuple[str, ...],
        calling_attempt_index: int,
        calling_fan_out_index: int | None,
        calling_branch_name: str | None,
    ) -> str | None:
        # Calling-node identity precedence:
        #   1. Exact-match leaf node at the calling key.
        #   2. Per-instance fan-out dispatch observation when the
        #      call originated inside a fan-out instance.
        #   3. Subgraph dispatch observations along the calling
        #      namespace prefix, walked longest-prefix-first.
        #   4. None — Trace becomes the implicit parent.
        # The dispatch fallbacks cover the wrapped-call cases the
        # exact-match miss would otherwise need a leaf-ancestor walk
        # to handle.
        key: _StackKey = (
            calling_namespace_prefix,
            calling_attempt_index,
            calling_fan_out_index,
            calling_branch_name,
        )
        observation = inv_state.open_observations.get(key)
        if observation is not None:
            return observation.handle.id
        # Per-instance fan-out dispatch. The dispatch map is keyed by the
        # lineage-aware _dispatch_key; reconstruct the top-level instance's key
        # (namespace[:1], the instance index, no branch axis) to match it. Only
        # the innermost fan_out_index is available here, so this resolves an LLM
        # call directly inside a top-level fan-out instance (the case the flat
        # ``namespace[:1] + (str(index),)`` key handled before the lineage keys).
        if calling_fan_out_index is not None and calling_namespace_prefix:
            instance_key = _dispatch_key(calling_namespace_prefix[:1], (calling_fan_out_index,), (None,))
            dispatch = inv_state.fan_out_instance_observations.get(instance_key)
            if dispatch is not None:
                return dispatch.handle.id
        # Subgraph dispatch, longest-prefix-first.
        for prefix_len in range(len(calling_namespace_prefix), 0, -1):
            prefix = calling_namespace_prefix[:prefix_len]
            sg = inv_state.subgraph_observations.get(prefix)
            if sg is not None:
                return sg.handle.id
        return None

    def _typed_event_metadata(
        self, event: LlmCompletionEvent | LlmFailedEvent, correlation_id: str | None
    ) -> dict[str, Any]:
        """Build the Generation observation's metadata dict from a
        typed LLM event. Shared between the success path
        (LlmCompletionEvent) and the failure path (LlmFailedEvent).
        Response-side metadata (finish_reason / response_model /
        response_id) renders for a completion and for a
        structured_output_invalid failure (its wire response was intact);
        every other failure category renders none, per the guard below."""
        metadata: dict[str, Any] = {}
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        # §8.4.2: the OA-emitted fan_out_index / branch_name scoping rows,
        # mirroring _observation_metadata so a Generation inside a fan-out
        # instance or branch carries the same scoping as its node observation.
        if event.fan_out_index is not None:
            metadata["fan_out_index"] = event.fan_out_index
        if event.branch_name is not None:
            metadata["branch_name"] = event.branch_name
        metadata["system"] = event.provider
        active_prompt = event.active_prompt
        if active_prompt is not None:
            metadata["prompt"] = {
                "name": active_prompt.name,
                "version": active_prompt.version,
                "label": active_prompt.label,
                "template_hash": active_prompt.template_hash,
                "rendered_hash": active_prompt.rendered_hash,
            }
        active_group = event.active_prompt_group
        if active_group is not None:
            metadata["prompt_group_name"] = active_group.group_name
        if event.caller_invocation_metadata is not None:
            _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        # Response-side metadata. A completion always carries it; a
        # structured_output_invalid failure also carries it (proposal 0082),
        # since its wire response was intact, so finish_reason (the truncation
        # signal) and the response identity render on the failed Generation too.
        # Every other failure category received no response and renders none.
        if isinstance(event, LlmCompletionEvent):
            renders_response_side = True
        else:
            # event narrows to LlmFailedEvent (the only other union member).
            renders_response_side = event.error_category == "structured_output_invalid"
        if renders_response_side:
            if event.finish_reason is not None:
                metadata["finish_reason"] = event.finish_reason
            if event.response_model is not None:
                metadata["response_model"] = event.response_model
            if event.response_id is not None:
                metadata["response_id"] = event.response_id
        return metadata

    def _usage_from_typed_event(self, event: LlmCompletionEvent | LlmFailedEvent) -> LangfuseUsage | None:
        """Map the typed event's Usage onto the Langfuse Usage record.
        Returns None when no usage was reported."""
        # Spec observability §8.4.3 (Langfuse usage mapping).
        usage = event.usage
        if usage is None:
            return None
        if usage.prompt_tokens is None and usage.completion_tokens is None and usage.total_tokens is None:
            return None
        return LangfuseUsage(
            input=usage.prompt_tokens,
            output=usage.completion_tokens,
            total=usage.total_tokens,
        )

    def _resolve_prompt_link_from_typed_event(self, event: LlmCompletionEvent | LlmFailedEvent) -> Any:
        """Case discrimination on the typed event's active_prompt
        snapshot."""
        # Spec observability §8.4.4.
        active_prompt = event.active_prompt
        if active_prompt is None:
            return None
        entities = getattr(active_prompt, "observability_entities", None)
        if not isinstance(entities, dict):
            return None
        return cast("dict[str, Any]", entities).get("langfuse_prompt")

    def _open_trace_for_typed_event(
        self,
        invocation_id: str,
        correlation_id: str | None,
        event: LlmCompletionEvent
        | LlmFailedEvent
        | ToolCallEvent
        | ToolCallFailedEvent
        | EmbeddingEvent
        | EmbeddingFailedEvent
        | RerankEvent
        | RerankFailedEvent,
    ) -> None:
        """Trace open path for a typed event (LLM completion / failure,
        tool execution, embedding, or rerank) arriving before any node-started
        event reached
        this observer. Synthesizes the minimal trace shape from the
        typed event's scoping fields (all read generically)."""
        if event.namespace:
            entry_node = event.namespace[0]
        else:
            entry_node = event.node_name or "openarmature.llm.complete"
        metadata: dict[str, Any] = {
            "entry_node": entry_node,
            "spec_version": self.spec_version,
            "implementation_name": self.implementation_name,
            "implementation_version": self.implementation_version,
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        if event.caller_invocation_metadata is not None:
            _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        self._client_trace(id=invocation_id, name=entry_node, metadata=metadata)
        self._inv_states[invocation_id] = _InvState(trace_id=invocation_id)

    def _maybe_truncate_for_input(self, value: Any) -> Any:
        # Returns the native value (list of message dicts) when it
        # fits the cap, or the truncated marker-bearing string when
        # it doesn't. The list-or-str union return is intentional per
        # spec §8.7: the unparseable JSON IS the truncation signal —
        # surfacing the marker preserves the diagnostic without
        # faking a parse, and the Langfuse UI renders the string view
        # rather than the structured-input view. Callers MUST NOT
        # assume the return value is JSON-parseable.
        serialized = self._serialize_payload_value(value)
        truncated = _truncate(serialized, self.payload_byte_cap)
        if truncated is None:
            return value  # fits cap, native shape preserved
        return truncated

    def _maybe_truncate_for_output(self, value: str) -> str:
        # generation.output is a plain string in Langfuse's shape;
        # apply the cap directly to the source string.
        truncated = _truncate(value, self.payload_byte_cap)
        return truncated if truncated is not None else value

    def _maybe_truncate_for_extras(self, value: dict[str, Any]) -> Any:
        # request_extras goes on metadata as a native dict when it
        # fits, or the truncated marker-bearing string when it
        # doesn't. The dict-or-str union return mirrors
        # _maybe_truncate_for_input's intentional shape per spec §8.7:
        # the unparseable JSON IS the truncation signal, and the
        # Langfuse UI renders the string view in that case.
        serialized = self._serialize_payload_value(value)
        truncated = _truncate(serialized, self.payload_byte_cap)
        if truncated is None:
            return value
        return truncated

    @staticmethod
    def _serialize_payload_value(value: Any) -> str:
        # Mirrors observability/otel/observer.py's _serialize_for_attribute
        # so both observers see the same string under the same cap.
        # ``default=str`` is the same safety net: an opaque tool result
        # JSON can't natively encode renders via str() rather than
        # raising inside the observer (no-op for the encodable payloads).
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _truncate(serialized: str, cap_bytes: int) -> str | None:
    # Returns None when the serialized form fits within cap_bytes,
    # or the truncated-with-marker string otherwise. Mirrors the OTel
    # observer's _truncate_for_attribute algorithm (UTF-8 code-point
    # boundary backtracking, marker append).
    encoded = serialized.encode("utf-8")
    full_length = len(encoded)
    if full_length <= cap_bytes:
        return None
    marker = _TRUNCATION_MARKER_TEMPLATE.format(m=full_length)
    marker_bytes = marker.encode("utf-8")
    target = cap_bytes - len(marker_bytes)
    if target <= 0:
        return marker
    boundary = target
    while boundary > 0 and (encoded[boundary] & 0b1100_0000) == 0b1000_0000:
        boundary -= 1
    return encoded[:boundary].decode("utf-8", errors="strict") + marker


__all__ = ["LangfuseObserver"]
