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
#   appear only when `disable_llm_payload=False`; the truncation
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
    InvocationCompletedEvent,
    InvocationStartedEvent,
    LlmCompletionEvent,
    MetadataAugmentationEvent,
    NodeEvent,
)
from openarmature.observability.lineage import is_strict_prefix
from openarmature.observability.llm_event import LLM_NAMESPACE, LlmEventPayload

from .client import (
    LangfuseClient,
    LangfuseGenerationHandle,
    LangfuseSpanHandle,
    LangfuseUsage,
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


@dataclass
class _OpenObservation:
    """An in-flight Langfuse observation pinned in the observer's state.

    Per proposal 0045: carries the observation's own
    ``fan_out_index_chain`` and ``branch_name_chain`` so the
    augmentation walk can apply §3.4's lineage-aware boundary rule
    (mirror of the OTel observer's ``_OpenSpan``)."""

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


def _empty_str_frozenset() -> frozenset[str]:
    """Typed empty frozenset factory for ``detached_subgraphs`` /
    ``detached_fan_outs`` defaults."""
    return frozenset()


def _apply_caller_metadata(metadata: dict[str, Any], caller_metadata: Mapping[str, Any]) -> None:
    """Merge caller-supplied invocation metadata into a Trace's or
    Observation's metadata bag at top level per observability §8.4.1
    + §8.4.2 (proposal 0034).

    Top-level placement is by spec: Langfuse UI filters on
    ``metadata.<key>`` directly, so caller-supplied entries become
    siblings to ``correlation_id`` / ``entry_node`` rather than
    nested under a ``user`` sub-object.

    Reserved-key collision with §8.4.1 / §8.4.2 keys
    (``correlation_id``, ``entry_node``, ``spec_version``,
    ``namespace``, etc.) is not currently checked here: the spec
    permits the rejection to happen at either boundary, and the
    ``invoke()`` API-boundary validation already rejects
    ``openarmature.*`` / ``gen_ai.*`` prefixed keys. Per-Langfuse-
    backend collision rejection is queued as a follow-up.
    """
    for key, value in caller_metadata.items():
        metadata[key] = value


def _subgraph_identity_at(event: NodeEvent, depth: int) -> str:
    """Return the compiled-subgraph identity for the wrapper at the
    given 1-based namespace depth, or the empty string when no
    identity is tracked at that depth.

    Per observability §5.3 + the coord-thread
    ``clarify-subgraph-name-semantics`` resolution: the empty-string
    fallback matches the spec's "if the implementation tracks one"
    clause for implementations / direct ``SubgraphNode(...)`` callers
    that don't wire an identity through. Conformance fixtures
    031/032/033 lock identity as the required value; the empty-string
    path keeps direct callers conformant with §5.3 but failing those
    fixtures.
    """
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
    fan_out_instance_observations: dict[tuple[str, ...], _OpenObservation] = field(
        default_factory=dict[tuple[str, ...], _OpenObservation]
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
    fan_out_instance_root_prefixes: set[tuple[str, ...]] = field(default_factory=set[tuple[str, ...]])
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
    """Observer-driven Langfuse mapping per spec observability §8.

    Construct with a :class:`LangfuseClient` — the bundled
    :class:`InMemoryLangfuseClient` for tests, or a real
    ``langfuse.Langfuse()`` instance for production. The observer
    handles the §6 event stream and emits Trace + Observation entities
    through the client.

    Constructor knobs:

    - ``client``: the Langfuse sink (Protocol-typed).
    - ``disable_llm_spans``: when ``True`` the observer skips
      Generation observations on LLM provider events.
    - ``disable_llm_payload``: default ``True`` per §8.9's "symmetric
      privacy posture" with the OTel observer. Gates
      ``generation.input`` / ``output`` / ``metadata.request_extras``
      emission.
    - ``payload_byte_cap``: per-attribute byte cap on the source
      payload string before parse-back. Mirrors the OTel observer's
      ``payload_max_bytes`` semantic — emission preserves the raw
      truncated string when the §5.5.5 marker is present (per §8.7).
      Default 64 KiB; same minimum (256 bytes) applies.
    - ``detached_subgraphs``: set of subgraph wrapper node names that
      run in their own Langfuse Trace per §8.5. Each such subgraph
      gets a fresh trace_id; the main Trace's dispatch observation
      surfaces the link via ``metadata.detached_child_trace_ids``.
    - ``detached_fan_outs``: set of fan-out node names whose instances
      each get their own Langfuse Trace. Same link mechanism on the
      fan-out node observation: each per-instance detached trace_id
      lands in the array.
    - ``disable_state_payload``: default ``True`` per §8.4.1 *Trace
      input/output sourcing* (proposal 0043). When ``True`` the
      observer does NOT serialize ``initial_state`` / final state
      directly onto ``trace.input`` / ``trace.output``; the minimal
      stub applies unless ``trace_input_from_state`` /
      ``trace_output_from_state`` overrides. When ``False`` the raw
      state object is serialized to the Trace fields, subject to
      ``payload_byte_cap`` truncation. Independent of
      ``disable_llm_payload`` — the two payloads carry distinct
      threat models (LLM-call transcript vs. application state).
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
      Defaults to ``openarmature.__version__``. Always-emit invariant
      inherited from §5.1 — not gated by ``disable_state_payload``,
      ``disable_llm_payload``, or any other privacy knob.

    The observer reads the spec version from the package at
    construction time. Safe to share across concurrent invocations
    and across resumes of the same correlation_id; per-invocation
    state isolation keys all internal maps by invocation_id.
    """

    client: LangfuseClient
    disable_llm_spans: bool = False
    disable_llm_payload: bool = True
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
        event: (
            NodeEvent
            | MetadataAugmentationEvent
            | InvocationStartedEvent
            | InvocationCompletedEvent
            | LlmCompletionEvent
        ),
    ) -> None:
        if isinstance(event, InvocationStartedEvent):
            self._handle_invocation_started(event)
            return
        if isinstance(event, InvocationCompletedEvent):
            self._handle_invocation_completed(event)
            return
        # Proposal 0049 typed LlmCompletionEvent (success path). Drives
        # the §5.5 Generation observation lifecycle for successful
        # provider calls. Failures don't emit this variant; they flow
        # through the sentinel error path below (a single sentinel
        # ``completed`` event — no started counterpart in v0.13.0+).
        if isinstance(event, LlmCompletionEvent):
            if not self.disable_llm_spans:
                self._handle_typed_llm_completion(event)
            return
        if isinstance(event, MetadataAugmentationEvent):
            self._handle_metadata_augmentation(event)
            return
        # LLM provider sentinel events: failure-path completed opens +
        # closes an ERROR-level Generation; everything else is a no-op
        # (success-path typed handler above owns the Generation).
        if event.namespace == LLM_NAMESPACE:
            if not self.disable_llm_spans:
                self._handle_llm_error_event(event)
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
        # themselves; this cache bridges).
        if event.fan_out_config is not None and event.fan_out_index is None:
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
            for prefix in list(inv_state.fan_out_instance_root_prefixes):
                if len(prefix) > len(event.namespace) and prefix[: len(event.namespace)] == event.namespace:
                    # Detached per-instance dispatches live in
                    # fan_out_instance_observations (same map as
                    # non-detached); close via the matching helper.
                    self._close_fan_out_instance_dispatch_observation(inv_state, prefix)
                    inv_state.fan_out_instance_root_prefixes.discard(prefix)
                    inv_state.detached_traces.pop(prefix, None)
        # Per spec proposal 0013 (v0.10.0): when the fan-out node's
        # own completion fires, close all per-instance dispatch
        # observations synthesized for it. Children-before-parents.
        if event.fan_out_index is None and event.fan_out_config is not None:
            for prefix in list(inv_state.fan_out_instance_observations.keys()):
                if len(prefix) > len(event.namespace) and prefix[: len(event.namespace)] == event.namespace:
                    self._close_fan_out_instance_dispatch_observation(inv_state, prefix)
            inv_state.fan_out_parent_node_name.pop(event.namespace, None)
            # Clear the detached-child-trace-ids accumulator for this
            # fan-out node — cyclic execution that re-enters the same
            # fan-out starts the next iteration with a fresh list
            # rather than appending to the previous iteration's
            # accumulator and overwriting the prior link metadata.
            inv_state.detached_child_trace_ids.pop(event.namespace, None)
        # Per proposal 0045: clean up the pb cache on a pb NODE's own
        # completion.  Same shape as the fan-out cleanup above.
        if event.parallel_branches_config is not None:
            inv_state.parallel_branches_parent_node_name.pop(event.namespace, None)

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

        # Fan-out instance dispatch observations.
        for key, observation in inv_state.fan_out_instance_observations.items():
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
            observation.handle.update(metadata=metadata_delta)

        # Open NODE observations.  Same as augmenter or strict
        # ancestor on the path; skip shared-parent NODE observations
        # (fan-out NODE / pb NODE) identified by presence in the
        # parent_node_name caches.
        for key, observation in inv_state.open_observations.items():
            ns, _ai, _fi, _bn = key
            if ns == aug_ns:
                if _observation_chain_on_path(observation, aug_fi_chain, aug_bn_chain):
                    observation.handle.update(metadata=metadata_delta)
                continue
            if not is_strict_prefix(ns, aug_ns):
                continue
            if ns in inv_state.fan_out_parent_node_name or ns in inv_state.parallel_branches_parent_node_name:
                continue
            if _observation_chain_on_path(observation, aug_fi_chain, aug_bn_chain):
                observation.handle.update(metadata=metadata_delta)

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
        self.client.trace(id=invocation_id, name=entry_node, metadata=metadata)
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
        self.client.trace(id=invocation_id, name=trace_name, metadata=metadata)
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
        if event.fan_out_index is not None and event.namespace:
            instance_key = event.namespace[:1] + (str(event.fan_out_index),)
            dispatch = inv_state.fan_out_instance_observations.get(instance_key)
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
            # Non-detached per-instance dispatch for the current
            # event's own fan-out instance gets opened below; skip
            # the regular subgraph path here so we don't double-open.
            if (
                depth == 1
                and event.fan_out_index is not None
                and (prefix + (str(event.fan_out_index),)) in inv_state.fan_out_instance_observations
            ):
                continue
            # Detached subgraph: the first segment matches a
            # configured detached_subgraphs name → mint a fresh
            # detached Trace + open the dispatch observation in it.
            if depth == 1 and prefix[0] in self.detached_subgraphs:
                self._open_detached_subgraph_trace(inv_state, correlation_id, prefix, event)
                continue
            # Detached fan-out: the fan-out instance gets its own
            # Trace per spec §8.5. The fan-out node's Span observation
            # in the parent Trace already exists (opened on the
            # fan-out node's started event); the detached dispatch
            # observation goes into the new Trace.
            if depth == 1 and event.fan_out_index is not None and prefix[0] in self.detached_fan_outs:
                self._open_detached_fan_out_instance_trace(inv_state, correlation_id, prefix, event)
                continue
            # Non-detached fan-out: synthesize per-instance dispatch
            # observation under the fan-out node observation (proposal
            # 0013 v0.10.0). Only triggers when the inner event is
            # inside a fan-out instance AND the fan-out node's
            # parent_node_name has been cached (i.e., the fan-out
            # node's own started event was seen).
            if (
                depth == 1
                and event.fan_out_index is not None
                and prefix[0] not in self.detached_fan_outs
                and prefix in inv_state.fan_out_parent_node_name
            ):
                self._open_fan_out_instance_dispatch_observation(inv_state, correlation_id, prefix, event)
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
        fan_out_open = self._find_fan_out_node_observation(inv_state, prefix)
        parent_observation_id = fan_out_open.handle.id if fan_out_open is not None else None
        parent_node_name = inv_state.fan_out_parent_node_name.get(prefix, prefix[-1])
        # Per-instance dispatch is synthesized from the first inner
        # event inside the instance subtree; inherit scalar metadata
        # from that event (same pattern as ``_open_subgraph_observation``).
        metadata: dict[str, Any] = {
            "namespace": list(prefix),
            "step": event.step,
            "attempt_index": 0,
            "fan_out_parent_node_name": parent_node_name,
            "fan_out_index": event.fan_out_index,
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
        instance_key = prefix + (str(event.fan_out_index),)
        # Per proposal 0045: chain sliced to instance-dispatch depth.
        chain_len = len(prefix)
        inv_state.fan_out_instance_observations[instance_key] = _OpenObservation(
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
        self.client.trace(id=detached_trace_id, name=wrapper_obs_name, metadata=detached_metadata)
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
        fan_out_open = self._find_fan_out_node_observation(inv_state, prefix)
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
        self.client.trace(
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
        instance_key = prefix + (str(event.fan_out_index),)
        chain_len = len(prefix)
        inv_state.fan_out_instance_observations[instance_key] = _OpenObservation(
            handle=handle,
            fan_out_index_chain=event.fan_out_index_chain[:chain_len],
            branch_name_chain=event.branch_name_chain[:chain_len],
        )
        inv_state.detached_traces[instance_key] = detached_trace_id
        inv_state.fan_out_instance_root_prefixes.add(instance_key)

    def _close_subgraph_observation(self, inv_state: _InvState, prefix: tuple[str, ...]) -> None:
        observation = inv_state.subgraph_observations.pop(prefix, None)
        if observation is None:
            return
        observation.handle.end()

    def _close_fan_out_instance_dispatch_observation(
        self, inv_state: _InvState, prefix: tuple[str, ...]
    ) -> None:
        observation = inv_state.fan_out_instance_observations.pop(prefix, None)
        if observation is None:
            return
        observation.handle.end()

    def _find_fan_out_node_observation(
        self, inv_state: _InvState, prefix: tuple[str, ...]
    ) -> _OpenObservation | None:
        # Find the fan-out node's open leaf observation at the given
        # prefix. Retry middleware wrapping a fan-out bumps the
        # attempt_index; this scans for any entry at ``prefix`` with
        # ``fan_out_index is None``. Only one such entry is open at a
        # time (retry opens and closes within an attempt's lifecycle).
        for key, observation in inv_state.open_observations.items():
            if key[0] == prefix and key[2] is None:
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

    # v0.13.0 (proposal 0049 + 0057): success-path Generation lifecycle
    # is driven by the typed LlmCompletionEvent — opened and closed in
    # one shot at typed-event arrival, with start_time back-dated by
    # latency_ms so the observation's duration reflects the adapter-
    # boundary measurement rather than dispatcher queue delay. Failure
    # path keeps a single sentinel NodeEvent (``completed`` phase
    # carrying error fields on its LlmEventPayload — LlmCompletionEvent
    # is success-only per proposal 0049 §3 alternative 3). The provider
    # dropped success-path sentinel emission entirely in this release,
    # so on success the typed event is the only signal the Generation
    # observation has to fire from; the failure path's sentinel
    # ``started`` was also dropped, leaving only ``completed``.
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
        if not self.disable_llm_payload:
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

    def _handle_llm_error_event(self, event: NodeEvent) -> None:
        """Emit an ERROR-level Generation observation from the sentinel
        NodeEvent on the failure path. Success-path sentinel completion
        is no longer emitted by the provider in v0.13.0; this handler
        only fires for failures."""
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        if event.phase != "completed":
            # Sentinel started becomes a no-op once the success-side
            # emission drops. Failures only emit the completed half.
            return
        if not isinstance(event.pre_state, LlmEventPayload):
            return
        payload = event.pre_state
        if payload.error_type is None:
            # Defensive — success path no longer emits the sentinel
            # pair; if a non-error sentinel completion slips through
            # (e.g., legacy custom provider not yet migrated), the
            # typed event handler owns the Generation.
            return
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        correlation_id = current_correlation_id()
        if invocation_id not in self._inv_states:
            self._open_trace(invocation_id, correlation_id, event)
        inv_state = self._inv_states[invocation_id]
        parent_observation_id = self._resolve_llm_parent_observation_id(
            inv_state,
            calling_namespace_prefix=payload.calling_namespace_prefix,
            calling_attempt_index=payload.calling_attempt_index,
            calling_fan_out_index=payload.calling_fan_out_index,
            calling_branch_name=payload.calling_branch_name,
        )
        metadata, model_parameters, input_value, _ = self._llm_metadata_and_payload(
            payload, correlation_id, phase="started"
        )
        target_trace_id = self._trace_id_for(
            inv_state, payload.calling_namespace_prefix, payload.calling_fan_out_index
        )
        handle = self.client.generation(
            trace_id=target_trace_id,
            name="openarmature.llm.complete",
            model=payload.model,
            model_parameters=model_parameters,
            input=input_value,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            prompt=self._resolve_prompt_link(payload),
        )
        # Error-category mapping: §8.4.2 + §8.4.3.
        end_kwargs: dict[str, Any] = {
            "level": "ERROR",
            "status_message": payload.error_category or payload.error_type,
        }
        handle.end(**end_kwargs)

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
        # Per-instance fan-out dispatch.
        if calling_fan_out_index is not None and calling_namespace_prefix:
            instance_key = calling_namespace_prefix[:1] + (str(calling_fan_out_index),)
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

    def _typed_event_metadata(self, event: LlmCompletionEvent, correlation_id: str | None) -> dict[str, Any]:
        """Build the Generation observation's metadata dict from the
        typed event. Mirrors _llm_metadata_and_payload's metadata
        construction but reads from LlmCompletionEvent fields, and
        combines started + completed phases into a single populated
        dict (the typed event carries everything at once)."""
        metadata: dict[str, Any] = {}
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
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
        # Asymmetric guard with _llm_metadata_and_payload below: the
        # typed event types caller_invocation_metadata as Mapping | None
        # while LlmEventPayload defaults to an empty mapping (never
        # None). Don't "normalize" the two paths without normalizing
        # the source types.
        if event.caller_invocation_metadata is not None:
            _apply_caller_metadata(metadata, event.caller_invocation_metadata)
        if event.finish_reason is not None:
            metadata["finish_reason"] = event.finish_reason
        if event.response_model is not None:
            metadata["response_model"] = event.response_model
        if event.response_id is not None:
            metadata["response_id"] = event.response_id
        return metadata

    def _usage_from_typed_event(self, event: LlmCompletionEvent) -> LangfuseUsage | None:
        """Map the typed event's Usage onto the Langfuse Usage record
        per §8.4.3. Returns None when no usage was reported."""
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

    def _resolve_prompt_link_from_typed_event(self, event: LlmCompletionEvent) -> Any:
        """§8.4.4 case discrimination on the typed event's active_prompt
        snapshot. Same logic as _resolve_prompt_link but reads from
        LlmCompletionEvent instead of LlmEventPayload."""
        active_prompt = event.active_prompt
        if active_prompt is None:
            return None
        entities = getattr(active_prompt, "observability_entities", None)
        if not isinstance(entities, dict):
            return None
        return cast("dict[str, Any]", entities).get("langfuse_prompt")

    def _open_trace_for_typed_event(
        self, invocation_id: str, correlation_id: str | None, event: LlmCompletionEvent
    ) -> None:
        """Trace open path for a typed LlmCompletionEvent arriving
        before any node-started event reached this observer.
        Synthesizes the minimal trace shape from the typed event's
        scoping fields."""
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
        self.client.trace(id=invocation_id, name=entry_node, metadata=metadata)
        self._inv_states[invocation_id] = _InvState(trace_id=invocation_id)

    def _llm_metadata_and_payload(
        self,
        payload: LlmEventPayload,
        correlation_id: str | None,
        *,
        phase: str,
    ) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
        # Returns (metadata, model_parameters, input, output) for the
        # generation(...) / .end(...) call. Phase-specific filtering
        # keeps the started call lean (input only) and the completed
        # call focused on the output + usage + response metadata.
        metadata: dict[str, Any] = {}
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        # gen_ai.system → metadata.system per §8.4.3
        metadata["system"] = payload.genai_system
        # Prompt-identity metadata (§8.4.4 always-on, independent of
        # whether a Langfuse Prompt entity link is established).
        active_prompt = payload.active_prompt
        if active_prompt is not None:
            metadata["prompt"] = {
                "name": active_prompt.name,
                "version": active_prompt.version,
                "label": active_prompt.label,
                "template_hash": active_prompt.template_hash,
                "rendered_hash": active_prompt.rendered_hash,
            }
        active_group = payload.active_prompt_group
        if active_group is not None:
            metadata["prompt_group_name"] = active_group.group_name
        _apply_caller_metadata(metadata, payload.caller_invocation_metadata)

        model_parameters: dict[str, Any] = {}
        request_params = payload.request_params or {}
        # Per §8.4.3: every gen_ai.request.<suffix> attribute lifts to
        # generation.modelParameters.<suffix> by inclusion. The §5.5.2
        # source set keys this on (temperature, max_tokens, top_p,
        # seed, frequency_penalty, presence_penalty, stop_sequences as
        # of v0.24.0); new request-param attrs added in future spec
        # versions flow through automatically.
        for key, value in request_params.items():
            model_parameters[key] = value

        # Input/output payload gated by disable_llm_payload (§8.7).
        input_value: Any = None
        output_value: Any = None
        if not self.disable_llm_payload:
            if phase == "started" and payload.input_messages is not None:
                # The payload's input_messages is already image-
                # redacted at the provider per §5.5.5 (inline image
                # bytes never reach the observer). Serialize and
                # compare against the configured cap; under cap the
                # native shape is fine, over cap §8.7 says preserve
                # the raw truncated string with the marker.
                input_value = self._maybe_truncate_for_input(payload.input_messages)
            if phase == "completed" and payload.output_content is not None:
                output_value = self._maybe_truncate_for_output(payload.output_content)
            if phase == "started" and payload.request_extras:
                # request_extras renders into metadata, not the input
                # field, per §8.4.3 (`metadata.request_extras`).
                metadata["request_extras"] = self._maybe_truncate_for_extras(dict(payload.request_extras))

        # Response metadata fields land on the completed call (§8.4.3).
        if phase == "completed":
            if payload.finish_reason is not None:
                metadata["finish_reason"] = payload.finish_reason
            if payload.response_model is not None:
                metadata["response_model"] = payload.response_model
            if payload.response_id is not None:
                metadata["response_id"] = payload.response_id

        return metadata, model_parameters, input_value, output_value

    def _usage_from_payload(self, payload: LlmEventPayload) -> LangfuseUsage | None:
        # Map OA usage fields onto the Langfuse Usage record per
        # §8.4.3. Returns None when no usage was reported (all three
        # token fields None) so the Generation observation reflects
        # absence rather than zeroed counts.
        if (
            payload.prompt_tokens is None
            and payload.completion_tokens is None
            and payload.total_tokens is None
        ):
            return None
        return LangfuseUsage(
            input=payload.prompt_tokens,
            output=payload.completion_tokens,
            total=payload.total_tokens,
        )

    def _resolve_prompt_link(self, payload: LlmEventPayload) -> Any:
        # §8.4.4 case discrimination: the trigger is whether the
        # prompt's source exposes a Langfuse Prompt reference, not
        # which specific backend produced it. PromptResult has
        # observability_entities['langfuse_prompt'] populated when
        # case 1 applies; absent otherwise.
        active_prompt = payload.active_prompt
        if active_prompt is None:
            return None
        # PromptResult is typed Any on LlmEventPayload to avoid a
        # cross-package import (see llm_event.py for the rationale);
        # read defensively.
        entities = getattr(active_prompt, "observability_entities", None)
        if not isinstance(entities, dict):
            return None
        return cast("dict[str, Any]", entities).get("langfuse_prompt")

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
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
