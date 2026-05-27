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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from openarmature.observability.llm_event import LLM_NAMESPACE, LlmEventPayload

from .client import (
    LangfuseClient,
    LangfuseGenerationHandle,
    LangfuseSpanHandle,
    LangfuseUsage,
)

if TYPE_CHECKING:
    from openarmature.graph.events import NodeEvent


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


# In-flight Span observation handle, keyed by the standard span-stack
# key (namespace, attempt_index, fan_out_index). Mirrors the OTel
# observer's _OpenSpan shape but holds a Langfuse handle instead of an
# OTel Span.
_StackKey = tuple[tuple[str, ...], int, int | None]


@dataclass
class _OpenObservation:
    """An in-flight Langfuse observation pinned in the observer's state."""

    handle: LangfuseSpanHandle | LangfuseGenerationHandle


def _empty_str_frozenset() -> frozenset[str]:
    """Typed empty frozenset factory for ``detached_subgraphs`` /
    ``detached_fan_outs`` defaults."""
    return frozenset()


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
    open_llm_observations: dict[str, _OpenObservation] = field(default_factory=dict[str, _OpenObservation])
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

    async def __call__(self, event: NodeEvent) -> None:
        # LLM provider events use a sentinel namespace per §5.5; route
        # them to the dedicated Generation path.
        if event.namespace == LLM_NAMESPACE:
            if not self.disable_llm_spans:
                self._handle_llm_event(event)
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
        inv_state.open_observations[key] = _OpenObservation(handle=handle)

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

    def _open_trace(self, invocation_id: str, correlation_id: str | None, event: NodeEvent) -> None:
        metadata: dict[str, Any] = {
            "entry_node": event.node_name,
            "spec_version": self.spec_version,
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        # §8.6 trace name: caller-supplied invocation label takes
        # precedence; entry-node name is the spec-recommended fallback.
        # The caller-supplied path lands in proposal 0034 (PR 4) — for
        # now only the fallback is wired.
        trace_name = event.node_name
        self.client.trace(id=invocation_id, name=trace_name, metadata=metadata)
        self._inv_states[invocation_id] = _InvState(trace_id=invocation_id)

    def _key_for(self, event: NodeEvent) -> _StackKey:
        return (event.namespace, event.attempt_index, event.fan_out_index)

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
                self._open_detached_subgraph_trace(inv_state, correlation_id, prefix)
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
            self._open_subgraph_observation(inv_state, correlation_id, prefix)

    def _open_subgraph_observation(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
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
        metadata: dict[str, Any] = {"subgraph_name": prefix[-1]}
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        handle = self.client.span(
            trace_id=inv_state.trace_id,
            name=prefix[-1],
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        inv_state.subgraph_observations[prefix] = _OpenObservation(handle=handle)

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
        metadata: dict[str, Any] = {
            "fan_out_parent_node_name": parent_node_name,
            "fan_out_index": event.fan_out_index,
        }
        if correlation_id is not None:
            metadata["correlation_id"] = correlation_id
        handle = self.client.span(
            trace_id=inv_state.trace_id,
            name=prefix[-1],
            metadata=metadata,
            parent_observation_id=parent_observation_id,
        )
        instance_key = prefix + (str(event.fan_out_index),)
        inv_state.fan_out_instance_observations[instance_key] = _OpenObservation(handle=handle)

    def _open_detached_subgraph_trace(
        self,
        inv_state: _InvState,
        correlation_id: str | None,
        prefix: tuple[str, ...],
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
        link_metadata: dict[str, Any] = {
            "subgraph_name": prefix[-1],
            "detached_child_trace_ids": [detached_trace_id],
        }
        if correlation_id is not None:
            link_metadata["correlation_id"] = correlation_id
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
        self.client.trace(id=detached_trace_id, name=prefix[-1], metadata=detached_metadata)
        dispatch_metadata: dict[str, Any] = {
            "subgraph_name": prefix[-1],
            "detached": True,
        }
        if correlation_id is not None:
            dispatch_metadata["correlation_id"] = correlation_id
        handle = self.client.span(
            trace_id=detached_trace_id,
            name=prefix[-1],
            metadata=dispatch_metadata,
            parent_observation_id=None,
        )
        inv_state.subgraph_observations[prefix] = _OpenObservation(handle=handle)
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
            link_metadata: dict[str, Any] = {
                "detached_child_trace_ids": list(ids_list),
            }
            if correlation_id is not None:
                link_metadata["correlation_id"] = correlation_id
            fan_out_open.handle.update(metadata=link_metadata)
        # Open the detached Trace + per-instance dispatch observation.
        detached_metadata: dict[str, Any] = {
            "detached_from_invocation_id": inv_state.trace_id,
            "fan_out_index": event.fan_out_index,
        }
        if correlation_id is not None:
            detached_metadata["correlation_id"] = correlation_id
        self.client.trace(
            id=detached_trace_id,
            name=prefix[-1],
            metadata=detached_metadata,
        )
        parent_node_name = inv_state.fan_out_parent_node_name.get(prefix, prefix[-1])
        dispatch_metadata: dict[str, Any] = {
            "fan_out_parent_node_name": parent_node_name,
            "fan_out_index": event.fan_out_index,
            "detached": True,
        }
        if correlation_id is not None:
            dispatch_metadata["correlation_id"] = correlation_id
        handle = self.client.span(
            trace_id=detached_trace_id,
            name=prefix[-1],
            metadata=dispatch_metadata,
            parent_observation_id=None,
        )
        instance_key = prefix + (str(event.fan_out_index),)
        inv_state.fan_out_instance_observations[instance_key] = _OpenObservation(handle=handle)
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
        # closed before they end. LLM observations → leaf nodes
        # (sorted deepest-first by namespace length) → per-instance
        # fan-out dispatches → subgraph dispatches.
        for call_id in list(inv_state.open_llm_observations.keys()):
            obs = inv_state.open_llm_observations.pop(call_id, None)
            if obs is not None:
                obs.handle.end()
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
        return metadata

    # ------------------------------------------------------------------
    # Generation observation lifecycle (LLM provider events)
    # ------------------------------------------------------------------

    def _handle_llm_event(self, event: NodeEvent) -> None:
        from openarmature.observability.correlation import (
            current_correlation_id,
            current_invocation_id,
        )

        if not isinstance(event.pre_state, LlmEventPayload):
            # Defensive — sentinel-namespaced events MUST carry an
            # LlmEventPayload per llm-provider / observability §5.5.
            return
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return
        payload = event.pre_state
        # The Trace MAY not exist yet if the LLM call fires before any
        # node `started` event has hit this observer (race-y under
        # tests that prepare via `prepare_sync` only). The in-memory
        # client tolerates create-on-demand; production SDK adapters
        # should too.
        if invocation_id not in self._inv_states:
            self._open_trace(invocation_id, current_correlation_id(), event)
        inv_state = self._inv_states[invocation_id]
        correlation_id = current_correlation_id()

        if event.phase == "started":
            parent_observation_id = self._resolve_llm_parent_observation_id(inv_state, payload)
            metadata, model_parameters, input_value, output_value = self._llm_metadata_and_payload(
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
                output=output_value,
                metadata=metadata,
                parent_observation_id=parent_observation_id,
                prompt=self._resolve_prompt_link(payload),
            )
            inv_state.open_llm_observations[payload.call_id] = _OpenObservation(handle=handle)
            return

        # completed: pop the started handle and finalize.
        observation = inv_state.open_llm_observations.pop(payload.call_id, None)
        if observation is None:
            return
        metadata, _model_parameters, _input_value, output_value = self._llm_metadata_and_payload(
            payload, correlation_id, phase="completed"
        )
        end_kwargs: dict[str, Any] = {"metadata": metadata}
        if output_value is not None:
            end_kwargs["output"] = output_value
        usage = self._usage_from_payload(payload)
        if usage is not None:
            end_kwargs["usage"] = usage
        # Error-category mapping: §8.4.2 + §8.4.3 (an LLM provider
        # error_category lands on the Generation observation's level
        # and statusMessage the same as on a Span observation).
        if payload.error_category is not None:
            end_kwargs["level"] = "ERROR"
            end_kwargs["status_message"] = payload.error_category
        observation.handle.end(**end_kwargs)

    def _resolve_llm_parent_observation_id(
        self, inv_state: _InvState, payload: LlmEventPayload
    ) -> str | None:
        # Calling-node identity comes from the payload (set at
        # dispatch time per llm-provider §5.5). Precedence:
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
            payload.calling_namespace_prefix,
            payload.calling_attempt_index,
            payload.calling_fan_out_index,
        )
        observation = inv_state.open_observations.get(key)
        if observation is not None:
            return observation.handle.id
        # Per-instance fan-out dispatch.
        if payload.calling_fan_out_index is not None and payload.calling_namespace_prefix:
            instance_key = payload.calling_namespace_prefix[:1] + (str(payload.calling_fan_out_index),)
            dispatch = inv_state.fan_out_instance_observations.get(instance_key)
            if dispatch is not None:
                return dispatch.handle.id
        # Subgraph dispatch, longest-prefix-first.
        for prefix_len in range(len(payload.calling_namespace_prefix), 0, -1):
            prefix = payload.calling_namespace_prefix[:prefix_len]
            sg = inv_state.subgraph_observations.get(prefix)
            if sg is not None:
                return sg.handle.id
        return None

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
