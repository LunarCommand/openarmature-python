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

    The observer reads the spec version from the package at
    construction time. Safe to share across concurrent invocations
    and across resumes of the same correlation_id; per-invocation
    state isolation keys all internal maps by invocation_id.
    """

    client: LangfuseClient
    disable_llm_spans: bool = False
    disable_llm_payload: bool = True
    payload_byte_cap: int = 65536
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
        key = self._key_for(event)
        if key in inv_state.open_observations:
            # Idempotent: a second started for the same (namespace,
            # attempt_index, fan_out_index) tuple is a no-op (matches
            # the OTel observer's behavior under retry-replay).
            return

        parent_observation_id = self._resolve_parent_observation_id(inv_state, event)
        metadata = self._observation_metadata(event, correlation_id)
        handle = self.client.span(
            trace_id=inv_state.trace_id,
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
        # Walk namespace ancestors looking for the innermost open
        # observation; fall back to None (Trace becomes the parent).
        # Subgraph dispatch / fan-out per-instance / detached-trace
        # parenting are deferred from this version of the observer
        # (no fixtures exercise them); future PRs add per-spec-§8.3
        # synthetic dispatch observations.
        for prefix_len in range(len(event.namespace) - 1, 0, -1):
            prefix = event.namespace[:prefix_len]
            for key, observation in inv_state.open_observations.items():
                if key[0] == prefix:
                    return observation.handle.id
        return None

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
            handle = self.client.generation(
                trace_id=inv_state.trace_id,
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
        # dispatch time per llm-provider §5.5). Resolve the calling
        # node's open observation; fall back to None (Trace parent)
        # if not found.
        key: _StackKey = (
            payload.calling_namespace_prefix,
            payload.calling_attempt_index,
            payload.calling_fan_out_index,
        )
        observation = inv_state.open_observations.get(key)
        if observation is not None:
            return observation.handle.id
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
        # Returns the native value when it fits the cap, or the
        # truncated string-with-marker when it doesn't. §8.7's
        # "raw truncated string" rule: the unparseable JSON IS the
        # truncation signal, surfacing the marker rather than faking
        # a parse.
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
        # fits; falls through to the raw truncated string when it
        # doesn't, matching §8.7's parse-fallthrough story.
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
