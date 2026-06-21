# Backend mappings (the OTel observer in this repo, future Langfuse /
# Datadog adapters) recognize the LLM_NAMESPACE sentinel and read these
# fields directly via attribute access.
#
# call_id is the per-call disambiguator: a UUIDv4 minted by the
# provider and shared across the started / completed event pair.
# Backend observers key their in-flight LLM-span maps by it so
# concurrent complete() calls (e.g., fan-out instances each calling
# the provider) don't collide on the single sentinel-namespace key.
#
# calling_namespace_prefix / calling_attempt_index /
# calling_fan_out_index carry the calling node's identity so the OTel
# observer can resolve §5.5 "parent under calling node" correctly
# under concurrent fan-out and retry. Populated from the engine's
# ContextVars at dispatch time; sentinel defaults when the provider
# is called outside any node body.
#
# active_prompt / active_prompt_group are dispatch-time snapshots of
# the prompts-context ContextVars (per friction-roundup #3). The
# delivery-worker task cannot read these ContextVars — its Context is
# snapshotted at invoke()-entry, before any node body opens a
# with_active_prompt block — so the snapshot has to travel on the
# payload.
#
# input_messages / output_content / request_params / request_extras
# source the §5.5.1 + §5.5.2 attributes. input_messages is the message
# list serialized to §3 plain-dict shape with ImageSourceInline already
# redacted (per §5.5.5 — inline bytes never leave the provider in
# event form, regardless of any observer-side flag). response_id /
# response_model source the §5.5.3 gen_ai.response.{id,model}
# attributes. genai_system sources gen_ai.system per §5.5.3 (default
# "openai"; overridable on the OpenAI-compatible provider).

"""LLM event payload exchanged between providers and observability backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openarmature.llm.messages import ToolCall

# Sentinel namespace the LLM provider emits to signal "this is an LLM
# event, not a regular node event." Backend mappings (the OTel observer
# in this repo, future Langfuse / Datadog adapters) recognise this
# value on ``NodeEvent.namespace`` and route to their LLM-specific
# span path. Lives here rather than under ``otel/observer.py`` so the
# core observability package doesn't pull the OTel backend into its
# import chain — anyone consuming ``LlmEventPayload`` from a custom
# provider needs the namespace value too, and shouldn't have to
# install the ``[otel]`` extra just for the constant.
LLM_NAMESPACE: tuple[str, ...] = ("openarmature.llm.complete",)


# LlmEventPayload uses plain Pydantic BaseModel (not openarmature.graph.State)
# so importing it doesn't transitively load the entire graph package.
# That lets providers in openarmature.llm import this type cleanly even
# though graph.middleware.retry imports from openarmature.llm.errors —
# subclassing State would create a circular load order. NodeEvent.pre_state
# is typed Any (per the comment in graph.events) so the State-subclass
# constraint isn't load-bearing here.
class LlmEventPayload(BaseModel):
    """Typed payload carried on ``NodeEvent.pre_state`` for the
    ``openarmature.llm.complete`` event pair an LLM provider emits
    around each ``complete()`` call.

    Observers subscribing to events with namespace
    :data:`openarmature.observability.LLM_NAMESPACE` read attributes
    directly off this payload. The OpenAI provider populates every
    field; third-party providers populate the subset they support.
    """

    # Extra fields rejected at construction; instance frozen so
    # observers can't mutate payload data after dispatch.
    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # Cache-stat fields sourced from Response.usage per spec proposal
    # 0047 (§5.5.3.1 OA-namespace cache attributes). Absent (None)
    # when the provider does not report cache stats; set to 0 when
    # the provider reports zero hits (the "reported miss" case,
    # distinct from absent). The OTel observer emits the
    # openarmature.llm.cache_read.input_tokens span attribute when
    # cached_tokens is populated; same conditional for
    # cache_creation.
    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    # error_category is the canonical llm-provider §7 category
    # (provider_unavailable, etc.) when the failed exception carried
    # one — the provider caller doesn't have a graph-engine §4
    # RuntimeGraphError to attach to NodeEvent.error, so failure
    # detail surfaces through these fields instead.
    error_type: str | None = None
    error_message: str | None = None
    error_category: str | None = None
    # Calling-node identity captured at dispatch time. The OTel
    # observer reads these to look up the calling node's span in its
    # invocation_id-scoped _open_spans map without relying on the
    # OTel current-span context (which under concurrent fan-out can
    # yield a sibling instance's span).
    calling_namespace_prefix: tuple[str, ...] = ()
    calling_attempt_index: int = 0
    calling_fan_out_index: int | None = None
    # Calling-node branch_name (pipeline-utilities §11). Mirrors the
    # other ``calling_*`` fields; the OTel observer's open-span key
    # widening (``_StackKey`` now includes ``branch_name``) needs this
    # to disambiguate concurrent same-named inner nodes across sibling
    # branches.
    calling_branch_name: str | None = None
    # Prompt-context snapshot captured at dispatch time. ``Any``
    # because the prompts package imports State indirectly; the typed
    # shapes are PromptResult / PromptGroup from openarmature.prompts.
    # Observers cast back at the read site.
    active_prompt: Any = None
    active_prompt_group: Any = None
    # Payload + request-config carrier. input_messages is already
    # image-redacted by the provider before reaching this struct;
    # request_params carries only the gen_ai.request.* fields;
    # request_extras carries the RuntimeConfig extras pass-through bag.
    input_messages: list[dict[str, Any]] | None = None
    output_content: str | None = None
    request_params: dict[str, Any] | None = None
    request_extras: dict[str, Any] | None = None
    response_id: str | None = None
    response_model: str | None = None
    genai_system: str = "openai"
    # Per proposal 0034 / observability §3.4 + §5.6: snapshot of
    # caller-supplied invocation metadata captured at LLM-event
    # dispatch time (in the calling node's Context). Backend
    # observers read from the snapshot rather than re-reading the
    # ContextVar at observer time — the OTel + Langfuse observers
    # run on the engine's ``deliver_loop`` task whose Context is
    # frozen at invoke time, so mid-invocation augmentations made
    # by node bodies running in the main engine task are NOT visible
    # there. The snapshot pattern mirrors the existing
    # ``calling_namespace_prefix`` / ``calling_attempt_index`` /
    # ``calling_fan_out_index`` fields.
    caller_invocation_metadata: dict[str, Any] = Field(default_factory=dict)


def serialize_tool_calls(tool_calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
    """The observability §5.5.5 tool-call serialization,
    ``[{id, name, arguments}, ...]``.

    The single home for the encoding, shared by the input-message
    payload (the provider's ``input.messages`` serialization, where the
    model's tool calls appear inside replayed assistant history) and the
    output tool-call attribute (the OTel observer's gated
    ``openarmature.llm.output.tool_calls``). Lives here rather than in a
    provider or observer module so both sides import one definition and
    the encoding can't drift between them.
    """
    return [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]


__all__ = ["LLM_NAMESPACE", "LlmEventPayload", "serialize_tool_calls"]
