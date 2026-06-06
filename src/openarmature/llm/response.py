# Spec: realizes llm-provider §6 (Response shape + RuntimeConfig).
# ``raw`` follows charter §3.1 principle 8 (Transparency over
# abstraction) — carries everything the provider returned, including
# fields the spec doesn't normalize (logprobs, content-filter detail,
# vendor extensions).

"""Response and RuntimeConfig.

The ``Response`` is what ``Provider.complete()`` returns: the
assistant message, a finish reason, optional usage, and the verbatim
parsed provider response. ``raw`` carries everything the provider
returned; including fields the abstraction doesn't normalize
(logprobs, content-filter detail, vendor-specific extensions) so
users who need them can reach through the abstraction directly.

``RuntimeConfig`` is the optional per-call sampling-parameter record.
Implementations MAY accept additional provider-specific fields; the
seven declared here are the mandated minimum.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .messages import AssistantMessage

# ``parsed`` may carry either a raw dict (when the caller passed a
# JSON-Schema dict as response_schema) or a Pydantic model instance
# (when the caller passed a BaseModel subclass). The latter is a
# per-language ergonomic — the runtime shape mirrors what the caller
# requested. Absent (None) on calls without response_schema and on
# tool-call responses regardless of whether response_schema was set.
ParsedValue = dict[str, Any] | BaseModel | None

# The five spec §6 finish-reason values. Modeled as a Literal union so
# pydantic rejects unknown values at parse time — provider responses
# carrying a non-standard value surface as ``provider_invalid_response``
# rather than silently passing through.
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


# Cache-stat fields (cached_tokens / cache_creation_tokens) are
# optional and default to None. The absent-vs-reported-zero distinction
# is observable: None means the provider did not report the field; 0
# means the provider reported the field with value zero (a "reported
# miss"). Each per-provider wire-format mapping documents which fields
# it sources.
class Usage(BaseModel):
    """Token-accounting record.

    Each field is a non-negative integer or ``None``. If the provider
    does not report token counts, ``prompt_tokens`` / ``completion_tokens``
    / ``total_tokens`` MUST be ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    total_tokens: int | None = Field(ge=0)
    # The count of input tokens that hit a prefix cache, sourced from
    # the provider's response. Absent (None) when the provider does
    # not report cache statistics; set to 0 when the provider reports
    # zero cache-hit tokens. Each wire-format mapping documents the
    # provider response field this value is sourced from.
    cached_tokens: int | None = Field(default=None, ge=0)
    # The count of input tokens written to the cache during the call.
    # Populated primarily by providers with explicit cache-control
    # surfaces that report a discrete cache-creation count alongside
    # cache reads. Absent (None) for providers that only report
    # implicit cache reads (the §8.1 OpenAI-compat mapping leaves this
    # field absent).
    cache_creation_tokens: int | None = Field(default=None, ge=0)


class Response(BaseModel):
    """The result of a ``Provider.complete()`` call.

    Attributes:
        message: The assistant message returned by the model.
            Always ``role: "assistant"``. May carry ``tool_calls``.
        finish_reason: One of ``"stop"``, ``"length"``, ``"tool_calls"``,
            ``"content_filter"``, ``"error"``.
        usage: The token record (all ``None`` if the provider didn't
            report usage).
        raw: The parsed provider response, populated on every successful
            return. Carries everything the provider returned; the
            normalized fields above are derived from it.
        parsed: The parsed-and-validated structured value when the call
            supplied a ``response_schema`` and the model returned
            structured content. ``None`` otherwise. The runtime type
            depends on the schema form the caller passed: ``dict`` for
            a JSON-Schema dict input, a ``BaseModel`` instance for a
            Pydantic class input.
    """

    # ``parsed`` is absent (None) on calls that didn't supply a
    # response_schema, and on responses whose finish_reason is
    # "tool_calls" — the tool-call path and the structured-content
    # path are mutually exclusive at the response level.
    #
    # message.content carries the model's content string verbatim.
    # parsed is the post-receive deserialization of that content
    # against the schema; the provider's content string is NOT
    # re-serialized from parsed.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage
    raw: dict[str, Any]
    parsed: ParsedValue = None
    # The provider's response id (e.g., OpenAI's ``chatcmpl-…``).
    # Surface as a typed field rather than asking callers to reach into
    # ``raw["id"]``; mirrors the gen_ai.response.id semconv attribute
    # the observability mapping (spec §5.5.3) emits onto the LLM span.
    # ``None`` when the provider didn't return one.
    response_id: str | None = None
    # The model identifier the provider returned (the ``model`` field
    # on the response body). May be more specific than the bound
    # request model — e.g., bound ``gpt-4o``, response carries
    # ``gpt-4o-2024-08-06``. Mirrors gen_ai.response.model per §5.5.3.
    # ``None`` when the provider didn't return one.
    response_model: str | None = None


# Spec §6 declared-field surface: seven optional fields. Undeclared
# fields supplied by callers MUST be forwarded to the wire body
# untouched (extras pass-through); declared fields with value ``None``
# MUST be omitted from the wire body (null-skip). Both rules are
# enforced by the §8 wire-format mapping, not by RuntimeConfig itself.
class RuntimeConfig(BaseModel):
    """Per-call sampling parameters and budget hints."""

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    # Per spec §6: setting ``seed`` does NOT guarantee determinism; see
    # §9. Best-effort only, useful for providers that support it.
    seed: int | None = None
    # Promoted from extras to a declared field in proposal 0032 (spec
    # v0.24.0). Cross-vendor: OpenAI / Mistral / Cohere accept this
    # name directly; Anthropic and Gemini map to vendor-specific
    # equivalents at the wire layer (per §8.2 / §8.3 when those land).
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # Declared field name matches the cross-vendor OpenTelemetry GenAI
    # semconv (``gen_ai.request.stop_sequences``) and Anthropic /
    # Gemini wire-key conventions. The §8.1 OpenAI-compatible wire
    # mapping translates ``stop_sequences`` to OpenAI's shorter body
    # key ``stop`` on emission. OpenAI is the outlier on the shorter
    # name; the declared layer matches the cross-vendor norm.
    stop_sequences: list[str] | None = None

    # Pure Python ergonomic, not a spec contract. The wire-layer
    # null-skip rule already drops ``None``-valued declared fields, so
    # this helper exists solely to let callers splat a dict whose
    # entries may be ``None`` without filtering at the call site.
    @classmethod
    def from_partial(cls, **kwargs: Any) -> RuntimeConfig:
        """Construct a config, dropping kwargs whose value is ``None``.

        >>> RuntimeConfig.from_partial(temperature=0.7, top_p=None).top_p is None
        True
        """
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


__all__ = [
    "FinishReason",
    "ParsedValue",
    "Response",
    "RuntimeConfig",
    "Usage",
]
