"""Response and RuntimeConfig (spec §6).

The ``Response`` is what ``Provider.complete()`` returns: the
assistant message, a finish reason, optional usage, and the verbatim
parsed provider response. Per charter §3.1 principle 8
"Transparency over abstraction", ``raw`` carries everything the
provider returned — including fields the spec doesn't normalize
(logprobs, content-filter detail, vendor-specific extensions) so
users who need them can reach through the abstraction directly.

``RuntimeConfig`` is the optional per-call sampling-parameter record.
Implementations MAY accept additional provider-specific fields; the
four declared here (temperature, max_tokens, top_p, seed) are the
spec-mandated minimum.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .messages import AssistantMessage

# The five spec §6 finish-reason values. Modeled as a Literal union so
# pydantic rejects unknown values at parse time — provider responses
# carrying a non-standard value surface as ``provider_invalid_response``
# rather than silently passing through.
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


class Usage(BaseModel):
    """Token-accounting record per spec §6.

    Each field is a non-negative integer or ``None``. If the provider
    does not report usage, all three MUST be ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class Response(BaseModel):
    """The result of a ``Provider.complete()`` call.

    Per spec §6:

    - ``message`` is the assistant message returned by the model.
      Always ``role: "assistant"``. May carry ``tool_calls``.
    - ``finish_reason`` is one of the five spec values.
    - ``usage`` is the token record (all ``None`` if the provider
      didn't report usage).
    - ``raw`` is the parsed provider response, populated on every
      successful return. Carries everything the provider returned —
      the normalized fields above are derived from it.
    """

    model_config = ConfigDict(extra="forbid")

    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage
    raw: dict[str, Any]


class RuntimeConfig(BaseModel):
    """Per-call sampling parameters and budget hints (spec §6).

    All four fields are optional. Implementations MAY accept
    additional provider-specific fields; this is the spec-mandated
    minimum.
    """

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    # Per spec §6: setting ``seed`` does NOT guarantee determinism; see
    # §9. Best-effort only, useful for providers that support it.
    seed: int | None = None


__all__ = [
    "FinishReason",
    "Response",
    "RuntimeConfig",
    "Usage",
]
