# Spec: realizes retrieval-provider §4 (EmbeddingResponse + EmbeddingUsage
# shapes and the response invariants) and §2 (the EmbeddingRuntimeConfig
# record). ``raw`` follows charter §3.1 principle 8 (Transparency over
# abstraction) -- it carries the parsed provider response verbatim
# alongside the normalized fields, so callers who need un-normalized data
# can reach through the abstraction. The ``input_type`` config field
# arrives with proposal 0077; ``dimensions`` is the one declared field
# at this pin.

"""EmbeddingResponse, EmbeddingUsage, and EmbeddingRuntimeConfig.

The ``EmbeddingResponse`` is what ``EmbeddingProvider.embed()`` returns:
one vector per input string in input order, the model identifier, usage,
the optional provider response id, the output dimensionality, and the
verbatim parsed provider response. ``raw`` carries everything the provider
returned.

``EmbeddingRuntimeConfig`` is the optional per-call request-parameter
record. Implementations may accept additional provider-specific fields
via the extras pass-through; ``dimensions`` is the declared field.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingUsage(BaseModel):
    """Token-accounting record for an embedding call.

    Carries ``input_tokens`` only; an embedding call has no output
    tokens (vectors are not tokens).
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)


class EmbeddingResponse(BaseModel):
    """The result of an ``EmbeddingProvider.embed()`` call.

    Attributes:
        vectors: One vector (a list of floats) per input string, in the
            order the inputs were supplied. ``len(vectors)`` equals the
            input length.
        model: The model identifier the provider returned; may be more
            specific than the bound identifier.
        usage: The token record.
        response_id: The provider-returned response id when present;
            ``None`` otherwise.
        dimensions: The output vector dimensionality; equals the length
            of each inner vector.
        raw: The parsed provider response, populated on every successful
            return. Carries everything the provider returned.
    """

    model_config = ConfigDict(extra="forbid")

    vectors: list[list[float]]
    model: str
    usage: EmbeddingUsage
    response_id: str | None = None
    dimensions: int
    raw: dict[str, Any]


# Spec §2 declared-field surface: an optional ``dimensions`` plus the
# extras pass-through bag (``extra="allow"``). Undeclared fields supplied
# by callers are forwarded to the wire body untouched by the §8 wire-format
# mapping; declared fields with value ``None`` are omitted on the wire.
class EmbeddingRuntimeConfig(BaseModel):
    """Per-call embedding request parameters."""

    model_config = ConfigDict(extra="allow")

    dimensions: int | None = None

    # Pure ergonomic, not a contract: lets callers splat a dict whose
    # entries may be ``None`` without filtering at the call site, mirroring
    # ``llm.response.RuntimeConfig.from_partial``.
    @classmethod
    def from_partial(cls, **kwargs: Any) -> EmbeddingRuntimeConfig:
        """Construct a config, dropping kwargs whose value is ``None``."""
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


__all__ = [
    "EmbeddingResponse",
    "EmbeddingRuntimeConfig",
    "EmbeddingUsage",
]
