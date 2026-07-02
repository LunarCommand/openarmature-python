# Spec: realizes retrieval-provider §4 (EmbeddingResponse + EmbeddingUsage
# shapes and the response invariants), §6 (RerankResponse + RerankUsage +
# ScoredDocument shapes and the rerank response invariants), and §2 (the
# EmbeddingRuntimeConfig + RerankRuntimeConfig records). ``raw`` follows
# charter §3.1 principle 8 (Transparency over abstraction) -- it carries
# the parsed provider response verbatim alongside the normalized fields, so
# callers who need un-normalized data can reach through the abstraction. The
# ``input_type`` config field arrives with proposal 0077; ``dimensions`` is
# the one declared embedding-config field at this pin. ``return_documents``
# is the one declared rerank-config field (proposal 0060).

"""EmbeddingResponse / RerankResponse types and their runtime configs.

The ``EmbeddingResponse`` is what ``EmbeddingProvider.embed()`` returns:
one vector per input string in input order, the model identifier, usage,
the optional provider response id, the output dimensionality, and the
verbatim parsed provider response. ``raw`` carries everything the provider
returned.

The ``RerankResponse`` is what ``RerankProvider.rerank()`` returns: the
scored documents sorted by relevance descending, the model identifier, the
optional usage record and response id, and the verbatim parsed provider
response.

``EmbeddingRuntimeConfig`` and ``RerankRuntimeConfig`` are the optional
per-call request-parameter records. Implementations may accept additional
provider-specific fields via the extras pass-through; ``dimensions`` is the
declared embedding field and ``return_documents`` the declared rerank field.
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


class RerankUsage(BaseModel):
    """Token-accounting record for a rerank call.

    Both fields default to ``None`` and are individually nullable: a
    provider may surface one figure and not the other (Cohere reports
    ``search_units`` but no token count; Voyage AI reports ``input_tokens``).
    """

    # Spec §6: both fields are individually nullable; a RerankUsage record is
    # present only when the provider surfaces at least one figure, and
    # implementations MUST NOT fabricate an all-null record.
    model_config = ConfigDict(extra="forbid")

    search_units: int | None = None
    input_tokens: int | None = None


class ScoredDocument(BaseModel):
    """A single scored result entry in a ``RerankResponse``.

    Attributes:
        index: The 0-based position of this document in the original input
            ``documents`` list. Load-bearing for caller-side lookup:
            ``documents[result.index]`` maps a result back to its input.
        relevance_score: The provider-assigned relevance score; higher =
            more relevant. Provider-specific scale (not normalized here).
        document: The echoed document text when the provider returns it;
            ``None`` otherwise. Never fabricated from the input documents.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    """The result of a ``RerankProvider.rerank()`` call.

    Attributes:
        results: The scored documents sorted by ``relevance_score``
            descending (most relevant first). Each entry's ``index`` keys
            back to the original input ``documents`` list.
        model: The model identifier the provider returned; may be more
            specific than the bound identifier.
        usage: The usage record, or ``None`` when the provider reports no
            usage object.
        response_id: The provider-returned response id when present;
            ``None`` otherwise.
        raw: The parsed provider response, populated on every successful
            return. Carries everything the provider returned.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[ScoredDocument]
    model: str
    usage: RerankUsage | None = None
    response_id: str | None = None
    raw: dict[str, Any]


# Spec §2 rerank runtime config: one declared field ``return_documents``
# (boolean, default False) plus the extras pass-through bag (``extra="allow"``).
# Undeclared fields supplied by callers are forwarded to the wire body by the
# §8 wire-format mapping, except for the provider-reserved keys a mapping
# manages itself (e.g. the Cohere mapping owns model / query / documents /
# top_n, so a caller extra cannot clobber them).
class RerankRuntimeConfig(BaseModel):
    """Per-call rerank request parameters."""

    model_config = ConfigDict(extra="allow")

    return_documents: bool = False

    # Pure ergonomic, not a contract: lets callers splat a dict whose
    # entries may be ``None`` without filtering at the call site, mirroring
    # ``EmbeddingRuntimeConfig.from_partial``.
    @classmethod
    def from_partial(cls, **kwargs: Any) -> RerankRuntimeConfig:
        """Construct a config, dropping kwargs whose value is ``None``."""
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


__all__ = [
    "EmbeddingResponse",
    "EmbeddingRuntimeConfig",
    "EmbeddingUsage",
    "RerankResponse",
    "RerankRuntimeConfig",
    "RerankUsage",
    "ScoredDocument",
]
