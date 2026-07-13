"""Shared wire-format helpers for the retrieval reference providers.

Pure, stateless primitives reused across the vendor wire mappings:
request-body shaping (the client-side ``input_type`` prefix), endpoint
normalization (``base_url``), and response parsing (the rerank document echo).
Types live in ``response.py``, the observability event builders in
``_events.py``, and the provider protocols in ``provider.py``; only the
mapping helpers belong here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from openarmature.llm.errors import ProviderInvalidResponse

from .provider import validate_embedding_response
from .response import EmbeddingResponse, EmbeddingUsage


def normalize_base_url(base_url: str, *, guard_prefix: str) -> str:
    """Strip a trailing slash from ``base_url`` and reject a doubled version prefix.

    Returns the trailing-slash-stripped host root. Raises :class:`ValueError`
    when the stripped value already ends in ``guard_prefix`` (e.g. ``"/v1"`` /
    ``"/v2"``).
    """
    # base_url is the host root; the provider appends the versioned routes
    # itself (e.g. /v1/embeddings, /v2/rerank), so a base_url that already ends
    # in the version prefix would produce a doubled /v1/v1 (or /v2/v2) path that
    # 404s. Guard the footgun at construction; trailing slashes are stripped.
    normalized = base_url.rstrip("/")
    if normalized.endswith(guard_prefix):
        raise ValueError(
            f"base_url should be the host root; the provider appends the "
            f"{guard_prefix} routes itself, so a trailing {guard_prefix} would "
            f"produce a doubled {guard_prefix}{guard_prefix} path. Got {base_url!r}."
        )
    return normalized


# retrieval-provider §8.1 client-side prefix rule, reused by §8.3: input_type
# "query" selects query_prefix, "document" selects document_prefix; any other
# value (including None / absent) selects no prefix -- the symmetric default.
def client_side_prefix(
    input_type: str | None,
    *,
    query_prefix: str | None,
    document_prefix: str | None,
) -> str | None:
    """Return the client-side prefix for ``input_type``, or ``None``.

    ``"query"`` maps to ``query_prefix``, ``"document"`` to ``document_prefix``;
    any other value (including ``None``) maps to ``None`` (no prefix).
    """
    if input_type == "query":
        return query_prefix
    if input_type == "document":
        return document_prefix
    return None


# Apply the §8.1 client-side prefix to an input list for input_type: prepend the
# resolved prefix to every string, or return the list unchanged when input_type
# resolves to no prefix. The single source of the "prefix the inputs" step
# shared by the OpenAI (§8.3) and TEI (§8.1) embed mappings -- keeps the copy /
# no-copy semantics from drifting between providers.
def apply_client_side_prefix(
    input_strings: list[str],
    input_type: str | None,
    *,
    query_prefix: str | None,
    document_prefix: str | None,
) -> list[str]:
    """Return ``input_strings`` prefixed per ``input_type``, or unchanged.

    Resolves the prefix via :func:`client_side_prefix`; when it is ``None`` the
    list is returned as-is (no defensive copy -- the result is only serialized
    onto the wire), otherwise a new list with the prefix prepended to each
    string.
    """
    prefix = client_side_prefix(input_type, query_prefix=query_prefix, document_prefix=document_prefix)
    if prefix is None:
        return input_strings
    return [prefix + s for s in input_strings]


# retrieval-provider §8 *Batch chunking* (the general embed rule), shared by
# every EmbeddingProvider whose wire enforces a per-call input cap (TEI /embed's
# max-client-batch-size, Cohere /v2/embed's 96-input cap, ...). When len(input)
# exceeds cap the mapping MUST split the inputs into consecutive <=cap slices
# (preserving order), issue one request per slice with EVERY field other than the
# chunked input list identical across slices, concatenate the per-chunk vectors
# IN INPUT ORDER (so §4's one-vector-per-input + input-order invariants hold
# across the whole call), and combine the per-chunk usage per §4's nullable
# usage contract -- sum input_tokens when the provider reports usage, else
# usage = null. response_id is the FIRST chunk's id (a single-request call uses
# that request's id). raw is that one response for a single-request call, or the
# LIST of per-chunk responses in request order for a chunked call (proposal
# 0096). embed_chunk owns the per-mapping wire shaping / POST / parse and returns
# (vectors, input_tokens, response_id, raw_body) for its slice -- so the loop /
# stitch / validation is mapping-agnostic. When len(input) <= cap this issues a
# single request (the single-iteration path). Valid because each input's
# embedding is independent of the others in its batch.
async def chunk_and_stitch_embed(
    input_strings: list[str],
    *,
    model: str,
    cap: int,
    embed_chunk: Callable[[list[str]], Awaitable[tuple[list[list[float]], int | None, str | None, Any]]],
) -> EmbeddingResponse:
    """Issue one embed request per ``<= cap`` chunk and stitch the vectors.

    ``embed_chunk`` sends one chunk's request and returns
    ``(vectors, input_tokens, response_id, raw_body)`` for that chunk; the
    per-mapping wire shaping, POST, and parse live in the closure. Returns the
    stitched :class:`EmbeddingResponse`.
    """
    stitched_vectors: list[list[float]] = []
    chunk_bodies: list[Any] = []
    input_tokens_total: int | None = None
    response_id: str | None = None
    for offset in range(0, len(input_strings), cap):
        chunk = input_strings[offset : offset + cap]
        chunk_vectors, chunk_tokens, chunk_id, chunk_body = await embed_chunk(chunk)
        # Per-chunk count MUST match the chunk's inputs before stitching: the
        # stitch is positional (no index re-basing), so a chunk returning the
        # wrong vector count would silently misalign vectors that a compensating
        # chunk lets the stitched total pass (§4 input-order).
        if len(chunk_vectors) != len(chunk):
            raise ProviderInvalidResponse(
                f"embedding response returned {len(chunk_vectors)} vectors for {len(chunk)} inputs"
            )
        stitched_vectors.extend(chunk_vectors)
        chunk_bodies.append(chunk_body)
        if offset == 0:
            response_id = chunk_id
        # Sum input_tokens across chunks; a chunk that omits usage does not
        # contribute, and usage stays null only when NO chunk reports it (§4 /
        # §8 batch-chunking step 4 -- never fabricate).
        if chunk_tokens is not None:
            input_tokens_total = (input_tokens_total or 0) + chunk_tokens
    # §4 cross-impl invariants (one vector per input, uniform dimensionality)
    # are enforced against the STITCHED result.
    dimensions = validate_embedding_response(stitched_vectors, len(input_strings))
    usage = EmbeddingUsage(input_tokens=input_tokens_total) if input_tokens_total is not None else None
    # raw is the verbatim deserialized response (§4 / §8 per proposal 0096, dict
    # | list): a single call carries that response's body; a chunked call carries
    # the LIST of per-chunk bodies in request order.
    raw: dict[str, Any] | list[Any] = chunk_bodies[0] if len(chunk_bodies) == 1 else chunk_bodies
    return EmbeddingResponse(
        vectors=stitched_vectors,
        model=model,
        usage=usage,
        response_id=response_id,
        dimensions=dimensions,
        raw=raw,
    )


# §6 (0097): the rerank document-echo shape rule, shared by every RerankProvider
# (Cohere / TEI / Jina). ScoredDocument.document carries the provider's string
# echo verbatim when present, null otherwise -- never fabricated from the input
# documents. Vendors echo it in several shapes: a bare string, a TextDoc object
# {"text": "..."} (Jina's return_documents form), a text-less object (an
# ImageDoc), or absent/null. Unwrap the string from a string or a TextDoc; a
# text-less object or absent/null yields null (an empty string is present --
# surfaced as "", not folded to null). A NON-object scalar (number / array /
# bool -- outside the documented anyOf[string, object, null]) is wire corruption
# and raises provider_invalid_response (§7), NOT folded to null. The verbatim
# echo is preserved on RerankResponse.raw regardless.
def document_echo(value: Any) -> str | None:
    """Extract the string document echo from a rerank result's documented shapes."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = cast("dict[str, Any]", value).get("text")
        return text if isinstance(text, str) else None
    raise ProviderInvalidResponse(
        f"rerank document echo must be a string, object, or null (got {type(value).__name__})"
    )


__all__ = [
    "apply_client_side_prefix",
    "chunk_and_stitch_embed",
    "client_side_prefix",
    "document_echo",
    "normalize_base_url",
]
