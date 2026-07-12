# Spec: realizes the retrieval-provider §5 RerankProvider protocol (POST
# /v2/rerank) and the §3 EmbeddingProvider protocol (POST /v2/embed) against the
# Cohere v2 API (retrieval-provider §8.4) -- the reference Cohere providers. The
# §7 error categories are shared with llm-provider; the retrieval-applicable
# subset (no unsupported_content_block, no structured_output_invalid) is mapped
# from the Cohere-shape HTTP error envelope below. Typed RerankEvent /
# EmbeddingEvent dispatch (with their failure variants) mirrors the sibling
# providers via current_dispatch(). The §8.4 embed half sends the mandatory
# input_type wire field (query -> search_query, document -> search_document,
# absent -> search_document, unrecognized -> pre-send provider_invalid_request),
# embedding_types ["float"] + truncate "NONE" explicitly (fail-loud), maps
# dimensions -> output_dimension, and chunk-and-stitches over Cohere's 96-input
# per-call cap. FOLLOW-UP: classify_http_error / base_url normalization are
# duplicated in spirit from the jina / openai retrieval providers; lifting a
# shared HTTP helper is a multi-provider follow-on.

"""Cohere-shape rerank + embedding providers.

``CohereRerankProvider`` issues ``POST {base_url}/v2/rerank`` and parses the
Cohere ``{id, model?, results: [{index, relevance_score, document?}], meta}``
envelope into a :class:`RerankResponse`. ``CohereEmbeddingProvider`` issues
``POST {base_url}/v2/embed`` (chunk-and-stitching across Cohere's 96-input
per-call cap when the input list is larger) and parses the Cohere
``{id, embeddings: {float: [[float, ...], ...]}, texts, meta}`` envelope into an
:class:`EmbeddingResponse`. ``base_url`` is the host root (the provider appends
``/v2/rerank`` / ``/v2/embed``), overridable for a proxy / private gateway.
``api_key`` is optional and sent as ``Authorization: Bearer <key>``. The
``transport`` parameter is the test seam (``httpx.MockTransport``).

A Cohere rerank instance and an embedding instance are distinct providers (one
model each) sharing the hosted endpoint.

The Cohere ``/v2/rerank`` wire has no ``return_documents`` parameter, so
``RerankRuntimeConfig.return_documents`` is a silent no-op on this wire (the
mapping sends no wire field for it). Cohere v2 does not echo document text, so
``ScoredDocument.document`` is null in practice; the parser still passes
through a ``document`` echo when a response carries one (a compatible gateway
or a fixture). ``max_tokens_per_doc`` rides the extras pass-through bag.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any, cast

import httpx

from openarmature.llm.errors import (
    LlmProviderError,
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderRateLimit,
    ProviderUnavailable,
)
from openarmature.observability.correlation import current_dispatch

from .._events import (
    build_embedding_event,
    build_embedding_failed_event,
    build_rerank_event,
    build_rerank_failed_event,
)
from .._wire import document_echo, normalize_base_url
from ..provider import (
    validate_embedding_input,
    validate_embedding_response,
    validate_rerank_input,
    validate_rerank_response,
)
from ..response import (
    EmbeddingResponse,
    EmbeddingRuntimeConfig,
    EmbeddingUsage,
    RerankResponse,
    RerankRuntimeConfig,
    RerankUsage,
    ScoredDocument,
)


def _classify_cohere_http_error(resp: httpx.Response) -> LlmProviderError:
    """Map a non-200 Cohere-shape rerank response to an error category.

    The rerank-applicable subset: 401/403 to auth,
    429 to rate_limit, 404 to invalid_model, 400/422 to invalid_request, and
    every other status to unavailable. Returns the exception (does not raise)
    so the caller raises with consistent traceback context.
    """
    status = resp.status_code
    try:
        body_raw = resp.json()
    except ValueError:
        body_raw = {}
    body: dict[str, Any] = cast("dict[str, Any]", body_raw) if isinstance(body_raw, dict) else {}
    # Cohere surfaces errors as a top-level ``message`` string; the OpenAI-shape
    # ``{error: {message}}`` envelope the fixtures also use is read as a
    # fallback so a compatible gateway's error body still yields a message.
    message_raw = body.get("message")
    if not isinstance(message_raw, str):
        error_block_raw = body.get("error")
        error_block: dict[str, Any] = (
            cast("dict[str, Any]", error_block_raw) if isinstance(error_block_raw, dict) else {}
        )
        message_candidate = error_block.get("message")
        message_raw = message_candidate if isinstance(message_candidate, str) else None
    message = message_raw

    if status in (401, 403):
        return ProviderAuthentication(message or f"HTTP {status}")
    if status == 429:
        return ProviderRateLimit(message or "HTTP 429")
    if status == 404:
        return ProviderInvalidModel(message or "model not found")
    if status in (400, 422):
        return ProviderInvalidRequest(message or f"HTTP {status}")
    return ProviderUnavailable(message or f"HTTP {status}")


# Absence-is-meaningful per observability §5.5.2: only caller-supplied keys
# appear in the event's request_params -- "the field was not supplied for this
# call", distinct from a supplied default. return_documents is the one declared
# rerank-config field (retrieval-provider §2); it is reported on the event when
# the caller explicitly set it, even though it is a silent no-op on the Cohere
# wire (§8.4). Unlike embedding, the rerank event's request_params do NOT feed
# the wire body -- return_documents has no Cohere wire key.
def _request_params_from_config(config: RerankRuntimeConfig | None) -> dict[str, Any]:
    """Extract the supplied rerank request parameters for the event."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if "return_documents" in config.model_fields_set:
        out["return_documents"] = config.return_documents
    return out


# §8.4 *Batch chunking (96-input cap)*: Cohere /v2/embed accepts at most 96
# inputs per request, so an over-cap embed call chunk-and-stitches over
# consecutive <=96 slices (the §8 general embed chunk-and-stitch rule; the
# fixed vendor cap, not a construction-configured chunk_size like TEI).
_COHERE_EMBED_MAX_INPUTS = 96

# §8.4 *input_type (mandatory wire field)*: the closed set the mapping
# recognizes, translated into Cohere's mandatory input_type wire value. Cohere's
# other input_type values (classification / clustering / image) ride the extras
# pass-through bag, not OA's input_type (widening input_type's normative value
# space is a §2 / 0077 protocol-level change, deferred until a consumer needs it).
_INPUT_TYPE_TO_COHERE: dict[str, str] = {
    "query": "search_query",
    "document": "search_document",
}


def _cohere_input_type(input_type: str | None) -> str:
    """Resolve the mandatory Cohere ``/v2/embed`` ``input_type`` wire value.

    ``"query"`` maps to ``"search_query"``, ``"document"`` to
    ``"search_document"``, and an absent ``input_type`` to ``"search_document"``
    (the bulk-indexing default -- the wire requires a value). Raises
    :class:`ProviderInvalidRequest` pre-send for an ``input_type`` outside the
    recognized set.
    """
    # Cohere v2 /v2/embed REQUIRES input_type on every request (unlike §8.1 /
    # §8.2, which omit the field when input_type is absent, and §8.3, which has
    # no field), so the mapping ALWAYS sends a value. An absent input_type MUST
    # map to search_document (§8.4 -- the wire needs a value; storing document
    # vectors is the dominant case); an unrecognized OA input_type is a pre-send
    # provider_invalid_request (§7), NOT silently coerced to the default.
    if input_type is None:
        return "search_document"
    resolved = _INPUT_TYPE_TO_COHERE.get(input_type)
    if resolved is None:
        raise ProviderInvalidRequest(
            f"Cohere input_type must be one of {sorted(_INPUT_TYPE_TO_COHERE)} (got {input_type!r}); "
            "other Cohere input_type values ride the extras pass-through bag"
        )
    return resolved


# Absence-is-meaningful per observability §5.5.2: only caller-supplied keys
# appear in the event's request_params -- "the field was not supplied for this
# call", distinct from a supplied default. For Cohere embed, input_type joins
# dimensions in this set. input_type in request_params is the caller's ORIGINAL
# value (query / document), NOT the resolved search_document default the wire
# carries -- the wire's mandatory default MUST NOT leak back onto the event, and
# the wire body is built separately.
def _embedding_request_params(config: EmbeddingRuntimeConfig | None) -> dict[str, Any]:
    """Extract the supplied embedding request parameters for the event."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if config.dimensions is not None:
        out["dimensions"] = config.dimensions
    if config.input_type is not None:
        out["input_type"] = config.input_type
    return out


def _nonneg_int(value: Any) -> int | None:
    """Return a non-negative int value, or None (bool excluded)."""
    # bool is an int subclass, so exclude it explicitly; a malformed value falls
    # back to None (the call succeeded; usage is secondary).
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _billed_units(body: dict[str, Any]) -> dict[str, Any] | None:
    """Return the Cohere ``meta.billed_units`` object, or None."""
    # Both /v2/rerank and /v2/embed report usage under meta.billed_units; the
    # descent + isinstance guards are shared, each caller reads its own figures.
    meta_raw = body.get("meta")
    if not isinstance(meta_raw, dict):
        return None
    billed_raw = cast("dict[str, Any]", meta_raw).get("billed_units")
    if not isinstance(billed_raw, dict):
        return None
    return cast("dict[str, Any]", billed_raw)


class CohereRerankProvider:
    """Cohere ``/v2/rerank`` wire-shape rerank provider.

    Construct with the bound rerank model and an optional API key +
    transport. ``base_url`` is the host root and defaults to the Cohere
    origin (``https://api.cohere.com``), overridable for a proxy / gateway.
    ``rerank()`` posts to ``/v2/rerank``.

    ``ready()`` verifies the bound model with a minimal one-document
    ``/v2/rerank`` probe. The Cohere ``/v2/rerank`` wire exposes no
    model-catalog probe (unlike the OpenAI-compatible embedding surface), so
    there is a single universal probe.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.cohere.com",
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "cohere",
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the host root; the provider appends /v2/rerank, so a
        # trailing /v2 would produce a doubled /v2/v2 path that 404s (the
        # sibling embedding provider guards the same footgun for /v1). Trailing
        # slashes are stripped.
        self.base_url = normalize_base_url(base_url, guard_prefix="/v2")
        self.model = model
        # ``genai_system`` surfaces as gen_ai.system on the rerank span.
        self._genai_system = genai_system
        self._populate_caller_metadata = populate_caller_metadata
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            transport=transport,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client (releases the connection pool)."""
        await self._client.aclose()

    async def ready(self) -> None:
        """Verify the bound rerank model is reachable and serving."""
        # A minimal one-document /v2/rerank probe surfaces
        # provider_invalid_model (404) / provider_unavailable (5xx) /
        # provider_authentication exactly as a real rerank() would.
        body = {"model": self.model, "query": "ready", "documents": ["ready"]}
        try:
            resp = await self._client.post("/v2/rerank", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_cohere_http_error(resp)

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        config: RerankRuntimeConfig | None = None,
    ) -> RerankResponse:
        """Score ``documents`` against ``query``, sorted by relevance."""
        dispatch = current_dispatch()
        call_id = str(uuid.uuid4())
        # Snapshot prompt context at dispatch time (the node task's context);
        # the delivery worker has a stale ContextVar view. Lazy import avoids
        # the prompts -> graph -> ... cycle.
        from openarmature.prompts.context import current_prompt_group, current_prompt_result

        active_prompt = current_prompt_result()
        active_prompt_group = current_prompt_group()
        documents_list = list(documents)
        request_params = _request_params_from_config(config)
        request_extras = dict(config.model_extra or {}) if config is not None else {}
        adapter_start = time.perf_counter()
        try:
            validate_rerank_input(query, documents_list, top_k)
            body = self._build_request_body(query, documents_list, top_k, request_extras)
            try:
                resp = await self._client.post("/v2/rerank", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"rerank request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_cohere_http_error(resp)
            response = self._parse_response(resp, documents_list, top_k)
        except LlmProviderError as exc:
            latency_ms_failed = (time.perf_counter() - adapter_start) * 1000.0
            if dispatch is not None:
                dispatch(
                    build_rerank_failed_event(
                        exc,
                        latency_ms_failed,
                        provider=self._genai_system,
                        model=self.model,
                        populate_caller_metadata=self._populate_caller_metadata,
                        call_id=call_id,
                        query=query,
                        documents=documents_list,
                        top_k=top_k,
                        request_params=request_params,
                        request_extras=request_extras,
                        active_prompt=active_prompt,
                        active_prompt_group=active_prompt_group,
                    ),
                )
            raise
        latency_ms = (time.perf_counter() - adapter_start) * 1000.0
        if dispatch is not None:
            dispatch(
                build_rerank_event(
                    response,
                    latency_ms,
                    provider=self._genai_system,
                    model=self.model,
                    populate_caller_metadata=self._populate_caller_metadata,
                    call_id=call_id,
                    query=query,
                    documents=documents_list,
                    top_k=top_k,
                    request_params=request_params,
                    request_extras=request_extras,
                    active_prompt=active_prompt,
                    active_prompt_group=active_prompt_group,
                ),
            )
        return response

    def _build_request_body(
        self,
        query: str,
        documents_list: list[str],
        top_k: int | None,
        request_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the /v2/rerank request body."""
        # Extras are merged FIRST so the bound model, the query, the documents,
        # and top_n always win: a caller's undeclared extra named "model" /
        # "query" / "documents" must not clobber the wire identity.
        # ``documents`` is the plain string-array form (Cohere v2 takes strings
        # only). ``top_n`` maps from top_k and is omitted when the caller passed
        # None. return_documents is NOT sent -- the Cohere wire has no such
        # field, so it is a silent no-op (§8.4); it stays a declared config
        # field (not an extra), so it never reaches request_extras.
        body: dict[str, Any] = {
            **request_extras,
            "model": self.model,
            "query": query,
            "documents": documents_list,
        }
        # top_n is the wire mapping of the top_k parameter, so it is a managed
        # key like model / query / documents: a caller extra named "top_n" must
        # not set it. Leaving it would send an effective top_n the response-count
        # validation (which keys off top_k) never checks. Drop any extras top_n;
        # top_k is the sole source.
        body.pop("top_n", None)
        if top_k is not None:
            body["top_n"] = top_k
        return body

    def _parse_response(
        self,
        resp: httpx.Response,
        documents_list: list[str],
        top_k: int | None,
    ) -> RerankResponse:
        """Parse the Cohere rerank envelope into a RerankResponse."""
        # Reads the §6 result shape ({index, relevance_score, document?}),
        # sorts by relevance_score descending (Cohere returns ranked results,
        # but §6 mandates the sort regardless), then validates the §6
        # invariants (valid index into the input documents, no duplicate index,
        # len(results) <= top_k when supplied) via validate_rerank_response.
        try:
            body_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("rerank response is not valid JSON") from exc
        if not isinstance(body_raw, dict):
            raise ProviderInvalidResponse("rerank response is not a JSON object")
        body = cast("dict[str, Any]", body_raw)
        results_raw = body.get("results")
        if not isinstance(results_raw, list):
            raise ProviderInvalidResponse("rerank response missing 'results' array")
        results = cast("list[Any]", results_raw)
        scored: list[ScoredDocument] = []
        for raw_entry in results:
            if not isinstance(raw_entry, dict):
                raise ProviderInvalidResponse("rerank response entry is not a JSON object")
            entry = cast("dict[str, Any]", raw_entry)
            index = entry.get("index")
            # bool is an int subclass, so exclude it explicitly.
            if not isinstance(index, int) or isinstance(index, bool):
                raise ProviderInvalidResponse("rerank response entry missing integer 'index'")
            score = entry.get("relevance_score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ProviderInvalidResponse(
                    "rerank response entry has a missing or non-numeric 'relevance_score'"
                )
            # Read the document echo per the shared §6 (0097) rule; never
            # auto-fill from the input documents. Cohere v2 does not echo, so
            # this is null in practice, but a compatible gateway may return one.
            document = document_echo(entry.get("document"))
            scored.append(ScoredDocument(index=index, relevance_score=float(score), document=document))
        # §6: sort by relevance_score descending before validating / returning.
        scored.sort(key=lambda s: s.relevance_score, reverse=True)
        validate_rerank_response(scored, len(documents_list), top_k)
        usage = self._parse_usage(body)
        response_id = body.get("id")
        model = body.get("model")
        return RerankResponse(
            results=scored,
            model=model if isinstance(model, str) else self.model,
            usage=usage,
            response_id=response_id if isinstance(response_id, str) else None,
            raw=body,
        )

    def _parse_usage(self, body: dict[str, Any]) -> RerankUsage | None:
        """Extract a RerankUsage from meta.billed_units, or None.

        A record is present only when the provider surfaces at least one usage
        figure; an all-null record is never fabricated.
        Cohere reports ``search_units``; ``input_tokens`` is read too so a
        Cohere-compatible backend that surfaces it is honored.
        """
        billed = _billed_units(body)
        if billed is None:
            return None
        search_units = _nonneg_int(billed.get("search_units"))
        input_tokens = _nonneg_int(billed.get("input_tokens"))
        if search_units is None and input_tokens is None:
            return None
        return RerankUsage(search_units=search_units, input_tokens=input_tokens)


class CohereEmbeddingProvider:
    """Cohere ``/v2/embed`` wire-shape embedding provider.

    Construct with the bound embedding model and an optional API key +
    transport. ``base_url`` is the host root and defaults to the Cohere
    origin (``https://api.cohere.com``), overridable for a proxy / gateway.
    ``embed()`` posts to ``/v2/embed``, chunk-and-stitching across Cohere's
    96-input per-call cap when the input list is larger.

    Cohere ``/v2/embed`` requires ``input_type`` on every request, so the
    mapping always sends a value: ``"query"`` becomes ``"search_query"``,
    ``"document"`` becomes ``"search_document"``, and an absent ``input_type``
    becomes ``"search_document"`` (the bulk-indexing default). An unrecognized
    ``input_type`` is rejected before the request is sent. ``embedding_types``
    always requests ``"float"`` (the mapping reads ``embeddings.float``) and
    ``truncate: "NONE"`` is sent explicitly (an over-length input errors rather
    than being silently truncated); ``dimensions`` maps to Cohere's
    ``output_dimension`` when set. Other precisions (``int8`` / ``base64`` /
    ...) and ``input_type`` values (``classification`` / ...) ride the extras
    pass-through bag.

    ``ready()`` verifies the bound model with a minimal one-input ``/v2/embed``
    probe. The Cohere ``/v2/embed`` wire exposes no model-catalog probe, so
    there is a single universal probe.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.cohere.com",
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "cohere",
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the host root; the provider appends /v2/embed, so a
        # trailing /v2 would produce a doubled /v2/v2 path that 404s (the
        # sibling rerank / jina / openai providers guard the same footgun).
        # Trailing slashes are stripped.
        self.base_url = normalize_base_url(base_url, guard_prefix="/v2")
        self.model = model
        # ``genai_system`` surfaces as gen_ai.system on the embedding span.
        self._genai_system = genai_system
        self._populate_caller_metadata = populate_caller_metadata
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            transport=transport,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client (releases the connection pool)."""
        await self._client.aclose()

    async def ready(self) -> None:
        """Verify the bound embedding model is reachable and serving."""
        # A minimal one-input /v2/embed probe surfaces provider_invalid_model
        # (404) / provider_rate_limit (429) / provider_unavailable (5xx) /
        # provider_authentication exactly as a real embed() would. input_type is
        # mandatory on the wire, so the probe sends the search_document default.
        body = {
            "model": self.model,
            "input_type": "search_document",
            "texts": ["ready"],
            "embedding_types": ["float"],
            "truncate": "NONE",
        }
        try:
            resp = await self._client.post("/v2/embed", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_cohere_http_error(resp)

    async def embed(
        self,
        input: Sequence[str],
        *,
        config: EmbeddingRuntimeConfig | None = None,
    ) -> EmbeddingResponse:
        """Embed ``input`` into one vector per string, in input order."""
        dispatch = current_dispatch()
        call_id = str(uuid.uuid4())
        # Snapshot prompt context at dispatch time (the node task's context);
        # the delivery worker has a stale ContextVar view. Lazy import avoids
        # the prompts -> graph -> ... cycle.
        from openarmature.prompts.context import current_prompt_group, current_prompt_result

        active_prompt = current_prompt_result()
        active_prompt_group = current_prompt_group()
        input_strings = list(input)
        request_params = _embedding_request_params(config)
        request_extras = dict(config.model_extra or {}) if config is not None else {}
        input_type = config.input_type if config is not None else None
        dimensions = config.dimensions if config is not None else None
        adapter_start = time.perf_counter()
        try:
            validate_embedding_input(input_strings)
            # Resolve the mandatory wire input_type before any POST: an
            # unrecognized value raises provider_invalid_request here (pre-send
            # validation), so no request is issued. The resolved value is
            # identical across every chunk (§8.4 batch chunking).
            resolved_input_type = _cohere_input_type(input_type)
            response = await self._embed_chunked(
                input_strings, resolved_input_type, dimensions, request_extras
            )
        except LlmProviderError as exc:
            latency_ms_failed = (time.perf_counter() - adapter_start) * 1000.0
            if dispatch is not None:
                dispatch(
                    build_embedding_failed_event(
                        exc,
                        latency_ms_failed,
                        provider=self._genai_system,
                        model=self.model,
                        populate_caller_metadata=self._populate_caller_metadata,
                        call_id=call_id,
                        input_strings=input_strings,
                        request_params=request_params,
                        request_extras=request_extras,
                        active_prompt=active_prompt,
                        active_prompt_group=active_prompt_group,
                    ),
                )
            raise
        latency_ms = (time.perf_counter() - adapter_start) * 1000.0
        if dispatch is not None:
            dispatch(
                build_embedding_event(
                    response,
                    latency_ms,
                    provider=self._genai_system,
                    model=self.model,
                    populate_caller_metadata=self._populate_caller_metadata,
                    call_id=call_id,
                    input_strings=input_strings,
                    request_params=request_params,
                    request_extras=request_extras,
                    active_prompt=active_prompt,
                    active_prompt_group=active_prompt_group,
                ),
            )
        return response

    async def _embed_chunked(
        self,
        input_strings: list[str],
        input_type: str,
        dimensions: int | None,
        request_extras: dict[str, Any],
    ) -> EmbeddingResponse:
        """Issue one /v2/embed per <=96 chunk and stitch the vectors."""
        # THE chunk-and-stitch (§8.4 *Batch chunking (96-input cap)* / the §8
        # general embed rule). When len(input) <= 96 this issues a single
        # request; otherwise it splits the inputs into consecutive <=96 slices,
        # issues one /v2/embed per slice with IDENTICAL per-call params (only
        # texts differs), concatenates the per-chunk embeddings.float IN INPUT
        # ORDER, sums meta.billed_units.input_tokens across chunks, and takes
        # response_id from the FIRST chunk. This is valid because each input's
        # embedding is independent of the others in its batch.
        stitched_vectors: list[list[float]] = []
        chunk_bodies: list[Any] = []
        input_tokens_total: int | None = None
        response_id: str | None = None
        for offset in range(0, len(input_strings), _COHERE_EMBED_MAX_INPUTS):
            chunk = input_strings[offset : offset + _COHERE_EMBED_MAX_INPUTS]
            body = self._build_request_body(chunk, input_type, dimensions, request_extras)
            try:
                resp = await self._client.post("/v2/embed", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"embedding request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_cohere_http_error(resp)
            chunk_vectors, chunk_tokens, chunk_id, chunk_body = self._parse_chunk(resp)
            # Per-chunk count MUST match the chunk's inputs before stitching: the
            # stitch is positional (no index re-basing), so a chunk returning the
            # wrong vector count would silently misalign vectors that a
            # compensating chunk lets the stitched total pass (§4 input-order).
            if len(chunk_vectors) != len(chunk):
                raise ProviderInvalidResponse(
                    f"embedding response returned {len(chunk_vectors)} vectors for {len(chunk)} inputs"
                )
            stitched_vectors.extend(chunk_vectors)
            chunk_bodies.append(chunk_body)
            if offset == 0:
                response_id = chunk_id
            # Sum input_tokens across chunks; a chunk that omits usage does not
            # contribute, and usage stays null only when NO chunk reports it
            # (§4 / §8 batch-chunking step 4 -- never fabricate).
            if chunk_tokens is not None:
                input_tokens_total = (input_tokens_total or 0) + chunk_tokens
        # §4 cross-impl invariants (one vector per input, uniform
        # dimensionality) are enforced against the STITCHED result.
        dimensions_out = validate_embedding_response(stitched_vectors, len(input_strings))
        usage = EmbeddingUsage(input_tokens=input_tokens_total) if input_tokens_total is not None else None
        # raw is the verbatim deserialized response (§4 / §8 per proposal 0096,
        # dict | list): a single /v2/embed call carries that response's object;
        # a chunked call carries the LIST of per-chunk objects in request order,
        # discriminated by whether the input exceeded the 96-input cap.
        raw: dict[str, Any] | list[Any] = chunk_bodies[0] if len(chunk_bodies) == 1 else chunk_bodies
        return EmbeddingResponse(
            vectors=stitched_vectors,
            model=self.model,
            usage=usage,
            response_id=response_id,
            dimensions=dimensions_out,
            raw=raw,
        )

    def _build_request_body(
        self,
        chunk: list[str],
        input_type: str,
        dimensions: int | None,
        request_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one /v2/embed request body for a chunk of inputs."""
        # Extras merge FIRST so the managed keys (model, input_type, texts,
        # truncate, output_dimension) always win: a caller's undeclared extra
        # named "texts" must not clobber the wire identity, and one named
        # "truncate" cannot defeat the fail-loud guarantee. input_type is the
        # resolved mandatory wire value (search_query / search_document).
        # truncate "NONE" is sent explicitly so an over-length input errors
        # rather than being silently truncated (fail-loud, §8.4).
        #
        # embedding_types is the ONE managed key that MERGES rather than
        # clobbers: the mapping reads embeddings.float, so "float" MUST be
        # present, but a caller's other precisions (int8 / uint8 / binary /
        # base64) ride the extras bag and are preserved -- "float" is merged in
        # only when a caller's embedding_types omits it (§8.4). A malformed /
        # empty embedding_types extra falls back to ["float"].
        caller_types = request_extras.get("embedding_types")
        if isinstance(caller_types, list) and caller_types:
            embedding_types = list(cast("list[Any]", caller_types))
            if "float" not in embedding_types:
                embedding_types.append("float")
        else:
            embedding_types = ["float"]
        body: dict[str, Any] = {
            **request_extras,
            "model": self.model,
            "input_type": input_type,
            "texts": list(chunk),
            "embedding_types": embedding_types,
            "truncate": "NONE",
        }
        # output_dimension is the wire mapping of the dimensions parameter, so it
        # is a managed key like model / texts: a caller extra named
        # "output_dimension" must not set it. dimensions is the sole source; drop
        # any extras copy, then set it only when supplied (omitted otherwise so
        # Cohere's model default applies).
        body.pop("output_dimension", None)
        if dimensions is not None:
            body["output_dimension"] = dimensions
        return body

    def _parse_chunk(
        self,
        resp: httpx.Response,
    ) -> tuple[list[list[float]], int | None, str | None, dict[str, Any]]:
        """Parse one /v2/embed chunk into (vectors, input_tokens, id, raw)."""
        # Cohere /v2/embed returns an object envelope {id, embeddings: {float:
        # [[float, ...], ...]}, texts, meta: {billed_units: {input_tokens}}}.
        # embeddings.float is a list of vector lists IN INPUT ORDER (positional
        # -- Cohere returns them in request order, NOT index-keyed like the
        # OpenAI / Jina data[] shape). Rejects non-numeric vector values (JSON
        # strings, bools) rather than coercing them.
        try:
            body_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("embedding response is not valid JSON") from exc
        if not isinstance(body_raw, dict):
            raise ProviderInvalidResponse("embedding response is not a JSON object")
        body = cast("dict[str, Any]", body_raw)
        embeddings_raw = body.get("embeddings")
        if not isinstance(embeddings_raw, dict):
            raise ProviderInvalidResponse("embedding response missing 'embeddings' object")
        float_raw = cast("dict[str, Any]", embeddings_raw).get("float")
        if not isinstance(float_raw, list):
            raise ProviderInvalidResponse("embedding response missing 'embeddings.float' array")
        rows = cast("list[Any]", float_raw)
        vectors: list[list[float]] = []
        for row in rows:
            if not isinstance(row, list) or not row:
                raise ProviderInvalidResponse("embedding response has a missing or empty vector")
            values = cast("list[Any]", row)
            # bool is an int subclass, and float("0.1") would silently accept a
            # string, so the strict isinstance check is what makes "non-numeric
            # is malformed" hold.
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
                raise ProviderInvalidResponse("embedding response has a non-numeric vector value")
            vectors.append([float(x) for x in values])
        input_tokens = self._parse_input_tokens(body)
        response_id = body.get("id")
        return vectors, input_tokens, response_id if isinstance(response_id, str) else None, body

    def _parse_input_tokens(self, body: dict[str, Any]) -> int | None:
        """Extract meta.billed_units.input_tokens, or None.

        A value is read only when the provider surfaces it; a token count is
        never fabricated.
        """
        billed = _billed_units(body)
        if billed is None:
            return None
        return _nonneg_int(billed.get("input_tokens"))


__all__ = ["CohereEmbeddingProvider", "CohereRerankProvider"]
