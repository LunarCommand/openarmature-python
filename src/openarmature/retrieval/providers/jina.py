# Spec: realizes the retrieval-provider §8.2 Jina hosted wire mapping --
# a §8 mapping in the family established by 0077 (TEI). Jina is one hosted
# API (base_url default https://api.jina.ai, origin-only): a JinaEmbeddingProvider
# (POST /v1/embeddings) and a JinaRerankProvider (POST /v1/rerank) are distinct
# instances (one model each) sharing the endpoint, each binding an API key sent
# as Authorization: Bearer <key> (§8.2 *Construction*). The §7 error categories
# are shared with llm-provider; 429 maps to provider_rate_limit (NOT unavailable),
# and Jina's over-length fail-loud (422 -> provider_invalid_request) is the surface
# where the truncation / truncate: false flag bites. Jina reports usage
# (total_tokens), unlike TEI's null, so both responses carry a usage record when
# the body surfaces one (never fabricated). input_type is realized as Jina's
# native `task` per the §8.2 closed set (query -> retrieval.query,
# document -> retrieval.passage); an unrecognized input_type is a pre-send
# provider_invalid_request. Jina enforces no per-call embed cap (server-side
# batching), so there is no client-side chunk-and-stitch (unlike TEI /rerank).
# Typed EmbeddingEvent / RerankEvent dispatch mirrors the sibling providers via
# current_dispatch(). FOLLOW-UP: the ContextVar identity snapshot + event
# construction + error classification are duplicated in spirit from the tei /
# cohere / openai retrieval providers; lifting shared helpers is a multi-provider
# follow-on.

"""Jina AI hosted embedding + rerank providers.

``JinaEmbeddingProvider`` issues ``POST {base_url}/v1/embeddings`` and parses
Jina's ``{model, usage, data: [{index, embedding}]}`` envelope into an
:class:`EmbeddingResponse`. ``JinaRerankProvider`` issues
``POST {base_url}/v1/rerank`` and parses Jina's
``{model, usage, results: [{index, relevance_score, document?}]}`` envelope
into a :class:`RerankResponse`. ``base_url`` is the Jina endpoint origin
(default ``https://api.jina.ai``; the provider appends ``/v1/embeddings`` /
``/v1/rerank``), overridable for a proxy / private gateway. ``api_key`` is
required and sent as ``Authorization: Bearer <key>``. The ``transport``
parameter is the test seam (``httpx.MockTransport``).

A Jina embedding instance and a rerank instance are distinct providers (one
model each) sharing the hosted endpoint.
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
from .._wire import document_echo, nonneg_int, normalize_base_url
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

# §8.2 input_type realization: the closed set the mapping recognizes, translated
# into Jina's native `task` field so Jina applies the model-appropriate
# query/passage representation server-side. An input_type outside this set is a
# pre-send provider_invalid_request (§7); Jina's other task values
# (text-matching / classification / clustering) ride the extras pass-through bag,
# not input_type (widening input_type's normative value space is a protocol-level
# change, deferred until a consumer needs it).
_INPUT_TYPE_TO_TASK: dict[str, str] = {
    "query": "retrieval.query",
    "document": "retrieval.passage",
}


def _classify_jina_http_error(resp: httpx.Response) -> LlmProviderError:
    """Map a non-200 Jina-shape response to an error category.

    Returns the exception (does not raise) so the caller raises with
    consistent traceback context.
    """
    # Jina surfaces errors as a top-level ``detail`` string (its ErrorResponse
    # and 422 HTTPValidationError shapes both carry it); read it for the message.
    status = resp.status_code
    try:
        body_raw = resp.json()
    except ValueError:
        body_raw = {}
    body: dict[str, Any] = cast("dict[str, Any]", body_raw) if isinstance(body_raw, dict) else {}
    detail = body.get("detail")
    message = detail if isinstance(detail, str) else None

    if status in (401, 403):
        return ProviderAuthentication(message or f"HTTP {status}")
    # §8.2 *Errors*: 429 (rate limit) -> provider_rate_limit, NOT
    # provider_unavailable (the misclassification fixture 022 pins).
    if status == 429:
        return ProviderRateLimit(message or "HTTP 429")
    if status == 404:
        return ProviderInvalidModel(message or "model not found")
    # §8.2 *Errors*: over-length / malformed request (422) ->
    # provider_invalid_request (the fail-loud surface, fixture 021). Jina lists
    # only 422 here (no 400 / 413, unlike the TEI / OpenAI sibling mappings), so
    # a 400 falls through to provider_unavailable per the §8.2 enumeration.
    if status == 422:
        return ProviderInvalidRequest(message or "HTTP 422")
    return ProviderUnavailable(message or f"HTTP {status}")


def _task_for_input_type(input_type: str | None) -> str | None:
    """Translate ``input_type`` into Jina's ``task`` per the closed set.

    Returns ``None`` when ``input_type`` is absent (task omitted -- the
    symmetric default). Raises :class:`ProviderInvalidRequest` pre-send for an
    ``input_type`` outside the recognized set.
    """
    if input_type is None:
        return None
    task = _INPUT_TYPE_TO_TASK.get(input_type)
    if task is None:
        raise ProviderInvalidRequest(
            f"Jina input_type must be one of {sorted(_INPUT_TYPE_TO_TASK)} (got {input_type!r}); "
            "other Jina task values ride the extras pass-through bag"
        )
    return task


# Absence-is-meaningful per observability §5.5.2: only caller-supplied keys
# appear in the event's request_params -- "the field was not supplied for this
# call", distinct from a supplied default. For Jina, input_type joins dimensions
# in this set. input_type does NOT feed the wire body verbatim -- it is realized
# as Jina's `task`, so the wire body is built separately.
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


# return_documents is the one declared rerank-config field (retrieval-provider
# §2); it is reported on the event when the caller explicitly set it. It maps to
# Jina's return_documents on the wire (§8.2), built separately from the event
# params.
def _rerank_request_params(config: RerankRuntimeConfig | None) -> dict[str, Any]:
    """Extract the supplied rerank request parameters for the event."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if "return_documents" in config.model_fields_set:
        out["return_documents"] = config.return_documents
    return out


class JinaEmbeddingProvider:
    """Jina ``/v1/embeddings`` wire-mapping embedding provider.

    Construct with the bound embedding model and the required API key; the
    ``base_url`` defaults to the Jina endpoint (``https://api.jina.ai``, origin
    only -- override for a proxy / private gateway). ``embed()`` posts to
    ``/v1/embeddings``.

    ``ready()`` verifies the bound model with a minimal one-input
    ``/v1/embeddings`` probe.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.jina.ai",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "jina",
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the endpoint origin; the provider appends the /v1 routes,
        # so a trailing /v1 would produce a doubled /v1/v1 path that 404s (the
        # sibling OpenAI / Cohere providers guard the same footgun). Trailing
        # slashes are stripped.
        self.base_url = normalize_base_url(base_url, guard_prefix="/v1")
        self.model = model
        # ``genai_system`` surfaces as gen_ai.system on the embedding span.
        self._genai_system = genai_system
        self._populate_caller_metadata = populate_caller_metadata
        # Jina is a hosted vendor; the API key is required and always sent as
        # Authorization: Bearer <key> (§8.2 *Construction*).
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
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
        # A minimal one-input /v1/embeddings probe surfaces
        # provider_invalid_model (404) / provider_rate_limit (429) /
        # provider_unavailable (5xx) / provider_authentication exactly as a real
        # embed() would.
        body = {"model": self.model, "input": ["ready"]}
        try:
            resp = await self._client.post("/v1/embeddings", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_jina_http_error(resp)

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
            # An unrecognized input_type raises provider_invalid_request here,
            # before the POST (pre-send validation); no request is issued.
            body = self._build_request_body(input_strings, input_type, dimensions, request_extras)
            try:
                resp = await self._client.post("/v1/embeddings", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"embedding request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_jina_http_error(resp)
            response = self._parse_response(resp, input_strings)
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

    def _build_request_body(
        self,
        input_strings: list[str],
        input_type: str | None,
        dimensions: int | None,
        request_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the /v1/embeddings request body.

        Realizes ``input_type`` as Jina's native ``task`` per the closed set;
        absent ``input_type`` omits ``task`` (the symmetric default).
        ``truncate: false`` is always sent (fail-loud).
        """
        # Extras merge FIRST so the managed keys (model, input, task, dimensions,
        # truncate) always win: a caller's undeclared extra named "input" must
        # not clobber the wire identity, and one named "truncate" cannot defeat
        # the fail-loud guarantee. truncate: false is sent explicitly so an
        # over-length input errors rather than being silently truncated (§8.2).
        task = _task_for_input_type(input_type)
        body: dict[str, Any] = {
            **request_extras,
            "model": self.model,
            "input": list(input_strings),
            "truncate": False,
        }
        if task is not None:
            body["task"] = task
        if dimensions is not None:
            body["dimensions"] = dimensions
        return body

    def _parse_response(
        self,
        resp: httpx.Response,
        input_strings: list[str],
    ) -> EmbeddingResponse:
        """Parse Jina's embeddings envelope into an EmbeddingResponse."""
        # Jina returns an object envelope {model, usage, data: [{index,
        # embedding}]}. Orders ``data`` by the per-entry ``index`` so output
        # position matches input position (like the OpenAI data[] shape, NOT
        # TEI's positional bare array), then validates the §4 invariants (count +
        # consistent dimensionality) via validate_embedding_response. The index
        # set MUST be a 0..n-1 permutation, and every entry MUST carry a
        # non-empty numeric embedding.
        try:
            body_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("embedding response is not valid JSON") from exc
        if not isinstance(body_raw, dict):
            raise ProviderInvalidResponse("embedding response is not a JSON object")
        body = cast("dict[str, Any]", body_raw)
        data_raw = body.get("data")
        if not isinstance(data_raw, list):
            raise ProviderInvalidResponse("embedding response missing 'data' array")
        data = cast("list[Any]", data_raw)
        entries: list[dict[str, Any]] = []
        indices: list[int] = []
        for raw_entry in data:
            if not isinstance(raw_entry, dict):
                raise ProviderInvalidResponse("embedding response entry is not a JSON object")
            entry = cast("dict[str, Any]", raw_entry)
            index = entry.get("index")
            # bool is an int subclass, so exclude it explicitly.
            if not isinstance(index, int) or isinstance(index, bool):
                raise ProviderInvalidResponse("embedding response entry missing integer 'index'")
            entries.append(entry)
            indices.append(index)
        if sorted(indices) != list(range(len(entries))):
            raise ProviderInvalidResponse("embedding response 'index' values are not a 0..n-1 permutation")
        ordered = sorted(entries, key=lambda e: cast("int", e["index"]))
        vectors: list[list[float]] = []
        for entry in ordered:
            embedding = entry.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ProviderInvalidResponse("embedding response entry has a missing or empty 'embedding'")
            values = cast("list[Any]", embedding)
            # Reject non-numeric vector values (JSON strings, bools) rather than
            # coercing them: bool is an int subclass, and float("0.1") would
            # silently accept a string, so the strict isinstance check is what
            # makes "non-numeric is malformed" hold.
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
                raise ProviderInvalidResponse("embedding response has a non-numeric vector value")
            vectors.append([float(x) for x in values])
        dimensions = validate_embedding_response(vectors, len(input_strings))
        # Jina reports usage (total_tokens) -> EmbeddingUsage.input_tokens; a
        # record is built when the body surfaces it, else usage = null (§4 --
        # MUST NOT fabricate; a Jina-compatible backend that omits usage yields
        # None). raw is the verbatim response dict (object envelope, so a dict).
        usage = self._parse_usage(body)
        response_id = body.get("id")
        model = body.get("model")
        return EmbeddingResponse(
            vectors=vectors,
            model=model if isinstance(model, str) else self.model,
            usage=usage,
            response_id=response_id if isinstance(response_id, str) else None,
            dimensions=dimensions,
            raw=body,
        )

    def _parse_usage(self, body: dict[str, Any]) -> EmbeddingUsage | None:
        """Extract an EmbeddingUsage from usage.total_tokens, or None.

        A record is present only when the provider surfaces a usable
        ``total_tokens``; a value is never fabricated.
        """
        usage_block = body.get("usage")
        if not isinstance(usage_block, dict):
            return None
        total_tokens = nonneg_int(
            cast("dict[str, Any]", usage_block).get("total_tokens"), field="total_tokens"
        )
        if total_tokens is None:
            return None
        return EmbeddingUsage(input_tokens=total_tokens)


class JinaRerankProvider:
    """Jina ``/v1/rerank`` wire-mapping rerank provider.

    Construct with the bound rerank model and the required API key; the
    ``base_url`` defaults to the Jina endpoint (``https://api.jina.ai``, origin
    only -- override for a proxy / private gateway). ``rerank()`` posts to
    ``/v1/rerank`` (a single request -- Jina batches server-side, so there is no
    client-side chunk-and-stitch).

    ``ready()`` verifies the bound model with a minimal one-document
    ``/v1/rerank`` probe.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.jina.ai",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "jina",
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the endpoint origin; the provider appends the /v1 routes,
        # so a trailing /v1 would produce a doubled /v1/v1 path that 404s.
        # Trailing slashes are stripped.
        self.base_url = normalize_base_url(base_url, guard_prefix="/v1")
        self.model = model
        # ``genai_system`` surfaces as gen_ai.system on the rerank span.
        self._genai_system = genai_system
        self._populate_caller_metadata = populate_caller_metadata
        # Jina is a hosted vendor; the API key is required and always sent as
        # Authorization: Bearer <key> (§8.2 *Construction*).
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
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
        # A minimal one-document /v1/rerank probe surfaces
        # provider_invalid_model (404) / provider_rate_limit (429) /
        # provider_unavailable (5xx) / provider_authentication exactly as a real
        # rerank() would.
        body = {"model": self.model, "query": "ready", "documents": ["ready"]}
        try:
            resp = await self._client.post("/v1/rerank", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_jina_http_error(resp)

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
        request_params = _rerank_request_params(config)
        request_extras = dict(config.model_extra or {}) if config is not None else {}
        return_documents = config.return_documents if config is not None else False
        adapter_start = time.perf_counter()
        try:
            validate_rerank_input(query, documents_list, top_k)
            body = self._build_request_body(query, documents_list, top_k, return_documents, request_extras)
            try:
                resp = await self._client.post("/v1/rerank", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"rerank request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_jina_http_error(resp)
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
        return_documents: bool,
        request_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the /v1/rerank request body."""
        # Extras merge FIRST so the managed keys (model, query, documents,
        # top_n, return_documents, truncation) always win. documents maps
        # directly onto the string array (§8.2; no per-document wrapping).
        # return_documents is sent EXPLICITLY: Jina's wire default is true but
        # OA's is False (§2), so the mapping sends the resolved OA value rather
        # than relying on Jina's default. truncation: false is sent explicitly
        # (fail-loud -- an over-length pair errors rather than silently
        # truncating). top_n maps from top_k, omitted when None.
        body: dict[str, Any] = {
            **request_extras,
            "model": self.model,
            "query": query,
            "documents": documents_list,
            "return_documents": return_documents,
            "truncation": False,
        }
        # top_n is the wire mapping of top_k, so it is a managed key: drop any
        # extras top_n (top_k is the sole source), then set it only when supplied.
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
        """Parse Jina's rerank envelope into a RerankResponse."""
        # Jina returns {model, usage, results: [{index, relevance_score,
        # document?}]} -- Cohere-shaped (relevance_score, NOT TEI's score).
        # Reads the document echo only when present (never auto-filled), sorts
        # by relevance_score descending (Jina returns ranked results, but §6
        # mandates the sort regardless), then validates the §6 invariants (valid
        # index into the input documents, no duplicate index, len(results) <=
        # top_k when supplied) via validate_rerank_response.
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
            # auto-fill from the input documents. Jina echoes text as a TextDoc
            # object ({"text": ...}), which document_echo unwraps to the string.
            document = document_echo(entry.get("document"))
            scored.append(ScoredDocument(index=index, relevance_score=float(score), document=document))
        # §6: sort by relevance_score descending before validating / returning.
        scored.sort(key=lambda s: s.relevance_score, reverse=True)
        validate_rerank_response(scored, len(documents_list), top_k)
        # Jina meters rerank by tokens -> RerankUsage.input_tokens; search_units
        # is always null (Jina reports no search-unit figure). raw is the
        # verbatim response dict.
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
        """Extract a RerankUsage from usage.total_tokens, or None.

        A record is present only when the provider surfaces a usable
        ``total_tokens``; ``search_units`` stays null (Jina meters by tokens),
        and an all-null record is never fabricated.
        """
        usage_block = body.get("usage")
        if not isinstance(usage_block, dict):
            return None
        total_tokens = nonneg_int(
            cast("dict[str, Any]", usage_block).get("total_tokens"), field="total_tokens"
        )
        if total_tokens is None:
            return None
        return RerankUsage(input_tokens=total_tokens)


__all__ = ["JinaEmbeddingProvider", "JinaRerankProvider"]
