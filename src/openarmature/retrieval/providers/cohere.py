# Spec: realizes the retrieval-provider §5 RerankProvider protocol against a
# Cohere v2 POST /v2/rerank endpoint (retrieval-provider §8.4) -- the reference
# rerank provider. The §7 error categories are shared with llm-provider; the
# rerank-applicable subset (no unsupported_content_block, no
# structured_output_invalid) is mapped from the Cohere-shape HTTP error
# envelope below. Typed RerankEvent / RerankFailedEvent dispatch mirrors the
# embedding path via current_dispatch(). FOLLOW-UP: classify_http_error /
# base_url normalization are duplicated in spirit from the embedding + llm
# OpenAI providers; lifting a shared HTTP helper is a multi-provider follow-on.

"""Cohere-shape rerank provider.

``CohereRerankProvider`` issues ``POST {base_url}/v2/rerank`` and parses the
Cohere ``{id, model?, results: [{index, relevance_score, document?}], meta}``
envelope into a :class:`RerankResponse`. ``base_url`` is the host root (the
provider appends ``/v2/rerank``), overridable for a proxy / private gateway.
The ``transport`` parameter is the test seam (``httpx.MockTransport``).

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

from .._events import build_rerank_event, build_rerank_failed_event, document_echo, normalize_base_url
from ..provider import validate_rerank_input, validate_rerank_response
from ..response import RerankResponse, RerankRuntimeConfig, RerankUsage, ScoredDocument


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


class CohereRerankProvider:
    """Cohere ``/v2/rerank`` wire-shape rerank provider.

    Construct with a base URL (host root), the bound rerank model, and an
    optional API key + transport. ``rerank()`` posts to ``/v2/rerank``.

    ``ready()`` verifies the bound model with a minimal one-document
    ``/v2/rerank`` probe. The Cohere ``/v2/rerank`` wire exposes no
    model-catalog probe (unlike the OpenAI-compatible embedding surface), so
    there is a single universal probe.
    """

    def __init__(
        self,
        *,
        base_url: str,
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
        meta_raw = body.get("meta")
        if not isinstance(meta_raw, dict):
            return None
        billed_raw = cast("dict[str, Any]", meta_raw).get("billed_units")
        if not isinstance(billed_raw, dict):
            return None
        billed = cast("dict[str, Any]", billed_raw)
        search_units = self._nonneg_int(billed.get("search_units"))
        input_tokens = self._nonneg_int(billed.get("input_tokens"))
        if search_units is None and input_tokens is None:
            return None
        return RerankUsage(search_units=search_units, input_tokens=input_tokens)

    @staticmethod
    def _nonneg_int(value: Any) -> int | None:
        """Return a non-negative int value, or None (bool excluded)."""
        # bool is an int subclass, so exclude it explicitly; a malformed value
        # falls back to None (the rerank succeeded; usage is secondary).
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None


__all__ = ["CohereRerankProvider"]
