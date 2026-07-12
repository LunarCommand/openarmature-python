# Spec: realizes the retrieval-provider §8.1 TEI (Text Embeddings Inference)
# wire mapping -- the first concrete §8 mapping (proposal 0077). TEI serves one
# model per instance and hosts embedding + cross-encoder rerankers as separate
# deployments, so the embedding surface (POST /embed) and the rerank surface
# (POST /rerank) are two distinct provider classes against two distinct
# base_urls (§8.1 *Construction*). The §7 error categories are shared with
# llm-provider; TEI's over-length fail-loud (413 / 422 -> provider_invalid_
# request) is the surface where §8.1 diverges from the OpenAI-shape mapping.
# Typed EmbeddingEvent / RerankEvent dispatch mirrors the sibling providers via
# current_dispatch(). TEI returns no usage object on either endpoint, so both
# responses carry usage = null (§4 / §6; the mapping MUST NOT fabricate a usage
# record). FOLLOW-UP: the ContextVar identity snapshot + event construction +
# error classification are duplicated in spirit from the openai / cohere
# retrieval providers; lifting shared helpers is a multi-provider follow-on.

"""HuggingFace Text Embeddings Inference (TEI) providers.

``TeiEmbeddingProvider`` issues ``POST {base_url}/embed`` and parses TEI's
bare vector-array response into an :class:`EmbeddingResponse`.
``TeiRerankProvider`` issues ``POST {base_url}/rerank`` (chunk-and-stitching
across TEI's ``max-client-batch-size`` when the candidate pool is larger) and
parses TEI's ``[{index, score, text?}]`` response into a
:class:`RerankResponse`. ``base_url`` is the TEI instance root (the provider
appends ``/embed`` / ``/rerank``). The ``transport`` parameter is the test
seam (``httpx.MockTransport``).

TEI hosts one model per deployment, and embedding models and cross-encoder
rerankers are different model families, so an embedding instance and a rerank
instance are distinct providers against distinct ``base_url``s.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
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
from ..provider import (
    validate_embedding_input,
    validate_embedding_response,
    validate_rerank_input,
    validate_rerank_response,
)
from ..response import (
    EmbeddingResponse,
    EmbeddingRuntimeConfig,
    RerankResponse,
    RerankRuntimeConfig,
    ScoredDocument,
)


def _classify_tei_http_error(resp: httpx.Response) -> LlmProviderError:
    """Map a non-200 TEI-shape response to an error category.

    Returns the exception (does not raise) so the caller raises with
    consistent traceback context.
    """
    # TEI surfaces errors as a top-level ``error`` string plus an
    # ``error_type`` string ({"error": str, "error_type": str}); the OpenAI-
    # shape {"error": {"message"}} envelope is read as a fallback so a
    # compatible gateway's error body still yields a message.
    status = resp.status_code
    try:
        body_raw = resp.json()
    except ValueError:
        body_raw = {}
    body: dict[str, Any] = cast("dict[str, Any]", body_raw) if isinstance(body_raw, dict) else {}
    message_raw = body.get("error")
    if isinstance(message_raw, dict):
        nested = cast("dict[str, Any]", message_raw).get("message")
        message = nested if isinstance(nested, str) else None
    else:
        message = message_raw if isinstance(message_raw, str) else None

    if status in (401, 403):
        return ProviderAuthentication(message or f"HTTP {status}")
    if status == 429:
        return ProviderRateLimit(message or "HTTP 429")
    if status == 404:
        return ProviderInvalidModel(message or "model not found")
    # §8.1 *Errors*: over-length / malformed request (413 / 422) ->
    # provider_invalid_request. 413 is the TEI-specific addition over the
    # sibling providers (payload-too-large on the fail-loud truncate path);
    # 400 / 422 join it as the malformed-request statuses.
    if status in (400, 413, 422):
        return ProviderInvalidRequest(message or f"HTTP {status}")
    return ProviderUnavailable(message or f"HTTP {status}")


# Absence-is-meaningful per observability §5.5.2: only caller-supplied keys
# appear in the event's request_params -- "the field was not supplied for this
# call", distinct from a supplied default. For TEI, input_type joins dimensions
# in this set (proposal 0077). Unlike the OpenAI mapping (where the event params
# coincide with the wire keys), input_type does NOT feed the wire body verbatim
# -- it is realized as TEI's prompt_name (or a client-side prefix), so the wire
# body is built separately.
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
# TEI's return_text on the wire (§8.1), built separately from the event params.
def _rerank_request_params(config: RerankRuntimeConfig | None) -> dict[str, Any]:
    """Extract the supplied rerank request parameters for the event."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if "return_documents" in config.model_fields_set:
        out["return_documents"] = config.return_documents
    return out


class TeiEmbeddingProvider:
    """TEI ``/embed`` wire-mapping embedding provider.

    Construct with a base URL (the TEI embedding instance root), the bound
    embedding model, and optionally an ``input_type_prompt_map`` binding
    ``input_type`` to TEI's native ``prompt_name`` (server-side prompts) and/or
    client-side ``query_prefix`` / ``document_prefix`` strings (the fallback for
    models without configured prompts). ``embed()`` posts to ``/embed``.

    ``ready()`` verifies the bound model with a minimal one-input ``/embed``
    probe (TEI serves no model catalog).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "tei",
        chunk_size: int = 32,
        input_type_prompt_map: Mapping[str, str] | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the TEI instance root; the provider appends /embed.
        # Trailing slashes are stripped. TEI has no version prefix (unlike the
        # OpenAI /v1 + Cohere /v2 mappings), so there is no doubled-prefix
        # footgun to guard.
        self.base_url = base_url.rstrip("/")
        self.model = model
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive (got {chunk_size})")
        # chunk_size is TEI's max-client-batch-size (default 32); it bounds the
        # /embed batch per the §8 batch-chunking rule. Client-side embed chunk-
        # and-stitch is proposal 0092, out of scope here -- chunk_size is bound
        # for construction parity so an operator can already pin it.
        self._chunk_size = chunk_size
        # input_type -> TEI prompt_name map (server-side prompts) with an
        # optional client-side prefix fallback (§8.1 input_type realization).
        self._input_type_prompt_map = dict(input_type_prompt_map) if input_type_prompt_map else None
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
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
        # A minimal one-input /embed probe surfaces provider_invalid_model
        # (404) / provider_unavailable (5xx) / provider_authentication exactly
        # as a real embed() would. TEI serves no /models catalog.
        body = {"inputs": ["ready"]}
        try:
            resp = await self._client.post("/embed", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_tei_http_error(resp)

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
            body = self._build_request_body(input_strings, input_type, dimensions, request_extras)
            try:
                resp = await self._client.post("/embed", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"embedding request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_tei_http_error(resp)
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
        """Build the /embed request body.

        Realizes ``input_type`` as TEI's native ``prompt_name`` (server-side
        prompts) from the construction ``input_type -> prompt_name`` map, or a
        client-side ``query_prefix`` / ``document_prefix`` prepend when only
        prefixes are configured. Absent ``input_type`` sends neither -- the
        symmetric default, byte-identical to the symmetric path.
        """
        # input_type realization per §8.1. ``truncate`` is NOT added -- the
        # /embed mapping relies on TEI's false default, keeping the body minimal
        # (§8.1). Extras merge FIRST so the managed keys (inputs, prompt_name,
        # dimensions) always win: a caller's undeclared extra named "inputs"
        # must not clobber the wire identity.
        inputs = list(input_strings)
        prompt_name: str | None = None
        if input_type is not None:
            if self._input_type_prompt_map is not None:
                # Server-side prompt: look up the mapped prompt_name. An
                # input_type absent from the map yields no prompt_name (the map
                # is operator-supplied, not a closed set).
                prompt_name = self._input_type_prompt_map.get(input_type)
            else:
                # Client-side prefix fallback: prepend for models without
                # configured prompts. Only "query" / "document" have prefixes.
                prefix = self._prefix_for(input_type)
                if prefix is not None:
                    inputs = [prefix + s for s in inputs]
        body: dict[str, Any] = {**request_extras, "inputs": inputs}
        if prompt_name is not None:
            body["prompt_name"] = prompt_name
        if dimensions is not None:
            body["dimensions"] = dimensions
        return body

    def _prefix_for(self, input_type: str) -> str | None:
        """Return the client-side prefix for ``input_type``, or ``None``."""
        if input_type == "query":
            return self._query_prefix
        if input_type == "document":
            return self._document_prefix
        return None

    def _parse_response(
        self,
        resp: httpx.Response,
        input_strings: list[str],
    ) -> EmbeddingResponse:
        """Parse TEI's bare vector-array response into an EmbeddingResponse."""
        # TEI /embed returns a bare JSON array of vector arrays in input order
        # ([[float, ...], ...]) -- no envelope, no id, no usage object. The
        # vectors are already in input order (position == input index), so no
        # index-keyed reordering is needed (unlike the OpenAI data[] shape).
        # Validates the §4 invariants (count + consistent dimensionality) via
        # validate_embedding_response, and rejects non-numeric vector values.
        try:
            body_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("embedding response is not valid JSON") from exc
        if not isinstance(body_raw, list):
            raise ProviderInvalidResponse("TEI /embed response is not a JSON array")
        rows = cast("list[Any]", body_raw)
        vectors: list[list[float]] = []
        for row in rows:
            if not isinstance(row, list) or not row:
                raise ProviderInvalidResponse("TEI /embed response has a missing or empty vector")
            values = cast("list[Any]", row)
            # Reject non-numeric vector values (JSON strings, bools) rather than
            # coercing them: bool is an int subclass, and float("0.1") would
            # silently accept a string, so the strict isinstance check is what
            # makes "non-numeric is malformed" hold.
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
                raise ProviderInvalidResponse("TEI /embed response has a non-numeric vector value")
            vectors.append([float(x) for x in values])
        dimensions = validate_embedding_response(vectors, len(input_strings))
        # TEI returns no usage object, so usage = null (§4 -- MUST NOT
        # fabricate). raw is the verbatim deserialized response: TEI's bare
        # vector array carried as a list, not wrapped (§4 / §8.1 per proposal
        # 0096 -- raw is dict | list, the top-level shape the provider returned).
        return EmbeddingResponse(
            vectors=vectors,
            model=self.model,
            usage=None,
            response_id=None,
            dimensions=dimensions,
            raw=rows,
        )


class TeiRerankProvider:
    """TEI ``/rerank`` wire-mapping rerank provider.

    Construct with a base URL (the TEI reranker instance root), the bound rerank
    model, and ``chunk_size`` (TEI's ``max-client-batch-size``, default 32).
    ``rerank()`` posts to ``/rerank``, chunk-and-stitching across ``chunk_size``
    when the candidate pool is larger.

    ``ready()`` verifies the bound model with a minimal one-document ``/rerank``
    probe (TEI serves no model catalog).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "tei",
        chunk_size: int = 32,
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the TEI instance root; the provider appends /rerank.
        # Trailing slashes are stripped. TEI has no version prefix, so no
        # doubled-prefix footgun to guard (cf. TeiEmbeddingProvider).
        self.base_url = base_url.rstrip("/")
        self.model = model
        # A non-positive chunk_size would make range(0, n, chunk_size) raise
        # (0) or loop zero times (negative -- silently dropping every document),
        # so guard at construction: a misconfigured cap fails loudly.
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive (got {chunk_size})")
        # chunk_size is TEI's max-client-batch-size (default 32); it bounds the
        # per-request document count for the mandatory rerank chunk-and-stitch
        # (§8.1 *Mandatory rerank batch chunking*).
        self._chunk_size = chunk_size
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
        # A minimal one-document /rerank probe surfaces provider_invalid_model
        # (404) / provider_unavailable (5xx) / provider_authentication exactly
        # as a real rerank() would. TEI serves no /models catalog.
        body = {"query": "ready", "texts": ["ready"], "truncate": False, "return_text": False}
        try:
            resp = await self._client.post("/rerank", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_tei_http_error(resp)

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        config: RerankRuntimeConfig | None = None,
    ) -> RerankResponse:
        """Score ``documents`` against ``query``, sorted by relevance.

        Chunk-and-stitches across ``chunk_size``: one ``/rerank`` request per
        consecutive ``<= chunk_size`` slice, absolute-position re-basing, a
        global re-sort by score descending, then ``top_k``.
        """
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
            response = await self._rerank_chunked(
                query, documents_list, top_k, return_documents, request_extras
            )
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

    async def _rerank_chunked(
        self,
        query: str,
        documents_list: list[str],
        top_k: int | None,
        return_documents: bool,
        request_extras: dict[str, Any],
    ) -> RerankResponse:
        """Issue one /rerank per chunk and stitch the results."""
        # THE chunk-and-stitch. When len(documents) <= chunk_size this issues a
        # single request; otherwise it splits the documents into consecutive
        # <= chunk_size slices, issues one /rerank per slice (same query), and
        # re-bases each chunk's response index to its absolute position before
        # concatenating. The GLOBAL sort + top_k are applied AFTER all chunks
        # are stitched -- never per-chunk. This is valid because a cross-encoder
        # scores each (query, document) pair independently of its batch.
        stitched: list[ScoredDocument] = []
        chunk_bodies: list[Any] = []
        for offset in range(0, len(documents_list), self._chunk_size):
            chunk = documents_list[offset : offset + self._chunk_size]
            body = self._build_request_body(query, chunk, return_documents, request_extras)
            try:
                resp = await self._client.post("/rerank", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"rerank request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_tei_http_error(resp)
            chunk_scored, chunk_body = self._parse_chunk(resp, len(chunk), offset)
            stitched.extend(chunk_scored)
            chunk_bodies.append(chunk_body)
        # §6: global sort by score descending, then honor top_k. Validate the
        # in-range + no-duplicate invariants across the FULL stitched set BEFORE
        # slicing (top_k=None) so a duplicate that lands in the dropped tail is
        # still caught; the slice then guarantees len(results) <= top_k.
        stitched.sort(key=lambda s: s.relevance_score, reverse=True)
        validate_rerank_response(stitched, len(documents_list), None)
        if top_k is not None:
            stitched = stitched[:top_k]
        # TEI returns no id / no usage on /rerank, so response_id + usage are
        # null (§6 -- MUST NOT fabricate a RerankUsage). raw is the verbatim
        # deserialized response (§6 / §8 per proposal 0096, dict | list): a
        # single /rerank call carries that response's bare array; a chunked call
        # carries the LIST of per-chunk responses in request order (nothing
        # lost), discriminated by whether the pool exceeded chunk_size.
        raw: dict[str, Any] | list[Any] = chunk_bodies[0] if len(chunk_bodies) == 1 else chunk_bodies
        return RerankResponse(
            results=stitched,
            model=self.model,
            usage=None,
            response_id=None,
            raw=raw,
        )

    def _build_request_body(
        self,
        query: str,
        chunk: list[str],
        return_documents: bool,
        request_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one /rerank request body for a chunk."""
        # texts maps directly onto the chunk documents (§8.1; no per-document
        # object wrapping). truncate is sent false EXPLICITLY (fail-loud -- an
        # over-length pair errors rather than silently truncating); return_text
        # tracks return_documents. Extras merge FIRST so the managed keys always
        # win: a caller extra named "truncate" cannot defeat the fail-loud
        # guarantee, and one named "query" / "texts" cannot clobber the wire
        # identity.
        return {
            **request_extras,
            "query": query,
            "texts": list(chunk),
            "truncate": False,
            "return_text": return_documents,
        }

    def _parse_chunk(
        self,
        resp: httpx.Response,
        chunk_len: int,
        offset: int,
    ) -> tuple[list[ScoredDocument], Any]:
        """Parse one TEI /rerank chunk response, re-basing indices to absolute.

        Returns the chunk's ScoredDocuments (with absolute indices) and the
        verbatim parsed chunk body.
        """
        # TEI /rerank returns a bare JSON array [{index, score, text?}] with
        # CHUNK-LOCAL indices. Each local index is re-based to its absolute
        # position (offset + local). A local index outside the chunk range is a
        # malformed response.
        try:
            body_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("rerank response is not valid JSON") from exc
        if not isinstance(body_raw, list):
            raise ProviderInvalidResponse("TEI /rerank response is not a JSON array")
        entries = cast("list[Any]", body_raw)
        scored: list[ScoredDocument] = []
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                raise ProviderInvalidResponse("rerank response entry is not a JSON object")
            entry = cast("dict[str, Any]", raw_entry)
            index = entry.get("index")
            # bool is an int subclass, so exclude it explicitly.
            if not isinstance(index, int) or isinstance(index, bool):
                raise ProviderInvalidResponse("rerank response entry missing integer 'index'")
            if index < 0 or index >= chunk_len:
                raise ProviderInvalidResponse(
                    f"TEI /rerank chunk-local index {index} out of range for chunk of {chunk_len}"
                )
            score = entry.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ProviderInvalidResponse("rerank response entry has a missing or non-numeric 'score'")
            # Read ``text`` only when present; never auto-fill from the input
            # documents list (§6 -- the provider's echo and the caller's input
            # are two different surfaces).
            text_raw = entry.get("text")
            document = text_raw if isinstance(text_raw, str) else None
            scored.append(
                ScoredDocument(index=offset + index, relevance_score=float(score), document=document)
            )
        return scored, entries


__all__ = ["TeiEmbeddingProvider", "TeiRerankProvider"]
