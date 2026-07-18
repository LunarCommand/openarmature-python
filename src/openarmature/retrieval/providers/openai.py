# Spec: realizes the retrieval-provider §3 EmbeddingProvider protocol
# against an OpenAI-compatible POST /v1/embeddings endpoint -- the
# reference embedding provider. The wire mapping proposal 0079 formalizes
# (base_url override, encoding_format) is pre-satisfied here; 0079 pins
# the remaining wire details + its fixtures 023-027. The §7 error
# categories are shared with llm-provider; the embedding-applicable subset
# (no unsupported_content_block, no structured_output_invalid) is mapped
# from the OpenAI-shape HTTP error envelope below. Typed EmbeddingEvent /
# EmbeddingFailedEvent dispatch mirrors the llm-provider §6 path via
# current_dispatch(). FOLLOW-UP: classify_http_error / base_url
# normalization are duplicated in spirit from llm.providers.openai;
# lifting a shared OpenAI-shape HTTP helper is a multi-provider follow-on.

"""OpenAI-compatible embedding provider.

``OpenAIEmbeddingProvider`` issues ``POST {base_url}/v1/embeddings`` and
parses the OpenAI ``{data: [{index, embedding}], model, usage}`` envelope
into an :class:`EmbeddingResponse`. ``base_url`` is the host root (the
provider appends ``/v1/embeddings`` and ``/v1/models``), overridable for
any OpenAI-compatible backend (vLLM, LocalAI, TEI's OpenAI surface). The
``transport`` parameter is the test seam (``httpx.MockTransport``).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any, Literal, cast

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

from .._events import build_embedding_event, build_embedding_failed_event
from .._wire import (
    apply_client_side_prefix,
    chunk_and_stitch_embed,
    nonneg_int,
    normalize_base_url,
)
from ..provider import validate_embedding_input
from ..response import EmbeddingResponse, EmbeddingRuntimeConfig

# §8.3 *Batch chunking (2048-input cap)*: OpenAI /v1/embeddings accepts at most
# 2048 inputs per request, so an over-cap embed call chunk-and-stitches over
# consecutive <=2048 slices (the §8 general embed rule; a fixed vendor cap like
# Cohere's 96, not a construction-configured chunk_size like TEI). OpenAI also
# enforces a summed-token ceiling per request, which the count-based rule does
# not address: an over-token chunk still fails loud as provider_invalid_request
# (§8's rule chunks by input count).
_OPENAI_EMBED_MAX_INPUTS = 2048


def _classify_embedding_http_error(resp: httpx.Response) -> LlmProviderError:
    """Map a non-200 OpenAI-shape embeddings response to an error category.

    The applicable subset: 401/403 to auth, 429 to rate_limit, 404 to
    invalid_model, 400/422 to invalid_request, and every other status to
    unavailable. Returns the exception (does not raise) so the caller
    raises with consistent traceback context.
    """
    status = resp.status_code
    try:
        body_raw = resp.json()
    except ValueError:
        body_raw = {}
    body: dict[str, Any] = cast("dict[str, Any]", body_raw) if isinstance(body_raw, dict) else {}
    error_block_raw = body.get("error")
    error_block: dict[str, Any] = (
        cast("dict[str, Any]", error_block_raw) if isinstance(error_block_raw, dict) else {}
    )
    message_raw = error_block.get("message")
    message = message_raw if isinstance(message_raw, str) else None

    if status in (401, 403):
        return ProviderAuthentication(message or f"HTTP {status}")
    if status == 429:
        # The Retry-After surface is wired when a 429 embedding fixture
        # lands; unfixtured at this pin, so retry_after stays unset.
        return ProviderRateLimit(message or "HTTP 429")
    if status == 404:
        return ProviderInvalidModel(message or "model not found")
    if status in (400, 422):
        return ProviderInvalidRequest(message or f"HTTP {status}")
    return ProviderUnavailable(message or f"HTTP {status}")


# Absence-is-meaningful per observability §5.5.2: only caller-supplied keys
# appear in the event's request_params -- "the field was not supplied for
# this call", distinct from a supplied zero. For OpenAI, input_type joins
# dimensions in this set (proposal 0079). input_type does NOT feed the wire
# body verbatim -- the OpenAI wire has no query/document field, so input_type
# is a wire no-op (realized only as a client-side prefix when one is bound),
# and the wire body is built separately.
def _request_params_from_config(config: EmbeddingRuntimeConfig | None) -> dict[str, Any]:
    """Extract the supplied embedding request parameters for the event."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if config.dimensions is not None:
        out["dimensions"] = config.dimensions
    if config.input_type is not None:
        out["input_type"] = config.input_type
    return out


_VALID_READINESS_PROBES = frozenset({"embed", "models", "both"})


class OpenAIEmbeddingProvider:
    """OpenAI ``/v1/embeddings`` wire-compatible embedding provider.

    Construct with the bound embedding model and an optional API key +
    transport. ``base_url`` is the host root and defaults to the OpenAI
    origin (``https://api.openai.com``), overridable for any OpenAI-compatible
    backend. ``embed()`` posts to ``/v1/embeddings``.

    The optional ``query_prefix`` / ``document_prefix`` bind the client-side
    asymmetric prefixes -- off by default (pure-symmetric OpenAI). When bound
    (for an asymmetric model served behind a compatible endpoint),
    ``input_type`` selects which prefix to prepend to each input before
    sending, since the OpenAI wire carries no query/document field.

    ``ready()`` verifies the bound model per the ``readiness_probe``
    argument:

    - ``"embed"`` (default): a one-input ``/v1/embeddings`` probe. Works
      against any OpenAI-compatible backend, including ones that do not
      serve the ``/v1/models`` catalog (e.g. TEI's OpenAI surface).
    - ``"models"``: a ``GET /v1/models`` catalog check. Cheaper (no embed
      billed), but requires the endpoint to serve the catalog.
    - ``"both"``: the catalog check, then the embed probe.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com",
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "openai",
        readiness_probe: Literal["embed", "models", "both"] = "embed",
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the host root; the provider appends the /v1 routes, so
        # a trailing /v1 would produce a doubled /v1/v1 path that 404s (the
        # sibling llm provider guards the same footgun). Trailing slashes are
        # stripped. Proposal 0079 pins the full base_url-override contract
        # (fixture 025).
        self.base_url = normalize_base_url(base_url, guard_prefix="/v1")
        self.model = model
        # ``readiness_probe`` modes are documented on the class docstring;
        # the default "embed" is the universal probe. Reject an unknown mode.
        if readiness_probe not in _VALID_READINESS_PROBES:
            raise ValueError(
                f"readiness_probe must be one of {sorted(_VALID_READINESS_PROBES)} (got {readiness_probe!r})"
            )
        self._readiness_probe = readiness_probe
        # Optional client-side asymmetric prefixes (§8.1, reused by §8.3): the
        # OpenAI wire has no query/document field, so input_type is realized as
        # a prefix prepend only when these are bound. Off by default (symmetric).
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
        # Dispatches on readiness_probe (see __init__): "embed" issues a
        # minimal /v1/embeddings call (universal across backends), "models"
        # the /v1/models catalog check, "both" runs catalog then embed.
        if self._readiness_probe in ("models", "both"):
            await self._probe_models()
        if self._readiness_probe in ("embed", "both"):
            await self._probe_embed()

    async def _probe_embed(self) -> None:
        # The universal probe: a one-input embed against /v1/embeddings.
        # Surfaces provider_invalid_model (404) / provider_unavailable (5xx)
        # / provider_authentication exactly as a real embed() would, against
        # any OpenAI-compatible backend (OpenAI, vLLM, TEI's OpenAI surface).
        body = {"model": self.model, "input": ["ready"]}
        try:
            resp = await self._client.post("/v1/embeddings", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_embedding_http_error(resp)

    async def _probe_models(self) -> None:
        # Catalog probe: GET /v1/models + bound-model presence check.
        # Cheaper than _probe_embed (no embed billed) but requires the
        # endpoint to serve the /v1/models catalog -- OpenAI / vLLM do, TEI
        # does not.
        try:
            resp = await self._client.get("/v1/models")
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"readiness probe failed: {exc}") from exc
        if resp.status_code != 200:
            raise _classify_embedding_http_error(resp)
        try:
            body_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("models response is not valid JSON") from exc
        if not isinstance(body_raw, dict):
            raise ProviderInvalidResponse("models response is not a JSON object")
        body = cast("dict[str, Any]", body_raw)
        data_raw = body.get("data")
        if not isinstance(data_raw, list):
            raise ProviderInvalidResponse("models response missing 'data' array")
        models = cast("list[Any]", data_raw)
        ids = [cast("dict[str, Any]", m).get("id") for m in models if isinstance(m, dict)]
        if self.model not in ids:
            seen = sorted(i for i in ids if isinstance(i, str))
            raise ProviderInvalidModel(f"model {self.model!r} not in catalog (seen: {seen})")

    async def embed(
        self,
        input: Sequence[str],
        *,
        config: EmbeddingRuntimeConfig | None = None,
    ) -> EmbeddingResponse:
        """Embed ``input`` into one vector per string, in input order."""
        dispatch = current_dispatch()
        call_id = str(uuid.uuid4())
        # Snapshot prompt context at dispatch time (the node task's
        # context); the delivery worker has a stale ContextVar view.
        # Lazy import avoids the prompts -> graph -> ... cycle.
        from openarmature.prompts.context import current_prompt_group, current_prompt_result

        active_prompt = current_prompt_result()
        active_prompt_group = current_prompt_group()
        input_strings = list(input)
        request_params = _request_params_from_config(config)
        request_extras = dict(config.model_extra or {}) if config is not None else {}
        input_type = config.input_type if config is not None else None
        dimensions = config.dimensions if config is not None else None
        adapter_start = time.perf_counter()
        try:
            validate_embedding_input(input_strings)
            response = await self._embed_chunked(input_strings, input_type, dimensions, request_extras)
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
        input_type: str | None,
        dimensions: int | None,
        request_extras: dict[str, Any],
    ) -> EmbeddingResponse:
        """Issue one /v1/embeddings per <=2048 chunk and stitch the vectors."""

        # THE chunk-and-stitch (§8.3 *Batch chunking (2048-input cap)* / the §8
        # general embed rule) via the shared chunk_and_stitch_embed helper. The
        # per-chunk closure builds the /v1/embeddings body with IDENTICAL
        # per-call params (only input differs), POSTs, classifies the HTTP
        # error, and parses the chunk into (vectors, input_tokens, id, model,
        # body); the helper loops the consecutive <=2048 slices, concatenates
        # the vectors IN INPUT ORDER, sums input_tokens, takes the response
        # identity from the FIRST chunk, and validates the §4 invariants against
        # the stitched result. When len(input) <= 2048 this issues a single
        # request. Valid because each input's embedding is independent of the
        # others in its batch. The client-side input_type prefix is applied
        # per-chunk inside _build_request_body, which is equivalent to prefixing
        # the whole list up front (the prefix is per-string).
        async def _embed_one(
            chunk: list[str],
        ) -> tuple[list[list[float]], int | None, str | None, str | None, dict[str, Any]]:
            body = self._build_request_body(chunk, input_type, dimensions, request_extras)
            try:
                resp = await self._client.post("/v1/embeddings", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"embedding request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_embedding_http_error(resp)
            return self._parse_chunk(resp)

        return await chunk_and_stitch_embed(
            input_strings,
            model=self.model,
            cap=_OPENAI_EMBED_MAX_INPUTS,
            embed_chunk=_embed_one,
        )

    def _build_request_body(
        self,
        input_strings: list[str],
        input_type: str | None,
        dimensions: int | None,
        request_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the /v1/embeddings request body.

        The OpenAI wire has no query/document field, so ``input_type`` is not
        realized on the wire -- it selects a client-side ``query_prefix`` /
        ``document_prefix`` prepend when one is bound, and is otherwise a wire
        no-op (``input`` sent verbatim). ``dimensions`` maps to the wire
        ``dimensions`` field (Matryoshka) when set.
        """
        # input_type is realized as a client-side prefix (§8.1, reused by §8.3):
        # the OpenAI wire has NO query/document/input_type/task field, so an
        # input_type with no bound prefix leaves the input VERBATIM (the
        # symmetric no-op). input_type MUST NEVER land on the wire. Extras merge
        # FIRST so the managed keys (model, input, dimensions) always win: a
        # caller's undeclared extra named "model" or "input" must not clobber
        # the wire identity. encoding_format ("base64") rides the extras bag as
        # a request pass-through only -- _parse_response decodes float embeddings
        # (not base64 strings), so a base64 response raises
        # provider_invalid_response; base64 is not an end-to-end response format.
        inputs = apply_client_side_prefix(
            input_strings,
            input_type,
            query_prefix=self._query_prefix,
            document_prefix=self._document_prefix,
        )
        body: dict[str, Any] = {**request_extras, "model": self.model, "input": inputs}
        if dimensions is not None:
            body["dimensions"] = dimensions
        return body

    def _parse_chunk(
        self,
        resp: httpx.Response,
    ) -> tuple[list[list[float]], int | None, str | None, str | None, dict[str, Any]]:
        """Parse one /v1/embeddings chunk into (vectors, input_tokens, id, model, raw)."""
        # Orders ``data`` by the per-entry ``index`` so output position matches
        # input position WITHIN THE CHUNK. The index set MUST be a 0..n-1
        # permutation of the chunk's inputs so each vector maps to exactly one
        # input, and every entry MUST carry a non-empty numeric embedding -- a
        # missing / empty / non-numeric vector is a malformed response, not a
        # zero-dim result. The §4 cross-input invariants (one vector per input,
        # uniform dimensionality) are validated by the caller against the
        # STITCHED result, and the per-chunk vector count by the helper.
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
            if not isinstance(index, int):
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
            # Reject non-numeric vector values (JSON strings, bools) rather
            # than coercing them: bool is an int subclass, and float("0.1")
            # would silently accept a string, so the strict isinstance check
            # is what makes "non-numeric is malformed" actually hold.
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
                raise ProviderInvalidResponse("embedding response has a non-numeric vector value")
            vectors.append([float(x) for x in values])
        # §4 (proposal 0093): usage is a record or null -- an absent or malformed
        # usage block yields usage = null, NEVER a fabricated record or a zero
        # (a zero would assert "zero tokens billed", a claim the provider never
        # made). A REPORTED zero is a different thing and does yield a record
        # carrying 0. The hosted OpenAI wire always reports usage; the null path
        # is what an OpenAI-compatible backend that omits it produces. Under
        # chunking the helper sums the per-chunk figures.
        usage_block = body.get("usage")
        input_tokens = (
            nonneg_int(cast("dict[str, Any]", usage_block).get("prompt_tokens"), field="prompt_tokens")
            if isinstance(usage_block, dict)
            else None
        )
        # An empty-string id / model carries no provider identifier (§4: the
        # reported identifier is used only when the body carries one), so treat
        # "" as absent -> None. For the model that lets the stitch fall back to
        # the bound identifier rather than surfacing "".
        response_id = body.get("id")
        model = body.get("model")
        return (
            vectors,
            input_tokens,
            response_id if isinstance(response_id, str) and response_id else None,
            model if isinstance(model, str) and model else None,
            body,
        )


__all__ = ["OpenAIEmbeddingProvider"]
