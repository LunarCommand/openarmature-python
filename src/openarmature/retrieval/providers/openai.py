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
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import httpx

from openarmature.graph.events import EmbeddingEvent, EmbeddingFailedEvent
from openarmature.llm.errors import (
    LlmProviderError,
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderRateLimit,
    ProviderUnavailable,
)
from openarmature.observability.correlation import (
    current_attempt_index,
    current_branch_name,
    current_correlation_id,
    current_dispatch,
    current_fan_out_index,
    current_invocation_id,
    current_namespace_prefix,
)
from openarmature.observability.metadata import AttributeValue, current_invocation_metadata

from ..provider import validate_embedding_input, validate_embedding_response
from ..response import EmbeddingResponse, EmbeddingRuntimeConfig, EmbeddingUsage


def _classify_embedding_http_error(resp: httpx.Response) -> LlmProviderError:
    """Map a non-200 OpenAI-shape embeddings response to an error category.

    The applicable subset: 401/403 to auth, 429 to rate_limit, 404 to
    invalid_model, 400/422 to invalid_request, and other 5xx to
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
# this call", distinct from a supplied zero. The input_type field joins this
# set with proposal 0077. For embedding the event-param names coincide with
# the wire-body keys, so the same dict feeds both the event and the body.
def _request_params_from_config(config: EmbeddingRuntimeConfig | None) -> dict[str, Any]:
    """Extract the supplied embedding request parameters for the event."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if config.dimensions is not None:
        out["dimensions"] = config.dimensions
    return out


_VALID_READINESS_PROBES = frozenset({"embed", "models", "both"})


class OpenAIEmbeddingProvider:
    """OpenAI ``/v1/embeddings`` wire-compatible embedding provider.

    Construct with a base URL (host root), the bound embedding model, and
    an optional API key + transport. ``embed()`` posts to
    ``/v1/embeddings``.

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
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        genai_system: str = "openai",
        readiness_probe: Literal["embed", "models", "both"] = "embed",
        populate_caller_metadata: bool = True,
    ) -> None:
        # base_url is the host root; the provider appends the /v1 routes, so
        # a trailing /v1 would produce a doubled /v1/v1 path that 404s (the
        # sibling llm provider guards the same footgun). Trailing slashes are
        # stripped. Proposal 0079 pins the full base_url-override contract
        # (fixture 025); lifting a shared base_url helper is a follow-up.
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            raise ValueError(
                f"base_url should be the host root (e.g. 'https://api.openai.com'); "
                f"the provider appends /v1/embeddings and /v1/models itself, so a "
                f"trailing /v1 would produce a doubled /v1/v1 path. Got {base_url!r}."
            )
        self.base_url = normalized
        self.model = model
        # ``readiness_probe`` modes are documented on the class docstring;
        # the default "embed" is the universal probe. Reject an unknown mode.
        if readiness_probe not in _VALID_READINESS_PROBES:
            raise ValueError(
                f"readiness_probe must be one of {sorted(_VALID_READINESS_PROBES)} (got {readiness_probe!r})"
            )
        self._readiness_probe = readiness_probe
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
        body = cast("dict[str, Any]", body_raw) if isinstance(body_raw, dict) else {}
        data_raw = body.get("data")
        models = cast("list[Any]", data_raw) if isinstance(data_raw, list) else []
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
        adapter_start = time.perf_counter()
        try:
            validate_embedding_input(input_strings)
            body = self._build_request_body(input_strings, request_params, request_extras)
            try:
                resp = await self._client.post("/v1/embeddings", json=body)
            except httpx.HTTPError as exc:
                raise ProviderUnavailable(f"embedding request failed: {exc}") from exc
            if resp.status_code != 200:
                raise _classify_embedding_http_error(resp)
            response = self._parse_response(resp, input_strings)
        except LlmProviderError as exc:
            latency_ms_failed = (time.perf_counter() - adapter_start) * 1000.0
            if dispatch is not None:
                dispatch(
                    self._build_embedding_failed_event(
                        exc,
                        latency_ms_failed,
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
                self._build_embedding_event(
                    response,
                    latency_ms,
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
        request_params: dict[str, Any],
        request_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the /v1/embeddings request body."""
        # Extras are merged FIRST so the bound model, the input list, and the
        # declared request params always win: a caller's undeclared extra
        # named "model" or "input" must not clobber the wire identity. The
        # request params are the same dict the event carries (dimensions for
        # embedding), keeping the wire body and the observed request_params
        # provably identical. Proposal 0077's input_type maps to a per-vendor
        # wire key at the wire layer.
        return {**request_extras, "model": self.model, "input": input_strings, **request_params}

    def _parse_response(
        self,
        resp: httpx.Response,
        input_strings: list[str],
    ) -> EmbeddingResponse:
        """Parse the OpenAI embeddings envelope into an EmbeddingResponse."""
        # Orders ``data`` by the per-entry ``index`` so output position
        # matches input position, then validates the §4 invariants (count +
        # consistent dimensionality, via validate_embedding_response). The
        # index set MUST be a 0..n-1 permutation so each vector maps to
        # exactly one input, and every entry MUST carry a non-empty numeric
        # embedding -- a missing / empty / non-numeric vector is a malformed
        # response, not a zero-dim result.
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
        dimensions = validate_embedding_response(vectors, len(input_strings))
        usage_block = body.get("usage")
        prompt_tokens = (
            cast("dict[str, Any]", usage_block).get("prompt_tokens")
            if isinstance(usage_block, dict)
            else None
        )
        # bool is an int subclass, so exclude it explicitly; a malformed or
        # absent usage falls back to 0 (the embed succeeded; usage is secondary).
        input_tokens = (
            prompt_tokens
            if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens >= 0
            else 0
        )
        response_id = body.get("id")
        model = body.get("model")
        return EmbeddingResponse(
            vectors=vectors,
            model=model if isinstance(model, str) else self.model,
            usage=EmbeddingUsage(input_tokens=input_tokens),
            response_id=response_id if isinstance(response_id, str) else None,
            dimensions=dimensions,
            raw=body,
        )

    def _build_embedding_event(
        self,
        response: EmbeddingResponse,
        latency_ms: float,
        *,
        call_id: str,
        input_strings: list[str],
        request_params: dict[str, Any],
        request_extras: dict[str, Any],
        active_prompt: Any,
        active_prompt_group: Any,
    ) -> EmbeddingEvent:
        """Construct the typed EmbeddingEvent for the success path.

        Sources identity / scoping from the calling-node ContextVars and
        outcome fields from the response.
        """
        namespace = current_namespace_prefix()
        node_name = namespace[-1] if namespace else ""
        invocation_id = current_invocation_id() or ""
        caller_metadata: Mapping[str, AttributeValue] | None = None
        if self._populate_caller_metadata:
            caller_metadata = dict(current_invocation_metadata())
        return EmbeddingEvent(
            invocation_id=invocation_id,
            correlation_id=current_correlation_id(),
            node_name=node_name,
            namespace=namespace,
            attempt_index=current_attempt_index(),
            fan_out_index=current_fan_out_index(),
            branch_name=current_branch_name(),
            provider=self._genai_system,
            model=self.model,
            response_id=response.response_id,
            response_model=response.model,
            usage=response.usage,
            latency_ms=latency_ms,
            input_strings=input_strings,
            input_count=len(input_strings),
            dimensions=response.dimensions,
            request_params=request_params,
            request_extras=request_extras,
            active_prompt=active_prompt,
            active_prompt_group=active_prompt_group,
            call_id=call_id,
            caller_invocation_metadata=caller_metadata,
        )

    def _build_embedding_failed_event(
        self,
        exc: LlmProviderError,
        latency_ms: float,
        *,
        call_id: str,
        input_strings: list[str],
        request_params: dict[str, Any],
        request_extras: dict[str, Any],
        active_prompt: Any,
        active_prompt_group: Any,
    ) -> EmbeddingFailedEvent:
        """Construct the typed EmbeddingFailedEvent for the failure path.

        ``error_type`` defaults to the exception class name (the
        "upstream exception class name" style).
        """
        namespace = current_namespace_prefix()
        node_name = namespace[-1] if namespace else ""
        invocation_id = current_invocation_id() or ""
        caller_metadata: Mapping[str, AttributeValue] | None = None
        if self._populate_caller_metadata:
            caller_metadata = dict(current_invocation_metadata())
        return EmbeddingFailedEvent(
            invocation_id=invocation_id,
            correlation_id=current_correlation_id(),
            node_name=node_name,
            namespace=namespace,
            attempt_index=current_attempt_index(),
            fan_out_index=current_fan_out_index(),
            branch_name=current_branch_name(),
            provider=self._genai_system,
            model=self.model,
            latency_ms=latency_ms,
            input_strings=input_strings,
            request_params=request_params,
            request_extras=request_extras,
            active_prompt=active_prompt,
            active_prompt_group=active_prompt_group,
            call_id=call_id,
            error_category=exc.category,
            error_type=type(exc).__name__,
            error_message=str(exc),
            caller_invocation_metadata=caller_metadata,
        )


__all__ = ["OpenAIEmbeddingProvider"]
