# Spec: realizes llm-provider §8.1 (OpenAI-compatible wire-format mapping) including
# the §8.1.3 wire-error mapping table.

"""OpenAI-compatible HTTPX-based provider.

Implements the :class:`Provider` Protocol against the OpenAI Chat
Completions wire format (``POST /v1/chat/completions``). The same
wire format is the de facto standard for vLLM, LM Studio, llama.cpp,
and other local LLM servers, so this provider talks to all of them
with the right ``base_url``.

**Error mapping:**

| OpenAI condition                                  | Category                     |
|---------------------------------------------------|------------------------------|
| ``ConnectError``/``ConnectTimeout``/``ReadTimeout``/network | provider_unavailable |
| HTTP 401, 403                                     | provider_authentication      |
| HTTP 404 with model-not-found body                | provider_invalid_model       |
| HTTP 503 with model-loading body                  | provider_model_not_loaded    |
| HTTP 429 (with ``Retry-After`` → ``retry_after``) | provider_rate_limit          |
| HTTP 400 (schema violation)                       | provider_invalid_request     |
| HTTP 5xx (other)                                  | provider_unavailable         |
| 200 OK that fails to parse into Response shape    | provider_invalid_response    |

**``ready()`` probe.** Hits ``GET /v1/models`` and:

- 401/403 → ``provider_authentication``.
- 5xx / connection error → ``provider_unavailable``.
- 200 + bound model in returned list → success.
- 200 + bound model NOT in list → ``provider_invalid_model``.

The ``provider_model_not_loaded`` distinction needs a server-specific
probe (LM Studio's loaded-vs-configured endpoint, vLLM's health
endpoint, llama.cpp's runtime-status endpoint) that this base
provider can't generically emit. Subclasses or purpose-built
local-server provider variants close that gap; the base
``OpenAIProvider`` documents the limitation here rather than silently
treating "model in catalog" as "model loaded."
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
import jsonschema
from pydantic import BaseModel, ValidationError

from openarmature.graph.events import NodeEvent
from openarmature.observability.correlation import (
    current_attempt_index,
    current_dispatch,
    current_fan_out_index,
    current_namespace_prefix,
)
from openarmature.observability.llm_event import LlmEventPayload

# ``current_prompt_group`` / ``current_prompt_result`` are imported
# lazily inside :meth:`OpenAIProvider.complete` to avoid a module-load
# cycle: ``openarmature.prompts.prompt`` imports ``RuntimeConfig`` from
# this package (for the ``SamplingConfig`` subclass), so a top-level
# import here would re-enter prompts.prompt before its types finish
# defining.
from ..errors import (
    LlmProviderError,
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderModelNotLoaded,
    ProviderRateLimit,
    ProviderUnavailable,
    ProviderUnsupportedContentBlock,
    StructuredOutputInvalid,
)
from ..messages import (
    AssistantMessage,
    ContentBlock,
    ForceTool,
    ImageBlock,
    ImageSourceInline,
    Message,
    SystemMessage,
    TextBlock,
    Tool,
    ToolCall,
    ToolChoice,
    UserMessage,
)
from ..provider import (
    strict_mode_supported,
    validate_message_list,
    validate_response_schema,
    validate_tool_choice,
    validate_tools,
)
from ..response import FinishReason, ParsedValue, Response, RuntimeConfig, Usage


class OpenAIProvider:
    """OpenAI Chat Completions wire-compatible provider.

    Construct with a base URL, model identifier, and optional API key
    + transport (an :class:`httpx.AsyncBaseTransport`). The
    ``transport`` parameter is the test seam; ``httpx.MockTransport``
    drives the conformance fixtures by intercepting HTTP calls and
    returning canned responses, exercising the same wire-mapping
    code production traffic would.

    **``base_url`` shape.** Pass the host root only — e.g.
    ``"https://api.openai.com"`` or ``"http://localhost:8000"``. The
    provider appends ``/v1/chat/completions`` and ``/v1/models``
    itself. A trailing ``/v1`` on ``base_url`` raises ``ValueError``:
    httpx joins paths by appending, so an unprefixed ``base_url``
    suffix would produce a doubled ``/v1/v1/...`` wire path that
    silently 404/405s on most backends (some — like Bifrost — return
    200 for ``GET /v1/v1/models`` while rejecting ``POST
    /v1/v1/chat/completions``, leaving the readiness probe green and
    every completion broken). Trailing slashes are stripped; other
    non-empty paths (proxy prefixes like ``/api/openai-proxy``) are
    left intact for intentional proxy setups.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
        force_prompt_augmentation_fallback: bool = False,
        genai_system: str = "openai",
    ) -> None:
        self.base_url = _validate_and_normalize_base_url(base_url)
        self.model = model
        # ``force_prompt_augmentation_fallback`` switches structured-output
        # calls from the native response_format wire path to the
        # prompt-augmentation fallback. Used for older OpenAI-compatible
        # servers (some vLLM/LM Studio/llama.cpp versions) that reject
        # or silently ignore response_format.
        self._force_prompt_augmentation_fallback = force_prompt_augmentation_fallback
        # ``genai_system`` surfaces as the ``gen_ai.system`` span attribute
        # per observability §5.5.3. The OpenAI Chat Completions wire format
        # is the de facto standard for vLLM, LM Studio, llama.cpp,
        # sglang, etc. — callers using this provider against a non-OpenAI
        # endpoint pass the appropriate identifier (e.g. ``"vllm"``).
        # No base_url-sniffing happens: the same host:port could be any of
        # those servers, and a wrong inference is worse than the explicit
        # opt-in.
        self._genai_system = genai_system
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"
        # The client is constructed eagerly; one client per provider is
        # the standard httpx idiom (connection pool reuse).
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            transport=transport,
            timeout=timeout,
        )

    @property
    def uses_prompt_augmentation_fallback(self) -> bool:
        """Whether ``complete(response_schema=...)`` builds the wire
        body via prompt augmentation (``True``) or the native
        ``response_format`` path (``False``).
        """
        return self._force_prompt_augmentation_fallback

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Optional; async clients
        garbage-collect cleanly, but explicit close is RECOMMENDED in
        long-lived services to release the connection pool promptly."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # ready() — pre-flight probe
    # ------------------------------------------------------------------

    async def ready(self) -> None:
        """Verify the bound model is reachable and listed by the
        provider. Hits ``GET /v1/models`` and matches ``self.model``
        against the returned ``data[].id`` entries."""
        try:
            resp = await self._client.get("/v1/models")
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc

        if resp.status_code in (401, 403):
            raise ProviderAuthentication(f"GET /v1/models returned {resp.status_code}")
        if 500 <= resp.status_code < 600:
            raise ProviderUnavailable(f"GET /v1/models returned {resp.status_code}")
        if resp.status_code != 200:
            raise ProviderUnavailable(f"GET /v1/models returned unexpected {resp.status_code}")

        try:
            body_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("GET /v1/models returned non-JSON body") from exc

        body = cast("dict[str, Any]", body_raw) if isinstance(body_raw, dict) else None
        models_list_raw = body.get("data") if body else None
        if not isinstance(models_list_raw, list):
            raise ProviderInvalidResponse("GET /v1/models response missing 'data' array")
        models_list = cast("list[Any]", models_list_raw)
        # Walk the catalog looking for our bound model. If we find it,
        # additionally consult an optional ``status`` field — local
        # servers (LM Studio, vLLM) include this to distinguish
        # "configured but not loaded" from "serving." Treat status
        # values containing "not_loaded" or "loading" as the model-
        # not-loaded condition; anything else (including absent) is
        # treated as ready. The substring set is best-effort and may
        # need expansion (e.g., "warming_up", "downloading") as more
        # local-server backends are exercised in the field.
        bound_entry: dict[str, Any] | None = None
        seen_ids: set[str] = set()
        for entry in models_list:
            if not isinstance(entry, dict):
                continue
            entry_dict = cast("dict[str, Any]", entry)
            entry_id = entry_dict.get("id")
            if not isinstance(entry_id, str):
                continue
            seen_ids.add(entry_id)
            if entry_id == self.model:
                bound_entry = entry_dict
        if bound_entry is None:
            raise ProviderInvalidModel(
                f"model {self.model!r} not in /v1/models catalog (seen: {sorted(seen_ids)})"
            )
        status_field = bound_entry.get("status")
        if isinstance(status_field, str):
            lower = status_field.lower()
            if "not_loaded" in lower or "loading" in lower:
                raise ProviderModelNotLoaded(
                    f"model {self.model!r} is configured but not loaded (status={status_field!r})"
                )

    # ------------------------------------------------------------------
    # complete() — single completion call
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        config: RuntimeConfig | None = None,
        response_schema: dict[str, Any] | type[BaseModel] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Response:
        """Single completion call.

        Pre-send validation runs first (per-message Pydantic +
        list-level invariants + response_schema shape check +
        ``tool_choice`` validation). HTTP errors map to canonical
        provider-error categories. The successful 200 body is parsed
        into a :class:`Response`; failure to parse raises
        ``provider_invalid_response``; failure to validate the response
        content against ``response_schema`` raises
        ``structured_output_invalid``.

        When ``response_schema`` is supplied as a Pydantic BaseModel
        subclass, ``Response.parsed`` is a validated instance of that
        class; when supplied as a JSON Schema dict,
        ``Response.parsed`` is the deserialized dict.

        ``tool_choice`` is validated against ``tools`` per spec §5:
        ``"required"`` and the ``ForceTool`` record both demand
        non-empty ``tools``, and ``ForceTool.name`` must appear in the
        supplied list. Violations raise ``provider_invalid_request``
        BEFORE any HTTP request is sent.
        """
        validate_message_list(messages)
        validate_tools(tools)
        # ``validate_tool_choice`` runs after ``validate_tools`` so the
        # name-membership check sees a structurally valid tools list.
        validate_tool_choice(tool_choice, tools)
        schema_dict, schema_class = _normalize_response_schema(response_schema)
        # On the fallback path, the wire-side messages list is an
        # augmented COPY of the caller's messages — original messages
        # MUST NOT be mutated. _augment_messages_with_schema_directive
        # builds a fresh list and does not modify the reused Message
        # instances in place; the caller's sequence is untouched.
        wire_messages: Sequence[Message] = messages
        if schema_dict is not None and self._force_prompt_augmentation_fallback:
            wire_messages = _augment_messages_with_schema_directive(messages, schema_dict)
        body = self._build_request_body(
            wire_messages,
            tools,
            config,
            schema_dict,
            # The fallback only governs structured-output calls; free-
            # form calls (schema_dict is None) must preserve any
            # caller-supplied response_format from RuntimeConfig extras.
            include_response_format=(schema_dict is None or not self._force_prompt_augmentation_fallback),
            tool_choice=tool_choice,
        )

        # Spec observability §5.5 LLM provider span: when an
        # observability backend is active in the current invocation,
        # emit a started/completed event pair around the wire call so
        # the backend can build a span. Queue-mediated dispatch
        # preserves spec §6 serial event ordering across all event
        # sources within an invocation. ``current_dispatch()`` returns
        # ``None`` outside an openarmature invocation (direct
        # provider use in scripts/tests), in which case the call
        # proceeds without span emission.
        #
        # ``call_id`` is minted once per ``complete()`` call and
        # threaded through both events of the pair. Backend
        # observers key their in-flight LLM-span maps by it so
        # concurrent ``complete()`` calls (e.g., fan-out instances
        # each calling this provider) don't collide on the
        # constant ``("openarmature.llm.complete",)`` sentinel.
        dispatch = current_dispatch()
        call_id = str(uuid.uuid4())
        # Capture prompt context AT DISPATCH TIME (in the node task's
        # context). The delivery worker (asyncio.create_task'd at
        # ``invoke()`` entry, before any node body runs) has a stale
        # ContextVar snapshot — reading ``current_prompt_result()``
        # from inside the observer in the worker task returns ``None``
        # even when a node body opened a ``with_active_prompt`` block.
        # Snapshot here; the observer reads from the event payload.
        # Lazy import: see module-level comment for the cycle reason.
        from openarmature.prompts.context import (
            current_prompt_group,
            current_prompt_result,
        )

        active_prompt = current_prompt_result()
        active_prompt_group = current_prompt_group()
        # Payload data the §5.5.1 / §5.5.2 / §5.5.3 attributes are
        # sourced from. Image redaction (per §5.5.5) happens inside
        # ``_serialize_messages_for_payload`` — image bytes never
        # leave the provider in event form. ``input_messages`` mirrors
        # the messages list the caller supplied; the wire-side body
        # may be augmented (schema directive on the fallback path),
        # but the OBSERVED messages are the spec-§3 logical inputs.
        serialized_messages = _serialize_messages_for_payload(messages)
        request_params = _request_params_from_config(config)
        request_extras = _request_extras_from_config(config)
        if dispatch is not None:
            dispatch(
                _make_llm_event(
                    "started",
                    call_id=call_id,
                    model=self.model,
                    genai_system=self._genai_system,
                    input_messages=serialized_messages,
                    request_params=request_params,
                    request_extras=request_extras,
                    active_prompt=active_prompt,
                    active_prompt_group=active_prompt_group,
                )
            )

        try:
            response = await self._do_complete(body, schema_dict, schema_class)
        except Exception as exc:
            if dispatch is not None:
                dispatch(
                    _make_llm_event(
                        "completed",
                        call_id=call_id,
                        model=self.model,
                        genai_system=self._genai_system,
                        error=exc,
                        input_messages=serialized_messages,
                        request_params=request_params,
                        request_extras=request_extras,
                        active_prompt=active_prompt,
                        active_prompt_group=active_prompt_group,
                    )
                )
            raise

        if dispatch is not None:
            dispatch(
                _make_llm_event(
                    "completed",
                    call_id=call_id,
                    model=self.model,
                    genai_system=self._genai_system,
                    finish_reason=response.finish_reason,
                    usage=response.usage,
                    input_messages=serialized_messages,
                    output_content=response.message.content or None,
                    request_params=request_params,
                    request_extras=request_extras,
                    response_id=response.response_id,
                    response_model=response.response_model,
                    active_prompt=active_prompt,
                    active_prompt_group=active_prompt_group,
                )
            )
        return response

    async def _do_complete(
        self,
        body: dict[str, Any],
        schema_dict: dict[str, Any] | None,
        schema_class: type[BaseModel] | None,
    ) -> Response:
        """Wire-call helper: separated from ``complete()`` so the
        LLM-provider span hook in ``complete()`` can wrap success and
        failure paths uniformly."""
        try:
            resp = await self._client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc

        if resp.status_code != 200:
            raise classify_http_error(resp)

        try:
            payload_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("POST /v1/chat/completions returned non-JSON body") from exc
        if not isinstance(payload_raw, dict):
            raise ProviderInvalidResponse("POST /v1/chat/completions returned a non-object body")
        return self._parse_response(cast("dict[str, Any]", payload_raw), schema_dict, schema_class)

    # ------------------------------------------------------------------
    # Request building (spec §8.1.1)
    # ------------------------------------------------------------------

    def _build_request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None,
        config: RuntimeConfig | None,
        schema_dict: dict[str, Any] | None,
        include_response_format: bool = True,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
        }
        if tools:
            body["tools"] = [_tool_to_wire(t) for t in tools]
        if config is not None:
            # Per spec §6 null-skip: each declared field with value
            # ``None`` is omitted from the wire body. Same-name keys
            # for the pre-0032 four; same-name for frequency_penalty /
            # presence_penalty per §8.1; ``stop_sequences`` renames to
            # OpenAI's body key ``stop`` per §8.1's only rename.
            if config.temperature is not None:
                body["temperature"] = config.temperature
            if config.max_tokens is not None:
                body["max_tokens"] = config.max_tokens
            if config.top_p is not None:
                body["top_p"] = config.top_p
            if config.seed is not None:
                body["seed"] = config.seed
            if config.frequency_penalty is not None:
                body["frequency_penalty"] = config.frequency_penalty
            if config.presence_penalty is not None:
                body["presence_penalty"] = config.presence_penalty
            if config.stop_sequences is not None:
                body["stop"] = config.stop_sequences
            # Pass-through any provider-specific extras (extra="allow"
            # on RuntimeConfig); spec §6 mandates implementations MUST
            # accept and forward undeclared fields untouched.
            extras = config.model_extra or {}
            for k, v in extras.items():
                body.setdefault(k, v)
        # response_format is omitted entirely on the fallback path —
        # the schema travels in the augmented system message instead.
        if schema_dict is not None and include_response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _derive_schema_name(schema_dict),
                    "schema": schema_dict,
                    "strict": strict_mode_supported(schema_dict),
                },
            }
        elif not include_response_format:
            # On the fallback path the §8.1.5.1 contract is "response_format
            # MUST NOT be on the wire." RuntimeConfig is extra="allow" so
            # a caller could pass response_format through via the extras
            # loop above; strip it here so the fallback contract holds
            # regardless of caller-supplied extras.
            body.pop("response_format", None)
        # Per §8.1.1 (proposal 0025): map the spec-level `tool_choice`
        # shape onto the OpenAI wire shape. ``None`` omits the field
        # entirely so the OpenAI provider's own default applies —
        # load-bearing for backward compat with pre-0025 callers. The
        # string-literal modes pass through verbatim; the ``ForceTool``
        # record renames ``type: "tool"`` → ``type: "function"`` and
        # nests the name under a ``function`` sub-object per OpenAI's
        # request shape.
        if tool_choice is not None:
            if isinstance(tool_choice, ForceTool):
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice.name},
                }
            else:
                body["tool_choice"] = tool_choice
        return body

    # ------------------------------------------------------------------
    # Response parsing (spec §8.1.2)
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        payload: dict[str, Any],
        schema_dict: dict[str, Any] | None,
        schema_class: type[BaseModel] | None,
    ) -> Response:
        try:
            choices = cast("list[dict[str, Any]]", payload["choices"])
            choice = choices[0]
            wire_msg = cast("dict[str, Any]", choice["message"])
            finish_reason_raw = choice["finish_reason"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderInvalidResponse(f"response missing required fields: {exc}") from exc
        finish_reason: str = finish_reason_raw if isinstance(finish_reason_raw, str) else "error"

        # Per §8.1.2 (and conformance fixture 005's
        # `function_call_legacy_finish_reason_mapping` case): the
        # legacy `finish_reason: "function_call"` value MUST be
        # normalized to the spec's `"tool_calls"`. This is a
        # finish_reason *value* rename only — the assistant message's
        # `tool_calls` list is already populated by
        # ``_wire_to_assistant_message`` from the `tool_calls` field
        # on the wire. We do NOT translate the deprecated single
        # `message.function_call` shape (no backend we target emits
        # it). Any other unknown finish_reason maps to `error`.
        if finish_reason == "function_call":
            finish_reason = "tool_calls"
        if finish_reason not in {"stop", "length", "tool_calls", "content_filter", "error"}:
            finish_reason = "error"
        finish_reason_typed = cast("FinishReason", finish_reason)

        # Build the assistant message. Tool calls under
        # finish_reason="error" may carry malformed argument JSON per
        # §3 — surface as `arguments=None` rather than raising.
        try:
            assistant_msg = _wire_to_assistant_message(
                wire_msg, lenient_args=(finish_reason_typed == "error")
            )
        except ProviderInvalidResponse:
            raise
        except Exception as exc:
            raise ProviderInvalidResponse(f"could not parse assistant message: {exc}") from exc

        # Usage is optional — build a Usage with all-None fields if
        # the provider didn't report it. Per spec §6, token counts MUST
        # be non-negative integers; a wire response that violates that
        # surfaces as ``provider_invalid_response`` rather than
        # silently passing through.
        usage_wire_raw = payload.get("usage")
        try:
            if isinstance(usage_wire_raw, dict):
                usage_wire = cast("dict[str, Any]", usage_wire_raw)
                usage = Usage(
                    prompt_tokens=usage_wire.get("prompt_tokens"),
                    completion_tokens=usage_wire.get("completion_tokens"),
                    total_tokens=usage_wire.get("total_tokens"),
                )
            else:
                usage = Usage(prompt_tokens=None, completion_tokens=None, total_tokens=None)
        except ValidationError as exc:
            raise ProviderInvalidResponse(f"invalid usage record: {exc}") from exc

        # Structured-output parsing. parsed is absent when no schema
        # was requested AND when the response is a tool-call response
        # — the tool-call path and structured-content path are
        # mutually exclusive at the response level.
        parsed: ParsedValue = None
        if schema_dict is not None and finish_reason_typed != "tool_calls":
            parsed = _parse_and_validate(assistant_msg.content, schema_dict, schema_class)

        # gen_ai.response.id / gen_ai.response.model semconv (spec
        # §5.5.3) read these off the Response. The wire fields are
        # optional — providers MAY omit either or both. ``None`` when
        # absent or not a string.
        response_id_raw = payload.get("id")
        response_id: str | None = response_id_raw if isinstance(response_id_raw, str) else None
        response_model_raw = payload.get("model")
        response_model: str | None = response_model_raw if isinstance(response_model_raw, str) else None

        return Response(
            message=assistant_msg,
            finish_reason=finish_reason_typed,
            usage=usage,
            raw=payload,
            parsed=parsed,
            response_id=response_id,
            response_model=response_model,
        )


# ---------------------------------------------------------------------------
# base_url validation
# ---------------------------------------------------------------------------


# Rejects base_urls that end in /v1 or /v1/ because httpx joins paths by
# appending — a base_url with a trailing /v1 produces a doubled /v1/v1/...
# wire path. The failure mode is sneaky: some backends (Bifrost was the
# motivating case) return 200 for GET /v1/v1/models while rejecting POST
# /v1/v1/chat/completions, so the readiness probe stays green while every
# completion fails. Strict rejection is safer than silent strip — it keeps
# the bug visible at construction time.
def _validate_and_normalize_base_url(base_url: str) -> str:
    """Validate ``base_url`` and return its normalized form.

    Strips trailing slashes. Raises :class:`ValueError` when the path
    component ends in ``/v1`` (with or without a trailing slash) — the
    provider appends ``/v1/`` segments itself, so a base_url with a
    ``/v1`` suffix would produce a doubled path on the wire. Other
    non-empty paths (e.g., proxy prefixes like ``/api/openai-proxy``)
    are left intact.
    """
    normalized = base_url.rstrip("/")
    path = urlparse(normalized).path
    if path == "/v1" or path.endswith("/v1"):
        raise ValueError(
            f"OpenAIProvider base_url must not end with '/v1' — the provider "
            f"appends '/v1/chat/completions' and '/v1/models' itself, and "
            f"httpx would produce a doubled '/v1/v1/...' wire path. Pass the "
            f"host root instead (e.g., 'https://api.openai.com'). "
            f"Got: {base_url!r}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


# Normalize a response_schema argument to a dict (plus the optional
# BaseModel subclass form for the post-parse instance return). Accepts
# either form per the Provider Protocol; raises ProviderInvalidRequest
# on invalid shapes (non-dict, non-object-top-level for the dict form;
# pre-validated by validate_response_schema).
def _normalize_response_schema(
    response_schema: dict[str, Any] | type[BaseModel] | None,
) -> tuple[dict[str, Any] | None, type[BaseModel] | None]:
    if response_schema is None:
        return None, None
    if isinstance(response_schema, type):
        # Defensive runtime check: the Protocol signature accepts
        # type[BaseModel], but Python doesn't enforce that at the call
        # boundary. Reject non-BaseModel classes with a canonical error
        # instead of letting AttributeError leak from model_json_schema.
        if not issubclass(response_schema, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ProviderInvalidRequest(
                f"response_schema: class form MUST be a Pydantic BaseModel subclass "
                f"(got {response_schema.__name__})"
            )
        schema_dict = response_schema.model_json_schema()
        validate_response_schema(schema_dict)
        return schema_dict, response_schema
    validate_response_schema(response_schema)
    return response_schema, None


# OpenAI's response_format.json_schema.name field is restricted to
# letters, digits, underscores, and dashes with a max length of 64
# characters. A JSON Schema title can be any string ("Person Record",
# "User's Profile", etc.), so verbatim use risks a 400 on the wire.
_OPENAI_SCHEMA_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# Derive a stable identifier for the JSON Schema for OpenAI's
# response_format.json_schema.name field. Uses the schema's `title`
# when it satisfies the provider's name constraints; otherwise derives
# a deterministic short hash so the same schema always produces the
# same name across calls. Sanitizing-in-place would silently mutate
# user intent; the hash is a more honest fallback.
def _derive_schema_name(schema: dict[str, Any]) -> str:
    title = schema.get("title")
    if isinstance(title, str) and _OPENAI_SCHEMA_NAME_RE.match(title):
        return title
    canonical = json.dumps(schema, sort_keys=True).encode("utf-8")
    return f"oa_schema_{hashlib.sha256(canonical).hexdigest()[:16]}"


# Parse the model's content string as JSON, then validate against
# the schema. The dict-schema path uses jsonschema; the BaseModel-class
# path uses Pydantic's native validator (which produces an instance
# of the supplied class).
def _parse_and_validate(
    content: str,
    schema_dict: dict[str, Any],
    schema_class: type[BaseModel] | None,
) -> ParsedValue:
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StructuredOutputInvalid(
            "response content is not valid JSON",
            response_schema=schema_dict,
            raw_content=content,
            failure_description=str(exc),
        ) from exc
    if not isinstance(loaded, dict):
        raise StructuredOutputInvalid(
            "response JSON is not an object",
            response_schema=schema_dict,
            raw_content=content,
            failure_description=f"top-level type is {type(loaded).__name__}, expected object",
        )
    parsed_dict = cast("dict[str, Any]", loaded)

    # Pydantic-class path: validate and return the BaseModel instance.
    if schema_class is not None:
        # Validate against the generated JSON Schema FIRST so the
        # class path enforces the same strict per-type checks as the
        # dict path. Pydantic's default model_validate is coercive
        # (it accepts "30" for an int field), which would silently
        # accept responses that fail the wire schema. Running
        # jsonschema first matches the dict-schema path's strictness;
        # model_validate then constructs the typed instance.
        try:
            jsonschema.validate(instance=parsed_dict, schema=schema_dict)
        except jsonschema.ValidationError as exc:
            raise StructuredOutputInvalid(
                "response failed JSON Schema validation",
                response_schema=schema_dict,
                raw_content=content,
                failure_description=_format_jsonschema_failure(exc),
            ) from exc
        except jsonschema.SchemaError as exc:
            raise StructuredOutputInvalid(
                "response could not be validated against the supplied schema",
                response_schema=schema_dict,
                raw_content=content,
                failure_description=str(exc),
            ) from exc
        try:
            return schema_class.model_validate(parsed_dict)
        except ValidationError as exc:
            raise StructuredOutputInvalid(
                "response failed Pydantic validation",
                response_schema=schema_dict,
                raw_content=content,
                failure_description=str(exc),
            ) from exc

    # Dict-schema path: jsonschema validation, return the dict.
    try:
        jsonschema.validate(instance=parsed_dict, schema=schema_dict)
    except jsonschema.ValidationError as exc:
        raise StructuredOutputInvalid(
            "response failed JSON Schema validation",
            response_schema=schema_dict,
            raw_content=content,
            failure_description=_format_jsonschema_failure(exc),
        ) from exc
    except jsonschema.SchemaError as exc:
        # Safety net: validate_response_schema's pre-validation should
        # have caught this, but any schema-side exception (including
        # ref-resolution failures via the `referencing` library) MUST
        # still map to the canonical taxonomy rather than leak raw.
        raise StructuredOutputInvalid(
            "response could not be validated against the supplied schema",
            response_schema=schema_dict,
            raw_content=content,
            failure_description=str(exc),
        ) from exc
    return parsed_dict


def _format_jsonschema_failure(exc: jsonschema.ValidationError) -> str:
    """jsonschema.ValidationError.message describes the value mismatch
    (e.g., "'30' is not of type 'integer'") but doesn't include the
    failing field path. Prefix with ``json_path`` (e.g., ``$.age``) so
    the failure_description string carries both, matching the dict-
    schema and class-schema paths.
    """
    return f"{exc.json_path}: {exc.message}"


_SCHEMA_DIRECTIVE_TEMPLATE = (
    "You MUST return only valid JSON that conforms to the following JSON Schema. "
    "Do not include prose, markdown fences, or any text outside the JSON object.\n\n"
    "JSON Schema:\n{schema_json}"
)


# Construct a fresh message list with a schema directive added. The
# directive is appended to the existing system message's content when
# present, or prepended as a new system message otherwise. The caller's
# original list is never mutated; Message instances are reused, and
# this helper does not modify them in place (the message models are
# not frozen Pydantic models, so the safety is structural, not
# enforced by the type). The serialized schema appears verbatim in
# the directive so callers that need to verify the directive
# references the schema (conformance harnesses, observability spans)
# can substring-match the canonical JSON form.
def _augment_messages_with_schema_directive(
    messages: Sequence[Message],
    schema_dict: dict[str, Any],
) -> list[Message]:
    directive = _SCHEMA_DIRECTIVE_TEMPLATE.format(schema_json=json.dumps(schema_dict, sort_keys=True))
    out: list[Message] = list(messages)
    if out and isinstance(out[0], SystemMessage):
        existing = out[0]
        merged = SystemMessage(content=f"{existing.content}\n\n{directive}")
        out[0] = merged
    else:
        out.insert(0, SystemMessage(content=directive))
    return out


def _message_to_wire(msg: Message) -> dict[str, Any]:
    """Spec §8.1.1 request mapping for one message."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, UserMessage):
        # Dual-shape user content (§8.1.1): string maps directly; a
        # content-block sequence maps to OpenAI's content-array form
        # per §8.1.1.1.
        if isinstance(msg.content, str):
            return {"role": "user", "content": msg.content}
        return {
            "role": "user",
            "content": [_block_to_wire(block) for block in msg.content],
        }
    if isinstance(msg, AssistantMessage):
        # Tool-call-only assistants emit ``"content": null`` on the
        # wire — that's the OpenAI convention for "no textual reply,
        # only tool calls." Don't "fix" this to an empty string; the
        # API rejects empty-string content alongside tool_calls.
        wire: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
        if msg.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # Canonical compact form (no inter-token spaces). Matches
                        # the spec's wire-mapping fixture (005, cases shape) and
                        # the form OpenAI itself emits.
                        "arguments": json.dumps(tc.arguments or {}, separators=(",", ":")),
                    },
                }
                for tc in msg.tool_calls
            ]
        return wire
    # Discriminated union exhausted: msg must be ToolMessage here.
    return {
        "role": "tool",
        "content": msg.content,
        "tool_call_id": msg.tool_call_id,
    }


# Spec §8.1.1.1: content-block to OpenAI content-array entry mapping.
# Both URL-referenced and inline-base64 image blocks go through
# OpenAI's `image_url` entry shape; the inline case is expressed as
# an RFC 2397 data: URI carrying media_type + base64_data. The
# `detail` hint goes on the wire only when explicitly set on the spec
# block (None on the spec block omits it from the wire; providers
# apply their own conceptual default of "auto").
def _block_to_wire(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if not isinstance(block, ImageBlock):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"unhandled content block type: {type(block).__name__}")
    if isinstance(block.source, ImageSourceInline):
        url = f"data:{block.media_type};base64,{block.source.base64_data}"
    else:
        url = block.source.url
    image_url: dict[str, Any] = {"url": url}
    if block.detail is not None:
        image_url["detail"] = block.detail
    return {"type": "image_url", "image_url": image_url}


def _tool_to_wire(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _wire_to_assistant_message(wire: dict[str, Any], *, lenient_args: bool) -> AssistantMessage:
    """Parse OpenAI-shaped assistant message into spec §3 form.

    When ``lenient_args=True`` (i.e. ``finish_reason == "error"``),
    tool calls with unparseable JSON arguments populate
    ``arguments=None`` instead of raising. Per spec §3 "Validation
    under finish_reason: error" — degraded responses surface what
    they can; repair is a caller concern.
    """
    content_raw = wire.get("content") or ""
    content: str = content_raw if isinstance(content_raw, str) else ""
    raw_tool_calls = cast("list[Any]", wire.get("tool_calls") or [])
    parsed_tool_calls: list[ToolCall] = []
    for raw in raw_tool_calls:
        if not isinstance(raw, dict):
            raise ProviderInvalidResponse("tool_call entry is not a dict")
        raw_dict = cast("dict[str, Any]", raw)
        function_raw_any: Any = raw_dict.get("function") or {}
        if not isinstance(function_raw_any, dict):
            raise ProviderInvalidResponse("tool_call.function is not a dict")
        function = cast("dict[str, Any]", function_raw_any)
        # Preserve provider-supplied id verbatim — spec §3 requires
        # implementations MUST NOT rewrite or normalize.
        tc_id = raw_dict.get("id")
        if not isinstance(tc_id, str):
            raise ProviderInvalidResponse("tool_call.id missing or not a string")
        name = function.get("name")
        if not isinstance(name, str):
            raise ProviderInvalidResponse("tool_call.function.name missing")
        arguments_str = function.get("arguments", "{}")
        if not isinstance(arguments_str, str):
            raise ProviderInvalidResponse("tool_call.function.arguments must be a JSON-encoded string")
        arguments: dict[str, Any] | None
        try:
            arguments = cast("dict[str, Any]", json.loads(arguments_str)) if arguments_str else {}
        except json.JSONDecodeError:
            if lenient_args:
                arguments = None
            else:
                raise ProviderInvalidResponse(
                    f"tool_call.function.arguments is not valid JSON: {arguments_str!r}"
                ) from None
        parsed_tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments))
    return AssistantMessage(
        content=content,
        tool_calls=parsed_tool_calls or None,
    )


def classify_http_error(resp: httpx.Response) -> LlmProviderError:
    """Map a non-200 ``httpx.Response`` from an OpenAI-shape API to
    the right canonical error category.

    Returns the exception (does not raise) so the caller can
    ``raise`` with consistent traceback context.

    Reusable by third-party Provider implementations targeting any
    OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server,
    etc.); the wire shape is stable across these and the helper
    saves implementers from reimplementing the mapping table.
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
    error_type = error_block.get("type")
    error_code = error_block.get("code")
    message_raw = error_block.get("message")
    message = message_raw if isinstance(message_raw, str) else None

    if status in (401, 403):
        return ProviderAuthentication(message or f"HTTP {status}")
    if status == 400:
        # Spec §8.1.3: HTTP 400 bodies that indicate the bound model
        # rejected a content block map to provider_unsupported_content_block
        # rather than the generic provider_invalid_request. The
        # detection rule is implementation-defined.
        if _looks_like_content_rejection(error_code, error_type, message):
            return ProviderUnsupportedContentBlock(
                message or "HTTP 400 (content block not supported)",
                block_type=_extract_rejected_block_type(error_code, message),
                reason=message,
            )
        return ProviderInvalidRequest(message or "HTTP 400")
    if status == 404:
        # 404 with model-not-found body → invalid_model.
        if error_code == "model_not_found" or _looks_like_model_not_found(error_type):
            return ProviderInvalidModel(message or "model not found")
        return ProviderUnavailable(message or "HTTP 404")
    if status == 429:
        retry_after = parse_retry_after(resp.headers.get("Retry-After"))
        return ProviderRateLimit(message or "HTTP 429", retry_after=retry_after)
    if status == 503:
        # 503 with model-loading body → not_loaded; otherwise
        # generic unavailability.
        if error_type == "model_not_loaded" or _looks_like_model_not_loaded(message):
            return ProviderModelNotLoaded(message or "model not loaded")
        return ProviderUnavailable(message or "HTTP 503")
    if 500 <= status < 600:
        return ProviderUnavailable(message or f"HTTP {status}")
    return ProviderUnavailable(message or f"HTTP {status}")


# Known OpenAI error codes for content-block rejections. Used by
# classify_http_error's 400 branch to route to
# ProviderUnsupportedContentBlock instead of ProviderInvalidRequest.
# The list is best-effort and evolves as OpenAI's error-code surface
# shifts; the substring fallback below catches near-misses.
_CONTENT_REJECTION_ERROR_CODES = frozenset(
    {
        "image_content_not_supported",
        "unsupported_image_media_type",
        "audio_content_not_supported",
        "video_content_not_supported",
        "unsupported_content_block",
    }
)


def _looks_like_content_rejection(
    error_code: object,
    error_type: object,
    message: str | None,
) -> bool:
    """Heuristic for HTTP 400 bodies that indicate the bound model
    rejected a content block (image / audio / video / unsupported
    media_type). Used to route to provider_unsupported_content_block
    rather than the generic provider_invalid_request."""
    if isinstance(error_code, str):
        if error_code in _CONTENT_REJECTION_ERROR_CODES:
            return True
        lower_code = error_code.lower()
        for block_type in ("image", "audio", "video"):
            if block_type in lower_code and ("not_supported" in lower_code or "unsupported" in lower_code):
                return True
    if isinstance(error_type, str) and error_type.lower() in {
        "image_parse_error",
        "image_content_not_supported",
    }:
        return True
    if isinstance(message, str):
        lower_msg = message.lower()
        if "does not support" in lower_msg and (
            "image" in lower_msg or "audio" in lower_msg or "video" in lower_msg
        ):
            return True
    return False


def _extract_rejected_block_type(error_code: object, message: str | None) -> str | None:
    """Pull a best-effort block-type identifier (``"image"`` / ``"audio"``
    / ``"video"``) out of an error code or message, for surfacing on
    ProviderUnsupportedContentBlock.block_type."""
    haystacks: list[str] = []
    if isinstance(error_code, str):
        haystacks.append(error_code.lower())
    if isinstance(message, str):
        haystacks.append(message.lower())
    for haystack in haystacks:
        for block_type in ("image", "audio", "video"):
            if block_type in haystack:
                return block_type
    return None


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value to a float seconds count.

    HTTP allows seconds-int OR HTTP-date; this implementation handles
    the seconds-int form (the OpenAI/vendor norm) and ignores
    HTTP-date.

    Reusable by third-party Provider implementations that need to
    surface ``Retry-After`` to ``ProviderRateLimit.retry_after``.
    """
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _looks_like_model_not_found(error_type: object) -> bool:
    """Heuristic for 404 bodies that indicate model-not-found without
    using the canonical ``model_not_found`` code."""
    if not isinstance(error_type, str):
        return False
    return "model" in error_type.lower() and "found" in error_type.lower()


def _looks_like_model_not_loaded(message: object) -> bool:
    """Heuristic for 503 messages that indicate the model is
    configured but not loaded."""
    if not isinstance(message, str):
        return False
    lower = message.lower()
    return "not loaded" in lower or "loading" in lower


# ---------------------------------------------------------------------------
# Observability §5.5 LLM provider span event helpers
# ---------------------------------------------------------------------------


# Inline image sources are redacted in this step per observability
# §5.5.5: ImageSourceInline → {"type": "inline_redacted",
# "byte_count": N} where N is the byte length of the original base64
# string. media_type stays at the image-block level per llm-provider
# §3.1.2; detail is preserved when present.
#
# Redaction lives here (provider-side) rather than observer-side so
# inline image bytes never leave the provider in event form —
# defense-in-depth that applies to every observer consuming the
# payload, not just OA's own. URL-form images pass through unchanged.
def _serialize_messages_for_payload(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Render a list of typed :class:`Message` instances into the
    plain-dict shape carried on ``LlmEventPayload.input_messages``."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                out.append({"role": "user", "content": msg.content})
            else:
                rendered_blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        rendered_blocks.append({"type": "text", "text": block.text})
                    else:  # ImageBlock
                        # The ImageBlock validator already guarantees
                        # media_type when source is inline.
                        if isinstance(block.source, ImageSourceInline):
                            byte_count = len(block.source.base64_data)
                            source_record: dict[str, Any] = {
                                "type": "inline_redacted",
                                "byte_count": byte_count,
                            }
                        else:
                            source_record = {"type": "url", "url": block.source.url}
                        image_record: dict[str, Any] = {
                            "type": "image",
                            "source": source_record,
                        }
                        if block.media_type is not None:
                            image_record["media_type"] = block.media_type
                        if block.detail is not None:
                            image_record["detail"] = block.detail
                        rendered_blocks.append(image_record)
                out.append({"role": "user", "content": rendered_blocks})
        elif isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls
                ]
            out.append(entry)
        else:  # ToolMessage
            out.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
    return out


# Only set fields appear in the result. Absence is meaningful per
# observability §5.5.2: "the field was not supplied for this call"
# — distinct from "supplied with a zero value."
def _request_params_from_config(config: RuntimeConfig | None) -> dict[str, Any]:
    """Extract the cross-vendor request parameters from a
    ``RuntimeConfig`` for emission as ``gen_ai.request.*`` attributes."""
    if config is None:
        return {}
    out: dict[str, Any] = {}
    if config.temperature is not None:
        out["temperature"] = config.temperature
    if config.max_tokens is not None:
        out["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        out["top_p"] = config.top_p
    if config.seed is not None:
        out["seed"] = config.seed
    # Three fields promoted in proposal 0032; surfaced under their
    # cross-vendor declared names so the observer emits
    # gen_ai.request.{frequency_penalty,presence_penalty,stop_sequences}.
    if config.frequency_penalty is not None:
        out["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        out["presence_penalty"] = config.presence_penalty
    if config.stop_sequences is not None:
        out["stop_sequences"] = config.stop_sequences
    return out


def _request_extras_from_config(config: RuntimeConfig | None) -> dict[str, Any]:
    """Return the ``RuntimeConfig`` extras pass-through bag as a plain
    dict; empty when no extras are set or when ``config`` is None."""
    if config is None:
        return {}
    return dict(config.model_extra or {})


# call_id MUST be the same string on the started/completed pair so
# the observer can match them under concurrency. The OTel observer
# (or any backend mapping) recognises the sentinel node_name +
# namespace and emits an LLM-specific span instead of a node span;
# backend-specific attribute extraction reads payload fields from
# pre_state directly.
def _make_llm_event(
    phase: Literal["started", "completed"],
    *,
    call_id: str,
    model: str,
    genai_system: str,
    finish_reason: FinishReason | None = None,
    usage: Usage | None = None,
    error: BaseException | None = None,
    input_messages: list[dict[str, Any]] | None = None,
    output_content: str | None = None,
    request_params: dict[str, Any] | None = None,
    request_extras: dict[str, Any] | None = None,
    response_id: str | None = None,
    response_model: str | None = None,
    active_prompt: Any = None,
    active_prompt_group: Any = None,
) -> NodeEvent:
    """Build a ``NodeEvent``-shaped record for the engine's delivery
    queue, populated as an ``openarmature.llm.complete`` event."""
    error_type: str | None = None
    error_message: str | None = None
    error_category: str | None = None
    if error is not None:
        error_type = type(error).__name__
        error_message = str(error)
        category = getattr(error, "category", None)
        if isinstance(category, str):
            error_category = category
    payload = LlmEventPayload(
        call_id=call_id,
        model=model,
        finish_reason=finish_reason,
        prompt_tokens=usage.prompt_tokens if usage is not None else None,
        completion_tokens=usage.completion_tokens if usage is not None else None,
        total_tokens=usage.total_tokens if usage is not None else None,
        error_type=error_type,
        error_message=error_message,
        error_category=error_category,
        calling_namespace_prefix=current_namespace_prefix(),
        calling_attempt_index=current_attempt_index(),
        calling_fan_out_index=current_fan_out_index(),
        active_prompt=active_prompt,
        active_prompt_group=active_prompt_group,
        input_messages=input_messages,
        output_content=output_content,
        request_params=request_params,
        request_extras=request_extras,
        response_id=response_id,
        response_model=response_model,
        genai_system=genai_system,
    )
    return NodeEvent(
        node_name="openarmature.llm.complete",
        namespace=("openarmature.llm.complete",),
        step=-1,
        phase=phase,
        pre_state=payload,
        post_state=None,
        error=None,
        parent_states=(),
    )


__all__ = [
    "OpenAIProvider",
    "classify_http_error",
    "parse_retry_after",
]
