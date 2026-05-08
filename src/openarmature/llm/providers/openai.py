"""OpenAI-compatible HTTPX-based provider (spec §8).

Implements the spec's :class:`Provider` Protocol against the OpenAI
Chat Completions wire format (``POST /v1/chat/completions``). The
same wire format is the de facto standard for vLLM, LM Studio,
llama.cpp, and other local LLM servers, so this provider talks to
all of them with the right ``base_url``.

**Error mapping (spec §8.3):**

| OpenAI condition                                  | Spec category                |
|---------------------------------------------------|------------------------------|
| ``ConnectError``/``ConnectTimeout``/``ReadTimeout``/network | provider_unavailable |
| HTTP 401, 403                                     | provider_authentication      |
| HTTP 404 with model-not-found body                | provider_invalid_model       |
| HTTP 503 with model-loading body                  | provider_model_not_loaded    |
| HTTP 429 (with ``Retry-After`` → ``retry_after``) | provider_rate_limit          |
| HTTP 400 (schema violation)                       | provider_invalid_request     |
| HTTP 5xx (other)                                  | provider_unavailable         |
| 200 OK that fails to parse into §6 shape          | provider_invalid_response    |

**``ready()`` probe.** Hits ``GET /v1/models`` and:

- 401/403 → ``provider_authentication``.
- 5xx / connection error → ``provider_unavailable``.
- 200 + bound model in returned list → success.
- 200 + bound model NOT in list → ``provider_invalid_model``.

The spec's ``provider_model_not_loaded`` distinction needs a
server-specific probe (LM Studio's loaded-vs-configured endpoint,
vLLM's health endpoint, llama.cpp's runtime-status endpoint) that
this base provider can't generically emit. Subclasses or
purpose-built local-server provider variants close that gap; the
base ``OpenAIProvider`` documents the limitation here rather than
silently treating "model in catalog" as "model loaded."
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, cast

import httpx
from pydantic import ValidationError

from openarmature.graph.events import NodeEvent
from openarmature.observability.correlation import current_dispatch

from ..errors import (
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderModelNotLoaded,
    ProviderRateLimit,
    ProviderUnavailable,
)
from ..messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    Tool,
    ToolCall,
    UserMessage,
)
from ..provider import validate_message_list, validate_tools
from ..response import FinishReason, Response, RuntimeConfig, Usage


class OpenAIProvider:
    """OpenAI Chat Completions wire-compatible provider.

    Construct with a base URL, model identifier, and optional API key
    + transport (an :class:`httpx.AsyncBaseTransport`). The
    ``transport`` parameter is the test seam — ``httpx.MockTransport``
    drives the conformance fixtures by intercepting HTTP calls and
    returning canned responses, exercising the same wire-mapping
    code production traffic would.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
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

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Optional — async clients
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
    ) -> Response:
        """Single completion call per spec §5.

        Pre-send validation runs first (per-message Pydantic + list-
        level invariants per §3 "Validation timing"). HTTP errors
        map to §7 categories per §8.3. The successful 200 body is
        parsed into a :class:`Response` per §8.2 — failure to parse
        raises ``provider_invalid_response``.
        """
        validate_message_list(messages)
        validate_tools(tools)
        body = self._build_request_body(messages, tools, config)

        # Spec observability §5.5 LLM provider span: when an
        # observability backend is active in the current invocation,
        # emit a started/completed event pair around the wire call so
        # the backend can build a span. Queue-mediated dispatch
        # preserves spec §6 serial event ordering across all event
        # sources within an invocation. ``current_dispatch()`` returns
        # ``None`` outside an openarmature invocation (direct
        # provider use in scripts/tests), in which case the call
        # proceeds without span emission.
        dispatch = current_dispatch()
        if dispatch is not None:
            dispatch(_make_llm_event("started", model=self.model))

        try:
            response = await self._do_complete(body)
        except Exception as exc:
            if dispatch is not None:
                dispatch(_make_llm_event("completed", model=self.model, error=exc))
            raise

        if dispatch is not None:
            dispatch(
                _make_llm_event(
                    "completed",
                    model=self.model,
                    finish_reason=response.finish_reason,
                    usage=response.usage,
                )
            )
        return response

    async def _do_complete(self, body: dict[str, Any]) -> Response:
        """Wire-call helper: separated from ``complete()`` so the
        LLM-provider span hook in ``complete()`` can wrap success and
        failure paths uniformly."""
        try:
            resp = await self._client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc

        if resp.status_code != 200:
            raise self._classify_http_error(resp)

        try:
            payload_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("POST /v1/chat/completions returned non-JSON body") from exc
        if not isinstance(payload_raw, dict):
            raise ProviderInvalidResponse("POST /v1/chat/completions returned a non-object body")
        return self._parse_response(cast("dict[str, Any]", payload_raw))

    # ------------------------------------------------------------------
    # Request building (spec §8.1)
    # ------------------------------------------------------------------

    def _build_request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None,
        config: RuntimeConfig | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_wire(m) for m in messages],
        }
        if tools:
            body["tools"] = [_tool_to_wire(t) for t in tools]
        if config is not None:
            if config.temperature is not None:
                body["temperature"] = config.temperature
            if config.max_tokens is not None:
                body["max_tokens"] = config.max_tokens
            if config.top_p is not None:
                body["top_p"] = config.top_p
            if config.seed is not None:
                body["seed"] = config.seed
            # Pass-through any provider-specific extras (extra="allow"
            # on RuntimeConfig); spec §6 permits implementations to
            # accept additional fields.
            extras = config.model_extra or {}
            for k, v in extras.items():
                body.setdefault(k, v)
        return body

    # ------------------------------------------------------------------
    # Response parsing (spec §8.2)
    # ------------------------------------------------------------------

    def _parse_response(self, payload: dict[str, Any]) -> Response:
        try:
            choices = cast("list[dict[str, Any]]", payload["choices"])
            choice = choices[0]
            wire_msg = cast("dict[str, Any]", choice["message"])
            finish_reason_raw = choice["finish_reason"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderInvalidResponse(f"response missing required fields: {exc}") from exc
        finish_reason: str = finish_reason_raw if isinstance(finish_reason_raw, str) else "error"

        # Per §8.2 (and conformance fixture 005's
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

        return Response(
            message=assistant_msg,
            finish_reason=finish_reason_typed,
            usage=usage,
            raw=payload,
        )

    # ------------------------------------------------------------------
    # Error classification (spec §8.3)
    # ------------------------------------------------------------------

    def _classify_http_error(self, resp: httpx.Response) -> Exception:
        """Map a non-200 ``httpx.Response`` to the right §7 category.
        Returns the exception (not raises) so the caller can ``raise``
        with consistent traceback context."""
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
            return ProviderInvalidRequest(message or "HTTP 400")
        if status == 404:
            # 404 with model-not-found body → invalid_model.
            if error_code == "model_not_found" or _looks_like_model_not_found(error_type):
                return ProviderInvalidModel(message or "model not found")
            return ProviderUnavailable(message or "HTTP 404")
        if status == 429:
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
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


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def _message_to_wire(msg: Message) -> dict[str, Any]:
    """Spec §8.1 request mapping for one message."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, UserMessage):
        return {"role": "user", "content": msg.content}
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
                        "arguments": json.dumps(tc.arguments or {}),
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


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value to a float seconds count.
    HTTP allows seconds-int OR HTTP-date; this implementation handles
    the seconds-int form (the OpenAI/vendor norm) and ignores
    HTTP-date."""
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


def _make_llm_event(
    phase: Literal["started", "completed"],
    *,
    model: str,
    finish_reason: FinishReason | None = None,
    usage: Usage | None = None,
    error: BaseException | None = None,
) -> NodeEvent:
    """Build a NodeEvent-shaped record for the engine's delivery
    queue. The OTel observer (or any backend mapping) recognises the
    sentinel ``node_name`` and ``namespace`` and emits an LLM-specific
    span instead of a node span. Backend-specific attribute extraction
    reads ``model``, ``finish_reason``, and ``usage`` from
    ``pre_state``'s ``llm_event`` payload.

    The pre_state field is reused as the carrier for LLM event detail
    because NodeEvent's shape is fixed (graph-engine §6) and adding
    ad-hoc fields would break observers that pattern-match on the
    existing shape. Backend mappings know to inspect
    ``event.pre_state['llm_event']`` when the namespace is
    ``("openarmature.llm.complete",)``.
    """
    payload: dict[str, Any] = {"model": model}
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    if usage is not None:
        payload["prompt_tokens"] = usage.prompt_tokens
        payload["completion_tokens"] = usage.completion_tokens
        payload["total_tokens"] = usage.total_tokens
    if error is not None:
        # The engine's NodeEvent.error type is RuntimeGraphError, but
        # llm-provider errors aren't graph-engine §4 categories. Carry
        # the exception detail in the payload instead so backends can
        # surface it without our needing to wrap as RuntimeGraphError.
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
        category = getattr(error, "category", None)
        if isinstance(category, str):
            payload["error_category"] = category
    return NodeEvent(
        node_name="openarmature.llm.complete",
        namespace=("openarmature.llm.complete",),
        step=-1,
        phase=phase,
        # ``pre_state`` is overloaded here as the LLM-event payload
        # carrier — see the docstring above.
        pre_state=cast("Any", {"llm_event": payload}),
        post_state=None,
        error=None,
        parent_states=(),
    )


__all__ = [
    "OpenAIProvider",
]
