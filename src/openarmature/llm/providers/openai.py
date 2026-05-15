# Spec: realizes llm-provider §8 (concrete OpenAI provider) including
# the §8.3 wire-error mapping table.

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

import httpx
import jsonschema
from pydantic import BaseModel, ValidationError

from openarmature.graph.events import NodeEvent
from openarmature.graph.state import State
from openarmature.observability.correlation import (
    current_attempt_index,
    current_dispatch,
    current_fan_out_index,
    current_namespace_prefix,
)

from ..errors import (
    LlmProviderError,
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderModelNotLoaded,
    ProviderRateLimit,
    ProviderUnavailable,
    StructuredOutputInvalid,
)
from ..messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    Tool,
    ToolCall,
    UserMessage,
)
from ..provider import (
    strict_mode_supported,
    validate_message_list,
    validate_response_schema,
    validate_tools,
)
from ..response import FinishReason, ParsedValue, Response, RuntimeConfig, Usage


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
        force_prompt_augmentation_fallback: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # ``force_prompt_augmentation_fallback`` switches structured-output
        # calls from the native response_format wire path to the
        # prompt-augmentation fallback. Used for older OpenAI-compatible
        # servers (some vLLM/LM Studio/llama.cpp versions) that reject
        # or silently ignore response_format.
        self._force_prompt_augmentation_fallback = force_prompt_augmentation_fallback
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
        response_schema: dict[str, Any] | type[BaseModel] | None = None,
    ) -> Response:
        """Single completion call.

        Pre-send validation runs first (per-message Pydantic +
        list-level invariants + response_schema shape check). HTTP
        errors map to canonical provider-error categories. The
        successful 200 body is parsed into a :class:`Response` —
        failure to parse raises ``provider_invalid_response``; failure
        to validate the response content against ``response_schema``
        raises ``structured_output_invalid``.

        When ``response_schema`` is supplied as a Pydantic BaseModel
        subclass, ``Response.parsed`` is a validated instance of that
        class; when supplied as a JSON Schema dict,
        ``Response.parsed`` is the deserialized dict.
        """
        validate_message_list(messages)
        validate_tools(tools)
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
        if dispatch is not None:
            dispatch(_make_llm_event("started", call_id=call_id, model=self.model))

        try:
            response = await self._do_complete(body, schema_dict, schema_class)
        except Exception as exc:
            if dispatch is not None:
                dispatch(_make_llm_event("completed", call_id=call_id, model=self.model, error=exc))
            raise

        if dispatch is not None:
            dispatch(
                _make_llm_event(
                    "completed",
                    call_id=call_id,
                    model=self.model,
                    finish_reason=response.finish_reason,
                    usage=response.usage,
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
    # Request building (spec §8.1)
    # ------------------------------------------------------------------

    def _build_request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None,
        config: RuntimeConfig | None,
        schema_dict: dict[str, Any] | None,
        include_response_format: bool = True,
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
            # On the fallback path the §8.5.1 contract is "response_format
            # MUST NOT be on the wire." RuntimeConfig is extra="allow" so
            # a caller could pass response_format through via the extras
            # loop above; strip it here so the fallback contract holds
            # regardless of caller-supplied extras.
            body.pop("response_format", None)
        return body

    # ------------------------------------------------------------------
    # Response parsing (spec §8.2)
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

        # Structured-output parsing. parsed is absent when no schema
        # was requested AND when the response is a tool-call response
        # — the tool-call path and structured-content path are
        # mutually exclusive at the response level.
        parsed: ParsedValue = None
        if schema_dict is not None and finish_reason_typed != "tool_calls":
            parsed = _parse_and_validate(assistant_msg.content, schema_dict, schema_class)

        return Response(
            message=assistant_msg,
            finish_reason=finish_reason_typed,
            usage=usage,
            raw=payload,
            parsed=parsed,
        )


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
    etc.) — the wire shape is stable across these and the helper
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


class _LlmEventState(State):
    """Typed payload for LLM-provider span events. Subclasses
    :class:`openarmature.graph.state.State` so the
    ``NodeEvent.pre_state: State`` contract holds — observers
    calling ``event.pre_state.model_dump()`` (or any other
    Pydantic-on-State method) work without the raw-dict overload
    that previously violated the schema.

    Backend mappings (the OTel observer in this repo, future
    Langfuse / Datadog adapters) recognize the
    ``("openarmature.llm.complete",)`` namespace sentinel and read
    these fields directly via attribute access.

    ``call_id`` is the per-call disambiguator: a UUIDv4 minted in
    ``OpenAIProvider.complete`` and shared between the started /
    completed event pair. Backend observers key their in-flight
    LLM-span maps by it so concurrent ``complete()`` calls (e.g.,
    fan-out instances each calling the provider) don't collide on
    a single sentinel-namespace key.

    ``calling_namespace_prefix``, ``calling_attempt_index``, and
    ``calling_fan_out_index`` carry the calling node's identity so
    the OTel observer can resolve the §5.5 "parent under calling
    node" contract correctly under concurrent fan-out and retry.
    Populated from the engine's ContextVars (set in
    ``_step_*_node`` around node-body execution); fall back to
    sentinel defaults (empty tuple, 0, ``None``) when the LLM
    provider is called outside any node body.
    """

    call_id: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # On error responses the provider caller doesn't have a
    # graph-engine §4 ``RuntimeGraphError`` to put in
    # ``NodeEvent.error``, so we surface the failure detail through
    # these fields instead. ``error_category`` is the canonical §7
    # llm-provider category (``provider_unavailable``, etc.) when
    # the failed exception carries one.
    error_type: str | None = None
    error_message: str | None = None
    error_category: str | None = None
    # Calling-node identity captured at dispatch time. The OTel
    # observer reads these to look up the calling node's span in
    # its (now-invocation_id-scoped) ``_open_spans`` map without relying on
    # the OTel current-span context (which under concurrent fan-out
    # can yield a sibling instance's span).
    calling_namespace_prefix: tuple[str, ...] = ()
    calling_attempt_index: int = 0
    calling_fan_out_index: int | None = None


def _make_llm_event(
    phase: Literal["started", "completed"],
    *,
    call_id: str,
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
    ``pre_state`` directly via attribute access.

    ``call_id`` MUST be the same string on the started/completed
    pair so the observer can match them under concurrency.
    """
    error_type: str | None = None
    error_message: str | None = None
    error_category: str | None = None
    if error is not None:
        error_type = type(error).__name__
        error_message = str(error)
        category = getattr(error, "category", None)
        if isinstance(category, str):
            error_category = category
    payload = _LlmEventState(
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
