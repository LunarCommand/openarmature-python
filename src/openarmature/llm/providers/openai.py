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

**``ready()`` probe.** Three modes selected by the constructor kwarg
``readiness_probe``:

- ``"chat_completions"`` (default) — issues ``POST /v1/chat/completions``
  with a minimal ``max_tokens=1`` body. Actually exercises the inference
  path, so OpenAI-compatible proxies (Bifrost, custom gateways) that
  return 200 on ``GET /v1/models`` but reject ``POST /v1/chat/completions``
  surface immediately rather than at first real call.
- ``"models"`` — hits ``GET /v1/models`` and verifies the bound model
  appears in the returned catalog. Cheaper (no completion tokens
  billed) but doesn't catch the wire-mismatch case above.
- ``"both"`` — runs the catalog probe first, then the chat probe.
  Catalog check short-circuits with the cleaner "model not in catalog"
  diagnostic before any billable call.

All three modes map non-200 responses through ``classify_http_error``
so the canonical error categories (``provider_authentication``,
``provider_unavailable``, ``provider_invalid_model``,
``provider_model_not_loaded``, ``provider_rate_limit``) surface
consistently regardless of which probe ran.

The previous default was ``"models"``; flipped to ``"chat_completions"``
because the catalog probe missed a real failure class (proxy wire-
format mismatch) in field use.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from openarmature._managed_extras import ManagedArm, apply_managed_extras
from openarmature._usage import nonneg_int

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
    _assistant_text,
)
from ..provider import strict_mode_supported
from ..response import FinishReason, ParsedValue, Response, RuntimeConfig, Usage
from ._base import _ProviderBase, _StructuredPlan
from ._common import (
    _augment_messages_with_schema_directive,
    _canonicalize_dict_keys,
    _parse_and_validate,
    parse_retry_after,
)

# Runtime guard for ``OpenAIProvider(..., readiness_probe=...)``. The
# Literal type narrows callers under static checkers but is not enforced
# at runtime, so an unknown string would silently no-op both dispatch
# branches in ``ready()`` and return None — a false-green readiness
# signal. Validate in ``__init__`` against this set instead.

_VALID_READINESS_PROBES = frozenset({"models", "chat_completions", "both"})

# §8.1 managed wire fields and their collision arms (0105 + 0108).
#
# The structural keys (model / messages / tools / tool_choice) are ALWAYS
# managed: 0105 §3.5 gives them no "while producing it" qualifier, unlike the
# conditionally-managed response_format / stream_options. So an extras key of a
# structural name is rejected even where the call produced no such field, which
# is what stops an extras `tools` on a no-tools call smuggling an unvalidated
# tool array past validate_tools.
#
# The rest are managed only WHEN the mapping produced them, which the call site
# enforces by filtering to keys present in the built body. A sampling field left
# None, or response_format off the structured-output path, is unmanaged and an
# extra of that name rides untouched. `stop` realizes the list-shaped
# stop_sequences, so it MERGES rather than collides.
_OPENAI_MANAGED_ARMS: dict[str, ManagedArm] = {
    "model": "reject",
    "messages": "reject",
    "tools": "reject",
    "tool_choice": "reject",
    "temperature": "reject",
    "top_p": "reject",
    "max_tokens": "reject",
    "seed": "reject",
    "frequency_penalty": "reject",
    "presence_penalty": "reject",
    "response_format": "reject",
    "stop": "merge",
}

# The structural managed-internal keys (0105 §3.5), managed regardless of whether
# the mapping produced them on this call. tools / tool_choice are absent from the
# body on a no-tools call, but an extras key of either name must still reject
# rather than inject; model / messages are always in the body.
_OPENAI_STRUCTURAL_KEYS = frozenset({"model", "messages", "tools", "tool_choice"})


@dataclass(frozen=True)
class _OpenAIPlan(_StructuredPlan):
    include_response_format: bool
    strict: bool


class OpenAIProvider(_ProviderBase[_OpenAIPlan]):
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

    _completions_path = "/v1/chat/completions"
    _logger_name = __name__

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
        readiness_probe: Literal["models", "chat_completions", "both"] = "chat_completions",
        populate_caller_metadata: bool = True,
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
        # ``readiness_probe`` selects which wire path ``ready()`` exercises.
        # The default ``"chat_completions"`` actually tests inference; the
        # opt-in ``"models"`` is the older catalog-only probe for
        # cost-sensitive cloud callers (every chat probe bills prompt
        # tokens). ``"both"`` runs catalog then chat for the strongest
        # signal at double the round-trip cost. Same explicit-opt-in
        # rationale as ``genai_system``: no base_url sniffing, since the
        # right probe shape depends on what's on the other end and a
        # wrong inference is worse than a wrong default.
        if readiness_probe not in _VALID_READINESS_PROBES:
            raise ValueError(
                f"readiness_probe must be one of {sorted(_VALID_READINESS_PROBES)} (got {readiness_probe!r})"
            )
        self._readiness_probe = readiness_probe
        # Proposal 0049's caller_invocation_metadata field is OPTIONAL
        # on the typed LlmCompletionEvent. The python implementation
        # defaults the opt-in to True because the bundled OTel and
        # Langfuse observers read the field to populate caller-metadata
        # span attributes (§5.6); leaving it off by default would
        # silently strip those attributes after the typed-event
        # migration. Pass ``populate_caller_metadata=False`` to suppress
        # the snapshot when no downstream consumer needs it.
        self._populate_caller_metadata = populate_caller_metadata
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

    # ------------------------------------------------------------------
    # ready() — pre-flight probe
    # ------------------------------------------------------------------

    async def ready(self) -> None:
        """Verify the bound model is reachable. Dispatches on the
        ``readiness_probe`` mode chosen at construction:

        - ``"chat_completions"`` (default) issues a ``max_tokens=1``
          chat call against ``POST /v1/chat/completions``.
        - ``"models"`` issues ``GET /v1/models`` and matches
          ``self.model`` against the returned ``data[].id`` entries.
        - ``"both"`` runs the catalog probe first (cheaper, surfaces
          model-not-in-catalog with the catalog diagnostic), then the
          chat probe.
        """
        if self._readiness_probe in ("models", "both"):
            await self._probe_models()
        if self._readiness_probe in ("chat_completions", "both"):
            await self._probe_chat_completions()

    async def _probe_models(self) -> None:
        """Catalog probe — ``GET /v1/models`` + bound-model presence
        check. Cheaper than the chat probe (no completion tokens
        billed) and surfaces the model-not-in-catalog case with the
        cleaner ``seen_ids`` diagnostic; misses wire-format mismatches
        on proxies that serve the catalog correctly but reject
        completions."""
        try:
            resp = await self._client.get("/v1/models")
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc

        if resp.status_code != 200:
            raise classify_http_error(resp)

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

    async def _probe_chat_completions(self) -> None:
        """Inference probe — ``POST /v1/chat/completions`` with a
        ``max_tokens=1`` body. Surfaces wire-format mismatches that
        the catalog probe can't see (the motivating case: Bifrost-
        style proxies that 200 on ``/v1/models`` but 405/404 on
        ``/v1/chat/completions``). Bills one prompt's worth of tokens
        on cloud endpoints, which is why this defaults on but is
        opt-out via ``readiness_probe="models"``."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
        }
        try:
            resp = await self._client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        if resp.status_code != 200:
            raise classify_http_error(resp)
        # Validate the response shape so a proxy answering 200 with an
        # error payload or non-OpenAI-shape JSON doesn't pass the probe.
        # Mirrors ``_do_complete``'s parse step. The returned Response
        # is discarded — the validation itself is the point.
        try:
            payload_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("POST /v1/chat/completions returned non-JSON body") from exc
        if not isinstance(payload_raw, dict):
            raise ProviderInvalidResponse("POST /v1/chat/completions returned a non-object body")
        self._parse_response(cast("dict[str, Any]", payload_raw), None, None)

    # ------------------------------------------------------------------
    # The mapping's parts of complete()
    # ------------------------------------------------------------------

    def _plan_structured_output(
        self,
        messages: Sequence[Message],
        schema_dict: dict[str, Any] | None,
        *,
        tools: Sequence[Tool] | None,
        tool_choice: ToolChoice | None,
    ) -> _OpenAIPlan:
        fallback = self._force_prompt_augmentation_fallback
        # On the fallback path, the wire-side messages list is an
        # augmented COPY of the caller's messages — original
        # messages MUST NOT be mutated.
        wire_messages: Sequence[Message] = messages
        if schema_dict is not None and fallback:
            wire_messages = _augment_messages_with_schema_directive(messages, schema_dict)
        # §8.1.5: the strict flag the request carries decides the path the
        # call reports, so the body and the path both read this one value.
        # Only the native path sends the flag.
        strict = schema_dict is not None and not fallback and strict_mode_supported(schema_dict)
        return _OpenAIPlan(
            messages=wire_messages,
            path=_structured_output_path(schema_dict, fallback=fallback, strict=strict),
            # The fallback only governs structured-output calls; free-form
            # calls (schema_dict is None) must preserve any caller-supplied
            # response_format from RuntimeConfig extras.
            include_response_format=schema_dict is None or not fallback,
            strict=strict,
        )

    def _stop_sequence_sent(self, body: Mapping[str, Any]) -> bool:
        return _carries_stop_sequence(body.get("stop"))

    def _classify_http_error(self, resp: httpx.Response) -> LlmProviderError:
        return classify_http_error(resp)

    # ------------------------------------------------------------------
    # Request building (spec §8.1.1)
    # ------------------------------------------------------------------

    def _build_request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None,
        config: RuntimeConfig | None,
        schema_dict: dict[str, Any] | None,
        *,
        plan: _OpenAIPlan | None = None,
        tool_choice: ToolChoice | None = None,
        collision_hint: Callable[[str], str | None] | None = None,
    ) -> dict[str, Any]:
        include_response_format = plan is None or plan.include_response_format
        strict = plan is not None and plan.strict
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
        # response_format is omitted entirely on the fallback path —
        # the schema travels in the augmented system message instead. It is
        # set BEFORE the extras reconciliation so it is a managed value the
        # collision rule can guard on the structured path.
        if schema_dict is not None and include_response_format:
            # Spec 0047 §8.1.5 / Q5 ack: response_format.json_schema.schema
            # is a user-supplied JSON Schema and flows through the same
            # canonicalization path as tool.parameters.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _derive_schema_name(schema_dict),
                    "schema": _canonicalize_dict_keys(schema_dict),
                    "strict": strict,
                },
            }
        # On the free-form call and the §8.1.5.1 prompt-augmentation fallback
        # path the mapping produces no response_format, so it is UNMANAGED there
        # (0105 §3.5): a caller's extras response_format rides untouched via the
        # reconciliation below rather than being stripped.
        # §8.1.1: map the spec `tool_choice` onto the OpenAI wire shape.
        # ``None`` omits the field so OpenAI's own default applies. String
        # modes pass through; ``ForceTool`` renames ``type: "tool"`` to
        # ``"function"`` and nests the name under a ``function`` object.
        if tool_choice is not None:
            if isinstance(tool_choice, ForceTool):
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice.name},
                }
            else:
                body["tool_choice"] = tool_choice
        # Managed-field collision (proposals 0105 + 0108, §6). Reconcile the
        # caller's extras against the managed wire fields the mapping just
        # produced: a colliding managed scalar/object is a matching no-op or a
        # conflicting reject (provider_invalid_request), stop merges, and every
        # unmanaged extra keeps §6's untouched pass-through. Canonicalize each
        # extra value first (0047 §8) so a dict-typed extra compares against the
        # already-canonical managed value on the reject arm and renders with
        # stable key ordering on pass-through. The structural keys (0105 §3.5)
        # are managed unconditionally; every other key is managed only when the
        # mapping actually produced it (present in the body), which realizes the
        # "while producing it" semantics for the declared-field realizations and
        # the conditionally-managed response_format.
        if config is not None and config.extras:
            extras = {k: _canonicalize_dict_keys(v) for k, v in config.extras.items()}
            managed: dict[str, ManagedArm] = {
                key: arm
                for key, arm in _OPENAI_MANAGED_ARMS.items()
                if key in body or key in _OPENAI_STRUCTURAL_KEYS
            }
            apply_managed_extras(body, extras, managed, collision_hint=collision_hint)
        # Spec 0047 §8 belt-and-suspenders: walk the assembled body
        # once more sorting any dict at every nesting level, in case
        # a future code path introduces a user-input boundary the
        # per-field canonicalization above doesn't cover. Cheap (the
        # body is small) and explicit.
        return _canonicalize_dict_keys(body)

    # ------------------------------------------------------------------
    # Response parsing (spec §8.1.2)
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        payload: dict[str, Any],
        schema_dict: dict[str, Any] | None,
        schema_class: type[BaseModel] | None,
        *,
        plan: _OpenAIPlan | None = None,
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
        finish_reason_typed = cast(FinishReason, finish_reason)

        # Build the assistant message. Tool calls under
        # finish_reason="error" may carry malformed argument JSON per
        # §3 — surface as `arguments=None` rather than raising.
        try:
            # §8.1.2: a non-empty refusal on a message with no tool calls is the
            # message's content, and the finish_reason is "content_filter" whatever
            # OpenAI reported. OpenAI sends a null or empty refusal on ordinary
            # messages, and that changes nothing. The message is copied rather than
            # changed in place because `raw` keeps the response as received, and the
            # check sits in this try because a malformed message is no dict.
            refusal = wire_msg.get("refusal")
            if isinstance(refusal, str) and refusal and not wire_msg.get("tool_calls"):
                wire_msg = {**wire_msg, "content": refusal}
                finish_reason_typed = "content_filter"
            assistant_msg = _wire_to_assistant_message(
                wire_msg, lenient_args=(finish_reason_typed == "error")
            )
        except ProviderInvalidResponse:
            raise
        except Exception as exc:
            raise ProviderInvalidResponse(f"could not parse assistant message: {exc}") from exc

        # Usage is optional -- build a Usage with all-None fields if the
        # provider didn't report it. Per spec §6 / proposal 0101 a malformed
        # counter (present on the wire but not a non-negative int) is treated
        # as not reported: nonneg_int nulls it rather than raising, the
        # completion succeeded and the verbatim value survives on ``raw``. When
        # every counter is malformed the record is {null, null, null} (§6
        # null-together); the typed event mirrors it (it reads response.usage).
        usage_wire_raw = payload.get("usage")
        if isinstance(usage_wire_raw, dict):
            usage_wire = cast("dict[str, Any]", usage_wire_raw)
            # cached_tokens sources from
            # usage.prompt_tokens_details.cached_tokens per spec §8.1.2; vLLM
            # and other OpenAI-compatible servers that surface implicit-cache
            # stats follow the same nesting. Defaults to None when
            # prompt_tokens_details is absent or when the nested cached_tokens
            # key is missing (preserves the absent-vs-reported-zero
            # distinction). cache_creation_tokens stays None -- OpenAI-compatible
            # providers do not report a discrete cache-creation count here.
            prompt_tokens_details_raw = usage_wire.get("prompt_tokens_details")
            prompt_tokens_details: dict[str, Any] | None = (
                cast("dict[str, Any]", prompt_tokens_details_raw)
                if isinstance(prompt_tokens_details_raw, dict)
                else None
            )
            # Conditional set: only pass cached_tokens when the wire reports the
            # nested key, so pydantic tracks it as "set" only then and
            # model_dump(exclude_unset=True) omits it when the provider didn't
            # report it. A reported-but-malformed cached_tokens is set to None
            # (reported as null), distinct from absent (not in the kwargs).
            usage_kwargs: dict[str, Any] = {
                "prompt_tokens": nonneg_int(
                    usage_wire.get("prompt_tokens"), field="prompt_tokens", source="llm provider"
                ),
                "completion_tokens": nonneg_int(
                    usage_wire.get("completion_tokens"), field="completion_tokens", source="llm provider"
                ),
                "total_tokens": nonneg_int(
                    usage_wire.get("total_tokens"), field="total_tokens", source="llm provider"
                ),
            }
            if prompt_tokens_details is not None and "cached_tokens" in prompt_tokens_details:
                usage_kwargs["cached_tokens"] = nonneg_int(
                    prompt_tokens_details["cached_tokens"], field="cached_tokens", source="llm provider"
                )
            usage = Usage(**usage_kwargs)
        else:
            usage = Usage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )

        # gen_ai.response.id / gen_ai.response.model semconv (spec
        # §5.5.3) read these off the Response. The wire fields are
        # optional — providers MAY omit either or both. ``None`` when
        # absent or not a string. Parsed before structured-output
        # validation so a structured_output_invalid failure can carry the
        # response identity (proposal 0082).
        response_id_raw = payload.get("id")
        response_id: str | None = response_id_raw if isinstance(response_id_raw, str) else None
        response_model_raw = payload.get("model")
        response_model: str | None = response_model_raw if isinstance(response_model_raw, str) else None

        # Structured-output parsing. parsed is absent when no schema
        # was requested AND when the response is a tool-call response
        # — the tool-call path and structured-content path are
        # mutually exclusive at the response level.
        parsed: ParsedValue = None
        if schema_dict is not None and finish_reason_typed != "tool_calls":
            try:
                parsed = _parse_and_validate(
                    _assistant_text(assistant_msg.content), schema_dict, schema_class
                )
            except StructuredOutputInvalid as exc:
                # The wire response is intact; attach its response-side
                # context so the failed event / §7 error carries finish_reason
                # (truncation triage), usage, and the response identity
                # (proposal 0082). output_content is already set by the helper.
                exc.finish_reason = finish_reason_typed
                exc.usage = usage
                exc.response_id = response_id
                exc.response_model = response_model
                raise

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
    # ``rstrip`` on the full URL is a no-op when a query string or
    # fragment follows the path (e.g., ``https://host/v1/?token=...``
    # ends in ``c`` so the URL-level rstrip leaves the parsed path's
    # trailing slash intact). Strip the parsed path itself so the
    # suffix check catches those shapes too.
    path = urlparse(normalized).path.rstrip("/")
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


# §8.1.5: OpenAI documents schema-constrained decoding only for strict: true,
# so a strict request reports "native" and any other native request
# "native_unenforced". The fallback reports "prompt_augmentation" whatever the
# request's extras carry (graph-engine §6).
def _structured_output_path(
    schema_dict: dict[str, Any] | None, *, fallback: bool, strict: bool
) -> str | None:
    if schema_dict is None:
        return None
    if fallback:
        return "prompt_augmentation"
    return "native" if strict else "native_unenforced"


# A matched stop sequence also reports "stop", so §5.5.7 skips an attempt whose
# request carried one. Read from the body as sent, after the extras' `stop` has
# been merged into the declared list or dropped as malformed. OpenAI's `stop` is
# a string or a list, and only an empty list carries none.
def _carries_stop_sequence(stop: Any) -> bool:
    return isinstance(stop, str) or (isinstance(stop, list) and len(cast("list[Any]", stop)) > 0)


def _message_to_wire(msg: Message) -> dict[str, Any]:
    """Request mapping for one message."""
    # Spec llm-provider §8.1.1.
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
        # §8.1.1 strip-on-send: thinking and redacted-thinking blocks have no
        # OpenAI wire form, so they are dropped and the text blocks go out as
        # one string.
        text = _assistant_text(msg.content)
        # Tool-call-only assistants emit ``"content": null`` on the
        # wire — that's the OpenAI convention for "no textual reply,
        # only tool calls." Don't "fix" this to an empty string; the
        # API rejects empty-string content alongside tool_calls. A message
        # the strip left with neither text nor tool calls sends "", since
        # §8.1.1 makes stripping raise no error.
        wire: dict[str, Any] = {"role": "assistant", "content": text or (None if msg.tool_calls else "")}
        if msg.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # Canonical compact form (no inter-token spaces). Matches
                        # the spec's wire-mapping fixture (005, cases shape) and
                        # the form OpenAI itself emits. ``sort_keys=True`` per
                        # spec 0047 §8 — tool-call arguments are a
                        # caller-supplied dict and the JSON-encoded string
                        # MUST be byte-stable across equivalent inputs.
                        "arguments": json.dumps(tc.arguments or {}, separators=(",", ":"), sort_keys=True),
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
    # Per llm-provider §8.1.1, the wire-byte stability rule covers
    # tool DEFINITIONS broadly — not just the parameters subtree.
    # Sort the function record's top-level keys + recursively
    # canonicalize the parameters JSON Schema.
    return {
        "type": "function",
        "function": _canonicalize_dict_keys(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        ),
    }


def _wire_to_assistant_message(wire: dict[str, Any], *, lenient_args: bool) -> AssistantMessage:
    """Parse an OpenAI-shaped assistant message into canonical form.

    When ``lenient_args=True`` (i.e. ``finish_reason == "error"``),
    tool calls with unparseable JSON arguments populate
    ``arguments=None`` instead of raising — degraded responses surface
    what they can; repair is a caller concern.
    """
    # Spec llm-provider §3: validation under finish_reason "error".
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


__all__ = [
    "OpenAIProvider",
    "classify_http_error",
    "parse_retry_after",
]
