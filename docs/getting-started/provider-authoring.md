# Authoring a custom Provider

`openarmature` ships an `OpenAIProvider` for any OpenAI Chat
Completions-compatible endpoint (vLLM, LM Studio, llama.cpp, the OpenAI
API itself). If you need to target a different wire format — Anthropic
Messages, Bedrock, an internal gateway, a hand-rolled inference service —
implement the `Provider` Protocol yourself.

This page walks through a minimal skeleton (~60 lines of real code) and the
contract a Provider has to satisfy.

## What you implement

The `Provider` Protocol is two async methods (spec §5):

- `async ready() -> None` — verifies the bound model is reachable. A
  successful return means the next `complete()` shouldn't raise §7
  categories that surface mismatched configuration or unloaded state.
- `async complete(messages, tools=None, config=None) -> Response` —
  performs a single completion. **Stateless** (no per-call state held
  across invocations), **reentrant** (safe to call concurrently), and
  **non-mutating** (`messages` MUST NOT be modified). The Provider does
  not loop on tool calls and does not retry — those are the caller's and
  middleware's jobs respectively.

## Skeleton

A minimal OpenAI-compatible provider targeting any `/v1/chat/completions`
endpoint. Compare with `openarmature.llm.OpenAIProvider` (~465 lines) to
see what a full implementation adds (tool-call wire mapping, observability
spans, the `/v1/models` catalog probe, retry-after parsing, lenient
argument parsing under `finish_reason="error"`, etc.).

```python
from collections.abc import Sequence
from typing import Any

import httpx
from openarmature.llm import (
    AssistantMessage,
    Message,
    ProviderInvalidResponse,
    ProviderUnavailable,
    Response,
    RuntimeConfig,
    SystemMessage,
    Tool,
    ToolMessage,
    Usage,
    UserMessage,
    classify_http_error,
    validate_message_list,
    validate_tools,
)


class MyProvider:
    def __init__(self, *, base_url: str, model: str, api_key: str | None = None) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=60.0
        )
        self.model = model

    async def ready(self) -> None:
        try:
            resp = await self._client.get("/v1/models")
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        if resp.status_code != 200:
            raise classify_http_error(resp)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        config: RuntimeConfig | None = None,
    ) -> Response:
        validate_message_list(messages)
        validate_tools(tools)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_msg_to_wire(m) for m in messages],
        }
        if config and config.temperature is not None:
            body["temperature"] = config.temperature

        try:
            resp = await self._client.post("/v1/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        if resp.status_code != 200:
            raise classify_http_error(resp)

        try:
            payload = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse("non-JSON response body") from exc

        choice = payload["choices"][0]
        wire_msg = choice["message"]
        usage = payload.get("usage", {})

        return Response(
            message=AssistantMessage(content=wire_msg.get("content") or ""),
            finish_reason=choice["finish_reason"],
            usage=Usage(
                # All three fields are required; pass ``None`` when the
                # provider doesn't report usage (spec §6 explicit).
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            raw=payload,
        )


def _msg_to_wire(msg: Message) -> dict[str, Any]:
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, UserMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, AssistantMessage):
        return {"role": "assistant", "content": msg.content or ""}
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id,
        }
    raise ValueError(f"unhandled message type: {type(msg).__name__}")
```

## Contract checklist

When you ship a Provider, the following MUST hold (spec §3, §5, §7):

**Statelessness + reentrancy.**

- [ ] `complete()` MUST NOT carry state across calls. Each call sees the
      full message list; there is no implicit conversation state.
- [ ] Multiple `complete()` calls MAY run concurrently on the same
      Provider instance. The HTTP client should be safe for concurrent
      use (httpx.AsyncClient is).

**Non-mutation.**

- [ ] `messages` passed to `complete()` MUST NOT be mutated. Build wire
      bodies from copies / projections; never modify the input.

**Boundary validation.**

- [ ] Call `validate_message_list(messages)` to enforce spec §3
      list-level invariants (non-empty list; `system` is optional but,
      when present, must be the first message; last must be `user` or
      `tool`; every `tool_call_id` matches an earlier assistant
      `ToolCall.id`).
- [ ] Call `validate_tools(tools)` if tools are accepted (duplicate-name
      check).

**Error mapping (spec §7).**

- [ ] Network failures (connection errors, timeouts) → `ProviderUnavailable`.
- [ ] HTTP 401/403 → `ProviderAuthentication`.
- [ ] HTTP 400 → `ProviderInvalidRequest`.
- [ ] HTTP 404 with model-not-found → `ProviderInvalidModel`; otherwise →
      `ProviderUnavailable`.
- [ ] HTTP 429 → `ProviderRateLimit` with `retry_after` from the header.
- [ ] HTTP 503 with model-loading → `ProviderModelNotLoaded`; otherwise →
      `ProviderUnavailable`.
- [ ] HTTP 5xx (other) → `ProviderUnavailable`.
- [ ] 200 OK that fails to parse into spec §6 shape →
      `ProviderInvalidResponse`.

For OpenAI-compatible endpoints, `classify_http_error` does the whole
non-200 mapping table for you; the skeleton above just delegates.

**Finish reasons (spec §6).**

- [ ] Return one of: `"stop"`, `"length"`, `"tool_calls"`,
      `"content_filter"`, `"error"`. Map the wire format's finish-reason
      vocabulary to these five.

## Beyond the skeleton

The skeleton omits things real providers usually need. Reach for
`openarmature.llm.OpenAIProvider` as a reference when you need any of:

- **Tool calls** — wire-mapping the `tool_calls` array on
  `AssistantMessage` to the provider's expected shape, parsing tool
  results back from `ToolMessage`s.
- **Observability spans** — opt-in `started`/`completed` events around
  the wire call so the OTel observer can build LLM spans (spec
  observability §5.5).
- **Lenient response parsing** under `finish_reason="error"` — degraded
  responses surface what they can; tool-call arguments that fail to
  parse populate `arguments=None` instead of raising.
- **Catalog-aware `ready()`** — `GET /v1/models` plus checking whether
  the bound model is in the returned catalog (and, for local servers
  like LM Studio, whether it's actually loaded).
- **`Retry-After` parsing** — use `parse_retry_after` (re-exported from
  `openarmature.llm`) to populate the `retry_after` field of
  `ProviderRateLimit` from the response header.

When in doubt, the openarmature spec at
[`openarmature-spec`](https://github.com/LunarCommand/openarmature-spec) is
the source of truth. The Python conformance fixtures under
`tests/conformance/test_llm_provider.py` exercise the wire mapping
end-to-end against the spec; a custom Provider passing those fixtures is
correctly implementing the contract.
