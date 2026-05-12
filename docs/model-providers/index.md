# Model Providers

A **Provider** is the seam between OpenArmature's graph engine and
any LLM backend — OpenAI's hosted API, an Anthropic Messages
endpoint, a local vLLM / LM Studio / llama.cpp server, or an
internal gateway. The engine doesn't know about LLMs; nodes call
providers, providers do the wire work.

## What ships

- **`OpenAIProvider`** — implements the OpenAI Chat Completions wire
  format (`POST /v1/chat/completions`). Talks to OpenAI itself plus
  the local servers that adopt the same format (vLLM, LM Studio,
  llama.cpp). One Provider class covers most real-world deployments.

For wire formats that aren't OpenAI Chat Completions (Anthropic
Messages, Bedrock, gateways with custom shapes), write your own.
See [Authoring a Provider](authoring.md).

## The contract

A Provider implements two async methods:

```python
from collections.abc import Sequence
from typing import Protocol

from openarmature.llm import Message, Response, RuntimeConfig, Tool


class Provider(Protocol):
    async def ready(self) -> None: ...
    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        config: RuntimeConfig | None = None,
    ) -> Response: ...
```

- **`ready()`** verifies the bound model is reachable. Pre-flight
  check, typically called once before invoking the graph.
- **`complete()`** performs a single completion call and returns the
  full `Response` — message, finish reason, token usage, raw wire
  payload.

### Behaviour guarantees

- **Stateless.** Every `complete()` call carries the full message
  list. There is no implicit conversation memory.
- **Reentrant.** Safe to call concurrently from many nodes. The
  underlying HTTP client is shared but task-safe.
- **Non-mutating.** Inputs (`messages`, `tools`) are never modified.
- **No tool-call loops.** When the model wants to call a tool, the
  Provider returns with `finish_reason="tool_calls"`. The caller
  executes the tool and makes a follow-on `complete()` with the
  result. The Provider does not re-enter itself.
- **No retry on transient errors.** That's middleware's job — wrap a
  node in `RetryMiddleware` or similar.

## Errors

Seven canonical error categories cover every failure mode:

| Error                       | Trigger                                       |
| --------------------------- | --------------------------------------------- |
| `ProviderAuthentication`    | 401 / 403 — bad key, expired token            |
| `ProviderUnavailable`       | 5xx, network failure, timeout                 |
| `ProviderInvalidModel`      | Bound model doesn't exist on the provider     |
| `ProviderModelNotLoaded`    | Model known but not currently serving         |
| `ProviderRateLimit`         | 429 (with `Retry-After` exposed)              |
| `ProviderInvalidResponse`   | 200 OK that fails to parse                    |
| `ProviderInvalidRequest`    | Malformed request (per-message or list-level) |

Three of these (`Unavailable`, `RateLimit`, `ModelNotLoaded`) are
exported in `TRANSIENT_CATEGORIES` — the canonical "safe to retry"
set used by the default retry-middleware classifier.

## A minimal example

Direct usage of an `OpenAIProvider` against a local server, without
the engine in the picture:

```python
import asyncio

from openarmature.llm import OpenAIProvider, UserMessage


async def main() -> None:
    provider = OpenAIProvider(
        base_url="http://localhost:8000/v1",  # any OpenAI-compatible endpoint
        model="some-model",
        api_key="optional-for-local-servers",
    )
    # await provider.ready()                   # pre-flight; needs a live endpoint
    # response = await provider.complete(
    #     messages=[UserMessage(content="Hello, world!")],
    # )
    # print(response.message.content)


asyncio.run(main())
```

In a real graph you'd construct one Provider at startup and let
nodes call it inside their bodies.

## Where to next

- **[Authoring a Provider](authoring.md)** — how to implement the
  Protocol for a non-default wire format. Includes a ~60-line
  skeleton + contract checklist.
- **[API reference: `openarmature.llm`](../reference/llm.md)** — the
  full public surface (Message types, Response, Usage, RuntimeConfig,
  error classes).
