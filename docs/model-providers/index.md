# Model Providers

A **Provider** is the seam between OpenArmature's graph engine and
any LLM backend (OpenAI's hosted API, an Anthropic Messages
endpoint, a local vLLM / LM Studio / llama.cpp server, or an
internal gateway). The engine doesn't know about LLMs; nodes call
providers, providers do the wire work.

## What ships

- **`OpenAIProvider`**: implements the OpenAI Chat Completions wire
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
from typing import Any, Protocol

from pydantic import BaseModel
from openarmature.llm import Message, Response, RuntimeConfig, Tool


class Provider(Protocol):
    async def ready(self) -> None: ...
    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        config: RuntimeConfig | None = None,
        response_schema: dict[str, Any] | type[BaseModel] | None = None,
    ) -> Response: ...
```

- **`ready()`** verifies the bound model is reachable. Pre-flight
  check, typically called once before invoking the graph.
- **`complete()`** performs a single completion call and returns the
  full `Response`: message, finish reason, token usage, raw wire
  payload, and (when `response_schema` is supplied) a parsed
  structured value on `Response.parsed`. See
  [Structured output](#structured-output) below.

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
- **No retry on transient errors.** That's middleware's job; wrap a
  node in `RetryMiddleware` or similar.

## Errors

Eight canonical error categories cover every failure mode:

| Error                       | Trigger                                                                |
| --------------------------- | ---------------------------------------------------------------------- |
| `ProviderAuthentication`    | 401 / 403 (bad key, expired token)                                     |
| `ProviderUnavailable`       | 5xx, network failure, timeout                                          |
| `ProviderInvalidModel`      | Bound model doesn't exist on the provider                              |
| `ProviderModelNotLoaded`    | Model known but not currently serving                                  |
| `ProviderRateLimit`         | 429 (with `Retry-After` exposed)                                       |
| `ProviderInvalidResponse`   | 200 OK that fails to parse                                             |
| `ProviderInvalidRequest`    | Malformed request (per-message or list-level)                          |
| `StructuredOutputInvalid`   | Response failed to parse as JSON or failed to validate against schema  |

Three of these (`Unavailable`, `RateLimit`, `ModelNotLoaded`) are
exported in `TRANSIENT_CATEGORIES`, the canonical "safe to retry"
set used by the default retry-middleware classifier.
`StructuredOutputInvalid` is non-transient by default; see
[Structured output](#structured-output) below.

## Structured output

`complete()` accepts an optional `response_schema` argument that
constrains the model's output to a caller-supplied shape. When set, the
provider tells the model on the wire to produce conforming output,
parses and validates the response, and surfaces the validated value on
`Response.parsed`. Parse or validation failures raise
`StructuredOutputInvalid`.

Two `response_schema` forms are accepted: a Pydantic class
(typed-instance return) and a raw JSON Schema dict (dict return). Same
wire shape underneath; pick the form that fits the call site.

```python
from typing import Literal

from pydantic import BaseModel
from openarmature.llm import OpenAIProvider, UserMessage


class Classification(BaseModel):
    intent: Literal["research", "summarize"]
    rationale: str


# Class form: parsed comes back as a Classification instance.
async def classify(provider: OpenAIProvider) -> Classification:
    response = await provider.complete(
        [UserMessage(content="Route: 'what is RAG?'")],
        response_schema=Classification,
    )
    assert isinstance(response.parsed, Classification)
    return response.parsed


# Dict form: parsed comes back as a plain dict.
async def plan_research(provider: OpenAIProvider) -> dict:
    response = await provider.complete(
        [UserMessage(content="Plan research for: 'what is RAG?'")],
        response_schema={
            "type": "object",
            "properties": {"topics": {"type": "array", "items": {"type": "string"}}},
            "required": ["topics"],
            "additionalProperties": False,
        },
    )
    assert isinstance(response.parsed, dict)
    return response.parsed
```

For the rendering of structured output into LLM-using node patterns
(routing on parsed fields, error handling, retry composition), see the
[LLMs concept page](../concepts/llms.md).

### Native and fallback wire paths

`OpenAIProvider` uses OpenAI's native `response_format` field on the
request body by default. Some OpenAI-compatible servers (older vLLM,
some LM Studio releases, llama.cpp variants) either reject
`response_format` or silently ignore it. Construct the provider with
`force_prompt_augmentation_fallback=True` to switch to a
prompt-augmentation path that prepends a system directive with the
serialized schema and parses-and-validates post-receive. The behavioral
contract is identical across both paths; the
`uses_prompt_augmentation_fallback` read-only property lets callers
inspect which path is active.

### Strict mode

OpenAI's native path supports a `strict: true` flag that engages
schema-constrained decoding. The provider passes `strict: true` when
the schema satisfies the strict-mode constraints and `strict: false`
otherwise; the full constraint list lives on the
[LLMs concepts page](../concepts/llms.md#strict-mode).
`strict_mode_supported(schema)` is exported from `openarmature.llm`
for callers wanting to check the heuristic directly. Either way, the
provider validates the response post-receive against the supplied
schema.

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

- **[Authoring a Provider](authoring.md)**: how to implement the
  Protocol for a non-default wire format. Includes a ~60-line
  skeleton + contract checklist.
- **[API reference: `openarmature.llm`](../reference/llm.md)**: the
  full public surface (Message types, Response, Usage, RuntimeConfig,
  error classes).
