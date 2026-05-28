# Self-hosted vLLM

`OpenAIProvider` talks to any server that implements OpenAI's Chat
Completions wire format (`POST /v1/chat/completions`) — including
self-hosted [vLLM](https://github.com/vllm-project/vllm). This page
walks through the configuration nuances specific to vLLM
deployments.

## The 30-second version

```python
import asyncio

from openarmature.llm import OpenAIProvider, RuntimeConfig, UserMessage


async def main() -> None:
    provider = OpenAIProvider(
        base_url="http://localhost:8000",   # host root only — no /v1
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=None,                       # vLLM doesn't require auth by default
        genai_system="vllm",                # surfaces on observability spans
    )
    messages = [UserMessage(content="hello")]
    config = RuntimeConfig(temperature=0.0, max_tokens=128)
    # await provider.ready()                          # pre-flight; needs a live endpoint
    # response = await provider.complete(messages, config=config)
    # print(response.message.content)
    _ = (messages, config)                            # used once the calls above are uncommented
    await provider.aclose()


asyncio.run(main())
```

That's it for the happy path. The rest of the page covers the
config nuances you'll hit in real deployments.

## `base_url` shape — host root only

vLLM serves on `http://<host>:<port>/v1/chat/completions` and
`http://<host>:<port>/v1/models`. `OpenAIProvider` appends the
`/v1/...` paths itself, so the `base_url` you pass MUST be the host
root — no `/v1` suffix:

```python
from openarmature.llm import OpenAIProvider

# Correct — host root only, provider appends /v1/...
provider = OpenAIProvider(
    base_url="http://localhost:8000",
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=None,
)

# Rejected at construction time — raises ValueError
try:
    OpenAIProvider(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key=None,
    )
except ValueError as exc:
    # "base_url must not end with '/v1' — the provider appends …"
    _ = exc
```

The check exists because httpx joins paths by appending, so an
unprefixed `/v1` on `base_url` produces a doubled `/v1/v1/...` wire
path that silently 404/405s on most backends. The provider rejects
the misconfiguration synchronously rather than letting it surface as
a wire failure at lifespan startup.

Trailing slashes are stripped; other non-empty paths (proxy prefixes
like `/api/openai-proxy`) are left intact for intentional reverse-
proxy setups.

## Authentication — typically off, optionally on

vLLM ships with auth off by default. Pass `api_key=None` for that
case. To enable auth on the vLLM side, launch with `--api-key
<your-key>` and pass the same value to `OpenAIProvider`:

```bash
# vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --api-key sk-vllm-local-secret
```

```python
from openarmature.llm import OpenAIProvider

provider = OpenAIProvider(
    base_url="http://localhost:8000",
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key="sk-vllm-local-secret",
)
```

A wrong or missing key surfaces as `ProviderAuthentication` (mapped
from 401/403) — the same error category as OpenAI cloud auth
failures, so retry / surface logic is portable across backends.

## `genai_system="vllm"` for the observability layer

The `genai_system` constructor kwarg sets the OTel `gen_ai.system`
span attribute. The default `"openai"` is correct for OpenAI's
hosted API; for self-hosted backends, set it to the backend name so
dashboards / cost-attribution UIs filter the traces correctly:

```python
from openarmature.llm import OpenAIProvider

provider = OpenAIProvider(
    base_url="http://localhost:8000",
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=None,
    genai_system="vllm",
)
```

Standard values for other backends running the same wire format:
`"vllm"`, `"lmstudio"`, `"llamacpp"`, `"sglang"`. No `base_url`
sniffing is done — the same host:port could be any of those servers,
and a wrong inference would be worse than the explicit opt-in.

## Older vLLM releases — `force_prompt_augmentation_fallback`

OpenAI's native structured-output path uses the `response_format`
field on the request body. Older vLLM releases either reject this
field or silently ignore it. If you're targeting structured output
on an older vLLM:

```python
from openarmature.llm import OpenAIProvider

provider = OpenAIProvider(
    base_url="http://localhost:8000",
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=None,
    genai_system="vllm",
    force_prompt_augmentation_fallback=True,
)
```

With the fallback flag set, the provider injects a JSON-Schema
directive into the system message instead of using
`response_format`. The wire body never carries `response_format`;
the model sees the schema in the prompt and is asked to produce
conforming JSON. Validation against the schema still runs on the
returned text — `StructuredOutputInvalid` surfaces when the model's
output doesn't match.

Recent vLLM releases (>=0.5.x) support `response_format` natively;
leave the flag at its default `False` for those.

## Readiness probe — `GET /v1/models`

`provider.ready()` hits `GET /v1/models` and:

- Matches the bound model against the returned `data[].id` entries;
  raises `ProviderInvalidModel` if absent.
- Consults an optional per-entry `status` field — if it contains
  `loading` or `not_loaded`, raises `ProviderModelNotLoaded`. Local
  servers that report load state (some LM Studio / vLLM builds) get a
  real not-loaded signal through this path.
- Maps 401/403 → `ProviderAuthentication`, 5xx / connection error →
  `ProviderUnavailable`.

**Limitation for vLLM specifically.** vLLM's `/v1/models` doesn't
populate a `status` field — it returns the configured model with a
200 even during a slow first-load. So the `status`-based not-loaded
detection above doesn't fire for vLLM; the probe confirms the model
name matches but can't tell warmed from cold. For deployments where
cold-load takes seconds to minutes, layer your own warm-up call
after `ready()`:

```python
from openarmature.llm import OpenAIProvider, RuntimeConfig, UserMessage


async def warm_up(provider: OpenAIProvider) -> None:
    await provider.ready()
    # Synthetic warm-up — sends a 1-token request to force the model
    # to finish loading before lifespan startup completes.
    await provider.complete(
        [UserMessage(content="ok")],
        config=RuntimeConfig(temperature=0.0, max_tokens=1),
    )
```

A more discriminating readiness contract is on the roadmap (see
post-release task: harden OpenAIProvider readiness probe).

## Tool calling on vLLM

vLLM supports OpenAI-style tool calling when launched with
`--enable-auto-tool-choice` and a tool-parser flag matching the
model family (e.g., `--tool-call-parser llama3_json` for Llama 3.1
Instruct). The wire shape is identical to OpenAI's; from
`OpenAIProvider`'s perspective, tool calls Just Work. The
[fundamentals → tool calling](../concepts/llms.md#tool-calling) page
covers the OA-side dispatch pattern; no vLLM-specific changes
needed.

```bash
# vLLM server — enable tool calling
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json
```

## Behaviour to be aware of

- **Concurrency**: vLLM batches requests internally. `OpenAIProvider`
  shares one `httpx.AsyncClient` per provider instance; concurrent
  `complete()` calls share the connection pool and round-trip
  through vLLM's batcher.
- **Token counting**: vLLM returns OpenAI-shaped `usage` blocks. OA
  records them on `Response.usage` and surfaces them as
  `gen_ai.usage.*` span attributes per observability §5.5.3.
- **`Retry-After` on 429**: vLLM emits 429 when its scheduler
  queues fill. `ProviderRateLimit.retry_after` is populated from the
  header. Retry middleware handles backoff if wrapped around the
  node.
- **Streaming**: not supported on `provider.complete()` (it's a
  single-completion call by contract). For streaming, write a
  capability against the same wire endpoint.
