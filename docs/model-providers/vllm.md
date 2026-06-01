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
model family. The wire shape is identical to OpenAI's; from
`OpenAIProvider`'s perspective, tool calls Just Work. The
[fundamentals → tool calling](../concepts/llms.md#tool-calling) page
covers the OA-side dispatch pattern; no vLLM-specific changes
needed.

```bash
# vLLM server with tool calling enabled
vllm serve <model-id> \
    --enable-auto-tool-choice \
    --tool-call-parser <parser-name>
```

The `--tool-call-parser` flag MUST match the model family's training
format; mismatches produce assistant messages that vLLM tries to
parse as tool calls and silently returns as content (or vice versa).
Common families:

| Model family                  | `--tool-call-parser` value |
|-------------------------------|----------------------------|
| Llama 3.x Instruct            | `llama3_json`              |
| Llama 4 (Maverick / Scout)    | `llama4_pythonic`          |
| Mistral Instruct families     | `mistral`                  |
| Hermes, Qwen 2.5 tool-use     | `hermes`                   |
| Qwen3 / Qwen3-Coder           | `qwen3_xml`                |
| DeepSeek V3                   | `deepseek_v3`              |
| GPT-OSS (20B / 120B)          | `openai`                   |

Anthropic Claude and Google Gemini models are proprietary cloud APIs,
not open weights; vLLM doesn't serve them, so they don't appear in
this table. Use their first-party endpoints (or an OpenAI-compatible
proxy) and skip the `--tool-call-parser` story entirely.

**Gemma (Google open weights).** Distinct from Gemini, but vLLM does
not currently ship a tool-call parser for the mainstream Gemma 2,
Gemma 3, or CodeGemma variants; tool calling is effectively
unsupported under vLLM for those. The one exception is Google's
specialized FunctionGemma (270M, edge-focused), which has its own
`functiongemma` parser. For general-purpose tool-calling workloads,
pick a model family from the table above rather than Gemma.

**Qwen3-VL specifically.** vLLM's docs don't currently document a
dedicated parser for the Qwen3-VL variants (`Qwen3-VL-30B-A3B`,
`Qwen3-VL-72B`). Check vLLM's release notes for the version you're
pinned to before assuming the Qwen3 row above carries over;
multimodal-instruct variants sometimes ship parser support behind
the text-instruct generation.

See vLLM's
[tool-calling docs](https://docs.vllm.ai/en/latest/features/tool_calling.html)
for the current full list; the set grows release-over-release.

## Production deployment

The 30-second snippet at the top of this page is enough for a local
dev box. Production deployments hit three additional gotchas worth
calling out.

### `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` against `OpenAIProvider`

`OpenAIProvider` keeps one `httpx.AsyncClient` per provider instance
and reuses connections across concurrent `complete()` calls per the
standard httpx pool idiom. vLLM's stock uvicorn keep-alive timeout
is 5 seconds; an idle pooled connection on the OA side can outlive
that window and the next request lands on a half-closed socket. The
visible symptom is `httpcore.RemoteProtocolError: Server
disconnected without sending a response` or
`httpx.RemoteProtocolError`, surfaced through `OpenAIProvider` as
`ProviderUnavailable`.

The fix is to widen vLLM's keep-alive window via the
`VLLM_HTTP_TIMEOUT_KEEP_ALIVE` env var (the value feeds uvicorn's
`timeout_keep_alive`). 300 seconds covers most pool idle windows in
practice:

```bash
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=300 vllm serve <model-id> --host 0.0.0.0 --port 8001
```

Same applies behind a reverse proxy: the proxy's keep-alive window
MUST be at least as wide as vLLM's. Otherwise the proxy closes
connections vLLM still considers alive and the OA-side pool reuses a
dead socket on the next call.

### systemd unit shape

For long-running vLLM workloads, a systemd unit is the canonical
launcher. The structural skeleton:

```ini
# /etc/systemd/system/vllm-<model>.service
[Unit]
Description=vLLM serving <model-id>
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=vllm
WorkingDirectory=/srv/vllm
EnvironmentFile=/etc/vllm/<model>.env
ExecStart=/srv/vllm/.venv/bin/vllm serve <model-id> \
    --host 0.0.0.0 --port 8001 \
    --enable-auto-tool-choice \
    --tool-call-parser <parser-name>
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

The `EnvironmentFile` pattern keeps `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`,
`CUDA_VISIBLE_DEVICES`, `HF_HOME`, and other deploy-specific vars
out of the unit file itself, which makes the unit shippable across
hosts without per-machine edits. `journalctl -u vllm-<model>` is
then the canonical log surface for production triage.

### Throughput knobs and OA concurrency

Three vLLM flags interact directly with how many concurrent
`complete()` calls an OA graph can land before vLLM starts 429-ing:

- `--max-model-len`: per-request context ceiling. Lower values fit
  more concurrent requests in the same KV-cache budget; higher
  values let individual requests carry longer prompts at the cost
  of concurrent capacity.
- `--max-num-seqs`: hard cap on concurrent sequences vLLM will
  schedule. Past this cap, the scheduler queues and (once queue
  fills) returns 429 with `Retry-After`.
- `--gpu-memory-utilization`: fraction of GPU VRAM vLLM may use.
  Higher values widen the KV-cache budget, which lets vLLM schedule
  closer to its `--max-num-seqs` cap before evicting in-flight
  sequences; the cap itself doesn't move. Tune cautiously to avoid
  OOM on the resident model weights.

OA's `OpenAIProvider` shares one connection pool across the whole
graph, so a fan-out with `concurrency=N` lands N simultaneous wire
calls. When `N` exceeds `--max-num-seqs` minus vLLM's other
in-flight traffic, expect `ProviderRateLimit` with
`retry_after` populated; wrap the LLM-calling node in
`RetryMiddleware` (or set `concurrency` explicitly on the fan-out)
to avoid head-of-line stalls.

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
