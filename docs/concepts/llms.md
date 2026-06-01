# LLMs

The graph engine has no concept of LLMs or tools. A node is just an
async function that reads typed state and returns a partial update.
Calling an LLM is one of the things a node can do during that call, the
same way it might read a file, hit a database, or invoke an internal
service. This page covers the patterns that emerge once you start
mixing LLM calls into graph nodes.

## LLM calls are async IO inside a node

Construct one [`Provider`](../reference/llm.md) when your application
owns its lifecycle (entry-point coroutine, FastAPI startup event,
lazy on-first-use) and share it across nodes. Each `complete()` call
carries the full message list and returns a
[`Response`](../reference/llm.md); the provider is stateless and
reentrant, so multiple nodes (or fan-out instances) can call into it
concurrently without coordination.

`OpenAIProvider` eagerly opens an `httpx.AsyncClient` in its
constructor; that client must be closed with `await provider.aclose()`
to release the connection pool. Constructing the provider as a
module-level side effect (`provider = OpenAIProvider(...)` at the top
of the file) leaks the client in tooling, tests, and docs-build
processes that import the module without running your shutdown path.
Prefer lazy construction or an explicit lifecycle hook.

```python
import os
from openarmature.llm import OpenAIProvider, UserMessage

_provider_instance: OpenAIProvider | None = None


def _get_provider() -> OpenAIProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OpenAIProvider(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
            model="gpt-4o-mini",
            api_key=os.environ["LLM_API_KEY"],
        )
    return _provider_instance


async def analyze(state: AnalysisState) -> dict:
    response = await _get_provider().complete(
        [UserMessage(content=state.text)],
    )
    return {"raw": response.message.content}


async def main() -> None:
    try:
        ...  # build graph, invoke
    finally:
        if _provider_instance is not None:
            await _provider_instance.aclose()
```

The provider goes wherever your application's other long-lived
dependencies go (dependency-injection container, factory, lazy
module-level cache), and you close it on the same lifecycle hook you
use for those. A FastAPI app uses `app.on_event("shutdown")`; a
script uses a `try/finally` around the entry-point coroutine.

A real graph hits LLMs from multiple nodes. The conventional shape:

```python
async def classify(state):    # one provider call
    response = await provider.complete(...)
    return {...}

async def extract(state):     # another provider call
    response = await provider.complete(...)
    return {...}

async def synthesize(state):  # a third
    response = await provider.complete(...)
    return {...}
```

The graph composes the order; the provider sees three independent
stateless calls. Conversational memory (if you want it) is the
caller's responsibility: thread it through state and pass the
accumulated message list into each call.

## Pre-flight readiness check

`Provider.ready()` is the optional pre-flight call you make before
your application starts taking real traffic. It raises one of the
canonical [`LlmProviderError`](../reference/llm.md) categories on
failure and returns `None` on success, so a typical startup hook
looks like:

```python
async def startup() -> None:
    provider = _get_provider()
    try:
        await provider.ready()
    except ProviderAuthentication:
        # Bad API key — fail fast at boot.
        raise
    except ProviderInvalidModel:
        # Bound model isn't served by this endpoint — same.
        raise
    except ProviderUnavailable:
        # Endpoint is down or unreachable — fail fast too.
        raise
```

`OpenAIProvider` ships three probe shapes selected via the
`readiness_probe` constructor kwarg:

- **`"chat_completions"`** (default) — issues `POST /v1/chat/completions`
  with a `max_tokens=1` body. Actually exercises the inference wire
  path. Strongest signal at the cost of one prompt's worth of tokens
  on cloud endpoints.
- **`"models"`** — issues `GET /v1/models` and verifies the bound
  model appears in the catalog. Cheaper (no completion billing) but
  blind to proxy wire-mismatch cases: some OpenAI-compatible proxies
  (Bifrost is the motivating example) serve `/v1/models` correctly
  while 405'ing the completions endpoint, so a green catalog probe
  doesn't prove `complete()` will work.
- **`"both"`** — runs the catalog probe first (cheap fail-fast on
  model-not-in-catalog with the cleaner `seen_ids` diagnostic), then
  the chat probe. Strongest signal at double the round-trip cost.

```python
# Local server (LM Studio, vLLM, llama.cpp) — chat probe is free.
provider = OpenAIProvider(
    base_url="http://localhost:8000",
    model="qwen2.5-coder",
    readiness_probe="chat_completions",  # default
)

# Cloud endpoint, cost-sensitive — opt back into the catalog-only probe.
provider = OpenAIProvider(
    base_url="https://api.openai.com",
    model="gpt-4o-mini",
    api_key=os.environ["LLM_API_KEY"],
    readiness_probe="models",
)
```

The chat probe is the default because the catalog probe's
false-green failure mode (Bifrost-style proxy mismatch) is silent at
boot but fatal at first real call, and that's worse than the extra
token spend for the small set of cost-sensitive callers who can opt
out explicitly.

## Structured output

Every LLM-using node that produces typed data ends up with the same
shape: render a prompt, call the model, parse the response as JSON,
validate it against the expected schema, retry on parse or validation
failure. Five steps of boilerplate that differ only in the schema and
the prompt.

Structured output collapses that into one parameter. Pass a
`response_schema` to `complete()` and the provider:

1. Tells the model on the wire to produce schema-conforming output.
2. Parses and validates the response against the schema.
3. Surfaces the validated value on `Response.parsed`.
4. Raises `StructuredOutputInvalid` on parse or validation failure.

Two forms are accepted: a Pydantic class (typed-instance return) and a
JSON Schema dict (raw-dict return). Same wire shape underneath.

### Pydantic class form

```python
from typing import Literal

from pydantic import BaseModel

class Classification(BaseModel):
    intent: Literal["research", "summarize"]
    rationale: str


async def classify(state):
    response = await provider.complete(
        [UserMessage(content=f"Route this query: {state.query!r}")],
        response_schema=Classification,
    )
    return {"classification": response.parsed}
```

`Response.parsed` is a validated `Classification` instance at
runtime; the framework calls `.model_json_schema()` under the hood
to derive the wire body and `.model_validate()` to deserialize the
response.

Static typing is shallower. `Response.parsed` is annotated as
`dict[str, Any] | BaseModel | None`, so a type checker won't narrow
to `Classification` from the `response_schema=Classification`
argument alone. Callers that want static field access either
`cast(Classification, response.parsed)`, narrow with `isinstance`,
or assign the value into a typed local. Generic `Response[T]` is on
the table as a follow-up.

### JSON Schema dict form

```python
async def research(state):
    response = await provider.complete(
        [UserMessage(content=f"Plan research: {state.query!r}")],
        response_schema={
            "type": "object",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}},
                "follow_up_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topics", "follow_up_questions"],
            "additionalProperties": False,
        },
    )
    return {"research_plan": response.parsed}
```

`Response.parsed` is a `dict[str, Any]` populated per the schema. Use
this when the shape is dynamic, generated, or borrowed from another
system that already speaks JSON Schema.

### Wire paths: native and fallback

Real `OpenAIProvider` traffic uses OpenAI's native `response_format`
field on the request body, so the model produces schema-conforming
output in one trip. Some OpenAI-compatible servers (older vLLM, some
LM Studio releases, llama.cpp variants) either reject `response_format`
with a 400 or silently ignore it. For those, construct the provider
with `force_prompt_augmentation_fallback=True`:

```python
provider = OpenAIProvider(
    base_url="http://localhost:8000",
    model="some-local-model",
    force_prompt_augmentation_fallback=True,   # opt into the fallback
)
```

In fallback mode the provider prepends a system directive containing
the serialized schema, omits `response_format` from the wire, and
parses-and-validates the response post-receive. The behavioral contract
is identical: `Response.parsed` populates the same way; failures raise
`StructuredOutputInvalid` the same way. The
`uses_prompt_augmentation_fallback` read-only property lets callers
inspect which path is active.

### Strict mode

OpenAI's native path supports a `strict: true` flag that engages the
model's schema-constrained decoding: the model literally cannot emit
non-conforming tokens. The framework decides `strict: true` vs
`strict: false` automatically based on whether your schema satisfies
strict-mode constraints. Either way, the framework validates the
response post-receive against the supplied schema; strict is a
wire-level optimization, not a correctness requirement.

`strict_mode_supported(schema)` (exported from `openarmature.llm`)
performs the deep recursive check. The heuristic is conservative:
anything not on the list below trips to `strict: false`:

- Top-level schema is `type: "object"`.
- For every nested object: `additionalProperties` is **explicitly**
  `false`, and every key in `properties` is listed in `required`.
- For every nested array: `items` is present and points to a
  verifiable schema (dict, or tuple-form list of dicts).
- Every branch of `anyOf` / `oneOf` / `allOf` independently satisfies
  the above.
- Internal `$ref` targets (`#/...` or bare `#`) resolve and their
  resolved schema passes. External refs (any other URI) and `$ref`
  cycles are handled conservatively.
- Primitive types (`string`, `integer`, `number`, `boolean`, `null`)
  are accepted as terminal: no nested structure to verify.
- Empty `{}` schemas and unrecognized-keyword schemas (`const`-only,
  `enum`-only, etc.) trip to non-strict; the walker can't statically
  verify them.

If you control the schema and want strict mode, the easiest path is to
set `additionalProperties: false` and put every property in `required`
on every object. Pydantic-derived schemas may need `model_config =
ConfigDict(extra="forbid")` on the class to get the
`additionalProperties: false` in the generated JSON Schema.

## Tool calling

Beyond producing typed text, an LLM call can request work from local
Python functions and resume with their results. The wire shape is a
turn-based loop driven entirely from the same `complete()` call: the
model emits `tool_calls`, the caller dispatches them to local
functions, appends `ToolMessage` responses, and re-calls. The graph
engine has no special concept of tools; the loop fits as a
conditional-edge cycle.

```python
from openarmature.llm import Tool

lookup_mission = Tool(
    name="lookup_mission",
    description="Look up factual records for a named lunar mission.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
)

response = await provider.complete(messages, tools=[lookup_mission, ...])
```

When the model decides to use one or more tools, the response carries
`finish_reason="tool_calls"` and `response.message.tool_calls` is a
list of `ToolCall(id, name, arguments)` records. `arguments` is a
parsed dict whose shape matches the corresponding tool's `parameters`
schema. The single edge case where `arguments` is `None` is
`finish_reason="error"` for unparseable model output.

The caller dispatches each call to its local function, appends one
`ToolMessage(content=..., tool_call_id=...)` per call to the message
list, and re-calls. The `tool_call_id` field MUST match the
`ToolCall.id` the model emitted so the model can pair its requests
with the responses. The next turn either emits more `tool_calls` or
returns a normal assistant content message signaling completion.

Wiring the loop as a graph cycle: a `call_llm` node, a
`dispatch_tools` node that resolves calls and appends
`ToolMessage`s, a conditional edge from `call_llm` that routes back
to `call_llm` when `tool_calls` are present and forward to a
termination node when they aren't. A turn cap on the routing function
prevents runaway loops on a model that stays in tool-calling forever.
See [`09 - Tool use`](../examples/09-tool-use.md) for the runnable
shape.

### Controlling tool-call behavior with `tool_choice`

By default the model decides whether and which tools to call.
`tool_choice` constrains that decision per call. Four modes:

- `"auto"` — the model decides. Equivalent to omitting the parameter
  when `tools` is non-empty.
- `"required"` — the model MUST call at least one tool. Useful for
  routing nodes that branch on tool selection.
- `"none"` — the model MUST NOT call tools, even if `tools` is
  supplied. Useful for guarded LLM calls or for explicitly disabling
  tool-calling without rebuilding a tools-less request.
- `ForceTool(name=...)` — the model MUST call the named tool exactly.

Pre-send validation catches the three failure modes (`required` with
empty tools, `ForceTool` with empty tools, `ForceTool.name` not in
the supplied list) and raises `ProviderInvalidRequest` before the
HTTP request is sent.

```python
from openarmature.llm import ForceTool

# Routing node: model MUST pick one of the supplied tools.
response = await provider.complete(
    messages, tools=[search, summarize], tool_choice="required"
)

# Forced specific tool: useful when the pipeline knows which tool
# the model should call next (e.g., a `dispatch_search` node).
response = await provider.complete(
    messages, tools=[search, summarize], tool_choice=ForceTool(name="search")
)
```

Not all providers honor `tool_choice` — confirm with your provider's
documentation. The `OpenAIProvider` maps the spec shape onto OpenAI's
wire shape per the §8.1.1 mapping table. Whether the model actually
honored the constraint is observable from the returned
`finish_reason` and `tool_calls` fields; the framework does NOT
re-validate the response against the constraint.

## Content blocks (multimodal user messages)

User messages carry content in one of two shapes: a plain text string,
or an ordered sequence of typed content blocks. The string form is the
common case. Blocks are how you mix non-text modalities into a single
turn. v1 defines two block types: text and image. Audio and video are
deferred to future proposals.

System, assistant, and tool messages stay text-string only. Image
inputs are user-only in v1; image outputs (assistant-message-borne
images, e.g. DALL-E-style generation) are out of scope.

### Text and image blocks

A text block is the array-form equivalent of a text-string message:
`TextBlock(text="describe this")`. A user message holding a single
text block is normatively equivalent to one with `content="describe
this"`.

An image block carries one source (URL or inline base64) plus an
optional `detail` hint:

```python
from openarmature.llm import (
    ImageBlock,
    ImageSourceInline,
    ImageSourceURL,
    OpenAIProvider,
    TextBlock,
    UserMessage,
)


async def describe_image(provider: OpenAIProvider) -> str:
    response = await provider.complete(
        [
            UserMessage(
                content=[
                    ImageBlock(
                        source=ImageSourceURL(url="https://example.com/diagram.png"),
                        detail="high",  # optional; omitted from wire when None
                    ),
                    TextBlock(text="What does this diagram show?"),
                ]
            )
        ]
    )
    return response.message.content
```

Block order is preserved on the wire. Providers vary in whether they
treat order as semantically meaningful (an image followed by its
describing text is a different signal from text followed by the
image); construct the sequence in the order you want the model to
perceive it.

### URL vs inline sources

- **URL source** (`ImageSourceURL`): the provider fetches the URL. Any
  scheme the provider documents support for is valid (`http(s)://`,
  `data:`, etc.). The framework passes it through unchanged.
- **Inline source** (`ImageSourceInline`): the image is sent as
  base64-encoded bytes in the request body. The `media_type` field on
  the surrounding `ImageBlock` is **required** for inline sources (and
  ignored for URL sources). The framework constructs an RFC 2397
  `data:<media_type>;base64,<bytes>` URI for the wire; it does not
  inspect, transcode, or re-encode the bytes.

OpenAI, Anthropic, and Google all accept `image/png`, `image/jpeg`,
and `image/webp` as guaranteed media types. `media_type` is typed as
`str | None`, so callers MAY pass additional `image/*` types when
they know the bound model supports them; portable code sticks to the
three.

### The `detail` hint

`detail` is a per-image hint to the provider about processing
fidelity: `"auto"`, `"low"`, or `"high"`. The class default is `None`,
which **omits the field from the wire** and lets the provider apply
its own default (conceptually `"auto"`). Setting `detail="auto"`
explicitly on the spec block forces the wire to carry an explicit
`"auto"`, usually unnecessary since the provider's default is the
same value.

### When the model can't handle the block

`provider_unsupported_content_block` raises when the bound model
rejects a content block type or media variant. Concrete cases:

- A text-only model (e.g., `gpt-3.5-turbo`) received an image block.
- The model supports images but not the requested `media_type`.
- The model supports the type but rejected the specific source variant
  (a URL the provider can't fetch, for example).

The error category is **non-transient**: retrying without changing
the request, the bound model, or the provider won't succeed. Userland
fallback patterns (e.g., a middleware that routes to a multimodal
provider on this category) compose cleanly against it.

`ProviderUnsupportedContentBlock` carries `block_type` ("image",
"audio", "video") and `reason` (the provider's human-readable
message) when those are recoverable from the rejection.

`OpenAIProvider` detects content rejection via the response body:
HTTP 400 with an error code like `image_content_not_supported` or a
message like "does not support image inputs." Pre-send capability
checks (failing fast before the wire trip when you know the model
doesn't support images) live above the provider as userland
middleware; the provider doesn't ship a static model-capability
catalog.

## Routing on parsed fields

A conditional edge is a function `state -> str` that names the next
node. The string can come from anywhere: a hard-coded rule, a lookup
table, the parsed output of an LLM call. The graph engine doesn't
distinguish.

This means LLM-driven routing and deterministic routing have the same
shape. A classifier node writes its parsed `Classification` to state;
the conditional edge reads `state.classification.intent` and returns
that string. The branches don't know whether the LLM or a regex
produced the discriminator.

```python
async def classify(state):
    response = await provider.complete(
        [UserMessage(content=f"Route: {state.query!r}")],
        response_schema=Classification,
    )
    return {"classification": response.parsed}


def route(state) -> str:
    return state.classification.intent


builder.add_conditional_edge("classify", route)
```

The same `route` function could read a feature flag, a config lookup,
or `"research" if "?" in state.query else "summarize"`. The branch
nodes don't change. Swapping a rule-based router for an LLM-based one
is a one-node change.

## Errors at the LLM boundary

Every provider call can fail. The
[`openarmature.llm` reference](../reference/llm.md) lists the canonical
error categories; this section covers how they compose with the rest
of the graph.

**Transient categories** (retry MAY succeed):
`ProviderRateLimit`, `ProviderUnavailable`, `ProviderModelNotLoaded`.
These are the canonical "wrap a node in `RetryMiddleware`" set; the
default classifier picks them up automatically via
`TRANSIENT_CATEGORIES`.

**Non-transient categories** (retry without changing the request will
not succeed): `ProviderAuthentication`, `ProviderInvalidModel`,
`ProviderInvalidRequest`, `ProviderInvalidResponse`,
`StructuredOutputInvalid`. These propagate up as `NodeException` so
the graph's error-recovery middleware (or the caller of `invoke()`)
can handle them.

`StructuredOutputInvalid` is the new one and worth a note. It fires
when a model returns content that fails to parse as JSON, or parses
but fails to validate against the supplied schema. The exception
carries the requested `response_schema`, the `raw_content` the model
produced, and a `failure_description`. It is non-transient by default
because a model that emits non-conforming output on a given prompt
usually emits the same non-conforming output on retry. Useful retry
strategies for this case involve changing the prompt or doubling
`max_tokens` rather than re-issuing the same call; that's a
middleware concern, not the provider's default.

```python
from openarmature.llm import StructuredOutputInvalid

async def classify_with_diagnostics(state):
    try:
        response = await provider.complete(
            [UserMessage(content=...)],
            response_schema=Classification,
        )
    except StructuredOutputInvalid as exc:
        log.warning(
            "schema-validation failure on classify",
            extra={
                "raw_content": exc.raw_content,
                "failure": exc.failure_description,
            },
        )
        raise
    return {"classification": response.parsed}
```

Callers wanting to retry validation failures specifically can
construct a `RetryMiddleware` with a custom classifier that adds
`structured_output_invalid` to the transient set. The default
classifier won't do this for them.

## Where to next

- [Model Providers](../model-providers/index.md) for the provider
  contract, the shipped `OpenAIProvider`, and the canonical error
  categories.
- [Authoring a Provider](../model-providers/authoring.md) for writing
  a provider against a non-OpenAI wire format (Anthropic Messages,
  Bedrock, internal gateway).
- [API reference: `openarmature.llm`](../reference/llm.md) for the
  full surface: message types, `Response`, `RuntimeConfig`, every
  error class, validation helpers.
- [Examples: 00 - Hello, world](../examples/00-hello-world.md) for a
  runnable graph exercising both `response_schema` forms in one
  pipeline.
- [Examples: 09 - Tool use](../examples/09-tool-use.md) for the
  agent-loop pattern with two local tools.
- [Examples: 07 - Multimodal prompt](../examples/07-multimodal-prompt.md)
  for content blocks alongside versioned prompts.
