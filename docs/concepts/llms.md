# LLMs

The graph engine has no concept of LLMs or tools. A node is just an
async function that reads typed state and returns a partial update.
Calling an LLM is one of the things a node can do during that call, the
same way it might read a file, hit a database, or invoke an internal
service. This page covers the patterns that emerge once you start
mixing LLM calls into graph nodes.

## LLM calls are async IO inside a node

Construct one [`Provider`](../reference/llm.md) at startup and share it
across nodes. Each `complete()` call carries the full message list and
returns a [`Response`](../reference/llm.md); the provider is stateless
and reentrant, so multiple nodes (or fan-out instances) can call into
it concurrently without coordination.

```python
import os
from openarmature.llm import OpenAIProvider, UserMessage

provider = OpenAIProvider(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
    model="gpt-4o-mini",
    api_key=os.environ["LLM_API_KEY"],
)


async def analyze(state: AnalysisState) -> dict:
    response = await provider.complete(
        [UserMessage(content=state.text)],
    )
    return {"raw": response.message.content}
```

The provider goes wherever your application's other long-lived
dependencies go: module-level constant, dependency-injection
container, factory function. It does not need to be constructed per
call, and constructing it cheaply (no eager network calls) means
import-time setup is fine.

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
performs the deep recursive check. The heuristic is conservative —
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
- [Examples: `00-hello-world`](https://github.com/LunarCommand/openarmature-python/tree/main/examples/00-hello-world)
  for a runnable graph exercising both `response_schema` forms in one
  pipeline.
