# Prompts

Named, versioned, content-addressed prompts. OpenArmature's
prompt-management capability separates *fetching* a template
from *rendering* it, lets you compose multiple backends with
explicit fallback, and propagates prompt identity to your
observability backend so trace UIs can pivot on the prompt
that produced a call.

Skip ahead to [a minimal example](#a-minimal-example) if you
want code first.

## The two halves: fetch and render

A `PromptBackend` knows how to find a template by `name` and
`label`; nothing more. A `PromptManager` composes one or more
backends and adds rendering on top:

```python
from openarmature.prompts import PromptManager, FilesystemPromptBackend

manager = PromptManager(FilesystemPromptBackend("./prompts"))

# Fetch returns a Prompt (the raw template + identity metadata).
prompt = await manager.fetch("greeting", "production")

# Render applies variables and returns a PromptResult (the
# rendered messages plus a content-addressed identity).
result = manager.render(prompt, {"user": "Alice"})

# Or do both in one shot:
result = await manager.get("greeting", "production", {"user": "Alice"})
```

Why two operations instead of one? Three reasons:

- **Inspect templates without binding variables.** Schema
  validation, prompt diffing, tooling that walks the prompt
  catalogue.
- **Cache templates separately from rendered output.** The
  fetch step is the I/O step; rendering is pure local
  computation.
- **Render the same template with different variables in
  tight loops.** Map-reduce over chunks, batch evaluation,
  fan-out fixtures.

The convenience `get()` operation gives you the single-call
shape when you want it without removing the separability.

## Refreshing cached prompts: `cache_ttl_seconds`

`fetch` and `get` take an optional `cache_ttl_seconds` that controls how
fresh a served prompt must be, for backends that maintain a client-side
cache:

- omitted / `None` keeps the backend's current behavior;
- `0` forces a fresh read past any cache;
- `N > 0` serves a cached entry only while it is younger than N seconds,
  re-reading the source once it ages past N.

A negative value is rejected. It is a read-side control: it governs which
cached entry may be served for this fetch, not whether or how results are
cached. Cacheless backends (the bundled filesystem backend) ignore it; the
bundled Langfuse backend forwards it to the Langfuse SDK's own prompt cache.

```python
# Always re-read from the backend, bypassing any cache:
fresh = await manager.fetch("greeting", "production", cache_ttl_seconds=0)

# Serve a cached entry only if it's under five minutes old:
recent = await manager.get(
    "greeting", "production", {"user": "Alice"}, cache_ttl_seconds=300
)
```

### A service-wide default

If most fetches want the same freshness bound, set it once at construction
with `default_cache_ttl_seconds` instead of passing `cache_ttl_seconds` on
every call:

```python
# `backend` is any backend with a client-side cache (e.g. the Langfuse backend);
# a cacheless backend ignores the TTL regardless of where it comes from.
manager = PromptManager(backend, default_cache_ttl_seconds=60)

# Uses the default (60s); no per-call value needed:
prompt = await manager.fetch("greeting", "production")

# A per-call value always wins, so this still force-refreshes:
fresh = await manager.fetch("greeting", "production", cache_ttl_seconds=0)
```

Resolution follows a precedence chain: an explicit per-call value (including
`0`) wins; otherwise the manager default applies; otherwise nothing is
forwarded and the backend's own caching governs. A negative default is
rejected at construction. Once a default is set, an omitted per-call value
resolves to it, so there is no per-call way to defer to the backend's own
behavior for a single fetch while a default is configured. Configure no
default, or pass an explicit value, if you need that.

## Prompt identity

Every `Prompt` carries five identity fields:

- `name`: your stable identifier (`"greeting"`).
- `version`: the backend's version string. Implementation-defined:
  a backend MAY use semver, monotonic integers, content
  hashes, git short-SHAs, or any stable identifier. The
  filesystem backend derives it from the template content
  hash.
- `label`: the slot the prompt was fetched from
  (`"production"`, `"latest"`, `"variant-a"`). The label is
  part of the query.
- `template_hash`: SHA-256 of the raw template source.
  Two prompts with different content always have different
  hashes.
- `fetched_at`: when the prompt was fetched. Cached
  backends preserve the original fetch time, not the
  cache-hit time.

The `name + version + label` triple identifies the prompt;
the `template_hash` lets you tell two prompts apart by
*content*, which matters when a vendor backend serves
different content under the same `latest` label over time.

A `PromptResult` propagates all of those, plus:

- `rendered_hash`: SHA-256 over the rendered messages.
  Same template + same variables → same hash. This is the
  cache-key value a memoization layer wants.
- `messages`: the rendered output as an LLM-ready
  `list[Message]`. Directly consumable by
  `Provider.complete()`.
- `variables`: what was applied. Audit-trail friendly.
- `rendered_at`: when the render happened. Distinct from
  `fetched_at`.

## Strict variables by default

A template that references a variable not in the mapping
raises `PromptRenderError`:

```python
prompt = await manager.fetch("greeting", "production")  # "Hello, {{ user }}! Today is {{ day }}."
manager.render(prompt, {"user": "Alice"})  # raises: "day" is undefined
```

This is intentional. Silently substituting empty strings for
missing variables masks bugs: a typo'd variable name produces
a working-but-wrong prompt, often invisibly. If you need
lenient behavior, wrap your variables in your own defaulting
layer before passing them to `render()`.

The Python implementation uses Jinja2's `StrictUndefined`. To opt
out, pass a different `Undefined` subclass at `PromptManager`
construction:

```python
import jinja2

manager = PromptManager(backend, jinja_undefined=jinja2.Undefined)
```

`jinja2.Undefined` renders a missing variable as the empty string;
`jinja2.ChainableUndefined` is the other common opt-out for
templates that walk nested attributes. Reach for these only when the
strict default is actively wrong for your workflow.

## Two variants: text and chat

`Prompt` is a discriminated union over `TextPrompt` and `ChatPrompt`:

- A `TextPrompt` carries a single `template: str` and renders to
  exactly one `UserMessage`. This is the simpler variant and the
  default for the filesystem backend; reach for it when the prompt
  is a single user instruction and you don't need role tagging.
- A `ChatPrompt` carries `chat_template: list[ChatSegment]`. Each
  segment is either a `ContentSegment` (a role-tagged content
  block: `system`, `user`, or `assistant`, carrying a text
  template OR a list of content-block templates for multimodal
  user messages) or a `PlaceholderSegment` (a slot the caller fills
  at render time with a `list[Message]`, useful for chat history
  injection).

`PromptManager.render(prompt, variables, placeholders=...)`
dispatches on the variant. For `TextPrompt` the `placeholders`
kwarg is ignored. For `ChatPrompt` each content segment renders
with the strict-undefined rule applied independently; placeholder
segments inject their caller-supplied message lists in order.

Backends can return either variant: the `LangfusePromptBackend`
maps Langfuse text prompts to `TextPrompt` and Langfuse chat
prompts to `ChatPrompt` with one `ContentSegment` per Langfuse
chat message. Discriminate at the call site with
`isinstance(prompt, ChatPrompt)` when you need variant-specific
behavior; most callers just pass the prompt back into `render()`.

## Per-prompt sampling parameters

A `Prompt` carries an optional `sampling` field: a `SamplingConfig`
sub-record mirroring `RuntimeConfig`'s seven declared fields
(`temperature`, `max_tokens`, `top_p`, `seed`, `frequency_penalty`,
`presence_penalty`, `stop_sequences`) plus the extras pass-through
bag. Backends that source per-prompt config (Langfuse's
`prompt.config`, a filesystem sidecar) populate it; backends that
don't leave it `None`.

```python
prompt = await manager.fetch("classify", "production")
if prompt.sampling is not None:
    response = await provider.complete(messages, config=prompt.sampling)
else:
    response = await provider.complete(messages)
```

`SamplingConfig` is a subclass of `RuntimeConfig`, so it splats
directly into `provider.complete()` without translation.
`PromptResult.sampling` carries the value verbatim from the source
`Prompt`; rendering doesn't touch it.

The `FilesystemPromptBackend` reads sidecar config when constructed
with `sampling_source="per-prompt-sidecar"` (reading
`<root>/<label>/<name>.config.json` next to each template) or
`sampling_source="unified"` (reading `<root>/prompt_configs.json`
once at construction, keyed by prompt name).

## Deployment-time label routing with `LabelResolver`

`PromptManager.fetch(name)` without an explicit `label` consults a
configured `LabelResolver` and falls back to `"production"`. This
lets one prompt be A/B-tested or canaried without code changes:
edit the resolver's data, not the call sites.

```python
from openarmature.prompts import MappingLabelResolver, PromptManager

resolver = MappingLabelResolver({
    "default": "production",
    "experimental_classifier": "staging",
    "extract_claims": "variant-a",
})
manager = PromptManager(backend, label_resolver=resolver)

# Resolver returns "staging", staging template fetched.
classify = await manager.fetch("experimental_classifier")
# Resolver returns "production" (the default), production fetched.
greet = await manager.fetch("greet")
# Explicit label bypasses the resolver entirely.
audit = await manager.fetch("greet", "audit")
```

`LabelResolver` is a Protocol with one method, `resolve(name) -> str`.
The reference implementation is `MappingLabelResolver`, but any
class with the right shape works (a JSON-file-backed resolver, a
remote-config-service-backed resolver).

## Composite backends and fallback

A manager constructed with multiple backends consults them in
order. The fallback rule distinguishes infrastructure failure
from logical absence:

```python
from openarmature.prompts import PromptManager
from openarmature_langfuse import LangfusePromptBackend  # hypothetical sibling

manager = PromptManager(
    LangfusePromptBackend(api_key=...),
    FilesystemPromptBackend("./prompts"),  # local fallback
)
```

- **`PromptStoreUnavailable` from a backend → try the next.**
  Network's down, vendor API is 5xx-ing, filesystem hiccupped,
  so the manager falls back. This is the "Langfuse is degraded,
  use the local copy" case.
- **`PromptNotFound` from a backend → STOP the chain.** The
  error propagates. This is the "operator deliberately deleted
  the prompt from Langfuse to retire it" case; falling back here
  would silently resurface a stale local copy under a name the
  operator wanted gone.
- **All backends `PromptStoreUnavailable` → manager raises
  `PromptStoreUnavailable`.** Everything's down.

The two error categories have different operational
meanings; the manager keeps them separated.

## Errors

Three categories cover every failure mode:

| Error                     | When                                                                | Transient |
| ------------------------- | ------------------------------------------------------------------- | --------- |
| `PromptNotFound`          | No prompt matches `(name, label)` in any backend (after §8 rules)   | No        |
| `PromptRenderError`       | Undefined variable, template parse error, coercion failure          | No        |
| `PromptStoreUnavailable`  | Backend infrastructure failure (network, I/O, vendor API)           | Yes       |

`PROMPT_TRANSIENT_CATEGORIES` is exported as a frozenset for
retry-middleware classifiers, matching the pattern
`openarmature.llm` uses with its `TRANSIENT_CATEGORIES`.

## PromptGroup: tracing related prompts together

A `PromptGroup` is a structural grouping of two or more
`PromptResult` instances under a stable `group_name`. The
group itself doesn't execute anything; it gives observability
a shared name to render related calls under.

```python
from openarmature.prompts import PromptGroup, with_active_prompt_group

classify = await manager.get("classify", variables={"input": user_query})
answer = await manager.get("answer", variables={"input": user_query, ...})

group = PromptGroup(group_name="classifier_chain", members=[classify, answer])
with with_active_prompt_group(group):
    # Every LLM call in this scope carries
    # openarmature.prompt.group_name="classifier_chain".
    classification = await provider.complete(classify.messages, ...)
    final = await provider.complete(answer.messages, ...)
```

Canonical patterns the primitive covers:

- **Multi-stage classification**: `[coarse, fine, answer]`.
- **RAG with reranking**: `[query_rewrite, retrieve, rerank, answer]`.
- **Self-correction loops**: `[generate, critique, revise]`.
- **Map-reduce over chunks**: `[chunk_classify_1..N, synthesize]`.

The N=2 case ("classifier + follow-up") is the simplest;
larger groups work under the same primitive. Constructing a
group with fewer than two members raises `PromptGroupInvalid`
at construction time, before any render or call; single-prompt
tagging is already served by the per-prompt observability
attributes below.

## Observability propagation

When an LLM call fires inside `with_active_prompt(result)` (or
`with_active_prompt_group(group)`), the OTel observer surfaces
six normative attributes on the `openarmature.llm.complete`
span:

- `openarmature.prompt.name`
- `openarmature.prompt.version`
- `openarmature.prompt.label`
- `openarmature.prompt.template_hash`
- `openarmature.prompt.rendered_hash`
- `openarmature.prompt.group_name`

Pattern:

```python
result = await manager.get("greeting", "production", {"user": "Alice"})
with with_active_prompt(result):
    response = await provider.complete(result.messages, ...)
```

Trace UIs can then pivot on `prompt.name`, filter on
`prompt.template_hash` to find every call that used a given
template version, or surface `prompt.group_name` to group
related calls into a single workflow view.

Nesting is innermost-wins. If you activate a result inside
another active result, the inner one wins for the duration
of the inner block.

### Backend-keyed observability entity references

A `Prompt` also carries an optional `observability_entities`
mapping for backend-keyed references to first-class entities
the prompt has been registered as in observability backends. The
spec-normative key is `langfuse_prompt`, holding the Langfuse SDK
`Prompt` reference. The Langfuse observer (when it ships) reads
this field to establish the native Generation → Prompt link
rather than reaching into the implementation-defined `metadata`
mapping. Backends that don't surface such references leave the
field `None`.

## Determinism and content-addressed caching

`render` is deterministic: same `Prompt`, same `variables` →
bytewise-identical `messages` and `rendered_hash` across
calls. This is the cache-key contract: `rendered_hash`
gives a downstream memoization layer the right equivalence
relation for free.

Templates MAY reference user-supplied variables that capture
nondeterministic values (`now=datetime.utcnow()`); the
determinism contract applies to the render operation given
fixed inputs, not to user-supplied variable content.

## A minimal example

```python
import asyncio
from pathlib import Path

from openarmature.prompts import FilesystemPromptBackend, PromptManager


async def main() -> None:
    manager = PromptManager(FilesystemPromptBackend(Path("./prompts")))
    result = await manager.get(
        "greeting",
        "production",
        variables={"user": "Alice"},
    )
    print(result.messages[0].content)         # rendered text
    print(result.rendered_hash)               # cache key


asyncio.run(main())
```

The filesystem backend layout is
`<root>/<label>/<name>.j2`; for the example above,
`./prompts/production/greeting.j2`.

## Prefix-cache friendly authoring (APC)

Inference engines that implement Automatic Prefix Caching
(vLLM with `--enable-prefix-caching`, OpenAI's hosted prompt
caching, llama.cpp's prefix reuse, others) skip recomputing
attention for token prefixes they have already processed in
a recent request. The cache hit is decided by **byte equality**
of the prefix. A single reordered key, a shuffled tool
definition, or a timestamp embedded in the system prompt
invalidates the cache and re-runs full attention from the
first changed byte.

OpenArmature handles the wire-byte half of this contract for
you. The OpenAI provider canonicalizes every user-supplied dict
on the wire — tool parameter schemas, response-format schemas,
`RuntimeConfig` extras, tool-call arguments — so equivalent OA
inputs produce byte-identical wire output regardless of dict
insertion order. Prompt rendering is deterministic by
construction: same `Prompt` plus same variables produces
byte-identical `PromptResult.messages` (spec
[prompt-management §13](https://github.com/LunarCommand/openarmature-spec/blob/main/spec/prompt-management/spec.md#13-determinism)).

Authoring discipline that maximizes APC hit rates is
out of OA's hands — it's about how you structure the prompts.
The spec's [llm-provider §14 *APC-friendly authoring
guidance*](https://github.com/LunarCommand/openarmature-spec/blob/main/spec/llm-provider/spec.md#14-apc-friendly-authoring-guidance-informative)
lists five informative patterns; the headline:

1. **Place variables and chat history at the end of templates.**
   Stable static prefix at the front maximizes cacheable bytes.
2. **No timestamps, UUIDs, or other nondeterministic values
   in static segments.** They poison the cache prefix on every
   request.
3. **Stable few-shot ordering.** Pick once, reuse across
   requests; don't shuffle.
4. **Sort retrieval results before injecting** when the
   downstream consumer doesn't care about order.
5. **Cache-friendly tool ordering.** Define tools in a stable
   order across calls.

### Debugging "the cache attribute isn't showing up"

When the OTel observer is running but
`openarmature.llm.cache_read.input_tokens` doesn't appear on
your `openarmature.llm.complete` spans, the cause is almost
always server-side: the inference engine either isn't
configured to surface cache stats, or isn't running with prefix
caching enabled at all.

- **vLLM**: launch with `--enable-prefix-caching` AND
  `--enable-prompt-tokens-details`. The first turns APC on;
  the second tells vLLM to populate
  `usage.prompt_tokens_details.cached_tokens` on the wire
  response. Both flags are required for the attribute to
  surface.
- **OpenAI hosted (Chat Completions)**: prompt caching is
  on automatically for prompts ≥1024 tokens; the
  `prompt_tokens_details.cached_tokens` field appears on
  qualifying responses without configuration.

OA's role is to source the field when present (provider-side)
and emit the attribute when populated (observer-side); without
the upstream signal, neither happens — and that's the right
behavior (per the spec's absent-vs-zero distinction, an absent
attribute means "the provider didn't report," not "zero
hits").

## What's out of scope (for now)

- **Specific vendor backends**: Langfuse, PromptLayer, etc.,
  ship as sibling packages (`openarmature-langfuse`, …). The
  core ships the protocol + a filesystem reference.
- **Prompt versioning workflows**: how versions are assigned,
  promoted, pinned. Per project. The spec defines the
  `version` field; the discipline is yours.
- **Cache invalidation policies**: `template_hash` and
  `rendered_hash` are the keys; the cache itself is a
  separate concern.
- **Prompt linting / evaluation**: quality checks belong to
  separate tools (or the future eval capability).
- **Multi-message render decomposition**: v1 emits a single
  `UserMessage` carrying the rendered text. If you need
  `system + user` splits, construct the messages list
  manually outside `render()` for now.

## Where to next

- **[Model Providers](../model-providers/index.md)**:
  what to pass `result.messages` into.
- **[API reference: `openarmature.prompts`](../reference/prompts.md)**:
  the full public surface.
