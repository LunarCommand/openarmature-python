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

## Prompt identity

Every `Prompt` carries five identity fields:

- `name` — your stable identifier (`"greeting"`).
- `version` — the backend's version string. Implementation-defined:
  a backend MAY use semver, monotonic integers, content
  hashes, git short-SHAs, or any stable identifier. The
  filesystem backend derives it from the template content
  hash.
- `label` — the slot the prompt was fetched from
  (`"production"`, `"latest"`, `"variant-a"`). The label is
  part of the query.
- `template_hash` — SHA-256 of the raw template source.
  Two prompts with different content always have different
  hashes.
- `fetched_at` — when the prompt was fetched. Cached
  backends preserve the original fetch time, not the
  cache-hit time.

The `name + version + label` triple identifies the prompt;
the `template_hash` lets you tell two prompts apart by
*content*, which matters when a vendor backend serves
different content under the same `latest` label over time.

A `PromptResult` propagates all of those, plus:

- `rendered_hash` — SHA-256 over the rendered messages.
  Same template + same variables → same hash. This is the
  cache-key value a memoization layer wants.
- `messages` — the rendered output as an LLM-ready
  `list[Message]`. Directly consumable by
  `Provider.complete()`.
- `variables` — what was applied. Audit-trail friendly.
- `rendered_at` — when the render happened. Distinct from
  `fetched_at`.

## Strict variables by default

A template that references a variable not in the mapping
raises `PromptRenderError`:

```python
prompt = await manager.fetch("greeting", "production")  # "Hello, {{ user }}! Today is {{ day }}."
manager.render(prompt, {"user": "Alice"})  # raises — "day" is undefined
```

This is intentional. Silently substituting empty strings for
missing variables masks bugs: a typo'd variable name produces
a working-but-wrong prompt, often invisibly. If you need
lenient behavior, wrap your variables in your own defaulting
layer before passing them to `render()`.

The Python implementation uses Jinja2's `StrictUndefined`.

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
  Network's down, vendor API is 5xx-ing, filesystem hiccupped —
  the manager falls back. This is the "Langfuse is degraded,
  use the local copy" case.
- **`PromptNotFound` from a backend → STOP the chain.** The
  error propagates. This is the "operator deliberately
  deleted the prompt from Langfuse to retire it" case —
  falling back here would silently resurface a stale local
  copy under a name the operator wanted gone.
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
retry-middleware classifiers — the same pattern
`openarmature.llm` uses with its `TRANSIENT_CATEGORIES`.

## PromptGroup — tracing related prompts together

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

- **Multi-stage classification** — `[coarse, fine, answer]`.
- **RAG with reranking** — `[query_rewrite, retrieve, rerank, answer]`.
- **Self-correction loops** — `[generate, critique, revise]`.
- **Map-reduce over chunks** — `[chunk_classify_1..N, synthesize]`.

The N=2 case ("classifier + follow-up") is the simplest;
larger groups work under the same primitive. The group rejects
empty and single-member shapes — single-prompt tagging is
already served by the per-prompt observability attributes
below.

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

## Determinism and content-addressed caching

`render` is deterministic: same `Prompt`, same `variables` →
bytewise-identical `messages` and `rendered_hash` across
calls. This is the cache-key contract — `rendered_hash`
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
`<root>/<label>/<name>.j2` — for the example above,
`./prompts/production/greeting.j2`.

## What's out of scope (for now)

- **Specific vendor backends** — Langfuse, PromptLayer, etc.,
  ship as sibling packages (`openarmature-langfuse`, …). The
  core ships the protocol + a filesystem reference.
- **Prompt versioning workflows** — how versions are assigned,
  promoted, pinned. Per project. The spec defines the
  `version` field; the discipline is yours.
- **Cache invalidation policies** — `template_hash` and
  `rendered_hash` are the keys; the cache itself is a
  separate concern.
- **Prompt linting / evaluation** — quality checks belong to
  separate tools (or the future eval capability).
- **Multi-message render decomposition** — v1 emits a single
  `UserMessage` carrying the rendered text. If you need
  `system + user` splits, construct the messages list
  manually outside `render()` for now.

## Where to next

- **[Model Providers](../model-providers/index.md)** —
  what to pass `result.messages` into.
- **[API reference: `openarmature.prompts`](../reference/prompts.md)** —
  the full public surface.
