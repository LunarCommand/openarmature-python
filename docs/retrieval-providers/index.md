# Retrieval Providers

A retrieval provider is the seam between OpenArmature's graph engine and
a vector or reranking backend (a hosted API like OpenAI, Cohere, or Jina,
or a self-hosted Text Embeddings Inference server). The engine does not
know about embeddings or rerankers; nodes call providers, providers do
the wire work. For what embedding and reranking are and how they fit a
pipeline, see the [Retrieval concept page](../concepts/retrieval.md); this
section is the catalog of what ships and how to run it.

## What ships

Four vendors ship as reference providers. Each embedding surface and each
rerank surface is a separate provider instance bound to one model:

| Provider | Embedding | Rerank | Default `base_url` |
|---|---|---|---|
| OpenAI-compatible | `OpenAIEmbeddingProvider` | not offered | `https://api.openai.com` |
| Cohere | `CohereEmbeddingProvider` | `CohereRerankProvider` | `https://api.cohere.com` |
| Jina | `JinaEmbeddingProvider` | `JinaRerankProvider` | `https://api.jina.ai` |
| TEI (self-hosted) | `TeiEmbeddingProvider` | `TeiRerankProvider` | none (pass your instance URL) |

A few provider-specific notes:

- **OpenAI-compatible** is embedding-only and symmetric. `base_url` is
  overridable, so the same provider drives any OpenAI-shaped embedding
  endpoint (vLLM, LocalAI, TEI's OpenAI surface). `dimensions` maps to the
  wire for Matryoshka models that support truncation.
- **Cohere** embedding requires an `input_type`; the mapping sends a
  sensible default when you omit it. Its reranker returns scores without
  echoing documents, so map results back to your candidates by
  `ScoredDocument.index`.
- **Jina** realizes `input_type` as its native task field over a fixed
  set of values and reports token usage on both surfaces.
- **TEI** is the self-hosted option: you pass the instance URL, it reports
  no usage, and its reranker chunks a large candidate pool the same way
  embedding chunks a large input list. See [Self-hosting TEI](tei.md).

For a backend none of these cover, write your own; see
[Authoring a Provider](authoring.md).

## The contract

Retrieval is two protocols, one async method each:

```python
from collections.abc import Sequence
from typing import Protocol

from openarmature.retrieval import (
    EmbeddingResponse,
    EmbeddingRuntimeConfig,
    RerankResponse,
    RerankRuntimeConfig,
)


class EmbeddingProvider(Protocol):
    async def ready(self) -> None: ...
    async def embed(
        self,
        input: Sequence[str],
        *,
        config: EmbeddingRuntimeConfig | None = None,
    ) -> EmbeddingResponse: ...


class RerankProvider(Protocol):
    async def ready(self) -> None: ...
    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
        config: RerankRuntimeConfig | None = None,
    ) -> RerankResponse: ...
```

- **`ready()`** verifies the bound model is reachable. Pre-flight check,
  typically called once before invoking the graph.
- **`embed()`** returns one vector per input string, in input order.
- **`rerank()`** scores the documents against the query and returns them
  sorted best-first.

### Behaviour guarantees

- **Input order.** `embed()` returns `len(input)` vectors, `vectors[i]`
  being the embedding of `input[i]`, regardless of how the provider
  paginated the request. All vectors share one dimensionality.
- **Arbitrary-length input.** `embed()` accepts any-length list; the
  mapping chunks under the provider's per-request cap and stitches the
  result. `rerank()` chunks a large candidate pool the same way.
- **Usage is a record or null.** `response.usage` is a token-accounting
  record when the provider reports one and `None` otherwise. The mapping
  never fabricates a record, a zero, or an estimate.
- **Reentrant and stateless.** Safe to call concurrently from many nodes;
  no implicit state carries between calls. Inputs are never mutated.
- **No retry on transient errors.** That is middleware's job; wrap the
  calling node in `RetryMiddleware` or similar.

## Errors

Retrieval calls raise the same canonical error categories as LLM calls
(from `openarmature.llm.errors`), mapped from the provider's HTTP
response. The retrieval-applicable subset:

| Error | Trigger |
|---|---|
| `ProviderAuthentication` | 401 / 403 (bad or missing key) |
| `ProviderRateLimit` | 429 |
| `ProviderInvalidModel` | 404 (bound model not found) |
| `ProviderInvalidRequest` | 400 / 413 / 422 (malformed or over-length request, empty input) |
| `ProviderUnavailable` | 5xx, network failure, timeout |
| `ProviderInvalidResponse` | 200 OK that fails to parse, or a malformed response body |

`ProviderUnavailable` and `ProviderRateLimit` are in
`TRANSIENT_CATEGORIES`, the canonical "safe to retry" set the default
retry-middleware classifier uses. A malformed usage figure does not fail
the call: the vectors or scores are sound, so the count is recorded as
unknown (`usage = None`) rather than raising.

## A minimal example

Direct usage of the providers, without the engine in the picture:

```python
import asyncio

from openarmature.retrieval import CohereRerankProvider, OpenAIEmbeddingProvider


async def main() -> None:
    embedder = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="sk-...")
    reranker = CohereRerankProvider(model="rerank-v3.5", api_key="...")
    try:
        vectors = (await embedder.embed(["the lunar south pole"])).vectors
        ranked = (await reranker.rerank("water on the Moon", ["...", "..."])).results
        print(len(vectors), [r.index for r in ranked])
    finally:
        await embedder.aclose()
        await reranker.aclose()


asyncio.run(main())
```

In a real graph you construct the providers once at startup and let nodes
call them inside their bodies, where their `EmbeddingEvent` / `RerankEvent`
reach any attached observer.

## Where to next

- **[Self-hosting TEI](tei.md)**: run Text Embeddings Inference for
  embeddings and reranking, from a first `docker run` to a production
  deployment, and wire the TEI providers to it.
- **[Authoring a Provider](authoring.md)**: implement the
  `EmbeddingProvider` or `RerankProvider` protocol for a backend none of
  the bundled providers cover.
- **[API reference: `openarmature.retrieval`](../reference/retrieval.md)**:
  the full public surface (the providers, response types, runtime configs).
