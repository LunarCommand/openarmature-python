# Retrieval

The retrieval capability covers the two building blocks of a search or
RAG pipeline: turning text into vectors (**embedding**) and reordering a
candidate list by relevance to a query (**reranking**). Like the LLM
capability, both are async IO you call from inside a node body, behind a
small provider protocol so the graph does not care which vendor is on the
other end.

Everything lives in `openarmature.retrieval`:

```python
from openarmature.retrieval import (
    OpenAIEmbeddingProvider,
    CohereRerankProvider,
    EmbeddingRuntimeConfig,
    RerankRuntimeConfig,
)
```

## Embedding text

An `EmbeddingProvider` turns a list of strings into a list of vectors,
one vector per input, in input order:

```python
provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", api_key="sk-...")

response = await provider.embed(["the lunar south pole", "the Sea of Tranquility"])

response.vectors      # list[list[float]], one per input, same order
response.dimensions   # the vector length (equal across all vectors)
response.model        # the model the provider actually served
response.usage        # an EmbeddingUsage record, or None (see below)
```

The input-order guarantee is the contract you build on: `vectors[i]` is
always the embedding of `input[i]`, no matter how the provider paginated
the request under the hood. `len(response.vectors) == len(input)` always
holds, and every vector has the same dimensionality.

`embed` takes an arbitrary-length list. You do not have to pre-chunk to
fit a provider's per-request cap; the mapping does that for you (see
[Long input lists](#long-input-lists-are-chunked-for-you)).

## Reranking candidates

A `RerankProvider` scores a set of candidate documents against a query
and returns them sorted best-first. This is the precision step after a
cheap-and-broad first pass (vector similarity, keyword search, whatever):

```python
provider = CohereRerankProvider(model="rerank-v3.5", api_key="...")

response = await provider.rerank(
    "where is there water ice on the Moon?",
    ["Regolith is abrasive dust.", "Ice sits in shadowed polar craters.", "..."],
    top_k=3,
)

for result in response.results:      # sorted by relevance_score, best first
    result.index                     # position in the input `documents` list
    result.relevance_score           # higher is more relevant
    result.document                  # the echoed text, or None (see below)
```

Results come back ranked. The mapping applies the sort even if a provider
returns them unsorted, so you can always trust `response.results[0]` to be
the best hit.

`result.index` is the load-bearing field: it points back into the
`documents` list you passed. Not every provider echoes the document text
(Cohere, for one, returns scores only), so `result.document` may be
`None`. Map back to your own candidate list by index rather than relying
on the echo:

```python
ranked = [candidates[result.index] for result in response.results]
```

`top_k` trims the returned results to the best K. Pass it when you only
want the top of the list; omit it to score every candidate.

## Query vs document: `input_type`

Some embedding models are **asymmetric**: they embed a search query and
the documents being searched with slightly different representations, and
mixing them up hurts recall. `EmbeddingRuntimeConfig.input_type` is the
portable knob for this:

```python
# When embedding the corpus you are searching over:
await provider.embed(passages, config=EmbeddingRuntimeConfig(input_type="document"))

# When embedding the user's query at search time:
await provider.embed([query], config=EmbeddingRuntimeConfig(input_type="query"))
```

Each provider realizes it the way its wire expects: on an asymmetric
model it selects the query or document representation; on a symmetric
model (OpenAI's) it is a no-op and the text is embedded verbatim. Setting
it costs nothing on the symmetric providers and keeps the same pipeline
correct if you later switch to an asymmetric one, so prefer setting it.

## Long input lists are chunked for you

Every hosted embedding API caps how many inputs one request may carry.
The mapping honors the contract that `embed` takes any-length input by
splitting a large list into consecutive slices under the cap, issuing one
request per slice, and stitching the vectors back together in input
order. This is invisible: you call `embed` with 10,000 strings and get
10,000 vectors, regardless of the provider's per-request limit.

Usage is combined across the slices: the token total is reported only
when every slice reported one, otherwise `usage` is `None` for the whole
call (an honest "unknown" rather than a partial count).

## Usage is a record or null

`response.usage` is an `EmbeddingUsage` (or `RerankUsage`) record when the
provider reported token accounting, and `None` when it did not. The
mapping never fabricates a usage record, a zero, or a client-side
estimate: a `None` means "the provider told us nothing," which is
different from a real, reported zero. Guard before reading it:

```python
if response.usage is not None:
    print(response.usage.input_tokens)
```

This matters because not every provider bills the same way. A local
Text Embeddings Inference server reports no usage at all, so `usage` is
`None`. Cohere's reranker reports `search_units` but no token count, so
its `RerankUsage` carries `search_units` with `input_tokens=None`.

## The bundled providers

Four vendors ship as reference providers: OpenAI-compatible (embedding
only), Cohere, Jina, and TEI. All four embed; Cohere, Jina, and TEI also
rerank. The [Retrieval Providers](../retrieval-providers/index.md) section
has the full table, the protocol contract, per-provider notes, and a guide
to [self-hosting TEI](../retrieval-providers/tei.md) for embeddings and
reranking on your own hardware. Writing your own is the same exercise as a
custom LLM provider: implement the `EmbeddingProvider` or `RerankProvider`
protocol and the graph treats it like any other.

## Observability

When you call `embed` or `rerank` from inside a node body, the provider
dispatches a typed `EmbeddingEvent` or `RerankEvent` (and a failed
variant on error) to any attached observer, exactly like LLM completions.
The bundled OTel observer renders an `openarmature.embedding.complete` or
`openarmature.rerank.complete` span; the Langfuse observer renders a
dedicated Embedding or Retriever observation. Token usage lands on the
span or observation only when the provider reported it, matching the
nullable-usage contract above.

Provider calls made outside a graph (for example an offline index build)
run fine but dispatch no events, since there is no observer context to
receive them.

## Putting it together

The [`retrieval-rag`](https://github.com/LunarCommand/openarmature-python/tree/main/examples/retrieval-rag)
example wires the whole pattern into a graph: it batch-embeds a corpus
into an index, then per query embeds the question, retrieves by cosine
similarity, reranks the shortlist, and grounds an LLM answer in the
reranked passages.

## Where to next

- [LLMs](llms.md) for the completion side that consumes retrieved context.
- [Observability](observability.md) for what the embedding and rerank
  spans carry.
- The [`openarmature.retrieval` reference](../reference/retrieval.md) for
  the full type surface.
