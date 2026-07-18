# Authoring a Provider

When you target a backend none of the bundled providers cover (a
different vendor API, an internal embedding gateway, a hand-rolled
inference service), implement the `EmbeddingProvider` or `RerankProvider`
protocol yourself. Each is one async method plus a readiness check, so a
minimum-viable provider is short.

If you are new to retrieval providers, read
[Retrieval Providers](index.md) for the contract guarantees first.

## Skeleton

A minimal embedding provider against an OpenAI-shaped `/v1/embeddings`
endpoint. Compare with `openarmature.retrieval.OpenAIEmbeddingProvider`
to see what a full implementation adds (input-list chunking, the
client-side `input_type` prefix, observability events, a readiness probe).

```python
from collections.abc import Sequence

import httpx

from openarmature.llm import (
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderRateLimit,
    ProviderUnavailable,
)
from openarmature.retrieval import (
    EmbeddingResponse,
    EmbeddingRuntimeConfig,
    EmbeddingUsage,
    validate_embedding_input,
    validate_embedding_response,
)


class MyEmbeddingProvider:
    def __init__(self, *, base_url: str, model: str, api_key: str | None = None) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=60.0)
        self.model = model

    async def ready(self) -> None:
        resp = await self._client.post(
            "/v1/embeddings", json={"model": self.model, "input": ["ready"]}
        )
        if resp.status_code != 200:
            raise _classify(resp)

    async def embed(
        self,
        input: Sequence[str],
        *,
        config: EmbeddingRuntimeConfig | None = None,
    ) -> EmbeddingResponse:
        input_strings = list(input)
        validate_embedding_input(input_strings)  # rejects an empty list

        body = {"model": self.model, "input": input_strings}
        try:
            resp = await self._client.post("/v1/embeddings", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc
        if resp.status_code != 200:
            raise _classify(resp)

        # A malformed 200 body (bad JSON, missing keys, wrong types) is a
        # provider_invalid_response, not a leaked KeyError/ValueError.
        try:
            payload = resp.json()
            # Order the returned data by index so vectors line up with input.
            entries = sorted(payload["data"], key=lambda e: e["index"])
            vectors = [[float(x) for x in e["embedding"]] for e in entries]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderInvalidResponse("malformed embeddings response") from exc
        # Validates the count and uniform dimensionality, returns the dim.
        dimensions = validate_embedding_response(vectors, len(input_strings))

        prompt_tokens = payload.get("usage", {}).get("prompt_tokens")
        usage = (
            EmbeddingUsage(input_tokens=prompt_tokens)
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0
            else None  # never fabricate a record the provider did not report
        )
        return EmbeddingResponse(
            vectors=vectors,
            model=payload.get("model", self.model),
            usage=usage,
            response_id=payload.get("id"),
            dimensions=dimensions,
            raw=payload,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _classify(resp: httpx.Response) -> Exception:
    status = resp.status_code
    if status in (401, 403):
        return ProviderAuthentication(f"HTTP {status}")
    if status == 404:
        return ProviderInvalidModel("model not found")
    if status == 429:
        return ProviderRateLimit("HTTP 429")
    if status in (400, 413, 422):
        return ProviderInvalidRequest(f"HTTP {status}")
    if status >= 500:
        return ProviderUnavailable(f"HTTP {status}")
    return ProviderInvalidResponse(f"HTTP {status}")
```

A `RerankProvider` is the same shape: a `rerank(query, documents, *,
top_k=None, config=None)` method that posts the query and documents,
validates the inputs with `validate_rerank_input`, builds `ScoredDocument`
results, and validates them with `validate_rerank_response` before
returning a `RerankResponse`.

## Contract checklist

When you ship a provider, the following must hold:

**Input order and shape.**

- [ ] `embed()` MUST return one vector per input, in input order, so
      `vectors[i]` is the embedding of `input[i]`. `validate_embedding_response`
      enforces the count and uniform dimensionality; ordering is yours to get
      right (sort by the wire's index, or trust a positional wire).
- [ ] `rerank()` results MUST be sorted by relevance, best first, and each
      `ScoredDocument.index` MUST point into the input `documents` list.
      `validate_rerank_response` enforces the index range and `top_k`.

**Non-mutation and reentrancy.**

- [ ] Inputs MUST NOT be mutated; build wire bodies from copies.
- [ ] Concurrent calls on one instance MUST be safe (httpx.AsyncClient is).

**Boundary validation.**

- [ ] Call `validate_embedding_input(input)` (rejects an empty list) or
      `validate_rerank_input(query, documents, top_k)` (rejects an empty
      query, empty documents, or non-positive `top_k`) before sending.

**Usage is a record or null.**

- [ ] Populate `response.usage` only when the provider reports token
      accounting. MUST NOT fabricate a record, a zero, or a client-side
      estimate; a provider that reports nothing yields `usage = None`. A
      malformed usage figure reads as not-reported, not as an error.

**Error mapping.**

- [ ] Network failures and timeouts to `ProviderUnavailable`; 401/403 to
      `ProviderAuthentication`; 404 (model not found) to
      `ProviderInvalidModel`; 429 to `ProviderRateLimit`; 400/413/422 to
      `ProviderInvalidRequest`; other 5xx to `ProviderUnavailable`; a 200
      that fails to parse to `ProviderInvalidResponse`.

**Rerank document echo.**

- [ ] If the wire echoes document text, put it on `ScoredDocument.document`
      verbatim; otherwise leave it `None`. Never reconstruct it from the
      input documents.

## Beyond the skeleton

The skeleton omits things real providers usually need:

- **Chunking for a capped wire.** Most embedding APIs cap the inputs per
  request. When `embed()` must accept an arbitrary-length list, split it
  into consecutive slices under the cap, issue one request per slice with
  every non-input field identical, concatenate the vectors in input order,
  and combine usage all-or-nothing (sum only when every slice reported a
  figure, else `None`). The bundled `OpenAIEmbeddingProvider` and
  `TeiEmbeddingProvider` do this through a shared internal helper; a custom
  provider implements the same loop. A `RerankProvider` chunks a large
  candidate pool the same way.

- **The `input_type` knob.** `EmbeddingRuntimeConfig.input_type`
  (`"query"` / `"document"`) selects the asymmetric-embedding
  representation. Realize it however your wire expects: a native request
  field, a server-side prompt name, or a client-side prefix prepended to
  each input. On a symmetric model it is a no-op.

- **Observability events.** Dispatch a typed `EmbeddingEvent` (or
  `RerankEvent`) on success and the failed variant alongside any error, so
  the bundled OTel and Langfuse observers render retrieval spans and
  observations. The shape mirrors the LLM path (see the
  [LLM provider authoring guide](../model-providers/authoring.md#beyond-the-skeleton)
  for the full dispatch sketch): capture the request-side data once, then
  on success dispatch an `EmbeddingEvent` built from the response, or on an
  error dispatch an `EmbeddingFailedEvent` **alongside** the raised
  exception (the exception still propagates; the event goes on the observer
  queue). A single call emits exactly one success or one failed event,
  never both.

  ```python
  import uuid

  from openarmature.graph import EmbeddingEvent
  from openarmature.observability.correlation import (
      current_dispatch,
      current_invocation_id,
      current_namespace_prefix,
  )


  # Inside embed(), after a successful parse:
  dispatch = current_dispatch()
  if dispatch is not None:
      namespace = current_namespace_prefix()
      dispatch(
          EmbeddingEvent(
              invocation_id=current_invocation_id() or "",
              node_name=namespace[-1] if namespace else "",
              namespace=namespace,
              provider="my-provider",
              model=self.model,
              response_model=response.model,
              response_id=response.response_id,
              usage=response.usage,
              input_strings=input_strings,
              input_count=len(input_strings),
              dimensions=response.dimensions,
              output_vectors=response.vectors,
              call_id=str(uuid.uuid4()),
              # plus the scoping fields: correlation_id, attempt_index,
              # fan_out_index, branch_name, latency_ms, request_params,
              # request_extras, active_prompt, active_prompt_group.
          )
      )
  ```

The conformance fixtures under `tests/conformance/test_retrieval_provider.py`
exercise the wire mapping end to end; a custom provider that passes them
matches the contract.
