# Examples

End-to-end demo projects for `openarmature`. Each is a standalone
`main.py` you can run against any OpenAI-compatible LLM endpoint
(OpenAI's public API, vLLM, LM Studio, llama.cpp server, etc.).

## Demos

### [`00-hello-world/`](./00-hello-world/main.py)

Classify a query with an LLM and route to one of two follow-up
nodes. Demonstrates: typed `State` with three reducer policies, the
`OpenAIProvider` from `openarmature.llm`, structured output via a
Pydantic class (`response_schema=Classification` → `Response.parsed`
as a `Classification` instance), conditional routing on a parsed
field, and a compile-time observer.

### [`01-routing-and-subgraphs/`](./01-routing-and-subgraphs/main.py)

Question-answering assistant — classify, then short-answer or
research-subgraph, then copy-edit. Demonstrates: conditional edges,
`SubgraphNode`, custom `ProjectionStrategy`, the `merge` reducer.

### [`02-explicit-subgraph-mapping/`](./02-explicit-subgraph-mapping/main.py)

Compare two topics by running the same analysis subgraph on each.
Demonstrates: `ExplicitMapping` for reusing one compiled subgraph at
multiple parent sites with disjoint parent fields.

### [`03-observer-hooks/`](./03-observer-hooks/main.py)

Add observability to a `draft → review → finalize` pipeline without
changing any node code. Demonstrates: `attach_observer`, `NodeEvent`,
namespace chaining across subgraph boundaries, function-shaped and
class-shaped observers, plus the `OTelObserver` running alongside
the plain observer (same hook, different backend).

### [`04-nested-subgraphs/`](./04-nested-subgraphs/main.py)

Question answering against a tiny baked-in document corpus, with two
levels of subgraph nesting: outer coordinator → doc-QA subgraph →
section-extract subgraph. A depth-aware observer prints the descent
and return.

### [`05-fan-out-with-retry/`](./05-fan-out-with-retry/main.py)

Summarize a batch of news headlines in parallel. Each per-headline
run goes through a `summarize → classify` subgraph wrapped in retry
middleware (transient failures don't tank the batch) and timing
middleware (per-instance duration captured alongside the fan-out
index). Demonstrates: `add_fan_out_node` with `items_field` mode,
`extra_outputs` collecting a parallel list, `instance_middleware`,
concurrency cap.

## Configuration

All demos configure their LLM client via env vars; OpenAI public-API
defaults shown:

| Env var | Default | Notes |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://api.openai.com` | **Host root only** — the provider adds the path. |
| `LLM_MODEL` | `gpt-4o-mini` | Any model the bound endpoint exposes. |
| `LLM_API_KEY` | (none) | Required; pass empty for local servers that don't authenticate. |

## Running

```bash
# From the repo root, install the examples dep group:
uv sync --group examples

# Demo 03 also wants the OTel SDK for its OTelObserver:
uv sync --group examples --all-extras

# Run any demo:
LLM_API_KEY=sk-... uv run python examples/00-hello-world/main.py
LLM_API_KEY=sk-... uv run python examples/01-routing-and-subgraphs/main.py "what year did the moon landing happen"
```

For a local OpenAI-compatible server (vLLM, LM Studio, llama.cpp,
etc.), point `LLM_BASE_URL` at the host root (e.g. `http://localhost:8000`)
and set `LLM_API_KEY` to whatever value the server expects (often
empty or a placeholder).
