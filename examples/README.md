# Examples

End-to-end demo projects for `openarmature`. Each is a standalone
`main.py` you can run against a local OpenAI-compatible LLM endpoint
(vLLM, LM Studio, llama.cpp server, etc.).

## Demos

### [`00-hello-world/`](./00-hello-world/main.py)

Classify a query with an LLM and route to one of two follow-up
nodes. Demonstrates: typed `State` with three reducer policies, the
`OpenAIProvider` from `openarmature.llm`, structured output via a
Pydantic class (`response_schema=Classification` → `Response.parsed`
as a `Classification` instance), conditional routing on a parsed
field, and a compile-time observer.

Configured via env vars (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`);
defaults to OpenAI public API with `gpt-4o-mini`.

### [`01-linear-pipeline/`](./01-linear-pipeline/main.py)

Minimal two-node graph (`plan → write`). Demonstrates: typed `State`,
the `append` reducer, static edges, `END`.

### [`02-routing-and-subgraphs/`](./02-routing-and-subgraphs/main.py)

Question-answering assistant — classify, then short-answer or
research-subgraph, then copy-edit. Demonstrates: conditional edges,
`SubgraphNode`, custom `ProjectionStrategy`, the `merge` reducer.

### [`03-explicit-subgraph-mapping/`](./03-explicit-subgraph-mapping/main.py)

Compare two topics by running the same analysis subgraph on each.
Demonstrates: `ExplicitMapping` for reusing one compiled subgraph at
multiple parent sites with disjoint parent fields.

### [`04-observer-hooks/`](./04-observer-hooks/main.py)

Add observability to a `draft → review → finalize` pipeline without
changing any node code. Demonstrates: `attach_observer`, `NodeEvent`,
namespace chaining across subgraph boundaries, both function-shaped
and class-shaped observers.

### [`05-nested-subgraphs/`](./05-nested-subgraphs/main.py)

Two levels of nested subgraphs (`outer → middle → inner`) with a
depth-aware observer printing the descent and return. Demonstrates
spec graph-engine §6 depth invariants.

## Running

```bash
# From the repo root, install the examples dep group:
uv sync --group examples

# Run any demo:
uv run python examples/01-linear-pipeline/main.py "the psychology of long walks"
```

All five demos expect an OpenAI-compatible LLM endpoint at
`http://localhost:8000/v1`. Edit `VLLM_BASE_URL` and `MODEL` at the
top of each `main.py` to point elsewhere.
