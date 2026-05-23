# Examples

End-to-end demos of `openarmature`, each framed around a small but
plausible use case. Read top to bottom for a guided tour, or jump
straight to whichever example covers the feature you're learning.

Every demo is a standalone `main.py` you can run against any
OpenAI-compatible LLM endpoint (OpenAI's public API, vLLM, LM Studio,
llama.cpp server, etc.). All code lives under
[`examples/`](https://github.com/LunarCommand/openarmature-python/tree/main/examples)
in the repo.

## Catalog

- [**00 - Hello, world**](00-hello-world.md). Classify a query and
  route to one of two follow-up nodes. Smallest possible LLM-routed
  pipeline; introduces typed state, reducers, conditional edges, and
  structured output.
- [**01 - Routing and subgraphs**](01-routing-and-subgraphs.md).
  Question-answering assistant that branches into a short-answer node
  or a research subgraph, then copy-edits the result.
- [**02 - Explicit subgraph mapping**](02-explicit-subgraph-mapping.md).
  Run the same analysis subgraph against two parent fields by mapping
  them explicitly at each call site.
- [**03 - Observer hooks**](03-observer-hooks.md). Attach
  observability to a `draft → review → finalize` pipeline without
  touching any node code, including OpenTelemetry spans.
- [**04 - Nested subgraphs**](04-nested-subgraphs.md). Question
  answering against a small baked-in document corpus with two levels
  of subgraph nesting.
- [**05 - Fan-out with retry**](05-fan-out-with-retry.md). Summarize a
  batch of news headlines in parallel, with retry + timing middleware
  wrapping each per-instance subgraph run.
- [**06 - Parallel branches**](06-parallel-branches.md). Enrich an
  article with three independent analyses (summary, sentiment, tags)
  running concurrently, each with its own state schema.
- [**07 - Multimodal prompt**](07-multimodal-prompt.md). Two analyses
  of a lunar-mission photograph using versioned prompt templates,
  multimodal user messages, and a prompt-group trace.
- [**08 - Checkpointing and migration**](08-checkpointing-and-migration.md).
  Resume a saved invocation under an upgraded state schema, with a
  v1→v2 migration backfilling new fields.
- [**09 - Tool use**](09-tool-use.md). Lunar-mission assistant that
  calls local Python tools to answer questions mixing fact recall and
  physics arithmetic.

## Configuration

All demos read their LLM client config from environment variables.
The OpenAI public-API defaults are:

| Env var         | Default                    | Notes                                                                            |
| --------------- | -------------------------- | -------------------------------------------------------------------------------- |
| `LLM_BASE_URL`  | `https://api.openai.com`   | Host root only. The provider adds the path.                                      |
| `LLM_MODEL`     | `gpt-4o-mini`              | Any model the bound endpoint exposes.                                            |
| `LLM_API_KEY`   | (none)                     | Required. Pass empty for local servers that don't authenticate.                  |

For a local OpenAI-compatible server (vLLM, LM Studio, llama.cpp,
etc.), point `LLM_BASE_URL` at the host root (e.g.
`http://localhost:8000`) and set `LLM_API_KEY` to whatever value the
server expects.

## Running locally

```bash
# Install the examples dep group.
uv sync --group examples

# Demos 03 and 07 also want the OTel SDK for their OTelObserver.
uv sync --group examples --all-extras

# Run any demo.
LLM_API_KEY=sk-... uv run python examples/00-hello-world/main.py
LLM_API_KEY=sk-... uv run python examples/01-routing-and-subgraphs/main.py "what year did the moon landing happen"
```
