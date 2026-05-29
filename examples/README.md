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
middleware (per-instance duration captured). Demonstrates:
`add_fan_out_node` with `items_field` mode, `extra_outputs`
collecting a parallel list, `instance_middleware`, concurrency cap.

### [`06-parallel-branches/`](./06-parallel-branches/main.py)

Enrich an article with three independent analyses (summary,
sentiment, topic tags) running concurrently. Each analysis is a
separate subgraph with its own state schema. The sentiment branch
wraps its subgraph in retry middleware; the other two run bare.
Demonstrates: `add_parallel_branches_node`, `BranchSpec` per branch
with input/output projection, heterogeneous branch state schemas,
per-branch middleware.

### [`07-multimodal-prompt/`](./07-multimodal-prompt/main.py)

Two independent analyses of a lunar-mission photograph — describe
the surface, describe the equipment — using versioned prompt
templates and a multimodal user message. Templates load from
`FilesystemPromptBackend` with a primary + fallback chain; both
renders are grouped under one observability `PromptGroup` so a trace
UI can render them as one logical unit. Image source switches
between `ImageSourceURL` and `ImageSourceInline(base64_data=...)`
via env var. Demonstrates: `PromptManager` with composite backends,
prompt fetch + render with template variables, `PromptGroup` +
`with_active_prompt_group`, `with_active_prompt` nesting,
multimodal `UserMessage` carrying both text and image content blocks.

### [`08-checkpointing-and-migration/`](./08-checkpointing-and-migration/main.py)

A lunar-mission planning pipeline that checkpoints after every step,
then resumes the saved record under an upgraded state schema. Phase
one invokes a v1 graph against `MissionPlanStateV1`; the
`SQLiteCheckpointer` (JSON mode) writes records to a temp DB. Phase
two registers a v1→v2 migration backfilling a new `risk_assessment`
field, builds a v2 graph with one new node, and resumes from the v1
invocation. Demonstrates: `SQLiteCheckpointer(serialization="json")`,
`with_checkpointer`, save-on-completed-event, `State.schema_version`,
`with_state_migration`, `invoke(resume_invocation=...)`.

### [`09-tool-use/`](./09-tool-use/main.py)

A lunar-mission assistant that calls local Python tools to answer
questions mixing fact recall and physics arithmetic. Defines two
tools (`lookup_mission` reading a baked-in record store,
`compute_delta_v` doing a Hohmann transfer), passes them to the
model via `complete(tools=...)`, dispatches `assistant.tool_calls`
to the local functions, and feeds the results back as
`ToolMessage` entries. The agent loop is a graph cycle:
`call_llm → dispatch_tools → call_llm` via a conditional edge, with
a hard turn cap to prevent runaway loops. Demonstrates: `Tool`
definitions with JSON Schema parameters, `complete(tools=...)`,
parsing `ToolCall` records, `ToolMessage(tool_call_id=...)` round-
trip, multi-turn tool-calling loop as a graph cycle.

## Configuration

All demos configure their LLM client via env vars; OpenAI public-API
defaults shown:

| Env var | Default | Notes |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://api.openai.com` | **Host root only** — the provider adds the path. |
| `LLM_MODEL` | `gpt-4o-mini` | Any model the bound endpoint exposes. |
| `LLM_API_KEY` | (none) | Required; pass empty for local servers that don't authenticate. |

The Langfuse observer and the Langfuse prompt backend read the standard
Langfuse SDK variables when pointed at a live Langfuse account;
`Langfuse()` reads them automatically, so no credentials appear in the
example code:

| Env var | Notes |
| --- | --- |
| `LANGFUSE_PUBLIC_KEY` | From your Langfuse project settings. |
| `LANGFUSE_SECRET_KEY` | From your Langfuse project settings. |
| `LANGFUSE_BASE_URL` | Langfuse host (e.g. `https://cloud.langfuse.com`); the SDK also accepts `LANGFUSE_HOST`. |

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
