# Examples

End-to-end demo projects for `openarmature`. Each is a standalone
`main.py` you can run against any OpenAI-compatible LLM endpoint
(OpenAI's public API, vLLM, LM Studio, llama.cpp server, etc.).

## Demos

Grouped by what they teach.

### Foundations

#### [`hello-world/`](./hello-world/main.py)

Classify a query with an LLM and route to one of two follow-up
nodes. Demonstrates: typed `State` with three reducer policies, the
`OpenAIProvider` from `openarmature.llm`, structured output via a
Pydantic class (`response_schema=Classification` -> `Response.parsed`
as a `Classification` instance), conditional routing on a parsed
field, and a compile-time observer.

### Composition

#### [`routing-and-subgraphs/`](./routing-and-subgraphs/main.py)

Question-answering assistant: classify, then short-answer or
research-subgraph, then copy-edit. Demonstrates: conditional edges,
`SubgraphNode`, custom `ProjectionStrategy`, the `merge` reducer.

#### [`explicit-subgraph-mapping/`](./explicit-subgraph-mapping/main.py)

Compare two topics by running the same analysis subgraph on each.
Demonstrates: `ExplicitMapping` for reusing one compiled subgraph at
multiple parent sites with disjoint parent fields.

#### [`nested-subgraphs/`](./nested-subgraphs/main.py)

Question answering against a tiny baked-in document corpus, with two
levels of subgraph nesting: outer coordinator -> doc-QA subgraph ->
section-extract subgraph. A depth-aware observer prints the descent
and return.

### Concurrency

#### [`fan-out-with-retry/`](./fan-out-with-retry/main.py)

Summarize a batch of news headlines in parallel. Each per-headline
run goes through a `summarize -> classify` subgraph wrapped in
retry middleware (transient failures don't tank the batch) and
timing middleware (per-instance duration captured). Demonstrates:
`add_fan_out_node` with `items_field` mode, `extra_outputs`
collecting a parallel list, `instance_middleware`, concurrency cap.

#### [`parallel-branches/`](./parallel-branches/main.py)

Enrich an article with three independent analyses (summary,
sentiment, topic tags) running concurrently. Each analysis is a
separate subgraph with its own state schema. The sentiment branch
wraps its subgraph in retry middleware; the other two run bare.
Demonstrates: `add_parallel_branches_node`, `BranchSpec` per branch
with input/output projection, heterogeneous branch state schemas,
per-branch middleware.

### Prompts

#### [`multimodal-prompt/`](./multimodal-prompt/main.py)

Two independent analyses of a lunar-mission photograph (describe
the surface, describe the equipment) using versioned prompt
templates and a multimodal user message. Templates load from
`FilesystemPromptBackend` with a primary + fallback chain; both
renders are grouped under one observability `PromptGroup` so a
trace UI can render them as one logical unit. Image source
switches between `ImageSourceURL` and
`ImageSourceInline(base64_data=...)` via env var. Demonstrates:
`PromptManager` with composite backends, prompt fetch + render
with template variables, `PromptGroup` +
`with_active_prompt_group`, `with_active_prompt` nesting,
multimodal `UserMessage` carrying both text and image content
blocks.

#### [`chat-with-multimodal/`](./chat-with-multimodal/main.py)

Four-turn lunar-mission conversation with conversation memory
threaded through `ChatPrompt` + `PlaceholderSegment`. One turn
attaches a photograph; the agent processes it without changing
the chat shape. Demonstrates: `ChatPrompt`, `ContentSegment`,
`PlaceholderSegment`, history threading via the `append` reducer
on `list[Message]`, conditional self-loop for multi-turn cycles.

### Tool use

#### [`tool-use/`](./tool-use/main.py)

A lunar-mission assistant that calls local Python tools to answer
questions mixing fact recall and physics arithmetic. Defines two
tools (`lookup_mission` reading a baked-in record store,
`compute_delta_v` doing a Hohmann transfer), passes them to the
model via `complete(tools=...)`, dispatches `assistant.tool_calls`
to the local functions, and feeds the results back as
`ToolMessage` entries. The agent loop is a graph cycle:
`call_llm -> dispatch_tools -> call_llm` via a conditional edge,
with a hard turn cap to prevent runaway loops. Demonstrates:
`Tool` definitions with JSON Schema parameters,
`complete(tools=...)`, parsing `ToolCall` records,
`ToolMessage(tool_call_id=...)` round-trip, multi-turn
tool-calling loop as a graph cycle.

### Retrieval

#### [`retrieval-rag/`](./retrieval-rag/main.py)

Retrieval-augmented answering over a lunar knowledge base, using the
two-stage retrieve-then-rerank pattern before generation. Batch-embeds a
small corpus once into an index (`OpenAIEmbeddingProvider.embed` over a
list, one vector per passage), then per query embeds the question, ranks
the corpus by cosine similarity for a broad shortlist, reranks that
shortlist with a cross-encoder (`CohereRerankProvider.rerank`) for
precision, and grounds an LLM answer in the reranked top passages. The
retrieve and rerank steps run as graph nodes, so their `EmbeddingEvent` /
`RerankEvent` reach any attached observer. Demonstrates:
`OpenAIEmbeddingProvider` and `CohereRerankProvider` from
`openarmature.retrieval`, the `input_type` query/document knob (a wire
no-op on symmetric OpenAI, meaningful on asymmetric providers), mapping
`ScoredDocument` results back by `.index`, and retrieval feeding an
`OpenAIProvider` answer node. Needs `OPENAI_API_KEY` and `COHERE_API_KEY`.

### Reliability

#### [`checkpointing-and-migration/`](./checkpointing-and-migration/main.py)

A lunar-mission planning pipeline that survives a simulated
mid-pipeline crash, then resumes the saved record under an
upgraded state schema. Phase one invokes a v1 graph against
`MissionPlanStateV1`; the `SQLiteCheckpointer` (JSON mode) writes
records to a temp DB synchronously after every node completes.
`size_crew` raises on its first call to simulate a transient
infrastructure failure; a second invoke with `resume_invocation=`
picks up cleanly. Phase two registers a v1->v2 migration
backfilling a new `risk_assessment` field, builds a v2 graph with
one new node, and resumes from the (now-completed) v1 invocation.
Demonstrates: `SQLiteCheckpointer(serialization="json")`,
`with_checkpointer`, save-on-completed-event, `NodeException` at
the invoke boundary, `State.schema_version`,
`with_state_migration`, `invoke(resume_invocation=...)`.

### Observability

#### [`observer-hooks/`](./observer-hooks/main.py)

Add observability to a `draft -> review -> finalize` pipeline
without changing any node code. Demonstrates: `attach_observer`,
`NodeEvent`, namespace chaining across subgraph boundaries,
function-shaped and class-shaped observers, plus the
`OTelObserver` running alongside the plain observer (same hook,
different backend).

#### [`langfuse-observability/`](./langfuse-observability/main.py)

Send LLM-call observability natively to Langfuse with a
prompt-linkage demonstration on a mission-briefing Q&A pipeline.
Demonstrates: `LangfuseObserver` attached at the graph level,
`LangfusePromptBackend` for prompt fetch, automatic
Generation -> Prompt link via `observability_entities`.

#### [`production-observability/`](./production-observability/main.py)

Dual OTel + Langfuse observers attached to one graph, caller hooks
deriving domain-shaped `trace.input` / `trace.output` from State,
built-in `TimingMiddleware` recording per-node duration via an
`on_complete` callback, multi-tenant caller-supplied metadata
propagating to both observers in one `invoke()` call. The
production-grade observability shape, end-to-end, with in-memory
captures so the demo prints what each backend would have ingested
without needing real production credentials.

## Configuration

All demos configure their LLM client via env vars; OpenAI public-API
defaults shown:

| Env var | Default | Notes |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://api.openai.com` | **Host root only**; the provider adds the path. |
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

# The observer-hooks, multimodal-prompt, chat-with-multimodal,
# and production-observability demos also want the OTel SDK for
# their OTelObserver:
uv sync --group examples --all-extras

# Run any demo:
LLM_API_KEY=sk-... uv run python examples/hello-world/main.py
LLM_API_KEY=sk-... uv run python examples/routing-and-subgraphs/main.py "what year did the moon landing happen"
```

For a local OpenAI-compatible server (vLLM, LM Studio, llama.cpp,
etc.), point `LLM_BASE_URL` at the host root (e.g. `http://localhost:8000`)
and set `LLM_API_KEY` to whatever value the server expects (often
empty or a placeholder).
