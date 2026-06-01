# OpenArmature

[![CI](https://github.com/LunarCommand/openarmature-python/actions/workflows/ci.yml/badge.svg)](https://github.com/LunarCommand/openarmature-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openarmature.svg?color=blue)](https://pypi.org/project/openarmature/)
[![spec](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/LunarCommand/openarmature-python/main/pyproject.toml&query=%24.tool.openarmature.spec_version&label=spec&color=9D4EDD)](https://github.com/LunarCommand/openarmature-spec)
[![python](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/LunarCommand/openarmature-python/main/pyproject.toml&label=python&color=blue)](https://pypi.org/project/openarmature/)
[![License](https://img.shields.io/pypi/l/openarmature.svg)](https://github.com/LunarCommand/openarmature-python/blob/main/LICENSE)

**Documentation:** [openarmature.ai](https://openarmature.ai)

### OpenArmature is a workflow framework for LLM pipelines and tool-calling agents.

Typed state, compile-time topology checks, observability, and crash-safe checkpoints are baked into the engine. The graph layer itself has no concept of LLMs or tools, so the same primitives drive deterministic ETL pipelines and tool-calling agents alike.

This Python package is the reference implementation. The behavioral contract is specified in [openarmature-spec](https://github.com/LunarCommand/openarmature-spec) and verified by conformance fixtures.

## Install

```bash
uv add openarmature                  # core
uv add 'openarmature[otel]'          # with OpenTelemetry observability
# or, with pip:
pip install openarmature
pip install 'openarmature[otel]'
```

## Why OpenArmature

**One framework, from LLM-infused workflows to tool-calling agents.**<br>
OpenArmature is built for LLM pipelines first: extract, classify, route, render, validate, persist, with the LLM dropped in wherever probability beats hand-written rules. Tool-calling agents (graphs that cycle back to an LLM node) and pure deterministic ETL (no LLM at all) sit at the two ends of the same spectrum. The graph engine has zero concept of LLMs, tools, or messages; those live at the node boundary behind a `Provider` Protocol. One platform for the whole gradient, instead of bending agent-shaped frameworks to fit workflow-shaped work.

**Crash-safe resume is first-class, by spec contract.**<br>
Every completed node is followed by a synchronous checkpoint save before the engine advances. Any node fails, the process dies, OOM kill, preemption: the next `invoke(resume_invocation=...)` picks up from the last saved state with a fresh `invocation_id` (audit trail) and the original `correlation_id` preserved (cross-system join). Explicit state-schema migration registration handles old in-flight checkpoints when the schema evolves. Built for preemptible compute, queue workers, and any environment where the process can die mid-step, not just for human-in-the-loop interrupt resume.

**Destination-pluggable observability, not anchored to a paid SaaS.**<br>
`OTelObserver` (in `openarmature[otel]`) emits the OpenTelemetry GenAI semantic conventions (`gen_ai.system`, `gen_ai.request.*`, `gen_ai.response.*`, `gen_ai.usage.*`) that Honeycomb, HyperDX, Phoenix, Datadog APM, Tempo, an open-source Jaeger, or your own OTLP collector all render natively without per-service shims. On top of that, a separate `LangfuseObserver` (in `openarmature[langfuse]`) provides a native mapping for teams who've chosen Langfuse: MIT-licensed, self-hostable, decoupled through a `LangfuseClient` Protocol so swapping it out is a single-file change. No coupling to a closed-source product owned by the framework vendor.

**The graph either compiles or it never runs.**<br>
`.compile()` rejects six categories of structural error before `invoke()` is reachable: unreachable nodes, dangling edges, conflicting reducers, no declared entry, mappings to undeclared state fields, multiple outgoing edges from one node. State schemas are frozen Pydantic models validated at every merge boundary. For a 30-node pipeline with conditional routing, the difference between "tests pass" and "tests pass on today's code path" is structural.

**There's a spec, not just code.**<br>
OpenArmature is defined by a public, language-agnostic [specification](https://github.com/LunarCommand/openarmature-spec) with conformance fixtures every reference implementation must pass. Behavior is bounded by the spec; implementations conform to it. Minor-version surprises around state merge, fan-out collection, or resume semantics live in proposals tracked openly, not in silent code changes between releases.

For the full feature catalog see [openarmature.ai/concepts](https://openarmature.ai/concepts/).

## Hello World

About a hundred lines that show the engine in action. Three reducer policies declared on one state class. Three LLM calls each returning typed structured output (Pydantic class on two, raw JSON Schema dict on the third). Conditional routing as a pure function of state, not a hidden state machine. An observer attached at compile time that sees every node boundary the engine emits. Requires Python 3.12 or later and an OpenAI-compatible endpoint (defaults to OpenAI public API; works against any local server too).

```python
import asyncio
import os
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from openarmature.graph import END, GraphBuilder, NodeEvent, State, append, merge
from openarmature.llm import OpenAIProvider, UserMessage
from pydantic import BaseModel, Field


class Classification(BaseModel):
    intent: Literal["research", "summarize"]
    rationale: str


class Summary(BaseModel):
    one_liner: str
    confidence: float


class PipelineState(State):
    query: str                                                # last_write_wins (default)
    classification: Classification | None = None              # set by classify
    research_plan: dict[str, Any] | None = None               # set by research (dict-schema form)
    summary: Summary | None = None                            # set by summarize
    sources: Annotated[list[str], append] = Field(            # appends across writes
        default_factory=list
    )
    metadata: Annotated[dict[str, str], merge] = Field(       # merges across writes
        default_factory=dict
    )


provider = OpenAIProvider(
    base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),  # host root; impl adds /v1
    model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    api_key=os.environ.get("LLM_API_KEY") or None,                      # empty → no-auth
)


async def classify(state: PipelineState) -> Mapping[str, Any]:
    response = await provider.complete(
        [UserMessage(content=f"Route to 'research' or 'summarize': {state.query!r}")],
        response_schema=Classification,                                  # class → instance
    )
    return {"classification": response.parsed, "metadata": {"classified_by": "llm"}}


async def research(state: PipelineState) -> Mapping[str, Any]:
    response = await provider.complete(
        [UserMessage(content=f"Plan research for {state.query!r}: list topics + follow-ups.")],
        response_schema={                                                # dict → dict
            "type": "object",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}},
                "follow_up_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topics", "follow_up_questions"],
            "additionalProperties": False,
        },
    )
    return {
        "research_plan": response.parsed,
        "sources": ["wikipedia", "arxiv"],
        "metadata": {"tool": "research"},
    }


async def summarize(state: PipelineState) -> Mapping[str, Any]:
    response = await provider.complete(
        [UserMessage(content=f"Summarize {state.query!r} in one sentence with confidence 0-1.")],
        response_schema=Summary,                                         # class → instance
    )
    return {"summary": response.parsed, "sources": ["cache"], "metadata": {"tool": "summarize"}}


def route(state: PipelineState) -> str:
    assert state.classification is not None
    return state.classification.intent


async def trace(event: NodeEvent) -> None:
    if event.phase == "completed" and event.error is None and event.post_state is not None:
        print(f"{event.node_name}: sources={event.post_state.sources}")


graph = (
    GraphBuilder(PipelineState)
    .add_node("classify", classify)
    .add_node("research", research)
    .add_node("summarize", summarize)
    .add_conditional_edge("classify", route)
    .add_edge("research", END)
    .add_edge("summarize", END)
    .set_entry("classify")
    .compile()
)
graph.attach_observer(trace)


async def main() -> None:
    try:
        final = await graph.invoke(PipelineState(query="what is RAG?"))
        print(f"\nclassification: {final.classification}")
        if final.research_plan is not None:
            print(f"research_plan: {final.research_plan}")
        if final.summary is not None:
            print(f"summary: {final.summary}")
    finally:
        await graph.drain()
        await provider.aclose()


asyncio.run(main())
```

Set `LLM_API_KEY=sk-...` and run. To swap providers, point `LLM_BASE_URL` and `LLM_MODEL` at OpenRouter, vLLM, LM Studio, llama.cpp, or anything else that speaks the OpenAI Chat Completions wire format. The example also lives at [`examples/00-hello-world/main.py`](./examples/00-hello-world/main.py); see [`examples/`](./examples/) for more runnable demos.

A few things to notice:

- **Three reducer policies on one state schema.** `query` / `classification` / `research_plan` / `summary` get the default `last_write_wins`. `sources` is `Annotated[list[str], append]`, so successive writes concatenate. `metadata` is `Annotated[dict[str, str], merge]`, so successive writes shallow-merge. The merge policy lives on the schema, once.
- **Structured output, two forms.** `response_schema=Classification` (a Pydantic class) returns `Response.parsed` as a validated `Classification` instance, typed end-to-end. `response_schema={...}` (a raw JSON Schema dict) returns `Response.parsed` as a plain dict. Same wire shape underneath; pick the form that fits.
- **Conditional routing on a parsed field.** `route` reads `state.classification.intent` and returns the next node's name. The graph engine doesn't care the discriminator came from an LLM; it would accept a deterministic rule with the same shape.
- **Observer sees both phases.** `trace` filters to `completed` events for brevity; the engine also delivers `started` events.
- **The graph either compiles or it doesn't.** Remove `.set_entry()` and `.compile()` raises `NoDeclaredEntry` before `invoke()` runs.

## Next steps

- **Quickstart**: build your first graph end-to-end. [openarmature.ai/getting-started](https://openarmature.ai/getting-started/)
- **Concepts**: typed state, reducers, graphs, composition, fan-out, parallel branches, LLMs, prompts, observability, checkpointing. [openarmature.ai/concepts](https://openarmature.ai/concepts/)
- **Model Providers**: implement the Provider Protocol for a custom LLM backend. [openarmature.ai/model-providers/authoring](https://openarmature.ai/model-providers/authoring/)
- **API reference**: auto-generated from docstrings. [openarmature.ai/reference](https://openarmature.ai/reference/)
- **Examples**: ten runnable demos with walk-throughs. [openarmature.ai/examples](https://openarmature.ai/examples/) (source at [./examples/](./examples/))
- **Spec**: behavioral contract this implementation conforms to. [LunarCommand/openarmature-spec](https://github.com/LunarCommand/openarmature-spec)

## For AI agents

If you're an AI agent working in code that uses openarmature, read the bundled agent docs before editing:

```bash
python -c "import openarmature; print(openarmature.__path__[0] + '/AGENTS.md')"
```

Or use the convenience CLI:

```bash
openarmature docs        # print the path to the bundled AGENTS.md
python -m openarmature docs  # same, via the module entry point
```

The file ships with the package and covers capability contracts, common patterns, non-obvious shapes, and an example index. Adopting projects can run `openarmature init` from the project root to append a discovery pointer block into their own `AGENTS.md` / `CLAUDE.md` so agent sessions in their codebase find the bundled file automatically.

The same patterns content is also available programmatically:

```python
import openarmature.patterns as patterns

patterns.list()                          # ['bypass-if-output-exists', ...]
patterns.get('bypass-if-output-exists')  # canonical recipe content (markdown)
```
