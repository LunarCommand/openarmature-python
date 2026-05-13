# openarmature

[![CI](https://github.com/LunarCommand/openarmature-python/actions/workflows/ci.yml/badge.svg)](https://github.com/LunarCommand/openarmature-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openarmature.svg)](https://pypi.org/project/openarmature/)
[![spec](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/LunarCommand/openarmature-python/main/pyproject.toml&query=%24.tool.openarmature.spec_version&label=spec&color=9D4EDD)](https://github.com/LunarCommand/openarmature-spec)
[![Python versions](https://img.shields.io/pypi/pyversions/openarmature.svg)](https://pypi.org/project/openarmature/)
[![License](https://img.shields.io/pypi/l/openarmature.svg)](https://github.com/LunarCommand/openarmature-python/blob/main/LICENSE)

**Documentation:** [openarmature.ai](https://openarmature.ai)

OpenArmature is a workflow framework for LLM pipelines and tool-calling
agents — typed state, compile-time topology checks, and observability
and crash-safe checkpoints baked into the engine. The graph layer
itself has no concept of LLMs or tools, so the same primitives drive
deterministic ETL pipelines and tool-calling agents alike.

This Python package is the reference implementation; the behavioral
contract is specified in [openarmature-spec](https://github.com/LunarCommand/openarmature-spec)
and verified by conformance fixtures.

## Install

```bash
uv add openarmature                  # core
uv add 'openarmature[otel]'          # with OpenTelemetry observability
# or, with pip:
pip install openarmature
pip install 'openarmature[otel]'
```

## Why OpenArmature

**State you can't accidentally mutate.**
State schemas are frozen Pydantic models. Nodes return partial
updates; the engine merges. The snapshot a node holds can't change
mid-execution, and assignment into state raises rather than silently
writing.

**Schema validation at every merge.**
Fields outside the declared schema fail at the merge boundary instead
of silently dropping. A node returning `{"plann": "..."}` (typo)
raises `StateValidationError` immediately, not three nodes downstream
when the field is read and doesn't exist.

**Merge policy on the schema, not the call site.**
Each state field declares its reducer (`last_write_wins`, `append`,
`merge`, or a user-defined callable) as part of the schema. Two nodes
writing the same field compose via the field's policy — once,
declaratively, instead of duplicated across call sites.

**Subgraphs compose with explicit data seams.**
Subgraphs run against their own state schema with `inputs` (additive
— opt in to share parent fields) and `outputs` (replacement — name
exactly what comes back) mappings. Parent fields don't leak in by
accident; subgraph fields don't slip out unless declared.

**Bad graphs don't compile.**
Dangling edges, unreachable nodes, conflicting reducers, no declared
entry, mappings to undeclared fields, multiple outgoing edges from
one node — six categories of structural error all fail at
`.compile()`, not at runtime mid-execution. The graph either
constructs cleanly or it doesn't reach `invoke()`.

**The graph engine has no concept of LLMs or tools.**
Validation, retry, recovery, structured output — those are
node-internal or middleware concerns. The same engine runs
deterministic ETL pipelines and tool-calling agents; the topology
layer doesn't pick a side.

**Determinism is a contract.**
Same input, same node implementations, same edge functions → same
final state and same observed node-execution order. The spec mandates
it; conformance fixtures verify it across every implementation.
Replay an audit run and get byte-identical state.

**Checkpoint saves are synchronous-by-contract.**
The engine awaits each save before advancing — a crash immediately
after a `completed` event cannot have lost the corresponding write.
Resume mints a fresh `invocation_id` (audit trail) while preserving
`correlation_id` (cross-system join key), so a recovered run is
traceable as a new attempt without losing the thread to the original
request.

**Observability that doesn't double-export.**
The OpenTelemetry mapping mandates a private `TracerProvider` —
preventing the trap where global-provider auto-instrumentation
libraries (OpenInference, Langfuse v3, etc.) emit duplicate spans
alongside the framework's. Your spans flow exactly where you point
them; no surprise fan-out to vendor backends you didn't configure.

## Hello World

A three-node classification pipeline. Three different reducer
policies on one state class, conditional routing as a pure function
of state, and an observer that sees every node boundary — without
any LLM setup. Requires Python ≥ 3.12.

```python
import asyncio
from typing import Annotated

from openarmature.graph import (
    END,
    GraphBuilder,
    NodeEvent,
    State,
    append,
    merge,
)
from pydantic import Field


class PipelineState(State):
    query: str                                                # last_write_wins (default)
    classification: str = ""                                  # last_write_wins
    sources: Annotated[list[str], append] = Field(            # appends across writes
        default_factory=list
    )
    metadata: Annotated[dict[str, str], merge] = Field(       # merges across writes
        default_factory=dict
    )


async def classify(state: PipelineState) -> dict:
    decision = "research" if "?" in state.query else "summarize"
    return {
        "classification": decision,
        "metadata": {"classified_by": "rule"},
    }


async def research(state: PipelineState) -> dict:
    return {
        "sources": ["wikipedia", "arxiv"],
        "metadata": {"tool": "search"},
    }


async def summarize(state: PipelineState) -> dict:
    return {
        "sources": ["cache"],
        "metadata": {"tool": "summarizer"},
    }


def route(state: PipelineState) -> str:
    return state.classification


async def trace(event: NodeEvent) -> None:
    if event.phase == "completed" and event.error is None:
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

final = asyncio.run(graph.invoke(PipelineState(query="what is RAG?")))
# classify: sources=[]
# research: sources=['wikipedia', 'arxiv']
```

A few things to notice in ~40 lines:

- **Three reducer policies on one state schema.** `query` and
  `classification` get the default `last_write_wins`. `sources` is
  `Annotated[list[str], append]` — successive writes concatenate.
  `metadata` is `Annotated[dict[str, str], merge]` — successive
  writes shallow-merge. The merge policy lives on the schema, once.
- **Conditional routing as a state function.** `route` reads
  `state.classification` and returns a node name. The graph engine
  doesn't care that this happens to be deterministic; it would
  accept an LLM-driven router with the same shape.
- **Observer sees both phases.** `trace` filters to `completed` events
  for brevity; the engine also delivers `started` events.
- **The graph either compiles or it doesn't.** Remove `.set_entry()`
  and `.compile()` raises `NoDeclaredEntry` before `invoke()` runs.

## Next steps

- **Quickstart** — build your first graph end-to-end:
  [openarmature.ai/getting-started](https://openarmature.ai/getting-started/)
- **Concepts** — typed state, reducers, composition, fan-out,
  checkpointing, observability:
  [openarmature.ai/concepts](https://openarmature.ai/concepts/)
- **Model Providers** — implement the Provider Protocol for a
  custom LLM backend:
  [openarmature.ai/model-providers/authoring](https://openarmature.ai/model-providers/authoring/)
- **API reference** — auto-generated from docstrings:
  [openarmature.ai/reference](https://openarmature.ai/reference/)
- **Examples** — runnable demos:
  [openarmature-python/examples/](https://github.com/LunarCommand/openarmature-python/tree/main/examples)
- **Spec** — behavioral contract this implementation conforms to:
  [LunarCommand/openarmature-spec](https://github.com/LunarCommand/openarmature-spec)
