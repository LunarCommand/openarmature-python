---
hide:
  - toc
---

# OpenArmature

A workflow framework for LLM pipelines and tool-calling agents. Typed
state, structural graph checks, and observability that doesn't require
buy-in from every node.

[![PyPI](https://img.shields.io/pypi/v/openarmature.svg?color=blue)](https://pypi.org/project/openarmature/){target="_blank" rel="noopener"}
[![spec](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/LunarCommand/openarmature-python/main/pyproject.toml&query=%24.tool.openarmature.spec_version&label=spec&color=9D4EDD)](https://github.com/LunarCommand/openarmature-spec){target="_blank" rel="noopener"}

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/LunarCommand/openarmature-python){ .md-button target="_blank" rel="noopener" }

---

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } &nbsp; __Workflows to agents, one engine__

    ---

    Built for LLM-infused pipelines first. Tool-calling agents (cycle
    back to an LLM node) and pure deterministic ETL sit at the two ends
    of the same spectrum.

-   :material-content-save:{ .lg .middle } &nbsp; __Crash-safe by contract__

    ---

    Synchronous checkpoint save after every node. Process dies
    mid-step, next `invoke(resume_invocation=...)` picks up from the
    last save with `correlation_id` preserved.

-   :material-eye:{ .lg .middle } &nbsp; __Pluggable observability__

    ---

    Native `OTelObserver` emits GenAI semantic conventions any OTLP
    backend renders. Separate `LangfuseObserver` for the Langfuse
    destination. No vendor lock-in to a paid SaaS.

-   :material-graph:{ .lg .middle } &nbsp; __Bad graphs don't compile__

    ---

    `.compile()` rejects six categories of structural error before
    `invoke()` is reachable: dangling edges, unreachable nodes,
    conflicting reducers, no entry, mappings to undeclared state
    fields, multiple outgoing edges.

-   :material-arrow-split-vertical:{ .lg .middle } &nbsp; __Parallelism, formalized__

    ---

    Fan-out with bounded concurrency and per-instance error policy.
    Parallel-branches runs N named subgraphs. Both nest with
    attribution-correct observability.

-   :material-language-python:{ .lg .middle } &nbsp; __Async-first, LLM-agnostic__

    ---

    asyncio-native throughout: every node, observer, and checkpointer
    is `async`. Use any LLM provider, any model, any external system.
    Drops directly into FastAPI lifespan hooks.

</div>

---

## Open specification

OpenArmature is defined by a public, language-agnostic specification,
not a Python-shaped opinion exported to other languages. Reference
implementations share conformance fixtures, so behavior stays identical
across languages, runtimes, and tooling stacks.

[Read the spec →](https://openarmature.org){target="_blank" rel="noopener"}
