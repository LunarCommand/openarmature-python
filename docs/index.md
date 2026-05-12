---
hide:
  - navigation
  - toc
---

# OpenArmature

A workflow framework for LLM pipelines and tool-calling agents — typed
state, structural graph checks, and observability that doesn't require
buy-in from every node.

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/LunarCommand/openarmature-python){ .md-button target="_blank" rel="noopener" }

---

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } &nbsp; __Typed, frozen state__

    ---

    State schemas are Pydantic models with `frozen=True` and
    `extra="forbid"`. Nodes can't mutate state — they return partial
    updates and the engine merges via per-field reducers.

-   :material-graph:{ .lg .middle } &nbsp; __Compile-time checks__

    ---

    Bad graph shapes (dangling edges, unreachable nodes, conflicting
    reducers, missing entry) fail at `.compile()`, not at run time.

-   :material-eye:{ .lg .middle } &nbsp; __Observable, opt-in__

    ---

    Attach an `Observer` to see every node boundary. Drop in the
    optional OTel mapping for spans + log correlation; logs carry
    `trace_id` / `span_id` / `correlation_id` automatically.

-   :material-content-save:{ .lg .middle } &nbsp; __Checkpointable__

    ---

    In-memory and SQLite `Checkpointer` backends ship in core. Crash at
    node N+1, resume from node N's saved state on the next invocation.

-   :material-arrow-split-vertical:{ .lg .middle } &nbsp; __First-class fan-out__

    ---

    Per-instance fan-out with bounded concurrency, error-policy choice,
    and observability events that attribute correctly per instance.

-   :material-language-python:{ .lg .middle } &nbsp; __Async-first, LLM-agnostic__

    ---

    The engine has no concept of LLMs or tools — those live at the node
    boundary. Use any provider, any model, any external system.

</div>

---

Built around an open, language-agnostic
[specification](https://github.com/LunarCommand/openarmature-spec).
A TypeScript implementation is on the roadmap; behaviour stays
identical across implementations via spec conformance fixtures.
