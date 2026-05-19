# Quickstart

Build and run a two-node graph in under a minute. No LLM required. This is
the smallest possible openarmature program so you can see every part of the
shape on one screen.

## Install

```bash
uv add openarmature
# or, with pip:
pip install openarmature
```

Requires Python ≥ 3.12.

## A minimal graph

Two nodes (`hello` → `world`), one shared field, the `append` reducer.

```python
import asyncio
from typing import Annotated

from openarmature.graph import END, GraphBuilder, State, append
from pydantic import Field


class S(State):
    log: Annotated[list[str], append] = Field(default_factory=list)


async def hello(_s: S) -> dict[str, list[str]]:
    return {"log": ["hello"]}


async def world(_s: S) -> dict[str, list[str]]:
    return {"log": ["world"]}


graph = (
    GraphBuilder(S)
    .add_node("hello", hello)
    .add_node("world", world)
    .add_edge("hello", "world")
    .add_edge("world", END)
    .set_entry("hello")
    .compile()
)

final = asyncio.run(graph.invoke(S()))
assert final.log == ["hello", "world"]
```

## What just happened

- **`S`** is the state schema, a frozen Pydantic model. Nodes can't mutate
  it; they return partial-update dicts and the engine merges them.
- **`append`** is the reducer attached to `log`. When `hello` returns
  `{"log": ["hello"]}`, the engine *appends* to the existing list rather
  than replacing it.
- **`add_node` + `add_edge`** declare the graph shape; **`END`** is the
  terminal sentinel imported from `openarmature.graph`.
- **`compile()`** runs structural checks at construction time (no dangling
  edges, no unreachable nodes, no duplicate reducers) and returns an
  immutable `CompiledGraph`. Bad shapes fail here, not at run time.
- **`invoke()`** runs the graph from `set_entry()` to `END` and returns the
  final state.

## Next

- [Concepts](../concepts/index.md): deeper on state, reducers, graphs,
  composition, fan-out, parallel branches, LLMs, prompts,
  observability, checkpointing.
- [Examples](../examples/index.md): ten runnable demos with
  walk-throughs, each driving an OpenAI-compatible LLM endpoint to
  do real work.
- [API reference](../reference/index.md): auto-generated from docstrings.
