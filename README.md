# openarmature-python

Python reference implementation of [OpenArmature](https://github.com/LunarCommand/openarmature-spec) — a workflow framework for LLM pipelines and tool-calling agents.

**Status:** alpha. Implemented against spec v0.8.2.

## Install

Not yet on PyPI. For local use, install from a checkout:

```bash
uv add --editable /path/to/openarmature-python
```

## Quick example

```python
import asyncio
from typing import Annotated

from pydantic import Field

from openarmature.graph import END, GraphBuilder, State, append


class S(State):
    log: Annotated[list[str], append] = Field(default_factory=list)


async def hello(_state: S) -> dict[str, list[str]]:
    return {"log": ["hello"]}


async def world(_state: S) -> dict[str, list[str]]:
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
print(final.log)  # ['hello', 'world']
```

See `tests/conformance/` for fixtures covering conditional routing, subgraph composition, and the canonical compile- and runtime-error categories.

## Spec

The spec lives in [`openarmature-spec`](https://github.com/LunarCommand/openarmature-spec) and is pinned here as a git submodule. Conformance fixtures from the spec are exercised by `tests/conformance/`.

The pinned spec version is recorded in `tool.openarmature.spec_version` (in `pyproject.toml`) and exposed as `openarmature.__spec_version__`.

## License

Apache-2.0.
