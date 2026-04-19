# CLAUDE.md

Orientation for Claude Code sessions in this repo. The `README.md` covers what the project is and how to use it; this file covers things that aren't obvious from reading the code.

## Spec is the source of truth

This repo is a Python implementation of [`openarmature-spec`](https://github.com/LunarCommand/openarmature-spec). Behavior is defined by the spec; this repo executes it.

- The spec lives at `openarmature-spec/` as a git submodule pinned to a released tag. Don't edit files in the submodule.
- To bump the spec: `cd openarmature-spec && git checkout <tag>`, then bump the three places that track the spec version (see below).
- Behavior changes that aren't already in the spec require a proposal in the spec repo first, not a PR here.

## Three places hold the spec version — keep them in sync

- `tool.openarmature.spec_version` in `pyproject.toml`
- `__spec_version__` in `src/openarmature/__init__.py`
- The submodule commit (must match a released tag, e.g. `v0.1.1`)

`tests/test_smoke.py` asserts the first two match. The third is enforced by convention.

## Test layout

- `tests/conformance/` — runs the spec's YAML fixtures against the engine via an adapter. Drives most of the behavior coverage.
- `tests/unit/` — fills coverage gaps the conformance suite doesn't reach: `edge_exception`, `reducer_error`, `state_validation_error`, `SubgraphNode.run`, projection variants, frozen-state mutation, etc.
- `tests/test_smoke.py` — version sync.

## Tooling

- `uv` for everything. Don't use `pip` directly.
- Pyright **strict mode** is enforced (`pyproject.toml`). Annotations are not optional.
- Ruff for lint + format. Pre-commit hook runs `ruff format` automatically — the file you committed may not be the file in the next diff.
- `pytest-asyncio` with `asyncio_mode = "auto"` — `async def test_...` works with no decorator.

## Common commands

```bash
uv run pytest -q                          # all tests
uv run pytest tests/conformance/ -v       # spec conformance only
uv run ruff check . && uv run ruff format # lint + format
uv run pyright src/ tests/                # type check
```

## Engine design notes that are easy to miss

- `State` is `frozen=True` AND `extra="forbid"`. Nodes that return an undeclared field surface as a `state_validation_error`, not a silent drop.
- Conditional edges over-approximate at compile time (a conditional from node X is treated as reaching every node), so the unreachable-node check is sound but not tight.
- Each node has exactly one outgoing edge. Branching is via conditional edges, not multiple statics.
- `END` is a distinct sentinel object, not a reserved string. Use the exported `END` constant.
