"""Minimal openarmature demo: 2-node graph (plan → write) driven by a local vLLM.

**Use case:** Take a topic (e.g. "the psychology of long walks") and produce
a short written piece — first plan a few angles, then write the article.

**Demonstrates:** The minimal graph shape — typed `State`, the `append`
reducer, static edges, `END`, a two-node linear `plan → write` pipeline.

Run with:
    uv run python main.py "the psychology of long walks"
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Annotated, Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import Field

from openarmature.graph import END, CompiledGraph, GraphBuilder, State, append

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ"

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")


# ----------------------------------------------------------------------------
# State schema
# ----------------------------------------------------------------------------
# `State` is the immutable, strictly-typed object that flows through the graph.
# Every node receives an instance and returns a partial-update dict; the engine
# merges each update via per-field reducers and re-validates the result.
#
# Why this shape? The graph is a pipeline of pure state transitions. Nodes
# read a snapshot, emit a diff, and never mutate shared state — which makes
# "what was state when node X ran?" a question with a single clean answer.
# Under the hood, `State` is a Pydantic BaseModel with
# `model_config = ConfigDict(frozen=True, extra="forbid")` pre-baked into the
# base class — you don't touch `model_config` yourself.
#
# Non-obvious bits (see _docs/concepts.md for more):
#   - Instances are FROZEN — nodes can't mutate `s.plan = ...`. They return
#     {"plan": ...} and the engine applies the reducer.
#   - Extra fields are FORBIDDEN — a node that returns {"typo": 1} raises
#     StateValidationError instead of silently dropping the key.
#   - Reducers attach per-field via Annotated[T, reducer]. Unannotated fields
#     default to `last_write_wins` (new value replaces old).
#   - Mutable defaults (list/dict) need Field(default_factory=...) — pydantic
#     gotcha, not openarmature-specific.


class GraphState(State):
    topic: str
    plan: str = ""
    output: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# LLM client helper (not openarmature — just consumer plumbing)
# ----------------------------------------------------------------------------
# Nothing below is graph-engine-specific; it's an OpenAI-compatible call to
# the local vLLM. Skim past this to the node functions if you're reading for
# openarmature concepts.


async def _chat(system: str, user: str) -> str:
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=user),
    ]
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()


# ----------------------------------------------------------------------------
# Node functions
# ----------------------------------------------------------------------------
# A node is `async def name(state) -> dict-of-partial-updates`. It reads the
# immutable state snapshot it was handed and returns *only* the fields it
# wants to change — the engine merges each field via its reducer.
#
# The idea: nodes are dumb and local. "I produced this plan." "I added this
# line to the trace." A node doesn't construct the next state, doesn't know
# what ran before it, and doesn't know what runs after. The graph is what
# composes them; each node stays a plain async function you could unit-test
# in isolation.
#
# For reducer-tracked fields (e.g. `trace: Annotated[list[str], append]`),
# return only the increment — `{"trace": ["plan"]}`, NOT
# `{"trace": s.trace + ["plan"]}`. The `append` reducer does the
# concatenation; returning the full list causes duplication.


async def plan_node(s: GraphState) -> Mapping[str, Any]:
    content = await _chat(
        system="You are a concise outliner. Respond with exactly 3 bullet points, no preamble.",
        user=f"Outline the following topic in 3 bullets:\n\n{s.topic}",
    )
    return {"plan": content, "trace": ["plan"]}


async def write_node(s: GraphState) -> Mapping[str, Any]:
    content = await _chat(
        system="You are an essayist. Write in prose, one tight paragraph, no bullet points, no headers.",
        user=(
            f"Topic: {s.topic}\n\nOutline:\n{s.plan}\n\n"
            "Write a short paragraph (~120 words) that expands on the outline."
        ),
    )
    return {"output": content, "trace": ["write"]}


# ----------------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------------
# `GraphBuilder` is a mutable builder; `.compile()` turns it into an
# immutable `CompiledGraph` that's ready to run. The chain below does four
# things:
#   1. Register nodes under names — `.add_node("plan", plan_node)`. The name
#      is what you reference in edges; the fn is the async callable.
#   2. Connect them with static edges — `.add_edge("plan", "write")` means
#      "after `plan` runs and its update is merged, go to `write`".
#   3. Terminate with END — `.add_edge("write", END)` marks `write` as the
#      last step, so the engine halts and `invoke()` returns.
#   4. Declare the entry point — `.set_entry("plan")` — where execution
#      begins.
#
# Then `.compile()` runs the structural checks (no unreachable nodes, no
# dangling edges, no duplicate reducers on a field, etc.) and returns a
# `CompiledGraph`. Any problem with the graph's shape surfaces HERE, not at
# runtime — failing fast at the construction boundary.
#
# Each node has exactly ONE outgoing edge. Branching isn't done with multiple
# static edges; it's done with a single `.add_conditional_edge(source, fn)` where
# `fn(state) -> next_node_name`. We're linear here, so no conditional.
#
# `END` is a sentinel object, not a reserved string. Import it from
# `openarmature.graph`; don't use the literal `"END"` — it would be treated
# as a node name and fail `DanglingEdge` at compile time.


def build_graph() -> CompiledGraph[GraphState]:
    return (
        GraphBuilder(GraphState)
        .add_node("plan", plan_node)
        .add_node("write", write_node)
        .add_edge("plan", "write")
        .add_edge("write", END)
        .set_entry("plan")
        .compile()
    )


# ----------------------------------------------------------------------------
# Invoking the graph
# ----------------------------------------------------------------------------
# `graph.invoke(initial_state)` runs the compiled graph from entry to END
# and returns the final state. It's `async` because nodes can (and usually
# do) perform IO.
#
# Initial state: construct an instance of your state class with the required
# fields filled in. Here that's just `GraphState(topic=topic)` — the other
# fields have schema-level defaults, so when `plan` (the entry node) is
# called it sees empty `plan`, empty `output`, and empty `trace`.
#
# `GraphBuilder` and `CompiledGraph` are generic on the state type —
# `build_graph()` is annotated as returning `CompiledGraph[GraphState]`, so
# `await graph.invoke(...)` returns a `GraphState`, not the base `State`.
# Typed field access (`final.topic`) works without a `cast()` on the return.
# (Earlier versions of the library required `cast(GraphState, ...)`; see
# `_docs/rough-edges.md` for the history.)


async def main() -> None:
    topic = " ".join(sys.argv[1:]) or "the psychology of long walks"
    graph = build_graph()
    final = await graph.invoke(GraphState(topic=topic))

    print(f"topic: {final.topic}\n")
    print(f"plan:\n{final.plan}\n")
    print(f"output:\n{final.output}\n")
    print(f"trace: {final.trace}")


if __name__ == "__main__":
    asyncio.run(main())
