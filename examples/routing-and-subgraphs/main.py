"""openarmature demo: conditional routing + subgraph with a custom projection.

**Use case:** A question-answering assistant. Classify the question, then
either give a one-shot quick answer or run a multi-step research
sub-pipeline (plan angles → gather notes → synthesize), then lightly
copy-edit the result.

**Demonstrates:** Conditional edges (state-driven routing) via
`add_conditional_edge`, subgraph composition via `add_subgraph_node`, a
custom `ProjectionStrategy` for the parent ↔ subgraph boundary, and the
`merge` reducer for dict accumulation.

Three graph features that `hello-world` only touched lightly:

  1. **Conditional edges.** The entry node classifies the question and the
     graph routes to one of two branches based on that classification.
  2. **Subgraphs.** One of those branches is an entire sub-graph (plan →
     gather → synthesize) wrapped as a single node in the parent graph.
  3. **A custom `ProjectionStrategy`.** The default projection (`FieldNameMatching`)
     doesn't carry parent state *into* the subgraph; the subgraph starts
     from its own schema's defaults. To pass the user's question in (and
     shape what comes back out), we write a `ProjectionStrategy` by hand.

LLM calls go through ``openarmature.llm.OpenAIProvider`` (same pattern as
``hello-world``) so the example reads as the recommended path rather
than as "openai with some openarmature on top."

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root
  only**; the provider adds the path itself.
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/routing-and-subgraphs
    LLM_API_KEY=sk-... uv run python main.py "what year did the moon landing happen"
    LLM_API_KEY=sk-... uv run python main.py "why is the lunar south pole strategically important?"
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    ProjectionStrategy,
    State,
    append,
    merge,
)
from openarmature.llm import OpenAIProvider, SystemMessage, UserMessage

# Lazy-initialized so importing this module (test harnesses, doc builders,
# IDE inspection) doesn't open an httpx.AsyncClient connection pool.
_provider_instance: OpenAIProvider | None = None


def _get_provider() -> OpenAIProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OpenAIProvider(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("LLM_API_KEY") or None,
        )
    return _provider_instance


# ----------------------------------------------------------------------------
# State schemas: one for the outer graph, one for the subgraph
# ----------------------------------------------------------------------------
# The outer graph and the subgraph each have their OWN `State` subclass. This
# is one of openarmature's stronger opinions: a subgraph isn't a namespace
# inside the parent's schema; it's a separate pipeline with its own field
# shape. The boundary between them is an explicit translation step
# (a `ProjectionStrategy`), not implicit aliasing.
#
# Why separate schemas? Two reasons:
#   - Subgraphs are reusable. The research sub-pipeline could be dropped into
#     a different outer graph (or run on its own) without the parent's fields
#     leaking into its type.
#   - Boundaries are auditable. To find "what does the subgraph see?" you read
#     one projection class, not a scattered naming convention.
#
# Both schemas below use the standard reducer set: `append` on the
# `trace` list, `merge` on a dict. Fields without an
# `Annotated[..., reducer]` get `last_write_wins` by default.


class AssistantState(State):
    """Outer graph: takes a question, routes it, returns a formatted answer."""

    question: str  # required input
    route: str = ""  # set by `classify`; read by the conditional edge
    answer: str = ""  # set by whichever branch ran, then polished by `format_final`
    trace: Annotated[list[str], append] = Field(default_factory=list)
    tallies: Annotated[dict[str, int], merge] = Field(default_factory=dict)


class ResearchState(State):
    """Subgraph: takes a question, produces a synthesized answer."""

    question: str = ""  # projected IN from the parent (see `QuestionProjection`)
    angles: list[str] = Field(default_factory=list)  # 3 angles picked by `plan_research`
    notes: dict[str, str] = Field(default_factory=dict)  # angle → note, produced by `gather`
    answer: str = ""  # final synthesis
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ----------------------------------------------------------------------------
# LLM helper
# ----------------------------------------------------------------------------
# Thin wrapper over Provider.complete that takes a system + user pair and
# returns the assistant's reply as a string. Keeps the node bodies focused
# on graph logic (state in → state update out) rather than provider
# plumbing. Production code would typically inline the call.


async def _chat(system: str, user: str) -> str:
    response = await _get_provider().complete(
        [SystemMessage(content=system), UserMessage(content=user)],
    )
    return (response.message.content or "").strip()


# ----------------------------------------------------------------------------
# Outer-graph nodes
# ----------------------------------------------------------------------------
# Standard node shape: `async def(state) -> dict`, returning ONLY the
# fields it wants to change. The engine applies per-field reducers and
# re-validates.
#
# Three things worth noticing as you read these:
#
#   (a) `classify` returns `route`. The conditional edge function further down
#       reads that field off the post-merge state and dispatches. The fact
#       that "routing decision" lives as a normal state field (not as some
#       special return channel) is important: it's visible, it's typed, it's
#       part of the trace, and you can inspect it in every downstream node.
#
#   (b) Every node contributes to `tallies` via the `merge` reducer. Each
#       return dict carries a small `{"tallies": {...}}` fragment; the
#       reducer accumulates them into one dict on the final state. This is
#       the same pattern used for metrics/counts across a pipeline:
#       compose by emitting fragments, not by read-modify-write.
#
#   (c) No node calls a subsequent node. `classify` doesn't know whether
#       `quick_answer` or the research subgraph runs next. Nodes stay
#       local; the graph decides the shape.


async def classify(s: AssistantState) -> Mapping[str, Any]:
    """Pick a branch: 'quick' for anything answerable off-the-cuff, 'research' otherwise."""
    content = await _chat(
        system=(
            "You are a router. Read the question and answer with exactly one word: "
            "'quick' if it can be answered in a sentence or two of general knowledge, "
            "'research' if it benefits from considering multiple angles. No punctuation."
        ),
        user=s.question,
    )
    route = "quick" if "quick" in content.lower() else "research"
    return {
        "route": route,
        "trace": ["classify"],
        "tallies": {"classify_calls": 1},
    }


async def quick_answer(s: AssistantState) -> Mapping[str, Any]:
    """Fast path: one LLM call, direct answer."""
    content = await _chat(
        system="Answer the question directly in one or two sentences. No preamble.",
        user=s.question,
    )
    return {
        "answer": content,
        "trace": ["quick_answer"],
        "tallies": {"quick_answers": 1},
    }


async def format_final(s: AssistantState) -> Mapping[str, Any]:
    """Final polish: take whatever the branch produced and tighten it."""
    content = await _chat(
        system="Lightly copy-edit this answer for clarity. Preserve meaning. Return only the edited text.",
        user=s.answer,
    )
    return {
        "answer": content,
        "trace": ["format_final"],
        "tallies": {"formatted": 1},
    }


# ----------------------------------------------------------------------------
# Conditional edge function
# ----------------------------------------------------------------------------
# The conditional edge fn is called with the state AFTER `classify`'s update
# has been merged; so `s.route` reflects what `classify` wrote. The return
# value MUST be a declared node name (as a string) OR the `END` sentinel.
# Anything else raises `RoutingError` at runtime.
#
# Small but important: the fn is synchronous. It's a routing decision, not a
# place to do IO. (If you need async routing logic, do it in the producing
# node and write the decision into a state field; which is exactly what
# `classify` does here.)
#
# Default case: we fall back to "quick_answer" if classify returned something
# unexpected. We could also return `END` to halt, or route to a dedicated
# error node. This is a design knob, not a library rule.


def route_from_classification(s: AssistantState) -> str:
    if s.route == "research":
        return "research"
    return "quick_answer"


# ----------------------------------------------------------------------------
# Subgraph: research pipeline (plan_research → gather → synthesize)
# ----------------------------------------------------------------------------
# The subgraph is itself a full openarmature graph. It has its own state
# class (`ResearchState`), its own nodes, its own edges, its own entry. When
# compiled, it becomes a `CompiledGraph`; and a `CompiledGraph` can be
# wrapped as a node in an outer graph via `builder.add_subgraph_node(...)`.
#
# Why use a subgraph here at all? You could flatten these three nodes into
# the outer graph; plan_research, gather, synthesize as peers of classify
# and quick_answer. You'd lose two things:
#
#   1. **Encapsulation.** The outer graph cares about "research produces
#      an answer." It doesn't need to know how. The subgraph hides the
#      plan → gather → synthesize shape; swapping its implementation (e.g.
#      add a "verify" step later) doesn't touch the outer wiring.
#
#   2. **Reusability.** This compiled subgraph is a plain Python value. The
#      same `research_subgraph` could be used from a different outer graph,
#      invoked directly for testing (`await research_subgraph.invoke(...)`),
#      or composed inside yet another subgraph.
#
# The `ResearchState` is intentionally narrower than `AssistantState`. The
# subgraph shouldn't know or care about `route` or `tallies`; those are
# outer-graph concerns. This is what separate schemas buys you.


async def plan_research(s: ResearchState) -> Mapping[str, Any]:
    """Pick 3 angles to explore."""
    content = await _chat(
        system=(
            "Given a question, propose 3 distinct angles worth investigating. "
            "Respond with exactly 3 lines, one angle per line, no numbering or bullets."
        ),
        user=s.question,
    )
    angles = [line.strip(" -*•\t") for line in content.splitlines() if line.strip()][:3]
    return {
        "angles": angles,
        "trace": ["plan_research"],
    }


async def gather(s: ResearchState) -> Mapping[str, Any]:
    """Produce a short note for each angle. One LLM call, formatted result."""
    angles_joined = "\n".join(f"- {a}" for a in s.angles)
    content = await _chat(
        system=(
            "For each angle below, write a 1-2 sentence note that speaks to it. "
            "Format your response as:\n"
            "ANGLE: <angle>\n"
            "NOTE: <note>\n\n"
            "Repeat for each angle. No preamble."
        ),
        user=f"Question: {s.question}\n\nAngles:\n{angles_joined}",
    )
    # Parse the ANGLE/NOTE blocks into a dict. Robust to extra whitespace; if
    # the model goes off-script we fall back to a single catch-all note.
    notes: dict[str, str] = {}
    current_angle: str | None = None
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("ANGLE:"):
            current_angle = line[len("ANGLE:") :].strip()
        elif line.upper().startswith("NOTE:") and current_angle is not None:
            notes[current_angle] = line[len("NOTE:") :].strip()
            current_angle = None
    if not notes:
        notes = {"general": content[:400]}
    return {
        "notes": notes,
        "trace": ["gather"],
    }


async def synthesize(s: ResearchState) -> Mapping[str, Any]:
    """Combine angle notes into a short paragraph answer."""
    notes_joined = "\n".join(f"- {a}: {n}" for a, n in s.notes.items())
    content = await _chat(
        system=(
            "Synthesize the notes below into one tight paragraph (~100 words) that "
            "answers the question. No bullets, no headers."
        ),
        user=f"Question: {s.question}\n\nNotes:\n{notes_joined}",
    )
    return {
        "answer": content,
        "trace": ["synthesize"],
    }


def build_research_subgraph() -> CompiledGraph[ResearchState]:
    return (
        GraphBuilder(ResearchState)
        .add_node("plan_research", plan_research)
        .add_node("gather", gather)
        .add_node("synthesize", synthesize)
        .add_edge("plan_research", "gather")
        .add_edge("gather", "synthesize")
        .add_edge("synthesize", END)
        .set_entry("plan_research")
        .compile()
    )


# ----------------------------------------------------------------------------
# Custom projection: wiring parent ↔ subgraph
# ----------------------------------------------------------------------------
# `FieldNameMatching` (the default) does one thing well and one thing not at
# all:
#
#   - `project_out`: GOOD. It looks at the subgraph's final state, picks the
#     fields whose names also exist on the parent, and returns them as a
#     partial update; the parent's reducers then merge. That's how the
#     subgraph's `trace` list flows back into the outer `trace` via the
#     outer's `append` reducer.
#
#   - `project_in`: DELIBERATELY LIMITED. It builds a fresh subgraph state
#     from its schema's defaults; `subgraph_state_cls()`. The parent's
#     state is ignored. Subgraphs don't see the outer world unless the
#     author opts in; encapsulation is the point.
#
# For this demo we absolutely need the question in the subgraph. So we write
# a projection class that implements the `ProjectionStrategy` Protocol (see
# `openarmature.graph.projection`). Two methods: `project_in` decides what
# the subgraph starts with; `project_out` decides what leaks back.
#
# Teaching moment: this is the pattern for ALL non-trivial subgraph use. The
# default is fine for "inner computation shares the parent's field names"
# cases; anything else needs a custom projection. There's no runtime check
# that this is well-formed; a projection that returns a field the parent
# doesn't declare will surface at the boundary as a `StateValidationError`
# (extra="forbid" on the parent catches it).


# noinspection PyMethodMayBeStatic
class QuestionProjection:
    """Pass `question` INTO the subgraph; pull `answer` and `trace` OUT.

     Signatures are typed directly against `AssistantState` and `ResearchState`
    ; `ProjectionStrategy[ParentT, ChildT]` is a generic Protocol, so
     structural conformance is checked at the `_: ProjectionStrategy[...]`
     annotation below without inheritance.
    """

    def project_in(
        self, parent_state: AssistantState, subgraph_state_cls: type[ResearchState]
    ) -> ResearchState:
        # Construct the subgraph's initial state with the parent's question.
        # All other subgraph fields use their schema defaults.
        return subgraph_state_cls(question=parent_state.question)

    # noinspection PyUnusedLocal
    def project_out(
        self,
        subgraph_final_state: ResearchState,
        parent_state: AssistantState,
        subgraph_state_cls: type[ResearchState],
    ) -> Mapping[str, Any]:
        # Bring `answer` back; merged via parent's `last_write_wins`.
        # Bring `trace` back; merged via parent's `append` reducer, which
        # concatenates the subgraph's trace entries after the parent's.
        # Bump a tally so we can see the research branch ran.
        return {
            "answer": subgraph_final_state.answer,
            "trace": subgraph_final_state.trace,
            "tallies": {"research_runs": 1},
        }


# Static type check: a `QuestionProjection` instance satisfies the generic
# `ProjectionStrategy[AssistantState, ResearchState]` Protocol. This line is
# a no-op at runtime but catches shape drift at type-check time.
_: ProjectionStrategy[AssistantState, ResearchState] = QuestionProjection()


# ----------------------------------------------------------------------------
# Outer graph construction
# ----------------------------------------------------------------------------
# Four things to notice below:
#
#   1. `.add_subgraph_node("research", ..., projection=QuestionProjection())`:
#      this is the only new method on `GraphBuilder` vs the hello-world
#      example. It registers a compiled graph as a node, under the given
#      name, with the given projection.
#
#   2. `.add_conditional_edge("classify", route_from_classification)`; the
#      conditional edge. Exactly one outgoing edge per node still applies;
#      a conditional IS that one edge. Compile will fail with
#      `MultipleOutgoingEdges` if you mix a static and a conditional from
#      the same source.
#
#   3. Both branches (`quick_answer` and `research`) merge back into
#      `format_final`. You can fan out and fan in freely; the single-
#      outgoing-edge rule is per node, not "no multiple predecessors".
#
#   4. `.compile()` at the end runs all the same structural checks as
#      before; PLUS the reachability check understands conditional edges
#      (conservatively: a conditional from X is treated as reaching every
#      node, which keeps the unreachable check sound).


def build_graph() -> CompiledGraph[AssistantState]:
    research_subgraph = build_research_subgraph()

    return (
        GraphBuilder(AssistantState)
        .add_node("classify", classify)
        .add_node("quick_answer", quick_answer)
        .add_subgraph_node("research", research_subgraph, projection=QuestionProjection())
        .add_node("format_final", format_final)
        .add_conditional_edge("classify", route_from_classification)
        .add_edge("quick_answer", "format_final")
        .add_edge("research", "format_final")
        .add_edge("format_final", END)
        .set_entry("classify")
        .compile()
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "why is the lunar south pole strategically important?"
    graph = build_graph()
    try:
        final = await graph.invoke(AssistantState(question=question))
        print(f"question: {final.question}")
        print(f"route:    {final.route}")
        print()
        print(f"answer:\n{final.answer}")
        print()
        print(f"trace:   {final.trace}")
        print(f"tallies: {final.tallies}")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
