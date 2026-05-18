"""openarmature demo: a lunar-mission assistant that calls local Python
functions as tools to answer fact and physics questions about Apollo /
Artemis missions.

**Use case:** A user asks something that mixes a factual recall ("when
did Apollo 13 splash down?") with a small computation ("what's the
delta-v for a Hohmann transfer from a 300 km Earth orbit to lunar
distance?"). Neither belongs in the model's prompt — facts get stale and
arithmetic is unreliable from the model alone — so the agent defines two
local tools and lets the model call them.

The agent loops: send messages + tools to the model, dispatch any
``tool_calls`` the model emits, feed the results back as
``ToolMessage`` entries, and call the model again. Loop terminates
when the assistant message has no ``tool_calls`` (the model is done
requesting tools) or after a hard turn cap.

**What's interesting in the implementation:**

- ``Tool(name, description, parameters)`` defines each function as a
  JSON Schema for the model. Both tools below use the standard
  ``type: object`` shape with ``required`` properties; the model
  receives this through ``complete(messages, tools=TOOLS)`` and
  decides which (if any) to invoke.
- The model's response carries ``finish_reason="tool_calls"`` and
  populates ``response.message.tool_calls`` with parsed
  ``ToolCall(id, name, arguments)`` records. The framework guarantees
  ``arguments`` is a parsed dict matching the tool's parameters
  schema (or ``None`` only under ``finish_reason="error"``).
- The dispatcher node parses each ``ToolCall``, runs the matching
  local Python function, and appends one
  ``ToolMessage(content=..., tool_call_id=...)`` per call. Spec
  requires the ``tool_call_id`` round-trip exactly so the model can
  pair its requests with the responses.
- The loop is just a conditional edge on the graph: ``call_llm`` →
  ``dispatch_tools`` → back to ``call_llm`` when the model wants
  more tools, or → ``present`` when it's done. No special "agent
  framework" abstraction — tool-calling composes with the existing
  graph mechanics.
- A ``MAX_TURNS`` cap prevents runaway loops if a model stays in
  tool-calling forever. Production agents typically pair the cap with
  an explicit termination tool or a fallback summarization step.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/09-tool-use
    LLM_API_KEY=sk-... uv run python main.py
    LLM_API_KEY=sk-... uv run python main.py "When was Apollo 17 launched?"
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    State,
    append,
)
from openarmature.llm import (
    AssistantMessage,
    Message,
    OpenAIProvider,
    SystemMessage,
    Tool,
    ToolMessage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tool 1 — lookup_mission: read a small baked-in fact-record for a
# named lunar mission. Stand-in for a real lookup against a doc store
# or knowledge base.
# ---------------------------------------------------------------------------

LUNAR_MISSIONS: dict[str, dict[str, str]] = {
    "Apollo 11": {
        "launch_date": "1969-07-16",
        "splashdown_date": "1969-07-24",
        "commander": "Neil Armstrong",
        "lunar_module_pilot": "Buzz Aldrin",
        "command_module_pilot": "Michael Collins",
        "result": "First crewed lunar landing.",
    },
    "Apollo 13": {
        "launch_date": "1970-04-11",
        "splashdown_date": "1970-04-17",
        "commander": "Jim Lovell",
        "lunar_module_pilot": "Fred Haise",
        "command_module_pilot": "Jack Swigert",
        "result": (
            "Aborted lunar landing after service-module oxygen tank rupture; "
            "safe return via free-return trajectory."
        ),
    },
    "Apollo 17": {
        "launch_date": "1972-12-07",
        "splashdown_date": "1972-12-19",
        "commander": "Eugene Cernan",
        "lunar_module_pilot": "Harrison Schmitt",
        "command_module_pilot": "Ronald Evans",
        "result": "Final Apollo lunar landing.",
    },
    "Artemis II": {
        "launch_date": "2026-04-01",
        "splashdown_date": "2026-04-10",
        "commander": "Reid Wiseman",
        "lunar_module_pilot": "n/a (no surface landing)",
        "command_module_pilot": "Victor Glover",
        "result": (
            "First crewed lunar flyby of the Artemis program; tested Orion "
            "spacecraft on a free-return trajectory."
        ),
    },
}


def lookup_mission(name: str) -> str:
    record = LUNAR_MISSIONS.get(name)
    if record is None:
        known = ", ".join(sorted(LUNAR_MISSIONS.keys()))
        return f"Unknown mission {name!r}. Known missions: {known}."
    return json.dumps(record)


# ---------------------------------------------------------------------------
# Tool 2 — compute_delta_v: Hohmann transfer delta-v between two
# circular orbits around a body with known gravitational parameter.
# The textbook formula; rough but illustrative.
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6378.0
EARTH_MU_KM3_S2 = 398600.4418  # Standard gravitational parameter for Earth.


def compute_delta_v(initial_altitude_km: float, final_altitude_km: float) -> str:
    """Hohmann transfer delta-v from initial_altitude_km to
    final_altitude_km, both above Earth's surface (so 0 = surface,
    300 = LEO, 384400 = lunar distance). Returns a JSON record with
    the two burns and the total."""
    r1 = initial_altitude_km + EARTH_RADIUS_KM
    r2 = final_altitude_km + EARTH_RADIUS_KM
    mu = EARTH_MU_KM3_S2
    dv1 = math.sqrt(mu / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = math.sqrt(mu / r2) * (1 - math.sqrt(2 * r1 / (r1 + r2)))
    total = abs(dv1) + abs(dv2)
    return json.dumps(
        {
            "first_burn_km_s": round(abs(dv1), 3),
            "second_burn_km_s": round(abs(dv2), 3),
            "total_delta_v_km_s": round(total, 3),
            "note": "Hohmann transfer between two coplanar circular Earth orbits.",
        }
    )


# ---------------------------------------------------------------------------
# Tool definitions for the LLM
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="lookup_mission",
        description="Look up factual record for a named historical or upcoming lunar mission.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Mission name (e.g., 'Apollo 11', 'Artemis II').",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="compute_delta_v",
        description=(
            "Compute the Hohmann transfer delta-v between two circular Earth orbits "
            "given their altitudes above Earth's surface in km. Returns the two burns "
            "and the total delta-v in km/s."
        ),
        parameters={
            "type": "object",
            "properties": {
                "initial_altitude_km": {
                    "type": "number",
                    "description": "Altitude of the starting circular orbit above Earth's surface, in km.",
                },
                "final_altitude_km": {
                    "type": "number",
                    "description": "Altitude of the destination circular orbit above Earth's surface, in km.",
                },
            },
            "required": ["initial_altitude_km", "final_altitude_km"],
            "additionalProperties": False,
        },
    ),
]


def dispatch(name: str, arguments: dict[str, Any]) -> str:
    """Route a tool call to its local Python function.

    Returns a string the agent loop wraps in a ``ToolMessage`` and
    feeds back to the model. Unknown tool names produce an error
    string rather than raising; the model handles the error in the
    next turn.
    """
    if name == "lookup_mission":
        return lookup_mission(arguments["name"])
    if name == "compute_delta_v":
        return compute_delta_v(
            initial_altitude_km=float(arguments["initial_altitude_km"]),
            final_altitude_km=float(arguments["final_altitude_km"]),
        )
    return f"Unknown tool {name!r}."


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MAX_TURNS = 5


class AgentState(State):
    question: str
    messages: list[Message] = Field(default_factory=list[Message])
    final_answer: str = ""
    tool_call_count: int = 0
    turn: int = 0
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def call_llm(s: AgentState) -> Mapping[str, Any]:
    response = await _get_provider().complete(s.messages, tools=TOOLS)
    return {
        "messages": [*s.messages, response.message],
        "turn": s.turn + 1,
        "trace": [f"call_llm[turn={s.turn + 1}]"],
    }


async def dispatch_tools(s: AgentState) -> Mapping[str, Any]:
    last = s.messages[-1]
    if not isinstance(last, AssistantMessage) or not last.tool_calls:
        raise RuntimeError("dispatch_tools entered without a tool-calling assistant message")
    tool_messages: list[Message] = []
    for tc in last.tool_calls:
        # ToolCall.arguments is None only under provider-reported
        # finish_reason="error" (unparseable args). In a real agent the
        # model sees the error string and either retries or bails;
        # either way the loop doesn't crash.
        if tc.arguments is None:
            result_text = (
                f"Tool {tc.name!r} could not be invoked: arguments were "
                f"unparseable. Retry with valid JSON arguments."
            )
        else:
            try:
                result_text = dispatch(tc.name, tc.arguments)
            except (KeyError, ValueError, TypeError) as exc:
                result_text = f"Tool {tc.name!r} failed with {type(exc).__name__}: {exc}"
        tool_messages.append(ToolMessage(content=result_text, tool_call_id=tc.id))
    return {
        "messages": [*s.messages, *tool_messages],
        "tool_call_count": s.tool_call_count + len(tool_messages),
        "trace": [f"dispatch_tools[{len(tool_messages)}]"],
    }


async def present(s: AgentState) -> Mapping[str, Any]:
    last = s.messages[-1]
    if isinstance(last, AssistantMessage) and last.content:
        return {"final_answer": last.content, "trace": ["present"]}
    return {
        "final_answer": "(model exited without final content)",
        "trace": ["present"],
    }


def route_after_llm(s: AgentState) -> str:
    # Hard turn cap: cut the loop even if the model wants more tools.
    # Production agents typically pair this with a fallback summarize
    # step that asks the model to "wrap up with what you have."
    if s.turn >= MAX_TURNS:
        return "present"
    last = s.messages[-1]
    if isinstance(last, AssistantMessage) and last.tool_calls:
        return "dispatch_tools"
    return "present"


def build_graph() -> CompiledGraph[AgentState]:
    return (
        GraphBuilder(AgentState)
        .add_node("call_llm", call_llm)
        .add_node("dispatch_tools", dispatch_tools)
        .add_node("present", present)
        .add_conditional_edge("call_llm", route_after_llm)
        .add_edge("dispatch_tools", "call_llm")
        .add_edge("present", END)
        .set_entry("call_llm")
        .compile()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_QUESTION = (
    "Tell me about Apollo 13. Then, separately, if I were planning a similar "
    "free-return-style mission and wanted to inject from a 300 km parking orbit "
    "to apogee at the Moon's mean distance (384,400 km above Earth's surface), "
    "roughly how much delta-v would that take?"
)


async def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION

    initial_messages: list[Message] = [
        SystemMessage(
            content=(
                "You are a helpful lunar-mission assistant. You have access to "
                "two tools: lookup_mission (factual records for named missions) "
                "and compute_delta_v (Hohmann transfer arithmetic between two "
                "Earth orbits). Use them when the answer benefits. Cite the tool "
                "outputs in your final summary."
            )
        ),
        UserMessage(content=question),
    ]

    print("=" * 72)
    print("Lunar-mission assistant — tool-calling loop")
    print("=" * 72)
    print()
    print(f"  question: {question}")
    print()

    graph = build_graph()
    try:
        final = await graph.invoke(AgentState(question=question, messages=initial_messages))
        print(f"  turns:     {final.turn}")
        print(f"  tools used: {final.tool_call_count}")
        print()
        print("  trace:")
        for step in final.trace:
            print(f"    - {step}")
        print()
        print("  final answer:")
        for line in final.final_answer.splitlines() or [""]:
            print(f"    {line}")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
