# Tool-dispatch-as-node

**Problem.** How do I run an agent tool-call loop?

## Approach

A node reads the assistant's last `tool_calls` from the running
message list, dispatches each to a local Python function, appends
`ToolMessage` records back to the message list via an
[`append` reducer](../concepts/state-and-reducers.md), and a
[conditional edge](../concepts/composition.md) loops back to the
LLM node if the model wants more turns. The exit is the
conditional edge routing to a `present` node (or `END`) when the
assistant returns no `tool_calls`.

No "agent framework" abstraction — the loop is just a graph cycle
on top of [`Tool`, `ToolCall`, `ToolMessage`](../concepts/llms.md).

## Snippet

```python
import json
from typing import Annotated
from pydantic import Field
from openarmature.graph import END, EndSentinel, GraphBuilder, State, append
from openarmature.llm import AssistantMessage, Message, Tool, ToolMessage


class AgentState(State):
    messages: Annotated[list[Message], append] = Field(default_factory=list)
    turn: int = 0


TOOLS = [
    Tool(
        name="lookup_mission",
        description="Look up Apollo or Artemis mission facts.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
]
MAX_TURNS = 5


async def call_llm(s: AgentState) -> dict:
    response = await provider.complete(s.messages, tools=TOOLS)
    return {"messages": [response], "turn": s.turn + 1}


async def dispatch_tools(s: AgentState) -> dict:
    assistant = s.messages[-1]
    assert isinstance(assistant, AssistantMessage)
    results: list[Message] = []
    for tc in assistant.tool_calls or ():
        output = await dispatch_one(tc.name, tc.arguments)  # str or JSON-serializable
        content = output if isinstance(output, str) else json.dumps(output)
        results.append(ToolMessage(content=content, tool_call_id=tc.id))
    return {"messages": results}


def route_after_llm(s: AgentState) -> str | EndSentinel:
    if s.turn >= MAX_TURNS:
        return "present"
    last = s.messages[-1]
    if isinstance(last, AssistantMessage) and last.tool_calls:
        return "dispatch_tools"
    return "present"


async def present(s: AgentState) -> dict:
    return {}  # final formatting / output


builder = (
    GraphBuilder(AgentState)
    .add_node("call_llm", call_llm)
    .add_node("dispatch_tools", dispatch_tools)
    .add_node("present", present)
    .add_conditional_edge("call_llm", route_after_llm)
    .add_edge("dispatch_tools", "call_llm")
    .add_edge("present", END)
    .set_entry("call_llm")
)
graph = builder.compile()
```

The `MAX_TURNS` cap prevents runaway loops; the conditional edge
short-circuits to `present` when the cap is hit or when the model
returns no `tool_calls`.

See [`examples/09-tool-use`](../examples/09-tool-use.md) for a
runnable version with full tool definitions, defensive handling
for malformed `ToolCall.arguments`, and trace output.

## When this is the right pattern

- The model needs to call local Python functions and react to
  their results.
- The loop is bounded — either by `MAX_TURNS`, by the model
  signaling it's done, or by both.
- Tool results are textual or JSON-serializable and fit cleanly
  into `ToolMessage.content`.

## When it isn't

- Tools have side effects you can't replay safely on resume. Wrap
  each side-effecting tool with the
  [bypass-if-output-exists](bypass-if-output-exists.md) pattern so
  a crashed run resumes without re-side-effecting.
- The "tools" are long-running async pipelines, not function
  calls. Model them as subgraphs and let the LLM node route via
  conditional edge to the right subgraph; the loop shape is the
  same but each "tool" is a full pipeline.
- You need streaming tool results back to the model mid-call. The
  current `Tool` / `ToolMessage` shape is request/response;
  streaming is out of scope for this pattern.

## Cross-references

- [LLMs concept page](../concepts/llms.md) — `Tool`, `ToolCall`,
  `ToolMessage` types and the `complete(messages, tools=...)`
  contract.
- [State and reducers](../concepts/state-and-reducers.md) —
  `append` reducer semantics.
- [`examples/09-tool-use`](../examples/09-tool-use.md) — runnable
  reference implementation.
- Spec: [llm-provider](https://openarmature.org/capabilities/llm-provider/)
