# Parameterized entry point

**Problem.** How do I start the graph at an arbitrary node?

## Approach

You don't. Make the "entry point" a state-level parameter instead.
A first router node passes through, and a
[conditional edge](../concepts/composition.md) routes to wherever
execution should begin. The graph stays a single graph; what
differs across runs is which branch the conditional edge takes.

Combine with [checkpointing](../concepts/checkpointing.md) if you
want resume-style behavior — skip nodes whose work is already
captured in state.

## Snippet

```python
from openarmature.graph import END, EndSentinel, GraphBuilder, State


class MissionState(State):
    starting_stage: str = "plan"  # "plan" | "execute" | "report"
    plan: str = ""
    execution_log: str = ""
    report: str = ""


def route_from_starting_stage(s: MissionState) -> str | EndSentinel:
    return s.starting_stage


async def router(s: MissionState) -> dict:
    return {}  # no state change; conditional edge below routes


async def plan(s: MissionState) -> dict:
    return {
        "plan": "Apollo-style free-return trajectory.",
        "starting_stage": "execute",
    }


async def execute(s: MissionState) -> dict:
    return {"execution_log": "Burn complete. Trajectory nominal."}


async def report(s: MissionState) -> dict:
    return {"report": "Mission objectives met."}


builder = (
    GraphBuilder(MissionState)
    .add_node("router", router)
    .add_node("plan", plan)
    .add_node("execute", execute)
    .add_node("report", report)
    .add_conditional_edge("router", route_from_starting_stage)
    .add_edge("plan", "execute")
    .add_edge("execute", "report")
    .add_edge("report", END)
    .set_entry("router")
)
graph = builder.compile()

# Start at the beginning:
await graph.invoke(MissionState())

# Or skip straight to execute, with the plan already in state:
await graph.invoke(MissionState(starting_stage="execute", plan="..."))
```

The caller pre-populates `starting_stage` (and any prerequisite
fields the chosen branch needs) and the graph routes accordingly.

## When this is the right pattern

- You have a few canonical entry points and the choice between
  them is data, not control flow.
- You want to skip work already done in a prior run — combine with
  [checkpointing](../concepts/checkpointing.md) to pick up where
  you left off.
- Your "different entry points" share state structure and most of
  the downstream graph.

## When it isn't

- "Start at node X" really means "run a different pipeline." Then
  it's a different compiled graph. Don't bend one graph into two;
  two graphs are easier to test and reason about.
- The number of entry points grows unboundedly. Then you're
  reimplementing routing — consider a higher-level dispatch layer
  that picks which graph to invoke.

## Cross-references

- [Composition: conditional edges](../concepts/composition.md)
- [Checkpointing](../concepts/checkpointing.md)
- Spec: [graph-engine](https://openarmature.org/capabilities/graph-engine/)
