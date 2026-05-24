# Session-as-checkpoint-resume

**Problem.** How do I keep multi-turn agent state across turns?

## Approach

The framework's [checkpointing](../concepts/checkpointing.md)
provides single-invocation crash resume out of the box. Multi-turn
state is the same primitive used differently: the application
keeps a stable `session_id → invocation_id` mapping, and each
turn calls `invoke(resume_invocation=<prior_invocation_id>)` to
pick up where the previous turn left off.

The checkpointer returns the prior state. The new turn proceeds
from there. Session-context fields that accumulate across turns
(message history, retrieved facts, running totals) use a `merge`
or `append` reducer so each turn's contribution adds to what's
already there rather than replacing it.

Each resume mints a new `invocation_id`; the `session_id` is the
join key the application maintains, typically as the
`correlation_id` on `invoke()` (which is preserved unchanged
across resume).

## Snippet

```python
from typing import Annotated
from pydantic import Field
from openarmature.checkpoint import SQLiteCheckpointer
from openarmature.graph import END, GraphBuilder, State, append, merge
from openarmature.llm import Message


class SessionState(State):
    messages: Annotated[list[Message], append] = Field(default_factory=list)
    facts: Annotated[dict[str, str], merge] = Field(default_factory=dict)
    last_user_input: str = ""


# ... define nodes that read s.messages, append to s.messages,
#     and merge into s.facts ...

checkpointer = SQLiteCheckpointer(path="./sessions.db")
graph = (
    GraphBuilder(SessionState)
    .add_node("plan", plan)
    .add_node("respond", respond)
    .add_edge("plan", "respond")
    .add_edge("respond", END)
    .set_entry("plan")
    .with_checkpointer(checkpointer)
    .compile()
)


# The application maintains its own session table mapping
# session_id -> latest invocation_id. OA's checkpointer doesn't
# know about sessions; the join is the application's
# responsibility. The session_id doubles as correlation_id so
# observability traces share the cross-turn join key.
async def handle_turn(session_id: str, user_input: str) -> str:
    initial = SessionState(last_user_input=user_input)
    prior_invocation_id = sessions_db.get_invocation_id(session_id)

    if prior_invocation_id is None:
        final = await graph.invoke(initial, correlation_id=session_id)
    else:
        final = await graph.invoke(
            initial, resume_invocation=prior_invocation_id
        )

    # Record the new invocation_id for next turn's resume.
    # Read it from the checkpointer's latest record for this
    # correlation_id; exact lookup is application-side bookkeeping.
    sessions_db.set_invocation_id(session_id, latest_for(session_id))

    return final.messages[-1].content
```

`sessions_db` is your application's session-state store (Postgres,
Redis, a flat file, whatever); the checkpointer holds the OA-side
state and the session table holds the join keys.

## When this is the right pattern

- Your application has long-lived sessions with multiple LLM turns
  and you want the prior state to be the starting point of the
  next turn.
- You're already running a checkpointer for crash resume — this
  pattern is "use it more."
- Cross-turn state has clean reducer semantics: `merge` for
  accumulating dicts, `append` for growing lists.

## When it isn't

- A session's "state" is bigger than fits comfortably in a single
  graph state shape. Split into multiple graphs and share an
  external store keyed by session.
- Turns are completely independent — there's no value in carrying
  state across them. Then just run each turn as a fresh invoke.
- The application already has its own state-management layer that
  conflicts with OA's frozen-state model. Use OA per-turn without
  cross-turn resume.

## Cross-references

- [Checkpointing](../concepts/checkpointing.md) — backend wiring,
  `resume_invocation`, schema migration.
- [State and reducers](../concepts/state-and-reducers.md) — `merge`
  and `append` reducer strategies.
- [`examples/08-checkpointing-and-migration`](../examples/08-checkpointing-and-migration.md) —
  single-resume baseline.
- Spec: [pipeline-utilities](https://openarmature.org/capabilities/pipeline-utilities/)
