"""openarmature demo: pull a structured mission record out of a prose
lunar-landing report, and correct the model when it answers in the wrong
shape.

**Use case:** A feed delivers lunar-landing reports as free prose. You want
one row per report: mission, operator, landing site, outcome, and the mass
delivered in kilograms. A JSON schema says exactly that, and the provider is
asked to honour it.

Models miss. They wrap JSON in a markdown fence, answer in prose because the
report was ambiguous, invent a field, or send ``"1,340 kg"`` where the schema
says a number. The useful move is not to fail the run and not to retry the
identical prompt, which reproduces the identical mistake. It is to show the
model what it returned, say what was wrong with it, and ask again.

That is what ``reask`` is: you supply the corrective message, because only you
know how to talk to your model about your schema. The framework supplies the
loop, the transcript, and the budget.

**What's interesting in the implementation:**

- ``complete(response_schema=...)`` asks the provider for a shape. When the
  reply cannot be parsed or does not validate, the call raises
  ``StructuredOutputInvalid`` rather than handing back a half-built object.
- ``LlmRetryConfig(reask=...)`` makes that failure retryable FOR THIS CALL.
  Without a builder it is terminal, which is the right default: a schema the
  model cannot satisfy is usually a bug in the schema, not a transient.
- The builder receives the exception and returns the correction as a string.
  It reads ``exc.output_content`` (verbatim, what the model actually sent) and
  ``exc.error_message`` (what the validator objected to). Quoting both back is
  what makes the second attempt different from the first.
- The framework appends the model's own reply as an ``assistant`` turn and the
  builder's string as a ``user`` turn, so the conversation stays
  role-alternating and the model sees its own mistake in context. It authors no
  prompt of its own; every word sent is yours.
- Reask shares the ``max_attempts`` budget with transient retries. A call that
  burns two attempts on malformed output has one left for a rate limit.
- ``MODE`` selects the posture. ``"reask"`` (default) supplies the builder.
  ``"off"`` omits it, so the first invalid reply ends the call and you can see
  what the builder is buying.

**Run it:**

    export LLM_API_KEY=sk-...
    uv run python examples/structured-output-reask/main.py

    MODE=off uv run python examples/structured-output-reask/main.py

Point ``LLM_BASE_URL`` at any OpenAI-compatible endpoint. ``LLM_MODEL``
defaults to a small fast model; a weaker one makes the reask path fire more
often, which is the interesting case to watch.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from typing import Any

from openarmature.graph import END, CompiledGraph, GraphBuilder, State
from openarmature.graph.middleware import deterministic_backoff
from openarmature.llm import (
    LlmRetryConfig,
    OpenAIProvider,
    RuntimeConfig,
    StructuredOutputInvalid,
    SystemMessage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# The report feed. In a real app these arrive from a wire service, an
# operator's status page, or a mission-log database.
# ---------------------------------------------------------------------------

REPORTS: list[str] = [
    (
        "Intuitive Machines confirmed this morning that IM-3 touched down "
        "intact at Reiner Gamma at 04:12 UTC, carrying 1,340 kilograms of "
        "instruments for the agency's swirl-magnetism survey."
    ),
    (
        "The Chandrayaan-4 sample return element separated as planned and is "
        "on its way home. Its lander remains at the south pole with roughly "
        "620 kg of hardware still on the surface, powered down for the night."
    ),
    (
        "Contact with Peregrine Flight 2 was lost during descent. Telemetry "
        "placed it near Sinus Viscositatis carrying 90 kg of payload. The "
        "operator has not yet confirmed whether the vehicle survived."
    ),
]

# The shape we want back. Deliberately strict: `mass_kg` is a number, not a
# string, which is the constraint a model is most likely to break by sending
# "1,340 kg" verbatim from the prose.
MISSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mission": {"type": "string"},
        "operator": {"type": "string"},
        "landing_site": {"type": "string"},
        "outcome": {"type": "string", "enum": ["landed", "lost", "unconfirmed"]},
        "mass_kg": {"type": "number"},
    },
    "required": ["mission", "operator", "landing_site", "outcome", "mass_kg"],
    "additionalProperties": False,
}

_EXTRACT_SYSTEM = (
    "You extract one structured mission record from a lunar-landing report. Answer with a JSON object only."
)


# ---------------------------------------------------------------------------
# The reask builder. This is the whole point of the example.
# ---------------------------------------------------------------------------


def build_correction(exc: StructuredOutputInvalid) -> str:
    """Compose the corrective message sent after an invalid reply.

    Receives the failure and returns what to say about it. The framework
    appends the model's own reply before this, so it reads as a conversation:
    the model answered, you point at the problem, it answers again.
    """
    # Both halves matter. The verbatim output lets the model see what it
    # actually sent rather than what it meant to send, and the validator's
    # complaint names the constraint it missed. A correction carrying neither
    # is just the original prompt again, which reproduces the original answer.
    return (
        f"That reply did not match the required schema. The validator said: "
        f"{exc.error_message}\n\n"
        f"You sent:\n{exc.output_content}\n\n"
        "Send the JSON object again, corrected. Return only the object, with "
        "no surrounding prose and no markdown fence. `mass_kg` must be a bare "
        'number in kilograms, so write 1340 rather than "1,340 kg". '
        "`outcome` must be exactly one of: landed, lost, unconfirmed."
    )


# ---------------------------------------------------------------------------
# Provider and state
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


class FeedState(State):
    reports: list[str] = []
    records: list[dict[str, Any]] = []
    failures: list[str] = []


def _retry_config(mode: str) -> LlmRetryConfig:
    # `reask` is what makes a schema failure retryable at all. Omit it and the
    # first invalid reply is terminal, which is the useful default: retrying an
    # unchanged prompt against an unchanged model gets the same answer.
    return LlmRetryConfig(
        max_attempts=3,
        backoff=deterministic_backoff(0.0),
        reask=build_correction if mode == "reask" else None,
    )


async def extract(s: FeedState) -> Mapping[str, Any]:
    """Extract one record per report, tolerating a wrong-shaped first reply."""
    provider = _get_provider()
    retry = _retry_config(os.environ.get("MODE", "reask"))
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for report in s.reports:
        try:
            response = await provider.complete(
                [SystemMessage(content=_EXTRACT_SYSTEM), UserMessage(content=report)],
                config=RuntimeConfig(temperature=0.0),
                response_schema=MISSION_SCHEMA,
                retry=retry,
            )
        except StructuredOutputInvalid as exc:
            # Reached when the budget runs out, or immediately when MODE=off.
            # The exception carries the last thing the model sent and why it
            # was rejected, which is what you want in the log.
            failures.append(f"{exc.error_message}: {exc.output_content[:120]}")
            continue
        records.append(json.loads(response.message.content or "{}"))

    return {"records": records, "failures": failures}


async def present(s: FeedState) -> Mapping[str, Any]:
    return {}


def build_graph() -> CompiledGraph[FeedState]:
    return (
        GraphBuilder(FeedState)
        .add_node("extract", extract)
        .add_node("present", present)
        .add_edge("extract", "present")
        .add_edge("present", END)
        .set_entry("extract")
        .compile()
    )


async def main() -> None:
    mode = os.environ.get("MODE", "reask")
    print("=== openarmature structured-output-reask demo ===")
    print(f"mode: {mode}" + ("  (no corrective builder)" if mode == "off" else ""))
    print(f"reports: {len(REPORTS)}")
    print()

    graph = build_graph()
    try:
        final = await graph.invoke(FeedState(reports=REPORTS))
    finally:
        await _get_provider().aclose()
        await graph.drain()

    for record in final.records:
        print(
            f"  {record.get('mission')}  |  {record.get('operator')}  |  "
            f"{record.get('landing_site')}  |  {record.get('outcome')}  |  "
            f"{record.get('mass_kg')} kg"
        )
    if final.failures:
        print()
        print("  unrecovered:")
        for f in final.failures:
            print(f"    {f}")

    print()
    print(f"extracted {len(final.records)} of {len(REPORTS)}")
    if mode == "off" and final.failures:
        print("Run without MODE=off to let the corrective builder retry these.")


if __name__ == "__main__":
    asyncio.run(main())
