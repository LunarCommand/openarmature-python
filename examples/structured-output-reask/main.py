"""openarmature demo: pull a structured mission record out of a prose
lunar-landing report, and recover when the reply arrives unusable.

**Use case:** A feed delivers lunar-landing reports as free prose. You want
one row per report: mission, operator, landing site, outcome, and the mass
delivered in kilograms. A JSON schema says exactly that, and the provider is
asked to honour it.

You also cap output tokens, because the records are small and you pay by the
token. That cap is a guess about the longest record you will ever need, and
the guess is sometimes wrong. When it is, the model stops mid-object and the
reply that arrives is a fragment: valid so far, parseable as nothing. The
schema boundary rejects it exactly as it rejects a model that answered in
prose or sent ``"1,340 kg"`` where a number was required.

Retrying the identical request reproduces the identical fragment, because
nothing about the request changed. Two things have to change, and they are
different kinds of thing:

- **What you say.** Show the model what came back and what was wrong with it.
  That is ``reask``: you supply the corrective message, because only you know
  how to talk to your model about your schema.
- **What it is allowed to spend.** A correction cannot help a reply that gets
  cut off at the same place. That is ``per_attempt_override``: the retry runs
  under a raised ceiling.

Supply only the first and the retry is better informed and still truncated.
This demo shows both arms so the difference is visible.

**Why the ceiling and not the wrong-shaped answer.** The ceiling is the
failure this demo can guarantee, on any endpoint and any model. The
wrong-shaped answer is the one you are more likely to meet, and whether you
meet it at all is a property of your serving stack rather than of your code.
An endpoint that enforces the schema during decoding cannot produce it. An
endpoint that ignores ``response_format``, or serves a weaker model behind
one, produces it routinely, and so does a proxy that drops the field on the
way through. Both failures arrive at the same exception and the same builder
handles both, which is the point: you do not have to know in advance which
kind of endpoint you are pointed at, and you keep working when someone moves
you to a different one.

**What's interesting in the implementation:**

- ``complete(response_schema=...)`` asks the provider for a shape. When the
  reply cannot be parsed or does not validate, the call raises
  ``StructuredOutputInvalid`` rather than handing back a half-built object.
  A truncated reply and a wrong-typed field arrive through the same door.
- ``LlmRetryConfig(reask=...)`` makes that failure retryable FOR THIS CALL.
  Without a builder it is terminal, which is the right default: a schema the
  model cannot satisfy is usually a bug in the schema, not a transient.
- The builder receives the exception and returns the correction as a string.
  It reads ``exc.output_content`` (verbatim, what the model actually sent),
  ``exc.error_message`` (what the reader objected to), and
  ``exc.finish_reason`` (``"length"`` when the ceiling ended the reply). This
  one branches on that last signal and says something different about a
  truncation than about a model that simply answered wrongly, which is the
  kind of judgement only the caller can make.
- ``LlmRetryConfig(per_attempt_override=...)`` is a schedule of partial
  configs applied to retries only. Attempt 0 runs the caller's config
  untouched; each retry merges the next entry over it. A schedule shorter
  than the retry count carries its last entry forward.
- The framework appends the model's own reply as an ``assistant`` turn and the
  builder's string as a ``user`` turn, so the conversation stays
  role-alternating and the model sees its own fragment in context. It authors
  no prompt of its own; every word sent is yours.
- Reask shares the ``max_attempts`` budget with transient retries. A call that
  burns two attempts on unusable output has one left for a rate limit.

**Run it:**

    export LLM_API_KEY=sk-...
    uv run python examples/structured-output-reask/main.py

    MODE=nocap uv run python examples/structured-output-reask/main.py
    MODE=off   uv run python examples/structured-output-reask/main.py

Three postures. ``reask`` (default) supplies the builder and raises the cap.
``nocap`` supplies the builder and leaves the cap alone, so every attempt is
cut off at the same place and the budget drains. ``off`` supplies neither, so
the first unusable reply ends the call.

Point ``LLM_BASE_URL`` at any OpenAI-compatible endpoint (the host root, not
its ``/v1`` path). ``LLM_MODEL`` defaults to a small fast model.
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

# The shape we want back. `mass_kg` is a number rather than a string, which is
# the constraint a model breaks by copying "1,340 kg" out of the prose.
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

# The cap the app runs under, and the one a retry is allowed to climb to. The
# tight cap sits below what a complete record of this shape needs, so it is
# the ceiling rather than the model that makes the first reply unusable.
TIGHT_MAX_TOKENS = 32
ROOMY_MAX_TOKENS = 256


# ---------------------------------------------------------------------------
# The reask builder. This is the whole point of the example.
# ---------------------------------------------------------------------------


def build_correction(exc: StructuredOutputInvalid) -> str:
    """Compose the corrective message sent after an unusable reply.

    Receives the failure and returns what to say about it. The framework
    appends the model's own reply before this, so it reads as a conversation:
    the model answered, you point at the problem, it answers again.
    """
    # Both halves of the exception matter. The verbatim output lets the model
    # see what it actually sent rather than what it meant to send, and the
    # validator's complaint names the constraint it missed. A correction
    # carrying neither is the original prompt again, which earns the original
    # answer.
    raw = exc.output_content.strip()
    # A reply that stops mid-object is a spend problem, not a comprehension
    # problem, and saying "that did not match the schema" about it would be
    # misleading. `finish_reason` is how the framework reports which one
    # happened: "length" means the ceiling ended the reply, so the model was
    # never given the chance to be wrong.
    if exc.finish_reason == "length":
        return (
            "That reply stopped before the object closed, so it could not be "
            f"read. The reader said: {exc.error_message}\n\n"
            f"You sent:\n{raw}\n\n"
            "Send the whole object this time. Keep every value as short as "
            "the report allows and add nothing beyond the required fields."
        )
    return (
        "That reply did not match the required schema. The validator said: "
        f"{exc.error_message}\n\n"
        f"You sent:\n{raw}\n\n"
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


async def _close_provider() -> None:
    # Only close what was built. Calling the constructor here to obtain
    # something to close would mask whatever failure kept the run from
    # building one.
    if _provider_instance is not None:
        await _provider_instance.aclose()


class FeedState(State):
    reports: list[str] = []
    records: list[dict[str, Any]] = []
    failures: list[str] = []


def _retry_config(mode: str) -> LlmRetryConfig:
    # `reask` is what makes a schema failure retryable at all. Omit it and the
    # first unusable reply is terminal, which is the useful default: retrying
    # an unchanged prompt against an unchanged model gets the same answer.
    #
    # The override is the other half. Without it the retry is better informed
    # and still capped at the length that truncated it, so `nocap` spends the
    # whole budget re-learning the same lesson.
    return LlmRetryConfig(
        max_attempts=3,
        backoff=deterministic_backoff(0.0),
        reask=build_correction if mode in {"reask", "nocap"} else None,
        per_attempt_override=([RuntimeConfig(max_tokens=ROOMY_MAX_TOKENS)] if mode == "reask" else None),
    )


async def extract(s: FeedState) -> Mapping[str, Any]:
    """Extract one record per report, recovering from an unusable first reply."""
    provider = _get_provider()
    retry = _retry_config(os.environ.get("MODE", "reask"))
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for report in s.reports:
        try:
            response = await provider.complete(
                [SystemMessage(content=_EXTRACT_SYSTEM), UserMessage(content=report)],
                config=RuntimeConfig(temperature=0.0, max_tokens=TIGHT_MAX_TOKENS),
                response_schema=MISSION_SCHEMA,
                retry=retry,
            )
        except StructuredOutputInvalid as exc:
            # Reached when the budget runs out, and immediately when no
            # builder was supplied. The exception carries the last thing the
            # model sent and why it was rejected, which is what belongs in
            # the log.
            failures.append(f"{exc.error_message}: {exc.output_content[:90]}")
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


_POSTURE = {
    "reask": "corrective message + raised token ceiling",
    "nocap": "corrective message, ceiling left alone",
    "off": "neither",
}


async def main() -> None:
    mode = os.environ.get("MODE", "reask")
    print("=== openarmature structured-output-reask demo ===")
    print(f"mode: {mode}  ({_POSTURE.get(mode, 'unknown mode')})")
    print(f"reports: {len(REPORTS)}")
    print(
        f"cap: {TIGHT_MAX_TOKENS} output tokens"
        + (f", raised to {ROOMY_MAX_TOKENS} on retry" if mode == "reask" else "")
    )
    print()

    graph = build_graph()
    try:
        final = await graph.invoke(FeedState(reports=REPORTS))
    finally:
        await _close_provider()
        await graph.drain()

    for record in final.records:
        print(
            f"  {record.get('mission')}  |  {record.get('operator')}  |  "
            f"{record.get('landing_site')}  |  {record.get('outcome')}  |  "
            f"{record.get('mass_kg')} kg"
        )
    if final.failures:
        print("  unrecovered:")
        for f in final.failures:
            print(f"    {f}")

    print()
    print(f"extracted {len(final.records)} of {len(REPORTS)}")
    if mode == "off":
        print("MODE=nocap adds the corrective message. Default adds the ceiling too.")
    elif mode == "nocap":
        print(
            "The model was told what went wrong and still had nowhere to put "
            "the answer. Run without MODE to raise the ceiling as well."
        )


if __name__ == "__main__":
    asyncio.run(main())
