"""openarmature demo: reach a vendor knob openarmature does not model, and
watch the guardrails that stop you reaching the wrong one.

**Use case:** You classify lunar telemetry alerts in bulk. Two things you want
are provider-specific rather than portable, so there is no first-class field for
either: ``service_tier`` to take the cheaper, slower lane for a batch nobody is
waiting on, and ``logit_bias`` to stop the model emitting a severity label your
team retired last quarter. Both are real OpenAI request fields. Neither means
anything on another provider.

``extras`` is where those go. It is a named container on the runtime config, and
whatever you put in it rides to the wire untouched. That is the whole feature,
and it exists so a provider-specific knob does not require either a fork or a
framework release.

The interesting part is what it refuses. A field openarmature already models is
managed, and putting it in ``extras`` too is an error rather than an override,
because two sources of truth for one wire field is a bug you want at the call
site and not in a trace three days later.

**What's interesting in the implementation:**

- ``RuntimeConfig(extras={...})`` carries anything the framework does not model.
  ``logit_bias`` and ``service_tier`` arrive on the request body verbatim.
- A key naming a field the mapping PRODUCED on this call is rejected. Setting
  ``temperature=0.2`` and also ``extras={"temperature": 0.9}`` raises
  ``ProviderInvalidRequest``, naming the key.
- Two nuances worth knowing, shown in the notes rather than demonstrated:
  a MATCHING value is a no-op rather than an error, since there is no ambiguity
  about what to send; and a sampling field you leave unset is not managed on that
  call, so ``extras={"temperature": 0.9}`` with no ``temperature=`` rides
  through. Managed means "produced by this call", not "nameable".
- ``model``, ``messages``, ``tools`` and ``tool_choice`` are structural and
  reject ALWAYS, even where the call produced no such field. That is what stops
  an ``extras`` tool array from bypassing tool validation entirely.
- ``stop`` MERGES instead of colliding, because it realizes the same thing as
  ``stop_sequences``. Both lists arrive, concatenated.

**Run it:**

    uv run python examples/provider-extras/main.py

No credentials needed. This demo installs a stub transport and prints the
outbound request body, because the shape of that body IS the subject: whether a
knob reached the wire, and what happened when it collided with one the framework
manages. A real endpoint would answer neither question any better, and the
refusals happen before any request is sent.

To watch a real provider accept the knobs, change ``_provider()`` to drop the
``transport=`` argument and supply a real ``base_url`` and ``api_key``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx

from openarmature.graph import END, CompiledGraph, GraphBuilder, State
from openarmature.llm import (
    OpenAIProvider,
    ProviderInvalidRequest,
    RuntimeConfig,
    SystemMessage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Telemetry alerts to classify. In a real app these arrive off a spacecraft
# telemetry bus or an alerting pipeline.
# ---------------------------------------------------------------------------

ALERTS: list[str] = [
    "Regolith intake auger current 18% above nominal for 40 seconds, then settled.",
    "South-pole relay lost carrier for 3 frames during Earth occultation.",
    "Battery bus B cell 4 reading 0.2V under its siblings at end of charge.",
]

_CLASSIFY_SYSTEM = "Classify a lunar telemetry alert. Answer with one word: routine, watch, or urgent."

# The retired label we no longer want the model to emit. In production these
# token ids come from the provider's tokenizer; the value is illustrative.
_RETIRED_LABEL_TOKEN = "24886"

# The two knobs this demo is here for. Neither is portable, so openarmature
# models neither, and neither needs a framework change to use.
VENDOR_KNOBS: dict[str, Any] = {
    # Cheaper, slower lane. Fine for a batch nobody is waiting on.
    "service_tier": "flex",
    # Discourage the retired severity label.
    "logit_bias": {_RETIRED_LABEL_TOKEN: -100},
}


# ---------------------------------------------------------------------------
# A stub transport, so the demo can show the outbound body without an account.
# ---------------------------------------------------------------------------

_sent_bodies: list[dict[str, Any]] = []


def _stub(request: httpx.Request) -> httpx.Response:
    _sent_bodies.append(json.loads(request.content))
    return httpx.Response(
        200,
        json={
            "id": "alert-1",
            "model": "stub",
            "choices": [{"message": {"role": "assistant", "content": "watch"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
        },
    )


def _provider() -> OpenAIProvider:
    return OpenAIProvider(
        base_url="http://telemetry-classifier.invalid",
        model="gpt-4o-mini",
        api_key="stub",
        transport=httpx.MockTransport(_stub),
    )


class AlertState(State):
    alerts: list[str] = []
    severities: list[str] = []
    refusals: list[str] = []


async def classify(s: AlertState) -> Mapping[str, Any]:
    """Classify each alert, sending the two vendor knobs alongside."""
    provider = _provider()
    severities: list[str] = []
    try:
        for alert in s.alerts:
            response = await provider.complete(
                [SystemMessage(content=_CLASSIFY_SYSTEM), UserMessage(content=alert)],
                # `temperature` is a field, so it goes in the field. The vendor
                # knobs have no field, so they go in `extras`. Putting
                # `temperature` in both is the error the next node demonstrates.
                config=RuntimeConfig(temperature=0.0, extras=VENDOR_KNOBS),
            )
            severities.append((response.message.content or "").strip())
    finally:
        await provider.aclose()
    return {"severities": severities}


async def show_guardrails(s: AlertState) -> Mapping[str, Any]:
    """Try the three things `extras` refuses, and record what it said."""
    refusals: list[str] = []

    async def _attempt(label: str, config: RuntimeConfig) -> None:
        provider = _provider()
        try:
            await provider.complete([UserMessage(content="ping")], config=config)
            refusals.append(f"{label}: accepted")
        except ProviderInvalidRequest as exc:
            refusals.append(f"{label}: refused, {exc}")
        finally:
            await provider.aclose()

    # 1. A managed field the call produced. Two sources of truth for one wire
    #    field, so it is refused rather than silently preferring one.
    await _attempt(
        "temperature in both",
        RuntimeConfig(temperature=0.2, extras={"temperature": 0.9}),
    )
    # 2. A structural key. Refused even though this call passes no tools, which
    #    is the point: an extras tool array would otherwise skip validation.
    await _attempt(
        "tools via extras",
        RuntimeConfig(extras={"tools": [{"type": "function"}]}),
    )
    # 3. Not a refusal. `stop` realizes the same wire field as `stop_sequences`,
    #    so the two merge instead of colliding.
    await _attempt(
        "stop merges rather than collides",
        RuntimeConfig(stop_sequences=["END OF ALERT"], extras={"stop": ["HALT"]}),
    )
    return {"refusals": refusals}


def build_graph() -> CompiledGraph[AlertState]:
    return (
        GraphBuilder(AlertState)
        .add_node("classify", classify)
        .add_node("show_guardrails", show_guardrails)
        .add_edge("classify", "show_guardrails")
        .add_edge("show_guardrails", END)
        .set_entry("classify")
        .compile()
    )


async def main() -> None:
    print("=== openarmature provider-extras demo ===")
    print(f"alerts: {len(ALERTS)}")
    print()

    graph = build_graph()
    try:
        final = await graph.invoke(AlertState(alerts=ALERTS))
    finally:
        await graph.drain()

    print("classified:")
    for alert, severity in zip(ALERTS, final.severities, strict=False):
        print(f"  [{severity}] {alert[:64]}...")

    print()
    print("the knobs that reached the wire:")
    first = _sent_bodies[0]
    for key in ("service_tier", "logit_bias", "temperature"):
        if key in first:
            print(f"  {key} = {first[key]!r}")

    print()
    print("what extras refuses:")
    for line in final.refusals:
        print(f"  {line}")

    print()
    print(
        "`extras` is an escape hatch for knobs openarmature does not model, not a\n"
        "second way to set the ones it does."
    )


if __name__ == "__main__":
    asyncio.run(main())
