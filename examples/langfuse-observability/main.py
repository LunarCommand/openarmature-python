"""openarmature demo: Langfuse observer + prompt linkage on a lunar mission Q&A pipeline.

**Use case:** A mission-briefing assistant answers questions about Apollo
and Artemis missions. The pipeline fetches a versioned prompt template,
renders it with the user's question, sends it to the model, and stores
the response. The team running this in production wants to validate the
prompt is doing what it should; see exactly what messages went out,
what the model returned, what the token usage was, and (critically)
which prompt version produced which response so they can A/B test
prompt revisions safely.

**Demonstrates:** The Langfuse-native observer that maps every node and
LLM call into a Langfuse Trace + Observation tree. The demo's prompt
backend simulates a Langfuse-aware source by attaching a sentinel
Langfuse Prompt entity reference to each rendered prompt; the Generation
observation picks that up and links back to the entity, which is how
production Langfuse dashboards thread "this generation came from prompt
v7 of `mission-briefing`" without you having to wire anything up
manually.

The example uses the bundled ``InMemoryLangfuseClient`` recorder so the
demo runs without a Langfuse account; at the end we print the captured
Trace + Observation tree. Swapping to a real ``langfuse.Langfuse()``
client is a one-line constructor change via ``LangfuseSDKAdapter`` (see
the comment near the observer build below). The adapter bridges the
``langfuse>=4.6`` Python SDK shape onto OA's ``LangfuseClient``
Protocol. Install with::

    pip install 'openarmature[langfuse]'

LLM calls go through ``openarmature.llm.OpenAIProvider``.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/langfuse-observability
    LLM_API_KEY=sk-... uv run python main.py "what year did Apollo 11 land"
    LLM_API_KEY=sk-... uv run python main.py "compare the Artemis II crew to Apollo 8's"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from typing import Any

from openarmature.graph import END, CompiledGraph, GraphBuilder, State
from openarmature.llm import OpenAIProvider
from openarmature.observability.langfuse import (
    InMemoryLangfuseClient,
    LangfuseObservation,
    LangfuseObserver,
    LangfuseTrace,
)
from openarmature.prompts import Prompt, PromptManager, PromptResult, TextPrompt
from openarmature.prompts.context import with_active_prompt

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
# Mock prompt backend with a Langfuse-source reference
# ----------------------------------------------------------------------------
# A real production setup would use the Langfuse Python SDK's
# ``LangfusePromptBackend`` (community / forthcoming sibling-package
# territory) which fetches from the Langfuse Prompts API and attaches
# the SDK's Prompt-entity reference to ``Prompt.observability_entities``.
# We stub that here so the demo doesn't need a Langfuse account: the
# sentinel string ``"lf-prompt-mission-briefing-v7"`` stands in for what
# would normally be an SDK Prompt-entity object.


class _MockLangfusePromptBackend:
    """In-memory PromptBackend that simulates a Langfuse-source by
    attaching a sentinel ``langfuse_prompt`` entity reference.

    The Langfuse observer reads
    ``Prompt.observability_entities['langfuse_prompt']`` when emitting
    the Generation observation. In production that key holds a real
    Langfuse SDK Prompt object; here it's a string sentinel so the
    captured Trace shows the linkage shape without needing a real SDK.
    """

    def __init__(self) -> None:
        now = datetime.now(UTC)
        self._prompt = TextPrompt(
            name="mission-briefing",
            version="v7",
            label="production",
            template=(
                "You are a lunar mission historian. Answer the following "
                "question in two short sentences with specific dates or "
                "crew names when relevant.\n\n"
                "Question: {{ question }}"
            ),
            template_hash="sha256:mission-briefing-v7",
            fetched_at=now,
            observability_entities={
                "langfuse_prompt": "lf-prompt-mission-briefing-v7",
            },
        )

    async def fetch(self, name: str, label: str = "production") -> Prompt:
        if name != "mission-briefing":
            from openarmature.prompts import PromptNotFound

            raise PromptNotFound(
                f"no prompt {name!r} in this demo backend",
                name=name,
                label=label,
                backend="mock-langfuse",
            )
        return self._prompt


# ----------------------------------------------------------------------------
# State + node
# ----------------------------------------------------------------------------


class BriefingState(State):
    question: str
    answer: str = ""
    prompt_version: str = ""


_PROMPT_MANAGER = PromptManager(_MockLangfusePromptBackend())


async def answer_briefing(s: BriefingState) -> dict[str, Any]:
    """Fetch the briefing prompt, render with the question, send to the LLM.

    ``with_active_prompt(rendered)`` is what makes the Generation
    observation surface the prompt-identity metadata + Langfuse Prompt
    entity link. The Langfuse observer reads the active PromptResult at
    dispatch time and threads it onto the Generation observation it
    emits.
    """
    rendered: PromptResult = await _PROMPT_MANAGER.get(
        "mission-briefing", "production", {"question": s.question}
    )
    provider = _get_provider()
    with with_active_prompt(rendered):
        response = await provider.complete(rendered.messages)
    return {
        "answer": response.message.content or "",
        "prompt_version": rendered.version,
    }


def build_graph() -> CompiledGraph[BriefingState]:
    return (
        GraphBuilder(BriefingState)
        .add_node("answer_briefing", answer_briefing)
        .add_edge("answer_briefing", END)
        .set_entry("answer_briefing")
        .compile()
    )


# ----------------------------------------------------------------------------
# Pretty-printer for the captured Langfuse Trace
# ----------------------------------------------------------------------------


def _format_trace(trace: LangfuseTrace) -> str:
    """Render the captured Trace + Observation tree as a human-readable string.

    Production Langfuse renders this same data in the web UI; the
    in-memory recorder gives us the same structured shape so we can
    print it to stdout for the demo.
    """
    lines: list[str] = []
    lines.append(f"Trace id={trace.id}")
    lines.append(f"      name={trace.name!r}")
    lines.append(f"      metadata={_format_metadata(trace.metadata)}")
    for obs in trace.children_of(None):
        _format_observation(lines, trace, obs, indent="  ")
    return "\n".join(lines)


def _format_observation(
    lines: list[str], trace: LangfuseTrace, obs: LangfuseObservation, indent: str
) -> None:
    summary = f"{indent}[{obs.type}] {obs.name!r} level={obs.level}"
    lines.append(summary)
    if obs.metadata:
        lines.append(f"{indent}  metadata={_format_metadata(obs.metadata)}")
    if obs.type == "generation":
        if obs.model is not None:
            lines.append(f"{indent}  model={obs.model!r}")
        if obs.usage is not None:
            lines.append(
                f"{indent}  usage=input:{obs.usage.input} output:{obs.usage.output} total:{obs.usage.total}"
            )
        if obs.prompt_entity_link is not None:
            lines.append(f"{indent}  prompt_entity_link={obs.prompt_entity_link!r}")
        if obs.output is not None:
            out_str = obs.output if isinstance(obs.output, str) else json.dumps(obs.output)
            lines.append(f"{indent}  output={out_str[:120]!r}{'...' if len(out_str) > 120 else ''}")
    for child in trace.children_of(obs.id):
        _format_observation(lines, trace, child, indent=indent + "  ")


def _format_metadata(metadata: dict[str, Any]) -> str:
    # Sort keys for stable demo output across runs. correlation_id is a
    # UUIDv4 so its exact value changes every run; truncate it for
    # readability without losing the "yes, this is set" signal.
    parts: list[str] = []
    for key in sorted(metadata):
        value = metadata[key]
        if key == "correlation_id" and isinstance(value, str) and len(value) > 12:
            value = f"{value[:8]}…"
        parts.append(f"{key}={value!r}")
    return "{" + ", ".join(parts) + "}"


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "what year did Apollo 11 land"

    # The bundled in-memory client captures everything the observer
    # would have sent to Langfuse; Trace, Observations, Generation
    # fields; without needing a Langfuse account. For production:
    #
    #     from langfuse import Langfuse
    #     from openarmature.observability.langfuse import LangfuseSDKAdapter
    #
    #     langfuse_client = Langfuse(
    #         public_key="pk-lf-...",
    #         secret_key="sk-lf-...",
    #         host="https://cloud.langfuse.com",
    #     )
    #     client = LangfuseSDKAdapter(langfuse_client)
    #
    # Validated against ``langfuse>=4.6,<5``. The adapter bridges
    # langfuse v4's unified ``start_observation`` API onto OA's
    # ``LangfuseClient`` Protocol; the observer code doesn't change.
    client = InMemoryLangfuseClient()

    # disable_provider_payload=False opts in to capturing the input messages
    # and output content on Generation observations. Default is True
    # for the same privacy reason the OTel observer's flag exists:
    # payloads may contain PII the operator hasn't audited. Flip it
    # deliberately here because the demo's whole point is showing what
    # the model saw and returned.
    observer = LangfuseObserver(client=client, disable_provider_payload=False)

    graph = build_graph()
    graph.attach_observer(observer)

    try:
        final = await graph.invoke(BriefingState(question=question))
    finally:
        # Required for short-lived processes: invoke() returns when the
        # graph reaches END regardless of whether the observer queue
        # has finished draining. Without drain() the last few
        # observation calls (the LLM completion's `.end()`, the node's
        # close) can be dropped on process exit.
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()

    print()
    print(f"question: {final.question}")
    print(f"answer:   {final.answer}")
    print(f"prompt:   mission-briefing {final.prompt_version}")
    print()
    print("─── captured Langfuse trace ─────────────────────────────────")
    # Exactly one Trace per invocation; the LangfuseObserver opens it
    # on the first node event and the trace id equals the framework-
    # minted invocation_id so cross-system lookups land directly.
    assert len(client.traces) == 1, f"expected 1 trace, got {len(client.traces)}"
    trace = next(iter(client.traces.values()))
    print(_format_trace(trace))


if __name__ == "__main__":
    asyncio.run(main())
