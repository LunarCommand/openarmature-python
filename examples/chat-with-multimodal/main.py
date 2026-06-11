"""openarmature demo: multi-turn chat with conversation memory and a
multimodal turn, using ChatPrompt + PlaceholderSegment.

**Use case:** Lunar mission Q&A assistant that maintains conversation
context across four turns. Turn 3 includes an attached photograph
(e.g., a Lunar Module on the surface): the user asks about it, the
agent processes the multimodal turn naturally without changing the
chat-history shape. Turns 1, 2, 4 are text-only.

**Demonstrates:** ChatPrompt + ContentSegment (system + user) +
PlaceholderSegment for chat-history injection. PromptManager.render
with the `placeholders` kwarg.
Multi-turn message threading through state with the `append`
reducer; the conversation history grows over turns and feeds back
into render() on each turn. The same chat template carries an
optional ImageURLBlockTemplate when the user's current turn includes
an image (lunar mission photograph), so multimodal turns work
without bespoke handling. Complementary to the tool-use example,
which exercises a different LLM-side primitive entirely.

**What's interesting in the implementation:**

- The chat template is built per-turn by `_build_chat_prompt(...)`,
  which switches the user `ContentSegment.content` between a single
  text template (text-only turn) and a `[TextBlockTemplate,
  ImageURLBlockTemplate]` list (multimodal turn). The system segment
  and the `PlaceholderSegment(placeholder="history")` slot are identical
  across both shapes; only the trailing user segment changes.
- Chat history lives on state as `history: Annotated[list[Message],
  append]`. After each turn the node appends two messages (the new
  user turn that just rendered + the assistant response) so the
  next turn's render() sees the full prior conversation.
- `PromptManager.render(prompt, placeholders={"history": state.history})`
  injects the message list at the placeholder slot. An empty
  list is valid (first-turn case): the rendered messages become
  just `[system, current_user_turn]` with no prior history.
- The graph is a single `respond` node with a conditional edge that
  loops back to itself until the script-supplied user turns are
  exhausted, then routes to END. Each loop iteration renders the
  current chat template, calls the LLM, and updates state.
- `LangfusePromptBackend` is intentionally not used here: chat
  history threading is the headline demonstration, not prompt
  backend complexity. The multimodal-prompt example owns the
  multi-backend prompt story (filesystem primary + fallback);
  the langfuse-observability example owns the Langfuse-backend
  integration.
- Error handling at the invoke() boundary. `main()` catches
  `NodeException` (the graph engine's wrapper) and inspects
  `exc.__cause__` (Python's standard exception chain) for
  `LlmProviderError` to surface the canonical category
  (`provider_rate_limit`, `provider_invalid_request`, etc.) in the
  error message. The image URL failure mode (OpenAI's
  fetcher hitting a CDN that blocks it) lands here as
  `provider_invalid_request`. Four legitimate places to handle
  this in production: caller-side `try / except NodeException`
  (shown here), call-level retry via `complete(retry=...)` for
  transient categories (also shown here, on the respond node),
  `RetryMiddleware` wrapping the whole respond node, or a
  `try / except LlmProviderError` inside the node body returning a
  fallback response.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. Host root only.
- ``LLM_MODEL`` defaults to ``gpt-4o-mini`` (a vision-capable model
  needed for the multimodal turn).
- ``LLM_API_KEY`` required (empty for local servers that don't
  authenticate, but the model MUST support vision blocks).
- ``IMAGE_URL`` overrides the default image URL. Default is a
  public-domain NASA photograph of the Apollo 16 Lunar Module
  "Orion" on the lunar surface, served from NASA's images-assets
  archive. OpenAI's vision pipeline downloads the image; some hosts
  (e.g., upload.wikimedia.org) block its fetcher with a
  ProviderInvalidRequest. images-assets.nasa.gov is known to work.

Run with:

    uv sync --group examples --all-extras

    # Clean conversation output only (default).
    LLM_API_KEY=sk-... uv run python examples/chat-with-multimodal/main.py

    # With OTel JSON spans streaming to stderr alongside the chat.
    LLM_API_KEY=sk-... uv run python examples/chat-with-multimodal/main.py --traces

(``--all-extras`` pulls in ``opentelemetry-sdk`` for the OTel observer.)
The conversation transcript streams to stdout as each turn closes,
with a short visual delay between turns (~``_TURN_DELAY_S``).  Pass
``--traces`` to also see the OTel observer attached and node + LLM
spans dumped to stderr; the OTel side is optional supporting
infrastructure, not the headline of this example (the
observer-hooks example owns that story).

The demo is illustrative only: it runs four pre-scripted user turns
sequentially in one process. A real chat-server runtime would
manage one invocation per turn with the chat history persisted
across sessions (e.g., via a checkpointer keyed on session_id);
that's the checkpointing-and-migration example's territory,
combined with this one's chat shape.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import Field

from openarmature.graph import (
    END,
    EndSentinel,
    GraphBuilder,
    NodeException,
    State,
    append,
)
from openarmature.graph.middleware import RetryConfig
from openarmature.llm import (
    AssistantMessage,
    LlmProviderError,
    Message,
    OpenAIProvider,
    RuntimeConfig,
    UserMessage,
)
from openarmature.observability.otel import OTelObserver
from openarmature.prompts import (
    ChatPrompt,
    ContentBlockTemplate,
    ContentSegment,
    ImageURLBlockTemplate,
    PlaceholderSegment,
    Prompt,
    PromptManager,
    TextBlockTemplate,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Default image: NASA public-domain photograph of the Apollo 16 Lunar
# Module "Orion" parked on the lunar surface during the first EVA,
# served from NASA's official images-assets archive (the canonical
# NASA media library).
#
# Important: OpenAI's vision pipeline downloads the image from this
# URL during the chat completion call.  Some CDNs (notably
# ``upload.wikimedia.org``) block OpenAI's image fetcher and return a
# ``ProviderInvalidRequest`` from the API.  ``images-assets.nasa.gov``
# is known to work; if you override ``IMAGE_URL``, point at a host
# that allows OpenAI's user agent.
DEFAULT_IMAGE_URL = "https://images-assets.nasa.gov/image/as16-113-18334/as16-113-18334~orig.jpg"


# ---------------------------------------------------------------------------
# Provider (lazy-init)
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
# User turn shape (script-driven)
# ---------------------------------------------------------------------------
# Each scripted turn is a question with an optional image URL.  The
# multimodal turn supplies an image_url; text-only turns leave it None.


@dataclass(frozen=True)
class UserTurn:
    text: str
    image_url: str | None = None


# ---------------------------------------------------------------------------
# Chat prompt construction
# ---------------------------------------------------------------------------
# A small in-process function rather than a backend.  The point of this
# example is the placeholder + segment shape, not backend wiring (07
# covers FilesystemPromptBackend; 10 covers Langfuse).  A real
# deployment would either:
#   - fetch the chat template from LangfusePromptBackend, or
#   - load it from a FilesystemPromptBackend chat-prompt sidecar once
#     the backend grows chat support (the current filesystem backend
#     only emits TextPrompt).

_SYSTEM_INSTRUCTIONS = (
    "You are a lunar-mission expert assistant.  Answer questions about "
    "Apollo and Artemis missions concisely and factually.  When the user "
    "attaches an image, describe what you see in the image and connect it "
    "to the mission context the user provided.  Keep responses to "
    "two or three sentences."
)

# Stable build-time stamp for the inline-constructed prompt.  ``fetched_at``
# is meaningful for prompts pulled from a remote backend (when did we last
# sync); for the inline-built prompt in this demo it's just "process
# startup" so a constant is more honest than ``datetime.now()`` per turn.
_PROMPT_BUILT_AT = datetime.now(UTC)


def _build_chat_prompt(text: str, image_url: str | None) -> ChatPrompt:
    """Build the chat template for one turn.

    System and history-placeholder segments are identical across turn
    shapes; only the trailing user segment changes:

    - Text-only turn: ``ContentSegment(role="user", content=text)``.
    - Multimodal turn: ``ContentSegment(role="user",
      content=[TextBlockTemplate, ImageURLBlockTemplate])``.

    Constructing the template per-turn keeps the example self-contained;
    a production deployment would fetch a versioned template from a
    PromptBackend and pass the image_url through variables instead.
    """
    user_content: str | list[ContentBlockTemplate]
    if image_url is not None:
        user_content = [
            TextBlockTemplate(text=text),
            ImageURLBlockTemplate(url=image_url),
        ]
    else:
        user_content = text
    return ChatPrompt(
        name="lunar-chat",
        version="v1",
        label="production",
        template_hash="sha256:lunar-chat-v1",
        fetched_at=_PROMPT_BUILT_AT,
        chat_template=[
            ContentSegment(role="system", content=_SYSTEM_INSTRUCTIONS),
            PlaceholderSegment(placeholder="history"),
            ContentSegment(role="user", content=user_content),
        ],
    )


# ---------------------------------------------------------------------------
# Prompt manager
# ---------------------------------------------------------------------------
# ``PromptManager.render(prompt, ...)`` accepts a ``Prompt`` directly, so
# the example calls render() with the inline-built ChatPrompt rather
# than round-tripping through a backend's fetch().  The manager
# constructor requires at least one backend, so a no-op stub satisfies
# the contract without participating in execution.  Production
# deployments would supply a real backend (LangfusePromptBackend etc.)
# and call ``manager.fetch(name, label)`` to retrieve the versioned
# prompt before rendering.


class _NoFetchBackend:
    """Stub backend purely to satisfy PromptManager's constructor.

    The example constructs ChatPrompt objects inline (see
    ``_build_chat_prompt``) and calls ``manager.render()`` directly, so
    ``fetch()`` is never invoked.
    """

    async def fetch(self, name: str, label: str = "production") -> Prompt:
        raise NotImplementedError("example constructs prompts inline; fetch not used")


_PROMPT_MANAGER = PromptManager(_NoFetchBackend())


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# ``history`` is the conversation memory: the running list of user +
# assistant Message pairs from all prior turns.  Declared with the
# ``append`` reducer so each respond-node update concatenates the two
# new messages (current user turn + assistant response) rather than
# overwriting prior history.
#
# ``user_turns`` is the pre-scripted list of turns the demo runs;
# ``next_turn_index`` advances by one per respond call.  In a real
# chat server this would not be on state; turns arrive one per
# invocation rather than as a pre-scripted batch.  Keeping the
# scripted shape here lets the demo run end-to-end without an
# interactive prompt.


class ChatState(State):
    user_turns: list[UserTurn]
    next_turn_index: int = 0
    history: Annotated[list[Message], append] = Field(default_factory=list[Message])


# Visual pacing between turns when printing the transcript.  Tiny
# delay so the human reader can follow the conversation as it
# arrives rather than seeing the full thing dump at once; tune via
# the constant rather than per-turn.
_TURN_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def respond(state: ChatState) -> dict[str, Any]:
    """Render the chat template for the current turn, call the LLM,
    append both the new user message and the assistant response to
    history.
    """
    turn = state.user_turns[state.next_turn_index]

    # Build a fresh ChatPrompt per turn (text-only or multimodal) and
    # render directly through the manager; no fetch round-trip needed
    # since we have the Prompt in hand.
    prompt = _build_chat_prompt(turn.text, turn.image_url)
    rendered = _PROMPT_MANAGER.render(
        prompt,
        variables={},
        placeholders={"history": state.history},
    )

    # Call-level retry: retry only the provider wire call on transient
    # categories (provider_unavailable, provider_rate_limit, ...), using
    # the default classifier and backoff. It is terminal-only, so the
    # node still sees exactly one completion (or one final failure) even
    # when an attempt was retried underneath. Contrast with a
    # RetryMiddleware on the node, which re-runs the whole node body
    # (re-render + re-send) on each retry.
    response = await _get_provider().complete(
        rendered.messages,
        config=RuntimeConfig(temperature=0.0, max_tokens=400),
        retry=RetryConfig(max_attempts=3),
    )

    # The rendered messages include [system, *history, current_user]
    # for THIS chat_template shape.  ``rendered.messages[-1]`` is the
    # current user turn because the user ContentSegment is the last
    # segment in ``_build_chat_prompt``'s template; if the template
    # ever grows a trailing assistant or system segment, this index
    # has to move.  Append (current_user, assistant_response) to
    # history so the next turn sees the full conversation.  The system
    # message is part of the template, not part of history.
    current_user_message = rendered.messages[-1]
    assert isinstance(current_user_message, UserMessage), (
        "expected rendered messages to end with the new user turn"
    )

    # Print the turn immediately so the conversation streams to the
    # reader as the graph executes; otherwise the chat would only
    # appear after invoke() returns.  Side effects inside a node body
    # are fine; the alternative (a custom observer reacting to
    # ``completed`` events) would be more "OA-native" but adds
    # boilerplate that distracts from this example's headline.
    print(_format_turn(state.next_turn_index, turn, response.message))
    await asyncio.sleep(_TURN_DELAY_S)

    return {
        "next_turn_index": state.next_turn_index + 1,
        "history": [current_user_message, response.message],
    }


# Single cap for both user text and assistant response in the trace
# transcript.  Keeps the printout scannable without privileging one
# side; either both sides truncate or neither.
_TRANSCRIPT_LINE_CAP = 240


def _truncate(s: str, cap: int = _TRANSCRIPT_LINE_CAP) -> str:
    if len(s) <= cap:
        return s
    return s[: cap - 3] + "..."


def _format_turn(turn_index: int, turn: UserTurn, assistant: AssistantMessage) -> str:
    image_tag = " [+image]" if turn.image_url is not None else ""
    user_short = _truncate(turn.text)
    assistant_short = _truncate(assistant.content or "")
    return f"\n--- Turn {turn_index}{image_tag} ---\nUSER:      {user_short}\nASSISTANT: {assistant_short}"


def route_after_respond(state: ChatState) -> str | EndSentinel:
    """Loop back for the next turn or exit when the scripted turns run out."""
    if state.next_turn_index < len(state.user_turns):
        return "respond"
    return END


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    return (
        GraphBuilder(ChatState)
        .add_node("respond", respond)
        .add_conditional_edge("respond", route_after_respond)
        .set_entry("respond")
        .compile()
    )


# ---------------------------------------------------------------------------
# Observer (console)
# ---------------------------------------------------------------------------
# OTel observer with a console exporter emits one span per node
# boundary.  Inside the respond node, the LLM provider emits the
# ``openarmature.llm.complete`` span carrying the GenAI semconv
# attributes (gen_ai.system, model, usage tokens) plus, per turn, the
# prompt identity if the manager's ``with_active_prompt`` scope is
# active. The demo runs without that scope wrapping to keep the
# loop tight.


def _build_observer() -> OTelObserver:
    exporter = ConsoleSpanExporter()
    processor = SimpleSpanProcessor(exporter)
    return OTelObserver(
        span_processor=processor,
        resource=Resource.create({"service.name": "openarmature-chat-multimodal"}),
    )


# ---------------------------------------------------------------------------
# Scripted conversation
# ---------------------------------------------------------------------------
# Four turns: a factual opener, a follow-up that depends on the first
# answer, a multimodal turn with an image, and a closing follow-up.
# The multimodal turn intentionally references "the image you just
# saw" in the next turn to confirm conversation memory carries the
# multimodal context across turns.


def _scripted_turns(image_url: str) -> list[UserTurn]:
    return [
        UserTurn(text="What was the primary objective of Apollo 11?"),
        UserTurn(text="And what year did it launch?"),
        UserTurn(
            text=("I have a photograph of the Lunar Module. What's distinctive about its design?"),
            image_url=image_url,
        ),
        UserTurn(
            text=("Given what you described about the LM, was that design reused on later Apollo missions?"),
        ),
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-turn chat demo with a multimodal turn. "
            "Conversation streams to stdout as each turn completes."
        )
    )
    parser.add_argument(
        "--traces",
        action="store_true",
        help=(
            "Attach the OTel observer with a console exporter so node + LLM spans "
            "stream to stderr as JSON. Off by default for a cleaner first-read; "
            "turn on to see the observability shape end-to-end."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    image_url = os.environ.get("IMAGE_URL", DEFAULT_IMAGE_URL)

    graph = build_graph()
    if args.traces:
        graph.attach_observer(_build_observer())

    initial = ChatState(user_turns=_scripted_turns(image_url))

    print("=== openarmature chat-with-multimodal demo ===")
    print(f"Image URL: {image_url}")
    print(f"Scripted turns: {len(initial.user_turns)}")
    if args.traces:
        print("OTel traces: ON (spans stream to stderr as each node closes)")
    print()

    # Catch the engine-level wrapper ``NodeException`` at the
    # ``invoke()`` boundary.  The underlying error is attached via
    # Python's standard exception-chaining as ``exc.__cause__``; if
    # it's an ``LlmProviderError`` we surface the canonical
    # ``.category`` string (``provider_rate_limit``,
    # ``provider_invalid_request``, etc.) so the failure mode is
    # immediately greppable.  This is one of four legitimate places
    # to handle the error; see the docstring for the others
    # (call-level ``complete(retry=...)``, ``RetryMiddleware`` wrapping
    # the node, ``try/except`` inside the node body).
    final: ChatState | None = None
    try:
        final = await graph.invoke(initial)
    except NodeException as exc:
        cause = exc.__cause__
        if isinstance(cause, LlmProviderError):
            category = cause.category
        else:
            category = type(cause).__name__ if cause is not None else "<unknown>"
        print()
        print(f"*** node {exc.node_name!r} failed ({category}): {cause} ***")
        print()
        print("Four places to handle this in production code:")
        print("  - Caller-side try/except NodeException (this example).")
        print("  - Call-level complete(retry=...) on the wire call (this example).")
        print("  - RetryMiddleware on the node for transient categories.")
        print("  - try/except inside the node body returning a fallback.")
    finally:
        await graph.drain()
        await _get_provider().aclose()

    if final is None:
        return

    print()
    print(
        f"=== history length: {len(final.history)} messages "
        f"({len(final.history) // 2} user/assistant turns) ==="
    )


if __name__ == "__main__":
    asyncio.run(main())
