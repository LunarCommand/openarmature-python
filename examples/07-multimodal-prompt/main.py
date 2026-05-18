"""openarmature demo: caption a historical lunar photograph using a
versioned prompt template plus a multimodal user message.

**Use case:** Given a photograph from a lunar mission and the mission's
name, describe what's visible in the image. The text instructions are
loaded from a versioned prompt template on disk so they can be edited,
diffed, and rolled out independently of the code. The image is passed
to the model alongside the rendered text as a multimodal user message.

This is the "prompt management + image input" shape — two openarmature
surfaces that compose cleanly. The prompt manager gives you traceable,
hashable, version-tagged instruction text; content blocks give you the
multimodal payload alongside it.

**What's interesting in the implementation:**

- ``FilesystemPromptBackend`` loads ``caption-lunar-image.j2`` from
  ``prompts/production/``. The layout is ``<root>/<label>/<name>.j2``;
  the ``label`` ("production" here) is the rollout channel.
- ``PromptManager(backend)`` wraps the backend. ``manager.get(name,
  variables={...})`` fetches and renders in one call, returning a
  ``PromptResult`` whose ``messages`` carries the rendered text and
  whose ``template_hash`` / ``rendered_hash`` identify exactly which
  template+variables produced this output.
- ``with_active_prompt(result)`` is a context manager. While it's
  active, OTel observers see ``openarmature.prompt.*`` attributes
  stamped onto any LLM-call span fired inside the block. No OTel
  observer is attached in this demo (keeps the output focused on the
  caption), but the wrapping is the canonical pattern for production.
- The rendered text becomes a ``TextBlock`` inside a multimodal
  ``UserMessage``; the image is a sibling ``ImageBlock`` carrying an
  ``ImageSourceURL``. The provider passes both to the model in one
  call.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini`` (a vision-capable model).
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).
- ``IMAGE_URL`` overrides the default image. Default is a public-domain
  NASA photograph of Buzz Aldrin on the lunar surface.

Run with:

    uv sync --group examples
    cd examples/07-multimodal-prompt
    LLM_API_KEY=sk-... uv run python main.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
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
    ImageBlock,
    ImageSourceURL,
    OpenAIProvider,
    TextBlock,
    UserMessage,
)
from openarmature.prompts import (
    FilesystemPromptBackend,
    PromptManager,
    with_active_prompt,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Default image: the iconic Apollo 11 photograph of Buzz Aldrin posing
# next to the deployed seismic experiment on the lunar surface. Hosted on
# Wikimedia Commons (public-domain NASA imagery).
DEFAULT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/9/98/Aldrin_Apollo_11_original.jpg"
DEFAULT_MISSION = "Apollo 11"

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


# Build the prompt manager once at import time. The manager is cheap to
# construct, holds no per-call state, and is safe to share across nodes.
_PROMPT_ROOT = Path(__file__).parent / "prompts"
_PROMPT_MANAGER = PromptManager(FilesystemPromptBackend(_PROMPT_ROOT))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class CaptionState(State):
    image_url: str
    mission: str
    caption: str = ""
    prompt_version: str = ""
    template_hash: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def caption(s: CaptionState) -> Mapping[str, Any]:
    # Load + render the template in one call. ``variables`` are strict:
    # an undefined name in the template raises PromptRenderError.
    rendered = await _PROMPT_MANAGER.get(
        "caption-lunar-image",
        variables={"mission": s.mission},
    )

    # The PromptResult's messages list carries the rendered text as a
    # UserMessage. Pull out the text and compose a multimodal user
    # message that also carries the image.
    rendered_msg = rendered.messages[0]
    assert isinstance(rendered_msg, UserMessage)
    rendered_text = rendered_msg.content
    assert isinstance(rendered_text, str)

    multimodal_message = UserMessage(
        content=[
            TextBlock(text=rendered_text),
            ImageBlock(source=ImageSourceURL(url=s.image_url)),
        ],
    )

    # ``with_active_prompt`` propagates the prompt identifiers via
    # ContextVar to any observer that cares. An OTel observer would
    # stamp openarmature.prompt.{name,version,label,template_hash,
    # rendered_hash} on the LLM-call span fired inside this block. No
    # observer is attached in this demo, but the wrapping is the
    # canonical pattern; leaving it out drops the audit trail.
    with with_active_prompt(rendered):
        response = await _get_provider().complete([multimodal_message])

    return {
        "caption": (response.message.content or "").strip(),
        "prompt_version": rendered.version,
        "template_hash": rendered.template_hash,
        "trace": ["caption"],
    }


def build_graph() -> CompiledGraph[CaptionState]:
    return (
        GraphBuilder(CaptionState)
        .add_node("caption", caption)
        .add_edge("caption", END)
        .set_entry("caption")
        .compile()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    image_url = os.environ.get("IMAGE_URL", DEFAULT_IMAGE_URL)
    mission = os.environ.get("MISSION", DEFAULT_MISSION)

    print("=" * 72)
    print("Caption a lunar photograph using a versioned prompt template")
    print("=" * 72)
    print()
    print(f"  mission:   {mission}")
    print(f"  image_url: {image_url}")
    print()

    graph = build_graph()
    try:
        final = await graph.invoke(CaptionState(image_url=image_url, mission=mission))
        print(f"  prompt:    caption-lunar-image @ {final.prompt_version}")
        print(f"  template:  {final.template_hash}")
        print()
        print("  caption:")
        print(f"    {final.caption}")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
