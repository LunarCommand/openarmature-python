"""openarmature demo: caption and identify a lunar mission photograph
using versioned prompt templates, a fallback prompt backend, and a
multimodal user message.

**Use case:** Given a photograph from a lunar mission, run two prompts
in sequence: first describe what's visible (``caption-lunar-image``),
then use that caption alongside the same image to identify the specific
mission (``identify-mission``). Both prompts are versioned templates on
disk; both renders are grouped under one observability ``PromptGroup``
so a trace UI can render them as a single logical unit.

The image can come from a public URL (default) or a local file (set
``IMAGE_PATH`` to use the inline base64 source instead). The
``PromptManager`` is wired with a primary + fallback
``FilesystemPromptBackend`` to demonstrate composite-backend
configuration; the fallback path fires only when the primary raises
``PromptStoreUnavailable`` (e.g., a remote Langfuse backend off-line).

**What's interesting in the implementation:**

- ``PromptManager(primary, fallback)`` accepts multiple backends. On
  every ``fetch``, the manager tries them in order: if a backend
  raises ``PromptStoreUnavailable`` the manager continues to the
  next; if it raises ``PromptNotFound`` the chain stops (the name is
  legitimately missing). The typical production shape is "Langfuse
  primary + local-filesystem fallback".
- ``FilesystemPromptBackend`` uses the ``<root>/<label>/<name>.j2``
  layout. The demo ships two prompts (``caption-lunar-image``,
  ``identify-mission``) under the primary backend's ``production``
  label, plus a sibling backend rooted at a different folder for the
  fallback demonstration.
- ``PromptGroup(group_name=..., members=[result_a, result_b])`` wraps
  two ``PromptResult`` instances under one observability identifier.
  ``with_active_prompt_group(group)`` propagates the group name via
  ContextVar; OTel observers stamp ``openarmature.prompt.group_name``
  onto every LLM-call span fired inside.
- ``with_active_prompt(result)`` (inside the group's scope) propagates
  the per-call prompt identifiers — name, version, label,
  template_hash, rendered_hash. The two layers compose: spans inside
  the group see both the group identifier AND the per-call prompt
  identifiers.
- The rendered text becomes a ``TextBlock`` inside a multimodal
  ``UserMessage``; the image is a sibling ``ImageBlock``. The image
  source is ``ImageSourceURL(url=...)`` by default; setting
  ``IMAGE_PATH`` switches to ``ImageSourceInline(base64_data=...)``
  with the file's bytes base64-encoded and an inferred ``media_type``.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini`` (a vision-capable model).
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).
- ``IMAGE_URL`` overrides the default URL. Default is a public-domain
  NASA photograph of Buzz Aldrin on the lunar surface.
- ``IMAGE_PATH`` overrides the URL with a local file path. The file's
  bytes go to the model via ``ImageSourceInline`` (base64) instead.

Run with:

    uv sync --group examples
    cd examples/07-multimodal-prompt
    LLM_API_KEY=sk-... uv run python main.py
    LLM_API_KEY=sk-... IMAGE_PATH=./my-photo.jpg uv run python main.py
"""

from __future__ import annotations

import asyncio
import base64
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
    ImageSource,
    ImageSourceInline,
    ImageSourceURL,
    OpenAIProvider,
    TextBlock,
    UserMessage,
)
from openarmature.prompts import (
    FilesystemPromptBackend,
    PromptGroup,
    PromptManager,
    PromptResult,
    with_active_prompt,
    with_active_prompt_group,
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
#
# Two backends are wired here:
#   - primary: ``prompts/`` — ships caption-lunar-image and
#     identify-mission.
#   - fallback: ``prompts_fallback/`` — ships shorter variants of
#     BOTH prompts so the safety net actually covers the whole
#     pipeline. The fallback path fires when the primary raises
#     ``PromptStoreUnavailable`` (e.g., a remote primary like
#     Langfuse times out); ``PromptNotFound`` from primary stops the
#     chain (the name is legitimately missing).
#
# In this demo both prompts live in primary, so the fallback path
# isn't exercised at runtime. The construction-time setup is the
# demonstrated thing; production code would replace primary with a
# remote backend (LangfusePromptBackend etc.) while keeping the
# filesystem one as the offline safety net.
_PROMPT_ROOT_PRIMARY = Path(__file__).parent / "prompts"
_PROMPT_ROOT_FALLBACK = Path(__file__).parent / "prompts_fallback"
_PROMPT_MANAGER = PromptManager(
    FilesystemPromptBackend(_PROMPT_ROOT_PRIMARY),
    FilesystemPromptBackend(_PROMPT_ROOT_FALLBACK),
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class CaptionState(State):
    # Exactly one of ``image_url`` / ``image_path`` is set when the
    # demo runs; the helper below picks the right ImageSource shape.
    image_url: str = ""
    image_path: str = ""
    mission: str
    caption: str = ""
    identified_mission: str = ""
    group_name: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Image source helper
# ---------------------------------------------------------------------------
# The image arrives either as a URL (default) or a local file path
# (``IMAGE_PATH`` env var). The helper picks the right ``ImageSource``
# shape: ``ImageSourceURL`` passes the URL through to the model
# unchanged; ``ImageSourceInline`` reads the file, base64-encodes the
# bytes, and requires a ``media_type`` on the parent ``ImageBlock``.

_EXTENSION_TO_MEDIA_TYPE = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _build_image_block(image_url: str, image_path: str) -> ImageBlock:
    if image_path:
        data = Path(image_path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        ext = Path(image_path).suffix.lower()
        media_type = _EXTENSION_TO_MEDIA_TYPE.get(ext)
        if media_type is None:
            raise RuntimeError(
                f"image extension {ext!r} not recognized; supported: "
                f"{sorted(_EXTENSION_TO_MEDIA_TYPE.keys())}"
            )
        source: ImageSource = ImageSourceInline(base64_data=encoded)
        return ImageBlock(source=source, media_type=media_type)
    return ImageBlock(source=ImageSourceURL(url=image_url))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _extract_rendered_text(rendered: PromptResult) -> str:
    """Pull the rendered text out of a single-UserMessage PromptResult,
    failing loudly if the contract shape changes."""
    rendered_msg = rendered.messages[0]
    if not isinstance(rendered_msg, UserMessage) or not isinstance(rendered_msg.content, str):
        raise RuntimeError(
            "PromptManager.render() returned an unexpected shape; expected a single "
            f"UserMessage with str content, got {type(rendered_msg).__name__} "
            f"with content type {type(rendered_msg.content).__name__}"
        )
    return rendered_msg.content


async def caption(s: CaptionState) -> Mapping[str, Any]:
    # Each node fetches + renders its own prompt. ``get`` is the
    # convenience shorthand for ``render(await fetch(...))``.
    rendered = await _PROMPT_MANAGER.get(
        "caption-lunar-image",
        variables={"mission": s.mission},
    )
    rendered_text = _extract_rendered_text(rendered)

    multimodal_message = UserMessage(
        content=[
            TextBlock(text=rendered_text),
            _build_image_block(s.image_url, s.image_path),
        ],
    )

    # ``with_active_prompt`` propagates the per-call prompt
    # identifiers (name, version, label, template_hash,
    # rendered_hash) via ContextVar. An OTel observer would stamp
    # those onto the LLM-call span fired inside the block. The
    # outer ``with_active_prompt_group`` (set in main()) ALSO stamps
    # a ``group_name`` onto the same span — the two layers compose
    # so observers see both per-call AND per-group attribution.
    with with_active_prompt(rendered):
        response = await _get_provider().complete([multimodal_message])

    return {
        "caption": (response.message.content or "").strip(),
        "trace": ["caption"],
    }


async def identify(s: CaptionState) -> Mapping[str, Any]:
    # Uses the caption produced by the previous node — so the render
    # happens here, not in main(). Same with_active_prompt wrapping;
    # the outer group context from main() still applies.
    rendered = await _PROMPT_MANAGER.get(
        "identify-mission",
        variables={"caption": s.caption},
    )
    rendered_text = _extract_rendered_text(rendered)

    multimodal_message = UserMessage(
        content=[
            TextBlock(text=rendered_text),
            _build_image_block(s.image_url, s.image_path),
        ],
    )

    with with_active_prompt(rendered):
        response = await _get_provider().complete([multimodal_message])

    identified = (response.message.content or "").strip().removeprefix("Mission:").strip()
    return {
        "identified_mission": identified,
        "trace": ["identify"],
    }


def build_graph() -> CompiledGraph[CaptionState]:
    return (
        GraphBuilder(CaptionState)
        .add_node("caption", caption)
        .add_node("identify", identify)
        .add_edge("caption", "identify")
        .add_edge("identify", END)
        .set_entry("caption")
        .compile()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    image_url = os.environ.get("IMAGE_URL", DEFAULT_IMAGE_URL)
    image_path = os.environ.get("IMAGE_PATH", "")
    mission = os.environ.get("MISSION", DEFAULT_MISSION)

    print("=" * 72)
    print("Caption + identify a lunar photograph")
    print("=" * 72)
    print()
    print(f"  mission:   {mission}")
    if image_path:
        print(f"  image:     {image_path} (inline / base64)")
    else:
        print(f"  image:     {image_url} (url)")
    print()

    # Pre-render both prompts with placeholder variables so the
    # PromptGroup can be built ONCE at invoke entry and set as the
    # outer observability context for the whole pipeline. The actual
    # per-call renders happen inside the nodes, picking up the real
    # ``caption`` variable that's only known after the first node
    # completes. The group's ``members`` list is a metadata hint
    # naming the two prompt slots; per-call wrapping inside the
    # nodes carries the exact-rendered identity for each call.
    caption_member = await _PROMPT_MANAGER.get(
        "caption-lunar-image",
        variables={"mission": mission},
    )
    identify_placeholder = await _PROMPT_MANAGER.get(
        "identify-mission",
        variables={"caption": "(provided at runtime)"},
    )
    group = PromptGroup(
        group_name="lunar-image-analysis",
        members=[caption_member, identify_placeholder],
    )

    graph = build_graph()
    try:
        # ``with_active_prompt_group`` propagates the group_name to
        # observers for the duration of the invoke. Inside the nodes,
        # ``with_active_prompt`` adds the per-call prompt identifiers
        # alongside it — both layers stamp attributes on the same
        # LLM-call span.
        with with_active_prompt_group(group):
            final = await graph.invoke(
                CaptionState(
                    image_url=image_url if not image_path else "",
                    image_path=image_path,
                    mission=mission,
                    group_name=group.group_name,
                )
            )

        print(f"  group:       {final.group_name}")
        print(f"  caption-prompt:   {caption_member.name} @ {caption_member.version}")
        print(f"  identify-prompt:  {identify_placeholder.name} @ {identify_placeholder.version}")
        print()
        print("  caption:")
        print(f"    {final.caption}")
        print()
        print(f"  identified mission:  {final.identified_mission}")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
