"""openarmature demo: two independent analyses of a lunar-mission
photograph using versioned prompt templates, a fallback prompt
backend, and a multimodal user message.

**Use case:** Given a photograph from a lunar mission, run two
independent analyses: describe the lunar surface visible
(``describe-surface``) and identify the equipment (``describe-equipment``).
Both prompts take the mission name as their only variable; neither
depends on the other's output. Both renders are grouped under one
observability ``PromptGroup`` so a trace UI can render the analyses
as one logical unit.

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
  layout. The demo ships two prompts (``describe-surface``,
  ``describe-equipment``) under the primary backend's ``production``
  label, plus matching variants in the fallback backend so the safety
  net covers both prompts.
- ``PromptGroup(group_name=..., members=[result_a, result_b])`` wraps
  two ``PromptResult`` instances under one observability identifier.
  Because the prompts are INDEPENDENT analyses of the same input,
  both can be rendered upfront with real variables; no placeholder
  renders, no asymmetric "first call computes the second's input"
  shape.
- ``with_active_prompt_group(group)`` propagates the group name via
  ContextVar; the attached ``OTelObserver`` stamps
  ``openarmature.prompt.group_name`` onto every LLM-call span fired
  inside the block. Confirm in the console output; the two
  ``openarmature.llm.complete`` spans both carry
  ``openarmature.prompt.group_name = "lunar-image-analysis"``.
- ``with_active_prompt(result)`` (inside the group's scope) propagates
  the per-call prompt identifiers; name, version, label,
  template_hash, rendered_hash. The two layers compose: each LLM
  span carries the group identifier AND the per-call prompt
  identifiers. The console output makes both visible.
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

    uv sync --group examples --all-extras
    cd examples/multimodal-prompt
    LLM_API_KEY=sk-... uv run python main.py
    LLM_API_KEY=sk-... IMAGE_PATH=./my-photo.jpg uv run python main.py

(``--all-extras`` pulls in ``opentelemetry-sdk`` for the OTel observer.)
"""

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
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
from openarmature.observability.otel import OTelObserver
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
#   - primary: ``prompts/``; ships describe-surface and
#     describe-equipment.
#   - fallback: ``prompts_fallback/``; ships shorter variants of
#     both prompts so the safety net covers the whole pipeline. The
#     fallback path fires when the primary raises
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


class AnalysisState(State):
    # Exactly one of ``image_url`` / ``image_path`` is set when the
    # demo runs; the helper below picks the right ImageSource shape.
    image_url: str = ""
    image_path: str = ""
    mission: str
    surface_description: str = ""
    equipment_description: str = ""
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


async def describe_surface(s: AnalysisState) -> Mapping[str, Any]:
    # Each node fetches + renders its own prompt. Both prompts take
    # only the ``mission`` variable, so neither depends on the other's
    # output; the two analyses are independent.
    rendered = await _PROMPT_MANAGER.get(
        "describe-surface",
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
    # a ``group_name`` onto the same span; the two layers compose
    # so observers see both per-call AND per-group attribution.
    with with_active_prompt(rendered):
        response = await _get_provider().complete([multimodal_message])

    return {
        "surface_description": (response.message.content or "").strip(),
        "trace": ["describe_surface"],
    }


async def describe_equipment(s: AnalysisState) -> Mapping[str, Any]:
    rendered = await _PROMPT_MANAGER.get(
        "describe-equipment",
        variables={"mission": s.mission},
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

    return {
        "equipment_description": (response.message.content or "").strip(),
        "trace": ["describe_equipment"],
    }


def build_graph() -> CompiledGraph[AnalysisState]:
    return (
        GraphBuilder(AnalysisState)
        .add_node("describe_surface", describe_surface)
        .add_node("describe_equipment", describe_equipment)
        .add_edge("describe_surface", "describe_equipment")
        .add_edge("describe_equipment", END)
        .set_entry("describe_surface")
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
    print("Lunar-mission image analysis (surface + equipment)")
    print("=" * 72)
    print()
    print(f"  mission:   {mission}")
    if image_path:
        print(f"  image:     {image_path} (inline / base64)")
    else:
        print(f"  image:     {image_url} (url)")
    print()

    # Pre-render both prompts with the real ``mission`` variable so
    # the PromptGroup can be built once at invoke entry. Both renders
    # are honest; the nodes use the same fetch+render path inside,
    # so no placeholder identities sneak into the group's metadata.
    surface_member = await _PROMPT_MANAGER.get(
        "describe-surface",
        variables={"mission": mission},
    )
    equipment_member = await _PROMPT_MANAGER.get(
        "describe-equipment",
        variables={"mission": mission},
    )
    group = PromptGroup(
        group_name="lunar-image-analysis",
        members=[surface_member, equipment_member],
    )

    # Attach an OTel observer with a console exporter so the prompt-
    # context attributes the ``with_active_prompt`` / ``_group``
    # blocks below propagate become visible. Every LLM-call span
    # printed to stdout will carry ``openarmature.prompt.group_name``
    # (from the group context) plus the per-call
    # ``openarmature.prompt.{name, version, label, template_hash,
    # rendered_hash}`` attributes. A production setup would point a
    # ``BatchSpanProcessor`` at a real OTLP endpoint instead.
    otel_observer = OTelObserver(
        span_processor=SimpleSpanProcessor(ConsoleSpanExporter()),
        resource=Resource.create({"service.name": "openarmature-demo-multimodal"}),
    )

    graph = build_graph()
    graph.attach_observer(otel_observer)
    try:
        # ``with_active_prompt_group`` propagates the group_name to
        # observers for the duration of the invoke. Inside the nodes,
        # ``with_active_prompt`` adds the per-call prompt identifiers
        # alongside it; both layers stamp attributes on the same
        # LLM-call span. The OTel observer above captures both.
        with with_active_prompt_group(group):
            final = await graph.invoke(
                AnalysisState(
                    image_url=image_url if not image_path else "",
                    image_path=image_path,
                    mission=mission,
                    group_name=group.group_name,
                )
            )

        print(f"  group:                {final.group_name}")
        print(f"  describe-surface:     {surface_member.name} @ {surface_member.version}")
        print(f"  describe-equipment:   {equipment_member.name} @ {equipment_member.version}")
        print()
        print("  surface description:")
        print(f"    {final.surface_description}")
        print()
        print("  equipment description:")
        print(f"    {final.equipment_description}")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
