"""Prompt and PromptResult records.

Two prompt variants land on this module — the existing single-string
Text-prompt (:class:`TextPrompt`, formerly ``Prompt``) and the
role-tagged Chat-prompt (:class:`ChatPrompt`) carrying a list of
:class:`ChatSegment` entries.  The user-facing union alias
:data:`Prompt` covers both; callers ``isinstance``-narrow at the
consumption point.
"""
# Proposal 0046 (prompt-management §3.1): Text + Chat prompt variants.

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openarmature.llm.messages import Message
from openarmature.llm.response import RuntimeConfig


# SamplingConfig mirrors RuntimeConfig's declared-fields-plus-extras
# shape so `prompt.sampling` splats directly into `provider.complete()`
# without per-field translation (spec §12 cross-spec touchpoint;
# proposal 0033). Subclass rather than alias so the type system
# distinguishes the two names — a fetch returning
# ``SamplingConfig | None`` is meaningfully different in signatures
# from a provider call's ``RuntimeConfig`` argument. The subclass is
# empty today; future divergence (e.g., fields meaningful for prompts-
# at-rest but not for direct provider calls) lands on SamplingConfig
# without touching RuntimeConfig.
class SamplingConfig(RuntimeConfig):
    """Per-prompt sampling configuration. Shape-compatible with ``RuntimeConfig``."""


# Spec §3.1 *Chat-prompt variant* — content-blocks-template shapes
# mirroring llm-provider §3.1 ContentBlock shapes with variable-
# substitutable text fields.  The v1 set covers user-message
# authoring blocks (text + image); thinking / redacted-thinking
# blocks are assistant-side round-trip content and don't sit on
# the authored-template surface.
class TextBlockTemplate(BaseModel):
    """Text content block template.  Renders to an llm-provider text
    block carrying the variable-substituted text."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ImageURLBlockTemplate(BaseModel):
    """URL image content block template.  Renders to an llm-provider
    URL image block; ``url`` is variable-substituted."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageInlineBlockTemplate(BaseModel):
    """Inline base64 image content block template.  Renders to an
    llm-provider inline image block; ``base64_data`` and
    ``media_type`` are variable-substituted."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_inline"] = "image_inline"
    base64_data: str
    media_type: str
    detail: Literal["auto", "low", "high"] | None = None


ContentBlockTemplate = Annotated[
    TextBlockTemplate | ImageURLBlockTemplate | ImageInlineBlockTemplate,
    Field(discriminator="type"),
]


# Spec §3.1 placeholder regex: ASCII identifier shape to avoid
# collision with backend placeholder syntax (e.g., Langfuse {{name}}).
_PLACEHOLDER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ContentSegment(BaseModel):
    """One role-tagged content segment of a chat prompt.

    ``role`` is one of the three canonical authoring roles from the
    Message shape; the fourth role (``"tool"``) is intentionally
    excluded — tool-result messages have a distinct per-message shape
    that doesn't map to a template-author surface.  Tool-loop content
    flows through placeholder segments instead.

    ``content`` is either a single text template (the common case) or
    an ordered non-empty list of :class:`ContentBlockTemplate` entries
    for multimodal user messages (text + image).  Image blocks are
    user-only — a non-user role with an image-block-containing list
    raises ``prompt_render_error`` at render time.  Construction-time
    validation here surfaces the same condition earlier for ergonomic
    feedback.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["content"] = "content"
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentBlockTemplate]

    @model_validator(mode="after")
    def _check_role_blocks(self) -> ContentSegment:
        # Per spec §11 / msg 07: render-time is the spec-normative
        # trigger; construction-time enforcement is permitted as an
        # "ergonomic-only bonus" for hand-built prompts.  Both fire
        # the same canonical errors.  Backends or harnesses that
        # need to construct intentionally-invalid prompts (e.g., for
        # round-trip render-time error tests) bypass these checks
        # via ``ContentSegment.model_construct(...)``.
        if isinstance(self.content, list):
            if not self.content:
                raise ValueError("content-blocks segment: block list MUST be non-empty")
            if self.role != "user":
                for block in self.content:
                    if isinstance(block, (ImageURLBlockTemplate, ImageInlineBlockTemplate)):
                        raise ValueError(f"image blocks are user-only; got role={self.role!r}")
        return self


class PlaceholderSegment(BaseModel):
    """A placeholder slot in a chat prompt.  At render time the caller
    supplies a ``list[Message]`` to inject in place of this segment;
    an empty list injects zero messages (valid; the first-turn case),
    while an absent mapping entry raises ``prompt_render_error``.

    The ``placeholder`` name MUST match
    ``[A-Za-z_][A-Za-z0-9_]*`` — ASCII identifier shape — to avoid
    collision with backend placeholder syntax.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["placeholder"] = "placeholder"
    placeholder: str

    @model_validator(mode="after")
    def _check_name(self) -> PlaceholderSegment:
        # Per spec §11 / msg 07: render-time is the spec-normative
        # trigger; construction-time enforcement is the optional
        # "ergonomic bonus" for hand-built prompts.  Backends or
        # harnesses constructing intentionally-invalid names
        # (e.g., to verify the render-time error path) bypass this
        # check via ``PlaceholderSegment.model_construct(...)``.
        if not _PLACEHOLDER_NAME_RE.match(self.placeholder):
            raise ValueError(f"placeholder name {self.placeholder!r} MUST match [A-Za-z_][A-Za-z0-9_]*")
        return self


ChatSegment = Annotated[
    ContentSegment | PlaceholderSegment,
    Field(discriminator="type"),
]


class _PromptBase(BaseModel):
    """Shared identity / metadata for both prompt variants.

    Attributes:
        name: Stable identifier within the backend.
        version: Backend-defined version string.  Two distinct version
            strings denote distinct prompt contents.
        label: The label under which this prompt was fetched
            (e.g., "production", "latest", "variant-a").
        template_hash: SHA-256 of the canonical serialization of the
            prompt's template surface.  Format ``"sha256:<hex>"``.
        fetched_at: Time the prompt was fetched from its backend.
            When a caching backend serves a cached result,
            ``fetched_at`` MUST reflect the original fetch time, not
            the cache hit time.
        sampling: Optional per-prompt sampling configuration.  Splats
            into ``provider.complete(config=...)`` without translation.
        observability_entities: Optional backend-keyed references to
            first-class entities the prompt has been registered as in
            observability backends.  Spec-normative key:
            ``langfuse_prompt`` (the Langfuse SDK Prompt-entity ref).
        metadata: Optional backend-supplied metadata.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    label: str
    template_hash: str
    fetched_at: datetime
    sampling: SamplingConfig | None = None
    observability_entities: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class TextPrompt(_PromptBase):
    """An unrendered single-string template plus identity metadata.

    Renders to a single :class:`UserMessage` carrying the substituted
    template text.  Text-prompts render to exactly one Message with
    ``role: "user"``; multi-message and multimodal prompts go through
    :class:`ChatPrompt`.

    ``placeholders`` passed to ``PromptManager.render`` are ignored
    for Text-prompt rendering.
    """

    kind: Literal["text"] = "text"
    template: str


class ChatPrompt(_PromptBase):
    """A role-tagged, multi-segment chat prompt.

    ``chat_template`` is an ordered list of :class:`ChatSegment`
    entries — content segments carrying a role + content (text
    template or content-blocks template) and placeholder segments
    carrying a name that the caller fills at render time with a
    ``list[Message]``.  The rendered :class:`PromptResult.messages`
    is the in-order concatenation per segment.
    """

    kind: Literal["chat"] = "chat"
    chat_template: list[ChatSegment]

    @model_validator(mode="after")
    def _check_chat_template(self) -> ChatPrompt:
        # Per spec §11 / msg 07: render-time is the spec-normative
        # trigger; construction-time enforcement is the optional
        # "ergonomic bonus".  Backends or harnesses constructing
        # intentionally-invalid chat templates (e.g., to verify the
        # render-time error path) bypass via
        # ``ChatPrompt.model_construct(...)``.
        seen: set[str] = set()
        for seg in self.chat_template:
            if isinstance(seg, PlaceholderSegment):
                if seg.placeholder in seen:
                    raise ValueError(f"duplicate placeholder name {seg.placeholder!r} in chat_template")
                seen.add(seg.placeholder)
        return self


# Public union alias.  Per spec §3.1 a Prompt is one-of Text/Chat;
# discriminate via ``isinstance(prompt, ChatPrompt)`` at consumption
# sites.  The ``kind`` literal on each variant is the explicit type
# tag for Pydantic discriminator use (e.g., backend deserialization).
Prompt = Annotated[
    TextPrompt | ChatPrompt,
    Field(discriminator="kind"),
]


class PromptResult(BaseModel):
    """The rendered output of applying variables to a prompt.

    Carries the rendered ``Message`` sequence (ready to pass to
    ``Provider.complete()``) plus the source prompt's identity
    metadata and a ``rendered_hash`` that captures the rendered
    content.

    The ``rendered_hash`` is the cache-key value most useful to
    downstream consumers: two renders with the same template AND
    the same variables produce the same hash.

    Attributes:
        name: Propagated from the source Prompt.
        version: Propagated from the source Prompt.
        label: Propagated from the source Prompt.
        template_hash: Propagated from the source Prompt.
        rendered_hash: SHA-256 of the canonical serialization of
            the rendered messages list.
        messages: Ordered non-empty sequence of ``Message`` records.
        variables: Variable mapping used to render. v1 policy:
            pass-through unchanged (no automatic redaction). Keys
            are always preserved; future redaction policies would
            redact values, never strip keys.
        fetched_at: Propagated from the source Prompt.
        rendered_at: Time this PromptResult was rendered. Distinct
            from ``fetched_at``: a single fetched prompt may render
            many times.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    label: str
    template_hash: str
    rendered_hash: str
    messages: list[Message] = Field(min_length=1)
    variables: dict[str, Any]
    fetched_at: datetime
    rendered_at: datetime
    # Per spec §4: propagated verbatim from the source Prompt.
    # Rendering does NOT modify or reinterpret either field.
    sampling: SamplingConfig | None = None
    observability_entities: dict[str, Any] | None = None
