"""PromptManager; user-facing fetch + render + composite-fallback."""

from __future__ import annotations

import base64
import binascii
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import jinja2
from pydantic import ValidationError

from openarmature.llm.messages import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    ImageSourceInline,
    ImageSourceURL,
    Message,
    SystemMessage,
    TextBlock,
    UserMessage,
)

from .backend import PromptBackend
from .errors import PromptNotFound, PromptRenderError, PromptStoreUnavailable
from .hashing import compute_rendered_hash, compute_template_hash
from .label_resolver import SPEC_FALLBACK_LABEL, LabelResolver
from .prompt import (
    ChatPrompt,
    ContentSegment,
    ImageInlineBlockTemplate,
    ImageURLBlockTemplate,
    PlaceholderSegment,
    Prompt,
    PromptResult,
    TextBlockTemplate,
    TextPrompt,
)

_log = logging.getLogger(__name__)

# Render-time placeholder regex check (proposal 0046 §11).
_PLACEHOLDER_NAME_RE_RENDER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PromptManager:
    """Composes one or more PromptBackends and exposes fetch + render.

    Users interact with the manager; backends are an implementation
    detail of construction. The manager owns:

    - ``fetch``: consults backends in order per §9 (was §8) fallback semantics.
    - ``render``: synchronous local string transform; produces a
      ``PromptResult``.
    - ``get``: convenience: ``render(await fetch(...), variables)``.

    Constructor knobs:

    - ``label_resolver``: optional ``LabelResolver`` consulted by
      :meth:`fetch` / :meth:`get` when no explicit ``label`` argument
      is supplied (§6 step-2 of the fallback chain).
    - ``jinja_undefined``: Jinja ``Undefined`` subclass for render-time
      variable resolution. Default ``StrictUndefined`` matches spec
      §8 (was §7); pass ``jinja2.ChainableUndefined`` or any other
      ``Undefined`` subclass to opt out of strict-by-default rendering.
    """

    def __init__(
        self,
        *backends: PromptBackend,
        label_resolver: LabelResolver | None = None,
        jinja_undefined: type[jinja2.Undefined] = jinja2.StrictUndefined,
    ) -> None:
        if not backends:
            raise ValueError("PromptManager requires at least one backend")
        self._backends: tuple[PromptBackend, ...] = backends
        self._label_resolver = label_resolver
        # autoescape disabled by design: render output goes to an LLM
        # API call (plain text), not an HTML response. The env is
        # per-manager (was module-level) so jinja_undefined can be
        # overridden per-instance.
        self._render_env = jinja2.Environment(
            undefined=jinja_undefined,
            autoescape=False,
            keep_trailing_newline=True,
        )
        # template_hash → compiled jinja2 Template. Per-manager,
        # unbounded. Correct by construction: template_hash is
        # content-derived, so a backend returning updated content
        # surfaces a fresh hash and a fresh cache entry. An LRU
        # eviction policy can land if benchmarks ever show memory
        # pressure; typical apps have O(10) prompts.
        self._template_cache: dict[str, jinja2.Template] = {}

    def _resolve_label(self, label: str | None, name: str) -> str:
        # Spec §6 fallback chain:
        #   1. Explicit label argument wins (caller pinned it).
        #   2. Resolver is consulted when one was configured.
        #   3. Spec fallback "production" when neither applies.
        if label is not None:
            return label
        if self._label_resolver is not None:
            return self._label_resolver.resolve(name)
        return SPEC_FALLBACK_LABEL

    async def fetch(self, name: str, label: str | None = None) -> Prompt:
        """Consult composed backends in order, applying §9 (was §8) fallback.

        Label is resolved per §6's three-step chain: explicit
        argument > configured ``LabelResolver`` > spec fallback
        ``"production"``.

        - First successful fetch wins; further backends are not consulted.
        - ``PromptNotFound`` from any backend STOPS the chain: the
          error propagates. Logical absence MUST NOT silently
          substitute a stale alternative.
        - ``PromptStoreUnavailable`` from a backend continues to the
          next. After ALL backends are exhausted with unavailable
          failures, the manager raises ``PromptStoreUnavailable``.
        """
        resolved_label = self._resolve_label(label, name)
        causes: list[BaseException] = []
        for backend in self._backends:
            try:
                return await backend.fetch(name, resolved_label)
            except PromptNotFound:
                raise
            except PromptStoreUnavailable as exc:
                causes.append(exc)
                _log.warning(
                    "prompt backend %r unavailable for (%r, %r); falling back",
                    backend,
                    name,
                    resolved_label,
                )
                continue
        if not causes:
            # Unreachable under current control flow: the constructor
            # guarantees ``len(self._backends) >= 1`` and the only
            # fall-through path from the for-loop appends to
            # ``causes``. Explicit guard rather than ``assert`` so
            # the invariant holds under ``python -O`` (asserts get
            # stripped) — a future change that silently swallowed an
            # exception in the loop would surface here as a clear
            # RuntimeError instead of an opaque IndexError on the
            # next line.
            raise RuntimeError(
                "PromptManager.fetch internal invariant violated: no backends consulted but loop exhausted"
            )
        raise PromptStoreUnavailable(
            f"all prompt backends unavailable for ({name!r}, {resolved_label!r})",
            name=name,
            label=resolved_label,
            backends_tried=[type(b).__name__ for b in self._backends],
            causes=list(causes),
        ) from causes[-1]

    def render(
        self,
        prompt: Prompt,
        variables: Mapping[str, Any] | None = None,
        *,
        placeholders: Mapping[str, Sequence[Message]] | None = None,
    ) -> PromptResult:
        """Apply ``variables`` (and optionally ``placeholders``) and return a PromptResult.

        Render is synchronous; no I/O.  Variables are strict by
        default per §8: a template reference to a name not in
        ``variables`` raises ``PromptRenderError``.

        For a :class:`TextPrompt`, ``placeholders`` is ignored per
        spec §6 ("a Text-prompt renders to exactly one Message with
        ``role: "user"`` and ``content`` equal to the rendered
        template text").  Implementations MUST NOT raise on a
        non-empty ``placeholders`` mapping passed alongside a Text
        prompt.

        For a :class:`ChatPrompt`, the chat_template is rendered
        segment-by-segment per spec §6 — content segments substitute
        ``variables`` into the text (or per-block content) and
        produce one Message per segment; placeholder segments inject
        the caller-supplied ``list[Message]`` from
        ``placeholders[<name>]``.  An empty injected list is valid
        (the chat-history "first turn" case); an unfilled placeholder
        name raises ``prompt_render_error``.
        """
        variables = dict(variables or {})
        placeholders = placeholders or {}
        if isinstance(prompt, ChatPrompt):
            return self._render_chat(prompt, variables, placeholders)
        return self._render_text(prompt, variables)

    def _render_text(self, prompt: TextPrompt, variables: dict[str, Any]) -> PromptResult:
        rendered_text: str
        try:
            rendered_text = self._render_template_text(prompt.template_hash, prompt.template, variables)
        except PromptRenderError as exc:
            raise self._wrap_render_error(prompt, variables, exc) from exc

        try:
            messages: list[Message] = [UserMessage(content=rendered_text)]
            rendered_hash = compute_rendered_hash(messages)
        except ValidationError as exc:
            raise PromptRenderError(
                f"rendered output invalid for ({prompt.name!r}, {prompt.label!r}): {exc}",
                name=prompt.name,
                version=prompt.version,
                label=prompt.label,
                variables=variables,
                description=str(exc),
            ) from exc

        return self._build_result(prompt, variables, messages, rendered_hash)

    def _render_chat(
        self,
        prompt: ChatPrompt,
        variables: dict[str, Any],
        placeholders: Mapping[str, Sequence[Message]],
    ) -> PromptResult:
        # Spec §11 *Duplicate placeholder names*: enforce at render
        # time (the spec-normative trigger point).
        seen_placeholders: set[str] = set()
        for seg in prompt.chat_template:
            if isinstance(seg, PlaceholderSegment):
                if seg.placeholder in seen_placeholders:
                    raise PromptRenderError(
                        f"duplicate placeholder name {seg.placeholder!r} in "
                        f"chat_template of ({prompt.name!r}, {prompt.label!r})",
                        name=prompt.name,
                        version=prompt.version,
                        label=prompt.label,
                        variables=variables,
                        description=f"duplicate placeholder {seg.placeholder!r}",
                    )
                if not _PLACEHOLDER_NAME_RE_RENDER.match(seg.placeholder):
                    raise PromptRenderError(
                        f"invalid placeholder name {seg.placeholder!r} in "
                        f"chat_template of ({prompt.name!r}, {prompt.label!r})",
                        name=prompt.name,
                        version=prompt.version,
                        label=prompt.label,
                        variables=variables,
                        description=(f"placeholder {seg.placeholder!r} MUST match [A-Za-z_][A-Za-z0-9_]*"),
                    )
                seen_placeholders.add(seg.placeholder)

        messages: list[Message] = []
        for idx, segment in enumerate(prompt.chat_template):
            if isinstance(segment, PlaceholderSegment):
                if segment.placeholder not in placeholders:
                    raise PromptRenderError(
                        f"unfilled placeholder {segment.placeholder!r} "
                        f"in chat_template[{idx}] of ({prompt.name!r}, {prompt.label!r})",
                        name=prompt.name,
                        version=prompt.version,
                        label=prompt.label,
                        variables=variables,
                        description=f"placeholder {segment.placeholder!r} not supplied",
                    )
                injected = placeholders[segment.placeholder]
                messages.extend(injected)
                continue
            assert isinstance(segment, ContentSegment)
            try:
                message = self._render_content_segment(prompt, segment, variables, idx)
            except PromptRenderError:
                raise

            messages.append(message)

        # Spec §11 *Final rendered messages MUST be non-empty*: a
        # chat_template that yields zero rendered Messages (e.g.,
        # only placeholder segments whose lists were all empty)
        # raises prompt_render_error.
        if not messages:
            raise PromptRenderError(
                f"chat_template of ({prompt.name!r}, {prompt.label!r}) "
                f"rendered to zero messages (§11 non-empty rule)",
                name=prompt.name,
                version=prompt.version,
                label=prompt.label,
                variables=variables,
                description="rendered messages list is empty",
            )

        try:
            rendered_hash = compute_rendered_hash(messages)
        except ValidationError as exc:
            raise PromptRenderError(
                f"rendered output invalid for ({prompt.name!r}, {prompt.label!r}): {exc}",
                name=prompt.name,
                version=prompt.version,
                label=prompt.label,
                variables=variables,
                description=str(exc),
            ) from exc

        return self._build_result(prompt, variables, messages, rendered_hash)

    def _render_content_segment(
        self,
        prompt: ChatPrompt,
        segment: ContentSegment,
        variables: dict[str, Any],
        idx: int,
    ) -> Message:
        if isinstance(segment.content, str):
            try:
                # Cache by SHA-256 of the source text — process-stable
                # and content-derived, so identical segment text across
                # prompts shares one compiled jinja Template.  Cheaper
                # than re-parsing per render; predictable across runs.
                rendered_text = self._render_template_text(
                    compute_template_hash(segment.content),
                    segment.content,
                    variables,
                )
            except PromptRenderError as exc:
                raise self._wrap_render_error(prompt, variables, exc) from exc
            if not rendered_text:
                raise PromptRenderError(
                    f"empty content segment at chat_template[{idx}] of ({prompt.name!r}, {prompt.label!r})",
                    name=prompt.name,
                    version=prompt.version,
                    label=prompt.label,
                    variables=variables,
                    description=f"segment[{idx}] rendered to empty text",
                )
            return _build_message_from_text(segment.role, rendered_text)

        # content-blocks template — render-time role-block compat
        # check (spec §11): image blocks are user-only.
        if segment.role != "user":
            for block_idx, block_tmpl in enumerate(segment.content):
                if isinstance(block_tmpl, (ImageURLBlockTemplate, ImageInlineBlockTemplate)):
                    raise PromptRenderError(
                        f"image block at chat_template[{idx}].content[{block_idx}] "
                        f"of ({prompt.name!r}, {prompt.label!r}) requires "
                        f"role=user; got role={segment.role!r}",
                        name=prompt.name,
                        version=prompt.version,
                        label=prompt.label,
                        variables=variables,
                        description=(f"image blocks are user-only; segment[{idx}] has role={segment.role!r}"),
                    )
        blocks: list[ContentBlock] = []
        for block_idx, block_tmpl in enumerate(segment.content):
            try:
                rendered_block = self._render_content_block(
                    block_tmpl, variables, key=f"chat[{idx}].block[{block_idx}]"
                )
            except PromptRenderError as exc:
                raise self._wrap_render_error(prompt, variables, exc) from exc
            if rendered_block is None:
                raise PromptRenderError(
                    f"empty text block at chat_template[{idx}].content[{block_idx}] of "
                    f"({prompt.name!r}, {prompt.label!r})",
                    name=prompt.name,
                    version=prompt.version,
                    label=prompt.label,
                    variables=variables,
                    description=f"block[{block_idx}] rendered to empty text",
                )
            blocks.append(rendered_block)
        if not blocks:
            raise PromptRenderError(
                f"empty block list at chat_template[{idx}] of ({prompt.name!r}, {prompt.label!r})",
                name=prompt.name,
                version=prompt.version,
                label=prompt.label,
                variables=variables,
                description=f"segment[{idx}] block list is empty",
            )
        # Role-block compatibility was enforced at ContentSegment
        # construction time; image blocks are user-only.  Build the
        # Message directly per role.
        try:
            return _build_message_from_blocks(segment.role, blocks)
        except ValidationError as exc:
            raise PromptRenderError(
                f"rendered content-blocks segment invalid at "
                f"chat_template[{idx}] of ({prompt.name!r}, {prompt.label!r}): {exc}",
                name=prompt.name,
                version=prompt.version,
                label=prompt.label,
                variables=variables,
                description=str(exc),
            ) from exc

    def _render_content_block(
        self,
        block: TextBlockTemplate | ImageURLBlockTemplate | ImageInlineBlockTemplate,
        variables: dict[str, Any],
        *,
        key: str,
    ) -> ContentBlock | None:
        """Render a single content-block template.  Returns None when
        a text block renders to the empty string (caller surfaces
        §11 empty-text-block error)."""
        if isinstance(block, TextBlockTemplate):
            rendered = self._render_template_text(compute_template_hash(block.text), block.text, variables)
            if not rendered:
                return None
            return TextBlock(text=rendered)
        if isinstance(block, ImageURLBlockTemplate):
            rendered_url = self._render_template_text(compute_template_hash(block.url), block.url, variables)
            return ImageBlock(
                source=ImageSourceURL(url=rendered_url),
                detail=block.detail,
            )
        # ImageInlineBlockTemplate
        rendered_data = self._render_template_text(
            compute_template_hash(block.base64_data), block.base64_data, variables
        )
        rendered_media_type = self._render_template_text(
            compute_template_hash(block.media_type), block.media_type, variables
        )
        # Security: validate that the rendered base64 is decodable.
        # Catches mid-template-substitution mishaps (e.g., a variable
        # value that breaks the encoding) at render time rather than
        # at the LLM provider boundary, where the error would be
        # provider-specific and harder to attribute.
        try:
            base64.b64decode(rendered_data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise PromptRenderError(
                f"inline image block at {key!r} produced invalid base64 after substitution: {exc}",
                name="",
                version="",
                label="",
                variables=variables,
                description=f"inline image base64 invalid: {exc}",
            ) from exc
        return ImageBlock(
            source=ImageSourceInline(base64_data=rendered_data),
            media_type=rendered_media_type,
            detail=block.detail,
        )

    def _render_template_text(self, cache_key: str, source: str, variables: dict[str, Any]) -> str:
        """Render ``source`` through the per-manager jinja env.  Raises
        ``PromptRenderError`` with a thin description on undefined
        variables / template errors; callers re-wrap with prompt
        identity for the final exception."""
        try:
            template = self._template_cache.get(cache_key)
            if template is None:
                template = self._render_env.from_string(source)
                self._template_cache[cache_key] = template
            return template.render(**variables)
        except jinja2.UndefinedError as exc:
            raise PromptRenderError(
                f"undefined variable: {exc}",
                name="",
                version="",
                label="",
                variables=variables,
                description=str(exc),
            ) from exc
        except jinja2.TemplateError as exc:
            raise PromptRenderError(
                f"template error: {exc}",
                name="",
                version="",
                label="",
                variables=variables,
                description=str(exc),
            ) from exc

    def _wrap_render_error(
        self,
        prompt: TextPrompt | ChatPrompt,
        variables: dict[str, Any],
        thin: PromptRenderError,
    ) -> PromptRenderError:
        return PromptRenderError(
            f"render failure for ({prompt.name!r}, {prompt.label!r}): {thin.description or thin}",
            name=prompt.name,
            version=prompt.version,
            label=prompt.label,
            variables=variables,
            description=thin.description or str(thin),
        )

    def _build_result(
        self,
        prompt: TextPrompt | ChatPrompt,
        variables: dict[str, Any],
        messages: list[Message],
        rendered_hash: str,
    ) -> PromptResult:
        return PromptResult(
            name=prompt.name,
            version=prompt.version,
            label=prompt.label,
            template_hash=prompt.template_hash,
            rendered_hash=rendered_hash,
            messages=messages,
            variables=variables,
            fetched_at=prompt.fetched_at,
            rendered_at=datetime.now(UTC),
            # Defensive copy of the two mutable propagated fields.
            sampling=prompt.sampling.model_copy() if prompt.sampling is not None else None,
            observability_entities=(
                dict(prompt.observability_entities) if prompt.observability_entities is not None else None
            ),
        )

    async def get(
        self,
        name: str,
        label: str | None = None,
        variables: Mapping[str, Any] | None = None,
        *,
        placeholders: Mapping[str, Sequence[Message]] | None = None,
    ) -> PromptResult:
        """Convenience equivalent to ``render(await fetch(name, label), variables)``.

        ``label`` follows the same three-step resolution as :meth:`fetch`.
        ``placeholders`` is forwarded to :meth:`render`.
        """
        prompt = await self.fetch(name, label)
        return self.render(prompt, variables, placeholders=placeholders)


def _build_message_from_text(role: str, text: str) -> Message:
    if role == "system":
        return SystemMessage(content=text)
    if role == "assistant":
        return AssistantMessage(content=text)
    return UserMessage(content=text)


def _build_message_from_blocks(role: str, blocks: list[ContentBlock]) -> Message:
    # Image blocks are user-only per llm-provider §3.1.2.  Non-user
    # roles flatten their text-block list into a single text content
    # string; an image block in a non-user role would be a violation
    # of both construction-time validation AND render-time role-block
    # compatibility — caught by ``_render_content_segment`` before
    # this helper is reached.  We assert here as a defense-in-depth
    # invariant so harness-built prompts that bypass construction
    # validators surface the violation instead of silently dropping
    # the image's data.
    if role in ("system", "assistant"):
        for block in blocks:
            assert isinstance(block, TextBlock), (
                f"content-blocks segment with role={role!r} MUST carry "
                f"text blocks only; got {type(block).__name__}"
            )
        text = "\n".join(b.text for b in blocks if isinstance(b, TextBlock))
        if role == "system":
            return SystemMessage(content=text)
        return AssistantMessage(content=text)
    return UserMessage(content=blocks)
