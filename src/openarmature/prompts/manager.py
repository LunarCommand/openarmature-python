"""PromptManager; user-facing fetch + render + composite-fallback."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import jinja2
from pydantic import ValidationError

from openarmature.llm.messages import Message, UserMessage

from .backend import PromptBackend
from .errors import PromptNotFound, PromptRenderError, PromptStoreUnavailable
from .hashing import compute_rendered_hash
from .label_resolver import SPEC_FALLBACK_LABEL, LabelResolver
from .prompt import Prompt, PromptResult

_log = logging.getLogger(__name__)


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
        variables: dict[str, Any] | None = None,
    ) -> PromptResult:
        """Apply ``variables`` to ``prompt.template`` and return a PromptResult.

        Render is synchronous; no I/O. Variables are strict by
        default per §7: a template reference to a name not in
        ``variables`` raises ``PromptRenderError``.

        The render output is always a single ``UserMessage`` carrying
        the rendered text in v1. Multi-message decomposition (system
        + user split) is deferred to a follow-on; callers needing
        that today fetch the raw template and construct the messages
        list manually.
        """
        variables = variables or {}

        rendered_text: str
        try:
            template = self._template_cache.get(prompt.template_hash)
            if template is None:
                template = self._render_env.from_string(prompt.template)
                self._template_cache[prompt.template_hash] = template
            rendered_text = template.render(**variables)
        except jinja2.UndefinedError as exc:
            raise PromptRenderError(
                f"undefined variable rendering ({prompt.name!r}, {prompt.label!r}): {exc}",
                name=prompt.name,
                version=prompt.version,
                label=prompt.label,
                variables=variables,
                description=str(exc),
            ) from exc
        except jinja2.TemplateError as exc:
            raise PromptRenderError(
                f"template error rendering ({prompt.name!r}, {prompt.label!r}): {exc}",
                name=prompt.name,
                version=prompt.version,
                label=prompt.label,
                variables=variables,
                description=str(exc),
            ) from exc

        # Boundary-wrap the Pydantic-validation step around message
        # construction. A template that renders to an empty string
        # (e.g., ``{{ x if x else '' }}`` with ``x=None``) parses
        # cleanly through Jinja2 but ``UserMessage(content="")``
        # raises ValidationError per messages.py's non-empty rule.
        # That counts as a render failure under §10's "variable's
        # value is not coercible" framing.
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
            # Caching backends re-serve the same Prompt instance, so
            # aliasing would let a caller's mutation of the result
            # corrupt the backend's cached state.
            sampling=prompt.sampling.model_copy() if prompt.sampling is not None else None,
            observability_entities=(
                dict(prompt.observability_entities) if prompt.observability_entities is not None else None
            ),
        )

    async def get(
        self,
        name: str,
        label: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> PromptResult:
        """Convenience equivalent to ``render(await fetch(name, label), variables)``.

        ``label`` follows the same three-step resolution as :meth:`fetch`.
        """
        prompt = await self.fetch(name, label)
        return self.render(prompt, variables)
