"""PromptManager — user-facing fetch + render + composite-fallback."""

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
from .prompt import Prompt, PromptResult

_log = logging.getLogger(__name__)

# Module-level singleton. Stateless given the configuration (no
# filters, globals, or per-call mutation), and jinja2.Environment is
# documented as thread-safe for compile + render — so a single
# shared instance avoids re-parsing the env config on every render
# call. autoescape disabled by design: render output goes to an LLM
# API call (plain text), not an HTML response.
_RENDER_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


class PromptManager:
    """Composes one or more PromptBackends and exposes fetch + render.

    Users interact with the manager; backends are an implementation
    detail of construction. The manager owns:

    - ``fetch`` — consults backends in order per §8 fallback semantics.
    - ``render`` — synchronous local string transform; produces a
      ``PromptResult``.
    - ``get`` — convenience: ``render(await fetch(...), variables)``.
    """

    def __init__(self, *backends: PromptBackend) -> None:
        if not backends:
            raise ValueError("PromptManager requires at least one backend")
        self._backends: tuple[PromptBackend, ...] = backends

    async def fetch(self, name: str, label: str = "production") -> Prompt:
        """Consult composed backends in order, applying §8 fallback.

        - First successful fetch wins; further backends are not consulted.
        - ``PromptNotFound`` from any backend STOPS the chain — the
          error propagates. Logical absence MUST NOT silently
          substitute a stale alternative.
        - ``PromptStoreUnavailable`` from a backend continues to the
          next. After ALL backends are exhausted with unavailable
          failures, the manager raises ``PromptStoreUnavailable``.
        """
        last_unavailable: PromptStoreUnavailable | None = None
        for backend in self._backends:
            try:
                return await backend.fetch(name, label)
            except PromptNotFound:
                raise
            except PromptStoreUnavailable as exc:
                last_unavailable = exc
                _log.warning(
                    "prompt backend %r unavailable for (%r, %r); falling back",
                    backend,
                    name,
                    label,
                )
                continue
        assert last_unavailable is not None
        raise PromptStoreUnavailable(
            f"all prompt backends unavailable for ({name!r}, {label!r})",
            name=name,
            label=label,
            backends_tried=[type(b).__name__ for b in self._backends],
        ) from last_unavailable

    def render(
        self,
        prompt: Prompt,
        variables: dict[str, Any] | None = None,
    ) -> PromptResult:
        """Apply ``variables`` to ``prompt.template`` and return a PromptResult.

        Render is synchronous — no I/O. Variables are strict by
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
            template = _RENDER_ENV.from_string(prompt.template)
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
        )

    async def get(
        self,
        name: str,
        label: str = "production",
        variables: dict[str, Any] | None = None,
    ) -> PromptResult:
        """Convenience equivalent to ``render(await fetch(name, label), variables)``."""
        prompt = await self.fetch(name, label)
        return self.render(prompt, variables)
