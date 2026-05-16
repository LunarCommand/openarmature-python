"""Context variables for propagating prompt identity to observability.

Spec §11 leaves the propagation mechanism implementation-defined.
This module provides the Python implementation: two ``ContextVar``s
plus two context managers (``with_active_prompt`` and
``with_active_prompt_group``) that observers read to surface the
normative ``openarmature.prompt.*`` and
``openarmature.prompt.group_name`` span attributes.

Nesting policy: innermost-wins. When two ``with_active_prompt``
contexts nest, the inner result is the active one for the
duration of the inner block; the same applies to
``with_active_prompt_group``. This matches Python's natural
``ContextVar`` token-stacking behavior; spec §11 doesn't mandate
a nesting policy.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .group import PromptGroup
from .prompt import PromptResult

_active_prompt: ContextVar[PromptResult | None] = ContextVar(
    "openarmature_active_prompt",
    default=None,
)
_active_prompt_group: ContextVar[PromptGroup | None] = ContextVar(
    "openarmature_active_prompt_group",
    default=None,
)


@contextmanager
def with_active_prompt(result: PromptResult) -> Iterator[None]:
    """Mark ``result`` as the active prompt for downstream LLM calls.

    When the observability extra is installed and an LLM call fires
    inside this context, the OTel observer surfaces
    ``openarmature.prompt.name`` / ``version`` / ``label`` /
    ``template_hash`` / ``rendered_hash`` on the LLM-call span.

    Nesting is innermost-wins.
    """
    token = _active_prompt.set(result)
    try:
        yield
    finally:
        _active_prompt.reset(token)


@contextmanager
def with_active_prompt_group(group: PromptGroup) -> Iterator[None]:
    """Mark ``group`` as the active prompt group for downstream LLM calls.

    When an LLM call fires inside this context, the OTel observer
    surfaces ``openarmature.prompt.group_name`` on the LLM-call
    span, alongside any per-prompt attributes from a concurrently
    active ``with_active_prompt``.

    Nesting is innermost-wins.
    """
    token = _active_prompt_group.set(group)
    try:
        yield
    finally:
        _active_prompt_group.reset(token)


def current_prompt_result() -> PromptResult | None:
    """Return the innermost active PromptResult, or ``None``."""
    return _active_prompt.get()


def current_prompt_group() -> PromptGroup | None:
    """Return the innermost active PromptGroup, or ``None``."""
    return _active_prompt_group.get()
