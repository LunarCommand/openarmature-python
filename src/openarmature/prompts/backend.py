"""PromptBackend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .prompt import Prompt


@runtime_checkable
class PromptBackend(Protocol):
    """Backend protocol — implementations and sibling packages plug into this.

    A PromptBackend exposes one operation: ``fetch`` a prompt by name
    and label. Backends do NOT render; rendering is the manager's
    concern.

    Operation semantics:

    - ``fetch()`` MUST be reentrant: multiple concurrent calls on the
      same backend are permitted.
    - ``fetch()`` does NOT render or otherwise mutate the template.
    - ``fetch()`` MUST raise ``PromptNotFound`` when no prompt matches
      ``(name, label)``.
    - ``fetch()`` MUST raise ``PromptStoreUnavailable`` when the
      backend is unreachable (network failure, filesystem I/O error,
      vendor API timeout).

    Backends MAY cache their own results internally. When a backend
    serves a cached result, the returned Prompt's ``template_hash``
    MUST still be correct for the served template (caching MUST NOT
    break content-addressing), and ``fetched_at`` MUST reflect the
    original fetch time, not the cache hit time.
    """

    async def fetch(self, name: str, label: str = "production") -> Prompt: ...
