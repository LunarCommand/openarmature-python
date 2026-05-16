"""Error categories for the prompt-management capability."""

from __future__ import annotations

from typing import Any, ClassVar

PROMPT_NOT_FOUND = "prompt_not_found"
PROMPT_RENDER_ERROR = "prompt_render_error"
PROMPT_STORE_UNAVAILABLE = "prompt_store_unavailable"

# Mirrors openarmature.llm.errors.TRANSIENT_CATEGORIES. Retry-middleware
# classifiers MAY import this to identify transient prompt-management
# failures by category.
PROMPT_TRANSIENT_CATEGORIES: frozenset[str] = frozenset({PROMPT_STORE_UNAVAILABLE})


class PromptError(Exception):
    """Base for prompt-management errors. Subclasses set ``category``
    to one of the canonical identifier strings."""

    category: ClassVar[str]


class PromptNotFound(PromptError):
    """Raised when no prompt matches ``(name, label)``.

    Non-transient: retrying the same name + label will not succeed
    without changing the backends or the prompt store contents.
    """

    category = PROMPT_NOT_FOUND

    name: str
    label: str
    backend: str | None

    def __init__(
        self,
        *args: Any,
        name: str,
        label: str,
        backend: str | None = None,
    ) -> None:
        super().__init__(*args)
        self.name = name
        self.label = label
        self.backend = backend


class PromptRenderError(PromptError):
    """Raised when render fails: undefined variable under strict
    handling, template parse error, or variable-coercion failure.

    Carries the source prompt's identity plus the variable mapping
    and a description of the render failure.
    """

    category = PROMPT_RENDER_ERROR

    # v1 policy on ``variables``: pass-through unchanged (no automatic
    # redaction). Callers wanting redaction wrap their variables
    # before passing to render. Keys MUST be preserved if a future
    # redaction policy lands; only values may be redacted.
    name: str
    version: str
    label: str
    variables: dict[str, Any]
    description: str

    def __init__(
        self,
        *args: Any,
        name: str,
        version: str,
        label: str,
        variables: dict[str, Any],
        description: str,
    ) -> None:
        super().__init__(*args)
        self.name = name
        self.version = version
        self.label = label
        self.variables = variables
        self.description = description


class PromptStoreUnavailable(PromptError):
    """Raised when backend infrastructure fails: network unreachable,
    filesystem I/O error, vendor API 5xx, vendor API timeout.

    Transient: the same fetch may succeed when the backend recovers.
    ``PromptManager.fetch`` raises this only after ALL composed
    backends raise it.
    """

    category = PROMPT_STORE_UNAVAILABLE
