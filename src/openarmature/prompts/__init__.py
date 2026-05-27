"""Prompt-management capability; fetch, render, and trace named prompts."""

from .backend import PromptBackend
from .backends import FilesystemPromptBackend
from .context import (
    current_prompt_group,
    current_prompt_result,
    with_active_prompt,
    with_active_prompt_group,
)
from .errors import (
    PROMPT_NOT_FOUND,
    PROMPT_RENDER_ERROR,
    PROMPT_STORE_UNAVAILABLE,
    PROMPT_TRANSIENT_CATEGORIES,
    PromptError,
    PromptNotFound,
    PromptRenderError,
    PromptStoreUnavailable,
)
from .group import PromptGroup
from .hashing import compute_rendered_hash, compute_template_hash
from .label_resolver import SPEC_FALLBACK_LABEL, LabelResolver, MappingLabelResolver
from .manager import PromptManager
from .prompt import Prompt, PromptResult, SamplingConfig

__all__ = [
    "PROMPT_NOT_FOUND",
    "PROMPT_RENDER_ERROR",
    "PROMPT_STORE_UNAVAILABLE",
    "PROMPT_TRANSIENT_CATEGORIES",
    "SPEC_FALLBACK_LABEL",
    "FilesystemPromptBackend",
    "LabelResolver",
    "MappingLabelResolver",
    "Prompt",
    "PromptBackend",
    "PromptError",
    "PromptGroup",
    "PromptManager",
    "PromptNotFound",
    "PromptRenderError",
    "PromptResult",
    "PromptStoreUnavailable",
    "SamplingConfig",
    "compute_rendered_hash",
    "compute_template_hash",
    "current_prompt_group",
    "current_prompt_result",
    "with_active_prompt",
    "with_active_prompt_group",
]
