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
    PROMPT_GROUP_INVALID,
    PROMPT_NOT_FOUND,
    PROMPT_RENDER_ERROR,
    PROMPT_STORE_UNAVAILABLE,
    PROMPT_TRANSIENT_CATEGORIES,
    PromptError,
    PromptGroupInvalid,
    PromptNotFound,
    PromptRenderError,
    PromptStoreUnavailable,
)
from .group import PromptGroup
from .hashing import compute_rendered_hash, compute_template_hash
from .label_resolver import SPEC_FALLBACK_LABEL, LabelResolver, MappingLabelResolver
from .manager import PromptManager
from .prompt import (
    ChatPrompt,
    ChatSegment,
    ContentBlockTemplate,
    ContentSegment,
    ImageInlineBlockTemplate,
    ImageURLBlockTemplate,
    PlaceholderSegment,
    Prompt,
    PromptResult,
    SamplingConfig,
    TextBlockTemplate,
    TextPrompt,
    TokenBudget,
)

__all__ = [
    "PROMPT_GROUP_INVALID",
    "PROMPT_NOT_FOUND",
    "PROMPT_RENDER_ERROR",
    "PROMPT_STORE_UNAVAILABLE",
    "PROMPT_TRANSIENT_CATEGORIES",
    "SPEC_FALLBACK_LABEL",
    "ChatPrompt",
    "ChatSegment",
    "ContentBlockTemplate",
    "ContentSegment",
    "FilesystemPromptBackend",
    "ImageInlineBlockTemplate",
    "ImageURLBlockTemplate",
    "LabelResolver",
    "MappingLabelResolver",
    "PlaceholderSegment",
    "Prompt",
    "PromptBackend",
    "PromptError",
    "PromptGroup",
    "PromptGroupInvalid",
    "PromptManager",
    "PromptNotFound",
    "PromptRenderError",
    "PromptResult",
    "PromptStoreUnavailable",
    "SamplingConfig",
    "TextBlockTemplate",
    "TextPrompt",
    "TokenBudget",
    "compute_rendered_hash",
    "compute_template_hash",
    "current_prompt_group",
    "current_prompt_result",
    "with_active_prompt",
    "with_active_prompt_group",
]
