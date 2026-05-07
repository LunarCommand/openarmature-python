"""openarmature.llm — llm-provider capability per spec proposal 0006.

Public surface: typed ``Message`` / ``Tool`` / ``Response``, the
``Provider`` Protocol, the canonical error categories, and an
OpenAI-compatible provider. Users write::

    from openarmature.llm import (
        AssistantMessage,
        OpenAIProvider,
        Provider,
        SystemMessage,
        Tool,
        ToolCall,
        UserMessage,
    )

All seven §7 error categories and the canonical ``TRANSIENT_CATEGORIES``
frozenset are also re-exported here so callers writing custom retry
classifiers don't have to reach into ``openarmature.llm.errors``.
"""

from .errors import (
    PROVIDER_AUTHENTICATION,
    PROVIDER_INVALID_MODEL,
    PROVIDER_INVALID_REQUEST,
    PROVIDER_INVALID_RESPONSE,
    PROVIDER_MODEL_NOT_LOADED,
    PROVIDER_RATE_LIMIT,
    PROVIDER_UNAVAILABLE,
    TRANSIENT_CATEGORIES,
    LlmProviderError,
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderModelNotLoaded,
    ProviderRateLimit,
    ProviderUnavailable,
)
from .messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from .openai import OpenAIProvider
from .provider import Provider, validate_message_list, validate_tools
from .response import FinishReason, Response, RuntimeConfig, Usage

__all__ = [
    "PROVIDER_AUTHENTICATION",
    "PROVIDER_INVALID_MODEL",
    "PROVIDER_INVALID_REQUEST",
    "PROVIDER_INVALID_RESPONSE",
    "PROVIDER_MODEL_NOT_LOADED",
    "PROVIDER_RATE_LIMIT",
    "PROVIDER_UNAVAILABLE",
    "TRANSIENT_CATEGORIES",
    "AssistantMessage",
    "FinishReason",
    "LlmProviderError",
    "Message",
    "OpenAIProvider",
    "Provider",
    "ProviderAuthentication",
    "ProviderInvalidModel",
    "ProviderInvalidRequest",
    "ProviderInvalidResponse",
    "ProviderModelNotLoaded",
    "ProviderRateLimit",
    "ProviderUnavailable",
    "Response",
    "RuntimeConfig",
    "SystemMessage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "Usage",
    "UserMessage",
    "validate_message_list",
    "validate_tools",
]
