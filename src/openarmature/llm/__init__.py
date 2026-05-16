# Spec: this package implements the llm-provider capability (spec
# proposal 0006).

"""openarmature.llm — LLM provider abstraction.

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

All seven error categories and the canonical ``TRANSIENT_CATEGORIES``
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
    PROVIDER_UNSUPPORTED_CONTENT_BLOCK,
    STRUCTURED_OUTPUT_INVALID,
    TRANSIENT_CATEGORIES,
    LlmProviderError,
    ProviderAuthentication,
    ProviderInvalidModel,
    ProviderInvalidRequest,
    ProviderInvalidResponse,
    ProviderModelNotLoaded,
    ProviderRateLimit,
    ProviderUnavailable,
    ProviderUnsupportedContentBlock,
    StructuredOutputInvalid,
)
from .messages import (
    AssistantMessage,
    ContentBlock,
    ImageBlock,
    ImageSource,
    ImageSourceInline,
    ImageSourceURL,
    Message,
    SystemMessage,
    TextBlock,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from .provider import (
    Provider,
    strict_mode_supported,
    validate_message_list,
    validate_response_schema,
    validate_tools,
)
from .providers import OpenAIProvider, classify_http_error, parse_retry_after
from .response import FinishReason, ParsedValue, Response, RuntimeConfig, Usage

__all__ = [
    "PROVIDER_AUTHENTICATION",
    "PROVIDER_INVALID_MODEL",
    "PROVIDER_INVALID_REQUEST",
    "PROVIDER_INVALID_RESPONSE",
    "PROVIDER_MODEL_NOT_LOADED",
    "PROVIDER_RATE_LIMIT",
    "PROVIDER_UNAVAILABLE",
    "PROVIDER_UNSUPPORTED_CONTENT_BLOCK",
    "STRUCTURED_OUTPUT_INVALID",
    "TRANSIENT_CATEGORIES",
    "AssistantMessage",
    "ContentBlock",
    "FinishReason",
    "ImageBlock",
    "ImageSource",
    "ImageSourceInline",
    "ImageSourceURL",
    "LlmProviderError",
    "Message",
    "OpenAIProvider",
    "ParsedValue",
    "Provider",
    "ProviderAuthentication",
    "ProviderInvalidModel",
    "ProviderInvalidRequest",
    "ProviderInvalidResponse",
    "ProviderModelNotLoaded",
    "ProviderRateLimit",
    "ProviderUnavailable",
    "ProviderUnsupportedContentBlock",
    "Response",
    "RuntimeConfig",
    "StructuredOutputInvalid",
    "SystemMessage",
    "TextBlock",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "Usage",
    "UserMessage",
    "classify_http_error",
    "parse_retry_after",
    "strict_mode_supported",
    "validate_message_list",
    "validate_response_schema",
    "validate_tools",
]
