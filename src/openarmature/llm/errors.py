# Spec: realizes llm-provider §7 (seven canonical error categories).

"""Errors raised by an llm-provider implementation.

A provider call (``ready()`` or ``complete()``) MAY raise one of
seven canonical category errors. Each error class carries a
``category`` class attribute matching the canonical string identifier
so callers can dispatch on the category without matching exception
types directly.

This module is also the single source of truth for the canonical
category strings — :data:`TRANSIENT_CATEGORIES` lives here, and
``openarmature.graph.middleware.retry``'s default classifier imports
it.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Canonical category strings (llm-provider spec §7)
# ---------------------------------------------------------------------------

PROVIDER_AUTHENTICATION = "provider_authentication"
PROVIDER_UNAVAILABLE = "provider_unavailable"
PROVIDER_INVALID_MODEL = "provider_invalid_model"
PROVIDER_MODEL_NOT_LOADED = "provider_model_not_loaded"
PROVIDER_RATE_LIMIT = "provider_rate_limit"
PROVIDER_INVALID_RESPONSE = "provider_invalid_response"
PROVIDER_INVALID_REQUEST = "provider_invalid_request"


# Per spec §7 "Retry classification": these three categories are
# *transient* — a retry MAY succeed. The other four
# (`provider_authentication`, `provider_invalid_model`,
# `provider_invalid_request`, `provider_invalid_response`) are
# non-transient and MUST NOT be retried by the default classifier.
#
# Note: ``finish_reason: "error"`` is also transient per spec §7, but
# that's a Response-level signal rather than an exception category, so
# it isn't part of this set — the default retry middleware operates on
# raised exceptions.
TRANSIENT_CATEGORIES: frozenset[str] = frozenset(
    {
        PROVIDER_RATE_LIMIT,
        PROVIDER_UNAVAILABLE,
        PROVIDER_MODEL_NOT_LOADED,
    }
)


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class LlmProviderError(Exception):
    """Base for all llm-provider errors. Each subclass carries a
    ``category`` class attribute matching one of the canonical
    category strings above.

    Provider-originated errors SHOULD preserve the underlying provider
    exception as ``__cause__`` so callers can reach the wire-level
    detail when needed.
    """

    category: str


class ProviderAuthentication(LlmProviderError):
    """Auth failed — invalid key, expired token, missing credentials."""

    category = PROVIDER_AUTHENTICATION


class ProviderUnavailable(LlmProviderError):
    """Provider is unreachable — network failure, 5xx error, DNS, timeout."""

    category = PROVIDER_UNAVAILABLE


class ProviderInvalidModel(LlmProviderError):
    """The bound model does not exist on this provider. Terminal —
    retry will not succeed without changing the bound model."""

    category = PROVIDER_INVALID_MODEL


class ProviderModelNotLoaded(LlmProviderError):
    """The bound model is known to the provider but is not currently
    serving (e.g., a local vLLM/LM Studio/llama.cpp server has the
    model configured but not loaded). Distinct from
    ``provider_invalid_model`` because retry MAY succeed once loading
    completes."""

    category = PROVIDER_MODEL_NOT_LOADED


class ProviderRateLimit(LlmProviderError):
    """Provider returned a rate-limit response (HTTP 429 or equivalent).

    When the provider supplies a ``Retry-After`` header (or its
    equivalent), the parsed seconds-to-wait surfaces on
    :attr:`retry_after`. ``None`` if the provider didn't include one.
    """

    category = PROVIDER_RATE_LIMIT
    retry_after: float | None

    def __init__(self, *args: Any, retry_after: float | None = None) -> None:
        super().__init__(*args)
        self.retry_after = retry_after


class ProviderInvalidResponse(LlmProviderError):
    """Provider returned a malformed response that cannot be parsed
    into the expected :class:`Response` shape (missing required
    fields, invalid tool_calls structure, invalid JSON)."""

    category = PROVIDER_INVALID_RESPONSE


class ProviderInvalidRequest(LlmProviderError):
    """The request was malformed before sending (per-role message
    constraints violated, ``tool_call_id`` does not match an earlier
    assistant tool call, duplicate tool names, etc.). Raised by the
    implementation's pre-send validation, not by the provider."""

    category = PROVIDER_INVALID_REQUEST


__all__ = [
    "PROVIDER_AUTHENTICATION",
    "PROVIDER_INVALID_MODEL",
    "PROVIDER_INVALID_REQUEST",
    "PROVIDER_INVALID_RESPONSE",
    "PROVIDER_MODEL_NOT_LOADED",
    "PROVIDER_RATE_LIMIT",
    "PROVIDER_UNAVAILABLE",
    "TRANSIENT_CATEGORIES",
    "LlmProviderError",
    "ProviderAuthentication",
    "ProviderInvalidModel",
    "ProviderInvalidRequest",
    "ProviderInvalidResponse",
    "ProviderModelNotLoaded",
    "ProviderRateLimit",
    "ProviderUnavailable",
]
