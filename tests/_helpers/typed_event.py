"""Shared test helpers for constructing typed LLM event instances.

Replaces 20+ kwargs of boilerplate at each call site with a one-liner
plus the overrides relevant to the test. Used by unit tests against
the OTel and Langfuse observers; conformance-harness unit tests have
their own variant with conformance-specific defaults.
"""

from __future__ import annotations

from typing import Any

from openarmature.graph.events import LlmCompletionEvent, LlmFailedEvent


def make_typed_event(**overrides: Any) -> LlmCompletionEvent:
    """Build a ``LlmCompletionEvent`` with neutral defaults; ``overrides``
    swap individual fields for the test case."""
    base: dict[str, Any] = {
        "invocation_id": "inv-1",
        "correlation_id": None,
        "node_name": "ask",
        "namespace": ("ask",),
        "attempt_index": 0,
        "fan_out_index": None,
        "branch_name": None,
        "provider": "openai",
        "model": "test-m",
        "response_id": None,
        "response_model": None,
        "usage": None,
        "latency_ms": 10.0,
        "finish_reason": "stop",
        "input_messages": [],
        "output_content": None,
        "request_params": {},
        "request_extras": {},
        "active_prompt": None,
        "active_prompt_group": None,
        "call_id": "cc-1",
        "caller_invocation_metadata": None,
    }
    base.update(overrides)
    return LlmCompletionEvent(**base)


def make_failed_event(**overrides: Any) -> LlmFailedEvent:
    """Build a ``LlmFailedEvent`` with neutral defaults; ``overrides``
    swap individual fields for the test case. Mirrors ``make_typed_event``
    on the shared field set; failure-specific defaults are
    ``provider_unavailable`` category, the upstream class name as
    ``error_type``, and a generic message."""
    base: dict[str, Any] = {
        "invocation_id": "inv-1",
        "correlation_id": None,
        "node_name": "ask",
        "namespace": ("ask",),
        "attempt_index": 0,
        "fan_out_index": None,
        "branch_name": None,
        "provider": "openai",
        "model": "test-m",
        "latency_ms": 10.0,
        "input_messages": [],
        "request_params": {},
        "request_extras": {},
        "active_prompt": None,
        "active_prompt_group": None,
        "call_id": "cc-1",
        "error_category": "provider_unavailable",
        "error_type": "ProviderUnavailable",
        "error_message": "service down",
        "caller_invocation_metadata": None,
    }
    base.update(overrides)
    return LlmFailedEvent(**base)
