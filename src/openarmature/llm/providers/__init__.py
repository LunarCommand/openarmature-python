"""Concrete provider implementations.

Each provider talks to a specific LLM backend's wire format and
satisfies the :class:`openarmature.llm.Provider` Protocol. The current
catalog ships :class:`OpenAIProvider` (OpenAI Chat Completions wire
format — also covers vLLM, LM Studio, llama.cpp). Anthropic / Bedrock /
gateway-shaped providers will land here in later phases.

Users typically import from the package root::

    from openarmature.llm import OpenAIProvider

This subpackage exists to keep the provider catalog grouped and to
mirror the ``openarmature.graph.middleware`` layout.
"""

from .openai import OpenAIProvider

__all__ = ["OpenAIProvider"]
