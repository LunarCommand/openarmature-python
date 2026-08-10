# Categorized errors for the Langfuse observability mapping. Mirrors the
# llm/errors.py pattern: a ``category`` class attribute carrying the canonical
# string so callers dispatch on the category rather than matching the message.
#
# Spec basis: observability §6 payload-leak invariant (proposal 0116). Raised on
# the raise arm -- OA establishes that its payload-bearing Langfuse observations
# would reach a TracerProvider shared with the application, and the caller has
# not accepted a shared provider.

"""Categorized errors for the Langfuse observability mapping."""

from __future__ import annotations


class ObservabilityError(Exception):
    """Base for errors raised by the bundled observability backends.

    Distinct from the graph-engine, llm-provider, and prompt-management
    hierarchies: these are raised while wiring or driving an observability
    backend, not while running a graph or calling a provider.
    """


class LangfuseProviderIsolationUnavailable(ObservabilityError):
    """OA cannot keep its payload-bearing Langfuse observations off a
    TracerProvider shared with the application, and the caller has not accepted
    a shared provider, so construction fails loud rather than leaking.

    Carries the canonical ``category`` string so callers can dispatch on it
    without matching the message.
    """

    category = "langfuse_provider_isolation_unavailable"
