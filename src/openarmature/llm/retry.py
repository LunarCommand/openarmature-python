"""LLM-completion-scoped call-level retry configuration.

Proposal 0095 extends the generic pipeline-utilities §6.1 retry record with
call-level behaviors that are inherently completion-specific (per-attempt
sampling; message reask). Per the proposal's Resolved-at-Accept ruling, these
ride an llm-provider-scoped *superset* of the §6.1 record rather than widening
the framework-agnostic record with LLM-only fields. ``LlmRetryConfig`` is that
superset: a caller passes it to ``complete(retry=...)`` (it IS-A ``RetryConfig``,
so the existing signature accepts it), and the call-level loop reads the extra
fields when present. A plain ``RetryConfig`` preserves the current
transient-only, byte-identical-replay behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openarmature.graph.middleware.retry import RetryConfig

if TYPE_CHECKING:
    from openarmature.llm.response import RuntimeConfig


@dataclass(frozen=True)
class LlmRetryConfig(RetryConfig):
    """The §6.1 ``RetryConfig`` extended for LLM-completion call-level retry
    (proposal 0095, llm-provider §7.1).

    - ``per_attempt_override``: a retry schedule of ``RuntimeConfig``
      partial-overrides applied to RETRIES only. Attempt 0 uses the caller's
      base ``config`` unmodified; retry ``i`` (attempt ``i+1``) merges
      ``per_attempt_override[i]`` onto the base (the override's set fields
      replace, unspecified fields inherited). When the schedule is shorter
      than the retry count, the last entry carries forward. The canonical form
      is an escalating temperature schedule (e.g. ``[RuntimeConfig(temperature=0.3),
      RuntimeConfig(temperature=0.6)]``). ``complete()`` never mutates the
      caller's ``config`` -- each attempt config is a fresh copy.
    """

    # ``from __future__ import annotations`` keeps this annotation a string at
    # runtime, so RuntimeConfig stays a TYPE_CHECKING-only import (the dataclass
    # never evaluates it) -- no graph -> llm.response module-load edge.
    per_attempt_override: list[RuntimeConfig] | None = None
