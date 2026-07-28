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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openarmature.graph.middleware.retry import RetryConfig
from openarmature.llm.errors import StructuredOutputInvalid

if TYPE_CHECKING:
    from openarmature.llm.response import RuntimeConfig


# Proposal 0095b: a caller-supplied corrective-message builder for
# structured-output reask. Given the raised StructuredOutputInvalid (the 0082
# error surface -- ``exc.raw_content`` is the model's invalid output,
# ``exc.failure_description`` the reason), it returns the correction text OA
# appends as a user message. OA authors no prompt of its own (charter §3.1
# principle 7); the caller owns every word. Sync, mirroring classifier / backoff
# -- pure string rendering. Passing the whole exception matches the classifier /
# on_retry precedent and stays forward-compatible as the 0082 surface grows.
# The builder MUST return a str: if it raises or returns a non-str, the loop
# re-raises the original structured_output_invalid (its terminal LlmFailedEvent
# still fires) rather than leaking the builder's error.
ReaskBuilder = Callable[[StructuredOutputInvalid], str]


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
    - ``reask``: an opt-in structured-output reask builder (proposal 0095b).
      When present, the call-level loop treats a ``structured_output_invalid``
      failure as retryable FOR THIS CALL (without a custom classifier), and on
      each such failure appends two messages to a working transcript -- the
      model's raw output as an ``assistant`` message, then this builder's
      returned correction as a ``user`` message -- so the retry is informed
      rather than a byte-identical replay. The transcript accumulates across
      reask retries and consumes the ``max_attempts`` budget. Absent a builder,
      ``structured_output_invalid`` is non-transient and raises on the first
      occurrence.
    """

    # ``from __future__ import annotations`` keeps this annotation a string at
    # runtime, so RuntimeConfig stays a TYPE_CHECKING-only import (the dataclass
    # never evaluates it) -- no graph -> llm.response module-load edge.
    per_attempt_override: list[RuntimeConfig] | None = None
    reask: ReaskBuilder | None = None
