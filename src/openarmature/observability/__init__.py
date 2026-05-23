# Spec: this package implements the observability capability —
# core ContextVar primitives from observability §3; backend mappings
# (OTel here) realize §4-§7. Split mirrors charter §3.1 principle 5
# (core defines contracts; specific backends implement them).

"""openarmature.observability: cross-backend observability surface.

Two layers:

- **Core** (this module + ``correlation.py``): always available, no
  extra dependencies. Exposes :func:`current_correlation_id` and
  :func:`current_active_observers`; the ``ContextVar`` primitives
  that every backend mapping consumes.
- **Backend mappings** (under ``observability.otel`` and future
  ``observability.langfuse`` etc.): gated behind optional
  dependencies (``pip install openarmature[otel]``). Importing the
  subpackage without the extras installed raises an informative
  ``ImportError`` pointing the caller at the install command.

At v1.0 launch the backend mappings will lift into sibling packages
(``openarmature-otel``, ``openarmature-langfuse``); until then they
live here under per-backend subpackages so the layering is
established up front.
"""

from .correlation import (
    current_active_observers,
    current_attempt_index,
    current_correlation_id,
    current_dispatch,
    current_fan_out_index,
    current_invocation_id,
    current_namespace_prefix,
)

# v0.17.0 (proposal 0024 / friction-roundup #9): publish the LLM event
# contract so third-party Provider implementations and custom observers
# can interoperate against a stable shape. The namespace constant lives
# on the otel observer module (canonical site for the sentinel value);
# LlmEventPayload lives in its own module under observability so the
# dependency direction stays one-way (providers depend on observability
# primitives, not the reverse).
from .llm_event import LlmEventPayload
from .otel.observer import LLM_NAMESPACE  # noqa: E402

__all__ = [
    "LLM_NAMESPACE",
    "LlmEventPayload",
    "current_active_observers",
    "current_attempt_index",
    "current_correlation_id",
    "current_dispatch",
    "current_fan_out_index",
    "current_invocation_id",
    "current_namespace_prefix",
]
