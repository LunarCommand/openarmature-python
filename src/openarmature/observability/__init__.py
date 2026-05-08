"""openarmature.observability — cross-backend observability surface.

Two layers:

- **Core** (this module + ``correlation.py``): always available, no
  extra dependencies. Exposes :func:`current_correlation_id` and
  :func:`current_active_observers` — the spec observability §3
  ``ContextVar`` primitives that every backend mapping consumes.
- **Backend mappings** (under ``observability.otel`` and future
  ``observability.langfuse`` etc.): gated behind optional
  dependencies (``pip install openarmature[otel]``). Importing the
  subpackage without the extras installed raises an informative
  ``ImportError`` pointing the caller at the install command.

The split mirrors charter §3.1 principle 5: core defines the
contracts; specific backends implement them. At v1.0 launch the
backend mappings will lift into sibling packages
(``openarmature-otel``, ``openarmature-langfuse``) — until then
they live here under per-backend subpackages so the layering is
established up front.
"""

from .correlation import current_active_observers, current_correlation_id

__all__ = [
    "current_active_observers",
    "current_correlation_id",
]
