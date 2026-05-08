"""OpenTelemetry backend mapping for openarmature observability.

Per charter §3.1 principle 5: the core defines observability contracts
(``correlation_id`` ContextVar, dispatch primitives); specific
backends implement them. This subpackage is extras-gated — install
with ``pip install openarmature[otel]`` to bring in
``opentelemetry-api`` and ``opentelemetry-sdk``.

Importing this subpackage without the extras installed raises an
informative :class:`ImportError` pointing the caller at the install
command. We do NOT do partial fallbacks (e.g., a stubbed observer
that silently no-ops) — the user opted in to OTel by importing
``openarmature.observability.otel``, so a clean failure on missing
deps is preferable to silent-broken behavior.

Public surface:

- :class:`OTelObserver` — the observer-driven span lifecycle implementation
  per spec observability §6 RECOMMENDED path.
- :func:`install_log_bridge` — helper to wire the OTel Logs SDK to
  the stdlib ``logging`` root with ``correlation_id`` injection.

Plan to lift into a sibling ``openarmature-otel`` package at the v1.0
launch alongside ``openarmature-eval``; until then this subpackage
holds the implementation in-tree.
"""

from __future__ import annotations

try:
    from .logs import install_log_bridge
    from .observer import OTelObserver
except ImportError as exc:  # pragma: no cover - exercised by extras-not-installed path
    if "opentelemetry" in str(exc):
        raise ImportError(
            "openarmature.observability.otel requires the optional `otel` extras. "
            "Install with: pip install 'openarmature[otel]'"
        ) from exc
    raise

__all__ = [
    "OTelObserver",
    "install_log_bridge",
]
