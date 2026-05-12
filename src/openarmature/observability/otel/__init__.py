# Spec: this subpackage is the OTel backend mapping. Realizes the
# observer-driven span lifecycle from observability spec §6 (RECOMMENDED
# path) and the logs-correlation glue from §7. Layering follows charter
# §3.1 principle 5 (core defines contracts; backends implement them).
# Plan: lift into a sibling ``openarmature-otel`` package at v1.0 launch
# alongside ``openarmature-eval``.

"""OpenTelemetry backend mapping for openarmature observability.

This subpackage is extras-gated — install with
``pip install openarmature[otel]`` to bring in ``opentelemetry-api``
and ``opentelemetry-sdk``.

Importing this subpackage without the extras installed raises an
informative :class:`ImportError` pointing the caller at the install
command. We do NOT do partial fallbacks (e.g., a stubbed observer
that silently no-ops) — the user opted in to OTel by importing
``openarmature.observability.otel``, so a clean failure on missing
deps is preferable to silent-broken behavior.

Public surface:

- :class:`OTelObserver` — observer-driven span lifecycle.
- :func:`install_log_bridge` — helper to wire the OTel Logs SDK to
  the stdlib ``logging`` root with ``correlation_id`` injection.
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
