"""A real Langfuse client for the isolation fixtures, with its egress removed."""

# Spec basis: conformance-adapter §6.4 (proposals 0115 / 0116). §6.4 calls for a
# provider-faithful double, and for `mode: supplied` that is the protocol-shaped
# fake in `langfuse_provider_fake.py`. For `mode: credentials` the double is the
# REAL SDK client, because openarmature constructs it itself and wraps it in
# LangfuseSDKAdapter, which drives the SDK's private surface: start_observation,
# _otel_tracer, _create_remote_parent_span, _get_otel_trace_id, and the real
# langfuse._client.span classes for back-dated observations. Faking that surface
# is chasing a private API, and it was tried: each fake internal satisfied
# uncovered another, and provider observations would not have landed in a recorded
# side regardless, because the adapter builds real SDK span objects for them.
#
# So everything stays real except the network. The SDK attaches its own
# LangfuseSpanProcessor -- a batching OTLP exporter -- to whatever TracerProvider
# it is handed, which in a conformance run means retries against an unreachable
# API and background threads. Replacing that one class leaves the client, the
# adapter path, the provider binding, and the per-credential singleton genuinely
# real, and the fixture's own exporter still receives every span because it is a
# separate processor on the same provider.
#
# This is strictly more faithful than a fake, and it is the reason the fixtures
# mean anything: the private-API coupling they exercise is coupling openarmature
# already ships (see `tests/unit/test_langfuse_sdk_internals.py`, which enumerates
# it), not coupling the harness invented.

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from opentelemetry.sdk.trace import SpanProcessor

# Credentials the harness constructs with. Any non-blank pair works: nothing
# authenticates, and `from_credentials` rejects blanks at the boundary precisely
# so a blank cannot fall through to the SDK's ambient LANGFUSE_* environment
# credentials.
CONFORMANCE_PUBLIC_KEY = "pk-lf-conformance"
CONFORMANCE_SECRET_KEY = "sk-lf-conformance"

# Nothing listens here. Belt and braces alongside the processor swap below: if a
# future SDK grows a second egress path, it fails fast against a closed port
# rather than reaching a real project.

CONFORMANCE_HOST = "http://127.0.0.1:9"


class _NoEgressSpanProcessor(SpanProcessor):
    """Stands in for LangfuseSpanProcessor, dropping everything."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


@contextlib.contextmanager
def langfuse_sdk_without_egress() -> Iterator[None]:
    """Run with the SDK's exporting span processor replaced, and its cache clean.

    The per-credential client cache is process-global and outlives a case, so it
    is cleared on both sides: a stale entry would hand the next case a client
    bound to the previous case's provider and silently decide its isolation
    verdict.
    """
    import unittest.mock

    from langfuse._client import resource_manager

    # Reaching into a private module, deliberately and in one place. The shipped
    # adapter already depends on this SDK's internals (enumerated and guarded in
    # tests/unit/test_langfuse_sdk_internals.py); the harness adds no new coupling
    # beyond swapping the one class that would otherwise talk to the network.
    prior_processor = resource_manager.LangfuseSpanProcessor  # pyright: ignore[reportPrivateImportUsage]
    prior_instances = dict(resource_manager.LangfuseResourceManager._instances)
    resource_manager.LangfuseSpanProcessor = _NoEgressSpanProcessor  # type: ignore[assignment, misc]
    resource_manager.LangfuseResourceManager._instances.clear()
    # The SDK resolves its base URL from LANGFUSE_BASE_URL / LANGFUSE_HOST BEFORE
    # falling back to the host argument, so an ambient variable outranks the
    # unreachable host below and would point the API, score-ingestion and media
    # clients at a real project. We keep those variables set for live validation,
    # so this is a plausible local state rather than a hypothetical.
    cleared = {k: v for k, v in os.environ.items() if k.startswith("LANGFUSE_")}
    try:
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            for key in cleared:
                os.environ.pop(key, None)
            yield
    finally:
        os.environ.update(cleared)
        # Shut every client this block created before dropping it. Each resource
        # manager starts consumer threads, holds an httpx pool and registers an
        # atexit handler; clearing the cache alone orphans all of that.
        for manager in list(resource_manager.LangfuseResourceManager._instances.values()):
            with contextlib.suppress(Exception):
                manager.shutdown()
        resource_manager.LangfuseResourceManager._instances.clear()
        resource_manager.LangfuseResourceManager._instances.update(prior_instances)
        resource_manager.LangfuseSpanProcessor = prior_processor  # type: ignore[assignment, misc]


def prime_credential_on(tracer_provider: Any) -> Any:
    """Construct a client for the conformance credential first, as an
    application would, so openarmature is not the first constructor for it.

    This is `preexisting_same_key_client`. Because the SDK's own per-credential
    cache is in play, the discard it models is the real one: openarmature's later
    construction is handed THIS client on THIS provider and the isolated provider
    it asked for is dropped. That is the mechanism the whole isolation contract
    exists to detect, and no fake could have reproduced it faithfully.
    """
    from langfuse import Langfuse

    return Langfuse(
        public_key=CONFORMANCE_PUBLIC_KEY,
        secret_key=CONFORMANCE_SECRET_KEY,
        host=CONFORMANCE_HOST,
        tracer_provider=tracer_provider,
    )


__all__ = [
    "CONFORMANCE_HOST",
    "CONFORMANCE_PUBLIC_KEY",
    "CONFORMANCE_SECRET_KEY",
    "langfuse_sdk_without_egress",
    "prime_credential_on",
]
