"""The private Langfuse SDK surface openarmature's shipped adapter depends on."""

# openarmature declares `langfuse>=4.6,<5`, and inside that range its shipped
# adapter reaches for private SDK symbols: the client class itself, the
# `_resources` / `_tracing_enabled` pair the isolation verdict rests on, the
# per-credential cache behind the already-cached warning, the `_otel_tracer` and
# `_create_remote_parent_span` internals behind every back-dated observation, and
# the four `langfuse._client.span` classes it constructs. A minor upgrade inside
# our own declared range could break any of them.
#
# Until now the only guard was the live-account integration test, which CI
# deselects (`addopts = ["-m", "not integration"]`), so in practice nothing
# checked them on a normal run. These tests are the CI-visible half: they do not
# need credentials or a network, and they fail loudly on a rename rather than
# letting a client quietly classify `undetectable` or an observation quietly stop
# being emitted.
#
# openarmature.org/compatibility tracks the SDK version last verified and already
# records the bound-provider read as resting on non-portable internals. This file
# is the executable counterpart to that row.

from __future__ import annotations

import inspect

import pytest

langfuse = pytest.importorskip("langfuse", reason="the langfuse extra is optional")


def test_the_client_class_is_where_the_adapter_binds_it() -> None:
    # The adapter binds `Langfuse` at module import, which is why a harness must
    # patch the ADAPTER's name to substitute a client. Anything that moves this
    # breaks that substitution silently.
    from openarmature.observability.langfuse import adapter

    assert adapter.Langfuse is langfuse.Langfuse


def test_the_bound_provider_is_readable_through_resources() -> None:
    # adapter._classify_isolation walks client._resources.tracer_provider. If it
    # goes missing every client classifies `undetectable`, payload silently
    # switches off behind one WARNING, and conformance.toml keeps publishing
    # langfuse_bound_provider_detection = true with nothing going red.
    from langfuse._client.resource_manager import LangfuseResourceManager

    assert "self._resources" in inspect.getsource(langfuse.Langfuse)
    assert "self.tracer_provider" in inspect.getsource(LangfuseResourceManager)


def test_the_tracing_enabled_flag_classify_isolation_branches_on_exists() -> None:
    # _classify_isolation reads client._tracing_enabled FIRST and short-circuits to
    # `isolated` when it is false. If it disappears, getattr's default of True keeps
    # the code running while the short-circuit silently stops applying.
    assert "self._tracing_enabled" in inspect.getsource(langfuse.Langfuse)


def test_the_per_credential_cache_is_where_the_adapter_looks() -> None:
    # adapter._public_key_is_cached reads this to warn that a second construction
    # for a credential silently discards its options. A rename would not fail; the
    # warning would simply stop firing.
    from langfuse._client.resource_manager import LangfuseResourceManager

    assert isinstance(LangfuseResourceManager._instances, dict)


@pytest.mark.parametrize(
    ("symbol", "spelling"),
    [("_otel_tracer", "self._otel_tracer"), ("_create_remote_parent_span", "def _create_remote_parent_span")],
)
def test_the_back_dated_observation_internals_exist(symbol: str, spelling: str) -> None:
    # Every provider observation carrying a start_time goes through
    # _start_back_dated_observation, which needs both. Losing either does not
    # raise to the caller: the graph observer isolates observer errors, so the
    # observation simply stops being emitted and a leak assertion reads clean.
    #
    # Checked in the source rather than with hasattr, because `_otel_tracer` is an
    # INSTANCE attribute and a class-level hasattr would report it missing on a
    # perfectly good SDK.
    assert spelling in inspect.getsource(langfuse.Langfuse), (
        f"langfuse.Langfuse.{symbol} is gone; adapter._start_back_dated_observation depends "
        f"on it, and its absence surfaces only as a swallowed observer warning"
    )


@pytest.mark.parametrize(
    "name", ["LangfuseGeneration", "LangfuseTool", "LangfuseEmbedding", "LangfuseRetriever"]
)
def test_the_span_classes_the_adapter_constructs_exist(name: str) -> None:
    span_module = pytest.importorskip("langfuse._client.span")
    assert hasattr(span_module, name)


def test_the_installed_version_is_within_the_declared_range() -> None:
    # Non-vacuity for everything above: the checks are only meaningful against a
    # version we claim to support. This also surfaces drift between what is
    # installed and what openarmature.org/compatibility records as verified.
    import re
    from importlib.metadata import version

    installed = version("langfuse")
    # Regex rather than int() on the split parts: a PEP 440 two-component
    # pre-release such as 4.7rc1 attaches its suffix to the MINOR, so splitting
    # raises ValueError on a version that is inside our declared range. A bare
    # ValueError from a guard test also reads identically to a genuine range
    # violation, which is the more expensive confusion.
    match = re.match(r"^(\d+)\.(\d+)", installed)
    assert match is not None, (
        f"cannot parse a major.minor out of langfuse version {installed!r}; this guard cannot "
        f"confirm the installed SDK is inside the declared range"
    )
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (4, 6), f"langfuse {installed} is below the declared floor of 4.6"
    assert major < 5, f"langfuse {installed} is outside the declared range (<5)"
