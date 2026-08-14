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

import importlib
import inspect
import re
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

_CONFORMANCE_TOML = Path(__file__).resolve().parents[2] / "conformance.toml"

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


def _declared() -> dict[str, Any]:
    # Read at collection time to parametrize, so a missing section surfaces as a
    # named failure rather than a bare KeyError from inside pytest's collector.
    with _CONFORMANCE_TOML.open("rb") as handle:
        manifest = tomllib.load(handle)
    entry = cast("dict[str, Any]", manifest.get("external_dependencies", {})).get("langfuse")
    assert entry is not None, (
        "conformance.toml has no [external_dependencies.langfuse] section. It is the published "
        "record of the private SDK surface this implementation depends on, and the source these "
        "guards parametrize over; without it nothing checks that surface."
    )
    missing = sorted({"requires", "verified", "verified_on", "internals"} - set(entry))
    assert not missing, f"[external_dependencies.langfuse] is missing required keys: {missing}"
    return cast("dict[str, Any]", entry)


def _resolve(path: str) -> tuple[Any, str]:
    """Split a dotted path into the deepest importable owner and the final name."""
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            owner: Any = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:-1]:
            # Asserted rather than left to getattr: a renamed intermediate (a
            # class, say) would otherwise surface as a bare AttributeError, and
            # the message IS this guard's product. Losing one of these paths shows
            # up nowhere else, because the graph observer swallows observer errors.
            assert hasattr(owner, attr), (
                f"{path}: {attr!r} is missing from {owner!r}, so the rest of the path cannot be "
                f"resolved. openarmature's shipped adapter depends on this path."
            )
            owner = getattr(owner, attr)
        return owner, parts[-1]
    raise AssertionError(
        f"no importable module in {path!r}; the declared internal names a module that no "
        f"longer exists in the installed SDK"
    )


@pytest.mark.parametrize("path", _declared()["internals"])
def test_each_declared_internal_still_exists(path: str) -> None:
    # Parametrized over conformance.toml's `internals` list rather than a copy of
    # it, so the PUBLISHED surface and the ENFORCED surface are the same list.
    # Restating it here would let the public record drift from what is checked,
    # which is the failure this whole guard exists to prevent.
    #
    # Losing any of these does not raise to the caller: the graph observer
    # isolates observer errors, so an observation simply stops being emitted and
    # a leak assertion reads clean.
    owner, name = _resolve(path)
    if hasattr(owner, name):
        return
    # Instance attributes (`self._resources`, `self._otel_tracer`) are not on the
    # class, so a bare hasattr would report a perfectly good SDK as broken.
    source = inspect.getsource(owner)
    assert f"self.{name}" in source, (
        f"{path} is gone from the installed langfuse SDK. openarmature's shipped adapter "
        f"depends on it, and its absence surfaces only as a swallowed observer warning."
    )


def test_the_declared_internals_cover_what_the_adapter_imports() -> None:
    # The completeness half: the per-path checks above verify what IS declared and
    # can say nothing about what the adapter depends on but nobody declared.
    #
    # Parsed from the actual import statements rather than matched against names
    # written out here. A hardcoded list is not a completeness check: an earlier
    # version enumerated four span classes, so a fifth added later would have gone
    # unnoticed by the very test meant to catch that. A substring search over the
    # module source would also match a name in prose, and a suffix match would
    # accept the same name exported by a different module.
    adapter_source = inspect.getsource(importlib.import_module("openarmature.observability.langfuse.adapter"))
    imported = set(re.findall(r"from\s+langfuse\._client\.span\s+import\s+(\w+)", adapter_source))
    assert imported, (
        "found no `from langfuse._client.span import ...` in the adapter. Either it stopped "
        "importing private span classes, in which case the declared internals should shrink, "
        "or this pattern no longer matches and the check is reading nothing."
    )
    declared = set(_declared()["internals"])
    missing = sorted(
        f"langfuse._client.span.{name}"
        for name in imported
        if f"langfuse._client.span.{name}" not in declared
    )
    assert not missing, (
        f"the adapter imports {missing} but conformance.toml does not declare them, so a rename "
        f"upstream would go both unguarded and unpublished"
    )


def test_the_installed_version_is_within_the_declared_range() -> None:
    # Non-vacuity for everything above: the checks are only meaningful against a
    # version we claim to support. This also surfaces drift between what is
    # installed and what openarmature.org/compatibility records as verified.
    import re
    from importlib.metadata import version

    installed = version("langfuse")
    declared = _declared()
    assert installed == declared["verified"], (
        f"conformance.toml publishes langfuse {declared['verified']} as the verified version "
        f"but {installed} is installed. Either move the pin deliberately and update `verified` "
        f"and `verified_on`, or restore the lock; the published number must be the tested one."
    )
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
