"""The conformance harness's `requires_capability` audience gate."""

# Spec basis: conformance-adapter §5.5 (proposal 0116). The gate decides whether a
# fixture case applies to this adapter. Its failure mode is silence -- a gate that
# skips too much reports green while asserting nothing -- so the not-silent
# properties are tested directly rather than inferred from the fixture runs.

from __future__ import annotations

import pytest

from tests.conformance.harness.capabilities import (
    adapter_capabilities,
    assert_some_case_ran,
    capability_skip_reason,
)


def test_declared_detection_capability_matches_the_installed_sdk() -> None:
    # The declaration is a public claim in conformance.toml, and asserting it
    # against its own literal would only prove the file parses. What makes this
    # adapter detection-capable is the probe in the adapter: it reads
    # client._resources.tracer_provider, an SDK internal the adapter itself
    # treats as possibly-absent. So the claim is checked against the SDK actually
    # installed -- an upgrade that drops or renames that attribute silently
    # demotes every client to `undetectable`, and this is what goes red.
    import inspect

    langfuse = pytest.importorskip(
        "langfuse",
        reason="the declared capability is a claim about the installed Langfuse SDK",
    )
    resource_manager = pytest.importorskip("langfuse._client.resource_manager")

    assert adapter_capabilities()["langfuse_bound_provider_detection"] is True

    # The two hops of _classify_isolation's probe: client._resources, then
    # .tracer_provider on it. A rename or removal of either changes this source,
    # which is the point -- the declaration must break loudly rather than let
    # every client quietly classify `undetectable`.
    client_src = inspect.getsource(langfuse.Langfuse)
    manager_src = inspect.getsource(resource_manager.LangfuseResourceManager)
    hint = (
        "conformance.toml declares langfuse_bound_provider_detection = true, but the installed "
        "langfuse SDK no longer exposes the internal adapter._classify_isolation reads. The raise "
        "arm that declaration advertises is unreachable and every client would classify "
        "`undetectable`; fix the probe or drop the claim."
    )
    assert "self._resources" in client_src, f"Langfuse no longer sets _resources. {hint}"
    assert "self.tracer_provider" in manager_src, (
        f"LangfuseResourceManager no longer sets tracer_provider. {hint}"
    )


def test_ungated_case_runs() -> None:
    assert capability_skip_reason(None) is None
    assert capability_skip_reason({}) is None


def test_matching_gate_runs() -> None:
    assert capability_skip_reason({"langfuse_bound_provider_detection": True}) is None


def test_mismatched_gate_is_a_recognized_skip() -> None:
    reason = capability_skip_reason({"langfuse_bound_provider_detection": False})
    assert reason is not None
    assert "recognized skip" in reason


def test_undeclared_capability_is_an_error_not_a_skip() -> None:
    # The load-bearing property. §5.5 permits treating an undeclared capability as
    # absent, which would make a newly-added capability name switch off every case
    # that gates on it -- green, and asserting nothing. Here it raises instead.
    with pytest.raises(AssertionError, match="does not declare"):
        capability_skip_reason({"some_future_capability": True})


def test_non_boolean_requirement_is_rejected_not_coerced() -> None:
    # bool("false") is True, so coercing would select the capable arm for a
    # quoted fixture value and assert the wrong half of the contract in silence.
    with pytest.raises(AssertionError, match="not a boolean"):
        capability_skip_reason({"langfuse_bound_provider_detection": "false"})


def test_fixture_that_ran_nothing_fails() -> None:
    with pytest.raises(AssertionError, match="asserted nothing"):
        assert_some_case_ran("158-x", 0, {"a": "gated", "b": "gated"})


def test_wholly_deferred_fixture_fails_too() -> None:
    # The exclusion channels are checked together: a fixture every one of whose
    # cases is deferred is as empty as one entirely gated out, and reads as green
    # either way. It belongs in the fixture-level deferral table instead.
    with pytest.raises(AssertionError, match="asserted nothing"):
        assert_some_case_ran("158-x", 0, {"a": "per-case deferral", "b": "per-case deferral"})


def test_duplicate_case_names_cannot_hide_an_empty_run() -> None:
    # Two excluded cases sharing a name collapse to ONE entry in `excluded`. A
    # guard comparing len(excluded) against the case total would come up short
    # and pass; counting executions is immune to how the names collide.
    with pytest.raises(AssertionError, match="asserted nothing"):
        assert_some_case_ran("158-x", 0, {"<unnamed>": "gated"})


def test_zero_case_fixture_fails() -> None:
    # A fixture with `cases: []` loops zero times and excludes nothing, so the
    # message has nothing to quote -- but it still asserted nothing.
    with pytest.raises(AssertionError, match="no cases at all"):
        assert_some_case_ran("158-x", 0, {})


def test_failure_message_names_each_exclusion_reason() -> None:
    # The message has to distinguish the channels, or a reader cannot tell a stale
    # deferral from a wrong capability declaration.
    with pytest.raises(AssertionError) as excinfo:
        assert_some_case_ran("158-x", 0, {"a": "per-case deferral", "b": "recognized skip: ..."})
    assert "a: per-case deferral" in str(excinfo.value)
    assert "b: recognized skip" in str(excinfo.value)


def test_one_execution_is_enough_to_pass() -> None:
    assert_some_case_ran("158-x", 1, {"a": "gated"})


def test_unexcluded_fixture_passes() -> None:
    assert_some_case_ran("157-x", 2, {})
