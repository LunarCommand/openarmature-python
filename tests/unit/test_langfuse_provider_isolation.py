"""Langfuse client ownership + TracerProvider isolation (0114 / 0116)."""

# Spec basis: observability §6 / §8.9 (proposals 0114 + 0116, the payload-leak
# invariant). Behavior ships ahead of the v0.110.0 pin; these unit tests pin the
# credentials-in construction, the per-credential isolated-provider reuse, the
# post-construct isolation classification, and the raise / suppress / opt-out
# arms. The end-to-end provider-leak assertions are conformance fixtures 157/158.

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

pytest.importorskip("langfuse")

from openarmature.observability.langfuse import (  # noqa: E402
    LangfuseObserver,
    LangfuseProviderIsolationUnavailable,
)
from openarmature.observability.langfuse import adapter as _adapter_mod  # noqa: E402
from openarmature.observability.langfuse.adapter import (  # noqa: E402
    ISOLATION_ISOLATED,
    ISOLATION_LEAKED,
    ISOLATION_SHARED_ACCEPTED,
    ISOLATION_UNDETECTABLE,
    LangfuseSDKAdapter,
)

_ADAPTER = "openarmature.observability.langfuse.adapter"


@pytest.fixture(autouse=True)
def _clear_provider_registry() -> Any:  # pyright: ignore[reportUnusedFunction]
    # The isolated-provider registry is process-wide; clear it around each test so
    # reuse assertions and fresh-build assertions do not cross-contaminate.
    _adapter_mod._ISOLATED_PROVIDERS.clear()
    yield
    _adapter_mod._ISOLATED_PROVIDERS.clear()


def _langfuse_binding_passed(**kwargs: Any) -> MagicMock:
    # OA won the singleton: the SDK bound the provider we handed it.
    client = MagicMock()
    client._resources.tracer_provider = kwargs["tracer_provider"]
    return client


def _langfuse_binding_other(**kwargs: Any) -> MagicMock:
    # The singleton returned a client bound to someone else's provider.
    client = MagicMock()
    client._resources.tracer_provider = object()
    return client


def _langfuse_no_resources(**kwargs: Any) -> MagicMock:
    # A future SDK that does not expose the binding at all.
    return MagicMock(spec=[])


def _langfuse_tracing_disabled(**kwargs: Any) -> MagicMock:
    # Tracing disabled: no provider bound, nothing exports (not a hidden binding).
    client = MagicMock()
    client._resources.tracer_provider = None
    client._tracing_enabled = False
    return client


def _hook(state: Any) -> Any:
    return state


# --- adapter: construction + isolation classification -------------------------


def test_isolate_default_binds_dedicated_provider_status_isolated() -> None:
    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_passed) as mock_lf:
        adapter = LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"), host="h")
    assert mock_lf.call_args.kwargs["tracer_provider"] is not None
    assert adapter._isolation_status == ISOLATION_ISOLATED


def test_singleton_binding_other_provider_status_leaked() -> None:
    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_other):
        adapter = LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert adapter._isolation_status == ISOLATION_LEAKED


def test_binding_not_exposed_status_undetectable() -> None:
    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_no_resources):
        adapter = LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert adapter._isolation_status == ISOLATION_UNDETECTABLE


def test_accept_shared_provider_binds_ambient_status_shared_accepted() -> None:
    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_other) as mock_lf:
        adapter = LangfuseSDKAdapter.from_credentials(
            public_key="pk", secret_key=SecretStr("sk"), accept_shared_provider=True
        )
    assert mock_lf.call_args.kwargs["tracer_provider"] is None
    assert adapter._isolation_status == ISOLATION_SHARED_ACCEPTED


def test_isolated_provider_reused_per_public_key() -> None:
    providers: list[Any] = []

    def capture(**kwargs: Any) -> MagicMock:
        providers.append(kwargs["tracer_provider"])
        client = MagicMock()
        client._resources.tracer_provider = kwargs["tracer_provider"]
        return client

    with patch(f"{_ADAPTER}.Langfuse", side_effect=capture):
        LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
        LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
        LangfuseSDKAdapter.from_credentials(public_key="other", secret_key=SecretStr("sk"))
    assert providers[0] is providers[1]  # same key reuses one provider
    assert providers[2] is not providers[0]  # a different key gets its own


def test_managed_tracer_provider_kwarg_rejected() -> None:
    from opentelemetry.sdk.trace import TracerProvider

    with patch(f"{_ADAPTER}.Langfuse"):
        with pytest.raises(ValueError, match="tracer_provider"):
            LangfuseSDKAdapter.from_credentials(
                public_key="pk", secret_key=SecretStr("sk"), tracer_provider=TracerProvider()
            )


def test_mode_a_supplied_client_is_wrapped_not_replaced() -> None:
    sentinel: Any = object()
    adapter = LangfuseSDKAdapter(sentinel)
    assert adapter._client is sentinel
    assert adapter._isolation_status is None  # caller owns the provider (mode a)


# --- observer: the raise / suppress / opt-out policy --------------------------


def _patched_adapter(status: str) -> Any:
    stub = MagicMock()
    stub._isolation_status = status
    return patch.object(LangfuseSDKAdapter, "from_credentials", return_value=stub)


def test_observer_payloads_off_never_raises_even_on_leak() -> None:
    # Default disable_provider_payload=True: no payloads, so an un-isolatable
    # client is harmless.
    with _patched_adapter(ISOLATION_LEAKED):
        obs = LangfuseObserver.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert obs.disable_provider_payload is True


def test_observer_payloads_on_leak_raises() -> None:
    with _patched_adapter(ISOLATION_LEAKED):
        with pytest.raises(LangfuseProviderIsolationUnavailable):
            LangfuseObserver.from_credentials(
                public_key="pk", secret_key=SecretStr("sk"), disable_provider_payload=False
            )


def test_observer_payloads_on_isolated_proceeds() -> None:
    with _patched_adapter(ISOLATION_ISOLATED):
        obs = LangfuseObserver.from_credentials(
            public_key="pk", secret_key=SecretStr("sk"), disable_provider_payload=False
        )
    assert obs.disable_provider_payload is False


def test_observer_undetectable_suppresses_all_channels_and_warns(caplog: Any) -> None:
    with _patched_adapter(ISOLATION_UNDETECTABLE):
        with caplog.at_level("WARNING", logger="openarmature.observability"):
            obs = LangfuseObserver.from_credentials(
                public_key="pk",
                secret_key=SecretStr("sk"),
                disable_provider_payload=False,
                disable_state_payload=False,
                trace_input_from_state=_hook,
            )
    # Suppress-all: every construction-time channel forced off (fail-safe).
    assert obs.disable_provider_payload is True
    assert obs.disable_state_payload is True
    assert obs.trace_input_from_state is None
    assert "suppressing all provider and state payloads" in caplog.text


def test_observer_accept_shared_provider_warns_and_proceeds(caplog: Any) -> None:
    with _patched_adapter(ISOLATION_SHARED_ACCEPTED):
        with caplog.at_level("WARNING", logger="openarmature.observability"):
            obs = LangfuseObserver.from_credentials(
                public_key="pk",
                secret_key=SecretStr("sk"),
                accept_shared_provider=True,
                disable_provider_payload=False,
            )
    assert obs.disable_provider_payload is False  # proceeds (acknowledged leak)
    assert "acknowledged" in caplog.text


def test_observer_forwards_langfuse_kwargs_and_observer_kwargs() -> None:
    with patch.object(LangfuseSDKAdapter, "from_credentials") as mock_fc:
        mock_fc.return_value._isolation_status = ISOLATION_ISOLATED
        obs = LangfuseObserver.from_credentials(
            public_key="pk",
            secret_key=SecretStr("sk"),
            langfuse_kwargs={"environment": "prod"},
            disable_llm_spans=True,
        )
    assert mock_fc.call_args.kwargs["environment"] == "prod"
    assert mock_fc.call_args.kwargs["accept_shared_provider"] is False
    assert obs.disable_llm_spans is True


def test_state_payload_channel_on_leak_raises() -> None:
    # 0117: the state payload is a leak channel independent of the provider knob.
    with _patched_adapter(ISOLATION_LEAKED):
        with pytest.raises(LangfuseProviderIsolationUnavailable):
            LangfuseObserver.from_credentials(
                public_key="pk", secret_key=SecretStr("sk"), disable_state_payload=False
            )


def test_supplied_hook_on_leak_raises() -> None:
    # A supplied trace_input_from_state hook emits regardless of the knob, so a
    # supplied hook is a live channel even under the default privacy posture.
    with _patched_adapter(ISOLATION_LEAKED):
        with pytest.raises(LangfuseProviderIsolationUnavailable):
            LangfuseObserver.from_credentials(
                public_key="pk", secret_key=SecretStr("sk"), trace_input_from_state=_hook
            )


def test_all_channels_off_on_leak_does_not_raise() -> None:
    # Fully-locked-down posture: no construction-time channel live, so an
    # un-isolatable client is harmless at construction (the error-message channel
    # is handled per-emission, verified end-to-end by conformance fixture 158).
    with _patched_adapter(ISOLATION_LEAKED):
        obs = LangfuseObserver.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert obs.disable_provider_payload is True


@pytest.mark.parametrize(
    "status,expected",
    [
        (ISOLATION_LEAKED, True),
        (ISOLATION_UNDETECTABLE, True),
        (ISOLATION_ISOLATED, False),
        (ISOLATION_SHARED_ACCEPTED, False),
        (None, False),
    ],
)
def test_omit_harvested_error_gate(status: Any, expected: bool) -> None:
    # The per-emission gate: a failed observation's harvested error_message /
    # error_type is dropped only when OA has not established isolation.
    obs = LangfuseObserver(client=MagicMock(_isolation_status=status))
    assert obs._omit_harvested_error() is expected


def test_tracing_disabled_classified_isolated_not_undetectable() -> None:
    # A tracing-disabled client has no provider but exports nothing, so it is
    # no-leak (ISOLATED), not the suppress floor (UNDETECTABLE).
    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_tracing_disabled):
        adapter = LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert adapter._isolation_status == ISOLATION_ISOLATED
