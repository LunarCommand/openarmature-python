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


def test_accept_shared_provider_binds_the_ambient_provider() -> None:
    # Opt-out with a real ambient provider: OA binds THAT provider (never None,
    # which would let the SDK claim the process-global slot).
    from opentelemetry.sdk.trace import TracerProvider

    ambient = TracerProvider()

    def bind_ambient(**kwargs: Any) -> MagicMock:
        client = MagicMock()
        client._resources.tracer_provider = kwargs["tracer_provider"]
        return client

    with patch(f"{_ADAPTER}._resolve_shared_provider", return_value=ambient):
        with patch(f"{_ADAPTER}.Langfuse", side_effect=bind_ambient) as mock_lf:
            adapter = LangfuseSDKAdapter.from_credentials(
                public_key="pk", secret_key=SecretStr("sk"), accept_shared_provider=True
            )
    assert mock_lf.call_args.kwargs["tracer_provider"] is ambient
    assert adapter._isolation_status == ISOLATION_SHARED_ACCEPTED


def test_accept_shared_provider_with_no_ambient_provider_isolates_instead(caplog: Any) -> None:
    # Nothing registered to share: handing the SDK None would make it construct
    # and globally register its own provider, capturing the one-shot global slot,
    # so OA isolates instead and says so.
    with patch(f"{_ADAPTER}._resolve_shared_provider", return_value=None):
        with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_passed) as mock_lf:
            with caplog.at_level("WARNING", logger="openarmature.observability"):
                adapter = LangfuseSDKAdapter.from_credentials(
                    public_key="pk", secret_key=SecretStr("sk"), accept_shared_provider=True
                )
    assert mock_lf.call_args.kwargs["tracer_provider"] is not None
    assert adapter._isolation_status == ISOLATION_ISOLATED
    assert "no TracerProvider is registered yet" in caplog.text


def test_opted_in_but_actually_isolated_is_classified_isolated() -> None:
    # Membership, not the flag: when the SDK resolved the client onto OA's own
    # isolated provider, the opt-out must not mark it shared (which would emit a
    # false warning and disable the error gate).
    with patch(f"{_ADAPTER}._resolve_shared_provider", return_value=None):
        with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_passed):
            adapter = LangfuseSDKAdapter.from_credentials(
                public_key="pk", secret_key=SecretStr("sk"), accept_shared_provider=True
            )
    assert adapter._isolation_status == ISOLATION_ISOLATED


def test_blank_credentials_are_rejected_at_the_boundary() -> None:
    # The SDK falls back to ambient LANGFUSE_* env credentials for a blank value,
    # so a blank field must fail rather than silently authenticate elsewhere.
    with patch(f"{_ADAPTER}.Langfuse"):
        with pytest.raises(ValueError, match="public_key"):
            LangfuseSDKAdapter.from_credentials(public_key="  ", secret_key=SecretStr("sk"))
        with pytest.raises(ValueError, match="secret_key"):
            LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr(""))


def test_cached_public_key_warns_that_client_config_is_discarded(caplog: Any) -> None:
    # The SDK caches one client per credential, so a second construction drops
    # this call's host / options; that discard must not be silent.
    with patch(f"{_ADAPTER}._public_key_is_cached", return_value=True):
        with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_passed):
            with caplog.at_level("WARNING", logger="openarmature.observability"):
                LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert "cached client" in caplog.text


def test_sample_rate_is_applied_to_the_isolated_provider() -> None:
    # The SDK applies sample_rate only on a provider it builds, which isolation
    # bypasses, so the ratio has to land on OA's provider or sampling is a no-op.
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_passed) as mock_lf:
        LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"), sample_rate=0.25)
    provider = mock_lf.call_args.kwargs["tracer_provider"]
    assert isinstance(provider.sampler, TraceIdRatioBased)


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
    assert "suppressing the provider and state payloads you enabled" in caplog.text


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


def test_error_message_follows_the_payload_flag_not_the_isolation_status() -> None:
    # 0118: the flag governs the harvested error message. With payloads off it is
    # never emitted, whatever the provider turned out to be -- including a plain
    # caller-supplied client, which records no isolation status at all.
    for status in (ISOLATION_ISOLATED, ISOLATION_SHARED_ACCEPTED, None):
        obs = LangfuseObserver(client=MagicMock(_isolation_status=status))
        assert obs.disable_provider_payload is True  # the default posture
        assert obs._emits_harvested_error_message() is False


def test_error_message_emits_with_payloads_on_and_an_isolated_provider() -> None:
    obs = LangfuseObserver(
        client=MagicMock(_isolation_status=ISOLATION_ISOLATED), disable_provider_payload=False
    )
    assert obs._emits_harvested_error_message() is True


@pytest.mark.parametrize("status", [ISOLATION_LEAKED, ISOLATION_UNDETECTABLE])
def test_error_message_withheld_on_a_provider_not_established_as_isolated(status: str) -> None:
    # Not a second gate: the §6 arms already closed the flag for these statuses
    # (suppress-all), or refused construction outright, so the message cannot ride.
    obs = LangfuseObserver(client=MagicMock(_isolation_status=status))
    obs.disable_provider_payload = False  # a caller reopening it post-construction
    assert obs._emits_harvested_error_message() is False


def test_tracing_disabled_classified_isolated_not_undetectable() -> None:
    # A tracing-disabled client has no provider but exports nothing, so it is
    # no-leak (ISOLATED), not the suppress floor (UNDETECTABLE).
    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_tracing_disabled):
        adapter = LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert adapter._isolation_status == ISOLATION_ISOLATED


def test_tracing_disabled_on_a_foreign_provider_is_still_no_leak() -> None:
    # Tracing off exports nothing whatever provider the cached manager holds, so
    # it must not classify LEAKED and refuse the call.
    def disabled_on_foreign(**kwargs: Any) -> MagicMock:
        client = MagicMock()
        client._resources.tracer_provider = object()
        client._tracing_enabled = False
        return client

    with patch(f"{_ADAPTER}.Langfuse", side_effect=disabled_on_foreign):
        adapter = LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert adapter._isolation_status == ISOLATION_ISOLATED


# --- the guard applies to every path to an OA-constructed client --------------


def test_adapter_built_client_is_guarded_through_the_plain_constructor() -> None:
    # The arms live in __post_init__, so handing a from_credentials adapter to the
    # ordinary constructor is guarded exactly like the observer factory.
    with patch(f"{_ADAPTER}.Langfuse", side_effect=_langfuse_binding_other):
        adapter = LangfuseSDKAdapter.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    with pytest.raises(LangfuseProviderIsolationUnavailable):
        LangfuseObserver(client=adapter, disable_provider_payload=False)


def test_payload_knob_reopened_after_construction_is_still_blocked() -> None:
    # The knobs are public fields on a mutable dataclass; emission consults the
    # isolation status so a channel re-opened afterwards cannot leak.
    obs = LangfuseObserver(client=MagicMock(_isolation_status=ISOLATION_LEAKED))
    obs.disable_provider_payload = False
    assert obs._emits_provider_payload() is False


def test_state_channel_reopened_after_construction_falls_back_to_the_stub() -> None:
    from openarmature.graph.events import InvocationStartedEvent

    obs = LangfuseObserver(client=MagicMock(_isolation_status=ISOLATION_LEAKED))
    obs.disable_state_payload = False
    obs.trace_input_from_state = _hook
    resolved = obs._resolve_trace_input(
        InvocationStartedEvent(
            initial_state={"secret": "pii"},
            invocation_id="inv-1",
            correlation_id=None,
            entry_node="start",
        )
    )
    assert resolved == {"entry_node": "start"}  # the minimal stub, no state


# --- behavioral: what actually lands on the observation -----------------------


async def test_failed_tool_observation_omits_message_under_the_default_posture() -> None:
    # The gate's effect, not just its predicate. error_type is NOT gated (0118) --
    # it is the only failure discriminator a tool observation has, since a tool
    # failure carries no error category -- but the message is withheld and must
    # not be smuggled into statusMessage in its place.
    from openarmature.graph.events import ToolCallFailedEvent
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id
    from openarmature.observability.langfuse import InMemoryLangfuseClient

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)  # disable_provider_payload defaults True
    token = _set_invocation_id("inv-leak")
    try:
        await observer(
            ToolCallFailedEvent(
                invocation_id="inv-leak",
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-1",
                tool_name="get_weather",
                tool_call_id="call_1",
                arguments={"city": "Paris"},
                latency_ms=3.0,
                error_type="ValueError",
                error_message="rejected SSN 123-45-6789",
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-leak"].observations if o.type == "tool")
    assert obs.level == "ERROR"
    assert "error_message" not in obs.metadata
    assert obs.metadata.get("error_type") == "ValueError"  # ungated classification token
    assert obs.status_message is None


async def test_failed_tool_observation_keeps_error_message_with_payloads_on() -> None:
    # The converse: with payloads enabled the message reports normally, so the
    # gate does not break legitimate error triage.
    from openarmature.graph.events import ToolCallFailedEvent
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id
    from openarmature.observability.langfuse import InMemoryLangfuseClient

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, disable_provider_payload=False)
    token = _set_invocation_id("inv-ok")
    try:
        await observer(
            ToolCallFailedEvent(
                invocation_id="inv-ok",
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-2",
                tool_name="get_weather",
                tool_call_id="call_2",
                arguments={"city": "Paris"},
                latency_ms=3.0,
                error_type="TimeoutError",
                error_message="tool timed out",
            )
        )
    finally:
        _reset_invocation_id(token)

    obs = next(o for o in client.traces["inv-ok"].observations if o.type == "tool")
    assert obs.metadata.get("error_message") == "tool timed out"
    assert obs.status_message == "tool timed out"


# The Tool test above covers one of the four gated sites. These cover the other
# three: without them, reverting any of the LLM / Embedding / Retriever handlers
# to the retired isolation predicate passes the whole suite.


async def _one_observation(observer: LangfuseObserver, client: Any, event: Any, obs_type: str) -> Any:
    from openarmature.observability.correlation import _reset_invocation_id, _set_invocation_id

    token = _set_invocation_id(_INV_DEFAULT)
    try:
        await observer(event)
    finally:
        _reset_invocation_id(token)
    return next(o for o in client.traces[_INV_DEFAULT].observations if o.type == obs_type)


_INV_DEFAULT = "inv-default-posture"


async def test_failed_llm_generation_omits_message_under_the_default_posture() -> None:
    from openarmature.graph.events import LlmFailedEvent
    from openarmature.observability.langfuse import InMemoryLangfuseClient

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)  # disable_provider_payload defaults True
    obs = await _one_observation(
        observer,
        client,
        LlmFailedEvent(
            invocation_id=_INV_DEFAULT,
            correlation_id=None,
            node_name="call_llm",
            namespace=("call_llm",),
            attempt_index=0,
            fan_out_index=None,
            branch_name=None,
            provider="openai",
            model="m",
            latency_ms=1.0,
            input_messages=[],
            request_params={},
            request_extras={},
            active_prompt=None,
            active_prompt_group=None,
            call_id="cc-1",
            error_category="provider_unavailable",
            error_message="upstream said: prompt was 'secret'",
        ),
        "generation",
    )
    assert "error_message" not in obs.metadata
    assert obs.status_message == "provider_unavailable"  # the category still rides


async def test_failed_embedding_omits_message_under_the_default_posture() -> None:
    from openarmature.graph.events import EmbeddingFailedEvent
    from openarmature.observability.langfuse import InMemoryLangfuseClient

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    obs = await _one_observation(
        observer,
        client,
        EmbeddingFailedEvent(
            invocation_id=_INV_DEFAULT,
            correlation_id=None,
            node_name="embed",
            namespace=("embed",),
            attempt_index=0,
            fan_out_index=None,
            branch_name=None,
            provider="openai",
            model="m",
            latency_ms=1.0,
            input_strings=["x"],
            request_params={},
            request_extras={},
            active_prompt=None,
            active_prompt_group=None,
            call_id="cc-2",
            error_category="provider_unavailable",
            error_message="upstream said: input was 'secret'",
        ),
        "embedding",
    )
    assert "error_message" not in obs.metadata
    assert obs.status_message == "provider_unavailable"


async def test_failed_rerank_omits_message_under_the_default_posture() -> None:
    from openarmature.graph.events import RerankFailedEvent
    from openarmature.observability.langfuse import InMemoryLangfuseClient

    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client)
    obs = await _one_observation(
        observer,
        client,
        RerankFailedEvent(
            invocation_id=_INV_DEFAULT,
            correlation_id=None,
            node_name="rerank",
            namespace=("rerank",),
            attempt_index=0,
            fan_out_index=None,
            branch_name=None,
            provider="cohere",
            model="m",
            latency_ms=1.0,
            query="q",
            documents=["d"],
            document_count=1,
            top_k=1,
            request_params={},
            request_extras={},
            active_prompt=None,
            active_prompt_group=None,
            call_id="cc-3",
            error_category="provider_unavailable",
            error_message="upstream said: query was 'secret'",
        ),
        "retriever",
    )
    assert "error_message" not in obs.metadata
    assert obs.status_message == "provider_unavailable"


def test_undetectable_under_the_default_posture_neither_raises_nor_warns(caplog: Any) -> None:
    # With no channel live the suppress arm takes nothing away, so warning there
    # would contradict the documented "an un-isolatable client is harmless".
    with _patched_adapter(ISOLATION_UNDETECTABLE):
        with caplog.at_level("WARNING", logger="openarmature.observability"):
            obs = LangfuseObserver.from_credentials(public_key="pk", secret_key=SecretStr("sk"))
    assert obs.disable_provider_payload is True
    assert "cannot establish" not in caplog.text
