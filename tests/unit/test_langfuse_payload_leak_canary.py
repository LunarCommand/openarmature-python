"""Property test: no harvested content reaches an un-isolated Langfuse provider."""

# Spec basis: observability §6 / §8.4.x (proposals 0114 / 0116 / 0117 / 0118, the
# payload-leak invariant).
#
# This asserts the INVARIANT rather than the individual emission sites. Every
# harvested input carries a unique sentinel; the whole captured Langfuse tree is
# then searched for it. A site-by-site test can only cover the channels someone
# already thought of, so a newly added emission (or one the enumeration missed)
# slips through silently; this fails the moment any harvested value reaches a
# provider OA could not establish as isolated, whatever site emitted it.
#
# The isolated case asserts the sentinel IS present per channel, so the test
# cannot pass by driving nothing.

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langfuse")

from openarmature.graph.events import (  # noqa: E402
    CaughtException,
    FailureIsolatedEvent,
    InvocationCompletedEvent,
    InvocationStartedEvent,
    LlmFailedEvent,
    ToolCallFailedEvent,
)
from openarmature.observability.correlation import (  # noqa: E402
    _reset_invocation_id,
    _set_invocation_id,
)
from openarmature.observability.langfuse import (  # noqa: E402
    InMemoryLangfuseClient,
    LangfuseObserver,
)
from openarmature.observability.langfuse.client import (  # noqa: E402
    ISOLATION_ISOLATED,
    ISOLATION_LEAKED,
    ISOLATION_UNDETECTABLE,
)

# One sentinel per harvested channel, so a leak names which channel leaked.
STATE_IN = "CANARY-state-input-8f21"
STATE_OUT = "CANARY-state-output-3b77"
LLM_MSG = "CANARY-llm-error-5c04"
TOOL_ARG = "CANARY-tool-argument-9d13"
TOOL_MSG = "CANARY-tool-error-1a56"
MARKER_MSG = "CANARY-isolated-exception-7e92"

ALL_SENTINELS = (STATE_IN, STATE_OUT, LLM_MSG, TOOL_ARG, TOOL_MSG, MARKER_MSG)

_INV = "inv-canary"


async def _drive_every_channel(observer: LangfuseObserver) -> None:
    """Feed one event per harvested channel through the observer."""
    token = _set_invocation_id(_INV)
    try:
        await observer(
            InvocationStartedEvent(
                initial_state={"payload": STATE_IN},
                invocation_id=_INV,
                correlation_id=None,
                entry_node="start",
            )
        )
        await observer(
            LlmFailedEvent(
                invocation_id=_INV,
                correlation_id=None,
                node_name="call_llm",
                namespace=("call_llm",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                provider="openai",
                model="test-model",
                latency_ms=1.0,
                input_messages=[],
                request_params={},
                request_extras={},
                active_prompt=None,
                active_prompt_group=None,
                call_id="cc-llm",
                error_category="provider_unavailable",
                error_message=LLM_MSG,
            )
        )
        await observer(
            ToolCallFailedEvent(
                invocation_id=_INV,
                correlation_id=None,
                node_name="run_tool",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                call_id="cc-tool",
                tool_name="lookup",
                tool_call_id="call_1",
                arguments={"query": TOOL_ARG},
                latency_ms=1.0,
                error_type="ValueError",
                error_message=TOOL_MSG,
            )
        )
        await observer(
            FailureIsolatedEvent(
                event_name="node_failed",
                namespace=("run_tool",),
                attempt_index=0,
                fan_out_index=None,
                branch_name=None,
                pre_state={},
                post_state={},
                caught_exception=CaughtException(category="node_exception", message=MARKER_MSG, chain=()),
            )
        )
        await observer(
            InvocationCompletedEvent(
                final_state={"payload": STATE_OUT},
                status="completed",
                final_node="end",
                invocation_id=_INV,
                correlation_id=None,
            )
        )
    finally:
        _reset_invocation_id(token)


def _captured_text(client: InMemoryLangfuseClient) -> str:
    """Everything the client recorded, flattened, so the search covers any field
    a future emission might use rather than the ones enumerated today."""
    traces: list[Any] = []
    for trace in client.traces.values():
        traces.append(
            {
                "trace": {k: v for k, v in vars(trace).items() if k != "observations"},
                "observations": [vars(obs) for obs in trace.observations],
            }
        )
    return json.dumps(traces, default=str)


def _observer_on(status: str | None) -> tuple[LangfuseObserver, InMemoryLangfuseClient]:
    client = InMemoryLangfuseClient()
    if status is not None:
        client._isolation_status = status  # type: ignore[attr-defined]
    # Every payload channel opened: the strongest configuration a caller can ask
    # for, so the isolation floor is what has to hold the line.
    observer = LangfuseObserver(
        client=client,
        disable_provider_payload=False,
        disable_state_payload=False,
    )
    return observer, client


async def test_no_harvested_content_reaches_an_undetectable_provider() -> None:
    # UNDETECTABLE is the arm where an observer still exists with channels
    # requested: the binding cannot be established, so construction closes every
    # channel instead of refusing. Nothing harvested may reach the client.
    observer, client = _observer_on(ISOLATION_UNDETECTABLE)
    await _drive_every_channel(observer)
    captured = _captured_text(client)
    leaked = [s for s in ALL_SENTINELS if s in captured]
    assert not leaked, f"harvested content reached an un-isolated provider: {leaked}"


def test_a_leaked_provider_refuses_construction_rather_than_emitting() -> None:
    # On LEAKED the guarantee is stronger than suppression: with any channel live
    # the observer never comes into existence, so there is nothing to emit from.
    from openarmature.observability.langfuse import LangfuseProviderIsolationUnavailable

    with pytest.raises(LangfuseProviderIsolationUnavailable):
        _observer_on(ISOLATION_LEAKED)


async def test_marker_exception_never_reaches_langfuse_even_when_isolated() -> None:
    # The failure-isolation marker is a graph-mechanism span no mapping table
    # covers, so its exception message is prohibited outright rather than gated.
    observer, client = _observer_on(ISOLATION_ISOLATED)
    await _drive_every_channel(observer)
    assert MARKER_MSG not in _captured_text(client)


async def test_channels_are_actually_exercised_when_isolated() -> None:
    # Non-vacuity: without this, a workload that silently stopped driving a
    # channel would make the leak assertions above pass for the wrong reason.
    observer, client = _observer_on(ISOLATION_ISOLATED)
    await _drive_every_channel(observer)
    captured = _captured_text(client)
    missing = [s for s in (STATE_IN, STATE_OUT, LLM_MSG, TOOL_ARG, TOOL_MSG) if s not in captured]
    assert not missing, f"channel not exercised, so the leak test proves nothing: {missing}"


@pytest.mark.parametrize("status", [ISOLATION_LEAKED, ISOLATION_UNDETECTABLE])
async def test_reopening_channels_after_construction_still_leaks_nothing(status: str) -> None:
    # The knobs are public fields on a mutable dataclass; enforcement is at
    # emission, so re-opening a channel afterwards must not reopen the leak.
    client = InMemoryLangfuseClient()
    client._isolation_status = status  # type: ignore[attr-defined]
    observer = LangfuseObserver(client=client)
    observer.disable_provider_payload = False
    observer.disable_state_payload = False
    observer.trace_input_from_state = lambda state: state
    observer.trace_output_from_state = lambda state: state
    await _drive_every_channel(observer)
    captured = _captured_text(client)
    leaked = [s for s in ALL_SENTINELS if s in captured]
    assert not leaked, f"a channel re-opened after construction leaked: {leaked}"


def test_adapter_built_client_is_guarded_on_every_construction_path() -> None:
    # The bypass class: any observer over an OA-constructed client must be
    # guarded, not only one built through the observer's own factory.
    from openarmature.observability.langfuse import LangfuseProviderIsolationUnavailable

    leaked_client = MagicMock(_isolation_status=ISOLATION_LEAKED)
    with pytest.raises(LangfuseProviderIsolationUnavailable):
        LangfuseObserver(client=leaked_client, disable_provider_payload=False)
    with pytest.raises(LangfuseProviderIsolationUnavailable):
        LangfuseObserver(client=leaked_client, disable_state_payload=False)
    with pytest.raises(LangfuseProviderIsolationUnavailable):
        LangfuseObserver(client=leaked_client, trace_input_from_state=lambda s: s)
