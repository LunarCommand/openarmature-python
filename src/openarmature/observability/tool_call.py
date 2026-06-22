# Spec: realizes the graph-engine §6 tool-call instrumentation scope
# (proposal 0063, spec v0.69.0). A node-body primitive the caller wraps
# a tool execution in; OA observes the execution and dispatches a typed
# ToolCallEvent (success) or ToolCallFailedEvent (failure) at outcome
# time. OA does NOT run, select, loop, retry, or feed back tools
# (llm-provider §1) -- the caller runs the tool inside the scope.
#
# Shape follows the existing node-body context-manager precedent
# (prompts.context.with_active_prompt / with_active_prompt_group): a sync
# @contextmanager. A sync ``with`` brackets the awaited tool call in its
# body fine, and everything the scope does (capture identity, mint
# call_id, time, dispatch) is synchronous. Identity is captured at scope
# ENTRY (the §6 scope-entry-identity rule; the inline case is the trivial
# instance where entry and outcome share one context). Dispatch goes
# through ``current_dispatch()``, the same path set_invocation_metadata /
# FailureIsolationMiddleware use; it is None outside an invocation (no
# observers), in which case the body still runs and only the event is
# skipped.
#
# v1 ships this inline bracketing form only; the deferred start/complete
# split (result lands in a later turn) is a spec MAY, not yet needed.

"""Tool-call instrumentation scope: ``with_tool_call``."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


# Sentinel distinguishing "the caller never reported a result" from "the
# tool returned None" -- a forgotten set_result() resolves to a null
# result rather than masquerading as a real one.
class _Unset:
    pass


_UNSET = _Unset()


class ToolCallScope:
    """Handle yielded by :func:`with_tool_call`.

    The caller reports the tool's return value via :meth:`set_result` so
    the success event can carry it. ``call_id`` is OA's per-execution
    correlation token (minted when the scope is entered), exposed for the
    caller to correlate a deferred completion if needed.
    """

    __slots__ = ("call_id", "_result")

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self._result: Any = _UNSET

    def set_result(self, value: Any) -> None:
        """Report the tool's return value to the scope."""
        self._result = value


@contextmanager
def with_tool_call(
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    tool_call_id: str | None = None,
) -> Iterator[ToolCallScope]:
    """Instrument a tool execution inside a node body.

    Wrap the caller's tool execution in this scope and report the
    result via :meth:`ToolCallScope.set_result`::

        with with_tool_call("get_weather", {"city": "Paris"}, tool_call_id="call_abc") as scope:
            result = await get_weather(city="Paris")
            scope.set_result(result)

    On clean exit a :class:`~openarmature.graph.events.ToolCallEvent` is
    dispatched carrying the reported result; on an exception a
    :class:`~openarmature.graph.events.ToolCallFailedEvent` is dispatched
    (with the exception's type + message) and the exception **re-raises**
    -- the scope observes, it does not swallow. OA does not run the tool,
    choose it, loop, or feed the result back to the model; those stay in
    the caller's graph.

    ``arguments`` is the observability representation of the call inputs
    (for an LLM-originated call, the parsed ``ToolCall.arguments``); it is
    independent of how the caller actually invokes the tool.
    ``tool_call_id`` links back to the ``LlmCompletionEvent.output_tool_calls``
    entry this execution satisfies, or ``None`` for a standalone
    instrumented function. ``arguments`` and the result are payload;
    observer-side gating (``disable_provider_payload``) applies at
    rendering.
    """
    from openarmature.graph.events import ToolCallEvent, ToolCallFailedEvent

    from .correlation import (
        current_attempt_index,
        current_branch_name,
        current_correlation_id,
        current_dispatch,
        current_fan_out_index,
        current_invocation_id,
        current_namespace_prefix,
    )
    from .metadata import current_invocation_metadata

    # Scope-entry identity (§6): the node that initiated the execution.
    namespace = current_namespace_prefix()
    node_name = namespace[-1] if namespace else ""
    invocation_id = current_invocation_id() or ""
    correlation_id = current_correlation_id()
    attempt_index = current_attempt_index()
    fan_out_index = current_fan_out_index()
    branch_name = current_branch_name()
    caller_metadata = dict(current_invocation_metadata())
    call_id = uuid.uuid4().hex
    dispatch = current_dispatch()

    scope = ToolCallScope(call_id)
    start = time.perf_counter()
    try:
        yield scope
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        if dispatch is not None:
            dispatch(
                ToolCallFailedEvent(
                    invocation_id=invocation_id,
                    correlation_id=correlation_id,
                    node_name=node_name,
                    namespace=namespace,
                    attempt_index=attempt_index,
                    fan_out_index=fan_out_index,
                    branch_name=branch_name,
                    call_id=call_id,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    latency_ms=latency_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    caller_invocation_metadata=caller_metadata,
                )
            )
        # Observe, don't swallow: the exception propagates to the caller.
        raise
    latency_ms = (time.perf_counter() - start) * 1000.0
    if dispatch is not None:
        result = None if isinstance(scope._result, _Unset) else scope._result
        dispatch(
            ToolCallEvent(
                invocation_id=invocation_id,
                correlation_id=correlation_id,
                node_name=node_name,
                namespace=namespace,
                attempt_index=attempt_index,
                fan_out_index=fan_out_index,
                branch_name=branch_name,
                call_id=call_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=result,
                latency_ms=latency_ms,
                caller_invocation_metadata=caller_metadata,
            )
        )


__all__ = ["ToolCallScope", "with_tool_call"]
