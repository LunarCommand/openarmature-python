"""Where the bundled observers parent a provider call: under the calling node
when the node body makes it, and under the nearest enclosing wrapper when the
node's middleware makes it, whatever the scheduling."""

# Spec: observability §5.5 *Lineage-resolved parent* and *Resolution is
# structural*, and the §8.4.3 promise that the OTel and Langfuse trees agree.
#
# No barrier, unlike the fixtures. Their `await_event_delivery` keeps the calling
# node's span closed while the provider event resolves, which hides a resolver
# that picks the calling node whenever its span is open. Measured against that
# resolver: at `pre` OTel parents the no-yield call under `guard`, since it opens
# node spans when `started` is dispatched, while Langfuse opens observations on
# delivery and gets it right, so there the same bug is latent rather than
# reachable. At `post` both observers parent every variant under `guard`: the
# node's completed event waits for the middleware chain to return, so its span is
# still open. The `post` cases also fail an engine that leaves the body flag set
# after `node.run`.

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openarmature.graph import END, CompiledGraph, GraphBuilder, State
from openarmature.llm import OpenAIProvider, UserMessage
from openarmature.observability.langfuse import InMemoryLangfuseClient, LangfuseObserver
from openarmature.observability.otel import OTelObserver

_BODY = {
    "id": "cc",
    "object": "chat.completion",
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

_VARIANTS = pytest.mark.parametrize(
    ("phase", "yield_after_call"),
    [("pre", False), ("pre", True), ("post", False), ("post", True)],
    ids=["pre-no-yield", "pre-sleep0", "post-no-yield", "post-sleep0"],
)


class _Top(State):
    items: list[int] = [0]
    results: list[int] = []


class _Leaf(State):
    item: int = 0
    result: int = 0


def _provider() -> OpenAIProvider:
    return OpenAIProvider(
        base_url="http://mock-llm.test",
        model="m",
        api_key="t",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, content=json.dumps(_BODY).encode(), headers={"Content-Type": "application/json"}
            )
        ),
    )


def _wrapped_guard_graph(
    provider: OpenAIProvider, *, phase: str, yield_after_call: bool
) -> CompiledGraph[_Top]:
    """A fan-out `a` whose one instance runs `guard`. Guard's node middleware
    makes one provider call, before `next()` at `pre` or after it at `post`."""

    async def _guard(s: _Leaf) -> dict[str, Any]:
        return {"result": s.item + 1}

    async def _call() -> None:
        await provider.complete([UserMessage(content="guardrail check")])
        if yield_after_call:
            await asyncio.sleep(0)

    async def _wrapper(state: Any, next_call: Any) -> Any:
        if phase == "pre":
            await _call()
            return await next_call(state)
        result = await next_call(state)
        await _call()
        return result

    leaf = (
        GraphBuilder(_Leaf)
        .add_node("guard", _guard, middleware=[_wrapper])
        .add_edge("guard", END)
        .set_entry("guard")
        .compile()
    )
    return (
        GraphBuilder(_Top)
        .add_fan_out_node(
            "a",
            subgraph=leaf,
            items_field="items",
            item_field="item",
            collect_field="result",
            target_field="results",
        )
        .add_edge("a", END)
        .set_entry("a")
        .compile()
    )


async def _otel_spans(
    graph: CompiledGraph[Any], initial: State, provider: OpenAIProvider
) -> list[ReadableSpan]:
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    graph.attach_observer(observer)
    try:
        await graph.invoke(initial)
        await graph.drain()
    finally:
        observer.shutdown()
        await provider.aclose()
    return list(exporter.get_finished_spans())


def _otel_llm_parent(spans: list[ReadableSpan]) -> ReadableSpan:
    (llm,) = [s for s in spans if s.name == "openarmature.llm.complete"]
    assert llm.parent is not None, "the provider span has no parent"
    by_id = {ctx.span_id: s for s in spans if (ctx := s.get_span_context()) is not None}
    return by_id[llm.parent.span_id]


@_VARIANTS
async def test_wrapper_call_parents_under_the_fan_out_instance_on_otel(
    phase: str, yield_after_call: bool
) -> None:
    provider = _provider()
    graph = _wrapped_guard_graph(provider, phase=phase, yield_after_call=yield_after_call)
    parent = _otel_llm_parent(await _otel_spans(graph, _Top(), provider))
    assert parent.name != "guard", "a call from guard's middleware parented under guard's own span"
    # The instance dispatch span, not the fan-out node span that shares its name.
    assert (parent.attributes or {}).get("openarmature.fan_out.parent_node_name") == "a", (
        f"expected the fan-out instance span of `a`, got {parent.name!r} with {dict(parent.attributes or {})}"
    )


@_VARIANTS
async def test_wrapper_call_parents_under_the_fan_out_instance_on_langfuse(
    phase: str, yield_after_call: bool
) -> None:
    provider = _provider()
    graph = _wrapped_guard_graph(provider, phase=phase, yield_after_call=yield_after_call)
    client = InMemoryLangfuseClient()
    graph.attach_observer(LangfuseObserver(client=client))
    try:
        await graph.invoke(_Top())
        await graph.drain()
    finally:
        await provider.aclose()
    (trace,) = client.traces.values()
    by_id = {o.id: o for o in trace.observations}
    (generation,) = [o for o in trace.observations if o.type == "generation"]
    assert generation.parent_observation_id is not None, "the generation has no parent observation"
    parent = by_id[generation.parent_observation_id]
    assert parent.name != "guard", "a call from guard's middleware parented under guard's own observation"
    assert (parent.metadata or {}).get("fan_out_parent_node_name") == "a", (
        f"expected the fan-out instance observation of `a`, got {parent.name!r} with {parent.metadata}"
    )


async def test_a_body_call_after_a_nested_invocation_parents_under_its_own_node() -> None:
    # The engine marks the body by setting and resetting a flag, so a body that
    # runs another graph gets its own value back. A nested node step that clears
    # the flag on exit instead would leave this second call reading as a
    # middleware call, parented under the invocation span. Killed by replacing
    # `_step_function_node`'s reset with a False set.
    class _S(State):
        n: int = 0

    async def _noop(_s: _S) -> dict[str, Any]:
        return {}

    inner = GraphBuilder(_S).add_node("inner", _noop).add_edge("inner", END).set_entry("inner").compile()
    provider = _provider()

    async def _outer(_s: _S) -> dict[str, Any]:
        await inner.invoke(_S())
        await provider.complete([UserMessage(content="after the nested run")])
        return {}

    graph = GraphBuilder(_S).add_node("outer", _outer).add_edge("outer", END).set_entry("outer").compile()
    parent = _otel_llm_parent(await _otel_spans(graph, _S(), provider))
    assert parent.name == "outer", f"a call from outer's body parented under {parent.name!r}"


async def test_middleware_in_a_nested_invocation_does_not_inherit_the_outer_body() -> None:
    # Each node step marks its own middleware as outside the body, so the inner
    # graph's middleware is not read as part of the outer body that runs it.
    # Inheriting the outer True would bind this post-phase call to `x`, whose span
    # is still open at `post`. Killed by dropping the False set where
    # `_step_function_node` begins.
    class _S(State):
        n: int = 0

    provider = _provider()

    async def _x(_s: _S) -> dict[str, Any]:
        return {}

    async def _post_call(state: Any, next_call: Any) -> Any:
        result = await next_call(state)
        await provider.complete([UserMessage(content="from x's middleware")])
        return result

    inner = (
        GraphBuilder(_S)
        .add_node("x", _x, middleware=[_post_call])
        .add_edge("x", END)
        .set_entry("x")
        .compile()
    )
    exporter = InMemorySpanExporter()
    observer = OTelObserver(span_processor=SimpleSpanProcessor(exporter))
    inner.attach_observer(observer)

    async def _outer(_s: _S) -> dict[str, Any]:
        await inner.invoke(_S())
        await inner.drain()
        return {}

    graph = GraphBuilder(_S).add_node("outer", _outer).add_edge("outer", END).set_entry("outer").compile()
    try:
        await graph.invoke(_S())
        await graph.drain()
    finally:
        observer.shutdown()
        await provider.aclose()
    parent = _otel_llm_parent(list(exporter.get_finished_spans()))
    assert parent.name == "openarmature.invocation", (
        f"a call from x's middleware parented under {parent.name!r}"
    )
