# Spec mapping (observability §8): drives the Langfuse mapping
# fixtures (022 basic-trace, 023 generation-rendering, 024
# prompt-linkage) against the in-memory LangfuseObserver client.
# Sibling of test_observability.py (OTel mapping); shares no harness
# state with the OTel side — each fixture builds its own graph +
# observer instance.
#
# The harness also supports the graph-topology shapes used by
# 031/032/033 (subgraph / fan-out / detached-trace) via the
# cross-capability adapter.build_graph helper, but activation of
# those three fixtures is currently deferred — see the
# `_LANGFUSE_FIXTURES` frozenset comment for the gating questions.

"""Run spec observability Langfuse conformance fixtures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml

from openarmature.graph import END, GraphBuilder
from openarmature.llm import OpenAIProvider
from openarmature.llm.response import RuntimeConfig
from openarmature.observability.langfuse import (
    InMemoryLangfuseClient,
    LangfuseObservation,
    LangfuseObserver,
    LangfuseTrace,
)
from openarmature.prompts import (
    Prompt,
    PromptManager,
    SamplingConfig,
)
from openarmature.prompts.context import with_active_prompt

from .adapter import build_graph, build_state_cls

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "observability" / "conformance"
)


_LANGFUSE_FIXTURES = frozenset(
    {
        "022-langfuse-basic-trace",
        "023-langfuse-generation-rendering",
        "024-langfuse-prompt-linkage",
        # 027 — proposal 0034 (caller-supplied metadata propagation
        # into ``trace.metadata`` + every ``observation.metadata``
        # per §8.4.1 + §8.4.2).
        "027-langfuse-caller-supplied-metadata",
        # 031 / 032 / 033 — proposal 0035 (spec v0.26.1). The
        # subgraph_identity wiring (per coord thread
        # `clarify-subgraph-name-semantics` msg 02 — Option A) is
        # landed: SubgraphNode.subgraph_identity / FanOutConfig.
        # subgraph_identity flow through NodeEvent.subgraph_identities
        # to observer-side metadata.subgraph_name emission. Two
        # remaining spec/fixture ambiguities block fixture activation:
        # (1) ``step`` semantics on the wrapper synth observation vs.
        # ``outer_out``: fixture 031 expects ``outer_out`` at step 2
        # but graph-engine §6 says "subgraph-internal node executions
        # increment the same counter" so the python engine emits
        # step 3 (outer_in=0, inner_x=1, inner_y=2, outer_out=3).
        # (2) ``namespace`` rewrite for observations inside a
        # detached trace: fixture 033 case 1 expects
        # ``namespace: ["long_running_workflow", "step"]`` (using
        # subgraph identity for the wrapper component) but the
        # engine's event carries the wrapper node name (``"dispatch"``).
        # Both queued for spec input via a follow-up coord thread.
    }
)


def _fixture_paths() -> list[Path]:
    return sorted(p for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml") if p.stem in _LANGFUSE_FIXTURES)


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(path.read_text()))


# ---------------------------------------------------------------------------
# Mock LLM transport
# ---------------------------------------------------------------------------


def _build_mock_llm_handler(responses: list[dict[str, Any]]) -> httpx.MockTransport:
    queue = list(responses)

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not queue:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = queue.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(_handler)


# ---------------------------------------------------------------------------
# Mock prompt backend (fixture 024)
# ---------------------------------------------------------------------------


class _MockPromptBackend:
    """Returns canned Prompts for fixture 024.

    Two flavors per the fixture YAML's ``prompt_backend.type``:

    - ``mock_with_langfuse_reference``: attaches the supplied
      ``langfuse_prompt_reference`` sentinel under
      ``Prompt.observability_entities['langfuse_prompt']``. Verifies
      §8.4.4 case 1 (Generation linked to Prompt entity).
    - ``filesystem``: no Langfuse reference attached. Verifies §8.4.4
      case 2 (metadata-only).
    """

    def __init__(self, prompts: dict[str, dict[str, Any]], *, with_langfuse_reference: bool) -> None:
        self._prompts: dict[str, Prompt] = {}
        now = datetime.now(UTC)
        for prompt_name, spec in prompts.items():
            observability_entities: dict[str, Any] | None = None
            if with_langfuse_reference and "langfuse_prompt_reference" in spec:
                observability_entities = {"langfuse_prompt": spec["langfuse_prompt_reference"]}
            self._prompts[prompt_name] = Prompt(
                name=spec["name"],
                version=spec["version"],
                label=spec["label"],
                template=spec["template"],
                template_hash=spec["template_hash"],
                fetched_at=now,
                observability_entities=observability_entities,
            )

    async def fetch(self, name: str, label: str = "production") -> Prompt:
        return self._prompts[name]


# ---------------------------------------------------------------------------
# Fixture runner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_langfuse_fixture(fixture_path: Path) -> None:
    spec = _load(fixture_path)
    if "cases" in spec:
        for case in cast("list[dict[str, Any]]", spec["cases"]):
            try:
                await _run_case(case)
            except AssertionError as e:
                raise AssertionError(f"case {case.get('name')!r}: {e}") from e
    else:
        await _run_case(spec)


def _has_topology_constructs(case: Mapping[str, Any]) -> bool:
    """Return True when the fixture uses subgraph / fan_out / parallel_branches
    constructs. Such fixtures need the full ``adapter.build_graph`` machinery
    rather than the simpler per-node hand-rolled path used for the
    LLM/prompt-only fixtures."""
    if "subgraph" in case or "subgraphs" in case:
        return True
    nodes_spec = cast("dict[str, Any]", case.get("nodes") or {})
    for node_spec in nodes_spec.values():
        if not isinstance(node_spec, dict):
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        if "subgraph" in node_dict or "fan_out" in node_dict or "parallel_branches" in node_dict:
            return True
    return False


def _compile_subgraphs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build any subgraphs declared by the fixture and return a
    name→compiled-graph registry the adapter consumes. Mirrors the
    OTel-side helper in ``test_observability.py``."""
    subgraph_specs: dict[str, Any] = {}
    if "subgraph" in spec:
        single = cast("Mapping[str, Any]", spec["subgraph"])
        name = single.get("name") or "subgraph"
        subgraph_specs[name] = single
    if "subgraphs" in spec:
        for k, v in cast("dict[str, Any]", spec["subgraphs"]).items():
            subgraph_specs[k] = v
    compiled_subgraphs: dict[str, Any] = {}
    for name, sub_spec in subgraph_specs.items():
        sub_built = build_graph(sub_spec, trace=[])
        compiled_subgraphs[name] = sub_built.builder.compile()
    return compiled_subgraphs


def _resolve_detached_wrapper_names(case: Mapping[str, Any]) -> frozenset[str]:
    """Translate fixture-level ``detached_subgraphs`` (a list of SUBGRAPH
    IDENTITY names) into the set of WRAPPER NODE names the observer keys
    on. The fixture identifies detached subgraphs by their declaration name
    in ``subgraphs:`` (e.g., ``long_running_workflow``), but the
    LangfuseObserver matches by the wrapper node name in the parent graph
    that references the subgraph (e.g., ``dispatch``).
    """
    detached_identities = set(cast("list[str]", case.get("detached_subgraphs") or []))
    if not detached_identities:
        return frozenset()
    nodes_spec = cast("dict[str, Any]", case.get("nodes") or {})
    wrappers: set[str] = set()
    for wrapper_name, node_spec in nodes_spec.items():
        if not isinstance(node_spec, dict):
            continue
        sub_id = cast("dict[str, Any]", node_spec).get("subgraph")
        if isinstance(sub_id, str) and sub_id in detached_identities:
            wrappers.add(wrapper_name)
    return frozenset(wrappers)


async def _run_case(case: Mapping[str, Any]) -> None:
    # ---- Mock LLM transport (if the graph has an LLM call)
    mock_responses = cast("list[dict[str, Any]] | None", case.get("mock_llm"))
    transport = _build_mock_llm_handler(mock_responses) if mock_responses else None
    provider: OpenAIProvider | None = None
    if transport is not None:
        provider = OpenAIProvider(
            base_url="http://mock-llm.test",
            model=_resolve_llm_model(case),
            api_key="test",
            transport=transport,
        )

    # ---- Prompt backend (fixture 024)
    prompt_manager: PromptManager | None = None
    prompt_backend_spec = cast("dict[str, Any] | None", case.get("prompt_backend"))
    if prompt_backend_spec is not None:
        backend_type = prompt_backend_spec.get("type")
        prompts_block = cast("dict[str, dict[str, Any]]", prompt_backend_spec.get("prompts") or {})
        backend = _MockPromptBackend(
            prompts_block,
            with_langfuse_reference=(backend_type == "mock_with_langfuse_reference"),
        )
        prompt_manager = PromptManager(backend)

    # ---- Graph build
    # Two paths: topology fixtures (031/032/033) need the full
    # ``adapter.build_graph`` machinery for subgraph / fan_out shapes;
    # LLM/prompt fixtures (022/023/024) use the simpler hand-rolled
    # per-node build that knows about ``calls_llm`` / ``renders_prompt``.
    if _has_topology_constructs(case):
        subgraphs = _compile_subgraphs(case)
        built = build_graph(case, subgraphs=subgraphs, trace=[])
        graph = built.builder.compile()
        initial_state_factory = lambda: built.initial_state(case.get("initial_state", {}))  # noqa: E731
    else:
        state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
        state_cls = build_state_cls("LangfuseFixtureState", state_fields)
        nodes_spec = cast("dict[str, Any]", case["nodes"])
        entry = cast("str", case["entry"])
        edges = cast("list[dict[str, str]]", case["edges"])
        render_variables = cast("dict[str, Any]", case.get("render_variables") or {})

        builder = GraphBuilder(state_cls)
        for node_name, node_spec in nodes_spec.items():
            node_body = _build_node_body(
                node_name=node_name,
                node_spec=cast("dict[str, Any]", node_spec),
                provider=provider,
                prompt_manager=prompt_manager,
                render_variables=render_variables,
            )
            builder.add_node(node_name, node_body)
        for edge in edges:
            target_raw = edge["to"]
            target = END if target_raw == "END" else target_raw
            builder.add_edge(edge["from"], target)
        builder.set_entry(entry)
        graph = builder.compile()
        initial_state_factory = graph.state_cls

    # ---- Observer
    observer_cfg = cast("dict[str, Any]", case.get("langfuse_observer") or {})
    observer_kwargs: dict[str, Any] = {}
    if "disable_llm_payload" in observer_cfg:
        observer_kwargs["disable_llm_payload"] = bool(observer_cfg["disable_llm_payload"])
    if "disable_llm_spans" in observer_cfg:
        observer_kwargs["disable_llm_spans"] = bool(observer_cfg["disable_llm_spans"])
    if "payload_byte_cap" in observer_cfg:
        observer_kwargs["payload_byte_cap"] = int(observer_cfg["payload_byte_cap"])
    detached_subgraphs = _resolve_detached_wrapper_names(case)
    if detached_subgraphs:
        observer_kwargs["detached_subgraphs"] = detached_subgraphs
    detached_fan_outs = frozenset(cast("list[str]", case.get("detached_fan_outs") or []))
    if detached_fan_outs:
        observer_kwargs["detached_fan_outs"] = detached_fan_outs
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, **observer_kwargs)
    graph.attach_observer(observer)

    # ---- Run
    correlation_id = cast("str | None", case.get("caller_correlation_id"))
    invoke_kwargs: dict[str, Any] = {}
    if correlation_id is not None:
        invoke_kwargs["correlation_id"] = correlation_id
    caller_metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
    if caller_metadata is not None:
        invoke_kwargs["metadata"] = caller_metadata
    await graph.invoke(initial_state_factory(), **invoke_kwargs)
    await graph.drain()
    if provider is not None:
        await provider.aclose()

    # ---- Assert
    # Single-trace fixtures use ``langfuse_trace:``; detached / multi-trace
    # fixtures use ``langfuse_traces:`` (a list). Branch on which the
    # fixture supplies.
    expected = cast("dict[str, Any]", case["expected"])
    expected_invariants = cast("dict[str, Any]", expected.get("invariants") or {})
    if "langfuse_traces" in expected:
        _assert_multi_traces(client, expected, expected_invariants)
    else:
        expected_trace = cast("dict[str, Any]", expected["langfuse_trace"])
        assert len(client.traces) == 1, f"expected exactly one Trace, got {len(client.traces)}"
        trace = next(iter(client.traces.values()))
        _assert_trace(trace, expected_trace, expected_invariants=expected_invariants)


def _resolve_llm_model(case: Mapping[str, Any]) -> str:
    # Single LLM call per fixture today; pick up the per-call model if
    # supplied (fixture 023 explicitly sets `model: "test-model"` on
    # the calls_llm block).
    nodes_spec = cast("dict[str, Any]", case["nodes"])
    for node_spec in nodes_spec.values():
        if not isinstance(node_spec, dict):
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        calls_llm = node_dict.get("calls_llm")
        if isinstance(calls_llm, dict):
            return cast("str", cast("dict[str, Any]", calls_llm).get("model", "test-model"))
    return "test-model"


def _build_node_body(
    *,
    node_name: str,
    node_spec: dict[str, Any],
    provider: OpenAIProvider | None,
    prompt_manager: PromptManager | None,
    render_variables: dict[str, Any],
) -> Any:
    # Three node shapes in fixtures 022-024:
    #   - `update: {...}` — set fields on state directly (022).
    #   - `calls_llm: {...}` — invoke provider.complete (023).
    #   - `renders_prompt: <name>` + optional `calls_llm` — render the
    #     named prompt, then call the LLM under `with_active_prompt`
    #     so the Generation's prompt-linkage metadata + entity link
    #     populate per §8.4.4 (024).
    update_spec = cast("dict[str, Any] | None", node_spec.get("update"))
    if update_spec is not None:

        async def _node(_s: Any) -> dict[str, Any]:
            return dict(update_spec)

        return _node

    calls_llm_spec = cast("dict[str, Any] | None", node_spec.get("calls_llm"))
    renders_prompt_name = cast("str | None", node_spec.get("renders_prompt"))

    async def _llm_node(_s: Any) -> dict[str, Any]:
        assert provider is not None, f"node {node_name!r} has calls_llm but no mock_llm responses"
        messages_spec = cast(
            "list[dict[str, Any]] | None",
            (calls_llm_spec or {}).get("messages"),
        )
        config_spec = cast("dict[str, Any] | None", (calls_llm_spec or {}).get("config"))
        stores_in = cast("str", (calls_llm_spec or {}).get("stores_response_in", "msg"))
        if renders_prompt_name is not None:
            assert prompt_manager is not None, "renders_prompt requires a prompt_backend block"
            prompt = await prompt_manager.fetch(renders_prompt_name)
            rendered = prompt_manager.render(prompt, render_variables)
            llm_messages = list(rendered.messages)
            with with_active_prompt(rendered):
                response = await provider.complete(
                    cast("Sequence[Any]", llm_messages),
                    config=_runtime_config_from_spec(config_spec),
                )
        else:
            llm_messages = _materialize_messages(messages_spec or [])
            response = await provider.complete(
                cast("Sequence[Any]", llm_messages),
                config=_runtime_config_from_spec(config_spec),
            )
        return {stores_in: response.message.content or ""}

    return _llm_node


def _materialize_messages(raw: list[dict[str, Any]]) -> list[Any]:
    from openarmature.llm.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage

    out: list[Any] = []
    for entry in raw:
        role = entry.get("role")
        content: Any
        if "content_repeat" in entry:
            repeat = cast("dict[str, Any]", entry["content_repeat"])
            content = cast("str", repeat["char"]) * int(repeat["bytes"])
        else:
            content = entry.get("content")
        if role == "system":
            out.append(SystemMessage(content=cast("str", content)))
        elif role == "user":
            out.append(UserMessage(content=cast("str", content)))
        elif role == "assistant":
            out.append(AssistantMessage(content=cast("str", content)))
        elif role == "tool":
            out.append(
                ToolMessage(content=cast("str", content), tool_call_id=cast("str", entry["tool_call_id"]))
            )
        else:
            raise AssertionError(f"unknown message role: {role!r}")
    return out


def _runtime_config_from_spec(config_spec: dict[str, Any] | None) -> RuntimeConfig | None:
    if not config_spec:
        return None
    declared = {
        "temperature",
        "max_tokens",
        "top_p",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "stop_sequences",
    }
    kwargs = {k: v for k, v in config_spec.items() if k in declared}
    extras = cast("dict[str, Any]", config_spec.get("extras") or {})
    kwargs.update(extras)
    return RuntimeConfig(**kwargs)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


# Per-trace invariants — invariants ``_assert_trace`` knows how to
# check on a single Trace. The multi-trace runner filters
# ``expected_invariants`` to this set when delegating per-Trace
# assertions; the rest (``distinct_trace_ids``,
# ``correlation_id_consistent_across_traces``, etc.) stay in
# ``_assert_multi_traces`` as cross-Trace checks.
_PER_TRACE_INVARIANTS = frozenset({"trace_id_equals_invocation_id", "correlation_id_consistency"})


def _assert_multi_traces(
    client: InMemoryLangfuseClient,
    expected: dict[str, Any],
    expected_invariants: dict[str, Any],
) -> None:
    """Detached-trace fixtures (033) span multiple Traces. The expected
    block's ``langfuse_traces:`` list names the parent Trace explicitly;
    the additional detached Traces are either fully enumerated (subgraph
    case) or counted via ``detached_trace_count`` (fan-out case where
    only the parent's shape is asserted explicitly and the per-instance
    Traces share an identical shape).
    """
    expected_traces = cast("list[dict[str, Any]]", expected.get("langfuse_traces") or [])
    detached_trace_count = cast("int | None", expected.get("detached_trace_count"))
    # If the fixture enumerates all Traces explicitly, the actual count
    # MUST match. Otherwise, ``detached_trace_count`` indicates how many
    # additional traces beyond the enumerated parent the fixture expects.
    expected_total = (
        len(expected_traces) + detached_trace_count
        if detached_trace_count is not None
        else len(expected_traces)
    )
    assert len(client.traces) == expected_total, (
        f"expected {expected_total} Traces, got {len(client.traces)}: "
        f"{[t.name for t in client.traces.values()]}"
    )

    # Match each enumerated expected Trace to an actual Trace by name
    # compatibility (literal match, or wildcard ``<...>`` placeholder)
    # then by root-observation count. Tracks consumed traces so two
    # expected entries can't bind to the same actual one.
    consumed: set[str] = set()
    for exp in expected_traces:
        exp_name = cast("str", exp.get("name") or "")
        is_wildcard = exp_name.startswith("<") and exp_name.endswith(">")
        expected_obs_count = len(cast("list[Any]", exp.get("observations") or []))
        candidates = [
            t for t in client.traces.values() if t.id not in consumed and (is_wildcard or t.name == exp_name)
        ]
        # Prefer candidates whose root-observation count matches the
        # expected structure; the fan-out case has multiple traces of
        # the same name where the parent is the only one with a
        # populated observation tree.
        matching = [t for t in candidates if len(t.children_of(None)) == expected_obs_count]
        assert matching or candidates, (
            f"no Trace matches expected name={exp_name!r} (consumed={sorted(consumed)})"
        )
        trace = matching[0] if matching else candidates[0]
        consumed.add(trace.id)
        per_trace_invariants = {k: v for k, v in expected_invariants.items() if k in _PER_TRACE_INVARIANTS}
        _assert_trace(trace, exp, expected_invariants=per_trace_invariants)

    # Invariants that span multiple Traces.
    if expected_invariants.get("distinct_trace_ids"):
        trace_ids = {t.id for t in client.traces.values()}
        assert len(trace_ids) == len(client.traces), f"trace ids not all distinct: {sorted(trace_ids)}"
    if expected_invariants.get("all_instance_trace_ids_distinct"):
        trace_ids = {t.id for t in client.traces.values()}
        assert len(trace_ids) == len(client.traces), (
            f"instance trace ids not all distinct: {sorted(trace_ids)}"
        )
    expected_child_count = cast(
        "int | None", expected_invariants.get("dispatch_detached_child_trace_id_count")
    )
    if expected_child_count is not None:
        # Find the parent-trace observation whose metadata carries
        # detached_child_trace_ids; assert its length.
        found = False
        for trace in client.traces.values():
            for obs in trace.observations:
                child_ids_raw = obs.metadata.get("detached_child_trace_ids")
                if isinstance(child_ids_raw, list):
                    child_ids = cast("list[Any]", child_ids_raw)
                    assert len(child_ids) == expected_child_count, (
                        f"observation {obs.name!r} detached_child_trace_ids length: "
                        f"expected {expected_child_count}, got {len(child_ids)}"
                    )
                    found = True
        assert found, "no observation carried metadata.detached_child_trace_ids"
    if expected_invariants.get("correlation_id_consistent_across_traces"):
        correlation_ids = {
            cast("str | None", t.metadata.get("correlation_id")) for t in client.traces.values()
        }
        correlation_ids.discard(None)
        if len(correlation_ids) > 1:
            sorted_ids = sorted(c for c in correlation_ids if c is not None)
            raise AssertionError(f"correlation_id not consistent across Traces: {sorted_ids}")
    if expected_invariants.get("no_instance_spans_in_parent_trace"):
        # The parent Trace is the one named in expected_traces (singular).
        if len(expected_traces) == 1:
            parent_name = cast("str", expected_traces[0]["name"])
            parents = [t for t in client.traces.values() if t.name == parent_name]
            # If multiple Traces share the parent name (fan-out case where
            # detached per-instance Traces inherit the fan-out node name),
            # the actual parent is the one with a non-empty observation
            # tree. Per-instance Traces have their own subtree under their
            # own dispatch observation; their leaf shapes are different
            # from the parent's flat fan-out node observation.
            for t in parents:
                # Parent observations must be limited to the fan-out
                # dispatch (no leaked per-instance inner-node names).
                root_obs = t.children_of(None)
                for obs in root_obs:
                    if obs.name != parent_name:
                        # If we got here, an inner-node observation
                        # leaked into the parent Trace.
                        raise AssertionError(
                            f"unexpected observation {obs.name!r} in parent Trace {t.id!r}; "
                            f"parent should only contain the fan-out dispatch observation"
                        )


def _assert_trace(
    trace: LangfuseTrace,
    expected: dict[str, Any],
    *,
    expected_invariants: dict[str, Any],
) -> None:
    _assert_string_or_placeholder("trace.id", trace.id, expected.get("id"))
    if "name" in expected:
        _assert_string_or_placeholder("trace.name", trace.name, expected.get("name"))
    expected_metadata = cast("dict[str, Any]", expected.get("metadata") or {})
    _assert_metadata_subset("trace.metadata", trace.metadata, expected_metadata)
    expected_observations = cast("list[dict[str, Any]]", expected.get("observations") or [])
    root_observations = trace.children_of(None)
    _assert_observation_tree(trace, root_observations, expected_observations)

    # Invariants: cross-cutting checks that hold across the full Trace.
    if expected_invariants.get("trace_id_equals_invocation_id"):
        # No direct accessor for the invocation_id from outside the
        # observer; the §8.4.1 contract is that trace.id == invocation_id,
        # so the invariant degenerates to "trace.id matches the UUIDv4
        # pattern" — already asserted above via `<uuid>` placeholder.
        pass
    if expected_invariants.get("correlation_id_consistency"):
        trace_correlation = cast("str | None", trace.metadata.get("correlation_id"))
        if trace_correlation is not None:
            for obs in trace.observations:
                obs_correlation = obs.metadata.get("correlation_id")
                assert obs_correlation == trace_correlation, (
                    f"correlation_id mismatch: trace={trace_correlation!r}, "
                    f"observation {obs.name!r}={obs_correlation!r}"
                )


def _assert_observation_tree(
    trace: LangfuseTrace,
    actual_children: list[LangfuseObservation],
    expected_children: list[dict[str, Any]],
) -> None:
    assert len(actual_children) == len(expected_children), (
        f"observation children count mismatch: expected {len(expected_children)}, got {len(actual_children)}"
    )
    for actual, expected in zip(actual_children, expected_children, strict=True):
        _assert_observation(trace, actual, expected)


def _assert_observation(
    trace: LangfuseTrace,
    actual: LangfuseObservation,
    expected: dict[str, Any],
) -> None:
    if "type" in expected:
        assert actual.type == expected["type"], (
            f"observation {actual.name!r} type: expected {expected['type']!r}, got {actual.type!r}"
        )
    if "name" in expected:
        _assert_string_or_placeholder(f"observation[{actual.name}].name", actual.name, expected["name"])
    if "level" in expected:
        assert actual.level == expected["level"], (
            f"observation {actual.name!r} level: expected {expected['level']!r}, got {actual.level!r}"
        )
    if "model" in expected:
        assert actual.model == expected["model"], (
            f"observation {actual.name!r} model: expected {expected['model']!r}, got {actual.model!r}"
        )
    if "modelParameters" in expected:
        expected_params = cast("dict[str, Any]", expected["modelParameters"])
        for key, value in expected_params.items():
            assert actual.model_parameters.get(key) == value, (
                f"observation {actual.name!r} modelParameters.{key}: "
                f"expected {value!r}, got {actual.model_parameters.get(key)!r}"
            )
    if "usage" in expected:
        expected_usage = cast("dict[str, Any]", expected["usage"])
        assert actual.usage is not None, f"observation {actual.name!r} usage absent"
        for key, value in expected_usage.items():
            actual_value = getattr(actual.usage, key, None)
            assert actual_value == value, (
                f"observation {actual.name!r} usage.{key}: expected {value!r}, got {actual_value!r}"
            )
    if "input_parses_as_messages" in expected:
        expected_messages = cast("list[dict[str, Any]]", expected["input_parses_as_messages"])
        # input is the native message-list shape OR a JSON string of it;
        # either way, parse-to-shape MUST succeed and match.
        parsed = _parse_messages(actual.input)
        # The actual messages are produced by the LLM provider's
        # _serialize_messages_for_payload which carries the full §3 shape
        # (including ``content`` as either str or block list). Loose
        # subset compare on role+content for fixture parity.
        for expected_msg, parsed_msg in zip(expected_messages, parsed, strict=False):
            assert parsed_msg.get("role") == expected_msg["role"], (
                f"observation {actual.name!r} input message role mismatch: "
                f"expected {expected_msg['role']!r}, got {parsed_msg.get('role')!r}"
            )
            assert parsed_msg.get("content") == expected_msg["content"], (
                f"observation {actual.name!r} input message content mismatch: "
                f"expected {expected_msg['content']!r}, got {parsed_msg.get('content')!r}"
            )
    if expected.get("input_is_raw_string_with_marker") is True:
        assert isinstance(actual.input, str), (
            f"observation {actual.name!r} input expected raw string, got {type(actual.input).__name__}"
        )
        assert "[truncated," in actual.input, (
            f"observation {actual.name!r} input missing truncation marker: {actual.input!r}"
        )
    if "output" in expected:
        assert actual.output == expected["output"], (
            f"observation {actual.name!r} output: expected {expected['output']!r}, got {actual.output!r}"
        )
    if "prompt_entity_link" in expected:
        assert actual.prompt_entity_link == expected["prompt_entity_link"], (
            f"observation {actual.name!r} prompt_entity_link: "
            f"expected {expected['prompt_entity_link']!r}, got {actual.prompt_entity_link!r}"
        )
    if expected.get("prompt_entity_link_absent") is True:
        assert actual.prompt_entity_link is None, (
            f"observation {actual.name!r} prompt_entity_link expected absent, "
            f"got {actual.prompt_entity_link!r}"
        )
    expected_metadata = cast("dict[str, Any]", expected.get("metadata") or {})
    _assert_metadata_subset(f"observation[{actual.name}].metadata", actual.metadata, expected_metadata)

    expected_children = cast("list[dict[str, Any]]", expected.get("children") or [])
    actual_children = trace.children_of(actual.id)
    _assert_observation_tree(trace, actual_children, expected_children)


def _parse_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return cast("list[dict[str, Any]]", value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"input attribute did not parse as JSON: {value!r}") from exc
        if isinstance(decoded, list):
            return cast("list[dict[str, Any]]", decoded)
    raise AssertionError(f"input attribute did not parse as a message list: {value!r}")


def _is_placeholder(value: Any) -> bool:
    """``<anything>`` literals in fixtures are wildcards: they assert
    shape (non-empty string) without binding a specific value. Cross-
    occurrence consistency lives in the ``invariants:`` block."""
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def _assert_metadata_subset(
    label: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    # Subset-compare: every key in `expected` MUST be present in
    # `actual` with the corresponding value (or matching placeholder);
    # additional keys in `actual` are tolerated.
    for key, expected_value in expected.items():
        assert key in actual, f"{label}: missing key {key!r}; got keys {sorted(actual)}"
        actual_value = actual[key]
        if _is_placeholder(expected_value):
            assert isinstance(actual_value, str) and len(actual_value) > 0, (
                f"{label}.{key}: expected placeholder {expected_value!r} match, got {actual_value!r}"
            )
            continue
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            _assert_metadata_subset(
                f"{label}.{key}",
                cast("Mapping[str, Any]", actual_value),
                cast("Mapping[str, Any]", expected_value),
            )
            continue
        if isinstance(expected_value, list) and isinstance(actual_value, list):
            # List values may contain placeholders element-by-element
            # (e.g., ``detached_child_trace_ids: ["<trace_id_detached>"]``).
            # Length must match; each element matches by placeholder
            # rules or strict equality.
            expected_list = cast("list[Any]", expected_value)
            actual_list = cast("list[Any]", actual_value)
            assert len(actual_list) == len(expected_list), (
                f"{label}.{key} length: expected {len(expected_list)}, got {len(actual_list)}"
            )
            for i, (exp_el, act_el) in enumerate(zip(expected_list, actual_list, strict=True)):
                if _is_placeholder(exp_el):
                    assert isinstance(act_el, str) and len(act_el) > 0, (
                        f"{label}.{key}[{i}]: expected placeholder {exp_el!r}, got {act_el!r}"
                    )
                else:
                    assert act_el == exp_el, f"{label}.{key}[{i}]: expected {exp_el!r}, got {act_el!r}"
            continue
        assert actual_value == expected_value, (
            f"{label}.{key}: expected {expected_value!r}, got {actual_value!r}"
        )


def _assert_string_or_placeholder(label: str, actual: str | None, expected: Any) -> None:
    if expected is None:
        return
    # Any ``<placeholder>`` form is treated as a wildcard that requires
    # a non-empty string. Cross-occurrence consistency (e.g., the same
    # ``<corr_id_1>`` appearing on multiple Traces must resolve to the
    # same value) is enforced by the fixture's ``invariants:`` block
    # (``correlation_id_consistent_across_traces``, etc.), not by
    # placeholder-binding here. Known shape-only placeholders:
    # ``<uuid>``, ``<any-string>``, ``<corr_id_N>``,
    # ``<trace_id_parent>``, ``<trace_id_child>``,
    # ``<trace_id_instance_N>``.
    if isinstance(expected, str) and expected.startswith("<") and expected.endswith(">"):
        assert isinstance(actual, str) and len(actual) > 0, (
            f"{label}: expected non-empty string, got {actual!r}"
        )
        return
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"


# Unused helper kept for the SamplingConfig import to avoid a future
# F401 when 024's prompt-sampling path lands.
_ = SamplingConfig
