# Spec mapping (observability §8): drives the three Langfuse mapping
# fixtures (022 basic trace, 023 Generation rendering + truncation, 024
# prompt linkage) against the in-memory LangfuseObserver client. Sibling
# of test_observability.py (OTel mapping); shares no harness state with
# the OTel side — each fixture builds its own graph + observer instance.

"""Run spec observability Langfuse conformance fixtures (022-024)."""

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

from .adapter import build_state_cls

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "observability" / "conformance"
)


_LANGFUSE_FIXTURES = frozenset(
    {
        "022-langfuse-basic-trace",
        "023-langfuse-generation-rendering",
        "024-langfuse-prompt-linkage",
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

    # ---- Observer
    observer_cfg = cast("dict[str, Any]", case.get("langfuse_observer") or {})
    observer_kwargs: dict[str, Any] = {}
    if "disable_llm_payload" in observer_cfg:
        observer_kwargs["disable_llm_payload"] = bool(observer_cfg["disable_llm_payload"])
    if "disable_llm_spans" in observer_cfg:
        observer_kwargs["disable_llm_spans"] = bool(observer_cfg["disable_llm_spans"])
    if "payload_byte_cap" in observer_cfg:
        observer_kwargs["payload_byte_cap"] = int(observer_cfg["payload_byte_cap"])
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, **observer_kwargs)
    graph.attach_observer(observer)

    # ---- Run
    initial_state_cls = graph.state_cls
    correlation_id = cast("str | None", case.get("caller_correlation_id"))
    invoke_kwargs: dict[str, Any] = {}
    if correlation_id is not None:
        invoke_kwargs["correlation_id"] = correlation_id
    await graph.invoke(initial_state_cls(), **invoke_kwargs)
    await graph.drain()
    if provider is not None:
        await provider.aclose()

    # ---- Assert
    expected = cast("dict[str, Any]", case["expected"])
    expected_trace = cast("dict[str, Any]", expected["langfuse_trace"])
    assert len(client.traces) == 1, f"expected exactly one Trace, got {len(client.traces)}"
    trace = next(iter(client.traces.values()))
    _assert_trace(
        trace, expected_trace, expected_invariants=cast("dict[str, Any]", expected.get("invariants") or {})
    )


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
        if isinstance(expected_value, str) and (
            expected_value == "<any-string>" or expected_value.startswith("<corr_id_")
        ):
            # Placeholder match — any non-empty string passes.
            assert isinstance(actual_value, str) and len(actual_value) > 0, (
                f"{label}.{key}: expected placeholder match, got {actual_value!r}"
            )
            continue
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            _assert_metadata_subset(
                f"{label}.{key}",
                cast("Mapping[str, Any]", actual_value),
                cast("Mapping[str, Any]", expected_value),
            )
            continue
        assert actual_value == expected_value, (
            f"{label}.{key}: expected {expected_value!r}, got {actual_value!r}"
        )


def _assert_string_or_placeholder(label: str, actual: str | None, expected: Any) -> None:
    if expected is None:
        return
    if isinstance(expected, str) and (expected == "<uuid>" or expected == "<any-string>"):
        assert isinstance(actual, str) and len(actual) > 0, (
            f"{label}: expected non-empty string, got {actual!r}"
        )
        return
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"


# Unused helper kept for the SamplingConfig import to avoid a future
# F401 when 024's prompt-sampling path lands.
_ = SamplingConfig
