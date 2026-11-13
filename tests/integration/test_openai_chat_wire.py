"""Integration tests for the OpenAI chat wire mapping's structured-output paths.

Gated by the ``OPENAI_API_KEY`` env var: the tests run only when it is set.
Skipped in CI and local runs that don't have it in scope; runs end-to-end
against the live ``/v1/chat/completions`` endpoint when invoked with the key set.

``base_url`` is read from ``OPENAI_BASE_URL`` (default ``https://api.openai.com``,
origin only), and the bound model from ``OPENAI_CHAT_MODEL`` (default
``gpt-4o-mini``). The model must support strict structured outputs and accept
``max_tokens`` and ``temperature``, which rules out the reasoning models.

Each call runs inside a one-node graph with an observer collecting the LLM
events, since the provider dispatches them only inside an invocation.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from typing import Any

import pytest

from openarmature.graph import (
    END,
    GraphBuilder,
    LlmCompletionEvent,
    LlmFailedEvent,
    LlmRetryAttemptEvent,
    State,
)
from openarmature.llm import (
    OpenAIProvider,
    RuntimeConfig,
    StructuredOutputInvalid,
    UserMessage,
)
from openarmature.observability import STRUCTURED_OUTPUT_SCHEMA_NOT_HONORED, event_name_of

_API_KEY = os.environ.get("OPENAI_API_KEY")
_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

requires_openai = pytest.mark.skipif(
    not _API_KEY, reason="Requires OPENAI_API_KEY (live OpenAI-compatible key)"
)

_QUESTION = "Who was the first person to walk on the moon, and in what year?"

_STRICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["name", "year"],
    "additionalProperties": False,
}

# Not strict: no additionalProperties: false, and "year" is not required.
_LOOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["name"],
}


class _CallState(State):
    done: bool = False


class _Collector:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def __call__(self, event: Any) -> None:
        self.events.append(event)

    def of(self, kind: type[Any]) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]


async def _complete_in_graph(
    *,
    schema: dict[str, Any],
    fallback: bool = False,
    config: RuntimeConfig | None = None,
) -> tuple[Any, StructuredOutputInvalid | None, _Collector]:
    provider = OpenAIProvider(
        base_url=_BASE_URL,
        model=_CHAT_MODEL,
        api_key=str(_API_KEY),
        force_prompt_augmentation_fallback=fallback,
    )
    outcome: dict[str, Any] = {"response": None, "error": None}

    async def call(_s: _CallState) -> Mapping[str, Any]:
        try:
            outcome["response"] = await provider.complete(
                [UserMessage(content=_QUESTION)], config=config, response_schema=schema
            )
        except StructuredOutputInvalid as exc:
            outcome["error"] = exc
        return {"done": True}

    graph = GraphBuilder(_CallState).add_node("call", call).add_edge("call", END).set_entry("call").compile()
    collector = _Collector()
    try:
        await graph.invoke(_CallState(), observers=[collector])
    finally:
        await graph.drain()
        await provider.aclose()
    return outcome["response"], outcome["error"], collector


def _assert_moon_answer(parsed: dict[str, Any]) -> None:
    assert "armstrong" in str(parsed["name"]).lower()


def _is_json_object(content: str) -> bool:
    try:
        return isinstance(json.loads(content), dict)
    except ValueError:
        return False


@pytest.mark.integration
@requires_openai
@pytest.mark.parametrize(
    ("schema", "path"),
    [
        pytest.param(_STRICT_SCHEMA, "native", id="strict"),
        pytest.param(_LOOSE_SCHEMA, "native_unenforced", id="not-strict"),
    ],
)
async def test_a_native_call_reports_the_path_its_strict_flag_sets(schema: dict[str, Any], path: str) -> None:
    """A schema-constrained call parses and validates, and its attempt and
    completion events report the path the request took."""
    response, error, collector = await _complete_in_graph(schema=schema)

    assert error is None, f"call failed: {error!r}"
    _assert_moon_answer(response.parsed)
    [attempt] = collector.of(LlmRetryAttemptEvent)
    [completion] = collector.of(LlmCompletionEvent)
    assert attempt.structured_output_path == path
    assert completion.structured_output_path == path


@pytest.mark.integration
@requires_openai
async def test_the_fallback_reports_prompt_augmentation_and_the_directive_it_sent() -> None:
    """With the fallback forced, the events report prompt_augmentation and
    describe the messages as sent, the schema directive included."""
    response, error, collector = await _complete_in_graph(
        schema=_STRICT_SCHEMA, fallback=True, config=RuntimeConfig(temperature=0)
    )

    assert error is None, f"call failed: {error!r}"
    _assert_moon_answer(response.parsed)
    [attempt] = collector.of(LlmRetryAttemptEvent)
    [completion] = collector.of(LlmCompletionEvent)
    for event in (attempt, completion):
        assert event.structured_output_path == "prompt_augmentation"
        assert event.input_messages[0]["role"] == "system"
        assert "JSON Schema" in str(event.input_messages[0]["content"])


@pytest.mark.integration
@requires_openai
async def test_a_truncated_native_reply_logs_no_schema_not_honored_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A strict call cut off by max_tokens fails structured_output_invalid with
    finish_reason length, reports the native path, and logs no
    schema_not_honored record."""
    with caplog.at_level(logging.DEBUG):
        _response, error, collector = await _complete_in_graph(
            schema=_STRICT_SCHEMA, config=RuntimeConfig(max_tokens=5)
        )

    # Every other arm of the diagnostic's condition holds here: native path, no
    # stop sequence, content that is not a JSON object. So only the
    # finish_reason arm keeps the record off. A complete object would keep it
    # off by itself, which is why the content is checked.
    assert error is not None, "expected the truncated reply to fail validation"
    assert error.finish_reason == "length"
    assert not _is_json_object(error.output_content), error.output_content
    [failed] = collector.of(LlmFailedEvent)
    assert failed.error_category == "structured_output_invalid"
    assert failed.structured_output_path == "native"
    assert [r for r in caplog.records if event_name_of(r) == STRUCTURED_OUTPUT_SCHEMA_NOT_HONORED] == []
