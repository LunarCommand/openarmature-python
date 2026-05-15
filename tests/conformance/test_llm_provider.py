"""Run every spec llm-provider conformance fixture against OpenAIProvider.

The fixtures (``spec/llm-provider/conformance/``) describe a provider's
behavior in terms of OpenAI Chat Completions wire-format mock
responses + expected ``Provider.complete()`` / ``Provider.ready()``
outcomes. The harness drives the real :class:`OpenAIProvider` via
``httpx.MockTransport`` so the wire-mapping path (spec §8) is
exercised end-to-end — fixture 005 explicitly tests that mapping, so
mocking at the Provider boundary would skip what we want to verify.

Fixture shapes the harness handles:

- Top-level fixture with ``mock_provider`` + ``calls`` (e.g. 001, 002).
- Cases-shape with shared/per-case ``mock_provider`` and a single
  ``call:`` (e.g. 004 error-categories).
- ``ready`` operation calls (fixture 007).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml
from pydantic import ValidationError

from openarmature.llm import (
    AssistantMessage,
    LlmProviderError,
    Message,
    OpenAIProvider,
    ProviderInvalidRequest,
    ProviderRateLimit,
    Response,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "llm-provider" / "conformance"
)


# Fixtures whose implementation lands in a later PR of the 5-proposal batch.
# Skip-marked here so a green test run at this commit means "everything we
# claim to implement passes." Each subsequent PR drops its own rows as it
# lands the underlying support.
_DEFERRED_FIXTURES: dict[str, str] = {
    # proposal 0015 — multimodal images (PR-2 of the batch)
    "009-content-blocks-text-only-equivalence": "0015 multimodal images (PR-2)",
    "010-content-blocks-image-url": "0015 multimodal images (PR-2)",
    "011-content-blocks-image-inline-base64": "0015 multimodal images (PR-2)",
    "012-content-blocks-image-detail-hint": "0015 multimodal images (PR-2)",
    "013-content-blocks-mixed-order-preserved": "0015 multimodal images (PR-2)",
    "014-content-blocks-validation-empty-sequence": "0015 multimodal images (PR-2)",
    "015-content-blocks-validation-empty-text-block": "0015 multimodal images (PR-2)",
    "016-content-blocks-unsupported-by-model": "0015 multimodal images (PR-2)",
    "017-content-blocks-system-message-text-only": "0015 multimodal images (PR-2)",
    "018-content-blocks-image-source-missing": "0015 multimodal images (PR-2)",
    "019-content-blocks-invalid-detail-value": "0015 multimodal images (PR-2)",
    "020-content-blocks-inline-image-missing-media-type": "0015 multimodal images (PR-2)",
    # proposal 0016 — structured output (this PR; wired up later in the
    # commit sequence). These rows are removed in the commit that drives
    # the structured-output fixtures.
    "021-structured-output-success": "0016 structured output (this PR; not yet wired)",
    "022-structured-output-parse-failure": "0016 structured output (this PR; not yet wired)",
    "023-structured-output-validation-failure": "0016 structured output (this PR; not yet wired)",
    "024-structured-output-non-transient": "0016 structured output (this PR; not yet wired)",
    "025-structured-output-with-tool-calls": "0016 structured output (this PR; not yet wired)",
    "026-structured-output-openai-wire-mapping-native": "0016 structured output (this PR; not yet wired)",
    "027-structured-output-openai-wire-mapping-fallback": "0016 structured output (this PR; not yet wired)",
    "028-structured-output-no-schema-regression": "0016 structured output (this PR; not yet wired)",
}


def _fixture_paths() -> list[Path]:
    return sorted(CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


# ---------------------------------------------------------------------------
# Mock-transport machinery
# ---------------------------------------------------------------------------


def _build_handler(
    responses: list[Mapping[str, Any]],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Build an httpx.MockTransport that hands back the configured
    responses in order, plus a list that captures each outbound
    request for round-trip assertions."""
    captured: list[httpx.Request] = []
    iterator = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        try:
            spec = next(iterator)
        except StopIteration as exc:
            raise AssertionError(f"mock provider exhausted at request {len(captured)}") from exc
        if spec.get("connection_failure"):
            # Simulate connect error before any HTTP response.
            raise httpx.ConnectError("simulated connection failure")
        status = int(spec.get("status", 200))
        body = spec.get("body")
        headers_raw = cast("Mapping[str, Any]", spec.get("headers") or {})
        headers = {str(k): str(v) for k, v in headers_raw.items()}
        if body is None:
            return httpx.Response(status, headers=headers)
        return httpx.Response(
            status,
            headers=headers,
            content=json.dumps(body).encode("utf-8"),
            extensions={},
        )

    return httpx.MockTransport(handler), captured


def _build_provider(
    mock_provider_cfg: Mapping[str, Any],
    *,
    model: str = "test-model",
) -> tuple[OpenAIProvider, list[httpx.Request]]:
    # Some fixtures (007 ready-check) use ``health_endpoint`` instead
    # of ``responses`` — that's the same shape, just one entry.
    if "health_endpoint" in mock_provider_cfg:
        responses: list[Mapping[str, Any]] = [cast("Mapping[str, Any]", mock_provider_cfg["health_endpoint"])]
    else:
        responses = cast(
            "list[Mapping[str, Any]]",
            mock_provider_cfg.get("responses") or [],
        )
    transport, captured = _build_handler(responses)
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model=model,
        api_key="test-key",
        transport=transport,
    )
    return provider, captured


# ---------------------------------------------------------------------------
# Message + tool translation: fixture YAML -> typed objects
# ---------------------------------------------------------------------------


def _build_message(raw: Mapping[str, Any]) -> Message:
    role = raw["role"]
    if role == "system":
        return SystemMessage(content=cast("str", raw["content"]))
    if role == "user":
        return UserMessage(content=cast("str", raw["content"]))
    if role == "assistant":
        tool_calls_raw = raw.get("tool_calls")
        tool_calls: list[ToolCall] | None = None
        if tool_calls_raw:
            tool_calls = [
                ToolCall(
                    id=cast("str", tc["id"]),
                    name=cast("str", tc["name"]),
                    arguments=tc.get("arguments"),
                )
                for tc in cast("list[Mapping[str, Any]]", tool_calls_raw)
            ]
        return AssistantMessage(
            content=cast("str", raw.get("content") or ""),
            tool_calls=tool_calls,
        )
    if role == "tool":
        return ToolMessage(
            content=cast("str", raw["content"]),
            tool_call_id=cast("str", raw["tool_call_id"]),
        )
    raise AssertionError(f"unknown message role in fixture: {role!r}")


def _build_tools(raw_list: list[Mapping[str, Any]] | None) -> list[Tool] | None:
    if not raw_list:
        return None
    return [
        Tool(
            name=cast("str", t["name"]),
            description=cast("str", t["description"]),
            parameters=cast("dict[str, Any]", t["parameters"]),
        )
        for t in raw_list
    ]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_response_matches(actual: Response, expected: Mapping[str, Any]) -> None:
    """Verify ``actual`` matches the fixture's ``expected.response``
    block. ``raw_check.required_keys`` is enforced as a presence-only
    check (other keys MAY also be present)."""
    expected_msg = cast("Mapping[str, Any]", expected.get("message") or {})
    if "content" in expected_msg:
        # YAML loads `null` as None — the spec sometimes uses null for
        # tool-call-only messages; our AssistantMessage stores empty
        # string for that case.
        expected_content = expected_msg["content"]
        actual_content = actual.message.content or None
        assert actual_content == expected_content, (
            f"content mismatch: actual={actual_content!r}, expected={expected_content!r}"
        )
    if "tool_calls" in expected_msg:
        expected_tool_calls = cast("list[dict[str, Any]] | None", expected_msg["tool_calls"])
        if expected_tool_calls is None:
            assert actual.message.tool_calls is None, (
                f"expected no tool_calls, got {actual.message.tool_calls}"
            )
        else:
            assert actual.message.tool_calls is not None, "expected tool_calls but got none"
            actual_tcs = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in actual.message.tool_calls
            ]
            assert actual_tcs == expected_tool_calls, (
                f"tool_calls mismatch: actual={actual_tcs}, expected={expected_tool_calls}"
            )
    if "finish_reason" in expected:
        assert actual.finish_reason == expected["finish_reason"]
    if "usage" in expected:
        expected_usage = expected["usage"]
        actual_usage = actual.usage.model_dump()
        assert actual_usage == expected_usage, (
            f"usage mismatch: actual={actual_usage}, expected={expected_usage}"
        )
    raw_check = expected.get("raw_check")
    if raw_check:
        required = cast("list[str]", raw_check.get("required_keys") or [])
        for key in required:
            assert key in actual.raw, f"raw missing required key {key!r}"


def _assert_raises_matches(
    excinfo: pytest.ExceptionInfo[LlmProviderError],
    expected: Mapping[str, Any],
) -> None:
    err = excinfo.value
    assert err.category == expected["category"], (
        f"category mismatch: actual={err.category!r}, expected={expected['category']!r}"
    )
    if "retry_after_seconds" in expected:
        assert isinstance(err, ProviderRateLimit)
        assert err.retry_after == expected["retry_after_seconds"]


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_llm_provider_fixture(fixture_path: Path) -> None:
    fixture_id = fixture_path.stem
    if fixture_id in _DEFERRED_FIXTURES:
        pytest.skip(f"{fixture_id}: {_DEFERRED_FIXTURES[fixture_id]}")
    spec = _load(fixture_path)

    if "cases" in spec:
        for case in cast("list[dict[str, Any]]", spec["cases"]):
            case_name = case.get("name", "<unnamed>")
            try:
                await _run_one_case(case)
            except AssertionError as e:
                raise AssertionError(f"case {case_name!r}: {e}") from e
        return

    await _run_one_case(spec)


async def _run_one_case(spec: Mapping[str, Any]) -> None:
    """Run one fixture or one case from a cases-shape fixture.

    A case has either:
    - top-level ``calls: [...]`` (multi-call fixture, e.g. 001, 002)
    - top-level ``call: {...}`` (single-call case, e.g. each 004 sub-case)
    - top-level ``mock_provider:`` configures the wire mock
    """
    mock_cfg = cast("Mapping[str, Any]", spec.get("mock_provider") or {})
    provider, _captured = _build_provider(mock_cfg)
    try:
        for call_spec in _iter_calls(spec):
            await _run_one_call(provider, call_spec)
    finally:
        await provider.aclose()


def _iter_calls(spec: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield each call dict with its ``expected`` block attached.

    Two shapes the fixtures use:
    - ``calls: [{operation, messages, expected, ...}]`` — call and
      expected are siblings inside each call entry.
    - ``call: {operation, messages, ...}`` + sibling ``expected: ...``
      — the case-shape, where expected lives alongside the call.
    Both are normalised here to a flat dict where ``expected`` is on
    the call.
    """
    if "calls" in spec:
        yield from cast("list[Mapping[str, Any]]", spec["calls"])
    elif "call" in spec:
        call = dict(cast("Mapping[str, Any]", spec["call"]))
        if "expected" in spec and "expected" not in call:
            call["expected"] = spec["expected"]
        yield call
    else:
        raise AssertionError("fixture has neither `calls` nor `call` block")


async def _run_one_call(provider: OpenAIProvider, call_spec: Mapping[str, Any]) -> None:
    operation = call_spec.get("operation", "complete")
    expected = cast("Mapping[str, Any]", call_spec.get("expected") or {})

    if operation == "complete":
        # Per spec §3 "Validation timing" — complete() validates at
        # the boundary. Pydantic also catches per-role constraints at
        # message construction time. The fixture treats both layers as
        # the same boundary check (the user's mental model is "calling
        # complete() with a malformed input raises"), so wrap the
        # construction in the raises path so a pydantic ValidationError
        # surfaces as ProviderInvalidRequest.
        if "raises" in expected:
            with pytest.raises(LlmProviderError) as excinfo:
                try:
                    messages = [
                        _build_message(m) for m in cast("list[Mapping[str, Any]]", call_spec["messages"])
                    ]
                    tools = _build_tools(cast("list[Mapping[str, Any]] | None", call_spec.get("tools")))
                except ValidationError as ve:
                    raise ProviderInvalidRequest(str(ve)) from ve
                await provider.complete(messages, tools)
            _assert_raises_matches(excinfo, expected["raises"])
        else:
            messages = [_build_message(m) for m in cast("list[Mapping[str, Any]]", call_spec["messages"])]
            tools = _build_tools(cast("list[Mapping[str, Any]] | None", call_spec.get("tools")))
            response = await provider.complete(messages, tools)
            _assert_response_matches(response, cast("Mapping[str, Any]", expected.get("response") or {}))
        return

    if operation == "ready":
        if "raises" in expected:
            with pytest.raises(LlmProviderError) as excinfo:
                await provider.ready()
            _assert_raises_matches(excinfo, expected["raises"])
        else:
            await provider.ready()
            success = expected.get("success")
            if success is not None:
                assert bool(success), "fixture marked ready() expected.success: false but call succeeded"
        return

    raise AssertionError(f"unknown operation in fixture: {operation!r}")
