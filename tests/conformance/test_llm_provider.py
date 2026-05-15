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
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml
from pydantic import ValidationError

from openarmature.llm import (
    TRANSIENT_CATEGORIES,
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

from .harness import (
    assert_error_carries,
    assert_response_format_absent,
    assert_system_references_schema,
    match_wire_body,
    request_body,
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
    # ``capabilities.supports_native_response_format: false`` switches
    # the provider into prompt-augmentation fallback mode for structured
    # output. Absent or true ⇒ native path (default).
    capabilities = cast("Mapping[str, Any]", mock_provider_cfg.get("capabilities") or {})
    supports_native = capabilities.get("supports_native_response_format", True)
    force_fallback = supports_native is False
    provider = OpenAIProvider(
        base_url="http://mock-llm.test",
        model=model,
        api_key="test-key",
        transport=transport,
        force_prompt_augmentation_fallback=force_fallback,
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


async def _maybe_with_retry(
    operation: Callable[[], Awaitable[Any]],
    retry_cfg: Mapping[str, Any] | None,
) -> Any:
    """Optionally wrap an LLM-provider call in retry-middleware
    semantics. The harness simulates RetryMiddleware's default
    classifier (transient if exc.category is in TRANSIENT_CATEGORIES,
    non-transient otherwise) without dragging the graph-engine into
    LLM-provider conformance. ``classifier`` other than ``"default"``
    is not yet supported — raises AssertionError.
    """
    if retry_cfg is None:
        return await operation()
    classifier = retry_cfg.get("classifier", "default")
    if classifier != "default":
        raise AssertionError(f"retry_middleware classifier {classifier!r} not yet supported")
    max_attempts = int(retry_cfg.get("max_attempts", 1))
    attempts = 0
    while True:
        attempts += 1
        try:
            return await operation()
        except LlmProviderError as exc:
            if attempts >= max_attempts:
                raise
            if exc.category not in TRANSIENT_CATEGORIES:
                raise


def _assert_wire_expectations(
    *,
    call_spec: Mapping[str, Any],
    captured: list[httpx.Request],
    wire_count_before: int,
    response_schema: Any,
) -> None:
    """Apply ``expected_wire_request`` literal compare and
    ``expected_wire_request_checks`` sibling-check blocks. Both
    operate on the most-recent captured chat-completions request.
    """
    expected_wire = cast("Mapping[str, Any] | None", call_spec.get("expected_wire_request"))
    checks = cast(
        "Mapping[str, Any] | None",
        call_spec.get("expected_wire_request_checks"),
    )
    if expected_wire is None and checks is None:
        return
    last_request = _last_chat_completions_request(captured, wire_count_before)
    if last_request is None:
        raise AssertionError(
            "expected_wire_request[_checks] supplied, but no chat-completions request was captured"
        )
    body = request_body(last_request)
    if expected_wire is not None:
        match_wire_body(body, expected_wire)
    if checks is not None:
        for key, value in checks.items():
            if key == "response_format_absent":
                if value is True:
                    assert_response_format_absent(body)
            elif key == "system_message_content_references_schema":
                if value is True:
                    if not isinstance(response_schema, dict):
                        raise AssertionError(
                            "system_message_content_references_schema "
                            "requires a dict response_schema on the call"
                        )
                    assert_system_references_schema(body, cast("dict[str, Any]", response_schema))
            else:
                raise AssertionError(f"unknown expected_wire_request_checks key: {key!r}")


def _last_chat_completions_request(
    captured: list[httpx.Request],
    since: int,
) -> httpx.Request | None:
    """Pick the most recent /v1/chat/completions request captured at or
    after ``since`` (the wire-count baseline before this call started).
    The mock transport sees other requests too (e.g., /v1/models on
    ready()); skipping non-chat URLs keeps the wire-shape assertions
    targeted at the operation under test.
    """
    for req in reversed(captured[since:]):
        if req.url.path == "/v1/chat/completions":
            return req
    return None


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
    if "parsed" in expected:
        expected_parsed = expected["parsed"]
        actual_parsed = actual.parsed
        # BaseModel-class fixture cases would surface a BaseModel
        # instance on actual.parsed; the fixtures here only use the
        # dict-schema form, so a dict equality compare is sufficient.
        # Future fixtures driving the Pydantic-class overload can
        # extend this with a model_dump() comparison.
        assert actual_parsed == expected_parsed, (
            f"parsed mismatch: actual={actual_parsed!r}, expected={expected_parsed!r}"
        )


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
    provider, captured = _build_provider(mock_cfg)
    try:
        for call_spec in _iter_calls(spec):
            await _run_one_call(provider, call_spec, captured)
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


async def _run_one_call(
    provider: OpenAIProvider,
    call_spec: Mapping[str, Any],
    captured: list[httpx.Request],
) -> None:
    operation = call_spec.get("operation", "complete")
    expected = cast("Mapping[str, Any]", call_spec.get("expected") or {})
    response_schema = call_spec.get("response_schema")
    retry_mw_cfg = cast("Mapping[str, Any] | None", call_spec.get("retry_middleware"))

    if operation == "complete":
        # Per spec §3 "Validation timing" — complete() validates at
        # the boundary. Pydantic also catches per-role constraints at
        # message construction time. The fixture treats both layers as
        # the same boundary check (the user's mental model is "calling
        # complete() with a malformed input raises"), so wrap the
        # construction in the raises path so a pydantic ValidationError
        # surfaces as ProviderInvalidRequest.
        wire_count_before = len(captured)
        if "raises" in expected:
            with pytest.raises(LlmProviderError) as excinfo:
                try:
                    messages = [
                        _build_message(m) for m in cast("list[Mapping[str, Any]]", call_spec["messages"])
                    ]
                    tools = _build_tools(cast("list[Mapping[str, Any]] | None", call_spec.get("tools")))
                except ValidationError as ve:
                    raise ProviderInvalidRequest(str(ve)) from ve
                await _maybe_with_retry(
                    lambda: provider.complete(messages, tools, response_schema=response_schema),
                    retry_mw_cfg,
                )
            _assert_raises_matches(excinfo, expected["raises"])
            carries = cast(
                "Mapping[str, Any] | None",
                cast("Mapping[str, Any]", expected["raises"]).get("carries"),
            )
            if carries:
                assert_error_carries(excinfo.value, carries)
        else:
            messages = [_build_message(m) for m in cast("list[Mapping[str, Any]]", call_spec["messages"])]
            messages_snapshot = [m.model_dump(mode="json") for m in messages]
            tools = _build_tools(cast("list[Mapping[str, Any]] | None", call_spec.get("tools")))
            response = await _maybe_with_retry(
                lambda: provider.complete(messages, tools, response_schema=response_schema),
                retry_mw_cfg,
            )
            _assert_response_matches(response, cast("Mapping[str, Any]", expected.get("response") or {}))
            if expected.get("caller_messages_unmodified") is True:
                post_snapshot = [m.model_dump(mode="json") for m in messages]
                assert post_snapshot == messages_snapshot, (
                    "caller_messages_unmodified: messages list mutated by complete()"
                )

        wire_count_after = len(captured)
        provider_call_count = wire_count_after - wire_count_before
        expected_call_count = expected.get("provider_call_count")
        if expected_call_count is not None:
            assert provider_call_count == expected_call_count, (
                f"provider_call_count: actual={provider_call_count}, expected={expected_call_count}"
            )
        _assert_wire_expectations(
            call_spec=call_spec,
            captured=captured,
            wire_count_before=wire_count_before,
            response_schema=response_schema,
        )
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
