"""Run every spec llm-provider conformance fixture against OpenAIProvider.

The fixtures (``spec/llm-provider/conformance/``) describe a provider's
behavior in terms of OpenAI Chat Completions wire-format mock
responses + expected ``Provider.complete()`` / ``Provider.ready()``
outcomes. The harness drives the real :class:`OpenAIProvider` via
``httpx.MockTransport`` so the wire-mapping path (spec §8.1) is
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
from typing import Any, Literal, cast

import httpx
import pytest
import yaml
from pydantic import ValidationError

from openarmature.llm import (
    TRANSIENT_CATEGORIES,
    AssistantMessage,
    ForceTool,
    LlmProviderError,
    Message,
    OpenAIProvider,
    ProviderInvalidRequest,
    ProviderRateLimit,
    Response,
    RuntimeConfig,
    SystemMessage,
    Tool,
    ToolCall,
    ToolChoice,
    ToolMessage,
    UserMessage,
)

from ._deferral import skip_if_deferred
from .harness import (
    assert_error_carries,
    assert_response_format_absent,
    assert_system_references_schema,
    assert_tool_choice_absent,
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
    # Proposal 0037 (Anthropic Messages mapping) shipped in spec v0.28.0
    # but python marks it not-yet in conformance.toml — the Anthropic
    # provider isn't implemented in this release. 043 (the OpenAI side
    # stripping anthropic thinking-block content) waits with it.
    "033-anthropic-basic-message-round-trip": "Anthropic provider not implemented (0037 not-yet)",
    "034-anthropic-tool-call-flow": "Anthropic provider not implemented (0037 not-yet)",
    "035-anthropic-image-content-blocks": "Anthropic provider not implemented (0037 not-yet)",
    "036-anthropic-tool-choice-modes": "Anthropic provider not implemented (0037 not-yet)",
    "037-anthropic-runtime-config-mapping": "Anthropic provider not implemented (0037 not-yet)",
    "038-anthropic-max-tokens-required": "Anthropic provider not implemented (0037 not-yet)",
    "039-anthropic-error-mapping": "Anthropic provider not implemented (0037 not-yet)",
    "040-anthropic-structured-output-native": "Anthropic provider not implemented (0037 not-yet)",
    "041-anthropic-structured-output-fallback": "Anthropic provider not implemented (0037 not-yet)",
    "042-anthropic-thinking-block-round-trip": "Anthropic provider not implemented (0037 not-yet)",
    "043-openai-strips-thinking-blocks": "Anthropic provider not implemented (0037 not-yet)",
    # Proposal 0038 (Google Gemini wire-format mapping) shipped in spec
    # v0.32.0 but python marks it not-yet — the Gemini provider isn't
    # implemented in this release.
    "044-gemini-basic-message-round-trip": "Gemini provider not implemented (0038 not-yet)",
    "045-gemini-function-call-flow": "Gemini provider not implemented (0038 not-yet)",
    "046-gemini-image-content-blocks": "Gemini provider not implemented (0038 not-yet)",
    "047-gemini-tool-choice-modes": "Gemini provider not implemented (0038 not-yet)",
    "048-gemini-runtime-config-mapping": "Gemini provider not implemented (0038 not-yet)",
    "049-gemini-error-mapping": "Gemini provider not implemented (0038 not-yet)",
    "050-gemini-structured-output-native": "Gemini provider not implemented (0038 not-yet)",
    "051-gemini-structured-output-fallback": "Gemini provider not implemented (0038 not-yet)",
    "052-gemini-thought-signature-round-trip": "Gemini provider not implemented (0038 not-yet)",
    "053-cross-provider-signature-strip": "Gemini provider not implemented (0038 not-yet)",
    # ----- v0.12.0 cycle spec-pin bump (v0.38.0 -> v0.45.0) -------------
    # Proposal 0047 (implicit prefix-cache wire-byte stability, v0.39.0)
    # — wire-byte hashing across providers. Queued for v0.13.0 LLM
    # provider hardening batch.
    "054-openai-wire-byte-stability": ("Proposal 0047 wire-byte stability; queued for v0.13.0"),
    "055-anthropic-wire-byte-stability": (
        "Proposal 0047 wire-byte stability; queued for v0.13.0 (also Anthropic-pending)"
    ),
    # Proposal 0050 (call-level retry, v0.42.0) — three fixtures
    # exercise the new ``retry`` kwarg on ``complete()``. Queued for
    # v0.14.0 retry & reliability primitives batch.
    "056-call-level-retry-transient": ("Proposal 0050 call-level retry; queued for v0.14.0"),
    "057-call-level-retry-exhaustion": ("Proposal 0050 call-level retry; queued for v0.14.0"),
    "058-call-level-retry-non-transient-no-retry": ("Proposal 0050 call-level retry; queued for v0.14.0"),
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
    readiness_probe: Literal["models", "chat_completions", "both"] = "chat_completions"
    if "health_endpoint" in mock_provider_cfg:
        health_endpoint = cast("Mapping[str, Any]", mock_provider_cfg["health_endpoint"])
        responses: list[Mapping[str, Any]] = [health_endpoint]
        # The spec fixture intentionally leaves the probe shape to the
        # implementation (007 comment: "the implementation's chosen probe").
        # Pick the OpenAIProvider readiness_probe mode that matches the
        # fixture's mocked path so the mock is actually exercised. Fixtures
        # 007's cases all mock ``/v1/models``, so the catalog probe is what
        # they verify; a future fixture mocking ``/v1/chat/completions``
        # would automatically route to the chat probe. A missing ``path``
        # field leaves us at the OpenAIProvider default; all current
        # fixtures populate ``path``, so this branch is unreachable in
        # practice and the fallthrough is only defensive.
        endpoint_path = health_endpoint.get("path")
        if endpoint_path == "/v1/models":
            readiness_probe = "models"
        elif endpoint_path == "/v1/chat/completions":
            readiness_probe = "chat_completions"
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
        readiness_probe=readiness_probe,
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
        # Per spec §3, user content is str OR a list of content blocks.
        # Pydantic's discriminated union on the block ``type`` field
        # parses each dict in the list to the right TextBlock /
        # ImageBlock variant automatically.
        return UserMessage(content=raw["content"])
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


def _build_tool_choice(raw: Any) -> ToolChoice | None:
    """Translate a fixture's ``tool_choice:`` value into the
    :class:`ToolChoice` discriminated-union value.

    Two YAML shapes per spec proposal 0025:

    - String: ``auto`` / ``required`` / ``none`` — passes through
      verbatim.
    - Dict: ``{type: tool, name: X}`` — constructed into a
      :class:`ForceTool` record. The wire-side rename (``tool`` →
      ``function``) happens inside the provider, not at parse time.

    Returns ``None`` when the fixture omits ``tool_choice``; the
    provider's own default applies on the wire.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return cast("ToolChoice", raw)
    if isinstance(raw, dict):
        return ForceTool.model_validate(raw)
    raise AssertionError(f"unrecognized tool_choice shape in fixture: {raw!r}")


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
            elif key == "tool_choice_absent":
                if value is True:
                    assert_tool_choice_absent(body)
            elif key == "system_message_content_references_schema":
                if value is True:
                    if not isinstance(response_schema, dict):
                        raise AssertionError(
                            "system_message_content_references_schema "
                            "requires a dict response_schema on the call"
                        )
                    assert_system_references_schema(body, cast("dict[str, Any]", response_schema))
            elif key.endswith("_absent"):
                # Generic ``<field>_absent: true`` assertion: the
                # wire body MUST NOT carry the named field. Used by
                # fixture 032 (null-skip) to verify unset declared
                # fields don't serialize as JSON null.
                if value is True:
                    field = key[: -len("_absent")]
                    if field in body:
                        raise AssertionError(
                            f"expected_wire_request_checks.{key}: "
                            f"field {field!r} present in wire body "
                            f"with value {body[field]!r}"
                        )
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
        actual_usage_full = actual.usage.model_dump()
        # Subset comparison when the fixture asserts about specific
        # usage fields: spec fixtures pin which fields MUST be present
        # with what values. Impl-extension fields outside the fixture's
        # expected set (e.g., the 0047 cache-stat fields on impls that
        # have adopted them but against fixtures that pre-date the
        # proposal) are ignored when the fixture doesn't assert about
        # them. A fixture key that's absent from actual surfaces as a
        # missing key in the filtered dict and fails the comparison;
        # the impl can't silently drop a field the spec requires.
        #
        # Non-mapping expected_usage (e.g., a fixture sets usage: null)
        # falls back to direct comparison so the assertion fires with a
        # clean shape mismatch rather than crashing on the subset filter.
        if isinstance(expected_usage, dict):
            actual_usage = {k: v for k, v in actual_usage_full.items() if k in expected_usage}
        else:
            actual_usage = actual_usage_full
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
    skip_if_deferred(fixture_id, _DEFERRED_FIXTURES)
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


# Keys that may live as siblings to a ``call:`` block in a cases-shape
# fixture but are conceptually call-level metadata. ``_iter_calls``
# copies these from the case into the yielded call so the test runner
# sees them in one place.
_CASE_LEVEL_CALL_KEYS = (
    "expected",
    "expected_wire_request",
    "expected_wire_request_checks",
    "response_schema",
    "retry_middleware",
)


def _iter_calls(spec: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield each call dict with its case-level metadata attached.

    Two shapes the fixtures use:
    - ``calls: [{operation, messages, expected, ...}]`` — call and
      expected are siblings inside each call entry.
    - ``call: {operation, messages, ...}`` + sibling ``expected: ...``
      (and possibly ``expected_wire_request:``, ``response_schema:``,
      ``retry_middleware:``) — the case-shape, where call-level
      metadata lives alongside the call. All sibling keys in
      ``_CASE_LEVEL_CALL_KEYS`` are folded into the call dict here so
      the runner reads them from one place. The nested ``call`` block
      takes precedence when both are present.
    """
    if "calls" in spec:
        yield from cast("list[Mapping[str, Any]]", spec["calls"])
    elif "call" in spec:
        call = dict(cast("Mapping[str, Any]", spec["call"]))
        for key in _CASE_LEVEL_CALL_KEYS:
            if key in spec and key not in call:
                call[key] = spec[key]
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
    config_block = call_spec.get("config")
    # YAML convention: `config.extras: {...}` is the sub-block for
    # undeclared (provider-specific) RuntimeConfig fields. Flatten it
    # into the kwargs splat so the extras land in RuntimeConfig's
    # model_extra rather than as a single `extras` key.
    if config_block:
        block = dict(cast("Mapping[str, Any]", config_block))
        extras_block = cast("Mapping[str, Any] | None", block.pop("extras", None))
        if extras_block:
            block.update(extras_block)
        config = RuntimeConfig(**block)
    else:
        config = None

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
                    tool_choice = _build_tool_choice(call_spec.get("tool_choice"))
                except ValidationError as ve:
                    raise ProviderInvalidRequest(str(ve)) from ve
                await _maybe_with_retry(
                    lambda: provider.complete(
                        messages,
                        tools,
                        config,
                        response_schema=response_schema,
                        tool_choice=tool_choice,
                    ),
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
            tool_choice = _build_tool_choice(call_spec.get("tool_choice"))
            response = await _maybe_with_retry(
                lambda: provider.complete(
                    messages,
                    tools,
                    config,
                    response_schema=response_schema,
                    tool_choice=tool_choice,
                ),
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
