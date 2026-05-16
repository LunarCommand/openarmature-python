"""Focused unit tests for the prompts subpackage.

The conformance suite (``tests/conformance/test_prompt_management.py``)
covers the spec's behavioral surface end-to-end against fixtures
001-012. These unit tests fill gaps the conformance fixtures don't
exercise directly: per-class construction validation,
FilesystemPromptBackend disk I/O, hashing helpers, context-variable
propagation, and the empty-string-render boundary wrap.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from openarmature.llm.messages import Message, UserMessage
from openarmature.prompts import (
    PROMPT_NOT_FOUND,
    PROMPT_RENDER_ERROR,
    PROMPT_STORE_UNAVAILABLE,
    PROMPT_TRANSIENT_CATEGORIES,
    FilesystemPromptBackend,
    Prompt,
    PromptError,
    PromptGroup,
    PromptManager,
    PromptNotFound,
    PromptRenderError,
    PromptResult,
    PromptStoreUnavailable,
    compute_rendered_hash,
    compute_template_hash,
    current_prompt_group,
    current_prompt_result,
    with_active_prompt,
    with_active_prompt_group,
)

# ---------------------------------------------------------------------------
# Error class hierarchy + categories
# ---------------------------------------------------------------------------


def test_error_categories_match_spec() -> None:
    assert PromptNotFound.category == "prompt_not_found"
    assert PromptRenderError.category == "prompt_render_error"
    assert PromptStoreUnavailable.category == "prompt_store_unavailable"
    assert PROMPT_NOT_FOUND == "prompt_not_found"
    assert PROMPT_RENDER_ERROR == "prompt_render_error"
    assert PROMPT_STORE_UNAVAILABLE == "prompt_store_unavailable"


def test_transient_categories_contains_only_store_unavailable() -> None:
    assert PROMPT_TRANSIENT_CATEGORIES == frozenset({PROMPT_STORE_UNAVAILABLE})


def test_prompt_not_found_carries_identity_attributes() -> None:
    exc = PromptNotFound("nope", name="greeting", label="production", backend="local")
    assert exc.name == "greeting"
    assert exc.label == "production"
    assert exc.backend == "local"
    assert isinstance(exc, PromptError)


def test_prompt_render_error_carries_identity_and_variables() -> None:
    exc = PromptRenderError(
        "boom",
        name="greeting",
        version="v1",
        label="production",
        variables={"user": "Alice"},
        description="undefined: day",
    )
    assert exc.name == "greeting"
    assert exc.version == "v1"
    assert exc.label == "production"
    assert exc.variables == {"user": "Alice"}
    assert exc.description == "undefined: day"


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def test_template_hash_is_deterministic_and_prefixed() -> None:
    a = compute_template_hash("Hello, {{ user }}!")
    b = compute_template_hash("Hello, {{ user }}!")
    assert a == b
    assert a.startswith("sha256:")
    assert len(a) == len("sha256:") + 64


def test_template_hash_differs_for_different_inputs() -> None:
    a = compute_template_hash("Hello!")
    b = compute_template_hash("Goodbye!")
    assert a != b


def test_rendered_hash_is_deterministic() -> None:
    msgs: list[Message] = [UserMessage(content="Hello, Alice!")]
    a = compute_rendered_hash(msgs)
    b = compute_rendered_hash(msgs)
    assert a == b
    assert a.startswith("sha256:")


def test_rendered_hash_differs_for_different_message_content() -> None:
    msgs_a: list[Message] = [UserMessage(content="Hello, Alice!")]
    msgs_b: list[Message] = [UserMessage(content="Hello, Bob!")]
    a = compute_rendered_hash(msgs_a)
    b = compute_rendered_hash(msgs_b)
    assert a != b


# ---------------------------------------------------------------------------
# Type construction
# ---------------------------------------------------------------------------


def _make_prompt(template: str = "Hello, {{ user }}!") -> Prompt:
    return Prompt(
        name="greeting",
        version="v1",
        label="production",
        template=template,
        template_hash=compute_template_hash(template),
        fetched_at=datetime.now(UTC),
    )


def test_prompt_extra_fields_forbidden() -> None:
    with pytest.raises(ValueError, match="extra"):
        Prompt.model_validate(
            {
                "name": "greeting",
                "version": "v1",
                "label": "production",
                "template": "Hi",
                "template_hash": "sha256:abc",
                "fetched_at": datetime.now(UTC),
                "unknown_field": "not allowed",
            }
        )


def test_prompt_group_rejects_zero_members() -> None:
    with pytest.raises(ValueError, match="at least two"):
        PromptGroup(group_name="g", members=[])


def test_prompt_group_rejects_one_member() -> None:
    prompt = _make_prompt()
    pr = PromptResult(
        name=prompt.name,
        version=prompt.version,
        label=prompt.label,
        template_hash=prompt.template_hash,
        rendered_hash="sha256:abc",
        messages=[UserMessage(content="x")],
        variables={},
        fetched_at=prompt.fetched_at,
        rendered_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="at least two"):
        PromptGroup(group_name="g", members=[pr])


def test_prompt_group_accepts_two_or_more_members() -> None:
    prompt = _make_prompt()
    pr = PromptResult(
        name=prompt.name,
        version=prompt.version,
        label=prompt.label,
        template_hash=prompt.template_hash,
        rendered_hash="sha256:abc",
        messages=[UserMessage(content="x")],
        variables={},
        fetched_at=prompt.fetched_at,
        rendered_at=datetime.now(UTC),
    )
    PromptGroup(group_name="g", members=[pr, pr])
    PromptGroup(group_name="g", members=[pr, pr, pr])


# ---------------------------------------------------------------------------
# PromptManager — construction + render edge cases
# ---------------------------------------------------------------------------


def test_manager_requires_at_least_one_backend() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        PromptManager()


def test_render_empty_string_output_maps_to_prompt_render_error() -> None:
    # The boundary-wrap from the spec-agent's concern: a template that
    # renders cleanly to "" through Jinja2 would construct
    # UserMessage(content="") which Pydantic rejects.
    prompt = _make_prompt(template="{{ x if x else '' }}")

    class _NullBackend:
        async def fetch(self, name: str, label: str = "production") -> Prompt:
            return prompt

    manager = PromptManager(_NullBackend())
    with pytest.raises(PromptRenderError) as exc_info:
        manager.render(prompt, {"x": None})
    assert exc_info.value.name == "greeting"
    assert exc_info.value.label == "production"


def test_render_propagates_identity_fields() -> None:
    prompt = _make_prompt()

    class _Backend:
        async def fetch(self, name: str, label: str = "production") -> Prompt:
            return prompt

    manager = PromptManager(_Backend())
    result = manager.render(prompt, {"user": "Alice"})
    assert result.name == prompt.name
    assert result.version == prompt.version
    assert result.label == prompt.label
    assert result.template_hash == prompt.template_hash
    assert result.fetched_at == prompt.fetched_at
    assert result.variables == {"user": "Alice"}
    assert len(result.messages) == 1


# ---------------------------------------------------------------------------
# FilesystemPromptBackend
# ---------------------------------------------------------------------------


async def test_filesystem_backend_fetch_success(tmp_path: Path) -> None:
    label_dir = tmp_path / "production"
    label_dir.mkdir()
    (label_dir / "greeting.j2").write_text("Hello, {{ user }}!", encoding="utf-8")

    backend = FilesystemPromptBackend(tmp_path)
    prompt = await backend.fetch("greeting", "production")
    assert prompt.name == "greeting"
    assert prompt.label == "production"
    assert prompt.template == "Hello, {{ user }}!"
    assert prompt.template_hash == compute_template_hash("Hello, {{ user }}!")
    # version derived from first 12 hex chars of template_hash
    assert prompt.version == prompt.template_hash.removeprefix("sha256:")[:16]


async def test_filesystem_backend_fetch_missing_file_raises_not_found(tmp_path: Path) -> None:
    backend = FilesystemPromptBackend(tmp_path)
    with pytest.raises(PromptNotFound) as exc_info:
        await backend.fetch("missing", "production")
    assert exc_info.value.name == "missing"
    assert exc_info.value.label == "production"
    assert exc_info.value.backend == str(tmp_path)


async def test_filesystem_backend_io_error_raises_store_unavailable(tmp_path: Path) -> None:
    # Mock ``Path.read_text`` to raise a generic ``OSError`` so the
    # test isolates the OSError-but-not-FileNotFoundError branch
    # without depending on platform-specific filesystem semantics
    # (Linux surfaces NotADirectoryError for "file where directory
    # expected"; Windows can surface PermissionError or other
    # OSError subclasses).
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "foo.j2").write_text("template", encoding="utf-8")
    backend = FilesystemPromptBackend(tmp_path)
    with patch("pathlib.Path.read_text", side_effect=OSError("simulated I/O error")):
        with pytest.raises(PromptStoreUnavailable):
            await backend.fetch("foo", "production")


# ---------------------------------------------------------------------------
# Context-variable propagation
# ---------------------------------------------------------------------------


def _make_prompt_result() -> PromptResult:
    prompt = _make_prompt()
    return PromptResult(
        name=prompt.name,
        version=prompt.version,
        label=prompt.label,
        template_hash=prompt.template_hash,
        rendered_hash="sha256:rendered",
        messages=[UserMessage(content="hi")],
        variables={"user": "Alice"},
        fetched_at=prompt.fetched_at,
        rendered_at=datetime.now(UTC),
    )


def test_current_prompt_result_default_is_none() -> None:
    assert current_prompt_result() is None


def test_with_active_prompt_sets_and_resets() -> None:
    pr = _make_prompt_result()
    assert current_prompt_result() is None
    with with_active_prompt(pr):
        assert current_prompt_result() is pr
    assert current_prompt_result() is None


def test_with_active_prompt_innermost_wins() -> None:
    outer = _make_prompt_result()
    inner = _make_prompt_result()
    with with_active_prompt(outer):
        assert current_prompt_result() is outer
        with with_active_prompt(inner):
            assert current_prompt_result() is inner
        assert current_prompt_result() is outer
    assert current_prompt_result() is None


def test_with_active_prompt_group_default_none_and_sets() -> None:
    pr1 = _make_prompt_result()
    pr2 = _make_prompt_result()
    group = PromptGroup(group_name="g", members=[pr1, pr2])
    assert current_prompt_group() is None
    with with_active_prompt_group(group):
        assert current_prompt_group() is group
    assert current_prompt_group() is None


async def test_active_prompt_visible_from_nested_async_function() -> None:
    pr = _make_prompt_result()

    async def _read_in_task() -> PromptResult | None:
        await asyncio.sleep(0)
        return current_prompt_result()

    with with_active_prompt(pr):
        result = await _read_in_task()
    assert result is pr


# ---------------------------------------------------------------------------
# PromptManager fallback semantics (gaps the fixtures don't cover)
# ---------------------------------------------------------------------------


async def test_manager_fetch_first_match_short_circuits() -> None:
    """Once a backend returns a Prompt, later backends are not consulted."""
    prompt = _make_prompt()

    class _Hit:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch(self, name: str, label: str = "production") -> Prompt:
            self.calls += 1
            return prompt

    class _Second:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch(self, name: str, label: str = "production") -> Prompt:
            self.calls += 1
            return prompt

    first = _Hit()
    second = _Second()
    manager = PromptManager(first, second)
    await manager.fetch("greeting", "production")
    assert first.calls == 1
    assert second.calls == 0


async def test_manager_render_signature_returns_user_message() -> None:
    prompt = _make_prompt()

    class _Backend:
        async def fetch(self, name: str, label: str = "production") -> Prompt:
            return prompt

    manager = PromptManager(_Backend())
    result = manager.render(prompt, {"user": "Alice"})
    assert isinstance(result.messages[0], UserMessage)
    msg_content: Any = result.messages[0].content
    assert msg_content == "Hello, Alice!"
