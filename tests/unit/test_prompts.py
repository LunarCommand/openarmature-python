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
    PROMPT_GROUP_INVALID,
    PROMPT_NOT_FOUND,
    PROMPT_RENDER_ERROR,
    PROMPT_STORE_UNAVAILABLE,
    PROMPT_TRANSIENT_CATEGORIES,
    FilesystemPromptBackend,
    Prompt,
    PromptError,
    PromptGroup,
    PromptGroupInvalid,
    PromptManager,
    PromptNotFound,
    PromptRenderError,
    PromptResult,
    PromptStoreUnavailable,
    TextPrompt,
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
    assert PromptGroupInvalid.category == "prompt_group_invalid"
    assert PROMPT_NOT_FOUND == "prompt_not_found"
    assert PROMPT_RENDER_ERROR == "prompt_render_error"
    assert PROMPT_STORE_UNAVAILABLE == "prompt_store_unavailable"
    assert PROMPT_GROUP_INVALID == "prompt_group_invalid"


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


def _make_prompt(template: str = "Hello, {{ user }}!") -> TextPrompt:
    return TextPrompt(
        name="greeting",
        version="v1",
        label="production",
        template=template,
        template_hash=compute_template_hash(template),
        fetched_at=datetime.now(UTC),
    )


def test_prompt_extra_fields_forbidden() -> None:
    with pytest.raises(ValueError, match="extra"):
        TextPrompt.model_validate(
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


def test_prompt_result_rejects_empty_messages() -> None:
    prompt = _make_prompt()
    with pytest.raises(ValueError):
        PromptResult(
            name=prompt.name,
            version=prompt.version,
            label=prompt.label,
            template_hash=prompt.template_hash,
            rendered_hash="sha256:abc",
            messages=[],
            variables={},
            fetched_at=prompt.fetched_at,
            rendered_at=datetime.now(UTC),
        )


def test_prompt_group_rejects_zero_members() -> None:
    # Categorized prompt_group_invalid (proposal 0080), not a bare ValueError:
    # PromptGroupInvalid is not a ValueError, so pydantic propagates it
    # unwrapped from the model validator rather than folding it into a
    # ValidationError.
    with pytest.raises(PromptGroupInvalid, match="at least two") as exc_info:
        PromptGroup(group_name="g", members=[])
    assert exc_info.value.category == "prompt_group_invalid"


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
    with pytest.raises(PromptGroupInvalid, match="at least two") as exc_info:
        PromptGroup(group_name="g", members=[pr])
    assert exc_info.value.category == "prompt_group_invalid"


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
        async def fetch(
            self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
        ) -> Prompt:
            return prompt

    manager = PromptManager(_NullBackend())
    with pytest.raises(PromptRenderError) as exc_info:
        manager.render(prompt, {"x": None})
    assert exc_info.value.name == "greeting"
    assert exc_info.value.label == "production"


def test_render_propagates_identity_fields() -> None:
    prompt = _make_prompt()

    class _Backend:
        async def fetch(
            self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
        ) -> Prompt:
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


async def test_fetch_rejects_negative_cache_ttl_seconds() -> None:
    prompt = _make_prompt()

    class _Backend:
        async def fetch(
            self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
        ) -> Prompt:
            return prompt

    manager = PromptManager(_Backend())
    with pytest.raises(ValueError, match="cache_ttl_seconds must be >= 0"):
        await manager.fetch("greeting", "production", cache_ttl_seconds=-1)


# ---------------------------------------------------------------------------
# Proposal 0086: PromptManager service-wide default_cache_ttl_seconds
# ---------------------------------------------------------------------------


class _RecordingCacheTtlBackend:
    # Records the cache_ttl_seconds the manager forwarded on the last fetch,
    # so the resolution precedence can be asserted at the backend boundary.
    def __init__(self, prompt: Prompt) -> None:
        self._prompt = prompt
        self.last_cache_ttl_seconds: int | None = None

    async def fetch(
        self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
    ) -> Prompt:
        self.last_cache_ttl_seconds = cache_ttl_seconds
        return self._prompt


def test_construction_rejects_negative_default_cache_ttl() -> None:
    # A negative service-wide default is invalid and is rejected at
    # construction, before any fetch.
    backend = _RecordingCacheTtlBackend(_make_prompt())
    with pytest.raises(ValueError, match="default_cache_ttl_seconds must be >= 0"):
        PromptManager(backend, default_cache_ttl_seconds=-1)


async def test_default_cache_ttl_applies_when_per_call_omitted() -> None:
    # Precedence step 2: an omitted per-call value resolves to the manager
    # default, which is forwarded to the backend verbatim.
    backend = _RecordingCacheTtlBackend(_make_prompt())
    manager = PromptManager(backend, default_cache_ttl_seconds=60)
    await manager.fetch("greeting", "production")
    assert backend.last_cache_ttl_seconds == 60


async def test_explicit_none_selects_default_cache_ttl() -> None:
    # Resolution is presence-independent: an explicit None selects the
    # default exactly as an omitted argument does.
    backend = _RecordingCacheTtlBackend(_make_prompt())
    manager = PromptManager(backend, default_cache_ttl_seconds=60)
    await manager.fetch("greeting", "production", cache_ttl_seconds=None)
    assert backend.last_cache_ttl_seconds == 60


async def test_per_call_zero_overrides_positive_default_cache_ttl() -> None:
    # Precedence step 1: an explicit per-call value wins, so a 0 force-fresh
    # overrides a positive manager default.
    backend = _RecordingCacheTtlBackend(_make_prompt())
    manager = PromptManager(backend, default_cache_ttl_seconds=60)
    await manager.fetch("greeting", "production", cache_ttl_seconds=0)
    assert backend.last_cache_ttl_seconds == 0


async def test_no_default_forwards_none_cache_ttl() -> None:
    # Precedence step 3: with no manager default and no per-call value, None
    # is forwarded, so the backend's own caching governs (unchanged).
    backend = _RecordingCacheTtlBackend(_make_prompt())
    manager = PromptManager(backend)
    await manager.fetch("greeting", "production")
    assert backend.last_cache_ttl_seconds is None


async def test_zero_default_cache_ttl_is_valid_and_forwarded() -> None:
    # A default of 0 is valid (force-fresh-always): accepted at construction,
    # and an omitted per-call value resolves to 0, forwarded to the backend.
    backend = _RecordingCacheTtlBackend(_make_prompt())
    manager = PromptManager(backend, default_cache_ttl_seconds=0)
    await manager.fetch("greeting", "production")
    assert backend.last_cache_ttl_seconds == 0


async def test_get_inherits_default_cache_ttl() -> None:
    # get() delegates to fetch(), so the manager default resolves for get()
    # too when the per-call value is omitted.
    backend = _RecordingCacheTtlBackend(_make_prompt())
    manager = PromptManager(backend, default_cache_ttl_seconds=60)
    await manager.get("greeting", "production", {"user": "Alice"})
    assert backend.last_cache_ttl_seconds == 60


# ---------------------------------------------------------------------------
# FilesystemPromptBackend
# ---------------------------------------------------------------------------


async def test_filesystem_backend_fetch_success(tmp_path: Path) -> None:
    label_dir = tmp_path / "production"
    label_dir.mkdir()
    (label_dir / "greeting.j2").write_text("Hello, {{ user }}!", encoding="utf-8")

    backend = FilesystemPromptBackend(tmp_path)
    prompt = await backend.fetch("greeting", "production")
    assert isinstance(prompt, TextPrompt)
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
        task = asyncio.create_task(_read_in_task())
        result = await task
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

        async def fetch(
            self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
        ) -> Prompt:
            self.calls += 1
            return prompt

    class _Second:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch(
            self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
        ) -> Prompt:
            self.calls += 1
            return prompt

    first = _Hit()
    second = _Second()
    manager = PromptManager(first, second)
    await manager.fetch("greeting", "production")
    assert first.calls == 1
    assert second.calls == 0


def test_manager_render_caches_compiled_templates_by_hash() -> None:
    prompt = _make_prompt()

    class _Backend:
        async def fetch(
            self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
        ) -> Prompt:
            return prompt

    manager = PromptManager(_Backend())
    manager.render(prompt, {"user": "Alice"})
    assert prompt.template_hash in manager._template_cache  # noqa: SLF001
    cached = manager._template_cache[prompt.template_hash]  # noqa: SLF001
    # Second render reuses the same compiled Template instance.
    manager.render(prompt, {"user": "Bob"})
    assert manager._template_cache[prompt.template_hash] is cached  # noqa: SLF001


async def test_manager_render_signature_returns_user_message() -> None:
    prompt = _make_prompt()

    class _Backend:
        async def fetch(
            self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
        ) -> Prompt:
            return prompt

    manager = PromptManager(_Backend())
    result = manager.render(prompt, {"user": "Alice"})
    assert isinstance(result.messages[0], UserMessage)
    msg_content: Any = result.messages[0].content
    assert msg_content == "Hello, Alice!"


# Wish 5 (proposal 0033 python-side ergonomic): the StrictUndefined
# default matches spec §8 (was §7), but callers MAY opt out by passing
# a different Jinja Undefined subclass at PromptManager construction.


def test_manager_jinja_undefined_opt_out_renders_empty_for_missing_var() -> None:
    import jinja2

    from openarmature.prompts import PromptManager

    prompt = TextPrompt(
        name="opt_out",
        version="v1",
        label="production",
        template="Hello, {{ user }}!",
        template_hash="sha256:opt-out",
        fetched_at=datetime.now(UTC),
    )
    manager = PromptManager(_StubBackend(prompt), jinja_undefined=jinja2.Undefined)
    result = manager.render(prompt, {})  # `user` deliberately omitted
    msg_content: Any = result.messages[0].content
    # Default Jinja Undefined renders to empty string; StrictUndefined
    # would have raised PromptRenderError here.
    assert msg_content == "Hello, !"


# Wish 1 (proposal 0033 python-side ergonomic): FilesystemPromptBackend
# accepts a ``layout`` constructor flag. ``per-label`` (default) keeps
# v0.5.0 behavior; ``flat`` reads `<root>/<name>.j2` ignoring label and
# returns the requested label on the resulting Prompt verbatim.


async def test_filesystem_backend_flat_layout(tmp_path: Path) -> None:
    (tmp_path / "greet.j2").write_text("Hello, {{ user }}!", encoding="utf-8")
    backend = FilesystemPromptBackend(tmp_path, layout="flat")

    # Both label requests return the same template; .label echoes the request.
    p_prod = await backend.fetch("greet", "production")
    p_stage = await backend.fetch("greet", "staging")

    assert isinstance(p_prod, TextPrompt) and isinstance(p_stage, TextPrompt)
    assert p_prod.template == p_stage.template == "Hello, {{ user }}!"
    assert p_prod.label == "production"
    assert p_stage.label == "staging"


# Spec §5 informative filesystem-sidecar convention. The
# FilesystemPromptBackend opts in via ``sampling_source``.


async def test_filesystem_backend_per_prompt_sidecar(tmp_path: Path) -> None:
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "summarize.j2").write_text("Summarize: {{ text }}", encoding="utf-8")
    (tmp_path / "production" / "summarize.config.json").write_text(
        '{"temperature": 0.0, "max_tokens": 256, "extras": {"repetition_penalty": 1.05}}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(tmp_path, sampling_source="per-prompt-sidecar")
    prompt = await backend.fetch("summarize", "production")

    assert prompt.sampling is not None
    assert prompt.sampling.temperature == 0.0
    assert prompt.sampling.max_tokens == 256
    # Vendor extra rides through the extras-allow bag.
    assert (prompt.sampling.model_extra or {}).get("repetition_penalty") == 1.05


async def test_filesystem_backend_unified_sampling(tmp_path: Path) -> None:
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    (tmp_path / "production" / "extract.j2").write_text("Extract: {{ text }}", encoding="utf-8")
    (tmp_path / "prompt_configs.json").write_text(
        '{"classify": {"temperature": 0.0}, "extract": {"temperature": 0.7, "max_tokens": 1024}}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(tmp_path, sampling_source="unified")

    classify = await backend.fetch("classify", "production")
    extract = await backend.fetch("extract", "production")

    assert classify.sampling is not None
    assert classify.sampling.temperature == 0.0
    assert extract.sampling is not None
    assert extract.sampling.max_tokens == 1024


# Proposal 0083: the token_budget sub-object rides the SAME sidecar as
# sampling; ``token_budget_source`` gates it with the same three values.


async def test_filesystem_backend_token_budget_per_prompt_sidecar(tmp_path: Path) -> None:
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    # One sidecar carries both surfaces: flat sampling keys + a token_budget
    # sub-object sibling. sampling reads the flat keys and MUST NOT swallow
    # token_budget; token_budget reads only the sub-object.
    (tmp_path / "production" / "classify.config.json").write_text(
        '{"temperature": 0.2, "token_budget": {"input_max_tokens": 10}}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(
        tmp_path,
        sampling_source="per-prompt-sidecar",
        token_budget_source="per-prompt-sidecar",
    )
    prompt = await backend.fetch("classify", "production")

    assert prompt.token_budget is not None
    assert prompt.token_budget.input_max_tokens == 10
    assert prompt.token_budget.total_max_tokens is None
    # sampling excludes the token_budget sub-object (not a sampling field).
    assert prompt.sampling is not None
    assert prompt.sampling.temperature == 0.2
    assert "token_budget" not in (prompt.sampling.model_extra or {})


async def test_filesystem_backend_token_budget_absent_sidecar(tmp_path: Path) -> None:
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    (tmp_path / "production" / "classify.config.json").write_text(
        '{"temperature": 0.2}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(
        tmp_path,
        sampling_source="per-prompt-sidecar",
        token_budget_source="per-prompt-sidecar",
    )
    prompt = await backend.fetch("classify", "production")

    # Sidecar present but declares no token_budget -> None.
    assert prompt.token_budget is None
    assert prompt.sampling is not None


async def test_filesystem_backend_token_budget_source_none_leaves_budget_none(tmp_path: Path) -> None:
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    (tmp_path / "production" / "classify.config.json").write_text(
        '{"token_budget": {"total_max_tokens": 25}}',
        encoding="utf-8",
    )

    # Default token_budget_source="none": the sidecar budget is not sourced.
    backend = FilesystemPromptBackend(tmp_path, sampling_source="per-prompt-sidecar")
    prompt = await backend.fetch("classify", "production")

    assert prompt.token_budget is None


async def test_filesystem_backend_token_budget_unified(tmp_path: Path) -> None:
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    # The unified prompt_configs.json entry carries the token_budget sub-object
    # alongside the flat sampling keys, exactly like the per-prompt sidecar.
    (tmp_path / "prompt_configs.json").write_text(
        '{"classify": {"temperature": 0.0, '
        '"token_budget": {"input_max_tokens": 10, "total_max_tokens": 40}}}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(
        tmp_path,
        sampling_source="unified",
        token_budget_source="unified",
    )
    prompt = await backend.fetch("classify", "production")

    assert prompt.token_budget is not None
    assert prompt.token_budget.input_max_tokens == 10
    assert prompt.token_budget.total_max_tokens == 40
    assert prompt.sampling is not None
    assert prompt.sampling.temperature == 0.0


async def test_filesystem_backend_malformed_token_budget_is_fallback_eligible(tmp_path: Path) -> None:
    # A malformed advisory token_budget (a string, not an object) surfaces as the
    # fallback-eligible PromptStoreUnavailable -- NOT a bare exception that would
    # bypass PromptManager's multi-backend fallback and crash the LLM call.
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    (tmp_path / "production" / "classify.config.json").write_text(
        '{"token_budget": "1000"}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(tmp_path, token_budget_source="per-prompt-sidecar")
    with pytest.raises(PromptStoreUnavailable):
        await backend.fetch("classify", "production")


async def test_filesystem_backend_negative_bound_is_fallback_eligible(tmp_path: Path) -> None:
    # A bound failing validation (negative; TokenBudget requires ge=0) is also
    # converted to the fallback-eligible PromptStoreUnavailable, not a raw
    # pydantic ValidationError.
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    (tmp_path / "production" / "classify.config.json").write_text(
        '{"token_budget": {"input_max_tokens": -5}}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(tmp_path, token_budget_source="per-prompt-sidecar")
    with pytest.raises(PromptStoreUnavailable):
        await backend.fetch("classify", "production")


async def test_filesystem_backend_empty_token_budget_collapses_to_none(tmp_path: Path) -> None:
    # An empty / all-null token_budget object is "no bound declared" -> None,
    # not a non-null all-null record (parity with the langfuse backend).
    (tmp_path / "production").mkdir()
    (tmp_path / "production" / "classify.j2").write_text("Classify: {{ topic }}", encoding="utf-8")
    (tmp_path / "production" / "classify.config.json").write_text(
        '{"token_budget": {}}',
        encoding="utf-8",
    )

    backend = FilesystemPromptBackend(tmp_path, token_budget_source="per-prompt-sidecar")
    prompt = await backend.fetch("classify", "production")
    assert prompt.token_budget is None


def test_langfuse_token_budget_from_config() -> None:
    # Direct coverage of the langfuse backend's config sourcing (proposal 0083):
    # present -> TokenBudget; absent / non-object / no recognized bound / all-null
    # -> None; a malformed bound is tolerated (dropped to None) -- the intentional
    # divergence from the filesystem backend's fail-loud sidecar posture.
    from openarmature.prompts import TokenBudget
    from openarmature.prompts.backends.langfuse import _token_budget_from_config

    assert _token_budget_from_config({"token_budget": {"input_max_tokens": 10}}) == TokenBudget(
        input_max_tokens=10
    )
    assert _token_budget_from_config(None) is None
    assert _token_budget_from_config({}) is None
    assert _token_budget_from_config({"token_budget": "1000"}) is None
    assert _token_budget_from_config({"token_budget": {"unknown": 5}}) is None
    assert _token_budget_from_config({"token_budget": {"input_max_tokens": None}}) is None
    assert _token_budget_from_config({"token_budget": {"input_max_tokens": -5}}) is None


# LabelResolver fallback chain — covered by fixture 015 end-to-end,
# but the resolver class is python-only and a focused unit test
# documents the precedence rules in code.


def test_mapping_label_resolver_per_name_override() -> None:
    from openarmature.prompts import MappingLabelResolver

    resolver = MappingLabelResolver({"default": "production", "experimental": "staging"})
    assert resolver.resolve("experimental") == "staging"


def test_mapping_label_resolver_default_override() -> None:
    from openarmature.prompts import MappingLabelResolver

    resolver = MappingLabelResolver({"default": "canary", "other": "staging"})
    assert resolver.resolve("anything-not-listed") == "canary"


def test_mapping_label_resolver_spec_fallback_when_no_default() -> None:
    from openarmature.prompts import MappingLabelResolver

    resolver = MappingLabelResolver({"experimental": "staging"})
    assert resolver.resolve("anything-not-listed") == "production"


# ---------------------------------------------------------------------------
# Proposal 0046: construction-time §11 enforcement (ergonomic bonus
# atop the spec-normative render-time checks per spec msg-07 Q3).
# ---------------------------------------------------------------------------


def test_placeholder_segment_rejects_invalid_name_at_construction() -> None:
    # Spec §3.1 placeholder name regex: ``[A-Za-z_][A-Za-z0-9_]*``.
    # PlaceholderSegment enforces this at construction time as a
    # faster-feedback ergonomic bonus.
    from pydantic import ValidationError

    from openarmature.prompts import PlaceholderSegment

    with pytest.raises(ValidationError, match=r"placeholder name '1history' MUST match"):
        PlaceholderSegment(placeholder="1history")


def test_placeholder_segment_accepts_valid_name() -> None:
    from openarmature.prompts import PlaceholderSegment

    seg = PlaceholderSegment(placeholder="chat_history_v2")
    assert seg.placeholder == "chat_history_v2"


def test_chat_prompt_rejects_duplicate_placeholder_at_construction() -> None:
    # Spec §3.1: placeholder names MUST be unique within a single
    # chat_template.  Construction-time enforcement.
    from datetime import UTC, datetime

    from pydantic import ValidationError

    from openarmature.prompts import (
        ChatPrompt,
        ContentSegment,
        PlaceholderSegment,
    )

    with pytest.raises(ValidationError, match=r"duplicate placeholder name 'history'"):
        ChatPrompt(
            name="dup",
            version="v1",
            label="production",
            template_hash="sha256:dup-v1",
            fetched_at=datetime.now(UTC),
            chat_template=[
                ContentSegment(role="system", content="hi"),
                PlaceholderSegment(placeholder="history"),
                PlaceholderSegment(placeholder="history"),
            ],
        )


def test_content_segment_rejects_image_in_non_user_role_at_construction() -> None:
    # Spec §11 role-block compatibility: image blocks are user-only.
    # Construction-time enforcement.
    from pydantic import ValidationError

    from openarmature.prompts import ContentSegment, ImageURLBlockTemplate, TextBlockTemplate

    with pytest.raises(ValidationError, match=r"image blocks are user-only"):
        ContentSegment(
            role="system",
            content=[
                TextBlockTemplate(text="Context:"),
                ImageURLBlockTemplate(url="https://example.invalid/diagram.png"),
            ],
        )


def test_content_segment_rejects_empty_block_list_at_construction() -> None:
    from pydantic import ValidationError

    from openarmature.prompts import ContentSegment

    with pytest.raises(ValidationError, match=r"block list MUST be non-empty"):
        ContentSegment(role="user", content=[])


async def test_chat_segment_template_cache_is_content_stable() -> None:
    # Regression for cache-key stability: rendering the SAME segment
    # text twice MUST hit the cached compiled jinja Template; the
    # cache key derives from a SHA-256 of the segment source so it's
    # stable across process restarts (not the salted built-in
    # ``hash()``).
    from datetime import UTC, datetime

    from openarmature.prompts import (
        ChatPrompt,
        ContentSegment,
        PromptManager,
    )

    backend = _DummyBackend()
    manager = PromptManager(backend)
    prompt = ChatPrompt(
        name="cached",
        version="v1",
        label="production",
        template_hash="sha256:cached-v1",
        fetched_at=datetime.now(UTC),
        chat_template=[
            ContentSegment(role="system", content="hi {{ user }}"),
            ContentSegment(role="user", content="ask {{ q }}"),
        ],
    )

    # First render seeds the cache.
    manager.render(prompt, {"user": "Alice", "q": "?"})
    cached_keys_after_first = set(manager._template_cache.keys())  # noqa: SLF001

    # Second render — same content text → MUST NOT add new entries.
    manager.render(prompt, {"user": "Bob", "q": "?"})
    cached_keys_after_second = set(manager._template_cache.keys())  # noqa: SLF001

    assert cached_keys_after_first == cached_keys_after_second, (
        f"chat-segment cache leaked entries on second render; "
        f"new keys: {cached_keys_after_second - cached_keys_after_first!r}"
    )

    # All cache keys for chat segments are SHA-256 strings (start
    # with the ``sha256:`` prefix that ``compute_template_hash``
    # emits) — process-stable, NOT salted python ``hash()`` ints.
    chat_segment_keys = cached_keys_after_first
    assert chat_segment_keys, "expected chat-segment cache entries to be populated"
    for key in chat_segment_keys:
        assert isinstance(key, str), f"cache key {key!r} is not a string"
        assert key.startswith("sha256:"), (
            f"chat-segment cache key {key!r} is not SHA-256-derived; "
            f"likely regressed to process-randomized hash()"
        )


class _DummyBackend:
    async def fetch(
        self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
    ) -> Any:
        raise NotImplementedError


async def test_inline_image_block_rejects_invalid_base64_at_render() -> None:
    # Security hardening: a content-blocks template substitutes a
    # variable into the base64_data field; if the resulting string
    # isn't valid base64 (e.g., a stray comma from a CSV-like
    # variable, padding mangled, non-base64-alphabet chars), the
    # render-time check raises ``prompt_render_error`` rather than
    # letting the malformed payload reach the LLM provider where it
    # would surface as a provider-specific decode error.
    from datetime import UTC, datetime

    from openarmature.prompts import (
        ChatPrompt,
        ContentSegment,
        ImageInlineBlockTemplate,
        PromptManager,
        PromptRenderError,
        TextBlockTemplate,
    )

    backend = _DummyBackend()
    manager = PromptManager(backend)
    prompt = ChatPrompt(
        name="bad_image",
        version="v1",
        label="production",
        template_hash="sha256:bad-image-v1",
        fetched_at=datetime.now(UTC),
        chat_template=[
            ContentSegment(
                role="user",
                content=[
                    TextBlockTemplate(text="Describe:"),
                    ImageInlineBlockTemplate(
                        base64_data="{{ raw }}",
                        media_type="image/png",
                    ),
                ],
            ),
        ],
    )

    with pytest.raises(PromptRenderError, match=r"base64"):
        manager.render(prompt, {"raw": "not!valid base64!!"})


class _StubBackend:
    """Minimal PromptBackend that returns a single canned prompt."""

    def __init__(self, prompt: Prompt) -> None:
        self._prompt = prompt

    async def fetch(
        self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
    ) -> Prompt:
        return self._prompt


# ---------------------------------------------------------------------------
# Proposal 0047 §13: cross-variable substring stability
# ---------------------------------------------------------------------------


def test_cross_variable_substring_stability_text_prompt() -> None:
    # Spec 0047 §13 *Determinism* — *Cross-variable substring stability*
    # (normative clause): the static substring of a rendered output —
    # the portion not derived from variable substitution — MUST be
    # identical across renders that differ ONLY in unrelated variable
    # bindings. Two renders of the same template with different
    # user-bound values flanking a common static segment must agree on
    # that static segment byte-for-byte. Jinja2's StrictUndefined render
    # path satisfies this naturally; the test pins the contract so a
    # future render-time mutation (e.g., introducing context-aware
    # whitespace normalization) would fail loud rather than silently
    # break APC hit rates.
    template = "system: classify the input.\nuser: {{ user_text }}\n\ncontext: {{ context }}\n"
    prompt = _make_prompt(template)
    manager = PromptManager(_StubBackend(prompt))

    result_a = manager.render(prompt, {"user_text": "alice", "context": "ctx1"})
    result_b = manager.render(prompt, {"user_text": "bob", "context": "ctx2"})

    rendered_a = result_a.messages[0].content
    rendered_b = result_b.messages[0].content
    assert isinstance(rendered_a, str) and isinstance(rendered_b, str)

    # The static prefix (everything before the first substitution) MUST
    # be byte-identical across renders.
    static_prefix = "system: classify the input.\nuser: "
    assert rendered_a.startswith(static_prefix)
    assert rendered_b.startswith(static_prefix)
    # The static infix between the two substitutions MUST be byte-
    # identical too.
    static_infix = "\n\ncontext: "
    assert static_infix in rendered_a
    assert static_infix in rendered_b
    # Confirm the substitutions actually landed in their slots (so the
    # test is verifying substring stability, not just unconditional
    # equality on a degenerate render).
    assert "alice" in rendered_a and "bob" in rendered_b
    assert "ctx1" in rendered_a and "ctx2" in rendered_b


def test_cross_variable_substring_stability_chat_prompt() -> None:
    # Spec 0047 §13's substring stability rule applies to the multi-
    # segment chat-prompt variant too — the proposal's normative text
    # calls out "system prefix text, few-shot exchange text, segment
    # role markers" explicitly. Each rendered segment's static portions
    # (the role marker shape + the inter-segment formatting + the
    # template's literal substrings) MUST be byte-identical across
    # renders that differ only in variable bindings.
    from openarmature.prompts import ChatPrompt, ContentSegment

    chat_prompt = ChatPrompt(
        name="classifier",
        version="v1",
        label="production",
        chat_template=[
            ContentSegment(role="system", content="Classify the input as ham or spam."),
            ContentSegment(role="user", content="Subject: {{ subject }}\n\nBody: {{ body }}"),
        ],
        template_hash="sha256:chat-v1",
        fetched_at=datetime.now(UTC),
    )
    manager = PromptManager(_StubBackend(chat_prompt))

    result_a = manager.render(chat_prompt, {"subject": "alice's email", "body": "hello"})
    result_b = manager.render(chat_prompt, {"subject": "bob's email", "body": "world"})

    # Both renders MUST produce the same segment shape (same number of
    # messages, same roles in the same order).
    assert len(result_a.messages) == len(result_b.messages)
    for msg_a, msg_b in zip(result_a.messages, result_b.messages, strict=True):
        assert type(msg_a) is type(msg_b)
    # Static (non-substituted) system segment MUST be byte-identical.
    sys_a = result_a.messages[0].content
    sys_b = result_b.messages[0].content
    assert sys_a == sys_b
    # User segment's static infix between the two substitutions MUST
    # be byte-identical.
    user_a = result_a.messages[1].content
    user_b = result_b.messages[1].content
    assert isinstance(user_a, str) and isinstance(user_b, str)
    static_prefix = "Subject: "
    static_infix = "\n\nBody: "
    assert user_a.startswith(static_prefix) and user_b.startswith(static_prefix)
    assert static_infix in user_a and static_infix in user_b
    # Confirm the substitutions actually differ — guards against a
    # degenerate-equality false pass.
    assert "alice's email" in user_a and "bob's email" in user_b
    assert user_a.endswith("hello") and user_b.endswith("world")
