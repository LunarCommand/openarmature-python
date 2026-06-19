"""Unit tests for the Langfuse-backed PromptBackend (text prompts)."""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

pytest.importorskip("langfuse")

from langfuse.api import NotFoundError, ServiceUnavailableError  # noqa: E402
from langfuse.model import (  # noqa: E402
    ChatPromptClient,
    Prompt_Chat,  # pyright: ignore[reportPrivateImportUsage]
    Prompt_Text,  # pyright: ignore[reportPrivateImportUsage]
    TextPromptClient,
)

from openarmature.prompts import PromptManager  # noqa: E402
from openarmature.prompts.backends.langfuse import LangfusePromptBackend  # noqa: E402
from openarmature.prompts.errors import (  # noqa: E402
    PromptNotFound,
    PromptStoreUnavailable,
)

pytestmark = pytest.mark.asyncio


def _text_client(
    *,
    name: str = "greeting",
    version: int = 3,
    prompt: str = "Hello {{ user }}",
    config: dict[str, Any] | None = None,
    labels: list[str] | None = None,
    tags: list[str] | None = None,
) -> TextPromptClient:
    return TextPromptClient(
        Prompt_Text(
            type="text",
            name=name,
            version=version,
            prompt=prompt,
            config=config or {},
            labels=["production"] if labels is None else labels,
            tags=tags or [],
        )
    )


def _chat_client(*, name: str = "chatty", version: int = 1) -> ChatPromptClient:
    return ChatPromptClient(
        Prompt_Chat(
            type="chat",
            name=name,
            version=version,
            prompt=cast(Any, [{"role": "system", "content": "hi {{ user }}"}]),
            config={},
            labels=["production"],
            tags=[],
        )
    )


class _FakeClient:
    """Stands in for ``langfuse.Langfuse`` exposing only ``get_prompt``."""

    def __init__(self, *, result: Any = None, exc: BaseException | None = None) -> None:
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, str]] = []
        self.last_cache_ttl_seconds: int | None = None

    def get_prompt(
        self, name: str, *, label: str = "production", cache_ttl_seconds: int | None = None, **_: Any
    ) -> Any:
        self.calls.append((name, label))
        self.last_cache_ttl_seconds = cache_ttl_seconds
        if self._exc is not None:
            raise self._exc
        return self._result


async def test_fetch_text_prompt_maps_to_prompt() -> None:
    from openarmature.prompts import TextPrompt

    client = _text_client(prompt="Hello {{ user }}", version=7, tags=["greeting"])
    backend = LangfusePromptBackend(_FakeClient(result=client))

    prompt = await backend.fetch("greeting", "production")

    assert isinstance(prompt, TextPrompt)
    assert prompt.name == "greeting"
    assert prompt.version == "7"
    assert prompt.label == "production"
    assert prompt.template == "Hello {{ user }}"
    assert prompt.template_hash.startswith("sha256:")
    assert prompt.observability_entities is not None
    assert prompt.observability_entities["langfuse_prompt"] is client
    assert prompt.metadata is not None
    assert prompt.metadata["langfuse_version"] == 7
    assert prompt.metadata["langfuse_tags"] == ["greeting"]


async def test_fetch_passes_label_through() -> None:
    fake = _FakeClient(result=_text_client())
    backend = LangfusePromptBackend(fake)

    await backend.fetch("greeting", "staging")

    assert fake.calls == [("greeting", "staging")]


async def test_fetch_forwards_cache_ttl_seconds_to_sdk() -> None:
    # The langfuse backend forwards the read-side cache control to the
    # SDK's own get_prompt cache; an absent value forwards None (the
    # SDK's default TTL), preserving prior behavior.
    fake = _FakeClient(result=_text_client())
    backend = LangfusePromptBackend(fake)

    await backend.fetch("greeting", "production", cache_ttl_seconds=42)
    assert fake.last_cache_ttl_seconds == 42

    await backend.fetch("greeting", "production")
    assert fake.last_cache_ttl_seconds is None


async def test_chat_prompt_maps_to_chat_prompt() -> None:
    # Per proposal 0046 (v0.38.0): Langfuse chat prompts now map to
    # a ChatPrompt with one ContentSegment per Langfuse chat message
    # (placeholder markers map to PlaceholderSegment).
    from openarmature.prompts import ChatPrompt, ContentSegment

    backend = LangfusePromptBackend(_FakeClient(result=_chat_client()))

    prompt = await backend.fetch("chatty", "production")

    assert isinstance(prompt, ChatPrompt)
    assert prompt.name == "chatty"
    assert prompt.version == "1"
    assert len(prompt.chat_template) == 1
    segment = prompt.chat_template[0]
    assert isinstance(segment, ContentSegment)
    assert segment.role == "system"
    assert segment.content == "hi {{ user }}"


def _chat_client_with_raw_prompt(
    raw_prompt: list[Any], *, name: str = "ext", version: int = 1
) -> ChatPromptClient:
    """Build a ChatPromptClient whose ``.prompt`` carries the raw
    entries we want to test against — bypassing the SDK's own
    ``__init__`` filter (which would otherwise drop everything our
    backend doesn't recognize before our code sees it).  Simulates
    a future SDK that emits shapes our current backend hasn't
    learned to handle."""
    client = _chat_client(name=name, version=version)
    client.prompt = cast(Any, raw_prompt)
    return client


async def test_chat_prompt_with_unsupported_entry_fails_closed_at_fetch() -> None:
    # Regression: an unknown entry shape in a Langfuse chat prompt
    # MUST raise PromptNotFound at fetch time rather than silently
    # drop, otherwise a Langfuse SDK extension would produce a
    # degraded rendered prompt with no signal to the caller —
    # exactly the kind of bug that changes model behavior invisibly.
    chat_result = _chat_client_with_raw_prompt(
        [
            {"role": "system", "content": "ok"},
            # Unknown shape — not a placeholder, role missing.
            {"weird_extension": "future_sdk_thing"},
        ],
        name="unknown_shape",
    )
    backend = LangfusePromptBackend(_FakeClient(result=chat_result))

    with pytest.raises(PromptNotFound, match=r"unsupported role/content shape"):
        await backend.fetch("unknown_shape", "production")


async def test_chat_prompt_with_placeholder_missing_name_fails_closed_at_fetch() -> None:
    # Sibling regression: a placeholder marker missing the ``name``
    # key MUST raise PromptNotFound at fetch time.
    chat_result = _chat_client_with_raw_prompt(
        [
            {"role": "system", "content": "set up"},
            {"type": "placeholder"},  # missing ``name``
        ],
        name="bad_placeholder_shape",
    )
    backend = LangfusePromptBackend(_FakeClient(result=chat_result))

    with pytest.raises(PromptNotFound, match=r"placeholder entry missing"):
        await backend.fetch("bad_placeholder_shape", "production")


async def test_chat_prompt_with_malformed_placeholder_fetches_then_raises_at_render() -> None:
    # Regression for the spec §11 timing contract: a Langfuse-stored
    # chat prompt carrying a placeholder name that VIOLATES the §3.1
    # regex (e.g., leading digit) MUST NOT raise at fetch time —
    # render is the spec-normative error trigger.  The backend uses
    # ``PlaceholderSegment.model_construct`` to bypass construction-
    # time validators so the offending name reaches the render path
    # before surfacing.
    from openarmature.prompts import ChatPrompt, PlaceholderSegment, PromptManager, PromptRenderError

    chat_result = _chat_client_with_raw_prompt(
        [
            {"role": "system", "content": "set up"},
            {"type": "placeholder", "name": "1history"},  # leading digit — invalid per §3.1
            {"role": "user", "content": "go"},
        ],
        name="bad_placeholder",
    )
    backend = LangfusePromptBackend(_FakeClient(result=chat_result))

    # Fetch MUST succeed — construction-time check is bypassed.
    prompt = await backend.fetch("bad_placeholder", "production")
    assert isinstance(prompt, ChatPrompt)
    placeholder_seg = next(seg for seg in prompt.chat_template if isinstance(seg, PlaceholderSegment))
    assert placeholder_seg.placeholder == "1history"

    # Render MUST raise — the §11 spec-normative trigger.
    manager = PromptManager(backend)
    with pytest.raises(PromptRenderError, match=r"invalid placeholder name '1history'"):
        manager.render(prompt)


async def test_not_found_maps_to_prompt_not_found() -> None:
    backend = LangfusePromptBackend(_FakeClient(exc=NotFoundError("nope")))

    with pytest.raises(PromptNotFound):
        await backend.fetch("missing", "production")


async def test_service_unavailable_maps_to_store_unavailable() -> None:
    backend = LangfusePromptBackend(_FakeClient(exc=ServiceUnavailableError()))

    with pytest.raises(PromptStoreUnavailable):
        await backend.fetch("greeting", "production")


async def test_transport_error_maps_to_store_unavailable() -> None:
    # A connect/read/timeout/network failure surfaces as a raw httpx
    # TransportError (no HTTP response to map to a typed SDK error); it
    # must become PromptStoreUnavailable so PromptManager can fall back.
    backend = LangfusePromptBackend(_FakeClient(exc=httpx.ConnectTimeout("timed out")))

    with pytest.raises(PromptStoreUnavailable):
        await backend.fetch("greeting", "production")


async def test_sampling_extracted_from_config() -> None:
    client = _text_client(config={"temperature": 0.0, "max_tokens": 256, "model": "gpt-4o"})
    backend = LangfusePromptBackend(_FakeClient(result=client))

    prompt = await backend.fetch("greeting", "production")

    assert prompt.sampling is not None
    assert prompt.sampling.temperature == 0.0
    assert prompt.sampling.max_tokens == 256
    # Non-sampling config keys are not lifted into sampling, but the
    # full config is preserved under metadata.
    assert prompt.metadata is not None
    assert prompt.metadata["langfuse_config"]["model"] == "gpt-4o"


async def test_no_sampling_config_yields_none() -> None:
    backend = LangfusePromptBackend(_FakeClient(result=_text_client(config={})))

    prompt = await backend.fetch("greeting", "production")

    assert prompt.sampling is None


async def test_fetched_prompt_renders_through_manager() -> None:
    backend = LangfusePromptBackend(_FakeClient(result=_text_client(prompt="Hi {{ user }}")))
    manager = PromptManager(backend)

    prompt = await manager.fetch("greeting", "production")
    result = manager.render(prompt, {"user": "Alice"})

    assert len(result.messages) == 1
    assert result.messages[0].content == "Hi Alice"
