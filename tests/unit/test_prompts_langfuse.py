"""Unit tests for the Langfuse-backed PromptBackend (text prompts)."""

from __future__ import annotations

from typing import Any, cast

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

    def get_prompt(self, name: str, *, label: str = "production", **_: Any) -> Any:
        self.calls.append((name, label))
        if self._exc is not None:
            raise self._exc
        return self._result


async def test_fetch_text_prompt_maps_to_prompt() -> None:
    client = _text_client(prompt="Hello {{ user }}", version=7, tags=["greeting"])
    backend = LangfusePromptBackend(_FakeClient(result=client))

    prompt = await backend.fetch("greeting", "production")

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


async def test_chat_prompt_raises_not_found() -> None:
    backend = LangfusePromptBackend(_FakeClient(result=_chat_client()))

    with pytest.raises(PromptNotFound) as excinfo:
        await backend.fetch("chatty", "production")

    assert excinfo.value.backend == "langfuse"
    assert "chat prompt" in str(excinfo.value)


async def test_not_found_maps_to_prompt_not_found() -> None:
    backend = LangfusePromptBackend(_FakeClient(exc=NotFoundError("nope")))

    with pytest.raises(PromptNotFound):
        await backend.fetch("missing", "production")


async def test_service_unavailable_maps_to_store_unavailable() -> None:
    backend = LangfusePromptBackend(_FakeClient(exc=ServiceUnavailableError()))

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
