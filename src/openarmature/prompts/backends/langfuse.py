"""Langfuse-backed PromptBackend (text prompts).

Fetches prompts from Langfuse's prompt registry through OA's
``PromptManager``. Gated behind the ``[langfuse]`` extra; import this
module only when ``langfuse`` is installed (``backends/__init__`` does
not import it, so the base package stays langfuse-free).

v1 supports Langfuse TEXT prompts. A Langfuse CHAT prompt raises
``PromptNotFound`` because OA's render produces a single user message
today; multi-message (chat) prompt support is tracked for a later
release.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Protocol

from langfuse.api import NotFoundError, ServiceUnavailableError
from langfuse.model import ChatPromptClient, TextPromptClient

from ..errors import PromptNotFound, PromptStoreUnavailable
from ..hashing import compute_template_hash
from ..prompt import Prompt, SamplingConfig


class LangfusePromptClient(Protocol):
    """The minimal Langfuse prompt-fetch surface this backend needs.

    ``langfuse.Langfuse`` satisfies it structurally (its ``get_prompt``
    has additional optional parameters), so callers pass a real client;
    tests can supply a lightweight fake.
    """

    def get_prompt(self, name: str, *, label: str = "production") -> TextPromptClient | ChatPromptClient: ...


# Langfuse prompt `config` keys that line up with SamplingConfig's
# declared fields. Only these are lifted into `Prompt.sampling`; the
# full config is preserved under `Prompt.metadata` so nothing is lost.
_SAMPLING_FIELDS = (
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "stop_sequences",
)


class LangfusePromptBackend:
    """Reads prompts from Langfuse's prompt registry.

    Constructed with a caller-supplied ``langfuse.Langfuse`` client, so
    it shares one client (one connection pool, one flush thread) with a
    :class:`~openarmature.observability.langfuse.LangfuseObserver` built
    on the same instance::

        from langfuse import Langfuse
        from openarmature.prompts import PromptManager
        from openarmature.prompts.backends.langfuse import LangfusePromptBackend

        client = Langfuse(public_key="pk-lf-...", secret_key="sk-lf-...")
        manager = PromptManager(LangfusePromptBackend(client))

    ``fetch`` is reentrant and does not render; the manager renders.
    The returned ``Prompt`` carries the raw Langfuse template (Langfuse
    ``{{var}}`` placeholders are Jinja2-compatible, so OA's render
    applies unchanged), plus the Langfuse SDK Prompt object under
    ``observability_entities['langfuse_prompt']`` so the observability
    Generation -> Prompt link fires automatically.
    """

    def __init__(self, client: LangfusePromptClient) -> None:
        self._client = client

    async def fetch(self, name: str, label: str = "production") -> Prompt:
        # The Langfuse SDK's get_prompt is synchronous (and does its own
        # client-side caching); run it off the event loop.
        result = await asyncio.to_thread(self._get_prompt, name, label)

        if isinstance(result, ChatPromptClient):
            raise PromptNotFound(
                f"prompt ({name!r}, {label!r}) is a Langfuse chat prompt; "
                "the Langfuse backend supports text prompts only in this "
                "release (multi-message prompt support is planned)",
                name=name,
                label=label,
                backend="langfuse",
            )

        template = result.prompt
        template_hash = compute_template_hash(template)
        return Prompt(
            name=name,
            version=str(result.version),
            label=label,
            template=template,
            template_hash=template_hash,
            fetched_at=datetime.now(UTC),
            sampling=_sampling_from_config(result.config),
            observability_entities={"langfuse_prompt": result},
            metadata=_metadata_from(result),
        )

    def _get_prompt(self, name: str, label: str) -> TextPromptClient | ChatPromptClient:
        try:
            return self._client.get_prompt(name, label=label)
        except NotFoundError as exc:
            raise PromptNotFound(
                f"prompt ({name!r}, {label!r}) not found in Langfuse",
                name=name,
                label=label,
                backend="langfuse",
            ) from exc
        except ServiceUnavailableError as exc:
            raise PromptStoreUnavailable(
                f"Langfuse unavailable fetching ({name!r}, {label!r}): {exc}",
                name=name,
                label=label,
            ) from exc


def _sampling_from_config(config: dict[str, Any] | None) -> SamplingConfig | None:
    if not config:
        return None
    declared = {k: config[k] for k in _SAMPLING_FIELDS if k in config}
    if not declared:
        return None
    return SamplingConfig(**declared)


def _metadata_from(result: TextPromptClient) -> dict[str, Any]:
    # Preserve Langfuse-side attribution. `config` is kept whole here
    # even though sampling fields are also lifted to `Prompt.sampling`,
    # so non-sampling config keys aren't dropped.
    meta: dict[str, Any] = {
        "langfuse_version": result.version,
        "langfuse_labels": result.labels,
        "langfuse_tags": result.tags,
    }
    if result.config:
        meta["langfuse_config"] = result.config
    if result.commit_message is not None:
        meta["langfuse_commit_message"] = result.commit_message
    return meta
