"""Langfuse-backed PromptBackend (text + chat prompts).

Fetches prompts from Langfuse's prompt registry through OA's
``PromptManager``. Gated behind the ``[langfuse]`` extra; import this
module only when ``langfuse`` is installed (``backends/__init__`` does
not import it, so the base package stays langfuse-free).

Both Langfuse TEXT and CHAT prompts are supported.  Text prompts
return a :class:`TextPrompt`; chat prompts return a
:class:`ChatPrompt` with one :class:`ContentSegment` per Langfuse
chat message.  Langfuse chat placeholders map to
:class:`PlaceholderSegment` entries.
"""
# Proposal 0046 (v0.38.0): Langfuse text + chat prompt support.

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import httpx
from langfuse.api import NotFoundError, ServiceUnavailableError
from langfuse.model import ChatPromptClient, TextPromptClient
from pydantic import ValidationError

from ..errors import PromptNotFound, PromptStoreUnavailable
from ..hashing import compute_template_hash
from ..prompt import (
    ChatPrompt,
    ChatSegment,
    ContentSegment,
    PlaceholderSegment,
    Prompt,
    SamplingConfig,
    TextPrompt,
    TokenBudget,
)


class LangfusePromptClient(Protocol):
    """The minimal Langfuse prompt-fetch surface this backend needs.

    ``langfuse.Langfuse`` satisfies it structurally (its ``get_prompt``
    has additional optional parameters), so callers pass a real client;
    tests can supply a lightweight fake.
    """

    def get_prompt(
        self, name: str, *, label: str = "production", cache_ttl_seconds: int | None = None
    ) -> TextPromptClient | ChatPromptClient: ...


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

# Langfuse prompt `config.token_budget` sub-object keys that line up with
# TokenBudget's declared fields (proposal 0083). Mirrors `_SAMPLING_FIELDS`:
# the budget lives under a `token_budget` sub-object in the Langfuse config
# (sibling to the flat sampling keys), matching the filesystem sidecar shape.
_TOKEN_BUDGET_FIELDS = (
    "input_max_tokens",
    "total_max_tokens",
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

    async def fetch(
        self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
    ) -> Prompt:
        # The Langfuse SDK's get_prompt is synchronous (and does its own
        # client-side caching); run it off the event loop. The proposal
        # 0072 cache_ttl_seconds control forwards to that SDK cache:
        # None = SDK default, 0 = no cache (fresh), N = N-second bound.
        result = await asyncio.to_thread(self._get_prompt, name, label, cache_ttl_seconds)

        if isinstance(result, ChatPromptClient):
            normalized = _normalized_langfuse_entries(result.prompt, name=name, label=label)
            chat_template = list(_chat_segments_from_normalized(normalized))
            template_hash = compute_template_hash(json.dumps(normalized, sort_keys=True))
            # ``ChatPrompt.model_construct`` is required (not the
            # plain constructor): pydantic re-runs validators on
            # nested field values when validating the outer model,
            # so a placeholder name we bypassed at the
            # ``PlaceholderSegment`` level would still trip the
            # regex check during ChatPrompt construction.  Bypass
            # the outer validators too so the malformed input
            # reaches render-time (the spec-normative §11 error
            # trigger).
            return ChatPrompt.model_construct(
                kind="chat",
                name=name,
                version=str(result.version),
                label=label,
                chat_template=chat_template,
                template_hash=template_hash,
                fetched_at=datetime.now(UTC),
                sampling=_sampling_from_config(result.config),
                token_budget=_token_budget_from_config(result.config),
                observability_entities={"langfuse_prompt": result},
                metadata=_metadata_from(result),
            )

        template = result.prompt
        template_hash = compute_template_hash(template)
        return TextPrompt(
            name=name,
            version=str(result.version),
            label=label,
            template=template,
            template_hash=template_hash,
            fetched_at=datetime.now(UTC),
            sampling=_sampling_from_config(result.config),
            token_budget=_token_budget_from_config(result.config),
            observability_entities={"langfuse_prompt": result},
            metadata=_metadata_from(result),
        )

    def _get_prompt(
        self, name: str, label: str, cache_ttl_seconds: int | None = None
    ) -> TextPromptClient | ChatPromptClient:
        try:
            return self._client.get_prompt(name, label=label, cache_ttl_seconds=cache_ttl_seconds)
        except NotFoundError as exc:
            raise PromptNotFound(
                f"prompt ({name!r}, {label!r}) not found in Langfuse",
                name=name,
                label=label,
                backend="langfuse",
            ) from exc
        except (ServiceUnavailableError, httpx.TransportError) as exc:
            # 503 plus transport-level failures (connect/read/timeout/
            # network): the SDK surfaces raw httpx errors when there's no
            # HTTP response to map to a typed error. Per the PromptBackend
            # contract these are unavailability, so the manager can fall
            # back. 4xx auth and other errors still propagate.
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


def _token_budget_from_config(config: dict[str, Any] | None) -> TokenBudget | None:
    # The budget is a `token_budget` sub-object in the Langfuse config;
    # only the declared TokenBudget fields are lifted (TokenBudget forbids
    # extras, so an unknown config key would otherwise reject the fetch).
    # Absent / non-object / no recognized bound -> no budget.
    if not config:
        return None
    raw = config.get("token_budget")
    if not isinstance(raw, dict):
        return None
    budget = cast("dict[str, Any]", raw)
    declared = {k: budget[k] for k in _TOKEN_BUDGET_FIELDS if k in budget}
    if not declared:
        return None
    # The Langfuse config is a remote-service payload: tolerate a malformed
    # advisory bound (e.g. a negative value) by dropping the budget rather than
    # failing the fetch. The filesystem backend, whose sidecar is operator-
    # authored, fails loud (fallback-eligible) instead -- an intentional
    # divergence by data-source trust model, flagged to spec for confirmation.
    try:
        parsed = TokenBudget(**declared)
    except ValidationError:
        return None
    # An all-null budget (bounds present but explicitly null) is "no bound
    # declared" -> None, not a non-null all-null record.
    if parsed.input_max_tokens is None and parsed.total_max_tokens is None:
        return None
    return parsed


def _normalized_langfuse_entries(raw: Iterable[Any], *, name: str, label: str) -> list[dict[str, Any]]:
    """Normalize a Langfuse ``ChatPromptClient.prompt`` list to OA
    canonical entry dicts.  Each output entry is either a content
    message ``{"role": ..., "content": ...}`` or a placeholder
    marker ``{"type": "placeholder", "name": ...}``.

    Fails closed on any entry whose shape this mapper doesn't
    recognize.  Silent skipping is the wrong posture for a fetch-
    side mapper: a Langfuse SDK extension (or a malformed entry)
    would otherwise produce a degraded rendered prompt with zero
    signal to the caller — exactly the kind of bug that changes
    model behavior invisibly.  ``PromptNotFound`` is the canonical
    "we got the prompt but couldn't fully deserialize it" signal,
    matching how the backend handles other fetch-side failures.

    ``name`` and ``label`` are threaded through purely for error
    context on the ``PromptNotFound`` carriers.
    """
    out: list[dict[str, Any]] = []
    for raw_entry in raw:
        if not isinstance(raw_entry, dict):
            raise PromptNotFound(
                f"Langfuse chat-prompt entry has unsupported shape: "
                f"expected dict, got {type(raw_entry).__name__}",
                name=name,
                label=label,
                backend="langfuse",
            )
        entry = cast("dict[str, Any]", raw_entry)
        entry_type = entry.get("type")
        if entry_type == "placeholder":
            placeholder_name = entry.get("name")
            if not isinstance(placeholder_name, str):
                raise PromptNotFound(
                    f"Langfuse placeholder entry missing or invalid 'name': {entry!r}",
                    name=name,
                    label=label,
                    backend="langfuse",
                )
            out.append({"type": "placeholder", "name": placeholder_name})
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        raise PromptNotFound(
            f"Langfuse chat-prompt entry has unsupported role/content shape: {entry!r}",
            name=name,
            label=label,
            backend="langfuse",
        )
    return out


def _chat_segments_from_normalized(
    entries: Iterable[dict[str, Any]],
) -> Iterable[ChatSegment]:
    """Map a normalized canonical entry list to OA
    :class:`ChatSegment` entries.  Placeholder segments use
    ``model_construct`` so a Langfuse-stored prompt with a
    malformed placeholder name (e.g., leading-digit) reaches the
    render path before raising — the normative render-time error
    trigger.  Content segments go through the normal pydantic
    constructor since their fields don't carry the same constraints
    that hand-built callers would benefit from catching earlier."""
    for entry in entries:
        if entry.get("type") == "placeholder":
            yield PlaceholderSegment.model_construct(
                type="placeholder",
                placeholder=entry["name"],
            )
        else:
            yield ContentSegment(role=entry["role"], content=entry["content"])


def _metadata_from(result: TextPromptClient | ChatPromptClient) -> dict[str, Any]:
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
