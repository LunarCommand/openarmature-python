"""Prompt and PromptResult records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from openarmature.llm.messages import Message


class Prompt(BaseModel):
    """An unrendered template plus identity metadata.

    A prompt carries enough information to be rendered, traced, and
    content-addressed without a backend round-trip. ``template`` is
    the raw template source string (Jinja2 syntax in Python);
    compilation happens on render so ``Prompt`` stays serializable
    and engine-agnostic.

    Attributes:
        name: Stable identifier within the backend.
        version: Backend-defined version string. Two distinct version
            strings denote distinct prompt contents.
        label: The label under which this prompt was fetched
            (e.g., "production", "latest", "variant-a").
        template: Raw template source.
        template_hash: SHA-256 of the raw template source. Format
            ``"sha256:<hex>"``.
        fetched_at: Time the prompt was fetched from its backend.
            When a caching backend serves a cached result,
            ``fetched_at`` MUST reflect the original fetch time, not
            the cache hit time.
        metadata: Optional backend-supplied metadata.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    label: str
    template: str
    template_hash: str
    fetched_at: datetime
    metadata: dict[str, Any] | None = None


class PromptResult(BaseModel):
    """The rendered output of applying variables to a prompt.

    Carries the rendered ``Message`` sequence (ready to pass to
    ``Provider.complete()``) plus the source prompt's identity
    metadata and a ``rendered_hash`` that captures the rendered
    content.

    The ``rendered_hash`` is the cache-key value most useful to
    downstream consumers: two renders with the same template AND
    the same variables produce the same hash.

    Attributes:
        name: Propagated from the source Prompt.
        version: Propagated from the source Prompt.
        label: Propagated from the source Prompt.
        template_hash: Propagated from the source Prompt.
        rendered_hash: SHA-256 of the canonical serialization of
            the rendered messages list.
        messages: Ordered non-empty sequence of ``Message`` records.
        variables: Variable mapping used to render. v1 policy:
            pass-through unchanged (no automatic redaction). Keys
            are always preserved; future redaction policies would
            redact values, never strip keys.
        fetched_at: Propagated from the source Prompt.
        rendered_at: Time this PromptResult was rendered. Distinct
            from ``fetched_at``: a single fetched prompt may render
            many times.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    label: str
    template_hash: str
    rendered_hash: str
    messages: list[Message] = Field(min_length=1)
    variables: dict[str, Any]
    fetched_at: datetime
    rendered_at: datetime
