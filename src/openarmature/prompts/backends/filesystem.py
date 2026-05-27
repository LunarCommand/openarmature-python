"""Reference filesystem PromptBackend."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from ..errors import PromptNotFound, PromptStoreUnavailable
from ..hashing import compute_template_hash
from ..prompt import Prompt, SamplingConfig


class FilesystemPromptBackend:
    """Reads prompts from a directory tree.

    Two layouts are supported via the constructor:

    - ``layout="per-label"`` (default): ``<root>/<label>/<name>.j2``.
      The ``label`` subdirectory keeps name-collisions across labels
      distinct (e.g., ``prompts/production/greeting.j2`` and
      ``prompts/staging/greeting.j2``). Spec §5 permits filesystem
      backends to interpret label as "a subdirectory or filename
      suffix"; this is the subdirectory variant.
    - ``layout="flat"``: ``<root>/<name>.j2``. The same template
      is returned regardless of which label was requested; the
      Prompt's ``label`` field is the requested label verbatim.
      Useful when label-based A/B routing is driven by a
      :class:`~openarmature.prompts.label_resolver.LabelResolver`
      rather than a directory tree.

    The ``version`` field is derived from the template content hash
    (first 16 hex chars of the SHA-256, ~64 bits) so two file
    contents map deterministically to two distinct version strings
    without needing a sidecar metadata file. The 16-char prefix puts
    the birthday-paradox collision boundary at ~4B distinct templates,
    well past any realistic single-backend exposure.

    Optional ``sampling_source`` populates ``Prompt.sampling`` from a
    sidecar file, per the spec §5 informative filesystem convention:

    - ``"none"`` (default): never populate ``sampling``.
    - ``"per-prompt-sidecar"``: read ``<name>.config.json`` from the
      same directory as the template (i.e., ``<root>/<label>/<name>.config.json``
      under ``per-label`` layout, ``<root>/<name>.config.json`` under
      ``flat``). A missing sidecar leaves ``sampling = None``.
    - ``"unified"``: read ``<root>/prompt_configs.json`` at backend
      construction time and key into it by prompt name. A name not in
      the unified map leaves ``sampling = None``. Construction raises
      :class:`PromptStoreUnavailable` if the file exists but cannot
      be parsed.

    This backend reads templates from disk on every fetch; no caching.
    """

    def __init__(
        self,
        root: Path,
        *,
        layout: Literal["per-label", "flat"] = "per-label",
        sampling_source: Literal["none", "per-prompt-sidecar", "unified"] = "none",
    ) -> None:
        self._root = root
        self._layout = layout
        self._sampling_source = sampling_source
        # Unified mode: load and parse at construction so the cost is
        # paid once. Backend instances are typically long-lived
        # process-wide singletons, so a single read on startup is
        # cheaper than re-reading per fetch. Per-prompt values typed
        # ``Any`` rather than ``dict[str, Any]`` so the runtime
        # isinstance guard in ``_resolve_sampling`` remains meaningful
        # — JSON files can have non-dict values under top-level keys.
        self._unified_sampling: dict[str, Any] | None = None
        if sampling_source == "unified":
            self._unified_sampling = self._load_unified_configs()

    def _load_unified_configs(self) -> dict[str, Any]:
        path = self._root / "prompt_configs.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromptStoreUnavailable(
                f"failed to load unified prompt_configs.json at {path}: {exc}",
                name="",
                label="",
            ) from exc
        if not isinstance(data, dict):
            raise PromptStoreUnavailable(
                f"unified prompt_configs.json at {path} is not a JSON object",
                name="",
                label="",
            )
        return cast(dict[str, Any], data)

    def _template_path(self, name: str, label: str) -> Path:
        if self._layout == "flat":
            return self._root / f"{name}.j2"
        return self._root / label / f"{name}.j2"

    def _sidecar_path(self, name: str, label: str) -> Path:
        if self._layout == "flat":
            return self._root / f"{name}.config.json"
        return self._root / label / f"{name}.config.json"

    def _resolve_sampling(self, name: str, label: str) -> SamplingConfig | None:
        if self._sampling_source == "none":
            return None
        if self._sampling_source == "unified":
            assert self._unified_sampling is not None
            raw = self._unified_sampling.get(name)
            if raw is None:
                return None
            if not isinstance(raw, dict):
                raise PromptStoreUnavailable(
                    f"unified prompt_configs.json entry for {name!r} is not a JSON object "
                    f"(got {type(raw).__name__})",
                    name=name,
                    label="",
                )
            entry = cast(dict[str, Any], raw)
            return _sampling_from_dict(entry)
        # per-prompt-sidecar
        path = self._sidecar_path(name, label)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromptStoreUnavailable(
                f"failed to load sidecar {path} for ({name!r}, {label!r}): {exc}",
                name=name,
                label=label,
            ) from exc
        if not isinstance(raw, dict):
            raise PromptStoreUnavailable(
                f"sidecar {path} is not a JSON object",
                name=name,
                label=label,
            )
        return _sampling_from_dict(cast(dict[str, Any], raw))

    async def fetch(self, name: str, label: str = "production") -> Prompt:
        """Read the prompt template and (optionally) its sidecar sampling config.

        Returns a ``Prompt`` whose ``version`` is the leading 16 hex
        chars of the template's SHA-256 and ``template_hash`` is the
        full digest. Raises ``PromptNotFound`` when the template is
        missing and ``PromptStoreUnavailable`` on other I/O errors.
        """
        path = self._template_path(name, label)
        try:
            template_source = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError as exc:
            raise PromptNotFound(
                f"prompt ({name!r}, {label!r}) not found under {self._root}",
                name=name,
                label=label,
                backend=str(self._root),
            ) from exc
        except OSError as exc:
            raise PromptStoreUnavailable(
                f"filesystem I/O error reading ({name!r}, {label!r}): {exc}",
                name=name,
                label=label,
            ) from exc

        sampling = await asyncio.to_thread(self._resolve_sampling, name, label)
        template_hash = compute_template_hash(template_source)
        version = template_hash.removeprefix("sha256:")[:16]
        return Prompt(
            name=name,
            version=version,
            label=label,
            template=template_source,
            template_hash=template_hash,
            fetched_at=datetime.now(UTC),
            sampling=sampling,
        )


def _sampling_from_dict(data: dict[str, Any]) -> SamplingConfig:
    # Top-level `extras` is flattened so caller-supplied vendor knobs
    # end up in SamplingConfig's extras-allow bag rather than as a
    # single literal `extras` key. Matches the YAML conformance-fixture
    # convention from llm-provider/032 + the spec §5 sidecar example.
    flat: dict[str, Any] = {k: v for k, v in data.items() if k != "extras"}
    extras = data.get("extras")
    if isinstance(extras, dict):
        for k, v in cast(dict[str, Any], extras).items():
            flat.setdefault(k, v)
    return SamplingConfig(**flat)
