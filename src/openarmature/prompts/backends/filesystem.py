"""Reference filesystem PromptBackend."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from ..errors import PromptNotFound, PromptStoreUnavailable
from ..hashing import compute_template_hash
from ..prompt import Prompt, SamplingConfig, TextPrompt, TokenBudget


class FilesystemPromptBackend:
    """Reads prompts from a directory tree.

    Two layouts are supported via the constructor:

    - ``layout="per-label"`` (default): ``<root>/<label>/<name>.j2``.
      The ``label`` subdirectory keeps name-collisions across labels
      distinct (e.g., ``prompts/production/greeting.j2`` and
      ``prompts/staging/greeting.j2``). A filesystem backend may
      interpret label as a subdirectory or filename suffix; this is
      the subdirectory variant.
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
    sidecar file, per the informative filesystem convention:

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

    Optional ``token_budget_source`` populates ``Prompt.token_budget``
    from the SAME sidecar file as ``sampling``. It takes
    the same three values with the same semantics; the budget lives as a
    ``token_budget`` sub-object sibling to the sampling keys inside the
    sidecar (``<name>.config.json`` / unified ``prompt_configs.json``), so
    one file carries both. ``sampling`` reads every sidecar key EXCEPT the
    ``token_budget`` sub-object; ``token_budget`` reads only that sub-object.

    This backend reads templates from disk on every fetch; no caching.
    """

    def __init__(
        self,
        root: Path,
        *,
        layout: Literal["per-label", "flat"] = "per-label",
        sampling_source: Literal["none", "per-prompt-sidecar", "unified"] = "none",
        token_budget_source: Literal["none", "per-prompt-sidecar", "unified"] = "none",
    ) -> None:
        self._root = root
        self._layout = layout
        self._sampling_source = sampling_source
        self._token_budget_source = token_budget_source
        # Unified mode: load and parse at construction so the cost is
        # paid once. Backend instances are typically long-lived
        # process-wide singletons, so a single read on startup is
        # cheaper than re-reading per fetch. Per-prompt values typed
        # ``Any`` rather than ``dict[str, Any]`` so the runtime
        # isinstance guard in ``_resolve_sampling`` remains meaningful
        # — JSON files can have non-dict values under top-level keys.
        # sampling + token_budget share the one unified file, so a single
        # load serves both when either source is unified.
        self._unified_sampling: dict[str, Any] | None = None
        if sampling_source == "unified" or token_budget_source == "unified":
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

    def _read_sidecar(self, name: str, label: str) -> dict[str, Any] | None:
        # Read + parse the per-prompt sidecar ONCE per fetch so sampling and
        # token_budget derive from a single consistent snapshot (proposal 0083);
        # reading each independently risked a torn read + a redundant parse.
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
        return cast(dict[str, Any], raw)

    def _resolve_sampling(self, name: str, sidecar: dict[str, Any] | None) -> SamplingConfig | None:
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
            return _sampling_from_dict(cast(dict[str, Any], raw))
        # per-prompt-sidecar: use the single pre-read snapshot.
        if sidecar is None:
            return None
        return _sampling_from_dict(sidecar)

    def _resolve_token_budget(
        self, name: str, label: str, sidecar: dict[str, Any] | None
    ) -> TokenBudget | None:
        # Mirrors ``_resolve_sampling`` gate-for-gate, reading from the SAME
        # sidecar snapshot (proposal 0083). The budget is the ``token_budget``
        # sub-object; ``_token_budget_from_dict`` pulls it out and returns None
        # when the sidecar carries no budget.
        if self._token_budget_source == "none":
            return None
        if self._token_budget_source == "unified":
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
            source: dict[str, Any] = cast(dict[str, Any], raw)
            source_label = ""
        else:  # per-prompt-sidecar: use the single pre-read snapshot.
            if sidecar is None:
                return None
            source = sidecar
            source_label = label
        # A malformed advisory budget (non-object sub-value, or a bound failing
        # validation) surfaces as PromptStoreUnavailable -- a fallback-eligible
        # domain error -- NOT a bare exception that would bypass PromptManager's
        # multi-backend fallback and hard-crash the LLM call over an advisory field.
        try:
            return _token_budget_from_dict(source)
        except (ValueError, ValidationError) as exc:
            raise PromptStoreUnavailable(
                f"malformed token_budget for prompt {name!r}: {exc}",
                name=name,
                label=source_label,
            ) from exc

    async def fetch(
        self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
    ) -> Prompt:
        """Read the prompt template and (optionally) its sidecar sampling config.

        Returns a ``Prompt`` whose ``version`` is the leading 16 hex
        chars of the template's SHA-256 and ``template_hash`` is the
        full digest. Raises ``PromptNotFound`` when the template is
        missing and ``PromptStoreUnavailable`` on other I/O errors.

        The filesystem backend is cacheless, so ``cache_ttl_seconds`` is
        accepted for protocol conformance and ignored.
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

        # Read the per-prompt sidecar at most once; both sampling and
        # token_budget derive from this one snapshot (proposal 0083).
        sidecar: dict[str, Any] | None = None
        if self._sampling_source == "per-prompt-sidecar" or self._token_budget_source == "per-prompt-sidecar":
            sidecar = await asyncio.to_thread(self._read_sidecar, name, label)
        sampling = self._resolve_sampling(name, sidecar)
        token_budget = self._resolve_token_budget(name, label, sidecar)
        template_hash = compute_template_hash(template_source)
        version = template_hash.removeprefix("sha256:")[:16]
        return TextPrompt(
            name=name,
            version=version,
            label=label,
            template=template_source,
            template_hash=template_hash,
            fetched_at=datetime.now(UTC),
            sampling=sampling,
            token_budget=token_budget,
        )


def _sampling_from_dict(data: dict[str, Any]) -> SamplingConfig:
    # Top-level `extras` is flattened so caller-supplied vendor knobs
    # end up in SamplingConfig's extras-allow bag rather than as a
    # single literal `extras` key. Matches the YAML conformance-fixture
    # convention from llm-provider/032 + the spec §5 sidecar example.
    # `token_budget` (proposal 0083) is a sibling sub-object read by
    # `_token_budget_from_dict`, not a sampling field, so it is excluded
    # here alongside `extras`.
    flat: dict[str, Any] = {k: v for k, v in data.items() if k not in ("extras", "token_budget")}
    extras = data.get("extras")
    if isinstance(extras, dict):
        for k, v in cast(dict[str, Any], extras).items():
            flat.setdefault(k, v)
    return SamplingConfig(**flat)


def _token_budget_from_dict(data: dict[str, Any]) -> TokenBudget | None:
    # The sidecar carries the budget as a `token_budget` sub-object sibling to
    # the sampling keys. Absent -> no budget (None). A non-object sub-value, or
    # a bound failing validation, raises (the caller converts it to the
    # fallback-eligible PromptStoreUnavailable). An all-null / empty budget
    # collapses to None: "no bound declared" is None (parity with the langfuse
    # backend + the None-when-no-budget contract), not a non-null all-null record.
    raw = data.get("token_budget")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"sidecar token_budget must be a JSON object, got {type(raw).__name__}")
    budget = TokenBudget(**cast(dict[str, Any], raw))
    if budget.input_max_tokens is None and budget.total_max_tokens is None:
        return None
    return budget
