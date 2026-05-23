"""Reference filesystem PromptBackend."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from ..errors import PromptNotFound, PromptStoreUnavailable
from ..hashing import compute_template_hash
from ..prompt import Prompt


class FilesystemPromptBackend:
    """Reads prompts from a directory tree.

    Layout convention: ``<root>/<label>/<name>.j2``. The ``label``
    subdirectory keeps name-collisions across labels distinct
    (e.g., ``prompts/production/greeting.j2`` and
    ``prompts/staging/greeting.j2``). Spec §5 permits filesystem
    backends to interpret label as "a subdirectory or filename
    suffix"; this backend picks subdirectory.

    The ``version`` field is derived from the template content hash
    (first 16 hex chars of the SHA-256, ~64 bits) so two file
    contents map deterministically to two distinct version strings
    without needing a sidecar metadata file. Per spec §3, this
    satisfies the "stable identifier" requirement. The 16-char
    prefix puts the birthday-paradox collision boundary at ~4B
    distinct templates; well past any realistic single-backend
    exposure. Higher-scale backends should widen further or pick a
    different stable identifier (semver from a sidecar metadata
    file, git short-SHAs, etc.).

    This backend reads from disk on every fetch; no caching. A
    caching backend (e.g., openarmature-langfuse) that returns
    cached results MUST preserve the original ``fetched_at`` on the
    returned Prompt, not the cache-hit time, per spec §3.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    async def fetch(self, name: str, label: str = "production") -> Prompt:
        """Read ``<root>/<label>/<name>.j2`` and return the prompt.

        Reads on every call (no caching). The returned prompt's
        ``version`` is the leading 16 hex chars of the template's
        SHA-256, and ``template_hash`` is the full digest. Raises
        ``PromptNotFound`` when the file is missing and
        ``PromptStoreUnavailable`` on any other I/O error.
        """
        path = self._root / label / f"{name}.j2"
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

        template_hash = compute_template_hash(template_source)
        version = template_hash.removeprefix("sha256:")[:16]
        return Prompt(
            name=name,
            version=version,
            label=label,
            template=template_source,
            template_hash=template_hash,
            fetched_at=datetime.now(UTC),
        )
