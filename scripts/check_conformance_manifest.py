"""Validate conformance.toml against the pinned spec submodule's proposals.

Failure modes caught:

- Accepted proposal in the spec has no entry in conformance.toml.
- Entry in conformance.toml refers to a proposal that doesn't exist in
  the spec, or refers to a proposal whose Status is not "Accepted"
  (e.g., Draft / Superseded — those are deliberately excluded from the
  manifest so the docs site doesn't claim impl status for unsettled
  proposals).
- Entry has an unknown `status` value or a malformed `since` version.

Read-only; intended for CI. Non-zero exit on any failure with a
human-readable diff. Runs under the repo's stdlib Python (>=3.12, so
`tomllib` is always available).
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "conformance.toml"
PROPOSALS_DIR = REPO_ROOT / "openarmature-spec" / "proposals"

ALLOWED_STATUSES = frozenset({"implemented", "partial", "textual-only", "not-yet"})
PROPOSAL_FILENAME_RE = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")
STATUS_LINE_RE = re.compile(r"^- \*\*Status:\*\*\s*(.+?)\s*$", re.MULTILINE)
SINCE_RE = re.compile(r"^\d+\.\d+\.\d+$")


def parse_spec_proposals() -> dict[str, str]:
    # Returns {proposal_id: status} for every proposal markdown file in
    # the pinned spec submodule. proposal_id is the 4-digit string used
    # as the manifest key; status is the literal value from the file's
    # `- **Status:** ...` header line.
    if not PROPOSALS_DIR.is_dir():
        sys.exit(
            f"::error::proposals dir not found at {PROPOSALS_DIR} — "
            "is the openarmature-spec submodule checked out?"
        )

    result: dict[str, str] = {}
    for path in sorted(PROPOSALS_DIR.iterdir()):
        m = PROPOSAL_FILENAME_RE.match(path.name)
        if not m:
            continue
        proposal_id = m.group(1)
        text = path.read_text(encoding="utf-8")
        status_match = STATUS_LINE_RE.search(text)
        if not status_match:
            sys.exit(f"::error::proposal {proposal_id} ({path.name}) has no `- **Status:** ...` header line")
        result[proposal_id] = status_match.group(1).strip()
    return result


def load_manifest() -> dict[str, dict[str, Any]]:
    # Returns {proposal_id: entry_dict} for every [proposals."NNNN"]
    # section in conformance.toml.
    if not MANIFEST_PATH.is_file():
        sys.exit(f"::error::manifest not found at {MANIFEST_PATH}")

    with MANIFEST_PATH.open("rb") as f:
        data = tomllib.load(f)

    proposals = data.get("proposals", {})
    if not isinstance(proposals, dict):
        sys.exit("::error::conformance.toml [proposals] table malformed")
    return cast(dict[str, dict[str, Any]], proposals)


def main() -> int:
    spec = parse_spec_proposals()
    manifest = load_manifest()

    accepted_ids = {pid for pid, status in spec.items() if status == "Accepted"}
    manifest_ids = set(manifest.keys())

    errors: list[str] = []

    missing = sorted(accepted_ids - manifest_ids)
    for pid in missing:
        errors.append(f"Accepted spec proposal {pid} has no entry in conformance.toml")

    extra = sorted(manifest_ids - accepted_ids)
    for pid in extra:
        if pid not in spec:
            errors.append(
                f"conformance.toml entry {pid} refers to a proposal that "
                f"doesn't exist in openarmature-spec/proposals/"
            )
        else:
            errors.append(
                f"conformance.toml entry {pid} refers to a proposal whose "
                f"spec Status is {spec[pid]!r}, not 'Accepted' — "
                f"drafts and superseded proposals should be omitted"
            )

    for pid in sorted(manifest_ids):
        entry = manifest[pid]
        status = entry.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(
                f"conformance.toml entry {pid} has unknown status {status!r} "
                f"(allowed: {sorted(ALLOWED_STATUSES)})"
            )
        # `since` is required for every status except `not-yet`.
        since = entry.get("since")
        if status == "not-yet":
            if since is not None:
                errors.append(
                    f"conformance.toml entry {pid} has status=not-yet but "
                    f"also a `since` field — drop `since` for not-yet entries"
                )
        else:
            if since is None:
                errors.append(f"conformance.toml entry {pid} has status={status!r} but no `since` field")
            elif not SINCE_RE.match(since):
                errors.append(
                    f"conformance.toml entry {pid} `since` value {since!r} is not in MAJOR.MINOR.PATCH form"
                )

    if errors:
        print("conformance.toml validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: {len(accepted_ids)} accepted proposals, {len(manifest_ids)} manifest entries, all consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
