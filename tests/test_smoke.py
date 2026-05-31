import re
import tomllib
from pathlib import Path

import pytest

import openarmature


def test_package_versions() -> None:
    assert openarmature.__version__ == "0.10.0"
    assert openarmature.__spec_version__ == "0.37.0"


def test_spec_version_matches_pyproject() -> None:
    # AGENTS.md flags __spec_version__, pyproject.toml's
    # [tool.openarmature].spec_version, and the submodule pin as
    # required to stay in sync. This test catches the pyproject ↔
    # runtime drift class; test_spec_version_matches_submodule_changelog
    # below catches the submodule side.
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text())
    pyproject_spec_version = config["tool"]["openarmature"]["spec_version"]
    assert openarmature.__spec_version__ == pyproject_spec_version


# Keep a Changelog heading: ``## [0.15.0]`` (with optional trailing
# date). The ``[Unreleased]`` entry uses a non-numeric tag and is
# skipped by this pattern.
_CHANGELOG_VERSION_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]")


def _read_latest_spec_version_from_changelog(path: Path) -> str:
    """Return the first non-``[Unreleased]`` versioned heading from a
    Keep-a-Changelog file. Raises :class:`AssertionError` if no
    versioned heading is present (the file is malformed for our
    purposes).
    """
    for line in path.read_text().splitlines():
        match = _CHANGELOG_VERSION_RE.match(line)
        if match:
            return match.group(1)
    raise AssertionError(f"no versioned heading found in {path}")


def test_spec_version_matches_submodule_changelog() -> None:
    # Third value AGENTS.md flags: the submodule pin (the spec
    # checkout the parent repo records). We verify by reading the
    # spec's CHANGELOG.md at the pinned commit and asserting the
    # latest versioned entry equals __spec_version__. CHANGELOG
    # parsing is more robust than ``git describe`` (no tag-fetch
    # dependency, works in any checkout shape) and the spec follows
    # Keep a Changelog so the format is stable.
    changelog_path = Path(__file__).resolve().parent.parent / "openarmature-spec" / "CHANGELOG.md"
    if not changelog_path.exists():
        pytest.skip("openarmature-spec/CHANGELOG.md is not present")
    submodule_latest = _read_latest_spec_version_from_changelog(changelog_path)
    assert openarmature.__spec_version__ == submodule_latest, (
        f"submodule's CHANGELOG latest is {submodule_latest}, but "
        f"__spec_version__ is {openarmature.__spec_version__}"
    )
