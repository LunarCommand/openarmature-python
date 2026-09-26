import re
import tomllib
from pathlib import Path

import pytest

import openarmature


def test_package_versions() -> None:
    assert openarmature.__version__ == "0.16.0"
    assert openarmature.__spec_version__ == "0.118.2"


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


def test_conformance_spec_pin_matches_spec_version() -> None:
    # conformance.toml's [manifest].spec_pin is a fourth pin-sync point
    # (alongside __spec_version__, pyproject, and the submodule changelog)
    # that no other check covered, so it silently drifted across the
    # v0.15.0 pin bumps. Assert it tracks __spec_version__ (the manifest
    # value carries a ``v`` prefix) so it can't drift again.
    conformance_path = Path(__file__).resolve().parent.parent / "conformance.toml"
    manifest = tomllib.loads(conformance_path.read_text())
    spec_pin = manifest["manifest"]["spec_pin"]
    assert spec_pin == f"v{openarmature.__spec_version__}", (
        f"conformance.toml [manifest].spec_pin is {spec_pin!r}, expected v{openarmature.__spec_version__}"
    )


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


def test_the_conformance_manifest_is_force_included_in_the_wheel() -> None:
    # The bundled AGENTS.md tells an agent to read `conformance.toml` for
    # per-proposal implementation status, which is the only artifact that can
    # answer "is this real in the version I have installed" for a behaviour with
    # no importable name. That instruction shipped for releases while the file
    # did not, so it resolved only for someone working in a clone. An agent
    # planning against an accepted proposal had no way to learn that half of it
    # was missing, which is exactly what `partial` records.
    #
    # Guarded here rather than by building a wheel in the suite: this catches the
    # entry being dropped or its paths going stale, which is how it would break.
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text())
    force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert "conformance.toml" in force_include, (
        "conformance.toml must be force-included in the wheel; AGENTS.md instructs "
        f"an agent to read it. force-include currently: {force_include}"
    )
    target = force_include["conformance.toml"]
    package = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"][0]
    package_name = Path(package).name
    assert target == f"{package_name}/conformance.toml", (
        f"the manifest must land beside the other bundled docs inside the package; "
        f"got {target!r}, expected {package_name}/conformance.toml"
    )
    source = pyproject_path.parent / "conformance.toml"
    assert source.exists(), f"force-include names a source that does not exist: {source}"


def test_agents_md_tells_an_agent_where_the_manifest_actually_is() -> None:
    # The pointer used to read "at the repo root", which is unresolvable from a
    # venv and is the half of the defect that made the missing file invisible:
    # an agent that could not find it had no reason to think it should be there.
    bundled = Path(__file__).resolve().parent.parent / "src" / "openarmature" / "AGENTS.md"
    text = bundled.read_text()
    assert "conformance.toml" in text, "AGENTS.md must still point at the manifest"
    assert "at the repo root" not in text, (
        "the manifest pointer must not be repo-relative: it resolves for a clone and "
        "not for anyone who installed the package"
    )
    assert "importlib.resources" in text, (
        "the pointer should name the access path that works from an installed package"
    )


def test_the_sdist_excludes_the_private_follow_up_notes() -> None:
    # `_tasks/` is kept out of git by `.git/info/exclude`, which hatch does not
    # read: it packages what is on disk. So the local follow-up notes were going
    # into the sdist and would have published to PyPI. They are working notes
    # that quote internal coordination, not part of the public artifact set.
    #
    # Guarded by config shape rather than by building an sdist in the suite,
    # because the way this breaks is someone editing or dropping the exclude.
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text())
    exclude = config["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]

    assert "_tasks/" in exclude, (
        "`_tasks/` must be excluded from the sdist. It holds private working notes, "
        f"and .git/info/exclude does not reach hatch. Current excludes: {exclude}"
    )
    # The pinned spec submodule is most of the tarball and reconstructs from the
    # version pin, so it is excluded too. Not a privacy matter, just size.
    assert "openarmature-spec/" in exclude, (
        f"the pinned spec submodule should stay out of the sdist; excludes: {exclude}"
    )
    # Deliberately NOT excluded, asserted so a future tidy-up does not quietly
    # drop them: tests let a packager verify a build from source, and the other
    # two are adopter-facing.
    for kept in ("tests/", "examples/", "docs/"):
        assert kept not in exclude, (
            f"{kept} is excluded from the sdist; it was kept on purpose, so if that "
            "changed the reasoning in pyproject.toml needs changing with it"
        )
