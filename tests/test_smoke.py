import subprocess
import tomllib
from pathlib import Path

import pytest

import openarmature


def test_package_versions() -> None:
    assert openarmature.__version__ == "0.5.0"
    assert openarmature.__spec_version__ == "0.15.0"


def test_spec_version_matches_pyproject() -> None:
    # AGENTS.md flags __spec_version__, pyproject.toml's
    # [tool.openarmature].spec_version, and the submodule pin as
    # required to stay in sync. This test catches the pyproject ↔
    # runtime drift class; test_spec_version_matches_submodule_pin
    # below catches the submodule side.
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text())
    pyproject_spec_version = config["tool"]["openarmature"]["spec_version"]
    assert openarmature.__spec_version__ == pyproject_spec_version


def test_spec_version_matches_submodule_pin() -> None:
    # The submodule's git HEAD must be at the v{__spec_version__}
    # tag, completing the three-place drift check from AGENTS.md.
    # Skips cleanly when the submodule isn't a git checkout (e.g.,
    # installed-package CI lanes pulling from PyPI sdists).
    spec_dir = Path(__file__).resolve().parent.parent / "openarmature-spec"
    if not (spec_dir / ".git").exists():
        pytest.skip("openarmature-spec is not a git checkout")
    try:
        result = subprocess.run(
            ["git", "-C", str(spec_dir), "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        pytest.fail(
            "submodule HEAD is not at any tag; bump it to "
            f"v{openarmature.__spec_version__} or update __spec_version__"
        )
    submodule_tag = result.stdout.strip()
    expected = f"v{openarmature.__spec_version__}"
    assert submodule_tag == expected, (
        f"submodule pinned at {submodule_tag}, but __spec_version__ is "
        f"{openarmature.__spec_version__} (expected tag {expected})"
    )
