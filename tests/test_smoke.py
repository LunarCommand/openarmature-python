import tomllib
from pathlib import Path

import openarmature


def test_package_versions() -> None:
    assert openarmature.__version__ == "0.5.0"
    assert openarmature.__spec_version__ == "0.15.0"


def test_spec_version_matches_pyproject() -> None:
    # AGENTS.md flags __spec_version__, pyproject.toml's
    # [tool.openarmature].spec_version, and the submodule pin as
    # required to stay in sync. The test_package_versions check above
    # only verifies internal consistency between __spec_version__ and
    # its asserted value, so the pyproject side can drift undetected.
    # This test catches that class of three-place drift.
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    config = tomllib.loads(pyproject_path.read_text())
    pyproject_spec_version = config["tool"]["openarmature"]["spec_version"]
    assert openarmature.__spec_version__ == pyproject_spec_version
