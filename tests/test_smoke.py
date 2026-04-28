import openarmature


def test_package_versions() -> None:
    assert openarmature.__version__ == "0.2.0"
    assert openarmature.__spec_version__ == "0.2.0"
