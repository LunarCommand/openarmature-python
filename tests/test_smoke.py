import openarmature


def test_package_versions() -> None:
    assert openarmature.__version__ == "0.5.0rc1"
    assert openarmature.__spec_version__ == "0.10.0"
