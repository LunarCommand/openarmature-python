import openarmature


def test_package_versions() -> None:
    assert openarmature.__version__ == "0.4.0rc0"
    assert openarmature.__spec_version__ == "0.8.2"
