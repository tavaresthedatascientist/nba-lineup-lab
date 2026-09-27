import pytest


@pytest.fixture(scope="session")
def bundle():
    from lineup_lab.features import build
    return build()
