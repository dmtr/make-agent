import pytest


def pytest_addoption(parser):
    parser.addoption("--e2e", action="store_true", default=False, help="Run end-to-end tests against the real API")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--e2e"):
        return
    skip_e2e = pytest.mark.skip(reason="e2e tests are skipped by default; pass --e2e to run them")
    for item in items:
        if item.get_closest_marker("e2e"):
            item.add_marker(skip_e2e)
