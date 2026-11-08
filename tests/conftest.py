"""Shared test setup."""

import pytest

from src.utils.client_cache import clear_clients


@pytest.fixture(autouse=True)
def _fresh_sdk_clients():
    # The pipeline keeps one SDK client per target for the whole run; tests
    # patch the SDK classes, so each starts without stored clients
    clear_clients()
    yield
    clear_clients()
