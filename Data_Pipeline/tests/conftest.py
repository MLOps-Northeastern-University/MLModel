"""For every test: no network. The tests run on synthetic participants (tests/helpers.py), and a test
that reached OSF would download real COPS zips, hundreds of MB each. Fake sessions (NoNetwork,
FolderSession in test_download.py) stand in for OSF where a test needs its listing."""

import pytest
import requests


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, method, url, *args, **kwargs):
        raise RuntimeError(f"a test tried to reach the network ({method} {url}); use a fake session")
    monkeypatch.setattr(requests.Session, "request", refuse)
