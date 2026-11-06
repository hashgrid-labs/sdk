import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_dna import FakeDNA  # noqa: E402


@pytest.fixture
def fake():
    return FakeDNA()


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for var in ("HASHGRID_API_KEY", "HASHGRID_DNA_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def anyio_backend():
    return "asyncio"
