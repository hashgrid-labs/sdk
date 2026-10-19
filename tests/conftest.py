import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_dna import KEY, URL, FakeDNA, HashEmbedder  # noqa: E402


@pytest.fixture
def fake():
    return FakeDNA()


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture(autouse=True)
def _no_env(monkeypatch, tmp_path):
    for var in ("HASHGRID_API_KEY", "HASHGRID_DNA_URL", "HASHGRID_EMBEDDER", "OPENAI_API_KEY", "VOYAGE_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HASHGRID_INDEX", str(tmp_path / "index.sqlite"))


@pytest.fixture
def anyio_backend():
    return "asyncio"
