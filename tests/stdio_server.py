"""The MCP server as ``hashgrid-mcp`` runs it, over stdio, with DNA mocked
in-process and a test embedder: for the end-to-end test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fake_dna import URL, FakeDNA, HashEmbedder  # noqa: E402

from hashgrid.dna import AsyncDNA, AsyncGrid  # noqa: E402
from hashgrid.index import Index  # noqa: E402
from hashgrid.mcp_server import build_server  # noqa: E402

fake = FakeDNA()
build_server(
    lambda: AsyncGrid(Index(HashEmbedder()), AsyncDNA(base_url=URL, transport=fake.transport()))
).run("stdio")
