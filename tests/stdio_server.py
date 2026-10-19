"""The MCP server as ``hashgrid-mcp`` runs it, over stdio, with DNA mocked
in-process and a test embedder: for the end-to-end test."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fake_dna import URL, FakeDNA, HashEmbedder  # noqa: E402

from hashgrid.dna import AsyncDNA  # noqa: E402
from hashgrid.mcp_server import Local, build_server  # noqa: E402
from hashgrid.index import Index  # noqa: E402

fake = FakeDNA()
local = Local(
    lambda: AsyncDNA(base_url=URL, transport=fake.transport()),  # key from env
    lambda: HashEmbedder(spec=os.environ.get("HASHGRID_EMBEDDER", "test:hash@64")),
    Index,
)
build_server(local).run("stdio")
