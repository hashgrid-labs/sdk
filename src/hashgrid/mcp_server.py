"""A local MCP server (stdio) for DNA over text: tools take and return text,
embedded here; only vectors go to DNA, and the text stays in a local sqlite
file. Run it with ``uvx --from 'hashgrid[local]' hashgrid-mcp``.

Environment: ``HASHGRID_API_KEY`` (required), ``HASHGRID_EMBEDDER`` (default
``fastembed:BAAI/bge-small-en-v1.5``), ``HASHGRID_INDEX`` (default
``~/.hashgrid/index.sqlite``), ``HASHGRID_DNA_URL`` (default the hosted API)."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .dna import (
    AsyncDNA,
    AsyncTextGrid,
    Conflict,
    DNAError,
    Edge,
    ModelMismatch,
    NotFound,
)
from .dna.errors import GET_A_KEY
from .embed import Embedder, from_spec
from .index import Index

INSTRUCTIONS = f"""DNA is a search engine that learns from feedback. Add items as text (add_item), search them
with a text query (search, then more_results for the next ones), and score how well each result fits
(score, -1 to 1): scores train DNA, so later searches rank better. Text stays on this machine; DNA only
sees embeddings. Set HASHGRID_API_KEY to your DNA API key. {GET_A_KEY}"""

MAXK = 50


def item_key(text: str) -> str:
    """The key an item gets when none is given: a hash of its text."""
    return "/items/" + hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


class Local:
    """The grid behind the tools, opened on first use, so the server starts
    (and lists its tools) without a key. One embedder per grid: the first
    one used is recorded in the index, and another one is refused."""

    def __init__(
        self,
        dna: Callable[[], AsyncDNA],
        embedder: Callable[[], Embedder],
        index: Callable[[], Index],
    ):
        self._make = (dna, embedder, index)
        self._lock = asyncio.Lock()
        self._open: tuple[AsyncTextGrid, Index, str] | None = None

    async def open(self) -> tuple[AsyncTextGrid, Index, str]:
        """The grid, the index, and the grid's name in the index."""
        async with self._lock:
            if self._open is None:
                make_dna, make_embedder, make_index = self._make
                dna = make_dna()
                embedder = make_embedder()
                grid = await dna.get_grid()
                index = make_index()
                name = f"{dna._http.base_url} {grid.name}"
                recorded = index.embedder(name)
                if recorded is None:
                    index.set_embedder(name, embedder.spec)
                elif recorded != embedder.spec:
                    raise ModelMismatch(
                        f"Grid {grid.name!r} was built with {recorded}, not "
                        f"{embedder.spec}: set HASHGRID_EMBEDDER={recorded}"
                    )
                self._open = AsyncTextGrid(dna, embedder), index, name
            return self._open


def _k(k: int) -> None:
    if not 1 <= k <= MAXK:
        raise ToolError(f"`k` must be 1 to {MAXK}")


async def _found(
    grid: AsyncTextGrid, index: Index, name: str, edges: list[Edge]
) -> list[dict[str, Any]]:
    """Results with their key and text, from the local map; an item added
    elsewhere has no text here, and its key comes from DNA."""
    out = []
    for e in edges:
        known = index.item(name, e.id)
        key, text = known if known else ((await grid.dna.get_node(e.id)).key, None)
        out.append({"id": e.id, "key": key, "text": text})
    return out


def build_server(local: Local) -> MCPServer:
    server = MCPServer(
        name="hashgrid",
        title="Hashgrid DNA (text, local)",
        website_url="https://dna.hashgrid.ai",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register ``fn`` as a tool, its docstring as the description, its
        errors as tool errors."""

        @functools.wraps(fn)
        async def run(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolError:
                raise
            except NotFound as ex:
                raise ToolError("Not found") from ex
            except (DNAError, ModelMismatch, ValueError, ImportError) as ex:
                raise ToolError(str(ex)) from ex

        return server.tool(description=inspect.cleandoc(fn.__doc__ or ""))(run)

    @tool
    async def add_item(text: str, key: str | None = None) -> dict[str, Any]:
        """Add an item to search, as text. `key` is a path unique in the grid (e.g. "/docs/intro"), so
        searches can narrow by prefix; left out, it is a hash of the text, and adding the same text again
        changes nothing. Returns the item's id and key."""
        grid, index, name = await local.open()
        key = key or item_key(text)
        try:
            node = await grid.add(text, key)
        except Conflict:
            known = index.by_key(name, key)
            if known and known[1] == text:
                return {"id": known[0], "key": key}
            raise ToolError(f"Key {key!r} is taken by another item") from None
        index.put(name, node.id, key, text)
        return {"id": node.id, "key": key}

    @tool
    async def search(query: str, k: int = 5, prefix: str = "") -> dict[str, Any]:
        """Search the items with a text query: the best `k` (1 to 50) among those whose key starts with
        `prefix` ("" for any), best first, each with its id, key and text. Returns them with the
        `query_id` to pass to more_results and score."""
        _k(k)
        grid, index, name = await local.open()
        q = await grid.query(query)
        index.put(name, q.id, None, query)
        edges = await grid.search(q, k, prefix)
        return {"query_id": q.id, "results": await _found(grid, index, name, edges)}

    @tool
    async def more_results(
        query_id: str, k: int = 5, prefix: str = ""
    ) -> list[dict[str, Any]]:
        """The next `k` results of a search (1 to 50), not shown before; empty when none are left."""
        _k(k)
        grid, index, name = await local.open()
        return await _found(grid, index, name, await grid.search(query_id, k, prefix))

    @tool
    async def score(query_id: str, item_id: str, score: float) -> dict[str, Any]:
        """Score how well item `item_id` fits search `query_id`, from -1 (wrong) to 1 (perfect): a result
        of the search, or any item. Only scores train DNA; a new score replaces the last."""
        grid, _, _ = await local.open()
        edge = await grid.score(query_id, item_id, score)
        return {"query_id": query_id, "item_id": edge.id, "score": edge.score}

    @tool
    async def delete_item(item_id: str) -> str:
        """Delete an item, its text and its scores."""
        grid, index, name = await local.open()
        await grid.dna.delete_node(item_id)
        index.delete(name, item_id)
        return "Deleted"

    return server


def main() -> None:
    local = Local(AsyncDNA, from_spec, Index)
    build_server(local).run("stdio")


if __name__ == "__main__":
    main()
