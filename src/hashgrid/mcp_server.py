"""A local MCP server (stdio) for DNA over content: tools take and return text
and image files, embedded here by the ``Index``; only vectors go to DNA, and
the content stays in the index. Run it with
``uvx --from 'hashgrid[local]' hashgrid-mcp``.

Environment: ``HASHGRID_API_KEY`` (else the key create_grid saved in the index), ``HASHGRID_EMBEDDER`` (default
``fastembed:BAAI/bge-small-en-v1.5``; for images
``fastembed:Qdrant/clip-ViT-B-32``), ``HASHGRID_INDEX`` (default
``~/.hashgrid/index.sqlite``), ``HASHGRID_DNA_URL`` (default the hosted API)."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
from typing import Any, Callable

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .dna import (
    AsyncDNA,
    AsyncGrid,
    Conflict,
    DNAError,
    GridKey,
    Item,
    ModelMismatch,
    NotFound,
    acreate_grid,
)
from .dna.client import _url
from .index import Content, Image, Index

INSTRUCTIONS = f"""DNA is a search engine that learns from feedback. Add items, as text or image files (add_item),
search them with text or an image (search, then more_results for the next ones), and score how well each
result fits (score, -1 to 1): scores train DNA, so later searches rank better. Content stays on this
machine; DNA only sees embeddings. No grid yet? create_grid makes one."""

MAXK = 50


def _k(k: int) -> None:
    if not 1 <= k <= MAXK:
        raise ToolError(f"`k` must be 1 to {MAXK}")


def _content(text: str | None, image: str | None) -> Content:
    if (text is None) == (image is None):
        raise ToolError("Pass `text` or `image` (a file path), one of them")
    if image is not None:
        img = Image(image)
        if not img.path or not img.path.is_file():
            raise ToolError(f"No image file at {image}")
        return img
    return text  # type: ignore[return-value]


def _out(item: Item) -> dict[str, Any]:
    """An item as the tools return it: its text, or its image file."""
    out: dict[str, Any] = {"id": item.id, "key": item.key}
    if isinstance(item.content, Image):
        out["image"] = str(item.content.path) if item.content.path else None
    else:
        out["text"] = item.content
    return out


class Local:
    """The grid behind the tools, opened on the first call, so the server
    starts (and lists its tools) without a key. Its key is
    ``$HASHGRID_API_KEY``, else the one saved in the index by create_grid."""

    def __init__(
        self,
        index: Callable[[], Index] = Index,
        *,
        base_url: str | None = None,
        transport: Any = None,
    ):
        self._index, self._base_url, self._transport = index, base_url, transport
        self._lock = asyncio.Lock()
        self.index: Index | None = None
        self.grid: AsyncGrid | None = None

    @property
    def url(self) -> str:
        return str(httpx.URL(_url(self._base_url) + "/"))  # as httpx gives base_url

    def _open_index(self) -> Index:
        if self.index is None:
            self.index = self._index()
        return self.index

    async def open(self) -> AsyncGrid:
        async with self._lock:
            if self.grid is None:
                index = self._open_index()
                key = os.environ.get("HASHGRID_API_KEY") or index.saved_key(self.url)
                if not key:
                    raise ToolError(
                        "No DNA grid yet: call create_grid to make one (no key needed), "
                        "or set HASHGRID_API_KEY to your key."
                    )
                dna = AsyncDNA(key, base_url=self._base_url, transport=self._transport)
                self.grid = AsyncGrid(index, dna)
            return self.grid

    async def create(self, name: str) -> GridKey:
        async with self._lock:
            index = self._open_index()
            if os.environ.get("HASHGRID_API_KEY"):
                raise ToolError("HASHGRID_API_KEY is set: this server uses that grid")
            if index.saved_key(self.url):
                raise ToolError("This server has a grid already; delete_grid it first")
            key = await acreate_grid(name, base_url=self._base_url, transport=self._transport)
            index.save_key(self.url, key.name, key.api_key)
            self.grid = None
            return key

    async def delete(self) -> str:
        grid = await self.open()
        name = (await grid.dna.get_grid()).name
        await grid.delete_grid()
        async with self._lock:
            self.grid = None
        return name


def build_server(local: Local | None = None) -> MCPServer:
    local = local or Local()
    server = MCPServer(
        name="hashgrid",
        title="Hashgrid DNA (local)",
        website_url="https://dna.hashgrid.ai",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    open_grid = local.open

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
            except Conflict as ex:
                raise ToolError("That key is taken by another item") from ex
            except (DNAError, ModelMismatch, ValueError, ImportError, OSError) as ex:
                raise ToolError(str(ex)) from ex

        return server.tool(description=inspect.cleandoc(fn.__doc__ or ""))(run)

    @tool
    async def add_item(
        text: str | None = None, image: str | None = None, key: str | None = None
    ) -> dict[str, Any]:
        """Add an item to search: `text`, or `image`, the path of an image file (images need an image
        embedder). `key` is a path unique in the grid (e.g. "/docs/intro"), so searches can narrow by
        prefix; left out, it is a hash of the content, and adding the same content again changes nothing.
        Returns the item's id and key."""
        content = _content(text, image)
        g = await open_grid()
        item = await g.add(content, key)
        return {"id": item.id, "key": item.key}

    @tool
    async def search(
        query: str | None = None, image: str | None = None, k: int = 5, prefix: str = ""
    ) -> dict[str, Any]:
        """Search the items with a text `query`, or an `image` file: the best `k` (1 to 50) among those
        whose key starts with `prefix` ("" for any), best first, each with its id, key and text (or image
        path). Returns them with the `query_id` to pass to more_results and score."""
        _k(k)
        content = _content(query, image)
        g = await open_grid()
        q, results = await g.ask(content, k, prefix)
        return {"query_id": q.id, "results": [_out(r) for r in results]}

    @tool
    async def more_results(
        query_id: str, k: int = 5, prefix: str = ""
    ) -> list[dict[str, Any]]:
        """The next `k` results of a search (1 to 50), not shown before; empty when none are left."""
        _k(k)
        g = await open_grid()
        return [_out(r) for r in await g.search(query_id, k, prefix)]

    @tool
    async def score(query_id: str, item_id: str, score: float) -> dict[str, Any]:
        """Score how well item `item_id` fits search `query_id`, from -1 (wrong) to 1 (perfect): a result
        of the search, or any item. Only scores train DNA; a new score replaces the last."""
        g = await open_grid()
        result = await g.score(query_id, item_id, score)
        return {"query_id": query_id, "item_id": result.id, "score": result.score}

    @tool
    async def delete_item(item_id: str) -> str:
        """Delete an item, its content in the index and its scores."""
        g = await open_grid()
        await g.delete(item_id)
        return "Deleted"

    @tool
    async def create_grid(name: str = "hashgrid") -> dict[str, Any]:
        """Create a DNA grid for this server, when it has none (no key needed): its API key is saved in the
        local index, and the other tools use it from then on."""
        key = await local.create(name)
        return {
            "name": key.name,
            "max_nodes": key.max_nodes,
            "saved_in": str(local.index.path if local.index else ""),
        }

    @tool
    async def delete_grid(confirm: str) -> str:
        """Delete this server's grid, every item and score in it and its content in the local index, for good.
        `confirm` must be the grid's name. Only when the user asks for it."""
        grid = await local.open()
        name = (await grid.dna.get_grid()).name
        if confirm != name:
            raise ToolError(f"`confirm` must be the grid's name, {name!r}")
        await local.delete()
        return f"Deleted grid {name!r}"

    return server


def main() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)  # a line per request otherwise
    build_server().run("stdio")


if __name__ == "__main__":
    main()
