"""A local MCP server (stdio) for DNA over content: tools take and return text
and image files, embedded here by the ``Index``; only vectors go to DNA, and
the content stays in the index. Run it with
``uvx --from 'hashgrid[local]' hashgrid-mcp``.

Environment: ``HASHGRID_API_KEY`` (required), ``HASHGRID_EMBEDDER`` (default
``fastembed:BAAI/bge-small-en-v1.5``; for images
``fastembed:Qdrant/clip-ViT-B-32``), ``HASHGRID_INDEX`` (default
``~/.hashgrid/index.sqlite``), ``HASHGRID_DNA_URL`` (default the hosted API)."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .dna import AsyncGrid, Conflict, DNAError, Item, ModelMismatch, NotFound
from .errors import GET_A_KEY
from .index import Content, Image

INSTRUCTIONS = f"""DNA is a search engine that learns from feedback. Add items, as text or image files (add_item),
search them with text or an image (search, then more_results for the next ones), and score how well each
result fits (score, -1 to 1): scores train DNA, so later searches rank better. Content stays on this
machine; DNA only sees embeddings. Set HASHGRID_API_KEY to your DNA API key. {GET_A_KEY}"""

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


def build_server(grid: Callable[[], AsyncGrid]) -> MCPServer:
    """The server over the grid ``grid()`` opens, on the first tool call, so
    it starts (and lists its tools) without a key."""
    server = MCPServer(
        name="hashgrid",
        title="Hashgrid DNA (local)",
        website_url="https://dna.hashgrid.ai",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    opened: list[AsyncGrid] = []
    lock = asyncio.Lock()

    async def open_grid() -> AsyncGrid:
        async with lock:
            if not opened:
                opened.append(grid())
        return opened[0]

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

    return server


def main() -> None:
    build_server(AsyncGrid).run("stdio")


if __name__ == "__main__":
    main()
