"""A grid of content: ``Grid`` (sync) and ``AsyncGrid``. The ``Index``
embeds on your machine and keeps the content; DNA gets the vectors only.

>>> from hashgrid.dna import Grid
>>> grid = Grid()                     # Index() and DNA() from the environment
>>> grid.add("Lisbon, a city of seven hills", key="/cities/lisbon")
>>> query, results = grid.ask("a hilly European capital", k=3)
>>> grid.score(query, results[0], 1.0)
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Mapping, Sequence

from ..errors import Conflict, ModelMismatch
from ..index import Index
from ..index.content import Content, content_key, same
from ..index.embedders import Kind
from .client import DNA, AsyncDNA, _id
from .models import Edge, Item, Node, Result


def _pairs(
    items: Mapping[str, Content] | Iterable[Content],
) -> list[tuple[str, Content]]:
    """``{key: content}``, or contents keyed by their hash."""
    if isinstance(items, Mapping):
        return list(items.items())
    return [(content_key(c), c) for c in items]


class _Core:
    """What both grids share: the index, the grid's name in it, and the
    check that vectors have the grid's length."""

    def __init__(self, index: Index | None):
        self.index = index if index is not None else Index()
        self.name: str | None = None
        self.dim: int | None = None

    def bind(self, base_url: Any, grid_name: str, first: list[Node]) -> None:
        self.name = f"{base_url} {grid_name}"
        self.index.bind(self.name)
        self.dim = len(first[0].emb) if first else None

    def check(self, vectors: Sequence[Sequence[float]]) -> None:
        for v in vectors:
            if self.dim is None:
                self.dim = len(v)
            elif len(v) != self.dim:
                raise ModelMismatch(
                    f"{self.index.spec} gives vectors of {len(v)} values, but this grid's "
                    f"have {self.dim}: use the embedder the grid was built with"
                )

    def item(self, id: str, key: str | None = None) -> Item:
        known = self.index.get(self.name or "", id)
        if known is None:
            return Item(id, key, None)
        return Item(id, known[0] if known[0] is not None else key, known[1])

    def existing(self, key: str, content: Content) -> Item | None:
        """The item already at ``key``, if it holds the same content."""
        known = self.index.by_key(self.name or "", key)
        if known and same(known[1], content):
            return Item(known[0], key, known[1])
        return None


class Grid:
    """A DNA grid of content (text, images: whatever the index's embedder
    takes), synchronously. ``Grid()`` opens ``Index()`` and ``DNA()``, both
    configured from the environment."""

    def __init__(self, index: Index | None = None, dna: DNA | None = None):
        self.dna = dna if dna is not None else DNA()
        self._core = _Core(index)

    @property
    def index(self) -> Index:
        return self._core.index

    def _open(self) -> _Core:
        if self._core.name is None:
            info = self.dna.get_grid()
            self._core.bind(self.dna._http.base_url, info.name, self.dna.list_nodes(limit=1))
        return self._core

    def _embed(self, contents: list[Content], kind: Kind) -> list[list[float]]:
        core = self._open()
        vectors = core.index.embed(contents, kind)
        core.check(vectors)
        return vectors

    def _create(self, key: str, content: Content, emb: list[float]) -> Item:
        core = self._core
        try:
            node = self.dna.create_node(emb, key=key)
        except Conflict:
            if found := core.existing(key, content):
                return found
            raise
        core.index.put(core.name or "", node.id, key, content)
        return Item(node.id, key, content)

    def add(self, content: Content, key: str | None = None) -> Item:
        """Add an item, findable by ``key``, a path unique in the grid; left
        out, a hash of the content, so adding it again returns the same item."""
        key = key or content_key(content)
        (emb,) = self._embed([content], "document")
        return self._create(key, content, emb)

    def add_many(self, items: Mapping[str, Content] | Iterable[Content]) -> list[Item]:
        """Add items, ``{key: content}`` or contents, embedded in one batch."""
        pairs = _pairs(items)
        embs = self._embed([c for _, c in pairs], "document")
        return [self._create(k, c, e) for (k, c), e in zip(pairs, embs)]

    def query(self, content: Content) -> Item:
        """A query from ``content``: it searches, and is never found."""
        (emb,) = self._embed([content], "query")
        node = self.dna.create_node(emb)
        self._core.index.put(self._core.name or "", node.id, None, content)
        return Item(node.id, None, content)

    def search(self, query: str | Item, k: int = 1, prefix: str = "") -> list[Result]:
        """The next ``k`` results of ``query`` not shown before, best first,
        among items whose key starts with ``prefix``; fewer when none are left."""
        edges: list[Edge] = []
        while len(edges) < k and (edge := self.dna.create_edge(query, prefix)):
            edges.append(edge)
        return [self._result(e) for e in edges]

    def ask(
        self, content: Content, k: int = 1, prefix: str = ""
    ) -> tuple[Item, list[Result]]:
        """A query from ``content`` and its first ``k`` results."""
        query = self.query(content)
        return query, self.search(query, k, prefix)

    def score(self, query: str | Item, item: str | Item, score: float) -> Result:
        """Score how well ``item`` fits ``query``, from -1 (wrong) to 1
        (perfect): a result, or any pair. Only scores train DNA."""
        return self._result(self.dna.put_edge(query, item, score))

    def results(self, query: str | Item) -> list[Result]:
        """Every result of ``query`` so far, with its score (``None`` if unscored)."""
        return [self._result(e) for e in self.dna.iter_edges(_id(query))]

    def get(self, item: str | Item) -> Item:
        """An item, with its content from the index."""
        node = self.dna.get_node(item)
        return self._open().item(node.id, node.key)

    def delete(self, item: str | Item) -> None:
        """Delete an item, its content in the index and its scores."""
        core = self._open()
        self.dna.delete_node(item)
        core.index.delete(core.name or "", _id(item))

    def _result(self, edge: Edge) -> Result:
        core = self._open()
        found = core.item(edge.id)
        if found.content is None and found.key is None:  # added elsewhere
            found = Item(edge.id, self.dna.get_node(edge.id).key, None)
        return Result(found.id, found.key, found.content, edge.score)


class AsyncGrid:
    """A DNA grid of content, asynchronously; embedding runs in a worker
    thread. The same methods as ``Grid``, awaited."""

    def __init__(self, index: Index | None = None, dna: AsyncDNA | None = None):
        self.dna = dna if dna is not None else AsyncDNA()
        self._core = _Core(index)
        self._lock = asyncio.Lock()

    @property
    def index(self) -> Index:
        return self._core.index

    async def _open(self) -> _Core:
        async with self._lock:
            if self._core.name is None:
                info = await self.dna.get_grid()
                first = await self.dna.list_nodes(limit=1)
                self._core.bind(self.dna._http.base_url, info.name, first)
        return self._core

    async def _embed(self, contents: list[Content], kind: Kind) -> list[list[float]]:
        core = await self._open()
        vectors = await asyncio.to_thread(core.index.embed, contents, kind)
        core.check(vectors)
        return vectors

    async def _create(self, key: str, content: Content, emb: list[float]) -> Item:
        core = self._core
        try:
            node = await self.dna.create_node(emb, key=key)
        except Conflict:
            if found := core.existing(key, content):
                return found
            raise
        core.index.put(core.name or "", node.id, key, content)
        return Item(node.id, key, content)

    async def add(self, content: Content, key: str | None = None) -> Item:
        key = key or content_key(content)
        (emb,) = await self._embed([content], "document")
        return await self._create(key, content, emb)

    async def add_many(
        self, items: Mapping[str, Content] | Iterable[Content]
    ) -> list[Item]:
        pairs = _pairs(items)
        embs = await self._embed([c for _, c in pairs], "document")
        return [await self._create(k, c, e) for (k, c), e in zip(pairs, embs)]

    async def query(self, content: Content) -> Item:
        (emb,) = await self._embed([content], "query")
        node = await self.dna.create_node(emb)
        self._core.index.put(self._core.name or "", node.id, None, content)
        return Item(node.id, None, content)

    async def search(
        self, query: str | Item, k: int = 1, prefix: str = ""
    ) -> list[Result]:
        edges: list[Edge] = []
        while len(edges) < k and (edge := await self.dna.create_edge(query, prefix)):
            edges.append(edge)
        return [await self._result(e) for e in edges]

    async def ask(
        self, content: Content, k: int = 1, prefix: str = ""
    ) -> tuple[Item, list[Result]]:
        query = await self.query(content)
        return query, await self.search(query, k, prefix)

    async def score(self, query: str | Item, item: str | Item, score: float) -> Result:
        return await self._result(await self.dna.put_edge(query, item, score))

    async def results(self, query: str | Item) -> list[Result]:
        return [await self._result(e) async for e in self.dna.iter_edges(_id(query))]

    async def get(self, item: str | Item) -> Item:
        node = await self.dna.get_node(item)
        return (await self._open()).item(node.id, node.key)

    async def delete(self, item: str | Item) -> None:
        core = await self._open()
        await self.dna.delete_node(item)
        core.index.delete(core.name or "", _id(item))

    async def _result(self, edge: Edge) -> Result:
        core = await self._open()
        found = core.item(edge.id)
        if found.content is None and found.key is None:  # added elsewhere
            found = Item(edge.id, (await self.dna.get_node(edge.id)).key, None)
        return Result(found.id, found.key, found.content, edge.score)
