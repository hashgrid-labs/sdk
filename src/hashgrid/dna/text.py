"""DNA from text: embed on this machine with an ``Embedder``, send only the
vectors. Text is never sent; keep it yourself, by key or id.

>>> from hashgrid.dna import DNA, TextGrid
>>> from hashgrid.embed import from_spec
>>> grid = TextGrid(DNA(), from_spec())
>>> grid.add("Lisbon, a city of seven hills", key="/cities/lisbon")
>>> query, results = grid.ask("a hilly European capital", k=3)
>>> grid.score(query, results[0], 1.0)
"""

from __future__ import annotations

import asyncio
from typing import Mapping, Sequence

from ..embed import Embedder
from .client import DNA, AsyncDNA, _id
from .errors import ModelMismatch
from .models import Edge, Node


class _Fit:
    """One model per grid: vectors must have the length of the grid's
    nodes. A grid records no model name, so two models of the same length
    are not told apart here."""

    def __init__(self, spec: str):
        self.spec = spec
        self.dim: int | None = None

    def known(self, dim: int | None) -> None:
        self.dim = dim

    def check(self, vectors: Sequence[Sequence[float]]) -> None:
        for v in vectors:
            if self.dim is None:
                self.dim = len(v)
            elif len(v) != self.dim:
                raise ModelMismatch(
                    f"{self.spec} gives vectors of {len(v)} values, but this grid's "
                    f"have {self.dim}: use the embedder the grid was built with"
                )


class TextGrid:
    """A grid over text, synchronously."""

    def __init__(self, dna: DNA, embedder: Embedder):
        self.dna, self.embedder = dna, embedder
        self._fit: _Fit | None = None

    def _check(self, vectors: list[list[float]]) -> list[list[float]]:
        if self._fit is None:
            self._fit = _Fit(self.embedder.spec)
            first = self.dna.list_nodes(limit=1)
            self._fit.known(len(first[0].emb) if first else None)
        self._fit.check(vectors)
        return vectors

    def add(self, text: str, key: str) -> Node:
        """Add an item: ``text`` embedded as a document, findable by ``key``."""
        (emb,) = self._check(self.embedder.embed_documents([text]))
        return self.dna.create_node(emb, key=key)

    def add_many(self, items: Mapping[str, str]) -> list[Node]:
        """Add items, ``{key: text}``, embedded in one batch."""
        keys = list(items)
        embs = self._check(self.embedder.embed_documents([items[k] for k in keys]))
        return [self.dna.create_node(e, key=k) for k, e in zip(keys, embs)]

    def query(self, text: str) -> Node:
        """A query node from ``text``: it searches, and is never found."""
        (emb,) = self._check(self.embedder.embed_queries([text]))
        return self.dna.create_node(emb)

    def search(self, query: str | Node, k: int = 1, prefix: str = "") -> list[Edge]:
        """The next ``k`` results of ``query`` not shown before, best first;
        fewer when the grid runs out."""
        out: list[Edge] = []
        while len(out) < k and (edge := self.dna.create_edge(query, prefix)):
            out.append(edge)
        return out

    def ask(
        self, text: str, k: int = 1, prefix: str = ""
    ) -> tuple[Node, list[Edge]]:
        """A query from ``text`` and its first ``k`` results."""
        query = self.query(text)
        return query, self.search(query, k, prefix)

    def score(self, query: str | Node, item: str | Node | Edge, score: float) -> Edge:
        """Score how well ``item`` fits ``query``, from -1 (wrong) to 1
        (perfect); a suggested result or any pair. Only scores train DNA."""
        return self.dna.put_edge(query, item, score)

    def results(self, query: str | Node) -> list[Edge]:
        """Every result of ``query`` so far, with its score (``None`` if unscored)."""
        return list(self.dna.iter_edges(_id(query)))


class AsyncTextGrid:
    """A grid over text, asynchronously; embedding runs in a worker thread."""

    def __init__(self, dna: AsyncDNA, embedder: Embedder):
        self.dna, self.embedder = dna, embedder
        self._fit: _Fit | None = None

    async def _check(self, vectors: list[list[float]]) -> list[list[float]]:
        if self._fit is None:
            fit = _Fit(self.embedder.spec)
            first = await self.dna.list_nodes(limit=1)
            fit.known(len(first[0].emb) if first else None)
            self._fit = fit
        self._fit.check(vectors)
        return vectors

    async def _docs(self, texts: list[str]) -> list[list[float]]:
        return await self._check(
            await asyncio.to_thread(self.embedder.embed_documents, texts)
        )

    async def add(self, text: str, key: str) -> Node:
        (emb,) = await self._docs([text])
        return await self.dna.create_node(emb, key=key)

    async def add_many(self, items: Mapping[str, str]) -> list[Node]:
        keys = list(items)
        embs = await self._docs([items[k] for k in keys])
        return [await self.dna.create_node(e, key=k) for k, e in zip(keys, embs)]

    async def query(self, text: str) -> Node:
        (emb,) = await self._check(
            await asyncio.to_thread(self.embedder.embed_queries, [text])
        )
        return await self.dna.create_node(emb)

    async def search(
        self, query: str | Node, k: int = 1, prefix: str = ""
    ) -> list[Edge]:
        out: list[Edge] = []
        while len(out) < k and (edge := await self.dna.create_edge(query, prefix)):
            out.append(edge)
        return out

    async def ask(
        self, text: str, k: int = 1, prefix: str = ""
    ) -> tuple[Node, list[Edge]]:
        query = await self.query(text)
        return query, await self.search(query, k, prefix)

    async def score(
        self, query: str | Node, item: str | Node | Edge, score: float
    ) -> Edge:
        return await self.dna.put_edge(query, item, score)

    async def results(self, query: str | Node) -> list[Edge]:
        return [e async for e in self.dna.iter_edges(_id(query))]
