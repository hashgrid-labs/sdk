"""Typed clients for the DNA HTTP API: ``DNA`` (sync) and ``AsyncDNA``. One
method per route, named after it; each request is built once, in ``_routes``,
and sent by either client."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Generic, Iterator, TypeVar

import httpx

from .. import __version__
from ..errors import GET_A_KEY, DNAError, error_for
from .models import Edge, GridInfo, Node

T = TypeVar("T")

DEFAULT_URL = "https://dna.hashgrid.ai/api/v1"
MAXN = 1024  # the largest page
DEFN = 32  # the default page


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()
"""For ``update_node``: leave the field as it is."""


@dataclass
class _Request(Generic[T]):
    method: str
    path: str
    parse: Callable[[httpx.Response], T]
    params: dict[str, Any] | None = None
    json: Any = None
    # Statuses that are an answer, not an error: status -> value.
    answers: dict[int, Any] = field(default_factory=dict)


def _id(value: Any) -> str:
    """An id, or anything with one: a ``Node``, ``Edge``, ``Item`` or ``Result``."""
    return value if isinstance(value, str) else value.id


def _page(cursor: str | None, limit: int) -> dict[str, Any]:
    if not 1 <= limit <= MAXN:
        raise ValueError(f"limit must be 1 to {MAXN}")
    return {"limit": limit, **({"cursor": cursor} if cursor is not None else {})}


def _score(score: float) -> dict[str, float]:
    if not math.isfinite(score):
        raise ValueError("score must be a finite number")
    return {"score": float(score)}


def _none(_: httpx.Response) -> None:
    return None


class _routes:
    """Every route, as a request to send."""

    @staticmethod
    def get_grid() -> _Request[GridInfo]:
        return _Request("GET", "/", lambda r: GridInfo.from_json(r.json()))

    @staticmethod
    def list_nodes(cursor: str | None, limit: int) -> _Request[list[Node]]:
        return _Request(
            "GET",
            "/nodes",
            lambda r: [Node.from_json(n) for n in r.json()],
            params=_page(cursor, limit),
        )

    @staticmethod
    def get_node(nid: str) -> _Request[Node]:
        return _Request("GET", f"/nodes/{nid}", lambda r: Node.from_json(r.json()))

    @staticmethod
    def create_node(emb: list[float], key: str | None) -> _Request[Node]:
        body: dict[str, Any] = {"emb": [float(x) for x in emb]}
        if key is not None:
            body["key"] = key
        return _Request("POST", "/nodes", lambda r: Node.from_json(r.json()), json=body)

    @staticmethod
    def update_node(nid: str, key: Any, emb: Any) -> _Request[Node]:
        body: dict[str, Any] = {}
        if key is not UNSET:
            body["key"] = key
        if emb is not UNSET:
            body["emb"] = [float(x) for x in emb]
        return _Request(
            "PATCH", f"/nodes/{nid}", lambda r: Node.from_json(r.json()), json=body
        )

    @staticmethod
    def delete_node(nid: str) -> _Request[None]:
        return _Request("DELETE", f"/nodes/{nid}", _none)

    @staticmethod
    def list_edges(nid: str, cursor: str | None, limit: int) -> _Request[list[Edge]]:
        return _Request(
            "GET",
            f"/nodes/{nid}/edges",
            lambda r: [Edge.from_json(e) for e in r.json()],
            params=_page(cursor, limit),
        )

    @staticmethod
    def get_edge(nid: str, eid: str) -> _Request[Edge]:
        return _Request(
            "GET", f"/nodes/{nid}/edges/{eid}", lambda r: Edge.from_json(r.json())
        )

    @staticmethod
    def create_edge(nid: str, prefix: str) -> _Request[Edge | None]:
        return _Request(
            "POST",
            f"/nodes/{nid}/edges",
            lambda r: Edge.from_json(r.json()),
            json={"prefix": prefix},
            answers={400: None},  # nothing left to suggest
        )

    @staticmethod
    def put_edge(nid: str, eid: str, score: float) -> _Request[Edge]:
        return _Request(
            "PUT",
            f"/nodes/{nid}/edges/{eid}",
            lambda r: Edge.from_json(r.json()),
            json=_score(score),
        )

    @staticmethod
    def update_edge(nid: str, eid: str, score: float) -> _Request[Edge]:
        return _Request(
            "PATCH",
            f"/nodes/{nid}/edges/{eid}",
            lambda r: Edge.from_json(r.json()),
            json=_score(score),
        )

    @staticmethod
    def delete_edge(nid: str, eid: str) -> _Request[None]:
        return _Request("DELETE", f"/nodes/{nid}/edges/{eid}", _none)


def _settings(api_key: str | None, base_url: str | None) -> tuple[str, str]:
    key = api_key or os.environ.get("HASHGRID_API_KEY")
    if not key:
        raise ValueError(
            f"Pass api_key= or set HASHGRID_API_KEY to your DNA API key (hg-…). {GET_A_KEY}"
        )
    url = base_url or os.environ.get("HASHGRID_DNA_URL") or DEFAULT_URL
    return key, url.rstrip("/")


def _headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "User-Agent": f"hashgrid-python/{__version__}",
    }


def _result(req: _Request[T], r: httpx.Response) -> T:
    if r.status_code in req.answers:
        return req.answers[r.status_code]
    if r.is_error:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise error_for(r.status_code, detail)
    return req.parse(r)


class DNA:
    """The DNA API, synchronously. The API key defaults to ``$HASHGRID_API_KEY``
    and the URL to ``$HASHGRID_DNA_URL``, else the hosted API.

    >>> dna = DNA()
    >>> item = dna.create_node([0.1, 0.2], key="/items/1")
    >>> query = dna.create_node([0.1, 0.3])
    >>> edge = dna.create_edge(query, prefix="/items/")   # search
    >>> dna.put_edge(query, edge, 1.0)                    # score
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        key, url = _settings(api_key, base_url)
        self._http = httpx.Client(
            base_url=url, headers=_headers(key), timeout=timeout, transport=transport
        )

    def _send(self, req: _Request[T]) -> T:
        r = self._http.request(req.method, req.path, params=req.params, json=req.json)
        return _result(req, r)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> DNA:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_grid(self) -> GridInfo:
        """``GET /``: the grid the API key opens."""
        return self._send(_routes.get_grid())

    def list_nodes(self, cursor: str | None = None, limit: int = DEFN) -> list[Node]:
        """``GET /nodes``: a page of nodes, in id order; for the next page pass
        the last id as ``cursor``."""
        return self._send(_routes.list_nodes(cursor, limit))

    def iter_nodes(self, page: int = MAXN) -> Iterator[Node]:
        """Every node, page by page."""
        cursor = None
        while nodes := self.list_nodes(cursor, page):
            yield from nodes
            cursor = nodes[-1].id

    def get_node(self, nid: str | Node) -> Node:
        """``GET /nodes/{nid}``."""
        return self._send(_routes.get_node(_id(nid)))

    def create_node(self, emb: list[float], key: str | None = None) -> Node:
        """``POST /nodes``: an item gets a ``key``, a path unique in the grid,
        so searches can find it; a query gets none."""
        return self._send(_routes.create_node(emb, key))

    def update_node(
        self, nid: str | Node, *, key: str | None = UNSET, emb: list[float] = UNSET
    ) -> Node:
        """``PATCH /nodes/{nid}``: a field left out is kept; ``key=None``
        removes the key, so the node can no longer be found."""
        return self._send(_routes.update_node(_id(nid), key, emb))

    def delete_node(self, nid: str | Node) -> None:
        """``DELETE /nodes/{nid}``: the node and every edge from or to it."""
        self._send(_routes.delete_node(_id(nid)))

    def list_edges(
        self, nid: str | Node, cursor: str | None = None, limit: int = DEFN
    ) -> list[Edge]:
        """``GET /nodes/{nid}/edges``: a page of a query's results so far."""
        return self._send(_routes.list_edges(_id(nid), cursor, limit))

    def iter_edges(self, nid: str | Node, page: int = MAXN) -> Iterator[Edge]:
        """Every edge of ``nid``, page by page."""
        cursor = None
        while edges := self.list_edges(nid, cursor, page):
            yield from edges
            cursor = edges[-1].id

    def get_edge(self, nid: str | Node, eid: str | Node | Edge) -> Edge:
        """``GET /nodes/{nid}/edges/{eid}``."""
        return self._send(_routes.get_edge(_id(nid), _id(eid)))

    def create_edge(self, nid: str | Node, prefix: str = "") -> Edge | None:
        """``POST /nodes/{nid}/edges``, the search: the best node not yet shown
        to ``nid``, among those whose key starts with ``prefix``, stored as an
        edge without a score; ``None`` when nothing is left."""
        return self._send(_routes.create_edge(_id(nid), prefix))

    search = create_edge

    def put_edge(
        self, nid: str | Node, eid: str | Node | Edge, score: float
    ) -> Edge:
        """``PUT /nodes/{nid}/edges/{eid}``: score any pair, from -1 to 1,
        creating the edge if DNA never suggested it."""
        return self._send(_routes.put_edge(_id(nid), _id(eid), score))

    score = put_edge

    def update_edge(
        self, nid: str | Node, eid: str | Node | Edge, score: float
    ) -> Edge:
        """``PATCH /nodes/{nid}/edges/{eid}``: score an existing edge."""
        return self._send(_routes.update_edge(_id(nid), _id(eid), score))

    def delete_edge(self, nid: str | Node, eid: str | Node | Edge) -> None:
        """``DELETE /nodes/{nid}/edges/{eid}``: forget a result, so it can be
        suggested again."""
        self._send(_routes.delete_edge(_id(nid), _id(eid)))


class AsyncDNA:
    """The DNA API, asynchronously; the same methods as ``DNA``, awaited."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        key, url = _settings(api_key, base_url)
        self._http = httpx.AsyncClient(
            base_url=url, headers=_headers(key), timeout=timeout, transport=transport
        )

    async def _send(self, req: _Request[T]) -> T:
        r = await self._http.request(
            req.method, req.path, params=req.params, json=req.json
        )
        return _result(req, r)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncDNA:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def get_grid(self) -> GridInfo:
        return await self._send(_routes.get_grid())

    async def list_nodes(
        self, cursor: str | None = None, limit: int = DEFN
    ) -> list[Node]:
        return await self._send(_routes.list_nodes(cursor, limit))

    async def iter_nodes(self, page: int = MAXN) -> AsyncIterator[Node]:
        cursor = None
        while nodes := await self.list_nodes(cursor, page):
            for n in nodes:
                yield n
            cursor = nodes[-1].id

    async def get_node(self, nid: str | Node) -> Node:
        return await self._send(_routes.get_node(_id(nid)))

    async def create_node(self, emb: list[float], key: str | None = None) -> Node:
        return await self._send(_routes.create_node(emb, key))

    async def update_node(
        self, nid: str | Node, *, key: str | None = UNSET, emb: list[float] = UNSET
    ) -> Node:
        return await self._send(_routes.update_node(_id(nid), key, emb))

    async def delete_node(self, nid: str | Node) -> None:
        await self._send(_routes.delete_node(_id(nid)))

    async def list_edges(
        self, nid: str | Node, cursor: str | None = None, limit: int = DEFN
    ) -> list[Edge]:
        return await self._send(_routes.list_edges(_id(nid), cursor, limit))

    async def iter_edges(self, nid: str | Node, page: int = MAXN) -> AsyncIterator[Edge]:
        cursor = None
        while edges := await self.list_edges(nid, cursor, page):
            for e in edges:
                yield e
            cursor = edges[-1].id

    async def get_edge(self, nid: str | Node, eid: str | Node | Edge) -> Edge:
        return await self._send(_routes.get_edge(_id(nid), _id(eid)))

    async def create_edge(self, nid: str | Node, prefix: str = "") -> Edge | None:
        return await self._send(_routes.create_edge(_id(nid), prefix))

    search = create_edge

    async def put_edge(
        self, nid: str | Node, eid: str | Node | Edge, score: float
    ) -> Edge:
        return await self._send(_routes.put_edge(_id(nid), _id(eid), score))

    score = put_edge

    async def update_edge(
        self, nid: str | Node, eid: str | Node | Edge, score: float
    ) -> Edge:
        return await self._send(_routes.update_edge(_id(nid), _id(eid), score))

    async def delete_edge(self, nid: str | Node, eid: str | Node | Edge) -> None:
        await self._send(_routes.delete_edge(_id(nid), _id(eid)))


__all__ = ["DNA", "AsyncDNA", "DNAError", "UNSET", "DEFAULT_URL"]
