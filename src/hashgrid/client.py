"""The DNA client: ``DNA`` (sync) and ``AsyncDNA`` for the grid and its
items, ``Query`` and ``AsyncQuery`` for a question asked of them. Each
request is built once, in ``_routes``, and sent by either."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Generic, Iterator, Mapping, Sequence, TypeVar, Union
from urllib.parse import quote

import httpx

from . import __version__
from .errors import GET_A_KEY, error_for
from .models import GridInfo, Item, Result, _from

T = TypeVar("T")

DEFAULT_URL = "https://dna.hashgrid.ai/api/v1"
PAGE = 100  # the largest page DNA lists
EMB = "application/vnd.hashgrid.emb+json"

Content = Union[str, bytes, Path, Mapping[str, Any], Sequence[float]]
"""What an item or a query holds: text (``str``), a record (a ``dict``), an
image (its ``bytes``, or a ``Path`` to its file) or an embedding from the
grid's model (a list of floats)."""


@dataclass
class _Request(Generic[T]):
    method: str
    path: str
    parse: Callable[[httpx.Response], T]
    params: dict[str, Any] | None = None
    json: Any = None
    content: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def send(self, http: httpx.Client) -> T:
        return _parse(self, http.request(self.method, self.path, params=self.params, json=self.json,
                                         content=self.content, headers=self.headers))

    async def asend(self, http: httpx.AsyncClient) -> T:
        return _parse(self, await http.request(self.method, self.path, params=self.params, json=self.json,
                                               content=self.content, headers=self.headers))


def _parse(req: _Request[T], r: httpx.Response) -> T:
    if r.is_error:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise error_for(r.status_code, detail)
    return req.parse(r)


def _none(_: httpx.Response) -> None:
    return None


def _key(item: str | Item) -> str:
    """An item's key, from the key or the item, quoted for the URL."""
    return quote(item if isinstance(item, str) else item.key, safe="/")


def _body(content: Content, type: str | None) -> tuple[bytes, dict[str, str]]:
    """A request body and its ``Content-Type``. Bytes go without one, and DNA
    tells the image format from them."""
    if isinstance(content, Path):
        content = content.read_bytes()
    if isinstance(content, str):
        data, found = content.encode(), "text/plain; charset=utf-8"
    elif isinstance(content, (bytes, bytearray)):
        data, found = bytes(content), None
    elif isinstance(content, Mapping):
        data, found = json.dumps(content).encode(), "application/json"
    elif isinstance(content, Sequence):
        data, found = json.dumps([float(v) for v in content]).encode(), EMB
    else:
        raise TypeError(f"Content is str, bytes, a Path, a dict or a list of floats, not {content.__class__.__name__}")
    mime = type or found
    return data, {"Content-Type": mime} if mime else {}


def _content(r: httpx.Response) -> Any:
    """Content as DNA returns it: text as ``str``, a record as a ``dict``, an
    embedding as a list of floats, an image as its ``bytes``."""
    mime = r.headers.get("content-type", "").partition(";")[0].strip()
    if mime == "text/plain":
        return r.text
    if mime in ("application/json", EMB):
        return r.json()
    return r.content


class _routes:
    """Every route, as a request to send."""

    @staticmethod
    def create_grid(name: str) -> _Request[str]:
        return _Request("POST", "/", lambda r: r.json()["api_key"], json={"name": name})

    @staticmethod
    def info() -> _Request[GridInfo]:
        return _Request("GET", "/", lambda r: _from(GridInfo, r.json()))

    @staticmethod
    def delete_grid() -> _Request[None]:
        return _Request("DELETE", "/", _none)

    @staticmethod
    def put(key: str, content: Content, type: str | None, if_match: str | None,
            if_none_match: str | None) -> _Request[Item]:
        data, headers = _body(content, type)
        if if_match is not None:
            headers["If-Match"] = if_match if if_match == "*" else f'"{if_match}"'
        if if_none_match is not None:
            headers["If-None-Match"] = if_none_match if if_none_match == "*" else f'"{if_none_match}"'
        return _Request("PUT", f"/items/{key}", lambda r: _from(Item, r.json()), content=data, headers=headers)

    @staticmethod
    def get(key: str) -> _Request[Any]:
        return _Request("GET", f"/items/{key}", _content)

    @staticmethod
    def delete(key: str) -> _Request[None]:
        return _Request("DELETE", f"/items/{key}", _none)

    @staticmethod
    def list_items(prefix: str, after: str | None, limit: int) -> _Request[tuple[list[Item], str | None]]:
        params = {"prefix": prefix, "limit": limit, **({"after": after} if after is not None else {})}
        return _Request("GET", "/items", lambda r: ([_from(Item, i) for i in r.json()["items"]], r.json()["next"]),
                        params=params)

    @staticmethod
    def query(content: Content, type: str | None) -> _Request[str]:
        data, headers = _body(content, type)
        return _Request("POST", "/queries", lambda r: r.json()["id"], content=data, headers=headers)

    @staticmethod
    def results(query: str, limit: int, prefix: str | Sequence[str], ungraded: bool) -> _Request[list[Result]]:
        params: dict[str, Any] = {"limit": limit, "prefix": [prefix] if isinstance(prefix, str) else list(prefix)}
        if ungraded:
            params["ungraded"] = "true"
        return _Request("GET", f"/queries/{query}/results",
                        lambda r: [_from(Result, x) for x in r.json()["results"]], params=params)

    @staticmethod
    def grade(query: str, key: str, score: float) -> _Request[float]:
        return _Request("PUT", f"/queries/{query}/grades/{key}", lambda r: r.json()["score"],
                        json={"score": score})

    @staticmethod
    def grades(query: str) -> _Request[dict[str, float]]:
        return _Request("GET", f"/queries/{query}/grades",
                        lambda r: {g["key"]: g["score"] for g in r.json()["grades"]})

    @staticmethod
    def delete_grade(query: str, key: str) -> _Request[None]:
        return _Request("DELETE", f"/queries/{query}/grades/{key}", _none)

    @staticmethod
    def delete_query(query: str) -> _Request[None]:
        return _Request("DELETE", f"/queries/{query}", _none)


def _url(base_url: str | None) -> str:
    return (base_url or os.environ.get("HASHGRID_DNA_URL") or DEFAULT_URL).rstrip("/")


def _headers(key: str | None = None) -> dict[str, str]:
    agent = {"User-Agent": f"hashgrid-python/{__version__}"}
    return {"Authorization": f"Bearer {key}", **agent} if key else agent


def _api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("HASHGRID_API_KEY")
    if not key:
        raise ValueError(f"Pass api_key= or set HASHGRID_API_KEY to your DNA API key (hg-…). {GET_A_KEY}")
    return key


def _query_id(content: Content | None, id: str | None) -> str | None:
    """``id`` to reopen a query, or None to create one from ``content``."""
    if (content is None) == (id is None):
        raise TypeError("Pass the query's content to create it, or id= to reopen it, one of them")
    return id


class DNA:
    """A DNA grid, synchronously, opened by its API key: ``api_key``, else
    ``$HASHGRID_API_KEY``. The URL defaults to ``$HASHGRID_DNA_URL``, else
    the hosted API.

    >>> dna = DNA(DNA.signup("mine"))   # keep the key: DNA(api_key) next time
    >>> dna.put("cities/lisbon", "Lisbon, a hilly coastal capital")
    >>> query = dna.query("a capital by the sea")
    >>> for result in query.results(limit=3):
    ...     query.grade(result, 1.0)
    """

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=_url(base_url), headers=_headers(_api_key(api_key)), timeout=timeout,
                                  transport=transport)

    @staticmethod
    def signup(name: str = "hashgrid", *, base_url: str | None = None, timeout: float = 30.0,
               transport: httpx.BaseTransport | None = None) -> str:
        """``POST /``: a new grid, no key needed. Returns its API key, shown
        only now: keep it, and open the grid with ``DNA(api_key)``."""
        with httpx.Client(base_url=_url(base_url), headers=_headers(), timeout=timeout, transport=transport) as http:
            return _routes.create_grid(name).send(http)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> DNA:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def info(self) -> GridInfo:
        """``GET /``: the grid's name, item limit and embedder."""
        return _routes.info().send(self._http)

    def delete_grid(self) -> None:
        """``DELETE /``: delete the grid and everything in it, for good; its
        key stops working."""
        _routes.delete_grid().send(self._http)

    def put(self, key: str, content: Content, *, type: str | None = None, if_match: str | None = None,
            if_none_match: str | None = None) -> Item:
        """``PUT /items/{key}``: write the item at ``key``, like ``tools/42``.
        ``type`` overrides the content type; ``if_none_match="*"`` writes
        only a new item, ``if_match=etag`` replaces only that version."""
        return _routes.put(_key(key), content, type, if_match, if_none_match).send(self._http)

    def put_many(self, items: Mapping[str, Content]) -> list[Item]:
        """Write ``{key: content}``, one item at a time."""
        return [self.put(key, content) for key, content in items.items()]

    def get(self, item: str | Item) -> Any:
        """``GET /items/{key}``: the item's content as DNA keeps it."""
        return _routes.get(_key(item)).send(self._http)

    def delete(self, item: str | Item) -> None:
        """``DELETE /items/{key}``: the item and its grades."""
        _routes.delete(_key(item)).send(self._http)

    def items(self, prefix: str = "") -> Iterator[Item]:
        """``GET /items``: every item whose key starts with ``prefix``, in key
        order, page by page."""
        after: str | None = None
        while True:
            page, after = _routes.list_items(prefix, after, PAGE).send(self._http)
            yield from page
            if after is None:
                return

    def query(self, content: Content | None = None, *, type: str | None = None, id: str | None = None) -> Query:
        """``POST /queries``: a question to ask of the items, from its
        ``content``; or, with ``id``, one asked before, so its grades add up."""
        if (found := _query_id(content, id)) is None:
            found = _routes.query(content, type).send(self._http)  # type: ignore[arg-type]
        return Query(self, found)


class Query:
    """A question asked of a grid's items: its ``id``, and what DNA answers."""

    def __init__(self, dna: DNA, id: str):
        self._http, self.id = dna._http, id

    def __repr__(self) -> str:
        return f"Query({self.id!r})"

    def results(self, limit: int = 32, prefix: str | Sequence[str] = "", *, ungraded: bool = False) -> list[Result]:
        """``GET /queries/{id}/results``: the items that fit best, best first,
        under ``prefix`` (or any of several); with ``ungraded``, only those
        not graded yet."""
        return _routes.results(self.id, limit, prefix, ungraded).send(self._http)

    def grade(self, item: str | Item, score: float) -> float:
        """``PUT /queries/{id}/grades/{key}``: how well ``item`` fits, from -1
        (wrong) to 1 (right). Returns the score as kept."""
        return _routes.grade(self.id, _key(item), score).send(self._http)

    def grades(self) -> dict[str, float]:
        """``GET /queries/{id}/grades``: ``{key: score}``."""
        return _routes.grades(self.id).send(self._http)

    def delete_grade(self, item: str | Item) -> None:
        """``DELETE /queries/{id}/grades/{key}``."""
        _routes.delete_grade(self.id, _key(item)).send(self._http)

    def delete(self) -> None:
        """``DELETE /queries/{id}``: the query and its grades."""
        _routes.delete_query(self.id).send(self._http)


class AsyncDNA:
    """``DNA``, asynchronously: the same methods, awaited."""

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._http = httpx.AsyncClient(base_url=_url(base_url), headers=_headers(_api_key(api_key)),
                                       timeout=timeout, transport=transport)

    @staticmethod
    async def signup(name: str = "hashgrid", *, base_url: str | None = None, timeout: float = 30.0,
                     transport: httpx.AsyncBaseTransport | None = None) -> str:
        async with httpx.AsyncClient(base_url=_url(base_url), headers=_headers(), timeout=timeout,
                                     transport=transport) as http:
            return await _routes.create_grid(name).asend(http)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncDNA:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def info(self) -> GridInfo:
        return await _routes.info().asend(self._http)

    async def delete_grid(self) -> None:
        await _routes.delete_grid().asend(self._http)

    async def put(self, key: str, content: Content, *, type: str | None = None, if_match: str | None = None,
                  if_none_match: str | None = None) -> Item:
        return await _routes.put(_key(key), content, type, if_match, if_none_match).asend(self._http)

    async def put_many(self, items: Mapping[str, Content]) -> list[Item]:
        return [await self.put(key, content) for key, content in items.items()]

    async def get(self, item: str | Item) -> Any:
        return await _routes.get(_key(item)).asend(self._http)

    async def delete(self, item: str | Item) -> None:
        await _routes.delete(_key(item)).asend(self._http)

    async def items(self, prefix: str = "") -> AsyncIterator[Item]:
        after: str | None = None
        while True:
            page, after = await _routes.list_items(prefix, after, PAGE).asend(self._http)
            for item in page:
                yield item
            if after is None:
                return

    async def query(self, content: Content | None = None, *, type: str | None = None,
                    id: str | None = None) -> AsyncQuery:
        if (found := _query_id(content, id)) is None:
            found = await _routes.query(content, type).asend(self._http)  # type: ignore[arg-type]
        return AsyncQuery(self, found)


class AsyncQuery:
    """``Query``, asynchronously: the same methods, awaited."""

    def __init__(self, dna: AsyncDNA, id: str):
        self._http, self.id = dna._http, id

    def __repr__(self) -> str:
        return f"AsyncQuery({self.id!r})"

    async def results(self, limit: int = 32, prefix: str | Sequence[str] = "", *,
                      ungraded: bool = False) -> list[Result]:
        return await _routes.results(self.id, limit, prefix, ungraded).asend(self._http)

    async def grade(self, item: str | Item, score: float) -> float:
        return await _routes.grade(self.id, _key(item), score).asend(self._http)

    async def grades(self) -> dict[str, float]:
        return await _routes.grades(self.id).asend(self._http)

    async def delete_grade(self, item: str | Item) -> None:
        await _routes.delete_grade(self.id, _key(item)).asend(self._http)

    async def delete(self) -> None:
        await _routes.delete_query(self.id).asend(self._http)
