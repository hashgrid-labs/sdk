"""The DNA wire format."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..index.content import Content


@dataclass(frozen=True)
class GridInfo:
    """The grid an API key opens: its ``name``, and the most nodes it holds,
    ``max_nodes`` (``None``: no limit)."""

    name: str
    max_nodes: int | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GridInfo:
        return cls(name=data["name"], max_nodes=data.get("max_nodes"))


@dataclass(frozen=True)
class GridKey:
    """A grid just created: its ``name``, its API key, which DNA shows only
    this once (keep it), and the most nodes it holds, ``max_nodes``
    (``None``: no limit)."""

    name: str
    api_key: str = field(repr=False)
    max_nodes: int | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GridKey:
        return cls(name=data["name"], api_key=data["api_key"], max_nodes=data.get("max_nodes"))


@dataclass(frozen=True)
class Node:
    """A node: its ``id`` (16 hex characters), its ``key`` (``None`` when it
    has none, and cannot be found by searches) and its embedding, ``emb``
    (empty when the reply leaves it out)."""

    id: str
    key: str | None
    emb: list[float]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Node:
        return cls(id=data["id"], key=data.get("key"), emb=data.get("emb") or [])


@dataclass(frozen=True)
class Edge:
    """An edge of a node, listed under it: ``id`` is the peer's, and
    ``score`` your grade, from -1 to 1, ``None`` for a result not scored yet."""

    id: str
    score: float | None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Edge:
        return cls(id=data["id"], score=data.get("score"))


@dataclass(frozen=True)
class Item:
    """A node of a ``Grid``, with its content from the local index: ``None``
    when this index does not have it (it was added elsewhere)."""

    id: str
    key: str | None
    content: Content | None


@dataclass(frozen=True)
class Result(Item):
    """A search result: an item, and your ``score`` of it for the query,
    ``None`` until you score it."""

    score: float | None = None
