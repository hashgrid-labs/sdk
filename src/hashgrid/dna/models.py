"""The DNA wire format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..index.content import Content


@dataclass(frozen=True)
class GridInfo:
    """The grid an API key opens."""

    name: str

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> GridInfo:
        return cls(name=data["name"])


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
