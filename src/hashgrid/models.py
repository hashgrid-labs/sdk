"""The DNA wire format."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeVar

T = TypeVar("T")


def _from(cls: type[T], data: dict[str, Any]) -> T:
    """``cls`` from a JSON object, ignoring fields it does not have."""
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})  # type: ignore[arg-type]


@dataclass(frozen=True)
class GridInfo:
    """The grid an API key opens: its ``name``, and the most items it holds,
    ``max_items`` (``None``: no limit)."""

    name: str
    max_items: int | None = None


@dataclass(frozen=True)
class Item:
    """An item as DNA lists it, without its content: its ``key``, ``type``,
    ``size`` in bytes, ``etag``, and ``ctime`` and ``mtime`` in microseconds
    since the epoch. An image's write also gives the ``width`` and
    ``height`` it is kept at."""

    key: str
    type: str
    size: int
    etag: str
    ctime: int
    mtime: int
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class Result(Item):
    """An item a query returned, with ``score``, DNA's estimate of the grade
    you would give it."""

    score: float = 0.0
