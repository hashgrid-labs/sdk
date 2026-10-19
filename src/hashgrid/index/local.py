"""The ``Index``: the local half of a grid. It embeds content with its
embedder and keeps the content in sqlite, by DNA id; DNA holds only the
vectors. It also records the embedder each grid was built with."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Sequence

from ..errors import ModelMismatch
from .content import Content, Image, modality
from .embedders import Embedder, Kind, embedder as _embedder


def default_path() -> Path:
    """``$HASHGRID_INDEX``, else ``~/.hashgrid/index.sqlite``."""
    return Path(os.environ.get("HASHGRID_INDEX") or Path.home() / ".hashgrid" / "index.sqlite")


class Index:
    """``Index()`` embeds with the local default (or ``$HASHGRID_EMBEDDER``)
    and keeps content in ``~/.hashgrid/index.sqlite`` (or ``$HASHGRID_INDEX``).
    ``Index("voyage:voyage-multimodal-3")`` picks another embedder by spec,
    ``Index(OpenAI(dimensions=512))`` passes one; ``path=":memory:"`` keeps
    nothing."""

    def __init__(self, embedder: str | Embedder | None = None, *, path: str | Path | None = None):
        self.embedder = embedder if isinstance(embedder, Embedder) else _embedder(embedder)
        path = default_path() if path is None else path
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS grids (
                grid TEXT PRIMARY KEY,
                embedder TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS items (
                grid TEXT NOT NULL,
                id TEXT NOT NULL,
                key TEXT,
                kind TEXT NOT NULL,
                text TEXT,
                path TEXT,
                data BLOB,
                PRIMARY KEY (grid, id)
            );
            CREATE INDEX IF NOT EXISTS items_key ON items (grid, key);
            """
        )

    @property
    def spec(self) -> str:
        return self.embedder.spec

    def embed(self, contents: Sequence[Content], kind: Kind) -> list[list[float]]:
        for c in contents:
            if modality(c) not in self.embedder.modalities:
                raise ValueError(
                    f"{self.spec} embeds {' and '.join(sorted(self.embedder.modalities))} only; "
                    "for images use Index('fastembed:Qdrant/clip-ViT-B-32') "
                    "or Index('voyage:voyage-multimodal-3')"
                )
        return self.embedder.embed(list(contents), kind) if contents else []

    def bind(self, grid: str) -> None:
        """Record this index's embedder for ``grid``, or refuse it if the grid
        was built with another."""
        row = self.db.execute("SELECT embedder FROM grids WHERE grid = ?", (grid,)).fetchone()
        if row is None:
            with self.db:
                self.db.execute("INSERT INTO grids (grid, embedder) VALUES (?, ?)", (grid, self.spec))
        elif row[0] != self.spec:
            raise ModelMismatch(
                f"This grid was built with {row[0]}, not {self.spec}: "
                f"use Index({row[0]!r}) (HASHGRID_EMBEDDER={row[0]})"
            )

    def put(self, grid: str, id: str, key: str | None, content: Content) -> None:
        if isinstance(content, str):
            row = ("text", content, None, None)
        else:
            row = ("image", None, str(content.path) if content.path else None, content.data)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO items (grid, id, key, kind, text, path, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (grid, id, key, *row),
            )

    @staticmethod
    def _content(kind: str, text: str | None, path: str | None, data: bytes | None) -> Content:
        if kind == "text":
            return text or ""
        return Image(path) if path else Image(data=data)

    def get(self, grid: str, id: str) -> tuple[str | None, Content] | None:
        """``(key, content)`` of node ``id``, if this index has it."""
        row = self.db.execute(
            "SELECT key, kind, text, path, data FROM items WHERE grid = ? AND id = ?", (grid, id)
        ).fetchone()
        return (row[0], self._content(*row[1:])) if row else None

    def by_key(self, grid: str, key: str) -> tuple[str, Content] | None:
        """``(id, content)`` of the item with ``key``."""
        row = self.db.execute(
            "SELECT id, kind, text, path, data FROM items WHERE grid = ? AND key = ?", (grid, key)
        ).fetchone()
        return (row[0], self._content(*row[1:])) if row else None

    def delete(self, grid: str, id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM items WHERE grid = ? AND id = ?", (grid, id))

    def close(self) -> None:
        self.db.close()
