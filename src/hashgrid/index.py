"""The local index: a map from DNA ids to text, in sqlite: DNA holds the vectors, this
holds what they mean, and it never leaves the machine. It also records the
embedder each grid was built with."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def default_path() -> Path:
    """``$HASHGRID_INDEX``, else ``~/.hashgrid/index.sqlite``."""
    return Path(os.environ.get("HASHGRID_INDEX") or Path.home() / ".hashgrid" / "index.sqlite")


class Index:
    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path is not None else default_path()
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS grids (
                grid TEXT PRIMARY KEY,
                embedder TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS texts (
                grid TEXT NOT NULL,
                id TEXT NOT NULL,
                key TEXT,
                text TEXT NOT NULL,
                PRIMARY KEY (grid, id)
            );
            CREATE INDEX IF NOT EXISTS texts_key ON texts (grid, key);
            """
        )

    def embedder(self, grid: str) -> str | None:
        row = self.db.execute(
            "SELECT embedder FROM grids WHERE grid = ?", (grid,)
        ).fetchone()
        return row[0] if row else None

    def set_embedder(self, grid: str, spec: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO grids (grid, embedder) VALUES (?, ?)", (grid, spec)
            )

    def put(self, grid: str, id: str, key: str | None, text: str) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO texts (grid, id, key, text) VALUES (?, ?, ?, ?)",
                (grid, id, key, text),
            )

    def text(self, grid: str, id: str) -> str | None:
        row = self.db.execute(
            "SELECT text FROM texts WHERE grid = ? AND id = ?", (grid, id)
        ).fetchone()
        return row[0] if row else None

    def item(self, grid: str, id: str) -> tuple[str | None, str] | None:
        """``(key, text)`` of node ``id``."""
        row = self.db.execute(
            "SELECT key, text FROM texts WHERE grid = ? AND id = ?", (grid, id)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def by_key(self, grid: str, key: str) -> tuple[str, str] | None:
        """``(id, text)`` of the item with ``key``."""
        row = self.db.execute(
            "SELECT id, text FROM texts WHERE grid = ? AND key = ?", (grid, key)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def delete(self, grid: str, id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM texts WHERE grid = ? AND id = ?", (grid, id))

    def close(self) -> None:
        self.db.close()
