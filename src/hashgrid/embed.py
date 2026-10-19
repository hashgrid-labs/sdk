"""Embedders: text to vectors, on your machine or with your own key. DNA only
ever receives the vectors.

Each embedder has a ``spec``, ``"provider:model"``, which names it in full:
use one per grid, as vectors from different models do not compare.

- ``fastembed:BAAI/bge-small-en-v1.5`` (the default): local, no account;
  ``pip install 'hashgrid[local]'``.
- ``openai:text-embedding-3-small``: ``$OPENAI_API_KEY``.
- ``voyage:voyage-3.5``: ``$VOYAGE_API_KEY``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import httpx

DEFAULT = "fastembed:BAAI/bge-small-en-v1.5"
MAXDIM = 2048  # the longest embedding DNA accepts


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Documents (items) and queries may be embedded
    differently, as retrieval models are trained to."""

    spec: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


def cache_dir() -> Path:
    """Where local models are kept: ``$HASHGRID_CACHE``, else ``~/.cache/hashgrid``."""
    return Path(os.environ.get("HASHGRID_CACHE") or Path.home() / ".cache" / "hashgrid")


class FastEmbed:
    """A local model run by ``fastembed`` (ONNX, CPU). The first use downloads
    the model (~70 MB for the default) into ``cache_dir()``."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.spec = f"fastembed:{model}"
        self.model = model
        self._m: Any = None

    def _load(self) -> Any:
        if self._m is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as ex:
                raise ImportError(
                    "The local embedder needs fastembed: pip install 'hashgrid[local]' "
                    "(with uvx: uvx --from 'hashgrid[local]' hashgrid-mcp)"
                ) from ex
            self._m = TextEmbedding(self.model, cache_dir=str(cache_dir() / "fastembed"))
        return self._m

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().passage_embed(list(texts))]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().query_embed(list(texts))]


class _Remote:
    """An embeddings API over HTTPS, with the user's own key."""

    url: str
    env: str
    batch = 128

    def __init__(self, model: str, api_key: str | None, timeout: float):
        key = api_key or os.environ.get(self.env)
        if not key:
            raise ValueError(f"Set {self.env} (or pass api_key=) to embed with {model}")
        self.model = model
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {key}"}, timeout=timeout
        )

    def _body(self, texts: list[str], kind: str) -> dict[str, Any]:
        raise NotImplementedError

    def _embed(self, texts: Sequence[str], kind: str) -> list[list[float]]:
        out: list[list[float]] = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch):
            r = self._http.post(self.url, json=self._body(texts[i : i + self.batch], kind))
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out += [d["embedding"] for d in data]
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "query")


class OpenAI(_Remote):
    """OpenAI embeddings, ``$OPENAI_API_KEY`` (``$OPENAI_BASE_URL`` for a
    compatible endpoint). Models longer than DNA's 2048 values, such as
    ``text-embedding-3-large``, are shortened with ``dimensions``."""

    env = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        super().__init__(model, api_key, timeout)
        if dimensions is None and model == "text-embedding-3-large":
            dimensions = MAXDIM
        self.dimensions = dimensions
        base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.url = base.rstrip("/") + "/embeddings"
        self.spec = f"openai:{model}" + (f"@{dimensions}" if dimensions else "")

    def _body(self, texts: list[str], kind: str) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        return body


class Voyage(_Remote):
    """Voyage AI embeddings, ``$VOYAGE_API_KEY``."""

    env = "VOYAGE_API_KEY"
    url = "https://api.voyageai.com/v1/embeddings"

    def __init__(
        self,
        model: str = "voyage-3.5",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        super().__init__(model, api_key, timeout)
        self.spec = f"voyage:{model}"

    def _body(self, texts: list[str], kind: str) -> dict[str, Any]:
        return {"model": self.model, "input": texts, "input_type": kind}


def from_spec(spec: str | None = None) -> Embedder:
    """The embedder a spec names: ``"fastembed:<model>"``, ``"openai:<model>"``
    (``"openai:<model>@<dimensions>"`` to shorten) or ``"voyage:<model>"``; a
    provider alone takes its default model. ``None``: ``$HASHGRID_EMBEDDER``,
    else the local default."""
    spec = spec or os.environ.get("HASHGRID_EMBEDDER") or DEFAULT
    provider, _, model = spec.partition(":")
    if provider == "fastembed":
        return FastEmbed(model) if model else FastEmbed()
    if provider == "openai":
        name, _, dims = model.partition("@")
        return OpenAI(name or "text-embedding-3-small", dimensions=int(dims) if dims else None)
    if provider == "voyage":
        return Voyage(model) if model else Voyage()
    raise ValueError(
        f"Unknown embedder {spec!r}: use fastembed:<model>, openai:<model> or voyage:<model>"
    )


__all__ = ["Embedder", "FastEmbed", "OpenAI", "Voyage", "from_spec", "DEFAULT"]
