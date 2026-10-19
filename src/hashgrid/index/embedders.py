"""Embedders: content to vectors, on your machine or with your own key. DNA
only ever receives the vectors.

Each embedder has a ``spec``, ``"provider:model"``, which names it in full,
and the ``modalities`` it takes. Use one per grid: vectors from different
models do not compare.

- ``fastembed:BAAI/bge-small-en-v1.5`` (the default): text, local, no account.
- ``fastembed:Qdrant/clip-ViT-B-32``, ``fastembed:jinaai/jina-clip-v1``: text
  and images in one space, local.
- ``openai:text-embedding-3-small``: text, ``$OPENAI_API_KEY``.
- ``voyage:voyage-3.5``: text; ``voyage:voyage-multimodal-3``: text and
  images; ``$VOYAGE_API_KEY``.

The local ones need ``pip install 'hashgrid[local]'``.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import httpx

from .content import Content, Image, modality

DEFAULT = "fastembed:BAAI/bge-small-en-v1.5"
MAXDIM = 2048  # the longest embedding DNA accepts

Kind = Literal["document", "query"]


@runtime_checkable
class Embedder(Protocol):
    """Turns content into vectors. Items are embedded as ``"document"`` and
    searches as ``"query"``, as retrieval models are trained to."""

    spec: str
    modalities: frozenset[str]

    def embed(self, contents: Sequence[Content], kind: Kind) -> list[list[float]]: ...


def cache_dir() -> Path:
    """Where local models are kept: ``$HASHGRID_CACHE``, else ``~/.cache/hashgrid``."""
    return Path(os.environ.get("HASHGRID_CACHE") or Path.home() / ".cache" / "hashgrid")


def _split(
    contents: Sequence[Content],
) -> tuple[list[tuple[int, str]], list[tuple[int, Image]]]:
    texts, images = [], []
    for i, c in enumerate(contents):
        (texts if modality(c) == "text" else images).append((i, c))
    return texts, images  # type: ignore[return-value]


def _merge(n: int, *parts: tuple[list[int], list[list[float]]]) -> list[list[float]]:
    out: list[list[float]] = [[] for _ in range(n)]
    for idx, vecs in parts:
        for i, v in zip(idx, vecs):
            out[i] = v
    return out


class FastEmbed:
    """A local model run by ``fastembed`` (ONNX, CPU). The first use downloads
    it into ``cache_dir()`` (~70 MB for the default)."""

    # Text and image towers trained into one space, by the name to use.
    CROSS_MODAL = {
        "Qdrant/clip-ViT-B-32": ("Qdrant/clip-ViT-B-32-text", "Qdrant/clip-ViT-B-32-vision"),
        "jinaai/jina-clip-v1": ("jinaai/jina-clip-v1", "jinaai/jina-clip-v1"),
    }

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.spec = f"fastembed:{model}"
        self.text_model, self.image_model = self.CROSS_MODAL.get(model, (model, None))
        self.modalities = frozenset({"text", "image"} if self.image_model else {"text"})
        self._text: Any = None
        self._image: Any = None

    @staticmethod
    def _fastembed() -> Any:
        try:
            import fastembed
        except ImportError as ex:
            raise ImportError(
                "Local embedders need fastembed: pip install 'hashgrid[local]' "
                "(with uvx: uvx --from 'hashgrid[local]' hashgrid-mcp)"
            ) from ex
        return fastembed

    def _texts(self, texts: list[str], kind: Kind) -> list[list[float]]:
        if self._text is None:
            fe = self._fastembed()
            self._text = fe.TextEmbedding(self.text_model, cache_dir=str(cache_dir() / "fastembed"))
        run = self._text.query_embed if kind == "query" else self._text.passage_embed
        return [v.tolist() for v in run(texts)]

    def _images(self, images: list[Image]) -> list[list[float]]:
        if self._image is None:
            fe = self._fastembed()
            self._image = fe.ImageEmbedding(self.image_model, cache_dir=str(cache_dir() / "fastembed"))
        from PIL import Image as PIL  # a fastembed dependency

        inputs = [str(im.path) if im.path else PIL.open(io.BytesIO(im.read())) for im in images]
        return [v.tolist() for v in self._image.embed(inputs)]

    def embed(self, contents: Sequence[Content], kind: Kind) -> list[list[float]]:
        texts, images = _split(contents)
        parts = []
        if texts:
            parts.append(([i for i, _ in texts], self._texts([t for _, t in texts], kind)))
        if images:
            parts.append(([i for i, _ in images], self._images([m for _, m in images])))
        return _merge(len(contents), *parts)


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
        self._http = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=timeout)

    def _body(self, contents: list[Content], kind: Kind) -> dict[str, Any]:
        raise NotImplementedError

    def embed(self, contents: Sequence[Content], kind: Kind) -> list[list[float]]:
        out: list[list[float]] = []
        contents = list(contents)
        for i in range(0, len(contents), self.batch):
            r = self._http.post(self.url, json=self._body(contents[i : i + self.batch], kind))
            r.raise_for_status()
            out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        return out


class OpenAI(_Remote):
    """OpenAI embeddings, text only, ``$OPENAI_API_KEY`` (``$OPENAI_BASE_URL``
    for a compatible endpoint). Models longer than DNA's 2048 values, such as
    ``text-embedding-3-large``, are shortened with ``dimensions``."""

    env = "OPENAI_API_KEY"
    modalities = frozenset({"text"})

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

    def _body(self, contents: list[Content], kind: Kind) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "input": contents}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        return body


class Voyage(_Remote):
    """Voyage AI embeddings, ``$VOYAGE_API_KEY``: a ``multimodal`` model takes
    text and images, the others text."""

    env = "VOYAGE_API_KEY"

    def __init__(
        self, model: str = "voyage-3.5", *, api_key: str | None = None, timeout: float = 60.0
    ):
        super().__init__(model, api_key, timeout)
        self.spec = f"voyage:{model}"
        self.multimodal = "multimodal" in model
        self.modalities = frozenset({"text", "image"} if self.multimodal else {"text"})
        path = "multimodalembeddings" if self.multimodal else "embeddings"
        self.url = f"https://api.voyageai.com/v1/{path}"

    def _body(self, contents: list[Content], kind: Kind) -> dict[str, Any]:
        if not self.multimodal:
            return {"model": self.model, "input": contents, "input_type": kind}
        inputs = [
            {"content": [{"type": "text", "text": c}]}
            if isinstance(c, str)
            else {"content": [{"type": "image_base64", "image_base64": c.data_url()}]}
            for c in contents
        ]
        return {"model": self.model, "inputs": inputs, "input_type": kind}


def embedder(spec: str | None = None) -> Embedder:
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
