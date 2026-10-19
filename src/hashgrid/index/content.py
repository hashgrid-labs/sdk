"""What an index holds: text, or an ``Image``."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class Image:
    """An image, from a file (``Image("cat.jpg")``) or from bytes
    (``Image(data=…)``). A file is kept by its path, bytes as they are."""

    path: Path | None = None
    data: bytes | None = None

    def __init__(self, path: str | Path | None = None, *, data: bytes | None = None):
        if (path is None) == (data is None):
            raise ValueError("Image takes a path or data=, not both")
        object.__setattr__(self, "path", Path(path).expanduser().resolve() if path else None)
        object.__setattr__(self, "data", data)

    def read(self) -> bytes:
        return self.data if self.data is not None else self.path.read_bytes()  # type: ignore[union-attr]

    @property
    def media_type(self) -> str:
        if self.path is not None:
            return mimetypes.guess_type(self.path.name)[0] or "image/png"
        data = self.read()
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        if data[:3] == b"GIF":
            return "image/gif"
        return "image/png"

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{base64.b64encode(self.read()).decode()}"


Content = Union[str, Image]
"""Anything an index embeds: text, or an image."""


def modality(content: Content) -> str:
    if isinstance(content, str):
        return "text"
    if isinstance(content, Image):
        return "image"
    raise TypeError(f"Expected text or an Image, not {type(content).__name__}")


def content_key(content: Content) -> str:
    """The key an item gets when none is given: a hash of its content, so
    adding the same content again finds the same item."""
    raw = b"text\0" + content.encode() if isinstance(content, str) else b"image\0" + content.read()
    return "/items/" + hashlib.blake2b(raw, digest_size=8).hexdigest()


def same(a: Content, b: Content) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        return a.read() == b.read()
    except OSError:
        return False
