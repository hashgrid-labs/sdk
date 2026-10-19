"""The local half of a grid: ``Index`` embeds content on your machine (or
with your own key) and keeps it, by DNA id; DNA only receives the vectors.
Content is text or an ``Image``."""

from .content import Content, Image
from .embedders import DEFAULT, Embedder, FastEmbed, OpenAI, Voyage, embedder
from .local import Index

__all__ = [
    "Index",
    "Image",
    "Content",
    "Embedder",
    "FastEmbed",
    "OpenAI",
    "Voyage",
    "embedder",
    "DEFAULT",
]
