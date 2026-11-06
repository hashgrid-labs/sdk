"""Hashgrid: the Python client for DNA, a search engine that learns from
your grades. ``DNA`` stores items and asks queries; a ``Query`` gives its
results and takes their grades. ``AsyncDNA`` and ``AsyncQuery``, awaited."""

__version__ = "2.0.0"

from .client import DEFAULT_URL, DNA, AsyncDNA, AsyncQuery, Content, Query  # noqa: E402
from .errors import (  # noqa: E402
    DNAError,
    Forbidden,
    Invalid,
    NotFound,
    PreconditionFailed,
    TooLarge,
    TooManyRequests,
    Unauthorized,
    UnsupportedType,
)
from .models import GridInfo, Item, Result  # noqa: E402

__all__ = [
    "DNA",
    "AsyncDNA",
    "Query",
    "AsyncQuery",
    "Content",
    "GridInfo",
    "Item",
    "Result",
    "DNAError",
    "Unauthorized",
    "Forbidden",
    "NotFound",
    "PreconditionFailed",
    "TooLarge",
    "UnsupportedType",
    "Invalid",
    "TooManyRequests",
    "DEFAULT_URL",
]
