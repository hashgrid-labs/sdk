"""Errors from the DNA API, one class per status that means something."""

from __future__ import annotations

GET_A_KEY = "No key yet? Create a grid, no key needed: hashgrid.dna.create_grid() returns its key."


class DNAError(Exception):
    """A request DNA refused: ``status`` is the HTTP status, ``detail`` its reason."""

    def __init__(self, status: int, detail: object):
        self.status, self.detail = status, detail
        super().__init__(f"{status}: {detail}")


class Unauthorized(DNAError):
    """401: no API key, or a wrong one."""

    def __str__(self) -> str:
        return f"{super().__str__()}. {GET_A_KEY}"


class Forbidden(DNAError):
    """403."""


class NotFound(DNAError):
    """404: no such node or edge."""


class Conflict(DNAError):
    """409: the key is taken by another node."""


class Invalid(DNAError):
    """422: a malformed request, e.g. an embedding longer than 2048 values."""


class ModelMismatch(Exception):
    """An embedder that does not fit the grid: its vectors have another
    length than the grid's, or another model is recorded for it."""


_BY_STATUS: dict[int, type[DNAError]] = {
    401: Unauthorized,
    403: Forbidden,
    404: NotFound,
    409: Conflict,
    422: Invalid,
}


def error_for(status: int, detail: object) -> DNAError:
    return _BY_STATUS.get(status, DNAError)(status, detail)
