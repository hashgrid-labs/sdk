"""Errors from the DNA API, one class per status that means something."""

from __future__ import annotations

GET_A_KEY = "No key yet? DNA.signup() makes a grid, no key needed, and returns its key."


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
    """403: the grid holds its item limit, or it is frozen."""


class NotFound(DNAError):
    """404: no such item, query or grade."""


class PreconditionFailed(DNAError):
    """412: ``if_match`` or ``if_none_match`` did not hold."""


class TooLarge(DNAError):
    """413: text or a record over 8 KB, or an image over 3 MB or 40 MP."""


class UnsupportedType(DNAError):
    """415: a content type DNA does not take."""


class Invalid(DNAError):
    """422: content that is not of its type, or a malformed request."""


class TooManyRequests(DNAError):
    """429: this address has created its share of grids for now; try later."""


_BY_STATUS: dict[int, type[DNAError]] = {
    401: Unauthorized,
    403: Forbidden,
    404: NotFound,
    412: PreconditionFailed,
    413: TooLarge,
    415: UnsupportedType,
    422: Invalid,
    429: TooManyRequests,
}


def error_for(status: int, detail: object) -> DNAError:
    return _BY_STATUS.get(status, DNAError)(status, detail)
