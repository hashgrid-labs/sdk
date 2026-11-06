"""An in-memory DNA behind an ``httpx.MockTransport``: items, queries and
grades, with the statuses the real API answers. Results rank by grade, then
by key; that is all the client needs to see."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any
from urllib.parse import unquote

import httpx

KEY = "hg-test"
URL = "https://dna.test/api/v1"


def _type(request: httpx.Request) -> str:
    mime = request.headers.get("content-type", "").partition(";")[0]
    if mime:
        return mime
    body = request.content
    return "image/png" if body.startswith(b"\x89PNG") else "text/plain"


class FakeDNA:
    def __init__(self) -> None:
        self.keys = {KEY}
        self.items: dict[str, dict[str, Any]] = {}
        self.queries: dict[str, dict[str, Any]] = {}
        self.grades: dict[tuple[str, str], float] = {}
        self.requests: list[httpx.Request] = []
        self._ids = itertools.count(1)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)  # serves sync and async clients alike

    def _content(self, request: httpx.Request) -> dict[str, Any]:
        body = request.content
        return {"type": _type(request), "size": len(body), "etag": hashlib.sha256(body).hexdigest()[:16],
                "ctime": 1, "mtime": 1, "body": body}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = unquote(request.url.path.removeprefix("/api/v1"))
        method = request.method
        if method == "POST" and path == "/":
            key = f"hg-{next(self._ids)}"
            self.keys.add(key)
            return httpx.Response(201, json={"name": json.loads(request.content)["name"], "api_key": key,
                                             "max_items": 1024, "embedder": "nomic-v1.5"})
        if request.headers.get("authorization") not in {f"Bearer {k}" for k in self.keys}:
            return httpx.Response(401, json={"detail": "Invalid API key"})
        if path == "/":
            if method == "GET":
                return httpx.Response(200, json={"name": "test", "max_items": 1024, "embedder": "nomic-v1.5"})
            self.keys.discard(request.headers["authorization"].removeprefix("Bearer "))
            return httpx.Response(204)
        if path == "/items":
            prefix, after = request.url.params.get("prefix", ""), request.url.params.get("after")
            limit = int(request.url.params.get("limit", 32))
            keys = [k for k in sorted(self.items) if k.startswith(prefix) and (after is None or k > after)]
            page = keys[:limit]
            return httpx.Response(200, json={"items": [self._meta(k) for k in page],
                                             "next": page[-1] if len(keys) > limit else None})
        if path.startswith("/items/"):
            return self._item(request, path.removeprefix("/items/"))
        if method == "POST" and path == "/queries":
            qid = f"{next(self._ids):016x}"
            self.queries[qid] = self._content(request)
            q = self.queries[qid]
            return httpx.Response(201, json={"id": qid, "type": q["type"], "size": q["size"], "etag": q["etag"],
                                             "ctime": 1})
        parts = path.removeprefix("/queries/").split("/", 2)
        qid = parts[0]
        if qid not in self.queries:
            return httpx.Response(404, json={"detail": "Not found"})
        if len(parts) == 1:
            self.queries.pop(qid)
            return httpx.Response(204)
        if parts[1] == "results":
            prefixes = request.url.params.get_list("prefix") or [""]
            ungraded = request.url.params.get("ungraded") == "true"
            keys = [k for k in self.items if any(k.startswith(p) for p in prefixes)
                    and not (ungraded and (qid, k) in self.grades)]
            keys.sort(key=lambda k: (-self.grades.get((qid, k), 0.0), k))
            limit = int(request.url.params.get("limit", 32))
            return httpx.Response(200, json={"results": [self._meta(k) | {"score": self.grades.get((qid, k), 0.0)}
                                                         for k in keys[:limit]]})
        if len(parts) == 2:
            return httpx.Response(200, json={"grades": [{"key": k, "score": s} for (q, k), s
                                                        in sorted(self.grades.items()) if q == qid]})
        key = parts[2]
        if key not in self.items:
            return httpx.Response(404, json={"detail": f"No item at {key}"})
        if method == "PUT":
            score = max(-1.0, min(1.0, json.loads(request.content)["score"]))
            created = (qid, key) not in self.grades
            self.grades[qid, key] = score
            return httpx.Response(201 if created else 200, json={"key": key, "score": score})
        if (qid, key) not in self.grades:
            return httpx.Response(404, json={"detail": "Not found"})
        self.grades.pop((qid, key))
        return httpx.Response(204)

    def _meta(self, key: str) -> dict[str, Any]:
        return {"key": key} | {k: v for k, v in self.items[key].items() if k != "body"}

    def _item(self, request: httpx.Request, key: str) -> httpx.Response:
        found = self.items.get(key)
        if request.method == "PUT":
            etag = found and found["etag"]
            if ("if-none-match" in request.headers and found) or (
                "if-match" in request.headers and request.headers["if-match"].strip('"') != etag
            ):
                return httpx.Response(412, json={"detail": "Precondition failed"})
            if len(request.content) > 8192 and _type(request) == "text/plain":
                return httpx.Response(413, json={"detail": "Text is at most 8192 bytes"})
            self.items[key] = self._content(request)
            return httpx.Response(200 if found else 201, json=self._meta(key))
        if found is None:
            return httpx.Response(404, json={"detail": "Not found"})
        if request.method == "GET":
            mime = found["type"] + ("; charset=utf-8" if found["type"] == "text/plain" else "")
            return httpx.Response(200, content=found["body"], headers={"content-type": mime})
        self.items.pop(key)
        for pair in [p for p in self.grades if p[1] == key]:
            self.grades.pop(pair)
        return httpx.Response(204)
