"""An in-memory DNA behind an ``httpx.MockTransport``: the routes and their
statuses as the real API answers them; search ranks by dot product, plus the
scores given."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from typing import Any

import httpx

KEY = "hg-test"
URL = "https://dna.test/api/v1"


class _Grid:
    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], float | None] = {}


class FakeDNA:
    """Grids by API key; ``KEY`` opens the first, ``POST /`` makes more."""

    def __init__(self, name: str = "test"):
        self.grids: dict[str, _Grid] = {KEY: _Grid(name)}
        self.requests: list[httpx.Request] = []
        self._ids = itertools.count(1)
        self._keys = itertools.count(1)
        self._grid = self.grids[KEY]
        self.lag = 0  # searches that miss a node after it is added, as before a tick
        self._unticked: dict[str, int] = {}

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        return self._grid.nodes

    @nodes.setter
    def nodes(self, value: dict[str, dict[str, Any]]) -> None:
        self._grid.nodes = value

    @property
    def edges(self) -> dict[tuple[str, str], float | None]:
        return self._grid.edges

    @edges.setter
    def edges(self, value: dict[tuple[str, str], float | None]) -> None:
        self._grid.edges = value

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _reply(self, status: int, body: Any = None) -> httpx.Response:
        return httpx.Response(status, json=body) if body is not None else httpx.Response(status)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1")
        body = json.loads(request.content) if request.content else {}
        m, q = request.method, request.url.params
        if path == "/" and m == "POST":  # no key needed
            name = body.get("name", "hashgrid")
            if not 1 <= len(name) <= 256:
                return self._reply(422, {"detail": "name must be 1 to 256 characters"})
            key = f"hg-new{next(self._keys)}"
            self.grids[key] = _Grid(name)
            return self._reply(201, {"name": name, "api_key": key})
        auth = request.headers.get("authorization", "")
        grid = self.grids.get(auth.removeprefix("Bearer "))
        if grid is None:
            return self._reply(401, {"detail": "Invalid API key"})
        self._grid = grid
        if path == "/" and m == "GET":
            return self._reply(200, {"name": grid.name})
        if path == "/" and m == "DELETE":
            del self.grids[auth.removeprefix("Bearer ")]
            return self._reply(204)
        if path == "/nodes" and m == "DELETE":
            ids = body["ids"]
            if len(ids) > 1024:
                return self._reply(422, {"detail": "at most 1024 ids"})
            for nid in ids:
                self.nodes.pop(nid, None)
            self.edges = {p: s for p, s in self.edges.items() if not set(p) & set(ids)}
            return self._reply(204)
        if path == "/nodes" and m == "GET":
            ids = sorted(i for i in self.nodes if i > q.get("cursor", ""))
            return self._reply(200, [self.nodes[i] for i in ids[: int(q.get("limit", 32))]])
        if path == "/nodes" and m == "POST":
            emb, key = body.get("emb"), body.get("key")
            if not emb or len(emb) > 2048 or not all(math.isfinite(x) for x in emb):
                return self._reply(422, {"detail": "Node embedding must have 1 to 2048 values"})
            if key and any(n["key"] == key for n in self.nodes.values()):
                return self._reply(409, {"detail": "Resource already exists"})
            nid = f"{next(self._ids):016x}"
            self.nodes[nid] = {"id": nid, "key": key, "emb": emb}
            if self.lag:
                self._unticked[nid] = self.lag
            return self._reply(201, self.nodes[nid])
        if mt := re.fullmatch(r"/nodes/([0-9a-f]{16})", path):
            nid = mt[1]
            if nid not in self.nodes:
                return self._reply(404, {"detail": "Resource not found"})
            if m == "GET":
                return self._reply(200, self.nodes[nid])
            if m == "PATCH":
                if "key" in body:
                    self.nodes[nid]["key"] = body["key"]
                if "emb" in body:
                    self.nodes[nid]["emb"] = body["emb"]
                return self._reply(200, self.nodes[nid])
            if m == "DELETE":
                del self.nodes[nid]
                self.edges = {p: s for p, s in self.edges.items() if nid not in p}
                return self._reply(204)
        if mt := re.fullmatch(r"/nodes/([0-9a-f]{16})/edges", path):
            nid = mt[1]
            if nid not in self.nodes:
                return self._reply(404, {"detail": "Resource not found"})
            if m == "GET":
                peers = sorted(p for (a, p) in self.edges if a == nid and p > q.get("cursor", ""))
                page = peers[: int(q.get("limit", 32))]
                return self._reply(200, [{"id": p, "score": self.edges[nid, p]} for p in page])
            if m == "POST":
                return self._search(nid, body.get("prefix", ""))
        if mt := re.fullmatch(r"/nodes/([0-9a-f]{16})/edges/([0-9a-f]{16})", path):
            nid, eid = mt[1], mt[2]
            if m == "GET":
                if (nid, eid) not in self.edges:
                    return self._reply(404, {"detail": "Resource not found"})
                return self._reply(200, {"id": eid, "score": self.edges[nid, eid]})
            if nid not in self.nodes or eid not in self.nodes:
                return self._reply(404, {"detail": "Resource not found"})
            if m in ("PUT", "PATCH"):
                if m == "PATCH" and (nid, eid) not in self.edges:
                    return self._reply(404, {"detail": "Resource not found"})
                created = (nid, eid) not in self.edges
                score = round(max(-1.0, min(1.0, body["score"])), 4)
                self.edges[nid, eid] = score
                return self._reply(201 if created and m == "PUT" else 200, {"id": eid, "score": score})
            if m == "DELETE":
                if self.edges.pop((nid, eid), "gone") == "gone":
                    return self._reply(404, {"detail": "Resource not found"})
                return self._reply(204)
        return self._reply(404, {"detail": "Not Found"})

    def _search(self, nid: str, prefix: str) -> httpx.Response:
        src = self.nodes[nid]["emb"]
        best = None
        for pid, n in self.nodes.items():
            if pid == nid or not n["key"] or not n["key"].startswith(prefix):
                continue
            if (nid, pid) in self.edges or len(n["emb"]) != len(src):
                continue
            if self._unticked.get(pid):
                self._unticked[pid] -= 1
                continue
            s = sum(a * b for a, b in zip(src, n["emb"]))
            if best is None or s > best[0]:
                best = (s, pid)
        if best is None:
            return self._reply(400, {"detail": "Failed to create edge. Try again later."})
        self.edges[nid, best[1]] = None
        return self._reply(201, {"id": best[1], "score": None})


class HashEmbedder:
    """A deterministic bag-of-words embedder, unit length. An image is
    embedded by the words in its bytes, so a test image of b"a cat" is found
    by the text "cat"."""

    def __init__(self, dim: int = 64, spec: str | None = None, images: bool = True):
        self.dim = dim
        self.spec = spec or f"test:hash@{dim}"
        self.modalities = frozenset({"text", "image"} if images else {"text"})
        self.calls: list[tuple[str, list]] = []

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, contents, kind):
        self.calls.append((kind, list(contents)))
        return [self._one(c if isinstance(c, str) else c.read().decode("latin-1")) for c in contents]
