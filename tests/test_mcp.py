"""The local MCP server: text-level tools over a mocked DNA, in-process and
over stdio as a client launches it."""

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from fake_dna import KEY, URL, FakeDNA, HashEmbedder
from hashgrid.dna import AsyncDNA
from hashgrid.mcp_server import Local, build_server, item_key
from hashgrid.index import Index

TOOLS = {"add_item", "search", "more_results", "score", "delete_item"}


async def call(client: Client, name: str, **arguments: Any) -> Any:
    result = await client.call_tool(name, arguments)
    if result.is_error:
        raise AssertionError(result.content[0].text)
    out = result.structured_content
    return out["result"] if isinstance(out, dict) and set(out) == {"result"} else out


async def error(client: Client, name: str, **arguments: Any) -> str:
    result = await client.call_tool(name, arguments)
    assert result.is_error, result
    return result.content[0].text


def _server(fake: FakeDNA, index: Index, embedder=None, key: str = KEY):
    local = Local(
        lambda: AsyncDNA(key, base_url=URL, transport=fake.transport()),
        lambda: embedder or HashEmbedder(),
        lambda: index,
    )
    return build_server(local)


@pytest.fixture
def index(tmp_path):
    return Index(tmp_path / "index.sqlite")


@pytest.mark.anyio
async def test_tools(fake, index):
    async with Client(_server(fake, index)) as client:
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS


@pytest.mark.anyio
async def test_add_search_score(fake, index):
    async with Client(_server(fake, index)) as client:
        lisbon = await call(client, "add_item", text="Lisbon is a hilly coastal capital", key="/cities/lisbon")
        oslo = await call(client, "add_item", text="Oslo has fjords and snow", key="/cities/oslo")
        pasta = await call(client, "add_item", text="Pasta with tomato sauce")
        assert pasta["key"] == item_key("Pasta with tomato sauce")

        found = await call(client, "search", query="a hilly coastal city", k=1, prefix="/cities/")
        assert found["results"] == [
            {"id": lisbon["id"], "key": "/cities/lisbon", "text": "Lisbon is a hilly coastal capital"}
        ]
        qid = found["query_id"]
        more = await call(client, "more_results", query_id=qid, k=5, prefix="/cities/")
        assert [r["id"] for r in more] == [oslo["id"]]
        assert await call(client, "more_results", query_id=qid, prefix="/cities/") == []

        scored = await call(client, "score", query_id=qid, item_id=oslo["id"], score=-0.5)
        assert scored == {"query_id": qid, "item_id": oslo["id"], "score": -0.5}
        assert fake.edges[qid, oslo["id"]] == -0.5

    # Text stays local: DNA got vectors, the sqlite file got the text.
    wire = " ".join(r.content.decode() for r in fake.requests)
    assert "Lisbon" not in wire and "hilly" not in wire
    assert index.text(f"{URL}/ test", lisbon["id"]) == "Lisbon is a hilly coastal capital"


@pytest.mark.anyio
async def test_add_same_text_twice(fake, index):
    async with Client(_server(fake, index)) as client:
        a = await call(client, "add_item", text="same")
        assert await call(client, "add_item", text="same") == a
        await call(client, "add_item", text="one", key="/k")
        assert "taken" in await error(client, "add_item", text="two", key="/k")
        assert len(fake.nodes) == 2


@pytest.mark.anyio
async def test_item_added_elsewhere_has_no_text(fake, index):
    fake.nodes["00000000000000ff"] = {"id": "00000000000000ff", "key": "/x", "emb": HashEmbedder()._one("x")}
    async with Client(_server(fake, index)) as client:
        found = await call(client, "search", query="x")
        assert found["results"] == [{"id": "00000000000000ff", "key": "/x", "text": None}]


@pytest.mark.anyio
async def test_delete(fake, index):
    async with Client(_server(fake, index)) as client:
        a = await call(client, "add_item", text="gone soon")
        assert await call(client, "delete_item", item_id=a["id"]) == "Deleted"
        assert fake.nodes == {}
        assert "Not found" in await error(client, "delete_item", item_id=a["id"])


@pytest.mark.anyio
async def test_errors(fake, index):
    async with Client(_server(fake, index, key="hg-wrong")) as client:
        assert "office@hashgrid.ai" in await error(client, "add_item", text="x")
    async with Client(_server(fake, index)) as client:
        assert "`k` must be 1 to 50" in await error(client, "search", query="x", k=0)


@pytest.mark.anyio
async def test_no_key(fake, index, monkeypatch):
    local = Local(lambda: AsyncDNA(base_url=URL, transport=fake.transport()), HashEmbedder, lambda: index)
    async with Client(build_server(local)) as client:
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS  # starts without one
        assert "HASHGRID_API_KEY" in await error(client, "add_item", text="x")


@pytest.mark.anyio
async def test_one_embedder_per_grid(fake, index):
    async with Client(_server(fake, index)) as client:
        await call(client, "add_item", text="x")
    other = HashEmbedder(spec="test:other@64")
    async with Client(_server(fake, index, other)) as client:
        msg = await error(client, "search", query="x")
        assert "built with test:hash@64" in msg and "HASHGRID_EMBEDDER=test:hash@64" in msg


@pytest.mark.anyio
async def test_stdio_end_to_end(tmp_path):
    """The server as a client launches it: a subprocess over stdio."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "stdio_server.py")],
        env={
            "HASHGRID_API_KEY": KEY,
            "HASHGRID_INDEX": str(tmp_path / "e2e-index.sqlite"),
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
    )
    async with Client(params) as client:
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS
        item = await call(client, "add_item", text="the quick brown fox", key="/animals/fox")
        await call(client, "add_item", text="a slow green turtle", key="/animals/turtle")
        found = await call(client, "search", query="quick fox", k=1)
        assert found["results"][0] == {"id": item["id"], "key": "/animals/fox", "text": "the quick brown fox"}
        scored = await call(client, "score", query_id=found["query_id"], item_id=item["id"], score=1)
        assert scored["score"] == 1.0
    assert Index(tmp_path / "e2e-index.sqlite").text(f"{URL}/ test", item["id"]) == "the quick brown fox"
