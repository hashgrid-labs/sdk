"""The local MCP server: content-level tools over a mocked DNA, in-process
and over stdio as a client launches it."""

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters

from fake_dna import KEY, URL, FakeDNA, HashEmbedder
from hashgrid.index import Index
from hashgrid.index.content import content_key
from hashgrid.mcp_server import Local, build_server

TOOLS = {"add_item", "search", "more_results", "score", "delete_item", "create_grid", "delete_grid"}


async def call(client: Client, tool: str, **arguments: Any) -> Any:
    result = await client.call_tool(tool, arguments)
    if result.is_error:
        raise AssertionError(result.content[0].text)
    out = result.structured_content
    return out["result"] if isinstance(out, dict) and set(out) == {"result"} else out


async def error(client: Client, tool: str, **arguments: Any) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error, result
    return result.content[0].text


@pytest.fixture
def index_path(tmp_path):
    return tmp_path / "index.sqlite"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("HASHGRID_API_KEY", KEY)


def _server(fake: FakeDNA, index_path, embedder=None):
    local = Local(
        lambda: Index(embedder or HashEmbedder(), path=index_path),
        base_url=URL,
        transport=fake.transport(),
    )
    return build_server(local)


@pytest.mark.anyio
async def test_tools(fake, index_path):
    async with Client(_server(fake, index_path)) as client:
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS


@pytest.mark.anyio
async def test_add_search_score(fake, index_path):
    async with Client(_server(fake, index_path)) as client:
        lisbon = await call(client, "add_item", text="Lisbon is a hilly coastal capital", key="/cities/lisbon")
        oslo = await call(client, "add_item", text="Oslo has fjords and snow", key="/cities/oslo")
        pasta = await call(client, "add_item", text="Pasta with tomato sauce")
        assert pasta["key"] == content_key("Pasta with tomato sauce")

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

    # Content stays local: DNA got vectors, the index got the text.
    wire = " ".join(r.content.decode() for r in fake.requests)
    assert "Lisbon" not in wire and "hilly" not in wire
    assert Index(HashEmbedder(), path=index_path).get(f"{URL}/ test", lisbon["id"])[1].startswith("Lisbon")


@pytest.mark.anyio
async def test_images(fake, index_path, tmp_path):
    cat = tmp_path / "cat.png"
    cat.write_bytes(b"\x89PNG a sleepy cat")
    async with Client(_server(fake, index_path)) as client:
        item = await call(client, "add_item", image=str(cat), key="/img/cat")
        await call(client, "add_item", text="a dog", key="/txt/dog")
        found = await call(client, "search", query="cat", k=1)
        assert found["results"] == [{"id": item["id"], "key": "/img/cat", "image": str(cat.resolve())}]
        by_image = await call(client, "search", image=str(cat), k=1)
        assert by_image["results"][0]["id"] == item["id"]
        assert "No image file" in await error(client, "add_item", image=str(tmp_path / "none.png"))
        assert "one of them" in await error(client, "add_item")
        assert "one of them" in await error(client, "search", query="x", image=str(cat))
    async with Client(_server(fake, tmp_path / "other.sqlite", HashEmbedder(images=False))) as client:
        assert "for images use" in await error(client, "add_item", image=str(cat))


@pytest.mark.anyio
async def test_add_same_content_twice(fake, index_path):
    async with Client(_server(fake, index_path)) as client:
        a = await call(client, "add_item", text="same")
        assert await call(client, "add_item", text="same") == a
        await call(client, "add_item", text="one", key="/k")
        assert "taken" in await error(client, "add_item", text="two", key="/k")
        assert len(fake.nodes) == 2


@pytest.mark.anyio
async def test_item_added_elsewhere_has_no_text(fake, index_path):
    fake.nodes["00000000000000ff"] = {"id": "00000000000000ff", "key": "/x", "emb": HashEmbedder()._one("x")}
    async with Client(_server(fake, index_path)) as client:
        found = await call(client, "search", query="x")
        assert found["results"] == [{"id": "00000000000000ff", "key": "/x", "text": None}]


@pytest.mark.anyio
async def test_delete(fake, index_path):
    async with Client(_server(fake, index_path)) as client:
        a = await call(client, "add_item", text="gone soon")
        assert await call(client, "delete_item", item_id=a["id"]) == "Deleted"
        assert fake.nodes == {}
        assert "Not found" in await error(client, "delete_item", item_id=a["id"])


@pytest.mark.anyio
async def test_errors(fake, index_path, monkeypatch):
    monkeypatch.setenv("HASHGRID_API_KEY", "hg-wrong")
    async with Client(_server(fake, index_path)) as client:
        assert "create_grid()" in await error(client, "add_item", text="x")
    monkeypatch.setenv("HASHGRID_API_KEY", KEY)
    async with Client(_server(fake, index_path)) as client:
        assert "`k` must be 1 to 50" in await error(client, "search", query="x", k=0)


@pytest.mark.anyio
async def test_no_key_then_create_grid(fake, index_path, monkeypatch):
    monkeypatch.delenv("HASHGRID_API_KEY")
    async with Client(_server(fake, index_path)) as client:
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS  # starts without one
        msg = await error(client, "add_item", text="x")
        assert "create_grid" in msg and "HASHGRID_API_KEY" in msg
        made = await call(client, "create_grid", name="mine")
        assert made == {"name": "mine", "saved_in": str(index_path)}
        assert "api_key" not in made  # kept in the index, not shown to the agent
        assert "has a grid already" in await error(client, "create_grid")
        item = await call(client, "add_item", text="hello")
    assert fake.grids["hg-new1"].nodes.keys() == {item["id"]}
    assert oct(index_path.stat().st_mode & 0o777) == "0o600"
    # A new server finds the saved key.
    async with Client(_server(fake, index_path)) as client:
        found = await call(client, "search", query="hello")
        assert found["results"][0]["text"] == "hello"
        assert "must be the grid's name" in await error(client, "delete_grid", confirm="nope")
        assert await call(client, "delete_grid", confirm="mine") == "Deleted grid 'mine'"
        assert "hg-new1" not in fake.grids
        assert "No DNA grid yet" in await error(client, "add_item", text="x")
        assert (await call(client, "create_grid"))["name"] == "hashgrid"  # a fresh one
    assert Index(HashEmbedder(), path=index_path).saved_key(f"{URL}/") == "hg-new2"


@pytest.mark.anyio
async def test_create_grid_refused_with_env_key(fake, index_path):
    async with Client(_server(fake, index_path)) as client:
        assert "HASHGRID_API_KEY is set" in await error(client, "create_grid")


@pytest.mark.anyio
async def test_one_embedder_per_grid(fake, index_path):
    async with Client(_server(fake, index_path)) as client:
        await call(client, "add_item", text="x")
    async with Client(_server(fake, index_path, HashEmbedder(spec="test:other"))) as client:
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
    index = Index(HashEmbedder(), path=tmp_path / "e2e-index.sqlite")
    assert index.get(f"{URL}/ test", item["id"]) == ("/animals/fox", "the quick brown fox")
