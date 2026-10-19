"""The HTTP client: each route's request and reply, sync and async."""

import json

import httpx
import pytest

from fake_dna import KEY, URL
from hashgrid.dna import (
    DNA,
    AsyncDNA,
    GridKey,
    acreate_grid,
    create_grid,
    Conflict,
    Edge,
    GridInfo,
    Invalid,
    Node,
    NotFound,
    Unauthorized,
)


@pytest.fixture
def dna(fake):
    with DNA(KEY, base_url=URL, transport=fake.transport()) as c:
        yield c


def test_key_from_env(monkeypatch, fake):
    with pytest.raises(ValueError, match="create_grid"):
        DNA()
    monkeypatch.setenv("HASHGRID_API_KEY", KEY)
    monkeypatch.setenv("HASHGRID_DNA_URL", URL)
    assert DNA(transport=fake.transport()).get_grid() == GridInfo("test")


def test_wrong_key(fake):
    with pytest.raises(Unauthorized, match="create_grid"):
        DNA("hg-wrong", base_url=URL, transport=fake.transport()).get_grid()


def test_requests_on_the_wire(dna, fake):
    item = dna.create_node([1.0, 0.0], key="/a")
    query = dna.create_node([1.0, 0.0])
    edge = dna.search(query, prefix="/")
    dna.score(query, edge, 0.5)
    sent = [(r.method, r.url.path, json.loads(r.content) if r.content else None) for r in fake.requests]
    assert sent == [
        ("POST", "/api/v1/nodes", {"emb": [1.0, 0.0], "key": "/a"}),
        ("POST", "/api/v1/nodes", {"emb": [1.0, 0.0]}),
        ("POST", f"/api/v1/nodes/{query.id}/edges", {"prefix": "/"}),
        ("PUT", f"/api/v1/nodes/{query.id}/edges/{item.id}", {"score": 0.5}),
    ]
    assert fake.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert fake.requests[0].headers["user-agent"].startswith("hashgrid-python/")


def test_nodes(dna):
    a = dna.create_node([0.1, 0.2], key="/a")
    assert a == Node(a.id, "/a", [0.1, 0.2])
    assert dna.get_node(a.id) == a
    with pytest.raises(Conflict):
        dna.create_node([0.1], key="/a")
    with pytest.raises(Invalid):
        dna.create_node([0.0] * 2049)
    assert dna.update_node(a, emb=[0.3, 0.4]).key == "/a"  # key kept
    assert dna.update_node(a, key=None).key is None  # key removed
    dna.delete_node(a)
    with pytest.raises(NotFound):
        dna.get_node(a)


def test_update_node_sends_only_given_fields(dna, fake):
    a = dna.create_node([0.1], key="/a")
    dna.update_node(a, key=None)
    assert json.loads(fake.requests[-1].content) == {"key": None}
    dna.update_node(a, emb=[0.2])
    assert json.loads(fake.requests[-1].content) == {"emb": [0.2]}


def test_pages(dna):
    ids = [dna.create_node([1.0], key=f"/{i}").id for i in range(5)]
    assert [n.id for n in dna.list_nodes(limit=2)] == ids[:2]
    assert [n.id for n in dna.list_nodes(cursor=ids[1], limit=2)] == ids[2:4]
    assert [n.id for n in dna.iter_nodes(page=2)] == ids
    with pytest.raises(ValueError):
        dna.list_nodes(limit=0)


def test_search_score_and_results(dna):
    near = dna.create_node([1.0, 0.0], key="/items/near")
    far = dna.create_node([0.0, 1.0], key="/items/far")
    dna.create_node([1.0, 0.0], key="/other/x")
    q = dna.create_node([0.9, 0.1])
    assert dna.create_edge(q, "/items/") == Edge(near.id, None)
    assert dna.create_edge(q, "/items/") == Edge(far.id, None)
    assert dna.create_edge(q, "/items/") is None  # nothing left: 400
    assert dna.update_edge(q, near, 1.0) == Edge(near.id, 1.0)
    assert dna.put_edge(q, far, -3) == Edge(far.id, -1.0)  # clamped by DNA
    assert dna.get_edge(q, near).score == 1.0
    assert {e.id: e.score for e in dna.iter_edges(q)} == {near.id: 1.0, far.id: -1.0}
    dna.delete_edge(q, far)
    assert [e.id for e in dna.list_edges(q)] == [near.id]
    with pytest.raises(ValueError):
        dna.put_edge(q, near, float("nan"))


def test_update_edge_needs_an_edge(dna):
    a, b = dna.create_node([1.0], key="/a"), dna.create_node([1.0])
    with pytest.raises(NotFound):
        dna.update_edge(b, a, 1.0)
    assert dna.put_edge(b, a, 1.0).score == 1.0


def test_other_errors_keep_status(fake):
    def boom(request):
        return httpx.Response(500, text="oops")

    from hashgrid.dna import DNAError

    with pytest.raises(DNAError) as ex:
        DNA(KEY, base_url=URL, transport=httpx.MockTransport(boom)).get_grid()
    assert ex.value.status == 500 and ex.value.detail == "oops"


@pytest.mark.anyio
async def test_async(fake):
    async with AsyncDNA(KEY, base_url=URL, transport=fake.transport()) as dna:
        assert await dna.get_grid() == GridInfo("test")
        item = await dna.create_node([1.0, 0.0], key="/a")
        q = await dna.create_node([1.0, 0.0])
        assert (await dna.search(q)).id == item.id
        assert await dna.search(q) is None
        assert (await dna.score(q, item, 1)).score == 1.0
        assert [e.id async for e in dna.iter_edges(q)] == [item.id]
        assert [n.id async for n in dna.iter_nodes()] == [item.id, q.id]
        await dna.delete_node(item)
        with pytest.raises(NotFound):
            await dna.get_node(item)


def test_create_and_delete_grid(fake):
    key = create_grid("mine", base_url=URL, transport=fake.transport())
    assert key == GridKey("mine", "hg-new1") and "hg-new1" not in repr(key)
    assert "authorization" not in fake.requests[-1].headers  # no key needed
    dna = DNA(key.api_key, base_url=URL, transport=fake.transport())
    assert dna.get_grid() == GridInfo("mine")
    dna.create_node([1.0], key="/a")
    dna.delete_grid()
    assert "hg-new1" not in fake.grids
    with pytest.raises(Unauthorized):
        dna.get_grid()
    assert create_grid(base_url=URL, transport=fake.transport()).name == "hashgrid"
    with pytest.raises(Invalid):
        create_grid("", base_url=URL, transport=fake.transport())


@pytest.mark.anyio
async def test_async_create_and_delete_grid(fake):
    key = await acreate_grid("mine", base_url=URL, transport=fake.transport())
    async with AsyncDNA(key.api_key, base_url=URL, transport=fake.transport()) as dna:
        assert (await dna.get_grid()).name == "mine"
        await dna.delete_grid()
    assert key.api_key not in fake.grids


def test_delete_nodes(dna, fake):
    a, b, c = (dna.create_node([1.0], key=f"/{i}") for i in "abc")
    dna.put_edge(a, b, 1.0)
    dna.delete_nodes([a, b.id, "00000000000000ff"])
    assert json.loads(fake.requests[-1].content) == {"ids": [a.id, b.id, "00000000000000ff"]}
    assert fake.requests[-1].method == "DELETE" and fake.requests[-1].url.path == "/api/v1/nodes"
    assert [n.id for n in dna.list_nodes()] == [c.id] and fake.edges == {}
    with pytest.raises(ValueError, match="1024"):
        dna.delete_nodes(["00000000000000ff"] * 1025)
