"""Grid: content embedded by the index, only vectors sent to DNA."""

import pytest

from fake_dna import KEY, URL, HashEmbedder
from hashgrid.dna import DNA, AsyncDNA, AsyncGrid, Conflict, Grid, Item, ModelMismatch, Result
from hashgrid.index import Image, Index
from hashgrid.index.content import content_key

DOCS = {
    "/cities/lisbon": "Lisbon is a hilly coastal capital",
    "/cities/oslo": "Oslo is a cold northern capital with fjords",
    "/food/pasta": "Pasta with tomato sauce",
}


def _grid(fake, embedder=None, path=":memory:"):
    return Grid(Index(embedder or HashEmbedder(), path=path), DNA(KEY, base_url=URL, transport=fake.transport()))


def test_defaults_from_env(monkeypatch, fake):
    monkeypatch.setenv("HASHGRID_API_KEY", KEY)
    grid = Grid()
    assert grid.index.spec == "fastembed:BAAI/bge-small-en-v1.5"
    assert grid.dna._http.headers["authorization"] == f"Bearer {KEY}"


def test_add_ask_score(fake):
    grid = _grid(fake)
    items = grid.add_many(DOCS)
    assert [(i.key, i.content) for i in items] == list(DOCS.items())
    query, results = grid.ask("hilly coastal city", k=2, prefix="/cities/")
    assert query == Item(query.id, None, "hilly coastal city")
    assert results == [
        Result(items[0].id, "/cities/lisbon", DOCS["/cities/lisbon"], None),
        Result(items[1].id, "/cities/oslo", DOCS["/cities/oslo"], None),
    ]
    assert grid.score(query, results[1], -1).score == -1.0
    assert [(r.key, r.score) for r in grid.results(query)] == [("/cities/lisbon", None), ("/cities/oslo", -1.0)]
    assert grid.search(query, k=5, prefix="/cities/") == []
    assert grid.get(items[2].id) == items[2]
    grid.delete(items[2])
    assert len(fake.nodes) == 3 and grid.index.get(grid._core.name, items[2].id) is None


def test_document_and_query_embedding(fake, embedder):
    grid = _grid(fake, embedder)
    grid.add("an item")
    grid.query("a query")
    assert embedder.calls == [("document", ["an item"]), ("query", ["a query"])]


def test_content_never_sent(fake, tmp_path):
    img = tmp_path / "cat.png"
    img.write_bytes(b"\x89PNG whiskers")
    grid = _grid(fake)
    grid.add_many(DOCS)
    grid.add(Image(img))
    grid.ask("hilly coastal city", k=3)
    wire = b" ".join(r.content for r in fake.requests)
    for word in [b"Lisbon", b"hilly", b"Pasta", b"whiskers", b"cat.png"]:
        assert word not in wire


def test_add_is_idempotent_by_content(fake):
    grid = _grid(fake)
    a = grid.add("same")
    assert a.key == content_key("same")
    assert grid.add("same") == a
    assert grid.add_many(["same", "other"])[0] == a
    grid.add("one", key="/k")
    with pytest.raises(Conflict):
        grid.add("two", key="/k")
    assert len(fake.nodes) == 3


def test_images_found_by_text(fake, tmp_path):
    cat, dog = tmp_path / "cat.png", tmp_path / "dog.png"
    cat.write_bytes(b"\x89PNG a sleepy cat")
    dog.write_bytes(b"\x89PNG a running dog")
    grid = _grid(fake)
    grid.add(Image(cat), key="/img/cat")
    grid.add(Image(dog), key="/img/dog")
    grid.add(Image(data=b"\x89PNG a cat in memory"), key="/img/mem")
    _, results = grid.ask("cat", k=2, prefix="/img/")
    assert {r.key for r in results} == {"/img/cat", "/img/mem"}
    assert {type(r.content) for r in results} == {Image}
    _, (best,) = grid.ask(Image(data=b"dog"), k=1)  # and by image
    assert best.content == Image(dog)


def test_item_added_elsewhere_has_no_content(fake):
    other = Grid(Index(HashEmbedder(), path=":memory:"), DNA(KEY, base_url=URL, transport=fake.transport()))
    other.add("x", key="/x")
    _, (r,) = _grid(fake).ask("x")
    assert (r.key, r.content) == ("/x", None)


def test_model_mismatch(fake, tmp_path):
    _grid(fake, HashEmbedder(64)).add("one", "/1")
    with pytest.raises(ModelMismatch, match="32 values.*64"):
        _grid(fake, HashEmbedder(32)).add("two", "/2")
    assert len(fake.nodes) == 1  # refused before sending
    path = tmp_path / "i.sqlite"
    _grid(fake, HashEmbedder(64), path).add("three")
    with pytest.raises(ModelMismatch, match="built with test:hash@64"):
        _grid(fake, HashEmbedder(64, spec="test:other"), path).add("four")


@pytest.mark.anyio
async def test_async(fake, tmp_path):
    grid = AsyncGrid(Index(HashEmbedder(), path=":memory:"), AsyncDNA(KEY, base_url=URL, transport=fake.transport()))
    items = await grid.add_many(DOCS)
    assert await grid.add(DOCS["/food/pasta"], key="/food/pasta") == items[2]
    query, results = await grid.ask("pasta tomato", k=1)
    assert results == [Result(items[2].id, "/food/pasta", DOCS["/food/pasta"], None)]
    assert (await grid.score(query, results[0], 1)).score == 1.0
    assert [r.score for r in await grid.results(query)] == [1.0]
    assert (await grid.get(items[0])).content == DOCS["/cities/lisbon"]
    await grid.delete(items[0])
    assert items[0].id not in fake.nodes
