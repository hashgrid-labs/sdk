"""Text helpers: text is embedded locally, only vectors are sent."""

import json

import pytest

from fake_dna import KEY, URL, HashEmbedder
from hashgrid.dna import DNA, AsyncDNA, AsyncTextGrid, ModelMismatch, TextGrid

DOCS = {
    "/cities/lisbon": "Lisbon is a hilly coastal capital",
    "/cities/oslo": "Oslo is a cold northern capital with fjords",
    "/food/pasta": "Pasta with tomato sauce",
}


@pytest.fixture
def grid(fake, embedder):
    return TextGrid(DNA(KEY, base_url=URL, transport=fake.transport()), embedder)


def test_add_and_ask(grid, fake, embedder):
    nodes = grid.add_many(DOCS)
    assert [n.key for n in nodes] == list(DOCS)
    query, results = grid.ask("hilly coastal city", k=2, prefix="/cities/")
    assert query.key is None
    assert [r.id for r in results] == [nodes[0].id, nodes[1].id]
    assert embedder.calls == [("document", list(DOCS.values())), ("query", ["hilly coastal city"])]
    grid.score(query, results[1], -1)
    assert {e.id: e.score for e in grid.results(query)} == {nodes[0].id: None, nodes[1].id: -1.0}
    assert grid.search(query, k=5, prefix="/cities/") == []


def test_text_never_sent(grid, fake):
    grid.add_many(DOCS)
    grid.ask("hilly coastal city", k=3)
    wire = " ".join(r.content.decode() for r in fake.requests)
    for text in [*DOCS.values(), "hilly coastal city"]:
        assert text not in wire
    for word in ["Lisbon", "hilly", "Pasta"]:
        assert word not in wire


def test_model_mismatch(fake):
    dna = DNA(KEY, base_url=URL, transport=fake.transport())
    TextGrid(dna, HashEmbedder(64)).add("one", "/1")
    other = TextGrid(dna, HashEmbedder(32))
    with pytest.raises(ModelMismatch, match="32 values.*64"):
        other.add("two", "/2")
    assert len(fake.nodes) == 1  # refused before sending


@pytest.mark.anyio
async def test_async(fake, embedder):
    grid = AsyncTextGrid(AsyncDNA(KEY, base_url=URL, transport=fake.transport()), embedder)
    nodes = await grid.add_many(DOCS)
    query, results = await grid.ask("pasta tomato", k=1)
    assert results[0].id == nodes[2].id
    assert (await grid.score(query, results[0], 1)).score == 1.0
    assert [e.score for e in await grid.results(query)] == [1.0]
