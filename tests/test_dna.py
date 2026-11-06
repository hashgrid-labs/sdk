"""DNA, Query and their async twins against the in-memory DNA."""

import re
from pathlib import Path

import pytest

import hashgrid
from fake_dna import KEY, URL
from hashgrid import (
    DNA,
    AsyncDNA,
    AsyncQuery,
    GridInfo,
    Item,
    NotFound,
    PreconditionFailed,
    Query,
    TooLarge,
    Unauthorized,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 8


def _dna(fake) -> DNA:
    return DNA(KEY, base_url=URL, transport=fake.transport())


def test_the_loop(fake):
    dna = _dna(fake)
    dna.put_many({"cities/lisbon": "Lisbon", "cities/oslo": "Oslo", "food/pasta": "Pasta"})
    query = dna.query("a capital by the sea")
    assert isinstance(query, Query) and len(query.id) == 16
    results = query.results(limit=2, prefix="cities/")
    assert [r.key for r in results] == ["cities/lisbon", "cities/oslo"]
    assert query.grade(results[1], 2) == 1.0  # clamped
    assert query.grade("cities/lisbon", -0.5) == -0.5
    assert [r.key for r in query.results(prefix="cities/")] == ["cities/oslo", "cities/lisbon"]
    assert query.results(prefix=["cities/", "food/"], ungraded=True)[0].key == "food/pasta"
    again = dna.query(id=query.id)  # the same question, reopened: no request
    assert again.grades() == {"cities/lisbon": -0.5, "cities/oslo": 1.0}
    again.delete_grade("cities/lisbon")
    assert query.grades() == {"cities/oslo": 1.0}
    query.delete()
    with pytest.raises(NotFound):
        query.results()
    with pytest.raises(TypeError):
        dna.query()
    with pytest.raises(TypeError):
        dna.query("q", id=query.id)


def test_content_types(fake, tmp_path):
    dna = _dna(fake)
    image = tmp_path / "a.png"
    image.write_bytes(PNG)
    dna.put("text", "Lisbon")
    dna.put("record", {"name": "Lisbon", "tags": ["sea"]})
    dna.put("emb", [0.5, 1, 2])
    dna.put("bytes", PNG)
    dna.put("path", Path(image))
    dna.put("json-as-text", '{"a": 1}', type="text/plain")
    types = {r.headers.get("content-type") for r in fake.requests}
    assert {"text/plain; charset=utf-8", "application/json", "application/vnd.hashgrid.emb+json"} <= types
    assert fake.requests[3].headers.get("content-type") is None  # DNA tells the image from its bytes
    assert dna.get("text") == "Lisbon"
    assert dna.get("record") == {"name": "Lisbon", "tags": ["sea"]}
    assert dna.get("emb") == [0.5, 1.0, 2.0]
    assert dna.get("bytes") == dna.get("path") == PNG
    assert dna.get("json-as-text") == '{"a": 1}'
    with pytest.raises(TypeError):
        dna.put("bad", 42)  # type: ignore[arg-type]


def test_items_and_keys(fake):
    dna = _dna(fake)
    first = dna.put("tools/a b", "a tool")
    assert isinstance(first, Item) and first.key == "tools/a b"
    assert fake.requests[-1].url.raw_path == b"/api/v1/items/tools/a%20b"
    for i in range(120):
        dna.put(f"tools/{i:03}", f"tool {i}")
    dna.put("agents/1", "an agent")
    keys = [i.key for i in dna.items("tools/")]
    assert len(keys) == 121 and keys == sorted(keys)
    assert len([r for r in fake.requests if r.url.path.endswith("/items")]) == 2  # two pages
    assert dna.get(first) == "a tool"
    dna.delete(first)
    with pytest.raises(NotFound):
        dna.get("tools/a b")


def test_conditional_writes_and_errors(fake):
    dna = _dna(fake)
    item = dna.put("a", "one", if_none_match="*")
    with pytest.raises(PreconditionFailed):
        dna.put("a", "two", if_none_match="*")
    with pytest.raises(PreconditionFailed):
        dna.put("a", "two", if_match="stale")
    assert dna.put("a", "two", if_match=item.etag).etag != item.etag
    with pytest.raises(TooLarge) as e:
        dna.put("big", "x" * 9000)
    assert e.value.status == 413
    with pytest.raises(NotFound):
        dna.query("q").grade("nope", 1)


def test_signup_info_delete(fake):
    api_key = DNA.signup("mine", base_url=URL, transport=fake.transport())
    assert api_key.startswith("hg-")
    dna = DNA(api_key, base_url=URL, transport=fake.transport())
    assert dna.info() == GridInfo("test", 1024)
    dna.delete_grid()
    with pytest.raises(Unauthorized, match="DNA.signup"):
        dna.info()


def test_key_from_env(monkeypatch, fake):
    with pytest.raises(ValueError, match="HASHGRID_API_KEY"):
        DNA()
    monkeypatch.setenv("HASHGRID_API_KEY", KEY)
    monkeypatch.setenv("HASHGRID_DNA_URL", URL)
    with DNA(transport=fake.transport()) as dna:
        assert dna.info().name == "test"
    assert fake.requests[-1].headers["user-agent"] == f"hashgrid-python/{hashgrid.__version__}"


@pytest.mark.anyio
async def test_async_loop(fake):
    api_key = await AsyncDNA.signup(base_url=URL, transport=fake.transport())
    async with AsyncDNA(api_key, base_url=URL, transport=fake.transport()) as dna:
        await dna.put_many({"cities/lisbon": "Lisbon", "cities/oslo": "Oslo"})
        query = await dna.query("by the sea")
        assert isinstance(query, AsyncQuery)
        results = await query.results(limit=1)
        assert await query.grade(results[0], 1) == 1.0
        assert await (await dna.query(id=query.id)).grades() == {"cities/lisbon": 1.0}
        assert [i.key async for i in dna.items("cities/")] == ["cities/lisbon", "cities/oslo"]
        assert await dna.get("cities/oslo") == "Oslo"
        await dna.delete("cities/oslo")
        await query.delete()
        assert (await dna.info()).max_items == 1024
        await dna.delete_grid()


def test_version_matches_the_package():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M)[1] == hashgrid.__version__


@pytest.mark.parametrize("sync, other", [(DNA, AsyncDNA), (Query, AsyncQuery)])
def test_sync_and_async_match(sync, other):
    public = lambda cls: {n for n in vars(cls) if not n.startswith("_")} - {"close", "aclose"}
    assert public(sync) == public(other)
