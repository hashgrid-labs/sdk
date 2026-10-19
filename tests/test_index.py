"""The index: content, embedders, and the local store."""

import json

import httpx
import pytest

from fake_dna import HashEmbedder
from hashgrid.dna import ModelMismatch
from hashgrid.index import DEFAULT, FastEmbed, Image, Index, OpenAI, Voyage, embedder
from hashgrid.index.content import content_key

PNG = b"\x89PNG\r\n\x1a\n a cat"


def _api(captured):
    def handle(request):
        body = json.loads(request.content)
        captured.append((str(request.url), request.headers["authorization"], body))
        n = len(body.get("input") or body.get("inputs"))
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(n)]
        return httpx.Response(200, json={"data": data[::-1]})  # any order

    return httpx.MockTransport(handle)


def _mock(e, captured):
    e._http = httpx.Client(headers=e._http.headers, transport=_api(captured))
    return e


def test_default_is_local_text(monkeypatch):
    idx = Index(path=":memory:")
    assert isinstance(idx.embedder, FastEmbed) and idx.spec == DEFAULT
    assert idx.embedder.modalities == {"text"}
    monkeypatch.setenv("HASHGRID_EMBEDDER", "fastembed:Qdrant/clip-ViT-B-32")
    clip = Index(path=":memory:").embedder
    assert clip.modalities == {"text", "image"}
    assert (clip.text_model, clip.image_model) == ("Qdrant/clip-ViT-B-32-text", "Qdrant/clip-ViT-B-32-vision")


def test_specs(monkeypatch):
    with pytest.raises(ValueError, match="Unknown embedder"):
        embedder("cohere:x")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        Index("openai", path=":memory:")
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        embedder("voyage")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-x")
    assert embedder("openai").spec == "openai:text-embedding-3-small"
    assert embedder("openai:text-embedding-3-large").spec == "openai:text-embedding-3-large@2048"
    assert embedder("openai:text-embedding-3-small@512").spec == "openai:text-embedding-3-small@512"
    assert embedder("voyage").modalities == {"text"}
    assert embedder("voyage:voyage-multimodal-3").modalities == {"text", "image"}


def test_openai_request():
    captured = []
    e = _mock(OpenAI("text-embedding-3-large", api_key="sk-x"), captured)
    assert e.embed(["a", "b"], "document") == [[0.0, 1.0], [1.0, 1.0]]
    url, auth, body = captured[0]
    assert url == "https://api.openai.com/v1/embeddings" and auth == "Bearer sk-x"
    assert body == {"model": "text-embedding-3-large", "input": ["a", "b"], "dimensions": 2048}


def test_voyage_text_request_and_batches():
    captured = []
    e = _mock(Voyage(api_key="pa-x"), captured)
    e.batch = 2
    assert len(e.embed(["a", "b", "c"], "query")) == 3
    assert [c[2] for c in captured] == [
        {"model": "voyage-3.5", "input": ["a", "b"], "input_type": "query"},
        {"model": "voyage-3.5", "input": ["c"], "input_type": "query"},
    ]


def test_voyage_multimodal_request():
    captured = []
    e = _mock(Voyage("voyage-multimodal-3", api_key="pa-x"), captured)
    e.embed(["a cat", Image(data=PNG)], "document")
    url, _, body = captured[0]
    assert url == "https://api.voyageai.com/v1/multimodalembeddings"
    assert body["input_type"] == "document"
    assert body["inputs"][0] == {"content": [{"type": "text", "text": "a cat"}]}
    image = body["inputs"][1]["content"][0]
    assert image["type"] == "image_base64" and image["image_base64"].startswith("data:image/png;base64,")


def test_text_only_embedder_refuses_images():
    idx = Index(HashEmbedder(images=False), path=":memory:")
    with pytest.raises(ValueError, match="for images use"):
        idx.embed([Image(data=PNG)], "document")


def test_fastembed_missing_says_how_to_install(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_fastembed(name, *a, **k):
        if name == "fastembed":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fastembed)
    with pytest.raises(ImportError, match=r"hashgrid\[local\]"):
        FastEmbed().embed(["x"], "query")


def test_image():
    with pytest.raises(ValueError):
        Image()
    with pytest.raises(ValueError):
        Image("a.png", data=b"x")
    assert Image(data=b"\xff\xd8\xff").media_type == "image/jpeg"
    assert Image("x.webp").media_type == "image/webp"
    assert content_key("a") != content_key(Image(data=b"a"))
    assert content_key("a") == content_key("a")


def test_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HASHGRID_INDEX", str(tmp_path / "sub" / "i.sqlite"))
    img = tmp_path / "cat.png"
    img.write_bytes(PNG)
    idx = Index(HashEmbedder())
    assert idx.path == tmp_path / "sub" / "i.sqlite"
    idx.put("g", "01", "/a", "text a")
    idx.put("g", "02", "/b", Image(img))
    idx.put("g", "03", None, Image(data=PNG))
    assert idx.get("g", "01") == ("/a", "text a")
    assert idx.get("g", "02") == ("/b", Image(img))
    assert idx.get("g", "03") == (None, Image(data=PNG))
    assert idx.by_key("g", "/b") == ("02", Image(img))
    assert idx.get("other", "01") is None
    idx.delete("g", "01")
    assert idx.get("g", "01") is None
    # It persists.
    assert Index(HashEmbedder()).get("g", "02") == ("/b", Image(img))


def test_bind_one_embedder_per_grid():
    idx = Index(HashEmbedder(), path=":memory:")
    idx.bind("g")
    idx.bind("g")
    other = Index(HashEmbedder(spec="test:other"), path=":memory:")
    other.db = idx.db
    with pytest.raises(ModelMismatch, match="built with test:hash@64"):
        other.bind("g")
    other.bind("h")


def test_saved_keys_and_drop(tmp_path):
    idx = Index(HashEmbedder(), path=tmp_path / "i.sqlite")
    assert oct((tmp_path / "i.sqlite").stat().st_mode & 0o777) == "0o600"
    assert idx.saved_key("https://dna.test/") is None
    idx.save_key("https://dna.test/", "mine", "hg-x")
    assert idx.saved_key("https://dna.test/") == "hg-x"
    idx.bind("https://dna.test/ mine")
    idx.put("https://dna.test/ mine", "01", "/a", "a")
    idx.put("https://dna.test/ other", "01", "/a", "a")
    idx.drop("https://dna.test/ mine")
    assert idx.get("https://dna.test/ mine", "01") is None
    assert idx.get("https://dna.test/ other", "01") == ("/a", "a")
    assert idx.saved_key("https://dna.test/") is None
