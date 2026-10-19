"""Embedders: specs, and the requests to OpenAI and Voyage."""

import json

import httpx
import pytest

from hashgrid import embed


def _api(captured):
    def handle(request):
        body = json.loads(request.content)
        captured.append((str(request.url), request.headers["authorization"], body))
        data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data[::-1]})  # any order

    return httpx.MockTransport(handle)


def test_default_is_local(monkeypatch):
    e = embed.from_spec()
    assert isinstance(e, embed.FastEmbed) and e.spec == embed.DEFAULT
    monkeypatch.setenv("HASHGRID_EMBEDDER", "fastembed:BAAI/bge-base-en-v1.5")
    assert embed.from_spec().spec == "fastembed:BAAI/bge-base-en-v1.5"


def test_unknown():
    with pytest.raises(ValueError, match="Unknown embedder"):
        embed.from_spec("cohere:x")


def test_keys_from_env(monkeypatch):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        embed.from_spec("openai")
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        embed.from_spec("voyage")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-x")
    assert embed.from_spec("openai").spec == "openai:text-embedding-3-small"
    assert embed.from_spec("openai:text-embedding-3-large").spec == "openai:text-embedding-3-large@2048"
    assert embed.from_spec("openai:text-embedding-3-small@512").spec == "openai:text-embedding-3-small@512"
    assert embed.from_spec("voyage").spec == "voyage:voyage-3.5"


def test_openai_request():
    e = embed.OpenAI("text-embedding-3-large", api_key="sk-x")
    captured = []
    e._http = httpx.Client(headers=e._http.headers, transport=_api(captured))
    assert e.embed_documents(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    url, auth, body = captured[0]
    assert url == "https://api.openai.com/v1/embeddings" and auth == "Bearer sk-x"
    assert body == {"model": "text-embedding-3-large", "input": ["a", "b"], "dimensions": 2048}


def test_voyage_request_and_batches():
    e = embed.Voyage(api_key="pa-x")
    e.batch = 2
    captured = []
    e._http = httpx.Client(headers=e._http.headers, transport=_api(captured))
    assert len(e.embed_queries(["a", "b", "c"])) == 3
    assert [c[2] for c in captured] == [
        {"model": "voyage-3.5", "input": ["a", "b"], "input_type": "query"},
        {"model": "voyage-3.5", "input": ["c"], "input_type": "query"},
    ]
    e.embed_documents(["d"])
    assert captured[-1][2]["input_type"] == "document"


def test_fastembed_missing_says_how_to_install(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_fastembed(name, *a, **k):
        if name == "fastembed":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fastembed)
    with pytest.raises(ImportError, match=r"hashgrid\[local\]"):
        embed.FastEmbed().embed_queries(["x"])
