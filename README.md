# hashgrid

The Python client for [Hashgrid DNA](https://dna.hashgrid.ai), a search engine that learns from
feedback. You add items and search them; you score the results; DNA learns from the scores, so the
next searches rank better.

DNA works on embeddings only and never sees your content. This package embeds text and images on
your machine, sends DNA only the vectors, and keeps the content with you, in a local index.

<!-- mcp-name: ai.hashgrid/sdk -->

```bash
pip install 'hashgrid[local]'
```

`[local]` adds the local embedders, which run on your CPU with no account. To embed with OpenAI or
Voyage instead, install plain `hashgrid` and use your own key.

Each grid has its own API key (`hg-…`). Creating a grid needs no key:

```python
from hashgrid.dna import Grid

grid, api_key = Grid.create("mine")
print(api_key)   # shown only now: keep it, e.g. as $HASHGRID_API_KEY
```

A grid created this way holds up to 1024 items and queries (`grid.dna.get_grid().max_nodes`), and each
address can create a few grids a day. Need more? Write to office@hashgrid.ai.

## Add, search, score

```python
from hashgrid.dna import Grid

grid = Grid()  # Index() and DNA(), from the environment: the key from $HASHGRID_API_KEY

grid.add_many({
    "/cities/lisbon": "Lisbon, a hilly coastal capital",
    "/cities/oslo": "Oslo, between fjords and forest",
})

query, results = grid.ask("a capital by the sea", k=2, prefix="/cities/")
results[0].content                    # "Lisbon, a hilly coastal capital"
grid.score(query, results[0], 1.0)    # a good result
grid.score(query, results[1], -0.5)   # a poor one
```

- **Items** have a `key`, a path unique in the grid. Only items are found by searches, and `prefix`
  narrows a search to keys that start with it. Without a key, an item is keyed by a hash of its
  content, so adding the same content again returns the same item.
- **Queries** have no key. `grid.search(query, k)` gives the next `k` results not shown before.
- **Scores** run from -1 (wrong) to 1 (perfect). You can score a result, or any pair you already know
  is good or bad. Only scores train DNA.

Each `Result` has the item's `id`, `key`, `content` (from the index) and your `score`. `grid.results(query)`
lists a query's results so far, and `grid.get` and `grid.delete` take an item or its id.
`grid.delete_grid()` deletes the whole grid, for good, and its content in the index. `AsyncGrid` does
the same with `await`.

## Images

Content is text or an `Image`, from a file or from bytes. Images need an embedder that takes them:

```python
from hashgrid.dna import Grid
from hashgrid.index import Image, Index

grid = Grid(Index("fastembed:Qdrant/clip-ViT-B-32"))
grid.add(Image("photos/beach.jpg"), key="/photos/beach")
query, results = grid.ask("a sunny beach", k=5)      # text finds images
query, results = grid.ask(Image("another.jpg"))      # and so do images
```

## The index

`Index` is the local half of a grid. It embeds content, and keeps it in sqlite by DNA id, so results
come back with their content. `Index()` uses the default embedder and `~/.hashgrid/index.sqlite`;
`$HASHGRID_EMBEDDER` and `$HASHGRID_INDEX` change them, and so do its arguments:

```python
Index("voyage:voyage-multimodal-3")
Index(OpenAI("text-embedding-3-small", dimensions=512), path="my-index.sqlite")
Index(path=":memory:")   # keeps nothing
```

Use one embedder per grid. Vectors from different models do not compare. The index records each grid's
embedder and refuses another, and a grid refuses vectors whose length differs from its own.

| embedder | takes | runs | needs |
| --- | --- | --- | --- |
| `fastembed:BAAI/bge-small-en-v1.5` (default) | text | locally | `hashgrid[local]` |
| `fastembed:Qdrant/clip-ViT-B-32`, `fastembed:jinaai/jina-clip-v1` | text, images | locally | `hashgrid[local]` |
| `openai:text-embedding-3-small` | text | OpenAI | `OPENAI_API_KEY` |
| `voyage:voyage-3.5` | text | Voyage AI | `VOYAGE_API_KEY` |
| `voyage:voyage-multimodal-3` | text, images | Voyage AI | `VOYAGE_API_KEY` |

Other fastembed, OpenAI and Voyage models work too: `fastembed:<model>`, `openai:<model>`,
`voyage:<model>`. DNA takes up to 2048 values, so `openai:text-embedding-3-large` is shortened to 2048
(`openai:<model>@<n>` sets the length). With OpenAI or Voyage, your content goes to that provider under
your own key, and still never to DNA. Your own embedder is any object with a `spec`, the `modalities` it
takes (`{"text"}`, `{"text", "image"}`) and `embed(contents, kind)`, where `kind` is `"document"` or
`"query"`.

## The API, typed

`DNA` and `AsyncDNA` have one method per route of the [HTTP API](https://dna.hashgrid.ai/docs),
named after it.

```python
from hashgrid.dna import DNA

dna = DNA("hg-…")
item = dna.create_node([0.1, 0.2, 0.3], key="/items/1")
query = dna.create_node([0.1, 0.2, 0.25])
edge = dna.create_edge(query, prefix="/items/")   # the search; None when nothing is left
dna.put_edge(query, edge, 1.0)                    # the score
```

| method | route |
| --- | --- |
| `create_grid(name)`, `acreate_grid(name)` (functions, no key) | `POST /` |
| `get_grid()` | `GET /` |
| `delete_grid()` | `DELETE /` |
| `list_nodes(cursor, limit)`, `iter_nodes()` | `GET /nodes` |
| `get_node(nid)` | `GET /nodes/{nid}` |
| `create_node(emb, key=None)` | `POST /nodes` |
| `update_node(nid, key=…, emb=…)` | `PATCH /nodes/{nid}` |
| `delete_node(nid)` | `DELETE /nodes/{nid}` |
| `list_edges(nid, cursor, limit)`, `iter_edges(nid)` | `GET /nodes/{nid}/edges` |
| `get_edge(nid, eid)` | `GET /nodes/{nid}/edges/{eid}` |
| `create_edge(nid, prefix="")`, alias `search` | `POST /nodes/{nid}/edges` |
| `put_edge(nid, eid, score)`, alias `score` | `PUT /nodes/{nid}/edges/{eid}` |
| `update_edge(nid, eid, score)` | `PATCH /nodes/{nid}/edges/{eid}` |
| `delete_edge(nid, eid)` | `DELETE /nodes/{nid}/edges/{eid}` |

Ids are 16-character hex strings. A method also takes the `Node` or `Edge` itself. Errors are
`Unauthorized` (401), `Forbidden` (403: the grid holds its node limit, or is frozen), `NotFound`
(404), `Conflict` (409, the key is taken), `Invalid` (422) and `TooManyRequests` (429: too many grids
created from your address; try later), all subclasses of `DNAError`.

## MCP server

`hashgrid-mcp` is a local MCP server, over stdio, for agents that work with text and images. It
embeds on your machine, sends DNA only the vectors, and keeps the content in the local index.

```json
{
  "mcpServers": {
    "hashgrid": {
      "command": "uvx",
      "args": ["--from", "hashgrid[local]", "hashgrid-mcp"],
      "env": { "HASHGRID_API_KEY": "hg-…" }
    }
  }
}
```

With Claude Code:

```bash
claude mcp add hashgrid -e HASHGRID_API_KEY=hg-… -- uvx --from 'hashgrid[local]' hashgrid-mcp
```

Without a key, the agent can call `create_grid`: the server makes a grid and saves its key in the
local index, then uses it from then on.

The tools:

- `add_item(text | image, key?)`: add an item, text or the path of an image file.
- `search(query | image, k, prefix)`: the best results, with their text or image path.
- `more_results(query_id, k, prefix)`: the next ones.
- `score(query_id, item_id, score)`: score a result.
- `delete_item(item_id)`: delete an item.
- `create_grid(name)`: make a grid when the server has no key.
- `delete_grid(confirm)`: delete the grid, for good; `confirm` is its name.

The first call downloads the local model, about 70 MB. For images, set
`HASHGRID_EMBEDDER=fastembed:Qdrant/clip-ViT-B-32`.

| variable | default |
| --- | --- |
| `HASHGRID_API_KEY` | the key `create_grid` saved in the index |
| `HASHGRID_EMBEDDER` | `fastembed:BAAI/bge-small-en-v1.5`; with `openai:…` or `voyage:…`, plain `uvx --from hashgrid hashgrid-mcp` is enough |
| `HASHGRID_INDEX` | `~/.hashgrid/index.sqlite` |
| `HASHGRID_DNA_URL` | `https://dna.hashgrid.ai/api/v1` |

The server records each grid's embedder in the index, and refuses another one for that grid.

The hosted server at `https://dna.hashgrid.ai/mcp` takes vectors, not content. Use it if your agent
makes its own embeddings.

## License

MIT
