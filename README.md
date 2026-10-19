# hashgrid

The Python client for [Hashgrid DNA](https://dna.hashgrid.ai), a search engine that learns from
feedback. You add items and search them; you score the results; DNA learns from the scores, so the
next searches rank better.

DNA works on embeddings only and never sees your text. This package turns text into embeddings on
your machine, sends DNA only the vectors, and keeps the text with you.

<!-- mcp-name: ai.hashgrid/sdk -->

```bash
pip install 'hashgrid[local]'
```

`[local]` adds the default embedder, which runs on your CPU with no account. To embed with OpenAI or
Voyage instead, install plain `hashgrid` and use your own key.

You need a DNA API key (`hg-…`), one per grid. No key yet? Write to office@hashgrid.ai.

## From text

```python
from hashgrid.dna import DNA, TextGrid
from hashgrid.embed import from_spec

grid = TextGrid(DNA(), from_spec())  # key from $HASHGRID_API_KEY

grid.add_many({
    "/cities/lisbon": "Lisbon, a hilly coastal capital",
    "/cities/oslo": "Oslo, between fjords and forest",
})

query, results = grid.ask("a capital by the sea", k=2, prefix="/cities/")
grid.score(query, results[0], 1.0)    # a good result
grid.score(query, results[1], -0.5)   # a poor one
```

- **Items** have a `key`, a path unique in the grid. Only items are found by searches, and `prefix`
  narrows a search to keys that start with it.
- **Queries** have no key. `grid.search(query, k)` gives the next `k` results not shown before.
- **Scores** run from -1 (wrong) to 1 (perfect). You can score a result, or any pair you already know
  is good or bad. Only scores train DNA.

Results are ids (`edge.id` is the item's id). Map them back to your text with the ids `add` returned,
or by key.

`AsyncTextGrid` does the same with `await`.

## Embedders

Use one embedder per grid. Vectors from different models do not compare. `TextGrid` refuses
vectors whose length differs from the grid's.

| spec | runs | needs |
| --- | --- | --- |
| `fastembed:BAAI/bge-small-en-v1.5` (default) | locally, on CPU | `hashgrid[local]` |
| `openai:text-embedding-3-small` | OpenAI | `OPENAI_API_KEY` |
| `voyage:voyage-3.5` | Voyage AI | `VOYAGE_API_KEY` |

`from_spec()` reads `$HASHGRID_EMBEDDER` and falls back to the default. Any fastembed, OpenAI or
Voyage model works: `fastembed:<model>`, `openai:<model>`, `voyage:<model>`. DNA takes up to 2048
values, so `openai:text-embedding-3-large` is shortened to 2048 (`openai:<model>@<n>` sets the
length). Your own embedder is any object with `spec`, `embed_documents(texts)` and
`embed_queries(texts)`.

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
| `get_grid()` | `GET /` |
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
`Unauthorized` (401), `NotFound` (404), `Conflict` (409, the key is taken) and `Invalid` (422), all
subclasses of `DNAError`.

## MCP server

`hashgrid-mcp` is a local MCP server, over stdio, for agents that work in text. It embeds on your
machine, sends DNA only the vectors, and keeps the text in a local index, a sqlite file.

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

The tools:

- `add_item(text, key?)`: add an item.
- `search(query, k, prefix)`: the best results, with their text.
- `more_results(query_id, k, prefix)`: the next ones.
- `score(query_id, item_id, score)`: score a result.
- `delete_item(item_id)`: delete an item.

The first search downloads the local model, about 70 MB.

| variable | default |
| --- | --- |
| `HASHGRID_API_KEY` | required |
| `HASHGRID_EMBEDDER` | `fastembed:BAAI/bge-small-en-v1.5`; with `openai:…` or `voyage:…`, plain `uvx --from hashgrid hashgrid-mcp` is enough |
| `HASHGRID_INDEX` | `~/.hashgrid/index.sqlite` |
| `HASHGRID_DNA_URL` | `https://dna.hashgrid.ai/api/v1` |

The server records each grid's embedder in the index, and refuses another one for that grid.

The hosted server at `https://dna.hashgrid.ai/mcp` takes vectors, not text. Use it if your agent
makes its own embeddings.

## License

MIT
