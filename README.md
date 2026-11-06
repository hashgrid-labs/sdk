# hashgrid

The Python client for [Hashgrid DNA](https://dna.hashgrid.ai), a search engine that learns from your
grades. You store items, ask a question, get the items that fit it best, and grade them with an LLM, a
smaller model or your users. Each grade improves the results for every question.

```bash
pip install hashgrid
```

## Sign up

Each grid has its own API key (`hg-…`). `DNA.signup` makes a grid, no key needed, and returns its key:

```python
from hashgrid import DNA

api_key = DNA.signup("mine")
print(api_key)   # shown only now: store it, e.g. as $HASHGRID_API_KEY
```

`DNA(api_key)` opens the grid, and `DNA()` opens it from `$HASHGRID_API_KEY`. A grid made this way holds up
to 1024 items, queries included (`dna.info().max_items`), and each address can sign up a few times a day.
Need more? Write to office@hashgrid.ai.

## The loop

```python
from hashgrid import DNA

dna = DNA()

dna.put("cities/lisbon", "Lisbon, a hilly coastal capital")
dna.put("cities/oslo", "Oslo, between fjords and forest")

query = dna.query("a capital by the sea")
results = query.results(limit=2, prefix="cities/")
results[0].key                      # "cities/lisbon"
query.grade(results[0], 1.0)        # a good result
query.grade(results[1], -0.5)       # a poor one
```

- **Items** live at keys, like `42` or `tools/42`. `prefix` narrows results and listings to keys that
  start with it. Results return keys and metadata, not content: map the keys back to your own data.
- **Queries** have an `id`. To ask the same question again, reopen it with `dna.query(id=query.id)`, so
  its grades add up.
- **Grades** run from -1 (wrong) to 1 (right). You can grade any item, not only those a query returned.
  Only grades train DNA, and a grade shows in results within a few seconds.

## Content

An item or a query is text (`str`), a record (a `dict`), an image (its `bytes`, or a `Path` to its file),
or an embedding from the grid's model (a list of floats). DNA embeds the rest itself.

```python
from pathlib import Path

dna.put("tools/search", {"name": "search", "description": "Search the web"})
dna.put("photos/beach", Path("beach.jpg"))
dna.put("raw", '{"a": 1}', type="text/plain")   # text that looks like JSON
```

DNA keeps only what it matches on: text and records up to 8 KB, images as a 512 px WebP. `dna.get(key)`
reads one item back, as kept.

## Everything else

| method | route |
| --- | --- |
| `DNA.signup(name)`, returns the API key | `POST /` |
| `dna.info()` | `GET /` |
| `dna.delete_grid()` | `DELETE /` |
| `dna.put(key, content, type=, if_match=, if_none_match=)`, `dna.put_many({key: content})` | `PUT /items/{key}` |
| `dna.get(key)` | `GET /items/{key}` |
| `dna.delete(key)` | `DELETE /items/{key}` |
| `dna.items(prefix)`, every page | `GET /items` |
| `dna.query(content)`, or `dna.query(id=…)` to reopen one | `POST /queries` |
| `query.results(limit, prefix, ungraded=)` | `GET /queries/{id}/results` |
| `query.grade(key, score)` | `PUT /queries/{id}/grades/{key}` |
| `query.grades()`, as `{key: score}` | `GET /queries/{id}/grades` |
| `query.delete_grade(key)` | `DELETE /queries/{id}/grades/{key}` |
| `query.delete()` | `DELETE /queries/{id}` |

Methods take a key or the `Item` / `Result` itself. `AsyncDNA` and `AsyncQuery` have the same methods,
awaited. Errors are subclasses of `DNAError`: `Unauthorized` (401), `Forbidden` (403, the grid is full),
`NotFound` (404), `PreconditionFailed` (412), `TooLarge` (413), `UnsupportedType` (415), `Invalid` (422)
and `TooManyRequests` (429).

For agents, DNA has a hosted MCP server at `https://dna.hashgrid.ai/mcp`.

## License

MIT
