# Use DNA from text and images

DNA searches embeddings and never sees your content. To use it from text or images, embed on your own
machine and send DNA only the vectors. The `hashgrid` Python package does this for you, and keeps your
content with you, in a local index.

## Install

```bash
pip install 'hashgrid[local]'
export HASHGRID_API_KEY=hg-…
```

`[local]` adds embedding models that run on your CPU, with no account. No API key yet? Write to
office@hashgrid.ai.

## Add, search, score

```python
from hashgrid.dna import Grid

grid = Grid()

grid.add_many({
    "/docs/install": "How to install the CLI on macOS and Linux",
    "/docs/billing": "Invoices, refunds and changing your plan",
})

query, results = grid.ask("I was charged twice", k=2, prefix="/docs/")
print(results[0].key, results[0].content)
grid.score(query, results[0], 1.0)
```

Results come back with their content, from the local index. Score the results that were right and the
ones that were wrong. DNA learns from every score, so the next searches rank better.

## Images

```python
from hashgrid.dna import Grid
from hashgrid.index import Image, Index

grid = Grid(Index("fastembed:Qdrant/clip-ViT-B-32"))
grid.add(Image("photos/beach.jpg"), key="/photos/beach")
query, results = grid.ask("a sunny beach", k=5)
```

A text query finds images, and an image finds images or text.

## Pick an embedder once per grid

Vectors from different models do not compare, so use one model per grid.

- `fastembed:BAAI/bge-small-en-v1.5`, the default: text, local.
- `fastembed:Qdrant/clip-ViT-B-32`: text and images, local.
- `openai:text-embedding-3-small`: text, with `OPENAI_API_KEY`.
- `voyage:voyage-3.5`, or `voyage:voyage-multimodal-3` for images: with `VOYAGE_API_KEY`.

Pass it to `Index`, or set `HASHGRID_EMBEDDER`. With OpenAI or Voyage, your content goes to that
provider under your own key, and still never to DNA.

## From an agent: the local MCP server

```bash
claude mcp add hashgrid -e HASHGRID_API_KEY=hg-… -- uvx --from 'hashgrid[local]' hashgrid-mcp
```

The agent gets five tools: `add_item`, `search`, `more_results`, `score` and `delete_item`. They take
and return text, or image file paths. The server embeds locally and keeps the content in the local
index, `~/.hashgrid/index.sqlite`. Only vectors go to DNA.

If your agent makes its own embeddings, use the hosted server at `https://dna.hashgrid.ai/mcp`
instead. It takes vectors.
