# Use DNA from text

DNA searches embeddings and never sees text. To use it from text, embed on your own machine and send
DNA only the vectors. The `hashgrid` Python package does this for you, and keeps your text with you.

## Install

```bash
pip install 'hashgrid[local]'
export HASHGRID_API_KEY=hg-…
```

`[local]` adds a small embedding model that runs on your CPU, with no account. No API key yet?
Write to office@hashgrid.ai.

## Add, search, score

```python
from hashgrid.dna import DNA, TextGrid
from hashgrid.embed import from_spec

grid = TextGrid(DNA(), from_spec())

items = grid.add_many({
    "/docs/install": "How to install the CLI on macOS and Linux",
    "/docs/billing": "Invoices, refunds and changing your plan",
})

query, results = grid.ask("I was charged twice", k=2, prefix="/docs/")
grid.score(query, results[0], 1.0)
```

Each result is an item's id. `items` holds each item's id and key, so you can map results back to
the text you already have. Score the results that were right and the ones that were wrong. DNA learns from every score,
so the next searches rank better.

## Pick an embedder once per grid

Vectors from different models do not compare, so use one model per grid.

- `fastembed:BAAI/bge-small-en-v1.5`, the default, runs locally.
- `openai:text-embedding-3-small` uses `OPENAI_API_KEY`.
- `voyage:voyage-3.5` uses `VOYAGE_API_KEY`.

Set it in `HASHGRID_EMBEDDER`, or pass it to `from_spec`. With OpenAI or Voyage, your text goes to
that provider under your own key, and still never to DNA.

## From an agent: the local MCP server

```bash
claude mcp add hashgrid -e HASHGRID_API_KEY=hg-… -- uvx --from 'hashgrid[local]' hashgrid-mcp
```

The agent gets five tools: `add_item`, `search`, `more_results`, `score` and `delete_item`. They take
and return text. The server embeds locally and keeps the text in a local index,
`~/.hashgrid/index.sqlite`. Only vectors go to DNA.

If your agent makes its own embeddings, use the hosted server at `https://dna.hashgrid.ai/mcp`
instead. It takes vectors.
