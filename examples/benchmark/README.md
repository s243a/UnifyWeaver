# Cross-Target Effective Distance Benchmark

## Overview

Computes effective distance from Wikipedia articles to root categories
via the category hierarchy, compiled to multiple targets (C# Query, Go,
AWK, Python). See `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_*.md`.

## Data Provenance

### Source: Simple English Wikipedia

All data comes from **Simple English Wikipedia** database dumps, downloaded
from `https://dumps.wikimedia.org/simplewiki/latest/`:

| File | Size | Contents |
|------|------|----------|
| `simplewiki-latest-page.sql.gz` | ~32 MB | Page table (page_id, title, namespace) |
| `simplewiki-latest-categorylinks.sql.gz` | ~27 MB | Category links (page_id → target_id, type) |
| `simplewiki-latest-linktarget.sql.gz` | ~36 MB | Link target resolution (target_id → title) |

These are stored in `data/simplewiki/` (not committed — too large).

### Why Simple Wikipedia?

The Cohere Wikipedia embeddings dataset (`Supabase/wikipedia-en-embeddings`)
used elsewhere in this project is sourced from Simple English Wikipedia.
Using simplewiki category data ensures article titles match.

### Schema Note (MediaWiki post-2024)

The categorylinks table uses a new schema where `cl_to` (category name string)
was replaced by `cl_target_id` (a reference to the `linktarget` table). The
parser resolves this JOIN:

    categorylinks.cl_target_id → linktarget.lt_id → linktarget.lt_title

### SQLite Database

Parsed dumps are stored in `data/simplewiki/simplewiki_categories.db`:

```
Tables:
  page             - 483K rows (articles ns=0: 392K, categories ns=14: 92K)
  linktarget       - 336K rows (category namespace targets only)
  categorylinks_raw - 2.2M rows (raw cl_from → cl_target_id)
  categorylinks    - 2.2M rows (resolved: cl_from → category_name)
    page:   1.9M (article → category)
    subcat: 297K (category → parent category)
    file:   250
```

## Scripts

| Script | Purpose |
|--------|---------|
| `parse_simplewiki_dump.py` | Parse Wikipedia SQL dumps → SQLite |
| `generate_facts_from_db.py` | Extract facts from SQLite → Prolog/TSV (no crawling) |
| `generate_facts.py` | Alternative: fetch from Wikipedia API (for small datasets) |
| `effective_distance.pl` | Benchmark Prolog program |

## Usage

```bash
# One-time: parse dumps into SQLite
python examples/benchmark/parse_simplewiki_dump.py

# Generate dev dataset (20 articles, Physics root)
python examples/benchmark/generate_facts_from_db.py \
    --articles /path/to/wikipedia_physics_articles.jsonl \
    --output data/benchmark/dev/ \
    --max-articles 20 \
    --root Physics

# Run benchmark in SWI-Prolog (reference output)
swipl -l examples/benchmark/effective_distance.pl \
      -l data/benchmark/dev/facts.pl \
      -g run_benchmark -t halt
```

## Dev Dataset Stats

- Root: Physics
- Articles: 19 (of 20 requested; 1 had no category under Physics)
- Relevant categories: 121 (trimmed from 76K descendants)
- Hierarchy edges: 198
- Total facts: 230
