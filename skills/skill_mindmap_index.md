# Skill: Mindmap Index

Build and manage indexes for mindmap lookup and cross-reference tracking.

## When to Use

- User asks "how do I find a mindmap by tree ID?"
- User wants to track which mindmaps link to each other
- User asks about index storage formats (JSON, TSV, SQLite)
- User needs to maintain index consistency after reorganization
- User wants to build backlinks for mindmaps

## Overview

The index system provides two types of indexes:

| Index Type | Purpose | Maps |
|------------|---------|------|
| **Forward Index** | Find mindmap by tree ID | tree_id → file path |
| **Reverse Index** | Find backlinks | tree_id → list of linking mindmaps |

## Quick Start

### Build Forward Index

```bash
# Build index from mindmap directory
python3 scripts/mindmap/build_index.py output/mindmaps/ -o index.json

# Different output formats
python3 scripts/mindmap/build_index.py output/mindmaps/ -o index.tsv   # awk-friendly
python3 scripts/mindmap/build_index.py output/mindmaps/ -o index.db    # SQLite
```

### Build Reverse Index (Backlinks)

```bash
# Find which mindmaps link to which
python3 scripts/mindmap/build_reverse_index.py output/mindmaps/ -o backlinks.json
```

## Storage Backends

The index system supports three storage formats with identical API:

### JSON Store

Human-readable, good for small-to-medium indexes.

```json
{
  "base_dir": "/path/to/mindmaps",
  "count": 1234,
  "index": {
    "10390825": "Gods_of_Earth_and_Nature/id10390825.smmx",
    "75009241": "Physics/id75009241.smmx"
  }
}
```

**Best for:** Development, debugging, small collections (<10K files)

### TSV Store

Tab-separated, awk-friendly for shell scripting.

```
# base_dir: /path/to/mindmaps
10390825	Gods_of_Earth_and_Nature/id10390825.smmx
75009241	Physics/id75009241.smmx
```

**Best for:** Shell scripts, Unix pipelines, grep/awk processing

```bash
# Quick lookup with awk
awk -F'\t' '$1 == "10390825" {print $2}' index.tsv

# Count entries
grep -v '^#' index.tsv | wc -l
```

### SQLite Store

Efficient for large indexes with frequent lookups.

```sql
-- Schema
CREATE TABLE mindmap_index (
    tree_id TEXT PRIMARY KEY,
    path TEXT NOT NULL
);
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

**Best for:** Large collections (>10K files), frequent random access, concurrent reads

```bash
# Quick lookup with sqlite3
sqlite3 index.db "SELECT path FROM mindmap_index WHERE tree_id='10390825'"
```

## Programmatic Usage

### Creating a Store

```python
from scripts.mindmap.index_store import create_index_store

# Auto-detect format from extension
store = create_index_store('index.json')   # JSON
store = create_index_store('index.tsv')    # TSV
store = create_index_store('index.db')     # SQLite

# With caching (preloads entire index into memory)
store = create_index_store('index.db', cache=True)
```

### Basic Operations

```python
# Lookup
path = store.get('10390825')  # Returns 'folder/id10390825.smmx' or None
exists = store.contains('10390825')  # Returns True/False

# Absolute path resolution
abs_path = store.resolve_path('10390825')  # base_dir + relative path

# Modify
store.set('12345678', 'newfolder/id12345678.smmx')
store.delete('12345678')

# Iterate
for tree_id, path in store.items():
    print(f"{tree_id} -> {path}")

# Count
print(f"Total entries: {store.count()}")
```

### Cached Store

For frequent lookups, wrap any backend with caching:

```python
from scripts.mindmap.index_store import CachedStore, SQLiteStore

# Create cached SQLite store
backend = SQLiteStore('index.db')
store = CachedStore(backend, preload=True)

# All lookups now hit memory cache
path = store.get('10390825')  # Fast, from cache

# Invalidate cache if external changes
store.invalidate('10390825')  # Single entry
store.invalidate()            # Entire cache
```

## Reverse Index (Backlinks)

The reverse index tracks which mindmaps link to each mindmap.

### Structure

```json
{
  "10390825": ["75009241", "12345678"],
  "75009241": ["10390825"]
}
```

Meaning: Mindmap 10390825 is linked to by mindmaps 75009241 and 12345678.

### Building

```bash
python3 scripts/mindmap/build_reverse_index.py output/mindmaps/ -o backlinks.json
```

### Usage

```python
from scripts.mindmap.index_store import ReverseIndex

# Load reverse index
reverse = ReverseIndex('backlinks.json')

# Find all mindmaps that link to this one
linkers = reverse.get_linkers('10390825')
print(f"Mindmaps linking to 10390825: {linkers}")

# Essential for rename operations
# Before renaming, find what needs updating
for linker_id in linkers:
    path = forward_index.get(linker_id)
    print(f"Need to update cloudmapref in: {path}")
```

## Integration with Other Tools

### With add_relative_links.py

```bash
# Build index first
python3 scripts/mindmap/build_index.py output/mindmaps/ -o index.json

# Then add cross-links using the index
python3 scripts/mindmap/add_relative_links.py output/mindmaps/*.smmx \
  --index index.json
```

### With rename_mindmap.py

```bash
# Rename uses both forward and reverse index
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps/id10390825.smmx \
  --titled \
  --index index.json
```

### With generate_mindmap.py

```bash
# Pass index to generator for automatic cloudmapref links
python3 scripts/generate_mindmap.py \
  --data pearltrees.jsonl \
  --output output/mindmaps/id12345.smmx \
  --index index.json
```

## Commands

### Build Forward Index
```bash
python3 scripts/mindmap/build_index.py <mindmap_dir> -o <output_file>
```

Options:
- `-o, --output` - Output file (extension determines format)
- `--recursive` - Scan subdirectories

### Build Reverse Index
```bash
python3 scripts/mindmap/build_reverse_index.py <mindmap_dir> -o <output_file>
```

### Query Index (SQLite)
```bash
# Count entries
sqlite3 index.db "SELECT COUNT(*) FROM mindmap_index"

# Find by tree ID
sqlite3 index.db "SELECT path FROM mindmap_index WHERE tree_id='10390825'"

# Find by path pattern
sqlite3 index.db "SELECT * FROM mindmap_index WHERE path LIKE '%Physics%'"
```

## Related

**Parent Skill:**
- `skill_mindmap_tools.md` - Master mindmap skill

**Sibling Skills:**
- `skill_mindmap_rename.md` - Rename with index updates
- `skill_mindmap_cross_links.md` - Cross-references using index

**Code:**
- `scripts/mindmap/build_index.py` - Forward index builder
- `scripts/mindmap/build_reverse_index.py` - Reverse index builder
- `scripts/mindmap/index_store.py` - Storage abstraction layer
