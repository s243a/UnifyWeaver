# Mindmap Tools

Tools for managing SimpleMind mindmap files (.smmx) with relative linking support.

## Overview

These tools enable cross-linking between mindmaps using relative paths (`cloudmapref`), allowing navigation between related mindmaps without relying on external URLs.

## Tools

### build_index.py

Build an index mapping tree IDs to mindmap file paths.

```bash
# Build index from mindmap directory
python3 scripts/mindmap/build_index.py output/mindmaps_curated/ -o output/mindmaps_curated/index.json

# Build TSV index (awk-friendly)
python3 scripts/mindmap/build_index.py output/mindmaps_curated/ -o output/mindmaps_curated/index.tsv
```

The index maps tree IDs (e.g., `"10390825"`) to relative file paths (e.g., `"Gods_of_Earth_and_Nature/id10390825.smmx"`).

### add_relative_links.py

Add `cloudmapref` links to existing mindmaps using the index.

```bash
# Preview changes (dry-run)
python3 scripts/mindmap/add_relative_links.py output/mindmaps_curated/*.smmx \
  --index output/mindmaps_curated/index.json --dry-run --verbose

# Apply changes
python3 scripts/mindmap/add_relative_links.py output/mindmaps_curated/*.smmx \
  --index output/mindmaps_curated/index.json

# Specify root directory for path computation
python3 scripts/mindmap/add_relative_links.py /tmp/repaired_map.smmx \
  --index output/mindmaps_curated/index.json \
  --root output/mindmaps_curated/
```

For each Pearltrees URL in a mindmap, looks up the corresponding local mindmap in the index and adds a `cloudmapref` attribute for local navigation.

### build_reverse_index.py

Build a reverse index tracking which mindmaps link to each mindmap (backlinks).

```bash
python3 scripts/mindmap/build_reverse_index.py output/mindmaps_curated/ \
  -o output/mindmaps_curated/reverse_index.json
```

Useful for:
- Finding all mindmaps that reference a given mindmap
- Updating links when a mindmap is renamed or moved

### rename_mindmap.py

Rename mindmap files and automatically update all `cloudmapref` references.

```bash
# Single file: auto-generate titled filename from root topic
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --titled

# Single file: explicit new name
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --new-name "Technology_10380971.smmx"

# Batch: rename all to titled format
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps_curated/ \
  --titled

# Preview changes without making them
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --titled --dry-run --verbose
```

Features:
- Generates `Title_ID.smmx` filenames from root topic text
- Scans for all mindmaps with `cloudmapref` pointing to the renamed file
- Updates relative paths (including `../../../` paths)
- Batch mode updates all references before renaming files
- Optional index update via `--index`

### index_store.py

Abstraction layer for index storage with pluggable backends:

- **JSONStore**: Human-readable, good for small indexes
- **TSVStore**: Tab-separated, awk-friendly for shell scripting
- **SQLiteStore**: Efficient for large indexes with frequent lookups
- **CachedStore**: Wrapper adding in-memory caching to any backend

```python
from mindmap.index_store import create_index_store

# Auto-detect format from extension
store = create_index_store('index.json')
store = create_index_store('index.tsv')
store = create_index_store('index.db')

# With caching
store = create_index_store('index.json', cache=True)

# Usage
path = store.get('10390825')  # Returns 'Gods_of_Earth_and_Nature/id10390825.smmx'
store.set('12345678', 'subfolder/id12345678.smmx')
```

## Integration with generate_mindmap.py

The main mindmap generator supports the `--index` option to create both URL and cloudmapref links during generation:

```bash
python3 scripts/generate_mindmap.py \
  --data reports/pearltrees_targets.jsonl \
  --cluster-url "https://www.pearltrees.com/s243a/people/id10390825" \
  --output output/mindmaps_curated/id10390825.smmx \
  --index output/mindmaps_curated/index.json
```

When `--index` is provided:
- Main nodes get `urllink` to Pearltrees website
- Child nodes (square, unlabeled) get `cloudmapref` to local mindmap
- Only trees found in the index get cloudmapref links

You can override this behavior with `--url-nodes`:
- `--url-nodes url` (default with index): URL on main, cloudmapref on child
- `--url-nodes map`: cloudmapref on main, URL on child
- `--url-nodes url-label`: cloudmapref on main, URL on labeled child

## Workflow

### Initial Setup

1. Generate mindmaps recursively:
   ```bash
   python3 scripts/generate_mindmap.py \
     --cluster-url "https://www.pearltrees.com/s243a" \
     --recursive --output-dir output/mindmaps_curated/
   ```

2. Build the index:
   ```bash
   python3 scripts/mindmap/build_index.py output/mindmaps_curated/ \
     -o output/mindmaps_curated/index.json
   ```

3. Add relative links to all mindmaps:
   ```bash
   python3 scripts/mindmap/add_relative_links.py output/mindmaps_curated/**/*.smmx \
     --index output/mindmaps_curated/index.json
   ```

### Regenerating a Single Mindmap

When regenerating a mindmap, use the index to include relative links:

```bash
python3 scripts/generate_mindmap.py \
  --cluster-url "https://www.pearltrees.com/s243a/science/id12345678" \
  --output output/mindmaps_curated/science/id12345678.smmx \
  --index output/mindmaps_curated/index.json
```

### Handling Renames/Moves

Use `rename_mindmap.py` to rename a mindmap and automatically update all references:

```bash
# Rename with auto-generated title from root topic
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --titled --dry-run

# Rename with explicit new name
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps_curated/id10380971.smmx \
  --new-name "Technology_10380971.smmx"

# Batch rename all mindmaps to titled format (Title_ID.smmx)
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps_curated/ \
  --titled --dry-run
```

The tool:
1. Finds all mindmaps with `cloudmapref` pointing to the renamed file
2. Updates their relative paths to the new filename
3. Renames the file
4. Optionally updates the index
