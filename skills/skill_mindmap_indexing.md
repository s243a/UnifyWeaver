# Skill: Mindmap Indexing

Sub-master skill for tracking mindmap locations and maintaining index consistency when files move or are renamed.

## When to Use

- User asks "how do I track mindmap locations?"
- User wants to find a mindmap by tree ID
- User asks about backlinks or reverse indexing
- User needs to rename mindmaps without breaking links
- User wants to maintain index consistency after reorganization

## Overview

The indexing system provides:

1. **Forward Index** - Map tree IDs to file paths for lookup
2. **Reverse Index** - Track backlinks (who references whom)
3. **Storage Backends** - JSON, TSV, SQLite with optional caching
4. **Rename Tool** - Rename files and automatically update all references

## Individual Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `skill_mindmap_index.md` | Forward/reverse indexes, storage backends | Building and querying indexes |
| `skill_mindmap_rename.md` | Rename files, update all references | Renaming without breaking links |

## Why Indexing Matters

When mindmaps reference each other via `cloudmapref` attributes:

```xml
<link cloudmapref="../Physics/id75009241.smmx"/>
```

These paths must stay valid when files move. The index system:

1. **Tracks locations** - Know where every mindmap is
2. **Finds references** - Know what links to what
3. **Updates on rename** - Fix all references automatically

## The Indexing Pipeline

### 1. Build Forward Index

Map tree IDs to file paths:

```bash
python3 scripts/mindmap/build_index.py output/mindmaps/ -o index.json
```

**Result:**
```json
{
  "index": {
    "10390825": "Physics/id10390825.smmx",
    "75009241": "Math/id75009241.smmx"
  }
}
```

### 2. Build Reverse Index

Track which mindmaps link to which (backlinks):

```bash
python3 scripts/mindmap/build_reverse_index.py output/mindmaps/ -o backlinks.json
```

**Result:**
```json
{
  "10390825": ["75009241", "12345678"],
  "75009241": ["10390825"]
}
```

Meaning: Mindmap 10390825 is referenced by mindmaps 75009241 and 12345678.

### 3. Rename with Index Update

Rename a file and update all references:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps/id10390825.smmx \
  --new-name "Quantum_Physics_10390825.smmx" \
  --index index.json
```

This:
1. Renames the file
2. Finds all mindmaps that reference it (via reverse index)
3. Updates their `cloudmapref` attributes
4. Updates the forward index

See `skill_mindmap_rename.md` for full details.

## Storage Backend Selection

| Format | Extension | Best For |
|--------|-----------|----------|
| **JSON** | `.json` | Development, small indexes, human-readable |
| **TSV** | `.tsv` | Shell scripts, awk/grep processing |
| **SQLite** | `.db` | Large indexes, frequent queries, concurrent access |

```bash
# Auto-detected by extension
python3 scripts/mindmap/build_index.py output/ -o index.json   # JSON
python3 scripts/mindmap/build_index.py output/ -o index.tsv    # TSV
python3 scripts/mindmap/build_index.py output/ -o index.db     # SQLite
```

See `skill_mindmap_index.md` for storage details.

## Common Workflows

### After Reorganizing Folders

When you move mindmaps into new folders:

```bash
# 1. Rebuild the forward index
python3 scripts/mindmap/build_index.py output/organized/ -o index.json

# 2. Update cloudmapref links to use new paths
python3 scripts/mindmap/add_relative_links.py output/organized/**/*.smmx \
  --index index.json --update
```

### Batch Rename to Titled Format

Convert `id12345.smmx` to `Title_12345.smmx`:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps/ \
  --titled \
  --index index.json
```

### Check Index Consistency

Verify index matches actual files:

```bash
# List entries in index
sqlite3 index.db "SELECT COUNT(*) FROM mindmap_index"

# Find orphaned entries (in index but file missing)
python3 scripts/mindmap/verify_index.py --index index.json --dir output/
```

## Integration with Other Tools

### With add_relative_links.py

```bash
# Index must exist before adding cross-links
python3 scripts/mindmap/build_index.py output/ -o index.json
python3 scripts/mindmap/add_relative_links.py output/*.smmx --index index.json
```

### With generate_mindmap.py

```bash
# Pass index to generator for automatic cloudmapref
python3 scripts/generate_mindmap.py \
  --data data.jsonl \
  --output output/id12345.smmx \
  --index index.json
```

### With mst_folder_grouping.py

```bash
# After reorganizing, rebuild index
python3 scripts/mindmap/mst_folder_grouping.py ... -o structure.json
python3 scripts/mindmap/apply_folder_structure.py ...
python3 scripts/mindmap/build_index.py output/organized/ -o index.json
```

## Related

**Parent Skill:**
- `skill_mindmap_tools.md` - Master mindmap skill

**Individual Skills:**
- `skill_mindmap_index.md` - Forward/reverse indexes, storage backends
- `skill_mindmap_rename.md` - Rename files, update references

**Sibling Sub-Masters:**
- `skill_mindmap_organization.md` - Folder organization
- `skill_mindmap_references.md` - Cross-linking

**Code:**
- `scripts/mindmap/build_index.py` - Forward index builder
- `scripts/mindmap/build_reverse_index.py` - Reverse index builder
- `scripts/mindmap/index_store.py` - Storage abstraction
- `scripts/mindmap/rename_mindmap.py` - Rename with updates
