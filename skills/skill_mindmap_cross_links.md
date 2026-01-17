# Skill: Mindmap Cross-Links

Add `cloudmapref` links between mindmaps for local navigation without external URLs.

## When to Use

- User wants mindmaps to link to each other
- User asks to "connect", "link", or "cross-reference" mindmaps
- User needs to rename mindmaps while preserving links

## Quick Start

```bash
# Build index, then add links
python3 scripts/mindmap/build_index.py output/mindmaps/ -o output/mindmaps/index.json
python3 scripts/mindmap/add_relative_links.py output/mindmaps/*.smmx --index output/mindmaps/index.json
```

## Commands

### Build Index
```bash
python3 scripts/mindmap/build_index.py "MINDMAP_DIR" -o "INDEX_FILE"
```

### Add Cross-Links
```bash
python3 scripts/mindmap/add_relative_links.py "MINDMAP_FILES" \
  --index "INDEX_FILE"
```

### Preview Changes (Dry Run)
```bash
python3 scripts/mindmap/add_relative_links.py "MINDMAP_FILES" \
  --index "INDEX_FILE" \
  --dry-run --verbose
```

### Rename Single Mindmap
```bash
# Auto-generate name from root topic
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap "MINDMAP_PATH" \
  --titled

# Explicit new name
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap "MINDMAP_PATH" \
  --new-name "New_Name.smmx"
```

### Batch Rename All
```bash
python3 scripts/mindmap/rename_mindmap.py \
  --batch "MINDMAP_DIR" \
  --titled \
  --dry-run
```

### Build Reverse Index (Backlinks)
```bash
python3 scripts/mindmap/build_reverse_index.py "MINDMAP_DIR" \
  -o "REVERSE_INDEX_FILE"
```

### Get All Options
```bash
python3 scripts/mindmap/build_index.py --help
python3 scripts/mindmap/add_relative_links.py --help
python3 scripts/mindmap/rename_mindmap.py --help
```

## Index Formats

| Extension | Format | Use Case |
|-----------|--------|----------|
| `.json` | JSON | Human-readable |
| `.tsv` | TSV | Shell scripting |
| `.db` | SQLite | Large collections |

## Workflow

1. Build index from mindmap directory
2. Add relative links using index
3. When renaming, use `rename_mindmap.py` to update all references

## Related

- `skill_mindmap_linking.md` - Link to Pearltrees
- `scripts/mindmap/README.md` - Full documentation
