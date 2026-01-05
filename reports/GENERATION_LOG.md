# JSONL Generation Log

This document tracks the generation of JSONL files from Pearltrees RDF exports.

## Generation Records

### 2026-01-03 (Generated from 2026-01-02 exports)

| Output File | Source RDF | Generated | Item Type | Records | Size |
|-------------|------------|-----------|-----------|---------|------|
| `pearltrees_targets_s243a_2026-01-02.jsonl` | `pearltrees_export_s243a_2026-01-02.rdf` | 2026-01-03 23:45 | pearl | 34,329 | 15.0 MB |
| `pearltrees_targets_s243a_groups_2026-01-02.jsonl` | `pearltrees_export_s243a_grous_2026-01-02.rdf` | 2026-01-03 23:46 | pearl | 20,836 | 11.6 MB |
| `pearltrees_targets_combined_2026-01-02_trees_only.jsonl` | Both RDF files (s243a primary) | 2026-01-03 23:46 | tree | 13,279 | 6.4 MB |

**Commands used:**
```bash
# Individual s243a pearls
python3 scripts/pearltrees_target_generator.py \
  context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  reports/pearltrees_targets_s243a_2026-01-02.jsonl --item-type pearl

# Individual groups pearls
python3 scripts/pearltrees_target_generator.py \
  context/PT/pearltrees_export_s243a_grous_2026-01-02.rdf \
  reports/pearltrees_targets_s243a_groups_2026-01-02.jsonl --item-type pearl

# Combined multi-account trees
python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_grous_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_trees_only.jsonl --item-type tree
```

### 2026-01-01 (Previous)

| Output File | Source RDF | Generated | Item Type | Filter | Records |
|-------------|------------|-----------|-----------|--------|---------|
| `pearltrees_targets_s243a.jsonl` | `pearltrees_export_s243a_2025-12-27.rdf` | 2026-01-01 00:01 | pearl | none | ~4.8MB |
| `pearltrees_targets_s243a_groups.jsonl` | `pearltrees_export_s243a_grous_2025-12-27.rdf` | 2026-01-01 00:01 | pearl | none | ~1MB |

## Source Files Location

- RDF exports: `/home/s243a/Projects/UnifyWeaver/context/PT/`
- JSONL outputs: `/home/s243a/Projects/UnifyWeaver/reports/`

## Generation Command

```bash
python scripts/pearltrees_target_generator.py <input.rdf> <output.jsonl> [--item-type pearl|tree] [--filter-path "..."]
```

## Notes

- The `s243a` export contains the main personal Pearltrees account
- The `groups` export contains team/group Pearltrees
- Item types: `pearl` = content items (links, notes), `tree` = folders/collections
- **Trees vs Pearls**: The combined trees file has fewer records (~13K) than individual pearl files (~55K total) because trees are containers (folders) while pearls are the actual content items inside them
- See `docs/PEARLTREES_TO_SIMPLEMIND_WORKFLOW.md` for full workflow documentation
