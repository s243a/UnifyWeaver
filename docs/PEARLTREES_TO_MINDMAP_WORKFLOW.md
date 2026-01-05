# Pearltrees to Mind Map Workflow

This document describes the complete pipeline for converting Pearltrees exports to mind map files.

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Pearltrees     │     │    JSONL        │     │   Embeddings    │     │   Mind Map      │
│  RDF Export     │ ──▶ │    Generation   │ ──▶ │   Generation    │ ──▶ │   Output        │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Supported Output Formats

### Primary Format: SMMX

The generator produces SMMX format natively. This is an XML-based format with rich feature support:

| Feature | Support |
|---------|---------|
| Native radial layout | Yes |
| Per-node layout mode | Yes |
| Linked maps (cloudmapref) | Yes |
| Custom node styles | Yes |
| Algorithmic positioning | Yes |

### Converting to Other Formats

Other formats can be obtained via the `--format` option, which generates SMMX internally then converts using `scripts/export_mindmap.py`:

```bash
python3 scripts/generate_mindmap.py --format mm ...      # FreeMind
python3 scripts/generate_mindmap.py --format opml ...    # OPML outline
python3 scripts/generate_mindmap.py --format graphml ... # GraphML
python3 scripts/generate_mindmap.py --format vue ...     # VUE (Visual Understanding Environment)
```

| Target Format | Extension | Notes |
|---------------|-----------|-------|
| FreeMind/Freeplane | `.mm` | Loses some styling, widely compatible |
| OPML | `.opml` | Outline only, no positions |
| GraphML | `.graphml` | Graph format, good for analysis tools |
| VUE | `.vue` | Visual Understanding Environment format |

**Manual conversion** is also possible:
1. Generate SMMX with `--format smmx` (default)
2. Convert using: `python3 scripts/export_mindmap.py input.smmx output.mm`

**Native layout** = the mind map software positions nodes automatically
**Algorithmic layout** = we compute and embed x,y coordinates in the file

---

## Step 1: Export from Pearltrees

Export your Pearltrees data as RDF from the Pearltrees website.

**Location:** `context/PT/`

**Naming convention:** `pearltrees_export_{account}_{date}.rdf`

Example files:
- `pearltrees_export_s243a_2026-01-02.rdf`
- `pearltrees_export_s243a_groups_2026-01-02.rdf`

## Step 2: Generate JSONL from RDF

### Option A: Single Account (with Pearls)

Use `pearltrees_target_generator.py` for single-account exports with pearl support:

```bash
# Generate pearls (bookmarks, links, notes)
python3 scripts/pearltrees_target_generator.py \
  context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  reports/pearltrees_targets_s243a_2026-01-02.jsonl \
  --item-type pearl

# Generate trees only (folders/collections)
python3 scripts/pearltrees_target_generator.py \
  context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  reports/pearltrees_targets_s243a_2026-01-02_trees.jsonl \
  --item-type tree
```

### Option B: Multi-Account (Trees Only)

Use `pearltrees_multi_account_generator.py` to combine multiple accounts with cross-account linking:

```bash
# Generate trees (folders/collections)
python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_groups_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_trees_only.jsonl \
  --item-type tree

# Generate pearls (bookmarks, links, notes)
python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_groups_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_pearls.jsonl \
  --item-type pearl
```

### Option C: Combined Trees + Pearls

For mind maps that include both folder structure (trees) and content items (pearls), combine the outputs:

```bash
# Generate both separately first (Option B above), then combine:
cat reports/pearltrees_targets_combined_2026-01-02_trees.jsonl \
    reports/pearltrees_targets_combined_2026-01-02_pearls.jsonl \
    > reports/pearltrees_targets_combined_2026-01-02_all.jsonl
```

### Understanding Trees vs Pearls

| Type | What it is | Count (example) |
|------|------------|-----------------|
| **Trees** | Folders/collections that organize content | ~13,000 |
| **Pearls** | Actual content items (bookmarks, links, notes) | ~55,000 |

The combined file has **fewer records** than individual pearl files because:
- Trees = containers (folders)
- Pearls = contents (bookmarks inside folders)

A single tree can contain many pearls, so there are always more pearls than trees.

### JSONL Schema

#### Tree records:
```json
{
  "type": "Tree",
  "target_text": "/10311468/2492215\n- s243a\n  - Hacktivism",
  "raw_title": "Hacktivism",
  "query": "Hacktivism",
  "cluster_id": "https://www.pearltrees.com/s243a",
  "tree_id": "2492215",
  "account": "s243a",
  "uri": "https://www.pearltrees.com/t/hacktivism/id2492215"
}
```

#### Pearl records:
```json
{
  "type": "PagePearl",
  "target_text": "...",
  "raw_title": "Article Title",
  "query": "locate_url(Article Title)",
  "cluster_id": "https://www.pearltrees.com/t/hacktivism/id2492215",
  "pearl_id": "#12345",
  "pearl_uri": "https://www.pearltrees.com/t/hacktivism/id2492215#item12345",
  "parent_tree_uri": "https://www.pearltrees.com/t/hacktivism/id2492215",
  "account": "s243a"
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `type` | "Tree" or "PagePearl" |
| `uri` | Unique identifier URL |
| `cluster_id` | Parent tree URL (used for hierarchy) |
| `tree_id` | Numeric ID extracted from URI |
| `target_text` | Hierarchical path + indented title list |
| `raw_title` | Display title |
| `account` | Account name (for multi-account) |

## Step 3: Generate Embeddings

Generate dual embeddings (Nomic + MiniLM) for semantic clustering:

```bash
# For trees only
python3 scripts/generate_dual_embeddings.py \
  --data reports/pearltrees_targets_combined_2026-01-02_trees.jsonl \
  --output models/dual_embeddings_combined_2026-01-02_trees.npz

# For combined trees + pearls
python3 scripts/generate_dual_embeddings.py \
  --data reports/pearltrees_targets_combined_2026-01-02_all.jsonl \
  --output models/dual_embeddings_combined_2026-01-02_all.npz
```

**Output:** `.npz` file containing:
- `input_nomic`: (N, 768) - Nomic embeddings of raw titles
- `input_alt`: (N, 384) - MiniLM embeddings of raw titles
- `output_nomic`: (N, 768) - Nomic embeddings of hierarchical context (materialized ID path + structured title list)

**Important:** The `output_nomic` embeddings are used for curated folder clustering because they encode the full hierarchical path context, not just the raw title. This produces better semantic folder organization.

**Time estimate:** ~1-2 minutes per 10,000 items

## Step 4: Generate Mind Maps

### Basic Generation (Single Map)

```bash
python3 scripts/generate_mindmap.py \
  --data reports/pearltrees_targets_combined_2026-01-02_trees_only.jsonl \
  --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
  --cluster-url "https://www.pearltrees.com/t/hacktivism/id2492215" \
  --output /tmp/hacktivism.smmx \
  --layout radial-auto
```

### Recursive Generation (Full Hierarchy)

```bash
python3 scripts/generate_mindmap.py \
  --data reports/pearltrees_targets_combined_2026-01-02_trees_only.jsonl \
  --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
  --cluster-url "https://www.pearltrees.com/s243a" \
  --recursive \
  --output-dir /tmp/full_output/ \
  --layout radial-auto
```

### With Curated Folders

Organizes output into semantic folders using KMeans clustering:

```bash
python3 scripts/generate_mindmap.py \
  --data reports/pearltrees_targets_combined_2026-01-02_trees_only.jsonl \
  --embeddings models/dual_embeddings_combined_2026-01-02_trees_only.npz \
  --cluster-url "https://www.pearltrees.com/s243a" \
  --recursive \
  --output-dir /tmp/curated_output/ \
  --curated-folders \
  --folder-count 100 \
  --parent-links \
  --titled-files \
  --layout radial-auto
```

---

## Command Reference

### Layout Options

| Option | Description | Format Support |
|--------|-------------|----------------|
| `radial-auto` | Delegate to native layout engine | SMMX, XMind |
| `radial` | Equal angular spacing per parent | All formats |
| `radial-freeform` | Force-directed optimization | All formats |

**When to use each:**
- `radial-auto`: Best for sharing; recipients see nicely-arranged maps without needing to know the software
- `radial`: Standard radial tree layout with computed positions
- `radial-freeform`: Organic, dense layouts (similar to original Pearltrees aesthetic)

### Folder Organization Options

| Option | Description |
|--------|-------------|
| `--curated-folders` | Enable semantic folder organization |
| `--folder-count N` | Target number of folder groups (default: 100) |
| `--folder-method kmeans\|mst-cut` | Clustering method |
| `--titled-files` | Use human-readable filenames |
| `--parent-links` | Add "back to parent" nodes in child maps |
| `--mst-folders` | Alternative: MST-based folder hierarchy |

### Linked Map Options (SMMX only)

| Option | Description |
|--------|-------------|
| `--parent-links` | Add navigation back to parent map |
| `--url-nodes url\|map\|url-label` | How to render URL nodes |

These options use the `cloudmapref` feature specific to SMMX format.

### Style Options (SMMX only)

| Option | Description |
|--------|-------------|
| `--tree-style` | Node shape for trees: `half-round`, `ellipse`, `rectangle`, `diamond` |
| `--pearl-style` | Node shape for pearls: `half-round`, `ellipse`, `rectangle`, `diamond` |

### General Options

| Option | Description |
|--------|-------------|
| `--max-depth N` | Limit recursion depth |
| `--primary-account NAME` | Specify primary account for root selection |
| `--save-folder-map FILE` | Save computed folder structure to JSON |
| `--load-folder-map FILE` | Load pre-computed folder structure |

---

## File Locations

| Type | Location | Naming Convention |
|------|----------|-------------------|
| RDF exports | `context/PT/` | `pearltrees_export_{account}_{date}.rdf` |
| JSONL data | `reports/` | `pearltrees_targets_{account}_{date}.jsonl` |
| Embeddings | `models/` | `dual_embeddings_{name}.npz` |
| Generation log | `reports/` | `GENERATION_LOG.md` |
| Output maps | `/tmp/` or custom | `{title}_id{tree_id}.{ext}` |

---

## Known Issues & Limitations

### 1. Folder Structure Uses Embedding Similarity

The `--curated-folders` option groups items by **embedding similarity**, not the original Pearltrees hierarchy. This can result in unexpected folder assignments (e.g., "Science" in "cynicism" folder).

**Root cause:** KMeans clustering groups semantically similar content, which may differ from user's original organization.

**Planned fix:** Option to use original `pt:parentTree` hierarchy from RDF.

### 2. Self-Referencing Trees

Trees that reference themselves (`cluster_id == uri`) are subtree roots. The code now correctly identifies these and uses path analysis to find the true root.

### 3. Orphan Trees

Trees not connected to the main hierarchy are placed in `_orphans/` folder. This often happens with:
- Cross-account trees not properly linked
- Subtree roots without parent connections

---

## Troubleshooting

### "No items found for cluster"

**Cause:** The cluster URL doesn't exist in the JSONL data.

**Fix:** Check that the URL matches exactly (including `https://` vs `http://`).

### Root is wrong tree

**Cause:** Self-referencing trees were being mishandled.

**Fix:** Updated in `build_user_hierarchy()` to use path analysis from `target_text`.

### Too many orphans

**Cause:** Parent-child relationships broken by self-references.

**Fix:** The code now skips self-references when building hierarchy.

---

## Example Full Workflow

### Trees Only (Folder Structure)

```bash
# 1. Export RDF from Pearltrees (manual step via website)

# 2. Generate trees JSONL
python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_grous_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_trees.jsonl \
  --item-type tree

# 3. Generate embeddings
python3 scripts/generate_dual_embeddings.py \
  --data reports/pearltrees_targets_combined_2026-01-02_trees.jsonl \
  --output models/dual_embeddings_combined_2026-01-02_trees.npz

# 4. Generate mind maps
python3 scripts/generate_mindmap.py \
  --data reports/pearltrees_targets_combined_2026-01-02_trees.jsonl \
  --embeddings models/dual_embeddings_combined_2026-01-02_trees.npz \
  --cluster-url "https://www.pearltrees.com/s243a" \
  --recursive \
  --output-dir /tmp/mindmap_output/ \
  --curated-folders \
  --parent-links \
  --titled-files \
  --layout radial-auto
```

### Combined Trees + Pearls (Full Content)

```bash
# 1. Export RDF from Pearltrees (manual step via website)

# 2. Generate trees and pearls JSONL separately
python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_grous_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_trees.jsonl \
  --item-type tree

python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_grous_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_pearls.jsonl \
  --item-type pearl

# 3. Combine into single file (trees first, then pearls)
cat reports/pearltrees_targets_combined_2026-01-02_trees.jsonl \
    reports/pearltrees_targets_combined_2026-01-02_pearls.jsonl \
    > reports/pearltrees_targets_combined_2026-01-02_all.jsonl

# 4. Generate embeddings for combined file
python3 scripts/generate_dual_embeddings.py \
  --data reports/pearltrees_targets_combined_2026-01-02_all.jsonl \
  --output models/dual_embeddings_combined_2026-01-02_all.npz

# 5. Generate mind maps with all content
python3 scripts/generate_mindmap.py \
  --data reports/pearltrees_targets_combined_2026-01-02_all.jsonl \
  --embeddings models/dual_embeddings_combined_2026-01-02_all.npz \
  --cluster-url "https://www.pearltrees.com/s243a" \
  --recursive \
  --output-dir /tmp/mindmap_output_all/ \
  --curated-folders \
  --parent-links \
  --titled-files \
  --layout radial-auto

# 6. Open output files in your preferred mind map software
```

---

## Related Documentation

- `reports/GENERATION_LOG.md` - Log of generated files
- `docs/MINDMAP_GENERATOR.md` - Generator script details
- `reports/PR_declarative_layout_system.md` - Layout system proposal
- `playbooks/rdf_data_source_playbook.md` - RDF processing with Prolog
