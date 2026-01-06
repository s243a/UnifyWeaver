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

#### AliasPearl/RefPearl records (cross-account links):
```json
{
  "type": "AliasPearl",
  "target_text": "...",
  "raw_title": "Link to Other Tree",
  "query": "locate_url(Link to Other Tree)",
  "cluster_id": "https://www.pearltrees.com/s243a",
  "pearl_id": "#67890",
  "pearl_uri": "https://www.pearltrees.com/s243a#item67890",
  "parent_tree_uri": "https://www.pearltrees.com/s243a",
  "account": "s243a",
  "alias_target_uri": "https://www.pearltrees.com/s243a_groups/academic-disciplines/id53344165"
}
```

**Note:** `alias_target_uri` is only present for AliasPearl and RefPearl types. This field contains the target tree URL that the alias points to, enabling cross-account hierarchy traversal during mind map generation.

### Key Fields

| Field | Description |
|-------|-------------|
| `type` | "Tree", "PagePearl", "AliasPearl", or "RefPearl" |
| `uri` | Unique identifier URL |
| `cluster_id` | Parent tree URL (used for hierarchy) |
| `tree_id` | Numeric ID extracted from URI |
| `target_text` | Hierarchical path + indented title list (with actual titles) |
| `raw_title` | Display title |
| `account` | Account name (for multi-account) |
| `alias_target_uri` | (AliasPearl/RefPearl only) Target tree URL for cross-account links |

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

| Array | Shape | Description |
|-------|-------|-------------|
| `input_nomic` | (N, 768) | Nomic embeddings of raw titles |
| `input_alt` | (N, 384) | MiniLM embeddings of raw titles |
| `output_nomic` | (N, 768) | Nomic embeddings of hierarchical context (`target_text`) |
| `titles` | (N,) | Raw title strings (object array) |
| `item_types` | (N,) | Item type: "Tree", "PagePearl", "AliasPearl", etc. |
| `tree_ids` | (N,) | Numeric tree/pearl IDs |
| `uris` | (N,) | Unique URIs for each item |
| `output_texts` | (N,) | Full `target_text` strings for debugging |

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

### 4. Cross-Account Linking via AliasPearl

When using multiple accounts (e.g., `s243a` and `s243a_groups`), content in one account can reference trees in another via AliasPearl/RefPearl links. The generator now follows these `alias_target_uri` links during recursive traversal, enabling full cross-account hierarchy generation.

**Example:** If `s243a` has an AliasPearl pointing to `s243a_groups/academic-disciplines/id53344165`, the generator will include that tree and its children in the output.

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

## When to Regenerate JSONL Files

The JSONL files need to be regenerated when:

1. **RDF Export Updated** - New exports from Pearltrees with additional content
2. **Schema Changes** - When fields are added/modified (e.g., `alias_target_uri` for cross-account links)
3. **Hierarchy Fixes** - Corrections to path generation or parent-child relationships

### Regeneration Commands

```bash
# 1. Regenerate trees JSONL
python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_groups_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_trees_only.jsonl \
  --item-type tree

# 2. Regenerate pearls JSONL
python3 scripts/pearltrees_multi_account_generator.py \
  --primary s243a \
  --account s243a context/PT/pearltrees_export_s243a_2026-01-02.rdf \
  --account s243a_groups context/PT/pearltrees_export_s243a_groups_2026-01-02.rdf \
  reports/pearltrees_targets_combined_2026-01-02_pearls.jsonl \
  --item-type pearl

# 3. Combine into all.jsonl
cat reports/pearltrees_targets_combined_2026-01-02_trees_only.jsonl \
    reports/pearltrees_targets_combined_2026-01-02_pearls.jsonl \
    > reports/pearltrees_targets_combined_2026-01-02_all.jsonl

# 4. Regenerate embeddings for combined file
python3 scripts/generate_dual_embeddings.py \
  --data reports/pearltrees_targets_combined_2026-01-02_all.jsonl \
  --output models/dual_embeddings_combined_2026-01-02_all.npz
```

### Verification

After regeneration, verify the output:

```bash
# Check record counts
wc -l reports/pearltrees_targets_combined_2026-01-02_*.jsonl

# Check for alias_target_uri in pearls
grep -c '"alias_target_uri"' reports/pearltrees_targets_combined_2026-01-02_pearls.jsonl

# Sample a record to verify structure
head -1 reports/pearltrees_targets_combined_2026-01-02_all.jsonl | python3 -m json.tool
```

---

## Repairing Incomplete Data

RDF exports often have incomplete data - trees exist but their children weren't exported. Pearltrees provides two additional export formats that can be used to repair missing data:

### Export Types

| Export Type | Contains | URI Info | Use Case |
|-------------|----------|----------|----------|
| **RDF** | Full schema | Complete URIs | Primary data source |
| **Zip Export** | HTML files per pearl | Full pearl URIs | Repair missing PagePearls |
| **HTML Export** | Netscape bookmarks | Only for AliasPearls | Discover hierarchy, find missing trees |

### Zip Export Repair

Zip exports contain HTML files for each PagePearl with both the Pearltrees URI and source URL.

**Script:** `scripts/repair_from_zip_export.py`

```bash
# Compare zip export with existing JSONL (see what's missing)
python3 scripts/repair_from_zip_export.py \
    --export-dir "context/PT/Zip_Exports/Academic disciplines" \
    --root-uri "https://www.pearltrees.com/s243a/academic-disciplines/id53344165" \
    --account s243a \
    --compare reports/pearltrees_targets_repaired.jsonl \
    --output /tmp/missing_items.jsonl

# Auto-merge repairs into existing JSONL (creates backup)
python3 scripts/repair_from_zip_export.py \
    --export-dir "context/PT/Zip_Exports/Academic disciplines" \
    --root-uri "https://www.pearltrees.com/s243a/academic-disciplines/id53344165" \
    --account s243a \
    --merge-into reports/pearltrees_targets_repaired.jsonl
```

**What it extracts:**
- PagePearl URIs (from HTML `see-medal` div)
- Source URLs (from `author-medal` div)
- Parent tree URIs (embedded in pearl URL: `.../id{tree_id}/item{pearl_id}`)
- Child tree URIs (discovered from pearls in subfolders)

### HTML Export Repair

HTML exports (Netscape bookmark format) contain folder hierarchy but limited URI info.

**Script:** `scripts/repair_from_html_export.py`

```bash
# Analyze HTML export structure
python3 scripts/repair_from_html_export.py \
    --html-export "context/PT/pearltrees_export_s243a_2026-01-05.html" \
    --analyze

# Compare with JSONL and find missing trees from AliasPearls
python3 scripts/repair_from_html_export.py \
    --html-export "context/PT/pearltrees_export_s243a_2026-01-05.html" \
    --compare reports/pearltrees_targets_repaired.jsonl \
    --extract-missing-trees /tmp/missing_trees.jsonl

# Build hybrid hierarchy with title paths + ID patterns
python3 scripts/repair_from_html_export.py \
    --html-export "context/PT/pearltrees_export_s243a_2026-01-05.html" \
    --compare reports/pearltrees_targets_repaired.jsonl \
    --build-hierarchy /tmp/hierarchy.jsonl
```

**What it extracts:**
- Folder hierarchy (nested `<DL>` tags)
- AliasPearl targets (links to `pearltrees.com` without `TARGET="_BLANK"`)
- External URLs (PagePearls, but no pearl_uri)

### Hybrid Hierarchy Format

The HTML export parser builds entries with both title paths and ID patterns:

```json
{
  "type": "Tree",
  "raw_title": "Academic disciplines",
  "title_path": ["s243a", "Main topic classifications", "Academic disciplines"],
  "id_path_pattern": "/90289478/69207017/53344165",
  "target_text": "/90289478/69207017/53344165\n- s243a\n  - Main topic classifications\n    - Academic disciplines"
}
```

When some IDs are unknown, regex patterns are used:
```
/90289478/[^/]+/53344165
```

This preserves semantic meaning for embeddings even without complete ID information.

### Repair Workflow

```
1. Export RDF (primary data)
2. Generate JSONL from RDF
3. Identify missing items (trees with 0 children)
4. Export Zip for specific trees → repair_from_zip_export.py
5. Export HTML for hierarchy info → repair_from_html_export.py
6. Merge repairs into JSONL
7. Regenerate mindmaps
```

### File Locations

| Type | Location | Naming Convention |
|------|----------|-------------------|
| Zip exports | `context/PT/Zip_Exports/` | `{Tree Name}/` (unzipped) |
| HTML exports | `context/PT/` | `pearltrees_export_{account}_{date}.html` |

---

## Related Documentation

- `reports/GENERATION_LOG.md` - Log of generated files
- `docs/MINDMAP_GENERATOR.md` - Generator script details
- `reports/PR_declarative_layout_system.md` - Layout system proposal
- `playbooks/rdf_data_source_playbook.md` - RDF processing with Prolog
