# Skill: MST Folder Grouping

Organize mindmaps into semantically coherent folder hierarchies using Minimum Spanning Tree partitioning.

## When to Use

- User wants to organize mindmaps by semantic similarity
- User asks to "cluster", "group", or "organize" mindmaps
- User needs folder structure for a collection

## Quick Start

```bash
python3 scripts/mindmap/mst_folder_grouping.py \
  --trees-only --target-size 10 --max-depth 5 \
  -o output/mst_folder_structure.json --verbose
```

## Commands

### Generate Folder Structure
```bash
python3 scripts/mindmap/mst_folder_grouping.py \
  --trees-only \
  --target-size 10 \
  --max-depth 5 \
  -o "OUTPUT_JSON"
```

### Test on Subset
```bash
python3 scripts/mindmap/mst_folder_grouping.py \
  --subset physics \
  --target-size 8 \
  --max-depth 3 \
  --verbose
```

### Use Curated Hierarchy
```bash
python3 scripts/mindmap/mst_folder_grouping.py \
  --tree-source curated \
  --target-size 10 \
  -o "OUTPUT_JSON"
```

### Hybrid Mode (Curated + Orphan Attachment)
```bash
python3 scripts/mindmap/mst_folder_grouping.py \
  --tree-source hybrid \
  --embed-blend 0.3 \
  --target-size 10 \
  -o "OUTPUT_JSON"
```

### Get All Options
```bash
python3 scripts/mindmap/mst_folder_grouping.py --help
```

## Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target-size` | 8 | Target items per folder |
| `--max-depth` | 4 | Maximum folder depth |
| `--tree-source` | mst | `mst`, `curated`, or `hybrid` |
| `--cluster-method` | multilevel | `multilevel` or `bisection` |

## Generate Mindmaps from Structure

After generating folder structure:

```bash
python3 scripts/mindmap/generate_mst_mindmaps.py \
  --mst-structure output/mst_folder_structure.json \
  --output output/mst_mindmaps/ \
  --root-name "My_Collection"
```

## Related

**Parent Skill:**
- `skill_bookmark_tools.md` - Bookmark tools sub-master

**Sibling Skills:**
- `skill_bookmark_filing.md` - File bookmarks to Pearltrees
- `skill_folder_suggestion.md` - Suggest folders for individual items

**Other Skills:**
- `skill_mindmap_cross_links.md` - Add cross-links between mindmaps
- `skill_train_model.md` - Train models for semantic grouping

**Documentation:**
- `scripts/mindmap/README.md` - Full documentation (see "MST Semantic Clustering" section)
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format specification

**Education (in `education/` subfolder):**
- `book-13-semantic-search/07_density_scoring.md` - Scoring and clustering
- `book-13-semantic-search/08_advanced_federation.md` - Federated clustering
- `book-13-semantic-search/14_scale_free_networks.md` - Network analysis
- `book-14-ai-training/05_training_pipeline.md` - Training pipeline concepts

**Code:**
- `scripts/mindmap/mst_folder_grouping.py` - Main MST partitioning script
- `scripts/mindmap/generate_mst_mindmaps.py` - Generate mindmaps from MST structure
- `scripts/mindmap/hierarchy_objective.py` - J = D/(1+H) objective function
