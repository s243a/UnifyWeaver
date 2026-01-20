# Skill: Mindmap & Bookmark Tools (Master)

Master skill for managing SimpleMind mindmaps, bookmark filing, folder organization, and Pearltrees integration.

## When to Use

- User asks "how do I work with mindmaps?"
- User wants to file bookmarks into Pearltrees
- User needs to organize collections into folder hierarchies
- User asks about linking mindmaps to Pearltrees
- User wants to visualize embedding spaces and clusters

## Skill Hierarchy

```
skill_mindmap_bookmark_tools.md (this file - MASTER)
│
├── skill_mindmap_tools.md (sub-master)
│   ├── skill_mindmap_organization.md
│   │   └── (mst_folder_grouping, folder_suggestion, hierarchy_objective)
│   ├── skill_mindmap_indexing.md
│   │   └── (mindmap_index, mindmap_rename)
│   └── skill_mindmap_references.md
│       ├── skill_mindmap_linking.md - Enrich mindmaps with Pearltrees links
│       └── skill_mindmap_cross_links.md
│
├── skill_bookmark_tools.md (sub-master)
│   ├── skill_bookmark_filing.md - File bookmarks into Pearltrees
│   ├── skill_folder_suggestion.md - Suggest folders for items
│   └── skill_mst_folder_grouping.md - Build folder hierarchies via MST
│
└── skill_density_explorer.md - Visualize embeddings, trees, density
```

## Overview

This master skill covers two complementary domains:

### Mindmap Management

Work with SimpleMind (.smmx) files:
- **Organization** - Semantically cluster mindmaps into folders
- **Indexing** - Track locations, handle renames
- **References** - Link mindmaps to each other and to Pearltrees

### Bookmark & Filing

Organize content into Pearltrees hierarchies:
- **Bookmark Filing** - Find the best folder for new bookmarks
- **Folder Suggestion** - Where should this item go?
- **MST Grouping** - Create folder structures from flat collections

### Shared Foundation

Both domains share:
- **Federated Projection Models** - Trained on Pearltrees hierarchy
- **Semantic Embeddings** - nomic, MiniLM for similarity
- **Density Explorer** - Visualize clusters and trees

## Quick Reference

### Mindmap Workflows

```bash
# Generate mindmaps from Pearltrees data
python3 scripts/generate_mindmap.py --data data.jsonl --output-dir output/

# Build index for cross-referencing
python3 scripts/mindmap/build_index.py output/ -o index.json

# Add cross-links between mindmaps
python3 scripts/mindmap/add_relative_links.py output/*.smmx --index index.json

# Enrich mindmap with Pearltrees links
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap output/physics.smmx \
  --trees reports/pearltrees_targets.jsonl \
  --projection-model models/federated.pkl \
  --output output/physics_enriched.smmx
```

### Bookmark Workflows

```bash
# Get folder suggestions for a bookmark
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated.pkl \
  --query "Machine Learning Tutorial" \
  --top-k 10 --tree

# File bookmark with LLM assistance
python3 scripts/bookmark_filing_assistant.py \
  --bookmark "ML Tutorial" \
  --url "https://example.com/ml" \
  --provider claude

# Organize flat collection into folders
python3 scripts/mindmap/mst_folder_grouping.py \
  --target-size 10 --max-depth 5 \
  -o folder_structure.json
```

### Visualization

```bash
# Start density explorer
python tools/density_explorer/flask_api.py --port 5000
cd tools/density_explorer/vue && npm run dev
# Open http://localhost:5173
```

## Model Integration

Both mindmap and bookmark tools use the same trained projection models:

| Model | Embedder | Use Case |
|-------|----------|----------|
| `pearltrees_federated_nomic.pkl` | Nomic (768D) | Best quality |
| `pearltrees_federated_single.pkl` | MiniLM (384D) | Faster inference |

Train models with `skill_train_model.md` from `skill_ml_tools.md`.

## Typical End-to-End Workflow

```bash
# 1. Train projection model (see skill_ml_tools.md)
python3 scripts/train_pearltrees_federated.py \
  reports/pearltrees_targets.jsonl \
  models/federated.pkl

# 2. Generate mindmaps from source data
python3 scripts/generate_mindmap.py --data source.jsonl --output-dir output/

# 3. Build index
python3 scripts/mindmap/build_index.py output/ -o index.json

# 4. Enrich with Pearltrees links
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap output/*.smmx \
  --projection-model models/federated.pkl

# 5. Organize into folders
python3 scripts/mindmap/mst_folder_grouping.py \
  --embeddings embeddings.npz \
  -o folder_structure.json

# 6. Visualize and explore
python tools/density_explorer/flask_api.py
```

## Child Skills

### Mindmap Tools (Sub-Master)
- `skill_mindmap_tools.md` - SimpleMind file management
  - `skill_mindmap_organization.md` - Folder organization
  - `skill_mindmap_indexing.md` - Index and rename
  - `skill_mindmap_references.md` - Links and cross-references

### Bookmark Tools (Sub-Master)
- `skill_bookmark_tools.md` - Filing and organization
  - `skill_bookmark_filing.md` - File to Pearltrees folders
  - `skill_folder_suggestion.md` - Suggest folders for items
  - `skill_mst_folder_grouping.md` - MST-based hierarchy creation

### Visualization
- `skill_density_explorer.md` - Embedding space visualization

## Related

**Sibling Masters:**
- `skill_server_tools.md` - Backend services, APIs
- `skill_gui_tools.md` - Frontend/GUI generation
- `skill_data_tools.md` - ML foundation (embeddings, training, inference)

**Key Dependencies:**
- `skill_ml_tools.md` - Provides trained projection models
- `skill_embedding_models.md` - Model selection for embeddings

**Documentation:**
- `docs/QUICKSTART_MINDMAP_LINKING.md` - End-to-end workflow
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format spec
- `scripts/mindmap/README.md` - Tool documentation

**Code:**
- `scripts/mindmap/` - Mindmap tools
- `scripts/bookmark_filing_assistant.py` - Bookmark filing
- `scripts/infer_pearltrees_federated.py` - Inference
- `tools/density_explorer/` - Visualization
