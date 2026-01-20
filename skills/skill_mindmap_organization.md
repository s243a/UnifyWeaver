# Skill: Mindmap Organization

Sub-master skill for semantically organizing mindmaps into folder hierarchies using embeddings and clustering.

## When to Use

- User asks "how do I organize my mindmaps?"
- User wants to create folder structures from flat collections
- User asks about semantic clustering or grouping
- User needs to evaluate hierarchy quality
- User wants to suggest where new items should go

## Overview

Organization transforms flat collections of mindmaps into meaningful hierarchical folder structures using:

1. **Semantic Embeddings** - Vector representations capturing meaning
2. **MST Partitioning** - Minimum Spanning Tree for natural clustering
3. **Hierarchy Evaluation** - J = D/(1+H) objective function
4. **Folder Suggestion** - Where to place new items

## Individual Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `skill_mst_folder_grouping.md` | Create folder hierarchies via MST | Organizing collections |
| `skill_folder_suggestion.md` | Suggest folder for new items | Adding to existing structure |
| `skill_hierarchy_objective.md` | Evaluate hierarchy quality | Tuning or comparing structures |

## The Organization Pipeline

### 1. Prepare Embeddings

Before organizing, you need embeddings for your mindmaps:

```bash
# Generate embeddings (nomic recommended for better clustering)
python3 scripts/generate_embeddings.py \
  --input mindmaps/*.smmx \
  --output embeddings.npz \
  --model nomic-ai/nomic-embed-text-v1.5
```

### 2. Build Folder Structure

Use MST partitioning to create semantically coherent groups:

```bash
python3 scripts/mindmap/mst_folder_grouping.py \
  --embeddings embeddings.npz \
  --target-size 10 \
  --max-depth 5 \
  -o folder_structure.json
```

**Key parameters:**
- `--target-size` - Ideal number of items per folder
- `--max-depth` - Maximum folder nesting depth
- `--trees-only` - Only include tree-type items (not individual pearls)

See `skill_mst_folder_grouping.md` for full details.

### 3. Evaluate Quality

Check if the hierarchy is well-structured:

```bash
python3 scripts/mindmap/hierarchy_objective.py \
  --tree folder_structure.json \
  --embeddings embeddings.npy
```

**Output:**
```
Hierarchy Statistics:
  Objective J: 0.0955  (lower = better)
  Semantic Distance D: 0.2341
  Entropy Gain H: 1.4532
```

See `skill_hierarchy_objective.md` for interpreting results.

### 4. Suggest Folders for New Items

When adding new mindmaps, find where they fit:

```bash
python3 scripts/mindmap/suggest_folder.py \
  --item "new_mindmap.smmx" \
  --structure folder_structure.json \
  --embeddings embeddings.npz
```

See `skill_folder_suggestion.md` for details.

## Choosing Organization Strategies

### MST vs J-Guided Trees

| Approach | Best For | Trade-offs |
|----------|----------|------------|
| **MST** | Fast initial organization | May produce long chains |
| **J-Guided** | Balanced hierarchies | Slower, better structure |

```bash
# Standard MST
python3 scripts/mindmap/mst_folder_grouping.py --tree-type mst ...

# J-Guided (optimizes hierarchy quality)
python3 scripts/mindmap/mst_folder_grouping.py --tree-type j-guided ...
```

### Flat vs Deep Hierarchies

| Structure | Parameters | Use Case |
|-----------|------------|----------|
| **Flat** | `--max-depth 2 --target-size 20` | Few large folders |
| **Deep** | `--max-depth 5 --target-size 8` | Many small folders |
| **Balanced** | `--max-depth 4 --target-size 12` | General purpose |

### Curated vs Automatic

```bash
# Fully automatic from embeddings
python3 scripts/mindmap/mst_folder_grouping.py --tree-source mst ...

# Start from curated hierarchy, attach orphans
python3 scripts/mindmap/mst_folder_grouping.py --tree-source hybrid ...

# Use existing curated structure only
python3 scripts/mindmap/mst_folder_grouping.py --tree-source curated ...
```

## Workflow Example

Complete organization workflow:

```bash
# 1. Generate embeddings
python3 scripts/generate_embeddings.py \
  --input output/mindmaps/*.smmx \
  --output datasets/mindmap_embeddings.npz

# 2. Create folder structure
python3 scripts/mindmap/mst_folder_grouping.py \
  --embeddings datasets/mindmap_embeddings.npz \
  --target-size 10 \
  --max-depth 4 \
  -o output/folder_structure.json

# 3. Evaluate quality
python3 scripts/mindmap/hierarchy_objective.py \
  --tree output/folder_structure.json \
  --embeddings datasets/mindmap_embeddings.npz

# 4. If J > 0.3, try J-guided tree
python3 scripts/mindmap/mst_folder_grouping.py \
  --embeddings datasets/mindmap_embeddings.npz \
  --tree-type j-guided \
  --target-size 10 \
  -o output/folder_structure_jguided.json

# 5. Move files to folders
python3 scripts/mindmap/apply_folder_structure.py \
  --structure output/folder_structure.json \
  --source output/mindmaps/ \
  --dest output/organized/
```

## Related

**Parent Skill:**
- `skill_mindmap_tools.md` - Master mindmap skill

**Individual Skills:**
- `skill_mst_folder_grouping.md` - MST-based folder organization
- `skill_folder_suggestion.md` - Suggest folders for items
- `skill_hierarchy_objective.md` - Evaluate hierarchy quality (J = D/(1+H))

**Related Skills:**
- `skill_embedding_models.md` - Choosing embedding models
- `skill_density_explorer.md` - Visualize clustering

**Code:**
- `scripts/mindmap/mst_folder_grouping.py` - Main organization tool
- `scripts/mindmap/suggest_folder.py` - Folder suggestion
- `scripts/mindmap/hierarchy_objective.py` - Quality evaluation
