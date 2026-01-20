# Skill: Bookmark Tools (Sub-Master)

Filing bookmarks into Pearltrees, suggesting folders for items, and building folder hierarchies from collections.

## When to Use

- User asks "where should I file this bookmark?"
- User wants to organize items into folders
- User asks about semantic folder suggestion
- User needs to create folder hierarchies from flat collections
- User wants to use MST-based organization

## Skill Hierarchy

```
skill_mindmap_bookmark_tools.md (parent)
└── skill_bookmark_tools.md (this file)
    ├── skill_bookmark_filing.md - File bookmarks into Pearltrees
    ├── skill_folder_suggestion.md - Suggest folders for items
    └── skill_mst_folder_grouping.md - Build folder hierarchies via MST
```

## Overview

Bookmark tools help organize content into semantic hierarchies:

| Skill | Input | Output | Use Case |
|-------|-------|--------|----------|
| `bookmark_filing` | URL + title | Pearltrees folder path | Save new bookmark |
| `folder_suggestion` | Single item | Best folder in hierarchy | Place item |
| `mst_folder_grouping` | Flat collection | Folder hierarchy | Organize collection |

All tools use **federated projection models** trained on Pearltrees hierarchy data.

## Quick Start

### File a Bookmark

```bash
# Get semantic candidates
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated.pkl \
  --query "Machine Learning Tutorial" \
  --top-k 10 --tree

# With LLM-assisted selection
python3 scripts/bookmark_filing_assistant.py \
  --bookmark "Machine Learning Tutorial" \
  --url "https://example.com/ml-tutorial" \
  --provider claude
```

### Suggest Folder for Item

```bash
python3 scripts/mindmap/suggest_folder.py \
  --tree-id "TREE_ID" \
  --verbose
```

### Build Folder Hierarchy

```bash
python3 scripts/mindmap/mst_folder_grouping.py \
  --trees-only \
  --target-size 10 \
  --max-depth 5 \
  -o folder_structure.json
```

## Individual Skills

### Bookmark Filing (`skill_bookmark_filing.md`)

File bookmarks into Pearltrees using semantic search and optional LLM selection.

**Features:**
- 93% Recall@1 with federated projection
- Multi-account support (s243a, s243a_groups)
- LLM providers: Claude, Gemini, OpenAI, Anthropic, Ollama
- MCP server integration

```bash
# Tree view of candidates
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated.pkl \
  --query "BOOKMARK_TITLE" \
  --top-k 10 --tree

# Filter by account
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated.pkl \
  --query "BOOKMARK_TITLE" \
  --account s243a_groups \
  --top-k 10
```

### Folder Suggestion (`skill_folder_suggestion.md`)

Suggest the best folder for an item using semantic similarity and Procrustes projection.

**Features:**
- Probability-based ranking
- Check folders for misplacements
- JSON output for automation

```bash
# Suggest by tree ID
python3 scripts/mindmap/suggest_folder.py --tree-id "TREE_ID"

# Suggest by title
python3 scripts/mindmap/suggest_folder.py --title "ITEM_TITLE"

# Check folder for misplacements
python3 scripts/mindmap/suggest_folder.py \
  --check-folder "FOLDER_PATH" \
  --threshold 0.5
```

### MST Folder Grouping (`skill_mst_folder_grouping.md`)

Organize collections into semantically coherent folder hierarchies using Minimum Spanning Tree partitioning.

**Features:**
- Multiple tree sources: MST, curated, hybrid
- Configurable target size and depth
- Multilevel or bisection clustering

```bash
# Standard MST grouping
python3 scripts/mindmap/mst_folder_grouping.py \
  --trees-only \
  --target-size 10 \
  --max-depth 5 \
  -o folder_structure.json

# Hybrid mode (curated + orphan attachment)
python3 scripts/mindmap/mst_folder_grouping.py \
  --tree-source hybrid \
  --embed-blend 0.3 \
  --target-size 10 \
  -o folder_structure.json
```

## Model Information

| Model | Account | Clusters | Use Case |
|-------|---------|----------|----------|
| `pearltrees_federated_single.pkl` | All | 51 | General search |
| `pearltrees_federated_s243a.pkl` | s243a | 275 | s243a-focused |
| `pearltrees_federated_s243a_groups.pkl` | s243a_groups | 48 | Cross-account |

Models trained with `skill_train_model.md` from `skill_ml_tools.md`.

## Integration with Mindmap Tools

Bookmark tools complement mindmap tools:

| Tool | Purpose | Shared Model |
|------|---------|--------------|
| `bookmark_filing` | File NEW bookmarks | Yes |
| `mindmap_linking` | Enrich EXISTING mindmaps | Yes |
| `folder_suggestion` | Place items in hierarchy | Yes |
| `mst_folder_grouping` | Create hierarchy structure | Yes |

Both use the same federated projection models and semantic matching approach.

## Common Workflows

### Organize New Collection

```bash
# 1. Generate embeddings
python3 scripts/generate_embeddings.py \
  --input items.jsonl \
  --output embeddings.npz

# 2. Build folder structure
python3 scripts/mindmap/mst_folder_grouping.py \
  --embeddings embeddings.npz \
  --target-size 10 \
  -o folder_structure.json

# 3. Evaluate quality
python3 scripts/mindmap/hierarchy_objective.py \
  --tree folder_structure.json \
  --embeddings embeddings.npz
```

### Batch Filing

```bash
# File multiple bookmarks
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated.pkl \
  --input bookmarks.txt \
  --output suggestions.jsonl
```

## Child Skills

- `skill_bookmark_filing.md` - Semantic bookmark filing with LLM
- `skill_folder_suggestion.md` - Folder suggestion for items
- `skill_mst_folder_grouping.md` - MST-based hierarchy creation

## Related

**Parent Skill:**
- `skill_mindmap_bookmark_tools.md` - Master for mindmaps and bookmarks

**Sibling Skills:**
- `skill_mindmap_tools.md` - SimpleMind file management
- `skill_density_explorer.md` - Visualization

**Dependencies:**
- `skill_ml_tools.md` - Provides trained projection models
- `skill_hierarchy_objective.md` - Evaluates hierarchy quality

**Code:**
- `scripts/infer_pearltrees_federated.py` - Semantic search
- `scripts/bookmark_filing_assistant.py` - LLM-assisted filing
- `scripts/mindmap/suggest_folder.py` - Folder suggestion
- `scripts/mindmap/mst_folder_grouping.py` - MST partitioning
