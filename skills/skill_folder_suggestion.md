# Skill: Folder Suggestion

Suggest the best folder for a mindmap using semantic similarity and Procrustes projection.

## When to Use

- User asks "where should this mindmap go?"
- User wants to check if mindmaps are in the right folder
- User needs to organize misplaced mindmaps

## Quick Start

```bash
python3 scripts/mindmap/suggest_folder.py --tree-id "TREE_ID" --verbose
```

## Commands

### Suggest Folder by Tree ID
```bash
python3 scripts/mindmap/suggest_folder.py --tree-id "TREE_ID"
```

### Suggest Folder by Title
```bash
python3 scripts/mindmap/suggest_folder.py --title "MINDMAP_TITLE"
```

### Check Folder for Misplacements
```bash
python3 scripts/mindmap/suggest_folder.py \
  --check-folder "FOLDER_PATH" \
  --threshold 0.5
```

### JSON Output
```bash
python3 scripts/mindmap/suggest_folder.py --tree-id "TREE_ID" --json
```

### Get All Options
```bash
python3 scripts/mindmap/suggest_folder.py --help
```

## Prerequisites

Build folder projections first:

```bash
python3 scripts/mindmap/build_folder_projections.py \
  --embeddings "EMBEDDINGS_FILE" \
  --index "INDEX_FILE" \
  --output "PROJECTIONS_DB"
```

## Output Interpretation

| Probability | Meaning |
|-------------|---------|
| >50% | Strong match |
| 25-50% | Good match, minor ambiguity |
| <25% | Ambiguous, multiple valid options |

## Example Output

```
Tree ID: 2492215
Current folder: Hacktivism

Suggested folders:
  1. Hacktivism: 51.2% (fit=0.9667) <-- current
  2. eyes-symbols-history-s243a: 20.3%
  3. Online_Hacktivists_groups: 18.6%
```

## Related

**Skills:**
- `skill_mst_folder_grouping.md` - Generate folder structure
- `skill_mindmap_linking.md` - Link mindmaps to Pearltrees
- `skill_train_model.md` - Train projection models
- `skill_semantic_inference.md` - General inference concepts

**Documentation:**
- `scripts/mindmap/README.md` - Full documentation (see "Folder Organization Tools" section)
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format specification

**Education (in `education/` subfolder):**
- `book-13-semantic-search/05_semantic_playbook.md` - Best practices
- `book-13-semantic-search/07_density_scoring.md` - Scoring methods
- `book-13-semantic-search/08_advanced_federation.md` - Federation architecture
- `book-13-semantic-search/15_zero_shot_path_mapping.md` - Zero-shot inference
- `book-14-ai-training/03_lda_projection.md` - Projection methods

**Code:**
- `scripts/mindmap/suggest_folder.py` - Main suggestion script
- `scripts/mindmap/build_folder_projections.py` - Build W matrices per folder
