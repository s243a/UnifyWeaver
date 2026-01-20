# Skill: Bookmark Filing

This skill enables filing bookmarks into the Pearltrees folder hierarchy using semantic search and optional LLM selection.

## Overview

Uses a federated projection model (93% Recall@1) to find semantically similar folders, then optionally calls an LLM for final selection based on hierarchical context.

## Commands

### Get Semantic Candidates (Tree View)
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated_single.pkl \
  --query "BOOKMARK_TITLE" \
  --top-k 10 --tree
```

### Get Semantic Candidates (JSON)
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated_single.pkl \
  --query "BOOKMARK_TITLE" \
  --top-k 10 --json
```

### Filter by Account
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated_single.pkl \
  --query "BOOKMARK_TITLE" \
  --account s243a_groups \
  --top-k 10 --tree
```

### Use Account-Specific Model
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated_s243a_groups.pkl \
  --query "BOOKMARK_TITLE" \
  --top-k 10 --tree
```

### Full Filing Assistant (with LLM)
```bash
python3 scripts/bookmark_filing_assistant.py \
  --bookmark "BOOKMARK_TITLE" \
  --url "OPTIONAL_URL" \
  --provider claude \
  --top-k 10
```

## LLM Providers

| Provider | Flag | Notes |
|----------|------|-------|
| Claude CLI | `--provider claude` | Cheapest with subscription |
| Gemini CLI | `--provider gemini` | Gemini headless |
| OpenAI API | `--provider openai` | Requires OPENAI_API_KEY |
| Anthropic API | `--provider anthropic` | Requires ANTHROPIC_API_KEY |
| Local Ollama | `--provider ollama` | Requires ollama installed |

## Model Information

| Model | Account | Clusters | Use Case |
|-------|---------|----------|----------|
| `pearltrees_federated_single.pkl` | All | 51 | General search |
| `pearltrees_federated_s243a.pkl` | s243a | 275 | s243a-focused |
| `pearltrees_federated_s243a_groups.pkl` | s243a_groups | 48 | Cross-account migration |

- **Data Path**: `reports/pearltrees_targets_full_multi_account.jsonl`
- **Accuracy**: 93% Recall@1 on training data, 99% Recall@5
- **Format**: See `docs/design/FEDERATED_MODEL_FORMAT.md`

## Tree Output Example

```
└── s243a
    └── s243a_groups @s243a_groups  ← Account jump
        └── s243a
            └── STEM
                └── AI & Machine Learning ★ #10 [0.280]
                    ├── Machine Learning ★ #9 [0.284]
                    │   └── Deep Learning ★ #2 [0.328]
                    └── Neural network architectures ★ #1 [0.328]
```

Stars (★) mark candidate folders with rank and score.

## Filing Decision Guidelines

When recommending a folder:
1. **Match specificity** - File in the most specific matching folder
2. **Consider hierarchy** - A parent folder may be better for general topics
3. **Check duplicates** - Similar-named folders may exist at different depths
4. **Account matters** - Group folders (s243a_groups) are shared

## MCP Integration

An MCP server is available at `scripts/mcp_bookmark_filing_server.py` exposing:
- `get_filing_candidates` - Get semantic search candidates
- `file_bookmark` - Get LLM recommendation for filing

## Related

**Parent Skill:**
- `skill_bookmark_tools.md` - Bookmark tools sub-master

**Sibling Skills:**
- `skill_folder_suggestion.md` - Suggest folders for items
- `skill_mst_folder_grouping.md` - Build folder hierarchies

**Other Skills:**
- `skill_train_model.md` - Train the federated projection model
- `skill_semantic_inference.md` - General inference concepts
- `skill_mindmap_linking.md` - Link mindmaps to Pearltrees (same projection model)

**Documentation:**
- `docs/design/FEDERATED_MODEL_FORMAT.md` - Model format specification
- `docs/QUICKSTART_MINDMAP_LINKING.md` - End-to-end workflow

**Education (in `education/` subfolder):**
- `book-13-semantic-search/16_bookmark_filing.md` - Bookmark filing workflow
- `book-13-semantic-search/01_introduction.md` - Semantic search overview
- `book-13-semantic-search/05_semantic_playbook.md` - Best practices
- `book-13-semantic-search/07_density_scoring.md` - Scoring methods
- `book-13-semantic-search/08_advanced_federation.md` - Federation architecture
- `book-14-ai-training/05_training_pipeline.md` - Model training

**Code:**
- `scripts/infer_pearltrees_federated.py` - Semantic search inference
- `scripts/bookmark_filing_assistant.py` - Full filing assistant with LLM
- `scripts/generate_account_training_data.py` - Filter JSONL by account
- `scripts/import_pearltrees_to_db.py` - Import RDF to SQLite
- `scripts/mcp_bookmark_filing_server.py` - MCP server
- `scripts/train_pearltrees_federated.py` - Model training

## Database Import

To enable showing existing bookmarks in candidate folders:

```bash
# Import your Pearltrees RDF exports
python3 scripts/import_pearltrees_to_db.py \
  --account s243a data/s243a.rdf \
  --account s243a_groups data/s243a_groups.rdf \
  --output pearltrees.db
```

Then use with filing assistant:

```bash
python3 scripts/bookmark_filing_assistant.py \
  --bookmark "Machine learning tutorial" \
  --db pearltrees.db
```

The LLM will see existing folder contents:
```
## Existing Bookmarks in Candidate Folders

**Deep Learning** (#2): "PyTorch tutorial", "Keras getting started"
**Machine Learning** (#3): "Scikit-learn guide", "ML overview"
```
