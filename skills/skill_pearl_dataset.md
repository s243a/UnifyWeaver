# Skill: Pearl Dataset Generation

Generate semantic search training datasets from Pearltrees RDF exports, including trees (folders), pearls (bookmarks), and cross-account references.

## When to Use

- User asks "how do I create training data from Pearltrees?"
- User wants to generate targets for semantic search
- User needs to export Pearltrees hierarchy for ML
- User asks about RDF parsing or materialized paths

## Overview

Pearl dataset generation transforms Pearltrees RDF exports into structured JSONL training data suitable for:

- Semantic search training
- Folder suggestion models
- Bookmark filing assistants
- Hierarchy visualization

## Quick Start

```bash
# Generate targets from single RDF
python scripts/generate_pearl_dataset.py \
  --rdf data/s243a.rdf \
  --output reports/pearltrees_targets.jsonl

# Multi-account with cross-references
python scripts/generate_pearl_dataset.py \
  --rdf data/s243a.rdf data/s243a_groups.rdf \
  --output reports/pearltrees_targets_multi.jsonl \
  --cross-account
```

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--rdf` | - | RDF file(s) to process |
| `--output` | - | Output JSONL path |
| `--query-style` | raw | Query format (raw, locate, file, similar) |
| `--trees-only` | false | Only include tree nodes |
| `--pearls-only` | false | Only include pearl items |
| `--cross-account` | false | Enable cross-account linking |
| `--path-filter` | - | Filter by path prefix |
| `--account` | - | Filter by account name |

## Query Styles

Different query formats for different use cases:

| Style | Format | Use Case |
|-------|--------|----------|
| `raw` | Just the title | General embedding |
| `locate` | `locate_node("title")` | Prolog-style queries |
| `locate_object` | `locate_object(tree, "title")` | Typed queries |
| `file` | `file_bookmark("url", "title")` | Bookmark filing |
| `similar` | `find_similar("title")` | Similarity search |
| `browse` | `browse_folder("title")` | Navigation queries |

```bash
# Locate style for Prolog queries
python scripts/generate_pearl_dataset.py \
  --rdf data/s243a.rdf \
  --query-style locate \
  --output reports/targets_locate.jsonl

# File style for bookmark filing
python scripts/generate_pearl_dataset.py \
  --rdf data/s243a.rdf \
  --query-style file \
  --pearls-only \
  --output reports/targets_file.jsonl
```

## Output Format

### Tree Entry

```json
{
  "data_type": "tree",
  "type": "Tree",
  "uri": "http://www.pearltrees.com/s243a/science/id12345",
  "raw_title": "Science",
  "target_text": "s243a > Root > Science",
  "query": "locate_node(\"Science\")",
  "cluster_id": "http://www.pearltrees.com/s243a/root/id1",
  "account": "s243a"
}
```

### Pearl Entry

```json
{
  "data_type": "pearl",
  "type": "PagePearl",
  "uri": "http://www.pearltrees.com/s243a/item/id67890",
  "raw_title": "Quantum Computing Basics",
  "external_url": "https://example.com/quantum",
  "target_text": "s243a > Root > Science > Physics > Quantum Computing Basics",
  "query": "file_bookmark(\"https://example.com/quantum\", \"Quantum Computing Basics\")",
  "cluster_id": "http://www.pearltrees.com/s243a/physics/id12346",
  "account": "s243a"
}
```

### RefPearl Entry (Cross-Reference)

```json
{
  "data_type": "pearl",
  "type": "RefPearl",
  "uri": "http://www.pearltrees.com/s243a/item/id99999",
  "raw_title": "Physics (ref)",
  "alias_target_uri": "http://www.pearltrees.com/other/physics/id55555",
  "target_text": "s243a > Root > References > Physics (ref) → @other_account",
  "cluster_id": "http://www.pearltrees.com/s243a/references/id44444",
  "account": "s243a"
}
```

## Materialized Paths

The `target_text` field contains the materialized path - the full hierarchy from root to item:

```
account > Root > Level1 > Level2 > ... > Item Title
```

For cross-account references, the path includes account boundary notation:

```
s243a > Root > Shared > Physics → @other_account
```

## Multi-Account Processing

### Account Filtering

Extract single account from multi-account dataset:

```bash
python scripts/generate_account_training_data.py \
  --input reports/pearltrees_targets_multi.jsonl \
  --account s243a \
  --output reports/pearltrees_targets_s243a.jsonl
```

### Cross-Account Linking

Enable to track references between accounts:

```bash
python scripts/generate_pearl_dataset.py \
  --rdf data/s243a.rdf data/friends.rdf \
  --cross-account \
  --output reports/targets_cross.jsonl
```

## Pearl Types

| Type | Description | Has URL |
|------|-------------|---------|
| `Tree` | Folder/collection | No |
| `PagePearl` | Web bookmark | Yes |
| `NotePearl` | Text note | No |
| `RefPearl` | Reference to another tree | No (has target URI) |
| `AliasPearl` | Alias to another item | No (has target URI) |
| `SectionPearl` | Section within tree | No |

## Synthetic RefPearl Generation

Generate RefPearls from parent-child relationships:

```bash
python scripts/generate_tree_refpearls.py \
  --input reports/pearltrees_targets.jsonl \
  --output reports/pearltrees_targets_with_refs.jsonl
```

This infers relationships from `cluster_id` fields and creates RefPearl records marked with `_source: "hierarchy_inference"`.

## Workflow Example

Complete dataset generation workflow:

```bash
# 1. Export RDF from Pearltrees (manual step)
# Download from Pearltrees settings > Export

# 2. Generate base dataset
python scripts/generate_pearl_dataset.py \
  --rdf data/export.rdf \
  --query-style locate \
  --output reports/base_targets.jsonl

# 3. Add synthetic references
python scripts/generate_tree_refpearls.py \
  --input reports/base_targets.jsonl \
  --output reports/targets_with_refs.jsonl

# 4. Generate embeddings
python scripts/generate_embeddings.py \
  --input reports/targets_with_refs.jsonl \
  --output datasets/embeddings.npz \
  --model nomic-ai/nomic-embed-text-v1.5

# 5. Train projection model
python scripts/train_pearltrees_federated.py \
  reports/targets_with_refs.jsonl \
  models/federated.pkl
```

## Path Filtering

Extract subset by path prefix:

```bash
# Only Science subtree
python scripts/generate_pearl_dataset.py \
  --rdf data/s243a.rdf \
  --path-filter "Science" \
  --output reports/science_targets.jsonl
```

## Troubleshooting

### "RDF parse error"

- Ensure RDF is valid XML (Pearltrees export)
- Check encoding (should be UTF-8)
- Try opening in browser to validate XML

### Missing pearls

- Check `--pearls-only` and `--trees-only` flags
- RefPearls/AliasPearls may be filtered by default
- Use `--include-refs` if available

### Duplicate entries

- Multi-account processing may include duplicates
- Use `cluster_id` to deduplicate if needed

## Related

**Parent Skill:**
- `skill_synthetic_data.md` - Synthetic data sub-master

**Sibling Skills:**
- `skill_qa_generation.md` - Generate Q&A pairs
- `skill_answer_tailoring.md` - Reword answers

**Downstream Skills:**
- `skill_train_model.md` - Train on generated data
- `skill_semantic_inference.md` - Use trained models
- `skill_bookmark_filing.md` - File bookmarks

**Code:**
- `scripts/generate_pearl_dataset.py` - Main generator
- `scripts/generate_tree_refpearls.py` - Synthetic RefPearls
- `scripts/generate_account_training_data.py` - Account filtering
- `scripts/pearltrees_multi_account_generator.py` - Multi-account parser
