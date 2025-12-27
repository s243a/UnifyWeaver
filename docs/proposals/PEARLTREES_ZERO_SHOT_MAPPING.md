# Pearltrees Zero-Shot Mapping via Minimal Transformation

**Status:** Implemented
**Date:** 2025-12-27
**Updated:** Multi-account support added

## Overview

This proposal outlines the implementation of a "Zero-Shot" mapping for Pearltrees data. The goal is to map a user query (e.g., "Physics Notes") to a structured "Materialized Path" representation of the target item.

By using the **Minimal Transformation (Procrustes)** approach, we learn rotation and scaling transformations that align the "Query Embedding Space" (Titles) with the "Path Embedding Space" (Hierarchy + Titles).

## The Target Representation

The target "Answer" string encodes both the hierarchical identity (IDs) and the semantic content (Titles) of the path.

**Single Account Format:**
```text
/root_id/child_id/leaf_id
- Root Title
  - Child Title
    - Leaf Title
```

**Multi-Account Format (with cross-account boundary):**
```text
/root_id/child_id/other_account_id@other_account/sub_id
- Root Title
  - Child Title
    - Other Account Title @other_account
      - Sub Title
```

## Scripts

### 1. Single-Account Generator
Parses a single Pearltrees RDF export.

**Script:** `scripts/pearltrees_target_generator.py`

```bash
python scripts/pearltrees_target_generator.py \
  path/to/export.rdf \
  reports/targets.jsonl \
  --query-template "locate_tree('{title}')" \
  --item-type tree \
  --filter-path "Physics"
```

### 2. Multi-Account Generator
Parses multiple Pearltrees RDF exports with cross-account linking.

**Script:** `scripts/pearltrees_multi_account_generator.py`

```bash
python scripts/pearltrees_multi_account_generator.py \
  --account s243a path/to/s243a.rdf \
  --account s243a_groups path/to/s243a_groups.rdf \
  reports/targets.jsonl \
  --query-template "locate_tree('{title}')"
```

**Features:**
- First `--account` is automatically the primary account
- Cross-account links via RefPearls (preferred) and AliasPearls
- "Drop zone" trees are skipped as entry points
- Account boundary notation (`@account_name`) in paths

### 3. Training
Learn per-tree Procrustes transformations using ModernBERT.

**Script:** `scripts/train_pearltrees_projection.py`

```bash
python scripts/train_pearltrees_projection.py \
  reports/targets.jsonl \
  models/projection.pkl
```

### 4. Evaluation
Compute ranking metrics (Recall@1, R@5, R@10, MRR).

**Script:** `scripts/evaluate_pearltrees_projection.py`

```bash
python scripts/evaluate_pearltrees_projection.py \
  models/projection.pkl \
  reports/targets.jsonl
```

### 5. Inference
Fast query inference with cached target embeddings (~30ms latency).

**Script:** `scripts/infer_pearltrees.py`

```bash
# Single query
python scripts/infer_pearltrees.py models/projection.pkl "locate_tree('Physics')"

# Interactive mode
python scripts/infer_pearltrees.py models/projection.pkl --interactive
```

## Results (Physics Subset - Trees)

| Metric | Value |
|--------|-------|
| Recall@1 | **72.24%** |
| Recall@5 | **94.30%** |
| Recall@10 | **96.20%** |
| MRR | **0.8261** |
| Query Latency | ~30ms |

## Why This Works

- **Code-Aware Embedding:** ModernBERT understands the syntax of function calls (`locate_tree(...)`) and file paths (`/root/child/...`).
- **Per-Tree Procrustes:** Separate transformations for each subtree, with softmax routing at inference.
- **Cached Embeddings:** Target embeddings stored with the model for fast search.
- **Cross-Account Support:** Paths can span multiple Pearltrees accounts with clear boundary markers.
- **Zero-Shot:** Once trained, the model can project *any* query into "Path Space" without retraining.

