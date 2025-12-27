# Pearltrees Zero-Shot Mapping via Minimal Transformation

**Status:** Implementation Ready
**Date:** 2025-12-26

## Overview

This proposal outlines the implementation of a "Zero-Shot" mapping for Pearltrees data. The goal is to map a user query (e.g., "Physics Notes") to a structured "Materialized Path" representation of the target item.

By using the **Minimal Transformation (Procrustes)** approach, we learn a global rotation and scaling that aligns the "Query Embedding Space" (Titles) with the "Path Embedding Space" (Hierarchy + Titles).

## The Target Representation

The target "Answer" string is constructed to encode both the hierarchical identity (IDs) and the semantic content (Titles) of the path.

**Format:**
```text
/root_id/child_id/leaf_id
- Root Title
  - Child Title
    - Leaf Title
```

This structure ensures that the embedding model perceives the hierarchical context.

## Implementation Steps

### 1. Data Preparation
We parse the Pearltrees RDF export to generate the Q/A pairs. We use a function-call style query format (e.g., `locate_node('Title')`) to distinguish these structural queries from standard semantic search.

**Script:** `scripts/pearltrees_target_generator.py`
**Input:** Pearltrees RDF file (`.rdf` or `.xml`)
**Output:** JSONL file with `query` and `target_text`.

**Usage:**
```bash
python scripts/pearltrees_target_generator.py "path/to/export.rdf" "reports/pearltrees_targets.jsonl" --query-template "locate_node('{title}')"
```

### 2. Training the Projection
We learn the linear transformation $W$ using **ModernBERT** embeddings. ModernBERT is chosen because it was trained on code and handles the structured, "code-like" nature of our queries (e.g., `locate_node('Hacktivism')`) and hierarchical paths better than standard models.

**Script:** `scripts/train_pearltrees_projection.py`
**Usage:**
```bash
# Uses ModernBERT (nomic-ai/nomic-embed-text-v1.5) by default
python scripts/train_pearltrees_projection.py "reports/pearltrees_targets.jsonl" "models/pearltrees_proj.pkl"
```

### 3. Inference (Zero-Shot Retrieval)
To retrieve items:
1. Formulate query as code: `locate_node('Physics Notes')`.
2. Embed with ModernBERT.
3. Project: $q' = q \times W$.
4. Search target index.

## Why This Works
- **Code-Aware Embedding:** ModernBERT understands the syntax of function calls (`locate_node(...)`) and file paths (`/root/child/...`), providing a sharper distinction for these structural intents.
- **Structure Preservation:** The Procrustes method preserves the semantic geometry of the query space.
- **Hierarchical Context:** The target strings explicitly encode the path.
- **Zero-Shot:** Once $W$ is learned, it can project *any* query into the "Path Space" without retraining.
