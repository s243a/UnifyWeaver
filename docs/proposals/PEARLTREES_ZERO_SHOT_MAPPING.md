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
We parse the Pearltrees RDF export to generate the Q/A pairs.

**Script:** `scripts/pearltrees_target_generator.py`
**Input:** Pearltrees RDF file (`.rdf` or `.xml`)
**Output:** JSONL file with `query` and `target_text`.

**Usage:**
```bash
python scripts/pearltrees_target_generator.py "path/to/export.rdf" "reports/pearltrees_targets.jsonl"
```

### 2. Training the Projection
We learn the linear transformation $W$ such that:
$$ \text{Embed}(\text{Query}) \times W \approx \text{Embed}(\text{MaterializedPath}) $$

This is done using the Orthogonal Procrustes problem (with scaling), which finds the optimal rotation/scaling analytically.

**Script:** `scripts/train_pearltrees_projection.py`
**Usage:**
```bash
# Requires sentence-transformers
python scripts/train_pearltrees_projection.py "reports/pearltrees_targets.jsonl" "models/pearltrees_proj.pkl"
```

### 3. Inference (Zero-Shot Retrieval)
To retrieve items for a new query:
1. Embed the query $q$.
2. Project it: $q' = q \times W$.
3. Search the index of "Materialized Path Embeddings" for the nearest neighbors to $q'$.

## Why This Works
- **Structure Preservation:** The Procrustes method preserves the semantic geometry of the query space.
- **Hierarchical Context:** The target strings explicitly encode the path, so the projection learns to "shift" the query towards this hierarchical representation.
- **Zero-Shot:** Once $W$ is learned, it can project *any* query into the "Path Space" without retraining, assuming the semantic relationship between "Title" and "Path" is consistent.
