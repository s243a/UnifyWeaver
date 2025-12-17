# Proposal: Seed Question Topology

**Status:** Proposed
**Date:** 2025-12-17
**Context:** Extends [SEMANTIC_PROJECTION_LDA.md](SEMANTIC_PROJECTION_LDA.md) and Knowledge Graph initiatives.

## Executive Summary

This proposal structures the LDA training data from "flat clusters of synonyms" into a **topology of related questions**. We introduce the concept of **ordered seeds** (`seed(0)`, `seed(1)`, ... `seed(n)`), where `seed(0)` represents the direct intent of the answer, and subsequent seeds represent related questions that expand the search space and diversity.

## Motivation

### 1. The "Bag of Questions" Problem
Currently, a cluster contains a flat list of queries (short, medium, long) assumed to mean roughly the same thing. This obscures the "center" of the topic and treats tangential queries the same as core queries.

### 2. Knowledge Graph Provenance
When building the Knowledge Graph (KG), knowing the specific angle or nuance of a question is critical. Linking an answer to a generic cluster is less precise than linking it to the specific `seed(n)` variation that bridges two topics.

### 3. Search Space Expansion
`seed(n+1)` is not just a rephrasing of `seed(n)`; it is a **related question**. This explicitly expands the search space to cover adjacent semantic territory without forcing the system to hallucinate connections.

## The Seed Topology

### Definition: Seed(0) - The Anchor
*   **Role**: The direct, original question that the Playbook Answer was created to solve.
*   **Property**: Highest semantic alignment with the Answer content.
*   **Usage**: The primary key for "Foundational" KG relations.

### Definition: Seed(n) - The Expansion
*   **Role**: A distinct question *related* to `seed(n-1)`.
*   **Property**: Introduces diversity (new vocabulary, new use case, slightly different intent) while remaining relevant to the Cluster.
*   **Usage**:
    *   **Search**: Captures user queries that are "close but not exact" matches to the core intent.
    *   **KG**: Acts as a bridge. `Cluster A: seed(2)` might be semantically identical to `Cluster B: seed(0)`, creating a transitional link that wouldn't exist at the cluster level.

## Proposed Schema Change

We will move from a flat `queries` object to an ordered `seeds` array.

**Current (Flat):**
```json
"queries": {
  "short": ["read csv", "parse csv"],
  "medium": ["how to read csv files", "parsing csv data"]
}
```

**Proposed (Topology):**
```json
"seeds": [
  {
    "order": 0,
    "text": "How do I read a CSV file?",
    "variants": ["read csv", "parsing csv data", "csv ingestion"],
    "notes": "Direct intent"
  },
  {
    "order": 1,
    "text": "How do I read a CSV file with headers?",
    "variants": ["csv with headers", "parse labeled csv"],
    "notes": "Related: specific format variation"
  },
  {
    "order": 2,
    "text": "How do I filter rows while reading a CSV?",
    "variants": ["csv filtering", "read csv where condition"],
    "notes": "Related: compositional logic"
  }
]
```

## Impact on Knowledge Graph

This structure allows for **Multi-Resolution Linking**:

1.  **Core-to-Core (Strong)**: Linking `Cluster A (seed 0)` to `Cluster B (seed 0)`.
    *   *Example:* `dotnet_compilation` is foundational to `csharp_examples`.
2.  **Periphery-to-Core (Nuanced)**: Linking `Cluster A (seed 2)` to `Cluster B (seed 0)`.
    *   *Example:* `read_csv_with_headers (seed 1)` might link to `json_source (seed 0)` via a "Transitional" link (handling headers -> handling schemas), which might be missed if comparing generic "CSV" to "JSON".

## Impact on Training

*   **Weighting**: `seed(0)` can be weighted higher for the "Centroid" calculation to ensure the projection remains stable.
*   **Smoothing**: `seed(n)` variants provide the "support" needed for the Smoothing Basis algorithm, defining the valid semantic boundaries of the cluster.
