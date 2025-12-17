# Proposal: Seed Question Topology

**Status:** Proposed
**Date:** 2025-12-17
**Context:** Extends [SEMANTIC_PROJECTION_LDA.md](SEMANTIC_PROJECTION_LDA.md) and Knowledge Graph initiatives.

## Executive Summary

This proposal structures the LDA training data from "flat clusters of synonyms" into a **topology of related questions**. We introduce the concept of **seed levels** (`seed(0)`, `seed(1)`, ... `seed(n)`), where the level tracks **how far the knowledge graph has expanded from the original question**.

**Key clarifications:**
- Seed level tracks **expansion distance**, not importance
- `seed(n)` and `seed(n+1)` are only related if they are actually related questions
- All seed levels are equally valid for training - no weighting by level
- Selecting questions by seed level helps ensure **diversity** when expanding clusters

## Motivation

### 1. The "Bag of Questions" Problem
Currently, a cluster contains a flat list of queries (short, medium, long) with no provenance tracking. When adding new questions, we can't tell:
- Which were the original seed questions
- Which were generated/discovered later
- How much diversity exists in the cluster

### 2. Knowledge Graph Expansion Tracking
The seed level tells us how much the knowledge graph has expanded from the original dataset:
- `seed(0)`: The original Q-A pairs - **everything built first** (many questions)
- `seed(1)`: Questions discovered by exploring from seed(0) questions
- `seed(2)`: Questions discovered by exploring from seed(1) questions
- And so on...

This is **provenance tracking**, not a relationship chain. The existing dataset is all seed(0).

### 3. Diversity Control
When adding new questions to a cluster:
- Picking only from `seed(n)` level ensures you're not duplicating existing variations
- Higher seed levels represent more distant explorations
- This prevents clusters from becoming echo chambers of similar questions

## The Seed Topology

### Definition: Seed(0) - The Original Data
*   **Role**: The original Q-A pairs that were built first - the existing dataset.
*   **Property**: There will be **many** seed(0) questions, as this represents everything built before expansion began.
*   **Usage**: The foundation from which the knowledge graph expands.

### Definition: Seed(n) - Expansion Depth
*   **Role**: Questions discovered at expansion depth n from the origin.
*   **Property**: NOT necessarily related to `seed(n-1)` - only related if the questions themselves are related.
*   **Meaning**: "This question was discovered n expansion steps from the original seed."
*   **Usage**:
    *   **Diversity**: Selecting from a specific seed level helps ensure variety when adding new questions.
    *   **Provenance**: Track how the cluster grew over time.
    *   **KG bridging**: A `seed(2)` question might connect to another cluster's `seed(0)`, creating cross-cluster links.

### Important: Seed Level ≠ Relationship

```
seed(0): "How do I read a CSV file?"
seed(1): "How do I read CSV with headers?"      ← related to seed(0)
seed(1): "How do I parse delimited data?"       ← related to seed(0), NOT to above seed(1)
seed(2): "How do I handle CSV encoding issues?" ← related to some seed(1), expansion continues
```

The seed level tracks **when** a question was discovered, not **what** it relates to.

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

*   **No Weighting by Seed Level**: All questions are equally valid training data regardless of seed level. The seed level is metadata for provenance, not importance.
*   **Diversity for Expansion**: When generating new training questions, selecting from a specific seed level helps ensure the new questions explore different semantic territory.
*   **Smoothing Basis Support**: All `seed(n)` questions contribute to defining the semantic boundaries of the cluster. More seed levels = broader coverage.
