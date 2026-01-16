# Proposal: Curation Quality Metrics for Projected Embeddings

## Overview

This proposal describes metrics for measuring curation quality and how curation improves tree structure when applied via blended embeddings. The goal is to establish reproducible quality measures that work across different curation sources (personal, crowd-sourced, expert taxonomies).

## Current Metrics in the Codebase

### 1. Ranking Metrics (`scripts/evaluate_pearltrees_projection.py`)

**What they measure:** How well the projection model retrieves the correct target given a query.

| Metric | Description | Current Use |
|--------|-------------|-------------|
| R@1 | Recall at 1 - correct target in top result | Projection accuracy |
| R@5 | Recall at 5 - correct target in top 5 | Retrieval quality |
| R@10 | Recall at 10 - correct target in top 10 | Broader retrieval |
| MRR | Mean Reciprocal Rank - average of 1/rank | Overall ranking quality |

**Current location:** `scripts/evaluate_pearltrees_projection.py:132-149`

**Curation quality application:**
- Higher R@K with fewer curated examples = better curation efficiency
- Compare MRR across curation sources to evaluate organizational quality
- Track MRR vs. curation size to find minimum viable curation

### 2. Cluster Quality Metrics (`scripts/analyze_federated_clusters.py`)

**What they measure:** How well-separated and meaningful the learned clusters are.

| Metric | Description | Current Use |
|--------|-------------|-------------|
| Top-1 softmax weight | Average weight on highest-scoring cluster | Routing confidence |
| Top-10 softmax weight | Cumulative weight on top 10 clusters | Cluster spread |
| Centroid similarity (mean) | Average pairwise similarity of centroids | Cluster distinctness |
| Centroid similarity (max) | Maximum pairwise similarity | Worst-case overlap |
| Effective rank | Spectral participation ratio | Intrinsic dimensionality |
| Over-segmentation flag | top-1 < 50% indicates too many clusters | Model health |

**Current location:** `scripts/analyze_federated_clusters.py:68-120`

**Curation quality application:**
- Well-curated data should produce high top-1 weights (confident routing)
- Lower centroid similarity = better-separated conceptual clusters
- Effective rank indicates how many distinct "directions" the curation captures

### 3. Hierarchy Quality Metrics (`scripts/mindmap/hierarchy_objective.py`)

**What they measure:** How well a tree structure matches semantic and information-theoretic principles.

| Metric | Description | Current Use |
|--------|-------------|-------------|
| D(T) | Semantic distance - parent-child coherence | Tree structure quality |
| H(T) | Entropy gain - information flow between levels | Hierarchical organization |
| J(T) | Combined objective (α·D + β·H) | Overall hierarchy quality |

**Current location:** `scripts/mindmap/hierarchy_objective.py:254-284`

**Curation quality application:**
- D(T): Good curation should produce low semantic distance (related concepts grouped)
- H(T): Good curation should show entropy increasing with depth (general→specific)
- J(T): Combined score for comparing curation strategies

---

## Proposed: Curation Quality Metrics

### 4. Curation Efficiency Ratio (NEW)

**Definition:**
```
CER = ΔQuality / ΔCuration_Size
```

Where:
- ΔQuality = improvement in J(T) or MRR from adding curation
- ΔCuration_Size = number of curated nodes/relationships added

**Purpose:** Measure how much quality improvement each curated item provides. High CER = efficient curation.

### 5. Generalization Gap (NEW)

**Definition:**
```
G = Quality(unseen_data) / Quality(seen_data)
```

**Purpose:** Measure how well the curation transfers to new data. G ≈ 1.0 means good generalization.

**Application to blended embeddings:**
- Compute J(T) on trees built from curated data
- Compute J(T) on trees built from foreign data (same embedding space)
- Ratio indicates transfer quality

### 6. Blend Sensitivity Analysis (NEW)

**Definition:**
```
S(b) = dJ(T)/db at blend level b
```

**Purpose:** Measure how tree quality changes with blend ratio.

**Interpretation:**
- S(b) > 0: More projection improves quality
- S(b) < 0: Too much projection hurts quality
- S(b) = 0: Optimal blend point

---

## Measuring Curation Impact on Tree Structure

### Experimental Protocol

1. **Baseline (blend=0):** Build tree using raw embeddings
   - Compute J(T), D(T), H(T)
   - This represents "no curation influence"

2. **Projected (blend=1):** Build tree using fully projected embeddings
   - Compute J(T), D(T), H(T)
   - This represents "full curation influence"

3. **Blended (blend=0.1 to 0.9):** Sweep blend values
   - Find optimal blend that maximizes J(T)
   - Plot quality curves

4. **Generalization test:** Repeat on held-out data
   - Verify improvement transfers to unseen concepts

### Expected Outcomes

| Scenario | D(T) | H(T) | J(T) | Interpretation |
|----------|------|------|------|----------------|
| Good curation | ↓ | ↑ | ↑ | Tighter clusters, better hierarchy |
| Poor curation | ↔ | ↔ | ↔ | No improvement over baseline |
| Overfitting | ↓ seen, ↑ unseen | - | ↓ on unseen | Memorized, didn't generalize |

---

## Implementation Plan

### Phase 1: Integrate Existing Metrics

1. Add J(T) computation to density explorer recompute
2. Display D(T), H(T), J(T) in UI after tree generation
3. Log metrics with blend value for analysis

### Phase 2: Curation Efficiency Analysis

1. Create `scripts/analyze_curation_efficiency.py`
2. Input: curation data, test data, embedding model
3. Output: CER curve, optimal curation size recommendation

### Phase 3: Blend Optimization

1. Add auto-blend feature to density explorer
2. Sweep blend values, find optimal via J(T)
3. Display recommended blend with confidence

### Phase 4: Comparative Analysis Tool

1. Compare multiple curation sources on same test data
2. Generate report: which curation style works best for which domains
3. Support A/B testing of curation strategies

---

## Relation to Reproducibility

While curations are personal/perspectival, quality metrics are objective:

| Aspect | Reproducible? | Notes |
|--------|---------------|-------|
| Curation content | No | Personal choice |
| Metric computation | Yes | Same algorithm, same results |
| Quality comparison | Yes | Can compare any two curations |
| Optimal blend | Yes | Determined by optimization |

This allows statements like:
- "Curation A achieves J(T)=0.85 with 500 nodes"
- "Curation B needs 2000 nodes for same quality"
- "Curation A generalizes better (G=0.95 vs G=0.78)"

---

## References

- `scripts/evaluate_pearltrees_projection.py` - Ranking metrics
- `scripts/analyze_federated_clusters.py` - Cluster analysis
- `scripts/mindmap/hierarchy_objective.py` - Hierarchy objective J(T)
- `docs/CLUSTER_COUNT_THEORY.md` - Theoretical basis for cluster selection
