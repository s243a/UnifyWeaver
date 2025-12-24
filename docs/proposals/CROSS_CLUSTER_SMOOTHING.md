# Cross-Cluster Smoothing for LDA Projection

## Overview

This document describes three complementary approaches for smoothing LDA projection matrices across clusters. These techniques address the challenge of sparse clusters (few questions per answer) by leveraging similarity between clusters.

## The Problem

In semantic search with LDA projection:
- Each cluster maps question embeddings to answer embeddings via a projection matrix W
- Sparse clusters (1-3 questions) lead to unstable/overfit projections
- Similar clusters should have similar projections, but are trained independently

## Solution Architecture

```
Raw Clusters (277)
      │
      ▼
┌─────────────────────────────────────────┐
│  1. Smoothing Basis (per-cluster)       │
│     - Shared orthogonal basis matrices  │
│     - Per-cluster coefficients          │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  2. Hierarchical Smoothing (federation) │
│     - Merge similar clusters            │
│     - Train at multiple resolution      │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  3. FFT Smoothing (frequency domain)    │
│     - Order clusters by similarity      │
│     - Low-pass filter across boundaries │
└─────────────────────────────────────────┘
      │
      ▼
  Smoothed Projections
```

## Approach 1: Smoothing Basis Projection

**File:** `src/unifyweaver/targets/python_runtime/smoothing_basis.py`

### Concept

Instead of independent W matrices per cluster, decompose projections into:
- **Shared basis:** K orthogonal matrices {G₁, G₂, ..., Gₖ}
- **Per-cluster coefficients:** αᵢ = [αᵢ₁, αᵢ₂, ..., αᵢₖ]
- **Reconstruction:** Wᵢ = Σₖ αᵢₖ Gₖ

### Algorithm

1. Initialize basis from per-cluster pseudoinverse solutions
2. Gram-Schmidt orthogonalization
3. Alternating optimization:
   - Fix basis → solve for coefficients (closed-form least squares)
   - Fix coefficients → gradient descent on basis

### Usage

```python
from smoothing_basis import SmoothingBasisProjection

smoother = SmoothingBasisProjection(num_basis=4, cosine_weight=0.5)
losses = smoother.train(clusters, num_iterations=100)
projected = smoother.project(query_embedding, temperature=0.1)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_basis` | 4 | Number of shared basis matrices |
| `cosine_weight` | 0.5 | Balance between MSE (0) and cosine (1) loss |

## Approach 2: Hierarchical Smoothing

**File:** `src/unifyweaver/targets/python_runtime/hierarchical_smoothing.py`

### Concept

Use federation-style cluster aggregation:
1. Build hierarchy by merging similar clusters
2. Train smoothing at each hierarchy level
3. Combine projections from multiple levels

```
Level 0:  C1  C2  C3  C4  C5  C6  ...  (277 clusters)
           \  /    \  /    \  /
Level 1:   SC1     SC2     SC3    ...  (~50 super-clusters)
             \      /        \
Level 2:      SSC1            SSC2 ...  (~10 mega-clusters)
```

### Algorithm

1. Agglomerative clustering on centroids (cosine distance)
2. Merge clusters at each threshold level
3. Train SmoothingBasisProjection at each level
4. Weighted combination of level projections

### Usage

```python
from hierarchical_smoothing import HierarchicalSmoothing

hs = HierarchicalSmoothing(
    num_levels=3,
    merge_thresholds=[0.4, 0.7],
    num_basis=3
)
stats = hs.train(clusters, num_iterations=50)
projected = hs.project(query_embedding)

# Or project at specific level
proj_fine = hs.project_at_level(query_embedding, level=0)
proj_coarse = hs.project_at_level(query_embedding, level=2)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_levels` | 3 | Number of hierarchy levels |
| `merge_thresholds` | [0.3, 0.5, 0.7] | Cosine distance thresholds per level |
| `level_weights` | [0.55, 0.27, 0.18] | Weights for combining levels |

## Approach 3: FFT Smoothing

**File:** `src/unifyweaver/targets/python_runtime/fft_smoothing.py`

### Concept

Treat clusters as a 1D signal in similarity space:
1. Order clusters by similarity (MST path)
2. Apply FFT to W matrices along cluster dimension
3. Low-pass filter (remove high-frequency noise)
4. Inverse FFT to get smoothed projections

### Key Insight

Similar clusters should have similar projections. High-frequency components in the cluster sequence represent noise/overfitting.

### Algorithm

1. Build MST on cluster centroids (cosine distance)
2. DFS traversal gives 1D ordering
3. Stack W matrices: shape (N, d, d)
4. FFT along N dimension
5. Zero out high frequencies (above cutoff)
6. Inverse FFT
7. Blend with original

### Usage

```python
from fft_smoothing import FFTSmoothingProjection, AdaptiveFFTSmoothing

# Fixed cutoff
fft_proj = FFTSmoothingProjection(cutoff=0.5, blend_factor=0.7)
fft_proj.train(clusters)
projected = fft_proj.project(query_embedding)

# Analyze frequency content
freq_info = fft_proj.get_frequency_analysis()
print(f"90% power at: {freq_info['freq_90_percent']:.1%}")

# Adaptive cutoff based on local density
adaptive = AdaptiveFFTSmoothing(min_cutoff=0.2, max_cutoff=0.7)
adaptive.train(clusters)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cutoff` | 0.5 | Fraction of frequencies to keep (0.5 = 50%) |
| `blend_factor` | 0.5 | Blend between original (0) and smoothed (1) |
| `min_cutoff` | 0.2 | Minimum cutoff for adaptive version |
| `max_cutoff` | 0.8 | Maximum cutoff for adaptive version |

## Comparison

| Approach | Best For | Complexity | Key Benefit |
|----------|----------|------------|-------------|
| Smoothing Basis | Per-cluster regularization | O(NK²) per iter | Shared structure discovery |
| Hierarchical | Multi-resolution patterns | O(N² log N) build | Coarse-to-fine control |
| FFT | Continuous interpolation | O(N log N) | Noise removal |

## Combining Approaches

For best results, use hierarchical + FFT:

```python
# 1. Build hierarchy and train
hs = HierarchicalSmoothing(num_levels=3)
hs.train(clusters)

# 2. Apply FFT smoothing to each level's W matrices
for level_idx, smoother in enumerate(hs.projections):
    # smoother.basis contains the trained basis matrices
    # Could apply FFT smoothing here if needed
    pass

# 3. Project with multi-level combination
result = hs.project(query)
```

## Benchmark Results

Tested on real training data: **218 clusters, 642 Q-A pairs, dim=64**

### Accuracy Comparison

| Method | P@1 | P@3 | Cosine | MSE |
|--------|-----|-----|--------|-----|
| **FFT (cutoff=0.5)** | **99.0%** | 90.3% | 0.917 | 0.0078 |
| FFT (cutoff=0.7) | 99.0% | 90.3% | 0.917 | 0.0077 |
| AdaptiveFFT | 99.0% | 90.3% | 0.916 | 0.0082 |
| MultiHeadLDA (baseline) | 94.0% | 85.3% | 0.869 | 0.0078 |
| SmoothingBasis K=16 | 94.0% | 85.3% | 0.848 | 0.0081 |
| SmoothingBasis K=8 | 89.0% | 80.3% | 0.711 | 0.0096 |
| Hierarchical L=3 | 85.0% | 76.0% | 0.624 | 0.0115 |
| SmoothingBasis K=2 | 49.0% | 43.0% | 0.376 | 0.0132 |

### Computational Cost

| Method | Train (ms) | Inference (us/query) | Complexity |
|--------|------------|----------------------|------------|
| MultiHeadLDA | 4.5 | 1,077 | O(N) |
| FFT (cutoff=0.5) | 92 | 1,610 | O(N log N) |
| AdaptiveFFT | 313 | 1,649 | O(N log N) |
| SmoothingBasis K=8 | 2,618 | 8,386 | O(NK²) |
| SmoothingBasis K=16 | 4,129 | 14,251 | O(NK²) |
| Hierarchical L=3 | 3,642 | 15,440 | O(N² log N) |

### Key Findings

1. **FFT smoothing is the winner**: 99% P@1, fast training (92ms), reasonable inference
2. **Baseline is strong**: Simple multi-head routing achieves 94% without smoothing
3. **SmoothingBasis needs high K**: Only matches baseline at K=16, but 1000x slower
4. **Hierarchical underperforms**: Cross-cluster merging actually hurts accuracy
5. **Adaptive FFT**: No benefit over fixed cutoff in this dataset

### Recommendations

| Use Case | Recommended Method |
|----------|-------------------|
| Production (accuracy + speed) | FFT (cutoff=0.5) |
| Low latency required | MultiHeadLDA baseline |
| Research/interpretability | SmoothingBasis K=8 |

### Running the Benchmark

```bash
python scripts/benchmark_smoothing.py
python scripts/benchmark_smoothing.py --max-clusters 50  # Quick test
python scripts/benchmark_smoothing.py --dim 128          # Higher dimension
```

## Training Data

Two tailored answer datasets were generated:

| Provider | Location | Pairs | Success Rate |
|----------|----------|-------|--------------|
| Claude (sonnet) | `training-data/tailored/` | 639 | 99.2% |
| Gemini (3-flash) | `training-data/tailored-gemini/` | 644 | 100% |

## Future Work

1. **Learnable FFT filters**: Train filter shape instead of fixed cutoff
2. **Cross-validation**: Automatic hyperparameter selection
3. **Integration with HNSW**: Use Go/Rust federation infrastructure
4. **Streaming updates**: Incremental smoothing for new clusters

## References

- `smoothing_basis.py`: Gradient-based basis sharing
- `hierarchical_smoothing.py`: Federation-style aggregation
- `fft_smoothing.py`: Frequency-domain filtering
- `projection.py`: Original single-W projection
