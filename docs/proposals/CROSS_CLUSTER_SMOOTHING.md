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

## Approach 4: Hybrid Smoothing Planner

**Files:**
- `src/unifyweaver/core/lda_smoothing_policy.pl`: Declarative Prolog policy
- `src/unifyweaver/targets/python_runtime/smoothing_planner.py`: Python bridge

### Concept

Combine approaches hierarchically based on the FFT's natural tree structure:

```
Root (all clusters) ─── FFT smoothing
    │
    ├── Segment A (confusable) ─── basis_k8 refinement
    │       │
    │       └── Sub-segment (still confusable) ─── basis_k4
    │
    └── Segment B (distinguishable) ─── baseline (skip refinement)
```

### Key Optimizations

1. **Depth-based transitions**: FFT at root, basis at mid-levels, baseline at leaves
2. **Distinguishability-based refinement**: Only refine where clusters are still confusable
3. **Soft constraints**: Regularize child projections toward parent's W
4. **Inference blending**: Combine parent and child projections

### Usage

```python
from smoothing_planner import HybridSmoothingProjection

hybrid = HybridSmoothingProjection(
    segment_threshold=0.3,        # Gap threshold for segments
    parent_weight_decay=0.5,      # Inference blend ratio
    parent_constraint_weight=0.3  # Training regularization
)

result = hybrid.train(clusters)
projected = hybrid.project(query_embedding)
```

### When to Use

| Scenario | Recommendation |
|----------|----------------|
| Simple deployment | FFT only (99% P@1, fastest) |
| Maximum accuracy | Hybrid with distinguishability optimization |
| Interpretability | SmoothingBasis at segment level |

See `LDA_SMOOTHING_THEORY.md` Section 7 for detailed explanation.

## Approach 5: Kernel-Based Smoothing

**File:** `src/unifyweaver/targets/python_runtime/smoothing_basis.py`

### Concept

Replace 1D FFT smoothing with multi-dimensional kernel smoothing that respects the full similarity structure between clusters.

See [KERNEL_SMOOTHING_THEORY.md](KERNEL_SMOOTHING_THEORY.md) for theoretical foundation.

### Available Methods

#### 5a. KernelSmoothingProjection
Direct kernel-weighted averaging of W matrices:
```python
W_smoothed_i = Σ_j K(centroid_i, centroid_j) W_j / Σ_j K(...)
```

Supported kernels:
- **Matérn-5/2** (recommended): C⁴ smooth, numerically stable
- **Matérn-3/2**: C² smooth
- **Gaussian RBF**: C^∞ smooth
- **Laplacian**: Exponential decay (use with caution)

#### 5b. UnifiedKernelBasisProjection
Full coupled optimization with graph Laplacian regularization:
```
L = Σ_i ||Q_i W_i - A_i||² + λ Σ_{i,j} K(i,j) ||W_i - W_j||²
```

Features:
- Matérn-5/2 kernel → sparse graph Laplacian
- K basis vectors with orthonormal constraint
- Conjugate Gradient solver with Jacobi preconditioning
- Optional k-NN sparsification for O(Nk) complexity

#### 5c. KernelSmoothedBasisProjection (Best Performance)
Sequential hybrid approach:
1. Train SmoothingBasisProjection (per-cluster optimization)
2. Apply Matérn-5/2 kernel smoothing to resulting W matrices
3. Blend between unsmoothed and smoothed

### Usage

```python
from smoothing_basis import (
    KernelSmoothingProjection,
    KernelSmoothedBasisProjection,
    SmoothingKernel,
)

# Direct kernel smoothing
ks = KernelSmoothingProjection(
    kernel=SmoothingKernel.MATERN_52,
    length_scale=0.5,
)
ks.train(clusters)
projected = ks.project(query_embedding)

# Hybrid: basis + kernel (recommended)
hybrid = KernelSmoothedBasisProjection(
    num_basis=8,
    length_scale=0.5,
    kernel_blend=0.5,  # 0=unsmoothed, 1=fully smoothed
)
hybrid.train(clusters)
projected = hybrid.project(query_embedding)
```

### Benchmark Results (hash embeddings, 20 clusters)

| Method | MRR | Rank | Cosine |
|--------|-----|------|--------|
| **Hybrid_K8_ls0.5_b0.5** | **0.5124** | 3.2 | 0.462 |
| Residual_K8_r0.01 | 0.5041 | 2.7 | 0.484 |
| FFT_c0.5 | 0.4987 | 2.7 | 0.483 |
| Kernel_Matern52_ls0.5 | 0.4859 | 3.4 | 0.442 |
| UnifiedCG_K8 | 0.3519 | 7.3 | 0.324 |

### Key Finding: Sequential > Coupled

The hybrid approach outperforms full coupled optimization because:
- **Coupled (UnifiedCG)**: Laplacian regularization fights data fidelity during joint optimization
- **Sequential (Hybrid)**: Each phase fully optimizes its objective without interference

This suggests that kernel smoothing works best as **post-processing** rather than as a joint constraint.

### Recommendations

| Use Case | Method | Why |
|----------|--------|-----|
| Best accuracy | KernelSmoothedBasis K=8 | Highest MRR |
| Fast + good | FFT cutoff=0.5 | O(N log N), strong baseline |
| Research | UnifiedCG | Study coupled optimization |

## Completed Implementation

### Policy Compilation (Task 2)

The Prolog smoothing policy is now compiled to Python, Go, and Rust:

```bash
python src/unifyweaver/codegen/lda_smoothing_policy_compiler.py --target all
```

Generated files:
- `src/unifyweaver/codegen/generated/smoothing_policy.py`
- `src/unifyweaver/codegen/generated/smoothing_policy.go`
- `src/unifyweaver/codegen/generated/smoothing_policy.rs`

### Benchmark Results (Task 3)

Hybrid planner vs single-level FFT on 50 clusters:

| Method | P@1 | Train (ms) | Inference (μs) |
|--------|-----|------------|----------------|
| FFT (single-level) | 99.0% | 45.0 | 645 |
| Hybrid (distinguishability) | 99.0% | 19.3 | 959 |
| Hybrid (full refinement) | 99.0% | 20.2 | 816 |

Both achieve same accuracy; hybrid is faster for small clusters because policy selects simpler technique.

### Density Confidence Experiments (Task 4)

See `scripts/density_confidence_experiments.py` for:
- Experiment 1: Confidence calibration (ECE = 0.0035)
- Experiment 2: Ranking improvement (+0.0014 MRR with w=1.0, τ=0.5)
- Experiment 3: Sparse region robustness analysis

### End-to-End Pipeline (Task 5)

`smoothing_pipeline.py` integrates all components:

```python
from smoothing_pipeline import SmoothingPipeline

pipeline = SmoothingPipeline()
pipeline.train(clusters)
results = pipeline.search(query_embedding, k=10)
```

## Future Work

### Priority 1: Deeper Benchmarking

Current benchmarks measure cluster-level accuracy (P@1, P@3). We need answer-level metrics:

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Cosine Similarity** | cos(projected_query, target_answer) | Measures projection quality directly |
| **MSE** | ‖projected_query - target_answer‖² | Captures magnitude errors |
| **Rank of Target** | Position of correct answer in results | End-to-end retrieval quality |
| **Recall@k** | Is correct answer in top-k? | Practical retrieval threshold |

**Tasks:**
- [ ] Add per-query cosine similarity to target answer (not just cluster centroid)
- [ ] Add MSE from projected query to ground-truth answer embedding
- [ ] Compare metrics across smoothing methods (FFT vs Basis vs Baseline)
- [ ] Measure variance across clusters (are some clusters harder?)
- [ ] Test with real embeddings (E5-small, ModernBERT)

### Priority 2: Hyperparameter Optimization

- [ ] **Cross-validation**: Automatic selection of FFT cutoff, blend factor
- [ ] **Learnable FFT filters**: Train filter shape instead of fixed cutoff
- [ ] Sensitivity analysis: How robust are results to parameter changes?

### Priority 3: Production Integration

- [ ] **Integration with HNSW**: Use Go/Rust federation infrastructure
- [ ] **Streaming updates**: Incremental smoothing for new clusters
- [ ] Serialization format for cross-language compatibility

### Priority 4: Advanced Features

- [ ] Hierarchical FFT: Apply smoothing at multiple granularities
- [ ] Adaptive cutoff per cluster based on local density
- [ ] Online learning: Update W matrices from user feedback

## References

### Theory Documents
- [SMOOTHING_BASIS_PROJECTION.md](SMOOTHING_BASIS_PROJECTION.md): Full basis formulation theory
  - Section "Connection to Constraint Geometry" documents **tangent space theory**
  - Section "Connection to Tensor Algebra" links gradients to outer products
- [KERNEL_SMOOTHING_THEORY.md](KERNEL_SMOOTHING_THEORY.md): Multi-dimensional smoothing theory
  - Why 1D FFT falls short for high-dimensional embeddings
  - Graph Laplacian, Green's functions, and kernel methods
  - Matérn kernels as numerically stable alternative
- [SEMANTIC_PROJECTION_LDA.md](SEMANTIC_PROJECTION_LDA.md): Outer product W = a ⊗ q formulation
- [LDA_SMOOTHING_THEORY.md](LDA_SMOOTHING_THEORY.md): Core LDA theory and loss functions

### Core Smoothing
- `smoothing_basis.py`: Gradient-based basis sharing + ResidualBasisProjection
- `hierarchical_smoothing.py`: Federation-style aggregation
- `fft_smoothing.py`: Frequency-domain filtering
- `projection.py`: Original single-W projection

### Policy & Planning
- `smoothing_planner.py`: Hybrid hierarchical planner
- `lda_smoothing_policy.pl`: Prolog declarative policy
- `lda_smoothing_policy_compiler.py`: Policy compiler

### Density & Pipeline
- `density_scoring.py`: KDE and flux-softmax (includes DensityIndex)
- `smoothing_pipeline.py`: End-to-end pipeline

### Benchmarks & Experiments
- `scripts/benchmark_hybrid_planner.py`: Hybrid planner comparison
- `scripts/density_confidence_experiments.py`: Paper experiments
