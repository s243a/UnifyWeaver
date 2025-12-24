# LDA Smoothing Theory: From Problem to Solution

This document develops the theory of cross-cluster smoothing for LDA projection, building from first principles to practical implementation.

## 1. The Core Problem

### 1.1 What We're Trying to Do

We have a semantic search system where:
- **Questions** are embedded into vectors: `q ∈ ℝᵈ`
- **Answers** are embedded into vectors: `a ∈ ℝᵈ`
- We want to find the best answer for a given question

The naive approach (cosine similarity between q and a) fails because questions and answers occupy different regions of embedding space—they're semantically related but not geometrically close.

### 1.2 The LDA Solution

**Linear Discriminant Analysis (LDA)** learns a projection matrix W that maps questions toward their answers:

```
projected_q = q @ W
similarity = cosine(projected_q, a)
```

This transforms the question embedding to be geometrically closer to its answer.

### 1.3 The Cluster Structure

Our data is organized into **clusters**—groups of related Q-A pairs sharing a common answer theme:

```
Cluster "prolog-installation":
  Q1: "How do I install SWI-Prolog?"
  Q2: "What Prolog does UnifyWeaver use?"
  Q3: "Where do I download Prolog?"
  A:  "UnifyWeaver uses SWI-Prolog..."
```

Each cluster has:
- Multiple questions: Q = [q₁, q₂, ..., qₙ] ∈ ℝⁿˣᵈ
- One (or few) answers: A ∈ ℝᵐˣᵈ (often m=1)

### 1.4 The Sparsity Problem

**This is the core challenge**: Most clusters have only 1-3 questions, but embedding dimension d=64 or higher.

To learn a projection W ∈ ℝᵈˣᵈ, we need to solve:

```
Q @ W ≈ A
```

With n=2 questions and d=64 dimensions, this system is **severely underdetermined**—infinitely many W matrices satisfy it, most of which generalize poorly.

```python
# The problem in code
Q = np.array([[...], [...]])  # 2 questions, 64 dims
A = np.array([[...]])         # 1 answer, 64 dims

# Least squares gives ONE solution, but it overfits
W = np.linalg.lstsq(Q, A)[0]  # Shape: (64, 64)

# This W memorizes Q→A but fails on new questions
```

## 2. Solution Approaches

We developed three approaches, each with different tradeoffs.

### 2.1 Approach 1: Multi-Head Baseline (No Smoothing)

The simplest approach: don't learn W at all. Instead:
1. Store each cluster's centroid: `c_i = mean(Q_i)`
2. Store each cluster's answer embedding: `a_i`
3. At query time, route to nearest centroid, return that answer

```python
class MultiHeadLDABaseline:
    def train(self, clusters):
        self.heads = []
        for Q, A in clusters:
            centroid = np.mean(Q, axis=0)
            answer = np.mean(A, axis=0)
            self.heads.append((centroid, answer))

    def project(self, query, temperature=0.1):
        # Soft routing via softmax over centroid similarities
        sims = [cosine(query, c) for c, a in self.heads]
        weights = softmax(sims / temperature)
        return sum(w * a for w, (c, a) in zip(weights, self.heads))
```

**Why it works**: Avoids the underdetermined system entirely. The "projection" is really just weighted interpolation of stored answers.

**Benchmark result**: 94% P@1, 4.5ms train, 1ms inference

### 2.2 Approach 2: Smoothing Basis

**Key insight**: Similar clusters should have similar projection matrices. We can share structure across clusters.

Instead of independent W_i per cluster, decompose:

```
W_i = Σₖ αᵢₖ Gₖ
```

Where:
- **Gₖ** are K shared basis matrices (learned)
- **αᵢ** are per-cluster coefficients (learned)

This reduces parameters from N×d² to K×d² + N×K.

#### Mathematical Formulation

**Objective**: Minimize reconstruction error across all clusters:

```
L = Σᵢ ||Qᵢ @ Wᵢ - Aᵢ||² + λ·regularization
```

**Alternating optimization**:
1. Fix basis {Gₖ}, solve for coefficients {αᵢ} (closed-form least squares)
2. Fix coefficients, update basis via gradient descent

```python
def solve_for_alpha(Q, A, basis):
    """Given basis, optimal coefficients are closed-form."""
    # Stack projections: P[:,k] = vec(Q @ G_k)
    P = np.column_stack([(Q @ G).ravel() for G in basis])
    # Solve: P @ α ≈ vec(A)
    alpha = np.linalg.lstsq(P, A.ravel())[0]
    return alpha

def update_basis(clusters, alpha, basis, lr):
    """Gradient descent on basis matrices."""
    for i, (Q, A) in enumerate(clusters):
        W_i = sum(alpha[i,k] * basis[k] for k in range(K))
        grad = compute_gradient(W_i, Q, A)
        for k in range(K):
            basis[k] -= lr * alpha[i,k] * grad
```

#### Why K Matters

K controls the expressiveness vs. regularization tradeoff:
- **K=1**: All clusters share one projection (too constrained)
- **K=N**: Each cluster independent (no sharing, overfits)
- **K<<N**: Sweet spot—clusters share structure

**Benchmark results**:

| K | P@1 | Train Time |
|---|-----|------------|
| 2 | 49% | 1.7s |
| 4 | 79% | 2.0s |
| 8 | 89% | 2.6s |
| 16 | 94% | 4.1s |

**Problem**: Even K=16 only matches baseline, while being 1000x slower.

### 2.3 Approach 3: FFT Smoothing

**Key insight**: If we order clusters by similarity, the sequence of W matrices should be "smooth"—nearby clusters should have similar W.

High-frequency variations in this sequence are noise/overfitting.

#### Step 1: Order Clusters via MST + DFS

The goal: Create a 1D path through cluster space where **similar clusters are adjacent**.

**Why this matters**: FFT assumes the signal is ordered meaningfully. Random ordering would make "frequency" meaningless. By ordering clusters by similarity, low frequencies capture global patterns and high frequencies capture local noise.

**Step 1a: Compute pairwise distances**

```python
from scipy.spatial.distance import pdist, squareform

# Cosine distance between all cluster centroids
# distance = 1 - cosine_similarity (so similar = low distance)
distances = squareform(pdist(centroids, metric='cosine'))

# Result: C×C symmetric matrix
# distances[i,j] ≈ 0 means clusters i,j are similar
# distances[i,j] ≈ 2 means clusters i,j are opposite
```

**Step 1b: Build Minimum Spanning Tree**

MST connects all nodes with minimum total edge weight—preferring edges between similar clusters:

```
Full graph (all pairs):        MST (minimum edges to connect all):

  C1────C2────C3                 C1────C2
   │╲  ╱│╲  ╱│                        │
   │ ╲╱ │ ╲╱ │         →              C3
   │ ╱╲ │ ╱╲ │                        │
   │╱  ╲│╱  ╲│                   C4───C5────C6
  C4────C5────C6

  (C² edges)                    (C-1 edges, similar clusters linked)
```

```python
from scipy.sparse.csgraph import minimum_spanning_tree

mst = minimum_spanning_tree(distances).toarray()
mst = mst + mst.T  # Make symmetric (MST returns directed)
```

**Step 1c: DFS traversal to get 1D ordering**

Walk the tree depth-first, always visiting the most similar unvisited neighbor:

```
MST structure:              DFS traversal path:

       C2                   Start at C2 (highest degree)
      ╱  ╲
    C1    C3                C2 → C1 → C4 → C5 → C6 → C3
    │      │
    C4    C5                Path visits similar clusters consecutively
           │
          C6
```

```python
def order_clusters_by_similarity(centroids):
    """Find 1D path that keeps similar clusters adjacent."""

    # Pairwise cosine distances
    distances = squareform(pdist(centroids, metric='cosine'))

    # Build MST
    mst = minimum_spanning_tree(distances).toarray()
    mst = mst + mst.T  # Make symmetric

    # Start DFS from highest-degree node (most connected)
    degrees = (mst > 0).sum(axis=1)
    start = np.argmax(degrees)

    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        order.append(node)

        # Visit neighbors sorted by distance (closest first)
        neighbors = np.where(mst[node] > 0)[0]
        for neighbor in sorted(neighbors, key=lambda n: mst[node, n]):
            dfs(neighbor)

    dfs(start)
    return np.array(order)
```

**Why MST + DFS works:**

| Step | What it does | Why |
|------|--------------|-----|
| Cosine distance | Measures cluster similarity | Semantic closeness |
| MST | Keeps only essential edges | Removes redundant connections |
| DFS | Creates linear path | Respects local similarity |

**Result**: A 1D ordering where large jumps in similarity are minimized:

```
Before (arbitrary):  C1, C2, C3, C4, C5, C6
                     │   │   │   │   │   │
Similarities:        Go  Py  Go  Rust Py  Go   ← topics jump around

After (MST+DFS):     C1, C3, C6, C2, C5, C4
                     │   │   │   │   │   │
Similarities:        Go  Go  Go  Py  Py  Rust  ← smooth transitions
```

Now when we apply FFT, "low frequency" means patterns shared across many adjacent (similar) clusters.

#### Step 2: FFT Analysis

Treat the ordered W matrices as a signal and analyze frequency content:

```python
# Stack W matrices in similarity order
W_stacked = np.stack([W[i] for i in order])  # Shape: (N, d, d)

# FFT along cluster dimension
freq = np.fft.fft(W_stacked, axis=0)

# Power spectrum shows frequency distribution
power = np.abs(freq) ** 2
```

**What frequencies mean**:
- **Low frequency**: Global patterns shared across many clusters
- **High frequency**: Local variations, often noise/overfitting

#### Step 3: Low-Pass Filter

Remove high-frequency components:

```python
def fft_lowpass(signal, cutoff=0.5):
    N = len(signal)
    freq = fft(signal, axis=0)

    # Zero out high frequencies
    mask = np.zeros(N)
    keep = int(N * cutoff)
    mask[:keep] = 1
    mask[-keep+1:] = 1  # Symmetric for real signal

    filtered = ifft(freq * mask.reshape(-1,1,1), axis=0)
    return np.real(filtered)
```

#### Step 4: Blend

Combine original and smoothed:

```python
W_final = (1 - blend) * W_original + blend * W_smoothed
```

**Why FFT works so well**:
1. Operates on per-cluster W matrices (doesn't need to solve underdetermined system)
2. Regularizes by removing noise, not by constraining structure
3. O(N log N) complexity—much faster than basis optimization

**Benchmark result**: 99% P@1, 92ms train

### 2.4 Approach 4: Hierarchical Smoothing

**Idea**: Build a hierarchy of clusters, train smoothing at each level.

```
Level 0:  C1  C2  C3  C4  C5  C6  (218 clusters)
           \  /    \  /    \  /
Level 1:   SC1     SC2     SC3   (merged)
             \      /
Level 2:      SSC1              (coarse)
```

At each level:
1. Merge similar clusters (agglomerative clustering)
2. Train SmoothingBasis on merged clusters
3. Combine projections from all levels

```python
def build_hierarchy(clusters, thresholds=[0.4, 0.7]):
    hierarchy = [clusters]
    current = clusters

    for threshold in thresholds:
        # Agglomerative clustering on centroids
        labels = fcluster(linkage(centroids), t=threshold)

        # Merge clusters with same label
        merged = merge_by_label(current, labels)
        hierarchy.append(merged)
        current = merged

    return hierarchy
```

**Why it underperforms**: Merging destroys cluster-specific information. A question about "Prolog installation" merged with "Python setup" loses precision.

**Benchmark result**: 85% P@1 (worse than baseline!)

## 3. Why FFT Wins

### 3.1 The Frequency Perspective

Our benchmark showed 90% of signal power is in the lowest 87% of frequencies:

```
Frequency analysis:
  90% power at: 86.7% of spectrum
  Recommended cutoff: 0.70
```

This means only ~13% of the W variation is high-frequency "noise"—but removing it boosts P@1 from 94% to 99%.

### 3.2 Comparison of Regularization Strategies

| Method | Regularization Type | Effect |
|--------|--------------------| -------|
| Baseline | None | Stores centroids only, no W |
| SmoothingBasis | Structural constraint | Forces W = Σ αₖGₖ |
| FFT | Frequency constraint | Removes high-freq noise |
| Hierarchical | Cluster merging | Loses precision |

FFT's advantage: It doesn't constrain the structure of W, just smooths it.

### 3.3 Computational Complexity

#### Simplified Overview

| Method | Training | Inference |
|--------|----------|-----------|
| Baseline | O(C·d) | O(C·d) |
| SmoothingBasis | O(iter·C·K²·d²) | O(C·K·d²) |
| FFT | O(C·d³) | O(C·d²) |
| Hierarchical | O(C²·d + levels·iter·C·K²·d²) | O(levels·C·K·d²) |

Where: C = clusters, d = dimension, K = basis count, iter = iterations

#### Detailed Step-by-Step Calculation

For our benchmark: **C=218 clusters, d=64, K=8, iter=50**

**1. Baseline (MultiHeadLDA)**

| Step | Operation | Formula | Count |
|------|-----------|---------|-------|
| Compute centroids | Mean of Q per cluster | C × n̄ × d | 218 × 3 × 64 = 42K |
| Store answers | Copy | C × d | 218 × 64 = 14K |
| **Total** | | | **56K ops** |

**2. FFT Smoothing**

| Step | Operation | Formula | Count |
|------|-----------|---------|-------|
| Q'Q per cluster | Matrix multiply | C × n̄ × d² | 218 × 3 × 4096 = 2.7M |
| Solve (Q'Q+λI)⁻¹Q'A | Cholesky + solve | C × d³ | 218 × 262K = 57M |
| Pairwise distances | Cosine distance | C² × d | 47K × 64 = 3.0M |
| MST | Prim's algorithm | C² | 47K |
| DFS ordering | Graph traversal | C | 218 |
| FFT | Along cluster dim | C × log₂(C) × d² | 218 × 8 × 4096 = 7.1M |
| Filter + IFFT | Same as FFT | C × log₂(C) × d² | 7.1M |
| Blend | Weighted sum | C × d² | 218 × 4096 = 0.9M |
| **Total** | | | **78M ops** |

**3. SmoothingBasis (K=8)**

| Step | Operation | Formula | Count |
|------|-----------|---------|-------|
| Initial W (lstsq) | Same as FFT step 1-2 | C × d³ | 57M |
| Orthogonalize basis | Gram-Schmidt | K² × d² | 64 × 4096 = 0.3M |
| Per iteration: | | | |
| → Solve for α | Lstsq per cluster | iter × C × K × d² | 50 × 218 × 8 × 4096 = 357M |
| → Compute loss | Forward pass | iter × C × n̄ × d² | 50 × 218 × 3 × 4096 = 134M |
| → Gradient | Backprop | iter × C × d² | 50 × 218 × 4096 = 45M |
| → Update basis | K gradient steps | iter × K × d² | 50 × 8 × 4096 = 1.6M |
| **Total** | | | **595M ops** |

**4. Hierarchical (L=3)**

| Step | Operation | Formula | Count |
|------|-----------|---------|-------|
| Build hierarchy | Agglom. clustering | C² × d | 3.0M |
| Train level 0 | SmoothingBasis(C=218) | (see above) | 595M |
| Train level 1 | SmoothingBasis(C≈50) | ~C/4 | 149M |
| Train level 2 | SmoothingBasis(C≈15) | ~C/15 | 40M |
| **Total** | | | **787M ops** |

#### Predicted vs Actual Timing

| Method | Predicted Ops | Actual Time | Ops/ms | Efficiency Factor |
|--------|---------------|-------------|--------|-------------------|
| Baseline | 56K | 4.5ms | 12K | 1.0× (reference) |
| FFT | 78M | 92ms | 848K | **70× faster than predicted** |
| SmoothingBasis K=8 | 595M | 2618ms | 227K | 19× faster than predicted |
| Hierarchical L=3 | 787M | 3642ms | 216K | 18× faster than predicted |

#### Explaining the Efficiency Factors

**Why FFT is 70× more efficient than op-count suggests:**

1. **Vectorization**: NumPy's FFT operates on entire d×d matrices at once
2. **Cache locality**: Sequential memory access in FFT vs random in basis methods
3. **No Python loops**: FFT is single C call, basis methods iterate in Python
4. **BLAS optimization**: `np.linalg.solve` is highly optimized

**Why SmoothingBasis/Hierarchical are only ~18× efficient:**

1. **Python loop overhead**: 50 iterations with per-cluster operations
2. **Memory allocation**: Creating intermediate arrays each iteration
3. **Cache misses**: Random access patterns in basis reconstruction

#### The Real Bottleneck

```
FFT Smoothing time breakdown (estimated):
├── Compute W matrices:  70ms (76%)  ← d³ dominates
├── Cluster ordering:    15ms (16%)  ← C² distance matrix
├── FFT + filter:         5ms  (5%)  ← Actually cheap!
└── Blend:                2ms  (2%)
```

The FFT itself (O(C log C × d²)) is only ~5% of total time. The d³ solve dominates.

### 3.4 Scalability Analysis

Our benchmark (218 clusters, 642 pairs) runs fast on modest hardware. But how do these approaches scale?

#### Projected Training Times

| Clusters | Pairs | FFT | SmoothingBasis K=8 | Hierarchical |
|----------|-------|-----|-------------------|--------------|
| 218 | 642 | 92ms | 2.6s | 3.6s |
| 1,000 | 3,000 | 0.5s | 55s | 80s |
| 10,000 | 30,000 | 6s | 90min | 130min |
| 100,000 | 300,000 | 70s | 150hrs | 220hrs |

*Assumptions: Same d=64, K=8, iter=50. FFT scales as C·d³, others as iter·C·K²·d²*

#### The Crossover Point

```
Small project (< 500 clusters):
├── All approaches viable
├── SmoothingBasis: interpretable basis matrices
├── Hierarchical: multi-resolution analysis
└── FFT: fastest, best accuracy

Medium project (500 - 5,000 clusters):
├── FFT: still under 5 seconds
├── SmoothingBasis: minutes, but tolerable for offline training
└── Hierarchical: slow, diminishing returns

Large project (> 10,000 clusters):
├── FFT: only viable option
├── SmoothingBasis: hours, impractical
└── Hierarchical: days, unusable
```

#### Practical Guidance

| Use Case | Scale | Recommendation |
|----------|-------|----------------|
| Single GitHub repo | 50-500 clusters | Any approach works; FFT recommended |
| Documentation site | 500-2,000 clusters | FFT or baseline |
| Enterprise knowledge base | 2,000-20,000 clusters | FFT only |
| Web-scale search | 100,000+ clusters | FFT + approximate methods |

#### Why This Matters

For a single project like UnifyWeaver (218 clusters), the 2.6s SmoothingBasis training is negligible—you could even use Hierarchical's 3.6s without concern. The benchmark differences feel academic.

But scale to a company-wide documentation system with 10,000 clusters, and suddenly:
- FFT: 6 seconds (acceptable)
- SmoothingBasis: 90 minutes (painful)
- Hierarchical: 2+ hours (unusable)

**The insight**: FFT's O(C·d³) scaling vs O(iter·C·K²·d²) doesn't matter at small scale, but becomes critical at large scale. Design for growth.

#### Partitioning Strategies

For very large datasets, you can partition into smaller sets and apply expensive methods within partitions:

**Manual partitioning**: Split by topic/domain, apply SmoothingBasis per partition
```
Enterprise KB (50,000 clusters)
├── Engineering docs (10,000) → SmoothingBasis: 90min
├── Sales docs (8,000)        → SmoothingBasis: 60min
├── HR docs (5,000)           → SmoothingBasis: 25min
└── ...
Total: ~4hrs (vs 150hrs for monolithic)
```

**FFT's natural partitioning**: The MST + DFS ordering creates implicit "line segments" of similar clusters:

```
FFT ordering:  C3 → C1 → C5 │ C2 → C4 → C8 │ C6 → C7 → C9
               ─────────────┼──────────────┼─────────────
               segment 1     segment 2      segment 3
               (Prolog docs) (Python docs)  (Go docs)
```

These segments emerge naturally from the similarity structure. This enables:

1. **Parallel FFT**: Apply smoothing to segments independently, then blend at boundaries
2. **Segment-local basis**: Use SmoothingBasis within segments (small C), FFT across segments
3. **Incremental updates**: New cluster joins nearest segment, local re-smooth only

**Hybrid approach** (theoretical):
```python
# Partition into segments via FFT ordering
segments = partition_by_similarity_gaps(fft_order, threshold=0.3)

# Expensive method within segments (small C each)
for seg in segments:
    seg.W = SmoothingBasis(seg.clusters, K=4).train()

# FFT across segment representatives
segment_reps = [seg.centroid_W for seg in segments]
smoothed_reps = fft_smooth(segment_reps)

# Propagate smoothing back to clusters
for seg, smooth_W in zip(segments, smoothed_reps):
    seg.blend(smooth_W, factor=0.3)
```

This gives SmoothingBasis interpretability with FFT scalability.

## 4. Implementation Details

### 4.1 Handling Sparse Clusters

The underdetermined system Q @ W = A needs regularization:

```python
def solve_regularized(Q, A, reg=0.01):
    """Ridge regression instead of least squares."""
    d = Q.shape[1]
    # (Q'Q + λI)W = Q'A
    return np.linalg.solve(Q.T @ Q + reg * np.eye(d), Q.T @ A)
```

For clusters with single answer but multiple questions, broadcast:

```python
if A.shape[0] == 1 and Q.shape[0] > 1:
    A = np.tile(A, (Q.shape[0], 1))  # Repeat answer for each question
```

### 4.2 Soft Routing at Inference

All methods use soft routing via temperature-scaled softmax:

```python
def soft_route(query, centroids, projections, temperature=0.1):
    sims = [cosine(query, c) for c in centroids]
    weights = softmax(np.array(sims) / temperature)
    return sum(w * (query @ W) for w, W in zip(weights, projections))
```

Lower temperature → sharper routing (approaches hard argmax).

### 4.3 FFT Cutoff Selection

The benchmark suggests cutoff=0.5 to 0.7 works well. Automatic selection:

```python
def auto_cutoff(W_stacked):
    freq = fft(W_stacked.reshape(len(W_stacked), -1), axis=0)
    power = np.abs(freq) ** 2
    cumulative = np.cumsum(power.sum(axis=1)) / power.sum()

    # Find frequency containing 90% of power
    freq_90 = np.searchsorted(cumulative, 0.9) / len(W_stacked)
    return min(0.7, freq_90 + 0.1)
```

## 5. Practical Recommendations

### 5.1 When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Production system | FFT (cutoff=0.5) |
| Latency-critical | Baseline (no smoothing) |
| Interpretability needed | SmoothingBasis (see basis matrices) |
| Research/experimentation | Try all, benchmark on your data |

### 5.2 Hyperparameter Guidance

**FFT Smoothing**:
- `cutoff`: Start with 0.5, tune in [0.3, 0.7]
- `blend_factor`: 0.6-0.8 works well

**SmoothingBasis** (if you must use it):
- `K`: Start with sqrt(N), increase if underfitting
- `iterations`: 50-100 usually sufficient
- `cosine_weight`: 0.5 balances MSE and cosine loss

### 5.3 Debugging Tips

1. **Check cluster sizes**: Very uneven sizes can hurt smoothing
2. **Visualize frequency spectrum**: Most power should be low-freq
3. **Compare to baseline**: If smoothing hurts, your clusters may be too diverse

## 6. Future Directions

### 6.1 Learnable FFT Filters

Instead of fixed cutoff, learn the filter:

```python
# Learnable frequency mask
mask = sigmoid(learnable_params)  # Per-frequency weights
filtered = ifft(fft(signal) * mask)
```

### 6.2 Attention-Based Routing

Replace softmax routing with learned attention:

```python
# Query-dependent routing
attention = softmax(query @ key_matrix / sqrt(d))
projection = sum(a * W for a, W in zip(attention, projections))
```

### 6.3 Online Updates

Incremental smoothing when new clusters arrive:

```python
def add_cluster(self, Q_new, A_new):
    # Find insertion point in similarity ordering
    # Local FFT update instead of full recomputation
```

## 7. Hierarchical Smoothing Planner

While FFT smoothing is the clear winner for single-level smoothing, we can do better by combining approaches hierarchically. The key insight: **optimize at the specificity depth where clusters become distinguishable**.

### 7.1 The Hierarchical Structure

FFT's MST + DFS ordering creates a natural tree structure with segments:

```
Root (all 218 clusters)
    │
    ├── Segment 0 (18 clusters) ─ similar, need refinement
    ├── Segment 1 (1 cluster)  ─ trivially distinguishable
    ├── Segment 2 (25 clusters) ─ distinct after FFT, no refinement
    ├── Segment 3 (12 clusters) ─ still confusable, need basis
    └── ...
```

### 7.2 Technique Selection Policy

We use a declarative policy (implemented in Prolog with Python fallback) to select techniques based on:

1. **Cluster count**: FFT efficient at scale, basis for medium, baseline for small
2. **Depth in tree**: FFT at shallow depths, basis at deeper levels
3. **Distinguishability**: Skip refinement where clusters already separable

```prolog
%% Rule: Large clusters at shallow depths -> FFT
recommended_technique(NodeId, fft) :-
    node(NodeId, ClusterCount, _, Depth, _),
    ClusterCount >= 30,
    Depth < 3.

%% Rule: Medium clusters at deeper levels -> basis
recommended_technique(NodeId, basis_k8) :-
    node(NodeId, ClusterCount, _, Depth, AvgPairs),
    ClusterCount >= 10, ClusterCount =< 50,
    Depth >= 1,
    AvgPairs >= 2.

%% Rule: Clusters already distinguishable -> skip refinement
clusters_distinguishable(NodeId) :-
    similarity_score(NodeId, Score),
    Score < 0.3.  % Low similarity = distinct clusters
```

### 7.3 Distinguishability-Based Optimization

The key optimization: **only refine where needed**.

```python
def compute_cluster_distinguishability(centroids):
    """
    Average pairwise cosine similarity within segment.

    High score (>0.7) = clusters confusable → needs refinement
    Low score (<0.3) = clusters distinct → skip refinement
    """
    sims = [1 - cosine(c1, c2) for c1, c2 in pairs(centroids)]
    return np.mean(sims)
```

Decision flow:

```
FFT Smoothing (root)
      │
      ├─ Segment A: sim=0.1 → distinguishable → SKIP (use baseline)
      │
      └─ Segment B: sim=0.8 → confusable → REFINE with basis
              │
              ├─ Sub-B1: sim=0.2 → now distinguishable → STOP
              └─ Sub-B2: sim=0.7 → still confusable → continue...
```

This saves compute by focusing refinement only where FFT alone isn't enough.

### 7.4 Soft Constraints for Non-FFT Methods

When using basis methods at deeper levels, we regularize toward the parent's projection:

```python
def apply_parent_constraint(child_proj, parent_proj, weight=0.3):
    """
    Blend child's learned W toward parent's smoothed W.

    This preserves global structure while allowing local refinement.
    """
    parent_W = parent_proj.smoothed_W[child_indices]

    # Re-fit child coefficients toward blended target
    target_W = (1 - weight) * child_W + weight * parent_W
    child_proj.coefficients = refit_to_target(target_W)
```

**Important**: Soft constraints only apply to basis/baseline methods. FFT already performs global smoothing, so constraining FFT children would be redundant.

### 7.5 Inference-Time Blending

At inference, we blend projections from multiple levels:

```python
def project(self, query, temperature=0.1):
    # Find nearest segment
    segment = find_nearest_segment(query)

    # Get segment projection
    segment_proj = self.projectors[segment.id].project(query)

    # Blend with parent (captures global patterns)
    if segment.id in self.parent_projector:
        parent_proj = self.projectors[parent_id].project(query)

        # Weighted blend: parent global + child local
        parent_weight = 0.5
        return parent_weight * parent_proj + (1 - parent_weight) * segment_proj

    return segment_proj
```

### 7.6 Complete Pipeline

```python
from smoothing_planner import HybridSmoothingProjection

# Initialize with configuration
hybrid = HybridSmoothingProjection(
    segment_threshold=0.3,        # Gap threshold for segment boundaries
    parent_weight_decay=0.5,      # Blend ratio at inference
    parent_constraint_weight=0.3  # Regularization strength during training
)

# Train: builds tree, queries policy, executes plan
result = hybrid.train(clusters)

# Result contains:
# - tree: hierarchical structure with distinguishability scores
# - plan: [(technique, node_id), ...] from policy
# - projectors: trained projector per node

# Inference: routes to segment, blends with parent
projected = hybrid.project(query_embedding, temperature=0.1)
```

### 7.7 Cost/Accuracy Tradeoff

| Approach | Training Cost | Inference Cost | Expected Accuracy |
|----------|---------------|----------------|-------------------|
| FFT only | O(N log N + N d²) | O(N) routing + 1 projection | 99% P@1 |
| Hierarchical (naive) | FFT + basis per segment | 2 projections + blend | ~99% P@1 |
| Hierarchical (optimized) | FFT + basis only where needed | 2 projections + blend | ~99% P@1, less compute |

The distinguishability optimization reduces training cost by skipping refinement for already-separable segments.

## 8. Summary

| Concept | Key Point |
|---------|-----------|
| **Problem** | Sparse clusters → underdetermined W learning |
| **Baseline** | Store centroids + answers, soft route (94% P@1) |
| **SmoothingBasis** | Share basis matrices, but expensive and only matches baseline |
| **FFT Smoothing** | Remove high-freq noise from W sequence (99% P@1, fast) |
| **Hierarchical (old)** | Merging hurts precision (85% P@1) |
| **Hierarchical Planner** | FFT at root + basis where confusable + skip where distinct |
| **Winner** | FFT smoothing, optionally refined with hierarchical planner |

The surprising finding: Simple frequency-domain smoothing beats sophisticated structural constraints. The hierarchical planner adds value by focusing refinement effort only where FFT alone isn't enough to distinguish clusters.
