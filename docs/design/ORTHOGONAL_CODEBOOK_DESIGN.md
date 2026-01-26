# Orthogonal Bivector Codebook Design

This document describes the orthogonal bivector codebook approach for fast semantic projection, suitable for mobile deployment.

## Overview

The federated model uses per-cluster transformation matrices (W) that rotate query embeddings to match target embeddings. While accurate, the full approach requires:
- Matrix exponential for bivector composition: O(d³)
- Weighted blending of multiple cluster projections

The **orthogonal codebook** approach approximates these rotations using orthogonal rotation planes that can be composed efficiently using the Rodrigues formula: O(K×d).

## Theory: Bivectors and Rotations

### Procrustes Rotation Matrices

Each cluster's W matrix is an orthogonal transformation (rotation ± reflection) computed via SVD:

```
W = V @ U.T  where  Q @ W ≈ A
```

Where Q is query embeddings and A is answer embeddings.

### Bivector Representation

Any rotation can be represented as a bivector B (antisymmetric matrix) where:

```
W = exp(B)  (matrix exponential)
```

The bivector encodes the rotation plane and angle. For a rotation in the plane spanned by vectors u and v with angle θ:

```
B = θ(u ⊗ v - v ⊗ u)
```

### The Key Insight: Orthogonal Planes Commute

If two bivectors rotate in **orthogonal planes**, their rotations commute:

```
exp(B₁ + B₂) = exp(B₁) × exp(B₂)  (when planes are orthogonal)
```

This means we can:
1. Build a codebook of K orthogonal rotation planes
2. Express any rotation as a weighted combination
3. Apply rotations independently using Rodrigues formula

### Rodrigues Formula

For a rotation in a single plane (u, v) with angle θ:

```
y = x·cos(θ) + (u×x)·sin(θ) + u·(u·x)·(1-cos(θ))
```

This is O(d) per plane, giving O(K×d) total vs O(d³) for matrix exp.

## Implementation Approaches

### Canonical Codebook (Recommended)

Uses axis-aligned rotation planes: dimensions (0,1), (2,3), (4,5), etc.

**Advantages:**
- Guaranteed orthogonality
- No PCA/orthogonalization step needed
- Transformer learns to route to appropriate combinations

**Results:**
- Training: Mean Cosine Sim 0.9997 vs orthogonal teacher
- Hit@1: 57.3% (matches raw embeddings within 0.2%)

### PCA-based Codebook

Extracts principal rotation planes from cluster bivectors via PCA.

**Process:**
1. Compute logm(W) for each cluster to get bivectors
2. Flatten to vectors and run PCA
3. Orthogonalize top-K components

**Results:**
- Validation: Mean Cosine Sim 0.2148 vs full rotation manifold
- Significant information loss during orthogonalization

**Conclusion:** Canonical codebook is preferred.

## Why Canonical Works Better: Commutativity

Both approaches use 64 rotation planes, so the difference isn't coverage—it's **commutativity**.

### The Learning Problem

With **canonical (axis-aligned) planes**:
- Planes are perfectly orthogonal by construction
- Rotation in plane i has **zero effect** on plane j
- Each plane is an independent "knob" the transformer can turn
- The learning problem is **decomposable**: learn each weight independently

With **PCA-orthogonalized planes**:
- Original bivectors are distorted to force orthogonality
- This distortion breaks semantic relationships
- Planes may not commute cleanly after distortion
- The transformer must learn complex interactions between planes

### Why This Matters

The transformer's routing head outputs 64 weights (one per plane). With canonical planes:
```
output = rotate(input, w₀·plane₀) ∘ rotate(..., w₁·plane₁) ∘ ...
```

Each wᵢ can be learned independently because planes don't interfere. With PCA-orthogonalized planes, changing w₀ might require compensating changes to w₁, w₂, etc.

## Why 64 Planes is Sufficient: Matryoshka Embeddings

The canonical approach works well with **64 planes** (covering dims 0-127) because Nomic embeddings use **Matryoshka representation learning**.

### Matryoshka Structure

Modern embedding models like `nomic-ai/nomic-embed-text-v1.5` structure information hierarchically:

| Dimensions | Content |
|------------|---------|
| 0-63 | Core semantic meaning |
| 64-127 | Refinement |
| 128-255 | Fine details |
| 256-767 | Diminishing returns |

This means embeddings can be **truncated** to fewer dimensions while preserving most semantic content.

### Implications for Plane Count

| Planes | Dims Covered | Trade-off |
|--------|--------------|-----------|
| 64 | 0-127 | Covers semantic core, fast |
| 128 | 0-255 | Better coverage, 2× slower |
| 384 | 0-767 | Complete basis, 6× slower |

With Nomic embeddings, 64 planes captures the dimensions that matter most.

### Warning: Non-Matryoshka Embeddings

**This result may not generalize to all embedding models.**

If using an embedding model without Matryoshka properties (where information is spread evenly across all dimensions), you may need:
- More planes (128 or 256) to maintain retrieval quality
- Or accept lower Hit@K performance

Embedding models known to support Matryoshka/truncation:
- `nomic-ai/nomic-embed-text-v1.5` ✓
- `sentence-transformers/all-MiniLM-L6-v2` (partial)
- OpenAI `text-embedding-3-*` ✓

Check your embedding model's documentation for truncation support before assuming 64 planes is sufficient.

## Hit@K Evaluation Results

Testing on asymmetric retrieval (project queries, targets stay raw):

| Approach | Hit@1 | Hit@5 | Hit@10 | Speed |
|----------|-------|-------|--------|-------|
| Raw embeddings | 57.5% | 82.6% | 89.0% | baseline |
| **Orthogonal** | **57.3%** | **82.6%** | **89.0%** | 27,713/s |
| Weighted baseline | 14.1% | 26.5% | 31.4% | 706/s |

**Key findings:**
- Orthogonal matches raw embedding quality (-0.2%)
- 39× faster than weighted baseline
- Baseline is poor in asymmetric mode (designed for symmetric)

## Training Pipeline

### Multi-Source Training

Train on multiple federated models while keeping their clustering:

```bash
python3 scripts/train_orthogonal_codebook.py \
  --train-multisource \
  --federated-models models/skills_qa_federated.pkl models/books_qa_federated.pkl \
  --codebook-method canonical \
  --n-components 64 \
  --epochs 50 \
  --save-transformer models/orthogonal_transformer.pt
```

**Why multi-source?**
- Preserves individual model clustering structure
- Avoids mixing clusters from different domains
- Combines query pools for more robust training

### Single-Source Training

```bash
python3 scripts/train_orthogonal_codebook.py \
  --train \
  --federated-model models/federated.pkl \
  --orthogonal-codebook models/codebook.npz \
  --save-transformer models/transformer.pt
```

## Model Files

| File | Size | Contents |
|------|------|----------|
| `orthogonal_codebook.npz` | ~303 MB | Bivectors, planes (u,v,θ), routing keys |
| `orthogonal_transformer.pt` | ~342 MB | Transformer weights + codebook |

### Codebook Contents

```python
{
  'bivectors': (64, 768, 768),      # Bivector matrices
  'plane_u': (64, 768),              # First basis vector per plane
  'plane_v': (64, 768),              # Second basis vector per plane
  'plane_theta': (64,),              # Rotation angles
  'codebook_keys': (64, 768),        # Routing keys
}
```

## Architecture

### FastOrthogonalTransformer

```
Input: query embedding (768,)
  → Input projection (768 → 768)
  → Transformer encoder (3 layers, 4 heads)
  → Routing head → (64,) weights
  → Scale head → (1,) scale
  → Apply weighted Rodrigues rotations
Output: projected embedding (768,)
```

**Parameters:** ~9.6M

### FastOrthogonalCodebook

Numpy-based codebook for efficient inference:
- Routes queries to top-K planes by cosine similarity
- Applies weighted Rodrigues rotations
- No matrix exponential needed

## When to Use Each Approach

| Approach | Use Case | Speed | Quality |
|----------|----------|-------|---------|
| **Orthogonal codebook** | Mobile/edge deployment | 27K/s | Excellent |
| Weighted baseline | Scalable, many clusters | 700/s | Good for symmetric |
| Full rotation | Maximum accuracy | 40/s | Best |

### Baseline isn't terrible

The weighted baseline (soft-routing through cluster W matrices) has advantages:
- Scales well to many clusters
- Good Hit@5 and Hit@10 in symmetric mode
- Simpler conceptually

However, for asymmetric RAG (project queries, search raw answers), the orthogonal approach is significantly better.

## Related Documentation

- `skill_train_model.md` - Basic federated training
- `skill_semantic_inference.md` - Inference with federated models
- `FEDERATED_MODEL_FORMAT.md` - Model file format
- `education/book-13-semantic-search/08_advanced_federation.md` - Federation theory

## Code

- `scripts/train_orthogonal_codebook.py` - Training and evaluation
- `FastOrthogonalCodebook` class - Numpy inference
- `FastOrthogonalTransformer` class - PyTorch training
