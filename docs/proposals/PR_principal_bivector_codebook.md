# Proposal: Principal Bivector Codebook for Rotation Transformers

## Problem Statement

Current rotation transformer distillation approaches face a computational tradeoff:

| Approach | Quality | Computation | Output Dim |
|----------|---------|-------------|------------|
| Weighted vectors (baseline) | Good | Fast | 768 |
| Full Givens rotation | Better | Expensive | 294,528 |
| Sparse Givens (384 planes) | Better | Expensive | 385 |

The rotation-based approach (PR #637) shows numerically improved results (0.776 cosine similarity vs previous methods) but requires:
- Computing optimal rotation parameters per sample (O(k) per sample)
- 384+ output dimensions for rotation angles
- Sequential Givens rotation application

## Proposed Solution: Principal Bivector Codebook

Instead of predicting raw rotation parameters, use a **learned codebook of principal rotation directions** derived from the federated model's cluster rotations.

### Key Insight

The 124 cluster rotations in the federated model span a low-dimensional subspace of SO(768). We can find the principal directions in this subspace and use them as a basis for all rotations.

### Architecture

```
Input: Query embedding x ∈ R^768
                ↓
        Transformer Encoder (3 layers)
                ↓
        Routing projection → r ∈ R^768
                ↓
        Cosine similarity to codebook keys
                ↓
        Top-K selection + normalize weights
                ↓
        Blend: B = Σᵢ wᵢ × Bᵢ (weighted bivector blend)
                ↓
        Scale prediction: s ∈ R
                ↓
        Output: y = exp(B) × x × s
```

### Codebook Construction

```python
import numpy as np
from scipy.linalg import logm
from sklearn.decomposition import PCA

def build_principal_bivector_codebook(cluster_rotations, n_components=64):
    """
    Build a codebook of principal rotation directions.

    Args:
        cluster_rotations: List of (768, 768) rotation matrices from federated model
        n_components: Number of principal bivectors (codebook size)

    Returns:
        basis: (n_components, 768, 768) antisymmetric matrices
        mean: (768, 768) mean rotation generator
    """
    # Extract rotation generators (bivectors as antisymmetric matrices)
    generators = []
    for W in cluster_rotations:
        # Log map: rotation matrix → bivector (antisymmetric matrix)
        A = logm(W)
        # Ensure antisymmetric (numerical stability)
        A = (A - A.T) / 2
        generators.append(A)

    # Flatten for PCA
    d = generators[0].shape[0]
    flat_generators = np.array([g.flatten() for g in generators])

    # PCA on rotation space
    pca = PCA(n_components=n_components)
    pca.fit(flat_generators)

    # Reshape principal components back to antisymmetric matrices
    basis = []
    for component in pca.components_:
        B = component.reshape(d, d)
        # Ensure antisymmetric
        B = (B - B.T) / 2
        basis.append(B)

    return np.array(basis), pca.mean_.reshape(d, d), pca.explained_variance_ratio_
```

### Transformer Output

The transformer predicts a routing vector + scale:
- Routing embedding for cosine similarity to codebook keys
- 1 scaling factor

### Routing Strategy: Cosine Similarity (Preferred)

Instead of softmax on arbitrary logits, use **cosine similarity** to codebook keys, matching how the federated model routes to clusters:

```python
class BivectorCodebookTransformer(nn.Module):
    def __init__(self, embed_dim, codebook_size, n_layers=3, n_heads=4, top_k=8):
        super().__init__()
        self.encoder = TransformerEncoder(embed_dim, n_layers, n_heads)
        self.routing_head = nn.Linear(embed_dim, embed_dim)  # project to routing space
        self.scale_head = nn.Linear(embed_dim, 1)
        self.top_k = top_k

        # Codebook bivectors: (codebook_size, d, d)
        self.register_buffer('codebook', None)

        # Codebook keys for routing: (codebook_size, embed_dim)
        # Could be cluster centroids or learned keys
        self.register_buffer('codebook_keys', None)

    def forward(self, x):
        # x: (batch, embed_dim)
        h = self.encoder(x)

        # Project to routing space
        routing_vec = self.routing_head(h)  # (batch, embed_dim)
        routing_vec = F.normalize(routing_vec, dim=-1)

        # Cosine similarity to codebook keys
        # codebook_keys: (codebook_size, embed_dim), normalized
        similarities = torch.mm(routing_vec, self.codebook_keys.T)  # (batch, codebook_size)

        # Top-K selection
        top_k_sim, top_k_idx = similarities.topk(self.top_k, dim=-1)  # (batch, k)

        # Normalize over top-K (not softmax - direct cosine weights)
        weights = top_k_sim / top_k_sim.sum(dim=-1, keepdim=True)  # (batch, k)

        # Gather top-K bivectors and blend
        # codebook: (codebook_size, d, d)
        top_k_bivectors = self.codebook[top_k_idx]  # (batch, k, d, d)
        B = torch.einsum('bk,bkij->bij', weights, top_k_bivectors)  # (batch, d, d)

        # Predict scale
        scale = self.scale_head(h).squeeze(-1)  # (batch,)

        return B, scale
```

### Why Cosine Similarity > Softmax

| Aspect | Softmax on Logits | Cosine Similarity |
|--------|-------------------|-------------------|
| Geometric meaning | None - arbitrary learned space | Distance in embedding space |
| Interpretability | Opaque | "How similar is query to each rotation" |
| Consistency | Different from federated routing | Matches federated model routing |
| Sparse top-K | Arbitrary cutoff | Natural similarity threshold |
| Temperature | Needs tuning | Inherent from embedding geometry |

### Codebook Keys

The codebook keys for routing can be:
1. **Cluster centroids** from federated model (most consistent)
2. **Mean of samples** that use each bivector
3. **Learned keys** (fine-tuned during training)

### Applying the Rotation

```python
def apply_bivector_rotation(x, B, scale):
    """
    Apply rotation via matrix exponential.

    Args:
        x: (batch, d) input vectors
        B: (batch, d, d) antisymmetric matrices (bivectors)
        scale: (batch,) scaling factors

    Returns:
        y: (batch, d) rotated and scaled vectors
    """
    # Matrix exponential: exp(B) gives rotation matrix
    R = torch.matrix_exp(B)  # (batch, d, d)

    # Apply rotation
    y = torch.einsum('bij,bj->bi', R, x)  # (batch, d)

    # Apply scale
    y = y * scale.unsqueeze(-1)

    return y
```

## Computational Benefits

| Aspect | Current (384 planes) | Proposed (D=64 codebook) |
|--------|---------------------|--------------------------|
| Transformer output | 385 | 65 |
| Rotation application | 384 sequential Givens | 1 matrix exp |
| Training target | Compute per-plane angles | Compute basis coefficients |
| Codebook | None | 64 × 768 × 768 (precomputed) |

### Complexity Analysis

**Current approach:**
- Transformer: O(d × n_planes) output
- Apply rotation: O(n_planes × d) sequential operations
- Total per sample: O(d × n_planes)

**Proposed approach:**
- Transformer: O(d × D) output where D << n_planes
- Blend codebook: O(D × d²) but highly parallelizable
- Matrix exp: O(d³) but batched and GPU-accelerated
- Total per sample: O(D × d² + d³) amortized

For D=64, d=768: Current is O(768 × 384) ≈ 295K ops per sample. Proposed is dominated by matrix exp which is highly optimized on GPU.

## Training Strategy

### Phase 1: Build Codebook (Offline)
1. Load federated model with 124 cluster rotations
2. Compute rotation generators (log map)
3. PCA to get D principal bivectors
4. Save codebook

### Phase 2: Project Teacher to Subspace

**Key insight**: Both teacher and student should operate in the same D-dimensional subspace.

```python
def project_teacher_to_subspace(teacher_bivector, basis):
    """
    Project teacher's full bivector onto the D-dimensional principal subspace.

    Args:
        teacher_bivector: (d, d) antisymmetric matrix from federated model
        basis: (D, d, d) principal bivector basis

    Returns:
        coefficients: (D,) projection onto each basis vector
    """
    # Flatten for dot product
    teacher_flat = teacher_bivector.flatten()
    basis_flat = basis.reshape(len(basis), -1)  # (D, d²)

    # Project: coefficients_i = <teacher, basis_i>
    coefficients = basis_flat @ teacher_flat

    return coefficients
```

This ensures:
1. **Errors are meaningful** - measured only in the D directions that matter
2. **Information criteria apply** - can use AIC/BIC to compare model sizes
3. **Same manifold** - teacher and student both constrained to learned subspace
4. **Justifies larger models** - if more capacity reduces error in the subspace, it's real improvement

### Phase 3: Train Transformer
1. For each (query, target) pair:
   - Compute teacher's bivector from federated model
   - **Project onto D-dimensional subspace** (not full space)
   - Train transformer to predict the D coefficients
2. Loss: MSE on coefficients (in subspace) + output reconstruction

### Phase 4: Optional Fine-tuning
- Allow gradients to flow through codebook
- Joint optimization of routing and codebook

### Information-Theoretic Model Selection

With teacher and student in the same subspace, we can properly apply information criteria.

**Note**: As introduced in [QA_TRANSFORMER_DISTILLATION.md](QA_TRANSFORMER_DISTILLATION.md), we use **model capacity (H^L)** as a proxy for effective parameter count rather than raw parameter count. Raw parameter count (millions) would dominate the AIC/BIC penalty term, making it useless for model comparison. Capacity represents the number of discrete routing patterns the model can express.

```python
def compute_model_criteria(model, val_loss, n_samples, capacity):
    """
    Compare models using information criteria.

    Args:
        model: The trained model
        val_loss: Validation loss (in the D-dimensional subspace)
        n_samples: Number of validation samples
        capacity: Model capacity H^L (e.g., 6³=216 for 3-layer, 6-head)
                  Used as proxy for effective parameter count

    See: QA_TRANSFORMER_DISTILLATION.md for capacity-based AIC rationale
    """
    # Log-likelihood proxy (assuming Gaussian errors in subspace)
    log_likelihood = -n_samples * val_loss

    # AIC: penalize capacity lightly
    # Using capacity (not raw params) keeps penalty balanced with likelihood
    aic = 2 * capacity - 2 * log_likelihood

    # BIC: penalize capacity more heavily
    bic = capacity * np.log(n_samples) - 2 * log_likelihood

    return {'aic': aic, 'bic': bic, 'val_loss': val_loss, 'capacity': capacity}
```

This allows principled comparison: a 3-layer model (capacity 216) is justified over a 2-layer model (capacity 36) if it reduces subspace error enough to overcome the capacity penalty.

## Expected Benefits

1. **Faster inference**: Smaller output dimension, batched operations
2. **Better generalization**: Codebook constrains rotations to learned manifold
3. **Interpretability**: Each codebook entry represents a semantic rotation direction
4. **Compatibility**: Works with existing federated model infrastructure

## Implementation Plan

1. [ ] Add `BivectorCodebook` class to `rotation_transformer.py`
2. [ ] Add `build_codebook_from_federated()` utility
3. [ ] Add `--codebook-mode` option to training script
4. [ ] Benchmark against current Givens approach
5. [ ] Document in skill file

## Preliminary Results

Training run with rotation-based federated teacher (3-layer, 4-head, 384 rotation planes):

| Epoch | Loss | Train Cosine |
|-------|------|--------------|
| 1 | 1.754 | 0.095 |
| 20 | 0.728 | 0.739 |
| 40 | 0.371 | 0.776 |
| 60 | 0.235 | 0.908 |
| 80 | 0.183 | 0.852 |
| 100 | 0.160 | 0.908 |

**Test evaluation**: 0.635 cosine similarity (±0.31)

### Note on Evaluation

The test script initially reported this as "Poor" because it compared transformer output against LDA projections. However, this comparison is **not meaningful** because:

1. The transformer was trained against the **rotation-based teacher** (which stays in the rotation manifold)
2. LDA projections use **standard output blending** (which leaves the rotation manifold)
3. These are fundamentally different targets - comparing them conflates model error with manifold mismatch

The correct evaluation compares transformer vs rotation-based teacher on held-out data. The 0.635 test cosine reflects some overfitting (train was 0.908), which can be addressed with:
- Proper train/test split using question rewording redundancy
- Early stopping
- Regularization tuning

**TODO**: Implement proper held-out evaluation against rotation-based teacher.

## Questions/Risks

1. **Matrix exponential cost**: For d=768, matrix exp is O(d³). May need approximations for speed.
2. **Codebook size D**: How many principal directions are needed? Start with D=64, tune.
3. **Sparse routing**: Top-K selection vs full softmax for efficiency?
4. **Gradient flow**: Should codebook be frozen or fine-tuned?

## Theoretical Justification: Why Rotation (Minimal Transformation)?

### The Problem with Standard Blending

Standard multi-head projection blends output vectors:
```
output = Σ wᵢ × (query @ Wᵢ)
       = query @ (Σ wᵢ × Wᵢ)
```

The weighted sum of rotation matrices `Σ wᵢ × Wᵢ` is **not** a rotation matrix. This leaves the rotation manifold and produces an arbitrary linear transform that:
- Distorts angles between vectors
- Distorts distances between vectors
- Doesn't preserve neighborhood relationships

Even normalizing the output only fixes magnitude, not the geometric distortion.

### Minimal Transformation Preserves Geometry

Rotation-based blending:
```
angles = Σ wᵢ × anglesᵢ
output = R(angles) @ query × scale
```

The blended angles still parameterize a valid rotation. This enforces an **isometry** (plus scale) that preserves:
- **Angles**: `cos(Rx, Ry) = cos(x, y)` for any vectors
- **Distances**: `||Rx - Ry|| = ||x - y||`
- **Structure**: Neighborhood relationships are maintained

### Information-Theoretic Argument

The embedding model (nomic, etc.) learned rich semantic structure through massive pretraining. By using rotation:
- **Preserve**: The geometric relationships already learned
- **Learn**: Only the rotation parameters to align query → answer space
- **Maximize**: Mutual information between input and output

Arbitrary transforms can distort pretrained geometry and require more data to relearn structure that was already there. The minimal transformation maximizes information preservation - any deviation from isometry is "adding noise" that must be justified by data.

### Continuity Hierarchy

1. **Basic continuity**: Close inputs → close outputs
2. **Lipschitz continuity**: Bounded rate of change
3. **Minimal transformation**: Preserves relative geometry (strictest)

This matters for semantic search: if query A is "between" queries B and C in embedding space, after projection A should still be "between" the projections of B and C. Standard blending can distort this; rotation preserves it.

## References

- Geometric algebra and bivector representation
- Lie group theory: SO(n) and its tangent space
- VQ-VAE and codebook learning
- Mixture of Experts sparse routing
- Isometry and metric preservation in embeddings

## Appendix A: Matrix Logarithm and the Rotation Manifold

### Why Matrix Logarithm?

The rotation group SO(n) is a **Lie group** - a smooth manifold with group structure. Every Lie group has an associated **Lie algebra** which is its tangent space at the identity.

For SO(n):
- **Lie group**: SO(n) = {R ∈ ℝⁿˣⁿ : RᵀR = I, det(R) = 1} (rotation matrices)
- **Lie algebra**: so(n) = {A ∈ ℝⁿˣⁿ : Aᵀ = -A} (antisymmetric matrices)

The **exponential map** connects them:
```
exp: so(n) → SO(n)
exp(A) = I + A + A²/2! + A³/3! + ... = R
```

The **logarithm map** is its inverse:
```
log: SO(n) → so(n)
log(R) = A  where  exp(A) = R
```

### Why This Matters

1. **Linear operations in the tangent space**: Antisymmetric matrices form a vector space. We can add, scale, and do PCA on them. Rotation matrices do not form a vector space (the sum of two rotations is not a rotation).

2. **Interpolation**: To interpolate between rotations R₁ and R₂:
   - Compute A₁ = log(R₁), A₂ = log(R₂)
   - Interpolate: A_t = (1-t)A₁ + tA₂
   - Result: R_t = exp(A_t) is a valid rotation for all t ∈ [0,1]

3. **Principal components**: PCA on rotation matrices is meaningless. PCA on their logarithms (antisymmetric matrices) finds principal rotation directions.

### The Bivector Connection

In geometric algebra, a **bivector** B represents an oriented plane. For rotations in ℝⁿ:
- A bivector B corresponds to an antisymmetric matrix A
- The rotation by angle θ in the plane of B is: R = exp(θA)
- Components A_ij (i < j) represent rotation in the (i,j) plane

This is why we call log(R) a "bivector" - it encodes both:
- **Which planes** to rotate in (non-zero components)
- **How much** to rotate (magnitude of components)

### Derivation: Logarithm of a Rotation Matrix

For a rotation matrix R ∈ SO(n), the logarithm can be computed via eigendecomposition.

**2D case** (simple):
```
R = [cos θ  -sin θ]    A = log(R) = [  0   -θ]
    [sin θ   cos θ]                 [  θ    0]

where θ = atan2(R₂₁, R₁₁)
```

**General case** (n dimensions):
1. Compute eigendecomposition: R = VΛV⁻¹
2. Eigenvalues of rotation are e^{±iθₖ} for rotation angles θₖ
3. log(R) = V·log(Λ)·V⁻¹ where log(e^{±iθ}) = ±iθ
4. Result is real and antisymmetric (imaginary parts cancel)

In practice, use `scipy.linalg.logm(R)` and enforce antisymmetry:
```python
A = scipy.linalg.logm(R)
A = (A - A.T) / 2  # Ensure antisymmetric (numerical stability)
```

### Citations

- **Lie Groups and Lie Algebras**: Hall, B.C. (2015). *Lie Groups, Lie Algebras, and Representations*. Springer. Chapter 2-3 cover the exponential map.

- **Matrix Exponential/Logarithm**: Higham, N.J. (2008). *Functions of Matrices: Theory and Computation*. SIAM. Chapter 11 covers the matrix logarithm.

- **Rotation Interpolation**: Shoemake, K. (1985). "Animating Rotation with Quaternion Curves". *SIGGRAPH*. While focused on quaternions, establishes why interpolation must happen in the tangent space.

- **SO(n) Geometry**: Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds*. Princeton. Chapter 3 covers SO(n) as a Riemannian manifold.
