# Proposal: Integrated Tangent Deviation Criterion for Graph Attachment

## Abstract

This proposal develops a geometrically-principled criterion for evaluating graph attachments in embedding spaces. By modeling normalized embeddings as points on a unit hypersphere, we derive bounds on tangent deviation using Riemannian geometry and define an integral criterion over elliptical regions of influence.

## 1. Introduction

When attaching orphan nodes to an existing graph structure (e.g., curated hierarchy), we need to balance:
1. **Semantic distance** - attach to nearby points in embedding space
2. **Structural preservation** - minimize distortion of local geometry

The tangent deviation metric captures structural distortion by comparing local differential structure before and after attachment. This proposal derives theoretical bounds and an integral optimization criterion.

## 2. Mathematical Background

### 2.1 Embeddings on the Unit Hypersphere

Modern embedding models (BERT, Sentence Transformers, Nomic) typically produce normalized vectors for cosine similarity computation [1]. Normalized embeddings lie on the unit hypersphere:

$$S^{n-1} = \{x \in \mathbb{R}^n : \|x\| = 1\}$$

**Distance measures:**
- Cosine distance: $d_c = 1 - \cos(\alpha) = 1 - x \cdot y$
- Geodesic distance: $d_g = \alpha = \arccos(x \cdot y)$
- Euclidean (chord): $d_e = \|x - y\| = \sqrt{2(1 - \cos\alpha)} = \sqrt{2d_c}$

For small angles: $d_g \approx d_e \approx \sqrt{2d_c}$

### 2.2 Riemannian Geometry of the Sphere

The unit sphere $S^{n-1}$ is a Riemannian manifold with constant sectional curvature $K = 1$ [2].

**Tangent space:** At point $p \in S^{n-1}$, the tangent space is:
$$T_p S^{n-1} = \{v \in \mathbb{R}^n : v \cdot p = 0\}$$

**Exponential map:** Maps tangent vectors to points on the sphere:
$$\exp_p(v) = \cos(\|v\|) \cdot p + \sin(\|v\|) \cdot \frac{v}{\|v\|}$$

**Parallel transport:** Moving a tangent vector along a geodesic. On a sphere, parallel transport around a closed loop of area $A$ rotates the vector by angle $\theta = K \cdot A = A$ (Gauss-Bonnet theorem) [3].

### 2.3 Taylor Series on Manifolds

For a function $f: M \to \mathbb{R}$ on a Riemannian manifold, the Taylor expansion at $p$ along geodesic with initial velocity $v$ is [4]:

$$f(\exp_p(tv)) = f(p) + t \cdot df_p(v) + \frac{t^2}{2} \nabla^2 f(v,v) + O(t^3)$$

The remainder (error) when using the first-order approximation:
$$R_1 = \frac{t^2}{2} \nabla^2 f(v,v) + O(t^3)$$

On a manifold with bounded curvature $|K| \leq \kappa$:
$$|R_1| \leq C \cdot \kappa \cdot t^2$$

for some constant $C$ depending on $f$ [5].

## 3. Tangent Deviation Metric

### 3.1 Definition

For a node $N$ with neighbor set $\mathcal{N}(N)$, the **tangent vector** at $N$ is the average direction to neighbors projected onto the tangent space:

$$\tau_N = \frac{1}{|\mathcal{N}(N)|} \sum_{i \in \mathcal{N}(N)} \Pi_{T_N}(x_i - x_N)$$

where $\Pi_{T_N}$ is projection onto $T_N S^{n-1}$.

For normalized embeddings, this simplifies to:
$$\tau_N = \frac{1}{|\mathcal{N}(N)|} \sum_{i \in \mathcal{N}(N)} (x_i - (x_i \cdot x_N) x_N)$$

**Tangent deviation** between two graph structures (curated vs. final):
$$\theta_N = 1 - \frac{\tau_N^{curated} \cdot \tau_N^{final}}{\|\tau_N^{curated}\| \|\tau_N^{final}\|}$$

This measures $1 - \cos(\angle(\tau^{curated}, \tau^{final}))$, ranging from 0 (identical) to 2 (opposite).

### 3.2 Connection to Differential Geometry

The tangent vector $\tau_N$ approximates the gradient of a "potential function" induced by the graph structure. The graph Laplacian $L = D - A$ is the discrete analog of the Laplace-Beltrami operator [6]:

$$\Delta f(p) \approx \sum_{j \sim i} w_{ij}(f(p_j) - f(p_i))$$

Tangent deviation measures how much the induced differential operator changes between graph structures.

## 4. Spherical Geometry Bounds

### 4.1 Curvature-Induced Tangent Rotation

On a unit sphere ($K = 1$), parallel transport of a tangent vector around a region of area $A$ rotates it by angle $A$ radians.

For an attachment creating edge $(N, O)$ with geodesic separation $d_g$, the "zone of influence" can be modeled as an ellipse with foci at $N$ and $O$.

**Ellipse approximation:** For small $d_g$, approximate as a circle of radius $r \approx d_g/2$:
$$A_{influence} \approx \pi r^2 = \frac{\pi d_g^2}{4}$$

**Intrinsic tangent deviation** due to sphere curvature alone:
$$\theta_{curvature} = K \cdot A = \frac{\pi d_g^2}{4} \approx 0.785 \cdot d_g^2$$

### 4.2 Maximum Bound

The tangent deviation should not significantly exceed the curvature-induced baseline. We propose:

$$\theta \leq c \cdot d_g^2$$

where $c \sim O(1)$ is a tunable constant. Setting $c = 1$:

$$\theta_{max} = d_g^2 = [\arccos(1 - d_c)]^2$$

**Table: Maximum tangent deviation by distance**

| Cosine dist $d_c$ | Geodesic $d_g$ (rad) | $\theta_{max}$ |
|-------------------|----------------------|----------------|
| 0.05 | 0.32 | 0.10 |
| 0.10 | 0.45 | 0.20 |
| 0.20 | 0.64 | 0.41 |
| 0.30 | 0.80 | 0.64 |
| 0.50 | 1.05 | 1.10 |

### 4.3 Second-Order Error Bound

From Taylor series analysis, the approximation error at distance $d_g$ with tangent deviation $\theta$:

$$\epsilon(d_g, \theta) \leq \frac{\theta \cdot d_g^2}{2}$$

This follows from the Taylor remainder bound on manifolds with the Hessian term related to $\theta$.

## 5. Integrated Criterion

### 5.1 Elliptical Region of Influence

When attaching orphan $O$ to node $N$, define the region of influence as an ellipse $\mathcal{E}$ with:
- **Foci:** $N$ and $O$
- **Semi-major axis:** $a$ (tunable, default $a = d_g$)
- **Definition:** $\mathcal{E} = \{P : d(P,N) + d(P,O) \leq 2a\}$

Points inside $\mathcal{E}$ are "near" the edge and affected by the attachment.

### 5.2 Weighted Integral Formulation

The integrated error over the ellipse:

$$E_{integrated} = \frac{1}{|\mathcal{E}|} \iint_{\mathcal{E}} \epsilon(r_P, \theta_P) \cdot w(P) \, dA$$

where:
- $r_P = $ distance from $P$ to nearest point on edge $(N, O)$
- $\theta_P = $ tangent deviation at $P$
- $w(P) = $ weight function (higher near edge center)

**Weight function:** Using the ellipse geometry:
$$w(P) = 1 - \frac{d(P,N) + d(P,O) - d(N,O)}{2a - d(N,O)}$$

This gives $w = 1$ at points on the line segment $NO$ and $w = 0$ at the ellipse boundary.

### 5.3 Practical Discrete Approximation

For a discrete graph with nodes $\{P_i\}$:

$$E_{integrated} \approx \frac{\sum_{P_i \in \mathcal{E}} \theta_{P_i} \cdot r_{P_i}^2 \cdot w(P_i)}{\sum_{P_i \in \mathcal{E}} w(P_i)}$$

**Algorithm:**
```python
def integrated_error(N, O, embeddings, adjacency, curated_adj, a_factor=1.0):
    """
    Compute integrated tangent deviation error over ellipse region.

    Args:
        N, O: Indices of attachment point and orphan
        embeddings: Normalized embedding matrix (N x D)
        adjacency: Current graph adjacency
        curated_adj: Original curated adjacency
        a_factor: Semi-major axis as multiple of d_g

    Returns:
        Integrated error (lower is better)
    """
    # Geodesic distance between N and O
    d_g = np.arccos(np.clip(embeddings[N] @ embeddings[O], -1, 1))
    a = a_factor * d_g  # Semi-major axis

    weighted_error = 0.0
    weight_sum = 0.0

    for P in range(len(embeddings)):
        # Distance to foci
        d_PN = np.arccos(np.clip(embeddings[P] @ embeddings[N], -1, 1))
        d_PO = np.arccos(np.clip(embeddings[P] @ embeddings[O], -1, 1))

        # Check if inside ellipse
        if d_PN + d_PO > 2 * a:
            continue

        # Weight (1 at center, 0 at boundary)
        w = 1 - (d_PN + d_PO - d_g) / (2*a - d_g + 1e-8)
        w = max(0, w)

        # Tangent deviation at P
        theta_P = tangent_deviation_at_node(P, adjacency, curated_adj, embeddings)

        # Distance to edge (approximate as min to either focus)
        r_P = min(d_PN, d_PO)

        # Error contribution: θ · r² / 2
        error_P = theta_P * r_P**2 / 2

        weighted_error += error_P * w
        weight_sum += w

    return weighted_error / weight_sum if weight_sum > 0 else 0.0
```

### 5.4 Optimization Criterion

For attachment decisions, minimize:

$$\mathcal{L}(N, O) = d_g(N, O) + \lambda \cdot E_{integrated}(N, O)$$

where:
- First term: semantic distance (prefer nearby attachments)
- Second term: integrated structural distortion
- $\lambda$: trade-off parameter (default $\lambda = 1$ for balanced weighting)

**Constraint form:** Alternatively, minimize $d_g$ subject to:
$$E_{integrated} \leq \epsilon_{max}$$

where $\epsilon_{max}$ is derived from the max bound:
$$\epsilon_{max} = \frac{c \cdot d_g^2 \cdot d_g^2}{2} = \frac{c \cdot d_g^4}{2}$$

## 6. Theoretical Properties

### 6.1 Scale Invariance

The criterion is scale-invariant in the following sense:
- Tangent deviation $\theta$ is dimensionless (cosine-based)
- Geodesic distance $d_g$ is in radians (bounded $[0, \pi]$)
- The ratio $\theta / d_g^2$ is a dimensionless "excess deviation"

### 6.2 Consistency with Sphere Geometry

When $\theta = \theta_{curvature} = \pi d_g^2 / 4$:
$$E_{integrated} \propto \frac{\pi d_g^2}{4} \cdot d_g^2 = \frac{\pi d_g^4}{4}$$

This represents the "baseline" error from sphere curvature alone.

### 6.3 Asymptotic Behavior

For small distances ($d_g \to 0$):
- $\theta_{max} \to 0$ (tight bound)
- $E_{integrated} \to 0$ (vanishing error)

For large distances ($d_g \to \pi/2$):
- $\theta_{max} \to \pi^2/4 \approx 2.47$
- Higher tolerance for tangent deviation at larger distances

## 7. Implementation Notes

### 7.1 Computational Complexity

- **Per-attachment evaluation:** $O(|\mathcal{E}|)$ where $|\mathcal{E}|$ is nodes in ellipse
- **Sparse optimization:** Only consider nodes within $2a$ geodesic distance
- **Incremental updates:** After attachment, only affected nodes need recomputation

### 7.2 Approximations for Efficiency

1. **k-NN ellipse:** Only consider k nearest neighbors to the edge
2. **Sampling:** Monte Carlo integration over ellipse region
3. **Cached tangents:** Precompute tangent vectors, update incrementally

## 8. Future Work

1. **Higher-order bounds:** Extend to second-order tangent information (curvature tensor)
2. **Non-spherical manifolds:** Generalize to embeddings on other manifolds (hyperbolic, product spaces)
3. **Adaptive $\lambda$:** Learn optimal trade-off from data
4. **Spectral analysis:** Connect to spectral graph theory (Cheeger inequality, spectral gap)

## References

[1] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.

[2] do Carmo, M. P. (1992). Riemannian Geometry. Birkhäuser Boston.

[3] Spivak, M. (1979). A Comprehensive Introduction to Differential Geometry, Vol. 2. Publish or Perish.

[4] Lee, J. M. (2018). Introduction to Riemannian Manifolds (2nd ed.). Springer.

[5] Absil, P. A., Mahony, R., & Sepulchre, R. (2008). Optimization Algorithms on Matrix Manifolds. Princeton University Press.

[6] Chung, F. R. K. (1997). Spectral Graph Theory. American Mathematical Society.

[7] Belkin, M., & Niyogi, P. (2003). Laplacian Eigenmaps for Dimensionality Reduction and Data Representation. Neural Computation, 15(6), 1373-1396.

[8] Tenenbaum, J. B., De Silva, V., & Langford, J. C. (2000). A Global Geometric Framework for Nonlinear Dimensionality Reduction. Science, 290(5500), 2319-2323.

---

## Appendix A: Derivation of Curvature-Induced Rotation

On a surface with Gaussian curvature $K$, the holonomy (rotation after parallel transport around a closed loop) is given by the Gauss-Bonnet theorem:

$$\oint_{\partial R} \kappa_g \, ds + \iint_R K \, dA = 2\pi \chi(R)$$

For a simply-connected region $R$ (topological disk), $\chi(R) = 1$ and geodesic curvature $\kappa_g = 0$ for geodesic boundary:

$$\iint_R K \, dA = 2\pi - \sum_i \alpha_i$$

where $\alpha_i$ are exterior angles. For a smooth region:

$$\text{holonomy} = \iint_R K \, dA$$

On unit sphere ($K = 1$): holonomy $= A$ (area of region).

## Appendix B: Geodesic Distance Conversion

For points $x, y$ on unit sphere with cosine similarity $\cos\alpha = x \cdot y$:

| Cosine sim | Cosine dist $d_c$ | Geodesic $d_g$ (rad) | Geodesic (deg) |
|------------|-------------------|----------------------|----------------|
| 1.00 | 0.00 | 0.00 | 0° |
| 0.95 | 0.05 | 0.32 | 18° |
| 0.90 | 0.10 | 0.45 | 26° |
| 0.80 | 0.20 | 0.64 | 37° |
| 0.70 | 0.30 | 0.80 | 46° |
| 0.50 | 0.50 | 1.05 | 60° |
| 0.00 | 1.00 | 1.57 | 90° |

Conversion: $d_g = \arccos(1 - d_c)$
