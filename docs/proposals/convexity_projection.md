# Proposal: Convexity-Blended Projection

## Motivation

The density explorer's 2D projection currently uses SVD, which maximizes variance
explained. This spreads points out well but doesn't optimize for *hierarchy
visibility*. When the root node sits at a density peak with branches radiating
outward, the most informative 2D view is one where that branching structure is
most visible — where the density contours at the root are maximally convex.

An equal-weight L^n consensus MST already produces high-confidence long branches
(Physics → List of physicists → David Lee, Physics → Nuclear Physics → Nuclear).
A convexity-optimized projection would complement this by finding the 2D viewing
angle that best displays those branches.

## Geometric Foundation

### The Ridge Model of Branches

At the root node, the density surface doesn't look like a simple peak. If
multiple branches radiate outward, the density forms a **star-shaped ridge
system** — like a mountain summit where several ridgelines extend outward:

- **Along each branch**: density falls off gently (a ridge — the gentle path)
- **Between branches**: density drops sharply (steep banks)
- **Contour lines**: convex (curving outward) in the gaps between branches,
  indicating the valleys between ridges

The key insight is that branch visibility is about **contour line convexity**,
not about the steepness of the density peak itself. A point where the gradient
falls off least steeply is a ridge — a gentle path with steep banks on either
side. The contour lines at such a point curve outward (are convex), tracing the
shape of the valley walls.

### Two Kinds of Curvature

The Hessian H of the density function measures curvature of the function itself:

- **Most negative eigenvalues** → directions of steepest descent → point into
  the valleys *between* branches
- **Least negative eigenvalues** (near zero) → directions of gentlest descent →
  point *along* branches (the ridges)

But we care about the curvature of the **level sets** (contour lines), which is
a different quantity. For a level set f(x) = c at a point where ∇f ≠ 0, the
principal curvatures are the eigenvalues of the **shape operator**:

```
S = -(I - n̂n̂ᵀ) · H · (I - n̂n̂ᵀ) / |∇f|
```

where n̂ = ∇f/|∇f| is the unit normal to the level set. The shape operator
projects the Hessian onto the tangent plane of the level set and scales by the
gradient magnitude.

For our purpose: we want the 2D viewing plane where the projected contour lines
are most convex at the root. This means maximizing the curvature of the level
set of the *projected* 2D density function.

### Choosing a 2D Plane as a Bivector

Choosing a 2D viewing plane means choosing a bivector u ∧ v (wedge product of
two orthonormal vectors). The projected density on this plane inherits curvature
from the full-dimensional density. The projected Hessian is:

```
H_projected = [uᵀHu  uᵀHv]
              [vᵀHu  vᵀHv]
```

The determinant det(H_projected) = (uᵀHu)(vᵀHv) - (uᵀHv)² measures the
Gaussian curvature of the density surface in the chosen view. However, what we
actually want is the curvature of the *contour lines* in the projected view,
which depends on both the projected Hessian and the projected gradient.

### Connection to Wedge Products and Exterior Algebra

The curvature of level sets in n dimensions is governed by a 2-form. In the
eigendecomposition of H, each pair of eigenvectors (e_i, e_j) defines a
curvature bivector with magnitude λ_i · λ_j. The Hodge dual of this 2-form is
an (n-2)-form representing the normal space to the viewing plane.

The curl in n dimensions is the exterior derivative of a 1-form, giving a
2-form (antisymmetric tensor built from wedge products). While the Hessian
itself is symmetric (and the curl of a gradient is zero: d(df) = 0), the
*choice of optimal viewing plane* is a problem on the Grassmannian Gr(2, n),
which is parameterized by bivectors. Finding the best 2D plane for contour
convexity is finding the bivector that maximizes a curvature functional —
this is genuinely exterior algebra applied to visualization.

### Connection to the Bivector Model

The Bivector Paired transformation model operates on bivectors as its fundamental
object. The optimal convexity viewing plane is also a bivector. There may be a
deeper relationship: the model's learned bivector transformations could be
rotating embeddings toward directions of maximal hierarchical curvature. This
would mean the model is implicitly learning the best viewing angle for hierarchy.

### Which Directions to Include

Because we want to see **ridges** (branches) in the 2D view, the optimal plane
should contain:

- Directions where the density *gradient changes character* — transitioning from
  gentle (along-branch) to steep (between-branch)
- These are the directions where contour lines have maximum curvature

In practice, this is likely a mix of SVD components. The top SVD directions
capture maximum variance (spread), while mid-range components may capture the
ridge structure better. The optimization must search across this spectrum.

## Algorithm

### Phase 1: Hessian and Gradient at Root

Compute both the gradient and Hessian of the KDE at the root point. For
Gaussian KDE with bandwidth h, these have closed-form expressions:

```python
def kde_gradient_and_hessian(x, data, bandwidth):
    """
    Compute gradient and Hessian of KDE at point x.

    For Gaussian kernel K(u) = exp(-||u||²/2):
      ∇f(x)  = -(1/n·h^(d+2)) Σᵢ (x-xᵢ)/h² · K((x-xᵢ)/h)
      H_f(x) =  (1/n·h^(d+2)) Σᵢ [(x-xᵢ)(x-xᵢ)ᵀ/h² - I] · K((x-xᵢ)/h)
    """
    n, d = data.shape
    grad = np.zeros(d)
    H = np.zeros((d, d))
    for i in range(n):
        diff = (x - data[i]) / bandwidth
        k = np.exp(-0.5 * np.sum(diff ** 2))
        grad -= diff * k / bandwidth
        H += (np.outer(diff, diff) - np.eye(d)) * k
    grad /= (n * bandwidth ** (d + 2))
    H /= (n * bandwidth ** (d + 2))
    return grad, H
```

We need both because contour line curvature depends on the gradient direction
(normal to the level set) and the Hessian (how the gradient changes).

**Problem:** This is O(N·D²) and D=768 makes the D² term expensive. A 768×768
Hessian has ~295K entries.

**Solution:** Work in a reduced subspace.

### Phase 2: Reduced Subspace

Instead of computing the full 768×768 Hessian, work in the top-k SVD subspace:

1. Compute SVD: U, S, Vt = svd(centered_embeddings)
2. Project to top-k components (e.g., k=20): X_k = centered @ Vt[:k].T
3. Compute KDE Hessian in k-dimensional space (20×20 = 400 entries)
4. Find top-2 eigenvectors of -H_k (directions of maximum curvature)
5. Map back to original space: optimal_plane = Vt[:k].T @ eigvecs[:, :2]

This is tractable: a 20×20 Hessian with 200-300 points costs ~20²×300 = 120K
operations.

### Phase 3: Blended Objective

The user controls the balance between variance and convexity. Both objectives
produce 2D coordinates, and we blend:

**Option A: Blend at 2D level**

```python
pts_var = svd_project_2d(embeddings)     # max variance
pts_conv = hessian_project_2d(embeddings) # max convexity
pts_blend = alpha * pts_var + (1-alpha) * pts_conv
```

Simple but may produce artifacts (two different coordinate systems being mixed).

**Option B: Blend in subspace (recommended)**

Within the top-k SVD subspace, find a 2D plane that optimizes:

```
score(u,v) = (w_var · variance(u,v)^n + w_conv · convexity(u,v)^n)^(1/n)
```

where:
- variance(u,v) = σ_u² + σ_v² (variance captured by the plane)
- convexity(u,v) = det(projected Hessian) or min eigenvalue
- w_var, w_conv are user weights (normalized to sum to 1)
- n is the L^n exponent

This requires optimization over the Grassmannian Gr(2, k). For k=20, this is a
36-dimensional space. Approaches:

- **Grid search over Givens rotations**: Parameterize 2D planes via sequences of
  Givens rotations. Sample densely enough for a good solution.
- **Gradient descent on Stiefel manifold**: Use Riemannian optimization (e.g.,
  pymanopt or manual retraction). More precise but adds a dependency.
- **Greedy selection**: Pick direction 1 as the direction maximizing
  w_var·σ² + w_conv·|λ|, then pick direction 2 as the orthogonal direction
  maximizing the same. Simple, fast, may not find the global optimum.

**Recommended**: Start with greedy selection (fast, no dependencies), add
gradient-based optimization later if needed.

### Phase 4: Convexity Metrics

For a candidate 2D plane (u, v), we project the gradient and Hessian, then
compute the curvature of the contour lines at the root in the projected view.

The projected gradient g = [uᵀ∇f, vᵀ∇f] and Hessian H_2×2 give the 2D density
function's local structure. The contour line curvature at a non-critical point
(where g ≠ 0) is:

```
κ_contour = (g_x² H_yy - 2 g_x g_y H_xy + g_y² H_xx) / |g|³
```

At or near a critical point (density peak, where g ≈ 0), the contour lines
become approximately elliptical and their curvature is governed by the Hessian
eigenvalues directly. Several metrics are available:

| Metric | Formula | Behavior |
|--------|---------|----------|
| Determinant | λ₁·λ₂ | Both directions must be convex; zero if either is flat |
| Min eigenvalue | min(\|λ₁\|, \|λ₂\|) | Weakest direction; ensures no flat axis |
| Trace (Laplacian) | \|λ₁\| + \|λ₂\| | Total curvature; allows one flat + one sharp |
| Geometric mean | √(\|λ₁·λ₂\|) | Balanced; less sensitive to outlier eigenvalues |
| Contour curvature | κ_contour above | Curvature of actual contour line at root |

**Recommendation**: For roots at density peaks (g ≈ 0), use determinant. For
roots on ridges (g ≠ 0, which is the more common and interesting case for branch
visualization), use contour curvature κ. The ridge case is where the "gentle
path with steep banks" geometry applies — the root sits on a density ridge with
branches extending outward.

## Interaction with Existing Controls

### Bandwidth Dependency

Convexity depends on KDE bandwidth. A tight bandwidth makes every point a sharp
peak; a wide bandwidth smooths everything out. The convexity measure must use the
same bandwidth as the density display, or the optimization would find a projection
that looks wrong when rendered.

**Solution**: Compute convexity using the auto-selected bandwidth (Scott's rule)
or the user's manual bandwidth. If the user changes bandwidth after applying
convexity projection, show a warning that the projection is stale.

### Root Selection

Different roots produce different optimal projections. This is a feature: "show
me the best view from this node's perspective." Changing the root should trigger
a re-optimization of the projection.

**Integration**: When the user sets a root and has convexity blend active,
recompute the projection automatically (or show a "Recompute" prompt).

### Relationship to Blend Tab

The Blend tab controls distance space blending (L^n across embedding, weights,
learned, wiki). The convexity projection is orthogonal — it controls which 2D
plane to project onto, not how distances are computed. Both can be active
simultaneously:

- Blend tab: "Use these distances for the tree"
- Projection tab: "View from this angle"

## UI Design

### New Projection Tab (or section in Data tab)

```
┌──────────────────────────────────────┐
│ PROJECTION                           │
│                                      │
│ Method: [Max Variance (SVD)      ▾]  │
│                                      │
│ ─── Convexity Blend controls ─────── │
│ (shown when method = Convexity Blend)│
│                                      │
│ Root for convexity: [Physics      ]  │
│ (auto-set from tree root selection)  │
│                                      │
│ Variance ◄═══|════► Convexity        │
│           [0.50]                     │
│                                      │
│ Metric: [Determinant            ▾]   │
│   Options: Determinant, Min eigen,   │
│            Trace, Geometric mean     │
│                                      │
│ n: [2.0]  Presets: [Linear] [Euclid] │
│                                      │
│ Subspace dim: [20]  (advanced)       │
│                                      │
│ ──────────────────────────────────── │
│ Variance explained: 7.6%             │
│ Convexity score: 0.043               │
│ (vs SVD variance: 7.6%, conv: 0.012) │
│                                      │
│ [Apply Projection]                   │
└──────────────────────────────────────┘
```

Key features:
- **Method dropdown**: Max Variance (SVD) is default, Convexity Blend shows the
  additional controls
- **Variance↔Convexity slider**: At 0.0, recovers SVD. At 1.0, pure max
  convexity. In between, L^n blend.
- **Metrics display**: Show both variance and convexity for the current
  projection, plus what SVD alone would give, so the user can see the tradeoff.
- **Root auto-sync**: Uses the tree root selection from the View tab.

### Projection Method Options

```
Max Variance (SVD)        ← current default
Convexity Blend           ← new
Custom Axis               ← existing (click two points)
MDS (distance preserving) ← existing for wiki physics
```

## Implementation Plan

### Backend (`density_core.py`)

1. **Add `kde_hessian_at_point()`**: Compute Hessian in reduced subspace
2. **Add `convexity_project_2d()`**: Find optimal 2D plane blending variance and
   convexity using greedy selection in top-k SVD subspace
3. **Extend `compute_density_manifold()`**: New projection mode
   "convexity_blend" with params: root_id, blend_alpha, convexity_metric,
   power_n, subspace_k

### Backend (`flask_api.py`)

4. **Add "convexity_blend" to projection_mode validation**
5. **Extract convexity params** from request (alpha, metric, n, k)

### Frontend (`index.html`)

6. **Add Convexity Blend option** to projection mode dropdown
7. **Add convexity controls** (shown/hidden based on mode selection)
8. **Display variance/convexity metrics** in the info bar

## Open Questions

1. **Greedy vs optimization**: Is greedy selection (pick best direction 1, then
   best orthogonal direction 2) good enough, or do we need Grassmannian
   optimization?

2. **Bandwidth coupling**: Should the convexity projection auto-recompute when
   bandwidth changes, or is that too expensive?

3. **Subspace dimension k**: Is k=20 sufficient? The variance falls off quickly
   (top-2 SVD captures ~7%), but curvature might live in higher components. Should
   we auto-select k based on cumulative variance threshold?

4. **Tab vs section**: Should this be a new "Projection" tab, or a section within
   the existing Data tab's projection controls?

5. **Performance**: For 300 points in 20D subspace, the Hessian computation is
   fast (~ms). But if we add grid search over rotations, it could take seconds.
   Is that acceptable for an "Apply" button workflow?

6. **Theoretical depth**: The proposal about bivector model transformations
   aligning with convexity directions is speculative. Should we add a diagnostic
   that compares the model's learned bivector with the optimal convexity bivector?
