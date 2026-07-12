# Joint square-root / QR conditioner — inverse-root updates by Householder triangularisation

Strictly, the factors below are covariance/information (precision) factors. A square root of the inverse
correlation matrix is the standardized, unit-variance special case; the same QR update applies after the state
and measurements are scaled to correlation coordinates.

## Scope

This is a numerically robust implementation of the **same dense correlated Gaussian conditioner** used by
`gaussian_condition_update`. It does not change the fusion model, make sources independent, or supply a new
confidence heuristic. Its purpose is to maintain a triangular square root of the **precision** matrix and to
accept new conditionally independent measurement blocks without explicitly inverting the posterior covariance.

The name is deliberately descriptive. Potter and Carlson are historical antecedents, but classical Carlson is
a triangular covariance-root recursion for scalar measurements. The vector, nonzero-cross-covariance algorithm
here is more accurately called a **joint square-root information / QR conditioner**.

Householder transformations are reflections; Givens transformations are plane rotations. Either can perform
the orthogonal triangularisation. Householder QR is attractive for dense blocks; Givens is attractive when rows
arrive one at a time.

For a fixed corpus/rung, `H`, `P`, `C`, and `R` are constant across rows. The coefficient pre-array and its
Householder reflectors can therefore be compiled once and reused; only the right-hand side changes per row.
That optimization is exact. By contrast, when a later measurement block is assimilated for the same state,
the new block must start from the previous `U_post`, because the first block changed `P`. Streaming independent
blocks this way is algebraically equivalent to one batch QR.

## 1. Correlated static measurement model

Use the PR #3648 residual convention:

```
e = truth - prior
v = measurement - H truth
Cov([e, v]) = [[P, C], [C^T, R]]
r = measurement - H prior = H e + v
```

Maintain a factor `U` such that

```
P^-1 = U^T U.
```

Conditioning `v` on `e` removes the prior/measurement cross-covariance:

```
Rc = R - C^T P^-1 C
J  = H + C^T P^-1
r | e ~ Normal(J e, Rc).
```

Using the maintained factor avoids forming `P^-1`:

```
UC = U C
Rc = R - (UC)^T (UC)
J  = H + (UC)^T U.
```

`Rc` is the Schur complement of `P`. A materially non-positive `Rc` means the proposed joint covariance is
invalid. Small round-off defects may be regularised; a false statistical covariance must not be hidden.

## 2. Householder pre-array

Let `L L^T = Rc`, and whiten the measurement block:

```
A = L^-1 J
b = L^-1 r.
```

For a zero-mean prior error, `z_prior = 0`. More generally `z_prior = U m_prior`. Form

```
[ U_prior   z_prior ]
[ A         b       ].
```

Apply Householder reflectors to the coefficient columns and the RHS together, without forming `Q`:

```
Q^T [ U_prior   z_prior ] = [ U_post   z_post ]
    [ A         b       ]   [ 0        residual ].
```

Then

```
Lambda_post = U_post^T U_post
e_post      = solve(U_post, z_post)
x_post      = x_prior + e_post.
```

The implementation normalises the diagonal of `U_post` to be non-negative, making the root deterministic
under row permutations up to floating-point error.

## 3. Block updates

If several likelihood blocks are conditionally independent, stack all whitened rows at once or feed them
sequentially by reusing `(U_post, z_post)` as the next prior. Batch and streamed QR are algebraically identical.

The independence condition is **block diagonality of `Rc`**, not raw `R`:

```
Rc_ij = R_ij - C_i^T P^-1 C_j.
```

Therefore raw measurement families can be uncorrelated while their conditional residuals remain coupled through
the shared prior. Estimate and test the off-block entries of `Rc` on held-out, node-disjoint calibration data
before enabling streamed bundle updates. The state prior is inserted exactly once; never repeat a complete
`[[P,C_i],[C_i^T,R_i]]` block for every bundle.

The streamed `(U_post,z_post)` state remains in one fixed state coordinate. If every `b_i` is formed relative
to the original prior origin, carry both `U_post` and `z_post`. If instead the state is recentered at the first
posterior solution `e_hat`, reset the information RHS to zero and shift a later whitened RHS to
`b_i - A_i e_hat`; carrying `z_post` and recentering would double-count the earlier block. With nonzero `C`,
that shift uses the conditional design `J` (through `A=L^-1 J`), not merely the physical observation matrix
`H`.

## 4. Relationship to other alternatives

- **Compiled dense-gain conditioner:** same posterior and the correct fixed-design throughput baseline; cache
  `K` and the posterior covariance, then apply only the affine innovation update per row.
- **JointPosterior:** different statistical model; can learn nonlinear interactions and must be compared on a
  common held-out decision metric.
- **Factored PoE / hand weights:** control only; unsafe when shared e5 or judge lineage correlates errors.
- **Softmax/MoE gating across experts:** learned nonlinear alternative, not a numerical square-root method.
- **Pooling within a judge:** the current hard `max` over directional or symmetric relation scores is separate
  from across-expert fusion. Hard max versus weighted or temperature/log-sum-exp pooling deserves its own
  ablation; the square-root conditioner accepts whichever calibrated channel definition wins.

## 5. Batched CPU/GPU implementation

`joint_square_root_conditioner_torch.py` implements two paths:

- a generic batched design path for different `(U,H,R,C)` matrices;
- a throughput path that compiles one fixed design with `torch.geqrf`, stores the compact Householder vectors,
  applies `Q^T` to many innovation right-hand sides with `torch.ormqr`, and never materialises `Q`.

The second path is a valid GPU formulation, but the matched static baseline changes the decision. If
`P,H,C,R` are fixed, the dense Gaussian gain and posterior covariance can be compiled once; each row then needs
only the affine innovation update. On the local GTX 1660 SUPER, that compiled dense path beats QR on both CPU
and CUDA in every tested fixed-design cell. The present two-state/four-measurement case is fastest on CPU dense;
large batches and states do benefit from CUDA, but CUDA dense remains substantially faster than CUDA QR.

QR's distinct value is carrying a numerically stable information root through changing/sequential blocks. A
later measurement block must start from `(U_post,z_post)`, so the original static gain is no longer reusable.
See `REPORT_joint_square_root_qr_benchmark.md` for the matched static table. A separate sequential benchmark is
required before choosing the root-threading implementation for production.

## 6. Acceptance tests

1. Match `gaussian_condition_update` for random valid dense joint covariances with nonzero `C`.
2. Match batch and sequential processing when `Rc` is block diagonal.
3. Be invariant to measurement-row permutations after root-sign normalisation.
4. Preserve a positive precision root on ill-conditioned but valid problems.
5. Reject a materially invalid Schur complement.
6. Measure numerical error versus the dense implementation in float64 and float32 before replacing production
   code; statistical metrics should be identical within numerical tolerance.
7. Sweep covariance scales and condition numbers logarithmically; an absolute float32 regularisation floor
   alone is not evidence of scale-independent stability.
