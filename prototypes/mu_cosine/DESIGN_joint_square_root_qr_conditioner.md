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

This is the precise correlation-to-inverse-root step.  If the **conditional** measurement-error model is
supplied as a correlation matrix `Kc` and marginal conditional standard deviations `s`, construct

```
Rc = diag(s) Kc diag(s).
```

If instead calibration supplies the raw joint correlation of prior error `e` and measurement noise `v`, first
restore covariance units blockwise,

```
P = diag(s_e) K_ee diag(s_e)
C = diag(s_e) K_ev diag(s_v)
R = diag(s_v) K_vv diag(s_v),
```

and only then form the Schur complement `Rc = R - C^T P^-1 C`.  With nonzero `C`, raw measurement correlation
`K_vv` is not generally the conditional correlation `Kc` that must be whitened.

The operator `W = L^-1` is a left square root of the measurement precision because

```
W^T W = Rc^-1
W Rc W^T = I.
```

The implementation should normally **apply** `W` with triangular solves (`solve(L, J)` and `solve(L, r)`) rather
than materialise either `W` or `Rc^-1`.  An explicit inverse square root is useful for inspection or interchange,
but is unnecessary work in the update itself.

There is one related multiplication that *is* valid: converting an existing state-covariance root into
standardized correlation coordinates.  If `P = D K D` and `U_P^T U_P = P^-1`, then
`U_K = U_P D` satisfies `U_K^T U_K = K^-1`.  This is a coordinate change within the same state space; it is not
how a new likelihood is assimilated.

The prior and measurement roots are stacked, not multiplied:

```
[ U ]^T [ U ] = U^T U + J^T Rc^-1 J.
[ WJ]   [ WJ]
```

Multiplying a Cholesky factor of `Rc^-1` by the previous root is generally dimensionally invalid (`Rc` lives in
measurement space while `U` lives in state space) and, even when the dimensions happen to match, represents a
product of precisions rather than the Bayesian sum.  `W @ J` is the valid multiplication; QR of the vertical
stack performs the required addition without forming normal equations.

### Diagonal loading

Near-singular empirical correlations can require a small diagonal load before Cholesky.  Loading must be in
covariance units and visible in diagnostics.  A scale-aware policy uses

```
scale = max(abs(eig(Rc)))
target = max(absolute_floor, relative_floor * scale)
delta = max(0, target - min(eig(Rc)))
Rc_stable = Rc + delta I
```

Here `target` is the desired minimum eigenvalue and `delta` is the amount actually added; they must not be
conflated.  Machine epsilon alone is not a scale-independent policy.  A tiny negative eigenvalue consistent with
round-off may be lifted; a materially indefinite `Rc` is evidence that the fitted joint covariance or the Schur
complement is invalid and must be rejected.  Reports and benchmark output should include `scale`, the raw
minimum eigenvalue, `target`, and `delta`, since increasing `delta` weakens the measurement statistically as
well as stabilising the factorisation.

For a zero-mean prior error, `z_prior = 0`. More generally `z_prior = U m_prior`. Form

```
[ U_prior   z_prior ]
[ A         b       ].
```

`z` is the **square-root information RHS**, not the canonical information vector.  The identities are

```
eta = U^T z
z   = U m
m   = solve(U, z).
```

Passing `eta` where this API expects `z` gives the wrong posterior mean.

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

For an empirical approximate partition, report at least the conditional correlation matrix, relative off-block
Frobenius mass, and the largest whitened off-block spectral norm
`||Rc_ii^-1/2 Rc_ij Rc_jj^-1/2||_2`.  Select any approximation threshold before the confirmatory evaluation and
validate its posterior/NLL sensitivity against full-`Rc` whitening.  The present streamed API assumes the caller
has already accepted that contract; automatic partition selection is future work.

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
