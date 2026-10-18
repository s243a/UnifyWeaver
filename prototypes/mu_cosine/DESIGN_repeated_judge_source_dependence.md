# Repeated-judge source dependence — topology bridge

## Purpose and claim boundary

The three-hop-core construction in `REPORT_repeated_judge_source_regions.md` is too wasteful to support the
planned campaign.  This amendment replaces hard cross-region separation with an explicit, topology-only
dependence model over the **full** exclusive source regions.  It is a no-spend structural bridge: it reads only
the two frozen graphs, creates no candidates or embeddings, consumes no historical outcomes, and calls no
judge.

Topology can identify a reproducible exposure geometry and its consequences under stipulated correlation
strengths.  It cannot identify residual covariance, candidate packability, an effect size, or end-to-end power.
Accordingly, this audit authorizes no downstream operation.  Its passing matrices become fixed inputs to a
separate source-dependent full-procedure null/power extension.  Only that later power gate may unlock the
attempted-input **identity** inventory needed to test exact candidate feasibility.  Candidate selection, Nomic,
live scoring, covariance promotion, independent batching, QR specialization, and CUDA remain further away.

## Frozen full-region construction

Retain the deterministic `K in {64,96,128}` induced-connected partitions from the prior audit, but use every
node assigned to a region.  Regions remain concentration, fold, and dependence-sensitivity units; they are not
renamed connected components and are not claimed independent.  True weak-component identity remains a
separate immutable diagnostic.

Every later three-row candidate component has four distinct graph endpoints and all four must share one
`source_region`.  A region contributes at most `floor(0.10 G)` components at registered
`G in {160,320,512,800}`.  The topology audit charges four arbitrary nodes per component.  This is only an
optimistic necessary capacity bound: history exclusions, the 32 campaign cells, endpoint-disjoint packing,
the adjacent edge, and the matched finite-distance negative can all reduce realized capacity.

For each `(corpus,K,G)`, form a deterministic exposure-aware diagnostic allocation.  Starting from zero
counts `n`, repeatedly add one component to the eligible region minimizing the exact increase in
`n.T E n`, namely `2(E n)_r + E_rr`, with stable region ID as the only tie-break.  A region is eligible until
the smaller of `floor(region_size/4)` and `floor(0.10 G)` is reached.  Failure to allocate all `G` fails closed.
This prospective greedy quota spreads components away from already exposed regions; it is not claimed to be
the global integer optimum.  It is a planned source-diversity target, not evidence that exact candidates
exist.  The eventual builder must reproduce or improve its registered information diagnostics with real
candidate counts.

## PSD region exposure

Let `P` be the row-stochastic random-walk transition on the complete frozen undirected graph, with an isolated
node retaining unit self-mass.  Let `U` be the `K x N` matrix whose row `r` is uniform over the nodes assigned
to source region `r`.  With frozen cumulative-walk weights

```text
w = (1, .5, .25, .125),
Z = sum_h sqrt(w_h) U P^h.
```

Normalize each nonzero row of `Z` and define

```text
E = Z_normalized Z_normalized.T.
```

`E` is symmetric positive semidefinite with unit diagonal by construction.  Because its features are
nonnegative, its entries are nonnegative.  An off-diagonal entry measures overlap between the average
radius-three landing profiles of two full regions.  It is an exposure proxy, not an empirical residual
correlation and not an upper bound for adversarially boundary-concentrated candidates.

For a diagnostic component allocation, let `H` be its one-hot `G x K` region-membership matrix.  The frozen
source-dependence sensitivity family is

```text
S = H E H.T,
C_rho = (1-rho) I_G + rho S,
rho in {0, .025, .05, .10, .20}.
```

This path is PSD and unit diagonal.  Distinct components in the same region have correlation `rho`; components
in different regions have correlation `rho E_rs`.  It deliberately includes within-region dependence instead
of pretending that a region label makes its components independent.

## Information diagnostics

For an equal-component scalar mean under `C_rho`, report the exact design effect and equivalent independent
component count

```text
DE_rho = 1.T C_rho 1 / G,
G_eff,rho = G^2 / (1.T C_rho 1) = G / DE_rho.
```

For a separable vector endpoint covariance `C_rho tensor Sigma`, the same value is the worst-linear-contrast
effective count after whitening by the average marginal `Sigma`.  A future nonseparable source/prompt model
must instead report

```text
G_eff,min = 1 / lambda_max(Sigma_bar^-1/2 V_bar Sigma_bar^-1/2),
```

where `V_bar` is the full covariance of the equal-component vector mean.  Source ESS is a necessary design
diagnostic, not a power calculation or a replacement for two-way prompt/source inference.

Also report the exposure spectrum, numerical rank, effective rank `(tr E)^2 / tr(E^2)`, off-diagonal
quantiles, and average hop-specific mass landing outside each source region.  This is not first-exit mass: a
walk may leave and return by hop `h`.  Report an allocation-free numerically guarded ESS floor too.  Bound
`n.T E n` above by the tighter of (a) `lambda_max(E)` times the largest feasible
`sum_r n_r^2`, obtained by filling the largest region capacities first, and (b)
`G max_r(E c)_r`, where `c` is the capacity vector.  Substituting that upper bound in the mean-variance formula
gives a lower ESS bound for any cap-feasible allocation.  The implementation upper-bounds `lambda_max(E)` by
the largest nonnegative row sum and guards floating reductions outward; it does not call an eigensolver-plus-
tolerance value a formal certificate.  These quantities describe the frozen geometry; no threshold is tuned
after seeing them.

## Gates and next stage

The narrow topology bridge passes for a `(corpus,K,G)` only when:

1. the four-endpoint optimistic capacity bound can supply `G` components;
2. the deterministic exposure-aware allocation supplies exactly `G`, respects the 10% cap, and uses at least 20
   source regions; and
3. `E` and every `C_rho` are finite, symmetric, PSD, and unit diagonal.

The audit does **not** choose a winning `K` or impose an uncalibrated ESS cutoff.  All `K` values that pass the
narrow bridge continue to the next stage.  Joint structural passage in both required corpora unlocks nothing
by itself.  The next work is explicitly two-stage:

1. **Stage A, still without history:** extend the complete repeated-judge null/power procedure using these
   registered diagnostic allocations and exposure matrices, source-atomic folds, synthetic split-contained
   prompt incidence, and two-way prompt/source inference.  Require family-wise null control and at least 80%
   power before the attempted-input identity inventory can be constructed.  That inventory is still not a
   live-campaign authorization.
2. **Stage B, after a Stage-A pass:** consume only immutable attempted-input identities, enumerate exact
   structural candidates after those exclusions, verify the 32 cells, endpoint-disjoint packing,
   finite-distance negative, cap, and region quotas, then replace the diagnostic lift with realized candidate
   exposure and prompt incidence.  The realized design must remain inside the powered Stage-A envelope or its
   full procedure must be recalibrated before candidate selection, Nomic, or spending.

Inference and conditioning use different conservative directions.  Power and uncertainty must cover a
simultaneous **upper** dependence/spectral envelope; understating dependence is anti-conservative.  If a
validated covariance is later supplied to the conditioner, its cross-item benefit path is shrunk toward block
using training-only held benefit and the existing `s_safe`/`delta_95` gates.  Entrywise lower confidence bounds
are not assumed PSD and are not substituted for either procedure.

## Alternatives rejected or deferred

| Alternative | Disposition | Reason |
|---|---|---|
| three-hop cores | reject for campaign construction | certified separation retained too little graph mass in both corpora |
| call full regions independent | reject | a partition is an engineering unit, not a stochastic independence proof |
| shortest-path Gaussian region kernel | reject | general graph-distance RBFs are not guaranteed PSD |
| average pairwise node-kernel matrix without an explicit feature map | reject | expensive and easier to implement with hidden normalization/PSD errors |
| hard-coded confidence weights or inverse-variance source weights | reject | correlated evidence requires the later calibrated joint posterior; confidence remains a margin gate |
| entrywise lower 95% covariance for inference | reject | can be non-PSD and understates sampling variance |
| region count or Kish ESS called power | reject | neither includes the selector, effect size, prompt dependence, multiplicity, nor candidate constraints |
| tune `K`, hop weights, or `rho` from this audit | reject | would use the diagnostic result to redefine its own prospective family |
| full repeated-judge power extension in this PR | defer to the next focused PR | first freeze and audit the source matrices; the synthetic extension can use the registered diagnostic allocations, then exact candidate incidence must be rechecked later |
