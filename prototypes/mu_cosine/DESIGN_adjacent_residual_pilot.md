# Adjacent-row residual pilot — component cross-fitting and stability bands

## Status

This executable amendment was frozen after inspecting graph incidence, campaign tags, and pair counts, but
before computing any adjacency-stratified residual statistic.  It implements the existing-data pilot proposed
in `DESIGN_adjacent_row_residual_correlation.md`.  No new judge calls are used.

The pilot is **descriptive and conditional on the existing campaign**.  It cannot license structured
cross-item covariance or distance-separated QR batching.  A confirmatory deployment claim requires a
purpose-built adjacency-enriched sample with independent repeated judge draws.

## Outcome-blind limitations found before residual analysis

After the operating-judge, Luna, graph, and e5 intersection, the two corpora contain:

| corpus | rows | same-descendant, adjacent-root pairs | nonadjacent pool | endpoint components with positives |
|---|---:|---:|---:|---:|
| exploratory | 1,000 | 84 | 680 | 29 |
| fresh | 700 | 206 | 571 | 40 |

The exploratory corpus is below even a rough 30-cluster threshold, components are unequal (largest 69 and 31
rows), and every corpus has only one residual field.  Component resampling therefore produces **stability
bands**, not population confidence or coverage intervals.

Adjacency is also aligned with campaign construction.  Positive pairs are overwhelmingly tag pairs
`h1-h2`, `h2-h3`, `h3-h4`, and `h4-h5`.  Within a fixed descendant there are no exact tag-pair-matched
nonadjacent controls.  The identified existing-data estimand is consequently **predictive adjacency plus hop
proximity**, not a pure causal adjacency effect.

## Component-cross-fitted residual field

Rows are partitioned by connected components of the exact endpoint-incidence graph.  Whole components, never
individual rows, are assigned to five outcome-blind folds.  The assignment balances row count, campaign tags,
and the number of outcome-blind adjacent-root pairs.  Assignment seeds are 20 through 29, disjoint from the
#3671 seed range.  Repeated assignments are stability probes on the same labels, not independent samples.

For each held component fold, refit on the other four folds:

1. graph-D, graph-S, and debiased-Luna calibration;
2. the joint `P/C/R` residual model and conditional design;
3. semantic/graph scaling and the regional kernel-ridge mean;
4. the within-item conditional residual covariance `B`.

For held row `i`, form

```text
q_i = v_i - C.T P^-1 e_i
z_i = B^-1/2 (q_i - m_hat_i).
```

All rows receive exactly one out-of-fold residual per assignment.  Any two primary rows share their descendant,
so component assignment guarantees that their residuals were produced by the same train-only fit.

The regional mean's ridge/kernel choice still uses exact row LOO within the training components.  The outer
component holdout prevents direct held leakage, but shared-endpoint dependence can affect hyperparameter choice.
The smooth-mean synthetic control and assignment stability must be reported; this pilot does not call that
selection population-calibrated.

## Pair records and estimands

For each two different rows sharing descendant `x`, let their roots be `a` and `b`.  The record is positive when
`a` and `b` share a direct graph edge in either direction.  Its matrix observation is

```text
G_ij = 0.5 * (z_i z_j.T + z_j z_i.T).
```

For every positive `p=(i,j)`, select up to three nonadjacent controls that share `x` and one anchor row:
`(i,k)` or `(j,k)`.  Rank candidates without outcomes by partner-tag distance, root-degree-bin difference,
frozen semantic distance, and finally stable row identity.  Average the selected control matrices with equal
weight and define

```text
A_p = G_ij - mean_control(G_anchor,k).
```

Report eligibility/exclusions and the achieved tag, degree-bin, and semantic imbalance.  Average `A_p` within
each exact-endpoint component, then give every positive component equal weight.  The primary scalar is frozen
before inspecting matrix entries:

```text
delta = mean_component(trace(A_component) / 4).
```

Report:

- the 4x4 positive and control cross-product means;
- their anchor-matched adjacency-minus-control contrast;
- every diagonal/channel entry and all ten unique symmetric entries;
- spectral norms of the two means and their contrast;
- counts by corpus, stratum, descendant, and endpoint component.

The component-macro point estimate describes the observed positive-component population.  Use 9,999
Rademacher component-multiplier draws on the centered component estimates.  Report pointwise and max-|t|
simultaneous stability bands over the primary trace and ten unique matrix entries, plus the spectral error
radius

```text
delta_stability = Q_0.95(||Delta_star - Delta_hat||_2).
```

These are explicitly labelled conditional component-resampling stability summaries, not population CIs.
Disable every CI-like gate when there are fewer than 30 effective components or one component has more than
10% weight.  Leave-one-positive-
component-out estimates expose domination by a single component.

## Low-capacity PSD comparator

For root `r`, let `psi(r)` be the unit-normalized binary incidence vector of `r` and its one-hop parents and
children.  For campaign row `i`, use the role-aware sparse feature

```text
Phi_i = one_hot(descendant_i) tensor psi(root_i)
K_adj = Phi Phi.T.
```

This Gram matrix is PSD, has unit diagonal, and only couples rows with the same descendant.  With the fold-fit
within-item block `B=L_B L_B.T`, test only the frozen trust path

```text
C_alpha = ((1-alpha) I + alpha K_adj) tensor I_4
R_alpha = (I tensor L_B) C_alpha (I tensor L_B).T
alpha in {0, .025, .05, .10, .20, .35, .50}.
```

`alpha=1` is diagnostic only because repeated or identical features can make the endpoint singular.  A fixed
outcome-blind within-descendant derangement `P K_adj P.T` is the equal-capacity topology-specificity
comparator; it is a common-mode diagnostic, not a null distribution.  Any alpha selection used for a claim
must be nested and its block-null calibration must repeat calibration, `P/C/q`, mean selection, `B`, and alpha
selection on the same component topology.

## Synthetic controls

Before interpreting real residuals, exercise the statistic on independent synthetic components with:

- block-null residuals;
- smooth mean only, removed by the same cross-fitting contract;
- planted positive-pair coupling at 0.04, 0.10, and 0.20;
- equal-energy label/geometry permutation;
- unequal component sizes matching the two real inventories.

Mechanism power with known mean and `B` must be separate from end-to-end nuisance estimation.  Failure of a
nominal detection target blocks interpretation but does not prove that good covariance information is useless.

## PSD-safe covariance consequence

An elementwise lower confidence band is not matrix-conservative: for a two-row correlation matrix, changing
`rho` moves the common and contrast eigenvalues in opposite directions, and entrywise bounds need not be PSD.
The eventual deployment adapter therefore has two separate safeguards:

1. shrink uncertain correlation geometry toward the block model on a PSD path;
2. inflate the resulting conditional noise covariance by a high-probability spectral error envelope.

For a validated estimate, the intended form is

```text
R_hat(alpha) = L_B [(1-alpha) I + alpha C_hat] L_B.T
R_safe       = R_hat(alpha_safe) + delta_95 I.
```

`alpha_safe` is selected by a lower bound on held-out benefit or by worst-case validation, not by lowering
individual covariance entries.  `delta_95` must come from a confirmatory procedure that refits all
outcome-dependent objects.  This pilot's `delta_stability` is diagnostic and must not be relabelled `delta_95`.

Likewise, a proposed block separation is safe only when an **upper** bound on

```text
||B_i^-1/2 R_ij B_j^-1/2||_2
```

is below a preregistered `epsilon_batch`.  The present pilot cannot establish that bound.

## Decision rule

This PR can conclude only one of:

- **promising descriptive structure:** both corpora have the same contrast direction, assignment signs are
  stable, leave-one-component-out behavior is not dominated, and synthetic mechanism power is adequate;
- **not resolved by existing data:** any of those conditions fails.

In both cases the deployment gate remains false.  The next confirmatory campaign should deliberately sample
`(anchor, adjacent positive, matched distant/hard negative)` triples, balance hop/tag strata by construction,
and obtain at least two independent calls from each judge family.
