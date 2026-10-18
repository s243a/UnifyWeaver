# Explicit adjacent-row residual correlation — follow-up design

## Status and question

This is a future, post-hoc design written after the covariance-sensitivity controls exposed mean/covariance
identifiability, but before computing adjacency-stratified residual statistics.  The only inspected inputs were
outcome-blind graph incidence/counts.

The question is distinct from the confirmed `Sigma(hop)` result:

- `Sigma(hop)` asks whether one row's within-item D/S error block depends on the hop between that row's node
  and root.
- This study asks whether two different rows scored by the same judge have correlated errors when their graph
  locations are adjacent.

The engineering endpoint is a validated distance-separated batching rule: keep the full within-item judge
block everywhere, and omit cross-item covariance only when predicted whitened coupling is negligible.

## Outcome-blind inventory

On the exact matched rows used by #3671:

| Cross-row relation | Exploratory (1,000 rows) | Fresh (700 rows) |
|---|---:|---:|
| Same root; descendant nodes one edge apart | 0 | 0 |
| Same descendant; roots one edge apart | 84 pairs / 139 rows | 206 pairs / 285 rows |
| Any endpoints one edge apart | 876 pairs / 519 rows | 688 pairs / 372 rows |
| At least one exact shared endpoint | 824 pairs / 582 rows | 841 pairs / 569 rows |

The primary existing-data contrast is therefore **same descendant, adjacent roots** versus **the same
descendant, nonadjacent roots**.  This controls the shared-descendant effect while varying direct root
adjacency.  The reverse direction (same root, adjacent descendants) requires a deliberately sampled corpus.

### Adjacent positives and distant negatives

Treat adjacency-enriched sampling as a contrastive design, analogous in role (not identical likelihood) to
negative sampling in word2vec:

```text
(anchor row, adjacent positive row, matched distant negative row).
```

Adjacent positives teach local smoothing and dataset topology.  Distant negatives identify where smoothing
must stop and supply candidate near-independent pairs for block batching.  Match negatives on shared
descendant/root family, row hop, campaign tag, and judge where possible; include hard negatives that are
semantically similar but graph-distant so the task cannot be solved from trivial domain differences.

The representation/mean objective and covariance objective remain separate.  Contrastive adjacency can train
the graph judge/model to know the dataset; cross-fitted residuals then determine whether same-judge error
coupling persists after that learned mean has been removed.

## Residual and split contract

Use the existing conditional residual convention

```text
e = truth - prior
v = measurement - H truth
q = v - C.T P^-1 e.
```

Calibration, graph scaling, regional mean, and within-item block covariance are fit on training components
only.  Outer partitions are connected-component-disjoint under exact shared endpoints; crossing rows are
dropped.  The regional mean is cross-fit before any adjacency covariance statistic is computed so a smooth
mean realization is not relabelled as random covariance.

The target remains GPT-5.5 operating-judge fidelity, not independent truth.  Existing labels make this a
post-hoc reuse study; a later purpose-built sample must be registered before new judge calls.

## Primary adjacency statistic

Whiten held residual rows by the train-fitted within-item block `B`:

```text
z_i = B^-1/2 (q_i - m_hat_i).
```

For each adjacent row-pair, use the symmetric cross-product

```text
G_ij = 0.5 * (z_i z_j.T + z_j z_i.T).
```

Compare its mean with matched same-descendant/nonadjacent-root controls.  Report the full channel matrix,
spectral norm, signs, and per-channel diagonal—not only one pooled correlation.  Inference uses a
component-preserving permutation within corpus/tag/hop strata (`K >= 1000`); rows are never bootstrapped as
IID observations.

## PSD predictive comparator

An adjacency matrix is not automatically PSD.  Build a row feature map from endpoint and one-hop-neighborhood
incidence (or a preregistered graph diffusion feature map), normalize rows, and use its Gram matrix

```text
K_adj = Phi Phi.T.
```

The deployable comparator selects only correlation trust

```text
alpha in {0, .025, .05, .10, .20, .35, .50, .75, 1}
```

at one frozen adjacency geometry.  Bandwidth/channel-shape variants remain oracle sensitivity diagnostics;
they do not enter deployment selection.  Nested selection must use a family-wise block-null threshold that
repeats mean fitting and preserves overlapping components.

Compare regional block, adjacency covariance, and a matched parameter-count nonadjacency kernel on full held
joint residual NLL.  Then pass the covariance to the joint square-root/QR conditioner and report posterior
NLL, decision log-loss/AURC, and prior/noise loading.

## Identification and repeated-judge extension

One residual field can confound a smooth fixed bias with a smooth random effect.  The existing-data study can
establish predictive adjacency structure but cannot cleanly separate those causes.  A confirmatory corpus
should therefore score the same adjacency-enriched items with at least two independent judge draws.  That
supports separate estimates of persistent item bias, within-run adjacent error coupling, and run-to-run judge
variation.

Synthetic controls must include:

- block null with regional mean refitting;
- smooth mean only;
- equal-energy permuted adjacency geometry;
- planted adjacency coupling at 0.04, 0.10, and 0.20;
- oracle-mean/known-marginal mechanism power separately from end-to-end nuisance estimation.

## Gate and batching consequence

Call explicit adjacency covariance promising only if its nested-selected NLL gain is positive on both corpora
and at least 8/10 component-disjoint partitions per corpus, its family-wise null is rejected, posterior NLL
does not worsen, and decision log-loss/AURC degradation stays within 0.01.

If the gate passes, estimate a conservative separation threshold from

```text
norm(B^-1/2 R_ij B^-1/2, 2) <= epsilon_batch
```

on fresh held components.  Items beyond that validated threshold may use independent/block updates; adjacent
connected components remain joint QR blocks.  If the gate fails, retain block batching for this adjacency
definition without claiming that all possible graph geometries are independent.
