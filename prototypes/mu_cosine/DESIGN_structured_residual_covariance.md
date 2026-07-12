# Semantic/graph residual covariance — evidence-gate protocol

## Status and question

This protocol is the statistical follow-up to the batched square-root/QR conditioner.  The conditioner can
already whiten a dense conditional measurement covariance and triangularise the resulting information rows.
The open question is whether campaign rows that are close in frozen semantic or graph-feature space actually
have enough *held predictive residual dependence* to justify a cross-item covariance model.

This is an evidence gate, not a production integration.  It deliberately keeps CUDA and an optimized
Kronecker inverse-root operator out of scope until a structured covariance improves held data.

The operating target is the existing GPT-5.5 campaign response.  Results therefore measure fidelity to that
judge, not truth or judge-independent quality.  No new judge calls are required.  An independent or
human-verified target remains necessary for a general-quality claim.

## Statistical object: condition first, then model cross-item residuals

For one campaign row, retain the sign convention used by the correlated Gaussian conditioner:

```text
e = truth - prior
v = measurement - H truth
Cov(e)   = P0
Cov(e,v) = C0
Cov(v)   = R0.
```

The within-item calibration split estimates `P0`, `C0`, and `R0`.  Conditioning measurement error on prior
error gives

```text
J0 = H + C0.T P0^-1
q  = v - C0.T P0^-1 e
Rc = Cov(q) = R0 - C0.T P0^-1 C0.
```

`q` is the object whose dependence across campaign rows is tested.  This ordering matters: fitting a
cross-item model to raw `R0` and then passing it to the conditional whitener would subtract the prior/shared
term twice.

For a held batch of `N` items, the information update uses

```text
prior error state:  e_batch ~ N(0, I_N kron P0)
conditional design: J_batch = I_N kron J0
conditional noise:  q_batch ~ N(mean_q, Rc_batch).
```

The measurement RHS is `measurement - H prior - mean_q`.  Dense `Rc_batch` is whitened once and passed to the
existing Householder information update.  Item-major flattening is fixed throughout: all measurement channels
for item 0, then all channels for item 1, and so on.

This first protocol leaves cross-item prior-error covariance at `I_N kron P0`.  A persistent regional error in
the *prior* belongs in `P`, or in an augmented regional-bias state, and is not silently relabelled as
measurement noise here.

## Split and leakage contract

For each corpus and split seed:

1. Use `node_disjoint_pair_split` with 40% held nodes and 64 candidate assignments.
2. Candidate balancing may use the campaign's sampling/neighbourhood stratum and pair incidence.  It must not
   use held D/S values, held residuals, or the operating judge's hard decision.
3. Drop every pair crossing the node partition.
4. Fit graph-D affine calibration, graph-S regression, Luna affine calibration, `P0/C0/R0`, feature
   standardisation, RBF bandwidths, regional-mean hyperparameters, and covariance parameters on train rows
   only.
5. Do not use held outcomes to choose numerical loading, optimization duration, kernel weights, or a model.

The primary run uses predeclared seeds 0 through 9.  Mean and SD across those node partitions are descriptive
partition stability, not an SE or population confidence interval.  A dense held covariance couples rows, so a
row or endpoint bootstrap is not attached to the joint Gaussian likelihood.

The run manifest must content-hash the scored campaign, Luna scores, model checkpoint, both E5 caches, the
exploratory graph TSV, and the fresh LMDB `data.mdb`.  LMDB `lock.mdb` is excluded because it is mutable
process state rather than graph content.  A CLI campaign override must be passed to both target and dataset
loaders; recording a path that was not actually consumed is a provenance failure.

## Outcome-blind row geometry

Two PSD RBF kernels are available on both corpora:

- **semantic:** the row feature is the normalized concatenation of frozen e5 `passage(node)` and
  `query(root)` embeddings;
- **graph-feature:** train-standardized `[hit_prob, inverse common-ancestor distance, shared-parent,
  shared-grandparent, ancestor-related]` features.

The second is called a graph-*feature* kernel, not a shortest-path or diffusion kernel.  Applying an RBF to an
arbitrary shortest-path distance matrix is not guaranteed PSD.  A true graph diffusion kernel or an RBF over
a provenance-tracked structural embedding is later work; the existing structural embedding does not cover
the fresh corpus.

Each RBF bandwidth is the median nonzero pair distance on train rows.  Held rows do not influence it.

## Separate regional mean from covariance

One response per judge/item cannot identify a persistent regional bias field and correlated random noise by
interpretation alone.  The protocol therefore compares them predictively.

A train-only kernel-ridge regional mean is selected by exact leave-one-out MSE across semantic,
graph-feature, and equal-mixture kernels and a fixed ridge grid.  The covariance models are fit to the
corresponding leave-one-out train residuals; held residuals are centered using the train-fitted prediction.
The global-mean block model remains an explicit baseline.

## Covariance models

All component matrices are PSD by construction (`B = L L.T`).  The statistical independent term is distinct
from the numerical diagonal loading applied, if necessary, by the existing whitener.

1. **`block_global`**

   ```text
   Rc = I_N kron B0
   ```

   with only the train global residual mean removed.

2. **`block_regional`**

   The same block form, fit after the cross-fitted regional mean.  The difference from `block_global` tests
   predictable local bias without claiming residual covariance.

3. **`separable_regional`**

   ```text
   Rc = I_N kron B0
        + (w_sem K_sem + w_graph K_graph) kron Bshared,
   w_sem,w_graph >= 0,  w_sem + w_graph = 1.
   ```

   This is the restricted model that could later support a compact diagonalization/operator path.

4. **`dense_lmc_regional`**

   ```text
   Rc = I_N kron B0
        + K_sem   kron B_sem
        + K_graph kron B_graph.
   ```

   This additive latent-kernel model is the flexible structured reference.  It is dense when materialized but
   is not an arbitrary unstructured covariance, which cannot be estimated or transferred to unseen held items
   from one residual field.

The structured component matrices are fit by deterministic pairwise composite Gaussian likelihood on train
rows.  The primary held score is nevertheless the full joint Gaussian likelihood of the held residual field.
The block model's analytic train objective is in original channel units, whereas the structured optimizer's
composite objective uses train-RMS-standardized channels; those train diagnostics are not comparable across
families.  Every held NLL below is evaluated in the same original units and is the valid model comparison.

Each structured fit retains the exact independent-block submodel as a candidate.  Dense LMC also retains the
fitted separable model as a candidate.  When an inherited structured factor is zero, Adam starts from a
material PSD seed rather than an effectively zero Cholesky factor, whose `B=L L.T` gradient would vanish.
Tests require escape from a zero block start and require dense LMC to beat the restricted family on a planted
nonseparable field.  These checks reduce optimizer false negatives; they do not turn a finite-step fit into a
global-optimality proof.

## Metrics and required diagnostics

Primary covariance score:

- held joint conditional-residual Gaussian NLL per scalar.

Secondary posterior scores after one joint QR update:

- mean bivariate marginal posterior NLL;
- state MSE;
- Mahalanobis squared error per state dimension;
- empirical 95% bivariate ellipse coverage.

Operating-judge decision diagnostics use the existing train-only D/S logistic bridge:

- accuracy;
- categorical log-loss;
- ECE with 10 equal-width confidence bins;
- AURC using top-1 minus top-2 probability margin.

Every covariance result must also report:

- relative off-item Frobenius mass;
- largest whitened off-item block spectral norm;
- minimum/maximum eigenvalue and condition number;
- absolute and relative diagonal loading actually applied;
- covariance dimension and conditioner wall time;
- train residual mean removed, selected regional kernel/ridge, RBF bandwidths, and optimization provenance.

## Predeclared engineering gate

This gate decides whether a later inverse-root/CUDA PR is warranted; it is not a significance test.

1. `dense_lmc_regional` must beat `block_regional` in held joint residual NLL on both corpora and on at least
   8 of 10 split seeds per corpus.
2. `separable_regional` must beat `block_regional` on both corpora and at least 8 of 10 seeds, and recover at
   least 80% of the dense-LMC NLL gain after macro-averaging the per-scalar gain equally across the 20
   predeclared corpus/split records (not weighting by held batch size):

   ```text
   (NLL_block - NLL_separable) / (NLL_block - NLL_dense) >= 0.80.
   ```

3. Separable posterior NLL may not be worse than the regional block model on either corpus.
4. Decision log-loss and margin AURC may not worsen by more than 0.01 absolute.
5. No model may require silently increasing the conditioner's declared relative-loading budget.

If dense LMC helps but the separable model fails, revise the kernel/statistical model rather than optimizing a
Kronecker operator.  If neither helps, retain the block conditioner.  If the separable model passes, the next
PR may implement item-kernel diagonalization, small per-eigenmode judge factors, exact correlated online-block
conditioning, and a matched end-to-end CUDA benchmark that includes whitening.

## Explicit non-claims and separate follow-ups

- This protocol does not establish human or judge-independent quality.
- It does not show that graph supervision familiarizes the model with a dataset.  That needs the separate
  training-time graph-supervision × final-fusion factorial ablation.
- It does not promote `JointPosterior` over the correlated Gaussian model.  `JointPosterior` remains the
  learned nonlinear decision comparator and should be revisited only with soft macro targets and nested
  calibration.
- It does not interpret a fitted kernel component as causal judge noise.  Held predictive performance is the
  evidence; the component name is a modeling label.
- It does not claim a CUDA crossover from the earlier compute-only benchmark.  The proposed covariance can
  make whitening the dominant cost, which must be included in any future timing claim.
