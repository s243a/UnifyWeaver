# Product-Kalman PoE and predicted-error covariance

*Design note, 2026-07-07. This is not a result report and not a preregistered claim. It records the proposed
relationship between product-of-experts aggregation, Kalman-style updates, and hop-conditioned error covariance after
the Sigma(hop) confirmatory run.*

## Question

The confirmatory Sigma(hop) result says that graph position can improve prediction of residual covariance for fuzzy
directional/symmetric labels. The next modeling question is whether this is really the right direction, or whether a
Product-of-Experts (PoE) / geometric-mean objective would be simpler.

The short answer is that PoE and joint covariance answer different questions:

- PoE supplies a consensus-style point estimate of `mu`, often read as a lower-support proxy.
- The product-of-complements / noisy-OR dual can supply an upper-support proxy.
- Direct `mu`, lower-product `mu`, and upper-product `mu` can be fused as correlated channels.
- Sigma(hop) supplies a predicted-error covariance `V(hop)`.
- A calibrated loss needs both `mu` and `V`.

This means a weighted sum of "PoE loss" and "joint loss" is not conceptually clean unless the two terms are explicitly
defined as parts of one likelihood. PoE is a candidate mean/prior mechanism; joint covariance is the error geometry
around that mean.

## What this says about the research direction

If the goal were only a conservative membership estimate, PoE might be enough: it can act like an AND-style
lower-confidence bound on `mu`. But if the goal is a calibrated loss or update rule, the model also needs uncertainty
in that bound. That is the part PoE does not provide automatically.

The Sigma(hop) result is therefore evidence that the project is asking the right second question. It does not only ask
"what is the best `mu`?" It asks "how wrong do we expect this `mu` to be, and how does that predicted error depend on
graph position?" A Kalman-style update needs that second object because the update rate is controlled by predicted
error, not only by the point estimate.

## Objects

Let a pair `(x, y)` have a latent operator vector:

```text
z = [asymmetric, symmetric, ...]^T
```

The sources are different noisy views of this latent vector:

```text
mu_model = model-predicted operator superposition
mu_graph = graph-predicted operator superposition
mu_judge = judge-predicted operator superposition
```

For transitive-parent relations, the observation is not assumed to be purely asymmetric or purely symmetric. It can be
modeled as a mixture:

```text
mu_T = H mu_z
```

where `H` may be a scalar weight vector for one observed channel or a matrix for multiple observed/judge channels.

## Preconditions and implementation boundaries

This is a design note, not an implementation recipe. The closed-form Kalman and Gaussian-PoE formulas below assume
experts are Gaussian, or are being approximated as Gaussian in the chosen coordinates, and that shared evidence is
represented in a covariance matrix rather than counted twice.

For independent Gaussian expert densities over the same latent state, PoE reduces to information-form fusion:

```text
Lambda_fused = sum_j Lambda_j
mu_fused     = Lambda_fused^-1 * sum_j Lambda_j mu_j
Sigma_fused  = Lambda_fused^-1
```

where `Lambda_j = Sigma_j^-1`. This precision-summation identity is exact only under the Gaussian approximation and
conditional independence of expert noise given the state. Correlated experts require a full joint covariance, a learned
joint-channel model, or a conservative unknown-correlation method such as generalized/weighted PoE or covariance
intersection. This note does not implement those mitigations; it points to the joint-channel Kalman variant as the
preferred direction for this project.

All covariance and precision matrices are assumed symmetric positive-definite after regularization. Implementations
should clamp near-zero variances to avoid infinite-precision experts, use a finite-evidence convention for `log` and
`logit` transforms such as `mu <- clip(mu, eps, 1 - eps)`, and preserve covariance validity with Joseph-form updates
or an SPD projection when floating-point arithmetic nudges a matrix out of range.

## Additive predicted-error model

The Gaussian/Kalman version treats the observed fuzzy label as:

```text
y = H z + epsilon
epsilon ~ N(0, R)
z ~ N(mu_z, Sigma_z)
```

The predictive distribution is:

```text
E[y]  = H mu_z
V_y   = H Sigma_z H^T + R
```

and the proper scoring loss is:

```text
L = 0.5 * (y - H mu_z)^T V_y^-1 (y - H mu_z) + 0.5 * log |V_y|
```

For a scalar transitive-parent mixture:

```text
mu_T      = w_asym * mu_asym + w_sym * mu_sym
sigma_T^2 = w^T Sigma_ops w + sigma_obs^2
L_T       = 0.5 * (y_T - mu_T)^2 / sigma_T^2 + 0.5 * log sigma_T^2
```

This is the "measured error divided by predicted error" form. The log-variance term is what prevents the model from
escaping by predicting huge uncertainty.

The Kalman update follows from the same objects:

```text
K       = P H^T (H P H^T + R)^-1
mu_post = mu_prior + K (y - H mu_prior)
P_post  = (I - K H) P (I - K H)^T + K R K^T
```

The `P_post` line is the Joseph-form covariance update. The pseudoinverse appears as a limiting case with weak prior
structure and simple noise. With correlated operators or correlated judge channels, the covariance-aware update is the
relevant generalization.

## Product-of-experts as a mean/prior mechanism

The product version treats model and graph superpositions as two experts:

```text
g_prior = prod_i mu_model,i ^ alpha_i * prod_j mu_graph,j ^ beta_j
```

Replacing the model expert with a judge expert gives a measurement-like channel:

```text
g_meas = prod_i mu_judge,i ^ alpha_i * prod_j mu_graph,j ^ beta_j
```

Important: `g_prior` and `g_meas` share the `mu_graph` factor. They must not be fused as independent prior and
measurement precisions, because that would double-count graph evidence. The split above is expository; an
implementation should either carry prior-measurement cross-covariance or use the joint-channel Kalman variant below.

This membership-space product can be useful, but it is a different object from a formal product of expert densities
and from `V_y`. In membership space `[0, 1]`, the product is an AND-like quantity and is at most 1. Values above 1
only make sense after moving to odds, likelihood ratios, precisions, or another positive evidence scale.

PoE can therefore give a lower-confidence bound or consensus point estimate of membership, but it does not by itself
say how uncertain that bound is. A calibrated NLL still needs an uncertainty model.

## Lower and upper product proxies

If the plain product is used as a lower-support proxy, the natural dual is a product of complements. For source
memberships `mu_i` and nonnegative powers `w_i`:

```text
mu_lower = prod_i mu_i ^ w_i
mu_upper = 1 - prod_i (1 - mu_i) ^ w_i
```

`mu_lower` is an AND-style proxy: all sources must support the relation. `mu_upper` is the corresponding noisy-OR /
non-rejection proxy: the relation remains plausible if any source supports it. The interval
`[mu_lower, mu_upper]` is a useful disagreement diagnostic, not automatically a calibrated credible interval. It
widens when sources disagree and narrows when they agree.

A weighted geometric mean is a softened version of the same family:

```text
mu_geo_lower = prod_i mu_i ^ alpha_i
mu_geo_upper = 1 - prod_i (1 - mu_i) ^ alpha_i
sum_i alpha_i = 1
```

With normalized exponents, `mu_geo_lower` is not a strict lower bound in the order-theoretic sense; it is a
conservative consensus statistic relative to additive averaging. Calibration still has to be learned or checked on
held-out data.

## Product error propagation

If a product is used inside the likelihood, its error needs its own variance. To first order by the delta method around
the operating point, and with poor accuracy near zero-boundary values:

```text
p = u v
Var(p) ~= v^2 Var(u) + u^2 Var(v) + 2uv Cov(u, v)
```

For a geometric mean:

```text
g = sqrt(uv)
Var(log g) ~= 0.25 * [Var(log u) + Var(log v) + 2 Cov(log u, log v)]
Var(g)     ~= g^2 Var(log g)
```

The covariance term is not optional in this project. Existing reports measured strong correlation among readouts, and
naive/factored PoE already underperformed joint heads. Dropping `Cov(u, v)` is exactly the overconfidence failure mode.

## Product-Kalman update in log space

A product-space Kalman update can be written by moving to log evidence:

```text
ell_prior = log g_prior
ell_meas  = log g_meas
```

With scalar prior variance `P_ell` and measurement variance `R_ell`:

```text
K_ell    = P_ell / (P_ell + R_ell)
ell_post = ell_prior + K_ell * (ell_meas - ell_prior)
g_post   = exp(ell_post)
```

In vector form:

```text
K        = P_ell H^T (H P_ell H^T + R_ell)^-1
ell_post = ell_prior + K (ell_meas - H ell_prior)
P_post   = (I - K H) P_ell (I - K H)^T + K R_ell K^T
```

`R_ell` is in log-evidence coordinates. If the available predicted-error model is `V(hop)` in `mu`-residual
coordinates, it must be propagated through the same link function used for `ell`. With link Jacobian `J` at the
operating point, a local approximation is:

```text
V_ell ~= J V_mu(hop) J^T
R_ell ~= C V_ell C^T + R_extra
```

where `C` projects the vector residual covariance into the scalar or vector product channel, and `R_extra` covers
measurement noise and linearization error. Treating `R_ell` as a free scalar while citing `V(hop)` would lose the
bridge from the confirmatory result.

Here `J` is the per-channel link Jacobian, such as `d log(mu) / d mu = 1 / mu` or
`d logit(mu) / d mu = 1 / (mu * (1 - mu))`, so the clipping convention in the preconditions is part of the
definition rather than cosmetic numerical hygiene. `C` may be the identity when `R_ell` is kept per-channel; otherwise
it is the projection from linked residual channels into the fused product channel.

This is a correlated PoE, not an independent PoE. The exponents, source weights, and covariance terms should be
learned or calibrated on held-out node-disjoint data.

## Kalman design variants

There are at least three reasonable Kalman-style designs, and they should be compared rather than conflated.

1. **Direct-mu Kalman.** The state is the latent relation strength, usually in `mu`, `logit(mu)`, or log-evidence
   coordinates. The model's direct `mu` readout is the prior or one measurement channel, and graph/judge evidence
   updates it with a covariance-aware gain.
2. **PoE-as-mu Kalman.** The PoE lower/consensus product supplies the prior point estimate `mu_prior`; a judge+graph
   product supplies a measurement-like channel. Sigma(hop) or another covariance head supplies the predicted-error
   terms that decide the update rate.
3. **Joint direct-mu + PoE Kalman.** The observation vector includes the direct `mu` readout, the lower-product proxy,
   the upper noisy-OR proxy, and any graph/judge channels. The covariance matrix `R` carries their correlations, so
   the update can use PoE information without double-counting the direct model signal.

A schematic joint-channel form is:

```text
s = [logit(mu_direct), log(mu_lower), logit(mu_upper), graph_features, judge_features]^T
s = H ell + epsilon
epsilon ~ N(0, R)
K = P H^T (H P H^T + R)^-1
ell_post = ell_prior + K (s - H ell_prior)
```

This schematic mixes `logit`, `log`, and raw feature channels, so it is not an exact linear-Gaussian observation
model as written. Read `H` as a per-channel quasilinear/statistical-linearization map at the current operating point,
with `R` absorbing both intrinsic noise and linearization residuals. A stricter implementation can instead put every
channel into one link space, such as all logit or all log-evidence coordinates, before applying a linear Kalman step.

This is the version most aligned with the existing `JointPosterior` lesson: PoE, direct `mu`, and graph/judge signals
are features of one correlated measurement vector. The Kalman gain should learn how much nonredundant information
each channel contributes.

## Why not just weight PoE and joint?

A simple objective such as:

```text
L = lambda * L_PoE + (1 - lambda) * L_joint
```

looks attractive, but it hides a type error unless the terms are carefully defined. If PoE produces `mu` and the joint
model produces `V`, then they are not rival scalar losses. They are two parts of one likelihood:

```text
mu = f_PoE(sources)
V  = f_Sigma(hop, corpus, sources)
L  = 0.5 * (y - mu)^T V^-1 (y - mu) + 0.5 * log |V|
```

If `L_PoE` means "diagonal covariance / independent experts," then it is a baseline or shrinkage target. If `L_joint`
means "full covariance," then a gate between them must be learned or validated by held-out NLL, not assigned from a
p-value or intuition. A bounded gate is a shrinkage choice; a power-likelihood coefficient is a temperature and needs
separate calibration.

## How Sigma(hop) fits

Sigma(hop) is not a competing mean estimator. It is a conditional predicted-error model:

```text
V(hop) = [[sigma_D(hop)^2, rho(hop) sigma_D(hop) sigma_S(hop)],
          [rho(hop) sigma_D(hop) sigma_S(hop), sigma_S(hop)^2]]
```

The confirmatory run showed that this `V(hop)` improves held-out residual NLL over a constant covariance baseline on
one fresh Wikipedia category slice. That supports the narrower claim needed here: graph position can carry information
about error geometry. A future Product-Kalman PoE can use that error geometry, but PoE does not replace it.

## Evaluation plan

Treat this as a modeling hypothesis, not a paper claim. Compare on held-out node-disjoint splits:

1. **Naive PoE:** independent/factored product, with and without separability weights.
2. **JointPosterior:** existing calibrated joint head over the source vector.
3. **Additive covariance model:** PoE or linear mean plus diagonal/constant covariance.
4. **Sigma-conditioned covariance:** mean model plus `V(hop)`.
5. **Product-Kalman PoE:** product/log-evidence mean plus learned product-space covariance/update.
6. **Joint direct-mu + PoE Kalman:** direct `mu`, lower-product, and upper-noisy-OR channels fused with a learned
   covariance matrix.

Report:

- held-out log loss / NLL;
- ECE with stated bins;
- AURC using margin gating with bootstrap confidence intervals;
- ablations for model, graph, judge, product, and covariance terms;
- source correlation matrices and separability, before any fusion claim.

A Product-Kalman PoE or joint-channel Kalman variant earns its keep only if it improves held-out log loss and
calibration against both the naive-PoE controls and the additive/joint covariance baselines.

## Guardrails

- Do not assume expert independence. Shared e5/model inputs make independence false by default; use a full learned
  `R`, generalized/weighted PoE, or covariance intersection rather than naive precision summation when cross-covariance
  is unknown.
- Do not use confidence level as a per-item weight; use calibrated margins for routing/abstention.
- Do not evaluate on the same judge family used to train without naming the alignment confound.
- Do not treat a constructed product target as ground truth unless the data source actually measured a joint event.
- Do not treat `[mu_lower, mu_upper]` as a calibrated confidence interval until it has been calibrated on held-out data.
- Do not tune the covariance gate on the same data used to claim a confirmatory effect.
- Do not collapse structural likelihood and measurement error: transitive uncertainty can be inherent likelihood,
  while judge disagreement can be measurement noise.

## Build path

1. Keep the synthetic overconfidence check in `test_product_kalman_poe_synthetic.py` as the first guardrail:
   shared expert noise should make naive independent PoE report variance below empirical error.
2. Keep finite product-space transforms in `product_space.py`: clipped `log_mu`/`logit_mu`, lower/upper
   product proxies, interval width, and first-order Jacobians for covariance propagation. *(Completed in PR #3529.)*
3. Use `test_product_space_monte_carlo.py` as the local nonlinear-statistics sanity check: CPU by default,
   optional CUDA for larger Monte Carlo runs. *(Completed in PR #3529; keep as a prototype diagnostic until a
   Product-Kalman implementation depends on it enough to promote into CI.)*
4. Keep `product_kalman.py` as the scalar/vector Gaussian-conditioning core in product-evidence coordinates:
   estimate residual covariances from calibration residuals, pass cross-covariance when channels share evidence,
   and use the Gaussian-conditioning covariance `P - Cov(x,y) S^-1 Cov(y,x)` rather than independent precision
   summation. *(Core helper added in PR #3530.)*
5. Use `product_kalman_calibration.py` to fit `P_ell`, `R_ell`, and prior-measurement cross-covariance
   from calibration residuals in linked evidence coordinates. The calibration helper slices one regularized joint
   covariance block matrix, and all scalar channels must be passed as explicit `(n, 1)` row matrices rather than
   ambiguous 1-D arrays. Calibration splits must be node-disjoint from training data and from the final evaluation
   split.
6. Use `product_kalman_split_table.py` before real-corpus runs when the table does not already have an audited split.
   It samples held-out unit values, emits only rows whose split-unit columns stay on one side of the split, and
   records omitted boundary rows in a manifest. This makes the later calibration/evaluation rows unit-disjoint
   instead of relying on an ad hoc row split.
7. Use `product_kalman_table_evaluation.py` as the real-corpus entry point when starting from explicit CSV/TSV
   calibration/evaluation rows: `--output-dir` writes a canonical bundle (`input.npz`, `input.manifest.json`,
   `scores.json`, `eval_artifacts.npz`, and `report.md`) in one auditable command. If given `--split-unit-cols`,
   it first writes `split_table.csv`/`split_table.tsv` and `split.manifest.json` in the same run directory, then
   evaluates that split-labeled table. Explicit artifact paths may override those defaults. Under the hood,
   `product_kalman_table_to_npz.py` records source-table hash, split counts, column groups, dimensions, ID
   overlap/duplicate counts, and `H`
   shape/values; `product_kalman_evaluation.py` then fits calibration blocks on one split, scores prior /
   zero-cross-covariance / correlated Product-Kalman predictions on a separate split, and keeps
   calibration/evaluation IDs disjoint. *(Synthetic harness added; real-corpus comparison pending.)*
8. Use `product_kalman_report.py` to render the input manifest, optional split manifest, and score JSON into a
   descriptive Markdown run note. The report is an audit artifact only: it records scores and provenance, but does
   not encode a decision rule or promote Product-Kalman without comparison against registered baselines.
9. Fit empirical Product-Kalman variants on those calibration blocks, then compare against `JointPosterior` and
   Sigma-conditioned covariance on a separate node-disjoint evaluation split; do not reuse the calibration
   residuals that set `R_ell` as the comparison set.
10. Only after the held-out comparison, decide whether this belongs in the training objective.

## Related local artifacts

- `DESIGN_uncertainty_estimation_playbook.md` — current rule: learned calibrated combiner, held-out node-disjoint,
  margin gate, and PoE as a control rather than an independence assumption.
- `REPORT_mu_posterior.md` — empirical evidence that joint heads beat factored PoE under correlated readouts.
- `DESIGN_two_judge_posterior.md` — two-judge posterior framing and covariance/soft-constraint interpretation.
- `REPORT_two_judge_posterior.md` — objective-integration discussion: PoE as shrinkage baseline, learned covariance
  correction only when it earns held-out NLL.
- `DESIGN_transitive_relations.md` — predicted-distribution loss for transitive relations and covariance caveats.
- `test_product_kalman_poe_synthetic.py` — runnable synthetic check that shared expert noise makes independent
  Gaussian PoE overconfident unless the full covariance is modeled.
- `product_space.py`, `test_product_space.py`, and `test_product_space_monte_carlo.py` — finite product-space
  transform helpers, closed-form Jacobian tests, and CPU/CUDA Monte Carlo checks for nonlinear covariance propagation.
- `product_kalman.py` and `test_product_kalman.py` — Gaussian conditioning/update core for scalar/vector
  product-evidence coordinates, with residual covariance fitting and explicit prior-measurement cross-covariance.
- `product_kalman_calibration.py` and `test_product_kalman_calibration.py` — calibration-split fitting of
  `P`, `R`, and cross-covariance blocks plus batch application and split-ID leakage guards.
- `product_kalman_table_to_npz.py` and `test_product_kalman_table_to_npz.py` — CSV/TSV-to-NPZ builder for
  evaluator input arrays with explicit split, ID, prior, measurement, and target columns, plus optional JSON input
  manifests for real-corpus provenance checks.
- `product_kalman_split_table.py` and `test_product_kalman_split_table.py` — split-label materializer for explicit
  Product-Kalman tables, with held-out unit sampling, strict boundary-row omission, and a split manifest.
- `product_kalman_table_evaluation.py` and `test_product_kalman_table_evaluation.py` — one-command table-to-input-
  artifacts-to-holdout-score/report runner, with optional split materialization and canonical `--output-dir` bundles
  for real-corpus Product-Kalman comparisons.
- `product_kalman_report.py` and `test_product_kalman_report.py` — descriptive Markdown report generator for
  Product-Kalman input manifests, optional split manifests, and score JSON artifacts.
- `product_kalman_evaluation.py` and `test_product_kalman_evaluation.py` — holdout comparison harness for
  prior, zero-cross-covariance, and correlated Product-Kalman scoring on disjoint splits, including JSON summaries
  and row-level NPZ artifacts for reproducible corpus-run diagnostics.
- `REPORT_sigma_hop_confirmatory.md` and `PAPER_sigma_hop_confirmatory.md` — confirmatory Sigma(hop) result and
  publication scaffold.
