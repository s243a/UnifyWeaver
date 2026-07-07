# Product-Kalman PoE and predicted-error covariance

*Design note, 2026-07-07. This is not a result report and not a preregistered claim. It records the proposed
relationship between product-of-experts aggregation, Kalman-style updates, and hop-conditioned error covariance after
the Sigma(hop) confirmatory run.*

## Question

The confirmatory Sigma(hop) result says that graph position can improve prediction of residual covariance for fuzzy
directional/symmetric labels. The next modeling question is whether this is really the right direction, or whether a
Product-of-Experts (PoE) / geometric-mean objective would be simpler.

The short answer is that PoE and joint covariance answer different questions:

- PoE supplies a consensus-style point estimate of `mu`.
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
```

The pseudoinverse appears as a limiting case with weak prior structure and simple noise. With correlated operators or
correlated judge channels, the covariance-aware update is the relevant generalization.

## Product-of-experts as a mean/prior mechanism

The product version treats model and graph superpositions as two experts:

```text
g_prior = prod_i mu_model,i ^ alpha_i * prod_j mu_graph,j ^ beta_j
```

Replacing the model expert with a judge expert gives a measurement-like channel:

```text
g_meas = prod_i mu_judge,i ^ alpha_i * prod_j mu_graph,j ^ beta_j
```

This can be useful, but it is a different object from `V_y`. In membership space `[0, 1]`, the product is an AND-like
quantity and is at most 1. Values above 1 only make sense after moving to odds, likelihood ratios, precisions, or
another positive evidence scale.

PoE can therefore give a lower-confidence bound or consensus point estimate of membership, but it does not by itself
say how uncertain that bound is. A calibrated NLL still needs an uncertainty model.

## Product error propagation

If a product is used inside the likelihood, its error needs its own variance. For two random estimates:

```text
p = x y
Var(p) ~= y^2 Var(x) + x^2 Var(y) + 2xy Cov(x, y)
```

For a geometric mean:

```text
g = sqrt(xy)
Var(log g) ~= 0.25 * [Var(log x) + Var(log y) + 2 Cov(log x, log y)]
Var(g)     ~= g^2 Var(log g)
```

The covariance term is not optional in this project. Existing reports measured strong correlation among readouts, and
naive/factored PoE already underperformed joint heads. Dropping `Cov(x, y)` is exactly the overconfidence failure mode.

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
K       = P_ell H^T (H P_ell H^T + R_ell)^-1
ell_post = ell_prior + K (ell_meas - H ell_prior)
```

This is a correlated PoE, not an independent PoE. The exponents, source weights, and covariance terms should be
learned or calibrated on held-out node-disjoint data.

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

Report:

- held-out log loss / NLL;
- ECE with stated bins;
- AURC using margin gating with bootstrap confidence intervals;
- ablations for model, graph, judge, product, and covariance terms;
- source correlation matrices and separability, before any fusion claim.

A Product-Kalman PoE earns its keep only if it improves held-out log loss and calibration against both the naive-PoE
controls and the additive/joint covariance baselines.

## Guardrails

- Do not assume expert independence. Shared e5/model inputs make independence false by default.
- Do not use confidence level as a per-item weight; use calibrated margins for routing/abstention.
- Do not evaluate on the same judge family used to train without naming the alignment confound.
- Do not treat a constructed product target as ground truth unless the data source actually measured a joint event.
- Do not tune the covariance gate on the same data used to claim a confirmatory effect.
- Do not collapse structural likelihood and measurement error: transitive uncertainty can be inherent likelihood,
  while judge disagreement can be measurement noise.

## Build path

1. Write a small synthetic test where two sources have known correlation and naive PoE becomes overconfident.
2. Implement a product-space transform that exposes `log_mu`, `logit_mu`, or likelihood-ratio coordinates explicitly.
3. Fit scalar/vector product-Kalman updates with learned `P_ell` and `R_ell`.
4. Compare against `JointPosterior` and Sigma-conditioned covariance on held-out node-disjoint splits.
5. Only after the held-out comparison, decide whether this belongs in the training objective.

## Related local artifacts

- `DESIGN_uncertainty_estimation_playbook.md` — current rule: learned calibrated combiner, held-out node-disjoint,
  margin gate, and PoE as a control rather than an independence assumption.
- `REPORT_mu_posterior.md` — empirical evidence that joint heads beat factored PoE under correlated readouts.
- `DESIGN_two_judge_posterior.md` — two-judge posterior framing and covariance/soft-constraint interpretation.
- `REPORT_two_judge_posterior.md` — objective-integration discussion: PoE as shrinkage baseline, learned covariance
  correction only when it earns held-out NLL.
- `DESIGN_transitive_relations.md` — predicted-distribution loss for transitive relations and covariance caveats.
- `REPORT_sigma_hop_confirmatory.md` and `PAPER_sigma_hop_confirmatory.md` — confirmatory Sigma(hop) result and
  publication scaffold.
