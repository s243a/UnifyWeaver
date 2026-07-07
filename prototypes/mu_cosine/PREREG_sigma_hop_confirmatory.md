# Pre-registration: confirmatory `Σ(hop)` test on a fresh Wikipedia category slice

*Registered 2026-07-07, before constructing or scoring the fresh confirmatory corpus. This freezes the single
confirmatory test for the hop-conditional covariance result reported in PR #3517 / `REPORT_two_judge_posterior.md`:
smooth `Σ(hop)` improved held-out continuous-μ residual NLL over constant-Σ on the exploratory `100k_cats`
multi-hop sample. The exploratory result was significant under the finally-adopted specification
(`p=0.001`, K=1000), but the specification was reached after iteration on the same ~250 pairs. This document is the
guardrail: one fresh corpus, one model comparison, one one-sided permutation test.*

## Claim Being Tested

**Pre-registered hypothesis:** on a Wikipedia category corpus that shares no category nodes with the exploratory
`100k_cats` graph, the smooth hop-conditional covariance model `Σ(hop)` will beat a constant-covariance baseline on
held-out continuous-μ residual likelihood.

Decision rule, frozen in advance:

- **Confirmed:** observed mean gain `NLL_constantΣ - NLL_Σ(hop) > 0` and the one-sided hop-shuffle permutation test
  gives `p < 0.01`.
- **Not confirmed:** otherwise. In that case the PR #3517 result remains an exploratory/post-selection finding, even
  if the fresh effect points in the same direction but misses the threshold.

There is no second confirmatory endpoint in this preregistration. Per-hop plots, parameter values, robustness runs,
and corpus diagnostics may be reported as descriptive context, but they do not change the decision rule.

## Frozen Model Family

Use the same continuous residual setup as `fit_hetero.py`.

1. Score each pair with the same judge model and prompt template used for PR #3517 (`gpt-5.5-low`). If that judge is
   unavailable, amend this preregistration before any fresh labels exist rather than substituting a new judge
   post-hoc.
2. Collapse fuzzy LLM labels to two continuous targets:
   - `D = max(subcategory, subtopic, element_of, super_category)`, using `mu_fwd` for directional relations.
   - `S = max(see_also, assoc)`, using symmetric `mu`.
3. Compute the model readouts used by the mean model:
   - `μ_D = max(μ_HIER(x,y), μ_HIER(y,x))`
   - `μ_S = μ_SYM(x,y)`
   - `d = hit_prob(parents, x, y)`
4. Fit marginal linear means for `D` and `S` on train only:
   - `D ~ [μ_D, μ_S, d, 1]`
   - `S ~ [μ_D, μ_S, d, 1]`
5. Compare bivariate-Gaussian NLL on the held-out residuals `(r_D, r_S)`.

The confirmatory model is the 6-parameter smooth covariance from `fit_hetero.py`:

```text
σ_D(hop) = exp(a_D + b_D * hop)
σ_S(hop) = exp(a_S + b_S * hop)
ρ(hop)   = tanh(c + e * hop)
```

Parameters are fit by MLE on train residuals only. Held-out NLL is evaluated with held-out pairs' true hop values.
No per-hop oracle bins, splines, isotonic fits, logistic alternatives, corpus-specific hand tuning, transformed hop
scales, or extra predictors enter the confirmatory decision.

**Baseline:** constant-Σ bivariate Gaussian on the same train residuals:

```text
σ_D = std(r_D_train)
σ_S = std(r_S_train)
ρ   = corr(r_D_train, r_S_train)
```

The primary statistic for each split is:

```text
gain = mean_NLL_constantΣ_heldout - mean_NLL_Σ(hop)_heldout
```

Positive gain means the hop-conditional covariance predicts held-out continuous labels better.

## Fresh Corpus

The confirmatory corpus must be selected before any fresh pair scoring, and it must be structurally and temporally
distinct from the exploratory `100k_cats` corpus.

Frozen selection rule:

1. Define the exploratory node block as every category title appearing in the PR #3517 `100k_cats/category_parent.tsv`
   graph, including both child and parent columns after the same title normalization used by the graph loader.
2. Build a candidate Wikipedia category graph from a later Wikipedia category dump than the one used for
   `100k_cats`. If the exact exploratory dump date is unavailable, use the earliest later dump available after
   2026-07-07 and record the dump identifier.
3. Remove every candidate edge whose child or parent title appears in the exploratory node block. Then remove any
   candidate node that is connected to an exploratory node by a retained edge, if such edges survive because of
   normalization differences.
4. Choose one top-level slice that is not represented in the exploratory sample's seed/root categories. The default
   slice is the first alphabetically, after normalization, among eligible high-level category subgraphs with at least
   300 descendant nodes and enough depth to supply hops 1 through 5. If that slice cannot produce the required pairs,
   take the next alphabetically eligible slice. Record the chosen root(s) before scoring.
5. Sample at least 250 multi-hop descendant/ancestor pairs from the retained graph, targeting roughly balanced hop
   counts over hops 1..5. Use the same `transitive_h{hop}` style labels and exclude duplicate unordered pairs.

Hard exclusion: no category title may appear in both the exploratory node block and the confirmatory scored pairs.
If this exclusion leaves fewer than 250 scoreable multi-hop pairs, stop and report the preregistered sampling failure
rather than relaxing the no-overlap rule.

## Split Protocol

Use the same descendant-disjoint split class that corrected the exploratory leakage risk:

- 40 random splits.
- 30% of descendant endpoints held out in each split.
- A descendant's h=1..5 pairs must all fall on the same side of a split.
- Ancestor endpoints are not required to be disjoint; this is descendant-disjoint, not both-endpoint-disjoint.
- Seeds are `0..39`, unless fewer valid splits survive minimum-size guards. If any split is skipped, report the count
  and reason.

Minimum valid split guard: at least 30 train pairs and 12 held-out pairs after all filters.

## Permutation Test

Use one one-sided hop-shuffle permutation test on the averaged statistic.

Observed statistic:

```text
T_obs = mean over valid splits of (NLL_constantΣ - NLL_Σ(hop))
```

Null:

```text
Shuffle hop labels across pairs, preserving pairs, targets, features, descendant split assignments, and all fitted
mean-model behavior. Refit the smooth Σ(hop) model under each shuffled-hop assignment and recompute the same averaged
gain statistic.
```

Permutation count: `K >= 1000`.

P-value:

```text
p = (1 + count(T_null >= T_obs)) / (K + 1)
```

Report:

- `T_obs`
- null mean
- null 95th percentile
- `K`
- exact permutation p-value above
- number of valid splits and mean held-out pairs per split

No repeated-split standard error is to be used as the confirmatory significance claim.

## No Post-hoc Forks

The following are explicitly out of scope for the confirmatory decision:

- changing the smooth `exp/tanh` functional form after seeing the fresh labels;
- swapping in spline, per-hop-bin, isotonic, or logistic covariance heads;
- changing `p < 0.01` to another threshold;
- changing held-out fraction, split unit, split count, or random seeds because the result is weak;
- adding judge features, prompt variants, corpus indicators, or hand-cleaned exclusions after inspecting failures;
- testing multiple fresh slices and reporting only the favorable one.

If a bug in the preregistered implementation is found before scoring, fix it and amend this file before the fresh
labels exist. If a bug is found after scoring, report both the preregistered result and the corrected exploratory
rerun, but do not relabel the corrected rerun as the preregistered confirmation unless the correction was purely
mechanical and blind to outcomes.

## Literature Grounding

This preregistration is deliberately conservative because the exploratory result sits at the intersection of three
well-known statistical pressures:

- **Covariance shrinkage.** Ledoit and Wolf's covariance-shrinkage work frames why a structured/smooth covariance can
  beat a noisy empirical covariance estimate in small samples: the target is not "more parameters"; it is a better
  conditioned estimator than raw per-hop covariance bins. The exploratory smooth-vs-oracle win is consistent with
  that shrinkage story, not proof that the log-linear/tanh form is true.
- **Heteroscedastic hierarchical shrinkage.** Kou and Yang's heteroscedastic hierarchical-linear shrinkage setting is
  the closest conceptual neighbor: variance differs across units/conditions, and useful estimators borrow structure
  rather than treating every group as independent. Here, hop is the conditioning variable and `Σ(hop)` is the
  structured borrowing device.
- **Garden of forking paths.** Gelman and Loken's warning is exactly the caveat in PR #3517: even a defensible final
  analysis can overstate evidence when details were selected after seeing the data. This document freezes the data,
  model, split, statistic, null, and threshold before the fresh labels exist.

References:

- Ledoit, O. and Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal
  of Multivariate Analysis*, 88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4
- Ledoit, O. and Wolf, M. (2012). "Nonlinear shrinkage estimation of large-dimensional covariance matrices."
  *Annals of Statistics*, 40(2), 1024-1060. https://doi.org/10.1214/12-AOS989
- Kou, S. C. and Yang, J. J. (2015). "Optimal shrinkage estimation in heteroscedastic hierarchical linear models."
  arXiv:1503.06262. https://arxiv.org/abs/1503.06262
- Gelman, A. and Loken, E. (2013). "The garden of forking paths: Why multiple comparisons can be a problem, even when
  there is no fishing expedition or p-hacking and the research hypothesis was posited ahead of time."
  https://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf

## Carry-forward Limitations

Even a confirmed result would inherit these limitations:

- **Single LLM judge.** Labels still come from one judge/prompt (`gpt-5.5-low`), not independent human adjudication
  or a second model with calibrated disagreement.
- **Descendant-disjoint, not both-endpoint-disjoint.** Shared ancestors can recur across train/held pairs, so graph
  topology may leave residual dependence.
- **Functional form as regularization.** The smooth `exp/tanh` covariance head is pre-registered as a regularizer and
  predictive model, not as a claim about the true generative form of semantic drift.
- **Graph-topological dependence.** Category graphs have shared ancestry, local density variation, and extraction
  artifacts. The permutation test breaks pair-hop association, but it does not make the graph an IID sample.
- **Corpus scope.** Confirmation on one fresh Wikipedia slice supports this Wikipedia-category regime. It does not by
  itself prove universality across SimpleMind, Pearltrees, or other concept graphs.

## Reporting Template

The later confirmatory report should include, at minimum:

```text
fresh corpus dump/root:
node-overlap check with 100k_cats:
n scored pairs:
hop counts:
judge/prompt/model:
valid descendant-disjoint splits:
mean held pairs/split:
observed mean gain:
hop-shuffle null mean:
hop-shuffle null 95%ile:
K:
permutation p:
decision: confirmed / not confirmed
```

Interpretation must follow the decision rule above. A miss at `p >= 0.01` is not a "near-confirmation"; it means the
effect stays exploratory until another preregistered fresh-corpus test is written before the next data collection.
