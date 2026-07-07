# Hop-Conditional Covariance in Wikipedia Category Relations

*Toward fuzzy concept-graph uncertainty modeling, with one pre-registered Wikipedia-category confirmation.*

*Manuscript scaffold, 2026-07-07. This draft consolidates the exploratory `Sigma(hop)` finding, the
pre-registration, and the fresh-corpus confirmation. It is intentionally conservative: one confirmed Wikipedia
category result, not a universal claim about all concept graphs.*

## Abstract

Wikipedia category-relation labels are often treated as independent scalar judgments: a pair is hierarchical,
associative, both, or neither. In fuzzy LLM-scored category labels, however, the uncertainty structure may vary with
where a pair sits in the graph. We test, within a Wikipedia category regime, whether the covariance of directional and
symmetric relation residuals varies with graph-hop distance. An exploratory Wikipedia multi-hop sample suggested that
a smooth hop-conditional covariance model, `Sigma(hop)`, improves held-out bivariate Gaussian residual likelihood over
a constant-covariance baseline. Because that specification was chosen after exploration, we pre-registered a single
confirmatory test on a fresh Wikipedia category slice with no node overlap against the exploratory graph. On 250
fresh `Behavior`-slice pairs,
balanced across hops 1..5 and scored by the same `gpt-5.5-low` prompt family, the preregistered test confirmed the
effect: mean held-out NLL gain `+0.059799`, hop-shuffle null mean `-0.009487`, null 95th percentile `+0.000456`,
`K=1000`, one-sided permutation `p < 0.001` (finite-K floor `0.000999`). The result supports hop-conditioned
uncertainty modeling for this
Wikipedia-category regime as a predictive NLL result, while retaining limitations from a single LLM judge,
descendant-disjoint rather than both-endpoint-disjoint splits, and graph-topological dependence.

## Claims

1. **Confirmed claim:** on a fresh no-overlap Wikipedia category slice, smooth `Sigma(hop)` predicts held-out fuzzy
   directional/symmetric residuals better than constant-Sigma under the preregistered one-sided hop-shuffle test.
2. **Interpretive claim:** the gain is evidence for hop-dependent uncertainty geometry, not proof that the log-linear
   variance and tanh-correlation functional form is generative truth.
3. **Scope claim:** the confirmed result is for one Wikipedia category regime. It is a confirmation of predictive
   likelihood under this evaluation setup, not a general validation of concept-graph uncertainty theory. It should
   motivate, not replace, separate confirmatory tests on SimpleMind, Pearltrees, or other concept graphs.

## Background

The upstream modeling problem is to estimate fuzzy relation memberships between two concept nodes. For a pair
`(x, y)`, the LLM judge gives directional memberships such as `subcategory`, `subtopic`, `element_of`, and
`super_category`, plus symmetric memberships such as `see_also` and `assoc`. Earlier joint-posterior work showed
that binary co-occurrence structure can matter, but the continuous fuzzy-label setting refined the question: after a
mean model explains the expected directional and symmetric scores, does the residual covariance carry useful
structure?

The exploratory answer was yes, but only when covariance was allowed to vary with hop. Low-hop Wikipedia category
pairs are usually strongly directional and weakly symmetric; deeper pairs blur toward looser association. A constant
covariance must average incompatible residual geometries across that transition. The proposed remedy is a small
smooth covariance head conditioned on shortest upward graph-hop distance.

This sits near three methodological literatures:

- covariance shrinkage, because a smooth covariance curve can beat noisy per-hop bins in small samples;
- heteroscedastic hierarchical shrinkage, because hop is a structured condition over which uncertainty is shared;
- garden-of-forking-paths concerns, because the exploratory specification was reached after iteration and needed a
  fresh preregistered confirmation.

## Model

For each scored pair, collapse fuzzy labels to two continuous targets:

```text
D = max(subcategory, subtopic, element_of, super_category)
S = max(see_also, assoc)
```

Compute three predictors from the existing model and graph:

```text
mu_D = max(mu_HIER(x,y), mu_HIER(y,x))
mu_S = mu_SYM(x,y)
d    = hit_prob(parents, x, y)
```

For each descendant-disjoint split, fit marginal linear means on train only:

```text
D ~ [mu_D, mu_S, d, 1]
S ~ [mu_D, mu_S, d, 1]
```

Then evaluate the bivariate Gaussian likelihood of held-out residuals `(r_D, r_S)` under two covariance models.

Baseline constant-Sigma:

```text
sigma_D = std(r_D_train)
sigma_S = std(r_S_train)
rho     = corr(r_D_train, r_S_train)
```

Confirmatory smooth `Sigma(hop)`:

```text
sigma_D(hop) = exp(a_D + b_D * hop)
sigma_S(hop) = exp(a_S + b_S * hop)
rho(hop)     = tanh(c + e * hop)
```

The primary split statistic is:

```text
gain = mean_NLL_constantSigma_heldout - mean_NLL_Sigma(hop)_heldout
```

Positive gain means the hop-conditional covariance predicts held-out fuzzy labels better.

## Exploratory Result

The exploratory multi-hop Wikipedia run used about 250 pairs from the original `100k_cats` graph. After review, the
analysis was corrected to use continuous fuzzy labels, descendant-disjoint splits, and a calibrated hop-shuffle
permutation test rather than repeated-split standard errors. Under the finally adopted specification, smooth
`Sigma(hop)` beat constant-Sigma by about `+0.094` held-out joint NLL, with a hop-shuffle permutation `p=0.001`
(`K=1000`).

That result was not treated as confirmatory because the final specification emerged through iterative analysis on the
same data. The appropriate next step was therefore a pre-registration before a new corpus existed.

## Pre-Registered Confirmation

The pre-registration froze one endpoint:

```text
Confirmed iff observed mean gain > 0 and one-sided hop-shuffle permutation p < 0.01.
```

It also froze the model family, split protocol, permutation null, judge/prompt family, and no-overlap sampling rule.
The fresh corpus had to share no category titles with the exploratory `100k_cats` graph.

The sampled fresh corpus was produced from the `enwiki_cats_correct` scoped LMDB with a real MediaWiki title layer.
The selected slice was `Behavior`, under `Main_topic_classifications`. After excluding exploratory-overlap and
admin-title nodes, the retained slice contained 75,901 nodes and 99,986 retained edges. The score-input corpus had
250 pairs, exactly 50 at each shortest upward hop 1..5, and zero node overlap with the exploratory graph.

## Confirmatory Result

```text
fresh corpus: enwiki_cats_correct scoped LMDB, selected root Behavior
node overlap with exploratory 100k_cats: 0
scored pairs: 250
hop counts: {1: 50, 2: 50, 3: 50, 4: 50, 5: 50}
judge/prompt: gpt-5.5-low via score_with_codex.py / score_inferred_tail.py prompt
valid descendant-disjoint splits: 40
mean held pairs/split: 75.0
constant-Sigma NLL: -0.604010
Sigma(hop) NLL: -0.663809
observed mean gain: +0.059799
hop-shuffle null mean: -0.009487
hop-shuffle null 95%ile: +0.000456
permutations: K=1000
permutation p: 0.000999
decision: confirmed
```

With `K=1000`, `p=0.000999` is the finite-permutation floor `(1 + 0) / (1000 + 1)`: no shuffled-hop null run reached
the observed gain. In prose this should usually be reported as `p < 0.001` or as the finite-K floor above, rather
than as more precision than the permutation design supports. The confirmatory gain is smaller than the exploratory
`+0.094` estimate, which is consistent with post-exploratory attenuation on a fresh slice; the pre-registered
decision rule required a positive gain and `p < 0.01`, not replication of the exploratory effect size.

As a descriptive stability check, 33 of 40 descendant-disjoint split gains were positive. The split-gain standard
deviation was `0.051246`, giving a descriptive split-resampling SE of `0.008103` and a bootstrap-over-splits 95%
interval of approximately `[+0.044183, +0.075277]`. These split-resampling numbers are not used as the significance
claim because the splits share one dataset; the confirmatory inference remains the hop-shuffle permutation test.

## Interpretation

The confirmed gain means that, in this fresh Wikipedia slice, graph-hop distance is associated with better prediction
of the covariance of fuzzy directional and symmetric residuals beyond the marginal mean model. In geometric terms, the
fitted residual uncertainty ellipse is not constant across hop: the decoupling transformation useful for shallow
category relations differs from that for deeper pairs, which may reflect semantic drift or other hop-dependent
structural factors. A constant-Sigma baseline averages these fitted geometries; `Sigma(hop)` provides a six-parameter
hop-conditioned regularizer.

This supports the design idea that uncertainty should sometimes be modeled as structured conditional covariance, not
only as independent scalar confidence. It also gives a cleaner statistical target for later model work: rather than
adding a generic cross pseudo-judge everywhere, use conditional covariance where held-out data show residual coupling
that changes with graph position.

This paper does not yet implement a Kalman-style update rule or a fully calibrated predicted-error update rate. A
product term or product-of-experts score can be a useful feature or baseline, but it is not by itself a likelihood for
the measured error unless the expert-error distribution and dependence structure are specified. The present
`Sigma(hop)` result is closer to an error-modeling step: it estimates how residual variance and correlation change
with hop after the mean model, while leaving full calibration of predicted error, expert covariance, and update-rate
control as future work.

## Limitations

- **Single, non-deterministic LLM judge.** The fresh labels come from one judge/prompt family. Confirmation of the
  statistical effect is not equivalent to human validation of every fuzzy relation label, and re-querying the same
  model name later may not reproduce the same scores. The hashed raw response file is the run record for judge
  outputs.
- **Descendant-disjoint splits only.** Descendant endpoints are disjoint across train and held-out sets, but ancestors
  may recur. Shared ancestors and graph topology can still induce residual dependence.
- **Functional form as regularization.** The `exp/tanh` covariance head is confirmed as a predictive regularizer here,
  not as the true generative law of semantic drift or of any other hop-dependent mechanism.
- **One fresh Wikipedia slice.** The result confirms the effect in a structurally distinct Wikipedia category slice.
  It does not prove the same curve applies to SimpleMind, Pearltrees, or all concept graphs. Hop-5 category pairs may
  also be more semantically heterogeneous than shallow pairs, so the confirmed result averages over a deliberately
  bounded hop range rather than proving uniform behavior at arbitrary depth.
- **Ephemeral local artifacts.** Raw scoring outputs and caches currently live under `/tmp/mu_data/...`; the committed
  durable record is the pre-registration, sampler/runner code, and confirmatory report.

## Figures And Tables To Add

1. Table: exploratory vs confirmatory result summary.
2. Figure: hop-wise empirical `(D, S)` scatter or covariance ellipses for the confirmatory `Behavior` slice.
3. Figure: hop-shuffle null distribution with observed statistic marked.
4. Figure: fitted `sigma_D(hop)`, `sigma_S(hop)`, and `rho(hop)` curves on representative splits.
5. Appendix table: candidate-slice selection manifest, including skipped `Academic_disciplines` and selected
   `Behavior`.

## Reproducibility Anchors

```text
pre-registration: prototypes/mu_cosine/PREREG_sigma_hop_confirmatory.md
fresh sampler: prototypes/mu_cosine/sample_sigma_hop_fresh_corpus.py
confirmatory runner: prototypes/mu_cosine/sigma_hop_confirmatory.py
confirmatory report: prototypes/mu_cosine/REPORT_sigma_hop_confirmatory.md
exploratory report: prototypes/mu_cosine/REPORT_two_judge_posterior.md
```

The preregistration was committed as `638825558` at `2026-07-06T21:06:22-06:00`, before the confirmatory report
commits on `2026-07-07`. A submission version should preserve or cite this git ordering explicitly.

Local run artifacts recorded by the report:

```text
score input: /tmp/mu_data/sigma_hop_fresh_pairs.tsv
sampling manifest: /tmp/mu_data/sigma_hop_fresh_manifest.json
raw judge responses: /tmp/mu_data/sigma_hop_fresh_responses_gpt55low.txt
ingested judge scores: /tmp/mu_data/sigma_hop_fresh_scored_gpt55low.tsv
retained-slice e5 cache: /tmp/mu_data/sigma_hop_behavior_slice_e5.pt
result JSON: /tmp/mu_data/sigma_hop_confirmatory_result.json
```

These hashes identify the completed local run only; because the files currently live under `/tmp/mu_data/...`, the
hash table is not a durable archive. Before external submission, the small text artifacts should be committed in a
reproducibility directory or deposited in a stable data store.

Small text-artifact hashes from the completed run:

| artifact | bytes | SHA-256 |
|---|---:|---|
| `/tmp/mu_data/sigma_hop_fresh_pairs.tsv` | 27,022 | `5a09cbd86a9608790b2620d6710c9799455d2590c22cc69790baf60e3827d60d` |
| `/tmp/mu_data/sigma_hop_fresh_manifest.json` | 2,931 | `704fcfa9b359dceb3a4ec7d2828df8492e520bbfc63153230b0d186377ae42b1` |
| `/tmp/mu_data/sigma_hop_fresh_responses_gpt55low.txt` | 95,026 | `ab306afe2c5a2450e2ee4de7d0b599a2010cf2b8e54736b4dfb6608df1a5df97` |
| `/tmp/mu_data/sigma_hop_fresh_scored_gpt55low.tsv` | 51,456 | `c256a1b7fb44c8537b78a163b7d0d8b0d660ae726815e1fedd21d699016e241e` |
| `/tmp/mu_data/sigma_hop_confirmatory_result.json` | 1,773 | `0186023e9a3dda844a733044c6798685123daee778cc6bba579c3b5ae340b3b5` |
| `/tmp/mu_data/sigma_hop_confirmatory_REPORT.md` | 941 | `17b957a93cad3c0ef6f294041b28b93ea57be7ef84682d1877ae1d58f0a3b89d` |

The retained-slice e5 cache is a large regenerable torch artifact, so the manuscript records its path and generation
rule rather than treating it as a text artifact to archive inline. It was generated with `build_e5_tables` from
`mu_attention.py` over all 75,901 retained `Behavior` slice node titles using `intfloat/e5-small-v2` query/passage
prefixes and batch size 128 on CUDA.

## References To Carry Forward

- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
  *Journal of Multivariate Analysis*, 88(2), 365-411.
- Ledoit, O. and Wolf, M. (2012). Nonlinear shrinkage estimation of large-dimensional covariance matrices.
  *Annals of Statistics*, 40(2), 1024-1060.
- Kou, S. C. and Yang, J. J. (2015). Optimal shrinkage estimation in heteroscedastic hierarchical linear models.
  arXiv:1503.06262.
- Gelman, A. and Loken, E. (2013). The garden of forking paths. Unpublished manuscript; submission drafts may also
  cite the later *American Scientist* discussion depending on venue expectations.
- Zheng, L. et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.

## Submission Readiness Checklist

- Commit or externally archive durable copies of the fresh score input, raw responses, manifest, and result JSON;
  this scaffold records hashes for the small text artifacts but does not archive them.
- Confirm the git ordering evidence in the final manuscript: preregistration commit `638825558` precedes fresh-corpus
  sampling/scoring and result reporting.
- Pin the judge model/API version if the provider exposes one; otherwise state that `gpt-5.5-low` is a run label and
  the hashed raw response file is the reproducible judge-output record.
- Generate the null-distribution and covariance-curve figures from the saved run artifacts.
- Decide whether to position this as a workshop note, arXiv technical report, or methods appendix for the broader
  UnifyWeaver relation-modeling work.
- Run one independent-judge audit sample, or explicitly frame single-judge labels as a limitation rather than a
  resolved validity claim.
- Decide whether SimpleMind/Pearltrees follow-ups belong in this paper or a separate cross-corpus extension.
