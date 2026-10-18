# Cheap-judge JointPosterior comparison: an explicit decision-space bridge

Status: implementation and declared primary protocol for the post-#3648 validation work.

## Why the obvious comparison is invalid

`campaign_scored.tsv` does not contain honest relation-class targets in `cur_rel`. All 1,770 rows have
`cur_rel=subcategory`; that field records the relation used by the structural sampler, while the campaign
deliberately includes siblings, cousins, and random pairs. Training or evaluating `JointPosterior` against
`cur_rel` would therefore measure the ability to predict a constant.

The operating judge does provide eight relation probabilities and eight relation-specific μ values. The dense
Gaussian conditioner in #3648 does **not** retain those eight values. It reduces them to two continuous states:

- `D = max(μ_element_of, μ_subcategory, μ_subtopic, μ_super_category)`
- `S = max(μ_see_also, μ_assoc)`

No honest decoder can recover which directional or symmetric subtype attained the maximum. Consequently this
work does not claim an eight-way relation comparison.

## Declared common decision space

The comparison aggregates the GPT-5.5 operating judge's probabilities before taking an argmax:

| macro decision | summed operating-judge probabilities |
|---|---|
| directional | element_of, subcategory, subtopic, super_category |
| symmetric | see_also, assoc |
| other | unknown, none |

This target is **GPT-5.5 macro-decision fidelity**, not independent truth. The full aggregated probability
vector, top-two margin, tie flag, and fixed-order tie resolution are persisted per comparison row; the hard
decision is used because the repository's `JointPosterior` is a hard-class discriminative head.

### Do not confuse within-judge pooling with across-expert fusion

The #3648 D/S target uses a hard maximum over relation-specific μ values emitted in **one GPT-5.5 response**.
It is a within-judge reduction, not “select the most confident expert.” JointPosterior versus correlated
Gaussian is a separate, downstream question about combining prior, graph, and Luna experts.

Hard max is inherited from #3648, not assumed optimal. The runner audits two bounded alternatives without
changing the primary estimator:

- probability-weighted pooling, normalizing the judge's relation probabilities within the directional or
  symmetric group;
- temperature-softmax pooling of relation μ values (default `T=0.10`), which approaches max as `T→0` but is
  smoother and remains between the group's minimum and maximum.

For each alternative it prints the full-matched-corpus D/S shift and correlation versus hard max, then refits
the train-only `D,S → macro decision` bridge and reports a finite-sample linear-reference metric. This is only
a same-response target-construction diagnostic; it has no paired selection interval.
A fair production ablation must rebuild *all* targets and sources—GPT-5.5, Luna, and model readouts—with the
same pooling rule, then refit covariance blocks on the same node split. Mixing a soft-pooled target with
hard-max prior or measurement channels would change the estimand and is prohibited. That end-to-end pooling
ablation remains follow-up work if a preregistered, paired diagnostic materially improves held log-loss/AURC.

## Same-split comparison

For each corpus, the shared `node_disjoint_pair_split` considers 64 seeded node—not row—assignments and chooses
one using retained-pair coverage plus macro-class strata. Pairs crossing the node partition are dropped. Thus
no held endpoint enters this follow-up's calibration or combiner fit. Candidate selection never inspects model
outcomes. This endpoint split is not edge-disjoint or k-hop-isolated from the ambient parent graph used for graph
features. It also does not prove the upstream `model_prod_namecond.pt` checkpoint never saw those node identities;
its campaign independence is audited, but endpoint independence from all earlier training is not.

Every calibration is fit on the same training rows:

1. campaign-independent model prior `(prior_D, prior_S)`;
2. affine graph-D calibration;
3. linear graph-S calibration from the four #3648 structural features;
4. global per-channel affine Luna calibration;
5. the dense joint error covariance `(P, C, R)`;
6. every discriminative or factored decision head.

The common source vector is:

```text
[prior_D, prior_S, graph_D, graph_S, luna_D, luna_S]
```

### Graph supervision has two roles

The graph judge is not only a last-mile measurement for the final posterior. Its nearly free structural labels
can also familiarize or adapt a model to a particular dataset: its entities, vocabulary, topology, and recurring
relation patterns. That representation-learning role may improve later priors or features even when graph_S has
little incremental value after Luna in a fixed downstream fusion ladder.

The held-row graph ablations in this work estimate only the first role: the immediate value of graph channels as
posterior inputs. They do not estimate the value of graph-generated supervision for dataset adaptation. A small
post-Luna increment must therefore not be interpreted as a reason to remove graph supervision from the broader
dataset-learning pipeline; that second role needs its own frozen-initialisation, matched-training ablation.

The methods are:

- **JointPosterior:** multinomial LR over the full source vector. This is the train-standardized learned
  combiner; its raw probability calibration is audited rather than assumed.
- **Dense correlated Gaussian:** fuse graph and Luna measurements into `N(D,S)`, preserving the fitted
  prior–measurement cross-covariance. A train-only multinomial bridge estimates `p(decision | D,S)`.
  Two-dimensional Gauss-Hermite quadrature reports `E[p(decision | Z)]` for `Z ~ N(μ_post,P_post)`, so
  posterior covariance affects confidence.
- **D/S linear reference:** apply the same train-only LR bridge to the held row's operating-judge D/S. It is
  not deployable and not a Bayes ceiling: its error also includes finite training data, LR misspecification,
  optimisation, and possible incoherence between relation probabilities and relation-specific μ values.
- **Factored controls:** equal-weight and separability-weighted products of one-dimensional histograms.
  They are included to expose the cost of pretending correlated sources are independent.

The prior-only, prior+graph, prior+Luna, and all-source rungs are fit for both learned families. This separates
the graph and cheap-judge contributions without changing held rows.

## Metrics and uncertainty

Report on the same held rows:

- accuracy;
- multiclass log-loss;
- ECE with 10 equal-width confidence bins;
- AURC gated by the top-1 minus top-2 posterior margin;
- a paired AURC difference, `JointPosterior(all) - Gaussian(all)`.

AURC intervals resample endpoint-connected components. Every row sharing either node is in the same bootstrap
block, avoiding the false independence assumption of a row bootstrap. The report includes the number of
blocks and largest block size; a corpus dominated by one component cannot support a useful block-bootstrap
interval and must be reported as such.

AURC is retained as the registered selective-risk metric, but it can over-weight early high-confidence errors
and has finite-sample estimator bias. Report conclusions require the paired interval and effective block counts;
an AUGRC calculation is a separately named robustness check rather than a post-hoc replacement.

Source separability and the full Pearson correlation matrix are printed from training rows only. A source earns
its keep through a same-held-set rung/ablation improvement, not through a training correlation alone.

## Scope and non-claims

- This comparison cannot identify whether the Gaussian or joint method recovers the correct eight-way relation.
- Its labels and D/S bridge share the GPT-5.5 operating judge; it measures fidelity to that judge.
- The D/S linear reference diagnoses the bridge/representation combination; it does not isolate max-pooling
  information loss or upper-bound a learned source combiner.
- Alternative within-judge pooling rows are same-response linear-reference diagnostics, not an across-expert
  fusion result or a continuous-target selection experiment.
- Seed 0 is the declared primary node split; additional seeds are refitted sensitivity analyses and must not
  be pooled as independent rows. One split-specific bootstrap is not evidence of partition robustness.
- No held labels are used for post-hoc temperature calibration. ECE and log-loss therefore audit the raw
  cross-entropy-trained probabilities; a future calibrator needs a nested train/calibration partition.
- A future soft-target head could train against the aggregated three-way probability vector. It should be a
  separately named estimator, not silently substituted for the existing `JointPosterior`.

Implementation: `run_cheap_judge_joint_posterior.py`. Focused tests:
`test_cheap_judge_joint_posterior.py`.
