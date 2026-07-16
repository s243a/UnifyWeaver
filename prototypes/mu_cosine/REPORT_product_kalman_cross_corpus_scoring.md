# Public Cross-Corpus Judge Scoring and Frozen-Title Sensitivity

Status: completed public-data scoring stage. This run produces enwiki and Pearltrees judge labels and a descriptive
title-sensitivity control. It does **not** fit, select, or validate a Product-Kalman model.

## Question

Before comparing graph regimes, can we produce durable, schema-validated judge labels while separating the effect
of frozen title corrections from the non-determinism of the LLM judge?

This matters because a typo or malformed title can look like semantic or lateral drift. It also matters for the
Product-Kalman design: repeat variation in the judge is measurement noise and belongs in a calibrated observation
error model, not in the graph signal by default.

## Frozen design

- Corpora: 250 balanced hop-1..5 pairs from the enwiki Main-topic-classifications subtree and 250 from the public
  Pearltrees principal-parent view.
- Judge: `gpt-5.5-low`, using the existing `score_inferred_tail.py` prompt and relation schema.
- Response contract: exact pair IDs; all eight relation objects and required fields present; every numeric value in
  `[0,1]`.
- Checkpointing: one atomic JSON checkpoint per 10-row batch, bound to source, prompt, model, effort, sandbox, and
  judge hashes/fields. Completed runs were replayed with zero judge calls through the strict validator.
- Title policy: frozen before these labels. Enwiki corrected 0/250 pairs; Pearltrees corrected 53/250 pairs.
- Noise control: the same 53 Pearltrees source rows were independently rescored with their unchanged raw titles.

SimpleMind was not sent to the external judge. The local frozen policy identifies 36/250 affected rows, but title
disclosure was not assumed merely because the data exist locally.

## Results

All 500 public raw rows and all 106 Pearltrees comparison rows passed the strict response schema. Enwiki's audited
view is a clean zero-change control: its full audited scored TSV is byte-identical to its raw scored TSV.

Every Pearltrees structured response changed in both independent comparisons (53/53 corrected-title runs and
53/53 unchanged-title repeats). Exact JSON change is therefore not evidence of a title effect; it primarily shows
that this judge tier is non-deterministic at the granularity recorded here.

For the established campaign features

```text
D = max(mu_fwd over subcategory, subtopic, element_of, super_category)
S = max(mu over see_also, assoc)
```

the paired descriptive shifts were:

| feature | corrected-title changed | raw-repeat changed | corrected MAE | repeat MAE | MAE ratio | corrected RMSE | repeat RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `D` | 41/53 | 37/53 | 0.0909 | 0.0711 | 1.28 | 0.1247 | 0.1085 |
| `S` | 46/53 | 39/53 | 0.0951 | 0.0653 | 1.46 | 0.1151 | 0.0867 |

The corrected-title run moved both features more than this one repeat control on average. The excess mean absolute
shift was `+0.0198` for `D` and `+0.0298` for `S`. This is suggestive that title repair matters, especially for the
lateral feature, but it is not a causal estimate: there is only one repeat control, both comparisons share the same
raw baseline, and judge draws are not exchangeably replicated enough for an uncertainty interval.

Hop-level ratios are heterogeneous rather than monotone. For example, corrected-title `D` MAE is below repeat noise
at hop 1 (`0.83x`) and above it at hop 4 (`2.24x`). Small affected counts per hop (9-13) and a single repeat make
these descriptive diagnostics, not evidence for hop-dependent typo drift.

## Interpretation

1. Title normalization is scientifically relevant. On these 53 reviewed Pearltrees pairs, corrected titles changed
   `D/S` more than one same-title repeat on average.
2. Judge noise is not small. A same-title rerun changed `D` on 37/53 and `S` on 39/53 pairs, with MAE around
   `0.07` and `0.065`. Treating one judge pass as error-free would confound measurement noise with corpus geometry.
3. The current result supports the uncertainty-estimation guardrail: estimate judge observation error on a
   calibration split and compare candidates by identity-disjoint held-out NLL/calibration. It does not promote PoE,
   Product-Kalman, or a graph topology feature.
4. The user's structural hypothesis remains open. A more tree-like principal-parent graph may offset noisier titles,
   but that tradeoff requires the later matched model evaluation; this scoring stage alone cannot decide it.

## Durable record

The complete public inputs and outputs are committed under
`repro/product_kalman_cross_corpus/`. `SHA256SUMS` verifies the bundle. Unlike earlier `/tmp`-only runs, the raw judge
responses themselves are durable, which is necessary because the judge is non-deterministic.

Key derived records:

- `enwiki_title_sensitivity.json`: explicit zero-correction control.
- `pearltrees_corrected_title_sensitivity.json`: corrected-title versus original response.
- `pearltrees_raw_repeat_sensitivity.json`: unchanged-title repeat versus original response.
- `pearltrees_title_vs_repeat_contrast.json`: deterministic side-by-side excess and ratio summaries.

## Next decision

The next empirical stage should estimate the learned joint posterior and Product-Kalman candidate on identity- or
node-disjoint calibration/evaluation partitions, using the audited full views. The comparison must report held-out
NLL, calibration, and margin-gated AURC against the existing `JointPosterior` baseline. SimpleMind can join that run
only after explicit approval to disclose its 250 local pair titles to the external judge, or after choosing a local
judge that keeps them on-device.
