# Confirmatory Sigma(hop) run: confirmed on a fresh Behavior slice

*Run 2026-07-07 against `PREREG_sigma_hop_confirmatory.md`, after sampling and scoring a fresh Wikipedia category
slice that shares no category nodes with the exploratory `100k_cats` graph.*

## Decision

The preregistered confirmatory decision is **confirmed**: smooth `Sigma(hop)` beat constant-Sigma on held-out
continuous-mu residual likelihood, and the one-sided hop-shuffle permutation test passed the frozen `p < 0.01`
threshold.

```text
fresh corpus dump/root: enwiki_cats_correct scoped LMDB; title layer mediawiki_page_titles; scope Main_topic_classifications; selected root Behavior
node-overlap check with 100k_cats: passed, 0 overlapping category titles
n scored pairs: 250
hop counts: {1: 50, 2: 50, 3: 50, 4: 50, 5: 50}
judge/prompt/model: gpt-5.5 with model_reasoning_effort=low via score_with_codex.py / score_inferred_tail.py prompt
valid descendant-disjoint splits: 40
mean held pairs/split: 75.0
observed mean gain: +0.059799
constant-Sigma NLL: -0.604010
Sigma(hop) NLL: -0.663809
hop-shuffle null mean: -0.009487
hop-shuffle null 95%ile: +0.000456
K: 1000
permutation p: 0.000999
decision: confirmed
skipped splits: 0
```

At `K=1000`, `p=0.000999` is the finite-permutation floor `(1 + 0) / (1000 + 1)`: no shuffled-hop null run reached
the observed mean gain.

## Artifacts

```text
score input: /tmp/mu_data/sigma_hop_fresh_pairs.tsv
sampling manifest: /tmp/mu_data/sigma_hop_fresh_manifest.json
raw judge responses: /tmp/mu_data/sigma_hop_fresh_responses_gpt55low.txt
ingested judge scores: /tmp/mu_data/sigma_hop_fresh_scored_gpt55low.tsv
retained-slice e5 cache: /tmp/mu_data/sigma_hop_behavior_slice_e5.pt
result JSON: /tmp/mu_data/sigma_hop_confirmatory_result.json
rendered run report: /tmp/mu_data/sigma_hop_confirmatory_REPORT.md
runner commit: 98efc2fdc Add LMDB graph support to sigma-hop confirmatory runner
```

The `/tmp/mu_data/...` paths above are local run artifacts, not durable repository records. The durable record in this
PR is this committed report plus the runner change needed to consume the LMDB feature graph.

The selected root was `Behavior`. The retained LMDB slice had 75,901 nodes after the preregistered no-overlap and
admin-title filtering. Sampling produced exactly 50 pairs at each shortest upward hop 1..5.

## Implementation Notes

Two mechanical issues were fixed before any gain/permutation result was computed:

- The confirmatory runner was originally TSV-only for feature-graph loading, while the fresh corpus came from the
  LMDB title graph. Commit `98efc2fdc` added an LMDB feature-graph adapter that reuses the sampler's retained-slice
  traversal. This changed no labels, model family, split protocol, permutation null, threshold, or decision rule.
- An endpoint-only e5 cache was insufficient because `Tokenizer` also embeds retained-slice ancestors. The final run
  used `/tmp/mu_data/sigma_hop_behavior_slice_e5.pt`, built over all 75,901 retained `Behavior` slice nodes with the
  same `intfloat/e5-small-v2` query/passage prefixes.

Two earlier run attempts stopped before producing a statistic: first because the clean worktree lacked the local
`model_prod.pt` artifact, then because the endpoint-only e5 cache lacked ancestor tokens. The completed run used the
exact local checkpoint at `/home/s243a/Projects/UnifyWeaver/prototypes/mu_cosine/model_prod.pt`.

## Interpretation

This closes the main post-exploratory caveat for the Wikipedia-category regime tested here: the smooth hop-conditional
covariance result transferred to a fresh, descendant-disjoint, no-overlap category slice under the frozen test. The
carry-forward limitations from the preregistration still apply: single LLM judge, descendant-not-both-endpoint split,
functional form as regularization, graph-topological dependence, and one Wikipedia slice rather than universality
across other concept graphs.
