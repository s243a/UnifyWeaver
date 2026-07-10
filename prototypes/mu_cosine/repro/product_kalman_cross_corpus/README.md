# Public cross-corpus campaign scoring artifacts

This directory is the durable public-data record for
`REPORT_product_kalman_cross_corpus_scoring.md`. It contains the 250-pair enwiki and Pearltrees campaign inputs,
canonical `gpt-5.5-low` responses, standard ingested score tables, frozen-title views, and the Pearltrees
unchanged-title repeat control. Verify every file with:

```bash
cd prototypes/mu_cosine/repro/product_kalman_cross_corpus
sha256sum -c SHA256SUMS
```

## Views

- `*_pairs_raw.tsv` and `*_score_input_raw.tsv`: deterministic sampler outputs before title policy application.
- `*_pairs_audited.tsv` and `*_score_input_audited.tsv`: the frozen pre-scoring title-policy views.
- `*_raw_responses.txt` and `*_raw_scored.tsv`: complete 250-row judge outputs and standard ingested tables.
- `pearltrees_corrected_title_*`: the 53 rows affected by the frozen Pearltrees title policy.
- `pearltrees_raw_repeat_*`: the same 53 source rows rescored with unchanged raw titles to expose judge repeat noise.
- `*_merged_audited_*`: full 250-row views formed by replacing affected raw responses with corrected-title responses.
- `*_sensitivity.json`, `*_deltas.tsv`, and `pearltrees_title_vs_repeat_contrast.json`: deterministic descriptive
  comparisons. The contrast is not a causal estimator.

The run manifests preserve their original `/tmp/mu_data/...` execution paths, while machine-specific sampler
source roots are normalized to `<repo>` and `<local-public-source>`. These are provenance fields,
not dependencies: the files named by the manifests are archived here under stable names, and `SHA256SUMS` is the
archive integrity contract.

## Scope and privacy

Only public enwiki and Pearltrees inputs are archived. SimpleMind campaign titles, mappings, and unscored inputs are
intentionally absent: they were neither sent to the external judge nor committed because their disclosure status
was not established. The local frozen policy identified 36 affected SimpleMind pairs, but no SimpleMind score or
scientific result exists in this run.

All judge outputs are non-deterministic observations from the named model tier, not reproducible model evaluations
in the bit-for-bit inference sense. The archived raw responses are the record of what this run observed.
