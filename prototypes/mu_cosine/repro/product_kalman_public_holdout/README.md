# Public Product-Kalman Holdout Artifacts

This directory archives the inputs and outputs for `REPORT_product_kalman_public_holdout.md`.

## Contents

- `*_features.tsv`: fixed 250-row source/target tables used by every candidate.
- `*_features_manifest.json`: source provenance, hashes, model identity, and corpus diagnostics.
- `*_result.json`: complete split manifests, primary metrics, bootstrap intervals, stability summaries, and decision.
- `*_result.md`: compact rendered primary-split summaries.
- `*_primary.npz`: primary-split indices, targets, predictions, covariances, and classifier probabilities.
- `SHA256SUMS`: checksums for every archived artifact except itself.

The e5 caches are not archived because the feature tables contain their fixed outputs. Local Pearltrees exports and
the enwiki LMDB are likewise omitted; hashes or regeneration descriptions are retained in the manifests.

## Reproduce The Statistical Evaluation

```bash
python3 prototypes/mu_cosine/run_product_kalman_public_holdout.py \
  --features prototypes/mu_cosine/repro/product_kalman_public_holdout/enwiki_features.tsv \
  --seeds 40 --boot 1000 \
  --json-out /tmp/enwiki_result.json \
  --md-out /tmp/enwiki_result.md \
  --npz-out /tmp/enwiki_primary.npz
```

Replace `enwiki` with `pearltrees` for the second corpus. Byte equality of JSON is not guaranteed across numerical
library versions, but the split assignments and frozen decision rule are deterministic.
