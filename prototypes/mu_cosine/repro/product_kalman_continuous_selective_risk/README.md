# Product-Kalman Continuous Selective-Risk Artifacts

This directory archives the exact outputs for `REPORT_product_kalman_continuous_selective_risk.md`.

## Contents

- `*_result.json`: full split manifests, primary permutation/bootstrap results, hop diagnostics, and decision.
- `*_result.md`: concise rendered corpus summary.
- `*_primary.npz`: primary pair IDs, hops, targets, posterior means/covariances, predicted risks, and realized losses.
- `SHA256SUMS`: checksums for every archived artifact except itself.

The evaluator consumes the fixed public feature tables in the sibling `product_kalman_public_holdout` archive.

## Reproduce

```bash
python3 prototypes/mu_cosine/run_product_kalman_continuous_selective_risk.py \
  --features prototypes/mu_cosine/repro/product_kalman_public_holdout/enwiki_features.tsv \
  --seeds 40 --permutations 1000 --boot 1000 \
  --json-out /tmp/enwiki_selective_result.json \
  --md-out /tmp/enwiki_selective_result.md \
  --npz-out /tmp/enwiki_selective_primary.npz
```

Replace `enwiki` with `pearltrees` for the second corpus.
