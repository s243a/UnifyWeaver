# LMDB Policy Cache Materialization Probe - 2026-06-12

This probe separates cache residency from exact histogram materialization cost.
A policy-resident node can be useful enough to keep, but still too expensive to
materialize exactly.  When exact materialization hits a cap, the probe classifies
that node as a closed-form approximation candidate.  The initial fallback model
is a compact `binomial_support_prior` over the known root-distance support.

## Short-Budget Smoke

```bash
python3 scripts/lmdb_policy_cache_materialization_probe.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_policy_cache_materialization_smoke \
  --target-depths 3 \
  --children-per-node 16 \
  --frontier-limit 80 \
  --targets-per-depth 2 \
  --max-ancestor-nodes 1500 \
  --max-ancestor-edges 8000 \
  --root-distance-cap 32 \
  --cache-slots 128 \
  --admit-l-min 6 \
  --admit-l-max 8 \
  --sweep-cache-slots 128 \
  --sweep-admit-l-max 8 \
  --boundary-budgets 1,2,3,4 \
  --path-cap 10000 \
  --expansion-cap 50000 \
  --seed enwiki-mtc-policy-cache-materialization-smoke \
  --output-dir /mnt/c/Users/johnc/Scratch/policy-cache-runtime
```

| boundary_budget | resident | exact | closed_form | too_short | unresolved | expansion_capped | mean_nodes |
|----------------:|---------:|------:|------------:|----------:|-----------:|-----------------:|-----------:|
| 1 | 48 | 9 | 0 | 39 | 0 | 0 | 4.4 |
| 2 | 48 | 25 | 0 | 23 | 0 | 0 | 21.6 |
| 3 | 48 | 34 | 0 | 14 | 0 | 0 | 117.3 |
| 4 | 48 | 46 | 0 | 2 | 0 | 0 | 625.5 |

Short suffix budgets are cheap, but many resident nodes are simply too far from
root to be useful at the smallest budgets.  By budget 4 almost all resident
nodes materialize exactly without caps on this tiny sample.

## Expensive-Materialization Smoke

```bash
python3 scripts/lmdb_policy_cache_materialization_probe.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_policy_cache_materialization_budget8_smoke \
  --target-depths 3 \
  --children-per-node 16 \
  --frontier-limit 80 \
  --targets-per-depth 1 \
  --max-ancestor-nodes 1500 \
  --max-ancestor-edges 8000 \
  --root-distance-cap 32 \
  --cache-slots 128 \
  --admit-l-min 6 \
  --admit-l-max 8 \
  --sweep-cache-slots 128 \
  --sweep-admit-l-max 8 \
  --boundary-budgets 8 \
  --path-cap 10000 \
  --expansion-cap 50000 \
  --seed enwiki-mtc-policy-cache-runtime-debug \
  --output-dir /mnt/c/Users/johnc/Scratch/policy-cache-runtime
```

| boundary_budget | resident | exact | closed_form | too_short | unresolved | expansion_capped | mean_nodes | mean_fallback_width |
|----------------:|---------:|------:|------------:|----------:|-----------:|-----------------:|-----------:|--------------------:|
| 8 | 52 | 1 | 51 | 0 | 0 | 51 | 49038.5 | 2.1 |

This validates the fallback framing: deeper exact materialization can become
expensive even for near-root resident nodes.  Those capped nodes are not useless;
they are candidates for compact closed-form state.  The current probe records a
`binomial_support_prior` with support from `(L_min, min(L_max, boundary_budget))`
and either a partial-histogram estimate of `p` or a support-midpoint default.

## Next Step

Use this probe to choose a mixed materialization policy:

- exact histogram when a short suffix budget materializes without caps;
- no cache entry when the suffix budget is shorter than `L_min`;
- closed-form approximation when exact materialization hits caps but support
  metadata is available;
- unresolved only when neither exact materialization nor support metadata is
  available.
