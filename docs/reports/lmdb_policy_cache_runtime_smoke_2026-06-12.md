# LMDB Policy Cache Runtime Smoke - 2026-06-12

This smoke connects the ancestor-cache residency policy to actual bounded parent
histogram runtime.  Policy-selected resident nodes are materialized as boundary
histograms, then full bounded search is compared with cached search on the same
targets.

## Command

```bash
python3 scripts/lmdb_policy_cache_runtime_benchmark.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_policy_cache_runtime_budget3_smoke \
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
  --boundary-budget 3 \
  --budgets 4,6 \
  --path-cap 10000 \
  --expansion-cap 50000 \
  --seed enwiki-mtc-policy-cache-runtime-budget3 \
  --output-dir /mnt/c/Users/johnc/Scratch/policy-cache-runtime
```

## Results

| slots | L_min | L_max | boundary_budget | targets | capped_cones | resident | materialized | mean_nodes | budget | mean_l1 | mean_node_ratio | mean_time_ratio | mean_cache_hits |
|------:|------:|------:|----------------:|--------:|-------------:|---------:|-------------:|-----------:|-------:|--------:|----------------:|----------------:|----------------:|
| 128 | 6 | 8 | 3 | 2 | 1 | 52 | 37 | 1077.5 | 4 | 0.000000 | 0.971 | 1.050 | 1.500 |
| 128 | 6 | 8 | 3 | 2 | 1 | 52 | 37 | 1077.5 | 6 | 0.115385 | 0.966 | 0.982 | 10.500 |

## Interpretation

The policy-selected cache can produce real cache hits once boundary histogram
materialization is capped to a short suffix budget.  With `boundary_budget=3`,
37 of 52 policy-resident entries materialized and cached search hit those
entries during target searches.

The result also shows the next constraint clearly: cache residency is not enough.
A prior diagnostic with `boundary_budget=8` materialized only the root entry
because most boundary histograms hit the expansion cap.  That means the policy
needs a materialization-cost gate, or separate exact/partial-cache modes.

The budget-4 row remained exact on this tiny sample but did not improve wall time.
The budget-6 row had more cache hits and a slight time improvement, but introduced
histogram error.  The next benchmark should sweep `boundary_budget` and admission
thresholds together, then reject settings with too much error or capped boundary
materialization.
