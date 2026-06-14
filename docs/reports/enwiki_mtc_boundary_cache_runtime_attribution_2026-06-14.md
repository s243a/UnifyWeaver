# Enwiki MTC Boundary Cache Runtime Attribution

This smoke run adds timing attribution to the cached side of `scripts/lmdb_parent_boundary_cache_benchmark.py`. The goal was to explain why validation rows with positive cache hits can still run slower than full search. Attribution mode adds timer calls and materializes parent lists for timed parent lookups, so use it to compare timing buckets inside the cached path rather than as the canonical non-instrumented runtime.

## Run

```bash
python3 scripts/lmdb_parent_boundary_cache_benchmark.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_boundary_cache_runtime_attribution_smoke \
  --boundary-depths 2,3 \
  --target-depths 4 \
  --children-per-node 64 \
  --frontier-limit 1000 \
  --boundaries-per-depth 40 \
  --targets-per-depth 8 \
  --include-target-ancestor-boundaries \
  --target-ancestor-boundary-limit 80 \
  --boundary-budget 8 \
  --boundary-builder recurrence \
  --budgets 8 \
  --path-cap 50000 \
  --expansion-cap 100000 \
  --collect-attribution \
  --seed enwiki-mtc-runtime-attribution-v1 \
  --output-dir docs/reports
```

Artifacts:

- `lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_runtime_attribution_smoke_20260614T004619Z.jsonl`
- `lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_cache_runtime_attribution_smoke_20260614T004619Z.md`

## Result

| budget | rows | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_hist_bins_spliced | mean_payload_bytes_read | mean_decode_ns |
|--------|-----:|----------------:|------------------:|--------------------:|---------------:|-----------------------:|------------------------:|---------------:|
| 8 | 8 | 1.664 | 58,170,564.5 | 96,875,805.8 | 67.250 | 22.750 | 6,491.000 | 345,789.6 |

Cached search was slower in this smoke despite many histogram cache hits. Decode and splice are not the primary cost: decode was about `0.4%` of cached wall time, and splice was about `0.1%`. Parent lookup was visible at about `7.7%`, and cache probing was larger than expected, but the largest bucket was still the residual cached traversal/interpreter overhead.

| budget | mean_cached_time_ns | mean_decode_ns | mean_splice_ns | mean_parent_lookup_ns | mean_probe_ns | mean_path_cap_check_ns | mean_attributed_ns | mean_unattributed_ns |
|--------|--------------------:|---------------:|---------------:|----------------------:|--------------:|-----------------------:|-------------------:|---------------------:|
| 8 | 96,875,805.8 | 345,789.6 | 56,676.6 | 7,469,802.5 | 15,685,032.9 | 45,101.1 | 23,602,402.8 | 73,273,403.0 |

## Interpretation

The result suggests the current cached Python DFS is not just paying decode/splice cost. It is still doing enough recursive traversal, dictionary probing, per-node bookkeeping, and Python-level control flow that the cache hit does not guarantee a faster row. The next optimization target should therefore be the cached traversal loop: reduce repeated cache probes, avoid repeated `sum(hist.values())` checks where possible, memoize decoded payloads when a boundary is hit more than once in a row, and consider a tighter iterative cached search path for runtime benchmarks.

The benchmark now reports these cached-side attribution fields in each comparison row: `cache_probe_ns`, `cache_splice_ns`, `cache_path_cap_check_ns`, `cached_parent_lookup_ns`, `cached_attributed_ns`, and `collect_attribution`.
