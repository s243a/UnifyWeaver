# Enwiki MTC Boundary Cache Hit Overhead

This run follows the runtime attribution work by applying three low-risk cached-search overhead reductions:

- build a merged boundary lookup once per benchmark run so each visited node does one boundary-cache probe instead of probing histogram and parametric caches separately;
- share a decoded cache-entry memo across benchmark rows, so repeated boundary hits reuse decoded histograms;
- track cached path mass incrementally instead of repeatedly scanning `hist.values()` for path-cap checks.

## Smoke Runs

Both runs used enwiki MTC root `7345184`, boundary depths `2,3`, target depth `4`, 8 targets, recurrence boundary construction, budget `8`, path cap `50000`, expansion cap `100000`, and seed `enwiki-mtc-runtime-attribution-v1`.

Artifacts:

- Attribution run: `lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_hit_overhead_smoke_20260614T013223Z.jsonl`
- Attribution summary: `lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_cache_hit_overhead_smoke_20260614T013223Z.md`
- Non-attributed runtime run: `lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_hit_overhead_runtime_smoke_20260614T013243Z.jsonl`
- Non-attributed runtime summary: `lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_cache_hit_overhead_runtime_smoke_20260614T013243Z.md`

## Result

| run | rows | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | mean_payload_bytes_read | mean_decode_ns | mean_decode_memo_hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prior attribution smoke | 8 | 1.664 | 58,170,564.5 | 96,875,805.8 | 67.250 | 6,491.000 | 345,789.6 | n/a |
| optimized attribution smoke | 8 | 1.723 | 54,650,839.1 | 94,337,192.2 | 67.250 | 806.000 | 77,292.8 | 58.875 |
| optimized non-attributed smoke | 8 | 1.116 | 53,727,860.1 | 60,036,218.4 | 67.250 | 806.000 | 68,960.9 | 58.875 |

The shared decode memo did what it was supposed to do: per-row decoded payload bytes dropped from `6491` to `806`, and decode time dropped from about `345.8us` to `77.3us` in the attributed run. The non-attributed runtime smoke is now much closer to break-even, with cached search about `11.6%` slower than full search for this small sample.

## Interpretation

The remaining overhead is not payload decode. The cache hit path still has enough traversal, Python recursion, cache-probe, and bookkeeping cost that it does not quite beat full search on this smoke. The next optimization should be structural rather than serialization-focused: a tighter cached traversal loop, fewer per-node bookkeeping operations, or a benchmark mode that starts from already-selected ancestor-boundary targets so cache hits replace larger suffixes more consistently.
