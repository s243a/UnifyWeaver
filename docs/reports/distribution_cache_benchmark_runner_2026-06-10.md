# Distribution Cache Benchmark Runner Smoke Report

Date: 2026-06-10

Branch: `codex/distribution-cache-benchmark-runner`

## Scope

This smoke report covers the first executable slice of
`docs/design/DISTRIBUTION_CACHE_BENCHMARK_PLAN.md`: tiny parent-only DAG
fixtures, exact full search, exact cached histogram cutoffs, and benchmark
summary output.

The run does not cover SimpleWiki, enwiki, child-direction paths, admission
pressure, or fitted distribution tails.

## Unit Tests

Command:

```text
python3 -m unittest tests/test_distribution_cache_parity.py tests/test_distribution_cache_benchmark.py
```

Result:

```text
.......
----------------------------------------------------------------------
Ran 7 tests in 0.012s

OK
```

## CLI Smoke Run

Command:

```text
python3 scripts/distribution_cache_benchmark.py --fixtures diamond --precompute-depths 0,1 --budgets 2 --output-dir /mnt/c/Users/johnc/Scratch/distribution-cache-benchmark --fail-on-error
```

Summary:

| fixture | D_pre | B_search | cache_nodes | cache_bytes | full_ms | cached_ms | speedup | hit_rate | exact_failures |
|---------|-------|----------|-------------|-------------|---------|-----------|---------|----------|----------------|
| diamond | 0 | 2 | 1 | 566 | 0.023077 | 0.021932 | 1.052 | 1.000 | 0 |
| diamond | 1 | 2 | 3 | 1242 | 0.019335 | 0.009459 | 2.044 | 1.000 | 0 |

The CLI emitted JSONL and markdown artifacts under:

```text
/mnt/c/Users/johnc/Scratch/distribution-cache-benchmark
```

## Result

Cached histogram mode matched full exact search for the tested fixture grid.
The smoke run reported zero exactness failures.
