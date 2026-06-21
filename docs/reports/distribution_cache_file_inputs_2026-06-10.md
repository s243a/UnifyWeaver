# Distribution Cache File Input Smoke Report

Date: 2026-06-10

Branch: `codex/distribution-cache-file-inputs`

## Scope

This report covers the first file-backed benchmark input path for
`scripts/distribution_cache_benchmark.py`.

The benchmark now accepts a TSV edge list with `child<TAB>parent` rows and can
select shallow reachable targets from a configurable benchmark root. The
checked-in sample uses `Category:Articles` as a SimpleWiki content-subtree root.

`Category:Articles` is not the universal root of all SimpleWiki categories. It
is a useful content subtree for early statistics. Broader roots such as
`Category:Categories` and `Category:Container_categories` require filtering:
admin/template/container regions create an inhomogeneous benchmark population,
and `Category:Container_categories` itself is a high-branching registry outlier
even though individual container categories may be valid subtree roots.

## Unit Tests

Command:

```text
python3 -m unittest tests/test_distribution_cache_parity.py tests/test_distribution_cache_benchmark.py
```

Result:

```text
........
----------------------------------------------------------------------
Ran 8 tests in 0.011s

OK
```

## File-Backed CLI Smoke Run

Command:

```text
python3 scripts/distribution_cache_benchmark.py --edge-file tests/fixtures/simplewiki_articles_parent_sample.tsv --graph-name simplewiki_articles_parent_sample --max-target-depth 2 --precompute-depths 0,1,2 --budgets 2,4 --output-dir /mnt/c/Users/johnc/Scratch/distribution-cache-file-inputs --fail-on-error
```

Summary:

| fixture | D_pre | B_search | cache_nodes | cache_bytes | full_ms | cached_ms | speedup | hit_rate | exact_failures |
|---------|-------|----------|-------------|-------------|---------|-----------|---------|----------|----------------|
| simplewiki_articles_parent_sample | 0 | 2 | 1 | 582 | 0.031429 | 0.041384 | 0.759 | 1.000 | 0 |
| simplewiki_articles_parent_sample | 0 | 4 | 1 | 582 | 0.022509 | 0.024272 | 0.927 | 1.000 | 0 |
| simplewiki_articles_parent_sample | 1 | 2 | 3 | 1308 | 0.023338 | 0.017010 | 1.372 | 1.000 | 0 |
| simplewiki_articles_parent_sample | 1 | 4 | 3 | 1308 | 0.026759 | 0.014937 | 1.791 | 1.000 | 0 |
| simplewiki_articles_parent_sample | 2 | 2 | 6 | 2493 | 0.023025 | 0.006952 | 3.312 | 1.000 | 0 |
| simplewiki_articles_parent_sample | 2 | 4 | 6 | 2493 | 0.022094 | 0.006121 | 3.610 | 1.000 | 0 |

The CLI emitted JSONL and markdown artifacts under:

```text
/mnt/c/Users/johnc/Scratch/distribution-cache-file-inputs
```

## Fixture-Mode CLI Smoke Run

Command:

```text
python3 scripts/distribution_cache_benchmark.py --fixtures diamond --precompute-depths 0,1 --budgets 2 --fail-on-error
```

Result: fixture mode completed with zero exactness failures.
