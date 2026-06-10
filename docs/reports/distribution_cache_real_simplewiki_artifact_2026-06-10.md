# Distribution Cache Real SimpleWiki Artifact Smoke Report

Date: 2026-06-10

Branch: `codex/simplewiki-lmdb-distribution-export`

## Scope

This report covers the first distribution-cache benchmark run against an
existing local SimpleWiki benchmark artifact.

The local artifact discovered under `~/Projects/UnifyWeaver` was:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv
```

That TSV is numeric-id based rather than human-readable category-title based.
Its `root_categories.tsv` identifies root id `2`, so the sampler and benchmark
were run with `--root 2`.

For future human-readable runs, this branch also adds
`scripts/export_distribution_cache_edges.py`, which exports resolved category
edges from `data/simplewiki/simplewiki_categories.db` after the existing
three-dump SimpleWiki parser has joined `categorylinks`, `linktarget`, and
`page`.

## Unit Tests

Command:

```text
python3 -m unittest tests/test_distribution_cache_parity.py tests/test_distribution_cache_benchmark.py tests/test_distribution_cache_subtree_sampler.py tests/test_export_distribution_cache_edges.py
```

Result:

```text
............
----------------------------------------------------------------------
Ran 12 tests in 0.012s

OK
```

## Sampler Smoke Run

Command:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file /home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv --root 2 --max-depth 2 --output /mnt/c/Users/johnc/Scratch/distribution-cache-lmdb-export/simplewiki_articles_root2_depth2.tsv --targets-output /mnt/c/Users/johnc/Scratch/distribution-cache-lmdb-export/simplewiki_articles_root2_depth2_targets.txt
```

Result:

```text
root=2
selected_nodes=14644
sampled_edges=14851
max_depth=2
wrote_edges=/mnt/c/Users/johnc/Scratch/distribution-cache-lmdb-export/simplewiki_articles_root2_depth2.tsv
wrote_targets=/mnt/c/Users/johnc/Scratch/distribution-cache-lmdb-export/simplewiki_articles_root2_depth2_targets.txt
```

## Benchmark Smoke Run

Command:

```text
python3 scripts/distribution_cache_benchmark.py --edge-file /mnt/c/Users/johnc/Scratch/distribution-cache-lmdb-export/simplewiki_articles_root2_depth2.tsv --graph-name simplewiki_articles_root2_depth2_numeric --root 2 --targets-file /mnt/c/Users/johnc/Scratch/distribution-cache-lmdb-export/simplewiki_articles_root2_depth2_targets.txt --target-limit 500 --precompute-depths 0,1,2 --budgets 2,4 --output-dir /mnt/c/Users/johnc/Scratch/distribution-cache-lmdb-export --fail-on-error
```

Summary:

| fixture | D_pre | B_search | cache_nodes | cache_bytes | full_ms | cached_ms | speedup | hit_rate | exact_failures |
|---------|-------|----------|-------------|-------------|---------|-----------|---------|----------|----------------|
| simplewiki_articles_root2_depth2_numeric | 0 | 2 | 1 | 566 | 1.602792 | 1.957799 | 0.819 | 1.000 | 0 |
| simplewiki_articles_root2_depth2_numeric | 0 | 4 | 1 | 566 | 1.553598 | 1.859647 | 0.835 | 1.000 | 0 |
| simplewiki_articles_root2_depth2_numeric | 1 | 2 | 13721 | 5289038 | 2.197429 | 0.800229 | 2.746 | 1.000 | 0 |
| simplewiki_articles_root2_depth2_numeric | 1 | 4 | 13721 | 5289038 | 2.854687 | 1.019099 | 2.801 | 1.000 | 0 |
| simplewiki_articles_root2_depth2_numeric | 2 | 2 | 14644 | 5605061 | 1.669414 | 0.624972 | 2.671 | 1.000 | 0 |
| simplewiki_articles_root2_depth2_numeric | 2 | 4 | 14644 | 5605061 | 1.722279 | 0.679698 | 2.534 | 1.000 | 0 |

Result: sampled real-artifact benchmark completed with zero exactness failures.
