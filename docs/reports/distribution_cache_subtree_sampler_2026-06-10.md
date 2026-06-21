# Distribution Cache Subtree Sampler Smoke Report

Date: 2026-06-10

Branch: `codex/simplewiki-distribution-subtree-sampler`

## Scope

This report covers the offline subtree sampler for distribution-cache
benchmarks. The sampler consumes a resolved category edge TSV with
`child<TAB>parent` rows, selects a bounded child-distance subtree from a
benchmark root, applies admin/container filters, and emits a sampled TSV plus
optional target list for `scripts/distribution_cache_benchmark.py`.

The smoke run uses the checked-in SimpleWiki-shaped sample rooted at
`Category:Articles`. It does not require the full SimpleWiki dump.

## Unit Tests

Command:

```text
python3 -m unittest tests/test_distribution_cache_parity.py tests/test_distribution_cache_benchmark.py tests/test_distribution_cache_subtree_sampler.py
```

Result:

```text
..........
----------------------------------------------------------------------
Ran 10 tests in 0.013s

OK
```

## Sampler Smoke Run

Command:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file tests/fixtures/simplewiki_articles_parent_sample.tsv --root Category:Articles --max-depth 2 --output /mnt/c/Users/johnc/Scratch/distribution-cache-subtree-sampler/simplewiki_articles_depth2.tsv --targets-output /mnt/c/Users/johnc/Scratch/distribution-cache-subtree-sampler/simplewiki_articles_depth2_targets.txt
```

Result:

```text
root=Category:Articles
selected_nodes=6
sampled_edges=5
max_depth=2
wrote_edges=/mnt/c/Users/johnc/Scratch/distribution-cache-subtree-sampler/simplewiki_articles_depth2.tsv
wrote_targets=/mnt/c/Users/johnc/Scratch/distribution-cache-subtree-sampler/simplewiki_articles_depth2_targets.txt
```

## Downstream Benchmark Smoke Run

Command:

```text
python3 scripts/distribution_cache_benchmark.py --edge-file /mnt/c/Users/johnc/Scratch/distribution-cache-subtree-sampler/simplewiki_articles_depth2.tsv --graph-name simplewiki_articles_depth2_sampled --targets-file /mnt/c/Users/johnc/Scratch/distribution-cache-subtree-sampler/simplewiki_articles_depth2_targets.txt --precompute-depths 0,1,2 --budgets 2,4 --output-dir /mnt/c/Users/johnc/Scratch/distribution-cache-subtree-sampler --fail-on-error
```

Summary:

| fixture | D_pre | B_search | cache_nodes | cache_bytes | full_ms | cached_ms | speedup | hit_rate | exact_failures |
|---------|-------|----------|-------------|-------------|---------|-----------|---------|----------|----------------|
| simplewiki_articles_depth2_sampled | 0 | 2 | 1 | 582 | 0.043441 | 0.040083 | 1.084 | 1.000 | 0 |
| simplewiki_articles_depth2_sampled | 0 | 4 | 1 | 582 | 0.024808 | 0.028276 | 0.877 | 1.000 | 0 |
| simplewiki_articles_depth2_sampled | 1 | 2 | 3 | 1308 | 0.022641 | 0.014949 | 1.515 | 1.000 | 0 |
| simplewiki_articles_depth2_sampled | 1 | 4 | 3 | 1308 | 0.029033 | 0.017334 | 1.675 | 1.000 | 0 |
| simplewiki_articles_depth2_sampled | 2 | 2 | 6 | 2493 | 0.023183 | 0.008233 | 2.816 | 1.000 | 0 |
| simplewiki_articles_depth2_sampled | 2 | 4 | 6 | 2493 | 0.020583 | 0.008666 | 2.375 | 1.000 | 0 |

Result: sampled file-backed benchmark completed with zero exactness failures.
