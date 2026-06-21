# Distribution Cache SimpleWiki Depth Grid Report

Date: 2026-06-10

Branch: `codex/simplewiki-distribution-depth-grid-report`

## Scope

This report runs the parent-only distribution-cache benchmark against the
existing local numeric SimpleWiki Articles artifact:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv
```

The artifact is numeric-id based. Its `root_categories.tsv` sidecar identifies
root `2`, so both sampler and benchmark runs use `--root 2`.

Generated TSVs, target lists, JSONL, and markdown summaries were written under
Scratch and are not committed:

```text
/mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid
```

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

## Sampling

Depth 2 command:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file /home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv --root 2 --max-depth 2 --output /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth2.tsv --targets-output /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth2_targets.txt
```

Depth 2 result:

```text
selected_nodes=14644
sampled_edges=14851
```

Depth 3 command:

```text
python3 scripts/sample_distribution_cache_subtree.py --edge-file /home/s243a/Projects/UnifyWeaver/data/benchmark/simplewiki_articles/category_parent.tsv --root 2 --max-depth 3 --output /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth3.tsv --targets-output /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth3_targets.txt
```

Depth 3 result:

```text
selected_nodes=14680
sampled_edges=14887
```

Depth 3 adds only 36 nodes and 36 edges over depth 2 in this artifact.

## Depth 2 Full-Target Grid

Command:

```text
python3 scripts/distribution_cache_benchmark.py --edge-file /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth2.tsv --graph-name simplewiki_articles_root2_depth2_full --root 2 --targets-file /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth2_targets.txt --precompute-depths 0,1,2,3 --budgets 2,4,6 --output-dir /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid --fail-on-error
```

| D_pre | B_search | cache_nodes | cache_bytes | speedup | exact_failures |
|------:|---------:|------------:|------------:|--------:|---------------:|
| 0 | 2 | 1 | 566 | 0.868 | 0 |
| 0 | 4 | 1 | 566 | 0.856 | 0 |
| 0 | 6 | 1 | 566 | 0.880 | 0 |
| 1 | 2 | 13721 | 5289038 | 2.632 | 0 |
| 1 | 4 | 13721 | 5289038 | 2.626 | 0 |
| 1 | 6 | 13721 | 5289038 | 2.549 | 0 |
| 2 | 2 | 14644 | 5605061 | 3.017 | 0 |
| 2 | 4 | 14644 | 5605061 | 2.860 | 0 |
| 2 | 6 | 14644 | 5605061 | 2.828 | 0 |
| 3 | 2 | 14644 | 5605061 | 2.802 | 0 |
| 3 | 4 | 14644 | 5605061 | 2.932 | 0 |
| 3 | 6 | 14644 | 5605061 | 2.909 | 0 |

## Depth 3 Full-Target Grid

Command:

```text
python3 scripts/distribution_cache_benchmark.py --edge-file /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth3.tsv --graph-name simplewiki_articles_root2_depth3_full --root 2 --targets-file /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid/simplewiki_articles_root2_depth3_targets.txt --precompute-depths 0,1,2,3 --budgets 2,4,6 --output-dir /mnt/c/Users/johnc/Scratch/distribution-cache-depth-grid --fail-on-error
```

| D_pre | B_search | cache_nodes | cache_bytes | speedup | exact_failures |
|------:|---------:|------------:|------------:|--------:|---------------:|
| 0 | 2 | 1 | 566 | 0.864 | 0 |
| 0 | 4 | 1 | 566 | 0.871 | 0 |
| 0 | 6 | 1 | 566 | 0.872 | 0 |
| 1 | 2 | 13721 | 5289038 | 2.549 | 0 |
| 1 | 4 | 13721 | 5289038 | 2.552 | 0 |
| 1 | 6 | 13721 | 5289038 | 2.609 | 0 |
| 2 | 2 | 14644 | 5605061 | 2.923 | 0 |
| 2 | 4 | 14644 | 5605061 | 2.798 | 0 |
| 2 | 6 | 14644 | 5605061 | 2.757 | 0 |
| 3 | 2 | 14680 | 5617390 | 2.756 | 0 |
| 3 | 4 | 14680 | 5617390 | 2.927 | 0 |
| 3 | 6 | 14680 | 5617390 | 2.937 | 0 |

## Interpretation

All exact-cache cells matched full exact search. The real artifact therefore
continues to validate the cached suffix semantics.

`D_pre=0` is slower than full exact search because the cache only contains the
root and still pays lookup/truncation overhead. It should remain a baseline,
not a recommended policy.

`D_pre=1` gives the main speedup jump, reaching roughly `2.55x` to `2.63x`
depending on depth and budget. `D_pre=2` adds a smaller improvement, mostly
landing around `2.76x` to `3.02x`. `D_pre=3` is flat on this artifact because
the sampled depth-3 graph adds very little structure beyond depth 2.

For the current SimpleWiki Articles numeric artifact, `D_pre=2` is the best
default candidate for exact parent-only cache precomputation. `D_pre=1` is the
lower-memory fallback, and `D_pre=3` does not yet justify itself on this sample.
