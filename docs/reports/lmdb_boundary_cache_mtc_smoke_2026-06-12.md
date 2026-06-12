# Enwiki MTC Boundary Cache Smoke Benchmark

Date: 2026-06-12

Branch: codex/mtc-boundary-cache-benchmark

## Scope

This smoke benchmark compares full bounded simple-path parent search against a boundary-cache variant on the title-resolved enwiki Main_topic_classifications artifact:

    artifact=/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident
    root_title=Main_topic_classifications
    root_id=7345184

The cache variant precomputes parent-path histograms for selected boundary nodes, then stops later target searches at those nodes and splices the cached histogram into the current path length.

This is exact on acyclic cones. On cyclic cones with simple-path semantics it is an approximation, because a node-only cached histogram does not know which nodes are already present on the current path. The benchmark therefore compares each cached result against the full bounded simple-path search and reports histogram error.

## Command

    python3 scripts/lmdb_parent_boundary_cache_benchmark.py --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident --root 7345184 --graph-name enwiki_mtc_boundary_cache_smoke --boundary-depths 2 --target-depths 3 --children-per-node 128 --frontier-limit 2000 --boundaries-per-depth 120 --targets-per-depth 30 --boundary-budget 6 --budgets 6,8 --path-cap 50000 --expansion-cap 100000 --seed enwiki-mtc-boundary-cache-v1 --output-dir /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic

Generated outputs:

    /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_smoke_20260612T012623Z.jsonl
    /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic/lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_cache_smoke_20260612T012623Z.md

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 35 |
| boundary | 2 | 1023 |
| target | 0 | 1 |
| target | 1 | 35 |
| target | 2 | 1023 |
| target | 3 | 2000 |

| boundary_nodes | cached_boundary_nodes | targets | boundary_budget |
|----------------|-----------------------|---------|-----------------|
| 120 | 120 | 30 | 6 |

## Boundary Cache Build

| entries | cached | mean_paths | mean_bins | mean_nodes_expanded | capped_entries |
|---------|--------|------------|-----------|---------------------|----------------|
| 120 | 120 | 20.567 | 4.192 | 16458.0 | 0 |

The cache is not free. Even with boundary budget 6, building 120 boundary histograms required a mean of 16,458 expanded nodes per boundary entry. This cost has to be amortized across repeated downstream searches before the refinement is worthwhile.

## Full Search Versus Boundary Cache

| budget | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf | mean_node_ratio | mean_cache_hits | full_capped | cached_capped |
|--------|------|---------|--------|--------|----------|-----------------|-----------------|-------------|---------------|
| 6 | 30 | 0.001333 | 0.000000 | 0.040000 | 0.000667 | 0.926 | 17.467 | 0 | 0 |
| 8 | 30 | 0.084876 | 0.639456 | 1.231884 | 0.040515 | 0.982 | 20.967 | 30 | 29 |

At budget 6, boundary splicing is nearly exact on this sample and reduces node expansions by about 7.4 percent. The small nonzero error is consistent with the simple-path caveat: a node-only boundary histogram may contain suffix paths that revisit nodes already seen in the prefix path.

At budget 8, both full and cached searches are mostly capped, so the comparison is not a clean accuracy estimate. Cache hits increase, but node expansion barely improves and distribution error is larger. This suggests the current boundary choice is useful only in the lower-budget regime unless boundary selection is made more workload-aware.

## Interpretation

Boundary caches are a refinement, not a default policy. They are most defensible when all of these are true:

    1. boundary histograms finish uncapped;
    2. downstream queries hit the same boundaries often enough to amortize cache-build cost;
    3. the active cone is acyclic enough, or the application tolerates measured approximation error;
    4. the budget is low enough that cached search avoids meaningful expansion work.

For exact simple-path semantics in cyclic enwiki cones, a node-only boundary histogram is not strictly exact. A stricter cache would need additional visited/boundary state, or it should be restricted to acyclic cones identified by a separate check. The current benchmark is valuable because it quantifies the error and cost reduction rather than assuming the cache is exact.

## Validation

- python3 -m unittest tests.test_lmdb_parent_boundary_cache_benchmark tests.test_lmdb_parent_histogram_benchmark tests.test_lmdb_parent_branching_diagnostic
- python3 scripts/lmdb_parent_boundary_cache_benchmark.py ... --graph-name enwiki_mtc_boundary_cache_smoke
- git diff --check
