# WAM-C Accumulated Runtime Cost Investigation

Date: 2026-05-28
Branch: `investigate/wam-c-accumulated-runtime-cost`

## Question

The accumulated WAM-C effective-distance targets were correct but much slower
than optimized Prolog at `10x`. The investigation goal was to identify whether
the dominant cost came from WAM dispatch, fact loading, repeated setup,
generated query shape, native-kernel integration, or fact lookup.

## Baseline

Command:

```sh
python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales 10x --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels,c-wam-accumulated-lmdb,c-wam-accumulated-no-kernels-lmdb --repetitions 1 --baseline-target prolog-accumulated --keep-temp --run-timeout-seconds 180
```

Result before the fix:

| Target | Median |
|---|---:|
| `prolog-accumulated` | 0.205s |
| `c-wam-accumulated` | 8.801s |
| `c-wam-accumulated-lmdb` | 8.255s |
| `c-wam-accumulated-no-kernels` | 8.849s |
| `c-wam-accumulated-no-kernels-lmdb` | 9.137s |

All C outputs matched Prolog. The native-kernel target was only 1.006x faster
than no-kernels for TSV facts, so foreign dispatch was not the primary issue.

## Finding

The generated accumulated runner repeatedly evaluates `category_ancestor/4`
for article-category/root pairs. Both paths used by that workload had the same
shape:

- the native WAM-C `category_ancestor` kernel scanned every registered
  `category_parent` edge for each DFS step;
- the no-kernel reference DFS scanned every loaded fact-source edge for each
  DFS step.

At `10x`, the fixture has only hundreds of article-category entries but several
thousand parent edges. The cost therefore came from repeated full edge-list
scans inside recursive ancestor traversal, not from fact loading or kernel
dispatch.

## Fix

The C runtime now builds a lazy sorted child index for category edges:

- `WamState` keeps `category_edges_by_child` for native category traversal.
- `WamFactSource` keeps `edges_by_child` for generated no-kernel reference
  traversal.
- Both indexes are invalidated when edges are appended and rebuilt lazily on the
  first lookup.
- `category_ancestor` and the generated reference DFS use child ranges instead
  of scanning the full edge list.

## Result

Command:

```sh
python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales 10x --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels,c-wam-accumulated-lmdb,c-wam-accumulated-no-kernels-lmdb --repetitions 3 --baseline-target prolog-accumulated --run-timeout-seconds 180
```

Result after the fix:

| Target | Median | Speedup vs Prolog |
|---|---:|---:|
| `prolog-accumulated` | 0.204s | 1.00x |
| `c-wam-accumulated` | 0.066s | 3.08x |
| `c-wam-accumulated-lmdb` | 0.069s | 2.95x |
| `c-wam-accumulated-no-kernels` | 0.068s | 3.00x |
| `c-wam-accumulated-no-kernels-lmdb` | 0.067s | 3.04x |

All outputs still matched. The remaining kernel/no-kernel delta is small, which
is expected because both now use the same indexed ancestor traversal shape.

## Next Work

The immediate accumulated-runtime blocker is resolved for `10x`. Sensible next
branches are now feature-parity work, such as adding floating-point WAM values
for weighted/A* native-kernel results, or a larger-scale artifact-layout sweep
if the benchmark focus returns to Wikipedia-sized data.
