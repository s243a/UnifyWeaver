# Reverse CSR Scale Sweep

**Snapshot date**: 2026-05-24.

This report checks whether reverse CSR memory remains small as the
synthetic graph grows. The goal is to separate the current Wikipedia
category/link case from denser datasets where memory can still become
the limiting factor.

## Command

```sh
python3 examples/benchmark/benchmark_reverse_csr_scale_sweep.py \
  --scale 1000x8 \
  --scale 10000x8 \
  --scale 50000x8 \
  --sample-parents 1000 \
  --iterations 5 \
  --seed 7
```

Each scale uses `parents x children_per_parent`, so `50000x8` has
400000 immediate `category_child(parent, child)` edges.

## Result

| parents | children/parent | edges | backend | median_ms | csr_bytes | csr_bytes/edge | csr_bytes/parent | build_s | offset_bytes | parent_lmdb_bytes | parent_lmdb_bytes/edge | phase1_lmdb_bytes |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 8 | 8000 | csr_sorted_array | 4.012461 | 49016 | 6.127000 | 49.016000 | 0.005984 | 0 | 188416 | 23.552000 | 385024 |
| 1000 | 8 | 8000 | csr_lmdb_offset | 3.815845 | 118685 | 14.835625 | 118.685000 | 0.014155 | 61440 | 188416 | 23.552000 | 385024 |
| 1000 | 8 | 8000 | lmdb | 3.470263 | 49016 | 6.127000 | 49.016000 | 0.005984 | 0 | 188416 | 23.552000 | 385024 |
| 10000 | 8 | 80000 | csr_sorted_array | 3.419292 | 481020 | 6.012750 | 48.102000 | 0.059963 | 0 | 1507328 | 18.841600 | 4579328 |
| 10000 | 8 | 80000 | csr_lmdb_offset | 4.734667 | 943906 | 11.798825 | 94.390600 | 0.076226 | 454656 | 1507328 | 18.841600 | 4579328 |
| 10000 | 8 | 80000 | lmdb | 3.708610 | 481020 | 6.012750 | 48.102000 | 0.059963 | 0 | 1507328 | 18.841600 | 4579328 |
| 50000 | 8 | 400000 | csr_sorted_array | 4.418097 | 2401022 | 6.002555 | 48.020440 | 0.331824 | 0 | 7352320 | 18.380800 | 22859776 |
| 50000 | 8 | 400000 | csr_lmdb_offset | 2.961144 | 4522789 | 11.306973 | 90.455780 | 0.381317 | 2113536 | 7352320 | 18.380800 | 22859776 |
| 50000 | 8 | 400000 | lmdb | 3.812495 | 2401022 | 6.002555 | 48.020440 | 0.331824 | 0 | 7352320 | 18.380800 | 22859776 |

## Interpretation

For this synthetic immediate-edge shape, sorted-array CSR converges near
6 bytes per edge and about 48 bytes per parent. The LMDB-offset index is
larger, converging near 11-12 bytes per edge and about 90 bytes per
parent. Parent-only LMDB is still larger in this sweep, converging near
18-19 bytes per edge.

The important cost is therefore not raw memory for current-scale
Wikipedia category/link artifacts. Build time and reuse count are more
important: at 400000 edges the sorted CSR build is 0.331824s and the
LMDB-offset build is 0.381317s. Both are fast enough that repeated
effective-distance variants with 100-500 child lookups per query can
plausibly amortize the build cost.

Local benchmark metadata includes an English Wikipedia-derived
`1m_cats` fixture with 1000000 hierarchy edges and 46846078 scanned
categorylinks rows. Using the 400000-edge synthetic ratios as a rough
projection:

| Scenario | Edge count | sorted CSR estimate | LMDB-offset CSR estimate |
| --- | ---: | ---: | ---: |
| category hierarchy only | 1000000 | ~6 MB | ~11 MB |
| all scanned categorylinks rows | 46846078 | ~281 MB | ~529 MB |

These are order-of-magnitude estimates from synthetic packed-int
artifacts, not a substitute for a real enwiki build. They do support the
working assumption that immediate-edge CSR memory is modest for the
Wikipedia-scale cases we are currently targeting.

That assumption is not universal. CSR stores only nonzero edges, but it
does not make dense data small. Memory can become significant for:

- transitive descendant or all-pairs reachability artifacts;
- dense or near-complete graph datasets;
- non-Wikipedia datasets with much larger edge counts;
- in-memory representations that expand packed arrays into object
  lists, maps, or per-edge heap allocations.

For those cases, `reverse_index(csr(...))` still needs cost-analyzer
guardrails. The current evidence only says that immediate reverse edges
for likely Wikipedia category/link workloads are small enough that build
time, query reuse, and page-cache interaction should dominate the
decision.
