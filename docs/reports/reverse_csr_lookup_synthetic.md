# Reverse CSR Lookup Synthetic Benchmark

**Snapshot date**: 2026-05-22.

This report records the first repeatable CSR-vs-LMDB measurement using a
generated Phase 1 LMDB fixture. The goal is not to claim final
performance; it is to make the measurement path reproducible without
depending on local simplewiki/enwiki artifacts.

## Command

```sh
tmpdir=$(mktemp -d /tmp/uw-csr-bench-XXXXXX)
python3 examples/benchmark/generate_synthetic_phase1_lmdb.py \
  "$tmpdir/phase1.lmdb" \
  --parents 10000 \
  --children-per-parent 8
python3 examples/benchmark/benchmark_reverse_csr_lookup.py \
  "$tmpdir/phase1.lmdb" \
  --csr-dir "$tmpdir/csr" \
  --sample-parents 1000 \
  --iterations 5 \
  --seed 7
```

## Result

| backend | sample_parents | iterations | total_children | median_ms | min_ms | max_ms | csr_artifact_bytes | phase1_lmdb_env_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| csr | 1000 | 5 | 8000 | 3.188616 | 3.180329 | 3.430460 | 480780 | 4579328 |
| lmdb | 1000 | 5 | 8000 | 3.690326 | 3.581263 | 3.842682 | 480780 | 4579328 |

`phase1_lmdb_env_bytes` is the size of the whole Phase 1 LMDB
environment file (`data.mdb`), not the size of only the parent-edge or
child-edge relation. It includes `category_parent`, `category_child`,
`meta`, stub sub-dbs, LMDB B-tree pages, and LMDB allocation overhead.
LMDB does not expose an exact per-subdb byte count through this
benchmark.

## Interpretation

The prototype CSR artifact is much smaller than the full Phase 1 LMDB
environment for this synthetic graph, but this is not a parent-only
LMDB comparison. In a memory-budgeted runtime, the priority resident or
memory-mapped structure remains the hot parent-edge store. A reverse CSR
artifact may become memory-resident only when memory budget allows, and
that in-memory representation should preserve the compact typed-array /
CSR shape rather than expanding into Python-style object lists.

The original Python CSR reader was slower than LMDB lookup because every
lookup opened, sought, and read from the values file. The reader now
keeps the values file open across lookup batches, which makes this
synthetic fixture measure positional lookup cost more directly.

The next optimization target is therefore no longer file-handle churn:

- optionally cache recently used child slices or blocks;
- compare parent-edge-only LMDB memory pressure against CSR residency;
- then consider WAM runtime integration for workloads that need deferred
  child expansion.
