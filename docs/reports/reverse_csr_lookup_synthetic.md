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
| csr | 1000 | 5 | 8000 | 12.104892 | 11.941214 | 12.224214 | 480780 | 4579328 |
| lmdb | 1000 | 5 | 8000 | 3.613239 | 3.578186 | 3.852191 | 480780 | 4579328 |

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

The current Python CSR reader is slower than LMDB lookup. That is
expected for the first reader: every lookup opens, seeks, and reads from
the values file.

The next optimization target is therefore the reader path, not the CSR
layout:

- keep the values file open across lookup batches;
- optionally cache recently used child slices or blocks;
- then rerun the same benchmark before considering WAM runtime
  integration.
