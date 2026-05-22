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

| backend | sample_parents | iterations | total_children | median_ms | min_ms | max_ms | artifact_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| csr | 1000 | 5 | 8000 | 12.104892 | 11.941214 | 12.224214 | 480780 |
| lmdb | 1000 | 5 | 8000 | 3.613239 | 3.578186 | 3.852191 | 4579328 |

## Interpretation

The prototype CSR artifact is much smaller than the Phase 1 LMDB file
for this synthetic graph, but the current Python CSR reader is slower
than LMDB lookup. That is expected for the first reader: every lookup
opens, seeks, and reads from the values file.

The next optimization target is therefore the reader path, not the CSR
layout:

- keep the values file open across lookup batches;
- optionally cache recently used child slices or blocks;
- then rerun the same benchmark before considering WAM runtime
  integration.
