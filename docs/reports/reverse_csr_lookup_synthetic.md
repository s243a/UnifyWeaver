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
  --parent-lmdb-dir "$tmpdir/parent_only.lmdb" \
  --sample-parents 1000 \
  --iterations 5 \
  --seed 7
```

## Result

| backend | sample_parents | iterations | total_children | median_ms | min_ms | max_ms | csr_artifact_bytes | parent_lmdb_env_bytes | phase1_lmdb_env_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| csr | 1000 | 5 | 8000 | 3.136484 | 3.076236 | 3.269940 | 480780 | 1507328 | 4579328 |
| lmdb | 1000 | 5 | 8000 | 3.525135 | 3.490397 | 3.926229 | 480780 | 1507328 | 4579328 |

`phase1_lmdb_env_bytes` is the size of the whole Phase 1 LMDB
environment file (`data.mdb`), not the size of only the parent-edge or
child-edge relation. It includes `category_parent`, `category_child`,
`meta`, stub sub-dbs, LMDB B-tree pages, and LMDB allocation overhead.
LMDB does not expose an exact per-subdb byte count through this
benchmark.

`parent_lmdb_env_bytes` is a separate size probe built by copying only
`category_parent` plus minimal Phase 1-compatible metadata/stub sub-dbs
into its own LMDB environment. It is not a child-lookup backend in this
benchmark. It exists to approximate the hot parent-edge store that would
have memory priority in an ancestor-first runtime.

## Interpretation

The prototype CSR artifact is smaller than both the full Phase 1 LMDB
environment and the parent-only LMDB size probe for this synthetic
graph. The parent-only LMDB remains the right priority resident or
memory-mapped structure in a memory-budgeted runtime: it is the hot
ancestor-kernel store. A reverse CSR artifact may become memory-resident
only when memory budget allows, and that in-memory representation should
preserve the compact typed-array / CSR shape rather than expanding into
Python-style object lists.

The original Python CSR reader was slower than LMDB lookup because every
lookup opened, sought, and read from the values file. The reader now
keeps the values file open across lookup batches, which makes this
synthetic fixture measure positional lookup cost more directly.

The next optimization target is therefore no longer file-handle churn:

- optionally cache recently used child slices or blocks;
- compare parent-edge-only LMDB memory pressure against CSR residency;
- then consider WAM runtime integration for workloads that need deferred
  child expansion.
