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
  --csr-index-backends sorted_array,lmdb_offset \
  --sample-parents 1000 \
  --iterations 5 \
  --seed 7
```

## Result

| backend | index_backend | sample_parents | iterations | total_children | median_ms | min_ms | max_ms | csr_artifact_bytes | csr_build_seconds | offset_index_bytes | parent_lmdb_env_bytes | phase1_lmdb_env_bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| csr_sorted_array | sorted_array | 1000 | 5 | 8000 | 3.413093 | 3.299319 | 3.427612 | 481020 | 0.060911 | 0 | 1507328 | 4579328 |
| csr_lmdb_offset | lmdb_offset | 1000 | 5 | 8000 | 2.649246 | 2.543666 | 2.820476 | 943906 | 0.075248 | 454656 | 1507328 | 4579328 |
| lmdb | n/a | 1000 | 5 | 8000 | 3.514109 | 3.445555 | 3.775466 | 481020 | 0.060911 | 0 | 1507328 | 4579328 |

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

The `lmdb_offset` CSR index backend is slightly faster than binary
search over the sorted `.idx` file in this run, but it pays for that with
extra preprocessing and bytes: build time rises from 0.060911s to
0.075248s, and artifact size rises from 481020 bytes to 943906 bytes.
That is the cost-analyzer tradeoff. At Wikipedia scale, the sorted CSR
index is expected to fit in memory, so lookup savings must be large
enough to justify the additional LMDB offset artifact and build work.

The original Python CSR reader was slower than LMDB lookup because every
lookup opened, sought, and read from the values file. The reader now
keeps the values file open across lookup batches, which makes this
synthetic fixture measure positional lookup cost more directly.

The next optimization target is therefore no longer file-handle churn:

- optionally cache recently used child slices or blocks;
- sweep CSR index backends across larger scales and fanout shapes;
- then consider WAM runtime integration for workloads that need deferred
  child expansion.
