# LMDB-Resident Interning: Philosophy

## Context

After the demand-filter and pre-filter parallelisation work landed (Phase L
appendix #4 in `WAM_PERF_OPTIMIZATION_LOG.md`), `query_ms` on the
`100k_cats` fixture collapsed to ~0 ms but `total_ms` plateaued at ~3.2 s.
That floor is not the WAM interpreter, not the FFI kernel, and not the
parallel section — it is the sequential **setup** phase: TSV parsing,
string atom interning, and `IntMap` construction over ~280k pairs.

The current LMDB-backed Main.hs compounds the cost: it parses the TSV
*and* checks whether `data.mdb` exists. If LMDB is present it skips only
the LMDB *write*, not the parse and intern-table rebuild. The Int IDs in
the LMDB file are deterministic only because the input order is
deterministic — the mapping itself is rebuilt every run.

This document explains why the answer is to make the LMDB **the** intern
table, not a second copy of data that lives next to it.

## Core principle: the database is the intern table

A persistent integer ID space is exactly the kind of thing a key-value
store is good at. LMDB already gives us:

- Memory-mapped reads (zero-copy lookups in the hot path).
- Atomic single-writer transactions (correctness during ingestion).
- Sub-databases (multiple namespaces in one file).
- `MDB_APPEND` / `MDB_APPENDDUP` flags (skip B-tree balance for sorted writes).

The previous design treated LMDB as a *cache* of edges that a separate
Python script populated, with the intern table reconstructed in-process
on every run. That was not wrong — it was a phase. The lesson is that
once the consumer trusts the IDs, the intern table belongs in the
database too.

After this change, the only string ↔ int translation that remains at
runtime is at the **boundary**: a CLI argument naming a root, and the
output formatting that turns result IDs back into category names. Both
are O(rows-of-output), not O(graph-size).

## Why streaming preprocessing

The Wikipedia categorylinks dump is sorted by `cl_from` (it is the
clustered primary index in MySQL). The existing
`src/unifyweaver/runtime/rust/mysql_stream/` crate already streams the
gzipped dump and yields parsed rows. We can intern strings on the fly
and write Int↔Int edges to LMDB *as they arrive*, with `MDB_APPENDDUP`
because the keys are already sorted.

That gives us:

- One read of the input dump.
- No intermediate TSV materialisation.
- O(unique-strings) memory for the intern map; no edge buffer.
- Sequential B-tree page writes for both edges and `i2s` (which is also
  monotonic by construction).

The approach scales from `100k_cats` (~5 MB intern map) to full enwiki
(~350 MB intern map, ~28 M edges) on a commodity dev box. The
constraint we are designing against is "preprocessing should not need
a different machine than the benchmark itself."

## Why Rust, not Haskell, for the ingester

The Haskell side already has a working LMDB binding (`lmdb >= 0.2.5`) and
could in principle do its own ingestion. We are not extending the
Haskell binary because:

1. The Rust crate **already exists** and parses the MySQL dump format
   correctly. Reusing it is one new sink trait and one new Cargo
   dependency.
2. Lazy I/O in Haskell makes streaming an attentive engineering exercise
   (`bytestring`, careful seq, watch for retainers). Rust's eager-by-default
   model reaches the same answer with less ceremony.
3. The same preprocessor will eventually feed the WAM-Rust and WAM-Elixir
   targets. Putting it in Rust gives us a single shared tool. Putting it
   in Haskell would either fork the logic or force the other targets to
   call into a Haskell binary.

Doing it in Rust is not a statement that Haskell is bad at this; it is
a statement that a one-time, single-process preprocessor sits naturally
in a single-purpose tool.

## Why we keep the TSV path

The string-keyed TSV path stays as a fallback for:

- Small fixtures where ingestion overhead is irrelevant
  (`benchmark_target_matrix` at scale-300, dev workflows).
- Tests that need to construct fixtures without LMDB tooling.
- Targets that have not adopted the LMDB layout yet.

The matrix bench will pick the LMDB-resident path automatically when
`fact_count` exceeds the existing `use_lmdb(auto)` threshold (50k).
Below that, the in-process intern-table build is faster than opening a
file and the difference doesn't matter anyway.

## Demand set: why this design doesn't address it

The demand set is computed at startup by a backward BFS from the root.
With LMDB-resident edges, that BFS becomes a sequence of LMDB cursor
walks instead of `IM.lookup` calls. On a memory-mapped LMDB the lookups
are roughly the same cost as in-memory `IntMap` lookups once the OS
page cache is warm — and for warm runs (the common case) it is warm by
construction.

So the demand set stays in memory as a small `IntSet` (typically <100k
Ints), built at startup by walking the LMDB. No precomputation of
demand sets per root, no caching strategy. If profiling later shows
demand-set construction is itself a bottleneck, the right answer is
either to walk in parallel or to memoise it next to the LMDB — but
neither is needed up front.

## What this rules out

- **A separate manifest file** living next to the LMDB. The intern map
  is in the database; there is one file.
- **An in-process atom intern table that grows at runtime.** The
  database is read-only after preprocessing. The runtime opens it and
  performs lookups; it does not mutate the ID space.
- **Implicit ID assignment by the runtime.** Two binaries reading the
  same LMDB see exactly the same IDs because the IDs are *in* the LMDB,
  not derived from input order at runtime.

## Refs

- `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md` — schema, CLI, runtime contract.
- `WAM_LMDB_RESIDENT_INTERNING_IMPLEMENTATION_PLAN.md` — phased steps.
- `WAM_PERF_OPTIMIZATION_LOG.md` — measurements that motivated this work,
  especially Phase L appendix #4.
- `src/unifyweaver/runtime/rust/mysql_stream/` — existing Rust parser.
- `templates/targets/haskell_wam/main.hs.mustache` — existing
  `int_atom_seeds(true)` mode that this work generalises.
