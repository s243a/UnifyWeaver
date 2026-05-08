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

## Two regimes: integer-keyed vs text-keyed

The MySQL dumps we consume have two distinct shapes, and they motivate
different handling:

- **Integer-keyed (enwiki categorylinks)**: the relevant columns are
  `cl_from` (the source page's MediaWiki `page_id`, an int32) and
  `cl_target_id` (the target category's `page_id`, also int32).
  *MediaWiki has already done the interning.* The pipeline writes
  Int → Int edges directly. No `s2i` / `i2s` intern table is needed at
  this layer; the upstream identifier *is* the canonical ID.
- **Text-keyed (SimpleWiki-derived `100k_cats` / `50k_cats`)**: the
  fixture build extracts category names as strings, with no
  pre-assigned integer IDs. The pipeline must intern as it ingests.
  The intern table belongs in the same LMDB file as the edges.

The current production path covers the integer-keyed regime correctly
(see "What already exists" below). The work this document motivates is
adding text-keyed support without disturbing the existing path.

## Why streaming preprocessing

The Wikipedia categorylinks dump is sorted by `cl_from` (it is the
clustered primary index in MySQL). The existing
`src/unifyweaver/runtime/rust/mysql_stream/` crate already streams the
gzipped dump and yields parsed rows. The existing Python consumer
already writes Int → Int edges with `MDB_DUPSORT`. The text-keyed
extension reuses the same streaming pattern: intern strings on the fly
and write Int↔Int edges *as they arrive*, with `MDB_APPENDDUP` because
the keys are already sorted.

That gives us:

- One read of the input dump.
- No intermediate TSV materialisation beyond the existing producer→consumer pipe.
- O(unique-strings) memory for the intern map; no edge buffer.
- Sequential B-tree page writes for both edges and `i2s` (which is also
  monotonic by construction).

The approach scales from `100k_cats` (~5 MB intern map) to full enwiki
(~350 MB intern map, ~28 M edges) on a commodity dev box. The
constraint we are designing against is "preprocessing should not need
a different machine than the benchmark itself."

## What already exists

This is **not** a green-field project. The streaming pipeline is in
production for the integer-keyed enwiki path:

- `src/unifyweaver/runtime/rust/mysql_stream/` — the Rust parser. ~555
  lines, validated against simplewiki (2.2 M rows, byte-exact match to
  the SQLite ground truth, ~28 MB/s gzipped on one core).
- `src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py` — the
  Python consumer. ~206 lines. Reads TSV from stdin, writes LMDB.
  Already handles filtering (`UW_FILTER_COL=4 UW_FILTER_VAL=subcat`),
  column projection (`UW_KEY_COL=0 UW_VAL_COL=6`), encoding
  (`int32_le`), and dupsort (`UW_LMDB_DUPSORT=1`).
- `examples/streaming/enwiki_category_ingest.pl` — the Prolog glue
  composing producer + consumer.
- AWK and C# consumer variants live alongside (`enwiki_category_ingest_awk.pl`,
  `enwiki_category_ingest_csharp.pl`), demonstrating the pluggable
  shape — see "Pluggable parsers" below.

The text-keyed extension is small relative to this surface: extend the
Python consumer to optionally write `s2i` / `i2s` / `meta` sub-dbs when
the producer is configured for text-keyed input, plus add `MDB_APPEND`
flags when the input is sorted. No new Rust crate, no new dependency
on `heed`; the Python `lmdb` package already supports both `append=True`
and named sub-dbs.

## Pluggable parsers — but Rust is canonical for benchmarks

The streaming-glue layer (`src/unifyweaver/glue/streaming_glue.pl`)
treats the parser as a declaratively-selected target. The same Prolog
pipeline runs unchanged whether the producer is the Rust crate or an
AWK script:

```prolog
% Rust:
:- declare_target(parse_mysql_rows/2, rust,
                  [leaf(true), native_crate(mysql_stream)]).

% AWK (one-line swap):
:- declare_target(parse_mysql_rows/2, awk,
                  [leaf(true),
                   script_path('src/.../parse_inserts.awk'),
                   input_filter(zcat),
                   awk_exec(gawk)]).
```

The AWK and C# variants already exist; a Haskell variant would slot
into the same pattern. **For performance benchmarking we deliberately
fix the parser to Rust** so the preprocessing cost is constant across
query targets. The query target is what we are measuring; the parser
is upstream of the measurement and any variation in its cost would
contaminate the comparison.

The pluggable-parser property is preserved in the design, but it is
not exercised in the perf path. Other deployments (a developer
sketching a new ingest with AWK; a customer who can't ship a Rust
binary) keep the option without paying for it.

## Why Python (not Rust) for the LMDB sink — for now

The existing consumer is `ingest_to_lmdb.py`, and that is where the new
intern-table writing belongs **for the current arc**. The Haskell side
already has a working LMDB binding (`lmdb >= 0.2.5`), and we could in
principle have written the LMDB sink in Rust or extended the Haskell
binary. We are not, today, because:

1. The Python sink **already exists** and already handles the writing
   side correctly (filter, project, dupsort, batched txns). Extending
   it adds named sub-dbs and `MDB_APPEND` — both supported by the
   Python `lmdb` package. No new language and no new dependency.
2. Lazy I/O in Haskell makes streaming an attentive engineering
   exercise (`bytestring`, careful seq, watch for retainers). The
   Python `lmdb` package's eager-by-default model reaches the same
   answer with less ceremony.
3. The same consumer feeds the WAM-Rust and WAM-Elixir backends today.
   Keeping the sink in Python means one shared tool, not three.

This is not a statement that Haskell is bad at this; it is a statement
that the one-time, single-process consumer sits naturally where it
already lives.

### Pluggable sinks mirror pluggable parsers

The pluggable-parser property has a symmetric pluggable-sink property
that we explicitly preserve. The `streaming_glue` layer treats both
endpoints the same way:

```prolog
% Today (Python sink — fast enough):
:- declare_target(ingest_to_lmdb/3, python,
                  [leaf(true),
                   script_path('src/.../ingest_to_lmdb.py'),
                   pip_packages([lmdb])]).

% Future (one-line swap to a Rust-direct-LMDB sink, no Python in the loop):
:- declare_target(ingest_to_lmdb/3, rust,
                  [leaf(true),
                   native_crate(lmdb_sink)]).
```

A Rust-native sink would skip the producer→consumer pipe entirely
(linking the parser and writer in-process) and avoid TSV serialisation
between the two stages. We are choosing not to build it now because
the current bottleneck is in the Haskell *runtime*, not in
preprocessing throughput — but the door is left open as a one-line
Prolog change. Same pattern, same guarantees.

The directional principle is: **language choice for any pipeline stage
is a declarative concern, not an architectural one.** This is true
today for the parser (Rust canonical, AWK alternate, Haskell
hypothetical) and remains true for the sink (Python today, Rust later
if and only if we measure the producer→consumer pipe as a real cost).

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
- `src/unifyweaver/runtime/rust/mysql_stream/` — Rust parser (canonical
  for benchmarks).
- `src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py` —
  Python LMDB consumer that the text-keyed extension lives in.
- `examples/streaming/enwiki_category_ingest.pl` — production Prolog
  glue using the Rust parser.
- `examples/streaming/enwiki_category_ingest_awk.pl` — AWK variant,
  demonstrating the one-line parser swap.
- `examples/streaming/enwiki_category_ingest_csharp.pl` — C# consumer
  variant.
- `src/unifyweaver/glue/streaming_glue.pl` — the declarative
  producer/consumer composer.
- `templates/targets/haskell_wam/main.hs.mustache` — existing
  `int_atom_seeds(true)` mode that this work generalises.
