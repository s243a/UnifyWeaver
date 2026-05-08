# LMDB-Resident Interning: Implementation Plan

For the *why*, see `WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md`. For the
*what*, see `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`. This
document records the ordered rollout.

## 0. Starting point

Current state on `main` after PR #1882 / #1901:

- `data/benchmark/100k_cats/lmdb_proj/src/Main.hs:131-180` parses two
  TSVs (~280k pairs) and rebuilds the intern table on every run, even
  when `data.mdb` exists. Sequential setup floor: ~3.2 s wall-time
  out of ~3.2 s total at `-N1` on `100k_cats`.
- `templates/targets/haskell_wam/main.hs.mustache` already has an
  `int_atom_seeds(true)` mode that skips TSV loading and reads
  pre-interned int-id files. Used by the enwiki path; not used by the
  matrix bench fixtures.
- **`src/unifyweaver/runtime/rust/mysql_stream/`** — Rust parser for
  MySQL INSERT dumps. ~555 lines, validated against simplewiki (2.2 M
  rows, byte-exact match to SQLite ground truth, ~28 MB/s gzipped on
  one core). Schema-agnostic, emits TSV to stdout.
- **`src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py`** —
  Python LMDB consumer. ~206 lines. Reads TSV from stdin, writes LMDB
  with filtering, projection, encoding (`int32_le`), and dupsort. Already
  in production for the integer-keyed enwiki ingest.
- **`examples/streaming/enwiki_category_ingest.pl`** — Prolog glue
  composing the Rust producer and Python consumer.
- **`examples/streaming/enwiki_category_ingest_awk.pl`** — AWK variant
  of the parser, demonstrating the one-line `declare_target` swap.
- **`examples/streaming/enwiki_category_ingest_csharp.pl`** — C#
  consumer variant.
- `src/unifyweaver/glue/streaming_glue.pl` — the producer/consumer
  composer.
- `examples/benchmark/prepare_effective_distance_large_scales.py` builds
  the `100k_cats` fixture by extracting from a SimpleWiki SQLite db.

The plan below **extends the existing Python consumer** with text-keyed
intern support. It does **not** add a new Rust crate or a new LMDB
binding; the parser stays untouched, and the consumer's existing
codepath stays the default for the integer-keyed enwiki regime.

## 1. Phase 1 — extend Python consumer with text-keyed intern support

**Branch:** `feat/lmdb-ingest-text-keyed-intern`

The Rust parser stays unchanged. All work in this phase is in
`src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py` and a
new Prolog example.

### 1.1 New env-var surface

Per `SPECIFICATION §3.2`, add the env vars `UW_INTERN_KEY`,
`UW_INTERN_VAL`, `UW_LMDB_S2I_DB`, `UW_LMDB_I2S_DB`, `UW_LMDB_META_DB`,
`UW_LMDB_APPEND`, `UW_COMPILE_TIME_ATOMS`, `UW_SOURCE_SHA`,
`UW_SCHEMA_VERSION`, `UW_FORCE_REINGEST`. All default off; existing
behaviour is preserved exactly.

### 1.2 Intern map + sub-db writes

When `UW_INTERN_KEY=1` (and/or `UW_INTERN_VAL=1`):

- Maintain a Python `dict[str, int]` for the in-memory intern map.
- On each row: look up the string in the dict; if absent, allocate the
  next ID (starting from `compile_time_atoms_count`), append to `i2s`
  with `append=True` (IDs are monotonic by construction), and record
  in the dict.
- On finalize: sort the dict by key, bulk-load `s2i` with
  `append=True`. Write `meta` keys. Commit.

When `UW_LMDB_APPEND=1` and the input is sorted, edge writes use
`append=True` for the corresponding sub-db.

### 1.3 Compile-time atoms sidecar

Read the optional one-atom-per-line file at `UW_COMPILE_TIME_ATOMS`.
Pre-populate `s2i` and `i2s` at the low IDs (0 .. N-1). Set
`meta.compile_time_atoms_count = N`. Subsequent runtime atoms get IDs
starting at N.

If the input contains a string that collides with a reserved
compile-time atom (e.g. literal `"true"`), it returns the existing
reserved ID — no new ID is allocated.

### 1.4 Idempotence + reingestion guard

On startup, if the LMDB exists and has `meta.schema_version` matching
`UW_SCHEMA_VERSION`, exit with a diagnostic unless
`UW_FORCE_REINGEST=1`. This prevents accidental double-ingestion
into a populated database.

### 1.5 New Prolog example

`examples/streaming/simplewiki_category_ingest_text.pl` — same shape
as `enwiki_category_ingest.pl` but configured for text-keyed input
(string columns, `UW_INTERN_KEY=1 UW_INTERN_VAL=1`, sub-db names per
the spec). Producer declaration is unchanged from the enwiki example.

### 1.6 Tests

- Unit test on the consumer with a small synthetic TSV:
  - Text-keyed input → check `s2i`, `i2s`, edges, `meta` keys.
  - Reserved-ID collision: input containing the literal string
    `"true"` → returns the reserved low ID, not a fresh one.
  - Idempotence: rerun without `UW_FORCE_REINGEST` → exits non-zero
    with a clear diagnostic.
- Round-trip test driven by the existing Prolog harness:
  produce → consume → read back → reconstruct edges → byte-equal.

### 1.7 Acceptance

Tests green. Spike measurement: ingest SimpleWiki categorylinks
(~280k pairs) end-to-end in well under the 3.2 s the current Haskell
setup phase takes. The integer-keyed enwiki path keeps producing
byte-identical LMDB output (no behaviour change when the new env
vars are unset).

## 2. Phase 2 — Haskell runtime support (`int_atom_seeds(lmdb)`)

**Branch:** `feat/wam-haskell-int-atom-seeds-lmdb`

### 2.1 Extend `InternTable` type

In `templates/targets/haskell_wam/wam_types.hs.mustache` (or wherever
`InternTable` lives) add the `LmdbBacked` constructor per
`SPECIFICATION §5`. Add `internAtom` / `lookupAtom` cases.

### 2.2 New mustache section in `main.hs.mustache`

Add `{{#int_atom_seeds_lmdb}} ... {{/int_atom_seeds_lmdb}}` blocks that
parallel the existing `{{#int_atom_seeds}}` blocks (template lines
129-212), but:

- Open the LMDB env once at startup.
- Read `meta` and validate `schema_version`.
- Build `fullInternTable = LmdbBacked ...`.
- Compute `seedCats` from a `roots` sub-db scan (or `category_parent`
  key set if the workload's seeds are "all children").
- Compute `articleCategories` by iterating the `article_category`
  sub-db.
- Compute `demandSet` via the LMDB-backed `computeDemandSetLmdb` from
  `SPECIFICATION §6`.

### 2.3 `wam_haskell_target.pl` option

Add an `int_atom_seeds(lmdb)` option that activates the new mustache
sections and emits the `lmdb` cabal dep (already conditional via
`use_lmdb(true)`).

### 2.4 Tests

Extend `tests/test_wam_haskell_target.pl`:

- `test_int_atom_seeds_lmdb_emits_env_open` — generated Main.hs opens
  the LMDB at startup.
- `test_int_atom_seeds_lmdb_skips_tsv_load` — generated code does **not**
  call `loadTsvPairs` when `int_atom_seeds(lmdb)` is set.
- `test_int_atom_seeds_lmdb_reads_meta` — generated code reads
  `meta.schema_version`.
- `test_int_atom_seeds_lmdb_demand_uses_lmdb` — generated demand BFS
  walks LMDB cursors (asserts the source string).

### 2.5 Acceptance

Tests green. Generated project for a small fixture builds and runs
end-to-end against a Phase 1 ingester output, with stdout sha matching
the existing TSV path.

## 3. Phase 3 — wire into matrix bench

**Branch:** `bench/wam-haskell-lmdb-resident`

### 3.1 Extend `prepare_effective_distance_large_scales.py`

After the existing fixture build, call `mysql_stream_lmdb` (or a
SQLite-input adapter for the SimpleWiki path) to produce
`data/benchmark/100k_cats/data.mdb`. Verify the LMDB matches the TSVs
by spot-checking a few edges.

### 3.2 Update `generate_wam_haskell_matrix_benchmark.pl`

When `fact_count >= 50000` and a `data.mdb` exists in the fixture,
emit `int_atom_seeds(lmdb)` instead of the default TSV path.

### 3.3 Validation

- Scale-300 matrix bench: output sha `70bbc9ffa4cf` (current).
- 100k_cats default root: output sha matches the current path.
- 100k_cats `Deaths_by_year` root: output rows count matches.

### 3.4 Acceptance

All matrix bench scales green. New baseline numbers recorded in
`WAM_PERF_OPTIMIZATION_LOG.md`.

## 4. Phase 4 — measurements

Single sweep at 100k_cats with default and `Deaths_by_year` roots,
`+RTS -N1/-N2/-N4`, 5 trials each. Compare:

| Metric | TSV path (now) | LMDB-resident |
|---|---|---|
| `loadMs` | ? | ? |
| `setupMs` (intern + parents index) | ? | ? |
| `query_ms` | 0–17 | should match |
| `total_ms` | 3.2 s (-N1) | target: < 1 s |
| Peak RSS | ~600 MB | should drop |

Append the table and a short reading to `WAM_PERF_OPTIMIZATION_LOG.md`
as Phase L appendix #5.

## 5. Phase 5 — cross-target reuse

Audit WAM-Rust and WAM-Elixir runtimes against the same LMDB layout.
The Rust target already uses LMDB for its FFI kernel; the Elixir
target's "lmdb int IDs" path is the closest analogue. If both can
adopt the same `s2i` / `i2s` / `meta` schema, the consumer serves all
three without per-target schema branches.

The Python consumer is already shared across the C# and AWK pipeline
variants (via the streaming-glue layer), so cross-target reuse is the
default rather than the exception. This phase is exploratory: the
deliverable is a memo confirming the schema needs no per-target
adjustment, plus issues tracking the rollout to WAM-Rust and
WAM-Elixir.

### 5.1 Optional follow-up: parser-language extension

The current parser variants are Rust (canonical for benchmarks) and
AWK. If a deployment ever needs a Haskell-language parser (e.g. for
single-binary distribution), it slots into the same `declare_target`
shape. This is **not** in scope for this implementation arc; it is
documented here so the option is preserved if the need arises.

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| Reserved ID drift between codegen and consumer | Single source of truth: the `UW_COMPILE_TIME_ATOMS` sidecar file consumed by the Python consumer, asserted at runtime via `meta.compile_time_atoms_count`. |
| Demand-set BFS over LMDB is slow without reverse edges | Phase 2 ships option 1 (build reverse adjacency in memory at startup); option 2 (write a reverse sub-db at ingestion time) is a follow-up if the in-memory build dominates. |
| Disk size: full enwiki LMDB might not fit a CI cache | Document the size, run small fixtures in CI, run enwiki only locally. |
| TSV path drift: the kept TSV fallback rots from disuse | Matrix bench keeps a small-scale TSV run alongside the LMDB run; if the TSV path breaks the bench fails. |
| Python `lmdb` package's `append=True` rejects unsorted keys | The streaming pass writes `i2s` in monotonic ID order (always sorted) and writes edges in input-order (sorted by `cl_from` for categorylinks); `s2i` is bulk-loaded after sort at finalize. |
| Parser-cost variation contaminating benchmarks | Benchmark harness fixes the parser to Rust via `declare_target`; pluggable-parser property exists but is not exercised in the perf path. |

## 7. Verification

End-to-end verification at every phase boundary:

1. Phase 1 alone: ingester produces an LMDB whose contents the existing
   `lmdbRawEdgeLookup` reader can iterate without error.
2. Phase 2 alone: a hand-built minimal LMDB (no real ingester) drives a
   generated Haskell binary to correct output on a 10-edge fixture.
3. Phase 3 alone: the fixture-build pipeline produces a real LMDB at
   100k_cats and the matrix bench passes.
4. Phase 4: numbers tighten or hold; no scale-300 sha drift.

If any phase regresses scale-300 sha `70bbc9ffa4cf`, that is the
phase to debug before moving on.
