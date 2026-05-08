# LMDB-Resident Interning: Implementation Plan

For the *why*, see `WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md`. For the
*what*, see `WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`. This
document records the ordered rollout.

## 0. Starting point

Current state on `main` after PR #1882:

- `data/benchmark/100k_cats/lmdb_proj/src/Main.hs:131-180` parses two
  TSVs (~280k pairs) and rebuilds the intern table on every run, even
  when `data.mdb` exists. Sequential setup floor: ~3.2 s wall-time
  out of ~3.2 s total at `-N1` on `100k_cats`.
- `templates/targets/haskell_wam/main.hs.mustache` already has an
  `int_atom_seeds(true)` mode that skips TSV loading and reads
  pre-interned int-id files. Used by the enwiki path; not used by the
  matrix bench fixtures.
- `src/unifyweaver/runtime/rust/mysql_stream/` parses MySQL INSERT
  dumps and emits TSV to stdout. ~555 lines, well-tested.
- `examples/benchmark/prepare_effective_distance_large_scales.py` builds
  the `100k_cats` fixture by extracting from a SimpleWiki SQLite db.

## 1. Phase 1 — Rust ingester (`mysql_stream` extension)

**Branch:** `feat/mysql-stream-lmdb-sink`

### 1.1 Add `heed` dependency

Edit `src/unifyweaver/runtime/rust/mysql_stream/Cargo.toml` to add
`heed = "0.20"` (or current stable). Keep it behind a feature flag
`lmdb-sink` so the existing TSV-stdout path has no new compile-time
dependency by default.

### 1.2 `IngestSink` trait

Refactor `iter_mysql_rows` consumers behind a small trait:

```rust
pub trait IngestSink {
    fn on_row(&mut self, row: &[Field]) -> Result<()>;
    fn finalize(self) -> Result<()>;
}
```

The existing TSV-stdout writer becomes `TsvSink`; the new sink is
`LmdbSink`. The streaming loop is unchanged.

### 1.3 `LmdbSink` implementation

New module `src/lmdb_sink.rs` (gated by `lmdb-sink` feature):

- Holds a `heed::Env`, the four named databases, an
  `IndexMap<String, u32>` for the in-memory intern map, and a write
  txn that is committed in `finalize`.
- `on_row` extracts (child, parent) per the configured column indices,
  applies any filters, interns both strings, writes the edge with
  `MDB_APPENDDUP`, and writes any new `i2s` entries with `MDB_APPEND`.
- `finalize` sorts the intern map by string and bulk-loads `s2i` with
  `MDB_APPEND`. Computes / writes `meta` keys. Commits the txn.

### 1.4 New binary `mysql_stream_lmdb`

Adds `[[bin]]` entry in `Cargo.toml`. CLI per
`SPECIFICATION §3`. Wraps `iter_mysql_rows` + `LmdbSink`.

### 1.5 Tests

- Unit tests on `LmdbSink`: small fixture (10 rows), check `s2i`,
  `i2s`, edges, `meta` keys.
- Round-trip test: run `mysql_stream_lmdb` on a synthetic dump, then a
  Rust reader iterates the LMDB and reproduces the input edges.
- Reserved-ID collision test: input dump containing the literal
  string `"true"`, verify it gets the reserved low ID.

### 1.6 Acceptance

`cargo test --features lmdb-sink` green. Spike measurement: ingest
SimpleWiki categorylinks (~280k pairs) end-to-end in well under the
3.2 s the current Haskell setup phase takes.

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
adopt the same `s2i` / `i2s` / `meta` schema, the ingester serves all
three without per-target schema branches.

This phase is exploratory; the deliverable is a memo on whether the
schema needs adjustment to be cross-target, and an issue tracking the
follow-up rollouts.

## 6. Risks

| Risk | Mitigation |
|------|-----------|
| `heed` API changes between minor versions | Pin a specific `heed` version in `Cargo.toml`. |
| `MDB_INTEGERKEY` incompat with BE bytes | Already chose **not** to use it (`SPECIFICATION §1.7`). |
| Reserved ID drift between codegen and ingester | Single source of truth: a sidecar text file consumed by both, asserted at runtime via `meta.compile_time_atoms_count`. |
| Demand-set BFS over LMDB is slow without reverse edges | Phase 1 ships option 1 (build reverse adjacency in memory at startup); option 2 (`--with-reverse-edges` writing a reverse sub-db) is a follow-up if the in-memory build dominates. |
| Disk size: full enwiki LMDB might not fit a CI cache | Document the size, run small fixtures in CI, run enwiki only locally. |
| TSV path drift: the kept TSV fallback rots from disuse | Matrix bench keeps a small-scale TSV run alongside the LMDB run; if the TSV path breaks the bench fails. |

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
