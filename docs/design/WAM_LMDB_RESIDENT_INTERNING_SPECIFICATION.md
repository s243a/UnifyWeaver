# LMDB-Resident Interning: Specification

This document describes the technical shape of the LMDB-resident
interning design. For *why* each decision, see
`WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md`. For the rollout sequence,
see `WAM_LMDB_RESIDENT_INTERNING_IMPLEMENTATION_PLAN.md`.

## 1. LMDB layout

A single `data.mdb` file (LMDB environment) with the following
sub-databases. All Int IDs are unsigned 32-bit, big-endian on disk
(matches the existing `lmdbRawEdgeLookup` packed-binary convention in
`templates/targets/haskell_wam/lmdb_fact_source.hs.mustache`).

### 1.1 `meta` (named sub-db)

Plain key-value, no `MDB_DUPSORT`. ASCII keys, fixed-shape values.

| Key | Value | Purpose |
|---|---|---|
| `schema_version` | uint32 BE | Currently `1`. Bumped on incompatible layout change. |
| `next_id` | uint32 BE | The next ID the ingester will allocate. Read-only at runtime. |
| `compile_time_atoms_count` | uint32 BE | Number of reserved well-known IDs at the low end. |
| `source_dump_sha256` | 64 ASCII hex bytes | SHA-256 of the input dump. Detects "did the input change?" |
| `build_timestamp` | ASCII ISO-8601 | When the file was created. Diagnostic only. |
| `cli_args` | ASCII | The exact CLI invocation. Diagnostic only. |

### 1.2 `s2i` (named sub-db)

String → Int.

- Key: UTF-8 bytes of the atom string.
- Value: uint32 BE.
- No dupsort.
- Sorted by string lexicographically (LMDB default `MDB_NOOVERWRITE` ordering).

The ingester writes this sub-db **after** the streaming pass, by
sorting the in-memory `HashMap<String, u32>` by string and bulk-loading
with `MDB_APPEND`. Before the bulk load the sub-db must be empty.

### 1.3 `i2s` (named sub-db)

Int → String.

- Key: uint32 BE.
- Value: UTF-8 bytes.
- No dupsort.
- Sorted by Int (monotonically by construction during ingestion).

The ingester writes this sub-db **during** the streaming pass with
`MDB_APPEND` because IDs are allocated monotonically.

### 1.4 `category_parent` (named sub-db, dupsort)

Int → Int (one-to-many).

- Key: uint32 BE (child ID).
- Value: uint32 BE (parent ID).
- `MDB_DUPSORT` + `MDB_INTEGERKEY` + `MDB_INTEGERDUP` (or fixed-size
  surrogates if `MDB_INTEGERKEY` proves incompatible with packed BE; see
  §1.7).
- Sorted by (child, parent).

The ingester writes this with `MDB_APPEND` + `MDB_APPENDDUP` because
the categorylinks dump is sorted by `cl_from` and the parser yields
rows in input order.

### 1.5 `article_category` (named sub-db, dupsort)

Int → Int (one-to-many).

- Same shape as `category_parent`.
- Required for the per-article aggregation step in
  `effective_distance` style queries.
- Optional in deployments that only need pure category-graph queries;
  ingester accepts a `--no-articles` flag.

### 1.6 `roots` (named sub-db)

Int set, stored as keys with empty values.

- Key: uint32 BE.
- Value: empty bytes.
- Sorted by Int.

Carries the set of "true root" categories that the runtime may use
when no root is specified on the CLI. Optional; the runtime still
accepts a `--root <name>` argument that is resolved via `s2i`.

### 1.7 Endianness and `MDB_INTEGERKEY`

LMDB's `MDB_INTEGERKEY` requires native-endian integer keys. The
existing Haskell `lmdbRawEdgeLookup` already uses **big-endian** packed
binary so cross-platform reads work. We keep that convention and do
**not** set `MDB_INTEGERKEY` on integer-keyed sub-dbs; LMDB sorts BE
integers correctly via byte-wise comparison. This loses a small
optimisation in LMDB's key comparator but keeps the existing Haskell
reader unchanged.

## 2. Reserved ID range

IDs `0`..`compile_time_atoms_count - 1` are reserved for compile-time
well-known atoms (`[]`, `.`, `true`, `fail`, etc.). The Prolog codegen
emits `compileTimeAtomTable` with these IDs at fixed positions. The
ingester:

1. Reads `compile_time_atoms_count` from a sidecar file or CLI flag
   (must match the codegen).
2. Pre-populates `s2i` and `i2s` with the well-known atoms at those IDs.
3. Sets `next_id` to `compile_time_atoms_count`.
4. Streams the dump, allocating IDs from `next_id` upward.

If a runtime atom string collides with a reserved well-known atom (e.g.
a category literally named `"true"`), the existing well-known ID is
returned; no new ID is allocated. This is the same contract
`internAtom` already enforces in-process.

## 3. CLI contract

A new binary in the existing `mysql_stream` crate (or a sibling crate
re-exporting the parser):

```
mysql_stream_lmdb \
  --output PATH \
  [--cols CHILD,PARENT] \
  [--filter COL=VALUE] \
  [--articles-cols COL,COL --articles-filter COL=VALUE] \
  [--articles] \
  [--compile-time-atoms FILE.txt] \
  [--map-size BYTES] \
  [--no-articles] \
  [--source-sha PRECOMPUTED_SHA] \
  INPUT.sql.gz
```

- `--cols 0,1` selects (child, parent) column indices in the categorylinks
  rows. Defaults are tuned for the SimpleWiki / enwiki categorylinks
  schema: `cl_from` and `cl_to`.
- `--filter 2=subcat` keeps only rows where column index 2 (i.e.
  `cl_type`) equals `subcat`. Multiple `--filter` flags AND together.
- `--compile-time-atoms` is a one-line-per-atom text file aligned with
  the codegen's compile-time table.
- `--map-size` is the LMDB env max size in bytes. The default is large
  enough for full enwiki; the user can shrink it for small fixtures.
- `--source-sha`, if provided, is written to `meta.source_dump_sha256`
  verbatim. If omitted the binary computes it while streaming.

The binary writes the LMDB at `--output`. If the directory exists and
`meta.schema_version` matches, the binary is idempotent: it exits
non-zero with a diagnostic unless `--force` is given.

## 4. Runtime contract

A new emit-mode in `wam_haskell_target.pl` — call it
`int_atom_seeds(lmdb)` (extending the existing `int_atom_seeds(true)`).
The generated `Main.hs`:

1. Opens the LMDB env at `factsDir ++ "/data.mdb"`. Reads
   `meta.schema_version`; bails if it doesn't match the codegen.
2. Reads `meta.compile_time_atoms_count`. Asserts it equals the codegen's
   compile-time atom count.
3. Builds `fullInternTable` directly from the `i2s` sub-db. The
   resulting `InternTable` is a thin wrapper: `iAtom` becomes a wrapper
   around `mdb_get s2i`, and the reverse lookup is `mdb_get i2s`. Both
   are O(log n) page walks on a memory-mapped file.
4. Resolves the root from the CLI argument via `s2i`.
5. Computes `seedCats` and `articleCategories` by scanning the
   `roots` sub-db (or the LMDB key set of `category_parent` if the
   workload's seeds are "all categories that are children").
6. Computes `demandSet` by backward BFS from `rootId` over LMDB cursor
   walks of `category_parent`.
7. Filters seeds via `filteredSeedCats = filter (IS.member . iAtom) seedCats`
   exactly as today.
8. Runs `parMap rdeepseq` over `filteredSeedCats`.
9. At output time, looks up each result Int via `i2s`.

The hot path performs LMDB lookups, not in-memory `IntMap` lookups. On
warm runs the OS page cache absorbs the lookups; on cold runs the
mmap'd page faults pull pages on demand. Both are faster than parsing
TSV strings.

## 5. Atom intern table representation

```haskell
data InternTable
  = InMemory { itForward :: !(Map.Map String Int)
             , itReverse :: !(IM.IntMap String)
             , itSize    :: !Int
             }
  | LmdbBacked { itEnv :: !MDB.Environment
               , itS2i :: !MDB.Database
               , itI2s :: !MDB.Database
               , itSize :: !Int  -- read once from meta.next_id
               }
```

`internAtom :: InternTable -> String -> Int`,
`lookupAtom :: InternTable -> Int -> String`, and
`displayValue` continue to work without callers caring which constructor
they are using. The compile-time atoms (IDs 0..N-1) are still in
`compileTimeAtomTable` and are checked first; only a miss falls through
to the LMDB lookup.

## 6. Demand set construction protocol

Given a memory-mapped LMDB `category_parent` and a target `rootId`, the
demand set is the backward-reachable set:

```haskell
computeDemandSetLmdb :: EdgeLookup -> Int -> IO IS.IntSet
computeDemandSetLmdb edges rootId = bfs (IS.singleton rootId) (IS.singleton rootId)
  where
    bfs visited frontier
      | IS.null frontier = return visited
      | otherwise = do
          children <- forM (IS.toList frontier) $ \nodeId ->
            edgesReverseLookup edges nodeId  -- list of children pointing here
          let newFrontier = IS.fromList (concat children) `IS.difference` visited
          bfs (IS.union visited newFrontier) newFrontier
```

The implementation requires a *reverse* edge lookup. Two options:

1. **Walk forward edges**: for each `(child, parent)` in the LMDB,
   accumulate `parent → [child]` in memory. O(edges) once at startup,
   matches the current `reverseAdj` IntMap construction. Acceptable for
   ≤ 1 M edges.
2. **Add a reverse sub-db** `parent_category` during ingestion. Pure
   space cost on disk (~2× the forward edge size). Removes the startup
   cost.

Option 1 is the default for v1; option 2 is gated behind a
`--with-reverse-edges` flag on the ingester. Either way the runtime
sees the same `EdgeLookup` interface.

## 7. Compatibility with existing `int_atom_seeds(true)`

The current `int_atom_seeds(true)` mode reads `seed_ids.txt` and
`root_ids.txt` from disk. It stays as-is. The new
`int_atom_seeds(lmdb)` mode supersedes it for fixtures that come with a
preprocessed LMDB; the old mode remains for hand-prepared int-id
files.

The two modes share the same downstream code (everything from
`fullInternTable` onward in `Main.hs`); they differ only in how they
populate it.

## 8. Output sha equivalence

The ingester is a deterministic function of the input dump (modulo
`build_timestamp` and `cli_args` which only live in `meta`). Two runs
of the ingester on the same input produce byte-identical edge,
`s2i`, and `i2s` sub-dbs.

The Haskell binary's stdout is a deterministic function of the LMDB it
reads. We require that scale-300 (and any covered scale) produces the
same `stdout_sha256` as the current TSV-resident path. Any divergence
is a bug.
