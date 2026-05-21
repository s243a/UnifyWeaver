# WAM-C++ LMDB FactSource: Design

**Status**: draft. Targets v1 implementation; the v2 / interning story
is sketched but not committed.

## TL;DR

The C++ WAM target gains an LMDB-backed fact source mirroring the
**C target**'s shape — load-everything-on-open semantics, atom-only
UTF-8 encoding, arity-2 only, optional via `WAM_CPP_ENABLE_LMDB`
build flag. A new `cpp_fact_sources` codegen option lets users
register predicates against an LMDB file at write time. The runtime
exposes a `LmdbFactSource` C++ class with `lookup_by_arg1` and
`stream_all` methods, called from generated dispatch code wired into
the existing fact-table path. No runtime changes to the WAM step
loop are needed — facts loaded from LMDB live in the same in-memory
table that `assertz` populates, so existing unification/indexing
machinery handles queries unchanged.

This v1 deliberately under-scopes for parity with the C target, not
the Rust/Haskell ones. The Rust/Haskell stack uses integer-interned
multi-sub-db packed-BE LMDB layouts
(`WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`) that are richer but
also coupled to Haskell's `lmdbRawEdgeLookup` convention and Rust's
crate-selection scaffolding. C++ adopting that schema is v2 work —
it pays off only after the simpler path is shipping and the layout
is empirically the bottleneck.

## Goal

"Parity with Rust and Haskell" means: a C++ WAM program can be
configured to back a Prolog predicate by an LMDB file on disk, query
that predicate via normal Prolog unification, and get the same
answers it would have got from inline facts. The user-visible
ergonomics should feel like the existing per-target fact-source
options (`scala_fact_sources`, `r_fact_sources`, `facts_lmdb`).

What it does **not** mean for v1:

- Bytewise-identical LMDB schema with Rust/Haskell. They use packed
  uint32 + DUPSORT; C++ v1 uses UTF-8 string pairs (like C).
- Sub-database support. v1 is single-DB-per-env.
- Probe-on-demand. v1 loads the whole DB into the in-memory fact
  table at open time (matches C).
- Arity > 2. v1 covers the dominant `edge(Child, Parent)` /
  `parent(Child, Parent)` use case.

## Scope: v1 vs deferred

| Concern | v1 (this design) | Deferred (v2+) |
|---|---|---|
| Encoding | UTF-8 atom strings | `tag:payload` (atom / int / float) — Haskell-style |
| Arity | 2 only | Generic arity via templated FactSource |
| Load strategy | Eager (load entire DB at open) | Lazy probe-on-demand cursor lookups |
| Sub-databases | Single unnamed DB per env | Named sub-DBs (`s2i`, `i2s`, `category_parent`, …) |
| Key shape | Raw bytes (UTF-8) | Packed big-endian uint32 |
| Concurrency | Single-process, single-load | Multi-reader transactions |
| Build | Optional via `WAM_CPP_ENABLE_LMDB` | Same |
| Crate / library choice | System `liblmdb` only | n/a (C++ has one canonical lib) |

## Why mirror C, not Rust / Haskell, for v1

| Property | C target | Rust target | Haskell target | Choice for C++ v1 |
|---|---|---|---|---|
| Schema | UTF-8 string pairs | Packed BE uint32 + DUPSORT + multi-sub-db | Same as Rust | **C-shape** |
| Linking | System liblmdb | Crate (lmdb-zero / heed) | System liblmdb | **System liblmdb** |
| Ingestion path | Seeder C file emitted alongside the codegen output | External Rust ingester binary | Streaming Haskell preprocessor | **Seeder C++ stub + bring-your-own file** |
| In-memory model | Edge table, linear scan | Memory-mapped + zero-copy reads | Memory-mapped + cached arg1 index | **Edge table, linear scan** |
| Build flag | `WAM_C_ENABLE_LMDB` | Cargo feature | cabal flag | `WAM_CPP_ENABLE_LMDB` |

Three reasons:

1. **The C target is the closest analog.** Same systems-language
   constraints, same FFI surface (raw `liblmdb`), same lifetime
   discipline. The pattern at `wam_c_target.pl:2390-2461` is
   directly translatable to C++ with RAII replacing manual cleanup.
2. **The Rust/Haskell schema is coupled to its consumers.** The
   packed-BE + sub-DB layout exists because the Haskell runtime
   expects `lmdbRawEdgeLookup` to read uint32 keys
   (`WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md:18-19`), and the
   Rust pipeline ingests directly into that shape. Adopting it in
   C++ means bringing along an ingester (or coupling to the Rust
   one). C-shape decouples cleanly.
3. **v1 unblocks measurement.** Once the eager-load, UTF-8 path
   ships, we have a baseline to compare against. Following the same
   logic as `WAM_RUST_LMDB_CRATE_DECISION.md:56-58` — we can't
   measure what we don't have. v2 (richer schema, lazy probing) gets
   prioritised when v1 hits a real ceiling, not before.

## Configuration surface

A new codegen option `cpp_fact_sources(+List)` flows through
`write_wam_cpp_project/3`:

```prolog
write_wam_cpp_project(
    [user:edge/2, user:query_descendant/2],
    [ emit_main(true),
      include_stdlib(lists_extra),
      cpp_fact_sources([
          source(edge/2, lmdb('graph.mdb')),
          source(other/2, lmdb('other.mdb', [db_name('items')]))
      ])
    ],
    OutDir).
```

`source(Pred/Arity, Spec)` entries with `Spec = lmdb(Path)` or
`lmdb(Path, OptList)` cause the codegen to:

- Skip emitting in-source clauses for `Pred/Arity` (the LMDB file is
  the source of truth).
- Emit a fact-source registration call in `init_runtime` that opens
  the LMDB env and loads all rows into the in-memory fact table for
  that key.

Sub-options recognised by v1:

| Option | Default | Notes |
|---|---|---|
| `db_name(Atom)` | unnamed main DB | Lets one env hold multiple predicates. |
| `read_only(true)` | `true` | The only mode v1 supports. Listed for forward-compat. |

Unknown options error at codegen time (don't silently ignore — the
generated binary would mislead the user).

## Runtime ABI

```cpp
class LmdbFactSource {
public:
    // Open an LMDB env at env_path. db_name == nullptr selects the
    // unnamed main DB. Throws LmdbError on open failure.
    LmdbFactSource(const std::string& env_path,
                   const char* db_name = nullptr);

    // RAII close — mdb_env_close in destructor.
    ~LmdbFactSource();

    LmdbFactSource(const LmdbFactSource&) = delete;
    LmdbFactSource& operator=(const LmdbFactSource&) = delete;

    // Iterate the entire DB once, calling sink(key, value) per row.
    // Used at load time to populate the WAM fact table.
    void stream_all(
        std::function<void(std::string_view, std::string_view)> sink);

    // Direct probe by first-arg key. Returns all values whose key
    // matches. Reserved for v2 (lazy probing); v1 only calls
    // stream_all.
    std::vector<std::string> lookup_by_arg1(std::string_view key);

private:
    MDB_env* env_ = nullptr;
    MDB_dbi  dbi_ = 0;
    bool     dbi_open_ = false;
};
```

Load-time integration (called from generated `init_runtime`):

```cpp
void cpp_load_lmdb_fact_source(
    WamState& state,
    const std::string& functor_arity_key,  // e.g. "edge/2"
    const std::string& env_path,
    const char* db_name)
{
    LmdbFactSource src(env_path, db_name);
    src.stream_all([&](std::string_view k, std::string_view v) {
        // Build the equivalent of assertz(edge(k, v)) into the
        // in-memory dynamic_db, keyed by functor_arity_key.
        auto cell = make_arity_2_term(functor_arity_key, k, v);
        state.dynamic_db[functor_arity_key].push_back(std::move(cell));
    });
}
```

Bytes-to-WAM-value rule (v1): bytes are interpreted as UTF-8 atom
text. The cell built per row is `Compound(functor_arity_key,
{Atom(k), Atom(v)})`. Same convention as the C target — no tag
prefix, no type inference, atoms only.

## Codegen integration

The generated `init_runtime` runs LMDB load before any user query
dispatches. For each `source(Pred/Arity, lmdb(Path, Opts))` entry,
emit one line:

```cpp
cpp_load_lmdb_fact_source(state, "edge/2", "graph.mdb", nullptr);
```

Query dispatch is **unchanged**. Because the loaded rows live in the
existing `dynamic_db` keyed by `"edge/2"`, the regular dynamic-call
dispatch path (the one `assertz` populates) finds them naturally —
indexing, backtracking, and CP-creation machinery are reused as-is.

No changes to:
- The WAM step loop.
- The dispatch arms in `step()`.
- The existing fact-table representation.
- Any user-facing predicate semantics.

This is the design's biggest leverage point. By loading into the
existing dynamic_db, we avoid the temptation to invent a parallel
fact path and the complexity that brings.

## Build integration

CMake feature detection in `src/unifyweaver/runtime/wam_cpp_runtime/`
(or the codegen's emitted CMakeLists):

```cmake
find_path(LMDB_INCLUDE_DIR lmdb.h)
find_library(LMDB_LIBRARY lmdb)
if(LMDB_INCLUDE_DIR AND LMDB_LIBRARY)
    set(WAM_CPP_ENABLE_LMDB ON)
    target_compile_definitions(wam_runtime PUBLIC WAM_CPP_ENABLE_LMDB)
    target_link_libraries(wam_runtime PUBLIC ${LMDB_LIBRARY})
    target_include_directories(wam_runtime PUBLIC ${LMDB_INCLUDE_DIR})
endif()
```

Gracefully degrades when liblmdb is absent: codegen still emits the
`#include <lmdb.h>`-guarded `LmdbFactSource` class, but it's behind
`#ifdef WAM_CPP_ENABLE_LMDB`. Without the flag, calling
`cpp_load_lmdb_fact_source` returns false and the dynamic_db stays
empty (mirrors C target's stub at
`wam_c_target.pl:2392-2398`).

User install path: `apt install liblmdb-dev` (Debian/Ubuntu),
`brew install lmdb` (macOS), or system equivalent. No vendoring —
liblmdb is small, stable, and broadly packaged.

## Comparison with Rust and Haskell impls

| Dimension | C++ v1 | Rust | Haskell |
|---|---|---|---|
| Linking | System liblmdb | Crate (lmdb-zero default, heed alt) | System liblmdb via Haskell FFI |
| Schema | UTF-8 strings, arity 2 | Packed BE uint32, multi-sub-db | Packed BE uint32, multi-sub-db |
| Load model | Eager (full table copy) | Lazy probe via cursor | Eager + arg1 index cache |
| Ingester | User-provided (any tool that writes UTF-8 kv) | Streaming Rust binary | Streaming Haskell preprocessor |
| Intern table | None (atoms-only) | s2i + i2s sub-DBs | s2i + i2s sub-DBs |
| Build option | `WAM_CPP_ENABLE_LMDB` | Cargo feature | cabal flag |
| Crate choice | n/a | `lmdb_crate(lmdb_zero|heed|auto)` | n/a |

## Future work: bridging to integer-interned LMDB

The Rust/Haskell schema
(`WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`) is strictly more
powerful for large graphs because uint32 keys + `MDB_DUPSORT` +
`MDB_INTEGERKEY` + memory-mapped cursor reads beat eager-load-strings
on memory and query latency at scale. The path forward when v1 hits
a ceiling:

1. **v1.5**: add `lookup_by_arg1` real implementation (cursor probe
   instead of pre-loaded linear scan). Keep the same UTF-8 schema.
   Gain: query latency on partially-bound calls.
2. **v2**: add a second codegen mode `lmdb(Path, [interned])` that
   reads the Rust/Haskell schema. Requires:
   - `s2i` / `i2s` sub-DB readers.
   - Packed big-endian uint32 decode.
   - DUPSORT cursor handling.
   This unlocks interop with existing Rust/Haskell-built LMDB
   files, which is the strongest version of "parity."
3. **v3**: arity-generic FactSource via templated wrapper, so
   non-arity-2 predicates can be backed by LMDB.

Each step is independently shippable.

## Resolved decisions (from PR #2319 review)

These were "open questions" in the initial design; the responses
below capture the resolution and the rationale.

### Schema must be encoded in the DB

LMDB itself has no schema — keys and values are bag-of-bytes. The
v1 design needs to encode column names somewhere so the file is
self-describing. Without that, an arity-2 LMDB file looks identical
to any other and there's no way to ask "what predicate is this for,
and what does each column mean?"

**v1 convention:** a `__meta__` sub-DB (one per env, named exactly)
with reserved ASCII keys. Required entries:

| Key | Value | Purpose |
|---|---|---|
| `schema_version` | `1` (ASCII) | Bumped on incompatible layout change |
| `predicate` | `edge/2` (ASCII) | Functor/arity this DB backs |
| `columns` | `child,parent` (ASCII) | Comma-separated column names matching the arity |

A v1 reader validates `schema_version == 1`, that `predicate`
matches what the codegen registered, and that `columns` parses
into exactly `arity` comma-separated names. Mismatch is a
load-time error, not a silent skip — the LMDB file is the source
of truth and a wrong schema is a user mistake worth surfacing.

For `db_name(Atom)` configs, each named sub-DB carries its own
`__meta__` keys (we use prefix `<db_name>:` on the meta keys, e.g.
`items:predicate`, since LMDB doesn't support a sub-DB-of-a-sub-DB).

### Duplicate-row ordering needs explicit semantics

Graph applications (the current motivation) have unique nodes, so
duplicate keys aren't an issue and any iteration order works. But
once LMDB backs general predicates with non-unique first args, the
order in which clauses are tried matters — Prolog programs depend
on it for cut behavior and "first-solution" patterns.

**v1 position:** for graph use cases we accept LMDB's natural sort
order (keys sorted lexicographically; values in insertion order
when `MDB_APPEND` was used during ingest). Document that this
**is** the contract and that users requiring strict insertion order
across re-ingests should either (a) include an explicit order
column in the schema or (b) wait for v2.

**v2 sketch:** support an explicit `order_by(Column)` sub-option:

```prolog
source(rule/3, lmdb('rules.mdb', [columns([id, lhs, rhs]),
                                    order_by(id)]))
```

The codegen would emit a load-time sort by the named column before
populating the dynamic_db, giving deterministic clause order
independent of ingestion path. Strictly v2 — out of scope for v1.

### Idempotent re-call

If `init_runtime` runs twice in the same process, the second call
short-circuits for any already-loaded LMDB source. Tracked via a
`loaded_lmdb_sources` set keyed on `(env_path, db_name)`.

Why idempotent rather than reload: in practice the only reason
`init_runtime` runs twice is test harness re-entry, where reload
would double-populate the dynamic_db and break unification. A
real "reload from disk" use case would want an explicit
`cpp_reload_lmdb_fact_source(state, key)` primitive, not silent
re-firing of init.

### liblmdb version floor

v1 targets liblmdb **0.9.16+**. Released 2014; ships in Ubuntu
18.04, Debian Buster, RHEL 8, macOS Homebrew. No known API breaks
since. The CMake feature detection accepts any 0.9.x library that
exposes `mdb_env_create`, `mdb_env_set_maxdbs`, `mdb_dbi_open`,
`mdb_cursor_get` — the four functions v1 actually calls.

## Future work: memory-mapped arrays as a parallel backend

For the eager-load v1 path, an LMDB file and a memory-mapped flat
array end up in roughly the same place: every row gets read into
the in-memory `dynamic_db` at startup. LMDB pays B-tree overhead
for ordered key lookups we don't use in eager mode; a flat
memory-mapped array is likely faster for that case.

**C# already does this.** `csharp_query_runtime/QueryRuntime.cs`
defines `MmapArrayRelationArtifactManifest` (line 403) and
`MmapArrayRelationArtifactProvider` (line 606), with format ID
`unifyweaver.mmap_array_relation.v1` (line 405) and physical
backend label `mmap_array` (line 413). It's the only target with
production MMA support today, so it sets the precedent.

**v3 proposal for C++:** add `mmap_array(Path)` as an alternative
spec under `cpp_fact_sources`, sibling to `lmdb(Path)`:

```prolog
cpp_fact_sources([
    source(edge/2,  lmdb('graph.mdb')),
    source(other/2, mmap_array('other.bin'))
])
```

Same FactSource abstraction (`stream_all` + future `lookup_by_arg1`),
different concrete implementation. For arity-2 atom data the MMA
file format is just `[uint32 key_len][key bytes][uint32 val_len][val
bytes]` records back-to-back, mmap'd with `MAP_PRIVATE`.

Long-term both backends coexist:

- **MMA** wins for eager-load, sequential-scan, write-once
  workloads — the dominant pattern today.
- **LMDB** wins when v1.5's lazy cursor probing lands and partial
  loads start mattering, when multiple processes need to share the
  file, or when interop with Rust/Haskell-built LMDB files matters
  (v2).

The C# design should be the C++ design's reference for the MMA
format, so any future MMA work cross-pollinates between targets
(and possibly lets one target consume another's artifacts).

## Future work: pre-sorted compound-key schema

When a user declares `order([arg(2)])` (or any non-trivial order),
the v2 runtime sorts the loaded bucket with `std::sort` — O(n log
n) at load time. The Phase 2 implementation (PR #2327) emits a
codegen-time warning advising the user that this cost exists and
can usually be avoided by designing the LMDB schema so the data
is already in the desired order.

The technique: encode the sort column directly into the LMDB key.
Instead of storing `arg1 -> arg2` (sorted by arg1, the LMDB
default), store `<arg2>-<unique-id> -> arg1`. LMDB then iterates
in arg2-sorted order natively. The `<unique-id>` discriminator
avoids LMDB key collisions when multiple rows share the same
sort-column value.

Tradeoff:

- **Wins**: zero sort cost at load; B-tree locality matches the
  query pattern; future `lookup_by_arg1` (v1.5) probes the
  intended column directly.
- **Costs**: keys are longer (one full atom + delimiter + id) so
  the LMDB on-disk size grows; intern tables that key on the LMDB
  key wouldn't deduplicate the prefix portion.

The intern-table cost is mitigated by interning only the
**discriminator** portion (the unique id), since the sort-column
prefix repeats across rows that share a sort value and is
typically a small set of distinct values. The
`<sort_column>-<id>` shape lets a separate `id -> long_value`
intern table reuse identifiers across the data DB and the intern
sub-DB without duplicating the long values.

This is strictly v2 work — it needs:

1. A new `cpp_fact_sources` source spec option (e.g.
   `lmdb(Path, [compound_key([sort_column, unique_id_column])])`)
   telling the codegen which delimiter to split on.
2. Runtime decode of the compound key into the real row columns
   before pushing to `dynamic_db`.
3. A migration story for users who already have flat-key LMDB
   files (probably an explicit reseed step).

The Phase 2 warning is the in-band hint that nudges users toward
this design before they get there.

## Cross-references

- C target reference impl: `src/unifyweaver/targets/wam_c_target.pl:2390-2461`
- C runtime ABI: `src/unifyweaver/runtime/wam_c_runtime/wam_runtime.h:121-125, 281-288`
- Scala FactSource abstraction (cleanest API surface):
  `src/unifyweaver/targets/wam_scala_target.pl:786-804`
- Rust crate decision (option pattern this design follows):
  `docs/design/WAM_RUST_LMDB_CRATE_DECISION.md`
- LMDB-resident interning (v2 target schema):
  `docs/design/WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`
- Roadmap context: `docs/WAM_TARGET_ROADMAP.md:106-114`
- R target encoding spec (TAB-encoded `tag:payload`, for v2
  reference): `docs/WAM_R_TARGET.md:340-426`
- C# memory-mapped array precedent (v3 reference):
  `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs:403, 606`
