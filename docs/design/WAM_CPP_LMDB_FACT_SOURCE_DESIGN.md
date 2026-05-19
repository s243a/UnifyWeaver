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

## Open questions

1. **Empty-DB semantics.** If the LMDB file exists but contains no
   rows for the configured predicate, should that succeed silently or
   surface a warning? C target: silent. Proposing: silent for v1,
   parity with C.
2. **Multiple predicates, one env.** `db_name` lets one env hold
   multiple named sub-DBs. v1 should support this (it's a one-liner)
   but I want to confirm the test fixture story before committing.
3. **Reload on re-call.** If `init_runtime` is called twice in the
   same process, do we re-open the LMDB env and reload? Or
   short-circuit on already-loaded? Proposing: idempotent — second
   call is a no-op. Document explicitly.
4. **liblmdb version floor.** LMDB has been stable since 0.9.x
   (~2011). v1 targets any 0.9.16+ (the version in Ubuntu 18.04).
   Reasonable?

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
