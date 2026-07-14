<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk multi-table stores — implementation plan (phase 8.9)

**Status**: plan, not yet implemented. Sequences the work for *multiple named
tables per store* — the point at which `select` / `use` becomes load-bearing
rather than sugar (`PLAWK_MULTIPASS_CACHE.md` §3.7(c), §5 phase 8.9). Grounded
in the concrete single-table assumptions in the current codegen, so the arc is
broken into small, independently-reviewable PRs rather than one large change.

## 1. Where we are (the blocking facts)

Two independent single-table assumptions must fall, and they are *not* in the
order the surface suggests. Storage routing is **not** the first obstacle — the
driver rejects multi-table programs before storage is ever reached.

- **The multi-pass driver was hardwired to one shared table — FIXED (PR 1).**
  The no-END reader driver in
  `examples/plawk/codegen/plawk_native_codegen.pl` now threads N tables (see PR
  1 below); a program with several `declare`s / row tables builds and runs
  in-memory. The END-for-in driver still assumes its single END-loop table,
  which is fine — a multi-table program takes the reader driver. (Before PR 1 a
  two-`declare` program failed the driver match and the build reported
  *"uses multi-pass features outside the current multi-pass surface"*, before
  any cache/storage code ran.)
- **One store path ≈ one table (storage) — FIXED (PR 3).** A multi-table LMDB
  store now routes each table to its own named sub-DB (`mdb_env_set_maxdbs` +
  named `mdb_dbi_open`), so tables sharing a `cache("store.lmdb")` path are
  isolated and durable. A multi-table *file* store stays a compile error (class
  A, PR 2).

Much of the plumbing was already list-shaped, which is why PR 1 was small: the
setup/free helpers and `plawk_cache_entries/6` iterate the table list, and the
row readers (`records of` / `rows of`) resolve their table by name via
`nth0(TableIndex, Tables, Table)`. PR 1 only had to drop the artificial
single-table cap and generalise `plawk_multipass_table_params/3` (which had had
clauses only for `[]` and `[_Table]`) to N tables.

## 2. The backend rule (fixed, from PLAWK_CACHE_BACKENDS.md)

- **Class A (file)** — single-table container. Multiple/named tables are an
  **unsupported → compile error**, by spec. So multi-table programs require an
  LMDB (class B) store; the file backend gets a *clean, specific* diagnostic.
- **Class B (LMDB)** — a store is **either** the unnamed default DB (one table)
  **or** named sub-DBs (multi-table), decided at **compile time**, never at
  runtime. Multi-table mode: `mdb_env_set_maxdbs(N)` + `mdb_dbi_open(txn,
  "<table>", MDB_CREATE, …)`; the plawk table name becomes the sub-DB name; the
  unnamed DB becomes the catalog and stores no plawk data.

## 3. PR sequence

Each step is its own PR, green regressions required before the next.

### PR 1 — Generalise the multi-pass driver to N in-memory tables — **LANDED**

The foundational refactor; no durability yet. Much of the no-END reader driver
was already list-shaped (`plawk_passes_tables/2` collected and `sort/2`-ed every
referenced table; the setup/free helpers and `plawk_cache_entries/6` iterate the
list; the row-reader pass-fns resolve their table by `nth0`). Only two spots
hardcoded a single table: an artificial `N =< 1` cap in `plawk_passes_tables/2`,
and `plawk_multipass_table_params/3` (clauses only for `[]` and `[_Table]`,
emitting one `%plawk_assoc_table_0`). Dropping the cap and generalising the
params to one `%plawk_assoc_table_<i>` per table — indexed by position in the
sorted list, so setup / params / per-pass planning stay consistent — is the
whole change. Every pass function now takes the full table set; a rule or reader
references its table by index. `tests/test_plawk_multitable.pl`: two bare
(schema-less) row tables with no cache; a schema'd `records of` table alongside
a bare positional table; a mixed i64-counter + row program; three row tables.
The full plawk row/cache suite stays green (N=1 lowers exactly as before — the
single-table clause is just the N=1 case of the general one). No storage, no new
syntax.

### PR 2 — Per-store table grouping + class-A compile error — **LANDED**

`check_multitable_store/2` in `examples/plawk/bin/plawk` groups declared cache
tables by store path; when a path carries ≥2 distinct table names it applies the
backend rule (`PLAWK_CACHE_BACKENDS.md`): a **file** store (class A) is
single-table — a permanent compile error pointing at `backend "lmdb"`; an
**lmdb** store is a clear *"not yet"* error (its named-sub-DB routing is PR 3)
rather than a silent overwrite of every table into the one unnamed DB. In-memory
tables (no `cache_table` entry) are untouched, so multiple bare tables, or one
backed table plus bare ones, still build. `tests/test_plawk_multitable_store.pl`
covers all four cases. (Note: this replaced the PR-1 test that had put two
schema'd tables in one file store — now correctly an error — with a
named-plus-bare mix.) Small and low-risk; it also fixes the confusing generic
rejection.

### PR 3 — LMDB named sub-DB storage routing — **LANDED**

The storage payoff. `wam_cache_lmdb.c` was refactored so each op has a core
taking `const char *subname` (NULL = unnamed default DB); a shared
`wam_cache_lmdb_open` sets `mdb_env_set_maxdbs` and opens the named
`mdb_dbi_open` (`MDB_CREATE` on commit) when a subname is given. Public
`…_lmdb_sub` / `…_lmdb_str_sub` entry points expose the named-sub-DB form; the
existing `…_lmdb` / `…_lmdb_str` are the `subname = NULL` case, so single-table
stores are byte-identical. The schema is stored under its distinguished key
**inside each named sub-DB**, so every table is independently self-describing
and validated. Codegen threads a `SubDb` field (`none` | `subdb(Ref)`) through
the cache entries: `plawk_multitable_paths/2` marks paths with ≥2 tables, each
such table gets a sub-DB-name global (the table name) and routes to the `_sub`
helpers; `plawk_cache_fn/5`, `plawk_cache_call_ir/7`, and the decls gained the
`SubDb` dimension. The bin/plawk gate now allows lmdb multi-table (only file
stays an error). Tests: `tests/test_plawk_multitable_lmdb.pl` (two row tables
durable across runs in separate sub-DBs; two i64 counter tables accumulating
independently; per-sub-DB schema-mismatch → exit 3). The PR-2
`lmdb_multitable_is_not_yet_error` test flips to `lmdb_multitable_builds_and_runs`.
Durable multi-table over LMDB, both i64 and row values.

### PR 4 — `as ns` namespace + `ns.table` references (parser) — **LANDED**

The name-resolution surface (`PLAWK_MULTIPASS_CACHE.md` §3.7). `cache("db" as
ns) { declare orders … }` qualifies each declared table to the dotted atom
`ns.orders` (parse-time, `plawk_qualify/3`); a `table_ident//1` rule parses
`ns.table` references to the same atom in every table-name position (the three
row readers, `over`, and the assoc write/inc/add + print-lookup targets). So the
qualified name flows uniformly as an ordinary table name, and the codegen only
needed two small changes: the sub-DB name is now the **local part** (after the
dot, `plawk_local_table_name/2`), and `plawk_multitable_paths/2` treats a
namespaced store as multi-table even with one table (so `as ns` really uses
sub-DBs). A namespaced *file* store is a compile error (class A can't hold named
sub-tables), alongside the ≥2-table file error. `tests/test_plawk_namespace.pl`:
parse (qualified declare, namespaced write, bare unchanged); a two-table
namespaced lmdb store durable across runs; a single-table namespace durable;
namespaced-file → exit 2. Bare-but-unique multi-table (PRs 1–3) still works
without a namespace; `as ns` is the collision-avoidance sugar. **Deferred**: the
`cache("db") as ns` (alias after the paren) spelling, the `global` modifier, and
namespaced tables in an `END` for-in — noted follow-ons, none needed for the
core surface.

### PR 5 — `use ns.table` selection (resolver) — **LANDED**

The finale: attach to a multi-table store with **no `declare`**, taking each
table's schema from its sub-DB at build time. This PR is self-contained (it
subsumes the never-merged single-table `use`-over-LMDB PR #3727): it ships the
LMDB schema probe (`wam_cache_lmdb_schema.c`) with an **optional sub-DB
argument** — given a sub-DB name it opens that named DB (`mdb_env_set_maxdbs` +
named `mdb_dbi_open`) and reads its `__wam_schema__` key, else the unnamed
default DB. The resolver (`examples/plawk/bin/plawk`) computes the store's
multi-table paths exactly as the codegen does (≥2 tables, or any namespaced
`ns.table`), and for a `use` on such a path passes the table's LOCAL name to the
probe; single-table `use` reads the unnamed DB (file backend unchanged, reads
its header directly). So `BEGIN cache("db" backend "lmdb" as ns) { use orders;
use items }` attaches both tables by reading each sub-DB's schema — the
"avoid-declare" surface for multi-table stores. Tests:
`tests/test_plawk_use_table_lmdb.pl` (single-table `use` over LMDB, from #3727)
and `tests/test_plawk_use_namespace.pl` (namespaced multi-table `use` with
**three-column** tables, including column arithmetic over the schema-read
fields). Depends on PR 3's per-sub-DB schema storage.

---

**Arc complete.** All five PRs have landed: multi-table stores work end to end —
the driver takes N tables, an lmdb store routes them to named sub-DBs, `as ns`
namespaces the references, and `use ns.table` attaches by reading each sub-DB's
schema. A multi-table *file* store is a clean class-A compile error throughout.
Deferred sugar (noted in PR 4): the `cache("db") as ns` spelling, the `global`
modifier, and namespaced tables in an `END` for-in.

## 4. Risks / decisions

- **Regression surface (PR 1).** The single-table driver underlies every
  existing multi-pass + cache test. The refactor must keep the one-table path
  byte-identical; the safest shape is "N tables where N=1 lowers exactly as
  today." Gate on the full suite.
- **maxdbs sizing (PR 3).** `mdb_env_set_maxdbs` is a fixed ceiling set before
  env open; start with a generous constant (e.g. 128) and document it as a
  later knob, matching the existing 1 GiB mapsize choice.
- **Mode is per store, compile-time (PR 2/3).** Never open both the unnamed DB
  and named sub-DBs for data in one store — the class-B unnamed-table hazard
  (sub-DB names live as keys in the unnamed DB). Single table → unnamed; ≥2 →
  named; decided once at compile time.
- **Schema location (PR 3/5).** Per-table schema moves from the store-wide key
  to a per-sub-DB key, so each named table is independently self-describing.
