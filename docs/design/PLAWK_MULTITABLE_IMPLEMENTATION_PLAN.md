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
- **One store path ≈ one table (storage).** Even once the driver accepts N
  tables, the LMDB backend writes every table to the **unnamed** DB, so several
  tables sharing a `cache("store.lmdb")` path overwrite each other on commit
  (`PLAWK_MULTIPASS_CACHE.md` §3.7). Multi-table storage needs named sub-DBs.

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

### PR 3 — LMDB named sub-DB storage routing

The storage payoff. Add sub-DB-aware commit/load to `wam_cache_lmdb.c`
(`…_lmdb_sub` / `…_lmdb_str_sub`, factored over a shared core taking a
`const char *subname`, NULL = unnamed): `mdb_env_set_maxdbs` + a named
`mdb_dbi_open` with `MDB_CREATE` on commit. Codegen routes a multi-table
store's tables to their sub-DB helpers (single-table stores keep the unnamed
path, so existing stores stay byte-compatible). Durable multi-table over LMDB,
both i64 and row (`_str`) values. Test mirrors the durable-rows suites with two
tables.

### PR 4 — `as ns` namespace + `ns.table` references (parser)

The name-resolution surface (`PLAWK_MULTIPASS_CACHE.md` §3.7): `cache("db" as
ns)` (and `cache("db") as ns`) declares a namespace; tables are referenced
`ns.table` from passes / `END`; a `global`-marked declaration lifts to the bare
namespace. Parser + name resolution only; lowers onto the PR-1..3 machinery
(the qualified name maps to a table + its sub-DB name). Bare-but-unique
multi-table (PRs 1–3) already works without this; `as ns` is collision-avoidance
sugar.

### PR 5 — `use ns.table` selection (resolver)

Extends the build-time `use` resolver (`examples/plawk/bin/plawk`,
`expand_cache_use`) so `use ns.table` attaches to one named sub-DB of an
existing multi-table store. The LMDB schema probe (`wam_cache_lmdb_schema.c`,
added for single-table `use` over LMDB) gains a sub-DB argument so it reads the
schema from the named sub-DB rather than the unnamed default. Depends on the
single-table `use`-over-LMDB work (PR #3727) and PR 3.

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
