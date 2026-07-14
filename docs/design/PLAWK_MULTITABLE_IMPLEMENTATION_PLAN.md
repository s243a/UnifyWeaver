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

- **The multi-pass driver is hardwired to one shared table.** In
  `examples/plawk/codegen/plawk_native_codegen.pl`, every
  `plawk_program_multipass_driver_ir/2` clause discovers a single table and
  threads it as one parameter: the END-for-in driver asserts
  `AssocPlan = assoc_plan([ArrayName], _)` and calls
  `plawk_cache_entries([ArrayName], …)` / `plawk_multipass_table_params([ArrayName], …)`;
  the no-END and reader drivers make the same one-table assumption. A program
  with two `declare`s therefore fails the driver match and the build reports
  *"uses multi-pass features outside the current multi-pass surface"* — before
  any cache/storage code runs. **This is the first thing to fix.**
- **One store path ≈ one table (storage).** Even once the driver accepts N
  tables, the LMDB backend writes every table to the **unnamed** DB, so several
  tables sharing a `cache("store.lmdb")` path overwrite each other on commit
  (`PLAWK_MULTIPASS_CACHE.md` §3.7). Multi-table storage needs named sub-DBs.

Encouragingly, some plumbing is already list-shaped: `plawk_multipass_table_params/3`
takes a *list* of tables, and the row readers (`records of` / `rows of`)
already resolve their table by name via `nth0(TableIndex, Tables, Table)`. The
work is to stop hardcoding `[ArrayName]` and thread the full table set.

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

### PR 1 — Generalise the multi-pass driver to N in-memory tables

The foundational refactor; no durability yet. Stop hardcoding a single table:
collect every table referenced by the passes/readers/END into a `Tables` list;
create, thread (as N params), and free each; route each pass's writes/reads and
each reader to its table by index. The END-for-in still iterates the one table
its loop names; the others simply live alongside. Deliverable: a **purely
in-memory** program with two `declare`d row tables, each written and read back
in one run. Highest regression risk (touches the core driver) — run the full
plawk row/cache suite each iteration. No storage, no new syntax.

### PR 2 — Per-store table grouping + class-A compile error

Compile-time bookkeeping the storage step needs: group cache tables by store
path; when a store has ≥2 tables, mark it multi-table and assign each table its
sub-DB name (the table name). On the **file** backend, a multi-table store is a
clean compile error citing the class-A rule (replacing the generic driver
rejection). Small, low-risk, and it fixes the confusing error today.

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
