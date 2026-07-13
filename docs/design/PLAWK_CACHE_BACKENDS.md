<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk cache backends: the default-table rule

**Status**: specification (design). Defines, once and backend-independently,
**which physical table a plawk store operation targets when the program does
not name one** — the "default table" — and how named tables map onto each
backend. Companion to `PLAWK_MULTIPASS_CACHE.md` (§3.5 secondary indexes,
§3.6 row-oriented records, §3.7 table lifecycle, phases 8.7–8.9). Nothing here
changes behaviour on its own; it is the contract the `use` / multi-table work
(phases 8.8–8.9) must implement, and the yardstick for judging a backend's
fit.

## 1. Why this spec exists

plawk's surface has a single logical table per `declare NAME` today, and will
grow named/namespaced tables (`use ns.table`, §3.5/§3.7). Every backend,
however, has its own idea of what a "table" is: a flat file *is* one table; an
LMDB environment has an unnamed default database plus optional named ones;
SQLite has only named tables and no default; Redis has no table concept at
all. If each backend improvised its own mapping, a program would not port
across backends and the durability/`use` semantics would drift.

This spec fixes one rule — *what the default table is, and how names map* —
derived from a small **classification** of backends by their native table
model. The classification also tells us, up front, which backends fit cleanly
(and which never will), so the surface stays coherent as backends come and go.

The guiding principles, in priority order:

1. **Portability of single-table programs.** A program that declares one
   unnamed table MUST run on every supported backend, mapping to that
   backend's most natural single/primary table.
2. **Idiomatic mapping.** Use the backend's own default/primary table where it
   has one; do not invent a phantom named table when the backend offers a
   real default.
3. **Least surprise + safety.** Never silently place data where a later reader
   would not look; respect backend-specific hazards (e.g. LMDB's catalog, §4).
4. **Predictable degradation.** When a program needs a capability a backend
   lacks (e.g. multiple tables on a single-table container), that is a
   **compile-time error**, never a silent reinterpretation.

## 2. Model: container, table, default table

- A **store** is what a `BEGIN cache("path" [backend "…"])` names — one
  backend instance addressed by a path.
- A **container** is the physical unit the backend persists (a file, an LMDB
  environment, a SQLite database file, a Redis keyspace).
- A **table** is a plawk associative table: a key → (i64 | row) map. A
  container holds one or more tables depending on the backend.
- The **default table** is the table a store operation targets when the
  program does not name one (today: any `declare NAME`, since the name is a
  source-only label — see §3.7 of the cache doc).

## 3. Backend classification and the default-table rule

Every backend falls into one of four classes by its native table model. The
default-table rule and the named-table mapping are fixed per class.

### Class A — single-table container (the container *is* the table)

The container holds exactly one table; there is no place for a second.
Examples: our flat/byte-valued file; a plain TSV/CSV file.

- **Default table** = the container itself. A `declare NAME` maps here
  regardless of `NAME` (the name is a label only).
- **Named/multiple tables** = **unsupported → compile error.** A program that
  asks for two tables, or a namespaced `ns.table`, in a class-A store MUST be
  rejected at compile time (principle 4), not silently collapsed onto one
  file (which is the *current* flat-file bug the spec forbids going forward).

### Class B — default table + optional named tables

The container has a native **unnamed/default** table that always exists, plus
named sub-tables the program may opt into. Examples: LMDB (unnamed DB + named
sub-DBs), BerkeleyDB (unnamed database + named subdatabases).

- **Default table** = the backend's native **unnamed** table.
- **Named tables** = the backend's native named sub-tables, keyed by the
  plawk table name.
- **Mode is per store** (see §4): a store either uses its unnamed table as the
  one default table, **or** uses named sub-tables (and treats the unnamed one
  as the catalog) — it does not mix program data across both.

### Class C — all tables named, no default

Every table must be named; there is no native default. Example: SQLite
(`CREATE TABLE name …`; `main`/`temp` are database/schema names, not a default
table).

- **Default table** = a **reserved table name** the runtime uses when the
  program names none. This spec fixes that name as **`plawk_default`** (a
  single, documented reserved identifier, so a store written by one plawk
  program is found by another). Programs MUST NOT declare a user table called
  `plawk_default` on a class-C backend.
- **Named tables** = real named tables, one per plawk table name.

### Class D — no table concept (flat keyspace)

The container is a single flat keyspace with no tables. Example: Redis (a
numbered keyspace; hashes and key prefixes are the only grouping).

- **Default table** = the whole keyspace under a reserved key **prefix**,
  fixed as **`plawk:`** (so plawk data is namespaced away from unrelated keys
  in a shared keyspace).
- **Named tables** = a per-table key prefix (`plawk:<table>:`), or a native
  grouping primitive if one fits better (e.g. a Redis hash per table). The
  concrete choice is deferred to if/when a class-D backend is actually built;
  the *rule* (reserved prefix for the default, per-name prefix for named) is
  fixed here so the surface is decided.

### Summary table

| Class | Native model                         | Default table            | Named tables            |
|-------|--------------------------------------|--------------------------|-------------------------|
| A     | one table per container              | the container            | **compile error**       |
| B     | unnamed default + named sub-tables   | the unnamed table        | native named sub-tables |
| C     | all tables named, no default         | reserved `plawk_default` | real named tables       |
| D     | flat keyspace, no tables             | reserved prefix `plawk:` | per-table key prefix    |

## 4. Cross-cutting rules

- **Class-B unnamed-table hazard (LMDB / BerkeleyDB).** The *names* of named
  sub-databases are themselves stored as keys in the unnamed database. A
  store therefore MUST commit to one mode: **single-table** programs use the
  unnamed DB as the default table and never open named sub-DBs; **multi-table**
  programs use named sub-DBs and treat the unnamed DB purely as the catalog
  (no program rows in it). The mode is decided by whether the program declares
  a single unnamed table or one/more named tables — not chosen at runtime.
- **Schema is per table.** Each table carries its own row schema
  (`PLAWK_MULTIPASS_CACHE.md` §3.6/8.7). On class A the schema lives in the
  single container header; on B/C/D it is stored per named table (and for the
  default table, under the reserved name/prefix). Opening a table with a
  mismatched `declare(cols)` is a clean failure (8.7), on every backend.
- **Attaching without declaring (`use`, phase 8.8).** `use TABLE as r` takes
  the columns from the stored schema of the selected table — the default
  table when unnamed, the named table otherwise — so a reader need not repeat
  `declare(cols)`. This is well-defined on every class *because* the default
  table is well-defined here.
- **Missing default is create-on-commit.** Consistent with `declare` being
  open-or-create (§3.7): if the default table does not yet exist in the
  container, a read sees it empty and a commit creates it.

## 5. Backend inventory: supported, planned, out of scope

### Currently supported

- **`file` (flat / byte-valued) — Class A.** The store file *is* the one
  table (`[schema?][count][key value|key vallen valbytes]…`). Default table =
  the file. Multiple/named tables on one file path are therefore a compile
  error under this spec (the current code silently overwrites — a bug this
  spec marks for correction when named tables land). Durable i64 and rows
  (phases 3, 8.4).
- **`lmdb` — Class B, single-table mode only (today).** We open the **unnamed
  default DB** with `MDB_NOSUBDIR`. Default table = the unnamed DB, matching
  the class-B rule. Named sub-DBs (multi-table mode) are not built yet.
  Durable i64 today; durable rows over LMDB are a follow-on.

### Planned

- **`lmdb` named sub-DBs — Class B, multi-table mode (phase 8.9).**
  `mdb_env_set_maxdbs(N)` + `mdb_dbi_open(txn, "<table>", …)`; the plawk table
  name becomes the sub-DB name; the unnamed DB becomes the catalog (§4). This
  is the reference implementation of the class-B multi-table rule.

### Out of scope (kept as spec test cases, not roadmap)

These are **not planned**, but each exercises a different corner of the rule,
so they are the cases we check the spec against. If the rule gives a clean,
surprise-free answer for each, the rule is sound.

- **SQLite — Class C.** No default table exists natively. Tests that the
  reserved-name rule (`plawk_default`) is necessary and sufficient: a
  single-table plawk program maps to `CREATE TABLE plawk_default (…)`; named
  programs map to real tables; the reserved name must be excluded from user
  declarations. This is the case that *proves* class C needs to exist as a
  distinct class rather than being folded into B.
- **Redis — Class D.** No tables at all, plus numbered logical databases and a
  shared keyspace. Tests the reserved-prefix rule (`plawk:`) and forces the
  question of isolation in a keyspace shared with non-plawk data — which is
  why the default is a prefix, not the bare keyspace.
- **BerkeleyDB — Class B.** Almost identical to LMDB (unnamed default DB +
  named subdatabases). Tests that class B generalises beyond LMDB and that the
  unnamed-table hazard (§4) is a *class* property, not an LMDB quirk.
- **Plain TSV/CSV file — Class A.** A human-readable single-table container.
  Tests that class A is about *cardinality* (one table per container), not
  about our specific binary format — the same "named tables ⇒ compile error"
  rule applies.

## 6. Worked checks against the spec

- *Single-table histogram on `file`:* `declare c` → class A default = the
  file. ✔ (matches today.)
- *Single-table histogram on `lmdb`:* `declare c` → class B default = the
  unnamed DB. ✔ (matches today.)
- *Two tables on `file`:* `declare a; declare b` → class A ⇒ **compile error**
  (one container, one table). ✔ (spec-mandated; corrects the current silent
  overwrite.)
- *Two tables on `lmdb`:* → class B multi-table ⇒ named sub-DBs `a`, `b`;
  unnamed DB is the catalog. ✔ (phase 8.9.)
- *Single table on SQLite:* → class C ⇒ `CREATE TABLE plawk_default`. ✔ (rule
  supplies the missing default.)
- *Single table on Redis:* → class D ⇒ keys under `plawk:`. ✔ (rule supplies a
  namespaced default in a shared keyspace.)

## 7. Open questions

- **Class-C reserved name / class-D reserved prefix bikeshed.**
  `plawk_default` / `plawk:` are placeholders; if a class-C/D backend is ever
  built, confirm they don't collide with common conventions and allow an
  override in the `cache(...)` surface.
- **Mode inference vs. explicit declaration (class B).** Deciding
  single-vs-multi-table mode from "declares one unnamed table" is implicit; an
  explicit marker (e.g. `cache("x" as ns)` from §3.5 always meaning
  multi-table) may be clearer. Resolve when 8.9 lands.
- **Cross-backend store migration.** Not a goal; a store's on-disk layout is
  backend-specific. Noted so nobody assumes a `file` store opens as `lmdb`.
