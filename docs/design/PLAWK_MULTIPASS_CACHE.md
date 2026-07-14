<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk multi-pass processing over a persistent cache

**Status**: partially implemented. Captures the determinism rationale, the
surface, the runtime ABI, and a phased rollout.

**Landed:**

- **Persistent cache** — `BEGIN cache("path") { declare NAME }`, file backend
  (phase 1) and LMDB backend (`backend "lmdb"`, phase 5, eager).
- **Multi-pass execution** — `pass { }` blocks over a shared assoc table
  (phase 2), including pure-scalar (no-table) programs.
- **Per-record output** in assoc programs (`print $N` / `print arr[$N]`) — the
  "normalise" shape, where pass 2 prints each record from pass 1's table.
- **Cross-pass scalars** (`acc += 1` / `acc += $N`, module-global accumulators).
- **Print arithmetic in f64** (`$2 / total`) → **grand-total normalise**; and
  **per-key aggregation** (`total[$1] += $2` + `$2 / total[$1]`) → **per-key
  normalise**.
- **Cache as the inter-pass channel** (phase 3) — a declared table loaded
  before pass 1, committed after the last pass; durable across runs.
- **`over TABLE` reader** (phase 4) — iterate a table's entries as the pass's
  record source.
- **Row-oriented tables** (phase 8, §3.6): a table whose value is a
  named-field **row**. Schema (`declare NAME(col type, …)`); writers
  `TABLE[$k] = $0` and `TABLE[$k] = row($a, $b)`; readers `records of TABLE
  as r` (`r["col"]`, by schema name), `rows of TABLE as r` (`r[N]`, by
  position), and `rows of TABLE` (no `as` → awk-native `$N`); column
  arithmetic in f64 (`r["amt"] / 2`); durable rows on the file **and LMDB**
  backends (row bytes persisted, phase 8.4); self-describing store — schema persisted and
  validated on open (phase 8.7); and `use NAME` to attach to an existing
  store without re-`declare` (phase 8.8).
- **Reader guards** (a `WHERE`-style row filter): any of the three row readers
  may wrap its `print` in `if (COND) …`, where `COND` compares a column to a
  literal — `r["col"] CMP L` (records), `r[N] CMP L` (positional),
  or `$N CMP L` (anon). The literal is an **integer** (i64 compare, six
  operators `== != < <= > >=`), a **decimal float** (`3.5`, `-1.5`; the column
  is read as a double and compared with `fcmp`, so fractional thresholds and
  values work), or a **string** (`"alice"`, `==` / `!=` only — a
  length-then-`memcmp` byte comparison). Comparisons combine with **`&&` / `||`**
  (short-circuit, `&&` tighter than `||`, left-associative, parens allowed),
  e.g. `if (r["amt"] > 100 && r["cust"] == "alice")`. Filtered rows never reach
  the print block (`tests/test_plawk_reader_guards.pl`,
  `tests/test_plawk_guard_float.pl`, `tests/test_plawk_guard_string.pl`,
  `tests/test_plawk_guard_bool.pl`).

**Not yet:** `rows of`'s `unsafe` / inline check-or-rename spec;
the `over prev` reader (phase 4
follow-on); the query reader (phase 6); `eager` / secondary indexes;
string-literal print fields. Future sketches: nested pass blocks (§3.8),
`while`/loop statements (§3.8), views (§3.9), contained non-determinism / a
search construct (§3.10). See the per-phase status tags in §5.

## Implemented surface (quick reference)

```awk
# persistent cache (phase 1/5): counts survive across runs of the binary
BEGIN cache("hist.db") { declare c }                 # file backend (default)
BEGIN cache("hist.lmdb" backend "lmdb") { declare c } # LMDB backend
{ c[$1]++ }
END { for (k in c) print k, c[k] }

# multi-pass (phase 2): read the input once per pass into a shared table
pass { c[$1]++ }
pass { c[$1]++ }        # input re-read; counts accumulate across passes
END { for (k in c) print k, c[k] }
```

```awk
# the "normalise" shape now works: pass 2 prints each record using the
# table pass 1 built (per-record output in an assoc program)
pass { c[$1]++ }
pass { print $1, c[$1] }     # over `a a b`: a 2 / a 2 / b 1
```

Multi-pass v1 lowers each `pass` to its own function (fixed per-record SSA
`%line` / loop labels are then function-local and cannot collide), creates
the shared assoc table once in `main`, threads it to each pass as a
parameter, and (for an END for-in) reads it back. **Per-record output** in
an assoc program is supported — a `print` in the record loop with a text
field `$N` or a table lookup `arr[$N]` — which is what makes pass-2 emit one
line per record from pass-1's table. It requires a **file** argument (each
pass re-opens it; stdin is not re-openable), a single shared table, and
always-rule pass bodies. A **cache-backed** shared table (`BEGIN
cache("path") { declare NAME }`) now works with multi-pass too (phase 3):
loaded before pass 1, committed after the last pass, durable across runs.
String-literal print fields, multiple tables, and reader selection are
follow-ons.

## TL;DR

awk is single-pass: one implicit main block runs once per record, then
`END`. That is the right shape when every record can be handled with only
the state accumulated so far. It is the wrong shape when a record's
handling depends on a *global* fact that is only known after the whole
input has been seen (a total, a max, a second key's value, a join).

The classic awk answer is `awk '...' file file` (name the file twice) or
`awk | sort | awk`. Both re-read and, in the pipeline case,
**re-serialize** the data at a text boundary — and in our world they also
**throw away every loaded grammar object and compiled foreign predicate**,
which must then be re-registered (at best) or recompiled (at worst) in the
second process.

This design keeps multiple passes **inside one native binary**: several
ordered *interaction blocks* (passes), a **persistent cache** (an
LMDB-backed key/value store) as the explicit channel between them, and —
the payoff — **loaded `.wamo` objects and compiled predicates persist
across all passes** because it is one process, one VM, one set of shims.

```awk
BEGIN cache("run.lmdb") { declare total }

pass { total[$1] += $2 }              # pass 1: accumulate into the store

pass { print $1, $2 / total[$1] }     # pass 2: normalise each record by the
                                       #         now-complete per-key total

END   { }
```

## 1. The determinism model (the reminder, and why it constrains this)

`PLAWK_PHILOSOPHY.md §2` states the stance precisely; the operative facts:

- **A plawk program is a deterministic Prolog predicate** compiled through
  the hybrid WAM/LLVM target. "Familiar surface, principled interior."
- **The per-record body (`{ }`) is deterministic by default.** Handlers are
  written to succeed exactly once and leave **no choicepoints**. Modes
  (`+`/`-`) flow through `demand_analysis` / `binding_state_analysis` and
  select deterministic builtin variants and input-indexed dispatch;
  structural determinism (cut, if-then-else, switch indexing, `musttail`)
  does the rest. A `:- det` *directive* is not consumed yet — determinism
  is achieved structurally.
- **This determinism is the performance story, not a nicety.** Streaming
  runs in *constant memory* because per-record state lives in native SSA
  slots (i64/double loop phis) and a fixed record buffer — not on a managed
  heap. A nondeterministic per-record body would force the engine to retain
  choicepoints (and the trail/heap they pin) *across records*, converting
  the constant-memory streaming loop into something that grows with the
  input. `PLAWK_PHILOSOPHY.md` calls out "residual choicepoints cost real
  time" as the gap to hand-tuned Rust.

**So the invariant we must not break:** *within the iteration brackets, work
is deterministic.* Backtracking is "contained and annotated" (the Haskell
`IO`-monad analogy), never the default in the hot loop. This is the general
**bounded-multiplicity** principle — contain non-determinism by collapsing it
at a boundary, the same shape as SQL `GROUP BY`, Prolog `findall`, and list
comprehensions; stated once in `UNIFYWEAVER_LANGUAGE_PRINCIPLES.md` (Principle
1) and applied throughout this doc.

### 1.1 Cross-iteration information under that invariant

Two needs, two horizons:

- **Within one pass** (already solved): carry state in **scalar slots**
  (loop-carried phis — `sum += $1`) and **assoc tables** (the in-memory
  i64 hash — `arr[k]++`, iterated by `for (k in arr)`). Both are
  deterministic accumulators threaded through the record loop; both are
  bounded by RAM and consumed at `END`.
- **Across passes** (the gap): a record in pass 2 needs a value that is only
  correct once pass 1 has seen *every* record (a grand total, a global max,
  the other side of a join). Today the only "after all records" hook is
  `END` — a single terminal block, not a second streaming pass — and the
  only cross-record store is the in-memory assoc table, which is lost at
  exit and capped by RAM.

The user framed the alternatives well:

1. **Save-for-later via a closure.** Defer a computation by closing over the
   value needed later. This is expressive but keeps the deferred state in
   the managed heap and does not, by itself, give a *second streaming pass*.
   It is the right tool for "compute this once, reuse it many times within a
   pass," not for "re-scan the input knowing a global fact."
2. **A loop in `BEGIN`.** `BEGIN` is for one-time setup (open handles, set
   `FS`/`CACHE`), not for a data pass — putting the main loop there fights
   the model and loses the per-record pattern/action ergonomics.
3. **Multiple interaction blocks (this design).** Make the second pass a
   first-class construct, and give it an explicit, durable channel to read
   what the first pass computed.

## 2. Why multi-pass-in-one-binary beats `awk | awk`

The decisive advantage — and the reason this belongs *in the engine*
rather than as a shell pipeline — is **artifact persistence across
passes**:

- A plawk program can load grammar objects (`DYNLOAD = "g.wamo"`), resolve
  named entries, cache their PCs, and JIT-compile grammars from source
  (`compile(...)`). All of that lives in **one VM inside one process**.
- With a shell pipeline, the second `awk`/binary starts cold: every
  `.wamo` must be re-opened and re-resolved, every `compile(...)` re-run,
  every foreign predicate re-registered. At a text `|` boundary the data is
  also **re-serialized** — the exact cost `PLAWK_PHILOSOPHY.md §1.1` names
  as *the* differentiator ("inter-stage serialization").
- Multi-pass keeps the loaded objects, the resolved entries, and the record
  layout **hot** between passes. Pass 2's `dyncall@decode(...)` hits the
  same already-resolved PC pass 1 used. Nothing is re-registered or
  recompiled.

The persistent cache is what carries *data* across the pass boundary; the
single-process model is what carries *code/artifacts* across it for free.

## 3. Surface

### 3.1 Passes

A program is `BEGIN`, then **one or more `pass { }` blocks in order**, then
`END`. A bare `{ }` (today's main) is sugar for a single `pass { }`, so
existing programs are unchanged.

```awk
BEGIN { ... }        # once, before pass 1
pass { ... }         # pass 1, per record
pass { ... }         # pass 2, per record
...
END   { ... }        # once, after the last pass
```

- Each pass runs the **same deterministic per-record contract** as today's
  main. Determinism is per-pass; nothing about multi-pass relaxes §1.
- **Loaded objects / compiled predicates persist across passes** (§2).

### 3.2 Variable scoping, and stores as backed `BEGIN` blocks

Two orthogonal properties — a scoping property (does a variable survive
into the next pass?) and a storage property (is it durable / shareable?) —
are expressed through **where** a variable is declared:

- **Pass-local by default.** A variable first used inside a `pass { }` is
  local to that pass and reset at the start of each pass. Its in-loop state
  lives in native SSA slots, as today.
- **`BEGIN`-declared variables persist across passes** (they are
  program-global). This is the in-memory cross-pass channel — fast,
  process-lifetime, RAM-bound. `sum` or an assoc `arr` introduced in a plain
  `BEGIN` carries its accumulated value from one pass into the next with no
  DB involved.
- **Backing is declared at the `BEGIN`-block level.** awk already permits
  multiple `BEGIN` blocks (they run in source order); here a `BEGIN` block
  optionally carries a **store**, and every variable declared in that block
  is durably backed by that store. A backed block is, in effect, "open this
  DB and bind these variables to it":

```awk
BEGIN { FS = "," }                       # plain: setup only, no backing

BEGIN cache("users.lmdb") {              # backed, NO alias -> global names:
    declare seen                          # seen / total are durable and
    declare total                         # referenced bare everywhere
}
BEGIN cache("stats.lmdb" as stats) {     # backed, aliased -> a NAMESPACE:
    declare work                          # tables are stats.work / stats.hist
    declare hist
    global runs                           # ...except `runs`, lifted to global
}

pass          { seen[$1]++ ; total += $2 }        # bare globals -> users.lmdb
pass          { stats.work[$1]++ ; runs++ }       # qualified table; bare global
END           { for (k in stats.hist) print k, stats.hist[k] }
```

The store name **is** the cross-process coordination unit: two programs
that open a `cache("shared.lmdb")` block with the same tables are, by
construction, sharing that store. This replaces the earlier per-variable
`as cache(...)` annotation — group by block, which is both DRY and
awk-idiomatic.

| declaration site | survives passes? | durable / cross-process? |
|---|---|---|
| used inside a `pass { }` | no (pass-local) | no |
| plain `BEGIN { … }` | yes (in-memory) | no |
| `BEGIN cache("db") { … }` | yes | yes — bound to `db` |

**Namespacing is opt-in via `as`.** Backing (durability) and namespacing
(name resolution) are separate axes:

- **No alias → global names.** A backed block without `as` contributes its
  tables to the **global** namespace — referenced bare everywhere, exactly
  as awk variables are today (just durable). Global names must be unique
  program-wide.
- **`as ns` → a namespace.** `cache("db" as ns)` makes `ns` a namespace; its
  tables are referenced **`ns.table`** from passes / `END` / other blocks
  (the module model — this is what lets a store hold many tables without
  name collisions, each a primary-key-indexed sub-database of the one
  environment). The alias may sit inside `cache(... as ns)` or after it
  (`cache("db") as ns`); both parse.
- **`global` escapes the prefix.** A declaration marked `global` inside a
  namespaced block is lifted to the global namespace — usable bare, no
  `ns.` prefix. So you namespace the bulk and hand-pick the few hot tables
  to use bare.

**Rules.**

1. **One store ↔ one backing block** within a program (two blocks on the
   same DB would silently share a keyspace). Cross-*process* sharing is
   fine — different programs each open the same store.
2. **Load-on-open.** Entering a backed block opens the store and binds its
   tables to the live contents, so a resumed run or a peer process sees
   what is already there.
3. **Materialisation is lazy by default** (the `WAM_LMDB_LAZY_*` axis):
   values are fetched from the store on access, not slurped at open — the
   larger-than-RAM story. A specific table can be marked **`eager`** to load
   fully at open when it is small and hot (`declare total eager`).

A backed `BEGIN` block may still contain setup statements (initialise
defaults for its own tables); it is not restricted to declarations.

### 3.3 The cache as an associative array

A table declared in a backed `BEGIN` block reads/writes through its store
instead of the in-memory hash, reusing the assoc-array ergonomics users
already know:

```awk
BEGIN cache("run.lmdb") { declare total }

pass { total[$1] += $2 }              # persistent accumulate (routes to run.lmdb)
pass { print $1, $2 / total[$1] }     # persistent read (pass 1 committed)
```

- Keys are interned atoms (text keys — the same keyspace field slices
  intern into) or i64 / blob bytes; values are i64 or a serialized
  record/blob. This mirrors the in-memory assoc table (`i64 -> i64`), so
  `arr[k]++`, `arr[k] = v`, `for (k in arr)`, and the stage-1/2/3 for-in
  bodies all carry over to a cache-backed table with the same codegen
  shapes — only the underlying get/put/iterate primitives change.
- **A commit barrier runs between passes.** Pass N reads the *committed*
  state of passes `< N`; it never observes a partial mid-pass write from
  its own pass unless it wrote the key itself earlier in the same record.
  This is what preserves "deterministic within the iteration brackets":
  the cache a pass reads is a fixed snapshot for the duration of that pass
  (except its own writes), not a racing target.

### 3.4 The reader for each pass

awk hardcodes one reader: records from the input stream. `PLAWK_PHILOSOPHY
§3` already abstracts this into a decoupled **Reader** role ("how to obtain
the next item"). Multi-pass makes that role **selectable per pass**, so a
pass is not forced to re-scan the original input:

```awk
pass { ... }                          # default: re-scan the original input
pass over total as k { print k, total[k] }  # iterate a table (LANDED)
pass over prev { ... }                # consume the previous pass's records
pass over query(Goal) { ... }         # each solution of a query is a record
```

- **`over input`** (the default) — re-open and re-scan the program's input.
- **`over TABLE`** (LANDED) — iterate a table's entries as records (the
  `for (k in arr)` iteration, but as the pass's record source), written
  `pass over TABLE as VAR { ... }`. Fields are **named** (name lookup): the
  loop key binds to `VAR`, its value reads as `TABLE[VAR]` — no positional
  `$1`/`$2`. The natural "process what the previous pass stored" shape,
  emitting one line per distinct key.
- **`over prev`** — the previous pass's **sink becomes this pass's source**.
  When pass N declares `over prev`, pass N−1's `print`/`writebin` no longer
  go to stdout; they **spool** (to a cache table or a temp stream) and pass
  N scans that spool after the barrier. This is the `awk | awk` pipeline
  semantics **in-process** — no text `|`, no re-serialization, artifacts
  stay hot — and, crucially, it does **not** need coroutines or break the
  "finish pass N−1, then start pass N" barrier: `over prev` is sugar over
  `over <spool>`, so it reuses the cache/spool machinery rather than adding
  a new mechanism. The final pass (no successor consuming it) prints to
  stdout as usual.
- **`over query(Goal)`** — drive the pass from the solutions of a Prolog
  query (riding the `call/1` meta-call + dynamic-DB machinery the JIT arc
  built): each solution is a record. This is the most "beyond awk" reader
  and the most advanced; staged last.

Writers mirror readers (a pass emits to stdout by default, or to the spool
that feeds the next `over prev` pass, or `writebin` to a file). Reader +
writer per pass is exactly the Reader/Writer decoupling the philosophy doc
describes, now instantiated per stage of an in-process pipeline.

### 3.5 Secondary indexes (deferred)

A table is primary-key indexed (the assoc key is the store key). A
**secondary index** lets a table be looked up by a non-key field, declared
alongside the table:

```awk
BEGIN cache("sales.lmdb" as sales) {
    declare orders                    # records keyed by order id (primary key)
    index orders by cust              # non-unique secondary index
    index orders by sku unique        # unique secondary index
}
```

Indexing a field requires the table's values to be **records with named
fields** (the record-value encoding tracked in §6), so the indexed field is
addressable.

**Unique vs non-unique — and why aggregation is mandatory.** A unique
secondary key identifies at most one record, so a lookup is a deterministic
single result. A non-unique key matches a *set* of records — inherently
multi-valued, and therefore exactly the nondeterminism §1 insists be
"contained and annotated." So a non-unique lookup **must be consumed by an
aggregation** that collapses the set to a deterministic value or a
deterministically-ordered array; using it raw (as a scalar) is a
compile-time error. This is not just ergonomics — it is what keeps the
multi-valued lookup from leaking a choicepoint into the hot loop. It is the
**bounded-multiplicity** principle in its SQL `GROUP BY` form — a set collapsed
by an aggregate at a boundary (`UNIFYWEAVER_LANGUAGE_PRINCIPLES.md`, Principle
1).

```awk
# unique index: one record
(cust, amt) = orders[sku: s]

# non-unique index: MUST aggregate
for (o in collect(orders[cust: c])) { ... }   # collect -> array, then iterate
n   = count(orders[cust: c])                    # how many match
tot = sum(orders[cust: c], amount)              # fold a field over the matches
```

`collect` yields the matches in a defined order (primary-key order), so the
`for`/`foreach` over the resulting array is deterministic — the same move
`foreach` / `for (k in arr)` already make. `count` / `sum` / `min` / `max` /
`avg` fold to a scalar directly.

**Store mapping.** A unique index is a plain sub-DB (secondary value →
primary key); a non-unique index is an `MDB_DUPSORT` sub-DB (secondary value
→ the set of primary keys), so `count` uses the cursor's dup count and
`collect` walks the dups. Indexes are maintained on write — a `put` also
updates the table's index sub-DBs (a write-amplification cost to keep in
mind). Deferred to a phase past v1; captured here so the surface is coherent
when it lands.

### 3.6 Row-oriented records (record-valued tables)

Everything above keys a table to a single `i64` value. The next step is a
table whose value is a **row** — a record with several named columns — so a
lookup returns the row and a program addresses a field by **column name**,
exactly as if the row were an associative array:

```awk
BEGIN cache("orders.db") { declare orders(cust str, amount i64) }   # schema

pass records of orders as r {          # iterate rows; r is a named-field row
    print r["cust"], r["amount"]       # field = row/column intersection
}
```

This is the "retrieve by key, get the row back as an assoc array keyed by
column name" shape the surface has been building toward — it is the
concrete meaning of the "record-valued tables" that §3.5 (secondary indexes)
and Phase 7 already depend on.

**Storage model (decided): a row-blob per key.** Each primary key maps to
one serialized record; a field read decodes that column from the blob using
the declared schema. One store entry per row keeps it aligned with the
durable / LMDB / larger-than-RAM direction (a row is one key→value pair, not
N fanned-out tables) and reuses the existing record (de)serialization
(`@wam_object_call_record` typecodes, the decode-into-struct machinery from
the for-in stage-3 work) rather than adding a parallel encoding. The rejected
alternative — one assoc table per column sharing a row-id key space — is
less new runtime but costs N store round-trips and N commits per row and
fights the single-entry-per-row durable model.

**The schema lives with the table.** `declare NAME(col type, …)` extends the
backed-`BEGIN` declaration with a column list (names + types; `str` / `i64`
to start). The schema is the single source of truth used to (de)serialize
and to resolve a column name to its slot. A bare `declare NAME` (no column
list) stays exactly today's `i64`-valued table.

**Two readers — a safe named one and a positional one.** These mirror the
Reader role (§3.4) but for row values, and they have deliberately distinct
contracts so neither is a grab-bag of modifiers:

- **`records of TABLE as r` — safe, named, schema-required. LANDED.**
  Columns are addressed **by name only**: `r["cust"]`. Requires a declared
  schema (that is what supplies the names and the decode layout — `r["col"]`
  resolves to the column's position and extracts that field of the stored
  row). This is the recommended, everyday form. Numeric addressing is not
  accepted here (that is the positional `rows of` reader's job, below).

- **`rows of TABLE as r` — positional. Core LANDED.** Columns are addressed
  **by position** (`r[1]`, `r[2]`, 1-indexed = field N of the stored row),
  for raw or ad-hoc stores; **no schema required**
  (`tests/test_plawk_rows_reader.pl`). Reuses the same row-decode function as
  `records of`, sourcing field indices from the literal positions. The
  `unsafe` modifier and inline check-or-rename spec below are a **follow-on**.
  - **`rows of TABLE` — no `as`, awk-native `$N`. LANDED.** Omitting the
    `as VAR` binding switches to awk field addressing: `pass rows of t {
    print $1, $2 }` — a stored row is a field-separated record, so `$N`
    addresses its Nth column (and arithmetic over `$N` works, in f64). The
    two spellings do not mix: no `as` ⇒ `$N`; `as VAR` ⇒ `VAR[N]`. This makes
    the schemaless positional reader read like plain awk over the stored rows.
  A schema spec may appear at the read site, and its role depends on whether
  an authoritative (`declare`d) schema already exists:
  - **schema present →** the spec is a **check**: the store's row shape
    (arity / types / names) must match, diagnosed at open, not as silent
    garbage later.
  - **no schema →** the reader must be marked **`unsafe`**, and the spec then
    **defines/renames** the positional columns into names (so `r["a"]`
    becomes usable even though the store carried no declared schema). The
    same spec syntax thus *checks* in the safe case and *names* in the
    unsafe case. `rows of` with named access but neither a declared schema
    nor `unsafe` is an error (you are naming columns you never defined).

  `unsafe` means precisely "I have no trusted schema — trust mine / skip the
  check"; it is confined to `rows of`. `records of` never takes `unsafe` and
  never addresses columns positionally.

**Field access.** `r["col"]` (string-literal column key) for named access;
`r[N]` (integer) for positional access under `rows of`. The `arr["str"]` /
`arr[N]` key surface already exists; what is new is that `r` resolves against
the row schema / positional slots rather than being a separate table.

**Column arithmetic (LANDED).** A print field in either reader may be an
arithmetic expression over columns and integer constants, e.g.
`r["amount"] * 2`, `r["amount"] / 4`, `r[2] / r[3]`. It is evaluated in **f64**
and printed with `%g` (the surface `/` is integer, so a bare column ratio
would truncate) — the same print-arithmetic path grand-total normalise uses,
with a column operand read via `@wam_atom_field_f64_value` on the row.
Tests: `tests/test_plawk_records_reader.pl`, `tests/test_plawk_rows_reader.pl`.

**Write side.** How a pass *builds* a row touches multi-column value
assembly and the commit path. Two producers have **LANDED**, both riding the
existing str-value assoc mechanism (the i64 value is the row's atom id), no
new storage runtime: `TABLE[$k] = $0` captures the whole current record
(the "capture / index / dedup by a key" case), and `TABLE[$k] = row($a, $b,
…)` builds a row from chosen input fields, in that order (project / reorder),
joined by the field separator so a reader recovers the columns. Both are
replace semantics, keyed by field k. A later pass reads the row back (`over
TABLE as k`, or `records of` / `rows of`). Field-wise writes
(`orders[$1]["amount"] = $2`) remain a follow-on.

**In-run and durable.** A row value is an atom **id** (process-local), so the
plain i64 cache — which persists the id — cannot restore a row in a fresh
process. A str-valued (row) table therefore commits the value **bytes** and
re-interns them on load, so rows are **durable across runs** (phase 8.4). This
holds on **both** backends: the file backend uses `@wam_cache_commit_str` /
`@wam_cache_load_str` (`tests/test_plawk_row_durable.pl`), and the **LMDB**
backend uses `wam_cache_commit_lmdb_str` / `wam_cache_load_lmdb_str`
(`tests/test_plawk_row_durable_lmdb.pl`), which `mdb_put` the row bytes under
the i64 key and store the declared schema under a distinguished non-8-byte key
(validated on open; data rows are skipped-by-size). A reader pass in a later
run sees rows a prior run committed. Keys stay i64 ids (content-stable
interning reproduces them; readers only iterate, and a re-writing pass
re-interns the same key). One gap remains on LMDB: `use NAME` (attach without
re-`declare`) needs a build-time resolver that reads the LMDB schema key —
`declare` works over LMDB today.

**Phasing** (each its own PR):
1. **Schema surface** — `declare NAME(col type, …)` parses and carries a row
   schema; bare `declare NAME` unchanged. No behavior change yet. **LANDED.**
2. **Row capture writer** — `TABLE[$k] = $0` stores the record as a row
   value; read back via `over TABLE`. In-run. **LANDED.**
3. **`records of TABLE as r`** — decode a stored row by the schema, name-only
   `r["col"]` read path (the safe, named reader). **LANDED**
   (`tests/test_plawk_records_reader.pl`): `r["col"]` resolves through the
   `declare TABLE(col type, …)` schema to the column's position and extracts
   that field of the stored row; an unknown column is unsupported (clean
   compile-time failure). In-run (rides the row-capture writer's storage).
4. **Byte-valued cache storage** — durable rows across runs (store row bytes,
   not the id). **LANDED (file + LMDB backends)**
   (`tests/test_plawk_row_durable.pl`, `tests/test_plawk_row_durable_lmdb.pl`):
   a str-valued (row) table commits the value BYTES and re-interns them on load
   — the file backend via `@wam_cache_commit_str` / `@wam_cache_load_str`, LMDB
   via `wam_cache_commit_lmdb_str` / `wam_cache_load_lmdb_str` — so a row
   committed by one run is read by a later
   run — proven with a reader-pass-then-writer program run twice. Keys stay
   i64 ids (content-stable interning reproduces them). Durable str values over
   the **LMDB** backend remain a follow-on (an lmdb row table currently rides
   the i64 lmdb path).
5. **`rows of` positional reader** — `r[N]` by position, no schema. **LANDED**
   (`tests/test_plawk_rows_reader.pl`). Its `unsafe` modifier + inline
   check-or-rename spec remain a follow-on.
6. **Richer row producers** — `row(...)` constructor / field-wise writes.
   The `row($a, $b, …)` constructor is **LANDED**
   (`tests/test_plawk_row_cons.pl`): `TABLE[$k] = row($a, $b, …)` stores a row
   built from the chosen input fields, in that order (projected / reordered),
   joined by the field separator so both readers recover the columns.
   Field-wise writes (`r["col"] = …`) remain a follow-on.

### 3.7 Table lifecycle: open-or-create, and selecting an existing table

**`declare` is open-or-create, not create-only.** `declare NAME` /
`declare NAME(col type, …)` inside `BEGIN cache("path") { … }` (a) always
builds a fresh in-memory table at startup, (b) if the store file at `"path"`
**already exists, loads it into that table** — a missing/empty file just
starts empty — and (c) commits the table back to the path at the end
(read-modify-write). So opening an existing store is not an error and does
not clobber on open: its contents are read in (this is the durability
mechanism), and the final commit overwrites the file with the merged result.

Two properties follow that are worth stating plainly, because they bound what
a "select an existing table" surface would mean:

- **The schema is source-only; it is not (yet) recorded in the store.** The
  column names/types come from `declare(cols)` in the program text. A store
  written by one program and read by another must repeat the same
  `declare(cols)`; nothing validates the stored bytes against it, so a
  mismatched re-declare silently mis-reads field offsets.
- **One store path ≈ one table.** The flat-file (and current byte-valued)
  format holds a single table's entries; multiple `declare`s in one
  `cache("path")` block all target the same file and would overwrite each
  other on commit. Genuine multi-table storage needs the namespace / LMDB
  named-sub-DB design (§3.5), still deferred.

**Should there be a `select` / `use` statement?** Yes, but it is only
meaningful once one of the two properties above changes, so it is planned as
a short sequence rather than a lone keyword:

- **(a) Persist the schema in the store. LANDED (file backend).** The
  byte-valued store header now carries the row schema
  (`name:type,name:type,…`); commit writes it, load reads it. When both the
  stored schema and the program's `declare(cols)` are present and differ, the
  open **fails cleanly** (`plawk: cache schema mismatch`, exit 3) instead of
  silently mis-reading field offsets. `tests/test_plawk_row_durable.pl`.
- **(b) `use NAME` (the `select`). LANDED (file backend).** A backed-BEGIN
  `use NAME` **attaches to an existing store and takes its schema from (a)** —
  no re-`declare(cols)`. The plawk build reads the store's persisted schema
  header at **compile time** and expands `use NAME` into the same
  `cache_table` + `cache_schema` a matching `declare NAME(cols)` would produce,
  so `records of` / `rows of` and the runtime schema check work unchanged. A
  missing / schema-less store is a compile error.
  `tests/test_plawk_use_table.pl`.
- **(c) Multiple named tables per store.** With the namespace design (§3.5,
  LMDB named sub-DBs), a store holds several tables and `use ns.orders`
  selects among them. This is the point at which `select` becomes load-bearing
  rather than sugar.

Which physical table a store operation targets when the program names none —
the **default table** — is specified per backend (and per backend *class*) in
`PLAWK_CACHE_BACKENDS.md`: the container itself on a single-table backend
(class A, our `file`), the native unnamed database on a default-plus-named
backend (class B, `lmdb`), a reserved name on an all-named backend (class C,
SQLite), a reserved key prefix on a table-less backend (class D, Redis). That
spec also fixes how named tables map, and uses out-of-scope backends
(SQLite/Redis/BerkeleyDB/TSV) as cases to check the rule. `use` (b) and
multi-table (c) implement it.

Ordering: (a) is the enabler and closes a correctness gap; (b) is the
`select`/`use` reader; (c) generalises to many tables. Recorded as phases
8.7–8.9 in §5.

### 3.8 Nested pass blocks (future TODO — brief spec, not planned)

Passes are a **flat, ordered sequence** today. A future extension is
**nested** `pass` blocks — a pass inside a pass. Recorded here as a sketch,
**not** on the near-term roadmap.

The key observation is that a nested pass is essentially a **loop**: the inner
block runs (repeatedly) within each step of the outer. That splits the idea
into two concerns that should not be conflated:

- **Iteration.** "Run this inner work N times" is what a general loop
  statement expresses. plawk today has `for (k in arr)` (assoc iteration) but
  **no C-style `while` / `for(;;)`** in the per-record body. A loop statement
  is the more natural, lighter-weight primitive for iteration, and is the
  likely near-term path if per-record looping is wanted — independent of the
  multi-pass / interaction-block model. Nesting `pass` blocks *just to loop*
  would be using the heavier construct for the lighter job.
- **Nested interaction scope.** The distinctive thing a nested `pass` block
  could add over a loop is its own *interaction scope* — potentially its own
  reader (`over` / `records of`) and/or its own cache/store per outer
  iteration. That is the only case where nesting earns its weight; it is
  advanced and rare.

**If pursued, the sketch:**
- **Scoping / shadowing.** An inner block introduces a nested variable scope:
  a name declared inside **shadows** the same name in the enclosing block
  (as in most block-scoped languages).
- **Reaching the parent.** A qualifier reads the enclosing block's namespace
  when a name is shadowed — e.g. `parent.name` or `parent[name]` (spelling
  TBD; `parent[…]` reads naturally alongside the existing `arr[key]`
  subscript). Multiple levels would chain (`parent.parent.name`) or index by
  depth.
- **Determinism.** The invariant of §1 still holds inside every block: work
  within the (possibly nested) iteration brackets is deterministic; nesting
  adds scopes, not choicepoints.

**Recommendation:** if a looping need arises first, add a `while` / loop
statement to the per-record body (the general primitive) rather than nested
`pass` blocks. Reserve nested `pass` blocks for genuine nested *interaction
scopes* (own reader/store), and design the shadowing + `parent.` reference
then. Neither is scheduled; this note exists so the surface stays coherent if
either is taken up.

**A unifying observation (for when `do`/`while` is designed).** These
constructs line up with pieces we already have: a `do { … }` block is almost
exactly a `pass { … }` block — a block of actions run (repeatedly) over a
record stream — and a `while (cond)` clause is essentially a **filter**, a
narrower cousin of the `WHERE`-style reader guard (now landed — see the Status
list): the guard
selects which rows a reader emits, while the `while` selects how long a loop
continues. That symmetry suggests the eventual `do`/`while` should share the
guard/condition machinery (a boolean predicate over the current record's
fields) rather than grow a parallel one.

**Loops and determinism — what a `while` actually gives back.** It is tempting
to say a loop "reintroduces non-determinism" that passes gave up. It does not,
and the precise version is worth recording because it clarifies what the
determinism invariant (§1) protects. Two independent axes were bundled when we
made passes deterministic:

- **Determinism vs non-determinism** — one well-defined result, no
  choicepoints / no backtracking, vs. a goal with *many* solutions that a
  solver searches.
- **Bounded vs unbounded** — a fixed structural sweep (one step per
  record/row, guaranteed to finish) vs. data-dependent iteration that runs
  until a runtime condition (and might not terminate).

A `pass` is **both** deterministic **and** bounded. A `while` loop keeps
determinism — same input, same output, still no backtracking — but drops
**boundedness**: the program decides how many times to run from runtime
values. So a loop does not hand back *non-determinism*; it hands back the
*expressive power non-determinism buys for free* — search, fixpoint,
enumerate-until — but under explicit, deterministic, step-by-step control. A
Prolog solver explores a solution space implicitly; a `while` lets you do the
same thing by hand. This is the same containment as Phase 6's aggregation
rule (the bounded-multiplicity principle,
`UNIFYWEAVER_LANGUAGE_PRINCIPLES.md`), seen from the other side: a
multi-solution goal is tamed either by
**folding it** (`collect`/`count` → one value, enumeration implicit) or by
**looping over it** (enumeration explicit, one deterministic step at a time).
The cost the loop pays is exactly the boundedness it drops: the
"constant-work-per-record, guaranteed-termination" property that made passes
analyzable. So if loops are added, the *default* should stay bounded and the
unbounded form be opt-in.

**Two iteration surfaces — `pass` vs `for-in`.** Iteration is not unique to
`pass`: `for (k in arr)` already loops, and the two share one underlying
iterator primitive. Documenting the difference now (even though nested `pass`
is unbuilt) keeps the surface coherent, because a future nested `pass` sits
squarely on this axis.

Both walk a table's occupied slots one step at a time; they differ in **what
each step binds** and **where the scope boundary sits**:

| | `pass … of TABLE as r { }` | `for (k in TABLE) { }` |
|---|---|---|
| **binds** | the whole **record** (`r["col"]` / `r[N]` / `$N`) | a single **key** `k` |
| **scope** | a fresh namespace per pass | shares the enclosing namespace |
| **host** | a top-level pass (its own reader/store) | a statement inside a body |

The **key-vs-record distinction is not fundamental**: a key is sufficient,
because you can always look the record back up (`TABLE[k]`) — so `for (k in
TABLE) { … TABLE[k] … }` and `pass records of TABLE as r { … r["col"] … }`
reach the same data. What differs is **ergonomics vs. flexibility**:

- **Returning a record** (the `pass … as r` reader) is **less verbose** — the
  fields are in hand, no re-lookup, and the guard/`WHERE` filter reads
  naturally (`if (r["amt"] > 100)`). The cost is a fixed shape: you get *this
  record*, decoded by *this* schema/position.
- **Returning a key** (`for-in`) is **more flexible** — the key can index
  *any* table (join across tables, cross-reference, dereference indirectly),
  or be used without ever materialising the record. The cost is the explicit
  `TABLE[k]` lookup (and its verbosity) at every use.

So neither subsumes the other: the record reader trades flexibility for
brevity, the key iterator trades brevity for reach. A nested `pass` (§3.8), if
built, is the record-returning form gaining its own nested scope — the natural
extension of the left column, not a new mechanism.

### 3.9 Views (future TODO — brief spec, not planned)

A **view** is a named, derived query over one or more tables — conceptually a
stored `records of … WHERE … ` (reader guards have landed; a view would pair
one with a projection). Recorded as a sketch, **not** scheduled.

Backend-adaptive, mirroring `PLAWK_CACHE_BACKENDS.md`:

- **If the backend has native views** (e.g. SQLite `CREATE VIEW`), define the
  view natively and let the backend evaluate it; a `use`/reader over the view
  name reads the backend's result set.
- **If it does not** (file, LMDB, Redis), a view is **materialised**: the
  named subset of columns (a projection, optionally filtered by a guard) is
  computed and **cached as its own row table** — i.e. a view degrades to "a
  derived table a pass builds", which the existing writer/reader/cache
  machinery already covers. Refresh semantics (eager on write vs. lazy on
  read vs. explicit) are the main open question.

So a view is either delegated to the backend or lowered to a cached
projection table; the surface (`view NAME as records of T where …`) is
uniform, and only the evaluation strategy is backend-specific. Depends on
reader guards (the `WHERE`) and, for multi-table sources, phase 8.9.

### 3.10 Contained non-determinism — a search construct (future TODO — sketch)

The loops discussion (§3.8) draws a line: a `while` gives back *unboundedness*
but stays deterministic. The natural question is whether we ever want the
*other* axis back — genuine **non-determinism** (choicepoints, backtracking,
multiple solutions) — and, if so, how to admit it without breaking the
inter-pass determinism invariant (§1). This section sketches a design. It is a
**future TODO, not scheduled**, recorded because the substrate makes it cheap
and the containment rule falls out of principles already in this document.

**Why it is worth wanting.** plawk compiles through the WAM — a machine whose
entire reason for existing is choicepoints, the trail, and backtracking. Today
we suppress all of that to get constant-memory streaming. But *search* problems
(constraint solving, combinatorial enumeration, joins expressed as relations,
rule evaluation — "find an assignment satisfying these facts") are exactly what
that machinery does well and what awk/SQL express badly. And the efficiency
intuition — "determinism is faster" — is only true for straight-line data
processing. For search, backtracking with first-argument **clause indexing**
(which this project already built, see the loader-side indexing work) is often
both terser *and* faster than a hand-rolled nest of loops carrying explicit
state. So the payoff is real, and it is the one place the logic-programming
interior earns its keep at the surface.

**The idea (the user's, refined).** Keep `=` as today: deterministic
assignment / bind-to-ground, committed, no choicepoint. Add a **second binding
mode** — spelled here `in`, e.g. `X in domain(...)` — that introduces a *logic
variable* ranging over a domain rather than a value. (The user suggested `is`;
that spelling collides with Prolog's arithmetic `is/2`, so `in` — echoing
CLP(FD)'s `X in 1..10` — reads better; final choice open.) Logic variables are
consumed by a **search construct** over a cartesian product of such variables,
pruned by constraints — idiomatically the encapsulated-search meta-predicates
Prolog already uses to make non-determinism safe (`findall/3`, `aggregate_all/3`,
`forall/2`). Two quantifier shapes, which the informal "forall / forevery"
naming blurs and which the design must keep distinct:

- **Existential (∃) — enumerate / pick.** `find` (first solution),
  `collect` (all solutions → an array, in solution order), `the` (require
  *exactly one*, else error). This is where search bridges back to the loop
  discussion: `collect` is precisely "materialise the solution set", after
  which an ordinary `for-in` or `pass` walks it deterministically.
- **Universal (∀) — test.** `forall(Gen, Cond)` = "every solution of `Gen`
  satisfies `Cond`", yielding a boolean. Standard Prolog `\+ (Gen, \+ Cond)`;
  a pure guard, no bindings escape.

**The containment invariant (the crux — the user's rules, and they are
right).** Non-determinism lives *strictly inside* a search construct inside a
single pass, and must be **collapsed to a deterministic value before the pass
boundary**. Concretely:

1. **Choicepoints never cross a pass boundary.** A pass still succeeds exactly
   once and leaves none behind — the §1 invariant is untouched *between*
   passes and in the streaming hot loop. All backtracking is confined to the
   search construct's brackets.
2. **A logic variable must be resolved before it leaves the search.** Either
   bound to a ground value (an existential `find`/`the`) or aggregated over its
   solutions (`collect`/`count`/`forall`). This is the *same* discipline as
   Phase 6's "a multi-valued lookup must be contained by an aggregation" — here
   promoted to a language-level rule.
3. **Unbound logic variables cannot enter imperative plawk.** They may be
   passed only to Prolog goals (predicates defined in a `BEGIN` block) and to
   the search construct's own constraints — never printed, stored in a table,
   or used in `=` arithmetic while unbound. Only the **ground projection** of a
   solved search flows into ordinary plawk values. In type terms, "unbound
   logic var" is a distinct kind that the collapse operators are the only way
   to eliminate.

**What actually triggers the non-determinism (and where it must live).** The
`where` is a **list comprehension**: `collect (A, B) where { A in nodes(g); B in
nodes(g); adjacent(A, B) }` is exactly Haskell's `[ (a,b) | a <- nodes, b <-
nodes, adjacent a b ]`. So it is worth being precise about which piece carries
the non-determinism. It is **not** the collapse keyword — `collect` is the
*eliminator* (one of `find`/`the`/`count`/`forall`), chosen for *how* to fold
the solution set. The **producers** are the generators (`A in …`) and any
multi-solution goal. Crucially, a goal's multiplicity is a property of its
**mode**, not the predicate: `adjacent(A, B)` with both unbound *generates*
pairs, `adjacent(a, B)` *extends* one, `adjacent(a, b)` is a yes/no *test*. So
the rule is enforced by mode analysis (`demand_analysis` /
`binding_state_analysis`, §1): a call carrying an **unbound logic variable** —
the only calls that can backtrack — must sit inside the `where`. The same
predicate called all-ground is an ordinary deterministic test, callable
anywhere. That relation-run-backwards (a test used as a generator) is the extra
power the search has over a plain comprehension guard. This is Principle 1 of
`UNIFYWEAVER_LANGUAGE_PRINCIPLES.md` (bounded multiplicity) in its
Prolog/comprehension form.

**Termination.** A general `while` (§3.8) is deterministic but may not
terminate. This search is the mirror image: non-deterministic but **bounded
when the domains are finite** — the solution space is a subset of the
cartesian product, so a finite `in`-domain gives a finite (if large) search
and guaranteed termination. That gives three well-understood computation
shapes, each giving up exactly one of the two guarantees:

| shape | determinism | boundedness | collapse at pass edge |
|---|---|---|---|
| `pass` / `for-in` | deterministic | bounded (one sweep) | — |
| `while` loop | deterministic | **unbounded** (data-driven) | must terminate |
| search (`in` + `find`/`collect`/`forall`) | **non-deterministic** | bounded (finite domains) | must resolve/aggregate |

**Sketch of the surface (illustrative, not final):**

```awk
BEGIN {
  # Prolog predicates supply the domains and the relation being searched
  adjacent(X, Y) :- ...
}
pass over graph as g {
  # search inside the pass; A,B are logic vars over a finite domain,
  # constrained by a BEGIN predicate; collapse to a ground array before print
  paths = collect (A, B) where { A in nodes(g); B in nodes(g); adjacent(A, B) }
  for (p in paths) print p          # pure plawk again: paths is ground
}
```

Everything left of the collapse operator (`collect`) is logic; everything
right of it is ordinary deterministic plawk. The pass exits with no
choicepoints.

**Assessment.** The idea has real merit and is well-formed: it is the standard
Prolog pattern (encapsulated search via `findall`/`aggregate_all`) surfaced as
a language feature, with the containment rule matching principles already in
this design (Phase 6 aggregation, §1 iteration-bracket determinism). It is also
clearly **advanced and not near-term** — it needs a logic-variable kind in the
type system, the collapse operators, a search evaluator over WAM choicepoints,
and finite-domain analysis for termination. Recorded as a sketch so the surface
stays coherent; sequenced *after* the `over query(Goal)` reader (Phase 6),
which introduces the first controlled non-determinism (a goal's solution set)
and is the smaller, prerequisite step.

## 4. Runtime: the cache ABI (the first build step)

The LLVM target has no persistence layer. We add a small **backend-agnostic
C ABI** the generated code calls, mirroring the shape of the existing
`@wam_assoc_i64_*` helpers so the assoc emitters can target it with minimal
change:

```
; open/close a named store (path from a BEGIN cache("...") block); flags
; carry the materialisation mode (lazy default, eager per-variable)
@wam_cache_open(i8* path, i32 flags) -> i8*        ; handle, null on failure
@wam_cache_close(i8* handle) -> void

; key/value are byte spans; value-out is borrowed until the next op
@wam_cache_put(i8* h, i8* key, i64 klen, i8* val, i64 vlen) -> i1
@wam_cache_get(i8* h, i8* key, i64 klen, i8** val_out, i64* vlen_out) -> i1
@wam_cache_del(i8* h, i8* key, i64 klen) -> i1

; the i64->i64 fast path (mirrors @wam_assoc_i64_inc / _get / _set), so a
; cache-backed histogram needs no (de)serialization
@wam_cache_i64_inc(i8* h, i64 key, i64 delta) -> i64   ; new value
@wam_cache_i64_get(i8* h, i64 key, i64* out) -> i1
@wam_cache_i64_set(i8* h, i64 key, i64 val) -> void

; iteration (drives `for (k in arr)` over a cache-backed table)
@wam_cache_iter_open(i8* h) -> i8*                 ; cursor
@wam_cache_iter_next(i8* cur, i64* key_out, i64* val_out) -> i1
@wam_cache_iter_close(i8* cur) -> void

; the inter-pass barrier
@wam_cache_commit(i8* h) -> void
```

**Backend.** The ABI is deliberately backend-agnostic:

- **v1 backend = LMDB** (link `liblmdb`), gated behind a build flag exactly
  as the other WAM targets gate it (`WAM_CPP_ENABLE_LMDB`,
  `facts_lmdb`, …). Load-everything-on-open is *not* required here — the
  memory-mapped B-tree is the point (larger-than-RAM working sets, OS
  paging, a durable/inspectable file between runs).
- **Fallback backend** (when LMDB is unavailable at build time): an
  in-process hash that serialises to a file on `commit`/`close`. Same ABI,
  so codegen never branches on the backend. This lets Phase 1 land the
  surface without hard-coupling the build to LMDB.

**Performance invariant preserved:** a program with no backed `BEGIN` block
emits **zero** cache IR and links nothing new — the same
"pay only for what you use" rule that gates every dynamic capability behind
per-site collection.

**Relationship to existing LMDB work.** The C/C++/Rust/Haskell WAM targets
already have LMDB *fact-source* designs (`WAM_CPP_LMDB_FACT_SOURCE_DESIGN`,
`WAM_LMDB_LAZY_*`, `WAM_LMDB_RESIDENT_INTERNING_*`). Those back a *Prolog
predicate* with on-disk facts for **query**. This cache is the *write* side
— a scratch KV store plawk populates and re-reads across passes. They
converge later: a committed cache is a natural fact source, so a pass-2
program could `dyncall`/query pass-1's output as facts. v1 keeps them
separate and does not adopt the interned multi-sub-db schema.

## 5. Phased rollout

Each phase is a shippable PR with tests. Phases 1–2 are independent enough
to land in either order, but 1 first is lower-risk (a primitive with a
single-pass test before any driver surgery).

- **Phase 1 — the cache runtime primitive + single-pass surface.** Add the
  `@wam_cache_*` ABI and the fallback (file-backed) backend; wire a
  `BEGIN cache("...") { declare NAME }` block so `arr[k]++` / `arr[k]=v` /
  `arr[k]` / `for (k in arr)` route through the store **within a single
  pass** (lazy materialisation; `eager` deferred to a later phase). Proves
  the primitive end to end (write, commit at `END`, read back in a second
  run) with no multi-pass machinery yet. Test: a histogram whose counts
  survive across two separate binary invocations against the same store.

- **Phase 2 — multiple `pass { }` blocks. LANDED (v1).** Parser:
  `program_passes(Begin, [pass(Rules), ...], End)` (PR-A). Driver (PR-B):
  each pass is emitted as its own function so the fixed per-record SSA
  (`%line`) and loop labels are function-local and cannot collide; `main`
  creates the shared assoc table once, threads it to each pass as a
  parameter, re-opens the input file per pass, and reads the table back in
  the END for-in. Verified: reading the input twice into a shared counter
  doubles the counts (`tests/test_plawk_multipass.pl`). v1 scope: text mode,
  a single shared table, always-rule pass bodies, `END { for (k in arr)
  print ... }`, a file argument. **Landed since:** per-record *output*
  passes (per-record `$N` / `arr[$N]` reads, `tests/test_plawk_assoc_print.pl`);
  cross-pass *scalars* via zero-initialised module globals — no loop-phi
  threading needed (`tests/test_plawk_crosspass_scalar.pl`); pure-scalar
  (no-table) multi-pass, where the shared-table parameter is dropped
  entirely and passes coordinate only through scalar globals; and
  arithmetic in a per-record print evaluated in f64 (`$2 / total`,
  `tests/test_plawk_normalise.pl`) — the surface `/` is integer, so print
  arithmetic promotes operands to double (fields via
  `@wam_atom_field_f64_value`, a scalar via `load`+`sitofp`, an int constant
  via `sitofp`) and prints with `%g`. Grand-total normalise
  (`pass { total += $2 } pass { print $1, $2 / total }`) now works with no
  assoc table at all. **Landed since:** per-key aggregation — associative
  add-assign `arr[$k] += DELTA` (`tests/test_plawk_perkey.pl`, the general
  form of `arr[$k]++`, mapping to `@wam_assoc_i64_inc` with the record's
  delta) folds a per-key sum in pass 1, and a table lookup as an arithmetic
  operand (`$2 / total[$1]`, interned key → `@wam_assoc_i64_get` → `sitofp`)
  lets pass 2 divide each record by its own key's total. Grand-total and
  per-key normalise both work end to end. **Deferred:** multiple tables, and
  non-field keys / deltas in add-assign. (Pairing multi-pass with a
  cache-backed table landed in phase 3, below.)

- **Phase 3 — cache as the inter-pass channel (durable payoff). LANDED
  (v1).** A table declared in a `BEGIN cache("path") { declare NAME }` block
  and shared by the passes is loaded from its store before pass 1 and
  committed after the last pass — so the same table is both the in-memory
  channel between passes and durable between separate runs of the binary;
  re-running loads the prior committed state and accumulates onto it. Wired
  into both multi-pass driver shapes (no-END normalise and END for-in) by
  reusing the single-pass cache machinery (`plawk_program_cache_tables` /
  `plawk_cache_entries` / the cache-aware `plawk_assoc_entry_setup_ir/3` /
  `plawk_cache_commit_lines`); the passes themselves are unchanged (they
  already reference the shared table). Works for the file backend and, when
  liblmdb is present, `backend "lmdb"` (`bin/plawk` links the C runtime and
  `-llmdb` for a multi-pass program too). Verified: a cache-backed per-key
  sum / histogram persists and accumulates across runs
  (`tests/test_plawk_multipass_cache.pl`). **Deferred:** the commit *barrier*
  between passes (v1 commits once, after the last pass — the in-memory table
  IS the live channel between passes, so an intermediate commit is only
  needed for cross-process coordination); the name resolution surface below;
  and one-store-per-block with multiple stores. Follow-on: `as ns`
  namespaces a store (tables `ns.table`,
  mapping to LMDB named sub-databases), no-alias blocks stay global, and
  `global` lifts a namespaced table to bare. Test: `total[$1] += $2` in
  pass 1; `print $1, $2 / total[$1]` in pass 2, verifying pass 2 sees
  complete, durably-committed totals; plus a namespaced `stats.work` table
  addressed from a pass and a `global` table used bare.

- **Phase 4 — configurable readers.** `over TABLE` **LANDED (v1)**;
  `over prev` deferred. `pass over TABLE as VAR { print ... }` iterates the
  table's occupied slots as the pass's record source instead of re-scanning
  the input — the "process what a previous pass stored" shape, emitting one
  line per distinct key rather than per input record. Fields are **named**
  (name lookup): the loop key binds to `VAR` and its value reads as
  `TABLE[VAR]` (no positional `$1`/`$2`). Implemented by dispatching the
  multi-pass driver's per-pass emission on the pass shape — a plain
  `pass { }` scans input, a `pass over` walks the shared table with the same
  body emitter the END for-in uses (`@wam_assoc_i64_iter_next` /
  `key_at` / `value_at`), as a `void` function `main` calls in sequence. It
  pairs with a cache-backed table (durable across runs) like the
  input-scanning passes. Test: `tests/test_plawk_over_table.pl`.
  **Deferred:** `over prev` (the previous pass's sink spools into this pass —
  the in-process `awk | awk`, sugar over `over <spool>`; needs a spool sink +
  redirect of the prior pass's writer); guarded / writebin over-bodies;
  string-literal print fields; multiple tables.

- **Phase 5 — LMDB backend + materialisation modes.** Swap the fallback for
  `liblmdb` behind the build flag; add a test that exceeds a small RAM cap
  to demonstrate the lazy memory-mapped path, and wire the per-variable
  `eager` marker (load fully at open). Test: an `eager` scalar/table reads
  without a per-access store hit; a lazy table stays memory-mapped.

- **Phase 6 — `over query(Goal)` + determinism guarantees.** The
  query-driven reader (each solution a record, on the `call/1` + dynamic-DB
  machinery); document and (optionally) check that pass bodies remain
  deterministic; surface the closure-based "compute-once, reuse" pattern;
  confirm commit-barrier snapshot semantics in a test. **Planned** in
  `PLAWK_QUERY_READER_IMPLEMENTATION_PLAN.md`: a code-grounded four-PR sequence.
  The key finding is that the goal-call surface is single-solution only, so the
  reader **materialises** the solution set (a `findall` collapse at the reader
  boundary — bounded-multiplicity, `UNIFYWEAVER_LANGUAGE_PRINCIPLES.md`
  Principle 1) and iterates it like `over TABLE`, rather than retaining live
  choicepoints (which §1 forbids in the hot loop). **PRs 1–3 landed**: the
  surface parses to `pass_query(query(Pred, Vars), Body)`, and an all-query
  program of any goal arity with a `print $K...` body runs end-to-end — the
  build injects a per-column wrapper `__plawk_query_pred_C(L) :- findall(VC,
  pred(V1..Vn), L)`, materialises each column into a table by position via
  `@wam_object_call_posarray` on the shared `%WamState`, and walks keys in
  order binding `$1..$n` (ordered, deterministic; a disjunctive goal yields
  every solution; columns stay aligned across the per-column runs for a pure
  goal). The body may reorder, repeat, or subset the printed columns. Guards
  in the body, mixed query + ordinary passes, and string columns are the next
  PRs.

- **Phase 7 — secondary indexes (§3.5).** `index TABLE by FIELD [unique]` on
  record-valued tables; unique lookups return one record; non-unique
  lookups require an aggregation (`collect` → array, `count`, `sum`/`min`/
  `max`/`avg` over a field). Backed by plain / `MDB_DUPSORT` index sub-DBs,
  maintained on write. The aggregation requirement is the determinism
  containment for a multi-valued lookup. Rides Phase 5 (needs the LMDB
  sub-DB backend) and record-valued cache entries (Phase 8).

- **Phase 8 — row-oriented records (record-valued tables) (§3.6).** A table
  whose value is a named-field **row** (model A). Sub-phases: (8.1) the
  `declare NAME(col type, …)` schema surface — **LANDED**; (8.2) the row
  capture writer `TABLE[$k] = $0` (str-value; in-run), read back via `over
  TABLE` — **LANDED** (`tests/test_plawk_row_capture.pl`); (8.3) the safe
  `records of TABLE as r` reader with name-only `r["col"]` decode by schema —
  **LANDED** (`tests/test_plawk_records_reader.pl`); (8.4) byte-valued cache
  storage for durable rows across runs — **LANDED (file + LMDB backends)**
  (`tests/test_plawk_row_durable.pl` for the file backend,
  `tests/test_plawk_row_durable_lmdb.pl` for LMDB via
  `wam_cache_{commit,load}_lmdb_str` — schema stored under a distinguished key
  and validated on open; `use NAME` over an LMDB store reads that schema at
  build time via a small liblmdb probe, single- and multi-table); (8.5) the
  positional `rows of`
  reader (`r[N]`, no schema) — **LANDED**
  (`tests/test_plawk_rows_reader.pl`; its `unsafe` / inline check-or-rename
  spec remain a follow-on); (8.6) richer row producers (`row(...)` /
  field-wise). Foundational for Phase 7 (secondary indexes need addressable
  named fields). Table lifecycle / selecting an existing table (§3.7):
  (8.7) persist the schema in the store + validate it on open (self-describing
  store, closes the schema-mismatch hole) — **LANDED (file backend)**
  (`tests/test_plawk_row_durable.pl`); (8.8) `use NAME` that attaches to an
  existing store and takes its schema from 8.7 (read at build time) — the
  `select` surface, no re-`declare` — **LANDED (file backend)**
  (`tests/test_plawk_use_table.pl`); (8.9) multiple named tables per store
  (namespaces / LMDB named sub-DBs, §3.5, per `PLAWK_CACHE_BACKENDS.md`) so
  `use ns.table` selects among them — **LANDED** (all five PRs in
  `PLAWK_MULTITABLE_IMPLEMENTATION_PLAN.md`): the multi-pass driver takes N
  tables (`tests/test_plawk_multitable.pl`); a multi-table *file* store is a
  class-A compile error while an *lmdb* store routes each table to its own named
  sub-DB — durable and isolated, both i64 and row values
  (`tests/test_plawk_multitable_store.pl`, `tests/test_plawk_multitable_lmdb.pl`);
  `cache("db" as ns)` namespaces a store so its tables are referenced `ns.table`
  (the local part is the sub-DB), a namespaced file store being a class-A error
  (`tests/test_plawk_namespace.pl`); and `use ns.table` attaches to a
  multi-table store with no re-`declare`, reading each sub-DB's schema at build
  time (`tests/test_plawk_use_namespace.pl`). (8.10) **reader guards** — a
  `WHERE`-style row filter on any of the three readers: `if (r["col"] CMP N)`
  (records), `if (r[N] CMP N)` (positional), `if ($N CMP N)` (anon), for
  the six operators `== != < <= > >=`. An integer literal lowers to an i64
  field extract + `icmp`; a decimal float literal (`3.5`) to an f64 extract +
  `fcmp`; a string literal (`"alice"`, `==`/`!=`) to a length-then-`memcmp`
  byte comparison; and comparisons combine with `&&`/`||` (short-circuit
  branches). Filtered rows never reach the print block — **LANDED**
  (`tests/test_plawk_reader_guards.pl`, `tests/test_plawk_guard_float.pl`,
  `tests/test_plawk_guard_string.pl`, `tests/test_plawk_guard_bool.pl`; string
  *ordering* (`<`/`>` on text) remains a follow-on).

## 6. Open questions

- **Input re-scan for pass N — decided.** A **seekable** input is simply
  **re-opened** for each pass (automatic). A **non-seekable stream** (stdin,
  a pipe, a socket) is *not re-openable*, and the runtime **never silently
  spools it** — hidden buffering of an unbounded stream is exactly the kind
  of implicit cost this design avoids. To get a second pass over stream
  data, the program must **explicitly capture it into a store**: pass 1
  writes the records to a declared backed table, and a later pass reads them
  with `over TABLE`. The stream→cache spool is thus a *declared* step, not a
  default — it reuses the ordinary store mechanism, so no new machinery, but
  the memory/disk cost is always visible in the source. A second `over
  input` pass on a non-seekable stream *without* such a capture is an error
  (diagnosed at compile time when the input is known to be a stream,
  otherwise at runtime). `over prev` is likewise explicit — it spools the
  previous pass's *emitted* records, which the program chose to emit.
- **`over prev` sink vs stdout.** When a pass is consumed by `over prev`,
  its `print` feeds the spool instead of stdout. Does it *also* echo to
  stdout (tee), or only feed the next pass? Proposal: only feed the next
  pass (a true pipeline stage); the final, unconsumed pass prints to stdout.
- **Value encoding for non-i64 cache values — direction set (§3.6).**
  Records/blobs need a serialization. v1: i64 and blob-bytes only.
  Structured **row** values (Phase 8) reuse the existing record
  (de)serialization (`@wam_object_call_record` typecodes) as one blob per
  key, decoded by the table's declared `declare NAME(col type, …)` schema.
- **Cache lifetime / cleanup.** A backed `BEGIN cache("path")` block names
  an explicit path, so the store is durable by construction (survives exit).
  Open: do we also want an *unnamed* backed block (`BEGIN cache { … }`) that
  gets an ephemeral temp store deleted at exit — useful for a pure
  spill-to-disk larger-than-RAM pass with no cross-run intent? Leaning yes,
  as a convenience.
- **`eager` granularity.** `eager` is proposed per-table inside a backed
  block. Open: also allow a whole block `BEGIN cache("db") eager { … }` when
  every table is small/hot? Cheap to add if wanted.
- **Secondary indexes.** Sketched in §3.5 (`index TABLE by FIELD [unique]`;
  non-unique lookups must be aggregated — `collect`/`count`/`sum`). Deferred
  to a phase past v1; primary-key-only until then, made explicit here.
- **Multi-process concurrency — held, deserves its own design.** A named
  store (`as`) is meant as a cross-process coordination unit, but the
  *concurrent-access* semantics are deliberately **not** settled here:
  simultaneous writers, reader/writer isolation (LMDB's single-writer +
  MVCC readers), whether the commit barrier is per-process or coordinated,
  and how a peer sees another process's mid-run commits. v1 is
  single-process, single-writer; genuine multi-process coordination is
  parked as a dedicated design task rather than decided in passing.

## 7. Relationship to the JIT/self-host roadmap

`PLAWK_JIT_ROADMAP.md`'s arc (runtime-loadable grammars, structured
returns, the eval/compile bootstrap) is complete. This is the first item of
a **new** arc — *beyond single-pass awk* — orthogonal to the JIT work but
compounding with it: the grammars and compiled predicates the JIT arc made
loadable are exactly the artifacts multi-pass keeps hot across passes.
