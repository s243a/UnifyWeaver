<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk multi-pass processing over a persistent cache

**Status**: partially implemented. Captures the determinism rationale, the
surface, the runtime ABI, and a phased rollout. **Landed:** the persistent
cache (file backend + `BEGIN cache("path") { declare NAME }` surface, phase
1); the LMDB backend (`backend "lmdb"`, phase 5, eager); multi-pass
execution (`pass { }` blocks over a shared assoc table, phase 2); and
per-record output in assoc programs (`print $N` / `print arr[$N]` in the
record loop), which gives the "normalise" shape — pass 2 prints each record
from pass 1's table; cross-pass scalars (`acc += 1` / `acc += $N`
folded in one pass and read in a later pass, backed by a zero-initialised
module global); pure-scalar (no-table) multi-pass, where a program carries
no assoc table at all and passes coordinate only through scalar globals;
arithmetic in prints (`$2 / total`) evaluated in f64 (the surface `/`
is integer, so a print expression is promoted to double and printed with
`%g`), which completes **grand-total normalise**
(`pass { total += $2 } pass { print $1, $2 / total }`); and per-key
aggregation via associative add-assign (`total[$1] += $2`) plus a table
lookup as an arithmetic operand (`$2 / total[$1]`), which completes
**per-key normalise** — each record divided by its own key's total. **Not
yet:** configurable readers (phase 4); the query reader (phase 6);
namespaces / `eager` / secondary indexes; string-literal print fields. See
the per-phase status tags in §5.

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
always-rule pass bodies. Combining a cache-backed table with multi-pass,
cross-pass *scalars*, string-literal print fields, multiple tables, and
reader selection are follow-ons.

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
`IO`-monad analogy), never the default in the hot loop.

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
pass { ... }                    # default: re-scan the original input
pass over total { ... }         # iterate a table (cache-backed or in-memory)
pass over prev { ... }          # consume the previous pass's emitted records
pass over query(Goal) { ... }   # each solution of a query is a record
```

- **`over input`** (the default) — re-open and re-scan the program's input.
- **`over TABLE`** — iterate a table's entries as records (the `for (k in
  arr)` iteration, but as the pass's record source). This is the natural
  "process what the previous pass stored" shape when the previous pass
  accumulated into a table.
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
multi-valued lookup from leaking a choicepoint into the hot loop.

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
  per-key normalise both work end to end. **Deferred:** multiple tables,
  non-field keys / deltas in add-assign, and pairing multi-pass with a
  cache-backed / `BEGIN`-declared table.

- **Phase 3 — cache as the inter-pass channel (durable payoff).** Combine
  1+2: a table declared in a `BEGIN cache("db")` block written in pass 1 and
  read in pass 2, with the commit barrier between passes; plus the
  one-store-per-block and load-on-open rules. Also introduce the name
  resolution surface: `as ns` namespaces a store (tables `ns.table`,
  mapping to LMDB named sub-databases), no-alias blocks stay global, and
  `global` lifts a namespaced table to bare. Test: `total[$1] += $2` in
  pass 1; `print $1, $2 / total[$1]` in pass 2, verifying pass 2 sees
  complete, durably-committed totals; plus a namespaced `stats.work` table
  addressed from a pass and a `global` table used bare.

- **Phase 4 — configurable readers.** `over TABLE` (iterate a table as the
  record source) and `over prev` (the previous pass's sink spools into this
  pass — the in-process `awk | awk`, sugar over `over <spool>`). Writers
  gain the spool sink. Test: pass 1 emits a filtered/derived stream, pass 2
  `over prev` consumes it.

- **Phase 5 — LMDB backend + materialisation modes.** Swap the fallback for
  `liblmdb` behind the build flag; add a test that exceeds a small RAM cap
  to demonstrate the lazy memory-mapped path, and wire the per-variable
  `eager` marker (load fully at open). Test: an `eager` scalar/table reads
  without a per-access store hit; a lazy table stays memory-mapped.

- **Phase 6 — `over query(Goal)` + determinism guarantees.** The
  query-driven reader (each solution a record, on the `call/1` + dynamic-DB
  machinery); document and (optionally) check that pass bodies remain
  deterministic; surface the closure-based "compute-once, reuse" pattern;
  confirm commit-barrier snapshot semantics in a test.

- **Phase 7 — secondary indexes (§3.5).** `index TABLE by FIELD [unique]` on
  record-valued tables; unique lookups return one record; non-unique
  lookups require an aggregation (`collect` → array, `count`, `sum`/`min`/
  `max`/`avg` over a field). Backed by plain / `MDB_DUPSORT` index sub-DBs,
  maintained on write. The aggregation requirement is the determinism
  containment for a multi-valued lookup. Rides Phase 5 (needs the LMDB
  sub-DB backend) and record-valued cache entries.

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
- **Value encoding for non-i64 cache values.** Records/blobs need a
  serialization. v1: i64 and blob-bytes only; structured records reuse the
  existing record (de)serialization (`@wam_object_call_record` typecodes)
  keyed by the same layout — deferred to when a use needs it.
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
