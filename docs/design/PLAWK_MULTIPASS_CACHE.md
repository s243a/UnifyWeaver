<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk multi-pass processing over a persistent cache

**Status**: design. No implementation yet. Captures the determinism
rationale, the surface, the runtime ABI, and a phased rollout so the
context survives across sessions. The LLVM target has **no cache/LMDB
layer today** — that runtime primitive is the first build step.

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
BEGIN { CACHE = "run.lmdb" }

pass { total[$1] += $2 }              # pass 1: accumulate into the cache

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

### 3.2 Variable scoping and persistence

Two orthogonal properties — a scoping property (does a variable survive
into the next pass?) and a storage property (is it durable / shareable?):

- **Pass-local by default.** A variable first used inside a `pass { }` is
  local to that pass and reset at the start of each pass. Its in-loop state
  lives in native SSA slots, as today.
- **`BEGIN`-declared variables persist across passes** (they are
  program-global). This is the in-memory cross-pass channel — fast,
  process-lifetime, RAM-bound. `sum` or an assoc `arr` introduced in
  `BEGIN` carries its accumulated value from one pass into the next with no
  DB involved.
- **Backing is a separate, opt-in property.** A persistent variable can
  additionally be **DB-backed** with `as cache`, which makes it durable
  (survives process exit) and larger-than-RAM. `as cache` uses the default
  store (`BEGIN { CACHE = "..." }`); **`as cache("path.lmdb")` names the
  store**, so coordinating processes can point one variable at a shared DB
  while the rest stay local. Persistence (BEGIN scope) and backing
  (`as cache`) compose independently:

| declaration | survives passes? | durable / cross-process? |
|---|---|---|
| `x` used inside a pass | no (pass-local) | no |
| `x` set in `BEGIN` | yes (in-memory) | no |
| `x` in `BEGIN` `as cache` | yes | yes — default DB (`CACHE`) |
| `x` in `BEGIN` `as cache("shared.lmdb")` | yes | yes — **named** DB |

This corrects an earlier conflation: declaring in `BEGIN` does **not** by
itself put a variable in the cache; it makes it persist *in memory*.
Backing is the explicit `as cache[(db)]` opt-in.

### 3.3 The cache as an associative array

A DB-backed table reads/writes through the persistent store instead of the
in-memory hash, reusing the assoc-array ergonomics users already know:

```awk
BEGIN { CACHE = "run.lmdb" ; declare total as cache }

pass { total[$1] += $2 }              # persistent accumulate
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

## 4. Runtime: the cache ABI (the first build step)

The LLVM target has no persistence layer. We add a small **backend-agnostic
C ABI** the generated code calls, mirroring the shape of the existing
`@wam_assoc_i64_*` helpers so the assoc emitters can target it with minimal
change:

```
; open/close a named store (path from BEGIN { CACHE = "..." })
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

**Performance invariant preserved:** a program with no `CACHE` and no
cache-backed table emits **zero** cache IR and links nothing new — the same
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
  `BEGIN { CACHE = "..." }` setting and a `declare NAME as cache`
  table so `arr[k]++` / `arr[k]=v` / `arr[k]` / `for (k in arr)` route
  through the store **within a single pass**. Proves the primitive end to
  end (write, commit at `END`, read back in a second run) with no
  multi-pass machinery yet. Test: a histogram whose counts survive across
  two separate binary invocations against the same cache file.

- **Phase 2 — variable scoping + multiple `pass { }` blocks.** Parser:
  `program(Begin, Passes, End)` where `Passes` is a list of rule-sets.
  Scoping (§3.2): pass-local by default (reset per pass); `BEGIN`-introduced
  variables persist in memory across passes. Driver: run the record loop
  once per pass, re-scanning input between passes, threading loaded objects
  / foreign wrappers / resolved entries across passes (they already live in
  process-global state, so this is mostly *not resetting* them). Test: the
  normalise-by-total example with an **in-memory, `BEGIN`-declared** total —
  impossible single-pass; here pass 1 fills it, pass 2 divides. (No cache
  needed for this phase — it exercises in-memory cross-pass persistence.)

- **Phase 3 — cache as the inter-pass channel (durable payoff).** Combine
  1+2: a `BEGIN`-declared `as cache` table written in pass 1 and read in
  pass 2, with the commit barrier between passes; and `as cache("path")`
  for a named store. Test: `total[$1] += $2` in pass 1; `print $1, $2 /
  total[$1]` in pass 2, verifying pass 2 sees complete, durably-committed
  totals.

- **Phase 4 — configurable readers.** `over TABLE` (iterate a table as the
  record source) and `over prev` (the previous pass's sink spools into this
  pass — the in-process `awk | awk`, sugar over `over <spool>`). Writers
  gain the spool sink. Test: pass 1 emits a filtered/derived stream, pass 2
  `over prev` consumes it.

- **Phase 5 — LMDB backend + larger-than-RAM story.** Swap the fallback for
  `liblmdb` behind the build flag; add a test that exceeds a small RAM cap
  to demonstrate the memory-mapped path.

- **Phase 6 — `over query(Goal)` + determinism guarantees.** The
  query-driven reader (each solution a record, on the `call/1` + dynamic-DB
  machinery); document and (optionally) check that pass bodies remain
  deterministic; surface the closure-based "compute-once, reuse" pattern;
  confirm commit-barrier snapshot semantics in a test.

## 6. Open questions

- **Input re-scan for pass N.** Re-open a seekable file is trivial; stdin is
  not seekable. Options: (a) require seekable input for multi-pass; (b)
  spool stdin to a temp file on pass 1; (c) only allow `over TABLE` /
  `over prev` (not `over input`) for the non-first pass. Leaning (a)+(c) for
  v1, (b) later.
- **`over prev` sink vs stdout.** When a pass is consumed by `over prev`,
  its `print` feeds the spool instead of stdout. Does it *also* echo to
  stdout (tee), or only feed the next pass? Proposal: only feed the next
  pass (a true pipeline stage); the final, unconsumed pass prints to stdout.
- **Value encoding for non-i64 cache values.** Records/blobs need a
  serialization. v1: i64 and blob-bytes only; structured records reuse the
  existing record (de)serialization (`@wam_object_call_record` typecodes)
  keyed by the same layout — deferred to when a use needs it.
- **Cache lifetime / cleanup.** Is the cache file ephemeral (temp, deleted
  at exit) or durable (named, survives)? Proposal: durable when `CACHE` is
  an explicit path, ephemeral (temp) when a program uses a cache-backed
  table without naming one.
- **Concurrency.** Single-process, single-threaded for v1 — LMDB's
  multi-reader story is out of scope until a use appears.

## 7. Relationship to the JIT/self-host roadmap

`PLAWK_JIT_ROADMAP.md`'s arc (runtime-loadable grammars, structured
returns, the eval/compile bootstrap) is complete. This is the first item of
a **new** arc — *beyond single-pass awk* — orthogonal to the JIT work but
compounding with it: the grammars and compiled predicates the JIT arc made
loadable are exactly the artifacts multi-pass keeps hot across passes.
