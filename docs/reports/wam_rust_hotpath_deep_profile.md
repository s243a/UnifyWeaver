<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM (uw-resolve) hot-path DEEP profile — per-site optimization targets

This drills into the categories the first profile
(`wam_rust_hotspot_profile_full_tier.md`, D91) found and attributes each hot
category to **concrete call sites, `Value` constructions and predicates**, so the
optimization rounds that follow have exact targets. D91 stopped at the category
breakdown (memory-mgmt ~61 % B2 / ~49 % B3, deref, "f/N" functor strings, the B3
`msort`); this report ranks the actual allocation sites (by DHAT), the deref
call-sites and the sort comparator, and — the piece D91 lacked — runs
callgrind **with `--cache-sim=yes`** to settle whether the allocator cost is
latency-bound or instruction-bound.

**Content SHA.** `origin/main` @ `ef148c548` (merge of #4223 — full merged lowered
tier, regions 1–5 + general recognizer, all default-ON; ledger through D93). The
generated crate `examples/pkg_resolver/rust/uw_resolve_wam` is the profiled
subject. MEASUREMENT ONLY: `resolver.pl`, the `wam_rust` target/spec and all
committed crate source were left unmodified. The instrumented binary is the
committed crate rebuilt **with debuginfo only** (`RUSTFLAGS=-g`, no source
change) into a throwaway target dir; DHAT/callgrind harness scripts live under a
scratch path and nothing but this report is committed.

## TL;DR — the three levers, now attributed to sites

1. **Intern functors AND atoms/var-names to `u32` (kill the String bucket).**
   DHAT shows the churn is *tiny name-Strings*, not big terms: **76 % of ALL B3
   allocation blocks and ~37 % of ALL B2 allocation blocks are atom / functor /
   variable-name `String` allocations** (`deref_var` clones, `deref_heap`'s
   `functor.to_string()`, `term_compare`/`display_functor_name` atom-name
   clones). Cache-sim proves this cost is **instruction-bound** (LL miss rate
   ~0.0 %), so cutting the *number* of `malloc`/`free`/`clone`/`rfind('/')`
   calls is exactly the win. Blast radius is large (the functor lives in the
   `Value` enum and is built at 400+ sites in the **generated** `lib.rs`), so it
   is a code-generator change, not a hand-edit.

2. **Cut `Value` allocation churn the existing sharing does not cover.** The
   `Arc<Vec<StackEntry>>` frame stack and structural-shared `Args` already make
   choice-point creation and `Value::clone` O(1) — confirmed: `StackEntry::clone`
   is only 1.9 % of B2 allocation bytes and `save_regs`' register snapshot is not
   a top site. The leftover churn is finer: (a) **`deref_heap` rebuilds term
   spines** (`Value::strv`/`Value::list` → fresh `Arc<Vec<Value>>`) whenever any
   sub-term derefs to a new cell — 48 MB / 1.04 M blocks in B2; (b) the
   **string-tagged trail**: `bind_var` does `format!("__binding__{}", var)` +
   `bindings.insert(var.to_string(), …)` = **two `String` mallocs per bind**, and
   `trail_binding` a third; (c) `deref_var` returns a cloned `Value` on every
   hop.

3. **Decorate-sort the B3 `msort`/`sort` builtin.** `msort/2` already
   pre-derefs each element once (O(n)), but its comparator `term_compare`
   **re-derefs both operands from scratch on every comparison** — `deref_heap` is
   called ~5.4 M times, and the sort comparator (`driftsort` + `dedup_by`) drives
   term_compare to **≈61 % of B3's total instructions**. A comparator that trusts
   the pre-deref removes the O(n log n) re-deref, leaving O(n).

## Methodology

Same two workloads as D91, same box, `LC_ALL=C.UTF-8`, release build with
`-g` (debuginfo does not change codegen semantics — B3 reproduces
`resolve_ms≈312`, `selection_size=10`; B2 subset timing scales to the D91 19.2 s
full run). Three instruments:

1. **DHAT** (`valgrind --tool=dhat`) — per-allocation-site counts, byte totals
   and access traffic. B2 on a **60-case** subset (alloc-site proportions are
   stable across cases, as D91 established for the census); B3 on the full 5 000-
   package `resolve_layered --bench`. The DHAT JSON's program-point tree was
   parsed to attribute every allocation to the first meaningful crate frame
   (skipping `malloc`/`RawVec`/`String::from` plumbing), then a second pass
   classified each site's allocations as `String` / `Vec` / `HashMap` / spine.
2. **Callgrind `--cache-sim=yes --branch-sim=no`** — self-Ir with the full
   memory-hierarchy columns (I1/D1/LL misses) D91 could not collect, plus
   `cfn=`/`calls=` records parsed for **caller attribution** of `deref_heap`,
   `deref_var` and `term_compare`. B2 60-case, B3 full.
3. Cross-checked against D91's `UW_PROF` scalar census (deref 142.8 M B2 /
   5.37 M B3, 479 531 backtracks B2, etc.) — self-Ir tables here reproduce
   D91's category rollups (memory ~61 %/49 %, deref self ~14 %/15 %), so the
   `-g` build is the shipping profile.

**Limits.** (a) DHAT and callgrind are separate runs; DHAT ranks *allocation
requests by site*, callgrind ranks *instructions executed*, so a site can be
high in one and low in the other (e.g. `step` is 22 % of B3 alloc *bytes* but
its *self*-Ir is ~0.4 % — it requests big term Vecs but the instructions run in
its callees). Both views are given and labelled. (b) B2 DHAT is a 60-case
subset and B2 callgrind a 60-case subset; absolute counts scale ~×43 to the full
2 600-case corpus. (c) Ir counts instructions, not cycles — but see the
cache-sim result below, which shows Ir is a *good* proxy here (≤5 % inflation),
overturning D91's caveat that the allocator's wall share was badly
under-counted. (d) Load-time allocations (`json::Parser`, `shared_wam_program`)
are included in the DHAT totals and are called out where they rank; they are not
part of the resolve hot path.

## Cache-sim result (the memory-hierarchy cost D91 lacked)

| workload | Ir | D1 miss rate | **LL miss rate** | LL misses | est. cycle inflation over Ir\* |
|---|---:|---:|---:|---:|---:|
| **B2** (60-case) | 2.219 B | 1.0 % | **0.0 %** | 80 238 | **~+4.7 %** |
| **B3** (5 k) | 4.184 B | 0.6 % | **0.0 %** | 1 081 424 | **~+5.3 %** |

\*`Ir + 10·D1miss + 100·LLmiss`, the standard first-order estimate.

**Reading — this changes the framing.** The working set fits in L2/L3: last-level
miss rate is ~0 % in both workloads. The allocator therefore does **not** cost
because of memory latency — it costs because `_int_malloc`/`_int_free`/`free`/
`malloc` are **48 % of B2 self-Ir and ~36 % of B3 self-Ir as raw instruction
count**. In B3 the few LL misses that exist are **60.7 % concentrated in
`_int_malloc`'s write traffic** (`DLmw`) — i.e. writing into freshly-allocated
pages — so even the miss cost tracks allocation *count*. **The lever is fewer
allocations, and Ir is a faithful proxy for the win.** D91's "wall share is very
likely higher" caveat is, on this box, worth only ~5 %.

## B2 — allocation sites (term differential; DHAT 60-case: 4,136,815 blocks / 135.2 MB)

### Top allocation sites by BLOCK COUNT (malloc/free call pressure)

| site | blocks | % of all blocks | bytes | dominant kind |
|---|---:|---:|---:|---|
| **`WamState::deref_heap`** | **1,620,878** | **39.2 %** | 49.9 MB | 72 % spine-`Vec` rebuild, 28 % `functor.to_string()` |
| **`WamState::deref_var`** | **998,406** | **24.1 %** | 4.0 MB | **100 % `String` clone** (Unbound/Atom name) |
| `WamState::step` | 503,008 | 12.2 % | 39.9 MB | 58 % term `Vec`, 18 % `String`, 24 % misc |
| `WamState::run` | 181,244 | 4.4 % | 1.0 MB | dispatch scratch |
| `WamState::put_reg` | 123,600 | 3.0 % | 10.0 MB | Y-reg `HashMap` insert (`name.to_string()`) |
| `WamState::backtrack` | 111,974 | 2.7 % | 9.0 MB | 80 % `String` clone, 20 % `saved_args`/stack `Vec` |
| `WamState::restore_regs` | 78,329 | 1.9 % | 0.35 MB | register `Value` clones |
| `WamState::trail_binding` | 69,559 | 1.7 % | 0.40 MB | **`key.to_string()`** (register-name trail tag) |
| `WamState::get_reg_raw` | 65,917 | 1.6 % | 0.29 MB | value clones |
| `WamState::resume_builtin` | 57,432 | 1.4 % | 6.9 MB | nondet builtin state (`saved_args`, `BuiltinState`) |
| `StackEntry::clone` | 27,749 | 0.7 % | 2.55 MB | Arc COW frame copy (bounded — sharing works) |
| `WamState::bind_var` | 13,796 | 0.3 % | 0.22 MB | **`format!("__binding__{}")` + insert** |
| `value::Value::clone` | 17,934 | 0.4 % | 0.08 MB | atomic `String` clones |

`deref_heap` + `deref_var` alone = **63.3 % of every allocation in B2.** The
`Arc` frame stack and `Args` sharing hold their weight: `StackEntry::clone` is
0.7 % of blocks / 1.9 % of bytes, and `save_regs`' snapshot never reaches the top
list — the coordinator's "already done" mechanisms are confirmed. The leftover
churn is the deref subsystem and the string-tagged trail.

### Top allocation sites by BYTES

`deref_heap` 49.9 MB (36.9 %) · `step` 39.9 MB (29.5 %, term Vecs) · `put_reg`
10.0 MB (7.4 %) · `backtrack` 9.0 MB (6.7 %) · `resume_builtin` 6.9 MB (5.1 %) ·
`deref_var` 4.0 MB (3.0 %) · `StackEntry::clone` 2.55 MB (1.9 %) · JSON load
(`Parser::value`) 2.0 MB (1.5 %). By bytes the picture is deref + term
construction; by count it is deref + tiny name-Strings.

### B2 self-Ir (callgrind, 2.219 B Ir) — the instruction view

Memory-management family = **48.1 % self-Ir**: `_int_malloc` 13.88 %, `_int_free`
12.05 %, `malloc` 7.88 %, `free` 5.16 %, `malloc_consolidate` 4.11 %,
`unlink_chunk` 1.88 %, `__rdl_alloc` 1.66 %, alloc-shim 1.48 %. Plus
`drop_in_place<Value>` 7.04 %, `String::clone` 3.95 %, `memcpy` 2.68 % →
**memory subsystem ≈ 61.8 %** (matches D91's 61.5 %). Non-allocator hot Rust
functions: `deref_heap` self **9.52 %**, `deref_var` **3.57 %**, functor "/"
parse (`CharSearcher::next_match_back` 2.17 % + `memrchr` 1.38 % + `functor_of`
2.13 %) **= 5.68 %**, `same_cell` 1.60 %, `restore_regs` 1.27 %. **D1 read misses
concentrate in `deref_var` (11.8 %), `memcpy` (13.0 %), `drop_in_place` (8.3 %+9.2 %)
and `String::clone` (5.6 %)** — the deref/clone/free churn — but they stay in
L2/L3 (LL ~0 %).

### B2 deref call-site attribution (callgrind `cfn`/`calls`)

`deref_heap` (1.76 M attributed calls) is **88.6 % self-recursion** — the
recursive descent into every sub-term of a term — with top-level entries from
`resume_builtin` (the nondet search driver), `execute_term_builtin`,
`term_compare` and `deref_list_arg`. `deref_var` (1.86 M calls) is **83.8 %
called from inside `deref_heap`** (the `self.deref_heap(&self.deref_var(a))` at
`state.rs:4689`/`4720`), then `get_reg` (4.2 %) and `step` (1.2 %).

**What this means:** each top-level deref of a term walks and clones the *entire*
tree, calling `deref_var` (a `String`-cloning hop) then `deref_heap` per element,
and it is re-invoked on the *same* terms across backtracks. For an already-ground
term this whole walk is waste. This is the memoization / ground-bit candidate:
`deref_heap` on a term known to be ground should be an O(1) identity.

## B3 — allocation sites (5 k `resolve_layered`; DHAT: 8,481,071 blocks / 111.8 MB)

### Top allocation sites by BLOCK COUNT

| site | blocks | % of all blocks | bytes | dominant kind |
|---|---:|---:|---:|---|
| **`WamState::deref_heap`** | **3,074,510** | **36.3 %** | 15.0 MB | **93 % `functor.to_string()`**, 7 % spine, 0 % Vec |
| **`WamState::deref_var`** | **2,938,770** | **34.7 %** | 6.9 MB | **100 % `String` clone** |
| **`WamState::display_functor_name`** | **507,458** | **6.0 %** | 0.51 MB | `name.to_string()` (functor normalise in comparator) |
| `WamState::step` | 347,509 | 4.1 % | 24.7 MB | term `Vec` (biggest by bytes) |
| `region_key_dep_rows_dispatch` | 210,056 | 2.5 % | 7.1 MB | native key-row build |
| **`WamState::term_compare`** | **172,458** | **2.0 %** | 0.77 MB | **100 % `String`** (atom-name clones for compare) |
| `json::Parser::value` (load) | 131,169 | 1.5 % | 9.3 MB | load-time, not resolve |
| `region_group_keyed_dispatch` | 57,613 | 0.7 % | 4.5 MB | native group |
| `put_reg` | 43,361 | 0.5 % | 3.2 MB | register writes |
| `backtrack` | 24,729 | 0.3 % | 1.4 MB | (B3 barely backtracks) |

`deref_heap` + `deref_var` = **71 % of every allocation in B3**, and the
interning-addressable set — `deref_var` (100 % String) + `deref_heap`'s functor
String (93 % of its 3.07 M) + `term_compare` (100 % String) +
`display_functor_name` — is **≈ 6.46 M of 8.48 M blocks = 76.2 % of ALL B3
allocations.** These are *tiny* Strings (avg 2–4 bytes: `"-"`, `"s"`, package
names) which is why they are only ~26 % of *bytes* but drive the `malloc`/`free`
call count that is ~36 % of self-Ir.

### B3 self-Ir (callgrind, 4.184 B Ir)

Memory family ≈ **34–38 %** (`_int_free` 12.08 %, `malloc` 8.52 %, `_int_malloc`
5.83 %, `free` 5.65 %, `__rdl_alloc` 1.82 %, + sub-threshold consolidate/unlink)
+ `drop_in_place<Value>` 6.82 % + `String::clone` 5.43 % + `memcpy` 3.05 % →
**memory subsystem ≈ 49–51 %** (matches D91's 49.3 %). Then: `deref_heap` self
**13.75 %** (`'2` 10.46 % + 3.29 %), functor "/" parse **`memrchr` 5.02 % +
`CharSearcher` 5.01 % = 10.03 %**, `deref_var` **5.44 %**, `same_cell` 3.24 %
(also strips `"f/N"`). **`_int_malloc` owns 60.7 % of all LL write-misses** —
allocation write-traffic — confirming the miss cost is allocation-count-driven.

### B3 deref + msort call-site attribution — the decorate-sort case

`deref_heap` (5,367,136 calls): 57.4 % self-recursion, 23.3 % direct entry,
**15.9 % from `term_compare`** (12.7 % `'2` + 3.2 %) — and by inclusive-Ir
`term_compare` drives **36.4 % of deref_heap's cost**, the single biggest
external driver.

`term_compare` (425,789 top-level calls): 59.5 % self-recursion (nested compound
compare), 20.3 % direct, and the sort machinery:

| caller of `term_compare` | calls | % calls | inclusive Ir | % of B3 total Ir |
|---|---:|---:|---:|---:|
| `driftsort::sort` (the `sort_by` comparator) | 61,448 | 14.4 % | **1.881 B** | **45.0 %** |
| `Vec::dedup_by` (the `sort/2` dedup) | 22,535 | 5.3 % | **0.695 B** | **16.6 %** |
| smallsort / quicksort / pivot | ~2,250 | 0.5 % | ~0.055 B | 1.3 % |

**The sort comparator path is ≈ 2.58 B of 4.184 B Ir = 61.6 % of all of B3** —
this *is* B3, and it reproduces D91's "`term_compare` 63 % inclusive". The
builtin at `state.rs:13354` (`"msort/2" | "sort/2"`) does:

```rust
let mut sorted: Vec<Value> = list.iter()
    .map(|v| self.deref_heap(&self.deref_var(v)))   // (A) pre-deref ONCE — O(n), already correct
    .collect();
sorted.sort_by(|a, b| self.term_compare(a, b));      // (B) but term_compare RE-derefs a and b
if op == "sort/2" { sorted.dedup_by(|a, b| self.term_compare(a, b) == Equal); }
```

and `term_compare` (`state.rs:13071`) opens with:

```rust
let da = self.deref_heap(&self.deref_var(a));   // redundant — a is already derefed by (A)
let db = self.deref_heap(&self.deref_var(b));
```

So every one of the ~84 000 comparisons re-walks and re-clones both operands
(and, at each `Str` node, calls `display_functor_name` → a fresh functor
`String`), even though (A) already produced fully-derefed values. The
pre-deref is O(n); the re-deref is O(n log n).

## The three levers as concrete target lists

### Lever 1 — Intern functors + atoms + var-names to `u32`

**Owns:** ~37 % of B2 alloc blocks, **76 % of B3 alloc blocks**; directly the
functor-"/" parse (**10.0 % B3 / 5.7 % B2 self-Ir**) and a large share of the
`String::clone` (4–5 % self-Ir) and `malloc`/`free` (36–48 % self-Ir) buckets.

**Exact sites to change:**
- `value.rs`: the `Value` enum — `Atom(String)`, `Str(String, Args)`,
  `Unbound(String)` become `u32`-keyed (interned symbol id). This is the root.
- The functor decoders that vanish entirely once the key is an id:
  `functor_of` (`state.rs:4371`), `display_functor_name` (`:12980`),
  `is_cons_functor` (`:2311`) and the **9 `rfind('/')`/`rsplit_once('/')` parse
  sites** — replaced by integer compares / a small id table.
- `deref_heap` (`:4665`): drop `functor.to_string()` (the 93 %-of-its-allocs
  cost in B3, 28 % in B2); `deref_var` (`:2322`): its clone becomes `Copy`.
- `term_compare` (`:13071`) atom/functor compare → integer compare;
  `bindings: HashMap<String, Value>` → `HashMap<u32, Value>`.
- **Blast radius (the thread-through):** `Value::Atom` is constructed at **~425
  sites** and `Value::Str`/`strv` at **~230**, the *majority inside the generated
  `lib.rs`* (4 791 `to_string()` calls in generated code). Interning therefore
  **must be done in the code generator** (`wam_rust_target.pl`), not by editing
  `lib.rs`, plus the runtime (`value.rs`/`state.rs`) and the JSON shim boundary
  (`shim/main.rs` `s()`/`atom()` intern at parse). A de-intern table is needed
  for `Display`/output. The `intern_atom`/`atom_intern: HashMap<String,u32>`
  machinery already exists (used today only for the FFI graph kernels, which
  measured ~7.9× there) — extend it to the `Value` representation.
- **Risk:** highest blast radius of the three; touches the generator, the
  runtime and every term-construction path. Do it as a generator change with the
  shim interning at the boundary so the id space is single-sourced.

**Expected ceiling:** removes the 10 % B3 / 5.7 % B2 parse self-Ir outright,
plus the name-String share of `malloc`/`free`/`String::clone`/`drop`. Because
those name-Strings are 76 % of B3 allocations and the cost is instruction-bound,
a **B3 malloc/free reduction on the order of half its 36 % self-Ir**, i.e. a
**~15–25 % B3 total** win, is defensible on the DHAT counts alone (before any
overlap with lever 3). **B2 ~11 % direct** (parse + name-String clones), matching
D91's estimate, with a secondary alloc-count cut on top.

### Lever 2 — Cut the `Value` churn the existing sharing does not cover

The `Arc<Vec<StackEntry>>` stack and `Args` structural sharing are confirmed
carrying their weight (see B2 table). The uncovered sites:

- **2a. `deref_heap` spine rebuild + re-deref (`state.rs:4665`).** Its 48 MB /
  1.04 M B2 allocations are fresh `Arc<Vec<Value>>` spines built when *any*
  sub-term derefs to a new cell (`Value::strv`/`Value::list` at `:4714`/`:4735`),
  and it is re-run on the same terms across the 479 531 backtracks. **Target:** a
  ground/resolved bit on terms (or a small deref memo keyed by cell identity) so
  a ground term's `deref_heap` is an O(1) identity returning the same `Args`
  (no new spine, no clone). Owns the largest non-allocator self-Ir in both
  workloads (`deref_heap` self 9.5 %/13.8 %) and feeds the allocator above it.
  **Moves both.**
- **2b. String-tagged trail / binding table (`bind_var` `:2465`,
  `trail_binding` `:4331`, `unwind_trail_to` `:2480`).** `bind_var` does
  **`format!("__binding__{}", var_name)` + `bindings.insert(var_name.to_string(),
  …)` = 2 `String` mallocs per bind**, `trail_binding` a `key.to_string()` per
  register trail. These are pure encoding overhead: the discriminant "is this a
  binding or a register entry" is carried as a **string prefix**. **Target:**
  make `TrailEntry.key` an enum (`Binding(u32) | Reg(u16)`) instead of a
  `String`; with lever 1 the var id is already a `u32`. Removes the
  `trail_binding` (69.5 K) + `bind_var` (13.8 K) + Y-reg `put_reg` (22.6 K)
  block sites and the `format!`/`to_string` on the hot bind path. **Moves B2**
  (4.6 M trail bindings / 479 K backtracks); negligible for B3.
- **2c. `deref_var` clone (`state.rs:2322`).** Returns a cloned `Value`
  (String clone) on every hop — 24 % of B2 / 35 % of B3 allocation blocks.
  Subsumed by lever 1 (the clone becomes `Copy`); until then, the by-reference
  `deref_chain` (`:4347`) already exists and more callers could use it.

**Expected ceiling:** 2a attacks the top non-allocator function and its
allocator tail in both workloads; 2b is a focused B2 bind-path win (small bytes,
but on the hottest B2 path). Independently shippable.

### Lever 3 — Decorate-sort the B3 `msort`/`sort` builtin

**Owns:** the sort comparator path = **≈61.6 % of all B3 Ir**
(`driftsort`→`term_compare` 45 % + `dedup_by`→`term_compare` 16.6 %).

**Exact change (one comparator, `state.rs:13354`–13367):** the elements are
already pre-derefed by the `.map(deref_heap(deref_var))` at `:13359`. Add a
`term_compare_derefed(a, b)` variant that **skips the two `deref_heap(deref_var())`
at `term_compare`'s head** (`:13073`–13074) and is used by the `sort_by`/`dedup_by`
closures. That removes ~2 top-level `deref_heap` walks per comparison — i.e. the
bulk of the 5.37 M `deref_heap` calls, the 2.94 M `deref_var` calls, and the
per-`Str`-node `display_functor_name` Strings — replacing O(n log n) re-deref
with the O(n) pre-deref already present. (Also apply to `sort/4` `:13369`,
`keysort/2` `:13428`, and `@<`/`compare/3` which call `term_compare` on raw
regs — those legitimately need one deref, so keep the deref there and only skip
it under the sort's already-derefed keys.)

**Expected saving:** the redundant re-deref is `~(1 − n/(2·comparisons))` of the
comparator's deref work; with ~84 000 comparisons over the sorted rows that is
**>90 % of the deref/alloc performed inside the comparator eliminated**. Since
that comparator is ~61 % of B3 and roughly two-thirds of its cost is the
head-deref + the functor-String it triggers, decorate-sort alone plausibly
removes **~30–40 % of B3 total Ir**; **combined with lever 1** (which turns the
residual functor compares into integer compares) the two together plausibly
**cut B3's dominant sort cost by half or more**, consistent with D91's "2c + #1
plausibly halve B3". **B2 barely sorts — ~0 there.**

## Ranked worklist for the optimization rounds

Ordered by expected impact per unit risk, with grounded ceilings.

| # | target (site) | workload | measured share it owns | expected ceiling | risk / blast radius |
|---|---|---|---|---|---|
| **1** | **Decorate-sort `msort`/`sort` comparator** (`state.rs:13354`, `13071`) | **B3** | comparator = **61.6 % of B3 Ir**; removes the redundant re-deref | **~30–40 % B3** alone | **Low** — one new comparator fn, elements already pre-derefed; keep raw-deref for `@<`/`compare/3` |
| **2** | **Intern functors+atoms+var-names → `u32`** (`value.rs` enum; generator `wam_rust_target.pl`; shim boundary) | both | **76 % B3 / 37 % B2 alloc blocks**; parse 10 %/5.7 % self-Ir | **~15–25 % B3, ~11 % B2** (overlaps #1 on B3) | **High** — 400+ construction sites in *generated* code ⇒ generator change; needs de-intern for output |
| **3** | **`deref_heap` ground-bit / memo** (`state.rs:4665`) | both | `deref_heap` self 9.5 % B2 / 13.8 % B3 + its 48 MB spine rebuilds | high but bounded; feeds the 48–61 % allocator | **Medium** — correctness of the ground invariant under backtracking |
| **4** | **Enum-tag the trail (`TrailEntry`), drop `format!`/`to_string`** (`state.rs:2465`, `4331`, `2480`, `360`) | **B2** | trail/bind name-String sites ~106 K blocks on the 4.6 M-bind path | main focused B2 bind-path win | **Low–Medium** — local to trail/bind; pairs with #2's var ids |
| **5** | Store/seek default for large catalogs (D89, ~10× at scale) | B3 | architectural, not codegen | ~10× at scale | Project decision |
| **6** | More genrec (declined predicates) | — | dispatch = 3.3 % B2 / 0.3 % B3 | **≤3 % B2, ~0 % B3** | do not lead with this (D91) |

**Sequencing note.** Do **#1 first** — it is low-risk, single-site, and owns the
largest single B3 share. Then **#2** (interning), whose B3 win overlaps #1's
residual functor cost but whose B2 win and alloc-count reduction stand alone; land
it as a generator change so the id space is single-sourced through the shim. **#3**
(deref ground-bit) and **#4** (trail enum) are the independent alloc-churn cuts
that #2 does not fully cover. **#1+#2+#3 together** target the B3 stack end-to-end
(sort comparator → functor ids → ground deref); **#2+#3+#4** target the B2 stack
(name-String churn → ground deref → trail tags).

## Appendix — key raw figures

**B2 DHAT (60-case):** 4,136,815 blocks / 135.2 MB. deref_heap 1,620,878 blk
(39.2 %) / 49.9 MB; deref_var 998,406 blk (24.1 %, 100 % String) / 4.0 MB; step
503,008 blk / 39.9 MB; put_reg 123,600 / 10.0 MB; backtrack 111,974 / 9.0 MB;
trail_binding 69,559; bind_var 13,796; StackEntry::clone 27,749 / 2.55 MB.
**B2 callgrind:** 2.219 B Ir; LL miss rate 0.0 % (80,238); mem family 48.1 %
self-Ir; deref_heap self 9.52 %; deref_var 3.57 %; functor parse 5.68 %.
deref_heap 88.6 % self-recursive; deref_var 83.8 % from deref_heap.

**B3 DHAT (5 k):** 8,481,071 blocks / 111.8 MB. deref_heap 3,074,510 blk
(36.3 %, 93 % functor String) / 15.0 MB; deref_var 2,938,770 blk (34.7 %, 100 %
String) / 6.9 MB; display_functor_name 507,458; step 347,509 / 24.7 MB (biggest
bytes); term_compare 172,458 (100 % String). Interning-addressable ≈ 6.46 M blk
= 76.2 %. **B3 callgrind:** 4.184 B Ir; LL miss rate 0.0 % (1,081,424; 60.7 % in
`_int_malloc` writes); mem subsystem ≈ 49–51 % self-Ir; deref_heap self 13.75 %;
functor parse (memrchr+CharSearcher) 10.03 %; deref_var 5.44 %. Sort comparator
(driftsort 1.881 B + dedup_by 0.695 B) = 2.576 B Ir = **61.6 % of B3**.
