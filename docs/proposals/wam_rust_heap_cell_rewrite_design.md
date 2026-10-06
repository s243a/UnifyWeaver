<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: heap-cell runtime rewrite, design and plan

**Date:** 2026-10-04, revised 2026-10-05 after external review, revised
2026-10-06 after a second external review. **Ledger:** D128.
**Status:** Design. The rewrite is decided; this document sets its
representation, API and order. §17 maps each finding of the first external
review to the sections that resolve it; §18 does the same for the second
(a code-completeness sweep). One decision is left to the project owner: fix
the confirmed baseline divergences from SWI-Prolog on main before the
rewrite (phase R−1, §12.1, recommended) or emulate them in the rewrite.
**Evidence:** `docs/reports/wam_rust_heap_cell_spike.md` (baseline profile,
three spike variants, gates). Its numbers are cited below as **[S]**. Code
citations are `file:line` at commit `e095a00` (runtime sources identical to
`7da1210` and to `d0c1c06`). `RT` is `src/unifyweaver/targets/rust_target.pl`,
`PA` is `templates/targets/rust_wam/par_aggregate.rs.mustache`. `T` is
`src/unifyweaver/targets/wam_rust_target.pl`, `ST` is
`templates/targets/rust_wam/state.rs.mustache`, `DB` is
`templates/targets/rust_wam/dynamic_db_methods.rs.mustache`, `LE` is
`src/unifyweaver/targets/wam_rust_lowered_emitter.pl`, `WT` is
`src/unifyweaver/targets/wam_target.pl`.

## 1. Summary

The Rust WAM target is ~17× slower than SWI-Prolog at N=40 (4.52 vs 0.26 ms)
and ~3.9× slower at N=5000 (68.9 vs 17.5 ms) on the transpiled package
resolver. The measurements say two separate things are wrong. The rewrite has
to fix both, and in a particular order.

1. **Fixed cost: interpreter mechanics.** A resolve runs ~21.7 K WAM
   instructions at every N, and each costs ~1,075 Ir **[S]**. That cost is in:
   - register access by `String` name, plus a register trail that nothing
     reads on this workload (10,790 entries per resolve, 0 consumed);
   - choice points that clone ~10 registers when ~3 are live;
   - read/write unification contexts kept on a copy-on-write `Arc<Vec<StackEntry>>`;
   - term construction through a scratch heap that ends with a scan of 200
     registers;
   - string-keyed label lookups.

   Three naive spike levers together cut 17.6% of Ir and 25% of warm wall
   time at N=40, all byte-identical **[S]**. That figure is cumulative over
   s1–s3; §12 gives what each pre-rewrite change can claim on its own.
2. **Slope: term representation.** All of the N-dependent cost is in builtins
   (55%) and the native Stage-2 regions (35.5%) working on `Arc`'d `Value`
   trees. Within that, `deref_heap` materialization is 59% inclusive,
   `term_compare` 29%, `Arc` drop 21%, malloc 14% and `"f/N"` functor
   decomposition 12% **[S]**. No change to variables, registers or choice
   points reaches this. Only a term representation that builtins can read in
   place, with integer functor ids and no refcounts, does.

**Decisions.**

- **Terms live in a WAM heap of 8-byte tagged `Cell`s.** Variables are heap
  cells. Structures and lists are heap blocks. Atoms, small integers and
  functors are immediates or ids.
- **`Value` stays as the boundary type**, with two strictly separated
  forms (§7.3, §9.2): a *live bridge handle*, valid only inside one bridged
  builtin call on one machine, and a *detached term*, which holds no heap
  handle and is the only form any holder may keep across a rollback.
- **Registers become numeric operands** emitted by the generator.
- **No A/X register trail.** Each rollback site declares the registers it
  restores (§6.3).
- **Conditional trailing on heap addresses and on environment slots**, with
  the thresholds HB and EB derived from an explicit stack of *rollback
  obligations*: every live choice point and every active non-CP scope
  (§5.2). There is no free-standing `Mark`.
- **Scopes have separate `rewind` and `close`.** Rewinding keeps a scope's
  protection, so a lowered multi-clause snapshot can be rewound once per
  clause (§5.3).
- **Choice-point boundaries are generations, not depths.** Every CP gets a
  generation number at push. Scopes, B0, ITE levels and the backtrack floor
  record generations, so a cut through a scope cannot leave a stale boundary
  (§5.2, §5.4 counterexample 3).
- **Each scope and choice point carries an execution snapshot** (E,
  protected environment top, continuation, cut state, register policy), not
  only heap and trail tops (§6.3).
- **Heap-top reset on backtrack.**
- **Clause choice points save only the arity's argument registers.** ITE
  guards, aggregate frames and builtin choice points save the compiler's live
  set, or today's dirty set until the compiler provides it (§5.5).
- **Environments are a plain stack protected by the youngest obligation**,
  as in the standard WAM, plus conditional trailing of slot writes into
  protected frames (§6.2).
- **ITE barrier semantics are carried over exactly**: named levels on the
  guard CP, the pending level, both capture shapes (§6.4).
- **Term operations are separate**: compatibility standard order, strict
  identity, unification, fresh copy (§7.2). The comparator reproduces today's
  quirks; corrections are separate behavior changes.
- **Read and write mode use the S register.**
- **Baseline divergences are fixed first (recommended, owner's decision).**
  The second external review confirmed, against SWI-Prolog, behaviors where
  today's runtime is wrong. The rewrite's heap invariant needs the
  backtrack floor anyway. Phase R−1 fixes them on main as small separate
  changes and re-freezes the byte-identity baseline (§12).

## 2. What the measurements rule in and out

| claim | evidence **[S]** | consequence |
| --- | --- | --- |
| The named-variable map is not the problem | heap-cell variables alone: −1.7% at N=40, +0.2% at N=5000 (an extra enum variant de-inlined hot helpers) | Variable cells only pay off as part of a full cell representation. Do not ship them alone, and do not run a dual representation in the rewrite |
| The register trail is dead weight on this workload | 10,790 entries per resolve, 0 consumed by `unwind_trail_to`; disabling it kept every gate byte-identical; −1.88 M Ir (−8.1%) on top of s2 | A/X registers are not trailed in the rewrite. Before that lands on main, every rollback site is audited and given explicit register saves where it needs them (R0b, §12). The spike did not prove this in general |
| Choice points over-save | 383 clause CPs per resolve save 10.25 dirty slots where 2.78 are live; arity-only save −1.83 M Ir (−7.9%) on top of s1, byte-identical, lib 260/260 | The generator emits arity on clause `try_me_else` and a live set on ITE guards (§5.5) |
| The copy-on-write stack is a floor | after s3: `make_mut` 0.92 M + `StackEntry` clone 0.50 M + drop 0.35 M = 9.2% of N=40 | No `UnifyCtx`/`WriteCtx`: the S register and a mode flag. Environments in a plain stack (§6) |
| The slope is the term representation | N=5000: builtins 213.7 M + regions 138.0 M + `reset_query` 19.1 M = 95% of 388.5 M; mechanics ~17.6 M, the same as at N=40 | Hot builtins must read and build heap cells natively for the slope to move (R4). Regions are the other 35.5% (R5, §12) |
| The interpreter count is flat in N | the same steps, backtracks, CPs, binds and trail entries at N=40 and N=5000 | Mechanics work sets the fixed cost; builtin and region work sets the slope |

The spike measured where cost is today. It did not measure what the same work
costs on cells. §14's ranges are targets, and §12's gate G0 measures the two
largest of them before R2 is written.

## 3. Target representation

### 3.1 `Cell`: 8 bytes, `Copy`, no `Drop`

```rust
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Cell(u64);   // low 3 bits = tag, high 61 bits = payload
```

| tag | name | payload | notes |
| ---: | --- | --- | --- |
| 0 | `REF` | heap address (u32 in practice) | a bound or aliased variable reference; registers and Y slots hold `REF`s to unbound variables, never `VAR`s. Two reserved addresses are sentinels for registers and Y slots only: `UNINIT` and `ABSENT` (§6.2) |
| 1 | `VAR` | `(n << 4) \| kind` | an **unbound** variable, stored only in the heap. `kind` indexes a fixed prefix table and `n` is the number printed in the name (§3.4). It prints exactly the name today's runtime prints, so variable text and the standard order of variables are byte-identical by construction |
| 2 | `ATOM` | atom id (the existing interner's `u32`) | `[]` is the single atom `[]`. The `List([])`/`Atom("[]")` aliasing goes away |
| 3 | `INT` | 61-bit signed | canonical: an integer that fits is always `INT`, never a `BOX` |
| 4 | `STR` | heap address of a `FUN` header | the arguments follow the header |
| 5 | `LIS` | heap address of the head cell; the tail is at +1 | the only cons form. The `"[\|]/2"` vs `"./2"` vs `Value::List` aliasing goes away |
| 6 | `FUN` | functor id | a header cell, only inside the heap. `functors: Vec<(AtomId, u32 arity)>` is interned like atoms, so there are no `"f/N"` strings and no `decomp` |
| 7 | `BOX` | `(index << 1) \| arena`: arena 0 indexes the machine's `boxes: Vec<Boxed>`, arena 1 the program's immutable constant-box arena (§3.3) | `Float(f64)`, `BigInt(i64)` outside the 61-bit range, `Bool(bool)` (kept distinct from the atom `true` to preserve today's semantics). The machine arena's top is saved in every obligation and truncated on rollback; the constant arena is never truncated |

`deref(c)` follows `REF`s until it reaches a non-`REF` cell. For an unbound
variable it returns `REF(addr)` of the `VAR` cell, so callers always have the
address to bind. Binding writes the target cell at `addr`.

**Bind direction.** Today `unify` binds its *first* argument to its second
(`ST:6457-6464`: `(Unbound(n1), other) => bind n1`). That decides which name
survives in an output with aliased variables. The rewrite keeps
first-to-second. The standard younger-to-older rule would trail less, but it
can change printed names, so it is not used unless a gate proves it
identical. Any direction is sound when trailing follows §5.

### 3.2 Heap structures, not `Arc`'d trees

**Decision: everything the program builds lives in the heap.** Boundary inputs
are copied into a base segment at the bottom of the heap.

Why not keep `Arc` trees and only change variables and registers:

- The slope is the tree representation itself **[S]**:
  - `deref_heap` rebuilds terms to normalize `"f/N"` functor keys and to
    resolve bound variables inside spines (59% at N=5000);
  - `Arc::drop_slow` 21%, malloc 14% and `decomp` 12%;
  - `reset_query` spends 19.1 M Ir freeing the previous query's trees.

  Heap cells have none of these: a structure is a contiguous block, a functor
  is an id, freeing is `heap.truncate`, and a builtin reads arguments in place.
- Copying a register or saving a choice point becomes a `memcpy` of `Copy`
  cells, with no refcount traffic. At N=40 the spike's cumulative wall gain
  (−25%) exceeded its Ir gain (−17.6%) **[S]**. That is one small-workload
  observation. It says nothing about the wall/Ir ratio at N=5000, where the
  spike's wall numbers are noise.
- The cost is that boundary terms must be copied in. The shim already builds
  the catalog term on every call (`catalog_term`), so building cells directly
  instead of `Value`s should cost about the same: O(N) once per resolve, which
  SWI pays too. G0 (§12) measures it. A later option is to keep the base
  segment across resolves when the catalog is unchanged.

Rejected alternative: a hybrid where ground boundary terms stay as `Arc<Value>`
behind a `BOX` cell. It keeps two term models in every builtin and every
unify arm, which is exactly the dual-representation tax the spike measured
(de-inlining). It also leaves the sort-family slope where it is, because the
catalog lists *are* the boundary terms. Not worth it.

### 3.3 Heap layout and lifetime

```
heap: Vec<Cell>
  [0 .. base)      base segment: call copy-in (catalog, requests). Rebuilt per call.
  [base .. H)      query heap: grows by bump, reset to an obligation's h on rollback
```

`reset_query` sets `H = base` (O(1)). Nothing outside the heap may keep a heap
address across a rollback below it (§5.6, §9.2).

Boxes have the same two layers plus a permanent one:

```
const_boxes: Arc<[Boxed]>   program constants (Float, BigInt, Bool in instructions); built once at load,
                            shared by every machine of the program (parser machines, par workers, clones)
boxes: Vec<Boxed>
  [0 .. boxes_base)         boxes of the base segment (call copy-in). Rebuilt per call.
  [boxes_base .. top)       query boxes: reset to an obligation's `boxes` on rollback
```

`reset_query` sets the box top to `boxes_base`. Every obligation's `boxes` is
at least `boxes_base`, so no rollback removes a box that a base-segment cell
references. Instruction constants are pre-encoded cells (§4); a constant that
needs a box is a `BOX` with arena bit 1, so it stays valid across every reset
and rollback and on every machine that shares the code. Today the same
constants are `Value`s inside `Instruction` (`instructions.rs.mustache:12`,
`:26`, `:30`, `:44`, `:46`, `:134`) and need no lifetime rule. Export of a
constant box copies its value into the `Detached`, as for any box.

**Reset-surviving holders.** A value that survives `reset_query` on the same
machine must be exported before the reset and imported after it. The one
case today is the parser machine's operator table: `read_term` runs
`canonical_op_table/1`, reads `ops`, calls `parser.reset_query()`, then puts
`ops` into the parser's A2 (`T:3248-3256`). In the rewrite `ops` is exported
to a `Detached` before the reset and imported (`PreserveName`; the table is
ground) after it. The table depends only on the program, so a later
optimization may cache the `Detached` per program.

There is **no garbage collector** (as today, in effect: `Arc` frees eagerly, the
heap does not). Deterministic forward recursion that never backtracks keeps its
garbage until the query ends. For the resolver this is bounded by the work
done. For other programs it is a real risk (§13).

### 3.4 Variable names

Today every fresh variable is a `Value::Unbound(name)`. The name is the
variable's identity, its text and its standard-order key. The rewrite keeps
the exact text. These are all the prefixes the runtime generates
(`grep` of `T`, `ST`, `DB`, `LE`, `RT`, `PA`, the shims and the benchmark generators):

| kind | name | created by | counter convention |
| ---: | --- | --- | --- |
| 0 | `_V<n>` | `PutVariable` (`T:365`), `RecurseCategoryAncestorPc` (`T:628`), lowered emitter (`LE:978`) | post: `n = var_counter`, then `var_counter += 1` |
| 1 | `_H<n>` | `UnifyVariable` write mode (`T:286`), `SetVariable` (`T:431`) | post |
| 2 | `_L<n>` | `length/2` fresh list (`T:2490-2491`) | pre: `var_counter += 1`, then `n = var_counter` |
| 3 | `_F<n>` | `functor/3` fresh arguments (`T:2604-2605`) | pre |
| 4 | `_C<n>` | `copy_term_walk` (`T:2941`), used by `copy_term/2` (`T:2772`), `assert` (`DB:399`) and dynamic call/clause/retract renaming (`DB:534`, `DB:536`, `DB:561`, `DB:912`, `DB:994`) | pre, on `self.var_counter` |
| 5 | `_RP<n>` | `copy_external_term_from` (`T:3327`): `read_term` results, clause variable environments (`DB:283`) | pre |
| 6 | `_M<n>` | `fresh_meta_var` (`T:7245-7247`): maplist/foldl/predsort helpers | pre |
| 7 | `_MB<n>` | `raise_builtin_error` (`T:5698-5699`) | pre |
| 8 | `_EC<n>` | `raise_iso_error` (`DB:223-228`) | pre |
| 9 | `_SE<n>` | `raise_read_syntax_error` (`DB:235-236`) | pre |
| 15 | any other text | boundary names: `_uw_shim_out` (both shims), `__PAR_IN` / `__PAR_VAL` (`par_aggregate.rs.mustache:21-22`), `_N` (`main.rs.mustache:131`), `_A<i>` (fact-dispatch default, `T:9900`), `_RP_ops` / `_RP_term` / `_RP_env` (the separate parser machine, `T:3243-3258`), `__wrapper_target` / `__wrapper_cost` and `__agg_target` / `__agg_cost` (the six hybrid wrappers, `RT:4552-4553`, `RT:4696-4697`, `RT:4912-4913`, `RT:5035-5036`, `RT:5179-5180`, `RT:5314-5315`), `Hops` (`examples/benchmark/generate_wam_effective_distance_benchmark.pl:927`), the `ParAggregate` input fallback that names a variable after its register operand (`T:1053`), test names (`tests/fixtures/wam_rust_dynamic_builtins.rs`, in-template tests) | `n` is the atom id of the full text |

Encoding: `VAR` payload `(n << 4) | kind`, 4-bit kind, 57-bit `n`. The cell
stores the number that appears in the name, not the counter at creation, so
both conventions fit. Each creation site keeps its own increment order. A
`Value::Unbound` name imported from outside is parsed into `(kind, n)` only
when it is exactly a canonical generated name (prefix, then decimal digits with
no leading zero); everything else is kind 15.

Variable standard order compares the name **text** bytes, as today
(`T:5493-5498`), so `_V10` sorts before `_V9`. The comparator renders both
names into stack buffers; it never compares `(kind, n)` numerically.

Naming policy for external `Unbound(name)` on import: one name→cell map per
import session (§7.3). All occurrences of a name inside one session, across
all the arguments of that session, become one variable. Two sessions never
share a variable by name. Today a name *is* a global identity, so code that
relies on finding a variable again by name in a later call must keep the cell
instead. Those sites are: the shims' `OUT_VAR` read-back, par_aggregate's
`IN_VAR`/`VAL_VAR` read-back (`par_aggregate.rs.mustache:33`, `:56`, `:168`, `:193`),
the bench driver's `_N`, the six hybrid wrappers' 42 `vm.bindings` reads and
removes (`RT:4477-5348`, §8), and the effective-distance benchmark's `Hops`
read-back (`generate_wam_effective_distance_benchmark.pl:944`, `:967-968`). R3
rewrites each to hold the cell returned by the import. If an external name
happens to equal a live generated name, today the two alias and in the
rewrite they do not. No current shim or fact source produces such names.

**Helpers that rebuild a variable from its name.** Some runtime code finds an
existing variable again by building `Value::Unbound(name)` from a name it
collected. Today that *is* the variable. In the rewrite a name imported in a
new session is a new variable, so these helpers must return the existing
cells instead (§7.5). They are part of the migration inventory:

| helper | used by | rewrite |
| --- | --- | --- |
| `collect_term_variable_names` + `variables_from_term` (`DB:309-338`) | `term_variables/2` (`T:2799-2816`), `numbervars/3` (`T:2817-2861`), `read_term` `variables(_)` (`T:3285-3289`) | `term_vars(c) -> Vec<Cell>`: `REF`s of distinct unbound variables in first-occurrence order |
| `singletons_from_term`, `copied_variable_source_names` (`DB:340-387`) | `read_term` `singletons(_)` | count occurrences per variable cell in the imported term; map cell → source name through the import session |
| `copy_named_variables_from_env`, `copy_variable_names_from_env` (`DB:262-307`) | `read_term` `variable_names(_)` | the pair values are the cells the shared parser import returned |
| `unifiable/3` reading binding names off the trail (`T:2897-2915`) | `unifiable/3` | ordered bind-event capture (§7.5) |

Otherwise `term_variables(f(X,X),[V]), V=a` leaves `X` unbound, `numbervars`
numbers copies instead of the input, and `variable_names` stops sharing with
the parsed term.

`var_counter` restoration is per scope type and matches today exactly (§5.7).

## 4. Registers and instruction encoding (generator)

- **Operands are numbers.**
  - `enum Reg { A(u8), X(u8) }` packed as `u16` (A and X stay separate banks,
    because the shared `wam_target.pl` allocates them separately);
  - Y operands become `Y(u8)` slot indices;
  - the generator emits **separate variants** for X and Y forms
    (`GetVariableX`, `GetVariableY`, …), so no step arm tests the register
    kind at run time;
  - `regs: [Cell; 200]`, indexed directly.
- **Control is resolved at generation time.**
  - `Call{pc: u32, arity: u8}` and `Execute{pc}`;
  - `TryMeElse{alt: u32, arity: u8}` for clause alternatives;
  - `TryMeElseIte{alt: u32, live: LiveSet}` for ITE guards and disjunctions
    (today's `L_ite_else_*` labels);
  - `Jump{pc}`.

  The `labels: HashMap<String, usize>` map stays only for the public
  entry lookup. It cost 0.79 M per resolve on the hot path **[S]**.
- **ITE barriers get level ids.** `GetLevel{id: LevelId}` and
  `CutTo{id: LevelId}`, where `LevelId` is the Y number the compiler reserved
  (`WT:2309-2333`). It is only a name for the level; it never addresses a slot
  (§6.4).
- **Builtins get ids.** `BuiltinCall{id: BuiltinId, arity}` dispatches through
  a table. The assembled runtime has **154 grouped dispatch arms covering 216
  predicate keys** (core 4/4, arith 3/8, io 62/87, type 9/9, term 25/29,
  ext 42/58, meta 9/21), plus three dynamic-DB routes outside
  `execute_builtin` (`retract/1`, `clause/2`, `current_predicate/1`,
  `T:742-746`, `T:836-840`, `T:7456-7464`). The generator builds
  `BuiltinId` from a machine-readable inventory extracted from the arms, not
  from a hand list. An unknown name
  becomes a `CallUnknown{name}` that keeps today's `warn_unresolved_goal`
  behaviour.
- **Constants are pre-encoded.** `GetConstant{c: Cell, a: Reg}`. Atom ids
  must be stable: the generator emits atom *text*, and a load-time table
  interns it once into a `Vec<Cell>` indexed by constant number. Instructions
  hold that number, or the cell after load. A Float, out-of-range integer or
  Bool constant is a `BOX` in the program's constant arena (§3.3), never in a
  machine's rollback-truncated box arena.
- **Switch tables:** `SwitchOnConstant` becomes a sorted `Vec<(Cell, u32)>`
  or an `FxHashMap<Cell, u32>`. `SwitchOnStructure` is keyed by `FunctorId`.
- **One functor namespace for the process.** The functor table follows the
  existing `Sym` interner (`value.rs.mustache:27`): global, append-only,
  canonical. So a `FunctorId` (and an `AtomId`) means the same functor on
  every machine, and a `Detached` that carries one is valid on the parser
  machine, in `par_aggregate` workers and in machine clones.
- **Fused instructions** (`Cons`, `NotMember`, `ListLengthLt`,
  `RecurseCategoryAncestorPc`, `ReturnAdd1`, …) keep their meaning with
  numeric operands.
- **Generator check (R1):** every clause that uses a Y operand, other than a
  level id, has an `Allocate`. Today the compiler reserves the barrier Y after
  the environment decision (`T:913-929` comment), but with level ids that Y is
  no longer a slot. The check makes §6.3's "nested runs never write an outer
  frame" an enforced invariant.

Spike evidence for this bucket: register decode, the register file, the
register trail and labels are ~18.5% of N=40, plus `step` self at ~76 Ir per
instruction **[S]**.

## 5. Trail, rollback obligations and backtracking

### 5.1 Trail entries

```rust
pub struct TrailEntry { loc: u32, old: Cell }   // loc: heap address, or env slot index with a high tag bit
```

A heap binding at `addr` pushes an entry **only if `addr < HB`**. An
environment slot write at slot index `k` pushes an entry **only if `k < EB`**.
HB and EB are derived from the rollback obligations (§5.2). Undo writes `old`
back. Storing the old `VAR` cell keeps the variable's kind and number, and so
its name, without a side table. An address-only heap trail (8 bytes smaller)
would need the name re-derived; decide by measurement in R2 (§16). The spike
skipped 36% of binds with a far more conservative mark **[S]**.

**A/X registers are never trailed.** Backtracking restores the saved
registers from the CP. A non-CP scope restores the registers its policy names
(§6.3). Undo runs before truncation: entries are undone in reverse order while
their addresses are still below the current H, then H is truncated.

An env-slot entry is undone only if its slot is below the target
obligation's `e_top`. A slot at or above it is dead after the rollback, and it
may belong to a frame allocated after the entry was written (§6.3, "Env-slot
entries of abandoned frames"). The paranoid build counts skipped entries.

### 5.2 Rollback obligations, HB and EB

A **rollback obligation** is anything that may later restore the machine to
an earlier state. There are two kinds:

- every live **choice point** (`cps`), and
- every active **non-CP scope** (`scopes`): `\+`, meta-calls, comparator
  calls, `catch/3`, trial unifications, lowered clause snapshots, lowered ITE
  conditions, region P2 snapshots, dynamic-DB and fact-table attempts.

Each obligation records `h`, `tr`, `boxes` and `e_top` (the protected
environment extent, §6.1). The machine keeps the two stacks separately:

```rust
cps:    Vec<ChoicePoint>   // youngest last; each carries gen: u64
scopes: Vec<ScopeRec>      // youngest last; ScopeRec { h, tr, boxes, e_top, cp_gen: u64, exec: ExecSnap, serial }
next_gen: u64              // generation of the next CP pushed; never reused within a query
hb: u32, eb: u32           // derived caches
fn derive(&mut self) {
    self.hb = max(self.cps.last().map_or(0, |c| c.h),     self.scopes.last().map_or(0, |s| s.h));
    self.eb = max(self.cps.last().map_or(0, |c| c.e_top), self.scopes.last().map_or(0, |s| s.e_top));
}
```

**CP generations.** Every pushed CP takes `gen = next_gen` and increments it.
Generations increase strictly from the bottom of `cps` to the top, because
CPs are only pushed on top and only removed from the top or by truncation.
`reset_query` sets `next_gen` to 0. Every boundary that today is a CP depth
is a generation in the rewrite: a scope's `cp_gen` (the `next_gen` at open),
B0, `pending_b0`, ITE levels and the backtrack floor (§5.5, §6.3, §6.4).
"Remove CPs above boundary `g`" means "pop while `top.gen ≥ g`", the same
cost as `truncate`. A depth says how many CPs were older; a generation says
which CPs were created after the boundary. The two agree until a cut removes
a CP below the boundary (a cut "through" it, L2). After that the depth is
stale and the generation is not (§5.4, counterexample 3).

**Invariant I1 (monotone tops).** Within each stack, `h` and `e_top` never
decrease from oldest to youngest. An obligation is opened at the current H,
and H only drops to the `h` of a surviving obligation. So the maximum over all
surviving obligations is the maximum of the two tops, and `derive` is O(1).

**Invariant I2 (boundaries).** For an active scope `s`, every CP with
`gen < s.cp_gen` was pushed before `s` opened, so its `h ≤ s.h`. Hence after
`rewind(&s)` removes every CP with `gen ≥ s.cp_gen`, no surviving CP has an
`h` above the rewound H. A depth boundary cannot give this after a cut
through `s`.

**Every transition recomputes HB and EB from what survives.** Nothing sets HB
from a saved copy.

| transition | obligations | HB, EB |
| --- | --- | --- |
| push a CP (clause, ITE guard, aggregate frame, builtin, fact/dynamic/foreign) | push onto `cps` with `gen = next_gen++` | derive (= current H, E top) |
| `retry_me_else` | unchanged | unchanged |
| `trust_me` | pop the top CP | derive |
| backtrack to CP `c` | rewind to `c` (§5.5); `c` stays | derive |
| builtin redo | rewind to `c`, pop `c`, derive, then resume (§5.6) | derive after the pop |
| any cut (`!/0`, `CutTo`, `CutIte`, cut in a meta-call, `catch/3`, `\+`, aggregates, `lowered_dispatch`) | remove CPs with `gen ≥` the target generation; **never removes a scope** | derive |
| open scope | push onto `scopes` with `cp_gen = next_gen` | derive (= current H) |
| `rewind(&s)` | remove CPs with `gen ≥ s.cp_gen`; `s` stays | derive (= `s.h`) |
| `close(s)` | pop `s` (must be the top scope) | derive |
| `push_cp_keep(&s)` | push a CP with `s`'s tops; `s` stays | unchanged |
| `push_cp_from(s)` | pop `s`, push a CP with `s`'s tops | unchanged |

**LIFO rules.**

- **L1.** Scopes close and rewind in LIFO order among scopes. A scope is a
  token owned by one Rust frame. Nested scopes belong to inner frames and are
  closed first. `rewind(&s)` and `close(s)` check (debug) that `s` is the top
  scope.
- **L2.** CPs may interleave with scopes freely. A cut removes CPs only, even
  CPs older than an active scope (a cut "through" a scope). A scope survives
  the cut and keeps HB at its own `h` or higher. Because its boundary is a
  generation, the cut does not change which CPs the scope owns. `close(s)`
  may leave CPs that were pushed inside `s`; they keep their own protection.
  A policy that restores E on close (§6.3) first removes them, because they
  would resume into frames the close drops.
- **L3.** While a scope is active, backtracking never resumes a CP older than
  that scope: a CP may be resumed only if `c.gen ≥ floor`. A scope that runs
  nested execution sets the floor to its `cp_gen` (today `call_goal_once`
  sets a depth floor, `T:7262-7271`). A CP pushed after a cut through the
  scope has a generation above the floor, so its redo is allowed. The
  paranoid build asserts L3 on every backtrack. Today several nested-execution
  sites set no floor (§12 R−1a).

L3 is what makes I1 hold for scopes. Without it, backtracking to an older CP
would set H below an active scope's `h`.

The paranoid build also counts "boundary crossed" events: a cut that removes
a CP with `gen < s.cp_gen` for some active scope `s`. Under today's depth
boundaries those are the only cases where the rewrite's behavior can differ
from the frozen build (§5.4, §10).

### 5.3 Scope API: open, rewind, close

```rust
#[must_use] pub struct Scope { idx: u32, serial: u32 }   // not Copy, not Clone
fn open_scope(&mut self, p: ScopePolicy) -> Scope;  // captures tops + exec snapshot per policy; derive
fn rewind(&mut self, s: &Scope);                    // undo trail to s.tr, H = s.h, boxes = s.boxes,
                                                    // remove CPs with gen >= s.cp_gen, restore exec snapshot per
                                                    // policy (incl. var_counter only if the policy says so);
                                                    // s stays active: HB, EB still >= s's tops
fn close(&mut self, s: Scope);                      // end s keeping current bindings (commit); derive
fn rewind_close(&mut self, s: Scope);               // rewind(&s) then close(s)
fn push_cp_keep(&mut self, s: &Scope, alt: u32, kind: CpKind); // a CP from s's tops; s stays (§5.6)
fn push_cp_from(&mut self, s: Scope, alt: u32, kind: CpKind);  // convert s into a CP (§5.6)
```

`rewind` is reusable: the same scope can be rewound before every clause
attempt, and every binding made after each rewind is still trailed. `close`
consumes the token. `Scope` asserts in debug builds that it was closed.
`push_cp_keep` exists for the one site that makes two CPs from one
pre-candidate snapshot (`dynamic_call_attempt`, §5.6); it does not consume
`s`, and the site must still end `s` with `push_cp_from` or `close`.

`ScopePolicy` says which parts of the execution snapshot (§6.3) the scope
captures and restores on `rewind` and on `close`. Each of today's ~96 rollback
sites maps to one policy (table in §6.3).

**Making `trail`, `heap`, `cps` and `scopes` private** turns a rollback that
bypasses this API into a compile error. The fields being private does not by
itself make HB correct; `derive` on every transition above does.

### 5.4 The review counterexamples

**Counterexample 1: a cut inside a meta-call lowers HB below an active scope.**

| step | event | `cps` (h) | `scopes` (h) | HB | effect |
| ---: | --- | --- | --- | ---: | --- |
| 1 | older clause CP | [20] | [] | 20 | |
| 2 | meta-call opens its scope at H=100 | [20] | [100] | 100 | |
| 3 | inner goal pushes a CP at H=110 | [20, 110] | [100] | 110 | |
| 4 | inner `!` cuts to the inner barrier (generation 1) | [20] | [100] | **100** | the old `Mark` design set HB = 20 here |
| 5 | bind variable at address 50 | | | | 50 < 100: **trailed** |
| 6 | the meta-call fails: `rewind_close` | [20] | [] | 20 | the trail entry restores address 50 |

The cut removes CPs only (L2) and `derive` sees the scope.

**Counterexample 2: a reusable snapshot must stay protected after a rewind.**
A lowered three-clause predicate opens one scope `s` at H=100 with an older CP
at h=20. Variable `V` is at address 60 (allocated after the CP, before `s`).

| step | event | HB | effect |
| ---: | --- | ---: | --- |
| 1 | `open_scope` | 100 | |
| 2 | clause 1 binds `V`, fails | 100 | 60 < 100: trailed |
| 3 | `rewind(&s)` | **100** | `V` unbound again; `s` stays active. The old design restored `HB = m.hb = 20` here |
| 4 | clause 2 binds `V`, fails | 100 | trailed |
| 5 | `rewind(&s)` | 100 | `V` unbound again |
| 6 | clause 3 succeeds | 100 | |
| 7 | `close(s)` | 20 | bindings kept |

Both are required tests (§11.2, Appendix A), plus a three-clause fallback test
in which clause 3 reads `V` and must see it unbound.

**Counterexample 3 (second review): a cut through a scope, then a new CP.**
With depth boundaries, a scope that captured depth 2 is wrong after a cut
removes both older CPs and the goal pushes a new one at depth 1: floor 2
refuses its redo, and `truncate(2)` on rewind leaves it alive above the
rewound H. The code path exists today: `catch/3` calls its goal through
`call_goal_value` without raising B0 (`T:6988`), so a `!` in the goal
truncates to the enclosing clause's B0 (`T:7376`), below catch's
`cp_depth` (`T:6985`). Example: `catch((!, member(X,[a,b])), E, true)` in a
clause with an older clause CP. The structural `;` branch (`T:7414-7418`) is
another: its left branch is not wrapped in `call_goal_once`, so a `!` there
cuts below the `;` scope's boundary.

With generations (two older CPs, gens 1 and 2, `h` 20 and 30; the enclosing
B0 is gen 1):

| step | event | `cps` (gen:h) | scope `s` | depth model (old) | generation model |
| ---: | --- | --- | --- | --- | --- |
| 1 | older CPs | [1:20, 2:30] | | | |
| 2 | open `s` at H=100, floor set | | `cp_gen` 3, floor 3 | `cp_depth` 2, floor 2 | |
| 3 | `!` cuts to B0 (gen 1) | [] | survives (L2) | | |
| 4 | `member/2` pushes a CP at H=105 | [3:105] | | depth 1 < floor 2 | gen 3 ≥ floor 3 |
| 5 | the goal fails; backtrack | | | redo refused | member redo allowed |
| 6 | the goal fails again later; `rewind(&s)`, H=100 | | | `truncate(2)` keeps [105]: a CP above H | removes gen ≥ 3: [] |
| 7 | success instead of 6: catch commits (`close` after removing gen ≥ 3) | [] | | `truncate(2)` keeps [105] | [] |

Step 7 is the first-solution commit that today's `catch/3` intends
(`T:6991-6994`); with today's stale depth the member CP survives the
commit. Generations change the result only where today uses a boundary that
a cut has crossed. With R−1e (`catch/3` opaque to cut, §12) the catch case
disappears from the baseline. The required test (§11.2, Appendix A): cut
through a scope, push a new CP, redo it, then rewind, and check that no CP
survives with `h` above H and that the redo ran.

### 5.5 Choice points

```rust
pub struct ChoicePoint {
    gen: u64,                          // generation (§5.2)
    alt: u32,                          // next clause PC (0 for builtin-redo kinds)
    h: u32, tr: u32, boxes: u32,       // heap, trail and box tops
    e: u32, e_top: u32,                // E and the protected env extent (§6.1)
    cp: u32,                           // continuation
    b0: u64,                           // saved cut barrier (today's cut_barrier), as a generation
    args: u32, n: u8, save: RegSave,   // saved registers in arg_stack[args..args+n]
    levels: Levels,                    // ITE barrier levels, 0–2 entries of (LevelId, gen) (§6.4)
    kind: CpKind,                      // Clause | IteGuard | Aggregate{..} | Builtin(BuiltinRedo) | Fact | Dynamic | Foreign | Naf
}
enum RegSave { Args, Live(LiveSet), Dirty(DirtyMask) }
```

- **Register policy per kind.**
  - `Clause`: A1..A<arity>. The spike confirmed this for clause CPs **[S]**.
  - `IteGuard`, `Aggregate`: the compiler's live set at the guard. Until the
    compiler exports it, today's dirty set (`save_regs`, `ST:4466`).
  - `Builtin`, `Fact`, `Dynamic`, `Foreign`, `Naf`: today's dirty set. A
    temporary that is live across an inline builtin call must survive
    backtracking into that builtin, so arity is not enough here.
- **Restore keeps today's clearing.** `restore_ax_regs` (`ST:4497`) sets
  every dirty slot to `Uninit`, then writes the saved ones. The rewrite keeps
  a dirty bitmask and clears dirty slots to `UNINIT`, so an instruction that
  reads an unsaved temporary still fails exactly as today. Dropping the clear
  is an R7 perf item behind the gates.
- **`backtrack`** (today `T:1420-1475`): undo the trail (heap and env-slot
  entries) to `tr`, truncate `H` and the boxes, restore the registers, set
  `E = e`, `CP = cp`, `B0 = b0`, clear `pending_b0` and `pending_level`, set
  `pc = alt`, derive. `var_counter` is **not** restored (§5.7).
- **Cut** removes the CPs with `gen ≥` the target generation and derives HB
  and EB (§5.2).

### 5.6 Builtin redo and retained cells

A builtin CP's redo data, and every Rust local that outlives a rollback, may
keep cells only under these rules.

**Retention by tag.** A retained cell is valid after a rollback to obligation
`o` only if:

| tag | condition |
| --- | --- |
| `REF`, `STR`, `LIS` | address < `o.h` |
| `BOX` | box index < `o.boxes` (boxes are a separate arena, truncated separately) |
| `VAR` | never retained directly; retain the `REF` to it |
| `ATOM`, `FUN` ids | always valid (interned, never truncated) |
| `INT` | always valid |
| other ids (clause index, fact-row index, stream handle) | valid while the owning table is unchanged; the redo record carries the table generation and checks it |

The paranoid build checks every retained cell against the table when the CP
is resumed.

**Store original references.** Redo data stores the cell as it was in the
register (for an unbound variable, its `REF`), not `deref` of it. If a
variable that existed before the CP is bound later, the binding is trailed and
undone by the rollback, so the original `REF` regains its meaning. A saved
`deref` taken after such a binding can point above `o.h`. Today `member/2`
already stores the raw register values (`T:2423-2438`).

**Capture tops and establish protection before candidate bindings.** Several
sites today capture `trail.len()`/`heap.len()` first, try the candidate
(which binds), and push the CP only after it succeeds:
`fact_table_attempt` (`T:4979-5013`), `apply_first_compatible_foreign_result`
(called from `finish_foreign_results`, `T:4228-4260`),
`dynamic_call_attempt` (`DB:556-598`), `dynamic_rule_body_attempt`
(`DB:681-708`), `dynamic_clause_attempt` (`DB:906-951`), and by the same
`cp_trail` shape `current_predicate_attempt` (`DB:828`) and
`dynamic_retract_attempt` (`DB:987`). With unconditional trailing that is fine.
With conditional trailing it is wrong: during the candidate, HB is still the
older obligation's `h`, so a binding of a variable between that `h` and the
captured heap top is not trailed, and backtracking to the later CP does not
undo it. In the rewrite each such site opens a scope **before** the candidate
and converts it with `push_cp_from(s, …)` on success, or `rewind_close(s)` on
failure. The CP inherits the scope's tops, and HB covered the candidate.

**Two CPs from one snapshot.** `dynamic_call_attempt` takes one snapshot
before renaming the candidate clause (`DB:556-559`). When the candidate
succeeds it pushes a later-clause CP if more clauses remain (`DB:564`) and
then a rule-body CP if the clause has a body (`DB:581`), both with the same
saved registers, stack, trail and heap tops. Backtracking resumes the
rule-body CP first (the next body solution), then the later-clause CP. The
rewrite keeps this order with one scope and both conversion forms:

```rust
let s = self.open_scope(ATTEMPT);                 // before Fresh(C) import (§9.2)
// import the clause, detach its template R (§9.2), bind head, solve body
match (more_clauses, has_body) {
    (true,  true)  => { self.push_cp_keep(&s, cont, Dynamic(next_idx));      // older
                        self.push_cp_from(s, cont, RuleBody(R, 1)); }        // younger
    (true,  false) =>   self.push_cp_from(s, cont, Dynamic(next_idx)),
    (false, true)  =>   self.push_cp_from(s, cont, RuleBody(R, 1)),
    (false, false) =>   self.close(s),
}
```

Both CPs carry the scope's `h`, `tr`, `boxes`, `e_top`, register save and
exec snapshot, so I1 holds with equal tops. Each gets its own generation in
push order. On failure the site does `rewind_close(s)`, as today's restore at
`DB:600-603`.

**The pop-before-resume transition.** Today `backtrack` pops a builtin CP and
then calls `resume_builtin` (`T:1462-1470`). The rewrite does the same in this
order:

1. rewind to the CP (trail, H, boxes, env, registers, control);
2. pop it and derive. HB drops to the next obligation's `h`;
3. resume. If more alternatives remain, the redo pushes its new CP **before**
   binding the next candidate (`member/2` and `between/3` already do), or
   uses the open-scope-then-convert form above;
4. a last alternative binds with the lower HB. That is sound: no surviving
   obligation can restore the variables between the lower HB and the old CP's
   `h`, because rolling back to any older obligation truncates them.

### 5.7 `var_counter` per scope type

Fresh names are numbered from `var_counter`. Restoring it changes which names
later variables get, and so their text and sort order. Today:

| scope type | restores `var_counter`? | evidence |
| --- | --- | --- |
| CP backtrack (all CP kinds) | **no** | `T:1420-1475` restores pc, trail, stack, heap, regs, cp, cut barrier only |
| builtin redo | no | same path |
| lowered clause snapshot: T4 `multi_clause_n`, F11 tail loop, `lowered_dispatch` decline | **yes** | `lo_restore_clause`, `ST:4536-4543` |
| Stage-2 region decline (9 dispatchers: `matching_deps`, `matching_versions`, `key_dep_rows`, `group_keyed`, `build_tree`, `dep_breaks`, `filter_satisfies`, `key_pkg_rows`, `tree_lookup`) | **yes** | `region_decline`, `ST:4724-4730` |
| `catch/3` (failure and ball) | no | `T:6968-7034` |
| `call_goal_once` and its users (`\+`, `forall`, maplist family, include/exclude, predsort) | no | `T:7258-7303` |
| `call_goal_value` `;` and `->` branches, `\+` | no | `T:7397-7435` |
| lowered ITE condition | no | `LE:847-855` |
| builtin trial unifications (`builtin_unify_member`, `copy_term/2`, `sort/4`, …) | no | `T:5727-5733`, `T:2774-2779` |
| dynamic-DB, fact-table and foreign-result attempts | no | `DB:556-600`, `T:4979-5020` |
| aggregates | no | aggregate frame is a CP |
| `reset_query` | sets 0 | `ST:4304` |

The rewrite gives `ScopePolicy` a `restore_var_counter` flag, set only for
the two "yes" rows. CPs never restore it. So a failed alternative or a negated
goal that creates variables leaves the counter advanced, exactly as today.
This table is about in-place rollback. Whole-machine snapshots move the
counter with the machine: an adopted dynamic-body solution brings its
counter, a discarded snapshot or `par_aggregate` worker takes its increments
with it (§9.4).

## 6. Environments and execution snapshots

### 6.1 The environment stack

```
estack: Vec<Cell>, frames: [prev_e, cp, n_y, y0 .. y(n-1)], E = current frame base
```

- **`e_top`.** Every obligation records `e_top = max(end of the current
  frame, EB)` when it is created. EB is derived like HB (§5.2).
- `Allocate`: the new frame goes at `max(end of frame E, EB)`, so it never
  overwrites a frame that a live CP or an active scope may resume into.
  `Deallocate` sets `E = prev_e`; the stack top never drops below EB.
- Y reads and writes index `estack[E + 3 + y]`. There is no `Arc`, no
  copy-on-write, no `YRegs` `resize`, and no frame search for the "topmost
  Env". Today `put_reg` alone is 1.33 M per resolve **[S]**.
- `put_variable Yn, Ai` allocates the variable **on the heap** and puts a
  `REF` in both `Yn` and `Ai`. Permanent variables are never unbound *in* the
  environment. That removes the "unsafe variable" case.
- If R−1d lands (§12), the frame also saves the caller's B0 at `Allocate` and
  restores it at `Deallocate`, as in the standard WAM:
  `[prev_e, cp, b0, n_y, y0 ..]`. Otherwise B0 stays outside frames, as today.

### 6.2 Y slot writes and the env-slot trail

Protecting the frame's *space* does not undo writes into its *slots*. Today
the CP's `Arc` stack snapshot undoes both. So the question is whether any
instruction writes a Y slot whose earlier content is still needed after a
rollback. Checked against the compiler (`WT`) and the runtime:

1. **Ordinary compiled code writes a Y slot only at the variable's first
   occurrence** (`get_variable Yn`, `put_variable Yn`, `unify_variable Yn`).
   Later occurrences read it (`put_value`, `get_value`, `unify_value`). A slot
   holding a live value is not overwritten on any single path.
2. **The runtime overwrites one live Y slot:** aggregate finalization writes
   the result into the result register even when it is a Y slot
   (`T:5142-5146`, and the par path at `T:1114-1118`). The rewrite binds the
   variable that the slot references and leaves the slot alone. The binding
   is trailed by the heap rule.
3. **First-occurrence writes happen after CPs in the same frame.** A variable
   first seen in an ITE condition or a disjunction branch is initialized
   after the guard CP. The compiler's variable map after an ITE is the Then
   branch's map (`WT:2360-2361`: `Vf = V2` in both cases), so the
   continuation can read a slot whose first write happened in a failed
   condition. Compiled with `ite_use_y_level(true)`:

   ```prolog
   p(A,R) :- ( q(A, X) -> true ; true ), r(X, R).
   ```
   ```
   get_level Y4 / try_me_else L_ite_else_1 / put_value Y1, A1 / put_variable Y2, A2
   call q/2 / cut Y4 / ... / L_ite_else_1: trust_me / ... / L_ite_cont_1: put_value Y2, A1
   ```

   If `q` fails, the else path reaches `put_value Y2` with `Y2` written by the
   failed condition. Today backtracking restores the stack snapshot, so `Y2`
   is absent again and `put_value` fails (`T:373-378`, `get_reg` → `None`).
   With a plain protected stack, `Y2` would keep a `REF` above the CP's `h`:
   a dangling reference after the heap reset.

**Consequence.** The rewrite trails slot writes into protected frames: a Y
write at slot index `k < EB` pushes an env-slot trail entry with the slot's
exact previous content. Rollback restores it. That reproduces the snapshot
for slots, including the failure in the example above. Writes into frames
allocated after the youngest obligation (the common case: right after
`Allocate`) are not trailed.

Today a Y slot has two empty states. `ABSENT` (no entry; reads fail) and
`UNINIT` (an entry holding `Uninit`, written when a trail-only rollback undoes
a register entry whose old value was `None`; `get_reg` returns
`Some(Uninit)`). CP backtracking restores `ABSENT` through the snapshot;
trail-only sites restore `UNINIT` (`ST:4273-4292`). The rewrite has a
sentinel for each and restores the exact previous content. The R0b audit
(§12) checks whether any trail-only site's later code reads such a slot; the
lowered ITE is the only trail-only site that runs framed code with Y writes,
and it does not follow this rule today (§6.6). Fixing the compiler's
`Vf = V2` rule is R−1c (§12). After it, a permanent variable first seen
inside an ITE and read after it is initialized before the guard, so item 3's
case no longer arises in compiled code; the env-slot trail stays for
soundness.

### 6.3 Execution snapshot

A rollback must restore control state as well as heap state. Each scope's
`ExecSnap` and each CP carry:

```rust
struct ExecSnap {
    e: u32, e_top: u32,              // E and protected env extent
    cp: u32, pc: u32,                // continuation and pc (only when the policy saves them)
    b0: u64,                         // cut barrier, as a CP generation (§5.2)
    pending_b0: Option<(u64, u32)>,  // today's pending_cut_barrier (generation, at_pc)
    pending_level: Option<(LevelId, u64)>,
    floor: u64,                      // backtrack floor, as a CP generation
    regs: RegSave,                   // None | Args(n) | Dirty(mask) | Live(set)
    var_counter: Option<u32>,        // only when the policy restores it (§5.7)
}
```

Today's sites, mapped onto policies. "rewind" is what a failure path
restores; "close" is what the success path restores.

| today's site | heap/trail | registers | env (E, e_top, slots) | cp / pc | cut state | var_counter | on close |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lo_clause_snapshot` / `lo_restore_clause` (`ST:4524-4543`): T4 clause attempts, F11 loop | yes | dirty set, clear-then-restore | yes (today: the stack `Arc`) | no | `b0` | yes | nothing |
| `lowered_dispatch` (`ST:4570-4597`) | via the clause snapshot on decline | dirty set | yes on decline | `cp` always, `pc` on decline | `b0`, `pending_b0` always | yes on decline | `cp`, `b0`, `pending_b0` |
| `call_goal_once` (`T:7258-7303`) | none (its callers rewind) | none | **yes, on success too**; prunes env-slot entries of dropped frames (below) | no (`call_goal_key` saves `pc`, `cp`) | `b0`, `floor` always | no | remove CPs with `gen ≥ cp_gen`; `E`, `e_top`, `b0`, `floor`; prune |
| `catch/3` (`T:6968-7034`) | yes | dirty set | yes | `cp` always | `b0` always; `floor` after R−1a | no | remove CPs with `gen ≥ cp_gen` (commit); `cp`, `b0` |
| `predsort_order` (`T:7311-7338`) | trail always, after reading the result | dirty set, always | via `call_goal_once` | no | via `call_goal_once` | no | registers |
| `\+`, `->` in `call_goal_value` (`T:7397-7435`) | trail | none | via `call_goal_once` | no | no | no | nothing |
| `;` fallback in `call_goal_value` (`T:7414-7418`) | trail | none | `E`, `e_top` on rewind (the left branch is not wrapped in `call_goal_once`) | no | no | no | nothing |
| region P2 snapshot (`region_*_dispatch`, `region_decline`) | trail, heap | none (G-1: registers never clobbered) | no | no | no | yes | nothing |
| lowered ITE condition (`LE:847-855`) | **see §6.6**: after R−1c, trail and heap; otherwise trail only, H kept, every bind trailed | today: the register trail for delegated arms, nothing for direct writes; after R0b: an explicit save of the registers the else branch reads | **no protection of the current frame**: direct Y writes are not trailed, as today | no | no | no | nothing |
| trial unifications | trail | none | no | no | no | no | nothing |
| `unifiable/3` trial (`T:2897-2915`) | trail, with ordered bind capture (§7.5) | none | no | no | no | no | always rewound |
| fact/dynamic/foreign attempts | trail, heap | dirty set | yes | no | no | no | converted to a CP |

Notes:

- **`call_goal_once` restores the environment on success.** It restores
  the stack snapshot even when the goal succeeds, so a nested run cannot
  Deallocate the caller's frame (the `sort_versions_desc` fix, comment at
  `T:7250-7257`). In the rewrite `close` restores `E` and `e_top`, so frames
  the nested run left above `e_top` are dropped; its CPs were already
  truncated. Today's snapshot also returned outer Y slots to their pre-call
  values. With E-relative Y addressing and the R1 generator check (§4), a
  nested run cannot write an outer frame's slots. The paranoid build copies
  `estack[..e_top]` at open and asserts it unchanged at close.
- **Trail-only sites truncate H on rewind in the rewrite**, where today they
  only unwind the trail. That is safe because unification allocates no heap
  cells. A site that keeps a cell computed inside a scope across its rewind
  must show that the cell is below `s.h` (debug assertion) or detach it first.
  The one known case is `unifiable/3` (`T:2904-2915`): its captured pairs are
  pre-existing cells, and the result list is allocated only after the rewind
  (§7.5). The lowered ITE is the exception to H truncation unless R−1c lands
  (§6.6).
- **Env-slot entries of abandoned frames.** Today `call_goal_once` removes
  the nested run's Y-register trail entries before restoring the stack,
  keeping binding entries and A/X entries (`T:7279-7301`). It has to, because
  today a register entry is undone into the *current* topmost frame, which
  after the restore is the caller's. The rewrite's env-slot entries are
  absolute, so they cannot hit the wrong frame by name, but an entry for a
  dropped frame is still useless and can name a slot that a later frame
  reuses. Two rules:
  1. **Prune on close.** A scope whose policy restores `E` and `e_top` on
     close first removes the CPs with `gen ≥ s.cp_gen` (L2), then drops every
     env-slot entry above `s.tr` whose slot is at or above `s.e_top`, then
     restores `E` and `e_top`. Heap entries above `s.tr` are kept: the
     caller's later rewind still has to undo bindings the nested run made to
     older variables (`predsort_order` reads `Order`, then rewinds). Entries
     for slots below `s.e_top` are kept too; by the R1 check (§4) there are
     none, and the paranoid build asserts it. No surviving obligation needs a
     dropped entry: every older obligation has `e_top ≤ s.e_top`, so those
     slots are dead for it, and every younger one is gone.
  2. **Skip dead slots on undo** (§5.1). An entry left by a frame that was
     abandoned without a close (a cut, then `Deallocate`) is undone only by
     obligations older than that frame. Such a frame was allocated at or
     above the obligation's `e_top` (`Allocate` places frames at `≥ EB`), so
     the slot is dead after that rollback and the undo skips it. A frame
     that is still live after the rollback is below `e_top` and was never
     reused while the obligation was active, because new frames go at
     `≥ EB ≥ e_top`.

  The paranoid build tags every env-slot entry with the frame's allocation
  serial and asserts on undo that the slot is either dead or still owned by
  that frame. Required test (§11.2): a `predsort` comparator containing an
  ITE whose condition writes a Y slot of the comparator's frame, followed by
  a second comparator call that reuses the frame space, then an outer
  rewind; the caller's Y slots and the second frame must be unchanged.
- **No current instruction overwrites an already-live Y slot**, except the
  aggregate result write in §6.2 item 2, which the rewrite removes. The other
  hazard, first writes after a CP, is covered by the env-slot trail.

### 6.4 If-then-else barriers

Today's ITE machinery (`T:909-977`, `T:1128-1190`, `T:1446-1461`), carried
over without semantic change:

| today | meaning | rewrite |
| --- | --- | --- |
| `ChoicePoint.levels: Vec<(String, usize)>` (`ST:776-789`) | named barrier levels: the Y name the compiler reserved, and the CP depth when `get_level` ran | `levels: Levels` with `(LevelId, u64 gen)`: the `next_gen` when `get_level` ran (§5.2); at most two per guard |
| `pending_level` | set by `GetLevel` when the next instruction is a `TryMeElse`/`TryMeElsePc` (shape 1); consumed by that push; cleared on backtrack | same, in `ExecSnap` and machine state |
| shape 1: `get_level B` before `try_me_else` | records the depth *before* the guard on the guard CP | same |
| shape 2: `get_level C` after `try_me_else`, only when the condition has a top-level `!` (`WT:2316-2333`) | attaches the depth *including* the guard to the guard CP | same |
| `CutTo(yn)` | search CPs from the top for the nearest CP carrying `yn` (latest entry first); truncate to its depth; not found → no-op | same search on `LevelId`; removes CPs with `gen ≥` the recorded generation, through the cut transition (§5.2), so HB and EB are derived |
| `CutIte` (legacy) | pop one CP | same, through the cut transition |
| `cut_barrier` | B0: set by `Allocate` from `pending_cut_barrier` if `at_pc == pc`, else the CP depth (`T:713-716`); raised to the CP depth after an aggregate frame push (`T:1033`); set to the entry depth by `call_goal_once` and `lowered_dispatch`; restored by backtrack | `b0`, same rules with generations in place of depths (each "CP depth" above becomes `next_gen` at that moment); saved in CPs and in each scope's `ExecSnap` |
| `pending_cut_barrier` | set by a clause `try_me_else` (not an `L_ite_else_*` label) and by `retry_me_else` to `(clause depth, pc + 1)`; cleared on backtrack | `pending_b0`; `TryMeElseIte` sets it to `None` as the label test does today |
| `!/0` | truncate to `cut_barrier` | same, through the cut transition |

**Why the named search is per-activation.** CPs are LIFO, so the current
activation's guard is younger than any guard of an enclosing activation,
including a recursive activation of the same predicate with the same
`LevelId`. The commit `cut B` runs right after the condition succeeds, while
the guard is still present: the condition's own cuts are bounded by `C`
(shape 2), a callee's `!` is bounded by the callee's B0, which is above the
guard, and meta-calls are opaque scopes. So the nearest match is the current
activation's guard. Today's code treats "not found" as a no-op; the rewrite
keeps that, and the paranoid build counts not-found events. A non-zero count
in the differential is investigated before R2 exits.

**Cuts and obligations.** Every cut variant above is a "cut" transition in
§5.2: it removes CPs only, and derives HB and EB. An ITE inside an aggregate
works because the aggregate frame raised B0 above itself (`T:1023-1033`), and
the guard is above the frame.

Required tests (§11.2): cut in a condition (both shapes, including
`\+ (G, !, fail)`), nested ITE in Then and in Else (`(A -> B ; C -> D ; E)`),
recursive ITE with the recursion inside the condition and inside Then, ITE
with a cut in its condition inside `findall/3`, and ITE inside a meta-call.

### 6.5 Read and write mode

`UnifyCtx`/`WriteCtx` disappear. `get_structure` sets `S` (read mode) or
pushes a `FUN` header at `H` (write mode), and `unify_*` reads `heap[S++]`
or pushes. `put_structure` writes `STR(H)` into the register immediately. The
200-register scan in `set_heap_or_list` and its `"f/N"` parse go away.

### 6.6 Lowered ITE condition

**Today** (`LE:838-855`). The lowered ITE records `trail.len()`, runs the
condition in a closure, and on failure calls `unwind_trail_to` and runs the
else branch. Inside the condition there are two kinds of write:

- **direct** emitted writes: `get_variable`, `get_value`, `put_variable`,
  `put_value`, `put_constant` (`LE:942-985`) call `put_reg`, which writes the
  topmost frame's Y slot or calls `set_reg_str`; neither trails
  (`ST:4346-4356`, `ST:2604-2610`);
- **delegated** writes: arms run through `vm.step(..)` (`get_structure`,
  `get_list`, `unify_*`, `put_structure`, `put_list`) trail the registers
  they write, as in the interpreter.

Binding entries are always trailed. So after a failed condition: bindings
are undone; slots and registers written by delegated arms are restored (a
Y slot whose old content was absent becomes `Uninit`, `ST:4273-4292`); slots
and registers written directly keep the condition's values; nothing on the
heap side is freed. A variable created in the condition stays usable,
unbound. This is not the uniform env-slot restoration of §6.2, and it is not
the interpreter's behavior:

```prolog
rundoite(X,R) :- ((Y=a, Y=b) -> R=then ; R=els), X=Y.
```

`lowered_rundoite_2` on `(c, els)` succeeds, as in SWI-Prolog; the
interpreter gives no solution, because its `PutVariable` trails the Y write
(`T:365-372`), backtracking to the guard restores the stack snapshot, and the
`put_value` for `X=Y` reads an absent slot (the §6.2 item 3 case).
`tests/test_wam_rust_lowered_ite_exec.pl:41`, `:117` pins the lowered result
(`true`).

**With R−1c (recommended).** The compiler initializes, before the guard,
every permanent variable first seen inside the ITE and read after it (§12).
Then no slot first written in the condition is read on the else path or
after the ITE, and the lowered ITE uses a plain scope:

- open: `tr`, `h`, `boxes`, `cp_gen`, and `e_top` covering the current frame,
  so condition writes into the frame are trailed like any protected write;
- rewind (condition failed): undo the trail, truncate H and boxes, remove
  CPs with `gen ≥ cp_gen`, restore the registers the else branch reads
  (R0b); restored slots get their exact previous content;
- close (condition succeeded): nothing.

Lowered and interpreted code then agree with each other and with SWI on
`rundoite` and on `ite_map` (§12).

**Without R−1c (emulation).** The rewrite must reproduce both behaviors: the
interpreter keeps §6.2's rule (it fails), and the lowered ITE uses a
dedicated `LoweredIte` policy:

- open: `tr`, `cp_gen`, `e_top` covering the current frame, and increment
  the machine's `trail_all` count. While it is non-zero, `derive` sets HB to
  `u32::MAX`, so every heap binding is trailed, as today's binding trail is
  unconditional;
- direct emitted writes use `set_y_untrailed(k, c)` (lowered code only; the
  paranoid build counts them), so they are not undone, as today;
- delegated writes go through the step arms and are trailed as protected
  slot writes. An entry whose old content is `ABSENT` is undone to `UNINIT`,
  the trail-only rule of `ST:4273-4292`;
- rewind: undo the trail to `tr`; **do not truncate H or boxes**, so a
  variable the condition created stays a valid unbound cell; remove CPs with
  `gen ≥ cp_gen`; restore registers per R0b;
- close: decrement `trail_all`.

`trail_all` is required because H is kept. Under the normal rule a variable
created in the condition is above the scope's opening H, so its binding
(`Y=a`) would not be trailed, and the else branch would see `Y` still bound.
The cost is confined to lowered ITE conditions. Keeping H is always sound:
nothing is truncated, so nothing dangles.

**Condition CPs.** Today a failed condition leaves any CP it pushed (for
example a `member/2` CP left by a delegated builtin before a later goal
failed): only the trail is unwound. That CP's saved trail top is then above
the unwound trail, so a later backtrack into it would not undo bindings the
else branch records below that top, which is unsound already. Both policies
remove such CPs on rewind. The R−1 audit adds a probe for a lowered condition
that leaves a CP and fails; if a configuration reaches it, the fix lands in
R−1 too.

**Gate.** `tests/test_wam_rust_lowered_ite_exec.pl` on the frozen build and
on the rewrite, plus an interpreter run of the same predicates compared case
by case. Before R−1c the expected difference on `rundoite(c,els)` is
recorded (lowered `true`, interpreter no solution); after it both succeed.
The lowered corpus compiles with default features only after R−1f; until
then it runs with `--no-default-features`.

## 7. Builtins: migration API

There are 154 grouped builtin arms (216 predicate keys) in the assembled
dispatch across the core, arith, io, type, term, ext and meta families (§4
has the per-family counts), plus the three dynamic-DB routes. By the second
external review's count, 149 arms (211 keys) read registers or build terms,
directly or through helpers; the other five are `true/0`, `fail/0`, `!/0`,
`nl/0`, `halt/0`. There are 10 builtin family templates
(`*_builtin.rs.mustache`, ~1.4 K lines) and the dynamic-DB methods
(`dynamic_db_methods.rs.mustache`, 1.0 K lines). Inside the builtins there are
389 reads of `get_reg_raw("A<n>")`, 213 `deref_heap` calls and ~320
`Value::strv`/`Value::list` constructions. A big-bang port of all of them by
hand is the main schedule risk, so the API is built for a two-speed migration.

### 7.1 Native API (hot builtins, regions, lowered code)

Reading terms must not hold a borrow of the machine while user code runs:
`maplist`, `include`, `foldl` and `predsort` call user predicates, which need
`&mut WamState`. So the native API has three shapes.

```rust
// 1. Copyable handles, no borrow.
#[derive(Copy, Clone)] pub struct StrRef { f: FunctorId, args: u32 }
#[derive(Copy, Clone)] pub struct ListCursor { at: Cell, stop: Option<u32> }

impl WamState {
    fn reg(&self, i: u8) -> Cell;  fn set_reg(&mut self, i: u8, c: Cell);
    fn deref(&self, c: Cell) -> Cell;
    fn tag_of(&self, c: Cell) -> Tag;                     // after deref
    // 2. Short-lived accessors that return copied cells.
    fn str_ref(&self, c: Cell) -> Option<StrRef>;
    fn arg(&self, s: StrRef, i: u32) -> Cell;
    fn list_cursor(&self, c: Cell) -> ListCursor;
    fn list_next(&self, k: &mut ListCursor) -> Step;      // Elem(Cell) | End | Tail(Cell)
    // 3. Bulk borrows, only for pure operations.
    fn view(&self, c: Cell) -> View<'_>;                  // View::Str(FunctorId, &[Cell]) etc.
    // construction, binding, scopes
    fn bind(&mut self, var_addr: u32, c: Cell);           // conditional trail inside
    fn mk_int(&mut self, n: i64) -> Cell; fn mk_float(&mut self, f: f64) -> Cell;
    fn mk_str(&mut self, f: FunctorId, args: &[Cell]) -> Cell;
    fn mk_list(&mut self, items: &[Cell], tail: Cell) -> Cell;
    fn fresh_var(&mut self, kind: VarKind) -> Cell;       // new unbound VAR on the heap, returns its REF;
                                                          // numbered from var_counter with the kind's
                                                          // pre/post convention (§3.4): _V/_H post, _L/_F/_M/_C/... pre
    fn functor_id(&mut self, name: AtomId, arity: u32) -> FunctorId;
    fn open_scope(&mut self, p: ScopePolicy) -> Scope; fn rewind(&mut self, s: &Scope);
    fn close(&mut self, s: Scope); fn rewind_close(&mut self, s: Scope);
}
type BuiltinFn = fn(&mut WamState, args: u8 /* arity; A1..An are the args */) -> BuiltinResult;
enum BuiltinResult { Fail, Succeed, SucceedWithRedo(BuiltinRedo), Throw(Detached) }
```

What survives what:

| handle | heap allocation (Vec growth) | nested execution (`call_goal_once`, comparator) | rollback |
| --- | --- | --- | --- |
| `Cell`, `StrRef`, `ListCursor` | survives: addresses are indices | survives, if the cells existed when the nested scope opened (they are below its `h`, and nothing inside can unbind what was bound before it) | survives a rewind of a scope opened **after** the handle was taken; invalid after a rewind of any older obligation |
| `View<'_>` bulk borrow | not possible: the borrow forbids `&mut` | not possible | not possible |

Cursor semantics match today's. `deref_list_arg` (`T:3374-3386`)
materializes the spine at entry and leaves unbound element variables as
names, which later deref to whatever a nested goal bound. A cursor reads each
element cell at step time and derefs it then, which gives the same answers.
The spine of a proper list cannot change while the cursor's scope is active.
For a list that is partial at entry, a builtin whose today semantics fix the
spine at entry records `stop` (the element count) and stops there.

Bulk borrows (`View::Str(&[Cell])`) are for pure operations only: compare,
identity, hashing, ground checks, copy-out. They make no `&mut` call while the
borrow lives. `predsort` copies the element cells into an owned `Vec<Cell>`
because the sort needs a buffer anyway. That is 8 bytes per element, with no
refcounts.

### 7.2 Term operations: order, identity, unification, copy

Today these are four different algorithms with different answers on the same
inputs. The rewrite exposes each separately, and each reproduces today's
behavior. Corrections (to ISO or SWI behavior) are separate behavior changes,
gated on their own.

```rust
fn compare_std(&self, a: Cell, b: Cell) -> Ordering;   // = term_compare / term_compare_derefed
fn identical(&self, a: Cell, b: Cell) -> bool;          // = terms_identical (==/2, regions)
fn unify(&mut self, a: Cell, b: Cell) -> bool;          // = unify; first-to-second binding
fn copy_fresh(&mut self, c: Cell) -> Cell;              // = copy_term/2 (_C<n>, pre-increment)
fn variant(&self, a: Cell, b: Cell) -> bool;            // = variant_terms
// bridge forms (§7.3)
fn unify_value(&mut self, c: Cell, v: &Value) -> bool;
fn unify_values(&mut self, a: &Value, b: &Value) -> bool; // both imported in one session
```

**`compare_std`: the compatibility comparator.** Derived from `term_compare`
(`T:5483-5560`) and `term_compare_derefed` (`T:5592-5665`), which agree.

| class | members | rule |
| ---: | --- | --- |
| 0 | unbound `VAR` (and `UNINIT`, named `""`) | name text bytes |
| 1 | `INT`, `BOX` Float, `BOX` BigInt | convert both to `f64` (`n as f64`); `partial_cmp`, with NaN treated as Equal; on Equal, Float before Integer. So integers that differ but map to the same `f64` (for example 2^53 and 2^53+1) compare Equal, and `sort/2` keeps only one |
| 2 | `ATOM` (incl. `[]`), `BOX` Bool | name text: atom text, `"true"`/`"false"` for Bool. So `Bool(true)` and the atom `true` compare Equal |
| 3 | `STR`, `LIS` | below |

Class 3:

- `STR` vs `STR`: arity, then functor name text, then arguments left to right.
- `LIS` vs `LIS`: heads, then tails. This equals today's element-wise compare
  with the shorter list first, and today's list-vs-cons-`Str` arms.
- **`LIS` vs `STR` with arity n ≠ 2:** `2.cmp(&n)` (or reversed).
- **`LIS` vs `STR` with arity 2.** Today's answer depends on whether the list
  is proper:
  - proper list (ends in `[]`): **Equal**. Today it is a `Value::List`, and
    the arms at `T:5557-5559` and `T:5662-5664` compare arity only. So `[a]`
    and `f(b,c)` compare Equal, and `sort/2` drops one of them;
  - partial list (ends in an unbound variable or a non-list): compare `"[|]"`
    with the functor name. Today `deref_heap` rebuilds a partial list as a
    `"[|]/2"` `Str` (`rebuild_partial`, `ST:6634-6640`), so it reaches the
    `Str` vs `Str` arm. So `[a|T]` sorts before `f(b,c)`.

  The comparator walks the list's spine to decide. That walk happens only in
  this rare case.

Sorting uses a stable sort (`sort_by`, as today), so the order of
Equal elements, and which one dedup keeps, does not change.

**`identical`: strict identity.** Derived from `terms_identical` and
`identical_derefed` (`T:5412-5441`). Variables: the same cell after deref.
Floats: exact `f64` `==` (so `0.0 == -0.0` and NaN is never identical).
Integer vs Float: never. Bool vs atom: never. `LIS` vs `STR`: never (the
`da == db` fallback is false). Compound: same functor and arity, then the
arguments.

**`unify`: compatibility unification.** From `unify` (`ST:6434-6522`). Float
vs Float succeeds when `|a − b| < f64::EPSILON`, an absolute epsilon. Integer
vs Float fails. Bool unifies only with Bool. Binding is first-to-second
(§3.1).

**`copy_fresh`.** From `copy_term_walk` (`T:2930-2960`). A depth-first walk,
arguments left to right and head before tail, gives each distinct source
variable one new `_C<n>` with a pre-increment, in first-occurrence order. The
cell walk visits variables in the same order as the `Value` walk.

**Which operation each caller uses:**

| caller | operation (today and in the rewrite) |
| --- | --- |
| `sort/2`, `msort/2`, `sort/4`, `keysort/2`, `setof` dedup | `compare_std == Equal` (`T:5089`, `T:5901`, `T:5947`) |
| `compare/3`, `@<` family | `compare_std` |
| `==/2`, `\==/2` | `identical` (`T:1592-1596`) |
| Stage-2 regions (`group_keyed`, `matching_deps`, …) | `identical` (`terms_identical`) |
| `list_to_set/2` | unify-and-commit against earlier elements (`builtin_unify_member`, `T:5727-5733`), not compare |
| `copy_term/2` | `copy_fresh`. Never `from_value(to_value(t))`, which would keep live variables |

Mechanically migrating `group_keyed` to `compare_std == Equal` would merge
keys that are distinct today (for example `[a]` and `f(b,c)`). The
reference-model tests (§11.2) check each operation separately.

### 7.3 Bridge API (cold builtins, first-pass compile of everything)

The bridge copies between cells and `Value` inside one bridged builtin call.
It produces **live** values, and they must not outlive the call.

```rust
pub struct LiveVar { machine: u32, addr: u32, epoch: u32 }   // Value::Var(LiveVar)
fn reg_value(&self, i: u8) -> Value;                // copy out; an unbound var becomes Value::Var(LiveVar)
fn import(&mut self) -> ImportSession<'_>;          // one name->cell map for every value imported through it
impl ImportSession<'_> { fn cell(&mut self, v: &Value) -> Cell; }
fn unify_value(&mut self, c: Cell, v: &Value) -> bool;     // one session
fn unify_values(&mut self, a: &Value, b: &Value) -> bool;  // both sides through one session
```

- **Live handle validation.** The machine has an id and an `epoch` that is
  incremented whenever H is lowered (backtrack, rewind, reset). A `LiveVar` is
  valid only if its `machine` matches, and either its address is below the H
  at the start of the current bridged call, or its `epoch` equals the current
  epoch. Cells that existed at the call's start cannot be truncated by the
  call's own scopes, which open at or above that H. Checked in debug builds.
- **Sharing across arguments.** A bridged builtin that converts several
  arguments in or out does it through one session, so a variable shared
  between arguments stays one variable. `Value::Unbound(name)` resolves
  through the session's name map (§3.4). `Value::Var(LiveVar)` resolves to
  its `REF`.
- **Two-Value unification.** `unify_values` replaces today's
  `self.unify(&a, &b)` on two `Value`s. `unify_value(Cell, &Value)` alone
  cannot replace every such call.
- **Mechanical edits.** `self.get_reg_raw("A1")` → `self.reg_value(1)`,
  `self.unify(&a, &b)` → `unify_values`. The bridge copies, so it is only
  allowed for builtins whose arguments stay small. **Gate:** at N=5000 the
  profile must show no bridged builtin above 1% (§11).

`Value` keeps its public shape (`Atom/Integer/Float/Str/List/Unbound/Bool`)
and gains `Var(LiveVar)`. A `Value` that contains a `Var` is live. Anything a
holder keeps is a `Detached` (§9.2), which cannot contain one.

### 7.4 Which builtins go native (phase R4)

Taken from the profile and from term size, not from popularity:

- the sort family (`sort/2`, `msort/2`, `sort/4`, `keysort/2`, `predsort/3`,
  `list_to_set/2`), `compare/3`, `==`/`\==`/`@<`-family;
- `length/2`, `nth0/nth1`, `member/2`, `memberchk/2`, `append/3`,
  `reverse/2`, `last/2`, `sum_list`-family, `exclude/include/maplist` helpers
  if runtime-implemented;
- `functor/3`, `arg/3`, `=../2`, `copy_term/2`, `is/2` and arithmetic
  comparison (evaluate on cells), `=/2`, `\=/2`, type tests;
- findall/bagof/setof/aggregate_all finalization (§9.2).

Everything else (OS, process, time, random, stream, filesystem, read_term,
format, atom/string conversion, assert/retract) starts bridged.

`length/2` (`_L`), `functor/3` (`_F`) and the maplist/foldl/predsort helpers
(`_M`) create variables. Native ports use `fresh_var(kind)` (§7.1), never a
formatted `Value::Unbound` name through the bridge.

### 7.5 Identity-preserving operations

Three operations need more than `unify`, `export` and import, because today
they recover variable identity from names or from the private trail.

**Variable collection** (`term_variables/2`, `numbervars/3`, `read_term`
`variables(_)`; today `collect_term_variable_names` + `variables_from_term`,
`DB:309-338`):

```rust
fn term_vars(&self, c: Cell, out: &mut Vec<Cell>);   // REF of each distinct unbound variable,
                                                    // depth-first, args left to right, head before tail
```

Distinct means the same cell after deref, which is what a distinct name is
today. The result holds the input's own variables: `term_variables/2` builds
its list from them, and `numbervars/3` binds exactly them to `'$VAR'(N)`
(`T:2817-2861`). Required tests: `term_variables(f(X,X),[V]), V=a` binds `X`;
`numbervars(f(X,Y),0,E)` binds `X` and `Y` themselves.

**Parser results** (`read_term` family, `T:3219-3298`, `DB:262-387`). Today
the parsed term and the parser's variable environment are copied with one
`_RP` rename map, and `singletons` and `variables` rebuild variables from
names. The rewrite uses one export session on the parser machine and one
import session in the caller:

1. on the parser machine, export the parsed term and then the `var_env`
   entries, in that order, through one export session, so one `Detached`
   holds all roots with shared variable indices;
2. in the caller, import every root through one `Fresh(RP)` session in the
   same order. `_RP<n>` numbers come out exactly as today: term variables
   first, in first-occurrence order, then any variable that occurs only in
   the environment;
3. build `variable_names` pairs from the source names and the imported
   cells, in today's order (`pairs.reverse()`, `DB:286`); build `variables`
   with `term_vars` on the imported term; build `singletons` by counting
   occurrences per cell in the imported term and mapping each cell to its
   source name through the session, with today's filtering (skip
   `_`-prefixed source names, unnamed variables print as `_`).

Required test: `read_term_from_atom('f(X)',T,[variable_names(['X'=V])]),
V=a, T==f(a)`.

**Ordered bind capture** (`unifiable/3`, `T:2897-2915`). Today it unifies the
two terms, reads the binding entries pushed since its mark, makes a pair
`Name = Value` from each binding name and its *raw* bound value, unwinds,
then unifies the output with the pair list. The raw value matters: for
`unifiable(f(X),f(Y),S)` the pair is `X=Y`, and a deep-dereferenced export
would give `Y=Y`. The trail and bindings become private, so the rewrite
exposes the capture:

```rust
fn unify_capture(&mut self, s: &Scope, a: Cell, b: Cell,
                 out: &mut Vec<(u32 /* var addr */, Cell /* value written */)>) -> bool;
```

It must run as the first operation inside a fresh trial scope `s`. It
records every bind event in order, as `(addr, cell written into heap[addr])`,
which is today's trail order. Because `s` was opened at the current H and
unification allocates nothing, every bound address is below `s.h` and every
written cell is a pre-existing cell or an immediate, so all of them survive
`rewind(&s)` (debug assertion). The builtin then rewinds, allocates
`'='(REF(addr), cell)` structures and the list with `mk_str`/`mk_list`, and
unifies the output. It never allocates before the rewind, which would
truncate the result. Required test: `unifiable(f(X),f(Y),S)` gives
`S = [X=Y]` with `X` and `Y` still unbound and distinct, as today.

## 8. Lowered emitter and Stage-2 regions

- **`wam_rust_lowered_emitter.pl` (1.1 K lines)** emits Rust that calls ~12
  `WamState` methods (`get_reg`, `put_reg`, `head_constant`,
  `match_reg_atom[_str]`, `unify`, `step`, `run`, `lo_clause_snapshot`,
  `lo_restore_clause`, `unwind_trail_to`). Port each one to the numeric/cell
  API:
  - `match_reg_atom` compares an `ATOM` id against a constant id resolved at
    load;
  - `head_constant` becomes `unify(reg, const_cell)`;
  - T4 and F11 snapshots (`LE:580-607`, `LE:804-815`) become one scope with
    the lowered-clause policy (§6.3): `open_scope` once, `rewind(&s)` before
    each later clause, `close(s)` on success, `rewind_close(s)` after the last
    failure. The F11 loop closes the iteration's scope before `continue` and
    opens a new one;
  - the lowered ITE (`LE:847-855`) becomes a scope with the policy of §6.6
    (plain after R−1c, `LoweredIte` emulation otherwise), including the
    explicit register save added in R0b;
  - lowered `call`/`execute` (`LE:1045-1062`) and `lowered_dispatch`
    (`ST:4574-4597`) run nested code and set the backtrack floor (L3). Today
    neither sets one, and the confirmed `lower_floor` divergence follows from
    it (§12 R−1a). Raising `cut_barrier` alone, as `lowered_dispatch` does,
    does not stop failure backtracking;
  - literal constants and fresh variables are emitted through the cell API
    (constant table, `fresh_var`), which also retires today's `String`
    literals (`LE:976-979`, `LE:1107-1115`, R−1f).

  This is required API migration: existing lowered configurations must keep
  working in the merge (§12).
- **Stage-2 regions** are hand-written in `state.rs.mustache`: about 1.6 K
  lines of region code (regions 1, 2, 3a, 3b, 4 and 5) plus about 1.0 K lines
  of stress tests. They are 1.46 M of N=40 and **138 M (35.5%) of N=5000**
  **[S]**. They walk catalog lists with `deref_shallow`, build `strv` terms and
  compare with `terms_identical`. A native port uses cursors, `mk_*` and
  `identical` (not `compare_std`). Their soundness gates G-1…G-5
  (`docs/reports/wam_rust_stage2_kimi_soundness_review.md`) depend on snapshot
  semantics. Under the obligation model the P2 minimal snapshot is a scope
  with the region policy: trail, heap, `var_counter`, no registers, no env.
  It is no weaker than today's, and HB now covers it.
  A region may also land on a tested compatibility path instead of natively
  (§12).
- **`rust_target.pl` hybrid wrappers** (`RT:4448-5348`, about 900 lines).
  Six generated wrapper bodies drive a foreign kernel by backtracking and
  read its results by temporary variable name: 42 `vm.bindings.get`/`remove`
  calls in all.

  | wrapper (generator clause) | lines | temp names | accumulates |
  | --- | --- | --- | --- |
  | shortest-path stream, arity 3 (`compile_rust_foreign_stream_wrapper_from_plan/…,3`) | `RT:4448-4586` | `__wrapper_target`, `__wrapper_cost` | `packed_results: Vec<Value>` |
  | A* stream, arity 4 (same, `,4`, `astar_kernel`) | `RT:4587-4737` | same | `packed_results` |
  | grouped min, arity 3 (`compile_rust_foreign_min_aggregate_wrapper_from_plan/…,3`, `grouped_min`) | `RT:4813-4986` | `__agg_target`, `__agg_cost` | `BTreeMap<String, f64>`, then `packed_results` |
  | scalar min, arity 3 (`scalar_min`) | `RT:4987-5074` | same | best `f64` |
  | A* grouped min, arity 4 | `RT:5075-5261` | same | `BTreeMap<String, f64>` |
  | A* scalar min, arity 4 | `RT:5262-5348` | same | best `f64` |

  Policy for every wrapper:
  - **names.** Create the two temporaries once per wrapper call with
    `fresh_var` (or one `ByName` import, so they keep their kind-15 names) and
    hold the returned cells. Each `bindings.get(name)` becomes
    `deref(cell)`; each `bindings.remove(name)` (clearing before the next
    start candidate) becomes a rollback, below.
  - **holders.** Open one scope after creating the temporaries and capturing
    the input/output argument cells (they predate the scope, so they survive
    every rewind). Per solution, read the target and cost cells and copy them
    out as ground values (atom text, `f64`) into the accumulator before the
    next `backtrack`. The accumulators are plain Rust data or ground `Value`s
    and hold no heap handle. Between start candidates, `rewind(&s)` replaces
    the name removal. At the end, `rewind_close(s)`, then hand the ground
    results to `finish_foreign_results` as today.
  - **floor.** The wrapper loop calls `vm.backtrack()` with no floor today;
    when the kernel's stream is exhausted it can resume an older caller CP.
    The scope sets the floor (L3) so the loop stops instead. R−1a checks this
    case on the frozen build.

## 9. Boundary

### 9.1 Shims and `call_pred`

`examples/pkg_resolver/rust/shim/main.rs` and `rust_store/shim/main.rs`
(~760 lines each) build `Value`s (`catalog_term`) and read results (`sel_json`
over `Value`). Keep that code. Replace `call_pred`'s body with:

```rust
vm.call(pred, &[cat, reqs]) -> Option<Value>
// reset_query; import every arg through one session into the base segment;
// set A1..An and a fresh output var (kept as a cell, not looked up by name);
// run; export the output as a Detached and convert it to a plain Value
```

Copy-out produces exactly the `Value` the shim formats today: the same atom
text, the same float `Display`, `[]` as an atom (an empty `Value::List` and
the atom `[]` print the same), cons cells as `Value::List`, and unbound
variables as `Value::Unbound(name)` with today's name. A partial list with an
unbound tail is still built as a `"[|]/2"` `Str`, to match today's
`deref_heap`. The `json.rs` files do not change.

### 9.2 Live handles and detached terms

Heap-top reset is sound only if no Rust-side value keeps a heap address above
an obligation's `h` across a rollback to it. A `Value` with `Var(LiveVar)` is
such an address. So the rewrite has two representations.

**Live bridge handle**: `Value::Var(LiveVar)` or a raw `Cell`. Valid only on
its own machine, and only under the rules in §5.6 and §7.3. Never stored in a
holder.

**Detached term**: no heap handles.

```rust
pub struct Detached { root: DTerm, vars: Box<[DVar]> }
enum DTerm { Atom(AtomId), Int(i64), Float(f64), BigInt(i64), Bool(bool),
             Str(FunctorId, Box<[DTerm]>), Cons(Box<(DTerm, DTerm)>), Var(u32) }
struct DVar { name: VarName, anchor: Option<(u32 machine, u32 addr)> }
```

- Variables are indices into `vars`, the term's identity map. Two
  occurrences of one variable share an index, so sharing within the term is
  kept.
- `name` is the variable's `(kind, n)`, so it prints as today.
- `anchor` is a **hint**, not a handle. On import it is honored only if the
  machine id matches, `addr < H`, and `heap[addr]` is an unbound `VAR` with
  exactly this name. Otherwise it is ignored. So a detached term is safe across
  rollback, machine destruction and transfer to another machine.

**Export** (`export(&self, cells: &[Cell]) -> Detached`, or a session for
several terms): walk the deref'd term, give each distinct variable one index,
record its name and an anchor.

**Export order matters where a term is reused.** Export reads the term as it
is now, bindings included. A holder that must re-apply a term's original
variable structure later has to export (or build) it before anything binds
its variables. The case today is the renamed dynamic clause kept for
`dynamic_rule_body` redo: `dynamic_call_attempt` renames the clause, applies
head and body, and only then stores `clause.clone()` in the rule-body CP
(`DB:562`, `DB:581-590`); `dynamic_rule_body_attempt` stores it again after
re-applying (`DB:692`, `DB:702`). Today that works because the stored
`Value` holds names whose bindings the backtrack undoes. Exported after
binding, `assertz((p(X) :- member(X,[a,b])))` would give the template
`p(a) :- member(a,[a,b])` after the first answer and lose `b`. The rewrite's
order, inside the attempt scope (§5.6):

1. take the stored clause `D` (a `Detached`, no anchors);
2. rename: produce the template `R`, which is `D` with its variables given
   their `_C<n>` names (pre-increment, first-occurrence order). `R` is built
   directly from `D`, or exported from the fresh import before anything is
   bound; either way no binding is visible in it;
3. import `R` with `PreserveName` (new cells carrying the names step 2
   assigned; the counter moved only in step 2, as in today's
   `copy_term_walk`), unify the head, then solve the body;
4. on success push the CPs; the rule-body redo record holds `R`;
5. every retry (`dynamic_rule_body_attempt`) imports the same `R` with
   `PreserveName`, re-applies it, and on success pushes a CP holding the
   same `R`, never a re-export.

`dynamic_clause_attempt` (`clause/2`) and `dynamic_retract_attempt` keep no
clause in their redo data (key, patterns, index), so they import `D` with
`Fresh(C)` per candidate and need no template.

**Import policies.** Every import is through a session, so one name→cell map
covers everything imported together.

| policy | for a `DVar` | today's behavior it reproduces |
| --- | --- | --- |
| `Anchored` | valid anchor → `REF(addr)`; else a new variable carrying the same name, no counter change; same name within the session → same variable | today a name is a global identity. An outer variable unbound at collection is still that variable afterwards; a variable created inside the goal keeps its unique name, because CP backtracking does not restore `var_counter` |
| `PreserveName` | a new variable carrying the same name, no counter change; same name → same variable | re-applying an already-renamed clause |
| `Fresh(kind)` | each distinct variable → a new name from `var_counter` with the kind's convention (§3.4), in first-occurrence order | `copy_term_walk`, `copy_external_term_from` |
| `ByName` | for `Value::Unbound(name)` from outside: same name → same variable in this session | shims, fact sources |

**Holders.**

| holder | today | rewrite: representation | import |
| --- | --- | --- | --- |
| `aggregate_acc` (findall/bagof/setof/aggregate_all) | `deref_var(deref_heap(v))` per solution (`T:1036-1041`), unbound variables kept by name; one machine-wide vector, cleared by every `BeginAggregate` (`T:981`), so a nested findall wipes the outer one's solutions | `Vec<Detached>`, exported at `EndAggregate` before the backtrack. After R−1b: one vector per active aggregate frame (a stack); without R−1b: the same single vector | finalization imports all solutions in one `Anchored` session, then builds the list, or sorts and dedups with `compare_std` (setof), or reduces |
| `thrown_ball` | deep-deref'd `Value` at `throw/1` (`T:6957-6966`) | `Detached`, exported at `throw` before any rollback | `catch/3`: `Anchored`, after the rewind (below) |
| error terms (`raise_builtin_error`, `raise_iso_error`, syntax errors) | `Value` with a fresh `_MB`/`_EC`/`_SE` variable | `Detached` built directly; the variable takes its name from `var_counter` (pre-increment) at creation | as a thrown ball |
| `dynamic_db` clauses | `Value` renamed to `_C<n>` at assert (`DB:399`) | `Detached`, no anchors | retrieval: `Fresh(C)` (`DB:561`, `DB:912`, `DB:994`) |
| redo data holding a renamed clause (`dynamic_rule_body`, `DB:581-590`, `DB:681-708`) | `Value` stored after the clause was applied | `Detached` template `R`, built **before** any head/body binding (export order above) | `PreserveName` on each retry; the same `R` is stored again, never re-exported |
| `vm.call` result (public) | `deref_heap` of the output | `Detached` → plain `Value` with `Unbound(name)` | none |
| `par_aggregate` inputs | parent `Value`s set into each fork's registers | `Detached` from the parent; a worker's captured external args and its enumerated input are imported in one session, so names shared between them stay one variable | `ByName`/`PreserveName` into each worker |
| `par_aggregate` results (`par_aggregate.rs.mustache:155-245`) | each worker's `Value`s; worker variable names come from each fork's counter | `Detached`, exported before each worker backtrack; anchors carry the worker's machine id (workers are forks, §9.4), so the parent ignores them | one `PreserveName` session for the whole batch, so equal names alias as today. Residual difference: today a worker name equal to a live parent name would alias with the parent variable; here it does not. Resolver par results are ground |
| `read_term` parser machine (`T:3238-3271`) | separate `WamState`; copied with `_RP` renaming | one `Detached` holding the parsed term and the `var_env` entries (§7.5) | one `Fresh(RP)` session for all roots |
| parser operator table `ops` (`T:3248-3256`) | `Value` read before `parser.reset_query()`, used after it | `Detached`, exported before the reset (§3.3) | `PreserveName` into the parser after the reset |
| instruction constants (`instructions.rs.mustache:12`, `:26`, `:30`, `:44`, `:46`, `:134`) | `Value`s inside `Instruction`, alive for the program | pre-encoded `Cell`s; boxed ones in the program's constant-box arena (§3.3) | none; valid on every machine of the program |
| `unifiable/3` substitution pairs (`T:2904-2915`) | raw `Value` pairs read from the trial trail, kept across the unwind | `Vec<(u32, Cell)>` from `unify_capture`, cells below the trial scope's `h` (§7.5) | none; the result list is built after the rewind |
| hybrid-wrapper accumulators (`RT:4555`, `:4705`, `:4915`, `:5040`, `:5188`, `:5325`) | `packed_results: Vec<Value>`, `BTreeMap<String, f64>`, best `f64`, filled across kernel backtracking | ground values only, copied out per solution before the next backtrack (§8) | none |
| `dynamic_body_solutions` (`DB:624-665`) | whole-machine clones, `*self = solution` | snapshot clone and adoption (§9.4): same machine id, scope records copied, the attempt scope's token stays valid | none |
| `BuiltinState` redo data | `Vec<Value>` | `Vec<Cell>` under §5.6, or `Detached` | per kind |
| global variables (`b_setval`/`nb_setval`) | not implemented | if added: `Detached` | `Anchored` for `b_`, `Fresh` for `nb_` |
| lowered/region locals across a nested `run` | `Value` | `Cell`s, valid only while the scope that was active when they were read stays active and is not rewound below them | none |

**`catch/3` ordering.** Today: snapshot, run the goal; on a ball, restore the
snapshot first, then unify the catcher with the ball (`T:6968-7034`). The
rewrite keeps that order:

1. `throw/1` exports the ball to a `Detached` immediately, while every cell
   it references is still valid, and sets `thrown_ball`;
2. the run loop aborts without backtracking, as today (`T:1405-1408`);
3. `catch/3` does `rewind(&s)` on its scope, which removes the CPs with
   `gen ≥ s.cp_gen` and restores trail, H, boxes, env, registers, `cp`,
   `b0`;
4. it imports the ball with `Anchored`. An outer variable that was unbound
   when thrown is unbound again after the rewind, so its anchor is accepted;
5. it unifies the catcher. On failure it rewinds that unification, puts the
   **same `Detached`** back in `thrown_ball`, and fails (rethrow). On success
   it runs the recovery.

A ball never holds a heap address that the rewind in step 3 could invalidate.

### 9.3 Fact sources and kernels

- `seek_fact_source.rs`, `csr_fact_source.rs` and the LMDB sources
  (`lmdb_fact_source_*`) hand rows to `fact_table_attempt` and
  `finish_foreign_results` as `Value`s (atoms, ints and floats; the seek
  source decodes `Float` rows, `seek_fact_source.rs.mustache:243`, `:265`).
  These become a `ByName` import on delivery, or direct `mk_*` (atoms and
  ints need no heap; a float takes a machine box). Delivery opens a scope
  before the candidate binding (§5.6).
- The shared decoded-row cache (`seek_fact_source.rs.mustache:334`, `:485`,
  `:648`) survives resets and is shared across cloned machines. It keeps
  ground `Value` rows (boundary data, no heap handles) and creates cells,
  including `BOX` floats, only at delivery. It never caches a cell or a box
  index.
- `boundary_cache.rs` (7.8 K lines) works on `u32` node ids and talks to the
  machine only through result delivery, so it needs only a thin change.
- Foreign kernels (`execute_foreign_predicate`, the native category-ancestor
  family) read registers as atoms and ints. Port them to `tag_of` and
  accessors.

### 9.4 Machine snapshot, adoption and fork

Two places copy a whole `WamState` today, for different purposes. The
derived `Clone` copies the new fields too (scopes, `next_gen`, B0, floor,
pending state, machine id), which is right for one and wrong for the other.
So the rewrite has two operations and no public derived `Clone`.

**Snapshot and adoption** (`dynamic_body_solutions`, `DB:624-665`;
`dynamic_eval_body_solution`, `DB:628`). The dynamic body solver clones the
machine per branch, runs the goal on the clone, records solution clones,
and the caller replaces itself with one: `*self = solution`. This runs
inside `dynamic_call_attempt`'s or `dynamic_rule_body_attempt`'s attempt
scope (§5.6).

- `snapshot(&self) -> WamState` copies everything, **keeping the machine
  id, epoch, scope records, serials and `next_gen`**. A snapshot is the same
  logical machine at a branch point: its heap prefix is identical, so its
  live handles and anchors mean the same thing.
- Scope tokens: the attempt scope's `ScopeRec` is copied with its index and
  serial, so the Rust frame's `Scope` token is valid on the adopted machine.
  Every scope the clone opened during the body has been closed before a
  solution is recorded (they belong to Rust frames that have returned).
  `adopt(&mut self, solution: WamState)` asserts (debug) that the solution's
  scope stack equals `self`'s by index and serial, then replaces `self`.
- Obligations and floor: clone-side execution is nested execution while the
  attempt scope is active, so it runs with the floor at that scope's
  `cp_gen` (L3). Today the solver's `vm.call_goal_value(&other)` (`DB:652`)
  sets no floor; its explicit `cp_depth` check (`DB:658`) covers only the
  enumeration loop. R−1a adds the floor on main (§12). Solution clones keep
  the CPs they pushed with `gen ≥` the solver's entry boundary removed, as
  today's `truncate(cp_depth)` (`DB:655`).
- Pending state (`pending_b0`, `pending_level`) and B0 are copied and
  adopted with the machine, as today.
- `var_counter` moves with the adopted machine; a discarded snapshot's
  increments are lost, as today (§5.7).
- Anchors exported from a discarded snapshot carry the same machine id. They
  stay harmless: an anchor is honored only if the address is below H and
  holds an unbound `VAR` with exactly the same name, which today's global
  name identity would treat as the same variable anyway.

**Fork** (`par_aggregate` workers: `base.clone()` and `*m = base.clone()`,
`PA:29`, `:52`, `:76`, `:93`, `:130`, `:156`, `:180`, `:209`, `:226`; the
labeled paths then call `choice_points.clear()` and `aggregate_acc.clear()`,
`PA:157-158`, `PA:181-182`).

- `fork(&self) -> WamState` copies the heap, boxes and environment stack
  (the inputs a worker reads) and gives the copy a **fresh machine id** and
  epoch, so the parent rejects worker anchors and live handles.
- It clears the CPs **and** the scope stack. No parent `Scope` token can
  name a worker record, and a worker never rewinds into a parent obligation.
- It resets B0, floor and `next_gen`-based boundaries to the worker's empty
  CP stack (B0 = floor = `next_gen`), clears `pending_b0`, `pending_level`
  and the aggregate accumulators, and derives HB and EB (both 0: the worker
  never rolls back below its start, and its results leave as `Detached`).
  Today the cleared CPs leave the inherited B0, pending state and floor
  pointing at depths that no longer exist; the fork makes them consistent.
- Each repeated reset of a worker (`*m = base.clone()`) is a new fork with a
  new id, so anchors from an earlier body run are rejected too.
- The generic collector (`src/unifyweaver/targets/rust_runtime/par_aggregate.rs`)
  holds only `M: Clone` and `Vec<V>`; WAM callers instantiate `M` through
  `fork` and `V = Detached`, and the generic substrate does not change.

## 10. What stays byte-identical

- **Program output** (every gate JSONL, scale `--bench` stdout) must be
  `cmp`-identical to the pre-rewrite build. The things that could break it,
  and how the design holds each:
  - variable names: the `VAR` kind and number with each site's increment
    convention (§3.4), and `var_counter` restored at exactly today's sites
    (§5.7);
  - standard order of variables: name text compare (§3.4);
  - which aliased variable survives: the bind direction is preserved (§3.1);
  - standard order of terms and sort dedup: `compare_std`, including the
    list-vs-arity-2 and int-to-`f64` cases (§7.2);
  - `==` and region keys: `identical` (§7.2);
  - float unification: the absolute-epsilon rule (§7.2);
  - float text: copy-out goes through the same `Value` `Display`;
  - `[]` and cons spellings: copy-out normalizes them to today's forms;
  - atom order in sorts: text compare, as today;
  - findall, catch and par results: detached terms with today's naming
    (§9.2);
  - control: the ITE barrier rules (§6.4) and the execution snapshots
    (§6.3);
  - error terms from `raise_iso_error`: built as `Detached` with today's
    variable names.
- **The baseline.** "The pre-rewrite build" is the build frozen **after**
  phase R−1 (§12). Each R−1 change re-freezes it explicitly. If the owner
  chooses not to land an R−1 item, the rewrite emulates that behavior
  (§12, "If not fixed first") and the frozen build stays as is for it.
- **Known residual differences**, each needing a gate case before R3 green:
  external names equal to live generated names (§3.4); non-ground
  `par_aggregate` results aliasing a parent variable by name (§9.2); CPs
  that today survive a rewind or commit because a cut crossed a depth
  boundary (`catch/3` without R−1e; the structural `;` fallback after the
  "cannot retry a nondeterministic left conjunct" warning; a lowered
  condition that leaves a CP, §6.6), which generations remove (§5.4). The
  paranoid build's "boundary crossed" counter must be zero on every gate
  corpus, or each event is explained.
- **The generated Rust source changes completely.** It is not compared.

## 11. Test strategy and gates

1. **Gates for every rewrite commit that claims "green"** (the same set as
   D120–D127):
   - term differential `run_differential_rust.sh` (2600 / 0 / 0);
   - term corpus (51/51);
   - store differential (503 / 0 / 0);
   - store corpus (51/51, plus identical to the term corpus);
   - `cmp` identity of all four JSONLs and of scale `--bench` stdout at
     N=40/250/1000/5000 against a **frozen pre-rewrite build** (keep its
     binaries and outputs as artifacts);
   - generated-crate `cargo test --release --lib`;
   - Rust WAM plunit (56 files: the same per-file rc and the same failing
     test names as the frozen base);
   - CI rust conformance (`member,builtins`).
2. **Reference-model and targeted tests** (new, in the generated crate,
   `cargo test` only). Each runs against the frozen build too where it can,
   so "same as today" is checked, not assumed.
   - **Obligation model** (Appendix A): random sequences of push CP, trust,
     cut to a generation, open scope, rewind, close, allocate variable, bind,
     backtrack, against a naive machine that trails every bind. After every
     step the heap, the env slots and HB/EB must match, and no CP may have
     `h` above H. Fixed cases: the three §5.4 counterexamples (counterexample
     3 is cut through a scope + new CP + redo + rewind) and the three-clause
     fallback.
   - **Abandoned environments** (§6.3): a `predsort` comparator with an ITE
     whose condition first-writes a Y slot of the comparator's frame, a
     second comparator call reusing that frame space, then an outer rewind;
     the caller's Y slots and the live frame must be unchanged, and the
     pruned entry count must equal the nested Y writes.
   - **Lowered ITE** (§6.6): `tests/test_wam_rust_lowered_ite_exec.pl`
     against the frozen build and an interpreter run of the same predicates.
   - **Dynamic DB** (§5.6, §9.2): `assertz((p(X) :- member(X,[a,b])))` then
     all solutions of `p(X)` give `[a,b]`; a dynamic predicate with two
     clauses whose first has a body that yields two solutions gives the body
     solutions before the second clause; a dynamic body that fails with an
     older CP present (floor).
   - **Identity** (§7.5): `term_variables(f(X,X),[V]), V=a` binds `X`;
     `numbervars` binds the input's variables; `read_term_from_atom('f(X)',T,
     [variable_names(['X'=V])]), V=a, T==f(a)`; `unifiable(f(X),f(Y),S)` gives
     `[X=Y]` with both unbound.
   - **Resets** (§3.3): two `read_term` calls on one machine (the operator
     table survives the parser reset); a program with a float, a big-integer
     and a Bool instruction constant run across several `vm.call`s, rollbacks
     and a par fork.
   - **Baseline divergences** (§12 R−1): each R−1 program, compared with
     SWI-Prolog after its fix, and with the re-frozen build in the rewrite.
   - **Execution snapshot**: a lowered clause that allocates an environment
     and fails, after which clause 2 must see the entry E, continuation and
     B0; a nested comparator run (`predsort`) whose caller's Y slots and E are
     unchanged afterwards; the §6.2 `p/2` shape (Y first written in a failed
     condition and read after the ITE), which must give the frozen build's
     result.
   - **Builtin redo lifetimes**: redo data holding a boxed float and a big
     integer across a backtrack (paranoid retention check); a fact-table and
     a dynamic-DB candidate that binds a variable allocated between the older
     CP's `h` and the attempt's heap top, then backtracks into the next
     candidate (that variable must be unbound again).
   - **Detached terms**: `findall(f(X), member(X, [A, B]), L)` with `A`, `B`
     unbound; a findall whose goal creates fresh variables; a findall
     sharing an outer unbound variable; `catch(throw(f(X)), f(Y), true)` with
     `X` created inside the goal and with `X` outer; rethrow through two catch
     frames; assert then call a clause with shared variables; a non-ground
     `par_aggregate` result; `vm.call` output with unbound variables.
   - **ITE**: the list in §6.4.
   - **Comparator and operations**: `[a]` vs `f(b,c)` (Equal; `sort/2` keeps
     one, `==` false, `group_keyed` keeps both); `[a|T]` vs `f(b,c)` (Less);
     2^53 vs 2^53+1 (Equal); `1` vs `1.0` (Float first); NaN; `Bool(true)`
     vs the atom `true`; `_V10` vs `_V9`; msort stability of Equal elements;
     float unify within `f64::EPSILON`; `copy_term/2` numbering with shared
     variables.
   - **Variable names**: a failing alternative that creates variables, then a
     fresh variable (counter not restored); a negated goal that creates
     variables; a T4 lowered clause retry and a region decline (counter
     restored).
   - **Cursors**: maplist whose goal binds later elements; maplist whose
     goal backtracks internally; predsort whose comparator allocates.
   - **Backtrack floor**: `catch/3` whose goal fails with an older CP
     present; lowered `call` and `execute` whose callee fails with an older
     CP present (both emit modes); a hybrid wrapper whose kernel stream is
     exhausted with an older CP present (§12 R−1a).
   - The existing property tests: keep today's `Value` algorithms for unify,
     `term_compare`, `terms_identical`, sort/dedup and `copy_term` as a
     test-only `term_ref` module; generate random terms (atoms with tricky
     text, big and small ints, floats, nested structures, partial lists,
     shared and aliased variables) and check that
     `export(op_cell(import(t)))` equals `op_ref(t)`, including variable
     names.
3. **A debug "paranoid" feature:**
   - trail every bind and every slot write (conditional trailing off);
   - check every retained cell against §5.6 at resume;
   - check L1 and L3 on every scope operation and backtrack;
   - check live handles (§7.3);
   - count `CutTo` not-found events (§6.4);
   - count "boundary crossed" cuts (§5.2) and skipped dead env-slot undos
     (§5.1);
   - tag env-slot entries with the frame's allocation serial and check them
     on undo (§6.3);
   - count `set_y_untrailed` writes if the `LoweredIte` emulation is used
     (§6.6);
   - poison truncated heap cells.

   The differential must give identical output with it on and off.
4. **Unit-test churn.** About half of the 260 lib tests poke internals:
   `bindings`, `TrailKey`, `YRegs`, `AxDirty`, D121–D127 models. Rewrite the
   behavioural ones on the new API. Delete the ones that test removed
   machinery, and say so in the commit.
5. **Perf gates per phase** use the Ir probe in §12. A phase that regresses
   N=5000 Ir by more than 2% versus the previous phase needs a written
   reason.

## 12. Phase plan

The user chose a big-bang rewrite: one long-lived branch, one merge, all gates
green. The order inside the branch still matters, and the spike sets it.
R−1 (recommended) and R0a/R0b (optional) land on main **before** the branch,
each as separately reviewed changes. G0 is a measurement gate before R2.

| phase | content | why here (numbers **[S]**) | exit gate |
| --- | --- | --- | --- |
| **R−1** (recommended, main; owner's decision) | §12.1: R−1f, then R−1a–R−1d, optionally R−1e, each its own D-row | the rewrite needs L3 anyway; byte identity should target corrected behavior | each item's SWI comparison; full gate set; explicit re-freeze of the byte-identity baseline after each |
| **R0a** (optional, main) | Arity-only register save at **clause-entry** `try_me_else`/`TryMeElsePc` (not `L_ite_else_*`). The generator emits the arity; no label parsing. Keeps the environment stack snapshot, `restore_ax_regs` clear-then-restore, and the register trail. **Not** ITE guards, aggregate frames, builtin/fact/dynamic/foreign CPs, `lo_clause_snapshot` or any lowered mid-clause snapshot: those keep `save_regs` | measured s1→s2: −1.83 M Ir (−7.9% of base), on top of s1; lib 260/260 in the spike | the full D127 gate set; lib tests; new tests: an unsaved temporary is `Uninit` in clause 2 as today; ITE guards still restore temporaries |
| **R0b** (optional, main) | Remove **A/X** register trail entries. Y entries stay until R2's env-slot trail replaces them. First: (1) an audit table of every rollback consumer (the 61 + 35 `unwind_trail_to` sites, plus `lo_restore_clause`, `catch/3`, `predsort_order`), each classified as *restores registers explicitly*, *reads no A/X register that the scope could have clobbered*, or *relies on register entries*; (2) explicit saves for the third class. The lowered ITE (`LE:838-855`) is a known candidate: its condition's `vm.step` arms trail the registers they write, and the else branch reads registers; (3) targeted tests, written and passing before the removal: a failed `call/1` disjunction whose left branch clobbers A1; a lowered ITE whose condition calls a predicate and whose else reads A1; `\+` then a temporary read; `forall`; include/exclude with a failing goal; `catch/3` goal failure | measured s2→s3: −1.88 M Ir (−8.1% of base) is an upper bound: s3 also dropped Y entries and was never lib-tested. The spike calls s3 unsound in general | the audit table in the PR; the targeted tests; the full D127 gate set; lib tests |
| **R1** | Generator: numeric `Reg` operands, X/Y split variants, PC-resolved control, `BuiltinId` (from the 216-key inventory), constant table with the constant-box arena, `TryMeElse{arity}`, `TryMeElseIte{live}`, `LevelId` operands, `FunctorId` switch keys, the Y-without-Allocate check. The other instruction emitters move with it: both benchmark generators (`examples/benchmark/generate_wam_effective_distance_benchmark.pl`, `generate_wam_rust_matrix_benchmark.pl`, both backends) and the runtime's textual instruction parser (`ST:7059-7063`) | bucket (b) is ~18.5% and `step` self ~8% of N=40. Everything after it is written against the new encoding, so it comes first | the crate compiles against R2's skeleton; generator plunit |
| **G0** (gate, before R2) | Prototype on the proposed representation, in a scratch crate like the spike: (1) one native sort/comparator path (`msort` + `sort/2` dedup with `compare_std`) on the N=5000 catalog lists; (2) one large region (`group_keyed`, 59.8 M today) on cells. Measure Ir, wall, **boundary copy-in and copy-out**, and **peak memory** | §14's N=5000 ranges for these two are unvalidated; the spike measured cost today, not on cells | the measured residuals, with boundary cost, are inside §14's targets, or §14 and the R4/R5 scope are revised before R2 starts |
| **R2** | Core machine: `Cell`, heap, `VAR` names, trail (heap + env slot), obligations and scopes, CP plus arg stack, env stack with EB, execution snapshots, ITE barriers, S register, every step arm, `backtrack`, cut, `run`. All builtins through the **bridge** | buckets (a), (c) and (d), about 32% of N=40, plus construction ~8% | builds and runs; the model and targeted tests (§11.2) pass |
| **R3** | Boundary: `vm.call` copy-in/out, both shims, fact sources, kernels, `par_aggregate` (fork, §9.4), detached holders (findall/`copy_term`/assert/exceptions/`read_term`/parser `ops`), all six `rust_target.pl` wrappers (§8), the benchmark driver and benchmark-generator read-backs (`_N`, `Hops`), `tests/fixtures/wam_rust_dynamic_builtins.rs`. The tracked crate `examples/pkg_resolver/rust/uw_resolve_wam/` is regenerated from the changed templates, never hand-ported | nothing runs end-to-end without it, so R2+R3 is the first point where the differential can run | **differential, corpus and byte identity green** with every builtin bridged |
| **R4** | Native hot builtins (§7.4) | 23 sort-family calls are 196 M of 388 M at N=5000; R2/R3 with a bridged sort is about today's cost, because today already materializes | the N=5000 profile shows no bridged builtin above 1% |
| **R5** | Lowered emitter on the cell API (required). Stage-2 regions: native, or a tested compatibility path (below) | regions are 138 M (35.5%) at N=5000 | region stress tests and gates; the G-1…G-5 re-check for each native region |
| **R6** | Remove the old internals: `bindings`, `Args` spine and `deref_memo` flags, `"f/N"` functor syms and `decomp`, `WriteCtx`/`UnifyCtx`, `YRegs`, the register-name tables. Cold builtin families stay bridged | code size, compile time; removes the inlining tax of a dual representation, which the spike saw as +7.6 M at N=5000 | full gates |
| **R7** | Store lane rebuild and gates, then perf pass: ITE live sets from the compiler, dropping the dirty-slot clear if gates allow, base-segment reuse across resolves | store gates are part of "green" | the full D127 gate set, plus Ir/wall report |

**What R0a and R0b can claim.** The spike variants are cumulative, and s3
includes s1. The two R0 changes correspond to the s1→s2 and s2→s3 steps:
about −1.83 M and at most −1.88 M Ir, roughly −16% of base together, measured
on top of s1 rather than on main. Neither the cumulative −17.6% Ir nor the
−25% wall applies to them. Their wall effect was not measured separately.

**Required for the merge, by kind.**

| requirement | why | what satisfies it |
| --- | --- | --- |
| API migration (correctness) | the merge must run every program that runs today | every builtin and region compiles and runs on the cell machine, natively or through the bridge; the lowered emitter emits the new API, so existing lowered configurations work; all gates green |
| Native hot builtins (performance) | without them the N=5000 slope does not move, and the bridge adds copies | R4 in the same merge: no bridged builtin above 1% at N=5000 |
| Native regions (performance) | 35.5% of N=5000 | optional for the merge. A region may land on a tested compatibility path: (a) disabled, so the region's predicates run interpreted (declining is always sound today), or (b) its existing logic on the bridge. Either needs its stress tests green. The merge then states which N=5000 target it claims (§14) |

### 12.1 R−1: baseline fixes (decision for the project owner)

The second external review compiled small programs with
`write_wam_rust_project/3` and compared them with SWI-Prolog. It confirmed
five behaviors where today's runtime differs from SWI, plus a lowered-ITE
divergence between the two Rust tiers. This revision re-ran all of them at
`d0c1c06` (SWI-Prolog 9.0.4; Rust interpreter and functions mode) and found
two more of the same kind. The programs, verbatim:

```prolog
% catch_floor: query all catch_floor(X)
catch_floor(X) :- catch(catch_bad, _, true), X=wrong. catch_floor(ok). catch_bad :- fail.
% lower_floor: query all lower_floor(X) (functions mode, default features)
lower_floor(X) :- lower_helper, X= -1. lower_floor(0). lower_helper :- lower_bad(2), 3=3. lower_bad(1).
% ite_map: query all ite_map(no,R)
ite_map(A,R) :- (ite_q(A,X)->true;true), ite_r(X,R). ite_q(yes,a). ite_r(a,a). ite_r(b,b).
% cut_after_call: query all cut_after_call(R)
cut_after_call(R) :- cut_q, !, cut_r(R). cut_after_call(fallback). cut_q :- cut_t, cut_t. cut_t. cut_r(first).
% nested: query nested(L)
nested(L) :- findall(A,(nested_a(A),findall(B,nested_b(A,B),_)),L).
nested_a(a). nested_a(b). nested_b(a,one). nested_b(b,two).
% rundoite: lowered_rundoite_2(c,els) directly, and all rundoite(c,els) interpreted
rundoite(X,R) :- ((Y=a,Y=b)->R=then;R=els), X=Y.
% found in this revision:
catch_cut(X) :- catch((!,member(X,[a,b])),_,true). catch_cut(c).
:- dynamic dq/1.   % run after assertz((dq(Y) :- dbad(Y)))
dyn_floor(X) :- dq(X), X = wrong. dyn_floor(ok). dbad(_) :- fail.
```

| case | SWI | Rust today | cause |
| --- | --- | --- | --- |
| `catch_floor` | `[ok]` | interpreter `[]` | `catch/3` meta-calls its goal (`T:6988`; nested run at `T:7476`) with no backtrack floor; the failing goal's nested run resumes the caller's clause CP, and catch cannot restore the lost alternative |
| `lower_floor` | `[0]` | functions mode `[]`, default features | lowered `call` runs a nested `vm.run()` with no floor (`LE:1051`); `lowered_dispatch` raises `cut_barrier` only (`ST:4581-4583`) |
| `ite_map` | `[a,b]` | interpreter `[]` | after an ITE the compiler keeps the Then branch's variable map (`WT:2358-2361`); the else path's `put_value` reads an absent Y (`T:373`) |
| `cut_after_call` | `[first]` | interpreter `[first,fallback]`, functions `[first]` | B0 is a machine register, set by `Allocate` (`T:713-716`) and not restored by `Deallocate` or `Proceed`; after `cut_q` returns, the `!` uses the B0 `cut_q`'s clause left. `lowered_dispatch` restores B0, hence the functions-mode result |
| `nested` | `[[a,b]]` | interpreter and functions `[[b]]` | every `BeginAggregate` clears the one `aggregate_acc` (`T:981`), and finalization clears it again (`T:5133`) |
| `rundoite(c,els)` | true | lowered `true`, interpreter no solution | §6.6 |
| `catch_cut` | `[a,b,c]` | interpreter prints `a`, then the enclosing failure-driven driver fails | `catch/3` does not make its goal opaque to cut (it does not set B0 the way `call_goal_once` does, `T:7262-7266`), so `!` truncates to the enclosing B0 (`T:7376`); with the clause having no environment, that B0 is the caller's |
| `dyn_floor` | prints `ok`, driver succeeds | prints `ok`, then the driver fails | probably the dynamic body solver's unfloored `vm.call_goal_value` (`DB:652`), same shape as `catch_floor`; R−1a confirms the cause |

The review collected solutions with `findall/3`. This revision used
failure-driven drivers (`( G, write(R), nl, fail ; true )`), which show the
floor cases differently: the answer is printed from inside the nested run,
then the driver itself fails. Same cause. The lowered corpus with atom
constants does not compile with default Cargo features (24 `String`/`Sym`
errors, reproduced), so the functions-mode atom cases ran with
`--no-default-features`.

**Recommendation: fix these on main first, as separate small D-rows, and do
not emulate them.** The rewrite's heap invariant needs the floor rule L3
(§5.2) anyway: a nested run that resumes a CP older than an active scope
sets H below that scope's `h`, and the scope's later rewind is then
meaningless. Emulating today's floorless runs needs a "crossed scope" state
that the paranoid build cannot check (below). The other items are cheap on
today's code and make the rewrite's byte-identity target the corrected
behavior.

| item | change on main | files | regression test (compared with SWI) |
| --- | --- | --- | --- |
| **R−1f** (first) | the lowered emitter emits atom constants as `Value::Atom("x".to_string())` (`LE:1114`) and fresh variables as `Value::Unbound(format!(..))` (`LE:978`), which do not compile under the default `intern` feature (D96: `Atom(Sym)`); emit `.into()` / the interner path instead. No behavior change | `wam_rust_lowered_emitter.pl` | the lowered tests (`tests/test_wam_rust_lowered_*.pl`, the cut-semantics `functions` mode) build with default features |
| **R−1a** | backtrack floor for every nested execution: `catch/3` goal and recovery, lowered `call`/`execute` and `lowered_dispatch`, the dynamic body solver, the hybrid-wrapper `backtrack` loop (§8); each saves the floor, sets it to its entry depth, restores it | `wam_rust_target.pl`, `wam_rust_lowered_emitter.pl`, `state.rs.mustache`, `dynamic_db_methods.rs.mustache`, `rust_target.pl` | `catch_floor` → `[ok]`; `lower_floor` → `[0]` in both emit modes; `dyn_floor` → driver succeeds; a wrapper with an older CP present |
| **R−1b** | one accumulator per active aggregate frame: `BeginAggregate` saves the outer accumulator in its frame and finalization restores it | `wam_rust_target.pl`, `state.rs.mustache` | `nested` → `[[a,b]]` in both emit modes |
| **R−1c** | a permanent variable first seen inside an ITE and read after it is initialized on every path. Preferred form: initialize it before the guard (a fresh variable in its Y slot), so later occurrences are not first occurrences; this also gives §6.6 its plain policy. Alternative: initialize it at the end of each branch whose map lacks it | `wam_target.pl` (shared) | `ite_map` → `[a,b]`; `rundoite(c,els)` succeeds in both tiers |
| **R−1d** | save B0 in the environment frame at `Allocate`, restore it at `Deallocate` (standard WAM) | `wam_rust_target.pl`, `state.rs.mustache` | `cut_after_call` → `[first]` in both emit modes |
| **R−1e** (optional) | make the `catch/3` goal opaque to cut: set B0 to the entry depth around the goal, as `call_goal_once` does. Catch stays first-solution (SWI's `catch/3` is re-entrant; that difference is out of scope) | `wam_rust_target.pl` | `catch_cut` → `[a,c]` (first-solution catch; SWI `[a,b,c]`) |

Rules for each item:

- its own D-row, PR and review; it changes only what the row says;
- the regression test runs the program under SWI and the Rust build and
  compares, in every emit mode the item touches;
- the full gate set (§11.1) stays green against SWI (term and store
  differential, corpora, plunit with the same failing-test set);
- then the byte-identity baseline is **re-frozen explicitly**: rebuild the
  frozen binaries and outputs, record the commit in the D-row, and keep the
  old ones until the next re-freeze. Where an item changes no resolver
  output, the re-freeze still records that fact.

R−1c changes the shared compiler, so every WAM target's output changes for
the affected shape. It needs those targets' suites, or a Rust-only variant
(the Rust target inserting the initialization when it emits the ITE). That
choice is part of the owner's decision. R−1e is optional because it is a
semantic change beyond the review's list; without it the rewrite emulates
the commit (below).

**If not fixed first.** What the rewrite must reproduce, per item:

- **R−1a:** floorless nested runs at those sites. A backtrack that resumes a
  CP with `gen <` an active scope's `cp_gen` marks every such scope
  *crossed* and drops it from `derive`. The owner's later rewind or close of
  a crossed scope restores only today's non-heap parts (E and `e_top`,
  registers, `cp`, B0) and skips trail and H, matching today's
  `unwind_trail_to(mark)` and `heap.truncate(mark)`, which are no-ops once
  the trail and heap are below the mark. Today's stack restore also
  resurrects the old Y slot contents; the rewrite cannot, because env slots
  are not snapshotted, so the emulation is exact only where no frame above
  the crossed CP's `e_top` was reused. The paranoid L3 check is disabled at
  these sites. Not recommended.
- **R−1b:** one machine-wide accumulator, as §9.2 states.
- **R−1c:** the interpreter keeps §6.2's env-slot rule (reproduces `[]`);
  the lowered ITE uses the `LoweredIte` emulation (§6.6).
- **R−1d:** B0 outside frames, in the execution snapshot (§6.1, §6.4).
- **R−1e:** `catch/3`'s close truncates by the depth recorded at open, not by
  generation, so a CP pushed after a cut through the scope survives the
  commit as today. Its rewind still uses the generation, for soundness; the
  difference from today on that path is listed in §10.
- **R−1f:** freeze the lowered baseline with `--no-default-features`; R5
  replaces the emitted code anyway.

## 13. Risks

1. **Size and drift.** The rewrite touches ~9.9 K template lines, ~11 K
   generator lines and ~6 K lines across the other templates, shims and
   `rust_target.pl` (§15). Main keeps moving: other agents land D-items in
   the same files. Mitigation: freeze perf work on `rust_wam` for the
   duration, rebase weekly, and keep R0a/R0b small so they land first.
2. **Conditional-trailing soundness.** A rollback that HB does not cover gives
   silent wrong answers. Mitigation:
   - HB and EB derived from all obligations on every transition (§5.2);
   - private `trail`/`heap`/`cps`/`scopes` fields and the scope API;
   - scope-before-candidate at every attempt site (§5.6);
   - the paranoid feature (always trail), A/B'd in CI on the differential;
   - the obligation model test (Appendix A).
3. **Heap-top reset and dangling cells.** Any cell kept above an obligation's
   `h` across a rollback dangles. Mitigation: §5.6's per-tag rules, live
   handle validation (§7.3), and `Detached` for every holder (§9.2).
4. **Byte identity of variables and order.** Names, `var_counter` restoration
   sites, the bind direction and the comparator's quirks must be preserved.
   §3.4, §5.7 and §7.2 specify each from today's code; §11.2 tests each.
5. **Bridge cost hiding in the slope.** A bridged builtin on a catalog-sized
   term is O(N) per call. Mitigation: the R4 gate (no bridged builtin above 1%
   at N=5000).
6. **No GC.** Long deterministic runs keep heap garbage until the query ends,
   where `Arc` used to free it eagerly. That is fine for the resolver. Other
   programs with long forward recursion over large data could grow memory.
   Mitigation: a heap cap with a clear error, a measured memory column in the
   scale bench (G0 starts it), and a later "heap compaction at deterministic
   points" item. Do not block the rewrite on it.
7. **Stage-2 region proofs** were written against the old snapshot semantics.
   Each native region needs a re-check (R5).
8. **`par_aggregate`.** Forking clones the heap, an O(H) `memcpy` versus
   today's O(1) `Arc` clones. At N=5000 the base segment is ~10⁵ cells
   (~0.8 MB) per fork. Measure it. The fallback is sharing the base segment
   read-only (`Arc<[Cell]>`) across forks.
9. **Unit-test churn** (§11.4), and plunit tests that grep generated Rust text
   (`test_wam_rust_target.pl`). They will need new expected strings.
10. **The spike's codegen lesson.** Adding a variant to a hot enum de-inlined
    `same_cell`/`deref_var` (+7.6 M Ir at N=5000). Mitigation: `Cell` is a
    `u64` with explicit `#[inline]` helpers, and the old enum is deleted in R6,
    not kept alongside.
11. **Estimates.** §14's ranges are unvalidated. Mitigation: G0 before R2.
12. **Moving baseline.** R−1 changes behavior on main, and each item
    re-freezes the byte-identity baseline. A rewrite branch cut before the
    last re-freeze compares against the wrong build. Mitigation: start the
    branch after R−1 (or record which items were declined and emulate them),
    and name the frozen commit in every R-phase report.
13. **Generation boundaries change edge behavior.** Where today a cut crosses
    a depth boundary, generations give different (sound) results (§5.4,
    §10). Mitigation: the paranoid "boundary crossed" counter on every gate
    corpus, and R−1e for the one known reachable case.

## 14. Expected gains (unvalidated targets)

Every range in this section is a **target**, not a measurement. The spike
measured where cost is today. It did not measure what dereference traversal,
heap construction, copy-in/out, heap growth or fork copying cost on cells.
Do not use these ranges as schedule or acceptance assumptions until G0 (§12)
has measured the two largest. Wall columns assume wall scales with Ir; the
spike does not establish that at N=5000.

Against base D127 (Ir per warm resolve 23.30 M / 388.51 M; warm wall 4.26 /
~64–69 ms):

**N=40.** The s3 floor of 19.20 M is the whole resolve after the three spike
levers. It splits into builtins 4.61 M, regions 1.46 M and mechanics plus
everything else 13.13 M **[S]**.

| part | s3 | target |
| --- | ---: | ---: |
| mechanics and remainder | 13.13 M | 2–4 M (~100–200 Ir per instruction) |
| builtins | 4.61 M | 2–3 M |
| regions | 1.46 M | 0.6–1.0 M |
| **total** | **19.20 M** | **4.6–8.0 M** |

That is 2.9–5.1× fewer Ir than base, about 0.85–1.5 ms if wall scales with
Ir, or still ~3–6× SWI (0.26 ms). The range depends on how tight `step` gets.
The WAT target's experience is that dispatch becomes the floor once handlers
are small.

**N=5000.**

| part | base | target | status |
| --- | ---: | ---: | --- |
| sort family | 196.2 M | 35–65 M | unvalidated; G0 measures it |
| other builtins | 17.5 M | 10–17.5 M | unvalidated |
| regions | 138.0 M | 40–70 M if native | unvalidated; G0 measures `group_keyed` |
| `reset_query` | 19.1 M | ~0 | follows from `H = base` |
| mechanics | ~17.6 M | ~4 M | scaled from the N=40 target |
| boundary copy-in/out | (inside today's shim) | not estimated | G0 measures it |
| **total** | **388.5 M** | **~90–157 M plus boundary** | |

With a boundary cost of up to ~10 M that is the planning range **100–160 M**
(2.4–3.9×; 16–28 ms if wall scales with Ir from 64–69 ms; ~0.9–1.6× SWI at 17.5 ms).

Other cases:

- **Regions on a compatibility path, sort family native:** about 190–225 M
  plus boundary (~1.6–2.0× with a boundary cost up to ~10 M), if the regions
  keep today's cost. If a region is disabled and its predicates run
  interpreted, the cost is not estimated: the regions exist because the
  interpreted path was slower.
- **R4 and R5 both bridged:** expect N=5000 near today (~0.9–1.1×) while
  N=40 still gains.

## 15. Files that must change

Sizes are current line counts. "Touch" is a rough share of each file that
changes.

| file | lines | touch | what |
| --- | ---: | --- | --- |
| `templates/targets/rust_wam/state.rs.mustache` | 9,917 | ~70% | machine core, deref/unify/compare/identity/copy, obligations and scopes, builtins (core/arith/io/type/term/ext/meta), regions, snapshots, tests |
| `templates/targets/rust_wam/value.rs.mustache` | 816 | ~40% | `Value` stays the boundary type and gains `Var(LiveVar)`. New `cell.rs` (~600 new): `Cell`, tags, accessors, cursors, functor table, boxes. New `detached.rs` (~300 new): `Detached`, export, import sessions |
| `templates/targets/rust_wam/instructions.rs.mustache` | 149 | 100% | numeric operand enum |
| `src/unifyweaver/targets/wam_rust_target.pl` | 11,064 | ~45% | step arms, `backtrack`, `run`, `execute_builtin` dispatch table, `unwind_trail_bindings_only`, instruction literal emission (regs, PCs, arity, live sets, level ids, builtin ids, constants), `call_goal_once`, findall/aggregate, catch/throw, foreign predicates, `resume_builtin` |
| `src/unifyweaver/targets/wam_rust_lowered_emitter.pl` | 1,115 | ~50% | emitted API calls, scopes for T4/F11/ITE |
| `src/unifyweaver/targets/rust_target.pl` | 14,234 | ~6% (~900 lines, `RT:4448-5348`) | the six WAM-hybrid wrapper bodies (§8): 42 `vm.bindings` reads/removes, temp-name creation, accumulators, the unfloored `backtrack` loop |
| `templates/targets/rust_wam/dynamic_db_methods.rs.mustache` | 1,028 | ~40% | assert/retract/clause on `Detached`, scope-before-candidate |
| `templates/targets/rust_wam/seek_fact_source.rs.mustache` | 1,047 | ~10% | row delivery |
| `templates/targets/rust_wam/csr_fact_source.rs.mustache`, `lmdb_fact_source_heed.rs.mustache`, `lmdb_fact_source_lmdb_zero.rs.mustache`, `materialisation_setup.rs.mustache`, `lazy_category_parents.rs.mustache` | 179 + 176 + 452 + 149 + 23 | ~10% | row delivery and registration |
| `templates/targets/rust_wam/boundary_cache.rs.mustache` | 7,816 | <3% | result delivery only (`u32` kernels unchanged) |
| `templates/targets/rust_wam/par_aggregate.rs.mustache` and `src/unifyweaver/targets/rust_runtime/par_aggregate.rs` | 263 + 271 | ~40% | fork, inputs and results as `Detached` |
| builtin family templates: `filesystem_permission`, `os_error`, `os_utility`, `process`, `process_context`, `process_metrics`, `process_resource`, `random`, `stream`, `time` (`*_builtin.rs.mustache`) | ~1,365 total | ~25% | bridged: register reads, unify outputs, scopes instead of `unwind_trail_to` (22 sites; 5 more in `dynamic_db_methods`) |
| `templates/targets/rust_wam/main.rs.mustache`, `lib.rs.mustache`, `Cargo.toml.mustache` | 278 + 15 + 16 | ~30% | the bench driver uses `vm.call`; feature list (remove `deref_memo`, `intern_sym_thread`, `trail_enum`; add `paranoid`) |
| `examples/pkg_resolver/rust/shim/main.rs` | 759 | ~10% | `call_pred` → `vm.call`. `json.rs` unchanged |
| `examples/pkg_resolver/rust_store/shim/main.rs` | 753 | ~10% | same |
| `examples/pkg_resolver/rust/build.pl`, `rust_store/build.pl` | small | maybe | only if build options change |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`, 56 files) | — | some | expected generated-text fragments |
| `tests/fixtures/wam_rust_dynamic_builtins.rs` | 829 | ~60% | a real Rust integration source copied into the generated crate by `tests/test_wam_rust_dynamic_builtins.pl:150`: old string `Instruction` constructors, `bindings` lookups, register writes, `Unbound` tests. Changing the plunit caller does not rewrite it |
| `examples/benchmark/generate_wam_effective_distance_benchmark.pl` | 1,100 | ~30% | emits old instruction shapes and label-resolved PCs (`:340-377`, `:444`, `:456-523`, `:548-752`); register writes; `Hops` read back by name (`:927`, `:944`, `:967-968`) → hold the cell |
| `examples/benchmark/generate_wam_rust_matrix_benchmark.pl` | 1,668 | ~30% | both emitted backends build old `GetConstant`/`Call`/`TryMeElse`/`BuiltinCall` variants with string operands and register writes (`:375-393`, `:404-654`, `:835-838`; duplicated driver `:1023-1041`, `:1052` on, `:1572-1575`) |
| `examples/pkg_resolver/rust/uw_resolve_wam/` (tracked generated crate) | — | regenerated | regenerated from the changed templates and generator, including the new `cell.rs`/`detached.rs` and feature list; never hand-ported. Its own tests (`tests/comparator_equiv.rs`, …) follow the generated-crate row |
| `src/unifyweaver/targets/wam_target.pl` | — | small, only with R−1c | the ITE variable-map fix, if the owner chooses the shared-compiler form (§12.1) |
| generated-crate tests (`tests/comparator_equiv.rs`, `tests/intern_stress.rs`, in-template `mod d1xx_*_tests`) | — | ~50% | rewritten on the new API (§11.4) |
| docs: `docs/WAM_RUST_STATUS.md`, this design, a per-phase report under `docs/reports/` | — | — | — |

**Do not change:** `examples/pkg_resolver/resolver.pl`,
`resolver_store.pl`, the shared `wam_target.pl` compiler (beyond optionally
emitting ITE live sets and clause arity, which is additive, and R−1c if the
owner chooses its shared form, which lands on main before the branch), the
WAT, Go and other targets.

## 16. Open questions

To settle in R1:

- Does the shared compiler already compute live registers at an ITE guard
  that can be exported? If not, use the dirty set for ITE CPs at first.
- Should the trail store the old cell always (8 bytes per entry), or only the
  address with the `VAR` name re-derived? Decide by measurement in R2.

**Settled by the second review.** The first revision listed four baseline
questions here (backtrack floor, ITE variable map, B0 after a call, nested
aggregates). The second external review answered all four by running them
against SWI-Prolog, and this revision reproduced the answers: each is a
divergence from SWI. They are now R−1a–R−1d (§12.1), with the lowered-ITE
tier difference (§6.6), `catch/3` cut locality (R−1e) and the lowered
emitter's `String`/`Sym` compile defect (R−1f).

**For the project owner, before the branch:**

- Land R−1 first (recommended) or emulate (§12.1, "If not fixed first"),
  item by item.
- R−1c in the shared compiler (all WAM targets change for the shape) or as a
  Rust-only insertion.
- R−1e, and separately whether `catch/3` should become re-entrant like
  SWI's (today it is first-solution by design, `T:6969-6974`; out of scope
  here).

**To settle during R−1 and R2:**

- `dyn_floor`: confirm that the dynamic body solver's unfloored
  `call_goal_value` (`DB:652`) is the cause (§12.1).
- The hybrid wrappers' `vm.backtrack()` loop (§8): confirm on the frozen
  build whether an exhausted kernel stream resumes an older caller CP.
- Can a lowered ITE condition leave a CP and then fail (§6.6)? If yes, fix
  it in R−1; both policies remove such CPs.
- `CutTo` not found: today a missing level is a no-op; with generations the
  nearest-match search is unchanged, but a match on an older activation's
  guard would now cut by generation. The paranoid counter (§6.4) decides
  whether this ever happens.
- Constant boxes: an arena bit in the `BOX` payload (§3.1) versus a
  per-machine copied prefix of the constant boxes. The bit keeps forks and
  parser machines free of copies; measure the extra branch in box access in
  R2.
- CP generation width: `u64` costs 8 bytes per CP and never overflows; a
  `u32` reset per query with an overflow check is the alternative. Decide by
  measurement in R2.

## 17. Review response (first external review)

An external review raised 12 findings. Each was checked against the code at
`e095a00` before changing the design.

| # | finding | verified | sections changed | resolution |
| ---: | --- | --- | --- | --- |
| 1 | HB ignores active non-CP marks when CPs disappear | yes: `!` inside a meta-call truncates to the meta-call's barrier (`T:7373-7378`), and the old §5.2 set HB from the top CP only | §1, §5.2, §5.3, §5.4, §6.4, App. A | Explicit obligation stacks (CPs and scopes). HB and EB derived from the surviving tops on every transition. Cuts remove CPs only. LIFO rules L1–L3. Counterexample 1 written out |
| 2 | `rollback(&Mark)` unsafe for reusable snapshots | yes: T4 and F11 restore one snapshot per clause (`LE:580-607`, `LE:804-815`) | §5.3, §5.4, §8 | `rewind(&s)` keeps the scope active; `close`/`rewind_close` consume it. Counterexample 2 and a three-clause fallback test |
| 3 | `to_value` is not a detached copy | yes: the old §7.2 exported `Value::Var(addr)` | §1, §7.3, §9.2 | Live handle (`LiveVar`, machine and epoch validated) vs `Detached` (identity map, name, anchor hint). Holders table with representation and import policy. `catch/3` exports at throw and imports after the rewind |
| 4 | Redo lifetime check incomplete | yes; also found capture–bind–push sites (`T:4979-5013`, `T:4234-4260`, `DB:556-598`, `DB:906-951`) where conditional trailing would miss bindings | §5.6 | Per-tag retention rules. Original `REF`s. Scope opened before each candidate and converted to the CP. The pop-before-resume HB transition |
| 5 | CP protection does not replace non-CP stack snapshots | yes: `lo_clause_snapshot` saves stack and cut barrier (`ST:4524-4532`); `call_goal_once` restores the stack on success too (`T:7303`) | §5.5, §6.1–6.3 | `e_top` in CPs and scopes. `ExecSnap` with E, `e_top`, `cp`/`pc`, cut state, floor, register policy. Per-site mapping table. Y-slot determination: no live slot is overwritten except aggregate finalization (removed); first writes after CPs exist and the compiler can read them later, so slot writes into protected frames are trailed |
| 6 | CP schema omits ITE barrier state | yes (`T:909-977`, `T:1128-1190`) | §4, §5.5, §6.4 | `levels`, `pending_level`, both capture shapes, `pending_b0`, B0 rules, `CutTo` search, carried over. Per-activation argument. Required ITE tests |
| 7 | serial/kind does not guarantee name identity | yes: ten generated prefixes, both increment conventions; CP backtrack does not restore `var_counter` (`T:1420-1475`) | §3.1, §3.4, §5.7, §10 | 4-bit kind table plus kind 15 for external text. Each site keeps its convention. `var_counter` restored only at `lo_restore_clause` and `region_decline`, as today. External-name policy per import session |
| 8 | Conventional comparison changes sort output | yes: arity-only `List` vs `Str` (`T:5557-5559`, `T:5662-5664`), int→`f64` with Float-first tie (`T:5501-5512`). Also found: a *partial* list vs `f/2` compares by name today, not by arity | §7.2, §10 | `compare_std` specified from both functions, including the proper/partial split, Bool-as-atom and NaN. Float unify epsilon and exact float identity stated. Corrections out of scope |
| 9 | API conflates compare, identity, copy | yes, with one nuance: sort-family dedup by compare *is* today's behavior (`T:5089`, `T:5901`, `T:5947`); the hazard is for `terms_identical` users | §7.2, §7.3 | Separate `compare_std`, `identical`, `unify`, `copy_fresh`, `variant`, `unify_value`, `unify_values`. Caller table. Import sessions share variables across arguments |
| 10 | Borrowed views need a reentrant traversal contract | yes: maplist family and predsort run user goals mid-traversal | §7.1 | Copyable `StrRef`/`ListCursor` plus short accessors that return cells. Survival table for allocation, nested execution, rollback. Bulk borrows only for pure operations |
| 11 | R0 needs separate safety arguments; R4/R5 are performance scope | yes: s3 was never lib-tested; s3 includes s1 | §2, §12 | R0a (arity-only clause-entry save, keeps snapshots and clearing, excludes ITE/aggregate/builtin/lowered) and R0b (A/X trail removal after an audit and targeted tests; Y entries stay). Gains restated per step. Required API migration vs required performance scope separated; regions may use a tested compatibility path |
| 12 | Estimates are hypotheses; N=40 subtotal wrong | yes: 19.20 M is the whole s3 resolve | §2, §3.2, §12, §14 | N=40 recomputed from 13.13 + 4.61 + 1.46 M (target 4.6–8.0 M). All ranges labelled unvalidated targets. G0 prototypes native sort and `group_keyed` with boundary cost and peak memory before R2 |

**Where the review is partly wrong.**

- Finding 12 says the spike calls the 25% vs 17.6% wall differences noise.
  The report calls only the **N=5000** wall differences noise
  (`docs/reports/wam_rust_heap_cell_spike.md`, the paragraph under the
  headline table). The N=40 wall gain comes from 15 interleaved rounds. The
  review's conclusion still holds: one small-workload ratio does not
  establish the N=5000 wall/Ir conversion. §3.2 and §14 now say so. The spike
  report needed no correction.
- Finding 9 implies that deduplicating with `compare == Equal` is itself a
  migration error. For the sort family it is exactly today's rule. The design
  keeps it there and uses `identical` everywhere today uses
  `terms_identical`.

The review's closing advice is followed: findings 1–5 are resolved in §5,
§6 and §9 before any core runtime code (R2), the first-to-second bind
direction is kept (§3.1), and detached terms hold no heap handle (§9.2).

## 18. Second review response

A second external review swept the revised design (`d0c1c06`) for
completeness against the code (runtime sources at `7da1210`). It accounted
for 97 direct restore expressions and 35 region-decline calls, 20 literal CP
pushes (8 after binding), every variable-creation site, the full builtin
dispatch and the repository-wide consumers of removed APIs, in seven tables.
It ranked ten gaps, G1–G4 as blocking. Each citation used below was checked
against the code; the baseline programs were re-run (§12.1).

| gap | finding | verified | sections changed | resolution |
| --- | --- | --- | --- | --- |
| G1 | cuts leave stale scope boundaries: a cut through a scope (L2) makes its CP depth, floor and saved floors stale, so a post-cut CP can be refused a redo or survive a rewind above H | yes: `catch/3` calls its goal without raising B0 (`T:6988`), `!` truncates to the inherited B0 (`T:7376`), catch later truncates to its own depth (`T:6985`, `T:6991-6994`, `T:7002-7021`); the structural `;` branch is a second path (`T:7414-7418`). Reproduced an observable effect: `catch_cut` (§12.1) | §1, §5.2, §5.3, §5.4, §5.5, §6.3, §6.4, §9.2, §10, §11, §12.1, App. A | CP generations. Scope boundaries, B0, `pending_b0`, ITE levels and the floor are generations; cuts and rewinds remove `gen ≥` the boundary. Invariant I2: no CP survives a rewind above H. Counterexample 3 works the depth-2 example through; required test cut + new CP + redo + rewind. R−1e (optional) makes `catch/3` opaque to cut |
| G2 | abandoned environments leave env-slot trail entries | yes: today `call_goal_once` removes nested Y entries (`T:7279-7301`); the design had no equivalent | §5.1, §6.3, §11, App. A | prune env-slot entries of dropped frames on close of an E-restoring scope, keeping heap entries; skip undo of slots at or above the target's `e_top`; paranoid frame-serial tags; required test |
| G3 | lowered ITE restoration differs from its baseline | yes: direct lowered writes call `put_reg` without trailing (`LE:942-985`, `ST:4346-4356`); `rundoite(c,els)` lowered `true`, interpreter no solution (re-run) | §6.2, §6.3, §6.6 (new), §8, §11, §12.1 | after R−1c a plain scope; otherwise the `LoweredIte` emulation (trail every bind, keep H, untrailed direct Y writes, delegated writes undo `ABSENT` to `UNINIT`). Gate: `tests/test_wam_rust_lowered_ite_exec.pl` plus an interpreter comparison |
| G4 | dynamic redo exports the clause after binding | yes: `clause.clone()` stored after `dynamic_apply_clause_solution` (`DB:562`, `DB:581-590`; again `DB:692-702`) | §9.2 | export order: rename to a template `R` before any binding, keep the same `R` on every retry with `PreserveName` |
| G5 | one snapshot, two CPs; snapshot/adoption vs fork | yes: `DB:564` and `DB:581` share `cp_regs`/`cp_stack`/tops; `*self = solution` (`DB:628`); worker `base.clone()` then `choice_points.clear()` (`PA:156-158`, `PA:180-182`) | §5.2, §5.3, §5.6, §5.7, §9.2, §9.4 (new) | `push_cp_keep(&s)` then `push_cp_from(s)`, order and tops stated. `snapshot`/`adopt` keep machine id, scope records and serials, floor at the attempt scope; `fork` gets a fresh id and clears CPs, scopes, B0, floor, pending state |
| G6 | name-reconstruction helpers | yes: `variables_from_term` rebuilds `Value::unbound(name)` (`DB:327-338`); singletons rebuild from runtime names (`DB:355-387`) | §3.4, §7.5 (new), §11 | `term_vars` returns the input's own cells; one parser export session and one `Fresh(RP)` import session for the term and the environment, in today's order; helpers added to the migration inventory |
| G7 | `unifiable/3` needs raw substitution capture | yes: reads binding names and raw bound values off the trail, then unwinds (`T:2897-2915`) | §6.3, §7.5, §9.2 | `unify_capture` inside a fresh trial scope: ordered `(addr, cell)` events, rewind, then allocate the pairs |
| G8 | two reset-surviving holders | yes: parser `ops` read before `parser.reset_query()` and used after (`T:3248-3256`); instruction constants are `Value`s that become boxed cells (`instructions.rs.mustache:12` ff.) | §3.1, §3.3, §4, §9.2, §11 | export/import `ops` across the reset; program-wide immutable constant-box arena (arena bit in `BOX`), per-call `boxes_base` floor |
| G9 | migration scope incomplete | yes: 42 `vm.bindings` reads/removes in six wrapper bodies (`RT:4448-5348`); the fixture is copied into the generated crate (`tests/test_wam_rust_dynamic_builtins.pl:150`); both benchmark generators emit old instruction shapes | §3.4, §8, §12, §15 | wrapper table with name, holder and floor policies; fixture, both benchmark generators and the regenerated tracked crate in §15 and R1/R3 |
| G10 | confirmed baseline incompatibilities | yes: all reproduced with SWI-Prolog 9.0.4 and the Rust build at `d0c1c06` (§12.1), plus `catch_cut` and `dyn_floor` | §1, §10, §11, §12.1 (new), §16 | phase R−1 (owner's decision, recommended): R−1f, R−1a–R−1d, optional R−1e, each a D-row with an SWI regression test and an explicit re-freeze; per-item emulation if not fixed |
| nit | builtin count | yes: 154 arms / 216 keys; five families match exactly by a scripted count of the generated crate, io and meta are consistent once multi-line arms are counted | §4, §7 | counts and inventory-driven `BuiltinId` |
| nit | Float in cached rows | yes (`seek_fact_source.rs.mustache:243`, `:265`, cache `:334`) | §9.3 | Float rows; cache keeps ground `Value`s, boxes made at delivery |
| nit | cross-machine `FunctorId` | yes: `Detached` carries `FunctorId` to parser machines and workers | §4 | one process-wide functor table, like the `Sym` interner |
| nit | native fresh-variable constructor | yes: `length/2`, `functor/3`, `fresh_meta_var` create variables | §7.1, §7.4 | `fresh_var(kind)` with each kind's convention |

**Where the review is partly wrong or imprecise.**

- G2's harm. With the rewrite's absolute slot addresses, a stale entry can
  only be undone by an obligation older than the abandoned frame. That frame
  was allocated at or above the obligation's `e_top` (`Allocate` places frames
  at `≥ EB`), so the slot is dead once that rollback completes: the entry
  cannot corrupt a live frame, as it can today with name-relative undo. It
  can still index past the stack's end, and the trail grows without bound in
  long `maplist`/`predsort` loops. The design specifies pruning and skipping
  anyway (§6.3).
- The review's area-4 row for `T:1053` calls it an aggregate witness
  fallback; it is the `ParAggregate` input fallback. Its `par_aggregate`
  clone lines (`PA:51`, `:91`, `:224`) are `:52`, `:93`, `:226`, and it omits
  `:76`, `:130`, `:209`. Neither changes a conclusion.
- The review says the B0, ITE-map and nested-aggregate questions "require no
  additional redesign". Under the recommended R−1 they become fixes on main,
  which changes the frame layout (R−1d) and the accumulator (R−1b); the
  emulation path keeps the review's reading.

**Beyond the review.** Two more divergences of the same kind (`catch_cut`,
`dyn_floor`, §12.1), an unfloored `backtrack` loop in the hybrid wrappers
(§8), condition CPs left by a failed lowered ITE (§6.6), and the design's
own misnamed foreign-attempt function (§5.6, now
`apply_first_compatible_foreign_result`).

## Appendix A: obligation model test

A small executable model, written as Python-like pseudocode. The real test is
a `cargo test` in the generated crate with a random driver. It runs the
production machine (`conditional`) beside a naive machine (`naive`) that
trails every bind and every slot write, and compares them after each step.

```python
class M:
    heap: list            # cells; VAR = None, else a value
    trail: list           # (addr, old)
    cps: list             # dicts: gen, h, tr
    scopes: list          # dicts: h, tr, cp_gen, serial
    next_gen: int
    floor: int            # a generation
    def hb(self):
        return max(self.cps[-1]["h"] if self.cps else 0,
                   self.scopes[-1]["h"] if self.scopes else 0)
    def alloc(self):  self.heap.append(None); return len(self.heap) - 1
    def bind(self, a, v, naive):
        if naive or a < self.hb(): self.trail.append((a, self.heap[a]))
        self.heap[a] = v
    def undo_to(self, tr, h):
        while len(self.trail) > tr:
            a, old = self.trail.pop()
            if a < len(self.heap): self.heap[a] = old
        del self.heap[h:]
    def push_cp(self):
        self.cps.append(dict(gen=self.next_gen, h=len(self.heap), tr=len(self.trail)))
        self.next_gen += 1
    def trust(self):     self.cps.pop()
    def remove_from(self, g):                          # every CP with gen >= g
        while self.cps and self.cps[-1]["gen"] >= g: self.cps.pop()
    def cut_to(self, g): self.remove_from(g)           # never touches scopes
    def backtrack(self):
        c = self.cps[-1]
        assert c["gen"] >= self.floor                  # L3
        assert not self.scopes or c["gen"] >= self.scopes[-1]["cp_gen"]
        self.undo_to(c["tr"], c["h"])
    def open(self):
        s = dict(h=len(self.heap), tr=len(self.trail), cp_gen=self.next_gen, serial=fresh())
        self.scopes.append(s); return s
    def rewind(self, s):
        assert self.scopes[-1] is s                    # L1
        self.remove_from(s["cp_gen"])
        self.undo_to(s["tr"], s["h"])                  # s stays active
        assert all(c["h"] <= len(self.heap) for c in self.cps)   # I2
    def close(self, s):
        assert self.scopes[-1] is s                    # L1
        self.scopes.pop()
```

Driver: a random walk over `alloc`, `bind` (to a random unbound address below
H), `push_cp`, `trust`, `cut_to` (a random generation at or below the
current B0 stand-in), `backtrack`, `open`, `rewind`, `close`, respecting L1
and L3. Both machines get the same ops. After every `backtrack`, `rewind` and
`close`, `conditional.heap == naive.heap`, and no CP has `h` above H. A
depth-based variant of the same model (boundaries as `len(cps)`) must fail
counterexample 3; that keeps the test honest.

Fixed cases:

```python
# Counterexample 1
grow_to(m, 20); m.push_cp(); grow_to(m, 100); s = m.open(); grow_to(m, 110); m.push_cp()
m.cut_to(1); m.bind(50, "x"); m.rewind(s); m.close(s)    # gen 1 is the inner CP
assert m.heap[50] is None
# Counterexample 2 and the three-clause fallback
grow_to(m, 20); m.push_cp(); grow_to(m, 100); s = m.open()   # V at address 60
m.bind(60, "c1"); m.rewind(s)
m.bind(60, "c2"); m.rewind(s)
assert m.heap[60] is None                              # clause 3 sees V unbound
m.close(s)
# Counterexample 3: cut through a scope, new CP, redo, rewind
grow_to(m, 20); m.push_cp(); grow_to(m, 30); m.push_cp()  # gens 0, 1
grow_to(m, 100); s = m.open(); m.floor = s["cp_gen"]      # cp_gen 2
m.cut_to(0)                                                # cut through s: cps == []
grow_to(m, 105); m.push_cp()                               # gen 2
x = m.alloc(); m.bind(x, "a"); m.backtrack()               # redo allowed: gen 2 >= floor 2
m.rewind(s)                                                # removes gen >= 2
assert m.cps == [] and len(m.heap) == 100
m.close(s)
```

The same model, extended with an env-slot array and EB, covers the §6.2
slot-write case: write a slot below EB inside a scope, rewind, and check the
slot's exact previous content (`ABSENT`, `UNINIT` or a `REF`). Extended with
frames, it covers §6.3's abandoned-frame rules: a nested scope that restores
E prunes its frames' entries on close, and an outer rewind skips entries for
slots at or above its `e_top`.
