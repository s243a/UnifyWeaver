<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: heap-cell runtime rewrite, design and plan

**Date:** 2026-10-04. **Ledger:** D128.
**Status:** Design. The rewrite is decided; this document sets its
representation, API and order.
**Evidence:** `docs/reports/wam_rust_heap_cell_spike.md` (baseline profile,
three spike variants, gates). Its numbers are cited below as **[S]**.

## 1. Summary

The Rust WAM target is ~17× slower than SWI-Prolog at N=40 (4.52 vs 0.26 ms)
and ~3.9× slower at N=5000 (68.9 vs 17.5 ms) on the transpiled package
resolver. The measurements say two separate things are wrong. The rewrite has
to fix both, and in a particular order.

1. **Fixed cost: interpreter mechanics.** A resolve runs ~21.7 K WAM
   instructions at every N, and each costs ~1,075 Ir **[S]**. That cost is in:
   - register access by `String` name, plus a register trail that nothing
     reads (10,790 entries per resolve, 0 consumed);
   - choice points that clone ~10 registers when ~3 are live;
   - read/write unification contexts kept on a copy-on-write `Arc<Vec<StackEntry>>`;
   - term construction through a scratch heap that ends with a scan of 200
     registers;
   - string-keyed label lookups.

   Three naive spike levers already cut 17.6% of Ir and 25% of warm wall time
   at N=40, all byte-identical **[S]**.
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
- **`Value` stays as the boundary type.** It is used at the boundary (shims,
  fact sources, dynamic DB, findall copies, exceptions) and by cold builtins
  through a copy-in/copy-out bridge.
- **Registers become numeric operands** emitted by the generator.
- **No register trail.**
- **Conditional trailing on heap addresses**, with HB maintained by a single
  `Mark` API.
- **Heap-top reset on backtrack.**
- **Choice points save only the arity's argument registers**, or the
  compiler's live set for if-then-else guards, into an argument stack.
- **Environments are a plain stack protected by the newest choice point**, as
  in the standard WAM.
- **Read and write mode use the S register.**

## 2. What the measurements rule in and out

| claim | evidence **[S]** | consequence |
| --- | --- | --- |
| The named-variable map is not the problem | heap-cell variables alone: −1.7% at N=40, +0.2% at N=5000 (an extra enum variant de-inlined hot helpers) | Variable cells only pay off as part of a full cell representation. Do not ship them alone, and do not run a dual representation in the rewrite |
| The register trail is dead weight | 10,790 entries per resolve, 0 consumed by `unwind_trail_to`; disabling it kept every gate byte-identical; −1.87 M Ir (−8.1%) | Registers are never trailed. The ~96 non-CP rollback sites save what they need (§5.3) |
| Choice points over-save | 383 clause CPs per resolve save 10.25 dirty slots where 2.78 are live; arity-only save −1.84 M Ir (−7.9%), byte-identical, lib 260/260 | The generator emits arity on `try_me_else` and a live set on ITE guards (§6) |
| The copy-on-write stack is a floor | after s3: `make_mut` 0.92 M + `StackEntry` clone 0.50 M + drop 0.35 M = 9.2% of N=40 | No `UnifyCtx`/`WriteCtx`: the S register and a mode flag. Environments in a plain stack (§7) |
| The slope is the term representation | N=5000: builtins 213.7 M + regions 138.0 M + `reset_query` 19.1 M = 95% of 388.5 M; mechanics ~17.6 M, the same as at N=40 | Builtins and regions must read and build heap cells natively in the same merge (§8, phase R4/R5) |
| The interpreter count is flat in N | the same steps, backtracks, CPs, binds and trail entries at N=40 and N=5000 | Mechanics work sets the fixed cost; builtin and region work sets the slope |

## 3. Target representation

### 3.1 `Cell`: 8 bytes, `Copy`, no `Drop`

```rust
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Cell(u64);   // low 3 bits = tag, high 61 bits = payload
```

| tag | name | payload | notes |
| ---: | --- | --- | --- |
| 0 | `REF` | heap address (u32 in practice) | a bound or aliased variable reference; registers and Y slots hold `REF`s to unbound variables, never `VAR`s |
| 1 | `VAR` | `(serial << 1) \| kind` | an **unbound** variable, stored only in the heap. `serial` is the current `var_counter` value and `kind` is `H` or `V` (or `_`, `G`, … for builtin-made variables, via a small kind table). It prints exactly the name the current runtime prints (`_V12`, `_H7`), so the text and **standard order of variables are byte-identical by construction**. The spike kept names this way and stayed identical **[S]** |
| 2 | `ATOM` | atom id (the existing interner's `u32`) | `[]` is the single atom `[]`. The `List([])`/`Atom("[]")` aliasing goes away |
| 3 | `INT` | 61-bit signed | outside that range, a `BOX` |
| 4 | `STR` | heap address of a `FUN` header | the arguments follow the header |
| 5 | `LIS` | heap address of the head cell; the tail is at +1 | the only cons form. The `"[\|]/2"` vs `"./2"` vs `Value::List` aliasing goes away |
| 6 | `FUN` | functor id | a header cell, only inside the heap. `functors: Vec<(AtomId, u32 arity)>` is interned like atoms, so there are no `"f/N"` strings and no `decomp` |
| 7 | `BOX` | index into `boxes: Vec<Boxed>` | `Float(f64)`, `BigInt(i64)` outside the 61-bit range, `Bool(bool)` (kept distinct from the atom `true` to preserve today's semantics). Its top is saved in the CP and truncated on backtrack |

`deref(c)` follows `REF`s until it reaches a non-`REF` cell. For an unbound
variable it returns `REF(addr)` of the `VAR` cell, so callers always have the
address to bind. Binding writes the target cell at `addr`. Unifying two
unbound variables binds the younger to the older, the standard heap rule that
keeps conditional trailing tight. **One exception for byte identity:** the
current runtime binds the *first* argument of `unify` to the second, and that
decides which name survives in an output with aliased variables. Keep that
direction unless a gate proves the younger-to-older rule identical. It costs
some extra trail entries, not correctness.

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
  cells, with no refcount traffic. Atomics are cheap in Ir but not in wall
  time. The spike's wall gains (−25%) exceeded its Ir gains (−17.6%) **[S]**.
- The cost is that boundary terms must be copied in. The shim already builds
  the catalog term on every call (`catalog_term`), so building cells directly
  instead of `Value`s costs about the same: O(N) once per resolve, which SWI
  pays too. A later option is to keep the base segment across resolves when
  the catalog is unchanged.

Rejected alternative: a hybrid where ground boundary terms stay as `Arc<Value>`
behind a `BOX` cell. It keeps two term models in every builtin and every
unify arm, which is exactly the dual-representation tax the spike measured
(de-inlining). It also leaves the sort-family slope where it is, because the
catalog lists *are* the boundary terms. Not worth it.

### 3.3 Heap layout and lifetime

```
heap: Vec<Cell>
  [0 .. base)      base segment: call_pred copy-in (catalog, requests). Rebuilt per call.
  [base .. H)      query heap: grows by bump, reset to CP.h on backtrack
```

`reset_query` sets `H = base` (O(1)). Nothing outside the heap may keep a heap
address across a backtrack to an older CP (§9.2).

There is **no garbage collector** (as today, in effect: `Arc` frees eagerly, the
heap does not). Deterministic forward recursion that never backtracks keeps its
garbage until the query ends. For the resolver this is bounded by the work
done. For other programs it is a real risk (§13).

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
  - `TryMeElse{alt: u32, arity: u8}`;
  - `TryMeElseIte{alt: u32, live: LiveSet}`;
  - `Jump{pc}`.

  The `labels: HashMap<String, usize>` map stays only for the public
  `WamState::new`/`call_pred` entry lookup. It cost 0.79 M per resolve on the
  hot path **[S]**.
- **Builtins get ids.** `BuiltinCall{id: BuiltinId, arity}` dispatches through
  a table. The ~187 builtin names are resolved by the generator, which already
  knows the set (`is_builtin_pred/2` plus the runtime's arms). An unknown name
  becomes a `CallUnknown{name}` that keeps today's `warn_unresolved_goal`
  behaviour.
- **Constants are pre-encoded.** `GetConstant{c: Cell, a: Reg}`. Atom ids
  must be stable: the generator emits atom *text*, and a load-time table
  interns it once into a `Vec<Cell>` indexed by constant number. Instructions
  hold that number, or the cell after load.
- **Switch tables:** `SwitchOnConstant` becomes a sorted `Vec<(Cell, u32)>`
  or an `FxHashMap<Cell, u32>`. `SwitchOnStructure` is keyed by `FunctorId`.
- **Fused instructions** (`Cons`, `NotMember`, `ListLengthLt`,
  `RecurseCategoryAncestorPc`, `ReturnAdd1`, …) keep their meaning with
  numeric operands.

Spike evidence for this bucket: register decode, the register file, the
register trail and labels are ~18.5% of N=40, plus `step` self at ~76 Ir per
instruction **[S]**.

## 5. Trail, marks and backtracking

### 5.1 Trail entries

```rust
pub struct TrailEntry { addr: u32, old: Cell }   // 12–16 bytes; `old` is the VAR cell (or the old value for rare destructive updates)
```

A variable binding pushes an entry **only if `addr < HB`**, where HB is the
heap top at the newest rollback point. Undo writes `old` back. Storing the old
`VAR` cell keeps the variable's serial and kind, and so its name, without a
side table. An address-only trail (8 bytes smaller) would need the serial
re-derived; decide by measurement in R2 (§16). The spike skipped 36% of binds
with a far more conservative mark **[S]**. With HB tied to heap addresses,
more will skip.

**Registers are never trailed.** Backtracking restores the saved argument
registers from the CP. Y slots are protected by the environment stack rule
(§7).

### 5.2 Choice points

```rust
pub struct ChoicePoint {
    alt: u32,           // next clause PC
    h: u32, tr: u32, boxes: u32,  // heap/trail/box tops -> HB = h
    e: u32, cp: u32,    // environment and continuation
    b0: u32,            // cut barrier
    args: u32, n: u8,   // saved argument registers in `arg_stack[args..args+n]`
    kind: CpKind,       // Clause | IteGuard | Aggregate{..} | Builtin(BuiltinRedo)
}
```

- Push: `arg_stack.extend_from_slice(&regs[..n])`. Restore: `regs[..n]
  .copy_from_slice(..)`. Nothing to drop. Temporaries beyond `n` are dead by
  WAM convention. The spike confirmed this for clause CPs **[S]**.
- **ITE guards and aggregate frames** need the registers that are live at the
  guard. The shared compiler knows them. Emit a `LiveSet` (a bitmask over
  A/X) on `TryMeElseIte`. Until the compiler provides it, fall back to "every
  register written since clause entry". The runtime can track that cheaply,
  as `AxDirty` does today.
- `backtrack`: undo the trail to `tr`, `H = h`, truncate the boxes, restore
  the args, `E = e`, `CP = cp`, `B0 = b0`, `pc = alt`. A `Builtin` kind
  resumes the builtin's redo state (cells inside it must point below `h`;
  §9.2).
- Cut: truncate the CP stack to `b0`. HB becomes the new top CP's `h`.

### 5.3 Non-CP rollback points: one `Mark` API

Today ~96 sites take `trail.len()` (and sometimes `heap.len()` /
`var_counter`) and later call `unwind_trail_to`:

- 61 in `wam_rust_target.pl`: `call_goal_once`, `\+`, `forall`, findall
  helpers, builtins;
- 35 in templates: the dynamic DB, the builtin families, and the
  lowered/region snapshots `lo_clause_snapshot`, `lo_restore_clause` and
  region P2 snapshots.

All of them move to:

```rust
pub struct Mark { tr: u32, h: u32, boxes: u32, hb: u32, var_counter: u32 }
fn mark(&mut self) -> Mark;          // HB = H (so later binds of older cells are trailed)
fn rollback(&mut self, m: &Mark);    // undo trail to m.tr, H = m.h, HB = m.hb, var_counter = m.var_counter
fn release(&mut self, m: Mark);      // commit: HB = max(top CP.h, m.hb)
```

Callers that need registers after a rollback save them explicitly. A lowered
clause snapshot saves A1..A<arity>, the same set the CP would. **Making
`trail` and `heap` private** turns every forgotten mark into a compile error.
This is the soundness lever for conditional trailing: a rollback point that
does not raise HB is a silent wrong answer.

## 6. Environments (Y registers)

```
estack: Vec<Cell>, frames: [prev_e, cp, n_y, y0 .. y(n-1)], E = current frame base
```

- `Allocate`: the new frame goes at `max(E_top, B.e_top)`, so it never
  overwrites a frame a live CP may resume into (the standard WAM rule).
  `Deallocate` sets `E = prev_e` and does not shrink below `B.e_top`.
- Y reads and writes index `estack[E + 3 + y]`. There is no `Arc`, no
  copy-on-write, no `YRegs` `resize`, and no frame search for the "topmost
  Env". Today `put_reg` alone is 1.33 M per resolve **[S]**.
- `put_variable Yn, Ai` allocates the variable **on the heap** and puts a
  `REF` in both `Yn` and `Ai`. Permanent variables are therefore never unbound
  *in* the environment. That removes the "unsafe variable" case and keeps
  trailing heap-only.
- `GetLevel`/`CutTo` keep today's design: the level is stored on the ITE CP,
  not in a Y slot, so the §8 hazard does not come back.
- `UnifyCtx`/`WriteCtx` disappear. `get_structure` sets `S` (read mode) or
  pushes a `FUN` header at `H` (write mode), and `unify_*` reads `heap[S++]`
  or pushes. `put_structure` writes `STR(H)` into the register immediately. The
  200-register scan in `set_heap_or_list` and its `"f/N"` parse go away.

## 7. Builtins: migration API

There are ~187 builtin arms in `state.rs` across the core, arith, io, type,
term, ext and meta families. There are 10 builtin family templates
(`*_builtin.rs.mustache`, ~1.4 K lines) and the dynamic-DB methods
(`dynamic_db_methods.rs.mustache`, 1.0 K lines). Inside the builtins there are
389 reads of `get_reg_raw("A<n>")`, 213 `deref_heap` calls and ~320
`Value::strv`/`Value::list` constructions. A big-bang port of all of them by
hand is the main schedule risk, so the API is built for a two-speed migration.

### 7.1 Native API (hot builtins, regions, lowered code)

```rust
pub enum View<'a> {                       // a dereferenced cell
    Var(u32),                             // heap address of the unbound VAR cell
    Atom(AtomId), Int(i64), Float(f64), Bool(bool),
    Str(FunctorId, &'a [Cell]),           // args in place, no copy
    List(u32),                            // head addr; tail at +1
}
impl WamState {
    fn reg(&self, i: u8) -> Cell;  fn set_reg(&mut self, i: u8, c: Cell);
    fn deref(&self, c: Cell) -> Cell;  fn view(&self, c: Cell) -> View<'_>;
    fn list_iter(&self, c: Cell) -> ListIter<'_>;   // yields deref'd elements; .tail() = Nil | Var(addr) | Other
    fn unify(&mut self, a: Cell, b: Cell) -> bool;
    fn bind(&mut self, var_addr: u32, c: Cell);     // conditional trail inside
    fn compare(&self, a: Cell, b: Cell) -> Ordering; // standard order; vars by name text from (serial, kind)
    fn mk_int(&mut self, n: i64) -> Cell; fn mk_float(&mut self, f: f64) -> Cell;
    fn mk_str(&mut self, f: FunctorId, args: &[Cell]) -> Cell;
    fn mk_list(&mut self, items: &[Cell], tail: Cell) -> Cell;
    fn functor_id(&mut self, name: AtomId, arity: u32) -> FunctorId;
    fn mark(&mut self) -> Mark; fn rollback(&mut self, m: &Mark); fn release(&mut self, m: Mark);
}
type BuiltinFn = fn(&mut WamState, args: u8 /* arity; A1..An are the args */) -> BuiltinResult;
enum BuiltinResult { Fail, Succeed, SucceedWithRedo(BuiltinRedo), Throw(Value) }
```

**Sorting**:

- copy the deref'd element cells into a `Vec<Cell>` and sort it with
  `compare`;
- `compare` on atoms compares interned text (ids are first-sight order, not
  alphabetical; this matches today's de-intern-to-name rule);
- `compare` on functors compares arity, then name text, then arguments in
  place;
- `msort`/`sort`/`predsort`/`keysort`/`sort/4` all share this;
- dedup compares cells with `compare == Equal`;
- the result list is built with one `mk_list`.

This removes `deref_list_arg`'s materialization, per-element `deref_heap`,
`Arc` traffic and `decomp`. Those are 196 M of 388 M Ir at N=5000 **[S]**.

### 7.2 Bridge API (cold builtins, first-pass compile of everything)

```rust
fn to_value(&self, c: Cell) -> Value;     // copy out; an unbound var becomes Value::Var(addr)
fn from_value(&mut self, v: &Value) -> Cell; // copy in; Value::Var(addr) -> REF(addr); Unbound(name) -> fresh var per name
fn reg_value(&self, i: u8) -> Value;      // = to_value(reg(i)), the replacement for get_reg_raw("A<i>")
fn unify_value(&mut self, c: Cell, v: &Value) -> bool;
```

`Value` keeps its public shape (`Atom/Integer/Float/Str/List/Unbound/Bool`)
and gains `Var(u32)`, a handle to a live heap variable, which is exactly the
spike's dual form. That way a bridged builtin body keeps working on `Value`
with mechanical edits: `self.get_reg_raw("A1")` → `self.reg_value(1)`,
`self.unify(&a, &b)` on `Value`s → `unify_value`. The bridge copies, so it is
only allowed for builtins whose arguments stay small. **Gate:** at N=5000 the
profile must show no bridged builtin above 1% (§11).

### 7.3 Which builtins go native (phase R4)

Taken from the profile and from term size, not from popularity:

- the sort family (`sort/2`, `msort/2`, `sort/4`, `keysort/2`, `predsort/3`,
  `list_to_set/2`), `compare/3`, `==`/`\==`/`@<`-family;
- `length/2`, `nth0/nth1`, `member/2`, `memberchk/2`, `append/3`,
  `reverse/2`, `last/2`, `sum_list`-family, `exclude/include/maplist` helpers
  if runtime-implemented;
- `functor/3`, `arg/3`, `=../2`, `copy_term/2`, `is/2` and arithmetic
  comparison (evaluate on cells), `=/2`, `\=/2`, type tests;
- findall/bagof/setof/aggregate_all finalization (§9.3).

Everything else (OS, process, time, random, stream, filesystem, read_term,
format, atom/string conversion, assert/retract) starts bridged.

## 8. Lowered emitter and Stage-2 regions

- **`wam_rust_lowered_emitter.pl` (1.1 K lines)** emits Rust that calls ~12
  `WamState` methods (`get_reg`, `put_reg`, `head_constant`,
  `match_reg_atom[_str]`, `unify`, `step`, `run`, `lo_clause_snapshot`,
  `lo_restore_clause`, `unwind_trail_to`). Port each one to the numeric/cell
  API:
  - `match_reg_atom` compares an `ATOM` id against a constant id resolved at
    load;
  - `head_constant` becomes `unify(reg, const_cell)`;
  - snapshots become `Mark` plus an A1..A<arity> save.
- **Stage-2 regions** are hand-written in `state.rs.mustache`: about 1.6 K
  lines of region code (regions 1, 2, 3a, 3b, 4 and 5) plus about 1.0 K lines
  of stress tests. They are 1.46 M of N=40 and **138 M (35.5%) of N=5000**
  **[S]**. They walk catalog lists with `deref_shallow`, build `strv` terms and
  compare with `terms_identical`. Port them to `View`/`list_iter`/`mk_*`/
  `compare` in phase R5. Their soundness arguments (the Kimi G-1…G-5 gates in
  `docs/reports/wam_rust_stage2_kimi_soundness_review.md`) depend on snapshot
  semantics. Re-check each against `Mark`: "the P2 minimal snapshot" becomes
  `Mark` and is no weaker.
- **`rust_target.pl` hybrid wrappers** (~4470–4580) read `vm.bindings` by
  temporary variable names. Rewrite them on `reg`/`to_value`.

## 9. Boundary

### 9.1 Shims and `call_pred`

`examples/pkg_resolver/rust/shim/main.rs` and `rust_store/shim/main.rs`
(~760 lines each) build `Value`s (`catalog_term`) and read results (`sel_json`
over `Value`). Keep that code. Replace `call_pred`'s body with:

```rust
vm.call(pred, &[cat, reqs]) -> Option<Value>
// reset_query; from_value each arg into the base segment; set A1..An and a fresh output var;
// run; to_value(output) on success
```

Copy-out produces exactly the `Value` the shim formats today: the same atom
text, the same float `Display`, `[]` as an atom (an empty `Value::List` and
the atom `[]` print the same), and cons cells as `Value::List`. A partial list
with an unbound tail is still built as a `"[|]/2"` `Str`, to match today's
`deref_heap`. The `json.rs` files do not change.

### 9.2 Values that outlive a backtrack

Heap-top reset is sound only if no Rust-side value keeps a heap address above
the CP's `h` across a backtrack to it. Each of the following keeps a `Value`
copy (`to_value`), never a raw `Cell`, or stores cells in a structure the CP
truncates:

| holder | today | rewrite |
| --- | --- | --- |
| `aggregate_acc` (findall/bagof/setof/aggregate_all) | materialized `Value` (`deref_heap`) | `to_value`, the same cost. Later: a findall segment of cells |
| `thrown_ball` | `Value` | `Value` |
| `BuiltinState` redo data | `Vec<Value>` | `Vec<Cell>` allowed only when the cells are below the CP's `h` (the CP was pushed *after* they were built). Otherwise `Value`. A debug assertion checks `addr < cp.h` |
| `dynamic_db` (assert/retract) | `Value` with named vars | `Value`. Retrieval renames into fresh heap vars (`from_value`) |
| `par_aggregate` fork results | `Value` | `Value`. Inputs are `Value`; the fork clones the machine (a heap `memcpy`) |
| global variables (b_/nb_setval, if implemented) | `Value` | `Value` |
| lowered/region locals across a nested `run` | `Value` | `Cell`s only within one `Mark` scope |

### 9.3 Fact sources and kernels

- `seek_fact_source.rs`, `csr_fact_source.rs` and the LMDB sources
  (`lmdb_fact_source_*`) hand rows to `fact_table_attempt` and
  `finish_foreign_results` as `Value`s (atoms and ints). These become
  `from_value` on delivery, or direct `mk_*` for atoms and ints (cheap, no
  heap).
- `boundary_cache.rs` (7.8 K lines) works on `u32` node ids and talks to the
  machine only through result delivery, so it needs only a thin change.
- Foreign kernels (`execute_foreign_predicate`, the native category-ancestor
  family) read registers as atoms and ints. Port them to `view`.

## 10. What stays byte-identical

- **Program output** (every gate JSONL, scale `--bench` stdout) must be
  `cmp`-identical to the pre-rewrite build. The things that could break it,
  and how the design holds each:
  - variable names and their standard order: kept by the `VAR` serial/kind
    (§3.1);
  - which aliased variable survives: the bind direction is preserved;
  - float text: copy-out goes through the same `Value` `Display`;
  - `[]` and cons spellings: copy-out normalizes them to today's forms;
  - atom order in sorts: text compare, as today;
  - error terms from `raise_iso_error`: built on `Value` and copied in.
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
2. **Reference-model property tests** (new, in the generated crate, `cargo
   test` only):
   - keep today's `Value` algorithms for unify, `term_compare`, sort/dedup and
     `copy_term` as a test-only `term_ref` module;
   - generate random terms (atoms with tricky text, big and small ints,
     floats, nested structures, partial lists, shared and aliased variables)
     and check that `to_value(op_cell(from_value(t)))` equals
     `op_ref(t)`, including variable names;
   - run random bind, mark, rollback and backtrack sequences against a naive
     always-trail machine, checking that conditional trailing gives the same
     state after every step (the D125/D127 model-test pattern).
3. **A debug "paranoid" feature:**
   - trail every bind (conditional trailing off);
   - assert every retained `Cell` is below the relevant `h`;
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
Phase R0 is optional and lands on main **before** the branch.

| phase | content | why here (numbers **[S]**) | exit gate |
| --- | --- | --- | --- |
| **R0** (optional, main) | Two pre-rewrite wins in today's runtime: arity-only clause CP save (−7.9%), and dropping the register trail with explicit register saves at the ~96 non-CP rollback sites (−8.1%, upper bound) | −17.6% Ir and −25% wall at N=40, byte-identical in the spike. They also prove the two semantic assumptions the rewrite relies on (registers are not trailed; clause CPs need only args) on main, in small diffs | the full D127 gate set |
| **R1** | Generator: numeric `Reg` operands, X/Y split variants, PC-resolved control, `BuiltinId`, constant table, `TryMeElse{arity}`, `TryMeElseIte{live}`, `FunctorId` switch keys | bucket (b) is ~18.5% and `step` self ~8% of N=40. Everything after it is written against the new encoding, so it comes first | the crate compiles against R2's skeleton; generator plunit |
| **R2** | Core machine: `Cell`, heap, `VAR` serials, trail, `Mark`, CP plus arg stack, env stack with B protection, S register, every step arm, `backtrack`, cut, `run`. All builtins through the **bridge** | buckets (a), (c) and (d), about 32% of N=40, plus construction ~8% | builds and runs; property tests pass |
| **R3** | Boundary: `vm.call` copy-in/out, both shims, fact sources, kernels, `par_aggregate`, findall/`copy_term`/assert/exceptions/`read_term`, `rust_target.pl` wrappers | nothing runs end-to-end without it, so R2+R3 is the first point where the differential can run | **differential, corpus and byte identity green** with every builtin bridged |
| **R4** | Native hot builtins (§7.3) | 23 sort-family calls are 196 M of 388 M at N=5000; R2/R3 with a bridged sort is about today's cost, because today already materializes | the N=5000 profile shows no bridged builtin above 1% |
| **R5** | Stage-2 regions and the lowered emitter on the cell API; re-check the Kimi gates | 138 M (35.5%) at N=5000 | region stress tests and gates |
| **R6** | Remove the old internals: `bindings`, `Args` spine and `deref_memo` flags, `"f/N"` functor syms and `decomp`, `WriteCtx`/`UnifyCtx`, `YRegs`, `AxDirty`, the register-name tables. Cold builtin families stay bridged | code size, compile time; removes the inlining tax of a dual representation, which the spike saw as +7.6 M at N=5000 | full gates |
| **R7** | Store lane rebuild and gates, then perf pass: conditional-trail rule, ITE live sets from the compiler, base-segment reuse across resolves | store gates are part of "green" | the full D127 gate set, plus Ir/wall report |

R2 and R3 are one milestone in practice. R4 and R5 must be in the same merge
as R2, because a merge without them would land the mechanics win (N=40) with
the slope unchanged and the bridge copies on top.

## 13. Risks

1. **Size and drift.** The rewrite touches ~9.9 K template lines, ~11 K
   generator lines and ~6 K lines across the other templates, shims and
   `rust_target.pl` (§15). Main keeps moving: other agents land D-items in
   the same files. Mitigation: freeze perf work on `rust_wam` for the
   duration, rebase weekly, and keep R0 small so it lands first.
2. **Conditional-trailing soundness.** A rollback point that does not raise HB
   gives silent wrong answers. Mitigation:
   - private `trail`/`heap` fields and the single `Mark` API;
   - the paranoid feature (always trail), A/B'd in CI on the differential;
   - model tests.
3. **Heap-top reset and dangling cells.** Any `Cell` kept above a CP's `h`
   across a backtrack dangles. Mitigation: the §9.2 table as a code-review
   checklist, debug assertions, and `Value` copies at every escape.
4. **Byte identity of variables.** Names, standard order and the bind
   direction must be preserved. The `VAR` serial design does it by
   construction. The bind direction stays first-to-second unless proven
   identical.
5. **Bridge cost hiding in the slope.** A bridged builtin on a catalog-sized
   term is O(N) per call. Mitigation: the R4 gate (no bridged builtin above 1%
   at N=5000).
6. **No GC.** Long deterministic runs keep heap garbage until the query ends,
   where `Arc` used to free it eagerly. That is fine for the resolver. Other
   programs with long forward recursion over large data could grow memory.
   Mitigation: a heap cap with a clear error, a measured memory column in the
   scale bench, and a later "heap compaction at deterministic points" item.
   Do not block the rewrite on it.
7. **Stage-2 region proofs** were written against the old snapshot semantics.
   Each needs a re-check (R5).
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

## 14. Expected gains (honest ranges)

Against base D127 (Ir per warm resolve 23.30 M / 388.51 M; warm wall 4.3 /
~64–69 ms):

| N | what moves | estimate | resulting Ir | speed-up | wall (est.) | vs SWI |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 40 | mechanics 19.2 M after s3 → ~2–4 M (~100–200 Ir per instruction); builtins 4.6 → 2–3 M; regions 1.5 → 0.6–1.0 M | 3–4.5× fewer Ir | **5–8 M** | **3–4.5×** | 1.0–1.8 ms | still ~4–7× SWI (0.26 ms) |
| 5000 | sort family 196 M → 35–65 M (no materialization, in-place compare); regions 138 M → 40–70 M; `reset_query` 19 M → ~0; mechanics ~17 M → ~4 M | 2.5–4× fewer Ir | **100–160 M** | **2.4–3.9×** | 17–28 ms | ~1.0–1.6× SWI (17.5 ms) |

The N=40 range depends on how tight `step` gets. The WAT target's experience
is that dispatch becomes the floor once handlers are small. The N=5000 range
depends on R4/R5 being native. With them bridged, expect N=5000 to land near
today (~0.9–1.1×) while N=40 still gains. The 25% wall gain on 17.6% fewer Ir
**[S]** suggests wall improves somewhat more than Ir, because refcount atomics
and cache misses go away. These ranges do not count that.

## 15. Files that must change

Sizes are current line counts. "Touch" is a rough share of each file that
changes.

| file | lines | touch | what |
| --- | ---: | --- | --- |
| `templates/targets/rust_wam/state.rs.mustache` | 9,917 | ~70% | machine core, deref/unify/compare, builtins (core/arith/io/type/term/ext/meta), regions, snapshots, tests |
| `templates/targets/rust_wam/value.rs.mustache` | 816 | ~40% | `Value` stays the boundary type and gains `Var(u32)`. New `cell.rs` (~600 new): `Cell`, tags, `View`, functor table, boxes |
| `templates/targets/rust_wam/instructions.rs.mustache` | 149 | 100% | numeric operand enum |
| `src/unifyweaver/targets/wam_rust_target.pl` | 11,064 | ~45% | step arms, `backtrack`, `run`, `execute_builtin` dispatch table, `unwind_trail_bindings_only`, instruction literal emission (regs, PCs, arity, live sets, builtin ids, constants), `call_goal_once`, findall/aggregate, foreign predicates, `resume_builtin` |
| `src/unifyweaver/targets/wam_rust_lowered_emitter.pl` | 1,115 | ~50% | emitted API calls |
| `src/unifyweaver/targets/rust_target.pl` | 14,234 | <2% (~150 lines near 4470–4580) | WAM-hybrid wrappers that read `vm.bindings` |
| `templates/targets/rust_wam/dynamic_db_methods.rs.mustache` | 1,028 | ~40% | assert/retract/clause on `Value`, renaming in |
| `templates/targets/rust_wam/seek_fact_source.rs.mustache` | 1,047 | ~10% | row delivery |
| `templates/targets/rust_wam/csr_fact_source.rs.mustache`, `lmdb_fact_source_heed.rs.mustache`, `lmdb_fact_source_lmdb_zero.rs.mustache`, `materialisation_setup.rs.mustache`, `lazy_category_parents.rs.mustache` | 179 + 176 + 452 + 149 + 23 | ~10% | row delivery and registration |
| `templates/targets/rust_wam/boundary_cache.rs.mustache` | 7,816 | <3% | result delivery only (`u32` kernels unchanged) |
| `templates/targets/rust_wam/par_aggregate.rs.mustache` and `src/unifyweaver/targets/rust_runtime/par_aggregate.rs` | 263 + 271 | ~40% | fork, inputs and results as `Value` |
| builtin family templates: `filesystem_permission`, `os_error`, `os_utility`, `process`, `process_context`, `process_metrics`, `process_resource`, `random`, `stream`, `time` (`*_builtin.rs.mustache`) | ~1,365 total | ~25% | bridged: register reads, unify outputs, `Mark` instead of `unwind_trail_to` (22 sites; 5 more in `dynamic_db_methods`) |
| `templates/targets/rust_wam/main.rs.mustache`, `lib.rs.mustache`, `Cargo.toml.mustache` | 278 + 15 + 16 | ~30% | the bench driver uses `vm.call`; feature list (remove `deref_memo`, `intern_sym_thread`, `trail_enum`; add `paranoid`) |
| `examples/pkg_resolver/rust/shim/main.rs` | 759 | ~10% | `call_pred` → `vm.call`. `json.rs` unchanged |
| `examples/pkg_resolver/rust_store/shim/main.rs` | 753 | ~10% | same |
| `examples/pkg_resolver/rust/build.pl`, `rust_store/build.pl` | small | maybe | only if build options change |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`, 56 files) | — | some | expected generated-text fragments |
| generated-crate tests (`tests/comparator_equiv.rs`, `tests/intern_stress.rs`, in-template `mod d1xx_*_tests`) | — | ~50% | rewritten on the new API (§11.4) |
| docs: `docs/WAM_RUST_STATUS.md`, this design, a per-phase report under `docs/reports/` | — | — | — |

**Do not change:** `examples/pkg_resolver/resolver.pl`,
`resolver_store.pl`, the shared `wam_target.pl` compiler (beyond optionally
emitting ITE live sets, which is additive), the WAT, Go and other targets.

## 16. Open questions to settle in R1

- Does the shared compiler already compute live registers at an ITE guard
  that can be exported? If not, use runtime dirty tracking for ITE CPs at
  first.
- Should the `VAR` kind table be the fixed set {`V`, `H`, `L`, `F`, `C`,
  `RP`, `EC`, `SE`, `MB`, `M`, …} that today's builtins use for fresh names,
  or a general `(prefix_id, n)`? The prefix set is closed, so a 4-bit kind
  with a table is enough.
- Should the trail store the old cell always (8 bytes per entry), or only the
  address with the `VAR` serial re-derived? Decide by measurement in R2.
