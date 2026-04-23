<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025-2026 John William Creighton (@s243a)
-->
# WAM-WAT Target Architecture

This document describes the architecture of the WAM-WAT target
(`src/unifyweaver/targets/wam_wat_target.pl`), which compiles
Prolog predicates — via a WAM text intermediate form — to
WebAssembly (`.wat`/`.wasm`).

The target exists alongside the shell/systems-language/query
backends documented in `docs/targets/overview.md`. Where other
backends emit source code for a target language, WAM-WAT emits a
self-contained WebAssembly module that interprets a WAM bytecode
stream at runtime.

This document covers:

- [Compilation pipeline](#compilation-pipeline)
- [Peephole optimization passes](#peephole-optimization-passes)
- [Instruction set](#instruction-set)
- [Runtime data structures](#runtime-data-structures)
- [Key optimizations](#key-optimizations)
- [Performance trajectory](#performance-trajectory)

For the historical performance baseline before the recent
optimization work, see
[`WAM_TERM_BUILTINS_PHASE_6_PERF.md`](WAM_TERM_BUILTINS_PHASE_6_PERF.md).

---

## Compilation pipeline

```
Prolog predicates
      │
      ▼
  wam_target.pl   ─────▶  WAM text ("try_me_else L2\nallocate\n...")
      │
      ▼
  wam_wat_target.pl
   ├─ parse WAM text → instruction terms + label markers
   ├─ peephole optimization pipeline (8 passes, ordered)
   ├─ extract instruction list + resolve labels → PC table
   ├─ encode each instruction to 20-byte hex record
   └─ assemble .wat module (templates + data segment)
      │
      ▼
  bench_suite.wat
      │
      ▼ (wat2wasm)
  bench_suite.wasm ─────▶ runtime interprets via $step dispatch
```

Each predicate is compiled independently. The full pipeline
orchestrator is `pass1_parse_predicates` in `wam_wat_target.pl`.

### WAM text → instruction terms

`wam_lines_to_instrs_with_labels` parses the textual WAM output
into a list of Prolog terms, preserving label markers as
`label(Name)` pseudo-instructions. Label markers carry **strings**
(from the text parse); instruction operands that reference labels
carry **atoms** (from `atom_string/2` normalization). The
peephole passes normalize for comparison when needed.

### Label resolution

After the peephole pipeline, `extract_instrs_and_labels` strips
label markers and records each label's PC (= index into the real
instruction list). `resolve_label/3` uses that table during
encoding to turn label references into absolute PCs.

### Data segment encoding

Each instruction becomes a 20-byte record in a WebAssembly data
segment placed at `code_base` (default offset 131072, i.e.
start of page 2):

```
  [tag:i32]  [op1:i64]  [op2:i64]
   4 bytes    8 bytes    8 bytes
```

`encode_instr_hex/4` produces the hex-escape string; `wat2wasm`
turns that into raw bytes in the compiled module.

---

## Peephole optimization passes

The pipeline runs 8 passes in order (see `pass1_parse_predicates`):

```
raw WAM instr list
       │
       ▼
1. peephole_neck_cut          ← try_me_else+guard+cut → neck_cut_test
       │
       ▼
2. peephole_nested_arith      ← 2-level arithmetic → chained fused ops
       │
       ▼
3. peephole_fused_arith       ← X is A+B → fused_is_add
       │
       ▼
4. peephole_direct_builtins   ← builtin_call(arg/3) → arg_direct
       │
       ▼
5. peephole_arg_to_a1         ← arg + put_value(_, A1) → arg_to_a1
       │
       ▼
6. peephole_arg_call_k        ← arg_to_a1 + setup + call → arg_call_K
       │
       ▼
7. peephole_tail_call_k       ← put_value×5 + dealloc + exec → tail_call_5
       │                          dealloc + proceed → deallocate_proceed
       │                          dealloc + builtin + proceed → combined
       │
       ▼
8. peephole_type_dispatch     ← 2/3-clause type-guarded → type_dispatch_a1
       │
       ▼
optimized instr list → encoding
```

### Pass order is load-bearing

Several passes depend on earlier ones having run:

- **arg_to_a1** depends on **direct_builtins** having already
  rewritten `builtin_call('arg/3', 3)` to `arg_reg_direct` /
  `arg_lit_direct`.
- **arg_call_k** depends on **arg_to_a1** producing `arg_to_a1_reg`
  / `arg_to_a1_lit`.
- **tail_call_k** depends on **arg_call_k** having consumed
  everything ending in `call(Pred, K)` that would also match the
  tail-call prefix.
- **type_dispatch** depends on **tail_call_k** having absorbed
  the `deallocate + builtin_call('!/0', 0) + proceed` tail of
  each clause — the peephole matches the post-fusion form.

### Each pass is a list rewrite

Passes have the shape `peephole_X(+InInstrs, -OutInstrs)`. They
walk the list linearly and match specific multi-instruction
windows. The conventional form:

```prolog
peephole_X([Pattern | Rest], [Rewritten | Out]) :- !, peephole_X(Rest, Out).
peephole_X([H | T],          [H | Out])         :- peephole_X(T, Out).
peephole_X([], []).
```

With a cut on the match so the fall-through doesn't re-match the
rewritten prefix.

### Mutual exclusion guards

`peephole_neck_cut` skips type-test guards (`integer/1`, `atom/1`,
etc.) — those are left for `peephole_type_dispatch` downstream,
which is strictly faster for such cases (it dispatches on tag
without even running allocate/guard). Both passes fire on the
same predicates only when their patterns are distinct.

---

## Instruction set

Instruction tags are declared as `instr_tag(Name, N)` facts in
`wam_wat_target.pl`. The tag ranges document the historical
layering:

### Base WAM (tags 0–30)

Standard Warren Abstract Machine instructions:

| Tag | Name | Purpose |
|-----|------|---------|
| 0–4 | get_constant, get_variable, get_value, get_structure, get_list | Head unification |
| 5–7 | unify_variable, unify_value, unify_constant | Structure-mode unification |
| 8–12 | put_constant, put_variable, put_value, put_structure, put_list | Argument register setup |
| 13–15 | set_variable, set_value, set_constant | Structure-mode writes |
| 16–17 | allocate, deallocate | Env frame management |
| 18–20 | call, execute, proceed | Control flow |
| 21 | builtin_call | Builtin dispatch (Op, Arity) |
| 22–24 | try_me_else, retry_me_else, trust_me | Clause selection / CP |
| 25 | neck_cut_test | Combined guard + cut (peephole) |
| 26 | cut_ite | Soft cut for if-then-else |
| 27 | jump | Unconditional jump |
| 28–29 | begin_aggregate, end_aggregate | Aggregation framing |
| 30 | nop | — |

### First-arg indexing (tags 31–35)

| Tag | Name | Purpose |
|-----|------|---------|
| 31 | switch_on_const | Value-based dispatch header (A1 or A2) |
| 32 | switch_entry | Data record scanned by switch_on_const |
| 33 | switch_on_struct | Compound-functor dispatch header |
| 34 | switch_struct_entry | Data record scanned by switch_on_struct |
| 35 | switch_on_term_hdr | Mixed constant + structure indexing |

### Fused arithmetic (tags 36–40)

| Tag | Name | Window fused |
|-----|------|--------------|
| 36 | fused_is_add | `put_value(Dest,A1)` + `put_structure(+/2,A2)` + set_value×2 + builtin_call(is/2) |
| 37 | fused_is_sub | Same for `-/2` |
| 38 | fused_is_mul | Same for `*/2` |
| 39 | fused_is_add_const | `+/2` with a literal operand |
| 40 | fused_is_mul_const | `*/2` with a literal operand |

### Direct-dispatch builtins (tags 41–45)

Skip the `$execute_builtin` br_table by calling the specific
`$builtin_*` directly:

| Tag | Name | Calls |
|-----|------|-------|
| 41 | arg_direct | `$builtin_arg` |
| 42 | functor_direct | `$builtin_functor` |
| 43 | copy_term_direct | `$builtin_copy_term` |
| 44 | univ_direct | `$builtin_univ` |
| 45 | is_list_direct | `$builtin_is_list` |

### arg/3 call family (tags 46–61)

These absorb the arg/3 + call setup pattern that dominates
term-walking predicates like `sum_ints_args/5` and
`term_depth_args/5`. See [arg_call family](#arg_call-family) below.

| Tag | Name | Pattern |
|-----|------|---------|
| 46 | arg_reg_direct | Inline arg/3 fast path (reg N) |
| 47 | arg_lit_direct | Inline arg/3 fast path (literal N) |
| 48 | arg_to_a1_reg | arg/3 + put_value(_, A1) |
| 49 | arg_to_a1_lit | Same for literal N |
| 50 | arg_call_reg_3 | arg_to_a1 + put_value(A2) + put_variable(A3) + call/3 |
| 51 | arg_call_lit_3 | Same for literal N |
| 52 | arg_call_reg_3_dead | As 50, ArgDest elided by liveness |
| 53 | arg_call_lit_3_dead | — |
| 54–57 | arg_call_{reg,lit}_1 / _dead | K=1 variants |
| 58–61 | arg_call_{reg,lit}_2 / _dead | K=2 variants (with IsVar flag for put_value vs put_variable A2) |

### Tail-call + clause-end family (tags 62–71)

| Tag | Name | Window fused |
|-----|------|--------------|
| 62 | tail_call_5 | `put_value×5 + deallocate + execute(Pred)` |
| 63 | deallocate_proceed | `deallocate + proceed` |
| 64 | tail_call_5_c1_lit | As 62 with literal first arg |
| 65 | deallocate_builtin_proceed | `deallocate + builtin_call(Op,N) + proceed` |
| 66–70 | deallocate_{arg,functor,copy_term,univ,is_list}_direct_proceed | Direct-dispatch variants |
| 71 | builtin_proceed | `builtin_call(Op,N) + proceed` (no deallocate) |

### First-arg tag dispatch (tag 72)

| Tag | Name | Purpose |
|-----|------|---------|
| 72 | type_dispatch_a1 | Tag-based first-arg indexing (atom/int/cmpd/default) |

### Encoding conventions

- **op1 (i64)**: multi-byte packed fields for register indices
  and/or small literals. Each reg index is 8 bits; up to 8 reg
  indices fit. Flags (1-bit) typically live at bit 32 or 33.
- **op2 (i64)**: typically a target PC (for control-flow
  instructions) or an arity/count field. `type_dispatch_a1` packs
  two 32-bit PCs into op2.

The detailed packing for each instruction is documented in its
encoder (search for `wam_instruction_to_wat_bytes(InstrName(...)`).

---

## Runtime data structures

### Memory layout

```
Page 0 (offset 0):
    Reserved for native WAT target (atoms, constants, etc.).

Page 1.5 (offset 65536):
    WAM state:
    - Registers A1..A32, X1..X32 (768 bytes, 12 bytes per reg)
    - Globals (pc, cp, heap_top, trail_top, stack_top, cp_count, …)

Page 1.5 +  98304:
    Choice-point stack.

Page 2 (offset 131072):
    Instruction data segment (CODE_BASE by default).

Page 3 (offset 196608):
    Heap (bump-allocated, backtrack-rewindable).
```

### Env frame layout (396 bytes)

Extended from the original 392 in PR #1531 (trail-aware neck_cut):

```
offset  size  field
──────  ────  ─────
  +0      4   prev_env_base  (caller's env_base)
  +4      4   CP              (return PC)
  +8    384   Y slots         (32 × 12 bytes each)
+392      4   trail_mark      (added for trail-aware neck_cut)
──────────
 = 396
```

`allocate` snapshots `trail_top` into `trail_mark` at frame entry.
`neck_cut_test` reads it on guard failure and calls `$unwind_trail`
to undo any trailed bindings from the pre-guard prelude.

### Choice-point frame layout

```
offset  size  field
──────  ────  ─────
  +0      4   next_pc         (alternative clause PC)
  +4      4   trail_mark      (trail_top at push time)
  +8      4   saved_cp        (caller's CP)
 +12      4   saved_heap_top
 +16      4   saved_env_base
 +20      4   retry_n         (nondet arg/3 iteration state; 0 otherwise)
 +24     96   8 saved A-regs  (12 bytes each)
──────────
 = 120
```

`push_choice_point` saves all of this. `retry_me_else` updates
`next_pc` only. `trust_me` pops the whole frame.

### Value cells (12 bytes each)

Every tagged value (register, Y slot, heap cell) is 12 bytes:

```
offset  size  field
──────  ────  ─────
  +0      4   tag
  +4      8   payload
```

Tags (defined by `val_tag` / `$val_store*` helpers):

| Tag | Meaning | Payload |
|-----|---------|---------|
| 0 | Atom | DJB2 hash (i64) |
| 1 | Integer | signed i64 |
| 2 | Float | IEEE 754 bits |
| 3 | Compound | heap address of functor header |
| 4 | List (cons) | heap address of head cell |
| 5 | Ref | heap address of the ref cell (for deref chains) |
| 6 | Unbound | 0 (self-ref) |
| 7 | Bool | 0 or 1 |

### Dispatch (`$step` + `$run_loop`)

```
$run_loop(code_base, num_instrs):
  loop:
    if halted → exit
    pc = get_pc()
    if pc < 0 or pc >= num_instrs → fail
    if $step(code_base, pc) → continue
    else if $backtrack() → continue
    else → fail

$step(code_base, pc):
  instr_addr = code_base + pc * 20
  tag = load i32 at instr_addr
  op1 = load i64 at instr_addr + 4
  op2 = load i64 at instr_addr + 12
  dispatch via br_table on tag → $do_<instr>(op1, op2)
```

The three-field inline fetch was added in PR #1533 based on
profiling data showing `$step` at ~50% of hot-loop time (the
fetch functions were previously separate calls).

---

## Key optimizations

### neck_cut_test

Collapses the common cut-deterministic 2-clause guard+cut
pattern into a single instruction that neither pushes a choice
point nor enters the `$execute_builtin` dispatch.

```wam-text
try_me_else L2
allocate
…prelude…
builtin_call(>/2, 2)       ← guard
[deallocate]                ← optional; compiler sometimes moves it earlier
builtin_call(!/0, 0)       ← cut
…clause 1 body…
L2: trust_me
…clause 2 body…
```

→

```
allocate
…prelude…
neck_cut_test(>/2, 2, L2)
…clause 1 body…
L2:
…clause 2 body…
```

The trail-aware variant (PR #1531) reads the `trail_mark` from
the env frame on guard failure and calls `$unwind_trail` so any
bindings made by the pre-guard prelude (`get_value`, etc.) are
undone before clause 2 runs.

Gated by:
- **`type_test_guard/1`**: skip if guard is a type test — those
  are better handled by `type_dispatch_a1`.

### type_dispatch_a1

Tag-based first-argument dispatch for predicates that select
clauses by A1's runtime type.

```prolog
term_depth(T, 0) :- integer(T), !.
term_depth(T, 0) :- atom(T), !.
term_depth(T, D) :- functor(T, _, A), …, D is MaxChild + 1.
```

Emits:

```
type_dispatch_a1(atom_tgt, int_tgt, cmpd_tgt, default_tgt)
```

`op1` packs `atom_tgt | (int_tgt << 32)`; `op2` packs
`cmpd_tgt | (default_tgt << 32)`. On dispatch, A1's tag selects
one of the four targets; any 0-target falls through.

Since `default_tgt` is always set by the peephole (to the
untyped fallback clause's label), every possible A1 tag
dispatches directly. The surviving `try_me_else` / `retry_me_else`
/ `trust_me` chain becomes dead code and is **removed** from the
output entirely by the peephole (PR #1526). `retry_me_else` and
`trust_me` runtime cases are `cp_count = 0`-guarded no-ops, so
their removal is semantically transparent.

### arg_call family

The single-biggest optimization on the `sum_ints_args/5` hot
loop: fuse the entire window

```
arg(I, T, A)              ← arg/3
put_value(A, A1)          ← pass A to next call's A1
put_value(Acc, A2)        ← accumulator
put_variable(Acc1, A3)    ← fresh return slot
call(sum_ints, 3)
```

into one `arg_call_reg_3` (or `arg_call_lit_3`) instruction that:
1. Does the arg/3 fast path inline (tag check + offset compute +
   value copy).
2. Loads A1, A2, A3 directly.
3. Saves CP and jumps to the callee's PC.

**Liveness variants** (`_dead` suffix) elide writing ArgDest's
Y-slot when `reg_used_before_clause_end/2` proves it unused
after the call. For `sum_ints_args/5`, the slot is dead (its
only role was feeding A1, which is done at fusion time).

**K-family** (tags 54–61) extends to 1-arg and 2-arg callees.
K=2 uses an IsVar flag bit to distinguish `put_value` (existing
value source) from `put_variable` (fresh return slot).

### tail_call_5

The mirror of arg_call — fuses the entire arity-5 tail-call
setup at clause end:

```
put_value(R1, A1)
put_value(R2, A2)
put_value(R3, A3)
put_value(R4, A4)
put_value(R5, A5)
deallocate
execute(Pred)
```

into `tail_call_5(R1,…,R5, Pred)`. Variant `tail_call_5_c1_lit`
handles the case where R1 is a literal (e.g., `put_constant(1, A1)`
in `sum_ints/3`'s recursive tail).

### Clause-end fusion family

Collapses various tails that end clauses (tags 63–71):

| Window | Instruction |
|--------|-------------|
| `deallocate + proceed` | `deallocate_proceed` |
| `deallocate + builtin_call + proceed` | `deallocate_builtin_proceed` |
| `deallocate + arg_direct + proceed` | `deallocate_arg_direct_proceed` (+ functor, copy_term, univ, is_list variants) |
| `builtin_call + proceed` (no dealloc) | `builtin_proceed` |

For each predicate with `allocate`, the clause must pay one
`$step` iteration for `deallocate` and another for `proceed`
(plus possibly one for a terminal builtin). These fusions
collapse 2–3 dispatches per clause end to 1.

---

## Performance trajectory

Cumulative improvement on `bench_sum_big` since the Phase 6
baseline (Apr 2026, 100K iter, Termux/aarch64, V8 JIT):

| Milestone | bench_sum_big ratio vs Phase 6 baseline |
|-----------|----------------------------------------|
| Phase 6 baseline | 1.00× |
| Initial fused arith + direct dispatch (PR #1456, #1463) | ~0.87× |
| arg_call family + liveness (PRs #1481–#1497) | ~0.69× |
| tail_call_5 + K-family extension (PRs #1501, #1510) | ~0.58× |
| type_dispatch_a1 family (PRs #1520, #1524, #1526) | ~0.52× |
| neck_cut_test fix + trail-aware variant (PRs #1528, #1531) | ~0.45× |
| **Current total** | **~−55%** |

Absolute times on current main (Apr 2026):

| Workload | ns/call |
|----------|---------|
| bench_true | ~55 |
| bench_is_arith | ~120 |
| bench_sum_small | ~1100 |
| bench_sum_medium | ~2700 |
| bench_sum_big | ~5500 |
| bench_term_depth | ~7000 |
| bench_fib10 | ~35000 |

### Where remaining time goes

Profiling (PR #1533) showed the dispatch loop (`$step`) at ~50%
of hot-loop time — the br_table + indirect-call-to-`$do_*`
structure is now the dominant cost. The individual `$do_*`
handlers are each <2% of total time; the fusion work has shifted
the cost distribution from per-instruction work into pure
dispatch overhead.

### Known limitations / future work

Documented in individual PR follow-up sections; highlights:

- **`$step` is hard to shrink further** — structural changes
  (inline `$do_*` bodies, batched steps, etc.) are bigger
  investments with uncertain payoffs.
- **3-clause neck_cut for `fib/3`** — would need new
  `try_get_constant_else` instructions to handle head-unification
  failure without a CP; ~10% ceiling on `bench_fib10`.
- **Move indexing/neck_cut to `wam_target.pl`** — share with
  other WAM backends (Go, Rust, LLVM, ILAsm, Elixir, JVM).
  No WAM-WAT bench impact; broadens ecosystem benefit.
- **Cooler measurement environment** — Termux thermal noise
  (±1-3%) obscures sub-percent wins. Desktop-Linux A/B would
  clarify marginal optimizations.

---

## Development guide

### Adding a new instruction

1. Choose the next free tag number and declare it in the
   `instr_tag/2` block (near the top of `wam_wat_target.pl`).
2. Add an encoder clause:
   ```prolog
   wam_instruction_to_wat_bytes(my_instr(Arg1, Arg2), Labels, Hex) :-
       instr_tag(my_instr, Tag),
       …encode Op1, Op2…,
       encode_instr_hex(Tag, Op1, Op2, Hex).
   ```
3. Add a WAT case:
   ```prolog
   wam_wat_case(my_instr,
   '  ;; comment
     (local $op1i i64)
     …WAT body…
     (i32.const 1)').
   ```
4. (Optional) Add a peephole clause to rewrite some existing
   pattern into the new instruction.
5. Run `swipl -g "consult(…test_wam_wat_target.pl), run_tests, halt"`.
6. Regenerate the bench: `swipl examples/wam_term_builtins_bench/generate_wat_bench.pl`
   + `wat2wasm bench_suite.wat -o bench_suite.wasm`.
7. Run `node run_bench.js bench_suite.wasm` to verify all 13
   workloads return OK.

### Running the profiler

See `examples/wam_term_builtins_bench/profile_one.js`:

```bash
cd examples/wam_term_builtins_bench
node --cpu-prof --cpu-prof-interval=100 profile_one.js \
     bench_suite.wasm bench_sum_big_0 300000
```

Generates a `CPU*.cpuprofile` file that can be analyzed with a
short Python script (see PR #1533 description for an example).

### A/B benchmarking

```bash
# Save baseline + branch wasm
cp examples/wam_term_builtins_bench/bench_suite.wasm /tmp/branch.wasm
git show main:examples/wam_term_builtins_bench/bench_suite.wasm > /tmp/main.wasm

# Run alternating pairs
for i in 1 2 3; do
  echo "=== Pair $i ==="
  echo "-- MAIN --"; node run_bench.js /tmp/main.wasm | grep bench_
  echo "-- BRANCH --"; node run_bench.js /tmp/branch.wasm | grep bench_
done
```

Use min-of-pair as the thermal-noise proxy: the minimum run of
each binary across N pairs is cleaner than the average on
thermally-constrained devices.
