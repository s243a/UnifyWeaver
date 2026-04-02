# WAM-to-Rust Transpilation: Implementation Plan

## Phase 0: Rust Binding Registry for Prolog Builtins (PREREQUISITE)

**Goal:** Define Rust equivalents for the Prolog builtins used by
`wam_runtime.pl`, so the native lowering pipeline can translate them.

**Scope:**
- Register bindings for `library(assoc)` operations → `HashMap`
- Register bindings for list operations → `Vec`
- Register bindings for arithmetic → native Rust operators
- Register bindings for `format/2` → `format!()`
- Register bindings for `=../2` → `Value` enum match/construct

**Changes:**
1. Create `src/unifyweaver/bindings/rust_wam_bindings.pl`:
   ```prolog
   :- module(rust_wam_bindings, [rust_wam_binding/5]).

   % rust_wam_binding(+PrologPred/Arity, +RustExpr, +ArgMap, +ReturnMap, +Props)
   rust_wam_binding(get_assoc/3, "~map.get(&~key).cloned()",
       [key-string, map-hashmap], [value-value], [pure]).
   rust_wam_binding(put_assoc/4, "{ let mut m = ~map.clone(); m.insert(~key, ~val); m }",
       [key-string, map-hashmap, val-value], [result-hashmap], [pure]).
   rust_wam_binding(empty_assoc/1, "HashMap::new()",
       [], [result-hashmap], [pure]).
   ```

2. Register these bindings with the Rust target's `is_builtin_goal`
   detection so they're recognized during native lowering.

**Effort:** Medium — mechanical mapping work.

**Risk:** Low — the mappings are straightforward.

## Phase 1: Mustache Templates for Rust WAM Crate

**Goal:** Define the Rust crate skeleton that the transpiled WAM runtime
will live in.

**Scope:**
- `templates/targets/rust_wam/Cargo.toml.mustache`
- `templates/targets/rust_wam/value.rs.mustache` — `Value` enum
- `templates/targets/rust_wam/state.rs.mustache` — `WamState` struct
- `templates/targets/rust_wam/instructions.rs.mustache` — `Instruction` enum
- `templates/targets/rust_wam/lib.rs.mustache` — module layout

**Changes:**
1. Create template files in `templates/targets/rust_wam/`
2. Register templates in `template_system.pl` via `template/2` facts
3. Add `compile_wam_rust_crate/3` to `wam_target.pl` or a new
   `wam_rust_target.pl` module that orchestrates template rendering
   + code generation

**Template composition strategy:**
- Larger templates (value.rs, state.rs) use Mustache sections
- Method bodies within templates are filled by native lowering
- `{{step_wam_match_arms}}` placeholder is filled by compiling
  `step_wam/3` clauses to Rust match arms

**Effort:** Medium — template authoring + registration.

**Risk:** Low — follows established template patterns.

## Phase 2: Transpile `step_wam/3` to Rust `match`

**Goal:** Compile the multi-clause `step_wam/3` predicate to a Rust
`match` expression, producing the core instruction dispatcher.

**Scope:**
- Extend `rust_target.pl` to handle compound-head multi-clause dispatch
  as Rust `match` on enum variants
- Each `step_wam` clause body → one match arm body, lowered via
  `clause_body_analysis` with Rust bindings
- Handle state threading: `wam_state(PC, R, S, H, T, CP, CPS, Code, L)`
  fields → `self.pc`, `self.regs`, `self.stack`, etc.

**Changes:**
1. Add `compile_match_dispatch_to_rust/4` in `rust_target.pl`:
   ```prolog
   compile_match_dispatch_to_rust(Pred, Arity, Clauses, RustCode) :-
       Clauses are all compound-first-arg,
       group by first-arg functor,
       each group → one match arm,
       each arm body → native_rust_clause_body.
   ```

2. Wire into `compile_predicate_to_rust_normal` as a new tier
   (between recursive patterns and single rules).

3. State field mapping:
   ```prolog
   wam_state_field(1, 'self.pc').
   wam_state_field(2, 'self.regs').
   wam_state_field(3, 'self.stack').
   wam_state_field(4, 'self.heap').
   wam_state_field(5, 'self.trail').
   wam_state_field(6, 'self.cp').
   wam_state_field(7, 'self.choice_points').
   wam_state_field(8, 'self.code').
   wam_state_field(9, 'self.labels').
   ```

**Effort:** High — this is the core compilation challenge.

**Risk:** Medium — clause body analysis handles most patterns, but
`step_wam` bodies use assoc operations that need the Phase 0 bindings.

**Depends on:** Phase 0, Phase 1.

## Phase 3: Transpile Helper Predicates

**Goal:** Compile the remaining `wam_runtime.pl` predicates to Rust
functions via native lowering.

**Scope:**
- `run_loop/2` → `fn run(&mut self) -> bool` (recursive loop)
- `backtrack/2` → `fn backtrack(&mut self) -> bool`
- `unwind_trail/4` → `fn unwind_trail(&mut self, saved: &[TrailEntry])`
- `eval_arith/5` → `fn eval_arith(&self, expr: &Value) -> f64`
- `deref_heap/3` → `fn deref_heap(&self, val: &Value) -> Value`
- `is_unbound_var/1` → `fn is_unbound(&self) -> bool` (on Value)
- `trail_binding/4` → `fn trail_binding(&mut self, key: &str)`
- `get_reg/3`, `put_reg/6` → register access methods
- `parse_instr/2` → instruction parser (if needed for string input)

**Changes:**
1. Each predicate → attempt native lowering first
2. For predicates that resist → WAM-compile as fallback (but most
   of these are simple enough for native lowering)
3. Wire into the template's `{{helper_functions}}` placeholder

**Effort:** Medium — most helpers are straightforward.

**Risk:** Low — these are simpler than `step_wam/3`.

**Depends on:** Phase 0.

## Phase 4: WAM Fallback Integration in Rust Target

**Goal:** When `compile_predicate_to_rust_normal` fails, fall back to
WAM compilation + Rust codegen.

**Scope:**
- Add final clause to `compile_predicate_to_rust_normal`:
  ```prolog
  compile_predicate_to_rust_normal(Pred, Arity, Options, RustCode) :-
      % All native tiers failed — fall back to WAM
      wam_target:compile_predicate_to_wam(Pred/Arity, Options, WamCode),
      compile_wam_instructions_to_rust(WamCode, RustCode).
  ```
- Generate Rust code that creates an `Instruction` array and calls
  the transpiled WAM runtime to execute it
- Register natively-lowered predicates as builtins so WAM-compiled
  code can call them via `BuiltinCall`

**Changes:**
1. `compile_wam_instructions_to_rust/2` — parse WAM instruction
   string → Rust `vec![Instruction::GetConstant(...), ...]` literal
2. Wrapper function generation:
   ```rust
   fn predicate_name(vm: &mut WamState, a1: Value, a2: Value) -> bool {
       static CODE: &[Instruction] = &[ ... ];
       vm.load(CODE);
       vm.set_reg("A1", a1);
       vm.set_reg("A2", a2);
       vm.run()
   }
   ```
3. Builtin dispatch table for natively-lowered predicates

**Effort:** Medium.

**Risk:** Medium — interop calling convention needs careful design.

**Depends on:** Phase 2, Phase 3.

## Phase 5: End-to-End Testing & Validation

**Goal:** Compile a mixed-complexity Prolog module to Rust, run it,
verify correctness.

**Scope:**
- Test module with facts, rules, recursive predicates, and predicates
  requiring WAM fallback
- `cargo test` validation of generated code
- Comparison of Prolog-runtime results vs Rust-compiled results

**Changes:**
1. Create `tests/test_wam_rust_transpilation.pl` — Prolog-side tests
   that compile predicates and verify the generated Rust
2. Create test Cargo project that exercises the generated code
3. CI integration

**Effort:** Medium.

**Risk:** Low — validation is mechanical.

**Depends on:** Phase 4.

## Priority and Dependencies

```
Phase 0 (Rust bindings for Prolog builtins) ← independent
  ↓
Phase 1 (Mustache templates for crate structure) ← independent
  ↓
Phase 2 (step_wam/3 → Rust match) ← depends on Phase 0, 1
  ↓
Phase 3 (helper predicates → Rust functions) ← depends on Phase 0
  ↓
Phase 4 (WAM fallback in Rust target) ← depends on Phase 2, 3
  ↓
Phase 5 (E2E testing) ← depends on Phase 4
```

Phases 0 and 1 can proceed in parallel. Phase 3 can proceed in
parallel with Phase 2 after Phase 0 is complete.

## Metrics

| Phase | Templates | Native Lowering | New Tests | Risk |
|-------|-----------|-----------------|-----------|------|
| 0 | 0 | 15+ bindings | 5 | Low |
| 1 | 5 files | 0 | 2 | Low |
| 2 | 0 | 1 major predicate | 5 | Medium |
| 3 | 0 | 10+ predicates | 10 | Low |
| 4 | 1 wrapper | 1 fallback path | 5 | Medium |
| 5 | 0 | 0 | 10+ | Low |

## Future Extensions

Once the Rust pipeline works, the same architecture extends to:

- **WAM → WAT (WebAssembly text)**: Replace Rust templates with WAT
  templates, reuse the same instruction → target lowering pipeline.
- **WAM → JVM (Jamaica/Krakatau)**: Replace with JVM bytecode
  templates. The `shared_logic` pattern from agent-loop would let
  instruction semantics be defined once and expanded per target.
- **WAM → C**: For embedded/native targets. The `Value` enum becomes
  a tagged union, `HashMap` becomes a hash table, etc.

The binding registry (Phase 0) and instruction semantics (Phase 2) are
the reusable core; only the templates and target-specific lowering rules
change per downstream target.
