# Code Generation Implementation Plan — Converging Templates and Lowering

## Context

TypR pioneered a hybrid approach: templates for scaffolding,
native lowering for logic, R wrapping for fallback. The composable
template system (input modes) was developed independently for all
19 targets. This plan converges them.

## Phase 0: What Exists Today (DONE)

- 19 targets with composable templates (all 5 input modes)
- TypR with native lowering + R fallback + type-aware templates
- `input_source.pl` with `base_seed_code`, `seed_statement`,
  `resolve_input_mode`, `detect_context`
- TypR's `empty_collection_expr`, `annotation_suffix`,
  `resolve_node_type` (TypR-specific)

## Phase 1: Generalize TypR's Type Awareness

**Goal:** Type-aware template variables for statically typed targets.

**Scope:** Rust, Go, C, C++, TypeScript, Kotlin, Scala, F#, Haskell

**Changes:**

1. Move `empty_collection_expr/3` from `typr_target.pl` to a shared
   `type_resolution.pl` module, extended with per-target clauses:
   ```prolog
   empty_collection_expr(atom, rust, "Vec::new()").
   empty_collection_expr(atom, go, "[]string{}").
   ```

2. Move `resolve_node_type/3` to `type_resolution.pl` — extracts
   node type from declared types or infers from asserted facts.

3. Extend `compile_tc_from_template/6` to compute type-aware
   variables when the target has `empty_collection_expr` defined:
   ```prolog
   (   empty_collection_expr(NodeType, Target, EmptyExpr)
   ->  append(BaseDict, [empty_collection=EmptyExpr, node_type=TypeName], Dict)
   ;   Dict = BaseDict
   )
   ```

4. Update Rust, Go, TypeScript, C++ templates to use `{{node_type}}`
   and `{{empty_collection}}` where currently hardcoded to strings.

**Effort:** Medium — mostly moving existing code + new facts.

**Risk:** Low — fallback is to ignore new variables (templates still
work with hardcoded types).

## Phase 2: TypR Uses Composable Input Templates

**Goal:** TypR's TC path gains input mode flexibility.

**Scope:** TypR target only

**Changes:**

1. Split `compile_typr_transitive_closure` into two parts:
   - Type-aware variable computation (node type, annotations,
     empty_collection_expr, seed_code)
   - Template rendering

2. For the rendering step, check for composable templates:
   - If `input(Mode)` specified and composable templates exist,
     compose from parts
   - Otherwise, use monolithic `transitive_closure.mustache` (current)

3. Create TypR composable input templates:
   - `tc_input_stdin.mustache` — `readLines(file("stdin"))` via
     `@{ }@` raw expression (TypR can't do I/O natively)
   - `tc_input_file.mustache` — `readLines(path)` via raw R
   - `tc_input_vfs.mustache` — `nb_read(cell, prop)` via raw R
   - `tc_input_function.mustache` — TypR function accepting pairs

4. Keep `tc_definitions.mustache` as the existing monolithic
   template with `{{seed_code}}` removed (it moves to
   `tc_input_embedded.mustache`).

**Effort:** Medium — TypR's definitions section has `@{ }@` blocks
that make splitting non-trivial.

**Risk:** Medium — TypR templates use raw R extensively, splitting
must preserve IIFE scoping.

## Phase 3: Native Lowering for Second Target

**Goal:** Prove the native lowering approach works beyond TypR.

**Candidate:** TypeScript or Python (both have clear goal→expression
mappings and large user bases).

**Changes:**

1. Create `native_lowering.pl` — shared framework for goal-by-goal
   translation:
   ```prolog
   native_lower(Target, GoalSeq, VarMap, Code) :-
       native_goal_sequence(Target, GoalSeq, VarMap, Code).
   ```

2. Register binding translations for the target:
   ```prolog
   target_binding(python, length/2, [L,N], "~N = len(~L)").
   target_binding(python, append/3, [A,B,C], "~C = ~A + ~B").
   ```

3. Integrate into `compile_dispatch`: if pattern is not a named
   recursion type and native lowering succeeds, use it.

4. Fallback: if native lowering fails, report unsupported (no
   wrapped fallback unless a fallback target exists).

**Effort:** High — requires building the native lowering framework
and populating bindings.

**Risk:** High — correctness of generated code depends on complete
binding coverage.

## Phase 4: Validation Pipeline

**Goal:** Catch bugs in generated code before runtime.

**Scope:** Targets with available compilers/checkers on the host.

**Changes:**

1. Add `validate_generated(Target, Code, Result)` predicate.
2. Integrate into `compile_recursive/3` as optional post-step:
   ```prolog
   (   member(validate(true), Options)
   ->  validate_generated(Target, Code, Result),
       (Result = ok -> true ; report_validation_error(Result))
   ;   true
   )
   ```
3. Start with Rust (`cargo check`) and TypeScript (`tsc --noEmit`),
   since these are available on most dev machines.

**Effort:** Low per target — just shell out to the compiler.

**Risk:** Low — validation is opt-in, failure is informational.

## Phase 5: Fallback Chains

**Goal:** Maximum predicate coverage for targets with natural
fallback languages.

**Scope:** TypeScript→JavaScript, Kotlin→Java, Jython→Python

**Changes:**

1. Register fallback chains:
   ```prolog
   fallback_target(typescript, javascript).
   fallback_target(kotlin, java).
   ```

2. When native lowering fails and a fallback exists:
   - Compile to fallback target
   - Embed via FFI mechanism (TypeScript: `eval()` or `Function()`,
     Kotlin: inline Java via `@JvmStatic`)

3. TypR's `@{ }@` mechanism is the model — each target needs
   its own embedding syntax.

**Effort:** High — requires understanding each target's FFI.

**Risk:** Medium — embedding foreign code can break type safety.

## Priority and Dependencies

```
Phase 0 (DONE)
  ↓
Phase 1 (type awareness) ← no dependencies, can start now
  ↓
Phase 2 (TypR composable) ← depends on Phase 1 for shared module
  ↓
Phase 3 (second native lowering) ← independent of Phase 2
  ↓
Phase 4 (validation) ← independent, can start anytime
  ↓
Phase 5 (fallback chains) ← depends on Phase 3 for framework
```

## Metrics

| Phase | Templates | Native Lowering | Targets Affected | New Predicates |
|-------|-----------|-----------------|------------------|----------------|
| 0 | 133 files | TypR only | 19 | — |
| 1 | 0 new | shared type module | 9 (typed langs) | ~30 |
| 2 | 4 new | — | TypR | ~5 |
| 3 | 0 new | 1 new target | 1 | ~50 |
| 4 | 0 | — | all with compilers | ~5 |
| 5 | 0 | — | 3 (TS, Kotlin, Jython) | ~15 |
