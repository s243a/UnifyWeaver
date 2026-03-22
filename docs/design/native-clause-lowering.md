# Native Clause Body Lowering: Design Document

## Philosophy

UnifyWeaver compiles Prolog predicates to target languages. Today, most targets
handle recursion patterns (factorial, fibonacci, list_sum) through multifile
dispatch handlers — specialized code generators for each pattern. This works
well for clean, textbook patterns.

However, real-world Prolog predicates have **complex clause bodies**: nested
conditionals, guard sequences, multi-result branches, disjunctions. When a
target can't natively lower a clause body, it falls back to generic or
interpreted code — losing the performance and idiom benefits of native
compilation.

The TypR target pioneered a **native clause body lowering** system across ~50
commits (PRs #826–#898) that decomposes arbitrary clause bodies into native
control-flow constructs. This document describes the architecture, specifies
the requirements for porting to other targets, and provides an implementation
plan.

### Core Principle: Try Native, Fall Back Gracefully

The TypR target follows a dual compilation strategy:

```
try native_clause_body(PredSpec, Clauses, Code)
  → success: emit native target syntax
  → failure: wrap fallback code in raw embedding block
```

This means adding native lowering never breaks existing functionality — it only
improves output quality for predicates the lowering can handle.

### Why This Matters

Without native lowering, a predicate like:

```prolog
classify(X, Result) :-
    X > 0, X < 10, Result = small.
classify(X, Result) :-
    X >= 10, Result = large.
```

gets compiled as opaque function calls or string matching rather than a clean
if-else chain in the target language. Native lowering produces:

```python
def classify(x):
    if x > 0 and x < 10:
        return "small"
    elif x >= 10:
        return "large"
```

---

## Reference: TypR Implementation

### Commit History (PRs #826–#898)

**Phase 1: Non-Recursive Control Flow (PRs #826–#869)**

| PR | Commit | Pattern |
|----|--------|---------|
| #829 | `d52796d6` | Native control flow (if-then-else base) |
| #834 | `7659e02b` | Multi-step control flow |
| #835 | `d86c9b59` | Nonlinear control flow |
| #837 | `213d4524` | Fanout control flow |
| #838 | `5f96fb9b` | Split-recombine control flow |
| #841 | `08309827` | Guarded alternative assignments |
| #843 | `8e4682f7` | Direct output alternatives |
| #844 | `a4cb5e30` | Multi-result alternatives |
| #851 | `739cd592` | General multi-result alternatives |
| #854 | `864a547a` | Nested guarded alternatives |
| #855 | `b3b2a727` | Nested multi-result guarded alternatives |
| #856 | `d6d329e1` | Two-level nested guarded alternatives |
| #857 | `45087509` | Branch-local alternatives |
| #859 | `6828e8e3` | Multi-branch rejoin bodies |
| #860 | `d52c02b6` | Asymmetric partial rejoin |
| #861–#863 | `422c5e73`–`badf2067` | Non-recursive gap audits |
| #868 | `94ad4002` | Guard-only if-then-else chains |
| #869 | `570aab2d` | Asymmetric if-then-else branches |

**Phase 2: Recursion Pattern Lowering (PRs #870–#883)**

| PR | Commit | Pattern |
|----|--------|---------|
| #870 | `3d30b18a` | Tail recursion loops |
| #872 | `77cf32e5` | Linear recursion loops |
| #873 | `da297d4e` | List linear recursion loops |
| #874 | `1c68f837` | N-ary linear recursion loops |
| #876 | `7015ad87` | Guarded linear recursion recombination |
| #878 | `ff73f7da` | Multistate linear recursion recombination |
| #881 | `6076aadb` | Asymmetric linear recursion recombination |
| #883 | `ba517b0d` | Multicall recursion audit |

**Phase 3: Structural Tree Recursion (PRs #885–#898)**

| PR | Commit | Pattern |
|----|--------|---------|
| #885 | `05d13f1a` | Structural tree recursion |
| #886 | `ea0f6b07` | N-ary structural tree recursion |
| #887 | `7e020b36` | Structural tree prework |
| #889 | `fb4c03ab` | Structural tree branching |
| #891 | `0ec7498e` | Asymmetric structural tree branching |
| #893 | `38819ab6` | Structural tree invariant updates |
| #895 | `ed6d1875` | Subtree context recursion |
| #897 | `792ccb03` | Branch-local recursive calls |
| #898 | `fdad7a2c` | Recursive branch bodies |

### Key Source Locations

All references to `src/unifyweaver/targets/typr_target.pl`:

| Lines | Component | Purpose |
|-------|-----------|---------|
| 1618–1644 | `build_typr_function/5` | Entry point: tries native, falls back to R |
| 1640–1644 | Decision point | `native_typr_clause_body` → success or `wrapped_r_body_expression` |
| 1745–1760 | `native_typr_clause_body/3` | Single + multi-clause dispatch |
| 1767–1776 | `native_typr_clause/4` | Individual clause → (Condition, Code) |
| 1780–1810 | `native_typr_goal_sequence/5` | Core: goals → conditions + lines + result |
| 1811–1825 | `native_typr_prefix_goals/9` | Guard vs output classification |
| 1827–1845 | `native_typr_guarded_tail_sequence/8` | Interleaved guard+output handling |
| 1848–1854 | `native_typr_output_goal/6` | Single-result output lowering |
| 1855–1918 | `native_typr_output_expr/7` | Expression lowering (conditionals, disjunctions) |
| 1928–1935 | `typr_if_then_else_goal/4` | Pattern match `(If -> Then ; Else)` |
| 1937–1957 | If-then-else with shared output | Detects shared variables across branches |
| 1979–1984 | Disjunction with shared output | Flattens `;` into alternatives |
| 1989–2028 | `native_typr_multi_result_output_goal/6` | Container create + extract pattern |
| 2349–2395 | Container/extraction helpers | `list(...)` creation, `.subset2` extraction |
| 2397–2421 | Guard predicates | Comparison ops, R binding calls |
| 2572–2614 | VarMap management | `ensure_typr_var`, `lookup_typr_var`, `build_head_varmap` |

### Test Coverage

The TypR test suite (`tests/test_typr_target.pl`, 1736 lines) contains 100+
tests covering:

- 35+ non-recursive control flow edge cases
- 25+ recursion pattern variants
- Type annotation modes (explicit, infer, off)
- Fallback behavior validation

---

## Specification

### What Native Clause Lowering Must Handle

**Tier 1: Essential (covers most predicates)**

1. **Multi-clause predicates → if-else chains**
   - Each clause becomes a branch
   - Head unification becomes condition
   - Body becomes branch body

2. **Guard separation**
   - Arithmetic comparisons (`>`, `<`, `>=`, `=<`, `=:=`)
   - Type checks (`integer(X)`, `atom(X)`)
   - Must precede assignments in generated code

3. **Single-result outputs**
   - `Var is Expr` → variable assignment
   - Direct binding → assignment

4. **Variable mapping (VarMap)**
   - Prolog variables → target identifiers
   - Thread through clause body
   - Head args → `arg1, arg2, ...` or named parameters

**Tier 2: Important (handles conditionals)**

5. **If-then-else lowering**
   - Pattern: `(Guard -> Then ; Else)` in clause body
   - Shared output variable detection
   - Emit native conditional

6. **Nested conditionals**
   - Two+ levels of if-then-else
   - Variable scoping at each level

7. **Disjunction flattening**
   - `(A ; B ; C)` → alternatives list
   - Shared output variables across alternatives

**Tier 3: Advanced (handles complex real-world code)**

8. **Multi-result branches**
   - Branches producing different numbers of outputs
   - Container pattern: pack results, extract after conditional
   - Asymmetric branch convergence

9. **Guard-after-output patterns**
   - Guards that refine intermediate computed values
   - Requires interleaved guard+output processing

10. **Branch-local variable scoping**
    - Variables created inside branches don't leak
    - VarMap forking and merging

### Target-Specific Adaptations

Each target needs idiom-appropriate translations:

| Concept | Python | Go | Rust |
|---------|--------|----|------|
| If-else chain | `if/elif/else` | `if/else if/else` | `if/else if/else` or `match` |
| Variable binding | `x = expr` | `x := expr` | `let x = expr;` |
| Multi-result container | `(a, b)` tuple | `struct{a,b}` or multi-return | `(a, b)` tuple |
| Container extraction | `a, b = result` | `a, b := result.a, result.b` | `let (a, b) = result;` |
| Guard expression | `if x > 0:` | `if x > 0 {` | `if x > 0 {` |
| Fallback embedding | N/A (Python is native) | `// fallback` comment | `// fallback` comment |

---

## Implementation Plan

### Phase 0: Shared Infrastructure

Before porting to any target, extract the target-independent logic from TypR
into a shared module.

**Create `src/unifyweaver/core/clause_body_analysis.pl`:**

Extract from `typr_target.pl`:
- Goal classification (guard vs output vs control flow)
- If-then-else pattern matching
- Disjunction flattening
- Shared output variable detection
- Multi-result container need detection

These predicates work on Prolog clause structure, not target syntax — they
should be shared.

### Phase 1: Python Target (Pilot)

**Why Python first:**
- Clean syntax maps naturally to Prolog control flow
- No type declarations needed (like Prolog)
- Tuple unpacking handles multi-result cleanly
- Most portable/testable — everyone has Python

**Implementation steps:**

1. Add `native_python_clause_body/3` to `python_target.pl`
   - Single clause: extract guards → emit `if` checks, then assignments
   - Multi clause: emit `if/elif/else` chain

2. Add `native_python_goal_sequence/5`
   - Reuse shared goal classification from Phase 0
   - Emit Python assignments (`x = expr`)
   - Emit Python guards (`if x > 0:`)

3. Add `native_python_output_expr/7`
   - If-then-else → `x = expr_then if guard else expr_else`
   - Disjunction → chain of `if/elif`

4. Add `native_python_multi_result/6`
   - Container: `(a, b) = (expr_a, expr_b)` tuple unpacking

5. Add fallback: if native lowering fails, use existing Python codegen

6. Add tests mirroring TypR's test suite

### Phase 2: Go Target

**Why Go second:**
- Explicit types make it a good test of the type inference path
- Multi-return is idiomatic (no container hacks needed)
- Already has substantial target code (~19K lines)

**Additional considerations:**
- Go requires explicit variable declarations (`var x int` or `:=`)
- Multi-return: `a, b := someFunc()` — natural fit
- Error handling: Go's `if err != nil` pattern may interact with guards

### Phase 3: Rust Target

**Why Rust third:**
- Important for sciREPL integration
- `match` expressions map well to multi-clause predicates
- Ownership semantics require careful variable management
- Pattern matching is more powerful than if-else chains

**Rust-specific opportunities:**
- Multi-clause → `match` arms (more idiomatic than if-else)
- Guard clauses → `if` guards on match arms
- Tuple destructuring for multi-result
- `let` bindings with shadowing for variable rebinding

### Phase 4: Remaining Targets

After Python/Go/Rust prove the architecture:
- **Ruby**: Similar to Python, `if/elsif/else`
- **C/C++**: Needs struct-based multi-result, explicit types
- **Java/Kotlin/Scala**: Class-based, but multifile handlers already work
- **Haskell**: Guards map naturally, pattern matching is native
- **Elixir**: Pattern matching in function heads is native

### Testing Strategy

For each target, port the TypR test predicates progressively:

**Tier 1 tests (add first):**
```prolog
% Simple guard chain
classify(X, small) :- X > 0, X < 10.
classify(X, large) :- X >= 10.

% Single if-then-else
abs_val(X, R) :- (X >= 0 -> R = X ; R is -X).

% Linear recursion with guard
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
```

**Tier 2 tests (add after Tier 1 works):**
```prolog
% Nested conditional
range_classify(X, R) :-
    (X < 0 -> R = negative
    ; (X =:= 0 -> R = zero ; R = positive)).

% Multi-result branch
min_max(X, Y, Min, Max) :-
    (X =< Y -> Min = X, Max = Y ; Min = Y, Max = X).
```

**Tier 3 tests (add last):**
```prolog
% Asymmetric branches, guard-after-output, branch-local scoping
% (Port from TypR's test_typr_target.pl lines 108–678)
```

### Success Criteria

For each target, native clause lowering is complete when:

1. All Tier 1 test predicates compile to native target syntax (no fallback)
2. All Tier 2 test predicates compile with proper conditional handling
3. Generated code produces correct results (runtime tests pass)
4. Fallback still works for unsupported patterns (no regressions)
5. Performance: native code is at least as fast as fallback code

---

## Appendix: TypR Architecture Diagram

```
compile_predicate_to_typr(Pred/Arity, Options, Code)
    │
    ├─ transitive_closure detected? → compile_typr_transitive_closure (template)
    │
    └─ compile_generic_typr
         │
         ├─ build_typr_function
         │    │
         │    ├─ TRY: native_typr_clause_body(PredSpec, Clauses, Body)
         │    │    │
         │    │    ├─ Single clause → native_typr_clause → native_typr_goal_sequence
         │    │    │                                          │
         │    │    │                                          ├─ native_typr_prefix_goals
         │    │    │                                          │    ├─ Guard goals → conditions
         │    │    │                                          │    └─ Output goals → assignments
         │    │    │                                          │
         │    │    │                                          ├─ native_typr_output_expr
         │    │    │                                          │    ├─ If-then-else → conditional
         │    │    │                                          │    ├─ Disjunction → alternatives
         │    │    │                                          │    └─ Arithmetic → expression
         │    │    │                                          │
         │    │    │                                          └─ native_typr_multi_result_output_goal
         │    │    │                                               ├─ Container creation
         │    │    │                                               └─ Extraction lines
         │    │    │
         │    │    └─ Multi clause → map native_typr_clause → if/else chain
         │    │
         │    └─ FALLBACK: wrapped_r_body_expression → @{ R_CODE }@
         │
         └─ format as TypR function with type annotations
```
