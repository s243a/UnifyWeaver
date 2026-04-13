# Proposal: Compile If-Then-Else to WAM Instructions

## Status

Open — discovered during optimized Prolog benchmarking (#1359).

## Problem

The WAM compiler (`wam_target.pl`) cannot compile Prolog if-then-else
constructs `(Cond -> Then ; Else)` or bare disjunctions `(A ; B)`.
The compiler's `compile_goals` treats `;` as a regular goal and tries
to compile it as `call ;/2`, which fails. The `clause_body_analysis`
module's `disjunction_alternatives` is called on the body during
analysis and enters an infinite recursion for deeply nested structures.

This affects **all WAM targets**: Haskell, Rust, WAT, and LLVM.

### Where this matters

The Prolog optimization passes (`prolog_target.pl`) generate predicates
like `power_sum_selected/3` that use if-then-else for dispatch:

```prolog
'category_ancestor$power_sum_selected'(A, B, C) :-
    (   nonvar(B)
    ->  'category_ancestor$power_sum_bound'(A, B, C)
    ;   'category_ancestor$power_sum_grouped'(A, B, C)
    ).
```

These predicates cannot be compiled to WAM, blocking the optimized
Prolog pipeline for WAM targets.

### How direct transpilation targets handle it

All 20+ direct transpilation targets (Rust, Python, Go, C, Java,
TypeScript, Haskell, etc.) handle if-then-else natively using
`clause_body_analysis:if_then_else_goal/4` to detect the pattern and
then emitting the target language's native `if/else` construct. No
choice points are needed because the target language handles control
flow directly.

## Proposed WAM Compilation

### If-Then-Else: `(Cond -> Then ; Else)`

The standard WAM approach uses a temporary choice point with cut:

```
    try_me_else L_else       % push choice point for Else
    <compile Cond goals>     % condition — may fail
    !/0                      % cut: commit to Then, remove Else CP
    <compile Then goals>     % then branch
    jump L_continue          % skip Else (needs new Jump instruction
                             % or can use Proceed if in tail position)
L_else:
    trust_me                 % pop the choice point
    <compile Else goals>     % else branch
L_continue:
```

If the condition succeeds, the cut removes the Else choice point and
execution continues with Then. If the condition fails, backtracking
restores to the Else choice point and executes Else.

### Bare Disjunction: `(A ; B)` (without `->`)

Same pattern but without cut:

```
    try_me_else L_alt
    <compile A goals>
    jump L_continue
L_alt:
    trust_me
    <compile B goals>
L_continue:
```

### Multi-way Disjunction: `(A ; B ; C)`

Flattened using `disjunction_alternatives` into a try/retry/trust chain,
same as multi-clause predicates.

## Implementation Notes

### New instruction needed: Jump

The WAM instruction set doesn't have an unconditional jump. Current
options:
- **Add `Jump Label`** — cleanest, new instruction in all WAM runtimes
- **Use anonymous predicates** — compile each branch as a separate
  anonymous clause and Call them. Avoids new instructions but adds
  call overhead.
- **Inline the branches** — only works for tail-position disjunctions

### Label generation

The compiler needs to generate unique labels for Else/Continue points.
These labels must not collide with clause labels (L_pred_arity_N).
Suggested format: `L_ite_<pred>_<arity>_<counter>`.

### Variable scoping

Variables bound in the Cond are visible in Then but not in Else (Else
backtracks to before Cond). Variables bound in Then or Else may need
to be unified with the continuation. The compiler needs to handle
permanent variable allocation across the branches.

### Cut interaction

The `!/0` in the if-then-else compilation is a "soft cut" (neck cut) —
it only removes the immediately enclosing choice point, not all choice
points up to the clause barrier. The current WAM runtime's `!/0`
implementation truncates to `wsCutBar`, which may be too aggressive.
Need to verify that the cut semantics are correct for nested
if-then-else.

### Which targets need changes

1. **`wam_target.pl`** — `compile_goals` needs to detect `;` / `->` and
   emit the try/cut/trust pattern instead of treating it as a call.
2. **Haskell WamRuntime.hs** — needs `Jump` instruction support (trivial).
3. **Rust state.rs** — needs `Jump` instruction support.
4. **WAT** — needs `Jump` instruction support.
5. **LLVM** — needs `Jump` instruction support.

The fix is primarily in `wam_target.pl` (shared). The target runtimes
need only the `Jump` instruction added.

## Alternatives

### Alternative: Desugar before WAM compilation

Transform `(Cond -> Then ; Else)` into separate clauses before WAM
compilation:

```prolog
pred_ite_1(Args) :- Cond, !, Then.
pred_ite_1(Args) :- Else.
```

This avoids new WAM instructions but requires synthesizing new
predicate names and adds call/return overhead.

### Alternative: Normalize in `normalize_goals`

Add a case to `normalize_goals` in `clause_body_analysis.pl` that
converts if-then-else to a Call to a synthesized helper predicate.
This is the desugar approach at the analysis level.

## Scope

- Compile `(Cond -> Then ; Else)` to WAM try/cut/trust pattern
- Compile `(A ; B)` to WAM try/trust pattern
- Handle nested disjunctions
- All WAM targets benefit (shared `wam_target.pl`)
- Does NOT cover: nested if-then-else optimization, indexing on
  condition outcome, or first-argument indexing within branches

## References

- `src/unifyweaver/targets/wam_target.pl:474` — `compile_goals` (missing `;` case)
- `src/unifyweaver/core/clause_body_analysis.pl:129` — `if_then_else_goal/4`
- `src/unifyweaver/core/clause_body_analysis.pl:139` — `disjunction_alternatives/2`
- `src/unifyweaver/targets/rust_target.pl:7134` — Rust if-then-else handler (reference)
- `src/unifyweaver/targets/wat_target.pl:1930` — WAT if-then-else handler (reference)
- #1359 — optimized Prolog benchmark (exposed the gap)
