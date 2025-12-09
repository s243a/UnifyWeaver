# Parameterized Queries (Query Mode) – Exploration Note

**Branch:** `feat/parameterized-queries-querymode`  
**Status:** Exploring; main remains unchanged

## Why
- Query mode today assumes head vars are unbound and seeded from relation scans. Recursive arithmetic (e.g., Fibonacci) fails because arguments must be computed before recursion.
- Generator mode already handles these patterns; we’re exploring parameterized queries so query mode can accept input-bound arguments and pre-recursion bindings.

## What’s being prototyped (on the feature branch)
- Mode declarations (`mode/1` with `+/-/?`) to mark input/output args (defaults to all-output for current behavior).
- IR/runtime extensions: parameter seed, bind-expr node (compute from bound vars before recursion), parameterized recursive refs.
- Codegen/runtime adjustments to support parameterized entry points while keeping the existing datalog path intact.

## Current state
- Branch `feat/parameterized-queries-querymode` has a WIP plan doc and a mode parser helper; no changes on main yet.
- Generator mode remains the fallback for recursive arithmetic; see `playbooks/csharp_generator_playbook.md` for a working Fibonacci example.

## Risks/considerations
- Correctness: interactions with negation/aggregates/recursion need careful testing.
- Scope: adds a parallel path; must preserve existing semantics when no modes are declared.
- Performance: expected neutral/positive for function-style queries (less scanning), but needs measurement.

## Next steps (on the feature branch)
- Implement mode parsing into plan metadata.
- Add seed + bind-expr + param-recursive-ref nodes; update codegen/runtime.
- Add tests: recursive arithmetic, negation, aggregates, mixed-mode predicates.

## Decision
- Keep main unchanged until the feature is stable; rely on generator mode for Fibonacci-like patterns in the meantime.
