# C# Generator Mode – Handoff for Review

## What we’re doing
- Generator mode emits standalone C# with a local fixpoint engine (`Solve()`): facts + `ApplyRule_*` + worklist loop.
- Supported features: joins, builtins, stratified negation; aggregates (`aggregate_all/3`: count/sum/min/max/set/bag; `aggregate_all/4` grouped sum/min/max/set/bag/count).
- Recent perf/UX touches:
  - Relation + arg0/arg1 indexing in `Solve()`; joins/aggregates prefer buckets, fallback to relation list.
  - Early evaluation of bound-only builtins/negation inside joins (safe—no ordering change when vars unbound).
  - Indexing can be disabled via `enable_indexing(false)` option.

## Relation to other targets
- C# query/streaming targets use the managed QueryRuntime; generator mode bypasses it and runs standalone.
- Generator shares helpers (`common_generator`: var maps, builtin translation) with other targets; the goal is to converge on a shared generator API.
- Proposal that captures the cross-target direction: `proposals/generator_mode_cross_target.md` (unified generator IR; shared helpers for joins/negation/aggregates).

## Why it could be a common API
- Fixpoint + `Fact` abstraction is minimal and portable; other targets (Python/Bash) already reuse `common_generator` pieces.
- If the generator IR (joins/negation/aggregates) is stabilized, we can render to multiple runtimes with consistent semantics and less duplicated code.
- Current C# generator is the most feature-complete standalone path and can serve as the reference.

## Known/perf gaps (not landed)
- Deeper argN indexing (beyond arg1) was explored but not merged; arg0/arg1 buckets remain the default.
- Constraint reordering is conservative (only bound-only builtins/negation move early). No global reordering or selectivity heuristics yet.
- Indexing toggle is coarse (on/off); no finer-grained knobs.

## Suggested review focus for Claude
- Validate semantics around aggregates (grouped count/min/max/set/bag), joins with negation, and indexing fallback.
- Consider portability of the generator IR to other targets per the proposal.
- Assess whether argN indexing or smarter constraint ordering is worth re-attempting, and how to gate it safely (order-independence flags, bound-var checks).
