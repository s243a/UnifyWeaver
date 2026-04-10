# WAM Term Builtins: Phase 0 Audit

**Date:** 2026-04-10
**Status:** Complete
**Branch:** `feat/wam-term-builtins`

This note records the findings of the Phase 0 audit described in
`WAM_TERM_BUILTINS_IMPLEMENTATION_PLAN.md`. The goal was to determine
whether any user-facing Prolog predicate in the repository currently
needs `functor/3`, `arg/3`, `=../2`, or `copy_term/2` to be transpilable,
and whether to proceed with Phases 1–6.

## Methodology

Searched every `.pl` file under `tests/`, `examples/`, `docs/`, and
`src/` for literal uses of the four builtins:

```
grep -rn 'functor(\|copy_term(\|=\.\.' <path>
grep -rn '\barg(' <path>
```

For each hit, classified as one of:

- **(H) Host-side harness**: runs inside SWI-Prolog at test time, not
  transpiled. Includes test goal construction, assertions about term
  shape, test-runner plumbing.
- **(C) Compile-time infrastructure**: part of UnifyWeaver's compilation
  machinery itself, runs in SWI at build time when transpiling. Not a
  transpiled predicate.
- **(U) User-facing transpiled predicate**: a Prolog predicate passed as
  input to `write_wam_*_project` / `compile_predicate_to_wam`. These are
  the ones that matter for the hypothesis.

## Findings

### Category H — host-side test harness (all hits)

All five files with hits in `tests/` are in this category:

| File | Builtin usage |
|------|---------------|
| `tests/core/test_csharp_query_target.pl` | 24x `=..` building test Head/Body terms to pass to the compiler |
| `tests/core/test_prolog_target.pl` | 9x `=..` building `Goal` terms for `call/1` test runners |
| `tests/core/test_semantic_dispatch.pl` | 4x `=..` destructuring goal wrappers in test dispatcher |
| `tests/integration/glue/test_visualization_glue.pl` | 1x `copy_term(Goal, FreshGoal)` for test isolation |
| `tests/test_go_goal_parallel.pl` | 1x `assertion(functor(Head, p, 2))` for term-shape check |

None of these are transpiled. They all run inside SWI-Prolog as part of
the test framework. The builtins here are supplied by SWI natively and
are used to construct/inspect Prolog terms that will then be handed off
to the WAM compiler — the compiler sees the *result* of `=..`, not the
`=..` call itself.

### Category C — compile-time infrastructure

Many files under `src/unifyweaver/` use these builtins (confirmed via
`grep -rln 'functor(\|copy_term(\|=\.\.' src/ --include='*.pl'`). All
sampled files fall into this category:

- `src/unifyweaver/targets/wam_target.pl` uses `=..` at line 609 to
  construct a mock goal for `is_guard_goal/2` — compile-time only
- `src/unifyweaver/targets/wam_target.pl` lines 52, 133, 144, 153, etc.
  use `functor/3` and `=..` to destructure user clause heads during
  compilation
- `src/unifyweaver/core/advanced/*.pl` pattern matchers and clause-body
  analyzers — compile-time term walking
- `src/unifyweaver/bindings/*_wam_bindings.pl` — compile-time target
  dispatch

These are the Prolog meta-programming idioms that UnifyWeaver itself is
built on. They run in SWI at build time; they are not transpiled.

### Category U — user-facing transpiled predicates

**Zero hits.**

The predicates that actually get compiled to WAM in the existing test
suites are all structural recursion patterns:

- `tc_ancestor/2`, `tc_descendant/2` — transitive closure
- `tc_distance/3`, `tc_parent_distance/4` — depth-bounded closure
- `tri_sum/2` — arithmetic accumulator recursion
- `tail_suffix/2`, `tail_suffixes/2` — list tail extraction
- `weighted_path/3`, `astar_weighted_path/4` — A* with weighted edges
- `min_semantic_dist/3`, `grouped_min_semantic_dist/3`,
  `filtered_adjusted_min_semantic_dist/3` — aggregate kernels

None of these use `functor/3`, `arg/3`, `=../2`, or `copy_term/2`. They
all work by clause-head pattern matching, which is the WAM's native
operation and requires no term introspection builtins.

### Indirect evidence from target specs

Two existing target specification docs already list `=../2` in their
intended builtin mapping tables, but neither actually implements it:

- `docs/design/WAM_GO_TRANSPILATION_SPECIFICATION.md:237` —
  `` `=../2` (univ) | `Compound` construction/destructure ``
- `docs/design/WAM_ILASM_TRANSPILATION_SPECIFICATION.md:278` —
  `` `=../2` (univ) | CompoundValue field access ``

This confirms the gap has been **documented intent** for at least two
targets, even though no code landed. The gap is known; it just hasn't
bit anyone hard enough to prioritize.

## Conclusion

The audit finding is **"genuinely bare" for existing code**, but the
gap is **acknowledged in target specs**. This matches the risk
contingency described in
`WAM_TERM_BUILTINS_IMPLEMENTATION_PLAN.md` under "Risk: Phase 0 finds
no real demand":

> Two responses:
> 1. **Shelve the plan** — document the finding, revisit when a real
>    demand appears.
> 2. **Construct demand** — write a new example that demonstrates where
>    the gap bites, and use that as the justification.

This plan takes **option 2: construct demand**. The reasoning:

1. **The gap is not hypothetical** — multiple target specs list `=../2`
   as intended, so the designers of those targets expected it to be
   implemented.
2. **The current test corpus is narrow by construction** — UnifyWeaver's
   existing tests focus on structural recursion patterns because that's
   what the pattern-matching compiler specializes in. Absence of these
   builtins in the test suite doesn't mean absence of demand in real
   Prolog code; it means the test suite hasn't yet tried to transpile
   anything meta-programming-flavored.
3. **The implementation cost is bounded** — per the spec, each of the
   four builtins is a few helpers per target. The Phase 2/4/5 plan is
   small enough that constructing demand via new test predicates is
   cheaper than the effort saved by shelving.
4. **Future-proofing has value** — landing these builtins now means
   that when a user *does* try to transpile a meta-interpreter, a
   serializer, or a memo-tabled predicate, it just works.

## Constructed demand: test predicate classes

Phases 2–5 will introduce new test predicates in each target's test
suite to exercise the builtins. At minimum:

### For `functor/3` and `arg/3`
```prolog
% Generic binary tree walker that reports functor + arity + first arg
walk_term(T, Name, Arity, FirstArg) :-
    functor(T, Name, Arity),
    Arity >= 1,
    arg(1, T, FirstArg).
```

### For `=../2`
```prolog
% Tiny meta-interpreter: runtime goal construction
invoke(Pred, Args, Result) :-
    Goal =.. [Pred | Args],
    call(Goal),
    Result = ok.
```

### For `copy_term/2`
```prolog
% Freeze a template before unification — the canonical use case
with_fresh_copy(Template, Result) :-
    copy_term(Template, Fresh),
    Fresh = Result.

% The sharing test — critical correctness check
sharing_test(X) :-
    copy_term(f(V, V), X).  % X should unify with f(Y, Y), not f(Y, Z)
```

These will be added to `test_wam_wat_target.pl`, `test_wam_rust_target.pl`,
and `test_haskell_target.pl` as Phases 2, 4, and 5 come online. Each
predicate is small enough to serve as both motivation and regression
test.

## Implications for subsequent phases

- **Phase 1 proceeds as planned** — extend `is_builtin_pred/2`. No
  regressions are expected because no existing predicate uses these
  builtins; the change is additive.
- **Phase 2/4/5 gain a new responsibility**: write the test predicates
  in `docs/examples/` or inline in test files, so future contributors
  see what the feature enables.
- **Phase 6 (perf investigation) becomes harder** — there is no current
  workload that falls back to host SWI because of these builtins,
  because there is no current workload that uses them at all. Phase 6
  will need to construct a synthetic benchmark. The philosophy doc
  already anticipates a negative result here, and the constructed
  benchmark should explicitly test the "many small copy_term in a
  memoization loop" scenario rather than re-running the existing
  effective-distance benchmark (which won't be affected).

## Decision

**Proceed with Phase 1.**
