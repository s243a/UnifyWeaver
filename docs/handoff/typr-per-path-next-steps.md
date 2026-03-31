# Handoff: Remaining TypR Per-Path Visited Work

## Goal
Hand off the next TypR per-path visited slice cleanly so another agent can
 continue from `main` without re-discovering the current boundary.

## Current Supported TypR Slice

`main` now supports conservative native TypR lowering for per-path visited
recursion with:

- detected `VisitedPos` rather than a fixed visited argument
- mode-driven input/output positions when `user:mode/1` is available
- one recursion-driving input plus conservative invariant non-visited inputs
- weighted variants with one direct node output plus additive numeric outputs
- native `*_from_vectors` runtime helpers
- native `input(stdin|file|vfs|function)` wrappers
- declared scalar runtime node parsing such as `integer` and `number`
- conservative pair-shaped runtime node parsing such as
  `pair(integer, integer)` and `pair(number, number)`

Primary implementation points:

- [src/unifyweaver/targets/typr_target.pl](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/src/unifyweaver/targets/typr_target.pl)
- [templates/targets/typr/per_path_visited.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/per_path_visited.mustache)
- [templates/targets/typr/ppv_definitions.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/ppv_definitions.mustache)
- [templates/targets/typr/ppv_input_stdin.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/ppv_input_stdin.mustache)
- [templates/targets/typr/ppv_input_file.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/ppv_input_file.mustache)
- [templates/targets/typr/ppv_input_vfs.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/ppv_input_vfs.mustache)
- [templates/targets/typr/ppv_input_function.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/ppv_input_function.mustache)

Relevant design references:

- [docs/design/PER_PATH_VISITED_IMPLEMENTATION_PLAN.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/design/PER_PATH_VISITED_IMPLEMENTATION_PLAN.md)
- [docs/design/TYPR_TARGET_DESIGN.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/design/TYPR_TARGET_DESIGN.md)
- [README.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/README.md)

## Confirmed Next Gap

The next real TypR gap is not another loader variation. It is helper or guard
goals over invariant inputs between the step relation and the recursive call.

Concretely, the current supported path assumes the recursive body is still
close to:

```prolog
step(Current, Next),
\+ member(Next, Visited),
pred(Next, ..., [Next|Visited])
```

The next failing slice is:

```prolog
step(Current, Next),
Next =< Limit,
\+ member(Next, Visited),
pred(Next, Limit, ..., [Next|Visited])
```

or an equivalent helper-goal form such as:

```prolog
step(Current, Next),
Allowed is Limit - 1,
Next =< Allowed,
\+ member(Next, Visited),
pred(Next, Limit, ..., [Next|Visited])
```

This should stay on the existing native TypR worker path. It should not fall
back to raw R or to an unrelated general-recursion matcher.

## Recommended Next Branch

- `feature/typr-per-path-invariant-guards-audit`

## Recommended Scope

Keep the next branch narrow:

- compile-time-seeded step relation first
- scalar node IDs first
- native TypR worker path only
- preserve existing `VisitedPos` and mode-driven input/output handling
- no broader runtime loader expansion in the same branch

## Proving Predicates

Use one or two tight probes:

```prolog
category_ancestor_limited(Cat, Ancestor, Hops, Limit, Visited) :-
    category_parent(Cat, Ancestor),
    Ancestor =< Limit,
    \+ member(Ancestor, Visited),
    Hops = 1.
category_ancestor_limited(Cat, Ancestor, Hops, Limit, Visited) :-
    category_parent(Cat, Mid),
    Mid =< Limit,
    \+ member(Mid, Visited),
    category_ancestor_limited(Mid, Ancestor, H1, Limit, [Mid|Visited]),
    Hops is H1 + 1.
```

Weighted follow-up:

```prolog
category_ancestor_weight_limited(Cat, Ancestor, Hops, Cost, Limit, Visited) :-
    ...
```

## Files Most Likely to Change

- [src/unifyweaver/targets/typr_target.pl](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/src/unifyweaver/targets/typr_target.pl)
- [templates/targets/typr/per_path_visited.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/per_path_visited.mustache)
- [templates/targets/typr/ppv_definitions.mustache](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/templates/targets/typr/ppv_definitions.mustache)
- [tests/test_typr_target.pl](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/tests/test_typr_target.pl)
- [tests/test_typr_toolchain.pl](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/tests/test_typr_toolchain.pl)
- [README.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/README.md)
- [docs/design/TYPE_SYSTEM_IMPL_PLAN.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/design/TYPE_SYSTEM_IMPL_PLAN.md)
- [docs/design/TYPE_SYSTEM_SPEC.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/design/TYPE_SYSTEM_SPEC.md)
- [docs/design/TYPR_TARGET_DESIGN.md](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/docs/design/TYPR_TARGET_DESIGN.md)

## Non-Goals for That Branch

Do not mix in:

- broader generic recursion lowering
- SCC work
- transitive-closure raw-R reduction
- richer runtime node-shape parsing beyond the current scalar and pair slices

## Do Not Touch

- [examples/sci-repl/prototype](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/examples/sci-repl/prototype)
- existing untracked `PR_DESCRIPTION_*` files
- [std.ty](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/std.ty)
