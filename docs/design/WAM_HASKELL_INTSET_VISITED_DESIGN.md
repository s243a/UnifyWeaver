# IntSet-Backed Visited Set: Design

> **Companion to** the mode-analysis arc (`WAM_HASKELL_MODE_ANALYSIS_*.md`,
> Phase F + Phase G in `WAM_PERF_OPTIMIZATION_LOG.md`). This doc proposes the
> next algorithmic perf step after the constant-factor lowerings landed in
> PRs #1664–#1681.

## Why now

`\+ member(X, V)` is the visited-set check pattern in graph-traversal
predicates like `category_ancestor/4`. The Phase G lowering replaced the
`put_structure member/2 + builtin_call \+/1` chain with a single
`NotMemberList` instruction, saving ~210 ns per call (dispatch overhead +
heap allocation for the goal term).

The microbenchmark showed a **14 % win**, the macro benchmark on
effective-distance shows a **17 % win**. Both are constant-factor.

The dominant remaining cost on this hot path is the `VList` walk itself:
`O(N)` per call, where N is the depth of the visited path. For deep
recursion (`max_depth=10` in the existing benchmark, 50+ for richer
graph workloads), this dominates.

This doc proposes an algorithmic fix: represent the visited set as
`IntSet` (a Patricia trie keyed by interned atom IDs) so membership
goes from `O(N)` to `O(log N)`, and insertion goes from `O(1)` (cons)
to `O(log N)` — but for `N << 100`, both ops are essentially constant
in practice.

## Approach

A new `Value` variant, two new instructions, and a small compile-time
detection step that opts a specific predicate-argument position into
the IntSet representation.

### Runtime: new `Value` variant

```haskell
data Value
  = ...
  | VSet !IS.IntSet           -- visited set, keys are interned atom IDs
```

`VSet` is structurally distinct from `VList` and never occurs implicitly.
It is constructed by the new instructions (below) and consumed by
`NotMemberSet`. Pattern matches on `Value` that don't recognise it
fall through to `Nothing` (consistent with how `VList` is treated by
non-list-aware handlers).

**Touch points** (all 112 `case Value` sites in `wam_haskell_target.pl`'s
template were inventoried during scoping):

- `derefVar` — VSet is a self-deref (no var inside, return as-is).
- `NFData Value` — `rnf (VSet s) = rnf s`.
- `Eq Value` — auto-derive works (IntSet has Eq).
- `Show Value` — auto-derive or custom for debugging.
- `addToBuilder`, `copyTermWalk`, `unifyVal` — VSet is opaque in these
  paths; either explicit `_ -> Nothing` rejection, or pass-through.
- Most `case` matches default to `_ -> Nothing` already, so the
  net diff is small.

### Runtime: three new instructions

```haskell
data Instruction
  = ...
  | BuildEmptySet !RegId         -- write VSet IS.empty into the named register
  | SetInsert !RegId !RegId !RegId  -- elemReg, inReg, outReg
  | NotMemberSet !RegId !RegId   -- elemReg, setReg
```

Step handlers:

```haskell
step !_ctx s (BuildEmptySet r) =
  Just (s { wsPC = wsPC s + 1
          , wsRegs = IM.insert r (VSet IS.empty) (wsRegs s) })

step !_ctx s (SetInsert eReg inReg outReg) =
  let mE = derefVar (wsBindings s) <$> IM.lookup eReg (wsRegs s)
      mIn = derefVar (wsBindings s) <$> IM.lookup inReg (wsRegs s)
  in case (mE, mIn) of
    (Just (Atom aid), Just (VSet s')) ->
      Just (s { wsPC = wsPC s + 1
              , wsRegs = IM.insert outReg (VSet (IS.insert aid s')) (wsRegs s) })
    _ -> Nothing

step !_ctx s (NotMemberSet eReg setReg) =
  let mE = derefVar (wsBindings s) <$> IM.lookup eReg (wsRegs s)
      mSet = derefVar (wsBindings s) <$> IM.lookup setReg (wsRegs s)
  in case (mE, mSet) of
    (Just (Atom aid), Just (VSet s')) ->
      if IS.member aid s' then Nothing else Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing
```

`Atom`-only constraint: VSet members must be interned atom IDs.
Mixed-type visited sets (atom + integer + struct) are not supported in
this representation — they fall back to `VList`. This is fine for the
canonical visited-set use case (categories, names, identifiers).

### Compile-time: opt-in via directive

Add a directive recognised at codegen time:

```prolog
:- visited_set(Pred/Arity, ArgN).
```

For example:

```prolog
:- visited_set(category_ancestor/4, 4).  % arg 4 (Visited) is a set
```

When the WAM compiler sees this, it:

1. **At the head of the predicate:** if the visited arg is a list literal
   (`[Cat]`), emit `BuildEmptySet` + a single `SetInsert` for each
   atom in the literal. Otherwise treat the input as already a `VSet`
   from the caller.

2. **At cons construction `[X|V]`:** if the cons target is the visited
   arg of a recursive call, emit `SetInsert X V V'` where V' is a fresh
   X-register. The recursive call passes V' instead of building a new
   `VList` cell.

3. **At `\+ member(X, V)` where V is the visited arg:** emit
   `NotMemberSet X V` instead of `NotMemberList X V`.

The compile-time recognition piggy-backs on the existing
`binding_state_analysis` pass: a new predicate
`is_visited_set_var(Var, ClauseHead, ArgN)` returns true when `Var` is
the N-th head argument of a predicate marked with `:- visited_set/2`.

### Bootstrap: how the visited list enters

In the source code:

```prolog
path_to_root(Article, Root, Hops) :-
    article_category(Article, Cat),
    category_ancestor(Cat, AncestorCat, CatHops, [Cat]),  % <-- starts with [Cat]
    ...
```

The initial `[Cat]` is a 1-element list literal. The compiler recognises
that `category_ancestor/4` arg 4 is a visited-set, and emits, in
`path_to_root/3`'s body for that goal:

```
build_empty_set X1
set_insert <reg-of-Cat>, X1, X1
... (rest of put_arguments)
call category_ancestor/4
```

Instead of:

```
put_list <Acat-reg>
set_value <reg-of-Cat>
set_constant []
... (rest of put_arguments)
call category_ancestor/4
```

The result: every call into `category_ancestor/4` passes a `VSet`
through the visited slot, and the recursive cons becomes `SetInsert`,
and `\+ member` becomes `NotMemberSet`.

## Expected speedup

For `category_ancestor`-style traversal at depth N:

- Existing path: `\+ member` is `O(N)` per call, `N` calls per path,
  → `O(N²)` per path of length `N`.
- IntSet path: `\+ member` is `O(log N)`, `N` calls per path,
  → `O(N log N)` per path.

For `max_depth=10` (current benchmarks): `100` vs `~30` ops per path,
~3× improvement on the visited-set portion.

For `max_depth=50`: `2500` vs `~280`, ~9× improvement.

The actual macro speedup will be smaller because visited-set work is
not 100 % of total time. At 1k scale with `max_depth=10`, the existing
mode-analysis arc lands ~17 % macro speedup; IntSet should add another
~1.5–3× on top of that, depending on whether visited-set dominates.

## Soundness

The lowering is sound only if every value flowing into the visited set
is an `Atom`. Two enforcement points:

1. **Compile-time (directive):** `:- visited_set(Pred/Arity, ArgN)` is
   user-asserted; if the user lies (visited contains non-atoms), the
   `SetInsert` step handler returns `Nothing` and the goal fails. No
   silent miscompilation.

2. **Runtime fallback:** if the user code at any point does
   `\+ member(X, V)` where `V` is unexpectedly a `VList` (e.g.
   constructed outside the recognised cons pattern), the analysis
   doesn't fire `NotMemberSet` and falls back to `NotMemberList`.

## Out-of-scope for the first IntSet PR

- **Auto-detection without directive.** Heuristically detecting "this
  argument is a visited set" from the body shape (cons + `\+ member`
  on the same arg) is doable but riskier. Defer to a follow-up.
- **Mixed visited representations.** A predicate that calls another
  with a `VList` visited but uses `VSet` locally is not supported.
  The directive applies uniformly across all callers.
- **Other set-shaped accumulators.** `bagof`/`setof`-style aggregation
  could benefit similarly; out of scope.
- **Conversion between `VList` and `VSet`.** If the user wants to
  iterate the visited set elsewhere, they must convert manually. No
  automatic `set_to_list` instruction.

## Test surface for the implementation PR

Mirroring the existing PRs in the arc:

- **Runtime unit tests** (`tests/core/test_wam_intset_runtime.pl`) —
  invoke `BuildEmptySet`, `SetInsert`, `NotMemberSet` step handlers
  directly with hand-crafted states.
- **Codegen unit tests** (`tests/core/test_wam_visited_set_lowering.pl`)
  — assert the WAM text contains `not_member_set` and `set_insert`
  when `:- visited_set/2` is declared.
- **Macro benchmark extension** — re-run
  `wam_effective_distance_macro_bench.pl` with the directive added,
  compare against the existing 17 % baseline.

## Related

- `WAM_HASKELL_MODE_ANALYSIS_PHILOSOPHY.md` — the analysis substrate
  this lowering builds on.
- `WAM_PERF_OPTIMIZATION_LOG.md` Phase G — the constant-factor `\+
  member` lowering this would compound with.
- Task #191 — implementation tracking.
