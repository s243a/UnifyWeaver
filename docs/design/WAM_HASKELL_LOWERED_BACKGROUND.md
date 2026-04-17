# WAM-Lowered Haskell: Background — Taxonomy of Haskell Targets

This document captures the discussion that precipitated the WAM-lowered
Haskell design (see `WAM_HASKELL_LOWERED_PHILOSOPHY.md`,
`WAM_HASKELL_LOWERED_SPECIFICATION.md`, and
`WAM_HASKELL_LOWERED_IMPLEMENTATION_PLAN.md`). It exists to ground the
design docs in the *actual current code*, not in hand-waving about
"modes," and to record the clarification of what "hybrid WAM with/without
FFI" actually means at the code level.

Nothing in this document is a design decision in its own right. It is
descriptive, not prescriptive. Read the philosophy/spec/plan docs for
the decisions; read this document to understand the state of the world
those decisions apply to.

## The three paths

UnifyWeaver has (or, in the case of path 3, is introducing) three
top-level code generation paths that produce Haskell from Prolog. They
are not variations of the same generator. They are three distinct
compilation strategies with different inputs, different intermediate
analyses, and different failure modes.

### Path 1 — Direct native Haskell lowering

- **Source file:** `src/unifyweaver/targets/haskell_target.pl`
- **Entry point:** `compile_predicate_to_haskell/3` (line 72), via
  `native_haskell_clause_body/3`.
- **What it does:** emits straight-line Haskell functions directly from
  the Prolog clause AST. No WAM involved at all. For example, a
  `parent/2` fact set becomes a Haskell function that pattern-matches
  on the arguments against a list of tuples.
- **When it fires:** only for predicates whose clauses the lowering
  analysis (`native_haskell_clause_body/3`) can statically reason
  about. Anything with complex backtracking, cut interactions,
  meta-call, or aggregation semantics falls through to path 2.
- **Emission shape:** something like
  ```haskell
  ancestor x y
    | parent x y = True
    | ...
  ```
  or, for facts:
  ```haskell
  parent :: [(Entity, Entity)]
  parent = [(Tom, Bob), (Bob, Jim)]
  ```
- **Why it is fast:** GHC sees a pure functional program with no
  interpretation layer, no boxed state, no dispatch loop. It optimizes
  the output as if a human wrote it.
- **Why it is narrow:** every Prolog feature the lowering analysis
  does not support causes the predicate to fall through.

### Path 2 — WAM-interpreted Haskell (current production path)

- **Source file:** `src/unifyweaver/targets/wam_haskell_target.pl`
- **Entry point:** `write_wam_haskell_project/3`.
- **What it does:** compiles Prolog → WAM IR (via
  `wam_target:compile_predicate_to_wam/3`) → a Haskell project
  containing `WamTypes.hs`, `WamRuntime.hs`, `Predicates.hs` (the
  instruction array and label map), and `Main.hs`. At runtime, the
  `run`/`step` functions in `WamRuntime.hs` walk the instruction array
  and execute each WAM instruction.
- **Runtime dispatch chain at `Call`:** this is where the "hybrid" and
  "FFI" terminology comes from. In `WamRuntime.hs` the `step` function
  handles `Call` via `wam_haskell_target.pl:621-629`, which is a chain
  with four branches:
  1. **`CallResolved`** — a pre-resolved PC jump. Built by
     `resolveCallInstrs` (`wam_haskell_target.pl:1308-1322`) at project
     load time. Only fires for predicates that have a label and are
     **not** in the `foreignPreds` list. This is the fast path — no
     string lookup, no FFI, just `wsPC := targetPC`.
  2. **`executeForeign`** — hand-written Haskell escape hatches.
     Pattern-matches on the predicate name and dispatches to a
     specialised Haskell helper. Currently the only predicate with an
     escape hatch is `category_ancestor/4`, which calls
     `nativeCategoryAncestor`. This is "the FFI."
  3. **`callIndexedFact2`** — generic indexed dispatch for 2-arg
     facts, backed by `wcForeignFacts`. Used for `category_parent/2`
     and any other 2-arg fact table the user wires up. This is *not*
     an FFI — it's an indexed lookup inside the WAM, and it's always
     beneficial when facts are populated.
  4. **`wcLabels` lookup** — fallback string lookup for any predicate
     that somehow slipped past `resolveCallInstrs`. In a well-formed
     project this branch is unreachable.
- **Why it is fast (today):** persistent data structures so
  backtracking is cheap (structural sharing, not copying), a hot/cold
  split of `WamState` into hot state + `WamContext`, pre-resolved
  `Call → CallResolved` so most calls do not hash strings,
  compile-time arity for `PutStructure`, cached length counters so
  cut and backtrack are O(1), and targeted FFI hot-spot escape
  hatches.
- **Why it is slow (today):** every instruction is dispatched through
  a `case instr of ...` at runtime in `step`. GHC cannot see through
  that dispatch to fuse adjacent instructions, inline predicate calls,
  or keep registers in unboxed lets. The interpreter pays per-
  instruction overhead forever.

### Path 3 — WAM-lowered Haskell functions (this design)

- **Source file:** `src/unifyweaver/targets/wam_haskell_lowered_emitter.pl`
  (new, does not exist yet).
- **Entry point:** `lower_predicate_to_haskell/4`, called from
  `write_wam_haskell_project/3` when `emit_mode(functions)` or
  `emit_mode(mixed(HotPreds))` is active.
- **What it does:** compiles Prolog → WAM IR → **one Haskell function
  per predicate**. The function body is straight-line Haskell that
  mirrors the WAM instruction sequence inline. No instruction array,
  no `step`/`run` interpreter loop for the lowered predicates.
- **What it preserves from path 2:** `WamState` and `WamContext` are
  still threaded through the lowered functions. Persistent data
  structures still hold the bindings, trail, choice points, and
  stack. The FFI/indexed-fact helpers are shared between paths —
  lowered code calls them via ordinary Haskell function calls.
  Backtracking still works across the path-2/path-3 boundary because
  choice points record interpreter PCs the same way.
- **Why it might be faster than path 2:** GHC sees the whole predicate
  as a single Haskell function, so it can fuse adjacent WAM
  instructions, track register values as unboxed lets, inline
  intra-predicate calls, and specialize the generated code the same
  way path 1 would have — but for predicates path 1 cannot reach.
- **Why it is a separate path, not a replacement for path 2:** path 2
  is the reference implementation — when a lowered predicate
  misbehaves, the first debugging question is "does the same input
  produce the same output through the interpreter?" Also, lowering
  has a generator-complexity and GHC-compile-time cost per predicate,
  so forcing every cold predicate through lowering is wasteful. See
  the philosophy doc §"Why this is a third path, not a replacement"
  for the full argument.

## What "hybrid WAM with/without FFI" means, precisely

This is the clarification that motivated writing this background doc.
"Hybrid WAM with FFI" and "hybrid WAM without FFI" are real and
meaningful, but narrower than a casual reading suggests. Specifically:

- "With FFI" means **`executeForeign` is pattern-matched for at least
  one predicate** and `wcForeignFacts`/`wcForeignConfig` are populated
  in `Main.hs` so the escape hatch can actually fire.
- "Without FFI" means `executeForeign` matches no predicates (or the
  foreign facts are not populated), so the `Call` dispatch chain
  skips that branch and falls through to indexed-fact dispatch or the
  label lookup.

`callIndexedFact2` is **not** part of this "with/without FFI" axis. It
is a separate runtime feature — generic indexed dispatch for 2-arg
facts — and it is always beneficial when facts are populated. Turning
it off would mean running `category_parent/2` through clause
enumeration, which is slower for no reason. Nobody turns it off in
practice.

So the current `wam_haskell_target.pl` output actually has **three
independently-toggleable runtime features at the `Call` site**, not
two:

1. `CallResolved` fast path (always on, built by `resolveCallInstrs`).
2. `executeForeign` escape hatch (on per-predicate, currently only
   `category_ancestor/4`).
3. `callIndexedFact2` indexed-fact dispatch (on when facts are
   populated).

All three are part of the path-2 generator output. The path-3 design
preserves access to (2) and (3) via shared Haskell helpers that
lowered code calls directly. (1) becomes redundant inside a lowered
predicate, because the "call target" is either a direct Haskell
function call (intra-lowered) or an `wsPC`-based re-entry into the
interpreter (lowered → interpreted).

## Why the philosophy doc calls path 3 "a JIT over path 2"

Because the mental model maps cleanly:

- Path 2's `step` function is the interpreter.
- Path 3's emitted per-predicate Haskell function is the specialized
  code the interpreter would have produced if it had inlined every
  dispatch for this specific instruction sequence ahead of time.
- Path 2 decides what to execute at runtime (one `case` per step);
  path 3 commits that decision at generator time.

The analogy breaks down in one important place: a real JIT discards
the interpreted version once specialization is complete. Path 3 does
**not** discard path 2. The interpreter stays as the reference
implementation, the cold-predicate fallback, and the "does the
lowered version produce the same answer" oracle.

## Code references (for future archaeology)

These are the specific line ranges you will want to read if you are
about to touch the lowering path. All paths relative to the repo
root.

| What you want to understand | Where |
|---|---|
| `Call` runtime dispatch chain | `src/unifyweaver/targets/wam_haskell_target.pl:621-629` |
| `executeForeign` entry | `src/unifyweaver/targets/wam_haskell_target.pl:482-524` |
| `callIndexedFact2` | `src/unifyweaver/targets/wam_haskell_target.pl:448-476` |
| `resolveCallInstrs` | `src/unifyweaver/targets/wam_haskell_target.pl:1308-1322` |
| `foreignPreds` list (where FFI escape hatches are named) | `src/unifyweaver/targets/wam_haskell_target.pl:1022` |
| `wcForeignFacts` population in `Main.hs` | `src/unifyweaver/targets/wam_haskell_target.pl:1032-1041` |
| `nativeCategoryAncestor` (the one current FFI target) | `src/unifyweaver/targets/wam_haskell_target.pl:411-433` |
| `native_haskell_clause_body/3` (path 1 entry) | `src/unifyweaver/targets/haskell_target.pl:72-102` |
| WAM instruction → Haskell (path 2 emission) | `src/unifyweaver/targets/wam_haskell_target.pl`, `wam_instr_to_haskell/2` |
| Interpreter `step` body (semantic reference for path 3) | `src/unifyweaver/targets/wam_haskell_target.pl`, the big `step` template string |

Line numbers are accurate as of the commit on `feat/wam-haskell-ffi-optimization`
at the time this background doc was written. If they have drifted,
search by symbol name.
