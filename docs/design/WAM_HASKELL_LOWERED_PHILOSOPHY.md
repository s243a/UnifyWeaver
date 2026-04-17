# WAM-Lowered Haskell: Philosophy

> **Background.** For a descriptive walk through the *current* Haskell
> code-generation paths — including exact source line references for the
> `Call` dispatch chain, `executeForeign`, `callIndexedFact2`, and
> `resolveCallInstrs` — see
> [`WAM_HASKELL_LOWERED_BACKGROUND.md`](WAM_HASKELL_LOWERED_BACKGROUND.md).
> This philosophy doc assumes you already know what "hybrid WAM with/without
> FFI" means at the code level. The background doc is where that language is
> pinned down.

## Context

UnifyWeaver currently has two production Haskell code-generation paths and
is introducing a third:

1. **Direct native Haskell lowering** (`src/unifyweaver/targets/haskell_target.pl`,
   via `native_haskell_clause_body/3`). Emits straight-line Haskell functions
   directly from the Prolog AST. Handles predicates whose clauses can be
   statically reasoned about. No WAM at all.

2. **WAM-interpreted Haskell** (`src/unifyweaver/targets/wam_haskell_target.pl`,
   current state). Compiles Prolog → WAM IR → a Haskell `[Instruction]`
   array plus label map, then a `step`/`run` dispatch loop interprets the
   array at runtime. This is the path the `feat/wam-haskell-*` branches
   have been optimizing: persistent data structures for choice-point
   snapshots, `WamState`/`WamContext` hot/cold split, pre-resolved
   `CallResolved`, an FFI escape hatch (`executeForeign`) for predicates
   where hand-written Haskell beats the interpreter, and an indexed
   dispatch for 2-arg facts (`callIndexedFact2`).

3. **WAM-lowered Haskell functions** (this doc, new). Same WAM IR as path
   2, but instead of emitting an instruction array interpreted at runtime,
   the generator emits **one Haskell function per Prolog predicate**. The
   function body is straight-line Haskell that mirrors the WAM semantics
   for each instruction inline. No instruction array, no `step`/`run`
   interpreter loop. `WamState`/`WamContext` still thread through so
   backtracking and choice points work across predicates.

This document explains *why* the third path is worth doing, *why* it is
strictly additive to the first two, and the design principles the
generator must respect.

## The gap between paths 1 and 2

Native Haskell lowering (path 1) wins by emitting code GHC can optimize as
if a human wrote it — let-bindings, strict pattern matches, no boxed
intermediate state. But it only fires for predicates whose clauses the
lowering analysis can handle. Anything with complex backtracking, cut
interactions, meta-call, or aggregation semantics falls through to the
WAM path.

WAM-interpreted Haskell (path 2) wins by being a faithful, predictable
execution model that handles every Prolog feature UnifyWeaver's WAM IR
supports. But the per-instruction `step` dispatch costs something even
after the `WamContext` split: GHC can't see through a `case instr of ...`
into the specific instruction each time, so it can't fuse adjacent
instructions, can't keep registers in unboxed lets, and can't inline
predicate calls.

The two paths have **different reasons for being fast** and **different
reasons for being slow**. Path 1 is fast per-instruction but narrow in
coverage. Path 2 is wide in coverage but pays dispatch overhead on every
instruction, on every step, forever.

Path 3 is the attempt to get **path 2's coverage with path 1's inner
loop**. The WAM compiler already knows how to produce correct WAM for
every predicate UnifyWeaver can handle. That WAM is a specialized little
straight-line program for each predicate — it just happens to be encoded
as an array of `Instruction` values that a runtime interpreter walks.
Emitting the same semantics as a Haskell function body with explicit
`case` on tags, explicit register binds, and explicit control flow lets
GHC see *the whole predicate at once*.

## Why this is a third path, not a replacement

The temptation when you describe path 3 is to say "so just delete path
2." That is wrong for several reasons, and stating them upfront prevents
an entire category of scope creep.

**1. Path 2 stays as the reference implementation.** The interpreter loop
in `WamRuntime.hs` is where the WAM semantics are *defined*. When a
lowered predicate misbehaves, the first debugging question is "does the
same input produce the same output when run through the interpreter?"
Deleting path 2 destroys the ground-truth oracle.

**2. Lowering has a cost per predicate.** Path 2 compiles one generator
output per project (an instruction array and a dispatch loop that works
for every predicate). Path 3 has to emit specialized code for each
predicate, which costs generator complexity *per predicate* and GHC
compile time *per predicate*. For a project with 200 predicates and 20
hot ones, forcing everything through path 3 is more total work than
letting the cold predicates stay interpreted.

**3. Some predicates won't lower cleanly.** The WAM has features the
lowered functions will have to handle — nested cut, aggregation frames
(`AggPush`/`AggPop`), indexed-fact dispatch via `wcForeignFacts`, FFI
escape hatches via `executeForeign`. The lowered emitter will probably
handle most of these, but there will be edge cases where interpreting is
simpler and safer than emitting. Having the interpreter as a fallback
means "this predicate didn't lower cleanly, fall back to the interpreter"
is a valid compile-time decision.

**4. It lets us measure the difference.** Having paths 2 and 3 both
available, with a per-project or per-predicate selector, is how we will
know path 3 is actually a win. If path 3 is slower than path 2 on some
workload we didn't predict, we need to be able to run the same
predicates through path 2 on the same data without rewriting the
generator. That requires the interpreter to stay.

The mental model is: **path 3 is a JIT over path 2**. Path 2 knows how
to run every predicate. Path 3 is "pre-specialize the interpreter's
dispatch loop for this specific predicate's instruction sequence." The
two are complementary: the generator picks which predicates to specialize
based on expected benefit (mostly: the ones on the hot path), and leaves
the rest interpreted.

## The selector, and why it defaults to interpreted

Path 3 will ship behind a selector. The user (or a calling program)
decides which path to take per project *and* optionally per predicate.

The selector hierarchy the generator checks, in order:

1. **Per-call option on `write_wam_haskell_project/3`:**
   `emit_mode(interpreter)`, `emit_mode(functions)`, or
   `emit_mode(mixed(HotPreds))` where `HotPreds` is a list of predicate
   indicators to lower.
2. **User-defined Prolog predicate:** a dynamic fact `user:wam_haskell_emit_mode/1`
   with the same allowed values. Lets a project assert its preferred
   default at load time without threading an option through every call site.
3. **Generator default:** `interpreter`.

The generator default is **`interpreter` (path 2)**, not `functions`,
for three reasons:

1. **Path 2 is the proven path.** Every benchmark, every correctness
   test, every regression run currently in `examples/benchmark/` and
   `tests/` targets the interpreter. The first shipping version of
   path 3 will *not* have comparable test coverage. Making path 3 the
   default before its test coverage catches up means every user of
   `write_wam_haskell_project/3` becomes an unwitting beta tester.

2. **Path 3 has new failure modes.** The generator emits Haskell, so a
   generator bug now manifests as a GHC compile error in the user's
   project, not a runtime error in a well-tested interpreter loop. That
   is a worse failure mode for users who are not working on the
   generator itself.

3. **Opt-in lets us measure honestly.** We want the answer to "is path
   3 faster than path 2 on workload X" to be a clean A/B comparison
   with no confounders. That is easier to set up when path 3 is
   something the benchmark explicitly requests, not something that
   happens silently when the generator feels like it.

The default should flip to `functions` only when:
- Path 3 passes every correctness test path 2 passes, on every
  benchmark in `examples/benchmark/` relevant to the Haskell target.
- Path 3 is faster than path 2 on the `effective_distance` 10k
  benchmark by at least 1.5× median total_ms.
- At least one externally-reported workload has used path 3 in
  `mixed(HotPreds)` mode without filing a correctness bug.

Until then, users who want path 3 ask for it explicitly.

## What path 3 must preserve from path 2

The WAM-lowered emitter is allowed to be a different code generator, but
it is **not** allowed to be a different execution model. Specifically:

- **Persistent data structures stay.** `wsBindings`, `wsTrail`, `wsCPs`,
  `wsStack` remain `IntMap`/list of persistent snapshots. A lowered
  predicate allocates new `WamState` records exactly when the interpreter
  would — the win comes from GHC seeing the updates inline, not from
  switching to mutation. The `IORef`/`STRef` rejection in
  `WAM_HASKELL_PERF_PHILOSOPHY.md` applies with equal force here.
- **Choice points snapshot the same fields.** A lowered `TryMeElse`
  still builds a `ChoicePoint` with the same bindings/trail/heap
  lengths. The emitter may inline the construction, but the shape of
  the snapshot must match what the interpreter would have built, so
  that a backtrack crossing a path-2→path-3 boundary is safe.
- **`WamContext` stays read-only.** Lowered predicates take `WamContext`
  as an argument and never modify it. The hot/cold split the interpreter
  relies on holds for lowered code too.
- **FFI hooks (`executeForeign`, `callIndexedFact2`) still apply.** A
  lowered predicate encountering what would have been `Call "cat_anc/4"`
  in the interpreter emits a direct Haskell call into the same
  `executeForeign` / `callIndexedFact2` helpers. The lowered code and
  the interpreter share those helpers verbatim.
- **Mixed mode must round-trip.** A lowered predicate calling an
  interpreted predicate (or vice versa) must produce the same final
  `WamState` as if both were interpreted. This is the one correctness
  invariant that makes `mixed(HotPreds)` viable. If it doesn't hold,
  the only shipping configurations are "all interpreted" or "all
  lowered," and that halves the value of the path.

## What path 3 is *not* trying to do

- **Not trying to beat native lowering (path 1)** on predicates that
  path 1 already handles. Path 1 wins by skipping the WAM entirely and
  handing GHC a clean functional program. Path 3 is a specialization of
  WAM semantics, so it will always carry slightly more baggage than
  path 1 (unboxed register tracking, CP machinery, explicit trail/heap
  state). If path 1 can lower a predicate, let it.
- **Not trying to JIT.** GHC does the optimization at ahead-of-time
  compile time. The emitter's job is to produce Haskell GHC will compile
  well, not to produce machine code or use GHC's internal APIs.
- **Not trying to replace the WAM IR.** The WAM IR is the input to this
  generator, and it is the same IR the other WAM-based targets
  (`wam_rust_target.pl`, `wam_c_target.pl`, etc.) consume. Changes to
  the IR have to justify themselves at that level, not inside this
  emitter.
- **Not trying to unify with path 1.** The two have different input
  shapes (Prolog AST vs WAM IR), different intermediate analyses, and
  different failure modes. They stay distinct.

## Summary

| Idea | Status | Reason |
|---|---|---|
| Three-path taxonomy (native / interpreted / lowered) | **New** | Each path wins for a different reason |
| Interpreter stays as reference implementation | **Required** | Ground-truth oracle, cold-predicate fallback |
| Default is `emit_mode(interpreter)` | **Initial** | Proven path, better failure modes, clean A/B |
| Lowered code shares persistent data structures | **Required** | Same backtracking model as interpreter |
| Lowered code shares FFI/indexed-facts helpers | **Required** | One set of escape hatches |
| Mixed mode (`mixed(HotPreds)`) | **Required** | Lets users specialize only what matters |
| Flip default to `functions` eventually | **Conditional** | Only after correctness + ≥1.5× speedup + external validation |
| Replace the interpreter | **Rejected** | Destroys the oracle and the fallback |
| Replace native lowering | **Rejected** | Different input, different wins |
| IORef/mutation in lowered code | **Rejected** | Same reason as in the interpreter |
