# WAM Term Builtins: Philosophy

## The question

Three hybrid WAM targets are now in the tree (Rust, Haskell, WAT). All three
share the same gap: none of them implements `functor/3`, `arg/3`, `=../2`,
`copy_term/2`, `assert/1`, or `retract/1`. Predicates that use these builtins
either fail to lower at all, or get pushed to the slowest fallback path
(host SWI-Prolog interop).

This branch is an **investigation**, not a foregone conclusion. The question
to answer is:

> Does adding these builtins improve UnifyWeaver in measurable ways — either
> by **expanding what can be transpiled** (correctness/reach) or by **letting
> more predicates run on the WAM fast path** (performance) — and if so, which
> builtins matter and in what order?

The reason this is worth asking carefully is that the existing WAM targets
have been productive *without* these builtins. The transitive closure /
recursion patterns UnifyWeaver targets are structural, not meta. They walk
arguments by clause-head pattern matching, not by calling `arg/3`. They build
results by unification, not by `=..`. So the gap has not been load-bearing
*so far*.

The hypothesis this document is built around: the gap **is** load-bearing
once you step outside the recursion-pattern niche, and a meaningful slice of
real Prolog code is currently un-transpilable for this reason alone.

## What the existing targets actually do

Both Rust and Haskell hybrid WAMs implement essentially the same builtin set:

- Cut (`!/0`) via cut-barrier truncation
- Arithmetic (`is/2`, `</2`, `>/2`, `=:=/2`, `=\=/2`, `=</2`, `>=/2`)
- Type checks (`var/1`, `atom/1`, `integer/1`, etc.)
- Negation as failure (`\+/1`)
- A handful of list operations (`length/2`, `member/2`)
- I/O (`write/1`, `nl/0`)

WAM-WAT matches this set. The four "term inspection" builtins
(`functor/3`, `arg/3`, `=../2`, `copy_term/2`) are absent everywhere. The two
"dynamic database" builtins (`assert/1`, `retract/1`) are also absent
everywhere, but for very different reasons (see "The two categories" below).

How does Rust achieve its impressive `category_ancestor` performance, then?
Not via term inspection — via **specialized WAM instructions**
(`BaseCategoryAncestor`, `RecurseCategoryAncestorPc`) that bypass the general
WAM machinery entirely. That tells us the path to perf wins on specific
bottlenecks is custom instructions, not generic introspection. Generic
introspection is for *transpiling reach*.

## The two categories

The six builtins the user named split cleanly into two groups:

### Group A: term inspection (pure functions on heap cells)

- `functor/3` — read or construct a compound's name and arity
- `arg/3` — random-access into a compound's arguments
- `=../2` — convert between `f(a,b,c)` and `[f,a,b,c]`
- `copy_term/2` — fresh-variable copy of a term

These are pure: they read (and possibly allocate fresh) heap cells. They do
not touch the dynamic database, the trail in interesting ways, or the clause
store. Their semantics are well-defined and identical across all Prolog
systems. The implementation is **mechanical** — walk the heap, allocate, bind.

This group is the subject of this plan.

### Group B: dynamic database (stateful, separate workstream)

- `assert/1`, `asserta/1`, `assertz/1`
- `retract/1`, `retractall/1`

These require something none of the existing targets have: a **runtime-extensible
clause store**. The current WAM targets compile every predicate to a fixed
data segment of WAM instructions, indexed by static labels. There is no path
for new clauses to enter the dispatch table at runtime.

Adding `assert`/`retract` would require:

1. A separate dynamic-predicate dispatch table (per-target)
2. Decisions about backtracking semantics (standard Prolog: asserts survive
   backtracking; retracts are undone — but only the *physical* removal, not
   the *logical* effect)
3. Re-indexing strategy when clauses are added
4. A test story for memoization/tabling patterns that motivate the feature

This is a substantially larger workstream than Group A, with its own
philosophy/spec/plan. **It is out of scope here** and tracked as a follow-up.

## Why Group A matters: the case for transpiling reach

Three classes of Prolog code currently fail to lower because of Group A:

1. **DSL interpreters and rule engines.** Anything that builds a goal at
   runtime via `Goal =.. [Pred|Args], call(Goal)` cannot be transpiled. This
   pattern is endemic in business-rule systems, configuration languages, and
   meta-interpreters.

2. **Generic walkers.** Predicates that operate on "any compound term"
   without knowing its functor in advance — pretty-printers, serializers,
   structural diff, generic equality variants — all need `functor/3` and
   `arg/3` to walk subterms.

3. **Tabling and memoization built on `copy_term`.** When user code
   implements its own memo table to avoid recomputation, it almost always
   uses `copy_term` to freeze the key without aliasing the live query
   variables. Without `copy_term`, the user-level memo silently shares state
   across iterations and breaks.

Each of these classes is a meaningful slice of real Prolog code. None of
them are exotic. A "Prolog-to-WASM transpiler" that cannot handle them is
selling itself short.

## Why Group A matters less for performance (probably)

The honest answer is: **probably not much**, with one possible exception.

The reasoning:

- The hot loops in UnifyWeaver's existing benchmarks (effective semantic
  distance, transitive closure, A* aggregates) don't use these builtins at
  all. Adding them won't speed those up.
- For predicates that *do* use them, the alternative today is "fall back to
  host SWI-Prolog over an FFI bridge". A native WAM implementation will
  almost certainly beat that bridge by a wide margin — but only on those
  predicates. The benchmark needle won't move on existing workloads.
- The exception is `copy_term/2` in tabling-heavy code. If a predicate
  currently emulates `copy_term` by re-running unification N times against
  fresh variables, a native O(N-heap-cells) `copy_term` is asymptotically
  better. But UnifyWeaver doesn't currently have such a workload to measure.

The conclusion is that the philosophy doc should set expectations honestly:
**this is a transpiling-reach feature with possible perf upside on specific
predicates, not a perf optimization**. Phase 6 of the implementation plan
is dedicated to validating or refuting that more rigorously.

## Why each builtin individually

### `functor/3`

The atomic operation. `arg/3` and `=../2` are both expressible in terms of
`functor` + heap walks. If only one of the four were to be implemented,
this would be it. In the WAM, it is essentially "read the tag and arity
fields of a heap cell" — O(1).

The construct mode (`functor(-T, +N, +A)`) is slightly more interesting:
it must allocate `A` fresh unbound cells on the heap and tag them as
arguments of a fresh compound. The implementation pattern already exists
in the heap allocator paths used by `put_structure`.

### `arg/3`

`arg(+N, +Term, ?A)` is `O(1)` heap access — read the Nth cell after the
compound header. Trivial once `functor/3` machinery exists.

The reason it gets its own builtin (rather than being expressed as `=..`)
is that `=..` allocates a list with one cons cell per argument, which is
wasteful for "I just need the third arg".

### `=../2` (univ)

Decompose mode is `functor/3` + `arg/3` in a loop, building a list. Compose
mode is `functor/3` build mode + `arg/3` write in a loop.

It's the most syntactically common of the four in user-facing code,
because `Goal =.. [Pred|Args], call(Goal)` is the canonical Prolog idiom
for runtime goal construction.

### `copy_term/2`

The hardest of the four. Requires:

- A traversal of the source term
- A temporary mapping (var-id → fresh-var-id) so that shared variables
  in the source remain shared in the copy (`copy_term(f(X,X), C)` must
  give `C = f(Y,Y)`, not `f(Y,Z)`)
- Heap allocation for every cell of the copy

Persistent-data-structure WAMs (Haskell) and Rc-snapshot WAMs (Rust)
both handle this cleanly because their underlying value representation
is already structural. WAT has neither — it has a flat linear-memory
heap. The `copy_term` helper for WAT will need an explicit work stack
(same pattern as `$unify_regs`) and a small temporary mapping table
allocated on the heap.

## Rejected alternatives

### "Implement them in user-level Prolog and let the WAM compile that"

Doesn't work. The four Group A builtins are mutually defining at the
runtime level — at least one of them must be primitive because they all
need to inspect heap-cell tags, which user-level Prolog cannot do directly.
And `copy_term` requires variable freshness, which only the runtime can
provide (the var counter is not exposed to user code).

### "Skip them and document the gap"

This is what the project does today. The cost is invisible (predicates
that would have used them simply aren't written, or get rewritten in awkward
ways) but real. The investigation question is whether that cost is high
enough to justify the implementation work.

### "Implement them only in WAM-WAT and not in Rust/Haskell"

Tempting (smallest blast radius) but creates a permanent capability
asymmetry between targets. Better to add them at the canonical WAM layer
(`is_builtin_pred/2` in `wam_target.pl`) and roll out to all three
backends, even if WAM-WAT goes first.

### "Add them as specialized WAM instructions, not as `builtin_call`"

This is how Rust handles `category_ancestor`. The performance argument is
that fixed-register builtin dispatch costs an extra indirection.

For these four, it doesn't matter — they're not on hot paths. The existing
`builtin_call` pattern is fine, and avoids growing the instruction set.
Reserve specialized instructions for cases where the perf cost is measured.

### "Add `assert`/`retract` too while we're here"

No. They are a fundamentally different feature (mutable clause store,
backtracking semantics, re-indexing, dynamic dispatch). Bundling them into
a "term builtins" PR would conflate two unrelated workstreams and make
the change much harder to review. They get their own future plan.

## Experiment-first stance

The implementation plan is structured to **fail fast** if the hypothesis is
wrong:

- **Phase 0** is a one-pass audit of the existing test corpus. If no test
  predicate uses these builtins, the transpiling-reach claim is weakened
  and we should reconsider.
- **Phase 2** implements in WAM-WAT only. If codegen is harder than expected
  or the runtime cost is surprising, we stop there and reconsider before
  porting to Rust/Haskell.
- **Phase 6** is an explicit perf comparison: pick one predicate that
  currently falls back to host SWI for using these builtins, re-benchmark
  with the WAM-only path, and write up the result honestly — even if it
  shows no improvement.

If Phase 6 says "no perf win, modest reach win", that is still a successful
outcome — it bounds the claim and tells future contributors what to expect.

## What "good enough" looks like

A successful term-builtins workstream means:

1. The four Group A builtins are implemented in WAM-WAT, Rust, and Haskell
2. The canonical `is_builtin_pred/2` in `wam_target.pl` lists them
3. There is at least one Prolog predicate per builtin that previously could
   not be transpiled and now can
4. Either: (a) a measured perf improvement on at least one predicate, or
   (b) a documented finding that no perf improvement was achievable on the
   existing benchmark suite, with the predicate-class examples that *would*
   benefit identified for future work
5. `assert`/`retract` are clearly tracked as separate follow-up work, with
   a stub design doc identifying the architectural questions

The success criterion is **honesty** about what was unlocked, not raw
benchmark numbers.
