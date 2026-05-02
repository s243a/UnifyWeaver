# Kernel Shape Recognition: Why Pattern-Matched Sub-Logic Beats Whole-Workload Optimization

## The observation

`effective_distance/3` is the high-level workload — given an article
and a root category, compute a Minkowski-power aggregated distance
over all weighted paths through the category graph. It composes a
per-seed weight sum, an article-fanout aggregation, and a final
exponentiation. None of those individually is a "primitive."

Yet the 52× speedup at scale-300
(`docs/design/WAM_PERF_OPTIMIZATION_LOG.md` Phase K) didn't come from
optimizing `effective_distance` as a whole. It came from recognizing
`category_ancestor/4` — a depth-bounded ancestor walk with a visited
set — as a sub-shape *inside* the larger structure, and replacing
that with native Go code. The composition above (aggregate over
kernel-emitted hop counts) stayed in WAM bytecode. The kernel didn't
need to know what workload it was inside; the workload didn't need to
know what kernel it was using.

This is the central design move of the WAM target's FFI-kernel
system: pattern matching on **sub-logic inside larger structures**,
not workload-level templates.

## "High-level" and "primitive" aren't opposites

Three useful parallels from neighbouring fields:

- **BLAS** doesn't recognize "linear regression." It recognizes
  matrix-matrix multiply — a sub-shape that linear regression,
  neural-net inference, image kernels, and PDE solvers all contain.
  Recognition is at the algebraic shape, not the application name.
- **SQL query optimizers** don't optimize "Q1 sales report." They
  recognize hash join, merge join, index seek — operational
  sub-shapes that every report query composes. The rewriter chain
  matches subtrees, not whole queries.
- **LLVM** recognizes induction-variable patterns, not "this is
  matrix multiply." Higher-level intent comes from libraries; the
  compiler stays at the structural level.

`category_ancestor` belongs to the same category as "matmul" or
"hash join": a domain-named version of an underlying recurring code
shape — depth-bounded graph traversal with a visited set. The
bench's name is bookkeeping; what's primitive is the shape.

## A note on the word *kernel*

The "kernel" in our context inherits from the BLAS / GPU /
compute-kernel tradition: a tight inner-loop primitive that larger
workloads call many times.

There's a satisfying second reading though. In algebra, the *kernel*
of a homomorphism is what gets factored out — the elements that map
to identity. By analogy, when we identify a kernel shape in a Prolog
predicate, we are identifying what *can be factored out* of the WAM
bytecode stream. What remains is the quotient: the workload-specific
composition logic above the primitive. Both senses converge — a
kernel is what the compiler can lift out of the dispatch path
because the rest is structural overhead around it.

## When a shape becomes worth a kernel

Three properties:

1. **Prolog-idiom verbosity in the WAM.** A single ancestor step
   expands to dozens of WAM instructions: `Allocate`, `PutVar`,
   `Call`, `\+ member` over the visited list, `is _ + 1`,
   `Deallocate`. At depth 10 with branching parent edges, this is
   thousands of dispatched instructions per query.
2. **Native-algorithm density.** The same step in native code is a
   handful of lines: a map lookup, a slice append, a recursive call.
   The cost ratio is two orders of magnitude.
3. **Clean boundary.** Inputs (`Cat`, `Root`, `Visited`, `MaxDepth`)
   and outputs (one `int64` per matching path) are knowable at the
   shape level — no escape hatches into the surrounding workload.

A shape with all three pays back its implementation cost across
every workload that contains it. A shape missing any of the three
doesn't.

## Why kernels compose where whole-workload templates don't

The `effective_distance` workload looks roughly like:

```prolog
effective_distance(A, R, D) :-
    aggregate(weight, S^(category_ancestor(A, R, _, S)), TotalW),
    D is TotalW ** (1 / Dimensionality).
```

A whole-workload template would have to recognize this exact
composition. A kernel system recognizes `category_ancestor` and
lets the aggregate-and-exponentiate stay in WAM bytecode. The kernel
returns hop counts; the WAM composes them. Both halves stay simple,
and any other workload that happens to compose `category_ancestor`
differently (`category_influence`, ad-hoc category queries) gets the
same speedup for free.

The WAM bytecode that *does* run is the small outer loop where
dispatch overhead is fixed and tractable. Phases B–H of the
performance log made *that* overhead cheap. Phase K removed the
inner loop's overhead entirely. Both layers were necessary.

## What the detector actually matches

`detect_category_ancestor/4` at
`src/unifyweaver/core/recursive_kernel_detection.pl:272` matches on
structural tells, not on names:

- At least two clauses (base + recursive).
- A base clause whose body contains `\+ member(_, _)` and a binary
  call sharing its first argument with the head — that binary call
  is the edge predicate.
- A recursive clause containing `\+ member(_, _)` and an `is _ + 1`
  accumulator (the hop counter).
- A `user:max_depth/1` fact in scope (the recursion bound).

A predicate named `find_ancestor_within/4` with the same shape would
match the same kernel. A predicate named `category_ancestor/4`
written differently — say, with the visited-set encoded as a sorted
tree, or the accumulator on a different argument — would not. The
detector matches *shape*, not *vocabulary*.

This is what "pattern matching sees sub-logic inside a larger
structure" means concretely: the detector ignores the surrounding
`effective_distance` framing entirely and looks only at the
recursion-shape signature of the inner predicate.

## The optimization ladder

| Layer | What it removes | Phase example |
|---|---|---|
| Constants | Per-instruction overhead | Phases B–H — env trimming, MaxYReg, Bindings as slice. Each WAM op got cheaper. |
| Structural runtime | Whole instruction *categories* | Phase D save-A+Y-skip-X — the X-reg snapshot disappeared. |
| Kernel substitution | Whole instruction *streams* | Phase K — the inner ancestor walk stopped going through WAM at all. |

Each layer multiplies with the layer below. The ~730× cumulative at
Phase K isn't `52 × 12.8`; it's the product *because* the prior
12.8× made the outer dispatch (still in WAM) cheap enough that the
kernel boundary became the dominant remaining cost — and then the
kernel substitution removed that too.

## Tradeoffs

- **Maintenance burden scales with target count.** Each new kernel
  needs an implementation in every target's runtime (Go, Rust,
  Haskell, Clojure, …). The shared registry at
  `recursive_kernel_detection.pl:81-87` is what makes this
  tractable: one detector, N back-ends, metadata-driven dispatch
  glue.
- **Pattern matching has false negatives.** A predicate that is
  *semantically* equivalent to `category_ancestor` but structurally
  different (alternate visited-set encoding, accumulator on a
  different argument) won't match. The detector is a syntactic
  filter, not a semantic one. This is the cost of staying in
  decidable territory.
- **Rare shapes don't pay.** A shape appearing in one workload only
  pays for itself if that workload is hot. The cross-target survey
  behind Phase K — frequency-weighted by benchmark references — was
  the right way to prioritize.
- **Kernel boundaries must stay clean.** A kernel that needs to call
  back into Prolog mid-execution (e.g., for a user-defined predicate
  inside its loop) loses most of the benefit. The cost asymmetry
  only holds when the inner loop is fully native.

## Practical takeaways

For anyone building a similar layered runtime:

1. Profile to find the *shape* of the dispatch cost, not just the
   percentages. "20% in the dispatcher" can mean "the dispatcher is
   slow" (constant-factor target) or "we're doing too much
   dispatch" (kernel target). The right answer changes the next
   move by orders of magnitude.
2. Look for sub-shapes inside workloads, not at the workload
   itself. The right primitive boundary is rarely the user's
   outermost goal.
3. Make detectors syntactic and registries shared. Per-target
   ad-hoc detection drifts; one detector with N back-ends scales.
4. Remember the ladder. Constant-factor work isn't wasted when you
   later land a kernel — it is the prerequisite that makes the
   kernel boundary cleanly dominate.

---

## References

- `src/unifyweaver/core/recursive_kernel_detection.pl` — shared
  kernel detectors and the metadata registry that all targets
  consume.
- `src/unifyweaver/targets/wam_rust_target.pl:1797` — Rust reference
  implementation of `compile_collect_native_category_ancestor_to_rust`.
- `src/unifyweaver/targets/wam_go_target.pl` — Go target's kernel
  dispatch (`go_supported_shared_kernel/1` and the
  `executeForeignPredicate` switch).
- `docs/design/RECURSIVE_TEMPLATE_LOWERING.md` — the AST-recursive
  template-then-lower architecture this builds on.
- `docs/design/WAM_TIERED_LOWERING.md` — the routing / strategy-menu
  framing that places kernel substitution within the broader tiered
  system.
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` Phase K — the worked
  example: 52× speedup at scale-300 from one new kernel.
