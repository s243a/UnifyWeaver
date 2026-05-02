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

## The senses of *kernel* — and why they converge here

The word "kernel" carries several technical meanings across
mathematics and computing, and a number of them line up uncannily
well with what the detector is doing. Each lens illuminates a
different facet of why this design works, so it is worth walking
through them.

**The seed-kernel.** Etymologically, "kernel" is Old English
*cyrnel*, the diminutive of *corn* — the inner seed of a nut or
fruit. The metaphor distinguishes the dense, hard, valuable inner
part from the variable surrounding shell. Different shells (apple,
peach, walnut) wrap around structurally similar seeds. The shell is
the part that varies between fruits; the kernel is the carrier of
essential structure. In our case `category_ancestor` is the seed,
and `effective_distance`, `category_influence`, and any future
workload that traverses the category graph are different shells.
The shell varies; the kernel is invariant.

**The compute-kernel (BLAS, CUDA, OS).** In numerical computing,
BLAS kernels are tight inner loops everything calls into — `dgemm`
for dense matrix multiply, `dgemv` for matrix-vector. CUDA kernels
are functions executed in parallel across thousands of threads. An
OS kernel is the trusted core that every userspace program
delegates to for primitive operations. The common thread is
*inversion of control*: the kernel doesn't know about its callers;
callers know how to invoke it. The kernel is a fixed point in the
call graph, with everything around it variable. Our
`category_ancestor` shares this property — once detected and
lowered, the implementation is fixed across every workload that
reaches it.

**The algebraic kernel.** In abstract algebra, the kernel of a
homomorphism `f: G → H` is the set of elements in `G` that map to
the identity in `H` — the things that get "factored out." The
First Isomorphism Theorem says `G/ker(f) ≅ image(f)`: once you
quotient by the kernel, what remains is exactly what the map can
see. For our purposes the WAM dispatcher is the homomorphism; the
kernel is what the dispatcher does *not* need to see because it
has been lifted out into native code; the quotient is the outer
workload logic the WAM still interprets. The intuition that
"factors have zeros" lands here directly — the kernel is what the
structural map sends to identity, what becomes invisible to it,
freeing the rest to focus on what varies. (For the logic-
programming substrate this is sharper than analogy; see "Why this
lands so well on logic programming" below.)

**The integral-transform kernel.** In `∫ K(x, y) f(y) dy`, the
kernel function `K(x, y)` defines the transform — Fourier,
Laplace, Hilbert. Same operator (the integral), different kernels,
different transforms. By analogy, the WAM dispatcher is the
operator; different kernels installed into it give us different
specialized runtimes for different workload families. A workload
heavy on transitive closure gets one effective runtime; a workload
heavy on A\* gets another; the dispatcher is the same.

**The kernel-method (ML) sense.** In machine learning, the kernel
trick computes inner products in a transformed feature space
*without materializing the transformed space*. You don't need to
construct `φ(x)` explicitly — you just need a kernel function
`K(x, y) = ⟨φ(x), φ(y)⟩`. Our system has the same flavour: we
don't need to materialize a full native re-implementation of
`effective_distance` to make it fast. We just need the kernel
that sits inside it. The composition above the kernel can stay
in (relatively) slow WAM bytecode because its instruction count
is small enough that the dispatch cost is bounded. The kernel is
the *only* place where native code is required.

What unites all these senses is a single structural move: **find
the invariant under variation, lift it out, express the rest as a
quotient.** The compute-kernel does this in the call graph. The
algebraic kernel does it in the homomorphism. The integral
transform does it across families of operators. The kernel method
does it between feature spaces. The pattern-matched kernel here
does it across Prolog workloads.

This convergence is not coincidence. Optimization, almost by
definition, is the search for the right factorization: what can
be hoisted, what can be shared, what is genuinely workload-
specific. "Kernel" is the word that several traditions
independently arrived at for naming the hoisted part. That all of
those traditions apply simultaneously to what
`recursive_kernel_detection.pl` is doing isn't a happy linguistic
coincidence — it is evidence that the detector is identifying the
right structural object. When several independent disciplines
borrow the same name for "the thing you can factor out," and that
name fits a fifth domain without strain, the underlying notion is
probably real.

## Why this lands so well on logic programming

The convergence above is at the level of analogy across domains.
For our specific substrate — Prolog and the broader logic-
programming family it represents — the convergence collapses into
*identity*. The algebraic-kernel reading isn't a metaphor we
reach for; it is the literal mechanism by which the optimizer
operates, because the language itself is an algebraic object.

A Prolog program is a set of Horn clauses. Each clause is a
conjunction of literals; each predicate is the disjunction of its
defining clauses. The propositional fragment of the program is
exactly a Boolean expression in (almost) disjunctive normal form,
extended to first-order via existentially-quantified body
variables. This means the equational toolkit of Boolean algebra
applies *directly*, not analogically:

- **Distributivity** — `(A ∧ B) ∨ (A ∧ C) = A ∧ (B ∨ C)`. When
  `category_ancestor(...)` appears as a subgoal in
  `effective_distance(...)`, `category_influence(...)`, and any
  future workload, lifting it into the kernel registry is the
  application of distributivity at the program-structure level.
  The "common factor" is literal, not metaphorical.
- **Idempotence** — `A ∧ A = A`. Common subgoal elimination
  within a clause body.
- **Absorption / subsumption** — `A ∨ (A ∧ B) = A`. Clause
  subsumption: if one clause's head is implied by another's,
  the redundant clause can be dropped.
- **De Morgan's laws** — apply to negation-as-failure transforms,
  including the `\+ member(_, _)` patterns the kernel detector
  matches on.

These are not analogies the optimizer reaches for. They are the
identities that govern Horn-clause manipulation. What is true at
the meta-level *is* true at the program level, because the
program *is* the algebraic object.

The kernel-of-homomorphism reading collapses cleanly too. Boolean
algebras are commutative rings (under XOR + AND), so the First
Isomorphism Theorem applies in its full ring-theoretic strength.
The kernel of a Boolean-algebra homomorphism is *exactly* the
elements mapping to ⊥ — bottom, the algebraic zero. The intuition
that factors have zeros isn't a lay-reader's mnemonic; it's a
literal statement of the relationship between what the WAM
dispatcher sees and what the kernel registry stores. The kernel
is what the dispatcher's structural map sends to identity, which
is exactly what we factor out into native code.

### Comparison with other substrates

Other paradigms don't have this clean alignment:

- **Imperative languages** (C, Go, Rust) carry implicit state,
  side effects, and aliasing. An optimizer has to prove an
  enormous amount of "this rewrite is safe under all possible
  aliasing scenarios" before it can apply even the simplest
  Boolean-style identity. Most of what conventional compilers do
  is loophole-closing around these conservatism barriers — escape
  analysis, points-to analysis, alias-set computation. The
  algebra is buried under operational concerns.
- **Functional languages** (Haskell, ML) get partway there.
  Referential transparency removes most of the aliasing
  conservatism, and lambda calculus has its own equational theory
  (alpha/beta/eta conversion, functor laws, monad laws). But the
  substrate isn't propositional Boolean algebra — it is a
  calculus of functions — so Boolean-factoring identities don't
  apply directly. You get *related* optimizations (let-floating,
  common subexpression elimination on values) — not the same
  ones.
- **Logic-programming languages** are the case where the
  substrate *is* propositional. Pure Prolog, Datalog, Mercury,
  Curry, λProlog, Answer Set Programming — all share this
  property to varying degrees. The equational toolkit of Boolean
  algebra applies because the program shape *is* a Boolean
  expression. Of these, Prolog is the practically-used member:
  the language with the working ecosystem, the bench corpus, the
  toolchain. The argument here generalises to the family; the
  reason it matters is that Prolog is where the engineering
  actually exists.

### The lineage

The optimization techniques that fall out of this substrate
choice have a direct lineage in digital logic and database
theory:

- **Espresso, BDDs, and multi-level logic minimization** are
  Boolean-factoring algorithms applied to gate-level circuits.
  Common subexpression elimination at the predicate level is the
  same operation applied one rung up the abstraction ladder.
- **SAT and SMT solvers** operate on Boolean formulas; a Horn-
  clause program is a Boolean formula, and entailment in pure
  Prolog is a SAT-style decision problem on the program's
  ground instantiation.
- **Datalog and relational algebra** are the database-theoretic
  cousins: relational query optimization (magic-set
  transformation, semi-naive evaluation, indexed-join
  selection) is essentially the same factoring problem applied
  to bottom-up evaluation of bounded Horn programs.
- **Theorem provers and proof assistants** (Coq, Lean) operate on
  even richer logical substrates, but the pure-propositional core
  shares the algebraic identity.

UnifyWeaver's `purity_certificate` machinery (referenced in
`docs/design/WAM_TIERED_LOWERING.md`) is essentially an algebraic
classifier — it identifies which predicates live in the pure
Boolean-algebra fragment (no cut, no I/O, no `assert`/`retract`)
and are therefore safe to manipulate using full algebraic
identities. Predicates that fail the certificate fall back to
conservative imperative-style compilation, exactly because the
algebra no longer holds. The certificate is the runtime evidence
that the substrate-alignment argument applies.

### What this means in practice

We aren't grafting a Boolean-factoring optimizer onto a language
that has to fight its substrate to allow it. We are using the
language's native algebraic structure to do what that structure
was always going to allow. The kernel detector is specialised
Boolean factoring; the kernel registry is the factor-out target;
the FFI dispatch is the factored call site. What looks from the
outside like a clever pattern-matching trick is, underneath, the
same thing Espresso does to a gate-level netlist — applied at the
predicate level instead of the gate level. The 52× Phase K
speedup isn't just a kernel substitution; it is what happens when
Boolean factoring on program structure meets a runtime that knows
how to take advantage of the resulting quotient.

This is also why kernel parity across targets compounds so well.
Each target's runtime has to handle the same factored quotient —
the outer workload logic the WAM still interprets. The kernel
implementations vary per target (Go, Rust, Haskell, Clojure all
need their own native version of `category_ancestor`), but the
*factorization* is invariant — it lives in the shared detector,
not in any one target. One algebraic transformation, N runtimes.
That is the leverage.

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
