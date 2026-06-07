# Recurrence Evaluation Strategy — Philosophy

## What this is

A target-agnostic compiler module that, given a detected recursive kernel and the workload signals available at compile time, picks the **evaluation strategy** the target should use to compute the kernel's result. *Evaluation strategy* names a point in a two-level decision tree:

- **Level 1 — strategy class**: how the result is computed in the large.
  - `per_query` — evaluate the recurrence lazily, one query at a time, on demand. Implemented by graph search (uni/bi/A*/Dijkstra).
  - `fixed_point` — evaluate the recurrence eagerly, materialising the answer relation in one pass. Implemented by Tarski-style iteration (semi-naive, naive; top-down or bottom-up sub-choices).
  - `cached` — precompute the answer once, serve queries from a lookup table.
  - `hybrid` — combine the above (e.g. graph-search seed + iterative refine; magic-set transformation + bottom-up iteration).

- **Level 2 — algorithm within the class.** Only meaningful within a class:
  - For `per_query`: unidirectional BFS/DFS, bidirectional A*, Dijkstra, A* with admissible heuristic.
  - For `fixed_point`: semi-naive vs naive; top-down vs bottom-up.
  - For `cached`: full materialisation vs partial vs lazy-with-eviction.

The module *only* picks the strategy. The target reads the chosen strategy and emits code accordingly. The strategy decision is target-agnostic; the code emission is target-specific.

## Why now

[`book-18-graph-algorithms`](../../education/book-18-graph-algorithms/) chapter 10 (*Pattern detection by rule*) names the missing piece concretely:

> The most concrete next step: have the compiler *infer* that an ancestor query with appropriate cost-model hints should pick the bidirectional template, *without* the user saying so. This requires: a pattern matcher that recognises ancestor-style left-recursive transitive-closure rules; a cost-model rule that prefers bidirectional search when the query is a single-pair lookup and a precomputed minimum-distance map is available; a strategy selector that combines the recognition and the cost model into a code-generation choice.

The pattern matcher already exists ([`recursive_kernel_detection.pl`](../../src/unifyweaver/core/recursive_kernel_detection.pl) — 7 working detectors). The `bidirectional_ancestor` template, native-call spec, and register layout are all plumbed end-to-end. The `maybe_upgrade_bidirectional/2` upgrade path exists in [`wam_fsharp_target.pl`](../../src/unifyweaver/targets/wam_fsharp_target.pl). What's missing is **the cost-model rule and the strategy selector** that decides when to invoke the upgrade. Today, the upgrade is option-driven (the user passes `kernel_mode(bidirectional)` explicitly). The forward direction is for the compiler to make the choice from workload hints.

## The core insight: recurrence vs strategy

A user's recursive predicate defines a relation via a **recurrence** — equivalently a difference equation, with the loop-breaking mechanism baked in (either combinatorially via `\+ member` or numerically via contraction). The recurrence *is* the specification of what to compute.

The compiler's job is to pick an **evaluation strategy** for that recurrence. Different strategies compute the same recurrence in different ways:

- **Lazy evaluation** = top-down query-driven walk through the recurrence. Implemented as graph search.
- **Eager evaluation** = bottom-up materialisation of the recurrence's least fixed point. Implemented as Tarski-style iteration.
- **Hybrid** = lazy outer, eager inner (or vice versa). Magic-set transformation, demand-driven Datalog, seed-and-refine.

These are *not different computations*. They are different evaluation orders for the same recurrence. Datalog made this point sixty years ago in distinguishing top-down SLD from bottom-up semi-naive — same program, different evaluation strategy, same answer.

The implication for compiler design: pick the strategy *separately* from defining the recurrence. The user writes the recurrence (Prolog clauses); the compiler picks the strategy (via cost-model rule + optional user hints). The two concerns separate cleanly.

### The orthogonality is not complete

A subtlety worth naming. Strategy classes are **structurally** orthogonal — you can imagine them as independent axes — but **convergence properties** of the recurrence cross-cut the classes:

- A recurrence with **combinatorial loop-breaking** (explicit visited set) terminates by construction. Any strategy that doesn't drop the visited tracking will terminate.
- A recurrence with **numeric loop-breaking** (contraction via diagonal dominance) terminates *iff* the operator is a contraction with rate strictly less than 1. The contraction rate determines iteration count under iterative strategies; it determines admissibility under direct-solve strategies; it bounds the residual under per-query strategies.

So the strategy selector needs to know about the recurrence's loop-breaking mechanism — combinatorial or numeric — because the choice constrains which strategies are admissible. This is the *one* place where strategy class and recurrence properties interact.

## Theory connection: `r` is the diagonal-dominance condition

The convergence ratio `r = b'/(b_eff · D)` from [`TREE_LIKENESS_INDEX.md`](TREE_LIKENESS_INDEX.md) §2 and [`TREE_LIKENESS_INDEX_THEORY.md`](TREE_LIKENESS_INDEX_THEORY.md) §2.3 is *exactly* the diagonal-dominance / contraction condition for the `d_wPow` recurrence. Theorem 2.3's bound `r/(1−r)` on the per-step contribution from longer paths is the geometric-series bound from a contraction with rate `r`. The "shortcuts are rare" empirical observation (design note §3.3) and the "iteration converges fast" theoretical observation are the same observation, viewed through the two lenses of metric structure and numeric contraction.

This is why the strategy selector takes the recurrence's contraction rate as input even when the chosen strategy is per-query. The contraction rate is the *property of the recurrence* that bounds the per-iteration residual for any iterative method — including lazy on-demand iteration like A* search exploring partial paths. The selector uses it as a signal for "how aggressive can the budget cap be?" and "how cheap will fixed-point iteration be when it lands?"

For a deeper treatment, see [book-18-graph-algorithms appendix B.7](../../education/book-18-graph-algorithms/13_appendix_b_internal_theory.md) (convergence robustness — feature and trap).

## Signals: intent versus data

Workload signals come from several sources at compile time. The selector must distinguish two kinds:

- **Data signals** describe properties of the world the cost model reasons over. Examples: graph statistics (`b_eff`, `D`, `r`), CSR availability, relation-policy declarations (`cardinality(...)`, `determinism(...)`), query-shape inference (single-pair vs all-pairs).
- **Intent signals** express the user's required outcome. Examples: caller's `kernel_mode(bidirectional)` option, manifest's `strategy(...)` entries.

Most "conflicts" between signals dissolve when this distinction is applied. A manifest hint `cardinality(small)` and a caller option `kernel_mode(bidirectional)` are not in conflict — the cardinality is data feeding the cost model (which would have preferred unidirectional), while the kernel-mode is intent overriding the cost model's preference. The cost model still consumed the cardinality; it just didn't get the final word.

Only when *two intent signals* point at incompatible outcomes is there a real conflict — and even then the conflict-avoidance hierarchy (below) tries several non-trivial resolutions before treating caller-wins as the fallback.

## Conflict-avoidance hierarchy

When intent signals appear to conflict, the selector walks a hierarchy:

1. **Classify signals.** Data versus intent. Most apparent conflicts dissolve here.

2. **Look for a compatible third option.** If intent A points at strategy X and intent B points at strategy Y, but there exists a strategy Z that satisfies both intents, pick Z. Concrete example: caller's implicit "fast" + manifest's explicit "exact-only" → A* with admissible heuristic is both fast and exact; no conflict.

3. **Disambiguate by scope.** If one intent applies at the algorithm-name scope (manifest) and another at the compile-call scope (caller option), the more specific scope can override the broader one without it being treated as conflict — that's just normal scoping rules.

4. **Make the unsatisfiable satisfiable.** If an intent is structurally unmet (caller wants bidirectional but no CSR), look for adjustments:
   - CSR buildable + workload large enough to amortise → emit a build-CSR step
   - CSR unbuildable → warn loud and fall back to unidirectional (warn-not-silent is important here; the caller stated intent and we couldn't honour it)
   - The silent fallback mode is for when *no* preference was stated, not when an unmet intent was

5. **Surface the decision trace.** Every selection emits a reasoning trace (stderr at compile-time, comment in generated code): chosen strategy, deciding factor, what was overridden, what alternatives were considered. Even when the next step (caller-wins) is the final answer, the trace explains *why* caller overrode the cost model.

6. **Caller-wins-with-loud-warning** is the actual fallback — only when steps 1–5 fail to resolve. Trace records: "caller's `kernel_mode(bidirectional)` overrode manifest's `strategy(unidirectional)`; reason for override unknown; consider reconciling."

This is more involved than a simple "caller wins" precedence rule, but it pays for itself the first time someone gets a surprising compilation choice and looks at the trace.

## Reasoning trace as a first-class concern

A compiler that picks strategies behind the user's back is debuggable *only if* the user can ask "why did you pick this?" and get a structured answer. The reasoning trace is the answer.

The trace is **structured data**, not freeform text. Each step in the conflict-avoidance hierarchy contributes a structured entry: which signal triggered, which rule fired, what alternative was considered and rejected. The trace is emitted both:

- **At compile time**, to stderr, as a human-readable rendering (one line per selection decision, with the deciding signal named).
- **In the generated code**, as a comment header on the kernel call site (so reading the generated code reveals the strategy and its reasoning without consulting the compiler log).

A later `unifyweaver explain <pred>` command can render the trace nicely (this is one of the gaps named in book-18 chapter 9 §user-discovery). For now, the structured trace exists and is emitted; the rendering is human-readable but not yet interactive.

## Architectural alternatives considered

### 1. F#-specific cost rule in `wam_fsharp_target.pl`

The smallest-surgery version: add a predicate inside the F# WAM target that decides when to upgrade and short-circuits the existing `maybe_upgrade_bidirectional/2` call. Cheap, fast, target-coupled.

Rejected because: the same decision logic will be wanted by `wam_haskell_target.pl`, `wam_c_target.pl`, and any future WAM-flavoured target. Embedding it in F# would force later duplication or extraction.

### 2. Inline the logic into the existing cost model (`cost_model.pl`)

The cost model is the natural-sounding home. Rejected because: the existing cost-model code (see [`COST_FUNCTION_PHILOSOPHY.md`](COST_FUNCTION_PHILOSOPHY.md) and [`CACHE_COST_MODEL_PHILOSOPHY.md`](CACHE_COST_MODEL_PHILOSOPHY.md)) is concerned with per-operation cost estimation for specific resolvers (cache strategy, scan strategy). Strategy selection across the full per-query/fixed-point/cached space is a different conceptual level — it *consumes* the cost model, it doesn't extend it.

The new module reuses cost-model primitives as helpers; it does not extend the cost model's responsibilities.

### 3. Hardcode the strategy in the algorithm manifest

The user always declares `strategy(...)` explicitly; the compiler never infers. Rejected because: this works for sophisticated users but offers nothing to new users or to cases where workload changes without warrant for a manifest edit. The redirectable-declarative bargain in book-18 ch2 names exactly this: the compiler should pick sensibly when the user doesn't, and the user should be able to redirect when the compiler picks wrong.

### 4. ML-based pattern-to-strategy classifier

Train a model on (recurrence shape, workload, target) → strategy. Rejected because: the pattern space is small and well-understood, the workload signals are sparse and structured, and debuggability matters more than recall. A rule-based selector with a clear hierarchy is the right tool for this size of problem. ML would be appropriate when the pattern space grows beyond hand-enumeration; it does not yet.

See book-18 chapter 10 §rule-based-vs-ml-based for the longer argument.

## What we picked

A new module `src/unifyweaver/core/recurrence_evaluation_strategy.pl` exposing `select_evaluation_strategy/3`. F# WAM target is the first consumer. The module is target-agnostic; later WAM targets (Haskell, C) consume it without modification. The level-1 decision tree (per-query / fixed-point / cached / hybrid) is structurally present in the API, but only the per-query branch is populated with real logic in the first round — fixed-point compilation for F# WAM is a separate (larger) piece of work, and adding the level-1 selector now would be wiring up a switch with no second position.

The conflict-avoidance hierarchy is implemented as a six-step resolution machine. The reasoning trace is structured and emitted at both compile-time (stderr) and runtime (generated-code comment).

## When this matters

The strategy selector matters whenever **the same recurrence admits multiple admissible evaluations** and the workload provides signal about which is better. Concretely:

- The bidirectional vs unidirectional choice for the F# WAM ancestor kernel — the immediate driver.
- Future fixed-point compilation for F# WAM — when it lands, the level-1 decision tree gets a real chooser.
- Cross-target strategy reuse — Haskell WAM, C WAM, and the C# Query Runtime can all consume the same strategy decisions.
- Cost-model-driven auto-selection in the presence of relation-policy and algorithm-manifest hints — the user gets sensible defaults without writing per-call optimisation specs.

It does *not* matter (and should not be invoked) for:

- One-shot toy programs where the cost of any strategy is trivial.
- Cases where the user's intent is unambiguous and the cost model is informational only.
- Targets that have a single supported strategy class (e.g. stream targets that only do per-query).

The module is designed to be a no-op when there's no real decision to make: if only one strategy is admissible, the selector picks it and emits a minimal trace ("only one admissible strategy: X"). The overhead is bounded by the cost of evaluating the cost-model rule once per compile.

## What this isn't doing (yet)

- **Fixed-point compilation for F# WAM.** The level-1 decision tree includes it as a branch but the F# WAM target doesn't yet emit fixed-point code for these kernels. Until that lands, the level-1 decision defaults to `per_query` (or honours an explicit `strategy(fixed_point)` from the manifest, which then dead-ends with a clear error).
- **Cached / lookup-table strategy.** Same situation. Stubbed in the API; not selectable from cost-model rules.
- **Hybrid strategies.** Magic-set transformation, demand-driven Datalog, seed-and-refine. Named in the design space; not implemented.
- **The `unifyweaver explain <pred>` command.** The structured trace is emitted; a nice renderer is not yet built.
- **Numeric-loop-breaking detection.** The infrastructure passes contraction rate `r` through to the selector, but the *detector* that infers `r` from clause structure (for, say, a PageRank-style predicate) is not built. For now, `r` is supplied via algorithm-manifest hints; auto-inference from the predicate body is future work.

These gaps are not bugs in the present design; they are the explicit shape of the first round. The module is sized for the full eventual decision tree, with most branches as documented stubs.

## See also

- [`book-18-graph-algorithms`](../../education/book-18-graph-algorithms/) — the broader narrative this design fits into. Particularly chapter 7 (the difference-equation pivot), chapter 9 (constraint-hint predicates), chapter 10 (pattern detection by rule), appendix B.7 (convergence robustness).
- [`TREE_LIKENESS_INDEX.md`](TREE_LIKENESS_INDEX.md) and [`TREE_LIKENESS_INDEX_THEORY.md`](TREE_LIKENESS_INDEX_THEORY.md) — the theory connection (the `r` ratio as diagonal-dominance condition).
- [`KERNEL_SHAPE_RECOGNITION.md`](KERNEL_SHAPE_RECOGNITION.md) — the upstream pattern-detection layer. Recurrence-evaluation-strategy *consumes* detected kernels.
- [`ALGORITHM_MANIFEST_SPECIFICATION.md`](ALGORITHM_MANIFEST_SPECIFICATION.md) — the algorithm-vs-optimisation split the selector reads.
- [`RELATION_POLICY_DECLARATIONS.md`](RELATION_POLICY_DECLARATIONS.md) — the relation-policy hints that become data signals.
- [`COST_FUNCTION_PHILOSOPHY.md`](COST_FUNCTION_PHILOSOPHY.md) and [`CACHE_COST_MODEL_PHILOSOPHY.md`](CACHE_COST_MODEL_PHILOSOPHY.md) — adjacent cost-model layers the selector composes with but does not extend.
- [`SCAN_STRATEGY_PHILOSOPHY.md`](SCAN_STRATEGY_PHILOSOPHY.md) — closest sibling design in shape (a strategy selector with its own philosophy / specification / implementation-plan triple).
- [`WAM_FSHARP_COST_ANALYZER_DESIGN.md`](WAM_FSHARP_COST_ANALYZER_DESIGN.md) and [`WAM_FSHARP_CSR_KERNEL_INTEGRATION.md`](WAM_FSHARP_CSR_KERNEL_INTEGRATION.md) — the F# WAM target context this work integrates into.
- [`RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md) — the formal spec.
- [`RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md`](RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md) — the implementation plan.
