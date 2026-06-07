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

**The collapse holds under specific conditions.** Strictly, lazy and eager strategies compute the same relation when the recurrence is **monotone over a complete lattice** — the setting of Datalog and of Tarski's fixed-point theorem. Non-monotone recurrences, probabilistic recurrences, and recurrences whose semantics depend on traversal order (e.g. some Prolog programs that rely on cut for correctness) may not admit both strategies. The selector treats `monotone(true)` as a precondition for any level-1 decision that crosses strategy classes; non-monotone recurrences fall back to whatever strategy the user explicitly declares, with no auto-selection across classes.

The implication for compiler design: pick the strategy *separately* from defining the recurrence. The user writes the recurrence (Prolog clauses); the compiler picks the strategy (via cost-model rule + optional user hints). The two concerns separate cleanly.

### The orthogonality is not complete

A subtlety worth naming. Strategy classes are **structurally** orthogonal — you can imagine them as independent axes — but two properties of the recurrence cross-cut the classes and determine which strategies are admissible for that recurrence:

- **Value domain** — whether the recurrence's output lives in a *combinatorial* lattice (boolean, finite set of tuples — the Datalog case) or a *numeric* lattice (real-valued, continuous — the PageRank / `d_wPow` case). The two classes have different termination criteria.
- **Loop-breaking mechanism** — how the recurrence avoids non-termination on cyclic inputs. Either *combinatorially* via an explicit visited-set (`\+ member` in the user's clauses) or *numerically* via contraction (the iteration operator is a contraction map with rate < 1).

The two axes interact with fixed_point admissibility in different ways:

- For **combinatorial recurrences** (boolean lattice, finite Herbrand base — the standard Datalog case), `fixed_point` is admissible whenever the recurrence is `monotone(true)`. Tarski's theorem on a finite lattice guarantees the least fixed point exists and is reached by bottom-up iteration in at most |L| steps. **No contraction-rate guarantee is needed.** Datalog-style transitive-closure recurrences are the canonical example: combinatorial, monotone, terminates without any `r < 1` requirement.
- For **numeric recurrences** (continuous lattice — `d_wPow`, PageRank, weighted graph statistics), `monotone(true)` alone is not enough — convergence over a continuous lattice requires a contraction guarantee. The recurrence must have `numeric_contraction_rate(R)` with `R < 1.0` for `fixed_point` to be admissible.

**Visited-set loop-breaking is about per-query termination, not fixed-point admissibility.** The user's `\+ member(N, Visited)` clauses are a top-down-traversal safety mechanism — they let a Prolog query terminate on cyclic input. Bottom-up Datalog evaluation doesn't consult them; it adds tuples until fixpoint and doesn't loop because the universe is finite. So the combinatorial loop-breaking signal `has_combinatorial_loop_break(true)` tells the selector that *per-query* strategies are safe on cyclic graphs; it doesn't directly enable `fixed_point` (the value-domain and monotonicity properties do that independently).

**Bellman-Ford-style algorithms are a different case.** Bellman-Ford computes single-source shortest paths via |V|−1 iterations of edge-relaxation; it is *numeric* (real-valued distances), *monotone* (distances only decrease), and terminates in a bounded number of iterations because no shortest path has more than |V|−1 edges (assuming no negative cycles). Its termination guarantee comes from this **iteration-count bound from graph structure**, not from contraction-rate and not from the user's visited-set tracking. UnifyWeaver does not currently capture this admissibility path — there is no `iteration_bound(N)` recurrence property; supporting Bellman-Ford-style fixed_point evaluation would require adding one, plus a corresponding `termination_guarantee/1` clause that admits it. The current kernel registry has no Bellman-Ford kernels, so this is a documented future-work gap rather than an immediate concern. The reviewers' suggestion to admit Bellman-Ford via the user's visited-set signal is technically off — that signal is about per-query top-down traversal safety; bottom-up Bellman-Ford fixed_point doesn't consult it.

**Monotonicity-cross-class restriction.** A separate concern: when `monotone(false)`, no strategy class can be *auto-selected across classes* — the selector restricts to whichever class the user explicitly declares via intent. Non-monotone recurrences may have evaluation-order-dependent semantics; the compiler does not pick between classes without guidance. This restriction is applied in the conflict-resolution phase (where intent is in scope), not at the admissibility check (which sees only the recurrence).

**Putting it together — fixed_point admissibility test (termination only):**

```
admissible(fixed_point, Recurrence) :-
    termination_guarantee(Recurrence).

termination_guarantee(Recurrence) :-
    value_domain(Recurrence, combinatorial), !.    % finite-state iteration always halts:
                                                   %   - monotone: reaches LFP in <= |L| steps (Tarski + finite)
                                                   %   - non-monotone: may oscillate, but in a detectable cycle
                                                   %     on finite state — evaluator stops + reports cycle
                                                   % Either way: termination is guaranteed; semantic meaning of
                                                   % the result is monotone-dependent (handled in Phase C).
termination_guarantee(Recurrence) :-
    value_domain(Recurrence, numeric),
    numeric_contraction_rate(Recurrence, R),
    R < 1.0.                                       % continuous lattice needs contraction; monotonicity is
                                                   % orthogonal to termination here too (a contraction
                                                   % terminates regardless of monotonicity, but the result-
                                                   % semantics again depends on monotonicity).
```

Note the pseudocode does *not* gate on `monotone(true)`. Admissibility is about whether the strategy will terminate; for combinatorial recurrences over a finite state space, iteration always terminates (either at a fixed point if monotone, or at a detectable cycle if not). For numeric recurrences, termination requires contraction. Monotonicity affects whether the iteration converges to the *least fixed point*, which is what fixed_point evaluation usually intends — but that's a *semantic* concern, not a termination concern, and it's handled at the conflict-resolution phase (Phase C) where the cross-class restriction lives.

**Caveat — cycle-detection is an aspiration, not a current guarantee.** The "detectable cycle on finite state" claim above is a property of the *evaluator*, not of the recurrence. UnifyWeaver's current `fixed_point` templates (when they exist; the F# WAM target doesn't yet have one) are monotone-only — they would loop on a non-monotone input rather than detect the cycle and stop. The admissibility test admits non-monotone combinatorial recurrences as a *future-compatibility* design choice; an implementation that wants to take advantage of it needs to add cycle-detection logic. For now, the cross-class restriction in Phase C is the only protection against the selector picking `fixed_point` for a non-monotone recurrence and producing a runaway evaluation.

In practice, UnifyWeaver's kernels (TC, ancestor, BFS variants, `d_wPow`) are all monotone, so the distinction is operational only in the edge case of a non-monotone recurrence the user wants to force into fixed-point evaluation. The Phase C restriction protects against accidental cross-class selection for non-monotone recurrences (the user gets a warning); admissibility alone doesn't reject them.

For Phase 1 of the implementation, almost all kernels are combinatorial (TC, ancestor, BFS variants), so the contraction-rate check rarely applies. The contraction-rate machinery is set up for when `d_wPow`-style numeric recurrences enter the picture (book-18 ch7's territory).

## Theory connection: `r` is conjectured to be the contraction rate

The convergence ratio `r = b'/(b_eff · D)` from [`TREE_LIKENESS_INDEX.md`](TREE_LIKENESS_INDEX.md) §2 and [`TREE_LIKENESS_INDEX_THEORY.md`](TREE_LIKENESS_INDEX_THEORY.md) §2.3 plays the role of the **spectral contraction rate** for the linearised `d_wPow` iteration operator. Theorem 2.3's bound `r/(1−r)` on the per-step contribution from longer paths is the geometric-series bound from a contraction with rate `r`. The "shortcuts are rare" empirical observation (design note §3.3) and the "iteration converges fast" theoretical observation are the same observation, viewed through the two lenses of metric structure and numeric contraction.

**This identification is a conjecture, not a theorem.** Rigorously identifying `r` with the spectral radius of the iteration operator requires a norm-specific argument that has not been constructed in the project to date. The intuition is sound — the friendship-paradox quantity `E[d²]/E[d]` is a known estimator for the spectral radius of a configuration-model random graph's adjacency matrix, and `b_eff` is a directional extension of this — but a precise proof would tighten three things at once: (i) which norm the spectral analysis is being done in, (ii) which operator linearisation of the `d_wPow` iteration is being analysed, and (iii) the assumption that the **weighted-degree distribution is approximately uncorrelated** (i.e. the configuration-model assumption applies to the weighted directed graph the iteration operator is built from — without this, the friendship-paradox estimator may diverge from the spectral radius even in expectation). The rigorous identification is tracked theory work, named explicitly as a future-work item in [book-18-graph-algorithms appendix B.7](../../education/book-18-graph-algorithms/13_appendix_b_internal_theory.md) (convergence robustness — feature and trap). Not blocking; this PR proceeds on the conjecture, with the hedge documented and the dependence on the conjecture made operationally visible (see below).

**Why this matters operationally.** The contraction-rate gating in the previous section (`numeric_contraction_rate ≥ 1` disqualifies fixed_point) makes `r` operationally consequential, not just rhetorical. The selector can refuse to choose fixed-point if `r ≥ 1`, which only makes sense if `r` reliably tracks what the iteration operator actually does. The conjecture, even pending verification, is the strongest concrete relationship between graph statistics and iteration-convergence we currently have. The cost-model rules treat `r` as an upper-bound estimator of contraction rate — robust to small estimator errors in the standard way (see [book-18 appendix B.7 §convergence-robustness](../../education/book-18-graph-algorithms/13_appendix_b_internal_theory.md)) — but the entire chain becomes more brittle if the conjecture is wrong in a substantive way. Tightening the proof is a real follow-up.

This is also why the strategy selector takes the recurrence's contraction rate as input even when the chosen strategy is per-query. The selector uses it as a signal for "how aggressive can the budget cap be?" and "how cheap will fixed-point iteration be when it lands?" — both of which are calibrated under the conjecture.

### Spectral connection in plainer terms

For readers approaching this from linear algebra rather than graph theory: an iteration `x_{k+1} = B·x_k + c` converges iff the **spectral radius** of `B` is less than 1 (spectral radius = largest absolute eigenvalue). **Diagonal dominance** is one sufficient condition for spectral radius < 1; the **condition number** governs how fast convergence happens — high condition number means slow even though convergence is guaranteed. All three are aspects of *spectral analysis* — properties of the operator's eigenvalue structure.

A scoping note on the condition number specifically: the definition `κ(B) = λ_max/λ_min` is the standard form for **symmetric positive-definite (SPD)** matrices. The `d_wPow` iteration operator is a *directed* weighted-graph operator and is generally non-symmetric; for non-symmetric / non-SPD operators, the relevant convergence-rate quantity is the ratio of the spectral abscissa to the minimum singular value (or equivalently, properties of the singular-value spectrum rather than the eigenvalue spectrum). The intuition — "well-conditioned converges fast, ill-conditioned converges slow" — carries over, but the exact formula does not. For UnifyWeaver's purposes, the relevant quantity in the asymptotic bound `r/(1−r)` is the geometric-series rate, which is the spectral-radius-analog (not the condition-number-analog) — so the condition-number aside is informational about adjacent ideas, not directly load-bearing for the design.

The conjecture we depend on: the friendship-paradox-corrected `b_eff` plays the spectral-radius role for the `d_wPow` iteration operator; `r = b'/(b_eff·D)` is the contraction rate; the bound `r/(1−r)` from theorem 2.3 is the geometric-series convergence rate. Rigorous identification is future work; the operational consequences of the conjecture (per-iteration residual bound, contraction-rate gating for fixed_point) are what the selector relies on.

## Signals: intent, declared data, inferred data

Workload signals come from several sources at compile time. The selector distinguishes three tiers:

- **Intent signals** — explicit user-stated requirements about the outcome. Examples: caller's `kernel_mode(bidirectional)` option, manifest's `strategy(...)` entries. Intent overrides the cost model. High confidence; the user is responsible for the meaning.
- **Declared data signals** — facts the user has *explicitly stated* about the world the cost model reasons over. Examples: `cardinality(large)` from a `relation_policy` declaration, an explicit `csr_path(_)` option naming a file the user has prepared. Declared data has high confidence; the cost model treats it as fact.
- **Inferred data signals** — facts the compiler has *inferred* from static analysis or the absence of declarations. Examples: `csr_buildable(true)` (inferred by checking the edge predicate has a buildable inverse), `query_pattern(single_pair)` (inferred from the predicate's mode declarations or call-site analysis). Inferred data has lower confidence; the cost model treats it as soft evidence with explicit confidence weights.

Most "conflicts" between signals dissolve when these tiers are applied. A manifest's `cardinality(small)` and a caller's `kernel_mode(bidirectional)` are not in conflict — the cardinality is declared-data feeding the cost model (which would have preferred unidirectional), while the kernel-mode is intent overriding the cost model's preference. The cost model still consumed the cardinality; it just didn't get the final word.

Only when *two intent signals* point at incompatible outcomes is there a real conflict — and even then the conflict-avoidance hierarchy (below) tries several non-trivial resolutions before treating caller-wins as the fallback.

**Inferred signals carry uncertainty.** The compiler's inferences can be wrong: a `csr_buildable(true)` based on static analysis might fail at runtime if the underlying data source is unreachable; a `query_pattern(single_pair)` inferred from mode declarations might be contradicted by how the predicate is actually called. The cost-model rules treat inferred data as soft evidence — they contribute to scoring but with explicit confidence weights, and the reasoning trace records the tier of each signal so a user reading the trace can see which signals were inferences and override them by adding a corresponding declared-data signal to their relation-policy or manifest.

## Conflict-avoidance hierarchy

The selector walks a hierarchy of resolution steps. The step labels below match the semantic names in [`RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md` §Phase C](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#phase-c--conflict-resolution); a separate signal-classification preprocessing step (`classify_signals`) and an always-on trace-emission invariant bracket the hierarchy.

**Preprocessing — `classify_signals`.** Workload signals are partitioned into intent / declared-data / inferred-data tiers (per the previous section). This is a static sort, not a resolution step; it just sets up the inputs.

**Resolution steps**, walked in order. The first one that resolves wins:

1. **`step_no_intent`** — If there are no intent signals at all, the cost-model output wins immediately. Trace: "no intent signals; cost-model preference applies."

2. **`step_intent_matches`** — If *every* intent signal is satisfied by the cost-model choice (per the intent-compatibility matrix in the SPEC), no conflict. Trace: "intent and cost-model agree." This step handles the multi-intent-all-matching case, not only the single-intent case.

3. **`step_third_option`** — Look for a compatible third option. If intent A points at strategy X and intent B points at strategy Y, but there exists a strategy Z that satisfies both intents, pick Z. Concrete example: caller's implicit "fast" + manifest's explicit "exact-only" → A* with admissible heuristic is both fast and exact; no conflict.

4. **`step_scope_disambiguation`** — If one intent applies at the algorithm-name scope (manifest) and another at the compile-call scope (caller option), the *more specific* (refined) scope wins *without it being treated as conflict*. Definition: intent A **refines** intent B iff A's strategy-set is a subset of B's strategy-set; A is the more specific intent and wins. Concrete example: manifest says `strategy(per_query(_))` (matches *any* per_query strategy — broad), caller says `kernel_mode(bidirectional)` (matches `per_query(bidirectional)` and `per_query(astar)` only — narrow); caller's strategy-set is a subset of manifest's, so caller refines manifest, and caller wins by specificity, not by override. When the two intents have **disjoint** strategy-sets (neither refines the other) this step does *not* fire — the resolution falls through to the satisfiability and caller-wins steps. (Terminology note: earlier drafts said "subsumes" — switched to "refines" because the natural reading of "A subsumes B" varies between type-theory and natural-language conventions; "A refines B" is unambiguous: A is more specific.)

5. **`step_satisfiability`** — If an intent is structurally unmet (caller wants bidirectional but no CSR), look for adjustments:
   - CSR buildable + workload large enough to amortise → emit a build-CSR adjustment step in the trace; the chosen strategy is `per_query(bidirectional)` with the build-CSR step as a precondition
   - CSR unbuildable → warn loud and fall back to unidirectional (warn-not-silent is important here; the caller stated intent and we couldn't honour it)
   - The silent fallback mode is for when *no* preference was stated, not when an unmet intent was

6. **`step_caller_wins`** — The actual fallback — only when steps 1–5 fail to resolve. Caller-wins-with-loud-warning. Trace records: "caller's `kernel_mode(bidirectional)` overrode manifest's `strategy(unidirectional)`; reason for override unknown; consider reconciling."

**Always-on invariant — trace emission.** Every selection produces a reasoning trace. Even when an early step (e.g. `step_no_intent`) resolves immediately, the trace records what fired and why. The trace is what makes the selector debuggable; see the next section for its structure and persistence.

This is more involved than a simple "caller wins" precedence rule, but it pays for itself the first time someone gets a surprising compilation choice and looks at the trace.

## Reasoning trace as a first-class concern

A compiler that picks strategies behind the user's back is debuggable *only if* the user can ask "why did you pick this?" and get a structured answer. The reasoning trace is the answer.

The trace is **structured data**, not freeform text. Each step in the conflict-avoidance hierarchy contributes a structured entry: which signal triggered, which rule fired, what alternative was considered and rejected.

**The selector returns the trace; the caller decides what to do with it.** The selector module itself does *not* write to stderr. It produces the structured trace as part of its return value. Target adapters (or thin wrappers around them) decide whether to emit the trace to stderr, to a logfile, to a structured-output channel, or nowhere at all. This keeps the selector pure-functional and testable; quiet-mode operation (no stderr output) is the default for tests and library use, while CLI invocations turn emission on.

Two renderings are provided as helpers:

- **Compile-time stderr rendering**, human-readable, one line per selection decision, with the deciding signal named. Adapter calls a helper predicate that takes the trace and writes to stderr.
- **Generated-code comment**, multi-line, inserted as a comment header on the kernel call site so reading the generated code reveals the strategy and its reasoning without consulting the compiler log. Target inserts this via a syntax-appropriate comment-prefix.

**Persistence note.** The structured trace is *in-process only* — it lives in the compiler's memory during compilation and is not persisted between runs. The generated-code comment is the only *human-readable* persistent artefact of the strategy decision. The stderr rendering, when emitted, is human-readable but ephemeral.

For a future `unifyweaver explain <pred>` command, the **primary path** is re-compilation in explain-mode: the explain tool re-runs the strategy selector with the same inputs and reads the structured trace directly from the selector's return value. This gives the explain tool full access to the structured data — every step name, every signal, every alternative considered — without the lossy round-trip through human-readable text.

Parsing the generated-code comment back into a structured trace is *not* the primary path. The comment is intentionally human-readable and not designed to be machine-parsed; treating it as a serialisation format would be fragile (whitespace, formatting drift, locale-dependent renderings would all break parsing). The comment is the user-facing record of "what was chosen"; the structured trace from re-compilation is the tool-facing one.

When the explain tool needs to *consult* historical decisions (e.g. "what did the compiler decide last time?"), the right design is a separate explicit cache — keyed by the (Recurrence, Workload) inputs — not parsing of generated comments. That cache is out of scope for this iteration.

This is one of the gaps named in book-18 chapter 9 §user-discovery. For now, the structured trace exists and the rendering helpers exist; an interactive `explain` command is future work.

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

The conflict-avoidance hierarchy is implemented as a six-step resolution machine returning a structured trace; the selector is pure-functional. Target adapters render the trace (to stderr at compile time, into generated-code comments) per their needs.

**Dependency direction.** The new module depends on `cost_model.pl`, `cost_function.pl`, `relation_policy.pl`, and `algorithm_manifest.pl` as inputs; it depends on `recursive_kernel_detection.pl` for the detected-kernel terms it consumes. The reverse direction never holds — none of those modules depends on `recurrence_evaluation_strategy.pl`. This one-directional dependency prevents circular loading at startup. F# WAM target (and future Haskell-WAM and C-WAM targets) call into the new module after running kernel detection; they do not get called back from it.

**Planned future consumers.** The C# parameterised query target (`csharp_query_target.pl`) is the existing realisation of bottom-up fixed-point compilation in the codebase. When the level-1 decision tree's `fixed_point(...)` branch gets a real chooser, the C# query target becomes the natural second consumer of `select_evaluation_strategy/3`. The selector's API is sized to accommodate this; the integration itself is future work.

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
