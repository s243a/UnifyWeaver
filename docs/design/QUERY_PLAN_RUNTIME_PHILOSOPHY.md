# Query-Plan Runtime: Philosophy

**Status**: Precursor design discussion. Captures architectural vocabulary
and open questions; does **not** document current C# implementation in
detail.  A follow-up `QUERY_PLAN_RUNTIME_SPECIFICATION.md` and
`_IMPLEMENTATION_PLAN.md` should be written after a code audit of
`src/unifyweaver/targets/csharp/` and the existing C# query-engine
education docs (see §7).

**Snapshot date**: 2026-05-19.

## 1. Context — why this doc exists

The Rust-LMDB benchmark arc (R5/R6, branches
`feat/wam-rust-bench-simplewiki` / `feat/wam-rust-bench-enwiki`)
surfaced a measurement that doesn't fit the naïve "language X is faster
than language Y" frame.  At simplewiki (297,283 edges) the Rust WAM
target beats the Haskell WAM target by roughly 7× on a single query.
At enwiki (9,932,244 edges) the relationship inverts: Haskell beats
Rust by roughly 200× on the same shape of workload.

The reversal is not explained by either language's evaluation strategy
or compiler quality.  Both targets are native-compiled, both wrap the
same memory-mapped LMDB C library, both walk the same demand-set BFS.
What differs is **when each bench materialises the demand-set edge
list** — and that turns out to dominate at scale.

The Haskell side has its own internal version of this finding (Phase
L#7 in `WAM_PERF_OPTIMIZATION_LOG.md`): the `resident` IntMap mode that
pre-loads edges into an in-memory index is ~2× **slower** than the
`resident_cursor` mode that streams from LMDB on demand, even at
simplewiki scale where the IntMap fits comfortably in RAM.  Same
language, two designs; the streaming design wins.

So the trade-off is **eager vs lazy materialisation**, not Rust vs
Haskell.  Either language can implement either side; the C# target
already implements both behind a single planner abstraction.  This doc
captures the architectural vocabulary for that abstraction and frames
the open questions that the eventual specification needs to answer.

## 2. The eager-vs-lazy choice in cost-model terms

For a single process invocation against a demand-bounded workload, the
wall-clock cost can be modelled as:

```
total ≈ M (eager materialisation, one-time) + N × ε (per-seed kernel work)
```

where `N` is the number of seeds per process and `M` and `ε` are
properties of the bench design plus the underlying graph.

For lazy designs, `M ≈ 0` and `ε` absorbs whatever streaming cost the
kernel would otherwise have paid up front:

```
total ≈ N × p           where p > ε but bounded
```

The two designs cross over at `N* = M / (p − ε)`.  Below `N*`, lazy
wins; above `N*`, eager wins.

### 2.1 Measured snapshot (May 2026)

Recorded from the R5/R6 Rust-LMDB sweeps and Haskell Phase L#7/8/9 for
direct cross-target comparison.  Both targets read the same LMDB
fixtures; both run single-process; both pin the same kernel shape
(`category_ancestor/4` with `max_depth(10)`).

| Scale | Edges | Demand set | `M` (Rust) | `ε` (Rust) | `p` (Haskell, lazy) | `N*` (crossover) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simplewiki | 297,283 | 14,680 | ~17 ms | ~3 µs | ~45 µs (sequential) | ~400 seeds |
| enwiki | 9,932,244 | 796,695 | ~140 s | ~4 µs | ~0.73 ms `-N4` | ~190,000 seeds |

The simplewiki cost model implies Rust wins for any `N` ≥ ~400 — and
since the bench iterates a large seed set already, Rust does win
(34 ms warm vs Haskell's 226 ms `-N1`).  The enwiki cost model implies
Rust only wins when a single process can batch >190,000 queries —
which is far outside the "user issues one query" workload most
deployments actually have.  For one-shot interactive workloads at
enwiki scale, Haskell's lazy cursor design beats Rust's eager design
by 200×.

### 2.2 The architectural delta

Rust's bench builds a `Vec<(String, String)>` of every demand-set edge
at startup, before any kernel iteration.  This is `M` work,
proportional to `|demand_set|` regardless of seed count.

Haskell's `resident_cursor` mode does not build such a Vec.  Each
kernel walk reads the next parent edge from an LMDB cursor on demand.
If a seed's upward walk only touches 5 edges before terminating, only
those 5 edges are read — the other ~796 k edges in the demand set are
never materialised.

The crossover math then explains the measurement reversal directly.
At simplewiki, `M` is small enough that paying it up front and amortising
across the workload still beats `N × p`.  At enwiki, `M` is enormous
relative to the per-query cost difference, so eager amortisation only
pays off at improbably-large batch sizes.

### 2.3 What this is not

This is not a story about garbage collection, FFI overhead, or
language-runtime efficiency.  It is not a story about Haskell
"catching up to" Rust with tuning.  It is a story about a runtime
design decision (when to materialise) that happens to be made
differently in each of our current bench implementations.

## 3. The query-engine pattern as a runtime planner

The C# target sidesteps the design-time eager/lazy choice by lifting
it to a runtime decision.  Based on the references in
`examples/benchmark/README.md`, the C# pipeline includes:

- A *shared internal materialisation planner layer* that covers both
  the relation-fact storage and the DAG/recursive predicates.
- *Relation-retention policy* that picks per relation between
  `relation_rows(format(tsv_grouped))`, `exact_hash_index([...])`, and
  `ReplayableBuffer` shapes.
- A *manifest* (`manifest.edn`) that records the resolved policy plus
  access contracts so generated artefacts are self-describing.
- A precedence ordering: benchmark-specific overrides take priority,
  then shared `preprocess/2` declarations, then generator defaults.

The key property: the *surface predicate* the user wrote in Prolog is
the same regardless of which plan the engine ultimately picks.  The
plan is a property of the deployed binary plus the workload at call
time, not of the source program.

This is the same shape as a database query optimiser.  The user writes
"give me ancestors with their effective distance"; the planner decides
whether to materialise the demand-set edge index, stream cursors, or
combine both (e.g., cache the hot prefix and stream the cold tail).
The plan can vary per query without changing the source.

### 3.1 Connection to the existing cost model

UnifyWeaver already has a cost-model and resolver layer for fact
storage and cache-tier selection:

| Selector | Picks between | Hooks into |
| --- | --- | --- |
| `cache_strategy(auto)` | `none`, `memoize`, `shared`, `two_level` | `cost_model.pl`, `workload_metadata` |
| `lmdb_cache_mode(auto)` | `per_hec`, `sharded`, `two_level` | same |
| `resident_auto` (matrix bench) | `resident` (IntMap) vs `resident_cursor` (lazy) | same |
| Fact predicate layout | `inline_data`, `external_source` (TSV), `lmdb` | `classify_fact_predicate/4` |

These are all **codegen-time** decisions made from compile-time
metadata.  The query-engine pattern lifts the same axis to **runtime**
decisions made from per-call parameter bindings.  Codegen still has
work to do — it has to *emit code that supports both plans*, where
today it emits one plan only.

### 3.2 Generalisation gradient

The progression looks like:

1. **Hand-tuned codegen mode** (`resident_cursor` vs `resident`):
   one variant per project, picked by the developer.
2. **Auto-selected codegen mode** (`resident_auto`): cost model picks
   the variant at codegen time based on compile-time metadata.
3. **Auto-selected runtime plan** (query-engine pattern): codegen emits
   both variants and the planner picks at query time based on actual
   parameter bindings + measured workload signals.

UnifyWeaver is at stage 2 for the WAM-Haskell target and at stage 3
for the C# target.  The Rust target is at stage 1 (single eager
variant); a future lazy-cursor Rust variant would land it at stage 2.
Stage 3 for non-C# targets is the open horizon this doc points at.

## 4. Three-axis selection

The cost-model + shape-analysis layer is choosing on at least three
distinct axes per predicate.  Conflating them obscures why the C#
planner doesn't cover everything.

### Axis 1 — Predicate kind

| Class | Can a planner operate? |
| --- | --- |
| Pure relations / fact tables | Yes — joins, projection, indexed lookup |
| Recognised recursive kernels (`recursive_kernel` declarations) | Yes — transitive closure, ancestor traversal, accumulator-passing patterns |
| Aggregations and reductions | Yes |
| Arbitrary user-defined recursive predicates | Sometimes — depends on shape analysis |
| Full Prolog with cut, side-effecting builtins, higher-order calls, complex unification | **No** — needs the WAM target |

The planner's reach is bounded by what shape analysis can prove.
Predicates outside the analysed envelope must route through the WAM
target, which is general-purpose by construction.

### Axis 2 — Mode tractability

A query plan is typically specialised for a *calling mode* — which
arguments are bound, which are free, sometimes with refinements like
"bound to a ground term" vs "bound to a partial term".

For arity-N predicate with M relevant per-argument modes, the
specialisation cross-product is up to `M^N` variants.  For arity 4
with bound/free modes, that's 16; for arity 7, 128; for arity 10, 1024.
Lists and structures multiply further.

If the planner pre-emits all specialisations, binary size grows
exponentially in arity.  Practical planners therefore either:

- **Restrict to a fixed canonical mode set** (e.g., only `(+, -, ...)`
  query-style calls, plus optionally `(+, +, -, ...)` membership-style
  calls).  Other calls fall back to generic dispatch.
- **Cap the arity** beyond which they refuse to specialise.  Above the
  cap, generic dispatch (essentially a mini-WAM register file) takes
  over.
- **Gate on compile-time mode analysis**: only specialise predicates
  where shape analysis can prove only a small set of modes will be
  called.

The WAM has no equivalent explosion: register tags carry the mode at
runtime, and one compiled body covers all modes.  The cost is per-call
tag dispatch overhead, not binary size.

### Axis 3 — Workload shape

This is the eager-vs-lazy axis from §2, plus all the related materialisation
choices (cache tier, replay buffer, indexed pre-build, etc.).  It is
specifically what the C# planner runtime-resolves.

### 4.1 The selection picture

The cost-model + shape-analysis layer decides, **per predicate**:

1. *Does it fit axis 1?*  If no → WAM target.
2. *Does it fit axis 2 within budget?*  If no → WAM target (or generic
   dispatch fallback inside the planner).
3. *Given the predicate passes 1 and 2, what plan should run for this
   call?*  This is axis 3 — the planner's runtime choice.

A predicate that passes all three is what the C# query engine handles
end-to-end.  Predicates that fail axis 1 or 2 still get compiled, but
through the WAM target, where they pay the per-call dispatch cost
in exchange for full generality.

## 5. Why the WAM target stays in the picture

The mode-instantiation explosion is the structural reason WAM
dispatch isn't subsumed by query-plan specialisation.

A query-plan target with mode pre-specialisation is faster on
hand-picked calls but cannot scale to "any arity, any mode, possibly
unknown at codegen time".  The WAM target *can*, at the cost of
runtime tag inspection per unification.  In practice the two should
coexist:

- WAM target: general-purpose backstop, compiled once per clause,
  bounded binary size.
- Query-plan target: specialised fast path, applies when shape
  analysis succeeds and mode set is bounded.
- Cost model: routes predicates to the appropriate target.

For some predicates the same source clause might be compiled through
*both* — the WAM target as a correctness backstop, the query-plan
target as the hot path for recognised modes, with the planner
choosing per call which to invoke.  Whether that dual-target shape is
worth the codegen complexity is one of the open questions in §6.

## 6. Open questions (deferred to post-audit spec)

These are explicitly the questions a code audit needs to answer
before this doc graduates to a specification:

1. **What does the C# planner actually enumerate?**  Which modes does
   it specialise on?  Which does it route to generic dispatch?  Is
   there an arity cap?  Where in the C# codebase does that decision
   live?
2. **What runtime primitives does the C# planner use?**  `ReplayableBuffer`,
   `relation_rows(format(tsv_grouped))`, `exact_hash_index(...)`,
   counted-path traversal — what's the minimal set?  Which of these
   are general-purpose vs benchmark-specific?
3. **What is the planner's input?**  Static (compile-time metadata
   only)?  Or also runtime signals (parameter bindings, observed
   selectivity, accumulated query-stream history)?
4. **What would a target-agnostic abstraction of this pattern look
   like?**  Which primitives generalise to Rust / Haskell / Go / Scala,
   and which are specifically C#-ergonomic?
5. **Do the existing C# education docs cover the architectural
   picture or are they tutorial-level walkthroughs?**  If the latter,
   what gaps does the specification need to fill?
6. **How does this interact with the scan-strategy P3 work?**  The
   warm-build core in `SCAN_STRATEGY_*.md` is doing related plan
   selection at compile time; the query-engine pattern is the runtime
   analogue.  How do they compose?
7. **Where is the right boundary between codegen-time and
   runtime-time planning?**  Some decisions (predicate-kind
   classification) are hard to defer to runtime cheaply.  Others
   (workload-shape selection) gain a lot from runtime data.  Where
   does each axis live?

## 7. References

### Cross-references inside UnifyWeaver

- `docs/design/CACHE_COST_MODEL_PHILOSOPHY.md` — cost-model framework
  that this work extends.
- `docs/design/SCAN_STRATEGY_PHILOSOPHY.md` + `_SPECIFICATION.md` +
  `_IMPLEMENTATION_PLAN.md` — scan-strategy triad; P3 (warm-build
  core) is the closest relative.
- `docs/design/COST_FUNCTION_PHILOSOPHY.md` — Green's-function /
  flux-style cost functions that feed the planner's tiebreaking.
- `docs/design/ALGORITHM_MANIFEST_SPECIFICATION.md` — the slot
  mechanism that registers algorithms (a candidate hook for plan
  selection at runtime).
- `docs/design/WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md` — interning
  regimes (export-id vs position-based); same conceptual frame
  (different choices at the same abstraction layer).
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` Phase L#7 / #8 / #9 / #14
  — Haskell benchmark history that motivates the eager-vs-lazy
  observation.
- `src/unifyweaver/core/cost_model.pl` — the cost-model resolvers
  (`cache_strategy(auto)`, `lmdb_cache_mode(auto)`).
- `src/unifyweaver/core/purity_certificate.pl` and
  `clause_body_analysis.pl` — shape analysis that informs axis 1 + 2.

### External / education

- `examples/benchmark/README.md` — the C# planner mentions (search for
  "internal materialisation planner", "ReplayableBuffer",
  "exact_hash_index", "relation-retention policy").  Authoritative
  for the surface env vars and the precedence ordering.
- C# query-engine education docs (location TBD as part of the post-audit
  follow-up).  User has not yet confirmed whether they cover the
  architectural questions in §6.

### Outside-project references

The pattern this doc describes is well-trodden ground in database
query optimisation and Datalog system design.  Reference points
worth checking when writing the specification:

- Postgres / SQL query planners — the plan-cache and specialisation
  thresholds are directly analogous.
- Datomic / DataScript — Datalog-style planners that operate on a
  recognised relational subset and refuse anything outside it.
- LogicBlox / LogiQL — production Datalog with an explicit plan
  optimiser.
- Coq / Lean tactic-mode dispatch — the analogue for proof terms
  rather than data terms.

## 8. Authorship & history

| Date | Author | Action |
| --- | --- | --- |
| 2026-05-19 | precursor draft | First version — captures architectural vocabulary and open questions emerging from the Rust R5/R6 bench arc and a Reddit-thread discussion about Haskell-vs-Rust performance framing. |
| — | — | _Next: specification doc after C# code audit._ |
