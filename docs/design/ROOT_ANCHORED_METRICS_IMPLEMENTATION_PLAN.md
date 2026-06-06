# Root-Anchored Metrics — Implementation Plan

Companion to `ROOT_ANCHORED_METRICS_PHILOSOPHY.md` and
`ROOT_ANCHORED_METRICS_SPECIFICATION.md`. Phased delivery, smallest-useful-first,
with a recommendation at each fork. **Minimum distance to root is Phase 1** — it
is the simplest, cycle-safe, useful on its own, and unlocks the branch-prune that
the other metrics also benefit from.

Guiding constraint: reuse existing machinery wherever it already fits
(`recursive_kernel_detection`, the shortest-path kernels, `demand_analysis`'s
cursor BFS over `category_child`, the visited-set instructions, the LMDB fact
source, `build_scoped_subtree_lmdb.py`, `cost_model`/`algorithm_manifest`).

## Phase 0 — Spec surface + recognition (no algorithm yet)

- Add a `root_metric/2` directive parser to a new
  `src/unifyweaver/core/root_metric.pl` (sibling of `demand_analysis.pl`):
  validate the tuple (§1 of the spec), normalise `succ → plus(1)`, default
  `cycles(bounded)`, `materialize(kernel)`.
- Expose `root_metric_spec(Name/Arity, Normalised)` for downstream consumers and
  a `root_metric_semiring(Aggregate, Combine, Semiring)` table (§3).
- Tests: `tests/core/test_root_metric.pl` — tuple validation, defaults,
  rejection of unknown combine/aggregate and negative depth.
- **No regression risk**: directive is inert until a target consumes it.

**Recommendation:** land Phase 0 alone first; it is pure analysis + tests and
defines the contract every later phase implements.

## Phase 1 — Minimum distance to root (in-kernel node-DP)

Goal: `min_dist_to_root/3` computed as a BFS-from-root node-DP, validated against
the naive Prolog reference on small graphs and run at scale on the coherent
enwiki fixture.

- **Algorithm**: single BFS from the root over the *reverse* edge direction
  (`down` `category_child` for an `up` `parent` metric), labelling each node with
  its first-seen depth = min distance. This is exactly the cursor BFS already in
  `demand_analysis` / the Rust `reachable_to_root` — extend it to *record depth*,
  not just membership.
- **Reuse**: the demand BFS infrastructure and `weighted_shortest_path3` /
  `astar_shortest_path4` kernels are the precedent; prefer extending the demand
  BFS (it already walks `category_child` with a visited set).
- **Reference oracle**: the naive `aggregate_all(min(...), dist_to_root, D)` run
  in SWI-Prolog on a tiny fixture; assert the DP matches.
- **Validate at scale**: on `enwiki_cats_correct` (root 7345184), the BFS is
  `O(edges)` once → seconds, vs the ~17 h path-enumeration. Capture the timing
  contrast as the headline result.
- Tests: tiny-graph equivalence (incl. a cycle, to show cycle-safety) +
  unreachable-node handling.

**Fork — where does the BFS run?** (a) in the generated kernel at query/load
time, or (b) as a one-shot at ingest (Phase 2). Phase 1 does (a); it is the
ad-hoc-root path and needs no schema change.

## Phase 2 — Materialise min-distance at ingest

Goal: for a fixed known root, store `node → min_dist` once so queries are lookups
+ prune.

- Extend `build_scoped_subtree_lmdb.py` (or a sibling `build_root_metric_lmdb.py`)
  to compute min-distance during the same BFS it already runs from the root and
  write a `metric_min_dist_to_root` sub-db (`int32_le node → int32_le dist`) plus
  a `meta` provenance entry (root, max_depth, aggregate) per §6 of the spec.
  *It is the same BFS — the scoped builder already visits every node from the
  root; it just isn't recording the depth yet.* Near-zero marginal cost.
- Runtime: a lookup stub (`Name(Root,Node,V)` → keyed get) + the branch-prune
  helper that consults the table. Wire an opt-in flag mirroring `WAM_DEMAND`
  (e.g. the kernel consults `metric_min_dist_to_root` when present).
- Tests: extend `tests/test_build_scoped_subtree_lmdb.py` to assert the metric
  sub-db matches a BFS oracle on the tiny fixture; codegen guard for the lookup
  stub.

**Recommendation:** default to ingest-materialisation (this phase) for a fixed
root, fall back to Phase 1's in-kernel DP for runtime-chosen roots — the same
both-and conclusion as demand-set-at-ingest.

## Phase 3 — Effective distance as a linear difference equation

Goal: replace the path-enumerating `effective_distance_sum` with the
length-bucketed node-DP (§5 of the spec), proving equality and the asymptotic
collapse.

- **Algorithm**: `count[N][L] = Σ_parents count[N'][L-1]` over the reverse BFS
  layers, `count[R][0]=1`, `L ≤ max_depth`; then
  `S(N) = Σ_L count[N][L]·(L+1)^(-n)`. Each node carries an
  `(max_depth+1)`-wide vector; one pass, `O(edges·max_depth)`.
- **Equivalence test**: on a small graph, assert the DP's `S(N)` equals the
  existing kernel's per-seed path-sum exactly (not approximately).
- **Scale**: run on `enwiki_cats_correct` and contrast with the ~24 ms/seed
  path-enumeration baseline; this is the quantitative payoff.
- **Cycles**: `cycles(bounded)` via `max_depth` truncation is the supported
  default; document the `decay<1` convergence note and leave exact cyclic linear
  solve as a non-goal.
- Materialisation: `f64` value sub-db (§6); same builder extension as Phase 2.

## Phase 4 — Maximum distance to root

Goal: `max_dist_to_root/3` (depth-bounded longest walk), the `(max,+)` instance.

- **Algorithm**: longest-walk DP over reverse BFS layers, capped at `max_depth`;
  with `cycles(scc)` optionally condense SCCs first.
- Lower priority than 1–3 (semantic is "specificity/depth," useful but not on the
  query critical path). Implement once the semiring template from Phases 1/3 is
  proven so it is mostly a parameter change.

## Phase 5 — Generalisation + (optional) inference

- Factor Phases 1/3/4 into one **semiring-parameterised lowering template** so a
  new metric is a new `(aggregate, combine)` row, not new traversal code.
- Optional: infer the `root_metric` shape from bare `aggregate_all/3` over a
  transitive closure (the spec's §7 out-of-scope item) as a convenience on top of
  the explicit directive.

## Integration points (where each phase touches the tree)

| Concern                    | Module / file                                              |
|----------------------------|-----------------------------------------------------------|
| directive + recognition    | `src/unifyweaver/core/root_metric.pl` (new)               |
| kernel shape detection     | `recursive_kernel_detection.pl`, `KERNEL_SHAPE_RECOGNITION`|
| reuse shortest-path kernels| existing `weighted_shortest_path3`, `astar_shortest_path4` |
| reverse BFS / demand walk  | `demand_analysis.pl`, Rust `reachable_to_root`            |
| ingest materialisation     | `build_scoped_subtree_lmdb.py` (+ metric sub-db)          |
| runtime lookup + prune     | LMDB fact source templates per target                     |
| cost/algorithm selection   | `cost_model.pl`, `algorithm_manifest.pl`                  |
| bench                      | the WAM matrix bench (replace the path kernel)            |

## Validation strategy (every phase)

1. **Oracle equivalence** on a tiny hand-checked fixture (incl. a cycle).
2. **Cross-check** the materialised table vs the in-kernel DP vs the naive Prolog.
3. **Scale run** on `enwiki_cats_correct` (root 7345184), reporting the timing
   contrast vs the path-enumeration baseline — and stating seed counts /
   denominators explicitly (per `feedback_perf_skepticism`; the old enwiki
   numbers were degenerate because the graph was a broken flat star).
4. **No-regression** on the existing WAM test suites.

## Recommendations summary

- **Order**: Phase 0 → 1 → 2 → 3 → (4, 5). Ship min-distance end-to-end
  (in-kernel then materialised) before touching flux.
- **Spec style**: explicit `root_metric/2` directive over inference (robust,
  matches the codebase); inference is a later nicety.
- **Where to compute**: materialise at ingest for a fixed root (default),
  in-kernel DP for runtime roots — both, mirroring demand-set-at-ingest.
- **Effective distance**: length-bucketed node-DP, depth-bounded; it is *exactly*
  the current quantity, not an approximation.
- **Cycles**: rely on `max_depth` (`bounded`) by default; `scc` only where
  longest-path semantics demand it.

## See also

`ROOT_ANCHORED_METRICS_PHILOSOPHY.md`, `ROOT_ANCHORED_METRICS_SPECIFICATION.md`,
`WAM_DEMAND_FILTER_IMPLEMENTATION_PLAN.md`, `SCAN_STRATEGY_IMPLEMENTATION_PLAN.md`,
`KERNEL_SHAPE_RECOGNITION.md`; memory `project_enwiki_correct_ingest`,
`project_demand_set_at_ingest`.
