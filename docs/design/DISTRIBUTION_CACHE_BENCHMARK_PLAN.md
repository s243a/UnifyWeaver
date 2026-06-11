# Distribution Cache Benchmark Plan

This plan turns `DISTRIBUTIONAL_FIT_POLICY.md` into an executable benchmark. The
goal is to measure when shallow precomputed path-statistic distributions pay for
themselves during budgeted path aggregate search.

The first experiment is deliberately parent-only and exact. Fitted tails,
child-direction paths, and approximate cumulative bases come later. The immediate
question is simpler:

> If we precompute exact parent-only distributions for nodes near the root, how
> much search work disappears when later path aggregate queries can stop at those
> cached nodes?

## 1. Hypothesis

For a fixed root, shallow exact distributions should have high reuse because many
deeper paths pass through a small set of root-near ancestors. A search with
budget `B_search` can stop when it reaches a cached ancestor `N` and evaluate the
cached suffix distribution over the remaining budget.

Expected pattern:

```text
D_pre = 0: no suffix cache; full exact path enumeration
D_pre = 1..2: small cache, high-value root-near suffix summaries
D_pre = 3..4: larger cache, higher hit rate, diminishing marginal returns
D_pre > 4: useful on enwiki only if high-reuse nodes continue to dominate
```

The main curve is marginal speedup per cached byte or per cached node.

## 2. Baseline scope

Start with the simplest semantics:

- graph: SimpleWiki category graph first, enwiki later;
- SimpleWiki subtree root: `Category:Articles`;
- enwiki-oriented root: `Category:Main_topic_classifications`, or the canonical
  root used by the existing tree-likeness fixtures;
- broader category roots: `Category:Container_categories` and
  `Category:Categories` need filtering before benchmark use because they include
  massive container/admin/template regions;
- individual container categories can still be valid subtree roots, but
  `Category:Container_categories` itself should be treated as a high-branching
  registry outlier rather than part of the same global branch-factor population;
- edge direction: parent-only paths toward the selected benchmark root;
- avoid mixing content and admin/container regions in one benchmark population;
  once admin categories enter the sample, branch-factor and support-width
  statistics are no longer globally homogeneous;
- path statistic: hop count `L`;
- distribution representation: exact histogram, plus optional exact cumulative
  bases derived from that histogram;
- cycle policy: use the same bounded/visited policy as the exact search oracle;
- approximation: none.

Use `scripts/sample_distribution_cache_subtree.py` to turn a resolved
`child<TAB>parent` category edge list into a bounded benchmark subtree and
target list before passing it to `scripts/distribution_cache_benchmark.py`.
The sampler owns root selection, depth capping, and admin/container filtering;
the benchmark runner owns measurement over the resulting TSV.

Use `scripts/export_distribution_cache_edges.py` when starting from the
resolved SimpleWiki SQLite database created by `examples/benchmark/parse_simplewiki_dump.py`.
Existing benchmark artifacts may already provide numeric `category_parent.tsv`
files; those can be sampled directly when run with the matching numeric root
from their `root_categories.tsv` sidecar.

Under those constraints, cached-distribution search should be semantically exact.
Any discrepancy against full exact search is a bug in orientation, budget
accounting, normalization, cycle policy, or path-statistic handling.

## 3. Experiment matrix

Run a grid over precompute depth and search budget:

```text
D_pre:    0, 1, 2, 3, 4, 5
B_search: 2, 4, 6, 8, 10
```

`D_pre` is root-outward parent distance. A node is cache-eligible when its
minimum parent distance from the root is `<= D_pre`.

`B_search` is the path aggregate budget from a queried node toward the root. In
the first experiment, the budget is hop count. Later versions may use weighted
parent cost or tuple statistics.

For each `(D_pre, B_search)` cell, run the same sampled target set under:

| Mode | Description |
|------|-------------|
| `full_exact` | Enumerate/aggregate parent-only paths with no distribution cache |
| `cached_histogram` | Stop at cached nodes and scan exact histogram bins up to remaining budget |
| `cached_mass_cdf` | Stop at cached nodes and use mass CDF for reachability-mass queries |
| `cached_basis` | Use mass + selected cumulative bases for bounded average or weighted power |

The cached node is a boundary condition, not merely a lookup table. For a target
node `V`, a hit at ancestor `A` replaces the suffix search from `A` to the root
only when the cached distribution has a compatible root, horizon, statistic,
admissibility policy, and target scope. Global root tables are reusable as broad
boundary conditions; per-query tables are reusable only inside the ancestor cone
they were computed for. The benchmark should therefore count cache hits by both
node and boundary compatibility, not just by node id.

The first pass may implement only `full_exact` and `cached_histogram`. Add CDF
modes once the raw suffix-cutoff semantics are verified.

## 4. Query sets

Use multiple target sets so the cache is not tuned to one workload:

| Set | Purpose |
|-----|---------|
| `near_root` | Nodes at parent distance 1-4; should hit shallow cache frequently |
| `mid_depth` | Nodes below main-topic layers; tests practical reuse |
| `deep_tail` | Deep category nodes; tests whether shallow cache still cuts work |
| `high_indegree` | Multi-parent nodes likely to have many root paths |
| `random_reachable` | Unbiased reachable sample for aggregate reporting |

For SimpleWiki, exact runs should be tractable enough to use larger samples. For
enwiki, start with small samples and carry forward only the modes that passed
SimpleWiki parity.

## 5. Measurements

Record per query:

```text
target_node
D_pre
B_search
mode
runtime_ms
nodes_expanded
edges_examined
paths_enumerated_or_aggregated
cache_hits
first_cache_hit_depth_from_target
remaining_budget_at_hit
histogram_bins_scanned
cumulative_basis_lookups
ancestor_cone_nodes
ancestor_cone_edges
exact_histogram_bytes
parametric_state_bytes_estimate
continuous_sample_points
continuous_sample_bytes_estimate
compression_ratio_estimate
tail_pruning_thresholds
tail_pruned_kept_bins
tail_pruned_dropped_bins
tail_pruned_dropped_mass
tail_pruned_functional_error
result_mass
result_first_moment
result_weighted_power
result_entropy_if_available
exact_result_reference
absolute_error
relative_error
```

Record per cache build:

```text
root
D_pre
eligible_nodes
cached_nodes
cache_bytes_raw_histogram
cache_bytes_cumulative_bases
build_runtime_ms
max_support_size
mean_support_size
p95_support_size
total_distribution_mass
```

Derived metrics:

```text
speedup = runtime_full_exact / runtime_cached
work_reduction = 1 - nodes_expanded_cached / nodes_expanded_full_exact
hit_rate = queries_with_cache_hit / total_queries
bytes_per_ms_saved = cache_bytes / (runtime_full_exact - runtime_cached)
marginal_speedup_per_depth = speedup(D_pre=n) - speedup(D_pre=n-1)
```

## 6. Correctness checks

For exact parent-only histograms, require zero semantic error against the full
exact oracle for all supported functionals:

- reachability mass under `B_search`;
- bounded average when mass is nonzero;
- tail mass within the finite horizon;
- weighted-power sum for configured `N`;
- entropy when computed from the exact histogram.

CDF and cumulative-basis modes must also match raw histogram scans exactly,
within floating-point tolerance:

```text
mass_cdf(B) == sum_{L <= B} histogram[L]
moment_cdf(B) == sum_{L <= B} L * histogram[L]
weighted_power_cdf(B) == sum_{L <= B} (L + 1)^(-N) * histogram[L]
interval_mass(B1, B2) == mass_cdf(B2) - mass_cdf(B1)
```

If any exact-cache mode differs from `full_exact`, fail the benchmark cell and
emit the path-statistic trace needed to diagnose the mismatch.

## 7. Cache admission variants

After the full-cache grid is working, test admission pressure. Instead of storing
all eligible nodes within `D_pre`, cap the cache by bytes or entries and compare:

| Policy | Expected behavior |
|--------|-------------------|
| `store_all` | Upper-bound speedup and storage cost |
| `overwrite_on_collision` | Baseline to demonstrate why blind overwrite loses useful root-near entries |
| `root_proximity` | Prefer lower parent distance to root |
| `reuse_score` | Prefer root proximity, descendant count, observed hits, and recompute cost |
| `score_with_hysteresis` | Same as `reuse_score`, but avoids churn on close scores |

The benchmark should report when a candidate loses admission but still serves the
current query. Admission policy affects shared-cache reuse, not correctness of
the current result.

## 8. Expansion protocol

Run the work in layers:

1. Tiny hand-checkable DAG fixtures.
2. SimpleWiki parent-only exact histograms.
3. SimpleWiki cached search cutoffs over the `(D_pre, B_search)` grid.
4. SimpleWiki cumulative-basis modes.
5. SimpleWiki admission-pressure policies.
6. Enwiki sampled parent-only runs.
7. Enwiki admission-pressure runs.
8. Fitted finite-support tails only after exact-cache behavior is stable.

Do not introduce child-direction paths until parent-only exact cache semantics
and admission policy are measured. Parent-only distributions can later bound or
prioritize child-inclusive searches.

## 9. Future directions

The first benchmark deliberately measures exact parent-only histograms. Later
passes can test cheaper summaries and approximations once exact-cache semantics
are stable.

One near-term extension is to compute scalar support bounds before deciding
whether to build a full distribution:

```text
L_min(v) = shortest parent-only path length from v to root
L_max(v) = longest parent-only path length from v to root, under the active cycle policy
support(P_v) <= [L_min(v), L_max(v)]
```

When the requested functional is only `min_support` or `max_support`, these are
not approximations; the scalar recurrence is the whole computation. When the
requested functional is a bounded aggregate, the bounds are pruning and planning
state rather than a replacement for `P_v`:

```text
if L_min(v) > remaining_budget: suffix contributes zero
if L_max(v) <= remaining_budget: suffix is fully inside the budget
if L_max(v) - L_min(v) is small: exact histogram is likely cheap
if L_max(v) - L_min(v) is large: prefer a cached boundary or fitted tail candidate
```

This lets deeper layers carry cheap min/max summaries while the planner decides
where full histograms are still worth materialising.

When the target-scoped ancestor cone is small, full histograms may remain cheap
even for nodes that are deep by root distance. The benchmark should therefore
separate global precompute depth from per-query ancestor-cone cost. A continuous
fit sampled onto a fixed grid, such as 100 points before FFT convolution, should
beat exact histograms only when the exact support or reuse pattern justifies
that grid cost. Otherwise it is primarily a storage/compression option after an
exact or sampled distribution has already been validated.

For extremely light tails, also test exact-prefix tail pruning before switching
to a parametric representation. The benchmark should sweep thresholds such as
`1e-2`, `1e-3`, and `1e-4`, then report how many suffix bins are removed and
how much mass, first moment, or weighted-power contribution is lost. This makes
pruning a functional-level decision rather than a visual judgment about the
shape of the histogram.

The same run should report parent-degree moments over the selected reachable
nodes:

```text
p(v) = number of parent choices for v
E[p]
E[p^2]
E[p^2] / E[p]
```

`E[p^2]/E[p]` is the size-biased parent branching signal. Low values near `1`
mean parent paths do not multiply much under the selected graph policy, which
supports carrying exact histograms deeper. Higher values, especially in deeper
root-distance buckets, indicate that exact histogram propagation should be gated
by support width, cache reuse, or a fitted distribution policy.

See `PARENT_BRANCHING_DISTRIBUTION_THEORY.md` for the binomial
small-branching approximation, FFT evaluation of compound convolutions, and the
shifted exponential / shifted Gamma larger-branching approximation.

Another future direction is approximation from bounds plus low-order statistics.
The min/max interval gives finite support, while observed branching and second
moment estimates can help choose a candidate family. If real parent-path
distributions tend toward a recognisable finite-support form, such as
binomial-like, truncated-normal, or truncated-geometric, the benchmark can add
approximation modes that compare fitted distributions against exact SimpleWiki
histograms before trying enwiki-scale runs. The fitted state must still be
anchored to the same root/boundary, horizon, path statistic, and target scope as
the exact state it replaces.

Treat fitted distributions as a later phase, not part of the first parity
harness. Scalar min/max bounds are safe to introduce earlier because they can be
validated directly against the exact oracle and used only for pruning decisions
until the approximation policy is measured.

## 10. Output artifacts

Each benchmark run should emit machine-readable records and one summary table:

```text
distribution_cache_benchmark_<graph>_<timestamp>.jsonl
distribution_cache_summary_<graph>_<timestamp>.md
```

The summary should include:

- speedup vs `D_pre` for each `B_search`;
- cache bytes vs `D_pre`;
- hit rate vs `D_pre`;
- marginal speedup per additional precompute depth;
- exactness failures, if any;
- best observed policy under a fixed storage budget.

## 11. Decision criteria

Use the benchmark to choose defaults:

```text
D_pre_default = smallest D_pre where marginal speedup falls below threshold
B_search_default = largest budget where exact cache cutoffs remain tractable
cache_policy_default = highest speedup under storage budget with no churn spike
cumulative_basis_default = bases whose lookup savings exceed storage cost
```

Initial likely defaults:

```text
D_pre_default: 2 on SimpleWiki, remeasure for enwiki
B_search_default: 4 or 6 for parent-only exact validation
cumulative bases: mass first; moment(1) and weighted_power(N) only for hot functionals
admission: root_proximity + reuse_score with hysteresis
```

These are hypotheses, not commitments. The benchmark exists to replace intuition
with measured cutoffs.
