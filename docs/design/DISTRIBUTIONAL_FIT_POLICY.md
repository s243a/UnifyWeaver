# Distributional Fit Policy

This note specifies the policy layer between exact path-statistic distributions and closed-form approximations. It is a companion to `ROOT_ANCHORED_METRICS_SPECIFICATION.md`, `RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`, `DISTRIBUTIONAL_COMPRESSION_THEORY.md`, and `DISTRIBUTIONAL_COMPRESSION_IMPLEMENTATION_PLAN.md`.

The problem: root-anchored metrics such as `d_wPow` often start with an exact finite histogram over path statistics. Near the root, or on small graphs, that histogram is cheap and should be kept exactly. Deeper in the graph, especially on enwiki, the support can grow enough that the runtime should switch to a compact representation. That switch must be explicit, diagnosable, and user-overridable.

## 1. Distribution state

The runtime should treat a node's path statistic as a representation choice:

```prolog
distribution_state(exact_histogram(Bins)).

distribution_state(tail_pruned_histogram(
    ExactPrefix,
    DroppedTailRange,
    DroppedTailMass,
    FunctionalErrorBounds)).

distribution_state(hybrid_truncated(
    ExactPrefix,
    TailFamily,
    TailRange,
    TailMass,
    Parameters,
    ErrorBounds)).

distribution_state(parametric(
    Family,
    SupportRange,
    TotalMass,
    Parameters,
    ErrorBounds)).
```

`Bins` and `ExactPrefix` are finite maps from statistic value to mass. The statistic may be a hop count, weighted length, or a tuple such as `(parent_hops, child_hops)`.

Every stored distribution also has an anchoring key. A node's distribution is
not just "the distribution for node `V`"; it is the distribution for `V`
relative to a root, boundary node, direction, statistic, and admissibility
policy:

```prolog
distribution_anchor(
    RootOrBoundary,
    Direction,
    PathStatistic,
    CyclePolicy,
    Horizon,
    ScopeKey).
```

For a global root table, `ScopeKey` can be `global(Root)`. For a per-query
calculation, `ScopeKey` may identify the ancestor cone of the target node. This
matters because a deeper query may only need the distribution induced by the
ancestors of the node of interest, not the distribution obtained by
materialising the whole graph.

For parent-only paths to a fixed root, the support is finite. Tail fits should therefore be finite-support fits, not infinite asymptotic tails:

```prolog
TailRange = range(KPlusOne, DMax)
```

`DMax` may be the exact longest parent path to root, the active finite horizon, or a conservative structural bound. If only a budget `B` is relevant, `DMax` may be `B` for that computation.

## 2. Default representation policy

The initial policy should be conservative:

```prolog
if support_size <= K:
    keep exact_histogram
else:
    keep exact prefix through K
    fit a truncated tail over K+1 .. DMax
```

`K = 10` is a reasonable first default because it keeps small hand-checkable histograms exact while preventing unbounded per-node vectors. It must be a configuration value, not a constant baked into generated code.

A light-tail histogram can sometimes be shortened before fitting any parametric
family. If the suffix mass after a candidate cut is below `epsilon_tail`, and
the discarded contribution to the requested functional is below its error
budget, store `tail_pruned_histogram` rather than a fitted tail. This keeps the
observed prefix exact while recording enough metadata to make the dropped suffix
visible in diagnostics. For binomial-like or extremely light tails, this can
remove a large fraction of bins; for skewed or medium/heavy tails, the same rule
will refuse to prune because the suffix mass or functional contribution remains
too large.

After cheap exact encodings are considered, the candidate ladder should stay
bounded and discrete first.  `truncated_geometric` remains useful for a visible
finite-support exponential tail, but it should not be the universal first
fallback.  Do not assume a Poisson family by default: parent path distributions
are finite, graph-constrained, and driven by branching structure rather than
independent arrival counts.

Candidate representations:

| Family | Use when |
|--------|----------|
| `tail_pruned_histogram` | The retained exact prefix plus dropped-tail certificate meets the error budget |
| `quantized_cdf_table` | Prefix-mass queries dominate and CDF error is the right certificate |
| `truncated_geometric` | Tail decays approximately exponentially after the exact prefix |
| `binomial` | Bounded excess-event support where mean/variance recover compatible `n,p` |
| `beta_binomial` | Bounded support with measurable over-dispersion beyond binomial variance |
| `truncated_discrete_normal` | Large-depth CLT regime after CDF/W1 validation |
| `mixture(binomial)` | Bounded discrete modes fit better than one family |
| `discretized_gmm` | Narrow residual spikes or bottleneck modes survive cheaper discrete fits |
| `empirical_sketch` | No simple family fits but quantile/CDF accuracy is enough |

The normal family should be treated as a large-depth approximation, not as the
default for rare excess-parent events.  For real deep nodes, repeated shifted
sums make binomial or normal approximations increasingly plausible by the
central limit theorem.  That is a reason to try those families earlier in deep
ancestor cones, not a reason to accept them without an error certificate.  In
near-chain SimpleWiki regimes, binomial or empirical discrete priors preserve
the skewed finite support more directly.  If measured parent degrees become
scale-free rather than Poisson-like, the policy should favor empirical
sketches, mixtures, or explicit hub handling instead of a single light-tailed
count family.

Gaussian mixtures are not deprioritized because they are wrong.  They are
deprioritized because the primary object is a bounded integer histogram, and
cheaper discrete encodings usually expose the same planner information with
fewer parameters.  A mixture of binomials with shared support uses `2K - 1`
parameters, while a Gaussian mixture needs roughly `3K - 1`.  GMMs should be
available as an escalation family when the residual error has sub-binomial-width
spikes, bottleneck modes, or other structure that cheaper discrete candidates
cannot pass through the CDF/W1 gate.

The family choice should be evidence-driven. Simplewiki is a calibration fixture: compute exact parent-only distributions there, fit candidate tails, and measure error. Enwiki is the stress case where the representation switch is expected to matter.

## 3. Metrics are functionals over the state

The representation policy must not assume the query asks for minimum distance. The distribution state is reusable; the reported metric is a functional over it.

Examples:

```prolog
metric_functional(min_support).
metric_functional(bounded_average(Budget)).
metric_functional(reachability_mass(Budget)).
metric_functional(tail_mass(Budget)).
metric_functional(entropy).
metric_functional(weighted_power_mean(N, Budget)).
```

Approximation error should be assessed against the selected functional. A tail fit that preserves reachability mass may still distort entropy; a fit that preserves bounded average may still move the minimum support point. The selector must therefore carry both:

```prolog
representation_policy(...)
metric_functional(...)
```

## 4. Cumulative bases are acceleration layers

A cumulative distribution function is useful, but it is not a replacement for
the distribution state. It accelerates budgeted queries whose weighting function
has been chosen in advance.

The basic mass CDF is:

```prolog
cumulative_basis(mass).
F0_v[B] = sum_{S <= B} P_v[S]
```

With that basis, reachability mass and interval mass are cheap:

```prolog
mass(S <= B) = F0_v[B]
mass(B1 < S <= B2) = F0_v[B2] - F0_v[B1]
```

Other expectation forms need their own cumulative basis:

```prolog
cumulative_basis(moment(1)).
F1_v[B] = sum_{S <= B} S * P_v[S]

cumulative_basis(weighted_power(N)).
FN_v[B] = sum_{S <= B} (S + 1)^(-N) * P_v[S]

cumulative_basis(custom(Name)).
FG_v[B] = sum_{S <= B} g_Name(S) * P_v[S]
```

This makes common functionals constant-time or interval-difference lookups, but
each stored basis costs memory or storage. The representation policy should
therefore decide which bases to materialise:

```prolog
cached_distribution(
    raw_state(DistributionState),
    cumulative_basis([mass, moment(1), weighted_power(N)]),
    storage_policy(StoragePolicy)).
```

The default should be conservative: always expose mass when the support is wide
enough to make repeated scans expensive; store first moments or weighted-power
bases only when the workload asks for those functionals often enough to justify
the space. For exact histograms with small support, scanning bins may be cheaper
than storing cumulative arrays.

For an arbitrary function `g(S)`, a CDF alone is insufficient. The runtime must
either scan the exact bins, use an analytic integral supplied by the parametric
family, evaluate a stored `custom(Name)` basis, approximate numerically, or emit
a diagnostic that the requested functional has no supported cumulative basis.

## 5. User override surface

The long-term design goal is that users can modify the policy without rewriting target code. The compiler should expose a predicate hook that selects or overrides the representation policy.

Proposed directive:

```prolog
:- distribution_fit_policy(Name/Arity, Options).
```

Example:

```prolog
:- distribution_fit_policy(root_path_distribution/3,
     [ exact_support_limit(10),
       default_tail_family(truncated_geometric),
       max_tail_error(0.01),
       selection_predicate(my_distribution_policy/5) ]).
```

The selection predicate has the shape:

```prolog
my_distribution_policy(+Node,
                       +DistributionSummary,
                       +MetricFunctional,
                       +CostSignals,
                       -RepresentationChoice).
```

`RepresentationChoice` is one of:

```prolog
exact_histogram
hybrid_truncated(Family, PrefixLimit)
parametric(Family)
delegate(DefaultPolicy)
```

The compiler may provide a default predicate:

```prolog
default_distribution_policy(Node, Summary, Functional, Signals, Choice)
```

User predicates can call the default and override only selected cases. This keeps the policy extensible without making target adapters depend on project-specific Prolog code.

## 6. Cost and diagnostics

The representation switch should be driven by both cost and value:

```prolog
switch_to_tail_fit if
    support_size > exact_support_limit
    or histogram_memory > memory_budget
    or propagation_cost > cost_budget
and
    estimated_functional_error <= max_tail_error
```

Diagnostics should be emitted into the recurrence strategy trace:

- exact support size;
- tail-pruning threshold, dropped range, dropped mass, and functional error;
- chosen family;
- fitted finite support range;
- materialised cumulative bases;
- estimated error for the selected functional;
- inherited parent approximation error before fitting;
- local fit error after compression;
- CDF and W1 error certificates when available;
- fallback reason if no family passed the threshold;
- fallback reason if a requested functional has no matching cumulative basis;
- whether a user selection predicate overrode the default.

The important invariant is that approximation is not silent. A query that uses a closed-form tail should leave a trace explaining why the representation changed.

## 7. Cached distributions as search boundary conditions

A cached distribution can act as a boundary condition for later path search. During a per-query path aggregate, if the traversal reaches a node `N` with a valid distribution state for the same root and statistic, the search does not need to enumerate below `N`. It can integrate the cached distribution over the remaining path budget and add that contribution to the aggregate.

Conceptually:

```prolog
remaining_budget = TotalBudget - CostSoFar
contribution = integrate_distribution(
    CachedDistributionAtN,
    MetricFunctional,
    remaining_budget)
```

This is expectation-like, but the integration is specific to the functional being computed. For reachability mass it can use the mass CDF up to the remaining budget. For a bounded average it needs both mass and first-moment cumulative bases. For `weighted_power_mean(N, Budget)` it needs the matching weighted-power basis. For entropy it needs either a raw distribution scan, an analytic family-specific entropy calculation over the finite slice, or a stored custom basis.

The cache hit is valid only when the cached state was built under compatible semantics:

- same root or an explicitly compatible boundary;
- same edge direction and path statistic;
- compatible cycle policy and path admissibility rules;
- compatible horizon or a horizon at least as wide as the remaining budget;
- compatible target scope, unless the cached entry is a global table that
  dominates the query's scoped ancestor cone;
- compatible representation policy, or a representation with known error bounds for the requested functional.

This turns exact or approximated distribution tables into reusable suffix summaries. Search remains exact when the cached distribution is exact and the requested functional can be evaluated exactly from the stored state. It becomes a controlled approximation when the cached distribution is a fitted representation, or when the requested functional is evaluated through an approximate basis.

For target-scoped evaluation, the boundary condition is ancestor-relative. If a
query asks for node `V`, a cached state for ancestor `A` is reusable only for the
portion of `V`'s parent-path search that reaches `A` under the same constraints.
The contribution is then the aggregate from `A` to the root, sliced by the
remaining budget. This is exact for exact states when `A`'s stored distribution
was computed over the same admissible ancestor cone, or over a broader global
cone whose extra paths cannot be reached from `V` through `A`.

## 8. Cache admission and eviction

Distribution caches should not use blind overwrite-on-collision. A collided insert
is a policy decision: keeping a root-near, high-reuse suffix summary may be more
valuable than admitting a newly computed deep node.

The cache should score both the incumbent and the candidate:

```prolog
cache_score(Node, Entry, Score) :-
    Score is expected_reuse(Node)
          * recompute_cost_saved(Entry)
          * root_proximity_bonus(Node)
          * accuracy_value(Entry)
          / storage_cost(Entry).
```

Useful score signals:

- parent distance to root, with a bonus for nodes closer to the root;
- estimated descendant or query-reuse count;
- observed hit frequency and recency;
- cost to recompute the distribution or scoped fixed-point cone;
- representation quality, with exact states usually worth more than fitted ones;
- cumulative-basis storage cost, since each materialised basis consumes space;
- semantic compatibility width, meaning how many likely queries can reuse the
  same root/statistic/cycle-policy/budget-horizon entry.

On collision, admit only if the candidate is meaningfully better:

```prolog
admit_candidate if candidate_score > incumbent_score * hysteresis
```

The hysteresis factor prevents churn when two similarly useful entries map to the
same slot. If the candidate loses, the runtime can still return the just-computed
value to the current query; it simply does not install it in the shared cache.

Eviction should also be layered. Raw exact distributions are expensive to
recompute and should generally outlive derived cumulative bases. Cumulative
bases can be evicted first because they can be rebuilt from raw state. Parametric
tail parameters are compact and may be worth retaining if their error bounds are
still valid. A practical eviction order is:

```prolog
1. cold custom cumulative bases
2. cold moment / weighted-power bases
3. cold mass CDFs when raw state is still available
4. approximate fitted states with weak reuse
5. exact raw states, especially near-root entries, only under pressure
```

For Wikipedia-style root-anchored metrics, the default bias should be: retain
near-root exact distributions, retain high-reuse ancestor nodes, and evict deep
low-hit cumulative bases before overwriting root-proximal entries.

## 9. Scoped fixed-point generation

Fixed-point distribution generation does not need to materialise the whole graph for a single node query. For a target node `V`, first restrict work to the ancestor cone relevant to `V`: all nodes that can reach the root by parent edges and can also reach `V` by reversing the parent relation. Then compute distributions from the root outward only inside that scoped subgraph, stopping when the target node's distribution has been produced or the configured depth/horizon is exhausted.

The parent-only shape is:

```prolog
scope(V) = ancestors_to_root(V) intersect descendants_from_root(root)

for each node U in root-outward order within scope(V):
    P_U[S + step] = aggregate over parents(U) that are also in scope(V)
```

Equivalently, when evaluating one node, every parent distribution is computed recursively on demand and memoised, but only for parents that are ancestors of the node under evaluation. This gives the fixed-point recurrence the same distributional semantics without forcing an all-nodes materialisation.

Three execution modes fall out:

| Mode | Use when |
|------|----------|
| `per_query_distribution` | One or a few target nodes; compute only the target's ancestor cone |
| `scoped_fixed_point` | Many targets in the same topical subtree; reuse the scoped distribution table |
| `global_materialized` | Fixed root and high query volume justify whole-graph precomputation |

This scoped mode is the bridge between graph search and global fixed-point evaluation. It preserves the recurrence form, but its worklist is cut down by the query's ancestor cone and by cached boundary distributions encountered during evaluation.

At deeper layers, full distribution materialisation is not always the right
first state. The planner should be allowed to compute scalar support bounds
before deciding whether to construct an exact histogram, fit a tail, or stop at
a cached boundary:

```prolog
support_bounds(
    min_path_stat(Min),
    max_path_stat(Max),
    exact_under_policy(Boolean),
    horizon(Horizon)).
```

For min-only or max-only functionals, these scalar recurrences are sufficient:

```prolog
Min_v = 1 + min(Min_parent)
Max_v = 1 + max(Max_parent)
```

with the obvious weighted-step variant for non-unit parent costs and the same
cycle/admissibility policy as the search oracle. For bounded aggregate
functionals, the bounds are not a replacement for the distribution, but they are
valuable pruning signals:

- if `Min_v > remaining_budget`, the whole suffix contributes zero;
- if `Max_v <= remaining_budget`, the whole suffix is within budget and can use
  a total-mass or total-functional summary when available;
- if `[Min_v, Max_v]` is narrow, exact histogram materialisation may be cheap;
- if the interval is wide, the policy can prefer a fitted representation or a
  deeper cached boundary.

This gives the planner a cheap intermediate representation for deep nodes. It
can carry min/max bounds through large parts of the graph, then materialise full
distributions only where the query workload, error budget, or cache score makes
the extra state worthwhile.

Ancestor-cone size is a separate admission signal. A node can be fairly deep in
root distance but still have a small target-scoped ancestor cone. In that case,
exact histograms may remain cheap because the planner only materialises the
distributions needed by the node of interest, not a global table. The first
calibration pass should therefore record both support_width and ancestor_cone_nodes;
a deep node with a small cone and narrow support is a
good exact-histogram candidate even when a global depth rule would reject it.

Continuous or sampled approximations should be charged as computation, not just
as storage. For example, a Gamma-like continuous approximation can be sampled
onto a finite grid before FFT convolution, but good accuracy may require tens or
hundreds of sample points. If the exact ancestor-scoped histogram has only a few
bins, sampling 100 points is unlikely to save compute. Its main value is then a
compact reusable representation, not avoiding the initial exact or sampled
construction.

Support width should be paired with a parent-branching signal before choosing
how deep to carry exact histograms. Let `p_v` be the number of parent choices for
node `v` under the active graph/filter/root policy. Then:

```text
mean_parent_degree = E[p]
size_biased_parent_branching = E[p^2] / E[p]
```

The ratio is the parent-only analogue of a traversal-effective branch factor: it
estimates the parent degree seen by a path that has already followed a parent
edge. It is not a replacement for exact histogram validation, but it is a useful
early warning signal. When `support_width` stays small and `E[p^2]/E[p]` stays
near `1`, exact histograms can often be carried deeper because many states are
one-point or nearly one-point distributions. When the ratio rises, especially in
deeper root-distance buckets, parent paths are likely to multiply and exact
histogram materialisation should require stronger evidence of reuse.

A first planner rule should be shaped like:

```text
materialize_exact(node) when
    support_width(node) <= exact_support_width_limit
    or root_distance(node) <= D_pre
    or expected_reuse(node) * saved_search_cost(node) > storage_cost(node)

defer_full_distribution(node) when
    support_width(node) is wide
    and bucket_size_biased_parent_branching is high
    and expected_reuse(node) is low
```

For SimpleWiki, the current measurements suggest the first condition dominates.
For enwiki, the parent-branching moment should be measured by root-distance
bucket before deciding how far exact histograms should be propagated.

`PARENT_BRANCHING_DISTRIBUTION_THEORY.md` gives the statistical interpretation:
small excess parent branching is binomial-like, while larger parent branching is
better treated as a compound/convolution model before fitting a closed form.

## 10. Open validation work

The next implementation-facing work is a parity harness:

The benchmark plan in `DISTRIBUTION_CACHE_BENCHMARK_PLAN.md` defines the first shallow precompute/search-budget grid for this work.

The first approximation harness is `scripts/distribution_fit_comparison.py`.
It keeps two concepts separate:

- realized distribution fits compare binomial and shifted-Gamma-style vectors
  against exact histograms for already selected nodes;
- depth-conditioned prior distributions use the size-biased excess-parent
  distribution to estimate whether histograms at future depths are likely to
  stay narrow enough to materialize cheaply before observing exact node
  histograms.

The current prior is stationary within the selected calibration set. A later
variant should allow layer-conditioned priors, where the excess-parent law
changes with root-distance bucket. That would handle cases where the average
parent branching signal declines as nodes move farther from the root, but it
should wait for deeper SimpleWiki and enwiki measurements.

1. exact parent-only histogram on tiny fixtures;
2. exact parent-only histogram on simplewiki samples;
3. fitted truncated-tail representation over the same nodes;
4. functional-level error checks for `min_support`, `bounded_average`, `reachability_mass`, `tail_mass`, `entropy`, and `weighted_power_mean`;
5. cumulative-basis tests showing mass, interval, moment, and weighted-power lookups agree with raw histogram scans;
6. cache-admission tests showing near-root/high-reuse entries survive collisions against lower-score entries;
7. policy-selection tests showing that a user predicate can override the default without changing target code.

Only after those checks pass should the policy be used for enwiki-scale materialisation.
