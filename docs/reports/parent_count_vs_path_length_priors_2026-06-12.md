# Parent-Count Priors Versus Path-Length Histograms

This note fixes terminology for the parent-only histogram/cache work.  The next
benchmark passes should keep three objects separate.

## Parent-Count Distribution

The parent-count distribution is over:

```text
P(v) = number of parents of node v
```

For root-conditioned planning, the more relevant variant is the number of
parents that can still reach the chosen root.  Full parent count and
root-reaching parent count should both be measured, because full count captures
raw category branching while root-reaching count captures branching that can
contribute to the current root cone.

For enwiki, this distribution can be substantially wider than the SimpleWiki
near-chain regime.  The size-biased signal:

```text
E[P^2] / E[P]
```

acts like a branching pressure seen by a path that arrives through an edge.  If
it is close to one, exact sparse histograms may stay cheap deep into the tree.
If it is around four, as in the stronger enwiki MTC measurements, exact
histograms can widen much faster and the cache planner should expect more
pressure.

Gamma-like, lognormal, or heavy-tail fits belong primarily to this
parent-count/branching object.  They describe variation in local branching, not
directly the finite path-length histogram for a node.

## Path-Length Distribution

The path-length distribution is over:

```text
H_v[L] = number of root-reaching parent paths from v with length L
```

For parent-only recurrence without cycle complications, a child histogram is a
shifted sum of parent histograms:

```text
H_v[L] = sum over parents p of H_p[L - 1]
```

A normalized version can be read as a probability distribution over path length,
but the unnormalized histogram is usually the operational object because path
mass matters for aggregate weights.

This path-length distribution is not expected to have the same family as the
parent-count distribution.  It is produced by repeated shifted sums/mixtures of
parent states.  With enough finite-variance layers and weak dependence, the
shape should drift toward binomial or normal-like behavior.  Skew remains
important when:

- depth is shallow;
- parent cones are dependent or share many ancestors;
- shortcuts and hubs dominate;
- cycle handling removes paths unevenly; or
- the parent-count tail is heavy enough to keep rare large effects visible.

A binomial approximation is not only a symmetric large-depth approximation.  For
small trial counts it can carry visible skew; for example, `n=10` with small
`p` is clearly skewed.  As trial count grows and no rare tail dominates, the
same family becomes closer to the usual normal approximation.

## Planning Prior

The planning prior is a forecast used to decide whether to compute, cache, or
approximate a node-local path-length histogram.  It may be derived from
parent-count statistics, but it is not itself a measured node histogram.

The current depth-conditioned prior uses root-reaching parent-count signals to
forecast path-length support and storage cost.  This is useful for admission
decisions:

- exact histogram if predicted and observed storage are cheap;
- capped histogram when exact recurrence is risky but still useful;
- parametric boundary state when a histogram is over budget; and
- skip cache when even the approximate state is not worth the slot.

The latest parametric boundary benchmark deliberately uses oracle mass: it
aligns a prior shape to a measured `L_min` and scales it by the measured path
count.  That isolates shape/storage behavior, but it is not a production mass
model.  A production pass needs to estimate mass separately from shape.

## Benchmark Implications

Future flags and reports should avoid a single ambiguous `family` label.  Better
names are:

- `parent_count_family` for fits over parent counts or size-biased branching;
- `path_length_family` for fits over node-local path histograms;
- `planning_prior_family` for the depth-conditioned forecast used by admission;
  and
- `mass_model` for the total path-count estimate used to unnormalize a
  probability distribution.

The next useful benchmark split is therefore:

```text
path_length_family = empirical_prior | binomial | normal | histogram
mass_model         = oracle | depth_prior | recurrence_estimate
```

Gamma should stay in the candidate set for parent-count or branching variation.
It may still be used as a sampled continuous approximation when the empirical
path-length histogram is wide, but it should not be the default claim for
enwiki path-length histograms unless the measured path-length errors support it.
