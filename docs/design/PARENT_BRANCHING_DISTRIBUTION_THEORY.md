# Parent Branching Distribution Theory

This note records the statistical model behind the distribution-cache planning
policy in `DISTRIBUTIONAL_FIT_POLICY.md` and the support-bounds benchmark in
`DISTRIBUTION_CACHE_BENCHMARK_PLAN.md`.

The immediate question is how far exact parent-path histograms should be carried
before the planner switches to scalar bounds, cached boundaries, or fitted
distribution states.

## 1. Exact object

For parent-only paths to a fixed root, the exact path-length histogram for node
`v` is the coefficient vector of a generating function:

```text
H_root(z) = 1
H_v(z) = z * sum(H_parent(z) for parent in parents(v))
```

The coefficient of `z^L` is the number or mass of admissible parent paths of
length `L` from `v` to the root. This recurrence is exact under the active graph
filter, root, edge direction, cycle policy, and path-admissibility rule.

Two quantities matter for planning:

```text
support_width(v) = L_max(v) - L_min(v) + 1
path_mass(v) = sum_L H_v[L]
```

Support width controls the number of histogram bins. Path mass controls the size
of the counts and the amount of path multiplicity hidden inside those bins. For
unweighted hop-count histograms, many distinct paths of the same length collapse
into one bin, so path mass can grow while histogram support remains narrow.

## 2. Parent branching signal

Let `p_v` be the number of parent choices for node `v` under the same graph
filter used by the benchmark. The measured parent-degree moments are:

```text
E[p]
E[p^2]
b_p = E[p^2] / E[p]
```

`b_p` is the size-biased parent branching signal. It estimates the parent degree
seen by a traversal that arrives at nodes through parent edges. The excess
branching signal is:

```text
epsilon = b_p - 1
```

For the current SimpleWiki depth-3 Articles sample:

```text
b_p = 1.028750
epsilon = 0.028750
```

That is close to a chain: most visited nodes have one parent, with rare extra
parent choices.

## 3. Small-branching approximation

When parent degree is mostly `1`, sometimes `2`, and rarely larger, the excess
branching can be approximated as a Bernoulli event per layer:

```text
X_i = 1 if step i introduces an extra parent branch
X_i = 0 otherwise
Pr(X_i = 1) = epsilon
Pr(X_i = 0) = 1 - epsilon
```

Then the number of extra-branch events over depth `n` is approximately:

```text
K_n = X_1 + ... + X_n
Pr(K_n = k) = C(n, k) * epsilon^k * (1 - epsilon)^(n-k)
E[K_n] = n * epsilon
Var(K_n) = n * epsilon * (1 - epsilon)
```

This is the binomial framing. It is useful for SimpleWiki-like samples because
`epsilon` is small and measured `max_p` is low.

For `epsilon = 0.028750`:

```text
n = 10:  E[K_n] = 0.2875
n = 25:  E[K_n] = 0.7188
n = 50:  E[K_n] = 1.4375
n = 100: E[K_n] = 2.8750
```

This says exact histograms may remain cheap for substantial depth when support
width also remains small. It does not say the histogram has `(1 + epsilon)^n`
bins. `(1 + epsilon)^n` is path-multiplicity pressure, while bin count is still
bounded by the observed finite support interval.

## 4. Compound branching approximation

The binomial model assumes each branching event contributes at most one extra
parent. For larger graphs, especially enwiki, nodes may have many parents. Then
the per-step excess should be modeled as a discrete random variable:

```text
Y_i = p_i - 1
Y_i in {0, 1, 2, ...}
```

The total excess branching pressure over `n` steps is:

```text
K_n = Y_1 + ... + Y_n
```

This is an n-fold convolution. If `G_Y(t)` is the probability-generating function
of the measured excess-parent distribution, then:

```text
G_K(t) = G_Y(t)^n
E[K_n] = n * E[Y]
Var(K_n) = n * Var(Y)
```

This is the more general form we should measure for enwiki. The binomial model
is the special case where `Y_i` is Bernoulli.

## 5. Gamma-style continuous approximation

For larger branching, the discrete convolution may become expensive or too noisy
to store directly. A continuous approximation can be useful as a planning model.
If each layer contributes a positive, narrow branching-cost distribution with
mean near `b_p`, then repeated convolution tends toward a smooth family. A Gamma
approximation is a plausible first candidate because sums of positive variables
with similar scale parameters remain Gamma-shaped:

```text
LayerBranch_i ~ Gamma(shape = alpha, scale = theta)
E[LayerBranch_i] = alpha * theta ~= b_p
Var(LayerBranch_i) = alpha * theta^2

Sum_n = LayerBranch_1 + ... + LayerBranch_n
Sum_n ~ Gamma(shape = n * alpha, scale = theta)
```

If the standard deviation is very narrow, `alpha` is large and the sum becomes
concentrated around `n * b_p`. In that regime, a closed-form approximation may
be good enough for planning thresholds, but the exact histogram or empirical
convolution should remain the validation oracle.

This Gamma framing is not a claim that Wikipedia parent paths are Gamma
distributed. It is a candidate approximation for the high-branching regime when
the per-layer branching cost is positive, moderately homogeneous, and repeatedly
convolved.

## 6. Histogram support versus path multiplicity

The main risk is mixing up two different growth processes:

```text
path_count_pressure(n) ~= b_p^n
extra_branch_events(n) ~= convolution of per-layer excess branching
histogram_bins(v) <= L_max(v) - L_min(v) + 1
```

For hop-count histograms, multiple paths with the same length share a bin.
Support width grows only when shortcut structure creates multiple distinct path
lengths. Parent branching alone can increase path mass without increasing bin
count.

For weighted path statistics, support can grow faster because different edge
weights create more distinct statistic values. In that case, the planner should
track both:

```text
support_width or support_cardinality
path_mass or path_multiplicity_pressure
```

## 7. Planner consequences

The exact-histogram policy should combine three signals:

```text
1. support width: number of bins to store and scan
2. parent branching: expected path-multiplication pressure
3. reuse value: expected saved search work from caching this node
```

A practical first rule is:

```text
if support_width <= exact_support_width_limit:
    store exact histogram
elif b_p_bucket is near 1 and expected_reuse is moderate:
    store exact histogram or exact sparse sketch
elif b_p_bucket is high and expected_reuse is low:
    store min/max bounds first
else:
    evaluate exact histogram, cached boundary, or fitted distribution by cost
```

For SimpleWiki, the measured sample has narrow support and `b_p` close to `1`,
so exact histograms can probably be propagated deeper than a root-distance-only
policy would suggest.

For enwiki, the same rule should be remeasured by root-distance bucket and by
topical/admin filter. High `E[p^2]/E[p]`, wide support intervals, or high
observed `max_p` should push the planner toward scalar bounds, scoped cached
boundaries, or fitted/compound approximations.

## 8. Validation plan

The next benchmark additions should measure:

```text
parent_degree_distribution by L_min bucket
excess_parent_distribution where Y = p - 1
empirical convolution of Y for small n
binomial approximation error when max_p <= 2 or 3
Gamma approximation error when branching is larger and positive
correlation between support_width and parent branching moments
correlation between path_mass and b_p^n
```

The pass/fail criterion is functional, not aesthetic: an approximation is useful
only if it predicts the policy decision made by exact histograms on SimpleWiki
and small enwiki samples.
