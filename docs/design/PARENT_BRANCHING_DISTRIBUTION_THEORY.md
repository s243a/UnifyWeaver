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

The coefficient of `z^L` is the count or mass of admissible parent paths of
length `L` from `v` to the root. In the current benchmark `H_v[L]` stores counts;
a probability-normalised variant would change the meaning of `path_mass` below.
The recurrence form assumes admissibility has already been enforced by the
active graph filter, root, edge direction, cycle policy, and path-admissibility
rule.

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
filter used by the benchmark. The benchmark moments are uniform over the
selected reachable nodes, not edge-weighted. The measured parent-degree moments
are:

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

For buckets where every measured node has at least one parent, `epsilon >= 0`
follows from the size-biased mean being at least the ordinary mean. Buckets with
zero mean parent degree, such as a root-only bucket, should report this ratio as
undefined rather than as a real branching factor.

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

This is the binomial framing. It assumes approximately independent per-layer
excess-branching events. Correlation inside topical graph regions can inflate
variance, so the binomial model should be treated as a near-chain approximation,
not an exact law. For a specific node, set `n` to the relevant root-distance
horizon: `L_min` gives a lower-bound estimate, `L_max` gives an upper-bound
estimate, and narrow support makes the difference small. It is useful for
SimpleWiki-like samples because `epsilon` is small and measured `max_p` is low.

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
bounded by the observed finite support interval; see section 7.

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

This is an n-fold convolution. `G_Y(t)` is the probability-generating function
of the excess distribution on `{0, 1, 2, ...}`, not the raw parent-degree
PGF on `{1, 2, 3, ...}`; using the raw-degree PGF would shift the convolved
support by `n`. In the stationary i.i.d. case:

```text
G_K(t) = G_Y(t)^n
E[K_n] = n * E[Y]
Var(K_n) = n * Var(Y)
```

For non-stationary layers, replace the power by a product of per-layer PGFs and
use `Var(K_n) = sum_i Var(Y_i)` under layer independence. This is the more
general form we should measure for enwiki. The binomial model is the special
case where `Y_i` is Bernoulli.

## 5. FFT convolution route

If the planner needs the compound distribution rather than only its first two
moments, the convolution can be evaluated in the frequency domain. For discrete
probability vectors:

```text
P(K_n) = P(Y_1) * P(Y_2) * ... * P(Y_n)
FFT(P(K_n)) = FFT(P(Y_1)) * FFT(P(Y_2)) * ... * FFT(P(Y_n))
```

where `*` on the first line is convolution and `*` on the second line is
pointwise multiplication.

If the per-layer excess-parent distribution is stationary inside a bucket, this
becomes:

```text
P(K_n) = P(Y) convolved with itself n times
FFT(P(K_n)) = FFT(P(Y))^n
```

For non-stationary buckets, the planner can multiply one transformed vector per
root-distance layer:

```text
FFT(P(K_n)) = product_i FFT(P(Y_i))
```

Implementation notes:

- zero-pad to at least the linear-convolution support length before applying the
  FFT, otherwise circular convolution will wrap tail mass into low bins;
- use direct convolution for tiny supports, since it is simpler and often faster;
- use FFT convolution when support is wide, many layers are composed, or the same
  distribution is repeatedly convolved; initial crossover thresholds such as
  support greater than roughly `32` bins or more than roughly `8` repeated
  self-convolutions are only profiling starting points;
- after inverse FFT, clamp small negative numerical noise relative to the maximum
  recovered value, then renormalise if the result is used as a probability
  distribution;
- keep exact integer or rational convolution for tiny validation fixtures.

The FFT route is an implementation strategy for the compound model, not a claim
about the final closed-form family. It should be validated against exact
histograms on SimpleWiki and sampled enwiki subgraphs before being used for
policy decisions.

## 6. Shifted exponential / Gamma tail approximation

For larger branching, the discrete convolution may become expensive or too noisy
to store directly. A continuous approximation can be useful as a planning model,
but it should be placed on the excess-parent variable rather than on raw parent
degree:

```text
p >= 1
Pr(p < 1) = 0
X = p - 1
X >= 0
E[X] = b_p - 1
```

For the current SimpleWiki sample:

```text
E[X] = 1.028750 - 1 = 0.028750
```

This support constraint makes a shifted exponential or shifted Gamma more
appropriate than an unshifted exponential over raw parent degree. In raw
parent-degree terms:

```text
p = 1 + X
X ~ Exponential(lambda)
```

or, more generally:

```text
p = 1 + X
X ~ Gamma(shape = alpha, scale = theta)
E[X] = alpha * theta ~= b_p - 1
Var(X) = alpha * theta^2
```

The convolved excess process over depth `n` is:

```text
S_n = X_1 + ... + X_n
```

If the `X_i` are exponential with a common scale, the sum is Gamma:

```text
S_n ~ Gamma(shape = n, scale = theta)
```

If the `X_i` are Gamma with a common scale, their sum is also Gamma:

```text
S_n ~ Gamma(shape = sum_i alpha_i, scale = theta)
```

The planner-facing constraint is not merely matching the mean. The tail must
fall off quickly after the useful exact-support budget. A natural budget is the
extra support beyond the shortest admissible path that the planner is willing to
keep exact, for example `exact_support_width_limit - 1` for hop-count support
bins. A fitted closed form is acceptable only when it satisfies a budgeted tail
condition such as:

```text
Pr(S_n > excess_support_budget) <= epsilon_tail
```

or an equivalent condition around the expected excess support:

```text
Pr(S_n > E[S_n] + slack) <= epsilon_tail
```

If the standard deviation is very narrow, `alpha` is large and the sum remains
concentrated near `E[S_n]`. In that regime, a shifted closed-form approximation
may be good enough for planning thresholds, but the exact histogram, direct
convolution, or FFT convolution should remain the validation oracle.

This shifted Gamma framing is not a claim that Wikipedia parent paths are Gamma
distributed. It is a candidate approximation for the high-branching regime when
the per-layer excess branching cost is non-negative, moderately homogeneous,
fast-decaying, and repeatedly convolved.

A provisional regime map is:

| Regime | Signal | Recommended model |
|--------|--------|-------------------|
| Near-chain | `b_p` near `1`, low `max_p` | Binomial small-branching model |
| Moderate branching | wider excess support or repeated composition | Empirical compound model, direct or FFT convolution |
| High branching | wide support where exact convolution is too costly | Shifted exponential / Gamma fit, validated by tail error |

The thresholds should be calibrated by SimpleWiki and enwiki samples rather than
hard-coded from this note.

## 7. Histogram support versus path multiplicity

The main risk is mixing up two different growth processes:

```text
path_count_pressure(n) ~= b_p^n  (stationary order-of-magnitude estimate)
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

## 8. Planner consequences

The exact-histogram policy should combine three signals:

```text
1. support width: number of bins to store and scan
2. parent branching: expected path-multiplication pressure
3. reuse value: expected saved search work from caching this node
```

Here `b_p_bucket` means the size-biased parent branching ratio computed over the
nodes in a given `L_min` root-distance bucket. A practical first rule is:

```text
if exact_error is not None:
    store min/max bounds first and emit a diagnostic
elif support_width <= exact_support_width_limit:
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

## 9. Validation plan

The next benchmark additions should measure:

```text
parent_degree_distribution by L_min bucket
excess_parent_distribution where Y = p - 1
empirical convolution of Y for small n
depth-conditioned prior distributions from the size-biased excess-parent law
expected support width and tail mass as a function of root-distance horizon
exact path-length histograms versus fitted binomial vectors
exact path-length histograms versus shifted-Gamma-style vectors
accuracy/cost tradeoff between binomial and Gamma approximations
binomial approximation error when max_p <= 2 or 3
direct convolution versus FFT convolution parity over the same empirical Y
FFT zero-padding and numerical-noise error bounds for wider supports
shifted exponential / shifted Gamma approximation error for X = p - 1
tail probability beyond the exact-support budget
zero-parent and root-only buckets where size-biased branching is undefined
cross-graph transferability of calibrated `b_p` thresholds
sensitivity of `epsilon_tail` to increasing root distance
correlation between support_width and parent branching moments
correlation between path_mass and b_p^n
```

The pass/fail criterion is functional, not aesthetic: an approximation is useful
only if it predicts the policy decision made by exact histograms on SimpleWiki
and small enwiki samples.

`scripts/distribution_fit_comparison.py` is the first executable version of
that comparison. It intentionally separates two related ideas. A realized
distribution fit approximates the statistics of a histogram that was actually
computed for a node. A depth-conditioned prior distribution is a Bayesian prior:
a forecast based on a root-distance horizon and the size-biased excess-parent
law before observing the node's exact histogram. This is the object used to
estimate whether future exact histograms are likely to be cheap enough to store.

The planner should use the prior distribution for admission thresholds such
as "carry exact histograms to depth `d` while the prior effective support is
narrow." The realized fit is still needed as validation: if binomial or Gamma
approximations do not match exact histograms on sampled nodes, then the prior
depth policy is using the wrong family even if its mean looks plausible.

The first prior model is stationary within the selected calibration set: the
same excess-parent law is convolved at each depth. A future non-stationary
variant can use a different law per root-distance layer,
`P(Y_i | L_min = i)`, and compose those layer priors. This would account for
the empirical pattern where the size-biased parent-branching signal can decline
farther from the root. That refinement is planner-relevant, but should be held
until deeper SimpleWiki and enwiki samples show that the stationary prior is
materially miscalibrated.
