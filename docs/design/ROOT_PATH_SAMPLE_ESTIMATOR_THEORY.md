# Root Path Sample Estimator Theory

This note specifies the sampling estimator used by
`scripts/lmdb_boundary_coverage_probe.py` and its relationship to cached
root-path histograms.  It complements `ROOT_ANCHORED_METRICS_SPECIFICATION.md`,
`DISTRIBUTIONAL_COMPRESSION_THEORY.md`, and
`PARENT_BRANCHING_DISTRIBUTION_THEORY.md`.

The main distinction is between:

1. the **proposal correction** needed because a random walk does not sample all
   paths uniformly, and
2. the **path-value kernel** that says which quantity the query wants to sum.

Both may be called a "weight" in informal discussion, but they are different
objects and should stay separate in code and reports.

## 1. Admissible parent paths

Fix a target node `v0`, root `r`, and path-length budget `B`.  A sampled
parent path is:

```text
pi = (v0, v1, ..., vL)
```

where each `v_{i+1}` is an eligible parent of `v_i`.  Eligibility is defined by
the active traversal policy:

```text
A_i = parents(v_i)
      minus nodes already visited on this path
      minus parents rejected by the root-cone or reachability filter
      minus any other active parent filter
d_i = |A_i|
```

The simple-path rule is per path: a node cannot appear twice in the same sampled
path.  The path terminates when it reaches the root, hits a cache boundary, runs
out of budget, or has no eligible parents.

## 2. Proposal correction

The current sampler chooses uniformly from the eligible parent set at each
step.  For a complete sampled prefix of length `L`, the proposal probability is:

```text
q(pi) = product_{i=0}^{L-1} 1 / d_i
```

The inverse-proposal correction is therefore:

```text
c(pi) = 1 / q(pi) = product_{i=0}^{L-1} d_i
```

This is the branch-product value currently reported by the probe as the sample
"weight".  It is not a path-length preference.  It only corrects for the fact
that high-branching prefixes are less likely to be sampled by a one-parent
random walk.

If a future sampler enumerates all prefixes exactly, or samples complete paths
uniformly from the admissible path set, this correction changes.  In the exact
enumeration case it disappears because every path is already counted once.

## 3. Path-value kernels

Let `g(pi)` be the value the query wants to sum over root-reaching paths.  Common
examples are:

```text
path count:              g(pi) = 1
path length numerator:   g(pi) = L
decayed length kernel:   g(pi) = b_p^(-L)
weighted power:          g(pi) = (L + 1)^(-n)
```

The unbiased Monte Carlo estimator for the path-space sum is:

```text
S_hat_g = (1 / N) * sum_{samples j} c(pi_j) * g(pi_j) * I[pi_j reaches r]
```

For a mean property `f(pi)` over root-reaching paths, estimate the numerator and
denominator separately:

```text
Num_hat = (1 / N) * sum_j c(pi_j) * f(pi_j) * I[root]
Den_hat = (1 / N) * sum_j c(pi_j) * I[root]
Mean_hat = Num_hat / Den_hat
```

For a weighted mean under a kernel `g`, use:

```text
Num_hat_g = (1 / N) * sum_j c(pi_j) * g(pi_j) * f(pi_j) * I[root]
Den_hat_g = (1 / N) * sum_j c(pi_j) * g(pi_j) * I[root]
Mean_hat_g = Num_hat_g / Den_hat_g
```

The ratio form has finite-sample bias but is consistent.  Reports should carry
standard errors, confidence intervals, and eventually effective sample size so
that high-variance branch products are visible.

## 4. Relationship to `b_p = E[P^2] / E[P]`

The parent-branching prior `b_p` is a planning statistic, not the per-path
proposal correction.  In a stationary approximation where every eligible parent
count is close to `b_p`:

```text
q(pi) ~= b_p^(-L)
c(pi) ~= b_p^L
```

This explains why `b_p^L` estimates search-space growth and why
`b_p^(-L)` estimates the probability of one particular length-`L` path under a
constant-branching random walk.  The actual unbiased sample correction is still
`product_i d_i`, because the real path sees local branch counts, not the global
average.

It is valid to choose a metric kernel:

```text
g_bp(pi) = b_p^(-L)
```

This is a decay or de-multiplication choice: it makes long paths contribute less
by the expected branching pressure.  Under the constant-branching approximation,
the sampled contribution becomes:

```text
c(pi) * g_bp(pi) ~= b_p^L * b_p^(-L) = 1
```

That cancellation is an intuition for why equal path scoring can be reasonable
after an explicit path-length decay.  It is not a replacement for the
branch-product correction when the sampler is a local random walk.

## 5. Boundary splice estimator

Boundary sampling stops at the first compatible cached boundary node `b` instead
of walking all the way to the root.  Suppose the sampled prefix is `rho`, length
`ell`, with remaining budget `R = B - ell`.  Its proposal correction is:

```text
c(rho) = product_{i=0}^{ell-1} d_i
```

Let the boundary cache store an unnormalised suffix histogram:

```text
H_b[k] = number of admissible suffix paths from b to r of length k
```

For root-path count under the remaining budget, the suffix value is the CDF
mass:

```text
M_b(R) = sum_{k <= R} H_b[k]
```

The boundary-hit contribution to the estimated root-path search space is:

```text
c(rho) * M_b(R)
```

For a path-length kernel `g(L) = b_p^(-L)`, the suffix cache needs the matching
cumulative basis:

```text
G_b(R) = sum_{k <= R} H_b[k] * b_p^(-k)
```

and the boundary contribution is:

```text
c(rho) * b_p^(-ell) * G_b(R)
```

The same pattern applies to `weighted_power(n)`, first moments, entropy-like
queries, or any custom functional: the boundary cache must store enough suffix
mass or suffix numerator state for that functional.  A plain mass CDF is enough
for reachability/count queries, but not for arbitrary functions of the suffix.

## 6. Compatibility requirements

A boundary splice is exact only when the suffix histogram was built under the
same semantics as the prefix traversal:

```text
same root
same edge direction
same budget or horizon convention
same parent filters and root-cone restrictions
same cycle/simple-path policy
same path statistic and cumulative basis
```

The simple-path condition is the hardest one.  A cached `H_b` that only forbids
repeating nodes inside the suffix can overcount full paths if the suffix returns
to a node already visited in the prefix.  Exact splicing for cyclic graphs would
need a suffix state conditioned on the prefix visited set, or a structural proof
that the boundary lies in an acyclic/root-monotone cone where suffixes cannot
return to the prefix.  In practical Wikipedia category cones, those violations
may be rare, but they should be reported as an approximation assumption rather
than hidden.

The current probe therefore treats filtered suffix splicing conservatively:
when a runtime reachability filter is active, it reports boundary hits but does
not report spliced suffix totals unless compatible suffix histograms are being
measured with the same filter semantics.

## 7. Validation plan

The next validation should compare exact enumeration and sample estimates on
small graphs where all paths are tractable:

```text
exact root path count
sampled root path count using c(pi)
exact mean root path length
sampled mean root path length
exact boundary-spliced mass for acyclic fixtures
sampled boundary-spliced mass
path-length kernels such as b_p^(-L) and (L + 1)^(-n)
```

For larger enwiki cones, reports should separate:

```text
proposal correction variance
path-value kernel variance
boundary hit probability
remaining-budget suffix mass
effective sample size
root-cone/filter compatibility
```

This keeps the sampling question independent from the cache-admission question:
`b_p^L` remains useful for planning expected search cost, while `product_i d_i`
is the estimator correction for the random-walk proposal actually used.
