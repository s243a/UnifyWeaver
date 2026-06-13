# Distributional Compression Theory

This note records the theory behind compressing exact path-statistic
histograms into cheaper finite-support representations.  It complements
`DISTRIBUTIONAL_FIT_POLICY.md`, `PARENT_BRANCHING_DISTRIBUTION_THEORY.md`, and
`RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`.

The central distinction is that recurrence histograms are exact computed
objects, not noisy samples.  Choosing a compact representation is therefore a
lossy-compression problem with explicit error certificates, not a statistical
model-selection problem in the usual AIC/BIC sense.

## 1. Exact object

For parent-only paths to a fixed root, the unnormalised recurrence is:

```text
H_root[0] = 1
H_v[L] = sum_{p in parents(v)} H_p[L - 1]
```

Equivalently, in generating-function form:

```text
H_root(z) = 1
H_v(z) = z * sum_{p in parents(v)} H_p(z)
```

`H_v[L]` stores path-count mass at length `L`.  If a future storage layer keeps
normalised distributions `P_v`, it must also store total path mass `N_v`, and
the recurrence becomes:

```text
N_v * P_v[L] = sum_{p in parents(v)} N_p * P_p[L - 1]
N_v = sum_{p in parents(v)} N_p
```

The mass recurrence collapses to a plain sum because each `P_p` sums to one.

The current benchmark keeps the unnormalised form because cache splicing needs
path-count mass directly.

## 2. Error recurrence before fitting

Suppose parent `p` has an exact histogram `H_p` and a stored approximation
`A_p`.  Its signed error is:

```text
E_p[L] = H_p[L] - A_p[L]
```

Before fitting any new approximation for child `v`, the propagated parent
error is already known by the same shifted-sum operator:

```text
E_pre_v[L] = sum_{p in parents(v)} E_p[L - 1]
```

The exact child histogram decomposes as:

```text
H_v = A_pre_v + E_pre_v
A_pre_v[L] = sum_{p in parents(v)} A_p[L - 1]
```

If the child is then compressed into `A_v`, the new local compression residual
is:

```text
R_v[L] = H_v[L] - A_v[L]
```

and the stored error certificate for `A_v` should cover both inherited error
and new fit error.  In exact recurrence materialisation, `E_pre_v = 0`.  In a
mixed exact/approximate cache, `E_pre_v` allows the planner to reject or tighten
a fit before spending work on a more expressive family.

For normalised forms, scale the parent errors by `N_p` before shifting:

```text
E_pre_v_mass[L] = sum_p N_p * (P_p[L - 1] - Q_p[L - 1])
```

### Scalar certificate recurrence

The signed-vector recurrence above is the exact statement, but persisting
`E_p` per node defeats compression: the residual vector is as large as the
exact histogram it certifies.  What the cache should persist are scalar
certificates, and those obey their own recurrence.  The normalised child is
the mass-weighted mixture of unit-shifted parents, and a unit shift changes
neither metric, so by the triangle inequality:

```text
epsilon_K(v)  <= sum_p (N_p / N_v) * epsilon_K(p)  + epsilon_K_local(v)
epsilon_W1(v) <= sum_p (N_p / N_v) * epsilon_W1(p) + epsilon_W1_local(v)
```

where the local term is the metric applied to the new compression residual
`R_v`.  Inherited error therefore propagates as a mass-weighted average of
parent certificates — one float per metric per node.  The full `E_pre_v`
vector is only needed as a diagnostic, for example to see where inherited
error concentrates after a fit rejection.

## 3. Error metrics

Prefix-mass queries use a CDF:

```text
F_H(t) = sum_{L <= t} H[L] / N
```

For a normalised exact distribution `P` and approximation `Q`, the most useful
certificates are:

```text
epsilon_K = max_t |F_P(t) - F_Q(t)|
epsilon_W1 = sum_t |F_P(t) - F_Q(t)|
```

`epsilon_K` is the direct per-query absolute error bound for prefix-mass
queries.  `epsilon_W1` is the one-dimensional Wasserstein-1 distance on integer
support and bounds Lipschitz aggregate error:

```text
|E_P[g(L)] - E_Q[g(L)]| <= Lip(g) * epsilon_W1
```

For bounded-variation step-like costs:

```text
|E_P[g(L)] - E_Q[g(L)]| <= variation(g) * epsilon_K
```

The compression gate should therefore be expressed as tolerances over
`epsilon_K`, `epsilon_W1`, or a functional-specific certificate.  State-count or
bin-count thresholds are cost triggers; they do not by themselves prove quality.

### Absolute versus relative certificates

The certificates above are absolute, and absolute certificates are the right
gate for additive aggregates.  When many small tail contributions are summed
into an `O(1)` total, each term's absolute error is bounded by the
certificate and products of small errors are second-order — the usual
justification for dropping higher-order terms in approximations.

The case that needs a different gate is when a product of tail masses is the
whole quantity, for example ranking two candidates whose scores are each
products of small survival probabilities.  Relative errors add under
multiplication: factors with relative errors `delta_i` give a product with
relative error approximately `sum_i delta_i`.  An absolute certificate
`epsilon_K = 0.01` on a tail of mass `0.001` is a 10x relative error on that
factor, easily enough to invert the ranking of two competing small products,
even though the absolute error of either product is tiny.  Queries of this
shape should be gated on relative CDF/survival error over the queried tail
region, not on absolute `epsilon_K`.

## 4. Fit ladder

The representation ladder should prefer the cheapest encoding that satisfies
the active error budget.

| Representation | Parameters / storage | Strength | Weakness |
|----------------|----------------------|----------|----------|
| Tail-pruned exact histogram | packed nonzero bins plus dropped-tail certificate | Exact where kept; no fitting | Cost grows with effective support |
| Quantised CDF table | one monotone fixed-point CDF value per retained point | Direct prefix queries; error bounded by the quantisation step (`<= 2^-17` at 16 bits) | Less useful for arbitrary PMF functionals |
| Binomial | `D`, `p`, mass, support | Very cheap, bounded discrete support | Under/over-dispersion and multimodality fail quickly |
| Beta-binomial | `D`, `alpha`, `beta`, mass, support | First upgrade for over-dispersion | CDF is a finite sum unless tabulated |
| Mixture of binomials | `K` weights and `K` probabilities, shared `D` | Bounded, discrete, modes-per-float efficient | Component width tied to location |
| Discretised Gaussian mixture | weights, means, variances, support | Handles sharp/narrow modes; cheap CDF via `erf` | Continuous prior on discrete data; more parameters per mode |

### Why Gaussian mixtures are not the default

Gaussian mixtures are useful but should be an escalation family rather than the
first fallback.

- The data are bounded integer histograms.  Binomial and beta-binomial families
  encode that support directly.
- Mixtures of binomials use `2K - 1` parameters when every component shares
  `D`; a Gaussian mixture needs `3K - 1` parameters.  For equal storage,
  binomial mixtures can represent more natural binomial-width modes.
- Per-node compression is judged by exact CDF/W1 error, not by sampling
  likelihood.  If a cheaper discrete family meets the error budget, a GMM adds
  no planner value.
- Gaussian mixtures are strongest when the residual has a sub-binomial-width
  spike or a narrow bottleneck mode.  That is an escalation condition, not the
  common case assumed by the parent-branching recurrence.

The design should therefore test GMMs, but only after tail-pruned histograms,
CDF tables, binomial, beta-binomial, and binomial mixtures have failed the
active error budget.

## 5. Deep-node CLT regime

For real deep nodes, binomial or normal approximations become more plausible
because the recurrence repeatedly sums shifted parent contributions.  When many
small independent or weakly dependent contributions accumulate, the
standardised path-length distribution should move toward the central-limit
regime: skew shrinks, local irregularities smooth out, and a compact
binomial-like or normal-like representation can be accurate.

This is not the same claim as saying that parent counts themselves are
binomial or normal.  Parent-degree distributions can be skewed, exponential, or
heavy-tailed, especially on enwiki.  The CLT argument applies to the path-count
distribution after repeated shifted sums, and it can fail around bottlenecks,
topic mixtures, hubs, cycles, or small effective ancestor cones.

The policy consequence is simple: depth and effective-convolution count are
signals for trying binomial or normal compression earlier in the candidate
ladder, but the representation still has to pass the CDF/W1 error gate.

## 6. Fit gates

BIC, AIC, chi-square, and held-out likelihood are not the right gate for
per-node exact recurrence histograms.  The question is not generalisation from
samples; it is whether a compact representation preserves the exact object
within a planner tolerance.

The gate should be:

```text
choose cheapest representation R such that:
    max_cdf_error(exact, R) <= epsilon_K
and optional:
    w1_cdf_error(exact, R) <= epsilon_W1
and optional:
    functional_error(exact, R, g) <= epsilon_g
```

Statistical model selection becomes relevant only for shared priors learned
across many nodes, or if histograms become sampled/noisy rather than exact.

## 7. Policy consequence

The old question "is this over 50 points?" should be replaced with:

```text
which representation is cheapest at the requested error tolerance?
```

For in-memory Python dictionaries, a 10-bin histogram may already be larger
than a small packed parametric state.  For packed LMDB encodings, the crossover
point is different.  The policy must compare byte estimates for the target
storage mode:

```text
cost_model = packed_lmdb | in_memory_runtime | diagnostic_json
```

Thresholds such as `max_recurrence_states = 50` remain useful as work triggers:
they decide when to try compression.  They should not decide whether the
compressed representation is acceptable; that requires the error certificate.
