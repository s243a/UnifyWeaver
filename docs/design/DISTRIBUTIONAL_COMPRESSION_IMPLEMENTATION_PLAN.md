# Distributional Compression Implementation Plan

This plan turns `DISTRIBUTIONAL_COMPRESSION_THEORY.md` into staged work for
the parent-path recurrence and boundary-cache tools.

## Phase 1: Error certificates

Add common helpers for comparing an exact histogram with a candidate
representation:

- `max_cdf_error`
- `w1_cdf_error`
- dropped-tail mass
- functional-specific error where the functional is known

Persist those certificates on boundary rows and fitted states:

```text
fit_error_max_cdf
fit_error_w1_cdf
fit_error_functional(Name)
inherited_error_max_cdf
inherited_error_w1_cdf
```

The inherited error fields are scalars computed by the certificate recurrence
(see "Scalar certificate recurrence" in the theory doc), not by materialising
signed error vectors — a per-node residual vector would cost as much as the
exact histogram it certifies:

```text
inherited_error_max_cdf(v) = sum_p (N_p / N_v) * total_error_max_cdf(p)
inherited_error_w1_cdf(v)  = sum_p (N_p / N_v) * total_error_w1_cdf(p)
total_error_*(v)           = inherited_error_*(v) + fit_error_*(v)
```

The full signed-error vector `E_pre_v[L] = sum_p E_p[L - 1]` (or
`sum_p N_p * (P_p[L - 1] - Q_p[L - 1])` for normalised parent states) is a
diagnostic-only computation for inspecting where inherited error concentrates
after a fit rejection.

## Phase 2: Cheap exact encodings

Before adding new parametric families, implement packed exact alternatives:

1. tail-pruned exact histogram;
2. quantised CDF table for prefix-mass workloads;
3. packed sparse histogram byte model for LMDB;
4. in-memory runtime byte model for Python/runtime dictionaries.

This phase clarifies the real crossover point.  A 50-float model should be
compared against packed bytes at equal error tolerance, not against a Python
dictionary bin count.

## Phase 3: Single-family fits

Implement and gate cheap one-family fits:

1. binomial;
2. beta-binomial;
3. discretised normal for deep CLT-like regions.

Binomial and normal approximations are expected to become more reasonable for
deep nodes where many shifted parent contributions have accumulated.  This is
the central-limit regime: local irregularities can smooth out, skew often
shrinks in standardised units, and a compact symmetric or nearly symmetric
family may be good enough.

The implementation should still gate by measured CDF/W1 error, because deep
nodes can contain bottlenecks, near-deterministic modes, or topical mixtures
that violate a single-family fit.

## Phase 4: Mixture fits

Add bounded discrete mixtures before Gaussian mixtures:

1. mixture of binomials with shared support `D`;
2. discretised Gaussian mixture as an escalation family.

Mixture-of-binomials is preferred first because it is bounded and discrete with
`2K - 1` parameters.  Gaussian mixtures remain useful when residual error
concentrates in a narrow sub-binomial-width mode or bottleneck spike.

A binomial mixture with shared `n = D` is identifiable only when
`D >= 2K - 1`, so the fitter must clamp `K <= (D + 1) / 2`.  On narrow
supports this binds before the float budget does, and an unclamped fitter
will chase degenerate component splits.

Fits should be initialised cheaply rather than started cold:

1. binomial: `p = mean / D` (method of moments, closed form);
2. beta-binomial: method of moments — solve the overdispersion `rho` from
   `var = D * p * (1 - p) * (1 + (D - 1) * rho)`, then
   `alpha = p * (1 - rho) / rho` and `beta = (1 - p) * (1 - rho) / rho`;
   if `rho <= 0` fall back to plain binomial;
3. mixtures: warm-start from the parent.  Because
   `H_v = sum_p shift(H_p)`, the mass-weighted union of parent components
   with means bumped by one step (`p_i <- p_i + 1/D`) is a near-converged
   initialisation, and parameters drift slowly down ancestor chains.

Each mixture fit must have a capped iteration budget.  EM is acceptable if it
is bounded and diagnostics report:

```text
fit_family
components
em_iterations
fit_error_max_cdf
fit_error_w1_cdf
fit_rejected_reason
```

## Phase 5: Policy selection

Replace threshold-only admission with cheapest-representation selection:

```text
candidates = [
    tail_pruned_histogram,
    quantized_cdf_table,
    binomial,
    beta_binomial,
    normal_or_discrete_normal,
    binomial_mixture,
    discretized_gmm
]

choose argmin bytes(candidate)
where candidate.error <= active_error_budget
```

`max_recurrence_states` and `max_effective_bins_after_trim` remain useful as
triggers for trying compression, but acceptance must be error-gated.

## Phase 6: Planner integration

Expose policy options through `distribution_fit_policy`:

```prolog
distribution_fit_policy(
    root_path_distribution/3,
    [ max_recurrence_states(50),
      max_effective_bins_after_trim(50),
      max_cdf_error(0.01),
      max_w1_error(0.05),
      candidate_families([
          tail_pruned_histogram,
          quantized_cdf_table,
          binomial,
          beta_binomial,
          discrete_normal,
          binomial_mixture,
          discretized_gmm
      ]),
      storage_cost_model(packed_lmdb) ]).
```

The default candidate order should be conservative and discrete-first.  Users
can opt into Gaussian mixtures or learned smoothers, but those should not be the
first default while simpler bounded discrete families pass the error gate.

## Phase 7: Learned smoothers

Gaussian-mixture smoothing or a tiny convolutional smoother can be evaluated
after deterministic candidates exist.  The key distinction is operational:

- inference may be cheap with a tiny model;
- training adds separate compute cost;
- model drift and validation become new responsibilities.

Learned smoothers should therefore be optional candidate families behind the
same error-certificate interface, not a replacement for the deterministic
ladder.
