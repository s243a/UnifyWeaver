# Graph-geometry synthetic v2 — matched coupling and predictive equivalence

## Status

**Frozen after the v1 mechanism audit failed and before running v2.**  The v1 protocol and failed conclusion
are retained as an instructive invalid-comparison diagnostic; they are not relabelled or discarded.  The
portable v1 artifact reported later was recomputed after the shared kernel core changed, so it is a labelled
reproduction rather than a claim that the original temporary artifact remained byte-for-byte unchanged.

## Why v1 could not answer the intended question

V1 planted the same correlation-path amplitude `alpha` for every unit-diagonal kernel.  On the fixed graph,
that did not plant the same covariance effect:

| kernel | off-diagonal RMS | maximum off-diagonal |
|---|---:|---:|
| closed neighborhood | 0.3063 | 0.8165 |
| walk-feature decay | 0.0720 | 0.2453 |
| heat | 0.1848 | 0.6027 |
| resolvent | 0.1120 | 0.3618 |
| deranged walk | 0.0720 | 0.2453 |

At common `alpha`, the closed-neighborhood truth therefore carried 4.25 times the RMS coupling energy of the
walk truth.  Detection and held gain were not comparable across families.

V1 also found kernel correlations `corr(heat,resolvent)=0.999`, `corr(closed,heat)=0.922`, and
`corr(closed,resolvent)=0.927` on this graph.  Exact family recovery among those three is not identified and is
not a scientifically useful gate.  Their predictive covariance can be nearly interchangeable.

Finally, v1's family-wise threshold controlled block-null selection (3.5--6% across the sizing runs), but at 12
training fields selected held gain was negative for all planted cases except closed `alpha=0.2`.  Increasing to
96 fields helped the stronger local/spectral truths but did not repair the unmatched weak walk signal.  This
motivates matched effect size; it does not authorize post-hoc removal of difficult scenarios.

## V2 effect parameterization

For unit-diagonal kernel `K`, define

```text
s_max(K) = max_{i != j} |K_ij|,
alpha(K, rho) = rho / s_max(K),
C(K, rho) = (1-alpha) I + alpha K.
```

The frozen target grid is maximum whitened off-item coupling

```text
rho in {0, .025, .05, .10, .20}.
```

All resulting amplitudes must be `<0.95`; otherwise that family/effect cell is declared ineligible before
simulation.  Report off-diagonal RMS as a secondary effect diagnostic.  Maximum coupling is primary because it
is also the quantity that governs the proposed independent-batching upper bound.

The selector searches the same family set and this family-specific `rho` grid.  Its full maximum train gain is
calibrated under block-null fields exactly as in v1.  Selection uses training fields only; scoring uses
independent held fields.

## Predictive equivalence classes

Before generating v2 residual fields, group candidate kernels using only outcome-blind upper-triangle kernel
correlation:

```text
local_spectral = {closed, heat, resolvent}  # every pair correlation >= .90 here
walk           = {walk_decay}
deranged       = {deranged_walk}
```

Exact family selection remains descriptive.  The primary recovery event is selection from the truth's
equivalence class.  Geometry specificity is assessed by comparing the best train-selected truth-family
covariance with an equal-energy derangement on held fields.

## Frozen v2 runs and gates

Primary run:

```text
replicates=200, calibration_draws=1000,
train_fields=48, held_fields=64, confidence=.95.
```

The 48-field count is frozen from the v1 field-count sizing grid `{12,24,48,96}` as the first count at which a
planted local/spectral case at the smaller v1 effect stopped having negative mean held gain.  It is a mechanism
setting, not a claim that 48 judge calls are required; the real campaign's independent components and repeated
calls need an end-to-end power model.

For `rho in {.10,.20}`:

1. block-null nonzero selection is at most 10%;
2. every planted **non-deranged** truth at `rho=.20` has positive mean selected held NLL gain;
3. the truth's predictive equivalence class is selected in at least 60% of replicates at `rho=.20`;
4. the train-selected truth family beats its equal-energy derangement on held fields in at least 80% of
   replicates at `rho=.20`;
5. lower `rho=.10` results are reported as measured power, not forced to pass;
6. all real-covariance, batching, and QR-deployment gates remain false.

The deranged truth is a negative-control recovery check and does not need positive deployment value.  Failure
of gates 2--4 means the candidate search is still too broad or underpowered for the confirmatory campaign.
