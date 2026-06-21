# LMDB Recurrence Threshold Fallback Smoke

Date: 2026-06-12

This smoke adds opt-in approximation thresholds on top of the recurrence
boundary builder.  The benchmark now measures recurrence state count and
tail-trimmed effective support bins before admission.  If either configured
limit is exceeded, and a parametric representation fits, the boundary row can
store a parametric approximation instead of a materialized histogram.

The recurrence histogram convention remains unnormalized: each bin stores path
count mass.  For normalized storage, the recurrence must sum `N_p * shifted(P_p)`
and persist the resulting `N_v` separately.

Common setup:

```text
--lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident
--root 7345184
--boundary-depths 1
--target-depths 2
--children-per-node 32
--frontier-limit 256
--boundaries-per-depth 12
--targets-per-depth 6
--boundary-budget 6
--boundary-builder recurrence
--budgets 6
--path-cap 20000
--expansion-cap 50000
--seed enwiki-mtc-recurrence-threshold-v1
--admission-policy baseline
--parametric-shape-model support-binomial
--parametric-mean-model midpoint
--parametric-mass-model oracle
--max-recurrence-states 50
--max-effective-bins-after-trim 4
--tail-epsilon 0.001
```

| boundary_nodes | histogram_cached | parametric_cached | states_over_limit | bins_over_limit | forced_parametric | mean_effective_bins_after_trim |
|---------------:|-----------------:|------------------:|------------------:|----------------:|------------------:|-------------------------------:|
| 12 | 1 | 11 | 11 | 8 | 11 | 4.333 |

| threshold_reason | rows |
|------------------|-----:|
| `recurrence_states_over_limit` | 3 |
| `recurrence_states_over_limit+effective_bins_over_limit` | 8 |
| `within_threshold` | 1 |

| budget | rows | mean_l1 | max_l1 | mean_cdf | mean_path_relative_error | mean_abs_path_delta |
|--------|-----:|--------:|-------:|---------:|-------------------------:|--------------------:|
| 6 | 6 | 0.590152 | 1.234043 | 0.275845 | 0.182966 | 7.000 |

The threshold mechanism is doing the intended admission work: the large
recurrence rows no longer enter the exact histogram cache.  The smoke also shows
that a simple midpoint support-binomial approximation can introduce visible
histogram error, even with oracle mass.  That makes the thresholds useful as
cost controls, but not sufficient as quality controls.  Future calibration
should compare binomial, gamma, and sampled/mixture smoothers under the same
threshold settings.

Gaussian-mixture smoothing, possibly via a tiny convolutional model, remains a
candidate approximation family.  It could be cheap at inference once trained,
but training adds extra computation and creates a separate validation problem,
so the deterministic threshold plumbing should remain independent of any learned
model.
