# Enwiki MTC Cap-Aware Precompute Estimator

Date: 2026-06-13

This report follows the boundary-cache validation run.  The previous estimator
used an uncapped `b^m` suffix model:

```text
saved_per_hit ~= b^suffix_hops - cached_eval_cost
```

The validation benchmark showed that this is too optimistic for operational
queries that stop at `path_cap` or `expansion_cap`.  Both the full search and
cached search can hit caps, so the full `b^m` suffix is never paid.

## Model Update

The estimator now supports:

```text
cap_mode = uncapped | path | expansion | measured

cap_limited_suffix_states =
    min(b^suffix_hops, cap_ceiling)

saved_per_hit =
    cap_limited_suffix_states * uncached_cost_per_state
    - cached_eval_cost
    - per_hit_decode_cost
    - splice_cost
```

It also separates:

- one-time storage/decode cost used to materialize or store a state;
- per-hit decode cost, disabled with `--decode-memoized`;
- splice cost per point/bin.

## Runs

Uncapped calibrated planner:

```text
docs/reports/enwiki_mtc_precompute_depth_estimator_calibrated_distribution_precompute_depth_estimator_summary.md
```

Cap-aware planner with a large measured-work ceiling:

```text
docs/reports/enwiki_mtc_precompute_depth_estimator_cap_aware_distribution_precompute_depth_estimator_summary.md
```

Cap-aware planner matched to the capped validation smoke:

```text
docs/reports/enwiki_mtc_precompute_depth_estimator_cap_aware_smoke_matched_distribution_precompute_depth_estimator_summary.md
```

## Interpretation

The uncapped calibrated planner still recommends boundary depths through 5:
the assumed skipped suffix work is large enough that precomputation pays even
after build and decode costs.

The cap-aware smoke-matched run uses a deliberately small measured-work ceiling
to represent the validation condition where both full and cached searches were
cap-limited.  In that regime, most boundary depths stop paying because the
cached query pays decode/splice overhead while the uncached query does not
actually traverse the full `b^m` suffix.

This matches the validation result qualitatively: cache hits occurred, but
measured cached/full time ratios were around `1.0` or worse.

## Consequence

There are now two distinct planning modes:

| mode | use case |
|------|----------|
| uncapped | decide where full exact suffix search would become too expensive |
| cap-aware | decide whether cached boundaries improve the actual bounded query workload |

For production bounded queries, the cap-aware mode should be preferred.  The
next calibration step is to replace the ad hoc measured-work ceiling with values
derived directly from boundary-cache validation rows, ideally per
`boundary_depth`.
