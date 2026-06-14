# Enwiki MTC Profile-Driven Precompute Estimator

This note compares the profile-driven estimator run against the earlier validation-calibrated estimator. The new run reads the root-conditioned branching profile from `lmdb_root_conditioned_branching_profile_enwiki_mtc_root_conditioned_branching_smoke_20260614T001041Z.jsonl` and uses `root_conditioned_parent_degree` as the planning prior.

## Inputs

| input | value |
|---|---:|
| expected queries | 1000 |
| target depth | 8 |
| cap mode | validation |
| profile nodes | 4999 |
| profile sample cap | 5000 |
| profile truncated | no |
| root-conditioned `b = E[p^2] / E[p]` | 2.164035 |
| earlier validation-calibrated default `b` | 4.255699 |

The profile provides depth-specific root-conditioned priors through depth 3: depth 1 uses `2.621622`, depth 2 uses `2.349706`, and depth 3 uses `2.105674`. Deeper estimator rows fall back to the profile overall `2.164035` prior.

## Result

The generated profile-driven estimator report is `enwiki_mtc_precompute_depth_estimator_root_conditioned_smoke_distribution_precompute_depth_estimator_summary.md`. Its recommendation is still to materialize only boundaries 0 and 1 under the current validation cap. Depths 2 through 6 have positive cache-hit rows, but measured cached execution was slower than full execution in the validation benchmark, so their clipped saved-per-hit is zero. Depth 7 has only zero-hit validation rows and remains unusable for validation capping.

The profile-driven prior changes the economic shape even when it does not change the final admission boundary. Expected hits decay faster inside the root-conditioned ancestor tree than in the older report's hand-tuned query-reach model at shallow depths: depth 1 is `381.443` expected hits in the profile-driven run rather than `1000.000`, and depth 2 is `162.337` rather than `600.000`. Build and suffix state estimates are also lower because the in-root effective branch factor is much smaller than the raw/global parent branching prior.

## Interpretation

This is the behavior we wanted from the profile input: raw parent branching is useful as an upper-level warning, but cache admission should be driven by the branching visible inside the root-conditioned ancestor search space. The current smoke still says not to admit deeper boundaries because the measured cache path is slower there, not because the branching prior is too pessimistic. The next validation target is therefore decode/splice/runtime overhead for positive-hit boundary rows, not another adjustment to the branch prior.
