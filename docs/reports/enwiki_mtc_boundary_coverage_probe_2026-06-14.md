# Enwiki MTC Boundary Coverage Probe

Date: 2026-06-14

This probe separates two questions that the runtime benchmark conflates under caps:

- exact shallow coverage: enumerate every simple parent-prefix until root, boundary, or path-length budget;
- sampled deeper coverage: sample simple parent walks and use branch-product weights to estimate path-space counts.

The direct DFS search still forbids repeated nodes in a path. Cycle edges are skipped and counted separately. The original smoke rows used the full parent graph and a budget of 4, so their budget-exhausted counts should not be read as evidence about root-reaching parent paths. The root-reachable rows below keep only parent candidates that can still reach `Category:Main_topic_classifications` within the remaining path budget.

## Results

| run | mode | parent_filter | target_depth | targets | path_length_budget | boundary_nodes | observed_terminal_prefixes | root_paths | boundary_hit_prefixes | budget_exhausted_prefixes | expansion_cap_hit_targets | observed_boundary_hit_fraction | weighted_terminal_estimate | weighted_boundary_fraction |
|-----|------|---------------|-------------:|--------:|-------------------:|---------------:|---------------------------:|-----------:|----------------------:|--------------------------:|--------------------------:|-------------------------------:|---------------------------:|---------------------------:|
| enwiki_mtc_boundary_coverage_probe_smoke | exact | all | 4 | 2 | 4 | 6 | 1656 | 9 | 23 | 1624 | 0 | 0.013889 | n/a | n/a |
| enwiki_mtc_boundary_coverage_probe_smoke | sample | all | 4 | 2 | 4 | 6 | 200 | 4 | 24 | 172 | 0 | 0.120000 | 759.510 per target | 0.010409 |
| enwiki_mtc_boundary_coverage_sample_depth6_smoke | sample | all | 6 | 2 | 8 | 16 | 500 | n/a | 4 | n/a | 0 | 0.008000 | 96190.394 per target | 0.000009 |
| enwiki_mtc_boundary_coverage_root_reachable_b10_20_two_targets | exact | root-reachable | 4 | 2 | 10 | 6 | 1355 | 1309 | 15 | 0 | 0 | 0.011070 | n/a | n/a |
| enwiki_mtc_boundary_coverage_root_reachable_b10_20_two_targets | exact | root-reachable | 4 | 2 | 20 | 6 | 31943 | 31589 | 81 | 0 | 1 | 0.002536 | n/a | n/a |
| enwiki_mtc_boundary_coverage_root_reachable_b10_20 | exact | root-reachable | 4 | 1 | 20 | 6 | 306 | 300 | 2 | 0 | 0 | 0.006536 | n/a | n/a |

The shallow exact run completed without path-count or expansion caps. It found that selected boundaries covered 23 of 1656 terminal prefixes, about 1.39%. The weighted sample on the same shallow shape estimated about 1.04%, close enough for a small 100-sample-per-target smoke.

The deeper sampled run shows why raw random-walk hit rates are not enough. The observed sample hit rate was 0.8%, but branch-product weighting estimated a much smaller path-space boundary fraction, about 0.0009%. The sampled boundary hits were reachable with remaining budget 4, but they represented low-weight regions of a much larger path-prefix space.

The root-reachable exact run is the better first-pass evidence for root-anchored parent paths. With budget 10, both sampled targets completed exactly and no prefixes exhausted the path budget. With budget 20, one high-branching target hit the expansion cap, but the enumerated prefixes still had zero budget exhaustion. This supports the concern that the earlier budget-4 exhausted-prefix count mostly reflected the overly small budget and unfiltered parent graph, not a real claim that many simple parent paths to root exceed 4 hops.

## Interpretation

For shallow nodes, exact enumeration is the preferred evidence, but it should be root-reachable filtered when the question is about paths to a specific root. For deeper nodes, sampled evidence should report both the raw random-walk hit rate and a weighted path-space estimate. Cache-performance claims should be framed against the weighted estimate unless the query runtime intentionally follows the same random-walk proposal.

## Artifacts

- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_boundary_coverage_probe_smoke_20260614T041713Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_boundary_coverage_probe_smoke_20260614T041713Z.jsonl`
- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_boundary_coverage_sample_depth6_smoke_20260614T041725Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_boundary_coverage_sample_depth6_smoke_20260614T041725Z.jsonl`
- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_boundary_coverage_root_reachable_b10_20_20260614T043647Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_boundary_coverage_root_reachable_b10_20_20260614T043647Z.jsonl`
- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_boundary_coverage_root_reachable_b10_20_two_targets_20260614T043659Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_boundary_coverage_root_reachable_b10_20_two_targets_20260614T043659Z.jsonl`
