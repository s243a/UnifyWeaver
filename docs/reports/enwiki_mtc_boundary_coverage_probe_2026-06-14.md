# Enwiki MTC Boundary Coverage Probe

Date: 2026-06-14

This probe separates two questions that the runtime benchmark conflates under caps:

- exact shallow coverage: enumerate every simple parent-prefix until root, boundary, or path-length budget;
- sampled deeper coverage: sample simple parent walks and use branch-product weights to estimate path-space counts.

The direct DFS search still forbids repeated nodes in a path. Cycle edges are skipped and counted separately.

## Results

| run | mode | target_depth | targets | path_length_budget | boundary_nodes | observed_terminal_prefixes | observed_boundary_hit_fraction | weighted_terminal_estimate | weighted_boundary_fraction |
|-----|------|-------------:|--------:|-------------------:|---------------:|---------------------------:|-------------------------------:|---------------------------:|---------------------------:|
| enwiki_mtc_boundary_coverage_probe_smoke | exact | 4 | 2 | 4 | 6 | 1656 | 0.013889 | n/a | n/a |
| enwiki_mtc_boundary_coverage_probe_smoke | sample | 4 | 2 | 4 | 6 | 200 | 0.120000 | 759.510 per target | 0.010409 |
| enwiki_mtc_boundary_coverage_sample_depth6_smoke | sample | 6 | 2 | 8 | 16 | 500 | 0.008000 | 96190.394 per target | 0.000009 |

The shallow exact run completed without path-count or expansion caps. It found that selected boundaries covered 23 of 1656 terminal prefixes, about 1.39%. The weighted sample on the same shallow shape estimated about 1.04%, close enough for a small 100-sample-per-target smoke.

The deeper sampled run shows why raw random-walk hit rates are not enough. The observed sample hit rate was 0.8%, but branch-product weighting estimated a much smaller path-space boundary fraction, about 0.0009%. The sampled boundary hits were reachable with remaining budget 4, but they represented low-weight regions of a much larger path-prefix space.

## Interpretation

For shallow nodes, exact enumeration is the preferred evidence. For deeper nodes, sampled evidence should report both the raw random-walk hit rate and a weighted path-space estimate. Cache-performance claims should be framed against the weighted estimate unless the query runtime intentionally follows the same random-walk proposal.

## Artifacts

- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_boundary_coverage_probe_smoke_20260614T041713Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_boundary_coverage_probe_smoke_20260614T041713Z.jsonl`
- `docs/reports/lmdb_boundary_coverage_probe_summary_enwiki_mtc_boundary_coverage_sample_depth6_smoke_20260614T041725Z.md`
- `docs/reports/lmdb_boundary_coverage_probe_enwiki_mtc_boundary_coverage_sample_depth6_smoke_20260614T041725Z.jsonl`
