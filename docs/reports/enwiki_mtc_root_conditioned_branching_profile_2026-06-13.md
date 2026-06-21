# Enwiki MTC Root-Conditioned Branching Profile Smoke

This smoke run profiles parent branching after retaining only nodes reached by child traversal from `Category:Main_topic_classifications` (`7345184`).  It is capped at `5000` retained nodes, so it is preprocessing evidence rather than the final full-dataset prior.

Generated outputs:

- `docs/reports/lmdb_root_conditioned_branching_profile_enwiki_mtc_root_conditioned_branching_smoke_20260614T001041Z.jsonl`
- `docs/reports/lmdb_root_conditioned_branching_profile_summary_enwiki_mtc_root_conditioned_branching_smoke_20260614T001041Z.md`

## Result

The capped profile retained `5000` nodes, examined `56867` child edges, and took about `3278 ms`.  It hit the node cap before the requested depth cap.

| scope | E[p] | E[p^2]/E[p] | max_p |
|-------|-----:|------------:|------:|
| raw_full_graph | 4.152 | 7.048550 | 202 |
| root_conditioned | 1.698 | 2.164035 | 9 |
| outside_root | 2.690 | 7.009775 | 197 |

This supports the hypothesis that global parent branching is too pessimistic for common-root ancestor searches.  The retained-parent prior is much lower, and the extreme parent hubs mostly live outside the root-conditioned parent set.

## Implication

For materialization planning, the estimator should eventually consume a preprocessing profile with root-conditioned `E[p^2]/E[p]` and depth buckets instead of the raw global parent-branching statistic.  The preprocessing cost still belongs in the amortization model: a full profile or full histogram materialization only pays off when enough queries will reuse the dataset-level prior/cache.

## Next Step

Run the same profiler without the `5000` node cap, or with larger caps such as `50000`, `250000`, and full traversal, to estimate how quickly the root-conditioned branching prior stabilizes and what the preprocessing cost curve looks like.
