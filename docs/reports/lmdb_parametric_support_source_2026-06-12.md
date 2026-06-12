# LMDB Parametric Support Source Smoke

Date: 2026-06-12

This smoke compares the existing measured histogram support for parametric
boundary states with a root-distance support bound.  Both runs used the same
enwiki `Category:Main_topic_classifications` sample, forced parametric boundary
states with `--max-histogram-bytes 8 --parametric-bytes 8`, and kept the sample
small to avoid repeating the heavier benchmark runs.

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
--budgets 6
--path-cap 20000
--expansion-cap 50000
--seed enwiki-mtc-parametric-support-source-v1
--admission-policy depth-prior
--safety-factor 1.25
--parametric-shape-model support-binomial
--parametric-mean-model midpoint
--parametric-mass-model oracle
```

| support_source | boundary_nodes | parametric_nodes | mean_bound_width | mean_width_delta | mean_min_delta | mean_max_delta | truncated | cycle_skipped | mean_l1 | mean_cdf | mean_path_relative_error |
|----------------|---------------:|-----------------:|-----------------:|-----------------:|---------------:|---------------:|----------:|--------------:|--------:|---------:|-------------------------:|
| measured | 12 | 12 | 3.667 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0.787838 | 0.393919 | 0.327485 |
| distance-bounds | 12 | 12 | 4.250 | 0.583 | 0.000 | 0.583 | 12 | 12 | 0.800536 | 0.400268 | 0.327485 |

The distance-bound support source is conservative on this sample: it preserves
the measured lower bound on average but extends the upper support by about 0.58
bins.  That makes the parametric histogram slightly wider while leaving the path
mass error unchanged under the oracle mass model.

The `truncated` and `cycle_skipped` flags are important.  The distance source is
a root-distance bound computed under a capped parent-distance walk; it is a
planning/support signal, not proof that all simple-path support has been
enumerated exactly.  For deeper samples this flag should be read as "usable
bound metadata with caveats", not as an exact recurrence witness.
