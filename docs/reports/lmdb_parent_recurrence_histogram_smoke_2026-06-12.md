# LMDB Parent Histogram Recurrence Smoke - 2026-06-12

This smoke tests the recurrence formulation for parent-path histograms:

```text
H_root[0] = 1
H_v[d] = sum over parents p of H_p[d - 1]
```

The recurrence shifts each direct parent distribution by one path step and sums
the shifted histograms.  On a parent-only DAG this is exact and avoids explicit
path enumeration.  On cyclic category cones, a node-only histogram cannot carry
the visited-set state required for exact unique-node path semantics, so cycle
encounters are flagged as `cycle_approximation`.

## Command

```bash
python3 scripts/lmdb_parent_recurrence_histogram_benchmark.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_parent_recurrence_smoke \
  --target-depths 3 \
  --children-per-node 16 \
  --frontier-limit 80 \
  --targets-per-depth 3 \
  --budgets 4,6,8 \
  --path-cap 10000 \
  --expansion-cap 50000 \
  --seed enwiki-mtc-parent-recurrence-smoke \
  --output-dir /mnt/c/Users/johnc/Scratch/parent-recurrence
```

## Results

| budget | rows | exact_matches | cycle_approx | mean_l1 | mean_cdf | mean_state_ratio | mean_time_ratio | dfs_capped | recurrence_capped |
|-------:|-----:|--------------:|-------------:|--------:|---------:|-----------------:|----------------:|-----------:|------------------:|
| 4 | 3 | 3 | 3 | 0.000000 | 0.000000 | 0.133 | 0.060 | 0 | 0 |
| 6 | 3 | 3 | 3 | 0.000000 | 0.000000 | 0.027 | 0.299 | 0 | 0 |
| 8 | 3 | 1 | 3 | 0.890547 | 0.611940 | 0.010 | 9.502 | 2 | 0 |

## Interpretation

For budgets 4 and 6, recurrence matched the DFS simple-path histograms exactly
on this small sample while evaluating far fewer states.  The cycle flag was set,
so these rows should be read as "matched despite cycle encounters", not as a
proof that the recurrence enforces unique-node path semantics.

The budget-8 row is a warning case.  DFS enumeration hit caps on 2 of 3 rows,
while recurrence continued as a finite shifted-sum computation.  The high error
there is not a clean recurrence-vs-exact comparison; it says that deeper budgets
need better validation and cycle handling.

## Update Rules

The recurrence can also be viewed as an incremental boundary-condition update:
when a parent distribution changes, child distributions can be updated by
subtracting the old shifted parent contribution and adding the new shifted
contribution.  A conservative propagation gate is:

```text
update child only when the changed parent support can extend or change the
child support under the child path horizon
```

One variant we discussed is to re-update a child only when the parent's maximum
root path length is greater than the child's current maximum root path length.
Other variants could use support overlap, tail mass, peak-relative tail
thresholds, or expected downstream cache hits.

## Refinement Direction

The recurrence should become the first materialization path for cached boundary
distributions.  DFS enumeration should remain as a validator for small cones and
as a residual source for calibration.

There are two different prior layers:

1. **Node-local recurrence priors.**  These are the histograms produced by the
   difference equation on the ancestor cone of a concrete target.  They are
   fairly exact when the cone is DAG-like and the recurrence state carries the
   active path constraints.  They are the right boundary conditions for a later
   path-aggregate search.
2. **Global or depth-conditioned planning priors.**  These are approximate
   distributions predicted from graph-level statistics such as the
   depth-conditioned parent branching profile, expected support width, tail
   decay, or fitted binomial/Gamma family.  They are not boundary conditions for
   a specific node.  They estimate whether carrying exact histograms to a given
   depth is likely to be worth the memory and materialization cost.

A later refinement can treat node-local recurrence histograms as priors and use
sampled or bounded path enumeration residuals to adjust them.  That could be
framed as a Monte Carlo correction or gradient-style optimization: compare
enumerated path mass to predicted mass, then adjust local distributions where
the residual is large.  This is not cheap, but it is a plausible way to improve
recurrence states without enumerating all paths.
