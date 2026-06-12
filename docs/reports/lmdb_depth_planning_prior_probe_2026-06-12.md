# LMDB Depth-Conditioned Planning Prior Probe

This report adds a bounded probe for the distinction between:

- node-local recurrence histograms, which are candidate boundary conditions for
  concrete ancestor cones; and
- global or depth-conditioned planning priors, which estimate whether exact
  histogram materialization is likely to be cheap enough before the node-local
  recurrence is built.

## Script

```text
scripts/lmdb_depth_planning_prior_probe.py
```

The probe samples targets from an LMDB category graph, buckets them by maximum
parent traversal distance to the root (`L_max`), builds a depth-conditioned prior
from root-reaching parent degree statistics in each bucket, and compares that
prior with finite-budget recurrence histograms for targets in the bucket.

The prior is intentionally not used as a boundary condition.  It is a planning
estimate for support width, tail-pruned storage, and rough binomial/Gamma
parameters.

## Smoke Command

```bash
python3 scripts/lmdb_depth_planning_prior_probe.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_depth_planning_prior_smoke24_uncapped \
  --target-depths 2,3,4 \
  --children-per-node 24 \
  --frontier-limit 160 \
  --targets-per-depth 3 \
  --max-parent-depth 24 \
  --max-prior-depth 24 \
  --path-cap 50000 \
  --expansion-cap 50000 \
  --seed enwiki-mtc-depth-prior-smoke24-v1 \
  --output-dir /mnt/c/Users/johnc/Scratch/depth-planning-prior
```

Outputs:

```text
/mnt/c/Users/johnc/Scratch/depth-planning-prior/enwiki_mtc_depth_planning_prior_smoke24_uncapped_depth_planning_prior_summary.json
/mnt/c/Users/johnc/Scratch/depth-planning-prior/enwiki_mtc_depth_planning_prior_smoke24_uncapped_depth_planning_prior_summary.md
```

## Results

Selection:

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 24 |
| 2 | 160 |
| 3 | 160 |
| 4 | 160 |

Prior bucket:

| L_max | targets | mean_root_p | b_root | mean_excess | prior_bins | prior_eff_bins | binom_p | gamma_shape | gamma_scale | prior_pruned_bytes |
|------:|--------:|------------:|-------:|------------:|-----------:|---------------:|--------:|------------:|------------:|-------------------:|
| 24 | 9 | 3.000 | 4.037037 | 3.037037 | 145 | 106 | 1.000000 | 51.989691 | 1.401987 | 1696 |

Prior compared with recurrence histograms:

| L_max | rows | mean_l1 | p95_l1 | mean_cdf | mean_realized_bins | mean_pred_eff_bins | mean_storage_ratio | capped | cycle_approx |
|------:|-----:|--------:|-------:|---------:|-------------------:|-------------------:|-------------------:|-------:|-------------:|
| 24 | 9 | 2.000000 | 2.000000 | 1.000000 | 20.333 | 106.000 | 5.369 | 4 | 9 |

Coverage:

| targets | comparable | skipped | buckets |
|--------:|-----------:|--------:|--------:|
| 9 | 9 | 0 | 1 |

## Interpretation

The `L_max=24` bucket is a useful planning signal, not an accuracy validation.
The recurrence rows are all cycle-approximate and four of nine still hit the
path cap even at `path_cap=50000`.  That means the exact histograms are not a
clean target for fitting quality.

The prior over-predicts retained support for this smoke: `106` effective prior
bins versus about `20.3` realized recurrence bins, or roughly `5.4x` the sparse
histogram storage.  For admission control this is conservative: it warns that a
deep enwiki bucket may be expensive before exact recurrence materialization is
attempted.  It should not be interpreted as a node-local distribution.

The binomial parameter is clamped at `p=1.0`, which is itself diagnostic.  The
bucket's size-biased excess parent branching is above one, so a simple
`Binomial(depth, p)` family cannot carry the prior mean without saturation.  That
supports keeping Gamma-like or empirical planning families in the candidate set
for enwiki-style branching.

## Validation

```bash
python3 -m unittest tests.test_lmdb_depth_planning_prior_probe
python3 scripts/lmdb_depth_planning_prior_probe.py ... --graph-name enwiki_mtc_depth_planning_prior_smoke24_uncapped
```
