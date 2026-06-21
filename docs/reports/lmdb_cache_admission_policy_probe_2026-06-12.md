# LMDB Cache Admission Policy Probe

This pass turns the depth prior calibration output into an explicit cache
admission policy.  The policy is still wired only into the probe/report path,
not the production cache materializer.

## Policy Actions

The helper returns one of four actions:

| action | meaning |
|--------|---------|
| `materialize_exact` | Build and cache the exact recurrence histogram. |
| `materialize_capped` | Build/cache a capped or cycle-approximate histogram because exactness is not guaranteed, but predicted storage is within budget. |
| `use_parametric_prior` | Store a compact closed-form/parametric prior instead of a histogram. |
| `skip_cache` | Do not cache this distribution under the current byte budget. |

The inputs are:

- empirical-prior predicted bytes;
- safety factor;
- recurrence cap/cycle flags;
- max allowed histogram bytes;
- realized histogram bytes when available; and
- estimated parametric-state bytes.

When realized histogram bytes are available, the policy uses them as an
under-prediction guard by comparing the byte budget against the larger of the
safety-adjusted prior estimate and the realized histogram size.

## Smoke Command

```bash
python3 scripts/lmdb_depth_planning_prior_probe.py \
  --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident \
  --root 7345184 \
  --graph-name enwiki_mtc_cache_admission_policy_smoke \
  --target-depths 1,2,3,4 \
  --children-per-node 24 \
  --frontier-limit 180 \
  --targets-per-depth 4 \
  --max-parent-depth 24 \
  --max-prior-depth 24 \
  --tail-epsilon 0.001 \
  --safety-factor 1.25 \
  --max-histogram-bytes 1024 \
  --parametric-bytes 64 \
  --path-cap 50000 \
  --expansion-cap 50000 \
  --seed enwiki-mtc-depth-prior-calibration-v1 \
  --output-dir /mnt/c/Users/johnc/Scratch/depth-planning-prior-admission
```

Outputs:

```text
/mnt/c/Users/johnc/Scratch/depth-planning-prior-admission/enwiki_mtc_cache_admission_policy_smoke_depth_planning_prior_summary.json
/mnt/c/Users/johnc/Scratch/depth-planning-prior-admission/enwiki_mtc_cache_admission_policy_smoke_depth_planning_prior_summary.md
```

## Results

The smoke used the same bounded enwiki MTC sampling shape as the calibration
pass and covered `16` comparable targets across `L_max` buckets `12`, `15`, and
`24`.

Cache admission actions:

| action | rows |
|--------|-----:|
| `materialize_capped` | 2 |
| `use_parametric_prior` | 14 |

Cache admission reasons:

| reason | rows |
|--------|-----:|
| `recurrence_not_exact_and_histogram_over_budget` | 14 |
| `recurrence_not_exact_but_within_budget` | 2 |

The result matches the intended policy framing.  All comparable recurrence rows
were cycle-approximate, and five rows in the `L_max=24` bucket still hit a cap.
With `--max-histogram-bytes 1024`, the shallow `L_max=12` and `L_max=15` rows
fit as capped materializations, while the deeper `L_max=24` rows fall back to a
parametric prior.

## Validation

```bash
python3 -m unittest tests.test_lmdb_depth_planning_prior_probe tests.test_parent_histogram_recurrence tests.test_lmdb_parent_branching_diagnostic
python3 -m py_compile scripts/lmdb_depth_planning_prior_probe.py tests/test_lmdb_depth_planning_prior_probe.py
python3 scripts/lmdb_depth_planning_prior_probe.py ... --graph-name enwiki_mtc_cache_admission_policy_smoke
```
