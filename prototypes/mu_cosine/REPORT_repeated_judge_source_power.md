# Repeated-judge source dependence — Stage-A power-harness report

## Bottom line

This PR implements and preregisters the no-history Stage-A source-dependent power harness.  It does **not**
report a power result: the immutable full discovery and confirmation runs have not been launched.  No attempted-
input history, real Nomic embedding, candidate identity, label, judge response, or model output was read, and
no judge/model call was made.  Every downstream authorization remains false.

The central correction is that source strength cannot be treated as one shared scalar or replaced by the
largest value.  For every frozen design `G>K`, `H E H.T-I` is indefinite.  Therefore `.20` is not a universal
Loewner upper envelope, and the exploratory and fresh corpora need not have the same strength.  The harness
covers all 25 ordered pairs from `eta_source,c={0,.025,.05,.10,.20}`.

The full run is intentionally deferred until this tooling and its cost are reviewed.  A representative
`K=128,G=800` audit-only pilot completed the complete 5-by-5 source-strength grid with the full candidate and
multiplier settings in 17.92 seconds for one null draw per null/pair cell and one planted evaluation replicate.
That validates execution and resource shape, not power, type-I error, or a campaign size.

## Frozen statistical procedure

The exact topology bridge supplies a strictly positive-definite, unit-diagonal region exposure matrix `E` and
registered component-to-region incidence `H` for each `(corpus,K,G)`.  Persistent error is generated with the
region-factor representation

```text
u_c = sqrt(1-eta_source,c) epsilon_c + sqrt(eta_source,c) H_c z_c,
Cov(u_c) = [(1-eta_source,c) I + eta_source,c H_c E_c H_c.T]
           tensor C_item tensor B.
```

Complete source regions are atomic in five outer and three stable inner folds.  Prompt blocks are made inside
one outer/inner signature, contain at most ten components, and spread repeated sources before duplicating one
inside a prompt.  Point selection uses equal-component marginal quasi-NLL; source and prompt dependence enter
the generator, folds, and an inflated PSD graph-aware prompt-plus-source multiplier.  That multiplier is a
calibrated working covariance, not a claim that it upper-bounds arbitrary real covariance.

One nominal max-t statistic spans both endpoints, both corpora, and all five inference source strengths per
corpus.  The full joint 20-coordinate statistic and lower bounds are recomputed for every generator-strength
pair.  Expensive corpus contributions are prepared once and reused; finalized corpuswise bounds are never
combined.

For each `(K,G,R)`, the selector barrier uses 1,999 draws in each of 50 cells: two nulls by 25 corpus-strength
pairs.  Each cell uses one-based order-statistic position 1,900 with strict `observed > threshold`, and the
operational threshold is the maximum over all cells.  Discovery then uses 200 replicates per scenario/pair.
It selects the smallest passing `G`, then the coarsest passing `K`, using one-sided exact
Clopper--Pearson bounds rather than raw rates.

Discovery unlocks nothing.  A fixed selected pair must pass a seed-disjoint second calibration and 200-
replicate confirmation.  Its Bonferroni family has `M=425` gates:

- six primary truths by 25 pairs: 150 primary-event and 150 paired topology-over-derangement gates;
- two nulls by 25 pairs: 50 false-promotion gates; and
- three cumulative-kernel deranged-DGP controls by 25 pairs: 75 gates.

At `alpha=.05/425` with 200 replicates, a power/topology gate needs at least `180/200` successes to have lower
bound at least `.80`, while a null/deranged gate permits at most `5/200` promotions to keep its upper bound at
most `.10`.  The max-t bounds remain nominal; the exact claims here are the finite null-selector rule and the
binomial rate bounds.

## Exact source bundle

The tracked topology report intentionally omitted its exact matrices and allocations.  Stage A therefore uses
the compact exact bundle `repro/repeated_judge_source_power/source_design.json`, which contains all six
exposure matrices and all 24 registered allocations:

- compact bundle: 969,914 bytes, SHA-256
  `da7c2ec6d003150aeb0465eb099508aea9918b495ff00ae25ea3f6e44cfe5fb9`;
- reviewed parent payload: 2,767,735 bytes, SHA-256
  `bf9a09c35e54bd36c2e7efea19c432ccf1e9105ff67c4154cfc1c6e744a843b2`.

Loading revalidates the parent and tracked-summary content records, exact matrix/allocation encodings, source-
strength grid, authorization block, and path-free canonical identity.  Every one of the 24 scientific designs
builds with Cholesky and passes the source-fold/prompt gates.  A future merely PSD matrix requires a prospective
square-root/rank-tolerance amendment; this runner does not clip or load a failed factor.

## Reuse, checkpoints, and provenance

For a fixed phase/configuration/scenario/replicate/corpus, all five generator strengths reuse the same base
component, region, call, request, wave, missingness, geometry, and latent-state streams.  The two corpora have
independent streams.  Ten expensive corpus worlds are computed once and combined at the same replicate index
into 25 joint worlds.  This common-random-number construction preserves every pair's distribution; arbitrary
dependence between pairwise gates is allowed by Bonferroni.

The runner caches exact designs and deterministic source splits once per worker/configuration, limits each
worker to one BLAS thread, bounds pending spawn work, and uses atomic content-hashed checkpoint shards.  Null
shards retain small corpus/pair maxima.  Power shards retain compact finalized pair records but deliberately do
not persist the large component-level corpus arrays.  Resume validates contiguous indices, seeds, thresholds,
pair grids, schemas, and the scientific fingerprint; a conflicting overwrite fails closed.

Provenance includes content hashes for the design, science, bundle loader, baseline harness, runner, exact
bundle, and reviewed parent, plus path-free Python/NumPy/BLAS identity.  Inputs and scientific files are
rechecked at worker startup and before output.  One pilot attempted while these files were still changing
failed that check and was discarded; the stable rerun below passed.

## Timing and workload

The representative audit-only pilot used `K=128,G=800,R=3`, both null types, the full five-by-five generator
grid, the default gamma/item-rho/mean-ridge grids, one null draw per cell, one
`cumulative_rho_0.10` evaluation replicate, 999 multiplier draws, two spawned workers, and checkpoints.

| measure | observed |
|---|---:|
| wall / user / system time | 17.92 s / 35.71 s / 7.18 s |
| GNU-time maximum RSS | 78,800 KiB |
| complete output | 620,435 bytes |
| output SHA-256 | `3b74736d8444e9c68b17621ac4f4e6f95c57c61b9adbfa339fee4a922677c195` |
| checkpoint tree | 5 files, 89,546 bytes |

GNU time reports the monitored process's maximum RSS; it is not an asserted sum across workers.  The output
and checkpoints live under `/tmp` and are not committed.  Because the pilot has only one draw and one
replicate, its observed rates and selected candidates are meaningless and every authorization is false.

The exact discovery has 18 registered configurations.  With corpus-world reuse it still requires 1,223,640
expensive corpus worlds, 1,799,100 cheap null-pair combinations, and 1,260,000 joint power-pair combinations.
A selected-pair confirmation adds 67,980 corpus worlds, 99,950 null-pair combinations, and 70,000 power-pair
combinations, for at most 1,291,620 expensive corpus worlds.  The runner submits 2,400 bounded chunks per
discovery configuration and projects 241 checkpoint files per configuration; discovery plus confirmation is
at most 45,600 chunk submissions and 4,580 checkpoint files.  The one-replicate pilot does not support a
credible wall-time forecast for that heterogeneous workload.  Schedule the immutable run only after code
review and report its observed resource use rather than extrapolating an ETA from this pilot.

## Verification

Three fresh pytest processes passed 188 related tests: 55 source-power science/bundle/runner tests, 84 inherited
source-dependence/source-region/capacity tests, and 49 inherited baseline science/runner tests.  The separation
is intentional: the inherited baseline spawn test fingerprints the BLAS libraries loaded in its process, so
collecting unrelated numerical modules into that same process can give the parent and spawned child different
loaded-runtime sets.  The baseline suite passes in its clean process, and the new runner's serial/parallel and
checkpoint-resume equivalence passes in the focused 55-test process.  `py_compile` and `git diff --check` also
pass.

The final audit found no statistical or reproducibility blocker.  Its one low-severity finding was that the
design promised recorded prompt/source diagnostics while the runner only enforced them internally.  The final
payload now records, per corpus, the exact integer prompt-by-source incidence matrix, row/column identities,
analysis signature of every prompt block, bipartite rank/components, maximum prompt source share, outer/inner
cluster summaries, and every split gate.  That audit-only addition explains the larger final pilot payload.

## Authorization and next stage

Only a complete exact discovery plus passing fixed-pair confirmation may set
`attempted_input_identity_inventory_unlocked=true`.  Custom, smoke, timing, incomplete, discovery-only, and
`R=4` runs cannot.  Candidate enumeration, real Nomic work, judge calls, live campaign, covariance deployment,
independent batching, QR specialization, and CUDA remain false in every Stage-A outcome.

If Stage A eventually confirms, Stage B is fixed-design: build the identity-only history inventory; enumerate
and gate the topology-only universe; use a revision-pinned Nomic cache first only for agreement-cell quotas;
freeze exact packing at the selected `(K,G,R=3)`; then compute the Nomic Gram on those immutable components and
rerun the full null/evaluation/confirmation procedure on realized `H`, `E`, folds, and prompts.  Cross-corpus
revision/wave/session/calibration/source incidence must also be audited.  Any shared random or fitted nuisance
requires an amended DGP and multiplier; a common pinned model/prompt alone is a fixed conditioned-on stratum.

`JointPosterior` remains the separately trained and held-out-calibrated decision comparator.  It is neither
this source kernel nor the joint square-root/QR conditioner.  Real covariance promotion and triangularization
remain downstream of repeated-residual calibration and the safety gates.

## Reproduction

Representative diagnostic pilot:

```bash
python prototypes/mu_cosine/run_repeated_judge_source_power.py \
  --config 128:800:3 \
  --source-eta-grid 0 .025 .05 .10 .20 \
  --null-types block_null source_smooth_mean_null \
  --scenarios cumulative_rho_0.10 \
  --null-draws 1 --power-replicates 1 --multiplier-draws 999 \
  --workers 2 \
  --checkpoint-dir /tmp/repeated_judge_source_power_G800_exactpilot_checkpoint_v4 \
  --out /tmp/repeated_judge_source_power_G800_exactpilot_v4.json \
  --audit-only
```

Eventual immutable run, only after review and with durable checkpoint storage:

```bash
python prototypes/mu_cosine/run_repeated_judge_source_power.py \
  --full-prereg \
  --workers 8 \
  --checkpoint-dir /path/to/immutable-source-power-checkpoints \
  --out /path/to/repeated_judge_source_power.json
```

Scientific overrides alongside `--full-prereg` are rejected.  Worker count, checkpoint/output paths, and
summary projection are operational only.  The normal CLI exits 2 unless the exact confirmation passes;
`--audit-only` permits a zero exit for diagnostics without changing authorization.
