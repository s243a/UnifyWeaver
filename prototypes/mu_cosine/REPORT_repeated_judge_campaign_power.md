# Repeated-judge graph covariance — no-spend campaign and power-harness report

## Bottom line

This PR produces **protocol and tooling, not a covariance result**.  It made no judge/model calls, selected no
new endpoint, and did not run the full preregistered simulation.  Consequently it reports no campaign size and
authorizes neither live scoring nor covariance deployment.

The main design change is token-efficient but statistically explicit batching.  One LLM request may contain up
to ten same-role rows from distinct endpoint components, amortizing the system prompt.  Stable prompt-block
membership defines the dependence/inference cluster; role-by-judge base order is independently deterministic
and repeats use spread rotations.  Shared-request covariance is part of the fitted/scored covariance rather
than being ignored after batching.

The tooling is suitable for reproducible dry runs.  It is deliberately not live-campaign-ready because the
repository-owned candidate builder, verified historical inventory, exact approved prompt/model/settings
contract, and list-position pilot/sensitivity do not yet exist.

## What is frozen

- three-row endpoint-disjoint components: anchor, directly adjacent, and distance-at-least-three/disconnected
  matched negative;
- two required synthetic corpora and planned `G={160,320,512,800}` components per required corpus at `R=3`,
  with `R=4` sensitivities at `G={320,800}`;
- 32 equal campaign cells per corpus, a 10% graph-source-component cap, five outer folds, and three stable inner
  folds;
- judges `gpt-5.5-low` and `gpt-5.6-luna`, with at least three fresh stateless waves per judge and a four-wave
  sensitivity;
- one deployment-eligible PSD family
  `K_gamma=gamma K_cumulative+(1-gamma)K_Nomic`, with the preregistered gamma/rho grid and one familywise
  selector maximum;
- residual Gaussian NLL and latent-state posterior NLL as distinct synthetic endpoints;
- block-null and smooth-mean-only false-promotion controls, planted cumulative/Nomic/mixture truths, and
  equal-energy topology derangements;
- `1999` null calibrations, `200` power replicates, and `999` prompt-block multiplier draws for a reported full
  run.

Only an exact `--full-prereg` run over the entire frozen grid may set the synthetic sizing fields.  Customized
or smoke runs hard-code those fields to false/null even if they happen to include every scenario.

## Request and response integrity

The materializer emits immutable split-contained prompt requests but never sends them.  Each request:

- carries no more than one row from a component and no more than ten rows total;
- includes a stable `row_id` that the eventual approved scorer/parser must echo;
- hashes the exact serialized request TSV bytes, model revision, prompt hash, settings, wave, role, and derived
  seed into `request_id`;
- remains inside one corpus/outer/global-inner signature;
- records its prompt block as the inference cluster; and
- requires provider request IDs, and nonempty provider response IDs, to be unique across logical attempts.

An otherwise frozen-shaped output is labeled
`protocol-shape-compatible-no-spend-inputs-unverified`, never “confirmatory-ready.”  Caller-supplied builder
hashes make supplied bytes reproducible but do not attest how the graph, embedding, or candidate pool was made.

## Synthetic procedure and claim boundary

Every replicate regenerates folds and prompt blocks, simulates repeat/call/request/wave/missingness effects,
selects mean ridge and graph covariance only inside training components, recalibrates nuisance covariance, and
scores untouched outer components.  The two synthetic corpus labels are independent copies of a generic DGP:
they enforce two-corpus multiplicity but do **not** simulate empirical corpus heterogeneity, real source-group
dependence, or the 32-cell sampling distribution.

The synthetic selector threshold is never reusable on real residuals.  Real analysis must recalibrate the
complete selector inside every outer-training prompt-block set.  JointPosterior remains the calibrated decision
comparator; it is not identified with a matrix factorization.  Final campaign `G`, deployment, independent
batching, QR specialization, and CUDA remain false/null until the real calibration, margin-AURC, source,
position, spectral-safety, and cross-batch gates pass.

## Exact prompt-block diagonalization

For the synthetic request schedule, every component in one prompt block has the same observed repeat schedule.
Its joint noise covariance is therefore exactly

```text
Sigma_m = I_m tensor A + J_m tensor B,
```

where `B` is the shared-request block.  An orthogonal component transform yields one collective mode with
covariance `A+mB` and `m-1` contrast modes with covariance `A`.  Residual scoring needs two 12-by-12 Cholesky
systems; posterior scoring adds two 6-by-6 systems.  This is an exact block diagonalization, not an independent-
rows approximation.

The fast path compares the complete boolean observation schedules, not counts alone.  Different schedules or
counts use the dense overlap-aware reference.  Candidates within `1e-12` of a strict-zero eligibility boundary
are recomputed densely without rounding or changing the `>0` rule.  Factors are cached only inside one fitted
nuisance object.

Dense equivalence tests at prompt-block sizes 1, 2, and 10 observed maximum residual/posterior differences of
`2.22e-16` and `3.33e-16`.  A local single-thread microbenchmark over ten 10-component blocks measured:

| endpoint | eigenmode path | dense path | local speedup |
|---|---:|---:|---:|
| residual NLL | 1.177 ms | 33.457 ms | 28.43x |
| posterior NLL | 1.815 ms | 40.362 ms | 22.24x |

This shortcut is specific to verified compound symmetry.  Arbitrary real prompt incidence may not commute with
the item kernel, so dense joint square-root/QR remains the general numerical reference.

## Compute audit and smoke runs

Before the eigenmode reduction, the exact full workload comprised 11,994 joint null draws, 16,800 joint power
replicates, 57,588 generated corpus worlds, about 1.03 million nuisance fits, and about 296.7 million
prompt-block endpoint evaluations.  Local measurements projected roughly 230--300 single-thread CPU-hours
(10--12 serial days).

The exact reduced prototype projected roughly 40--45 CPU-hours, or about 11--15 wall-clock hours on four
physical cores with one BLAS thread per worker.  This is a compute projection, not a completed run.  A
five-null-draw pilot at every `G` is still required before launching the full job.

Two diagnostic executions checked the operational path:

- default smoke, two workers: 40.15 s; an exact checkpoint resume took 0.02 s and reproduced SHA-256
  `17a9c7693a307fb15d0290f946d0475e3cafb956ffb377ca19398b8a21c3191d`;
- `G=160,R=3`, five null draws and one replicate for all 14 scenarios, four workers: 30.55 s.

Both were customized diagnostics.  Neither is a power estimate, and both retained false/null sizing and
deployment fields.

The runner uses atomic indexed shards, a hard null-threshold barrier before power jobs, canonical
scenario/replicate reconstruction, one BLAS thread per worker, content fingerprints, and start/end provenance
checks.  A changed preregistration, implementation, configuration, or checkpoint payload fails closed.

## Verification

- repeated-campaign/sampler/power/runner suites: 68 passed;
- inherited graph geometry, synthetic-v2, structured covariance, JointPosterior, and square-root conditioner
  suites: 97 passed, 9 optional CUDA skips;
- Python compilation and `git diff --check`: clean.

No ignored artifact, checkpoint, smoke JSON, or generated score input is committed.

## Required next work

1. Implement and review the repository-owned candidate builder, then freeze graph/Nomic/historical artifacts.
2. Freeze the exact approved live prompt, model revisions, settings, and keyed response parser.
3. Run a list-position engineering pilot; preregister a train-only position-by-role-by-judge adjustment and add
   a position-effect power sensitivity.
4. Run five null pilots at each registered `G`; review timing/equivalence, then run the immutable full simulation
   with checkpoints.
5. Only if the synthetic primary event passes, request a separate live-campaign authorization.  Real covariance
   still must pass calibration, decision log-loss, margin-AURC, source-component, topology, loading,
   `s_safe/delta_95`, and cross-batch safety gates before entering the joint QR conditioner.

## Reproduction

Diagnostic smoke:

```bash
python prototypes/mu_cosine/run_repeated_judge_power.py \
  --workers 2 \
  --checkpoint-dir /tmp/repeated-judge-smoke-checkpoint \
  --out /tmp/repeated-judge-smoke.json
```

Eventual immutable full run, after the remaining preregistered prerequisites and timing pilots are accepted:

```bash
python prototypes/mu_cosine/run_repeated_judge_power.py \
  --full-prereg \
  --workers 4 \
  --checkpoint-dir /path/to/immutable-checkpoints \
  --out /path/to/repeated-judge-power.json
```

Changing scientific settings alongside `--full-prereg` is rejected.  Worker count, checkpoint location, and
output path are operational only and do not enter the scientific JSON.
