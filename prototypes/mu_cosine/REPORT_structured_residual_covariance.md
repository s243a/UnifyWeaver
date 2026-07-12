# Structured conditional-residual covariance — evidence-gate report

**Run date:** 2026-07-11

**Status:** primary 10-seed run complete; preregistered engineering gate **failed**

**Decision:** retain independent item blocks in the joint square-root/QR conditioner.  Do not build the
semantic/graph inverse-root or CUDA path from this evidence.

## Bottom line

Frozen semantic and graph-feature proximity did not predict enough transferable cross-item conditional
residual covariance to justify a more complex conditioner.  Both structured models lost held joint residual
likelihood on average in both corpora, and their downstream posteriors were worse.  This remained true after
giving dense LMC an exact block fallback, an exact separable fallback, and a nonzero optimization start.

There is a different, strong signal: a train-fitted regional residual **mean** improved conditional-residual
NLL on every one of the 20 corpus/split records.  That is evidence for predictable local bias, not correlated
random measurement noise.  It deserves a separate validation, because its operating-judge decision metrics
did not improve consistently.

This result does not remove dense correlated batches from the QR conditioner.  The conditioner remains the
correct general mechanism when a defensible covariance is supplied.  It says that the particular
semantic/graph covariance model tested here is not a reason to optimize a larger inverse-root/CUDA path.

## Protocol

The split, models, metrics, and gate were frozen in `DESIGN_structured_residual_covariance.md` before the
first 10-seed result was inspected.

- Target: GPT-5.5 operating-judge D/S and macro-decision fidelity, **not ground truth**.
- Data: 1,000 matched exploratory rows and 700 matched fresh rows.
- Splits: seeds 0–9, 40% held nodes, strict node-disjoint retained pairs; crossing pairs dropped.
- Fitting: calibration, conditionalization, feature scaling, bandwidths, regional mean, covariance, and the
  decision bridge all use train rows only.
- Conditional residual: `q = v - C.T P^-1 e`, after removing within-item prior/measurement correlation.
- Primary comparison: full held joint Gaussian NLL per scalar in original channel units.
- Structured family: PSD separable kernel covariance and a more flexible additive LMC reference over frozen
  semantic and graph-feature RBF kernels.
- Optimizer safeguards: exact block candidate for both structured models, exact separable candidate for dense
  LMC, a material seed for zero inherited factors, best-iterate retention, and planted nonseparable recovery
  coverage.

Mean ± SD below describes variation across node partitions; it is not an SE or population confidence
interval.  A positive gain means the model named in the column is better.

### Implementation-audit chronology

After the first complete run, a read-only code audit found optimizer bookkeeping and zero-factor warm-start
risks plus incomplete loading/objective metadata.  Those implementation issues were fixed without changing
the data, split seeds, model families, metrics, or gate.  The hardened rerun added exact submodel fallbacks,
a material dense start, prior-loading diagnostics, and regression tests.  Its verdict and aggregate values
reproduced the first run to rounding.  All numbers below are from the hardened rerun.

## Primary residual-likelihood result

| Corpus | Regional mean gain over global block | Separable covariance gain over regional block | Dense LMC gain over regional block |
|---|---:|---:|---:|
| Exploratory | +0.181786 ± 0.042311 (10/10 positive) | -0.000575 ± 0.003075 (3/10) | -0.000469 ± 0.003044 (4/10) |
| Fresh | +0.150996 ± 0.031034 (10/10 positive) | -0.002777 ± 0.003890 (2/10) | -0.002923 ± 0.004101 (2/10) |

The regional block mean NLLs were -0.875663 ± 0.049187 (exploratory) and
-0.759485 ± 0.042587 (fresh).  Dense LMC did improve its train composite objective over the inherited
separable candidate on all 20 fits, but only by a very small amount, and that gain did not transfer to held
joint likelihood.  The negative result is therefore not caused by returning a worse optimizer start.

## Posterior and decision guardrails

All values are the mean gain of `separable_regional` over `block_regional`; positive is better.

| Corpus | Bivariate posterior NLL | Positive seeds | Decision log-loss | Margin AURC | Maximum prior/noise relative loading |
|---|---:|---:|---:|---:|---:|
| Exploratory | -0.073120 | 1/10 | -0.009036 | -0.000394 | 0 |
| Fresh | -0.145861 | 0/10 | -0.041428 | -0.011650 | 0 |

The structured models reduced empirical 95% ellipse coverage as well: regional block → separable was
0.861 → 0.848 on exploratory and 0.848 → 0.815 on fresh.  This is consistent with harmful confidence from a
cross-item covariance pattern that did not transfer.

The gate failed independently in several places:

1. Dense LMC did not have positive mean held gain or 8/10 positive seeds in either corpus.
2. Separable covariance did not have positive mean held gain or 8/10 positive seeds.
3. The 80% recovery ratio was undefined because the macro-average dense gain was nonpositive.
4. Posterior NLL worsened in both corpora.
5. Fresh decision log-loss and AURC exceeded the allowed 0.01 degradation.

The numerical-loading guardrail passed: neither the calibrated prior covariance nor any conditional-noise
covariance required diagonal loading.

## What the regional-mean result does and does not show

The selected train-only mean was graph-feature kernel ridge on 8/10 exploratory splits, an equal semantic /
graph-feature mixture on 2/10 exploratory splits, and the equal mixture on all 10 fresh splits.  Its held
residual-likelihood gain was large and directionally stable.

That does not yet make it a production component.  Relative to the global block baseline, the regional block
slightly improved mean posterior NLL (+0.00738 exploratory, +0.00604 fresh), but decision log-loss worsened by
0.00805 and 0.00267 respectively; fresh margin AURC worsened by 0.01030.  The next statistical experiment
should distinguish a transferable regional bias field from operating-judge-specific smoothing, preferably
with soft or independent targets and nested calibration.

It also does not test the broader purpose of the graph judge—helping a model learn a dataset.  That requires
the separate training-time graph-supervision × final-fusion factorial ablation; a post-hoc residual covariance
test cannot answer it.

## Numerical diagnostics

- Retained held batches ranged from 150–168 items exploratory and 99–116 fresh, corresponding to state
  dimensions 300–336 / 198–232 and measurement dimensions 600–672 / 396–464.
- Mean separable off-item Frobenius mass was 0.188 exploratory and 0.115 fresh.  Nonzero fitted mass alone was
  not evidence: it failed held likelihood.
- Maximum whitened off-item coupling averaged 0.0447 exploratory and 0.0380 fresh.
- Materialized separable covariance condition numbers ranged 6.57–37.83 exploratory and 7.29–36.73 fresh.
- The CPU conditioner call averaged about 0.052 s exploratory and 0.022 s fresh.  This is a diagnostic timing
  of conditioning after covariance materialization, not an end-to-end or CUDA benchmark.

## Limitations

- The target is one operating judge, not human truth or an independent quality measure.
- Ten node partitions measure partition stability on one campaign realization; they are not independent
  experiments or confidence intervals.
- The graph kernel is an RBF over explicit graph features, not a shortest-path or diffusion kernel.  The
  existing structural embedding lacks fresh-corpus coverage.
- Pairwise composite likelihood is a finite-step fitting device.  Exact submodel fallbacks and planted tests
  reduce optimizer failure risk but do not prove a global optimum.
- The regional mean is selected from the same train campaign and needs independent/nested validation before
  deployment.

## Recommendation and next work

1. Keep `I_item kron Rc` as the default covariance for batched QR conditioning.
2. Do not start the structured inverse-root/CUDA optimization PR now.
3. Validate the regional mean as a separate bias model, including an augmented regional-bias-state comparator
   and soft/independent decision targets.
4. Run the separate graph-supervision × final-fusion factorial experiment to measure dataset familiarization.
5. Reopen item-kernel diagonalization and matched CUDA timing only if a future preregistered covariance model
   passes held residual, posterior, decision, and loading guardrails.

`JointPosterior` remains a separate learned nonlinear decision comparator; this experiment neither promotes
nor rejects it, and it should not be conflated with the square-root/QR numerical conditioner.

## Reproduction and provenance

```bash
python3 -u prototypes/mu_cosine/run_structured_residual_covariance.py \
  --artifact-repo /home/s243a/Projects/UnifyWeaver \
  --ckpt /home/s243a/Projects/UnifyWeaver/prototypes/mu_cosine/model_prod_namecond.pt \
  --seeds 10 --fit-steps 150 --max-pairs 4096 --bridge-epochs 300 \
  --cpu-threads 1 \
  --out /tmp/structured_residual_covariance_10seeds_provenance.json
```

Input SHA-256:

- `campaign_scored.tsv`: `c2acf399aadd35c3797171d5b42d64e45b07055802092cc28becf308d460ef09`
- `campaign_scored_luna.tsv`: `a8d951b4fd05f0ca111fbe9d9c23881bb47b790ccb335731113bfd4ae77ffe6e`
- `model_prod_namecond.pt`: `c1cfc3a3827e42a1993f4286b6a881aee7ff10eb56a76367735b9ec8fdf11f7d`
- `campaign_100k_e5.pt`: `037396a1d6552892d3d6aa04b3cc12bc6d5e6a532d3697e28da1f28e04810700`
- `sigma_hop_behavior_slice_e5.pt`: `ab51eb5d07cdd3bed34112a1d92005fcd3f8c46245cf068a7ab0a225afc6cd6b`
- `data/benchmark/100k_cats/category_parent.tsv`:
  `4881beedfd876e3abb9f1783cbc3fb8a7350e108e3f531cc4de28ef9956dc8ec`
- `data/benchmark/enwiki_cats_correct/lmdb_scoped/data.mdb`:
  `3bcfe59a3f85870f377fad1ea77547f7c3566370f6172e27748f4f7ceba5d690`

LMDB `lock.mdb` is intentionally excluded because it is mutable process state, not graph content.  The runner
passes `--campaign` to both the target loader and the campaign-row loader and records the exact consumed path.

Exact hardened-run output SHA-256 (the JSON contains wall-clock diagnostics, so an otherwise identical rerun
need not have the same whole-file hash):
`c1962b81fa8db981802c05bd9023c21c36e3fde0b852b9e55bce9a7dfa5beea4`

The compact, tracked machine-readable gate and provenance record is
`repro/structured_residual_covariance/primary_gate_summary.json`.

Validation suite:

```text
101 passed, 9 skipped
```
