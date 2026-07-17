# Bias states, step 1: the batch hierarchical bias fit — acceptance evaluation

Implements DESIGN_bias_state_augmentation.md §5 build-plan item 1 (and nothing further: no δ̃ /
dual-space states, no recursive/drift version, no square-root/QR work — those are separately gated
or Codex's lane). Per-(judge, distance-bin, channel) OFFSET states, fit as one shrunk ridge
regression per (judge, channel) ON TOP of the retained global per-channel affine calibration
(train-split-only, `affine_calibrate`) — never replacing it (the #3648 tilt lesson). Gauge: the
operating judge gpt-5.5-low is pinned to 0; every number below is fidelity/bias RELATIVE TO
gpt-5.5-low, never "semantic accuracy".

**Headline (honest):** the treatment (affine + shrunk bin offsets) improves the frozen primary
metric on BOTH corpora with near-unanimous split-seed consistency (S-marginal NLL at the ALL rung:
+0.046 nats, 38/40 seeds positive exploratory; +0.057, 40/40 fresh), the gains survive the
out-of-fold covariance robustness check (§2b), and the fitted bins reproduce the measured
stratum-bias signs 31/32. But the pre-declared confirmatory check — the paired node-block bootstrap
interval on one fixed split — **includes zero on both corpora** ([−0.051, +0.109] /
[−0.056, +0.129]), so by the strict gate this is an ENCOURAGING NULL, not a confirmed win: the mean
effect (~0.05 nats) is consistent across partitions but small relative to held-node resampling
noise at n≈160 held pairs. No post-hoc tuning was done to chase the gate; `affine+bins` remains
shadow/opt-in (promotion=false, §2).

## 0. Data provenance — the campaign was REGENERATED post-reboot

A WSL reboot (2026-07-15) cleared /tmp, destroying the original dual-judge campaign artifacts
(original md5s in HANDOFF_PR3648.md §3). Recovery (2026-07-16/17, this session; /tmp/mu_data is now
a symlink to the durable ~/mu_data):

- `campaign_pairs.tsv`: regenerated with `sample_channel_campaign.py --per-corpus 1000` (seed 0,
  deterministic). Archived in the durable ~/mu_data with the manifest; SHA-256
  `8c18488ddd664c89ff22601d8d75b6e3565b7ca6da420b6857d5c8204303481a` (pairs) /
  `bf019108079cee567bc80116c60868cf66b12475faa71dd0dc98505c35972307` (campaign_manifest.json) —
  any confirmatory rerun must verify against THESE hashes, since the historical exclusion set is
  lost. md5 `749dd34220d0a321133eb420771489df` ≠ original `f782e405…` — the sampler
  excludes pairs already scored in earlier /tmp artifacts, which were also wiped and are not
  reconstructible (LLM outputs), so the exclusion set was empty this time. The RNG stream does not
  depend on the exclusion set; the new 2,000-pair sample (1,000/corpus; identical strata counts) is
  deterministic and self-consistent with everything below.
- e5 caches rebuilt: `campaign_100k_e5.pt` (8,964 nodes incl. ancestor cones);
  `sigma_hop_behavior_slice_e5.pt` (75,901 Behavior-slice nodes — matches the documented original
  count, REPORT_sigma_hop_confirmatory.md).
- Both judges re-scored on the regenerated pairs (`score_with_codex.py`, batch 10):
  gpt-5.5-low → `campaign_scored.tsv` (2,000/2,000 rows, 0 failed batches, 140 min;
  SHA-256 `6eaf77cf5d3673836ff2b952c4a06607c803e8799b12139cf63ac1c24e3714a0`);
  gpt-5.6-luna → `campaign_scored_luna.tsv` (2,000/2,000 rows, 0 failed, 114 min;
  SHA-256 `004311c0fb9a74b0687599c10988ebea0df7f81f4d36d7ea1d67e18fdb85dd9f`).
- **Consequence: these labels are NEW judge draws on a NEW (same-methodology) pair sample.**
  Numbers are NOT comparable row-for-row with REPORT_luna_campaign.md / REPORT_cheap_judge_baseline.md;
  §3 shows the tilt SHAPE reproduces. The prior checkpoint `model_prod_namecond.pt` survived
  (repo dir, campaign-independent, SHA-256 `c1cfc3a3…` per REPORT_luna_campaign provenance).

The regenerated campaign REPRODUCES the measured tilt (compare_judges_campaign.py, all 2,000
pair-matched rows — compare REPORT_luna_campaign §1):

| stratum | n | D corr | D bias | S corr | S bias |
|---|---|---|---|---|---|
| trans | 660 | +0.753 | **+0.075** | +0.323 | −0.162 |
| sib | 332 | +0.558 | −0.084 | +0.489 | −0.112 |
| cous | 334 | +0.534 | −0.039 | +0.569 | −0.089 |
| rand | 674 | +0.477 | −0.001 | +0.527 | −0.007 |
| ALL | 2000 | +0.859 | +0.004 | +0.716 | −0.089 |

Same shape as the original: +D bias transitive-only, −S bias everywhere, magnitudes shrinking with
distance.

## 1. The estimator (fit_bias_states.py)

- **Soft bins**: deterministic, outcome-blind kernel basis over graph features only (never labels).
  Centers = strata classes h1..h5 (ancestor rows, Gaussian kernel in hop units; the hop coordinate
  is clamped at 5, so h5 explicitly means "≥5 hops") and sib/cous/rand (non-ancestor rows, same
  kernel on a unit-spaced lateral coordinate over d_sym: sib=2, cous=4, rand=cap 13). One shared
  bandwidth τ, tuned on train rows only by deterministic ENDPOINT-DISJOINT 5-fold CV (each node is
  hashed to a fold; pairs whose endpoints land in different folds are excluded from CV scoring, so
  same-node residual correlation cannot straddle fit/validation and reward overfit bandwidths;
  accepted residue: the 2-parameter affine behind the residuals is fit on the full outer-train set,
  not per inner fold); τ→0 recovers hard switching. Rows with NO usable distance signal get the
  explicit `missing` basis state (never silently mapped to rand — "no signal" ≠ "measured
  unrelated").
- **Fit**: per (judge, channel) residuals r = calibrated_reading − y_5.5 on TRAIN rows;
  b = argmin ‖r − Wb‖²/σ²_r + ‖b‖²/prior_sd² (ridge = the weak zero-prior pseudo-measurement =
  batch hierarchical partial pooling; identical to the sequential filter under identity transition).
  σ²_r defaults to var(r) — an upper bound, so shrinkage is conservative. prior_sd = 0.10.
- **Fail closed**: per fit we print the unregularized design rank (Σw is support mass, NOT an
  effective sample size — rank is the honest check; a rank-deficient supported design triggers an
  explicit warning), each state's conditional posterior variance as an information ratio
  1 − post_var/prior_var, and the condition number of the supported design. The closing mechanism
  is the PRIOR itself (external-review correction, 2026-07-17): the applied offsets are the
  coherent joint ridge posterior, so a zero-information state keeps exactly its prior (mean 0,
  variance prior_sd²) and weakly-informed states are shrunk in proportion; below-floor states are
  flagged as prior-dominated, not surgically zeroed (a point mass at 0 is a claim the model does
  not make, and dropping columns would break the batch ≡ sequential-posterior equivalence). The
  full state covariance is exposed (`posterior_cov` / `BiasStates.correction_var`) so downstream
  fusion can price bias-estimation uncertainty — a `missing` row carries the full prior variance
  rather than being treated as known-unbiased; the step-1 ladder does not yet consume it
  (cross-row terms and propagation are the square-root/QR follow-up lane). Note the coordinatewise
  information ratio does not by itself establish joint identifiability under overlapping kernels —
  the rank and condition diagnostics are printed for exactly that reason.
- Judges fitted: luna (D, S) and graph (D, S) — the four measurement channels of the ladder. The
  5.5 gauge is structural (its labels are the target). 4 fits × 9 states = 36 states on ~350 train
  rows per split (~39 rows/state, thin-bin identifiability carried by the shrinkage + soft-w
  smoothing exactly as §1 of the design anticipates).
- 12 synthetic unit tests (`test_bias_states.py`): implied-correction sign/magnitude recovery,
  zero-support fallback, refit-after-drop correctness, missing≠rand, boolean-mask handling,
  determinism, hard-switch limit, and the affine-first guard (a pure slope error leaves nothing for
  the bins once the affine is retained). A 20-agent adversarial review ran on the implementation;
  all four verified correctness findings were fixed before the acceptance run.

## 2. Acceptance evaluation — node-disjoint ladder, frozen metric

Frozen BEFORE the run: primary metric = held-out **S-marginal NLL at the ALL rung** on the
node-disjoint ladder (`run_sym_channel_fusion.py --debias affine+bins`, 40 split seeds + paired
two-endpoint node-block bootstrap on the fixed seed-0 split, B=2000); control = the existing global
affine only, treatment = identical pipeline with the shrunk bin offsets applied to the four
measurement channels (offsets refit per split, train rows only). Secondary: D-marginal and joint
NLL. Gains are paired per row (control − treatment; positive = treatment better).

### Ladder (mean ± split SD across 40 node-disjoint splits; NLL ↓)

exploratory-campaign (1,000 rows; ~360 train / ~160 held per split):

| rung | joint (affine) | joint (+bins) | S (affine) | S (+bins) | D (affine) | D (+bins) |
|---|---|---|---|---|---|---|
| prior | +0.001 | +0.001 | −0.113 | −0.113 | +0.288 | +0.288 |
| +graph_D | −0.298 | −0.502 | −0.183 | −0.253 | −0.027 | −0.208 |
| +graph_D+graph_S | −0.802 | −0.942 | −0.687 | −0.700 | −0.096 | −0.230 |
| +graph_D+luna | −1.077 | −1.264 | −0.640 | −0.779 | −0.408 | −0.485 |
| **ALL** | **−1.220** | **−1.313** | **−0.784** | **−0.830** | **−0.426** | **−0.483** |

fresh-campaign (1,000 rows):

| rung | joint (affine) | joint (+bins) | S (affine) | S (+bins) | D (affine) | D (+bins) |
|---|---|---|---|---|---|---|
| prior | +0.098 | +0.098 | +0.000 | +0.000 | +0.185 | +0.185 |
| +graph_D | −0.252 | −0.370 | −0.015 | −0.067 | −0.167 | −0.285 |
| +graph_D+graph_S | −0.853 | −0.959 | −0.616 | −0.644 | −0.198 | −0.280 |
| +graph_D+luna | −1.068 | −1.250 | −0.538 | −0.698 | −0.476 | −0.534 |
| **ALL** | **−1.248** | **−1.334** | **−0.722** | **−0.778** | **−0.487** | **−0.529** |

### Paired control-vs-treatment gains (ALL rung; positive = bins help)

| effect | exploratory | fresh |
|---|---|---|
| **S-marginal [primary]** | **+0.046 ± 0.025 split SD; 38/40 seeds +; boot +0.038 [−0.051, +0.109]** | **+0.057 ± 0.021; 40/40 +; boot +0.042 [−0.056, +0.129]** |
| D-marginal | +0.058 ± 0.036; 39/40 +; boot +0.079 [−0.039, +0.180] | +0.042 ± 0.042; 34/40 +; boot +0.100 [−0.038, +0.221] |
| joint | +0.092 ± 0.038; 40/40 +; boot +0.101 [−0.039, +0.223] | +0.086 ± 0.048; 39/40 +; boot +0.134 [−0.035, +0.293] |

(split SD is descriptive Monte Carlo partition stability; the bootstrap CI is the pre-declared
confirmatory quantity, conditional on the fixed seed-0 fitted split. These are the numbers after
the external-review corrections — coherent-posterior fallback, endpoint-disjoint inner CV, hop≥5
clamp; they moved by ≤0.002 nats from the pre-correction run, i.e. the corrections did not change
the result.)

### 2b. Robustness: out-of-fold covariance estimation (`--covariance oof`)

External-review concern: in the default harness, R/C are fit on the same train rows the
calibrations saw, which is slightly optimistic for the arm with more fitted parameters. Rerunning
with R/C estimated from node-disjoint OUT-OF-FOLD residuals (3 inner folds, calibrations + bins
refit per fold, both arms symmetric):

| effect (oof) | exploratory | fresh |
|---|---|---|
| **S-marginal [primary]** | +0.047 ± 0.022; 39/40 +; boot +0.032 [−0.047, +0.098] | +0.054 ± 0.019; 40/40 +; boot +0.050 [−0.034, +0.126] |
| D-marginal | +0.058 ± 0.032; 40/40 +; boot +0.067 [−0.030, +0.154] | +0.045 ± 0.033; 36/40 +; boot +0.099 [−0.031, +0.218] |
| joint | +0.093 ± 0.033; 40/40 +; boot +0.085 [−0.036, +0.192] | +0.088 ± 0.039; 40/40 +; boot +0.143 [−0.012, +0.290] |

The control arm's ladder moves negligibly under oof (ALL joint −1.223 vs −1.220 exploratory), and
the treatment gains are preserved — the in-sample covariance optimism was NOT the source of the
observed gains. The gate outcome is unchanged (all primary CIs still include zero; fresh joint is
a near-miss at [−0.012, +0.290]).

**Verdict: encouraging null under the strict gate — NOT promoted.** Every point estimate is
positive, split-seed sign consistency is near-unanimous, and the effect survives being decomposed
into D/joint secondaries — but no bootstrap interval excludes zero (the intervals admit losses of
~0.04–0.05 nats). With ~160 held pairs per split, a +0.05-nat effect sits inside held-node
resampling noise; the split-seed consistency says the SIGN is stable to the partition, which is a
weaker (descriptive) claim the harness's own output labels as such. Per the pre-registered rule,
`affine+bins` remains SHADOW/OPT-IN: the harness default stays `--debias affine`, and Filing v1
consumers must opt in explicitly, knowing the gate was not cleared. A power upgrade (larger overlap
set, or pooling the bootstrap across corpora) is the honest next step if confirmation matters
before Filing v1.

```yaml
# machine-readable promotion state
experiment: bias_states_step1
primary_metric: s_marginal_nll_at_ALL
promotion: false
default_mode: affine        # affine+bins is shadow/opt-in
gate: paired node-block bootstrap CI excludes zero on both corpora
```

## 3. Per-bin posteriors vs the measured stratum table (sign check)

`fit_bias_states.py` (standalone CLI, node-disjoint seed-0 train split): **31/32 stratum×channel
signs reproduced** (luna.D and luna.S × 8 strata × 2 corpora). The single mismatch is fresh h1
luna.D, where the measured bias is −0.004 — statistically zero, so its sign is noise. The measured
positive-transitive / negative-lateral D tilt and the everywhere-negative S tilt both survive the
affine (they are per-bin structure, not slope), and the implied per-bin biases track the measured
values closely (e.g. exploratory luna.S: measured −0.162/−0.100/−0.064/−0.027 on h-pool/sib/cous/rand
vs implied −0.168/−0.084/−0.058/−0.026 at the fit's resolution).

New information the binned states surface (the design's stated motivation): a within-hop trend
appears as a DESCRIPTIVE, IN-SAMPLE fit diagnostic — e.g. fresh luna.D offsets run −0.169 (h1) →
−0.038 (h2) → +0.031 (h3) → +0.058 (h4) → +0.085 (h5), suggesting luna under-reads direct parent
links relative to 5.5 and over-reads deep transitive ones, structure a single per-stratum constant
(let alone a global affine) cannot express. Like the 31/32 sign check, this is measured on the same
train rows the offsets were fit on: it says the fit reproduces the measured bias pattern, not that
the pattern is confirmed out-of-sample — held-out confirmation is exactly what §2's gate tests (and
did not pass). Full posteriors, information ratios, and support masses for all four channels × both
corpora are in the run log (see Repro).

## 4. Fail-closed diagnostics (fixed seed-0 splits, both corpora)

- Unregularized design rank 8/8 supported states on every fit (40 splits × 4 channels × 2 corpora);
  the 9th state (`missing`) has zero support on this fully-in-graph campaign and its posterior
  therefore EQUALS its prior on every fit (the mean 1.0 prior-dominated states/split is exactly
  this state — under the corrected semantics it is flagged, and the coherent posterior keeps it at
  the prior mean 0 with the full prior variance).
- Condition number of the supported design: 9.3 (exploratory fixed split, τ=0.5) / 6.2 (fresh,
  τ=0.25); max across all 40 MC splits 129 / 1797 (larger τ ⇒ more column overlap; the ridge keeps
  the solve stable and the printed value is the unregularized honesty check).
- Per-state information ratios 0.46–0.97 on supported states — all comfortably above the 0.10
  floor on every split. Bandwidth chosen per split: mode τ=0.75 (range hard-switch 1e-6 to 1.0
  after the endpoint-disjoint CV correction) — most splits still prefer genuine soft sharing
  between neighboring bins over hard switching.

## 5. Caveats

- Labels regenerated post-reboot (§0): new judge draws, new pair sample; historical tables are a
  qualitative, not quantitative, reference. All original /tmp md5s are superseded by the hashes
  above.
- All estimates are relative to gpt-5.5-low (gauge); the human-verified gold subset remains the
  known, deferred upgrade for an absolute frame.
- The strict acceptance gate is not met (§2): `affine+bins` is shadow/opt-in (promotion=false),
  not the default. When Filing v1 needs per-stratum debiasing, opting in is a deliberate decision
  weighing this null against the demonstrably wrong global-only correction — not an automatic
  consequence of this PR.
- Covariance-estimation asymmetry (external review): in the default `--covariance train` mode the
  R/C blocks are fit on the same train rows the calibrations saw, which is slightly optimistic for
  whichever arm fits more parameters (the treatment's 36 offsets + bandwidth vs the control's
  handful). §2b reports the `--covariance oof` robustness check (node-disjoint out-of-fold
  residuals, both arms symmetric). Full marginalization of the bias posterior into R (using the
  exposed `correction_var`) is the principled upgrade, deferred to the square-root/QR lane.
- graph_D's large bin offsets (−0.28…+0.19 across hops) partly re-express the known nonlinearity of
  the d→D affine, not judge bias per se; the luna channels are the load-bearing case for the
  bias-state interpretation. graph_S's linear model already consumes the bin-defining features, so
  its offsets are small, as expected.
- σ²_r = var(r) overestimates row noise (contains the bin structure itself) → shrinkage is
  conservative; a one-step reestimate is a possible refinement, not done here.
- Single campaign; per-bin GAIN states, δ̃ dual-space states, and cross-campaign drift (process
  noise) are explicitly out of scope (spec §5 items 2–3).

## Repro

```
# data recovery (step 0; /tmp/mu_data → ~/mu_data symlink)
python3 sample_channel_campaign.py --per-corpus 1000
python3 prep_campaign_e5.py
# Behavior-slice cache: build_e5_tables over the load_feature_graph(fresh) slice nodes
python3 score_with_codex.py --pairs /tmp/mu_data/campaign_pairs.tsv --batch 10 \
    --out /tmp/mu_data/campaign_scored.tsv --responses /tmp/mu_data/campaign_responses.txt
python3 score_with_codex.py --pairs /tmp/mu_data/campaign_pairs.tsv --batch 10 \
    --model gpt-5.6-luna --judge gpt-5.6-luna \
    --out /tmp/mu_data/campaign_scored_luna.tsv --responses /tmp/mu_data/campaign_luna_responses.txt
python3 compare_judges_campaign.py                     # §0 stratum table

# estimator unit tests + standalone fit (§3 sign check, §4 diagnostics)
python3 -m pytest test_bias_states.py -q
python3 fit_bias_states.py                             # log: /tmp/mu_data/bias_states_cli_output.txt

# acceptance ladder (§2; control vs treatment, paired)
python3 run_sym_channel_fusion.py --debias affine+bins # log: /tmp/mu_data/ladder_treatment_output_v2.txt
# robustness (§2b): out-of-fold covariance, both arms symmetric
python3 run_sym_channel_fusion.py --debias affine+bins --covariance oof
                                                       # log: /tmp/mu_data/ladder_treatment_oof_output.txt
```
