# Two-judge posterior — first build (k=2 GLM): joint beats product, but the cross pseudo-judge doesn't add

*First build of `DESIGN_two_judge_posterior.md`. `fit_two_judge_posterior.py` fits the joint `P(D,S | features)`
over the directional (D) and symmetric (S) operators and walks the combiner ladder, measuring held-out (node-
disjoint) log-loss. 880 LLM-scored Wikipedia pairs (gpt-5.5-low, `wiki_rel_scored.tsv`), 20 splits. 2026-07-06.*

## Setup
- **Labels** (fuzzy LLM `mu[]`, so D & S can co-occur): D = max over {subcategory, subtopic, element_of,
  super_category}; S = max over {see_also, assoc}; binarised at 0.5. Joint class ∈ {00,01,10,11}.
- **Features** (the two judges + model): `μ_D` = model directional readout (max of μ_HIER either way),
  `μ_S` = model symmetric readout (μ_SYM), `d` = graph up-walk hit-prob.
- **Pseudo-judges:** `μ_D², μ_S²` (self / confidence rung), `μ_D·μ_S` (cross / correlation rung).
- **Metric:** held-out log-loss of the joint (D,S), mean over 20 node-disjoint splits.

## Result
```
n=880   D+ 45%   S+ 16%   BOTH(1,1) 4%   corr(D,S label) = −0.19
model                         held-out log-loss (20 splits)
product-of-marginals          0.7608 ± 0.110
joint linear (μ_D,μ_S,d)      0.7113 ± 0.114      ← the win
joint +self (μ_D²,μ_S²)       0.7105 ± 0.117
joint +CROSS (μ_D·μ_S)        0.7107 ± 0.117
```

## Two findings
1. **JOINT beats product-of-marginals (0.711 vs 0.761)** — the core claim of the design holds: the operators are
   correlated and modeling them jointly captures what product-of-marginals (which forces `P(D,S)=P(D)P(S)`) throws
   away. This reproduces the `mu_posterior` "joint > PoE" result specifically for the D/S operators.
2. **The explicit cross pseudo-judge (`μ_D·μ_S`) adds nothing** — the three joint rungs are identical within noise.

## What that refines in the design
The design leaned on the cross pseudo-judge as *the* large-n correlation term. Empirically, **the correlation here
is UNCONDITIONAL** — a baseline "a pair is *either* directional *or* symmetric" (`corr(D,S) = −0.19`, co-occurrence
only 4%). The **joint multinomial already captures unconditional correlation through its 4-class structure** (the
class intercepts encode the co-occurrence rate), so the explicit `μ_D·μ_S` feature is redundant. The cross
pseudo-judge earns its keep only for **feature-CONDITIONAL** correlation — where the D↔S coupling *varies with* the
readouts — which this h=1 data doesn't have.

So the essential second-order move is **joint modeling** (a multinomial over the operator outcome), *not* the
interaction feature. Restated on the ladder: product-of-marginals (independent) → **joint (captures unconditional
correlation) = the win** → +cross (captures conditional correlation = data-dependent, not needed here).

## CONTINUOUS μ reverses it — heteroscedasticity IS real (2026-07-06, user)
The analysis below binarised D,S at a threshold — which (user) is wrong for a fuzzy set: it discards the graded
membership and is threshold-dependent. Redone on the **continuous** μ (`corr(μ_D,μ_S)`, max over each relation
group; Pearson is shift-invariant so `μ−0.5` centering doesn't change r — the fix is *continuous vs binary*):

| hop | SimpleMind corr [95% CI] | Wikipedia corr [95% CI] |
|---|---|---|
| 1 | +0.20 [−0.22,+0.51] | **−0.83 [−0.91,−0.69]** |
| 2 | +0.23 [−0.22,+0.60] | **−0.70 [−0.83,−0.54]** |
| 3 | +0.51 [+0.16,+0.76] | −0.06 [−0.46,+0.28] |
| 4 | +0.51 [+0.26,+0.71] | −0.18 [−0.48,+0.12] |
| 5 | +0.49 [+0.22,+0.69] | +0.25 [−0.06,+0.53] |
| pooled | +0.41 [+0.26,+0.55] | −0.20 [−0.33,−0.06] |

- **Wikipedia is significantly HETEROSCEDASTIC** — `corr(μ_D,μ_S)` runs −0.83 (h1, compete) → +0.25 (h5), h1 & h5
  CIs **disjoint**. This is the **lateral-drift** prediction (user): climbing hops, the relation stops being
  strictly directional and drifts lateral, so D & S stop competing. The binarised analysis below **hid** this
  (threshold noise turned the clean trend into overlapping phi-coefficients — a measurement artefact, corrected).
- SimpleMind: strong positive throughout (pooled +0.41), hop-trend suggestive (+0.2→+0.5) but CIs overlap.
- **⇒ the cross pseudo-judge IS warranted** (hop-conditional, `μ_D·μ_S` coupled to `d`): the D↔S covariance
  genuinely varies across the space (Wikipedia), the QDA condition the h=1-only linear fit could not see. **Lesson:
  fit/measure on the continuous fuzzy μ, not binarised labels** — binarisation destroyed the very signal.

## [SUPERSEDED by the continuous analysis above] Is the cross term ever justified? — (binarised, noisy)
The cross pseudo-judge is the QDA/heteroscedastic term: it earns its keep ONLY if `Cov(D,S)` **varies across the
space** (LDA/linear suffices for a constant covariance — a Gaussian's score is linear, its correlation a single
change of basis absorbed into linear weights). Deciding check: does `corr(D,S)` vary with hop? **Point estimates
looked like it did — but with 95% bootstrap CIs (n≈45/hop) they don't:**

| | SimpleMind (per-hop → **pooled**) | Wikipedia (per-hop → **pooled**) |
|---|---|---|
| corr(D,S) | +0.24/+0.22/+0.42/+0.26/+0.20 → **+0.29 [+0.15,+0.43]** | −/−0.31/−0.10/−0.18/+0.18 → **−0.03 [−0.16,+0.09]** |

- **Hop-variation is NOT significant** — every per-hop CI overlaps every other (they are ±0.3 wide). The apparent
  swing was small-n noise; there is **no evidence the covariance is hop-conditional**, so the pseudo-judge is **not
  yet justified**. (Earlier draft over-read the point estimates — corrected.)
- **The real, significant signal is the CORPUS difference:** SimpleMind D & S genuinely **co-occur** (pooled +0.29,
  CI excludes 0); Wikipedia is **indistinguishable from independent** (pooled −0.03, CI includes 0). The `1996`-style
  "lateral drift ⇒ more correlation at depth" prior is **not** borne out (no significant hop trend).
- **⇒ the separation trick (joint modelling) is doing the work, without the pseudo-judge.** Whatever correlation
  exists (SimpleMind's real +0.29; Wikipedia's weak −0.19 at h=1) is CONSTANT enough to be captured **linearly** —
  a single change of basis folded into the joint's weights. The cross pseudo-judge earns its keep only under
  *significant* heteroscedasticity, which we do not observe. **Keep the separation trick; hold the pseudo-judge**
  until a corpus shows a covariance that provably varies (would need more data/hop than n≈45 to detect).

## Caveats & next
- **h=1 Wikipedia is the wrong regime for the cross term.** D↔S *anti*-correlate here (compete); the *positive
  co-occurrence* that would exercise a conditional cross-term appears at **deep hops** / concept graphs
  (`REPORT_multihop_direction`, SimpleMind ~2× symmetric mass). The cross pseudo-judge should be retested where D
  and S genuinely co-occur.
- **`μ_rev=0` clamp** (hard constraint) is not exercised by this D/S build; it enters the directional sub-model.
- **`LLM_op` is the label here, not a feature** — this is the *student/distillation* direction (predict the LLM
  from free features `μ_D, μ_S, d`). The teacher (LLM as a feature toward an independent op) needs a third label
  source; deferred.
- Next: (a) retest the cross term on the deep-hop / SimpleMind co-occurrence regime; (b) move `d` to a continuous
  walk on the Pearltrees multi-parent DAG (on h=1 categories `d` is near-binary and contributes little).

Repro: `fit_two_judge_posterior.py --scored wiki_rel_scored.tsv --e5-cache wiki_rel_e5.pt --graph 100k_cats/... --seeds 20`.
