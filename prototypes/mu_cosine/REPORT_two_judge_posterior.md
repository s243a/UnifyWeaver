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

## Hop-conditional CONFIDENCE Σ(hop) DOES help — but only the full covariance (`fit_hetero.py`, user)
Does *modelling* the hop-dependence improve held-out prediction? Continuous bivariate-Gaussian NLL of the (D,S)
residuals (mean model on `[μ_D,μ_S,d]`), 250 Wikipedia multi-hop pairs, 40 splits. **Key (user): the real
heteroscedasticity is in the CONFIDENCE (the diagonal `σ`), not just the correlation** — the direction is
*confident* at low hops and *ambiguous* at high hops. Measured (margin `μ_D−μ_S`): **0.62 (h1) → 0.11 (h5)** — at
h=1 μ_D=0.85≫μ_S=0.23; by h=5 they're indistinguishable.

| model | held-out joint NLL | vs constant ρ |
|---|---|---|
| (a) independent ρ=0 | −0.708 | — |
| (b) constant ρ | −0.721 | — |
| (c) ρ(hop) off-diagonal ALONE | −0.732 | +0.011 (+1.1σ) not sig |
| (d) σ(hop) confidence ALONE | −0.725 | +0.004 (+0.4σ) not sig |
| (e) σ(hop)+ρ(hop) — full Σ(hop), oracle per-hop bins | −0.782 | +0.061 (+4.9σ) HELPS |
| **(f) Σ(hop) PREDICTIVE — smooth σ(hop),ρ(hop) by MLE** | **−0.819** | **+0.098 (+8.5σ), and +3.6σ vs (e)** |

- **The full hop-dependent covariance is a significant win (+4.9σ vs constant, +6.4σ vs independent)** — even though
  *neither* `σ(hop)` nor `ρ(hop)` alone clears noise. They only help *together*: modelling the correlation with a
  wrong (constant) variance, or vice-versa, is misspecified; only the whole `Σ(hop)` is correct.
- **The driver is the CONFIDENCE term (`σ(hop)`)** — the earlier ρ-only test measured the off-diagonal and missed
  the diagonal, so it read "below noise." User's "compare learning rates" maps onto `σ(hop)`: the per-hop confidence
  IS the effective example weight; holding it constant (uniform LR) washes the effect out.
- **⇒ the second-order machinery earns its keep on multi-hop data — as the full hop-conditional Σ(hop)** (self
  pseudo-judges `μ_D²,μ_S²` = the σ diagonal, cross `μ_D·μ_S` = the ρ off-diagonal, both coupled to `d`), not the
  cross-term alone. This is the first *significant* evidence for the hop-conditional posterior. (The separation trick
  / constant-Σ remains the base; Σ(hop) adds a real, significant increment on top.)
- **BUILT (rung f): the predictive smooth `Σ(hop)`** — `σ_D(hop)=exp(a+b·hop)`, `σ_S(hop)`, `ρ(hop)=tanh(c+e·hop)`
  fit by MLE (no per-hop bins) — is **+8.5σ over constant AND +3.6σ over the oracle per-hop `Σ` (e)**. The smooth
  form *regularises* the covariance: it pools across hops instead of estimating `Σ` from ~35 pairs/bin, so the
  *buildable* model (Σ a learned function of the conditioning feature) beats the oracle. This is "`Σ(hop)` in the
  model," done — a hop-conditional covariance head with 6 extra parameters.

### WHY Σ(hop) beats constant Σ — the decoupling geometry rotates with hop (user)
The decoupling (whitening) transformation is a *function of hop*, and `Σ(hop)`'s **condition number reduces with
hop** (user's prediction):

| h | ρ | κ(Σ) | decoupling rotation |
|---|---|---|---|
| 1 | −0.83 | 11.0 | **−40°** |
| 3 | −0.06 | 7.8 | −1° |
| 5 | +0.25 | 5.3 | **+8°** |

Low hops: ill-conditioned (`ρ=−0.83` ⇒ D,S near-collinear) ⇒ strong decoupling rotation. High hops: decorrelated
(`ρ→0`), `Σ`→isotropic ⇒ ~identity transform. A **constant Σ must commit to one geometry** — but the true one
rotates −40°→~0° across hops, so it's a bad *average of incompatible geometries* (wrong at both ends); only `Σ(hop)`
fits each. Division of labor: at LOW hops the **correlation/decoupling** carries the info (joint ≫ product) while the
direction is certain; at HIGH hops `ρ→0` (joint≈product) but the **confidence/margin** degrades — `Σ(hop)` captures
both. (`κ` also mixes the variance ratio `σ_D/σ_S`; the *rotation* is the pure-correlation signal and reduces
monotonically.)

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

## SimpleMind cleanup — the depth co-occurrence was the ORG layer, not content (user, 2026-07-06)
`simplemind_clean.py`. Two confounds (user): (1) duplicate concept nodes — key case/hyphen/underscore variants of
the same title (500 keys → 464 titles, 36 merged); (2) chains climbing past the 6 `.smmx` map roots into an
**organisational super-layer** (`Applied mathematics`, `Cybernetics`, `Dynamical systems`, `Chaos theory` — the
ancestors of the map roots). Splitting the scored pairs by root type:

| root type | n | corr(μ_D,μ_S) |
|---|---|---|
| content-rooted (genuine concept parent) | 164 | +0.24 |
| **org-rooted** (broad organisational bucket) | 36 | **+0.82** |

By hop, the org-rooted pairs appear only at h≥4 — and they carry the high correlation:

| hop | content-rooted | org-rooted |
|---|---|---|
| 1–2 | +0.20, +0.23 | — |
| 4 | +0.09 (n=27) | +0.89 (n=13) |
| 5 | +0.08 (n=18) | +0.75 (n=22) |

**⇒ the "co-occurrence rises with depth" was the ORG layer, not the hierarchy.** On the **content** hierarchy `corr`
runs +0.2 (shallow) → +0.08 (deep) — *decreasing*: adjacent concepts are both nested & associated, deep content
ancestors decouple toward purely-directional. The rise came entirely from org-rooted pairs (root = a broad bucket,
where a concept is *weakly* directional *and* lateral, so both fire). Root-anchoring (content→map-root, drop the org
layer) is the reliable signal — and it does NOT show the strong heteroscedasticity the confounded read suggested.
A fully trustworthy SimpleMind correlation-vs-hop needs **re-sampling within-map on the deduped graph + re-scoring**
(deferred — a scoring run); `gen_mindmap_lineage.py` now canonicalises keys so the duplicates don't recur.

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

## SimpleMind re-validation on CLEAN within-map data (Part 2, 2026-07-06)
Regenerated per-map lineages with the improved `gen_mindmap_lineage.py` (488 chains, within-map, no cross-map org
super-layer), sampled within-map multi-hop pairs, re-scored (gpt-5.5-low, 200 pairs). `Σ(hop)` signature vs Wikipedia:

| | h1 | h3 | h5 | pooled corr |
|---|---|---|---|---|
| SimpleMind margin (μ_D−μ_S) | 0.21 | 0.20 | 0.13 | — |
| SimpleMind corr(μ_D,μ_S) | +0.17 | +0.07 | +0.43 | **+0.20 [+0.02,+0.36]** |
| SimpleMind κ(Σ) | 3.3 | 7.1 | 5.5 | — |
| Wikipedia margin | 0.62 | — | 0.11 | — |
| Wikipedia corr | −0.83 | — | +0.25 | −0.03 |
| Wikipedia κ(Σ) | 11.0 | — | 5.3 | — |

- **The "SimpleMind decouples at depth" hypothesis is NOT confirmed** — that was over-read from the (confounded)
  content-rooted split. On clean data SimpleMind corr is weakly *positive* (+0.20 pooled, CI excludes 0), roughly
  flat/rising, not decreasing. (Corrected.)
- **But the corpus-level `Σ(hop)` signatures genuinely differ (the real finding):** Wikipedia is *strongly*
  heteroscedastic (high directional confidence at low hop, margin 0.62; strong anti-corr −0.83; ill-conditioned
  κ=11; all relaxing with depth). SimpleMind is *flat* — **low directional confidence at every depth** (margin only
  0.21 at h1; μ_D=0.62, μ_S=0.41 both moderate), weak positive corr, better-conditioned (κ~3–7), little hop trend.
- **On SimpleMind D and S are never cleanly separated** — concepts read as both hierarchical *and* associative at
  all depths. Genuine (concepts are relationally richer than categories) OR the **LLM over-assigns `assoc` to
  concepts** (the user's judge-calibration dispute — open). Either way, a `Σ(hop)` head must learn a **corpus-specific
  curve** (steep-relaxing for Wikipedia, flat for SimpleMind), confirming the covariance is not universal.

## The `assoc` dispute, resolved: low-directional pairs are DATA errors, not LLM errors (user, 2026-07-06)
Inspecting the clean-SimpleMind pairs with the LOWEST directional signal (μ_D) — is it the LLM over-assigning
`assoc`, or genuine? It's genuine — the LLM is CORRECTLY rejecting bad "parents":
- **Title typo:** `Valve Body & Bonnet → "Values"` (×5) — "Values" is a corrupted "Valves"; LLM says `none=0.85`.
- **Organizational tag-nodes:** `Abel Kernel → "related"`, `"related" → Engineering` — "related" isn't a concept; `none`.
- **Spurious cross-topic lineage:** `Thermodynamics → Global bifurcations`, `Subjects of Learning → Derivative` — `none`.
- **Resource/website nodes:** `electronics-cooling.com → Fans & Blowers`, `Dortmund Data Bank → Mechanical Engineering`
  — genuinely lateral, LLM `assoc` correct. And `Number Theory → Complex Analysis` = related-not-hierarchical (`assoc`).

**⇒ the LLM is not over-lateraling concepts** — it correctly flags corrupted / organizational / spurious / resource
"parents" as non-directional. So SimpleMind's flat, low-directional-confidence `Σ(hop)` signature is a **DATA-QUALITY
artifact** (typos, tag-nodes, resource links, cross-topic lineage slips), not a judge-calibration problem. The user's
two intuitions are both upheld: the deliberate hierarchy IS the cleaner one, and the LLM is right — the *extraction*
is noisy. Fix upstream (dedup/typo-fix `Valves`, drop tag-nodes like "related", filter resource/URL nodes, fix
cross-topic lineage), not the judge.
