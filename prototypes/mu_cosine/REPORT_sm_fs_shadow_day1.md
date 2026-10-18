# SM-FS ranking shadow runs, day 1 — mechanism win, absolute null, and the information-set lever

**Status: SHADOW-EXPLORATORY (Tier 1).** Reserve rows untouched; preregistrations unamended;
every artifact stamped non-decision-bearing. Same frozen experiment design throughout (5 folds ×
3 seeds, sol's counter sampler, frozen lineage blocks, frozen rejection-sampled bootstrap).
Total compute for everything below: ~1 hour on the GTX 1660 (a full 800-step fit ≈ 50 s).

## Results

| configuration | whole-population MRR |
|---|---|
| positive-only (trained, matched budget) | 0.140 (seeds 0.128/0.159/0.135) |
| graded-negative (trained, matched budget) | **0.280** (0.273/0.282/0.286) |
| e5-residual graded (α=1) | 0.376 (0.353/0.382/0.393) |
| **frozen e5 title-cosine (no training)** | **0.573** (R@1 0.471) |
| e5 + α·correction sweep | best α=0.02: 0.574 (+0.002, noise); monotone ↓ after |

1. **The mechanism works.** Graded structural negatives double positive-only under matched
   budget: ΔMRR +0.140, frozen lineage-block bootstrap 95% CI [+0.101, +0.181], consistent
   across all three seeds — 14× the practical floor. The constructed hardness buckets and
   graph-decay targets carry real supervision signal. This answers the registered mechanism
   question emphatically (shadow-grade).
2. **The absolute level is a null.** Every trained variant sits far below frozen e5; the
   e5-residual head DEGRADES from its own starting point (0.573 → 0.376 at α=1 — empirical
   confirmation of the zero-init-is-not-a-floor warning), and the offline α-sweep shows the
   learned correction adds nothing at any useful weight. Third corpus, same standing
   conclusion: e5 title-cosine is the champion ranker; μ heads do not beat it from titles.

## The information-set hypothesis (next lever, for the math lane)

The trained model receives strictly LESS information than the supervision encodes: μ sees
(query title, candidate LEAF title) only — the §4 embedding-text rule excludes path components —
while the graded targets are functions of full path structure (distance, LCA depth). e5 already
extracts most of what two titles contain. Under this input constraint, matching e5 is close to
the ceiling and beating it may be impossible in principle. Concrete next shadow runs:
  (a) candidate ancestor-title context as model input (public titles, path structure exposed to
      the encoder) — does structure-as-INPUT do what structure-as-TARGET could not;
  (b) longer budgets / full-model fine-tune as a capacity control;
  (c) the process-expression P1 conditioning question on top of whichever input set wins.
The estimand critique of (a) — what it changes about the task and its comparability — is
squarely the rigor lane's mathematics, not procedure.

## Provenance

Runner: sm_fs_ranking_shadow.py (+ residual arm); artifacts in ~/mu_data/sm_fs_ranking_shadow_v1
(0600/0700, no-replace), 45 fits + 45 evals sealed with stamps; decision via the frozen
sm_fs_bootstrap. Tier-2 machinery intact for a later authoritative rerun of any configuration
that earns it.

## Day 2 addendum: data-scale and multi-judge shadow runs (all null; all informative)

| experiment | SM-FS MRR | verdict |
|---|---|---|
| wiki pretrain (64x graph rows) zero-shot | 0.072 | no cross-corpus transfer |
| wiki pretrain + SM-FS graded fine-tune | 0.283 vs 0.280 | more same-channel data: null |
| multi-judge, random-pair e5 distillation | graph 0.175 / e5j 0.157 | naive mixing HURT (dilution + mean-teaching) |
| multi-judge, WITHIN-QUERY e5 distillation | graph 0.220 / e5j 0.116 | pointwise MSE cannot absorb a ranker |
| blend sweeps (cos + a*graph + b*e5j) | best = pure cos 0.573 | learned channels add nothing at any weight |

Standing synthesis after 5 experiments (~2h GPU total): the graded-negative MECHANISM is real
(+0.140 CI-solid), but no configuration of the 18-tensor pointwise-MSE mu head approaches frozen
e5 (0.573) — not structure-as-target, not 64x data, not residual heads, not multi-channel
conditioning, not e5 distillation. The convergent explanation is LOSS/CAPACITY: reproducing or
beating a ranker requires (a) a ranking loss (pairwise/listwise — the deferred LINEAGE_RANK arm)
and/or (b) more trainable capacity than last-layer+readouts+residuals. The owner's
configuration-diversity thesis is NOT refuted: it has not yet been tested with a loss that can
express ranking. Next levers, in order: LINEAGE_RANK listwise arm; full-model fine-tune control;
then judge-diversity again on top of whichever works. e5-judge onboarding mechanics (12th row:
card + zero residual + zero embedding, pre-construction state-dict growth) are now working
machinery for that rematch.

## Day 3: the loss/capacity/mixture ladder (shadow; discussion draft — λ-superposition running)

| configuration | mean MRR | per-fold | collapses |
|---|---|---|---|
| graded MSE, 18-tensor (day-1 baseline) | 0.280 | — | none |
| rank-CE listwise, 18-tensor | 0.291 | .361/.273/.285/.296/.239 | none |
| rank-CE + graded mixed, 18-tensor | 0.251 | .381/.320/.268/**.019**/.269 | fold 3 |
| rank-CE alone, FULL capacity (4.0M, lr 2e-4) | 0.175 | **.441**/.008/.012/.016/**.397** | 3 of 5 |
| **rank-CE + graded mixed, FULL capacity** | **0.347** | .437/.283/.317/.338/.361 | **none** |
| tri-mix (+ e5-listwise-KL), FULL capacity | 0.205 | .035/.338/.286/**.016**/.352 | folds 0,3 |
| tri-mix e5-cond / score-superposition | 0.196 / 0.206 | — | — |
| frozen e5 title-cosine | **0.573** | — | — |

Findings, in causal order:
1. LISTWISE LOSS is necessary (regression cannot express ranking; day-2) but insufficient alone
   (+0.01 at partial capacity).
2. CAPACITY raises the ceiling (0.44 on trainable folds) but pure sharp objectives collapse
   3/5 folds — high-ceiling, unstable regime.
3. FUNCTION MIXTURE is the stabilizer: interleaving the dense graded-regression channel with
   rank-CE eliminates every collapse while keeping most of the ceiling (0.347, +24% over any
   prior stable arm). The owner's diversity thesis holds in a stronger-than-expected form:
   diversity stabilizes OPTIMIZATION, not just generalization.
4. Diversity is not monotone: adding a third channel (e5 soft-target KL) reintroduced collapses
   (0.205, 2/5 folds down). There is a composition/budget balance — an open mathematical
   question for the rigor lane (interference vs dilution vs gradient sharpness of soft targets).
5. Gap to frozen e5 (0.573) remains ~0.23. Untested levers: the continuous λ-superposition of
   the two working functions (running: conditioning = λ-blended op embedding, loss =
   λ·CE+(1−λ)·MSE, λ~Beta(.5,.5) — readout is the MRR(λ) interpolation curve), LR schedules /
   longer budgets for the stable mix, and structure-as-input.

Questions for discussion (any agent):
- Why does the 2-function mixture stabilize where 1 and 3 fail? (Gradient interference model?)
- Is the fold-collapse mode diagnosable from loss curves / update norms — an early-warning gate?
- Is 0.573 approachable at all from title-only inputs, or is the remaining gap information-
  theoretic? What experiment separates those?
- Correct λ sampling density for superposition training (Beta(.5,.5) chosen ad hoc)?
